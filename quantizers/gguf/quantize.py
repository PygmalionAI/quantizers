import subprocess
import os
import glob
import logging
import re

from rich.progress import Progress
from huggingface_hub import snapshot_download

from .utils import check_disk_space, upload_to_hub
from .imatrix import GGUFImatrix

logger = logging.getLogger(__name__)

class GGUFQuantizer:
    VALID_TYPES = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ2_XXS", "IQ2_XS",
                   "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M", "Q2_K", "Q2_K_S",
                   "IQ3_XXS", "IQ3_S", "IQ3_M", "Q3_K", "IQ3_XS", "Q3_K_S",
                   "Q3_K_M", "Q3_K_L", "IQ4_NL", "IQ4_XS", "Q4_K",
                   "Q4_K_S", "Q4_K_M", "Q5_K", "Q5_K_S", "Q5_K_M",
                   "Q6_K", "Q8_0", "F16", "BF16", "F32", "COPY"]


    def __init__(self, config):
        self.config = config

    def _get_types_to_process(self):
        types_to_process = [type_name.upper() for type_name in self.config['gguf_types']]
        if 'ALL' in types_to_process:
            exclude_types = self._get_exclude_types()
            types_to_process = [type_name for type_name in self.VALID_TYPES if type_name not in exclude_types]
        return types_to_process

    def _get_exclude_types(self):
        imatrix_path = self.config.get('imatrix', None)
        imatrix_required = [type_name for type_name in self.VALID_TYPES if "I" in type_name] + ["Q2_K_S"]
        exclude_from_all = ["F16", "BF16", "F32", "COPY"]
        if imatrix_required and not imatrix_path:
            logger.warning(f"imatrix not provided. Skipping quantization types: {imatrix_required}.")
            return exclude_from_all + imatrix_required
        else:
            return exclude_from_all

    def run(self):
        binary_path = os.path.join('third_party', 'llama.cpp', 'llama-quantize')
        output_directory = self.config.get('output_directory')
        output_name = self.config.get('output_base_name', 'gguf-model')
        input_model = self.config['input_model']
        imatrix_path = self.config.get('imatrix', None)

        GGUF_FILE = False
        LOCAL_DIR = False
        HF_REPO = False

        if os.path.isfile(input_model) and input_model.endswith('.gguf'):
            GGUF_FILE = True
        elif os.path.isdir(input_model) and os.path.exists(os.path.join(input_model, 'config.json')):
            LOCAL_DIR = True
        else:
            HF_REPO = True

        if HF_REPO:
            print(f"Model {input_model} not found locally. Checking Hugging Face Hub.")
            try:
                input_model = snapshot_download(input_model,
                                                # do not download .pth, .pt, .h5, .msgpack files
                                                ignore_patterns=["*.pth", "*.pt", "*.h5", "*.msgpack"])
            except Exception as e:
                print(f"Error downloading model from Hugging Face Hub: {e}")
                raise

        os.makedirs(output_directory, exist_ok=True)

        types_to_process = self._get_types_to_process()
        types_to_process = check_disk_space(input_model, types_to_process, output_directory)

        if LOCAL_DIR or HF_REPO:
            print("Exporting Hugging Face model to GGUF. This may take "
                  "a while depending on model size.")
            self._run_conversion_script(input_model)
            if HF_REPO:
                gguf_file = glob.glob(os.path.join(output_directory, 'converted.gguf'))[0]
            else:
                gguf_file = glob.glob(os.path.join(input_model, '*.gguf'))[0]
            input_model = gguf_file
            self.config['input_model'] = gguf_file

        if imatrix_path and imatrix_path.endswith('.txt'):
            print("Processing imatrix file. This will take a while.")
            imatrix = GGUFImatrix(self.config)
            imatrix.process()
            imatrix_path = os.path.join('artifacts', f"imatrix-{self.config.get('output_base_name', 'gguf-model')}.dat")

        for type_name in types_to_process:
            self._process_type(type_name, binary_path, input_model, output_directory, output_name, imatrix_path)

        if LOCAL_DIR or HF_REPO and not self.config.get('keep_gguf', False):
            print("Removing temporary GGUF file.")
            os.remove(input_model)

        if self.config.get('upload_to_hub', False):
            print("Uploading to the Hugging Face Hub.")
            upload_to_hub(output_directory, self.config)
            print("Uploaded to Hugging Face hub.")

    
    def _run_conversion_script(self, model_directory):
        convert_script_path = os.path.join('third_party', 'llama.cpp', 'convert-hf-to-gguf.py')
        output_directory = self.config.get('output_directory')
        outfile = os.path.join(output_directory, 'converted.gguf')
        command = ['python3', convert_script_path, model_directory, '--outfile', outfile]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            line = line.decode('utf-8').strip()
            logger.info(line)

    def _process_type(self, type_name, binary_path, input_model, output_directory, output_name, imatrix_path):
        if not input_model.endswith(".gguf"):
            raise ValueError("Input model must be a GGUF file.")

        if type_name not in self.VALID_TYPES:
            raise ValueError(f"Invalid quantization type for GGUF: {type_name}. Must be one of {self.VALID_TYPES}.")

        if "I" in type_name and not imatrix_path:
            raise ValueError("imatrix is recommended for I-quants. Please provide the path to the imatrix file.")

        output_file = os.path.join(output_directory, f"{output_name}.{type_name}.gguf")

        command = self._build_command(binary_path, input_model, output_file, type_name, imatrix_path)

        self._run_command(command, type_name)

    def _build_command(self, binary_path, input_model, output_file, type_name, imatrix_path):
        command = [binary_path]
        if imatrix_path:
            command.extend(["--imatrix", imatrix_path])
        command.extend([input_model, output_file, type_name])
        return command

    def _run_command(self, command, type_name):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        total = None
        task_id = None
        with Progress() as progress:
            for line in iter(process.stdout.readline, b''):
                line = line.decode('utf-8').strip()
                if line.startswith('['):
                    current, total = map(int, re.findall(r'\d+', line)[:2])
                    if task_id is None:
                        task_id = progress.add_task(f"[cyan]Quantizing model to {type_name}...", total=total)
                    progress.update(task_id, advance=(current - progress.tasks[task_id].completed))
                else:
                    logger.info(line)
