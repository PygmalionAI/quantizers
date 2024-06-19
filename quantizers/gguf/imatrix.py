import os
import subprocess
import re
import sys

from rich.progress import Progress
import torch
import gpustat

class GGUFImatrix:
    def __init__(self, config):
        self.config = config
        self.binary_path = os.path.join('third_party', 'llama.cpp', 'llama-imatrix')

    def process(self):
        input_model = self.config['input_model']
        imatrix_file = self.config.get('imatrix', None)
        output_base_name = self.config.get('output_base_name', 'gguf-model')
        imatrix_gpu = self.config.get('imatrix_gpu', False)

        if not imatrix_file or not imatrix_file.endswith(('.dat', '.txt')):
            raise ValueError("imatrix file must be a .dat or .txt file.")

        output_file = os.path.join('artifacts', f"imatrix-{output_base_name}.dat")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if imatrix_file.endswith('.dat'):
            pass
        elif imatrix_file.endswith('.txt'):
            if torch.cuda.is_available() and imatrix_gpu:
                num_layers = self._get_num_layers(input_model)
                num_layers_to_gpu = self._get_num_layers_to_gpu(num_layers, input_model)
                while num_layers_to_gpu > 0:
                    try:
                        self._run_imatrix_binary(input_model, imatrix_file,
                                                 output_file, num_layers_to_gpu)
                        break
                    except subprocess.CalledProcessError as e:
                        if  num_layers_to_gpu == 0:
                            print(f"imatrix creation failed: {e}")
                            sys.exit(1)
                        num_layers_to_gpu -= 1
            else:
                self._run_imatrix_binary(input_model, imatrix_file, output_file)

    def _get_num_layers(self, input_model):
        command = [self.binary_path, '-m', input_model, '-f', '/dev/null']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            line = line.decode('utf-8').strip()
            if line.startswith('llm_load_print_meta: n_layer'):
                return int(re.findall(r'\d+', line)[0])
        return None

    def _get_num_layers_to_gpu(self, num_layers, input_model):
        stats = gpustat.new_query()
        gpu_info = stats.gpus[0]
        free_vram = gpu_info.memory_free * 1024**2  # to bytes
        model_size = os.path.getsize(input_model) + 1 * 1024**3  # add 3gb wiggle room
        layer_size = model_size / num_layers
        num_layers_to_gpu = free_vram // layer_size
        return max(0, min(num_layers, num_layers_to_gpu))

    def _run_imatrix_binary(self, input_model, imatrix_file,
                            output_file, num_layers_to_gpu = None):
        command = [self.binary_path, '-m', input_model, '-f', imatrix_file, '-o', output_file]
        if num_layers_to_gpu is not None:
            print(f"Using {num_layers_to_gpu} layers on the GPU.")
            command.extend(['-ngl', str(num_layers_to_gpu)])
            command.extend(['-sm', 'none'])
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        total = None
        task_id = None

        with Progress() as progress:
            output = b''
            while chunk := process.stdout.read(1):
                if chunk == b'\n' or chunk == b',':
                    line = output.decode('utf-8').strip()
                    output = b''
                    if line.startswith('compute_imatrix: computing over'):
                        total = int(re.findall(r'\d+', line)[0])
                    elif line.startswith('[') and total is not None:
                        current = int(re.findall(r'\d+', line)[0])
                        if task_id is None:
                            task_id = progress.add_task(f"[cyan]Processing imatrix file...",
                                                        total=total)
                        progress.update(task_id, advance=(
                            current - progress.tasks[task_id].completed))
                else:
                    output += chunk
