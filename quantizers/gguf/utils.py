import shutil
import os
import glob
import time
import threading

from huggingface_hub import HfApi, whoami, create_repo
from huggingface_hub.utils import HfHubHTTPError
from InquirerPy import inquirer


QUANT_COMPRESSION_RATIOS = {
    "Q4_0": 0.29099, "IQ1_M": 0.13454, "IQ3_XS": 0.2189,
    "Q2_K": 0.19784, "Q4_1": 0.31926, "IQ3_XXS": 0.2038,
    "Q3_K_M": 0.2501, "IQ1_S": 0.12568, "Q3_K": 0.2501,
    "Q3_K_S": 0.228, "Q4_K": 0.3062, "Q5_0": 0.3493,
    "Q2_K_S": 0.186, "Q5_K_S": 0.3484, "Q8_0": 0.5315,
    "Q4_K_S": 0.292, "Q5_K": 0.3567, "IQ2_XS": 0.1621,
    "IQ2_XXS": 0.1493, "IQ4_XS": 0.2767, "Q5_1": 0.3776,
    "Q6_K": 0.4104, "IQ3_S": 0.2291, "IQ2_S": 0.1716,
    "IQ3_M": 0.2355, "Q4_K_M": 0.30622, "IQ2_M": 0.1834,
    "Q5_K_M": 0.3567, "IQ4_NL": 0.29112, "Q3_K_L": 0.2689,
}

def get_free_space(directory):
    total, used, free = shutil.disk_usage(directory)
    return free  # bytes


def get_model_size(input_path):
    if os.path.isfile(input_path) and input_path.endswith('.gguf'):
        return os.path.getsize(input_path)
    elif os.path.isdir(input_path):
        safetensors = glob.glob(os.path.join(input_path, '*.safetensors'))
        bins = glob.glob(os.path.join(input_path, '*.bin'))
        if safetensors:
            return sum(os.path.getsize(f) for f in safetensors)
        elif bins:
            return sum(os.path.getsize(f) for f in bins)
    else:
        raise ValueError(f"The specified input path {input_path} is not a valid GGUF file "
                         "or directory containing Hugging Face model files.")

def check_disk_space(input_path, types_to_process, output_directory):
    model_size = get_model_size(input_path)
    total_size = 0
    quant_sizes = {}
    for type_name in types_to_process:
        ratio = QUANT_COMPRESSION_RATIOS.get(type_name)
        if ratio is not None:
            quant_size = model_size * ratio * 1.02  # Add 2% wiggle room
            total_size += quant_size
            quant_sizes[type_name] = quant_size

    free_space = get_free_space(output_directory)
    print(f"Total free space: {free_space*1e-9:.2f} GB")
    print(f"Total estimated size of quantized models: {total_size*1e-9:.2f} GB")
    if total_size > free_space:
        print("Not enough free space on disk for all selected quants.")
        checklist_items = [{"name": f"{type_name} ({quant_size / 1e9:.2f} GB)", "value": type_name} for type_name, quant_size in quant_sizes.items()]
        to_remove = []
        def get_user_input():
            nonlocal to_remove
            to_remove = inquirer.checkbox(
                message="Select quants to remove. Press Space to "
                "select/deselect. Press Enter to confirm.",
                choices=checklist_items,
            ).execute()
        t = threading.Thread(target=get_user_input)
        t.start()
        t.join(timeout=20)
        if t.is_alive():
            print("No user input received. Exiting.")
            exit(1)
        for type_name in to_remove:
            types_to_process.remove(type_name)
            total_size -= quant_sizes[type_name]
        if total_size > free_space:
            print("Still not enough free space on disk. Please free up some space and try again.")
            exit(1)
    return types_to_process
    


def upload_to_hub(self, directory_path):
    hf_token = self.config.get('hf_token')
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token is required to upload to the hub.")

    hub_model_entity = self.config.get('hub_model_entity')
    if not hub_model_entity:
        username = whoami(token=hf_token)['name']
    else:
        username = hub_model_entity

    hub_model_name = self.config.get('hub_model_name', 'gguf-model')
    repo_id = f"{username}/{hub_model_name}"
    private = self.config.get('private', False)
    try:
        create_repo(repo_id, token=hf_token, private=private)
    except HfHubHTTPError as e:
        if e.response.status_code != 409:
            raise

    api = HfApi()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            api.upload_folder(
                folder_path=directory_path,
                repo_id=repo_id,
                repo_type="model",
                allow_patterns="*.gguf",
                multi_commits=True,
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                raise RuntimeError(f"Failed to upload to the hub: {e}")