import subprocess
import os

class LlamaCppQuantizer:
    def __init__(self, config):
        self.config = config

    def run(self):
        binary_path = os.path.join('third_party', 'llama.cpp', 'llama-quantize')
        output_directory = self.config['output_directory']
        input_model = self.config['input_model']
        types_mapping = {"Q4_0": 2, "Q4_1": 3, "Q5_0": 8, "Q5_1": 9}

        for type_name in self.config['llama_cpp_types']:
            output_file = os.path.join(output_directory, f"model-{type_name}.gguf")
            type_value = types_mapping[type_name]

            command = [
                    binary_path,
                    input_model,
                    output_file,
                    str(type_value)
                ]

            command = list(filter(None, command))

            subprocess.run(command, check=True)

