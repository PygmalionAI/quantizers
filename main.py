import yaml
from quantizers.llama_cpp import LlamaCppQuantizer

def run():
    with open('config.yaml') as file:
        config = yaml.safe_load(file)

        if config['llama_cpp']:
            quantizer = LlamaCppQuantizer(config)
            quantizer.run()

if __name__ == '__main__':
    run()
