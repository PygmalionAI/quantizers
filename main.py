import argparse
import yaml

from quantizers import GGUFQuantizer

def run(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)

        if config['gguf']['enabled']:
            quantizer = GGUFQuantizer(config['gguf'])
            quantizer.run()
        else:
            print("No quantizers enabled.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process config file.')
    parser.add_argument('config_file', type=str, help='Path to the config file')

    args = parser.parse_args()

    run(args.config_file)