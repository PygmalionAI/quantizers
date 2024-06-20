# Quantizers

Quantizers is a library that provides an easy-to-use interface for quantizing LLMs into various formats by using YAML configs.


## Supported Operating Systems
- [x] Linux
- [ ] Windows
- [ ] macOS

## Supported Quantizations
- [x] GGUF
- [ ] ExLlamaV2
- [ ] GPTQ
- [ ] AWQ
- [ ] AQLM
- [ ] QuIP
- [ ] QuIP#
- [ ] HQQ
- [ ] HQQ+
- [ ] SqueezeLLM
- [ ] Marlin
- [ ] EETQ
- [ ] SmoothQuant
- [ ] Bitsandbytes
- [ ] TensorRT-LLM

## Installation

To get started, clone the repo recursively:

```sh
git clone https://github.com/PygmalionAI/quantizers.git
cd quantizers
git submodule update --init --recursive
python3 -m pip install -e .
python3 -m pip install -r requirements.txt
```

To build with GPU support (currently for imatrix only), run this instead:

```sh
LLAMA_CUBLAS=1 python3 -m pip install -e .
```

## Usage

Only GGUF is supported for now. You will need a YAML config file. An example is provided in the [examples](/examples/) directory.

Once you've filled out your YAML file, run:

```sh
quantizers examples/gguf/config.yaml
```

## Contribution
At the moment, we don't accept feature contributions until we've finished supporting all the planned quantization methods. PRs for bug fixes and OS support are welcome!