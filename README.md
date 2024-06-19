# AutoQuant

A work-in-progress.

AutoQuant is a library that provides an easy-to-use interface for quantizing LLMs into various formats by using YAML configs.

To get started, clone the repo recursively:

```sh
git clone https://github.com/PygmalionAI/AutoQuant.git
cd AutoQuant
git submodule update --init --recursive
pushd third_party/llama.cpp && make -j4 && popd
python3 -m pip install -r requirements.txt
```

You can then use the library with `python3 main.py config.yaml`. See the [examples](/examples/) directory for some, well, examples. Currently, **only GGUF is supported.**