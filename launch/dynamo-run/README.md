# Dynamo service runner

`dynamo-run` is a tool for exploring the dynamo components.

## Setup

Libraries Ubuntu:
```
apt install -y build-essential libhwloc-dev libudev-dev pkg-config libssl-dev protobuf-compiler python3-dev
```

Libraries macOS:
```
brew install cmake protobuf

# install Xcode from App Store and check that Metal is accessible
xcrun -sdk macosx metal

# may have to install Xcode Command Line Tools:
xcode-select --install
```

Install Rust:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## Build

Navidate to launch/ directory
```
cd launch/
```

- Linux with GPU and CUDA (tested on Ubuntu):

```
cargo build --release --features mistralrs,cuda
```

- macOS with Metal:

```
cargo build --release --features mistralrs,metal
```

- CPU only:

```
cargo build --release --features mistralrs
```

The binary will be called `dynamo-run` in `target/release`
```
cd target/release
```

## Quickstart
### Automatically download a model from [Hugging Face](https://huggingface.co/models)

You may have to have an `HF_TOKEN` environment variable set.
```
./dynamo-run <HUGGING_FACE_ORGANIZATION/MODEL_NAME>
```
This will automatically download Qwen2.5 1B from Hugging Face (1 GiB download) and start it interactive text mode:
`./dynamo-run Qwen/Qwen2.5-1B-Instruct`

### Run a model from local file

For example one of these should be fast and good quality on almost any machine: https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF
E.g. https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q8_0.gguf

Download model file:
`curl -L -o Llama-3.2-1B-Instruct-Q4_K_M.gguf "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf?download=true"`


*Text interface*

`dynamo-run Llama-3.2-1B-Instruct-Q4_K_M.gguf` or path to a Hugging Face repo checkout instead of the GGUF.

*HTTP interface*

`dynamo-run in=http Llama-3.2-1B-Instruct-Q4_K_M.gguf`

List the models: `curl localhost:8080/v1/models`

Send a request:
```
curl -d '{"model": "Llama-3.2-1B-Instruct-Q4_K_M", "max_tokens": 2049, "messages":[{"role":"user", "content": "What is the capital of South Africa?" }]}' -H 'Content-Type: application/json' http://localhost:8080/v1/chat/completions
```

*Multi-node*

Node 1:
```
dynamo-run in=http out=dyn://llama3B_pool
```

Node 2:
```
dynamo-run in=dyn://llama3B_pool out=mistralrs ~/llm_models/Llama-3.2-3B-Instruct
```

This will use etcd to auto-discover the model and NATS to talk to it. You can run multiple workers on the same endpoint and it will pick one at random each time.

The `llama3B_pool` name is purely symbolic, pick anything as long as it matches the other node.

Run `dynamo-run --help` for more options.

## sglang

1. Setup the python virtual env:

```
uv venv
source .venv/bin/activate
uv pip install pip
uv pip install sgl-kernel --force-reinstall --no-deps
uv pip install "sglang[all]==0.4.2" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
```

2. Build

```
cargo build --release --features sglang
```

3. Run

Any example above using `out=sglang` will work, but our sglang backend is also multi-gpu and multi-node.

Node 1:
```
dynamo-run in=http out=sglang --model-path ~/llm_models/DeepSeek-R1-Distill-Llama-70B/ --tensor-parallel-size 8 --num-nodes 2 --node-rank 0 --dist-init-addr 10.217.98.122:9876
```

Node 2:
```
dynamo-run in=none out=sglang --model-path ~/llm_models/DeepSeek-R1-Distill-Llama-70B/ --tensor-parallel-size 8 --num-nodes 2 --node-rank 1 --dist-init-addr 10.217.98.122:9876
```

## llama_cpp

- `cargo build --release --features llamacpp,cuda`

- `dynamo-run out=llama_cpp --model-path ~/llm_models/Llama-3.2-3B-Instruct-Q6_K.gguf --model-config ~/llm_models/Llama-3.2-3B-Instruct/`

The extra `--model-config` flag is because:
- llama_cpp only runs GGUF
- We send it tokens, meaning we do the tokenization ourself, so we need a tokenizer
- We don't yet read it out of the GGUF (TODO), so we need an HF repo with `tokenizer.json` et al

If the build step also builds llama_cpp libraries into the same folder as the binary ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-cuda.so"), then `dynamo-run` will need to find those at runtime. Set `LD_LIBRARY_PATH`, and be sure to deploy them alongside the `dynamo-run` binary.

## vllm

Using the [vllm](https://github.com/vllm-project/vllm) Python library. We only use the back half of vllm, talking to it over `zmq`. Slow startup, fast inference. Supports both safetensors from HF and GGUF files.

We use [uv](https://docs.astral.sh/uv/) but any virtualenv manager should work.

Setup:
```
uv venv
source .venv/bin/activate
uv pip install pip
uv pip install vllm==0.7.3 setuptools
```

**Note: If you're on Ubuntu 22.04 or earlier, you will need to add `--python=python3.10` to your `uv venv` command**

Build:
```
cargo build --release --features vllm
```

Run (still inside that virtualenv) - HF repo:
```
./dynamo-run in=http out=vllm --model-path ~/llm_models/Llama-3.2-3B-Instruct/

```

Run (still inside that virtualenv) - GGUF:
```
./dynamo-run in=http out=vllm --model-path ~/llm_models/Llama-3.2-3B-Instruct-Q6_K.gguf --model-config ~/llm_models/Llama-3.2-3B-Instruct/
```

+ Multi-node:

Node 1:
```
dynamo-run in=text out=vllm ~/llm_models/Llama-3.2-3B-Instruct/ --tensor-parallel-size 8 --num-nodes 2 --leader-addr 10.217.98.122:6539 --node-rank 0
```

Node 2:
```
dynamo-run in=none out=vllm ~/llm_models/Llama-3.2-3B-Instruct/ --num-nodes 2 --leader-addr 10.217.98.122:6539 --node-rank 1
```

## Python bring-your-own-engine

You can provide your own engine in a Python file. The file must provide a generator with this signature:
```
async def generate(request):
```

Build: `cargo build --release --features python`

### Python does the pre-processing

If the Python engine wants to receive and returns strings - it will do the prompt templating and tokenization itself - run it like this:

```
dynamo-run out=pystr:/home/user/my_python_engine.py
```

- The `request` parameter is a map, an OpenAI compatible create chat completion request: https://platform.openai.com/docs/api-reference/chat/create
- The function must `yield` a series of maps conforming to create chat completion stream response (example below).
- If using an HTTP front-end add the `--model-name` flag. This is the name we serve the model under.

The file is loaded once at startup and kept in memory.

Example engine:
```
import asyncio

async def generate(request):
    yield {"id":"1","choices":[{"index":0,"delta":{"content":"The","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-1B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":" capital","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-1B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":" of","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-1B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":" France","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-1B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":" is","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-1B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":" Paris","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-1B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":".","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-1B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":"","role":"assistant"},"finish_reason":"stop"}],"created":1841762283,"model":"Llama-3.2-1B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
```

### Dynamo does the pre-processing

If the Python engine wants to receive and return tokens - the prompt templating and tokenization is already done - run it like this:
```
dynamo-run out=pytok:/home/user/my_python_engine.py --model-path <hf-repo-checkout>
```

- The request parameter is a map that looks like this:
```
{'token_ids': [128000, 128006, 9125, 128007, ... lots more ... ], 'stop_conditions': {'max_tokens': 8192, 'stop': None, 'stop_token_ids_hidden': [128001, 128008, 128009], 'min_tokens': None, 'ignore_eos': None}, 'sampling_options': {'n': None, 'best_of': None, 'presence_penalty': None, 'frequency_penalty': None, 'repetition_penalty': None, 'temperature': None, 'top_p': None, 'top_k': None, 'min_p': None, 'use_beam_search': None, 'length_penalty': None, 'seed': None}, 'eos_token_ids': [128001, 128008, 128009], 'mdc_sum': 'f1cd44546fdcbd664189863b7daece0f139a962b89778469e4cffc9be58ccc88', 'annotations': []}
```

- The `generate` function must `yield` a series of maps that look like this:
```
{"token_ids":[791],"tokens":None,"text":None,"cum_log_probs":None,"log_probs":None,"finish_reason":None}
```

- Command like flag `--model-path` which must point to a Hugging Face repo checkout containing the `tokenizer.json`. The `--model-name` flag is optional. If not provided we use the HF repo name (directory name) as the model name.

Example engine:
```
import asyncio

async def generate(request):
    yield {"token_ids":[791]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[6864]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[315]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[9822]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[374]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[12366]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[13]}
```

## trtllm

TensorRT-LLM. Requires `clang` and `libclang-dev`.

Build:
```
cargo build --release --features trtllm
```

Run:
```
dynamo-run in=text out=trtllm --model-path /app/trtllm_engine/ --model-config ~/llm_models/Llama-3.2-3B-Instruct/
```

Note that TRT-LLM uses it's own `.engine` format for weights. Repo models must be converted like so:

+ Get the build container
```
docker run --gpus all -it nvcr.io/nvidian/nemo-llm/trtllm-engine-builder:0.2.0 bash
```

+ Fetch the model and convert
```
mkdir /tmp/model
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir /tmp/model
python convert_checkpoint.py --model_dir /tmp/model/ --output_dir ./converted --dtype [float16|bfloat16|whatever you want] --tp_size X --pp_size Y
trtllm-build --checkpoint_dir ./converted --output_dir ./final/trtllm_engine --use_paged_context_fmha enable --gemm_plugin auto
```

The `--model-path` you give to `dynamo-run` must contain the `config.json` (TRT-LLM's , not the model's) and `rank0.engine` (plus other ranks if relevant).

+ Execute
TRT-LLM is a C++ library that must have been previously built and installed. It needs a lot of memory to compile. Gitlab builds a container you can try:
```
sudo docker run --gpus all -it -v /home/user:/outside-home gitlab-master.nvidia.com:5005/dl/ai-services/libraries/rust/nim-nvllm/tensorrt_llm_runtime:85fa4a6f
```

Copy the trt-llm engine, the model's `.json` files (for the model deployment card) and the `nio` binary built for the correct glibc (container is Ubuntu 22.04 currently) into that container.

## Echo Engines

Dynamo includes two echo engines for testing and debugging purposes:

### echo_core

The `echo_core` engine accepts pre-processed requests and echoes the tokens back as the response. This is useful for testing pre-processing functionality as the response will include the full prompt template.

```
dynamo-run in=http out=echo_core --model-path <hf-repo-checkout>
```

### echo_full

The `echo_full` engine accepts un-processed requests and echoes the prompt back as the response.

```
dynamo-run in=http out=echo_full --model-name my_model
```

### Configuration

Both echo engines use a configurable delay between tokens to simulate generation speed. You can adjust this using the `DYN_TOKEN_ECHO_DELAY_MS` environment variable:

```
# Set token echo delay to 1ms (1000 tokens per second)
DYN_TOKEN_ECHO_DELAY_MS=1 dynamo-run in=http out=echo_full
```

The default delay is 10ms, which produces approximately 100 tokens per second.
