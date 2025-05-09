# Dynamo Run

* [Quickstart with pip and vllm](#quickstart-with-pip-and-vllm)
    * [Automatically download a model from Hugging Face](#use-model-from-hugging-face)
    * [Run a model from local file](#run-a-model-from-local-file)
    * [Distributed system](#distributed-system)
* [Full usage details](#full-usage-details)
    * [Setup](#setup)
    * [mistral.rs](#mistralrs)
    * [llama.cpp](#llamacpp)
    * [Sglang](#sglang)
    * [Vllm](#vllm)
    * [TensorRT-LLM](#tensorrt-llm-engine)
    * [Echo Engines](#echo-engines)
    * [Write your own engine in Python](#write-your-own-engine-in-python)
* [Batch mode](#batch-mode)
* [Defaults](#defaults)
* [Extra engine arguments](#extra-engine-arguments)

`dynamo-run` is a CLI tool for exploring the Dynamo components, and an example of how to use them from Rust. It is also available as `dynamo run` if using the Python wheel.

It supports the following engines: mistralrs, llamacpp, sglang, vllm and tensorrt-llm. `mistralrs` is the default.

Usage:
```
dynamo-run in=[http|text|dyn://<path>|batch:<folder>] out=echo_core|echo_full|mistralrs|llamacpp|sglang|vllm|dyn://<path> [--http-port 8080] [--model-path <path>] [--model-name <served-model-name>] [--model-config <hf-repo>] [--tensor-parallel-size=1] [--base-gpu-id=0] [--extra-engine-args=args.json] [--router-mode random|round-robin]
```

Example: `dynamo run Qwen/Qwen3-0.6B`

Set environment variable `DYN_LOG` to adjust logging level, e.g. `export DYN_LOG=debug`. It has the same syntax as `RUST_LOG`, ask AI for details.

## Quickstart with pip and vllm

If you used `pip` to install `dynamo` you should have the `dynamo-run` binary pre-installed with the `vllm` engine. You must be in a virtual env with vllm installed to use this. To compile from source, see "Full documentation" below.

The vllm and sglang engines require [etcd](https://etcd.io/) and [nats](https://nats.io/) with jetstream (`nats-server -js`). Mistralrs and llamacpp do not.

### Use model from Hugging Face

This will automatically download Qwen3 4B from Hugging Face (16 GiB download) and start it in interactive text mode:
```
dynamo run out=vllm Qwen/Qwen3-4B
```

General format for HF download:
```
dynamo run out=<engine> <HUGGING_FACE_ORGANIZATION/MODEL_NAME>
```

For gated models (e.g. meta-llama/Llama-3.2-3B-Instruct) you have to have an `HF_TOKEN` environment variable set.

The parameter can be the ID of a HuggingFace repository (it will be downloaded), a GGUF file, or a folder containing safetensors, config.json, etc (a locally checked out HuggingFace repository).

### Run a model from local file

#### Step 1: Download model from Hugging Face
One of these models should be high quality and fast on almost any machine: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
E.g. https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

Download model file:
```
curl -L -o Llama-3.2-3B-Instruct-Q4_K_M.gguf "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true"
```
#### Run model from local file
**Text interface**
```
dynamo run Llama-3.2-3B-Instruct-Q4_K_M.gguf # or path to a Hugging Face repo checkout instead of the GGUF
```

**HTTP interface**
```
dynamo run in=http out=mistralrs Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

**List the models**
```
curl localhost:8080/v1/models
```

**Send a request**
```
curl -d '{"model": "Llama-3.2-3B-Instruct-Q4_K_M", "max_completion_tokens": 2049, "messages":[{"role":"user", "content": "What is the capital of South Africa?" }]}' -H 'Content-Type: application/json' http://localhost:8080/v1/chat/completions
```

### Distributed System

You can run the ingress side (HTTP server and pre-processing) on one machine, for example a CPU node, and the worker on a different machine (a GPU node).

You will need [etcd](https://etcd.io/) and [nats](https://nats.io) with jetstream installed and accessible from both nodes.

**Node 1:**

OpenAI compliant HTTP server, optional pre-processing, worker discovery.

```
dynamo-run in=http out=dyn://llama3B_pool
```

**Node 2:**

Vllm engine. Receives and returns requests over the network.

```
dynamo-run in=dyn://llama3B_pool out=vllm ~/llms/Llama-3.2-3B-Instruct
```

This will use etcd to auto-discover the model and NATS to talk to it. You can run multiple workers on the same endpoint and it will pick one at random each time.

The `llama3B_pool` name is purely symbolic, pick anything as long as it matches the other node.

Run `dynamo-run --help` for more options.

## Full usage details

`dynamo-run` is what `dynamo run` executes. It is also an example of what you can build in Rust with the `dynamo-llm` and `dynamo-runtime` crates. The following guide demonstrates how you can build from source with all the features.

### Setup

#### Step 1: Install libraries
**Ubuntu:**
```
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libssl-dev libclang-dev protobuf-compiler python3-dev cmake
```

**macOS:**
- [Homebrew](https://brew.sh/)
```
# if brew is not installed on your system, install it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
- [Xcode](https://developer.apple.com/xcode/)

```
brew install cmake protobuf

# Check that Metal is accessible
xcrun -sdk macosx metal
```
If Metal is accessible, you should see an error like `metal: error: no input files`, which confirms it is installed correctly.

#### Step 2: Install Rust
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

#### Step 3: Build

- Linux with GPU and CUDA (tested on Ubuntu):
```
cargo build --features cuda
```

- macOS with Metal:
```
cargo build --features metal
```

- CPU only:
```
cargo build
```

Optionally you can run `cargo build` from any location with arguments:

```
--target-dir /path/to/target_directory` # specify target_directory with write privileges
--manifest-path /path/to/project/Cargo.toml` # if cargo build is run outside of `launch/` directory
```

The binary will be called `dynamo-run` in `target/debug`
```
cd target/debug
```

Build with `--release` for a smaller binary and better performance, but longer build times. The binary will be in `target/release`.

### mistralrs

[mistral.rs](https://github.com/EricLBuehler/mistral.rs) is a pure Rust engine that is fast to run, fast to load, supports GGUF as well as safetensors, and runs well on CPU as well as GPU. For those reasons it is the default engine.

```
dynamo-run Qwen/Qwen3-4B
```

is equivalent to

```
dynamo-run in=text out=mistralrs Qwen/Qwen3-4B
```

If you have multiple GPUs, mistral.rs does automatic tensor parallelism. You do not need to pass any extra flags to dynamo-run to enable it.

### llamacpp

Currently [llama.cpp](https://github.com/ggml-org/llama.cpp) is not included by default. Build it like this:

```
cargo build --features llamacpp[,cuda|metal|vulkan] -p dynamo-run
```

```
dynamo-run out=llamacpp ~/llms/gemma-3-1b-it-q4_0.gguf
dynamo-run out=llamacpp ~/llms/Qwen3-0.6B-Q8_0.gguf # From https://huggingface.co/ggml-org
```

Note that in some cases we are unable to extract the tokenizer from the GGUF, and so a Hugging Face checkout of a matching model must also be passed. Dynamo will use the weights from the GGUF and the pre-processor (`tokenizer.json`, etc) from the `--model-config`:
```
dynamo-run out=llamacpp ~/llms/Llama-4-Scout-17B-16E-Instruct-UD-IQ1_S.gguf --model-config ~/llms/Llama-4-Scout-17B-16E-Instruct
```

If you have multiple GPUs, llama.cpp does automatic tensor parallelism. You do not need to pass any extra flags to dynamo-run to enable it.

### sglang

The [SGLang](https://docs.sglang.ai/index.html) engine requires [etcd](https://etcd.io/) and [nats](https://nats.io/) with jetstream (`nats-server -js`) to be running.

1. Setup the python virtual env:

```
uv venv
source .venv/bin/activate
uv pip install pip
uv pip install sgl-kernel --force-reinstall --no-deps
uv pip install "sglang[all]==0.4.2" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
```

2. Run

Any example above using `out=sglang` will work, but our sglang backend is also multi-gpu.

```
cd target/debug
./dynamo-run in=http out=sglang --model-path ~/llms/DeepSeek-R1-Distill-Llama-70B/ --tensor-parallel-size 8
```

To pass extra arguments to the sglang engine see *Extra engine arguments* below.

**Multi-GPU**

Pass `--tensor-parallel-size <NUM-GPUS>` to `dynamo-run`.

```
dynamo-run out=sglang ~/llms/Llama-4-Scout-17B-16E-Instruct/ --tensor-parallel-size 8
```

To specify which GPU to start from pass `--base-gpu-id <num>`, for example on a shared eight GPU machine where GPUs 0-3 are already in use:
```
dynamo-run out=sglang <model> --tensor-parallel-size 4 --base-gpu-id 4
```

**Multi-node:**

Dynamo only manages the leader node (node rank 0). The follower nodes are started in the [normal sglang way](https://docs.sglang.ai/references/deepseek.html#running-examples-on-multi-node).

Leader node:
```
dynamo-run out=sglang /data/models/DeepSeek-R1-Distill-Llama-70B/ --tensor-parallel-size 16 --node-rank 0 --num-nodes 2 --leader-addr 10.217.98.122:5000
```

All follower nodes. Increment `node-rank` each time:
```
python3 -m sglang.launch_server --model-path /data/models/DeepSeek-R1-Distill-Llama-70B --tp 16 --dist-init-addr 10.217.98.122:5000 --nnodes 2 --node-rank 1 --trust-remote-code
```

- Parameters `--leader-addr` and `--dist-init-addr` must match and be the IP address of the leader node. All followers must be able to connect. SGLang is using [PyTorch Distributed](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) for networking.
- Parameters `--tensor-parallel-size` and `--tp` must match and be the total number of GPUs across the cluster.
- `--node-rank` must be unique consecutive integers starting at 1. The leader, managed by Dynamo, is 0.

### vllm

Using the [vllm](https://github.com/vllm-project/vllm) Python library. Slow startup, fast inference. Supports both safetensors from HF and GGUF files, but is very slow for GGUF - prefer llamacpp.

The vllm engine requires requires [etcd](https://etcd.io/) and [nats](https://nats.io/) with jetstream (`nats-server -js`) to be running.

We use [uv](https://docs.astral.sh/uv/) but any virtualenv manager should work.

1. Setup:
```
uv venv
source .venv/bin/activate
uv pip install pip
uv pip install vllm==0.8.4 setuptools
```

**Note: If you're on Ubuntu 22.04 or earlier, you will need to add `--python=python3.10` to your `uv venv` command**

2. Build:
```
cargo build
cd target/debug
```

3. Run
Inside that virtualenv:

**HF repo:**
```
./dynamo-run in=http out=vllm ~/llms/Llama-3.2-3B-Instruct/

```

To pass extra arguments to the vllm engine see [Extra engine arguments](#extra_engine_arguments) below.

**Multi-GPU**

Pass `--tensor-parallel-size <NUM-GPUS>` to `dynamo-run`.

To specify which GPUs to use set environment variable `CUDA_VISIBLE_DEVICES`.

**Multi-node:**

vllm uses [ray](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#running-vllm-on-multiple-nodes) for pipeline parallel inference. Dynamo does not change or manage that.

Here is an example on two 8x nodes:
- Leader node: `ray start --head --port=6379`
- Each follower node: `ray start --address='<HEAD_NODE_IP>:6379`
- Leader node: `dynamo-run out=vllm ~/llms/DeepSeek-R1-Distill-Llama-70B/ --tensor-parallel-size 16`

The `--tensor-parallel-size` parameter is the total number of GPUs in the cluster. This is often constrained by a model dimension such as being a divisor of the number of attention heads.

Startup can be slow so you may want to `export DYN_LOG=debug` to see progress.

Shutdown: `ray stop`

#### TensorRT-LLM engine

To run a TRT-LLM model with dynamo-run we have included a python based [async engine] (/examples/tensorrt_llm/engines/agg_engine.py).
To configure the TensorRT-LLM async engine please see [llm_api_config.yaml](/examples/tensorrt_llm/configs/llm_api_config.yaml). The file defines the options that need to be passed to the LLM engine. Follow the steps below to serve trtllm on dynamo run.

##### Step 1: Build the environment

See instructions [here](/examples/tensorrt_llm/README.md#build-docker) to build the dynamo container with TensorRT-LLM.

##### Step 2: Run the environment

See instructions [here](/examples/tensorrt_llm/README.md#run-container) to run the built environment.

##### Step 3: Execute `dynamo run` command

Execute the following to load the TensorRT-LLM model specified in the configuration.
```
dynamo run out=pystr:/workspace/examples/tensorrt_llm/engines/trtllm_engine.py  -- --engine_args /workspace/examples/tensorrt_llm/configs/llm_api_config.yaml
```

### Echo Engines

Dynamo includes two echo engines for testing and debugging purposes:

#### echo_core

The `echo_core` engine accepts pre-processed requests and echoes the tokens back as the response. This is useful for testing pre-processing functionality as the response will include the full prompt template.

```
dynamo-run in=http out=echo_core --model-path <hf-repo-checkout>
```

Note that to use it with `in=http` you need to tell the post processor to ignore stop tokens from the template by adding `nvext.ignore_eos` like this:
```
curl -N -d '{"nvext": {"ignore_eos": true}, "stream": true, "model": "Qwen2.5-3B-Instruct", "max_completion_tokens": 4096, "messages":[{"role":"user", "content": "Tell me a story" }]}' ...
```

The default `in=text` sets that for you.

#### echo_full

The `echo_full` engine accepts un-processed requests and echoes the prompt back as the response.

```
dynamo-run in=http out=echo_full --model-name my_model
```

#### Configuration

Both echo engines use a configurable delay between tokens to simulate generation speed. You can adjust this using the `DYN_TOKEN_ECHO_DELAY_MS` environment variable:

```
# Set token echo delay to 1ms (1000 tokens per second)
DYN_TOKEN_ECHO_DELAY_MS=1 dynamo-run in=http out=echo_full
```

The default delay is 10ms, which produces approximately 100 tokens per second.

### Batch mode

`dynamo-run` can take a jsonl file full of prompts and evaluate them all:

```
dynamo-run in=batch:prompts.jsonl out=llamacpp <model>
```

The input file should look like this:
```
{"text": "What is the capital of France?"}
{"text": "What is the capital of Spain?"}
```

Each one is passed as a prompt to the model. The output is written back to the same folder in `output.jsonl`. At the end of the run some statistics are printed.
The output looks like this:
```
{"text":"What is the capital of France?","response":"The capital of France is Paris.","tokens_in":7,"tokens_out":7,"elapsed_ms":1566}
{"text":"What is the capital of Spain?","response":".The capital of Spain is Madrid.","tokens_in":7,"tokens_out":7,"elapsed_ms":855}
```

### Write your own engine in Python

Note: This section replaces "bring-your-own-engine".

The [dynamo](https://pypi.org/project/ai-dynamo/) Python library allows you to build your own engine and attach it to Dynamo.

The Python file must do three things:
1. Decorate a function to get the runtime
2. Register on the network
3. Attach a request handler

```
from dynamo.llm import ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker

# 1. Decorate a function to get the runtime
#
@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):

    # 2. Register ourselves on the network
    #
    component = runtime.namespace("namespace").component("component")
    await component.create_service()
    model_path = "Qwen/Qwen2.5-0.5B-Instruct" # or "/data/models/Qwen2.5-0.5B-Instruct"
    model_type = ModelType.Backend
    endpoint = component.endpoint("endpoint")
    # Optional last param to register_llm is model_name. If not present derives it from model_path
    await register_llm(model_type, endpoint, model_path)

    # Initialize your engine here
    # engine = ...

    # 3. Attach request handler
    #
    await endpoint.serve_endpoint(RequestHandler(engine).generate)

class RequestHandler:

    def __init__(self, engine):
        ...

    async def generate(self, request):
        # Call the engine
        # yield result dict
        ...

if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
```


The `model_path` can be:
- A HuggingFace repo ID. It will be downloaded and cached locally.
- The path to a checkout of a HuggingFace repo - any folder containing safetensor files as well as `config.json`, `tokenizer.json` and `tokenizer_config.json`.
- The path to a GGUF file, if your engine supports that.

The `model_type` can be:
- ModelType.Backend. Dynamo handles pre-processing. Your `generate` method receives a `request` dict containing a `token_ids` array of int. It must return a dict also containing a `token_ids` array and an optional `finish_reason` string.
- ModelType.Chat. Your `generate` method receives a `request` and must return a response dict of type [OpenAI Chat Completion](https://platform.openai.com/docs/api-reference/chat). Your engine handles pre-processing.
- ModelType.Completion. Your `generate` method receives a `request` and must return a response dict of the older [Completions](https://platform.openai.com/docs/api-reference/completions). Your engine handles pre-processing.

Here are some example engines:
- [vllm simple](https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/python/examples/hello_world/server_vllm.py)
- [sglang simple](https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/python/examples/hello_world/server_sglang.py)
- [vllm](https://github.com/ai-dynamo/dynamo/blob/main/launch/dynamo-run/src/subprocess/vllm_inc.py)
- [sglang](https://github.com/ai-dynamo/dynamo/blob/main/launch/dynamo-run/src/subprocess/sglang_inc.py)


### Defaults

The input defaults to `in=text`. The output will default to `out=mistralrs` engine, unless it is disabled with `--no-default-features` in which case vllm is used.

### Extra engine arguments

The vllm and sglang backends support passing any argument the engine accepts.

Put the arguments in a JSON file:
```
{
    "dtype": "half",
    "trust_remote_code": true
}
```

Pass it like this:
```
dynamo-run out=sglang ~/llms/Llama-3.2-3B-Instruct --extra-engine-args sglang_extra.json
```
