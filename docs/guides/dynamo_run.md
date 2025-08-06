# Running Dynamo CLI (`dynamo-run`)


With the Dynamo CLI, you can chat with models quickly using `dynamo-run`
`dynamo-run` is a CLI tool for exploring the Dynamo components. It's also an example of how to use components from Rust. If you use the Python wheel, it's available as `dynamo-run`.

It supports these engines: mistralrs, llamacpp, sglang, vllm, and tensorrt-llm. `mistralrs` is the default.

Usage:
```
dynamo-run in=[http|text|dyn://<path>|batch:<folder>] out=echo_core|echo_full|mistralrs|llamacpp|sglang|vllm|dyn [--http-port 8080] [--model-path <path>] [--model-name <served-model-name>] [--model-config <hf-repo>] [--tensor-parallel-size=1] [--context-length=N] [--num-nodes=1] [--node-rank=0] [--leader-addr=127.0.0.1:9876] [--base-gpu-id=0] [--extra-engine-args=args.json] [--router-mode random|round-robin|kv] [--kv-overlap-score-weight=1.0] [--router-temperature=0.0] [--use-kv-events=true] [--verbosity (-v|-vv)]
```

Example: `dynamo-run Qwen/Qwen3-0.6B`

Set the environment variable `DYN_LOG` to adjust the logging level; for example, `export DYN_LOG=debug`. It has the same syntax as `RUST_LOG`.

To adjust verbosity, use `-v` to enable debug logging or `-vv` to enable full trace logging. For example:

```bash
dynamo-run in=http out=mistralrs -v  # enables debug logging
dynamo-run in=text out=llamacpp -vv  # enables full trace logging
```

## Quickstart with pip and vllm

If you used `pip` to install `dynamo`, you have the `dynamo-run` binary pre-installed with the `vllm` engine. You must be in a virtual environment with vllm installed to use this engine. To compile from source, see [Full usage details](#full-usage-details) below.

The vllm and sglang engines require [etcd](https://etcd.io/) and [nats](https://nats.io/) with jetstream (`nats-server -js`). Mistralrs and llamacpp do not.

### Use model from Hugging Face

To automatically download Qwen3 4B from Hugging Face (16 GiB download) and to start it in interactive text mode:
```
dynamo-run out=vllm Qwen/Qwen3-4B
```

The general format for HF download follows this pattern:
```
dynamo-run out=<engine> <HUGGING_FACE_ORGANIZATION/MODEL_NAME>
```

For gated models (such as meta-llama/Llama-3.2-3B-Instruct), you must set an `HF_TOKEN` environment variable.

The parameter can be the ID of a HuggingFace repository (which will be downloaded), a GPT-Generated Unified Format (GGUF) file, or a folder containing safetensors, config.json, or similar (perhaps a locally checked out HuggingFace repository).

### Run a model from local file

To run a model from local file:
- Download the model from Hugging Face
- Run the model from local file

See the following sections for details.

#### Download model from Hugging Face
One of the models available from Hugging Face should be high quality and fast on almost any machine: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
For example, try https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

To download model file:
```
curl -L -o Llama-3.2-3B-Instruct-Q4_K_M.gguf "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true"
```
#### Run model from local file
To run the model:

*Text interface*
```
dynamo-run Llama-3.2-3B-Instruct-Q4_K_M.gguf # or path to a Hugging Face repo checkout instead of the GGUF file
```

*HTTP interface*
```
dynamo-run in=http out=mistralrs Llama-3.2-3B-Instruct-Q4_K_M.gguf
```
You can also list models or send a request:

*List the models*
```
curl localhost:8080/v1/models
```

*Send a request*
```
curl -d '{"model": "Llama-3.2-3B-Instruct-Q4_K_M", "max_completion_tokens": 2049, "messages":[{"role":"user", "content": "What is the capital of South Africa?" }]}' -H 'Content-Type: application/json' http://localhost:8080/v1/chat/completions
```

### Distributed System

You can run the ingress side (HTTP server and pre-processing) on one machine, for example a CPU node, and the worker on a different machine (a GPU node).

You will need [etcd](https://etcd.io/) and [nats](https://nats.io) with jetstream installed and accessible from both nodes.

**Node 1:** OpenAI compliant HTTP server, optional pre-processing, worker discovery:

```
dynamo-run in=http out=dyn
```

**Node 2:** Vllm engine. Receives and returns requests over the network:

```
dynamo-run in=dyn://llama3B.backend.generate out=vllm ~/llms/Llama-3.2-3B-Instruct
```

This uses etcd to auto-discover the model and NATS to talk to it. You can
run multiple instances on the same endpoint; it picks one based on the
`--router-mode` (round-robin by default if left unspecified).

Run `dynamo-run --help` for more options.

### Network names

The `in=dyn://` URLs have the format `dyn://namespace.component.endpoint`. For quickstart just use any string `dyn://test`, `dynamo-run` will default any missing parts for you. The pieces matter for a larger system.

* *Namespace*: A pipeline. Usually a model. e.g "llama_8b". Just a name.
* *Component*: A load balanced service needed to run that pipeline. "backend", "prefill", "decode", "preprocessor", "draft", etc. This typically has some configuration (which model to use, for example).
* *Endpoint*: Like a URL. "generate", "load_metrics".
* *Instance*: A process. Unique. Dynamo assigns each one a unique instance_id. The thing that is running is always an instance. Namespace/component/endpoint can refer to multiple instances.

If you run two models, that is two pipelines. An exception would be if doing speculative decoding. The draft model is part of the pipeline of a bigger model.

If you run two instances of the same model ("data parallel") they are the same namespace+component+endpoint but different instances. The router will spread traffic over all the instances of a namespace+component+endpoint. If you have four prefill workers in a pipeline, they all have the same namespace+component+endpoint and are automatically assigned unique instance_ids.

Example 1: Data parallel load balanced, one model one pipeline two instances.
```
Node 1: dynamo-run in=dyn://qwen3-32b.backend.generate out=sglang /data/Qwen3-32B --tensor-parallel-size 2 --base-gpu-id 0
Node 2: dynamo-run in=dyn://qwen3-32b.backend.generate out=sglang /data/Qwen3-32B --tensor-parallel-size 2 --base-gpu-id 2
```

Example 2: Two models, two pipelines.
```
Node 1: dynamo-run in=dyn://qwen3-32b.backend.generate out=vllm /data/Qwen3-32B
Node 2: dynamo-run in=dyn://llama3-1-8b.backend.generate out=vllm /data/Llama-3.1-8B-Instruct/
```

Example 3: Different endpoints.

The KV metrics publisher in VLLM adds a `load_metrics` endpoint to the current component. If the `llama3-1-8b.backend` component above is using patched vllm it will also expose `llama3-1-8b.backend.load_metrics`.

Example 4: Multiple component in a pipeline.

In the P/D disaggregated setup you would have `deepseek-distill-llama8b.prefill.generate` (possibly multiple instances of this) and `deepseek-distill-llama8b.decode.generate`.

For output it is always only `out=dyn`. This tells Dynamo to auto-discover the instances, group them by model, and load balance appropriately (depending on `--router-mode` flag). The old syntax of `dyn://...` is still accepted for backwards compatibility.

### KV-aware routing

**Setup**

Currently, only patched vllm supports KV-aware routing.

To set up KV-aware routing on patched vllm:

1. Ensure that `etcd` and `nats` (see [Quickstart with pip and vllm](#quickstart-with-pip-and-vllm)) are running and accessible from all nodes.
1. Create a virtualenv: `uv venv kvtest` and source its `activate`.
1. Use `pip` to **either**:
   1. Install Dynamo's vllm branch:
      ```
      uv pip install ai-dynamo-vllm
      ```
       **or**
   1. Install upstream vllm 0.8.4:
      ```
      uv pip install vllm==0.8.4
      ```
      And then patch it:
      ```
      cd kvtest/lib/python3.12/site-packages
      patch -p1 < $REPO_ROOT/container/deps/vllm/vllm_v0.8.4-dynamo-kv-disagg-patch.patch
      ```
1. Build the C bindings:
   ```
   cd $REPO_ROOT/lib/bindings/c
   cargo build
   ```
1. Put the library you just built on library path:
   ```
   export LD_LIBRARY_PATH=$REPO_ROOT/target/debug/
   ```
If you patched locally (instead of installing `ai-dynamo-vllm`), edit vllm's `platforms/__init__.py` to undo a patch change:
```
    #vllm_version = version("ai_dynamo_vllm")
    vllm_version = version("vllm")
```

**Start the workers**

The workers are started normally:

```
dynamo-run in=dyn://dynamo.endpoint.generate out=vllm /data/llms/Qwen/Qwen3-4B
```

**Start the ingress node**

```
dynamo-run in=http out=dyn --router-mode kv
```

The only difference from the distributed system above is `--router-mode kv`. The patched vllm announces when a KV block is created or removed. The Dynamo router run finds the worker with the best match for those KV blocks and directs the traffic to that node.

For performance testing, compare a typical workload with `--router-mode random|round-robin` to see if it can benefit from KV-aware routing.

The KV-aware routing arguments:

- `--kv-overlap-score-weight`: Sets the amount of weighting on overlaps with prefix caches, which directly contributes to the prefill cost. A large weight is expected to yield a better TTFT (at the expense of worse ITL). When set to 0, prefix caches are not considered at all (falling back to pure load balancing behavior on the active blocks).

- `--router-temperature`: Sets the temperature when randomly selecting workers to route to via softmax sampling on the router cost logits. Setting it to 0 recovers the deterministic behavior where the min logit is picked.

- `--use-kv-events`: Sets whether to listen to KV events for maintaining the global view of cached blocks. If true, then we use the `KvIndexer` to listen to the block creation and deletion events. If false, `ApproxKvIndexer`, which assumes the kv cache of historical prompts exists for fixed time durations (hard-coded to 120s), is used to predict the kv cache hit ratio in each engine. Set false if your backend engine does not emit KV events.

### Request Migration

In a [Distributed System](#distributed-system), you can enable [request migration](../architecture/request_migration.md) to handle worker failures gracefully. Use the `--migration-limit` flag to specify how many times a request can be migrated to another worker:

```bash
dynamo-run in=dyn://... out=<engine> ... --migration-limit=3
```

This allows a request to be migrated up to 3 times before failing. See the [Request Migration Architecture](../architecture/request_migration.md) documentation for details on how this works.

## Full usage details

 The `dynamo-run` is also an example of what can be built in Rust with the `dynamo-llm` and `dynamo-runtime` crates. The following guide shows how to build from source with all the features.

### Getting Started

#### Setup

##### Step 1: Install libraries
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

## Check that Metal is accessible
xcrun -sdk macosx metal
```
If Metal is accessible, you should see an error like `metal: error: no input files`, which confirms it is installed correctly.

##### Step 2: Install Rust
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

##### Step 3: Build

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
--target-dir /path/to/target_directory # specify target_directory with write privileges
--manifest-path /path/to/project/Cargo.toml # if cargo build is run outside of `launch/` directory
```

The binary is called `dynamo-run` in `target/debug`
```
cd target/debug
```

Build with `--release` for a smaller binary and better performance, but longer build times. The binary will be in `target/release`.



#### Defaults
The input defaults to `in=text`. The output defaults to `out=mistralrs` engine, unless it is disabled with `--no-default-features` in which case vllm is used.
### Running Inference with Pre-built Engines

#### mistralrs

[mistral.rs](https://github.com/EricLBuehler/mistral.rs) is a pure Rust engine that is fast to run, fast to load, supports GGUF as well as safetensors, and runs well on CPU as well as GPU. For those reasons it is the default engine.

```
dynamo-run Qwen/Qwen3-4B
```

is equivalent to

```
dynamo-run in=text out=mistralrs Qwen/Qwen3-4B
```

If you have multiple GPUs, mistral.rs does automatic tensor parallelism. You do not need to pass any extra flags to dynamo-run to enable it.

#### llamacpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) is built for CPU by default. For an optimized build pass the appropriate feature flag (highly recommended):

```
cargo build --features cuda|metal|vulkan -p dynamo-run
```

For GNU OpenMP support add the `openmp` feature. On Ubuntu this requires `libgomp1` (part of `build-essential`) at build and runtime.

```
cargo build --features cuda,openmp -p dynamo-run
```

```
dynamo-run out=llamacpp ~/llms/gemma-3-1b-it-q4_0.gguf
dynamo-run out=llamacpp ~/llms/Qwen3-0.6B-Q8_0.gguf # From https://huggingface.co/ggml-org
```

Note that in some cases we are unable to extract the tokenizer from the GGUF, and so a Hugging Face checkout of a matching model must also be passed. Dynamo uses the weights from the GGUF and the pre-processor (`tokenizer.json`, etc) from the `--model-config`:
```
dynamo-run out=llamacpp ~/llms/Llama-4-Scout-17B-16E-Instruct-UD-IQ1_S.gguf --context-length 32768 --model-config ~/llms/Llama-4-Scout-17B-16E-Instruct
```

If you have multiple GPUs, llama.cpp does automatic tensor parallelism. You do not need to pass any extra flags to dynamo-run to enable it.

#### sglang

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

Any example above using `out=sglang` can work, but our sglang backend is also multi-gpu.

```
cd target/debug
./dynamo-run in=http out=sglang --model-path ~/llms/DeepSeek-R1-Distill-Llama-70B/ --tensor-parallel-size 8
```

To pass extra arguments to the sglang engine see [Extra engine arguments](#extra-engine-arguments).

**Multi-GPU**

Pass `--tensor-parallel-size <NUM-GPUS>` to `dynamo-run`.

```
dynamo-run out=sglang ~/llms/Llama-4-Scout-17B-16E-Instruct/ --tensor-parallel-size 8
```

To specify the GPU to start from, pass `--base-gpu-id <num>`; for example, on a shared eight GPU machine where GPUs 0–3 are already in use:
```
dynamo-run out=sglang <model> --tensor-parallel-size 4 --base-gpu-id 4
```

**Multinode:**

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

#### vllm

Using the [vllm](https://github.com/vllm-project/vllm) Python library. Slow startup, fast inference. Supports both safetensors from HF and GGUF files, but is very slow for GGUF - prefer llamacpp.

The vllm engine requires [etcd](https://etcd.io/) and [nats](https://nats.io/) with jetstream (`nats-server -js`) to be running.

We use [uv](https://docs.astral.sh/uv/) but any virtualenv manager should work.

1. Setup:
```
uv venv
source .venv/bin/activate
uv pip install pip
uv pip install vllm==0.8.4 setuptools
```

```{note}
If you're on Ubuntu 22.04 or earlier, you must add `--python=python3.10` to your `uv venv` command.
```

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

To pass extra arguments to the vllm engine see [Extra engine arguments](#extra-engine-arguments).

vllm attempts to allocate enough KV cache for the full context length at startup. If that does not fit in your available memory pass `--context-length <value>`.

If you see an error similar to the following:
```text
2025-06-28T00:32:32.507Z  WARN dynamo_run::subprocess: Traceback (most recent call last):
2025-06-28T00:32:32.507Z  WARN dynamo_run::subprocess:   File "/tmp/.tmpYeq5qA", line 29, in <module>
2025-06-28T00:32:32.507Z  WARN dynamo_run::subprocess:     from dynamo.llm import ModelType, WorkerMetricsPublisher, register_llm
2025-06-28T00:32:32.507Z  WARN dynamo_run::subprocess: ModuleNotFoundError: No module named 'dynamo'
```
Then run
```
uv pip install maturin
pip install patchelf
cd lib/bindings/python
maturin develop
```
this builds the Python->Rust bindings into that missing dynamo module. Rerun dynamo-run, the problem should be resolved.

**Multi-GPU**

Pass `--tensor-parallel-size <NUM-GPUS>` to `dynamo-run`.

To specify which GPUs to use set environment variable `CUDA_VISIBLE_DEVICES`.

**Multinode:**

vllm uses [ray](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#running-vllm-on-multiple-nodes) for pipeline parallel inference. Dynamo does not change or manage that.

Here is an example on two 8x nodes:
- Leader node: `ray start --head --port=6379`
- Each follower node: `ray start --address=<HEAD_NODE_IP>:6379`
- Leader node: `dynamo-run out=vllm ~/llms/DeepSeek-R1-Distill-Llama-70B/ --tensor-parallel-size 16`

The `--tensor-parallel-size` parameter is the total number of GPUs in the cluster. This is often constrained by a model dimension such as being a divisor of the number of attention heads.

Startup can be slow so you may want to `export DYN_LOG=debug` to see progress.

Shutdown: `ray stop`

#### trtllm

Using [TensorRT-LLM's LLM API](https://nvidia.github.io/TensorRT-LLM/llm-api/), a high-level Python API.

You can use `--extra-engine-args` to pass extra arguments to LLM API engine.

The trtllm engine requires [etcd](https://etcd.io/) and [nats](https://nats.io/) with jetstream (`nats-server -js`) to be running.

##### Step 1: Build the environment

See instructions [here](https://github.com/ai-dynamo/dynamo/blob/main/examples/tensorrt_llm/README.md#build-docker) to build the dynamo container with TensorRT-LLM.

##### Step 2: Run the environment

See instructions [here](https://github.com/ai-dynamo/dynamo/blob/main/examples/tensorrt_llm/README.md#run-container) to run the built environment.

##### Step 3: Execute `dynamo-run` command

Execute the following to load the TensorRT-LLM model specified in the configuration.
```
dynamo-run in=http out=trtllm TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

#### Echo Engines

Dynamo includes two echo engines for testing and debugging purposes:

##### echo_core

The `echo_core` engine accepts pre-processed requests and echoes the tokens back as the response. This is useful for testing pre-processing functionality as the response includes the full prompt template.

```
dynamo-run in=http out=echo_core --model-path <hf-repo-checkout>
```

Note that to use it with `in=http` you need to tell the post processor to ignore stop tokens from the template by adding `nvext.ignore_eos` like this:
```
curl -N -d '{"nvext": {"ignore_eos": true}, "stream": true, "model": "Qwen2.5-3B-Instruct", "max_completion_tokens": 4096, "messages":[{"role":"user", "content": "Tell me a story" }]}' ...
```

The default `in=text` sets that for you.

##### echo_full

The `echo_full` engine accepts un-processed requests and echoes the prompt back as the response.

```
dynamo-run in=http out=echo_full --model-name my_model
```

##### Configuration

Both echo engines use a configurable delay between tokens to simulate generation speed. You can adjust this using the `DYN_TOKEN_ECHO_DELAY_MS` environment variable:

```
# Set token echo delay to 1ms (1000 tokens per second)
DYN_TOKEN_ECHO_DELAY_MS=1 dynamo-run in=http out=echo_full
```

The default delay is 10ms, which produces approximately 100 tokens per second.

#### Batch mode

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

#### Mocker engine

The mocker engine is a mock vLLM implementation designed for testing and development purposes. It simulates realistic token generation timing without requiring actual model inference, making it useful for:

- Testing distributed system components without GPU resources
- Benchmarking infrastructure and networking overhead
- Developing and debugging Dynamo components
- Load testing and performance analysis

**Basic usage:**

The `--model-path` is required but can point to any valid model path - the mocker doesn't actually load the model weights (but the pre-processor needs the tokenizer). The arguments `block_size`, `num_gpu_blocks`, `max_num_seqs`, `max_num_batched_tokens`, `enable_prefix_caching`, and `enable_chunked_prefill` are common arguments shared with the real VLLM engine.

And below are arguments that are mocker-specific:
- `speedup_ratio`: Speed multiplier for token generation (default: 1.0). Higher values make the simulation engines run faster.
- `dp_size`: Number of data parallel workers to simulate (default: 1)
- `watermark`: KV cache watermark threshold as a fraction (default: 0.01). This argument also exists for the real VLLM engine but cannot be passed as an engine arg.

```bash
echo '{"speedup_ratio": 10.0}' > mocker_args.json
dynamo-run in=dyn://dynamo.mocker.generate out=mocker --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --extra-engine-args mocker_args.json
dynamo-run in=http out=dyn --router-mode kv
```

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

The tensorrtllm backend also supports passing any argument the engine accepts. However, in this case config should be a yaml file.

```
backend: pytorch
kv_cache_config:
  event_buffer_max_size: 1024
```

Pass it like this:
```
dynamo-run in=http out=trtllm TinyLlama/TinyLlama-1.1B-Chat-v1.0 --extra-engine-args trtllm_extra.yaml
```

### Writing your own engine in Python

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
    model_path = "Qwen/Qwen3-0.6B" # or "/data/models/Qwen3-0.6B"
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
- A HuggingFace repo ID, optionally prefixed with `hf://`. It is downloaded and cached locally.
- The path to a checkout of a HuggingFace repo - any folder containing safetensor files as well as `config.json`, `tokenizer.json` and `tokenizer_config.json`.
- The path to a GGUF file, if your engine supports that.

The `model_type` can be:
- ModelType.Backend. Dynamo handles pre-processing. Your `generate` method receives a `request` dict containing a `token_ids` array of int. It must return a dict also containing a `token_ids` array and an optional `finish_reason` string.
- ModelType.Chat. Your `generate` method receives a `request` and must return a response dict of type [OpenAI Chat Completion](https://platform.openai.com/docs/api-reference/chat). Your engine handles pre-processing.
- ModelType.Completion. Your `generate` method receives a `request` and must return a response dict of the older [Completions](https://platform.openai.com/docs/api-reference/completions). Your engine handles pre-processing.

`register_llm` can also take the following kwargs:
- `model_name`: The name to call the model. Your incoming HTTP requests model name must match this. Defaults to the hugging face repo name, the folder name, or the GGUF file name.
- `context_length`: Max model length in tokens. Defaults to the model's set max. Only set this if you need to reduce KV cache allocation to fit into VRAM.
- `kv_cache_block_size`: Size of a KV block for the engine, in tokens. Defaults to 16.
- `user_data`: Optional dictionary containing custom metadata for worker behavior (e.g., LoRA configuration). Defaults to None.

Here are some example engines:

- Backend:
    * [vllm](https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/python/examples/hello_world/server_vllm.py)
    * [sglang](https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/python/examples/hello_world/server_sglang.py)
- Chat:
    * [sglang](https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/python/examples/hello_world/server_sglang_tok.py)

More fully-featured Backend engines (used by `dynamo-run`):
- [vllm](https://github.com/ai-dynamo/dynamo/blob/main/launch/dynamo-run/src/subprocess/vllm_inc.py)
- [sglang](https://github.com/ai-dynamo/dynamo/blob/main/launch/dynamo-run/src/subprocess/sglang_inc.py)

### Debugging

`dynamo-run` and `dynamo-runtime` support [tokio-console](https://github.com/tokio-rs/console). Build with the feature to enable:
```
cargo build --features cuda,tokio-console -p dynamo-run
```

The listener uses the default tokio console port, and all interfaces (0.0.0.0).

