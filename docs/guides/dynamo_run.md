# Dynamo Run

`dynamo-run` is a Rust binary that lets you easily run a model, explore the Dynamo components, and demonstrates the Rust API. It supports the `mistral.rs` and `llama.cpp` engines. `mistralrs` is the default engine.

It is primarily for development and rapid prototyping. For production use we recommend the Python wrapped components, see the main project README.

## Basics

Usage: See `dynamo-run --help`

Example: `dynamo-run Qwen/Qwen3-0.6B`

Set the environment variable `DYN_LOG` to adjust the logging level; for example, `export DYN_LOG=debug`. It has the same syntax as `RUST_LOG`.

To adjust verbosity, use `-v` to enable debug logging or `-vv` to enable full trace logging. For example:

```bash
dynamo-run in=http out=mistralrs <model> -v  # enables debug logging
dynamo-run in=text out=llamacpp <model> -vv  # enables full trace logging
```

### Use model from Hugging Face

To automatically download Qwen3 4B from Hugging Face (16 GiB download) and to start it in interactive text mode:
```
dynamo-run Qwen/Qwen3-4B
```

The general format for HF download follows this pattern:
```
dynamo-run out=<engine> <HUGGING_FACE_ORGANIZATION/MODEL_NAME>
```

For gated models (such as meta-llama/Llama-3.2-3B-Instruct), you must set an `HF_TOKEN` environment variable.

The parameter can be the ID of a HuggingFace repository (which will be downloaded) or a folder containing safetensors, config.json, or similar (perhaps a locally checked out HuggingFace repository).

### Run a model from local file

To run a model from local file:
- Download the model from Hugging Face
- Run the model from local file

See the following sections for details.

#### Download model from Hugging Face
This model available from Hugging Face should be high quality and fast on almost any machine: https://huggingface.co/Qwen/Qwen3-0.6B

To run the model:

*Text interface*
```
dynamo-run Qwen/Qwen3-0.6B
```

You can also pipe a prompt into `dynamo-run`:
```
echo 'What is the capital of Tuvalu?' | dynamo-run Qwen/Qwen3-0.6B --context-length 4096
```

*HTTP interface*
```
dynamo-run in=http out=mistralrs Qwen/Qwen3-0.6B
```
You can also list models or send a request:

*List the models*
```
curl localhost:8080/v1/models
```

*Send a request*
```
curl -d '{"model": "Qwen/Qwen3-0.6B", "max_completion_tokens": 2049, "messages":[{"role":"user", "content": "What is the capital of South Africa?" }]}' -H 'Content-Type: application/json' http://localhost:8080/v1/chat/completions
```

## Distributed System

You can run the ingress side (HTTP server and pre-processing) on one machine, for example a CPU node, and the worker on a different machine (a GPU node).

You will need [etcd](https://etcd.io/) and [nats](https://nats.io) with jetstream installed and accessible from both nodes. For development I run NATS like this: `nats-server -js --trace --store_dir $(mktemp -d)`.

**Node 1:** OpenAI compliant HTTP server, optional pre-processing, worker discovery:

```
dynamo-run in=http out=auto
```

**Node 2:** Engine. Receives and returns requests over the network:

```
dynamo-run in=dyn://llama3B.backend.generate out=mistralrs ~/llms/Llama-3.2-3B-Instruct
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
Node 1: dynamo-run in=dyn://qwen3-32b.backend.generate /data/Qwen3-32B
Node 2: dynamo-run in=dyn://qwen3-32b.backend.generate /data/Qwen3-32B
```

Example 2: Two models, two pipelines.
```
Node 1: dynamo-run in=dyn://qwen3-32b.backend.generate /data/Qwen3-32B
Node 2: dynamo-run in=dyn://llama3-1-8b.backend.generate /data/Llama-3.1-8B-Instruct/
```

Example 3: Different endpoints.

The KV metrics publisher in VLLM adds a `load_metrics` endpoint to the current component. If the `llama3-1-8b.backend` component above is using patched vllm it will also expose `llama3-1-8b.backend.load_metrics`.

Example 4: Multiple component in a pipeline.

In the P/D disaggregated setup you would have `deepseek-distill-llama8b.prefill.generate` (possibly multiple instances of this) and `deepseek-distill-llama8b.decode.generate`.

For output it is always only `out=auto`. This tells Dynamo to auto-discover the instances, group them by model, and load balance appropriately (depending on `--router-mode` flag). The exception is static workers, see that section.

### KV-aware routing

```
dynamo-run in=http out=auto --router-mode kv
```

The only difference from the distributed system above is `--router-mode kv`. vllm announces when a KV block is created or removed. The Dynamo router finds the worker with the best match for those KV blocks and directs the traffic to that node.

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

### Request Cancellation

When using the HTTP interface (`in=http`), if the HTTP request connection is dropped by the client, Dynamo automatically cancels the downstream request to the worker. This ensures that computational resources are not wasted on generating responses that are no longer needed.

For detailed information about how request cancellation works across the system, see the [Request Cancellation Architecture](../architecture/request_cancellation.md) documentation.

## Development

`dynamo-run` is also an example of what can be built in Rust with the `dynamo-llm` and `dynamo-runtime` crates. The following guide shows how to build from source with all the features.

### Step 1: Install libraries
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

### Step 2: Install Rust
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Step 3: Build

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


## Engines

The input defaults to `in=text`. The output defaults to `out=mistralrs` engine, unless it is disabled with `--no-default-features` in which case an engine that echo's back your input is used.

### mistralrs

[mistral.rs](https://github.com/EricLBuehler/mistral.rs) is a pure Rust engine that is fast to run and fast to load, and runs well on CPU as well as GPU. For those reasons it is the default engine.

```
dynamo-run Qwen/Qwen3-4B
```

is equivalent to

```
dynamo-run in=text out=mistralrs Qwen/Qwen3-4B
```

If you have multiple GPUs, `mistral.rs` does automatic tensor parallelism. You do not need to pass any extra flags to dynamo-run to enable it.

### Mocker engine

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
dynamo-run in=http out=auto --router-mode kv
```

### echo

The `echo` engine echoes the prompt back as the response.

```
dynamo-run in=http out=echo --model-name my_model
```

The echo engine uses a configurable delay between tokens to simulate generation speed. You can adjust this using the `DYN_TOKEN_ECHO_DELAY_MS` environment variable:

```
# Set token echo delay to 1ms (1000 tokens per second)
DYN_TOKEN_ECHO_DELAY_MS=1 dynamo-run in=http out=echo
```

The default delay is 10ms, which produces approximately 100 tokens per second.

### Other engines, multi-node, production

`vllm`, `sglang` and `trtllm` production grade engines are available in `components/backends`. They run as Python components, using the Rust bindings. See the main README.

`dynamo-run` is an exploration, development and prototyping tool, as well as an example of using the Rust API. Multi-node and production setups should be using the main engine components.

## Batch mode

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

## Writing your own engine in Python

The [dynamo](https://pypi.org/project/ai-dynamo/) Python library allows you to build your own engine and attach it to Dynamo. All of the main backend components in `components/backends/` work like this.

The Python file must do three things:
1. Decorate a function to get the runtime
2. Register on the network
3. Attach a request handler

```
from dynamo.llm import ModelInput, ModelType, register_llm
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
    model_input = ModelInput.Tokens # or ModelInput.Text if engine handles pre-processing
    model_type = ModelType.Chat # or ModelType.Chat | ModelType.Completions if model can be deployed on chat and completions endpoints
    endpoint = component.endpoint("endpoint")
    # Optional last param to register_llm is model_name. If not present derives it from model_path
    await register_llm(model_input, model_type, endpoint, model_path)

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

The `model_input` can be:
- ModelInput.Tokens. Your engine expects pre-processed input (token IDs). Dynamo handles tokenization and pre-processing.
- ModelInput.Text. Your engine expects raw text input and handles its own tokenization and pre-processing.

The `model_type` can be:
- ModelType.Chat. Your `generate` method receives a `request` and must return a response dict of type [OpenAI Chat Completion](https://platform.openai.com/docs/api-reference/chat).
- ModelType.Completions. Your `generate` method receives a `request` and must return a response dict of the older [Completions](https://platform.openai.com/docs/api-reference/completions).

`register_llm` can also take the following kwargs:
- `model_name`: The name to call the model. Your incoming HTTP requests model name must match this. Defaults to the hugging face repo name, or the folder name.
- `context_length`: Max model length in tokens. Defaults to the model's set max. Only set this if you need to reduce KV cache allocation to fit into VRAM.
- `kv_cache_block_size`: Size of a KV block for the engine, in tokens. Defaults to 16.
- `user_data`: Optional dictionary containing custom metadata for worker behavior (e.g., LoRA configuration). Defaults to None.

Here are some example engines:

- Backend:
    * [vllm](https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/python/examples/hello_world/server_vllm.py)
    * [sglang](https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/python/examples/hello_world/server_sglang.py)
- Chat:
    * [sglang](https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/python/examples/hello_world/server_sglang_tok.py)

More fully-featured Python engines are in `components/backends`.

## Debugging

`dynamo-run` and `dynamo-runtime` support [tokio-console](https://github.com/tokio-rs/console). Build with the feature to enable:
```
cargo build --features cuda,tokio-console -p dynamo-run
```

The listener uses the default tokio console port, and all interfaces (0.0.0.0).

