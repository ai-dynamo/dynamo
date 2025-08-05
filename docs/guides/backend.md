<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Writing Python Workers in Dynamo

This guide explains how to create your own Python worker in Dynamo.

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
- `migration_limit`: Maximum number of times a request may be [migrated to another Instance](../architecture/request_migration.md). Defaults to 0.
- `user_data`: Optional dictionary containing custom metadata for worker behavior (e.g., LoRA configuration). Defaults to None.

See `components/backends` for full code examples.

### Component names

A worker needs three names to register itself: namespace.component.endpoint

* *Namespace*: A pipeline. Usually a model. e.g "llama_8b". Just a name.
* *Component*: A load balanced service needed to run that pipeline. "backend", "prefill", "decode", "preprocessor", "draft", etc. This typically has some configuration (which model to use, for example).
* *Endpoint*: Like a URL. "generate", "load_metrics".
* *Instance*: A process. Unique. Dynamo assigns each one a unique instance_id. The thing that is running is always an instance. Namespace/component/endpoint can refer to multiple instances.

If you run two models, that is two pipelines. An exception would be if doing speculative decoding. The draft model is part of the pipeline of a bigger model.

If you run two instances of the same model ("data parallel") they are the same namespace+component+endpoint but different instances. The router will spread traffic over all the instances of a namespace+component+endpoint. If you have four prefill workers in a pipeline, they all have the same namespace+component+endpoint and are automatically assigned unique instance_ids.

Example 1: Data parallel load balanced, one model one pipeline two instances.
```
Node 1: namespace: qwen3-32b, component: backend, endpoint: generate, model: /data/Qwen3-32B --tensor-parallel-size 2 --base-gpu-id 0
Node 2: namespace: qwen3-32b, component: backend, endpoint: generate model: /data/Qwen3-32B --tensor-parallel-size 2 --base-gpu-id 2
```

Example 2: Two models, two pipelines.
```
Node 1: namespace: qwen3-32b, component: backend, endpoint: generate, model: /data/Qwen3-32B
Node 2: namespace: llama3-1-8b, component: backend, endpoint: generat, model: /data/Llama-3.1-8B-Instruct/
```

Example 3: Different endpoints.

The KV metrics publisher in VLLM adds a `load_metrics` endpoint to the current component. If the `llama3-1-8b.backend` component above is using patched vllm it will also expose `llama3-1-8b.backend.load_metrics`.

Example 4: Multiple component in a pipeline.

In the P/D disaggregated setup you would have `deepseek-distill-llama8b.prefill.generate` (possibly multiple instances of this) and `deepseek-distill-llama8b.decode.generate`.

