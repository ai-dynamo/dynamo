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
- `model_name`: The name to call the model. Your incoming HTTP requests model name must match this. Defaults to the hugging face repo name or the folder name.
- `context_length`: Max model length in tokens. Defaults to the model's set max. Only set this if you need to reduce KV cache allocation to fit into VRAM.
- `kv_cache_block_size`: Size of a KV block for the engine, in tokens. Defaults to 16.
- `migration_limit`: Maximum number of times a request may be [migrated to another Instance](../architecture/request_migration.md). Defaults to 0.
- `user_data`: Optional dictionary containing custom metadata for worker behavior (e.g., LoRA configuration). Defaults to None.

See `components/backends` for full code examples.

## Component names

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

## Migrate Ongoing Requests

A Python worker may need to be shut down promptly, for example when the node running the worker is to be reclaimed and there isn't enough time to complete all ongoing requests before the shutdown deadline.

In such cases, you can signal incomplete responses by raising a `GeneratorExit` exception in your generate loop. This will immediately close the response stream, signaling to the frontend that the stream is incomplete. With request migration enabled (see the [`migration_limit`](../architecture/request_migration.md) parameter), the frontend will automatically migrate the partially completed request to another worker instance, if available, to be completed.

> [!WARNING]
> We will update the `GeneratorExit` exception to a new Dynamo exception. Please expect minor code breaking change in the near future.

Here's an example of how to implement this in your `RequestHandler`:

```python
class RequestHandler:

    async def generate(self, request):
        """Generate response, with support for request migration"""
        for result in self.engine.generate_streaming(request):
            # Check if we need to migrate before yielding each token
            if is_shutting_down():
                # Raising GeneratorExit closes the stream and triggers migration
                raise GeneratorExit("Worker shutting down, migrating request")

            yield result
```

When `GeneratorExit` is raised, the frontend receives the incomplete response and can seamlessly continue generation on another available worker instance, preserving the user experience even during worker shutdowns.

For more information about how request migration works, see the [Request Migration Architecture](../architecture/request_migration.md) documentation.

## Request Cancellation

Your Python worker's request handler can optionally support request cancellation by accepting a `context` argument after the `request` argument. This context object allows you to check for cancellation signals and respond appropriately:

```python
class RequestHandler:

    async def generate(self, request, context):
        """Generate response with cancellation support"""
        for result in self.engine.generate_streaming(request):
            # Check if the request has been cancelled
            if context.is_stopped():
                # Stop processing and clean up
                break

            yield result
```

The context parameter is optional - if your generate method doesn't include it in its signature, Dynamo will call your method without the context argument.

For detailed information about request cancellation, including async cancellation monitoring and context propagation patterns, see the [Request Cancellation Architecture](../architecture/request_cancellation.md) documentation.
