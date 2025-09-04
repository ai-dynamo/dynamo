# Chapter 1: Building Your First Pipeline

In this chapter, we'll build a simple chat completion pipeline from scratch using Dynamo's runtime. You'll learn the core patterns for service communication and request flow.

## What We're Building

A minimal pipeline that takes OpenAI-style chat requests and generates responses:

```
Client → Frontend → Processor → Engine
         (HTTP)     (Tokenizer) (SGLang)
```

**Flow:**
1. **Frontend** (`frontend.py`) - HTTP API that accepts chat requests
2. **Processor** (`processor.py`) - Tokenizes text and orchestrates the pipeline
3. **Engine** (`engine.py`) - Runs SGLang for LLM inference

## Key Dynamo Concepts

### 1. Worker Registration

Every component starts with the `@dynamo_worker` decorator:

```python
from dynamo.runtime import DistributedRuntime, dynamo_worker

@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Your component logic here
```

This connects your component to the Dynamo runtime (NATS + etcd).

### 2. Services vs Clients

**Services** expose endpoints that others can call:
```python
# Creating a service (in processor.py and engine.py)
component = runtime.namespace("inference").component("processor")
await component.create_service()  # Registers in etcd

endpoint = component.endpoint("process")
await endpoint.serve_endpoint(your_async_generator_function)
```

**Clients** discover and call those services:
```python
# Creating a client (in frontend.py and processor.py)
client = await runtime.namespace("inference").component("processor").endpoint("process").client()
response = await client.generate(request_data)
```

The runtime handles service discovery through etcd automatically.

### 3. Data Transport

When components communicate via NATS, your data gets wrapped in an `Annotated` object. Extract it with `.data()`:

```python
async for chunk in response_stream:
    data = chunk.data()  # Extract the actual dict

    if "content" in data:
        print(data["content"])
```

### 4. Streaming Patterns

Services expose **async generators** for streaming responses:

```python
# Service side (engine.py)
async def generate(self, request):
    yield {"token_ids": [1, 2, 3]}
    yield {"token_ids": [4, 5]}
    yield {"finish_reason": "stop"}  # Always signal completion

# Client side (processor.py)
async for chunk in engine_response:
    data = chunk.data()
    if "finish_reason" in data:
        break  # Stream complete
```

**Key pattern:** Always yield a completion signal like `{"finish_reason": "stop"}`.

## Implementation Walkthrough

### Frontend (`frontend.py`)

The frontend is the simplest - it only creates clients, doesn't expose services:

- Uses FastAPI for HTTP handling
- Creates a client to the processor: `runtime.namespace("inference").component("processor").endpoint("process").client()`
- Collects streaming responses into a single OpenAI-format response
- **No service registration** - just HTTP → NATS conversion

### Processor (`processor.py`)

The processor does the coordination work:

- **Exposes a service:** `component("processor")` with `endpoint("process")`
- **Creates a client:** Talks to the engine for inference
- **Handles tokenization:** Uses HuggingFace transformers
- **Accumulates tokens:** Collects all tokens from engine, then detokenizes once for efficiency

Key insight: The processor serves as the pipeline orchestrator, handling the text ↔ token conversion boundary.

### Engine (`engine.py`)

The engine handles pure inference:

- **Exposes a service:** `component("engine")` with `endpoint("generate")`
- **Uses SGLang:** For actual LLM inference
- **Streams incrementally:** Only yields new tokens, not all tokens each time
- **Tracks state:** `num_output_tokens_so_far` to compute incremental outputs

## Running the Pipeline

Because discovery is handled by Dynamo, once you start these 3 components up, you can easily kill one of the components, make a change, and restart it without it affeecting any other components. Note the use of `configure_dynamo_logging` to set the logging level for Dynamo. This allows python and rust logging to look the same. To enable debug logging, simply set the environment variable `DYN_LOG_LEVEL=debug` and then run any of the components. 

```bash
# Terminal 1: Start the engine
python engine.py --model-path Qwen/Qwen2.5-0.5B-Instruct

# Terminal 2: Start the processor
python processor.py --model Qwen/Qwen2.5-0.5B-Instruct

# Terminal 3: Start the frontend
python frontend.py

# Terminal 4: Test it
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 25
  }'
```

## Understanding how Dynamo's discovery mechanism.

Under the hood, Dynamo uses NATS for communication between components and ETCD for component registration and discovery. In a terminal, run the following command after you've started the components:

```bash
etcdctl get --prefix instances
```

This should give you something like this:
```bash
instances/inference/engine/generate:694d97d100c7925a
{
  "component": "engine",
  "endpoint": "generate",
  "namespace": "inference",
  "instance_id": 7587887871106191962,
  "transport": {
    "nats_tcp": "inference_engine.generate-694d97d100c7925a"
  }
}
instances/inference/processor/process:694d97d100c79282
{
  "component": "processor",
  "endpoint": "process",
  "namespace": "inference",
  "instance_id": 7587887871106192002,
  "transport": {
    "nats_tcp": "inference_processor.process-694d97d100c79282"
  }
}
```

Note how the component, endpoint, and namespace are all things that we specified in our component code! This is how Dynamo knows how to route requests to the correct component.

## Understanding the Flow

1. **HTTP Request** → Frontend receives OpenAI chat request
2. **NATS Call** → Frontend calls `processor.generate(request)`
3. **Tokenization** → Processor converts messages to token IDs
4. **NATS Call** → Processor calls `engine.generate(tokens)`
5. **SGLang Inference** → Engine streams tokens back
6. **Accumulation** → Processor collects all tokens
7. **Detokenization** → Processor converts tokens back to text
8. **HTTP Response** → Frontend returns OpenAI-format response

The magic is that steps 2, 4 happen over NATS automatically - components don't know about HTTP or network details.

## Key Patterns You'll Use Everywhere

1. **Extract data first:** `data = chunk.data()`
2. **Handle errors:** Check for `"error"` fields in responses
3. **Async Generators:** All client -> service communication is via async generators

## What's Next

In Chapter 2, we'll add a router component to load balance between multiple engines, and explore more sophisticated request routing patterns.

The foundation you've learned here - services, clients, streaming, and data transport - applies to all Dynamo pipelines regardless of complexity.