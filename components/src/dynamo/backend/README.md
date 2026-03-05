# Dynamo Backend Framework

Base classes for implementing LLM backend workers in Dynamo. Uses the **Template Method pattern** to standardize the worker lifecycle while letting each backend plug in framework-specific logic.

## Package Layout

```
dynamo/backend/
    __init__.py      # Public API
    base.py          # Backend ABC
    handler.py       # Handler ABC
    args.py          # DynamoRuntimeConfig, DynamoRuntimeArgGroup
```

A working example lives in `dynamo/example_backend/` — run it with:

```bash
python -m dynamo.example_backend --model <model_path>
```

## Public API (`dynamo.backend`)

| Export | Purpose |
|---|---|
| `Backend` | ABC with lifecycle orchestration (`run()`) |
| `Handler` | ABC for request handlers (`generate()`) |
| `DynamoRuntimeConfig` | Base config dataclass (namespace, model, runtime settings, CLI argument support). Backend-specific settings go on the `backend` attribute. |
| `DynamoRuntimeArgGroup` | Reusable argparse group for common CLI flags |

## Worker Lifecycle

`Backend.run()` orchestrates these steps:

```
 1. pre_runtime_setup()          # Hook: pre-runtime configuration
 2. setup_runtime()              # Create DistributedRuntime
 3. setup_component()            # Create namespace/component/endpoint
 4. engine_context():            # Context manager for engine lifecycle
    5. create_engine()           # ABSTRACT: create the LLM engine
    6. setup_metrics()           # Hook: metrics collection
    7. is_non_leader_node()      # Hook: early exit for non-leader nodes
    8. create_handler()          # ABSTRACT: create request handler
    9. setup_kv_publishers()     # Hook: KV event publisher setup
   10. register_engine_routes()  # Hook: engine-specific routes
   11. register_and_serve()      # Register model + serve endpoint
   12. cleanup                   # Cancel metrics, handler cleanup
```

## Writing a New Backend

### 1. Directory Structure

Each backend is a standalone Python package under `dynamo/`:

```
dynamo/my_backend/
    __init__.py
    __main__.py     # Entry point: python -m dynamo.my_backend
    args.py         # Config + CLI argument parsing
    backend.py      # Backend subclass
    handlers.py     # Handler subclass
```

### 2. Configuration (`args.py`)

Inherit from `DynamoRuntimeConfig` and use `DynamoRuntimeArgGroup` for common CLI flags. Backend-specific fields go into a separate config object stored on `config.backend` so that standard Dynamo fields (`config.model`, `config.namespace`, …) are clearly separated from backend-specific ones (`config.backend.gpu_memory_fraction`):

```python
from dynamo.backend import DynamoRuntimeConfig, DynamoRuntimeArgGroup
from dynamo.common.configuration import ArgGroup

class MyBackendConfig:
    """Backend-specific settings — accessed via config.backend.<field>."""
    def __init__(self, gpu_memory_fraction: float = 0.9, max_batch_size: int = 256):
        self.gpu_memory_fraction = gpu_memory_fraction
        self.max_batch_size = max_batch_size

class Config(DynamoRuntimeConfig):
    pass

class MyBackendArgGroup(ArgGroup):
    def add_arguments(self, parser) -> None:
        group = parser.add_argument_group("MyBackend Options")
        group.add_argument("--gpu-memory-fraction", type=float, default=0.9)
        group.add_argument("--max-batch-size", type=int, default=256)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    DynamoRuntimeArgGroup().add_arguments(parser)
    MyBackendArgGroup().add_arguments(parser)
    args = parser.parse_args()
    config = Config.from_cli_args(args)
    config.backend = MyBackendConfig(
        gpu_memory_fraction=args.gpu_memory_fraction,
        max_batch_size=args.max_batch_size,
    )
    return config
```

With this pattern, any code that reads `config.backend.gpu_memory_fraction` is immediately recognizable as backend-specific, while `config.model` is clearly a standard Dynamo field.

### 3. Backend (`backend.py`)

Subclass `Backend` and implement the three abstract methods:

```python
from dynamo.backend import Backend

class MyBackend(Backend):
    async def create_engine(self):
        return my_framework.Engine(model=self.config.model)

    def create_handler(self, engine, component, endpoint):
        return MyHandler(engine=engine, component=component,
                         shutdown_event=self.shutdown_event)

    def get_health_check_payload(self, engine):
        return {"token_ids": [1], "sampling_options": {"temperature": 0.0},
                "stop_conditions": {"max_tokens": 1}}
```

#### Abstract Methods

| Method | Purpose |
|---|---|
| `create_engine()` | Create and return the LLM engine |
| `create_handler(engine, component, endpoint)` | Create and return a `Handler` subclass |
| `get_health_check_payload(engine)` | Return health check config dict |

#### Hooks (override as needed)

| Hook | When | Example Use |
|---|---|---|
| `pre_runtime_setup()` | Before runtime creation | Environment setup |
| `engine_context()` | Wraps engine lifecycle | Engines needing `async with` |
| `setup_metrics(endpoint)` | After engine creation | Prometheus metrics |
| `is_non_leader_node()` | After metrics setup | Multi-node: `rank > 0` |
| `handle_non_leader_node()` | If non-leader detected | Custom non-leader behavior |
| `setup_kv_publishers()` | After handler creation | KV event publishers |
| `register_engine_routes()` | After KV publishers | Extra RPC routes |
| `extract_runtime_config(engine)` | During model registration | Advertise capacity to frontend |
| `register_and_serve(handler, endpoint, engine)` | Serve phase | Custom serve pattern |
| `post_serve_cleanup()` | After serve exits | Sync cleanup |
| `async_post_serve_cleanup()` | After serve exits | Async cleanup |
| `_get_input_type()` | Model registration | `ModelInput.Text` for framework tokenizer |

### 4. Handler (`handlers.py`)

Subclass `Handler` and implement `generate()` as an async generator:

```python
from dynamo.backend import Handler

class MyHandler(Handler):
    def __init__(self, engine, component, shutdown_event=None):
        super().__init__(component=component, shutdown_event=shutdown_event)
        self.engine = engine

    async def generate(self, request, context):
        input_ids = request.get("input_ids", [])
        params = request.get("sampling_params", {})

        async with self._cancellation_monitor(
            context, abort_callback=lambda: self.engine.abort(context.id())
        ):
            num_output_tokens = 0
            async for output in self.engine.generate(input_ids, **params):
                result, num_output_tokens = self.process_generation_output(
                    output, num_output_tokens
                )
                yield result

    def cleanup(self):
        super().cleanup()
        self.engine.shutdown()
```

Key patterns:

- **Cancellation**: `self._cancellation_monitor(context, abort_callback)` handles client disconnects and server shutdown.
- **Output processing**: `self.process_generation_output(output, n)` extracts token deltas, logprobs, and finish reasons from cumulative engine outputs.
- **Tracing**: `self.get_trace_header(context)` returns a W3C `traceparent` header.
- **Cleanup**: Always call `super().cleanup()` to clean up KV publishers and temp directories.

### 5. Entry Point (`__main__.py`)

Each backend provides its own `__main__.py` so it can be launched with `python -m dynamo.<backend_name>`:

```python
import os

async def _run():
    from dynamo.my_backend.args import parse_args
    from dynamo.my_backend.backend import MyBackend

    config = parse_args()

    model_path = getattr(config, "model", None)
    if not config.served_model_name and model_path:
        config.served_model_name = model_path

    if model_path and not os.path.exists(model_path):
        from dynamo.llm import fetch_llm
        await fetch_llm(model_path)
    elif not model_path:
        raise ValueError("Please specify a model or model path using --model.")

    await MyBackend(config=config).run()

def main():
    import uvloop
    from dynamo.runtime.logging import configure_dynamo_logging
    configure_dynamo_logging()
    uvloop.run(_run())

if __name__ == "__main__":
    main()
```

## Common Patterns

### Disaggregated Serving (Prefill/Decode Split)

Pass `--disaggregation-mode prefill` or `--disaggregation-mode decode` via the CLI to declare the worker's role. Prefill mode OR's `ModelType.Prefill` into the endpoint types (e.g., `Chat | Prefill`) and disables tool/reasoning parsers. The default is `aggregated` (normal operation).

### Multi-Node Deployments

Override `is_non_leader_node()` to return `True` for non-rank-0 nodes. The default `handle_non_leader_node()` waits indefinitely (terminated via signals). Override it if non-leader nodes need metrics or KV publishing.

### KV Event Publishing

Set up publishers in `setup_kv_publishers()` and assign them to `handler.kv_publishers`. `Handler.cleanup()` shuts them down automatically.

### Framework Tokenizer Mode

If your engine handles tokenization internally, override `_get_input_type()` to return `ModelInput.Text`. This tells the frontend to send raw text instead of pre-tokenized input.
