# Dynamo Backend Framework

This module provides the base classes for implementing LLM backend workers in Dynamo. It uses the **Template Method pattern** to standardize the worker lifecycle while allowing each backend to plug in framework-specific logic.

## Architecture

```
dynamo.common.backend/
    __init__.py          # Public API re-exports
    base.py              # BaseBackend (lifecycle) + BackendConfig (config base)
    handler.py           # BaseHandler (request handling interface)
    args.py              # BackendCommonConfig, BackendCommonArgGroup, WorkerModeConfig
```

A minimal sample backend lives in `dynamo/worker/myengine/`. It echoes the request's input tokens repeated 5 times with no hook overrides — useful as a starting point for new backends.

## Core Classes

| Class | Purpose |
|---|---|
| `BaseBackend` | Abstract base with lifecycle orchestration (`run()` method) |
| `BaseHandler` | Abstract base for request handlers (`generate()` method) |
| `BackendConfig` | Base config with common fields (namespace, model, runtime settings) |
| `BackendCommonConfig` | Extended config used with `BackendCommonArgGroup` for CLI parsing |
| `BackendCommonArgGroup` | Reusable argparse argument group for common CLI flags |
| `WorkerModeConfig` | Config for worker mode flags (prefill, decode, multimodal, embedding) |
| `WorkerModeArgGroup` | Argparse group for worker mode flags |

## Worker Lifecycle

`BaseBackend.run()` orchestrates these steps in order:

```
 1. pre_runtime_setup()          # Hook: pre-runtime configuration
 2. setup_runtime()              # Create DistributedRuntime
 3. setup_component()            # Create namespace/component/endpoint
 4. engine_context():            # Context manager for engine lifecycle
    5. create_engine()           # ABSTRACT: create the LLM engine
    6. setup_metrics()           # Hook: metrics collection
    7. is_non_leader_node()      # Hook: check + early exit for non-leader nodes
    8. create_handler()          # ABSTRACT: create request handler
    9. setup_kv_publishers()     # Hook: KV event publisher setup
   10. register_engine_routes()  # Hook: engine-specific routes
   11. register_and_serve()      # Register model then serve endpoint
   12. cleanup                   # Cancel metrics, handler cleanup, post_serve_cleanup()
```

Subclasses **must** implement the abstract methods and **may** override any hook.

## Writing a New Backend

### 1. Directory Structure

Create a new directory under `dynamo/worker/`:

```
dynamo/worker/myengine/
    __init__.py
    args.py             # Config class + CLI argument parsing
    backends.py         # Backend class extending BaseBackend
    handlers.py         # Handler class(es) extending BaseHandler
```

> **Tip:** See `dynamo/worker/myengine/` for a working minimal example that implements only the required abstract methods with no hook overrides.

### 2. Configuration (`args.py`)

Define your config and CLI parsing. The simplest approach is to inherit from `BackendCommonConfig` and use `BackendCommonArgGroup`:

```python
from dynamo.common.backend import BackendCommonConfig, BackendCommonArgGroup
from dynamo.common.configuration.arg_group import ArgGroup

class MyEngineConfig:
    """Engine-specific fields."""
    gpu_memory_fraction: float = 0.9
    max_batch_size: int = 256

class Config(BackendCommonConfig, MyEngineConfig):
    """Full config combining common + engine-specific fields."""
    pass

class MyEngineArgGroup(ArgGroup):
    def add_arguments(self, parser) -> None:
        group = parser.add_argument_group("MyEngine Options")
        group.add_argument("--gpu-memory-fraction", type=float, default=0.9)
        group.add_argument("--max-batch-size", type=int, default=256)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    BackendCommonArgGroup().add_arguments(parser)
    MyEngineArgGroup().add_arguments(parser)
    args = parser.parse_args()
    return Config.from_args(args)
```

### 3. Backend (`backends.py`)

Implement the three abstract methods. Override hooks as needed.

```python
from dynamo.common.backend import BaseBackend

class MyEngineBackend(BaseBackend):
    # ── Required: abstract methods ──────────────────────────

    async def create_engine(self):
        """Create and return the LLM engine instance."""
        import myengine
        return myengine.Engine(
            model=self.config.model,
            gpu_memory_fraction=self.config.gpu_memory_fraction,
        )

    def create_handler(self, engine, component, endpoint):
        """Create and return the request handler."""
        return MyEngineHandler(
            engine=engine,
            component=component,
            shutdown_event=self.shutdown_event,
        )

    def get_health_check_payload(self, engine):
        """Return a HealthCheckPayload for liveness probes."""
        return MyEngineHealthCheckPayload(engine).to_dict()

    # ── Optional: hook overrides ────────────────────────────

    def extract_runtime_config(self, engine):
        """Extract runtime config so the frontend knows worker capacity."""
        from dynamo.llm import ModelRuntimeConfig
        return ModelRuntimeConfig(
            max_num_seqs=self.config.max_batch_size,
            max_num_batched_tokens=engine.max_num_batched_tokens,
        )
```

#### Abstract Methods (must implement)

| Method | Purpose |
|---|---|
| `create_engine()` | Create and return the framework-specific LLM engine |
| `create_handler(engine, component, endpoint)` | Create and return a `BaseHandler` subclass |
| `get_health_check_payload(engine)` | Return health check config dict for liveness probes |

#### Hook Methods (override as needed)

| Hook | When | Example Use |
|---|---|---|
| `pre_runtime_setup()` | Before runtime creation | Environment setup, global config |
| `engine_context()` | Wraps engine lifecycle | Engines requiring `async with` context management |
| `setup_metrics(endpoint)` | After engine creation | Register Prometheus metrics callbacks |
| `is_non_leader_node()` | After metrics setup | Return `True` if `rank > 0` in multi-node |
| `handle_non_leader_node()` | If non-leader detected | Custom behavior for non-leader nodes |
| `setup_kv_publishers()` | After handler creation | Set up KV event publishers for prefix caching |
| `register_engine_routes()` | After KV publishers | Register extra RPC routes on the runtime |
| `extract_runtime_config(engine)` | During model registration | Tell the frontend about max_seqs, block_size, etc. |
| `register_and_serve(handler, endpoint, engine)` | Serve phase | Custom serve pattern (e.g., concurrent gather) |
| `post_serve_cleanup()` | After serve exits | Sync cleanup |
| `async_post_serve_cleanup()` | After serve exits | Async cleanup (e.g., deferred handler teardown) |
| `_is_prefill()` / `_is_decode()` | During model registration | Declare disaggregation mode |
| `_get_input_type()` | During model registration | Return `ModelInput.Text` if engine handles tokenization |

#### Custom Lifecycle Ordering

The default `run()` order works for most backends. If your engine needs a different sequence (e.g., creating the handler before metrics, or warming up the engine before creating the handler), override `run()` and call the internal `_run_*` helpers directly:

```python
async def run(self):
    self.pre_runtime_setup()
    self.setup_runtime()
    component, endpoint = self.setup_component()

    async with self.engine_context() as engine:
        self.engine = engine

        # Custom: create handler before metrics (needed for KV publishers)
        self.handler = self._run_create_handler(engine, component, endpoint)
        await self._run_setup_metrics(endpoint)

        if await self._run_handle_non_leader():
            await self._run_cleanup()
            return

        self._run_setup_kv_publishers()
        self._run_register_engine_routes(self.runtime, self.handler)

        try:
            await self._run_serve(self.handler, endpoint, engine)
        finally:
            await self._run_cleanup()
```

### 4. Handler (`handlers.py`)

Implement `generate()` as an async generator that yields response chunks.

Here is the simplest possible handler (from `dynamo/worker/myengine/handlers.py`) — it echoes input tokens 5 times, one token per yield:

```python
from dynamo.common.backend import BaseHandler

_REPEAT_COUNT = 5

class MyEngineHandler(BaseHandler):
    """Echoes the request's input tokens repeated 5 times, one token at a time."""

    async def generate(self, request, context):
        input_ids = request.get("input_ids", [])
        output_ids = input_ids * _REPEAT_COUNT
        total = len(output_ids)

        for i, token_id in enumerate(output_ids):
            out = {"token_ids": [token_id]}
            if i == total - 1:
                out["finish_reason"] = "stop"
            yield out
```

A more realistic handler would wrap a real engine and use cancellation monitoring:

```python
class RealEngineHandler(BaseHandler):
    def __init__(self, engine, component, shutdown_event=None):
        super().__init__(component=component, shutdown_event=shutdown_event)
        self.engine = engine

    async def generate(self, request, context):
        input_ids = request.get("input_ids", [])
        params = request.get("sampling_params", {})

        async with self._cancellation_monitor(
            context, abort_callback=lambda: self.engine.abort(context.request_id)
        ):
            num_output_tokens = 0
            async for output in self.engine.generate(input_ids, **params):
                result, num_output_tokens = self.process_generation_output(
                    output, num_output_tokens
                )
                yield result

    def cleanup(self):
        """Clean up engine resources."""
        super().cleanup()
        self.engine.shutdown()
```

Key handler patterns:

- **Cancellation monitoring**: Use `self._cancellation_monitor(context, abort_callback)` to handle client disconnects and server shutdown.
- **Output processing**: Use the static `process_generation_output()` helper for engines that produce cumulative outputs with `.token_ids`, `.logprobs`, `.finish_reason`, and `.stop_reason` attributes.
- **Tracing**: Use `self.get_trace_header(context)` to get a W3C `traceparent` header for distributed tracing.
- **Cleanup**: Always call `super().cleanup()` to clean up KV publishers and temp directories.

### 5. Health Check (`health_check.py`)

Subclass `HealthCheckPayload` and set `self.default_payload` before calling `super().__init__()`:

```python
from dynamo.health_check import HealthCheckPayload

class MyEngineHealthCheckPayload(HealthCheckPayload):
    def __init__(self, engine=None):
        self.default_payload = {
            "token_ids": [1],  # BOS token
            "sampling_options": {"temperature": 0.0},
            "stop_conditions": {
                "max_tokens": 1,
                "stop": None,
                "stop_token_ids": None,
                "include_stop_str_in_output": False,
                "ignore_eos": False,
            },
        }
        super().__init__()
```

Users can override the payload at runtime with the `DYN_HEALTH_CHECK_PAYLOAD` environment variable.

### 6. Entry Point Integration

Register your backend in `dynamo/worker/main.py`:

```python
elif engine_type == "myengine":
    from dynamo.worker.myengine_v2.args import parse_args
    from dynamo.worker.myengine_v2.backends import MyEngineBackend

    config = parse_args()
    backend_cls = MyEngineBackend
```

And add `"myengine"` to `SUPPORTED_ENGINES`.

## Common Patterns

### Disaggregated Serving (Prefill/Decode Split)

Override `_is_prefill()` and `_is_decode()` to declare the worker's role. Use `WorkerModeConfig` / `WorkerModeArgGroup` for the CLI flags. The base `register_model()` method uses these to set `ModelType.Prefill` and disable tool/reasoning parsers for prefill workers.

### Multi-Node Deployments

Override `is_non_leader_node()` to return `True` for non-rank-0 nodes. The default `handle_non_leader_node()` waits indefinitely (the process is terminated via signals). Override it if non-leader nodes need to run metrics or KV publishing.

### KV Event Publishing

Set up publishers in `setup_kv_publishers()` and assign them to `handler.kv_publishers`. The base `BaseHandler.cleanup()` will shut them down automatically.

### Framework Tokenizer Mode

If your engine handles tokenization internally (text input instead of token IDs), override `_get_input_type()` to return `ModelInput.Text`. This tells the frontend to send raw text instead of pre-tokenized input.
