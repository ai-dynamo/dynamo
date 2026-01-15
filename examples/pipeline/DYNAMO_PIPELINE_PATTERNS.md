# Dynamo Pipeline Patterns

Dynamo's distributed runtime makes it easy to build arbitrary multi-stage pipelines. Any async Python function that yields values can become a distributed service endpoint.

## Serving a Stage

A stage is just a class with an async generator method:

```python
from dynamo._core import DistributedRuntime

class Stage1:
    async def generate(self, request, context):
        # Do work with request
        result = transform(request)
        yield result

# Register and serve the endpoint
runtime = DistributedRuntime(loop, "file", "tcp")
endpoint = runtime.namespace("pipeline").component("stage1").endpoint("generate")

await endpoint.serve_endpoint(Stage1().generate)
```

That's it. Your function is now a distributed service.

## Calling a Stage

To call another stage, get a client and call the method:

```python
# Connect to stage1
endpoint = runtime.namespace("pipeline").component("stage1").endpoint("generate")
client = await endpoint.client()
await client.wait_for_instances()

# Call it and stream results
stream = await client.generate(data, context=context)
async for response in stream:
    result = response.data()
```

## Building a Pipeline

Combine these patterns to create pipelines of any depth:

```python
class Stage1:
    def __init__(self, runtime):
        self.runtime = runtime

    async def initialize(self):
        # Connect to stage2
        endpoint = self.runtime.namespace("pipeline").component("stage2").endpoint("generate")
        self.stage2_client = await endpoint.client()
        await self.stage2_client.wait_for_instances()

    async def generate(self, request, context):
        # Transform input
        transformed = do_something(request)

        # Call stage2, passing context through
        stream = await self.stage2_client.generate(transformed, context=context)
        async for response in stream:
            yield response.data()
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Namespace** | Logical grouping (e.g., `"pipeline"`) |
| **Component** | A service within a namespace (e.g., `"stage1"`) |
| **Endpoint** | A callable method on a component (e.g., `"generate"`) |
| **Context** | Carries request metadata, enables cancellation |

## Why This Matters

- **No boilerplate**: Just write async generators
- **Automatic streaming**: `yield` becomes distributed streaming
- **Flexible topology**: Connect stages however you want
- **Context propagation**: Cancellation flows through the entire pipeline
- **Scale independently**: Each stage runs as its own process

## Pipeline Flow

```
┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
│ Client │ ──── │ Stage1 │ ──── │ Stage2 │ ──── │ Stage3 │
└────────┘      └────────┘      └────────┘      └────────┘
     │               │               │               │
     │   request     │   request     │   request     │
     │ ───────────►  │ ───────────►  │ ───────────►  │
     │               │               │               │
     │               │               │    yield      │
     │    yield      │    yield      │ ◄───────────  │
     │ ◄───────────  │ ◄───────────  │               │
     │               │               │               │
└────────────────────────────────────────────────────────┘
                      context flows through
```

## Parallel Pipeline

Stages can also be called in parallel and their results combined:

```
                      ┌────────┐
                 ┌──► │ Stage1 │ ──┐
┌────────┐      │    └────────┘    │      ┌────────┐
│ Client │ ──── │                  │ ──── │ Stage3 │
└────────┘      │    ┌────────┐    │      └────────┘
                 └──► │ Stage2 │ ──┘
                      └────────┘
```

```python
class Stage3:
    def __init__(self, runtime):
        self.runtime = runtime

    async def initialize(self):
        # Connect to both stages
        endpoint1 = self.runtime.namespace("pipeline").component("stage1").endpoint("generate")
        endpoint2 = self.runtime.namespace("pipeline").component("stage2").endpoint("generate")
        self.stage1_client = await endpoint1.client()
        self.stage2_client = await endpoint2.client()
        await self.stage1_client.wait_for_instances()
        await self.stage2_client.wait_for_instances()

    async def generate(self, request, context):
        # Call stage1 and stage2 in parallel
        stream1 = await self.stage1_client.generate(request, context=context)
        stream2 = await self.stage2_client.generate(request, context=context)

        # Gather results from both
        result1 = None
        result2 = None
        async for response in stream1:
            result1 = response.data()
        async for response in stream2:
            result2 = response.data()

        # Return combined result
        yield {"stage1": result1, "stage2": result2}
```
