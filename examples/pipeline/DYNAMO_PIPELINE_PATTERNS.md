# Dynamo Pipeline Patterns

Dynamo's distributed runtime makes it easy to build arbitrary multi-stage pipelines. Any async Python function that yields values can become a distributed service endpoint.

## Serving a Stage

A stage is just a class with an async generator method:

```python
from dynamo._core import DistributedRuntime

class MyStage:
    async def generate(self, request, context):
        # Do work with request
        result = transform(request)
        yield result

# Register and serve the endpoint
runtime = DistributedRuntime(loop, "file", "tcp")
component = runtime.namespace("my_app").component("my_stage")
endpoint = component.endpoint("generate")

await endpoint.serve_endpoint(MyStage().generate)
```

That's it. Your function is now a distributed service.

## Calling a Stage

To call another stage, get a client and call the method:

```python
# Connect to another stage
endpoint = runtime.namespace("my_app").component("other_stage").endpoint("generate")
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
class MiddleStage:
    def __init__(self, runtime):
        self.runtime = runtime

    async def initialize(self):
        # Connect to the next stage
        endpoint = self.runtime.namespace("app").component("next").endpoint("generate")
        self.next_client = await endpoint.client()
        await self.next_client.wait_for_instances()

    async def generate(self, request, context):
        # Transform input
        transformed = do_something(request)

        # Call next stage, passing context through
        stream = await self.next_client.generate(transformed, context=context)
        async for response in stream:
            yield response.data()
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Namespace** | Logical grouping (e.g., `"my_app"`) |
| **Component** | A service within a namespace (e.g., `"stage1"`) |
| **Endpoint** | A callable method on a component (e.g., `"generate"`) |
| **Context** | Carries request metadata, enables cancellation |

## Why This Matters

- **No boilerplate**: Just write async generators
- **Automatic streaming**: `yield` becomes distributed streaming
- **Flexible topology**: Connect stages however you want
- **Context propagation**: Cancellation flows through the entire pipeline
- **Scale independently**: Each stage runs as its own generate
