# Writing Python Workers in Dynamo

This guide explains how to create your own Python worker in Dynamo and deploy
it via `dynamo serve` or `dynamo deploy`, covering basic concepts as well as
advanced features like enabling KV routing and disaggregated serving.

For detailed information about `dynamo serve` infrastructure, see the
[Dynamo SDK Docs](deploy/dynamo/sdk/docs/README.md).

For a guide that walks through how to launch a vLLM-based worker with
implementation of Disaggregated Serving and KV-Aware Routing included,
see the [Dynamo Serve Guide](docs/guides/dynamo_serve.md).

## Basic Concepts

When deploying a python-based worker with `dynamo serve` or `dynamo deploy`, it is
a Python class based definition that requires a few key decorators to get going:
- `@service`: used to define a worker class
- `@dynamo_endpoint`: marks methods that can be called by other workers or clients

For more detailed information on these concepts, see the
[Dynamo SDK Docs](deploy/dynamo/sdk/docs/README.md).

### Worker Skeleton

Here is the rough outline of what a worker may look like in its simplest form:

```python
from dynamo.sdk import DYNAMO_IMAGE, dynamo_endpoint, service

@service(
    dynamo={
        "enabled": True,
        "namespace": "your_namespace",
    },
    image=DYNAMO_IMAGE,
)
class YourWorker:
    # Worker implementation
    # ...

    @dynamo_endpoint()
    async def your_endpoint(self, request: RequestType) -> AsyncIterator[ResponseType]:
        # Endpoint Implementation
        pass
```

Workers in Dynamo are identified by a `namespace/component/endpoint` naming schema.
When addressing this worker's endpoint with the `namespace/component/endpoint` schema
based on the definitions above, it would be: `your_namespace/YourWorker/your_endpoint`:
- `namespace="your_namespace"`: Defined in the `@service` decorator
- `component="YourWorker"`: Defined by the Python Class name
- `endpoint="your_endpoint"`: Defined by the `@dynamo_endpoint` decorator, or by default the name of the function being decorated.

For more details about service configuration, resource management, and dynamo endpoints,
see the [Dynamo SDK Docs](deploy/dynamo/sdk/docs/README.md).

### Request/Response Types

Request/Response types of endpoints can be defined arbitraily for your use case's needs, as long as
the client calling your worker matches the expectations.

Define your request and response types using Pydantic models:

```python
from pydantic import BaseModel

class RequestType(BaseModel):
    text: str
    # Add other fields as needed

class ResponseType(BaseModel):
    text: str
    # Add other fields as needed
```

For example, if putting your worker directly behind an OpenAI `http` service via `llmctl`,
you could define the Request/Response types to be Chat Completions objects, such as:
```python
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

class YourLLMWorker:
    @dynamo_endpoint(name="my_chat_completions_endpoint")
    async def generate(self, request: ChatCompletionRequest):
        # Endpoint Implementation
        pass
```

## Basic Worker Example

Here's a simple example of a worker that takes text in and streams text out
via custom RequestType/ResponseType definitions:

```python
import logging
from pydantic import BaseModel
from dynamo.sdk import DYNAMO_IMAGE, dynamo_endpoint, service

logger = logging.getLogger(__name__)

class RequestType(BaseModel):
    text: str

class ResponseType(BaseModel):
    text: str

@service(
    dynamo={
        "enabled": True,
        "namespace": "your_namespace",
    },
    image=DYNAMO_IMAGE,
)
class YourWorker:
    def __init__(self) -> None:
        logger.info("Starting worker...")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Generate tokens and stream them back"""
        logger.info(f"Worker endpoint received: {req.text}")
        text = f"{req.text}"
        for token in text.split():
            yield f"Backend: {token}"
```

To see a minimal worker example like the above used in a larger pipeline of
components, see the `dynamo serve`
[Hello World example](examples/hello_world).

### Client Example

Here's a simple example of a client that directly calls the example
worker above through Dynamo without any intermediate services:

```python
import asyncio
from pydantic import BaseModel
from dynamo.sdk import get_runtime

# These could also be imported from a shared file/definition
class RequestType(BaseModel):
    text: str

class ResponseType(BaseModel):
    text: str

async def call_worker():
    # Get the runtime
    runtime = await get_runtime()

    # Get a client to the worker endpoint
    client = await runtime.namespace("your_namespace").component("YourWorker").endpoint("generate").client()

    # Create a request
    request = RequestType(text="Hello, Dynamo!")

    # Call the dynamo endpoint exposed by the worker
    responses = await client.generate(request)
    for response in responses:
        print(f"Response: {response.text}")

if __name__ == "__main__":
    asyncio.run(call_worker())
```

If putting a worker defined to handle OpenAI objects like ChatCompletions
directly behind an OpenAI `http` service via `llmctl`, you could instead use
an OpenAI-based client (or `curl`) that communicates with the OpenAI HTTP Service
and internally routes the requests to the worker(s) instead.

In more advanced scenarios where your worker may operate on some other intermediate format
that may not directly match an OpenAI-like format, you could setup a separate processor worker
that does something like the following:
- Take in OpenAI Chat Completions requests from the HTTP service
- Convert requests from Chat Completions format to the RequestType format your worker expects
- Forward requests to the worker(s)
- Convert responses from the worker's ResponseType back into Chat Completions response format
- Forward responses back to client

This advanced scenario of a separate OpenAI Processor worker is demonstrated in this
[vLLM example](examples/llm/README.md).

For a more minimal example of deploying a pipeline of components with a custom
API that your client can communicate with, see the
[Hello World example](examples/hello_world/README.md).

## Advanced Features

### KV Routing for LLMs

KV-aware routing is a powerful feature of Dynamo that optimizes for routing
requests to specific workers while minimizing a specific KV-cache based cost function.

In its simplest form, all a worker needs to do to enable KV-aware routing is to
publish KV metrics for Dynamo's KV Router to consume:
```python
from dynamo.llm import KvMetricsPublisher

class YourWorker:
    def __init__(self):
        # Initialize metrics publisher from Dynamo
        self.metrics_publisher = KvMetricsPublisher()

        # (Optional) Initialize some metrics for the worker/class to track
        self.request_active_slots = 0

        ###
        ###  TODO: Verify this code, see if async_init/async_on_start needed
        ###

        # Send some dummy metrics at initialization time
        self.metrics_publisher.publish(
            0,     # request_active_slots
            1024,  # request_total_slots
            0,     # kv_active_blocks
            1024,  # kv_total_blocks
            0,     # num_requests_waiting
            0.0,   # gpu_cache_usage_perc
            0.0,   # gpu_prefix_cache_hit_rate
        )

        # (Optional) To expose a metrics endpoint on this component for
        # querying or visualizing the metrics.
        self.metrics_publisher.create_endpoint("YourWorker")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Generate tokens, update KV Cache metrics, and stream the tokens back"""
        # Increment the number of active requests on receiving one
        self.request_active_slots += 1

        self.metrics_publisher.publish(
            self.request_active_slots, # request_active_slots
            1024,  # request_total_slots
            0,     # kv_active_blocks
            1024,  # kv_total_blocks
            0,     # num_requests_waiting
            0.0,   # gpu_cache_usage_perc
            0.0,   # gpu_prefix_cache_hit_rate
        )
```

The granularity at which metrics are published is up to the backend/worker implementation.
For example, you may want to update metrics on every single generation step during token
generation, or you may only want to update once per request, depending on your use case.
Assuming long generation time or long output token sequence lengths, it would be more
accurate to publish metrics at every generation step.

For more details, see the [KV Cache Routing Guide](docs/kv_cache_routing.md).

### Disaggregated Serving

- TODO: Code snippets about core NIXL concepts for P/D disagg (@ptarasiewiczNV)

For more information on Disaggregated Serving, see the
[general guide](docs/disagg_serving.md) and [performance tuning guide](docs/guides/disagg_perf_tuning.md).

## Best Practices

1. **Resource Management**: Configure resource requirements based on your needs:
   ```python
   @service(
       resources={
           "cpu": "10",
           "memory": "20Gi",
           "gpu": "1",
       }
   )
   ```

2. **Async Operations**: Use async/await for I/O operations:
   ```python
   @dynamo_endpoint()
   async def generate(self, request):
       # Use async operations for better performance
       result = await self.some_async_operation()
   ```

## Additional Resources

- Check the [examples](examples/) directory for more detailed implementations
- Refer to the [Dynamo SDK Docs](deploy/dynamo/sdk/docs/README.md) for API details.
- For Disaggregated Serving, see the [general guide](docs/disagg_serving.md) and [performance tuning guide](docs/guides/disagg_perf_tuning.md).