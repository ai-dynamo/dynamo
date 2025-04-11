# Writing Python Workers in Dynamo

This guide explains how to create your own Python worker in Dynamo and deploy
it via `dynamo serve` or `dynamo deploy`, covering basic concepts and advanced
features like KV routing and disaggregated inference stages.

For detailed information about Dynamo's serving infrastructure, see the
[Dynamo SDK Docs](deploy/dynamo/sdk/docs/README.md).

For a guide that walks through how to launch a vLLM-based worker with
advanced features like Disaggregated Serving and KV-Aware Routing, see the
[Dynamo Serve Guide](docs/guides/dynamo_serve.md).

## Basic Concepts

When deploying a python-based worker with `dynamo serve` or `dynamo deploy`, it is
a Python class based definition that requires a few key decorators to get going:
- `@service`: used to define a worker class
- `@dynamo_endpoint`: marks methods that can be called by other workers or clients

Additionally, there are some environment variables to be aware of as well:
- `DYNAMO_IMAGE`: For container-based deployments, this should be set to a docker image
  containing `dynamo` and the necessary backends. Usually, this would be set to the
  resulting image built from running `./container/build.sh ...`

For more detailed information on these concepts, see the
[Dynamo SDK Docs](deploy/dynamo/sdk/docs/README.md).

### Worker Skeleton

Here is the rough outline of what a worker may look like in its simplest form:

```python
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

@dynamo_endpoint(name="my_chat_completions_endpoint")
async def generate(self, request: ChatCompletionRequest):
    # Implementation
    # ...
```

## Basic Worker Example

Here's a simple example of a worker that takes text in and returns text out
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
        "namespace": "demo",
    },
    image=DYNAMO_IMAGE,
)
class TextProcessor:
    def __init__(self) -> None:
        logger.info("Initializing TextProcessor")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Process the input text."""
        text = req.text
        logger.info(f"Processing text: {text}")
        # Add your processing logic here
        return ResponseType(text=f"Processed: {text}")
```

### Client Example

Here's a simple example of a client that directly calls the TextProcessor
worker through Dynamo without any intermediate services:

```python
import asyncio
from pydantic import BaseModel
from dynamo.sdk import get_runtime

# These could also be imported from a shared file/definition
class RequestType(BaseModel):
    text: str

class ResponseType(BaseModel):
    text: str

async def call_text_processor():
    # Get the runtime
    runtime = await get_runtime()

    # Get a client to the TextProcessor service
    client = await runtime.namespace("demo").component("TextProcessor").endpoint("generate").client()

    # Create a request
    request = RequestType(text="Hello, Dynamo!")

    # Call the process endpoint
    response = await client.generate(request)

    # Print the response
    print(f"Response: {response.text}")

# Run the async function
if __name__ == "__main__":
    asyncio.run(call_text_processor())
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

KV-aware routing is an essential feature of Dynamo that optimizes for routing
requests to specific workers while minimizing a specific KV-cache based cost function.

```
# TODO:
# 1. Highlight minimal snippets on touch points for KVMetrics/Aggregator/Indexer/etc.
#    to enable kv cache aware routing metrics in a custom worker.
# 2. Enhance KV Cache Routing guide to go into more detail on customizing python cost function.
```

For more details, see the [KV Cache Routing Guide](docs/kv_cache_routing.md).

### Disaggregated Serving

```python
# TODO:
# - Prefill/Decode worker snippets
# - NIXL specifics
```

TODO: Discuss local vs remote prefill
TODO: Decide on calling decode (interanlly calling prefill) vs. calling prefill -> decode
TODO: Link to disagg guide and performance tuning guide

- For more information on Disaggregated Serving, see the
  [general guide](docs/disagg_serving.md) and [performance tuning guide](docs/guides/disagg_perf_tuning.md).

## Best Practices

1. **Error Handling**: Always implement proper error handling and logging:
   ```python
   try:
       # Your logic here
   except Exception as e:
       logger.error(f"Error processing request: {e}")
       raise
   ```

2. **Resource Management**: Configure resource requirements based on your needs:
   ```python
   @service(
       resources={
           "cpu": "10",
           "memory": "20Gi",
           "gpu": "1",  # If needed
       }
   )
   ```

3. **Async Operations**: Use async/await for I/O operations:
   ```python
   @dynamo_endpoint()
   async def generate(self, request):
       # Use async operations for better performance
       result = await self.some_async_operation()
   ```

For more details about best practices and performance optimization, see the [Dynamo Serve Guide](docs/guides/dynamo_serve.md#best-practices).

## Additional Resources

- Check the [examples](examples/) directory for more detailed implementations
- Refer to the [Dynamo SDK Docs](deploy/dynamo/sdk/docs/README.md) for API details.
- For Disaggregated Serving, see the [general guide](docs/disagg_serving.md) and [performance tuning guide](docs/guides/disagg_perf_tuning.md).