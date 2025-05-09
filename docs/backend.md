# Writing Python Workers in Dynamo

This guide explains how to create your own Python worker in Dynamo and deploy
it via `dynamo serve` or `dynamo deploy`, covering basic concepts as well as
advanced features like enabling KV routing and disaggregated serving.

For detailed information about `dynamo serve` infrastructure, see the
[Dynamo SDK Docs](../deploy/sdk/docs/sdk/README.md).

For a guide that walks through how to launch a vLLM-based worker with
implementation of Disaggregated Serving and KV-Aware Routing included,
see the [Dynamo Serve Guide](../docs/guides/dynamo_serve.md).

## Basic Concepts

When deploying a python-based worker with `dynamo serve` or `dynamo deploy`, it is
a Python class based definition that requires a few key decorators to get going:
- `@service`: used to define a worker class
- `@dynamo_endpoint`: marks methods that can be called by other workers or clients

For more detailed information on these concepts, see the
[Dynamo SDK Docs](../deploy/sdk/docs/sdk/README.md).

### Worker Skeleton

Here is the rough outline of what a worker may look like in its simplest form:

```python
from dynamo.sdk import dynamo_endpoint, service

@service(
    dynamo={
        "namespace": "your_namespace",
    },
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
see the [Dynamo SDK Docs](../deploy/sdk/docs/README.md).

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

For example, if you deploy your worker directly behind an OpenAI HTTP (`http`) service
using `llmctl`, you can define the request and response types to correspond to
Chat Completions objects, such as the ones specified in the OpenAI API. For example:
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
# basic_worker.py
# This can be run standalone with `dynamo serve basic_worker:YourWorker`

from pydantic import BaseModel
from dynamo.sdk import dynamo_endpoint, service

logger = logging.getLogger(__name__)

class RequestType(BaseModel):
    text: str

class ResponseType(BaseModel):
    text: str

@service(
    dynamo={
        "namespace": "your_namespace",
    }
)
class YourWorker:
    def __init__(self) -> None:
        logger.info("Starting worker...")

    @dynamo_endpoint()
    async def generate(self, request: RequestType):
        """Generate tokens and stream them back"""
        logger.info(f"Worker endpoint received: {request.text}")

        for token in request.text.split():
            yield ResponseType(text=token).model_dump_json()
```

To see a minimal worker example like the above used in a larger pipeline of
components, see the `dynamo serve`
[Hello World example](../examples/hello_world).

### Client Example

Here's a simple example of a client that directly calls the example
worker above through Dynamo without any intermediate services:

```python
import asyncio
from pydantic import BaseModel
from dynamo.runtime import dynamo_worker, DistributedRuntime

# These could also be imported from a shared file/definition
class RequestType(BaseModel):
    text: str

class ResponseType(BaseModel):
    text: str

@dynamo_worker()
async def client_worker(runtime: DistributedRuntime):
    # Get a client to the worker endpoint from the distributed runtime
    client = await runtime.namespace("your_namespace").component("YourWorker").endpoint("generate").client()

    # Create a request
    request = RequestType(text="Hello, Dynamo!")

    # Call the dynamo endpoint exposed by the worker
    responses = await client.generate(request.model_dump_json())
    async for response in responses:
        print(response)

if __name__ == "__main__":
    asyncio.run(client_worker())
```

If there is an OpenAI `http` service in front of your worker and the worker
is defined to accept ChatCompletions-like requests, you could also use an
OpenAI-based client (or `curl`) that sends requests to the OpenAI HTTP Service,
and internally these requests would be routed to the attached worker endpoints instead.

In more advanced scenarios where your worker may operate on some other intermediate format
that may not directly match an OpenAI-like format, you could setup a separate processor worker
that does something like the following:
- Take in OpenAI Chat Completions requests from the HTTP service
- Convert requests from Chat Completions format to the RequestType format your worker expects
- Forward requests to the worker(s)
- Convert responses from the worker's ResponseType back into Chat Completions response format
- Forward responses back to client

This advanced scenario of a separate
[OpenAI Processor worker](../examples/llm/components/processor.py)
is demonstrated in this
[vLLM example](../examples/llm/).

For a more minimal example of deploying a pipeline of components with a custom
API that your client can communicate with, see the
[Hello World example](../examples/hello_world).

## Advanced Features

### KV Routing for LLMs

KV-aware routing is a powerful feature of Dynamo that optimizes for routing
requests to specific workers while minimizing a specific KV-cache based cost function.

In its simplest form, all a worker needs to do to enable KV-aware routing is to
publish KV metrics through the `KvMetricsPublisher`, which will be consumed
by a Dynamo KV Router through the `KvMetricsAggregator`:

```python
# kv_metrics_worker.py
# This can be run standalone with `dynamo serve kv_metrics_worker:YourWorker`

import logging
import random

from pydantic import BaseModel
from dynamo.llm import KvMetricsPublisher
from dynamo.sdk import dynamo_endpoint, service, dynamo_context

logger = logging.getLogger(__name__)

class RequestType(BaseModel):
    text: str

class ResponseType(BaseModel):
    text: str

@service(
    dynamo={
        "namespace": "your_namespace",
    }
)
class YourWorker:
    def __init__(self):
        # Initialize metrics publisher from Dynamo
        self.component = dynamo_context["component"]
        self.metrics_publisher = KvMetricsPublisher()
        # Register an endpoint for consumers of the KV Metrics
        # (KvMetricsAggregator) to listen/gather on.
        self.metrics_publisher.create_endpoint(self.component)

        # Initialize some metrics for the worker/class to track
        self.request_active_slots = 0
        self.request_total_slots = 1024
        self.kv_active_blocks = 0
        self.kv_total_blocks = 1024
        self.num_requests_waiting = 0
        self.gpu_cache_usage_perc = 0.0
        self.gpu_prefix_cache_hit_rate = 0.0

        # Publish some initial metrics to register
        # this worker as a candidate for KV Routing.
        self.metrics_publisher.publish(
            self.request_active_slots,
            self.request_total_slots,
            self.kv_active_blocks,
            self.kv_total_blocks,
            self.num_requests_waiting,
            self.gpu_cache_usage_perc,
            self.gpu_prefix_cache_hit_rate,
        )

    def publish_kv_metrics(self):
        # Populate the frequently changing metrics with random data for
        # demonstration. These values should be tracked by the implementation,
        # or queried from the underlying inference framework.
        self.kv_active_blocks = random.randint(0, 1024)
        self.num_requests_waiting = random.randint(0, 100)
        self.gpu_cache_usage_perc = random.uniform(0, 1.0)
        self.gpu_prefix_cache_hit_rate = random.uniform(0, 1.0)

        # Publish the metrics with the current state
        self.metrics_publisher.publish(
            self.request_active_slots,
            self.request_total_slots,
            self.kv_active_blocks,
            self.kv_total_blocks,
            self.num_requests_waiting,
            self.gpu_cache_usage_perc,
            self.gpu_prefix_cache_hit_rate,
        )

    @dynamo_endpoint()
    async def generate(self, request: RequestType):
        """Generate tokens, update KV Cache metrics, and stream the tokens back"""
        # Increment the number of active requests on receiving one
        self.request_active_slots += 1
        logger.info(f"Worker endpoint received: {request.text}")

        # Simulate each step of token generation
        for token in request.text.split():
            # Update the metrics with the current state
            self.publish_kv_metrics()

            print("Returning token:", token)
            yield ResponseType(text=token).model_dump_json()

        # Decrement the number of active requests when complete with one
        self.request_active_slots -= 1
```

The granularity at which metrics are published is up to the backend/worker implementation.
For example, you may want to update metrics on every single generation step during token
generation, or you may only want to update once per request, depending on your use case.
Assuming long generation time or long output token sequence lengths, it would be more
accurate to publish metrics at every generation step.

With the worker publishing KV metrics, you should now be able to connect it
to a KV Router through the `KvMetricsAggregator`.

These metrics can then be inputs to a cost function to determine which
of the available worker's the request should be routed to.

For a [python-based KV Router](../examples/llm/components/kv_router.py)
implementation, the router is like any other worker, and it can expose
an endpoint that can do arbitrary things based on your use case.

For example, you can initialize the `KvMetricsAggregator` and `KvIndexer`
in your class implementation:
```python
@service(
    dynamo={
        "namespace": "your_namespace",
    },
)
class Router:
    # ...

    @async_on_start
    async def async_init(self):
        self.runtime = dynamo_context["runtime"]

        # Initialize a listener/aggregator for collecting KV metrics
        # from the specified component (workers) publishing them
        kv_listener = self.runtime.namespace("your_namespace").component("YourWorker")
        await kv_listener.create_service()
        self.indexer = KvIndexer(kv_listener, self.args.block_size)
        self.metrics_aggregator = KvMetricsAggregator(kv_listener)
```

With this flexibility, you can also define your own cost function that takes the
KV metrics from the KvMetricsAggregator above and the set of available workers
as inputs, and returns which worker ID that the request should be routed to.
Since the router is like any other worker in this context, you can also track
your own custom metrics and use them in your cost function:

```python
@service(
    dynamo={
        "namespace": "your_namespace",
    },
)
class Router:
    # ...

    def _cost_function(
        self,
        scores: OverlapScores | None,
        metrics: AggregatedMetrics | None,
        token_length: int,
        custom_metrics: dict = {},
    ):
        """
        Args:
            scores (OverlapScores | None): The number of matching blocks between
                the request and the prefix cache of each worker.
            metrics (AggregatedMetrics | None): Several worker metrics polled
                by the `KvMetricsAggregator`, currently including the
                GPU cache usage, number of waiting requests, and the
                GPU prefix cache hit rate.
            token_length (int): The number of tokens in the request.
            custom_metrics (dict): Arbitrary (optional) data from your implementation.

        Returns:
            worker_id (str): The best worker ID based on the inputs.
        """

        # This is a dummy implementation for demonstration purposes, see the
        # llm/tensorrt_llm/hello_world examples for more realistic implementations.
        worker_ids = []

        # KV cache block hit scores
        for worker_id, score in scores.scores.items():
            print(f"{worker_id} # of matching KV Blocks of size {self.indexer.block_size()}: {score}")
            worker_ids.append(worker_id)

        # Aggregated KVMetrics published by workers
        for endpoint_metrics in metrics.endpoints:
            print(f"Endpoint metrics: {endpoint_metrics}")

        # Replace this random choice with your custom criteria to
        # select the best worker ID.
        best_worker_id = random.choice(worker_ids)

        return best_worker_id


    @dynamo_endpoint()
    async def generate(self, request: Tokens) -> AsyncIterator[WorkerId]:
        try:
            # lora_id is a placeholder for lora support, but not used in this example
            lora_id = 0
            scores = await self.indexer.find_matches_for_request(
                request.tokens, lora_id
            )
        except Exception as e:
            scores = {}
            print(f"Error finding matches: {e}")

        # Get published/aggregated KV Metrics
        metrics = await self.metrics_aggregator.get_metrics()

        # (Optional) Add custom metrics to consider in the cost function
        # The types and data used here are fully up to your implementation
        custom_metrics = {"my_custom_metric": 42}

        # Call custom cost function
        worker_id = self._cost_function(
            scores, metrics, len(request.tokens), custom_metrics
        )

        # Return worker ID selected from cost function
        yield f"{worker_id}"
```

Similarly, for running a Rust-based Router as a standalone binary
rather than as a Python Worker, see the
[WorkerSelector Trait](../lib/llm/src/kv_router.rs) trait, and the
[Router Component](../components/router/src/main.rs).

For more details on receiving and routing based on the worker's published KV
metrics, see the [KV Cache Routing Guide](../docs/kv_cache_routing.md).

### Disaggregated Serving

#### NIXL

NIXL (NVIDIA Inter-process Link) enables efficient GPU memory sharing between processes. In Prefill/Decode disaggregation, we use NIXL to transfer computed KV cache blocks from prefill workers to decode workers. Here are the core concepts:

1. **NIXL Agent Setup**
```python
from nixl._api import nixl_agent

class NixlConnector:
    def __init__(self, engine_id: str, rank: int):
        # Create unique NIXL agent for this worker
        self.nixl_wrapper = nixl_agent(str(uuid.uuid4()), None)
        self.engine_id = engine_id
        self.rank = rank
        self.block_len = None  # Will be set during registration
```

2. **Memory Registration and Transfer Preparation**
```python
def register_kv_caches(self, kv_cache: torch.Tensor):
    # Get block size from the KV cache tensor
    # Note: KV cache layout depends on specific attention implementation
    num_blocks, block_size, num_heads, head_dim = kv_cache.shape
    self.block_len = block_size * num_heads * head_dim * kv_cache.element_size()
    self.num_blocks = num_blocks

    # Register KV cache tensor with NIXL for sharing
    base_addr = kv_cache.data_ptr()
    region_len = num_blocks * self.block_len
    caches_data = [(base_addr, region_len, self.rank, "")]

    # Register memory regions with NIXL
    descs = self.nixl_wrapper.get_reg_descs(caches_data, "VRAM")
    self.nixl_wrapper.register_memory(descs)

    # Prepare local side of the transfer
    blocks_data = []
    for block_id in range(num_blocks):
        block_offset = block_id * self.block_len
        blocks_data.append((base_addr + block_offset, self.block_len, self.rank))

    # Create transfer descriptors and prepare for transfers
    self.local_blocks_descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")

    # Create transfer handle with block descriptors for future transfers
    self.local_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist("", self.local_blocks_descs)
```

3. **Remote Agent Communication**
```python
def get_agent_metadata(self):
    # Get metadata for sharing with other agents
    return self.nixl_wrapper.get_agent_metadata(), self.local_blocks_descs

def add_remote_agent(self, engine_id: str, agent_metadata: bytes, remote_blocks_descs: bytes):
    # Connect to remote NIXL agent
    agent_name = self.nixl_wrapper.add_remote_agent(agent_metadata)

    # Prepare remote side transfer handle using provided block descriptors
    self.remote_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist(agent_name, remote_blocks_descs)

    return agent_name

# Example usage:
# On decode worker:
decode_metadata, decode_blocks_descs = nixl_connector.get_agent_metadata()
# Share metadata with prefill worker through your communication channel

# On prefill worker:
nixl_connector.add_remote_agent(decode_engine_id, decode_metadata, decode_blocks_descs)
```

4. **KV Cache Transfer**
```python
def write_blocks(self, local_block_ids, remote_block_ids, notify_msg):
    # Initiate asynchronous transfer using block IDs
    # Block descriptors were specified during transfer preparation
    handle = self.nixl_wrapper.make_prepped_xfer(
        "WRITE",
        self.local_xfer_side_handle,
        local_block_ids,
        self.remote_xfer_side_handle,
        remote_block_ids,
        notify_msg
    )
    status = self.nixl_wrapper.transfer(handle)

# Example usage:
# On prefill worker:
nixl_connector.write_blocks([0, 3], [12, 16], "kv_transfer")
```

The NIXL connector provides:
- GPU memory registration for sharing between processes
- Connection establishment between Prefill and Decode workers
- Efficient block-based KV cache transfers
- Asynchronous transfer notifications

For a complete implementation example using NIXL for disaggregated serving, see the [vLLM example](../examples/llm/README.md).

#### Disaggregation in Dynamo

Aside from the NIXL specifics above, at its core, disaggregation in Dynamo builds
on the same concepts used for any Dynamo client<->worker or worker<->worker
interaction over the DistributedRuntime.

First you can define a worker for each as usual:
```python
class DecodeWorker:
    # ...

class PrefillWorker:
    # ...
```

Second, you decide when/how the (Decode) worker should decide to Prefill remotely
(by calling a separate Prefill worker), or decide to simply do the Prefill itself.
In some scenarios, it may be more efficient for the Decode worker to just do the
Prefill itself rather than do the extra communication, such as if the input
sequence length is below some small threshold. If you wanted to disable
disaggregation, the DecodeWorker could just always do the Prefill step as well.
```python
@service(
    dynamo={
        "namespace": "your_namespace",
    },
)
class DecodeWorker:
    def __init__(self):
        self.runtime = dynamo_context["runtime"]

        # Whether the decode worker should call a separate Prefill worker or not
        self.do_remote_prefill = True

        # Initialize client to PrefillWorker
        self.prefill_client = await self.runtime
                .namespace("your_namespace")
                .component("PrefillWorker")
                .endpoint("generate")
                .client()

    @dynamo_endpoint()
    async def generate(self, request):
        if self.do_remote_prefill:
            # Forward the request to the prefill worker
            prefill_response = await self.prefill_client.generate(...)

        # ... framework-specific decode logic ...

@service(
    dynamo={
        "namespace": "your_namespace",
    },
)
class PrefillWorker:
    def __init__(self):
        # ...

    @dynamo_endpoint()
    async def generate(self, request):
        # ... framework-specific prefill logic ...
```

Depending on the load distribution of requests and number of Prefill/Decode
worker instances, instead of directly forwarding requests to the Prefill
worker endpoint, it may be advantageous to send Prefill requests into a queue
that the Prefill workers can pull from on-demand instead. You can see an example
of that [here](../examples/hello_world/disagg_skeleton/components/prefill_worker.py).

For an introductory example on doing disaggregation with Dynamo using simple models, see
[this example](../examples/hello_world/disagg_skeleton).

For more information on Disaggregated Serving, see the
[general guide](../docs/disagg_serving.md) and [performance tuning guide](../docs/guides/disagg_perf_tuning.md).

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

- Check the [examples](../examples/) directory for more detailed implementations
- Refer to the [Dynamo SDK Docs](../deploy/sdk/docs/sdk/README.md) for API details.
- For Disaggregated Serving, see the [general guide](../docs/disagg_serving.md) and [performance tuning guide](../docs/guides/disagg_perf_tuning.md).
