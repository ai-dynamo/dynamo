# Dynamo Disaggregation: Separating Prefill and Decode for Enhanced Performance

The prefill and decode phases of LLM requests have different computation characteristics and memory footprints. Disaggregating these phases into specialized llm engines allows for better hardware allocation, improved scalability, and overall enhanced performance. For example, using a larger TP for the memory-bound decoding phase while a smaller TP for the computation-bound prefill phase allows both phases to be computed efficiently. In addition, for requests with long context, separating their prefill phase into dedicated prefill engines allows the ongoing decoding requests to be efficiently processed without being blocked by these long prefills.

Disaggregated execution of a request has three main steps:
1. Prefill engine computes prefill phase and generates KV cache
2. Prefill engine transfers the KV cache to decode engine, and
3. Decode engine computes decode phase.

However, not all requests’ prefill phases need to be computed in the remote prefill engine. If the prefill is short or the decode engine has a high prefix cache hit, often it is more efficient to prefill locally in the decode engine. The disaggregation design in Dynamo accounts for all these scenarios and features a flexible framework that delivers strong performance across various conditions.


## Design

```mermaid
sequenceDiagram
    participant D as Worker
    participant Q as PrefillQueue
    participant P as PrefillWorker

    Note over D: Request is routed to decode
    D->>D: Decide if prefill should be done locally or remotely

        D->>D: Allocate KV blocks
        D->>Q: Put RemotePrefillRequest on the queue

        P->>Q: Pull request from the queue
        P-->>D: Read cached KVs from Decode

        D->>D: Decode other requests
        P->>P: Run prefill
        P-->>D: Write prefilled KVs into allocated blocks
        P->>D: Send completion notification
        Note over D: Notification received when prefill is done
        D->>D: Schedule decoding
```

There are four main components in Dynamo disaggregation:
- Worker: execute prefill and decode requests
- Prefill worker: execute prefill requests only
- Disaggregated router: decide whether to prefill locally or remotely
- Prefill queue: cache and load balance the remote prefill requests

When worker receives a request, it first decides if the prefill should be done locally or remotely using the disaggregated router and allocates the KV blocks. If prefilling remotely, it then pushes a remote prefill request to the prefill queue. After that, the prefill worker pulls from prefill queue, reads KV blocks with prefix cache hit from the worker, computes the prefill, and writes the computed KV blocks back to the worker. Finally, the worker completes the remaining decoding.

## Conditional Disaggregation

Not all requests’ prefill phases need to be computed in the remote prefill engine. Disaggregated router decides whether the prefill phase of a request should be computed locally and globally at runtime based on the prefill length and prefill queue status. Specifically, a request is sent to remote prefill engine if the following two conditions are met:
1. The absolute prefill length without prefix cache hit is greater than a preset threshold. On the one hand, if the prefill length of a request is short, it can be efficiently computed in the decode engine by piggybacking chunked prefill requests with ongoing decode requests. On the other hand, if the prefix cache hit is long, the prefill becomes memory bound and hence can be more efficiently computed in the decode engine.
2. The number of remote prefill requests in the prefill queue is less than a preset threshold. When the prefill queue has a large number of prefill requests, it indicates that the prefill workers are lagging behind, and it is better to prefill locally until more prefill workers join.

Conditional disaggregation allows Dynamo to achieve high performance for dynamic workloads

## Prefill Queue

Prefill requests are computation bound (except for very short prefills) and should be executed in their dedicated iterations without any other requests to ensure fast TTFT. To balance the load across multiple prefill engines, Dynamo adopts a global prefill queue where workers push remote prefill requests and prefill workers pull and complete the requests one by one. The global prefill queue is implemented based on NATS stream to ensure high performance and availability.

## Efficient KV Transfer

```mermaid
sequenceDiagram
    participant D as Worker
    participant SD as WorkerScheduler
    participant SP as PrefillWorkerScheduler
    participant P as PrefillWorker

    Note over SD: KV blocks allocated
    SD->>SP: Issue remote prefill request <br> with KV block descriptors via prefill queue
    SP->>P: Add to in-flight batch

    P-->>D: Remote NIXL read for prefix hit KV blocks (non-block)
    P->>P: Execute prefill
    P-->>D: Remote NIXL write for comptued KV blocks (non-block)

    P->>SP: Notify finish
    SP->>SD: Notify finish
    SD->>D: Add to in-flight batch

    D->>D: Execute decode
```

The key to high-performance disaggregation is efficient KV transfer. Dynamo leverage NIXL to transfer KV cache directly from the VRAM of prefill engine to the VRAM of decode engine. In addition, the KV transfer is non-blocking, allowing GPU forward pass to serve other requests in addition to the KV transfer.

After the KV blocks are allocated, the worker scheduler sends the remote prefill requests, which contain the memory descriptors for the allocated KV blocks, to the prefill worker scheduler via prefill queue. This allows the prefill worker to read and write from the remote KV blocks without explicit handling in the remote worker engine, thanks to the RDMA read and write NIXL operations. Once the remote prefill is done, worker scheduler simply adds the decode request to the worker in-flight. This allows workers to execute forward passes of ongoing decode/prefill requests while waiting for the remote prefill to finish.

To reduce the size of memory descriptors, Dynamo applies two optimizations:
1. After each worker finishes its initialization and allocates all the KV cache pool, it stores the memory descriptor of all blocks (which is also referred to as the NIXL metadata) in ETCD, a distributed key-value store. Prefill workers load and cache the memory descriptors in one worker at the first time that it serves a remote prefill request issued by this worker. Thus, only the KV block ID instead of the full memory descriptor is needed when issuing the remote prefill request.

2. Dynamo promotes the memory allocator in the prefill engine to allocate continuous blocks and merge continuous blocks into larger blocks to reduce the total number of KV blocks.

For decode and prefill with different KV layouts (i.e., due to different TP), Dynamo applies a high-performance kernel that transposes the KV blocks into their matching layout in the KV receiver after the NIXL reads and before the NIXL writes.

## Runtime-Reconfigurable xPyD

The prefill queue and NIXL-based KV transfer design in Dynamo naturally allows runtime-reconfigurable xPyD. Workers and prefill workers can be added and removed at runtime without any system-level synchronization or overheads. New and existing prefill workers both just simply pull remote prefill requests from NATS prefill queue. The NIXL metadata of the new or existing workers (for new prefill workers) are lazily loaded and cached when necessary. Specifically, adding and removing workers and prefill workers is as easy as:

- Add worker: add NIXL metadata in ETCD.
- Remove worker: flush engine and delete NIXL metadata in ETCD.
- Add prefill worker: no explicit action needed.
- Delete prefill worker: flush engine.

### How this works under the hood

#### Auto-Discovery for new workers

In Dynamo, we use etcd (a distributed key-value pair store) as a way to register and discover new components. When a new decode/aggregated worker starts, it will add it's endpoint information to etcd allowing the router to discover it and route requests to it. For the KV-cache transfer process, newly added decode workers will put memory descriptors of their kv cache (used in NIXL transfer) in etcd. Newly added prefill workers also register with etcd for discovery and simply start pulling prefill requests from the global prefill queue after they spin up. Prefill workers will lazy-pull the descriptors when they start serving a remote prefill request for the first time.

You can watch this happen live by running the following:

```bash
# in terminal 1 - run the disaggregated serving example
dynamo serve graphs.disagg:Frontend -f ./configs/disagg.yaml
```

```bash
# in terminal 2 - watch the namespace in etcd
watch -cd etcdctl get --prefix <namespace>
```

You should see something like this show up as the disaggregated serving example starts up:

```bash
# worker information
dynamo/components/PrefillWorker/mock:694d967da694ea1e
{
  "component": "PrefillWorker",
  "endpoint": "mock",
  "namespace": "dynamo",
  "lease_id": 7587886413599009310,
  "transport": {
    "nats_tcp": "dynamo_prefillworker_0d6df828.mock-694d967da694ea1e"
  }
}
dynamo/components/Processor/chat/completions:694d967da694ea16
{
  "component": "Processor",
  "endpoint": "chat/completions",
  "namespace": "dynamo",
  "lease_id": 7587886413599009302,
  "transport": {
    "nats_tcp": "dynamo_processor_3816642d.chat/completions-694d967da694ea16"
  }
}
dynamo/components/VllmWorker/generate:694d967da694ea1a
{
  "component": "VllmWorker",
  "endpoint": "generate",
  "namespace": "dynamo",
  "lease_id": 7587886413599009306,
  "transport": {
    "nats_tcp": "dynamo_vllmworker_3f6fafd3.generate-694d967da694ea1a"
  }
}
dynamo/components/VllmWorker/load_metrics:694d967da694ea1a
{
  "component": "VllmWorker",
  "endpoint": "load_metrics",
  "namespace": "dynamo",
  "lease_id": 7587886413599009306,
  "transport": {
    "nats_tcp": "dynamo_vllmworker_3f6fafd3.load_metrics-694d967da694ea1a"
  }
}

# nixl metadata
dynamo/nixl_metadata/e318db87-be55-4c18-9829-8036e1e603e2
```

#### Graceful worker shutdown

Since worker information is stored in etcd, we can shutdown workers by simply revoking their etcd leases. After a lease is revoked:

- Decode/aggregated worker endpoints are immediately removed from etcd so that they would not accept new requests. They finish any in-flight requests, shut down their engine, and exit gracefully
- Prefill workers stop pulling from the prefill queue and exit gracefully after all pending remote kv cache writes finish

You can also visualize this by revoking a workers etcd lease while it has ongoing requests. Refer to this example script that does this: [revoke_lease.py](https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/python/examples/hello_world/revoke_lease.py).