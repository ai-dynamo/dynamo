# Service and Model Discovery Interface

## Background

This document is part of a series of docs attempting to replace the ETCD dependency in Dynamo using Kubernetes

Source: https://github.com/ai-dynamo/enhancements/blob/neelays/runtime/XXXX-runtime-infrastructure.md

ETCD usage in Dynamo can be categorized into the following:
- Endpoint Instance Discovery (Current focus)
- Model / Worker Discovery (Current focus)
- Worker Port Conflict Resolution
- Router State Sharing Synchronization

## ServiceDiscovery Interface

To de-couple Dynamo from ETCD, we define a `ServiceDiscovery` interface that can be satisfied by different backends (etcd, kubernetes, etc). In Kubernetes environments, we will use the Kubernetes APIs to implement this interface.

### Server-Side (Service Registry)

#### Methods
- `register_instance(namespace: str, component: str) -> InstanceHandle`
  - Registers a new instance of the given namespace and component. Returns an InstanceHandle.

- `InstanceHandle.instance_id() -> str`
  - Returns a unique identifier for the instance. (Similar to the primary lease id in ETCD)

- `InstanceHandle.set_metadata(metadata: dict) -> None`
  - Write the metadata associated with the instance.

- `InstanceHandle.set_ready(status: InstanceStatus) -> None`
  - Marks the instance as ready or not ready for traffic.


### Client-Side (Service Discovery)

#### Methods
- `list_instances(namespace: str, component: str) -> List[Instance]`
  - Lists all instances that match the given namespace and component. Returns a list of Instance objects.

- `watch(namespace: str, component: str) -> InstanceEventStream`
  - Watches for events for the set of instances that match `(namespace, component)`.
  - Returns a stream of events (InstanceAddedEvent, InstanceRemovedEvent)

- `Instance.metadata() -> dict`
  - Returns the metadata for a specific instance.

## Where will these APIs be used?

These APIs are intended to be used internally within the Rust codebase where there are currently calls to `etcd_client` for service discovery and model management.

Some examples of code that will be impacted:

Frontend Worker Discovery:
(How the frontend discovers workers and maintains inventory of model to worker mappings)
- run_watcher function at [lib/llm/src/entrypoint/input/http.rs](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/entrypoint/input/http.rs)
- ModelWatcher at [lib/llm/src/discovery/watcher.rs](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/discovery/watcher.rs)

Model registration (register_llm function):
(Initial worker bootstrapping as part of register_llm)
- LocalModel.attach at [lib/llm/src/local_model.rs](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/local_model.rs)

Dynamo Runtime:
(Used for registering on the runtime (Server) and for getting clients to components (Client))
- client.rs [lib/runtime/src/client.rs](https://github.com/ai-dynamo/dynamo/blob/main/lib/runtime/src/client.rs)
- endpoint.rs [lib/runtime/src/endpoint.rs](https://github.com/ai-dynamo/dynamo/blob/main/lib/runtime/src/endpoint.rs)

### Overall Flow

At a high level, these are the service discovery requirements in a given disaggregated inference setup. We note how the APIs defined above can be used to achieve this.

Credits: @itay

1. The frontend needs to know how to reach the decode workers
2. The frontend needs to know what model key to route to the above workers
3. The frontend needs to know some model specifics (like tokenizer) for the specified model
4. The decode worker needs to know how to reach the prefill workers (for the disagg transfer)

#### Frontend Discovers Decode Workers

##### Decode Worker Set Up

```python
# Register the instance
decode_worker = service_registry.register_instance("dynamo", "decode")
instance_id = decode_worker.instance_id() # Unique identifier for the instance

# Start up NATS listener for the endpoints of this component (or other transport). We may or may not need the instance id to setup the transport.
comp_transport_details = set_up_nats_listener(instance_id)

# Write metadata associated with the instance
metadata = {
    "model": {...}, # Runtime Info
    "transport": comp_transport_details, # Transport details for this component
    "mdc": {...}, #  Model Deployment Card
}

decode_worker.set_metadata(metadata)
decode_worker.set_ready("ready")
```

##### Frontend Discovers Decode Workers

```python
# Frontend start:
decode_workers = service_discovery.list_instances("dynamo", "decode")
for decode_worker in decode_workers:
    # Fetch the associated metadata for the instance to register the model in-memory model registry
    metadata = decode_worker.metadata()
    model = metadata["Model"]
    # map the instance to the model in the in-memory model registry ...

# Sets up watch to keep cache up to date
```

#### Frontend Needs to Know what Model Key -> Instance Mapping

Addressed above.

#### Frontend Needs to Know Some Model Specifics (like Tokenizer)

- Model card is duplicated on each decode worker in the metadata.

#### Decode Worker Needs to Know how to Reach Prefill Workers

- This is done in the exact same way as the way the frontend reaches the decode workers.


<!-- ### Simplifications and Assumptions -->

## API Reference

### register_instance

This method is used to register a new instance of the given namespace and component. Returns an InstanceHandle that can be used to manage the instance.

```python
# Example: Registration of a decode worker
# Server-side: Register the instance
decode_worker = service_registry.register_instance("dynamo", "decode")
instance_id = decode_worker.instance_id() # Unique identifier for the instance
```

#### Kubernetes Reference Impl
- Asserts the pod has namespace and component labels that match up with method args. Fails otherwise.
- Asserts there is a Kubernetes Service that will select on namespace and component labels. (More on this later)
- Returns an InstanceHandle object tied to the pod name. This can be used to fetch the unique identifier for the instance which can be used in setting up the transport.

Note: The instance registration is tied to the pod lifecycle.

### set_metadata

This method is used to set metadata associated with an instance.

```python
# Server-side: Set metadata for the instance
metadata = {
    "model": {
        "name": "Qwen3-32B",
        "type": "Completions",
        "runtime_config": {
            "total_kv_blocks": 24064,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 2048
        }
    },
    "transport": {...}, # Transport details for this component
    "mdc": {...}, # Model Deployment Card
}
decode_worker.set_metadata(metadata)
```

#### Implementation Details
- Updates an in-memory struct within the instance process
- Exposes the metadata via a `/metadata` HTTP endpoint on the component
- No external storage required.

### set_ready

This method is used to mark the instance as ready or not ready for traffic.

```python
# Context: decode worker has finished loading the model

decode_worker = service_registry.register_instance("dynamo", "decode")

# Setup handlers for the decode component, then setup transport.
decode_worker.set_metadata(metadata)

# When the server is ready to serve, mark the instance as ready for discovery.
decode_worker.set_ready("ready")
```

#### Kubernetes Reference Impl
- The readiness probe of the pod will proxy the result of the instance's readiness status. When the instance is ready for traffic, the readiness probe will return 200 and EndpointSlices will be updated to include the instance.
- Components that have called `watch` on this component will be notified when the instance is ready for traffic. They can use this instance to route traffic to using their transport.


### list_instances

This method is used to list all instances that match the given namespace and component. Returns a list of Instance objects.

```python
# Context: frontend startup - discover existing decode workers
decode_workers = service_discovery.list_instances("dynamo", "decode")
for decode_worker in decode_workers:
    # Fetch the associated metadata for the instance
    metadata = decode_worker.metadata()
    model = metadata["Model"]
    # register the model in-memory model registry
    # map the instance to the model
```

#### Kubernetes Reference Impl
- Queries EndpointSlices for the Service that selects on the namespace and component labels.
- Returns a list of Instance objects for all ready endpoints.

### watch

This method is used to watch for events for the set of instances that match (namespace, component). Returns a stream of events (InstanceAddedEvent, InstanceRemovedEvent) giving signals for when instances that match the subscription are created, deleted, or readiness status changes.

```python
# Context: frontend has watch setup on decode workers
stream = service_discovery.watch("dynamo", "decode")
for event in stream:
    switch event:
        case InstanceAddedEvent:
            # Fetch the associated metadata for the instance
            metadata = event.instance.metadata()
            model = metadata["Model"]
            # register the model in-memory model registry
            # map the instance to the model
```

#### Kubernetes Reference Impl
- Sets up a kubectl watch for EndpointSlices corresponding to the Service that selects on the namespace and component labels.
- Returns a stream of events that inform us when a READY pod matching the namespace and component is up.

### Instance.metadata()

This method is used to fetch the metadata associated with a specific instance. It makes an HTTP request to the instance's `/metadata` endpoint.

```python
# Client-side: Get metadata from a specific instance
decode_workers = service_discovery.list_instances("dynamo", "decode")
for decode_worker in decode_workers:
    # Fetch the associated metadata for the instance
    metadata = decode_worker.metadata()
    model = metadata["Model"]
    # register the model in-memory model registry
    # map the instance to the model
```

#### Implementation Details
- Makes an HTTP GET request to `/metadata` on the instance
- Returns the metadata payload stored in the component's in-memory struct
- No external storage lookup required - direct communication with the instance

## Kubernetes EndpointSlice Discovery Mechanism

This section provides a detailed explanation of how the ServiceDiscovery interface is implemented using Kubernetes EndpointSlices for service discovery.

### Core Components

The Kubernetes implementation relies on three main Kubernetes resources:

1. **Pods**: Each worker instance runs as a pod with specific labels
2. **Service**: Selects pods based on namespace and component labels
3. **EndpointSlices**: Automatically managed by Kubernetes, tracks ready pod endpoints

### Resource Structure

#### Pod Labels
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dynamo-decode-pod-abc123
  labels:
    nvidia.com/dynamo-namespace: dynamo
    nvidia.com/dynamo-component: decode
spec:
  readinessProbe:
    httpGet:
      path: /health
      port: 8080
    initialDelaySeconds: 5
    periodSeconds: 10
```

#### Service Selector
```yaml
apiVersion: v1
kind: Service
metadata:
  name: dynamo-decode-service
spec:
  selector:
    nvidia.com/dynamo-namespace: dynamo
    nvidia.com/dynamo-component: decode
  ports:
  - port: 8080 # The port itself is a dummy port. It is not used for traffic (although it could be).
    targetPort: 8080
    name: http
```

#### EndpointSlice (Auto-generated)
```yaml
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: dynamo-decode-service-abc123
  labels:
    kubernetes.io/service-name: dynamo-decode-service
    nvidia.com/dynamo-namespace: dynamo
    nvidia.com/dynamo-component: decode
addressType: IPv4
endpoints:
- addresses: ["10.1.2.3"]
  conditions:
    ready: true  # Based on pod readiness probe
  targetRef:
    kind: Pod
    name: dynamo-decode-pod-abc123
    namespace: default
ports:
- name: http
  port: 8080
  protocol: TCP
```

### Discovery Flow

1. **Instance Registration**: When `register_instance()` is called, the InstanceHandle is created and tied to the pod
2. **Metadata Setup**: When `set_metadata()` is called, the metadata is stored in-memory and exposed via `/metadata` endpoint
3. **Readiness Signaling**: When `set_ready("ready")` is called, the pod's readiness probe starts returning 200
4. **EndpointSlice Update**: Kubernetes automatically updates the EndpointSlice to mark the endpoint as ready
5. **Event Propagation**: Clients watching EndpointSlices receive events about the ready instance

### Event Mapping

The `watch()` method maps EndpointSlice events to ServiceDiscovery events:

| EndpointSlice Event | ServiceDiscovery Event | Description |
|---------------------|------------------------|-------------|
| Endpoint added with `ready: true` | `InstanceAddedEvent` | New instance is ready for traffic |
| Endpoint condition changes to `ready: true` | `InstanceAddedEvent` | Existing instance becomes ready |
| Endpoint condition changes to `ready: false` | `InstanceRemovedEvent` | Instance becomes unavailable |
| Endpoint removed from slice | `InstanceRemovedEvent` | Instance is terminated |

### Implementation Details

#### Instance ID Resolution
- **Instance ID**: Pod name (e.g., `dynamo-decode-pod-abc123`)
- **EndpointSlice Reference**: `targetRef.name` points to the pod name
- **Metadata Lookup**: HTTP request to `{instance_address}/metadata` for `Instance.metadata()`

#### Watch Implementation
```python
# Kubernetes watch setup for watch()
watch_filter = {
    "labelSelector": f"kubernetes.io/service-name=dynamo-{component}-service"
}
endpoint_slice_watch = k8s_client.watch_endpoint_slices(filter=watch_filter)

for event in endpoint_slice_watch:
    if event.type == "MODIFIED":
        for endpoint in event.object.endpoints:
            if endpoint.conditions.ready:
                emit_event(InstanceAddedEvent(
                    instance_id=endpoint.targetRef.name,
                    address=endpoint.addresses[0]
                ))
```

### Benefits of EndpointSlice Approach

1. **Native Kubernetes Integration**: Leverages built-in service discovery
2. **Automatic Cleanup**: Pods and EndpointSlices are cleaned up when instances terminate
3. **Scalability**: EndpointSlices handle large numbers of endpoints efficiently
4. **Consistency**: Kubernetes ensures eventual consistency across the cluster
5. **Health Integration**: Readiness probes directly control traffic eligibility

## Breakdown of the Implementation
- Remove ETCD* from top level APIs
- Define ServiceDiscovery interfaces in Rust
- Implementations:
    - Kubernetes
    - Existing impl for ETCD needs to be moved behind interface
    - TODO: Local/file system
- Re-factor existing code to use new interfaces ([See Where will these APIs be used?](#where-will-these-apis-be-used))