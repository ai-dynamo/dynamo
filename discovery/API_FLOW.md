## Server Side (`hello_world.py`)

| Python API Call | Rust Files & Methods |
|----------------|---------------------|
| `@dynamo_worker()` | **lib/runtime/src/distributed.rs**<br>- `DistributedRuntime::from_settings()`<br>- `DistributedRuntime::new()` |
| `runtime.namespace(namespace_name)` | **lib/runtime/src/component.rs**<br>- `impl Namespace { pub fn new() }`<br>**lib/runtime/src/component/namespace.rs**<br>- `impl Namespace { pub fn component() }` |
| `namespace.component(component_name)` | **lib/runtime/src/component.rs**<br>- `impl Component { pub fn new() }`<br>- `impl ComponentBuilder { pub fn from_runtime() }` |
| `await component.create_service()` | **lib/runtime/src/component/service.rs**<br>- `impl ServiceConfigBuilder { pub async fn create() }`<br>**lib/runtime/src/transports/nats.rs**<br>- `impl Client { pub fn service_builder() }`<br>- `impl Client { pub async fn start() }` |
| `component.endpoint(endpoint_name)` | **lib/runtime/src/component.rs**<br>- `impl Component { pub fn endpoint() }`<br>**lib/runtime/src/component/endpoint.rs**<br>- `impl Endpoint { pub fn new() }` |
| `await endpoint.serve_endpoint(content_generator)` | **lib/runtime/src/component/endpoint.rs**<br>- `impl EndpointConfigBuilder { pub async fn start() }`<br>**lib/runtime/src/pipeline/network/ingress/push_endpoint.rs**<br>- `impl PushEndpoint { pub async fn start() }`<br>**lib/runtime/src/transports/etcd.rs**<br>- `impl Client { pub async fn kv_create() }` |

## Client Side (`client.py`)

| Python API Call | Rust Files & Methods |
|----------------|---------------------|
| `@dynamo_worker()` | **lib/runtime/src/distributed.rs**<br>- `DistributedRuntime::from_settings()`<br>- `DistributedRuntime::new()` |
| `runtime.namespace("hello_world")` | **lib/runtime/src/component.rs**<br>- `impl Namespace { pub fn new() }` |
| `namespace.component("backend")` | **lib/runtime/src/component.rs**<br>- `impl Component { pub fn new() }` |
| `component.endpoint("generate")` | **lib/runtime/src/component.rs**<br>- `impl Component { pub fn endpoint() }` |
| `await endpoint.client()` | **lib/runtime/src/component/endpoint.rs**<br>- `impl Endpoint { pub async fn client() }`<br>**lib/runtime/src/component/client.rs**<br>- `impl Client { pub async fn new_dynamic() }` |
| `await client.wait_for_instances()` | **lib/runtime/src/component/client.rs**<br>- `impl Client { pub async fn wait_for_instances() }`<br>**lib/runtime/src/transports/etcd.rs**<br>- `impl Client { pub async fn kv_get_prefix() }` |
| `await client.generate(request)` | **lib/runtime/src/component/client.rs**<br>- `impl Client { pub async fn call_endpoint() }`<br>**lib/runtime/src/transports/nats.rs**<br>- `impl Client { pub async fn publish() }` |

## Core Infrastructure Files

These files provide the underlying infrastructure but aren't directly mapped to API calls:

| File | Purpose |
|------|---------|
| **lib/runtime/src/discovery.rs** | Service discovery and lease management |
| **lib/runtime/src/system_health.rs** | Health checking and monitoring |
| **lib/runtime/src/storage/key_value_store/etcd.rs** | etcd-based key-value store |
| **lib/runtime/src/component/registry.rs** | Component and service registry |

## Key etcd Paths

| Operation | Path |
|-----------|------|
| Endpoint Registration | `/services/{namespace}/{component}/{endpoint}-{lease_id}` |
| Instance Discovery | `/services/{namespace}/{component}` |

## Key NATS Subjects

| Operation | Subject |
|-----------|---------|
| Service Group | `{namespace_name}.{service_name}` |
| Endpoint | `{namespace_name}.{service_name}.{endpoint_name}-{lease_id_hex}` |