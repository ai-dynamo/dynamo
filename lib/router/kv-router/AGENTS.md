# lib/router/kv-router

`dynamo-kv-router` is the composition crate: it assembles the kernel and the stages into a running router, puts every transport (HTTP, ZMQ, the replica mesh, Prometheus) behind a feature, and is the one crate plugins name through `plugins::prelude`. Subdirectories keep their own `AGENTS.md` where one exists.

## Guardrails

- **Adding a dependency or an import.** Don't depend on `dynamo-runtime`, directly or behind a feature, and don't assume a Dynamo request, discovery, or event plane exists: processes that run no other Dynamo component use this crate.
- **Adding to `src/services/`.** Use brokerless transports (TCP, HTTP, ZMQ) and keep each service's protocol usable by a process that runs no other Dynamo component. Keep reusable routing, indexing, and tracking logic outside `src/services/`.

## Performance

PR numbers point to worked examples; open one only when you need to see how a rule was applied.

- **Adding a field to a request type that carries a prompt.** Add it to the flat wire struct (`inline_prompt_wire!` in `lib/router/scheduler/src/selection/types.rs`) and parse from bytes as `PromptJson` does. `#[serde(flatten)]` or an intermediate `serde_json::Value` buffers the whole body, which holds every prompt token (#17238).
- **Opening a ZMQ socket.** Use the pooled helpers in `src/services/common/zmq.rs` (`create_sub_socket_topics`, `connect_sub_socket`, `connect_dealer_socket`); a context per socket costs two threads per socket. `create_bound_pub_socket` is the exception: it takes its own context so its port is free as soon as its owner shuts down, so don't open one per worker. Open a per-worker socket that is rarely used on demand, as `replay_gap` dials its DEALER (#17253).
