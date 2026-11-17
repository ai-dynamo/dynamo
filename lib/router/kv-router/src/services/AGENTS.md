# lib/router/kv-router/src/services

These services must not depend on `dynamo-runtime`.

## Guardrails

- Do not assume a Dynamo request plane, discovery plane, or event plane exists.
- Do not add direct or optional `dynamo-runtime` dependencies to service code.
- Prefer brokerless transport primitives such as TCP, HTTP, and ZMQ.
- Keep service protocols usable by processes that do not run any other Dynamo components.
- Keep reusable routing, indexing, and tracking logic outside this directory.

## Performance

- **Adding a field to a request type that carries a prompt.** Add it to the flat wire struct (`inline_prompt_wire!` in `lib/router/scheduler/src/selection/types.rs`) and parse from bytes as `PromptJson` does. `#[serde(flatten)]` or an intermediate `serde_json::Value` buffers the whole body, which holds every prompt token (#17238).
- **Opening a ZMQ socket.** Use the helpers in `common/zmq.rs`, which share pooled contexts; a context per socket costs two threads per socket. Open a per-worker socket that is rarely used on demand, as `replay_gap` dials its DEALER (#17253).
