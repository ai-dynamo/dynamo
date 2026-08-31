# lib/kv-router/src/services

These services must not depend on `dynamo-runtime`.

## Guardrails

- Do not assume a Dynamo request plane, discovery plane, or event plane exists.
- Do not add direct or optional `dynamo-runtime` dependencies to service code.
- Prefer brokerless transport primitives such as TCP, HTTP, and ZMQ.
- Keep service protocols usable by processes that do not run any other Dynamo components.
- Keep reusable routing, indexing, and tracking logic outside this directory.
