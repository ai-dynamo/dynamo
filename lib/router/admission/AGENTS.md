# lib/router/admission

- This crate is the admission stage: the per-class policy queue and its stats, the resolved policy profiles, the classifier runtime with request lifecycles, the classifier plugin API (`RequestClassifier`, `ClassifyRequest`, the `WorkerView` it reads workers through), `AdmissionConfig`, and `AdmissionError`. The YAML policy file, the plugin registry, and the queue's HTTP exposure stay in `dynamo-kv-router`.
- Depend only on `dynamo-router-protocol` and `dynamo-tokens` among Dynamo crates. Never add `dynamo-runtime`, `dynamo-kv-router`, `dynamo-kv-indexer`, `dynamo-slot-tracker`, `dynamo-worker-selection`, axum, reqwest, zmq, or prometheus; the gate rejects them. Tokio is allowed for the queue's instants and the classifier's delivery task, never for sockets.
- The kernel is the only caller that admits, holds, or rejects; this crate exposes `&self` reads and the mutations the kernel drives, and never reaches for worker-selection rows except through its own `WorkerView`.
- In-tree hosts import this crate through `host`; classifier plugins import `dynamo_kv_router::plugins::prelude`. Items either module names are `pub` with a doc comment, never `#[doc(hidden)] pub`.
- Run `scripts/router_layering_check.sh` before committing.
