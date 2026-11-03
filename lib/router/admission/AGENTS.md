# lib/router/admission

`dynamo-admission` is the admission stage: admit, hold, or reject now. The per-class policy queue and its stats, the resolved policy profiles, the classifier runtime with request lifecycles, the classifier plugin API (`RequestClassifier`, `ClassifyRequest`, `WorkerView`), `AdmissionConfig`, and `AdmissionError`; the YAML policy file, the plugin registry, and the queue's HTTP exposure stay in `dynamo-kv-router`. The kernel is the only caller that admits, holds, or rejects.

- Depends on (Dynamo crates): `dynamo-router-protocol`, `dynamo-slot-tracker` (the fullness rule), `dynamo-tokens`.
- Must not depend on: other stages except `dynamo-slot-tracker`, the kernel, kv-router, the runtime, any transport (tokio for the queue's instants and the classifier's delivery task only).
- Hosts import this crate through its `host` module; classifier plugins import `dynamo_kv_router::plugins::prelude`.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
