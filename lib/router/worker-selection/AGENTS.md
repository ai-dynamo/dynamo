# lib/router/worker-selection

`dynamo-worker-selection` is the worker selection stage: pick one rank from a candidate table under a policy. The candidate rows and views, the `WorkerFilter`, `WorkerScorer`, and `WorkerPicker` plugin API, the composed `WorkerSelectionPolicy`, the candidate preparer, runtime updates, `SelectionConfig`, the selection trace, conditional disaggregation, and the LoRA filter. The kernel builds the rows and calls the policy; the registry and `plugins::prelude` live in `dynamo-kv-router`.

- Depends on (Dynamo crates): `dynamo-router-protocol`, `dynamo-session-affinity` (the affinity target), `dynamo-slot-tracker` (the fullness rule is that stage's output).
- Must not depend on: other stages except `dynamo-slot-tracker` and `dynamo-session-affinity`, `dynamo-kv-hints`, the kernel, kv-router, the runtime, any transport.
- Hosts import this crate through its `host` module; plugins import `dynamo_kv_router::plugins::prelude`.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`; the working rules for the public plugin API are in `README.md`.
