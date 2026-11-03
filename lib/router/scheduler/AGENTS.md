# lib/router/scheduler

`dynamo-scheduler` is the router's scheduler kernel: it runs the stages (index lookup, slot tracking, session affinity, worker selection, admission) atomically for one request and owns the request and response vocabulary, eligibility, the queue actor and `LocalScheduler`, the overlap and prefill-load analysis, the selection kernel, the validated kernel configuration, and the tracking hash. `protocols` holds the kernel's own types and re-exports nothing.

- Depends on (Dynamo crates): `dynamo-admission`, `dynamo-kv-events`, `dynamo-kv-hashing`, `dynamo-kv-hints`, `dynamo-kv-indexer`, `dynamo-multimodal`, `dynamo-router-protocol`, `dynamo-session-affinity`, `dynamo-slot-tracker`, `dynamo-tokens`, `dynamo-worker-selection`.
- Must not depend on: kv-router, the runtime, `dynamo-zmq`, any transport (axum, hyper, reqwest, zmq, prometheus); `dynamo-kv-router` composes it with those adapters.
- Hosts import this crate through its `host` module.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`; the kernel's invariants, lock order, and actor rules are in `README.md` under "Working rules".
