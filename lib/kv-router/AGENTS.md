# lib/kv-router

`dynamo-kv-router` is the composition crate: it assembles the kernel and the stages into a running router behind features (HTTP servers, the ZMQ KV-event listener and registry, the replica mesh, peer recovery, Prometheus exposition), carries the flat `KvRouterConfig` and the YAML policy file, and is the one crate plugins name through `plugins::prelude`. Subdirectories keep their own `AGENTS.md` where one exists.

- Depends on (Dynamo crates): the gate's composition row `dynamo-admission`, `dynamo-kv-events`, `dynamo-kv-hashing`, `dynamo-kv-hints`, `dynamo-kv-indexer`, `dynamo-multimodal`, `dynamo-router-protocol`, `dynamo-scheduler`, `dynamo-session-affinity`, `dynamo-slot-tracker`, `dynamo-worker-selection`; also `dynamo-tokens`, `dynamo-truthy`, and `dynamo-zmq` behind the ZMQ features.
- Must not depend on: the runtime; a transport crate outside a feature (with no features its tree carries none of axum, hyper, reqwest, zmq).
- Hosts import this crate directly; plugins import `plugins::prelude` and nothing else.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`; the migration map and the working rules are in `README.md`.
