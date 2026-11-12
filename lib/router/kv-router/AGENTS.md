# lib/router/kv-router

`dynamo-kv-router` is the composition crate: it assembles the kernel and the stages into a running router, puts every transport (HTTP, ZMQ, the replica mesh, Prometheus) behind a feature, and is the one crate plugins name through `plugins::prelude`. Subdirectories keep their own `AGENTS.md` where one exists.
