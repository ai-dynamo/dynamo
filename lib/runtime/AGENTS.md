<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `dynamo-runtime` — Agent Guide

`lib/runtime` is the crate every other Dynamo component is built on. It owns the
distributed object model — runtime, namespace, component, endpoint — and the three
communication planes that connect processes to each other. If you are looking for
where "request plane", "event plane", or "discovery plane" stop being words and start
being code, this crate is the answer.

Read [`../AGENTS.md`](../AGENTS.md) for the workspace map and the per-crate build
commands.

## The object model

`DistributedRuntime` (`src/distributed.rs`) is the top of a four-level hierarchy,
documented in the module docs of [`src/component.rs`](src/component.rs):

```text
DistributedRuntime      cluster-wide handle: transports, discovery, lifecycle
  └── Namespace         logical grouping that isolates one model deployment
        └── Component   discoverable logical unit of workers (a frontend, a worker)
              └── Endpoint   network-accessible service on that component
```

Two local types sit underneath it: `Runtime` (`src/runtime.rs`) owns the Tokio
runtimes and the cancellation token, and `Worker` (`src/worker.rs`) is the process
entrypoint that constructs one. All three are re-exported from
[`src/lib.rs`](src/lib.rs).

> [!NOTE]
> `DistributedRuntime` is not a process singleton. Constructing it twice yields
> independent instances with distinct discovery connection IDs. One per service
> replica is a soft invariant; multiple instances in one process exist for
> single-process test topologies and for the mocker. The doc comment on the struct in
> `src/distributed.rs` is the authoritative statement of this.

For the full picture — how the frontend, router, and workers compose over this model —
read [distributed runtime](../../docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/distributed-runtime.md)
and [architecture flow](../../docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture-flow.md).

## The three planes

The planes are independent: each is selected by its own environment variable, and any
combination is valid.

| Plane | Carries | Selected by | Options | Code |
|-------|---------|-------------|---------|------|
| Request | RPC between components (frontend → router → worker) | `DYN_REQUEST_PLANE` | `tcp` (default), `nats` | `src/transports/tcp.rs`, `src/transports/nats.rs`, `src/pipeline/network/` |
| Event | KV cache events, worker load metrics, sequence tracking — pub/sub | `DYN_EVENT_PLANE` | `zmq` (default), `nats` | `src/transports/event_plane/` |
| Discovery | Where components register and find each other | `DYN_DISCOVERY_BACKEND` | `etcd` (default), `kubernetes`, `file`, `mem` | `src/discovery/`, `src/transports/etcd/` |

The selection logic for all three lives in `src/distributed.rs`:
`RequestPlaneMode::from_env` reads `DYN_REQUEST_PLANE`, `DistributedConfig::from_settings`
reads `DYN_DISCOVERY_BACKEND`, and the `DYN_EVENT_PLANE` mapping is deliberately kept as
a single function so that every caller gets the same answer. Extend the parsing there,
not at a call site.

Canonical prose for each plane, including deployment guidance and the interaction
between them:

- [Request plane](../../docs/fern/pages/developer-guide/knowledge-base/concepts/communication-planes/request-plane.md)
- [Event plane](../../docs/fern/pages/developer-guide/knowledge-base/concepts/communication-planes/event-plane.md)
- [Discovery plane](../../docs/fern/pages/developer-guide/knowledge-base/concepts/communication-planes/discovery-plane.md)

## Source layout

| Path | Contents |
|------|----------|
| `src/distributed.rs` | `DistributedRuntime`, `DistributedConfig`, plane and backend selection |
| `src/component/` | `Component`, `Namespace`, `Endpoint`, `Client`, registry |
| `src/transports/` | `tcp.rs`, `nats.rs`, `zmq.rs`, `etcd/`, `event_plane/` |
| `src/discovery/` | Discovery trait plus the etcd/KV-store, Kubernetes, and mock backends |
| `src/pipeline/` | Engine pipeline: nodes, `network/ingress`, `network/egress`, `PushRouter`, `RouterMode` |
| `src/metrics/`, `src/system_status_server.rs` | Prometheus registry and the system status HTTP server |
| `src/system_health.rs`, `src/health_check.rs` | Health reporting |
| `src/storage/`, `src/protocols/`, `src/utils/` | KV store abstraction, wire protocols, shared helpers |
| `docs/rayon-tokio-strategy.md` | Why compute work is split between Rayon and Tokio |
| `examples/` | Runnable examples, including `hello_world` |

## Working in this crate

```bash
cargo build -p dynamo-runtime
cargo test  -p dynamo-runtime
```

Changes here reach Python through the PyO3 extension module, not through source: the
`dynamo.runtime` package re-exports `DistributedRuntime`, `Endpoint`, `Client`, and
`Context` from the compiled `dynamo._core`. After a change that Python callers depend
on, rebuild the module or the Python side will keep running the old code:

```bash
cd lib/bindings/python && maturin develop --uv
```

Because everything depends on this crate, treat changes to the plane selection, the
component hierarchy, or the wire protocols as cross-cutting: a rename here is a rename
in `lib/llm`, the bindings, and the Python components. `lib/llm/CLAUDE.md` records a
worker/frontend wire-compatibility policy that constrains protocol changes visible
across process boundaries — read it before changing anything on the wire.
