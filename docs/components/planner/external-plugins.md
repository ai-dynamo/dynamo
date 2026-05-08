---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: External Plugins
---

This page covers how to ship a plugin that runs **outside** the
planner's Python process — typically as a sidecar container in the
same Pod, or as a separate Deployment in the cluster.

For in-process plugins (Python classes registered via
`register_internal`), see the [Planner Design](../../design-docs/planner-design.md)
section on the orchestrator path; this page is only for the
cross-process model.

> The DEP-XXXX plugin architecture has two registration models for
> external plugins; both go through the same `PluginRegistryServer`
> internals (auth, dedup, circuit breaker), so behaviour is identical
> after the registration call. The difference is only in *how* the
> registration call is made.

## Two registration models

### Model 1 — Static config (W1)

The planner reads a list of plugin endpoints from its config at
startup and registers each one itself.

| When to use | Trade-offs |
|---|---|
| Plugin set is fixed at deploy time. ConfigMap-driven. | Adding/removing a plugin requires a planner restart. No partial liveness — if the plugin Pod isn't reachable when the planner boots, that entry's registration fails (logged, doesn't crash) and stays unregistered. |

`PlannerConfig` example (the relevant slice):

```yaml
scheduling:
  use_orchestrator: true
  external_plugins:
    - plugin_id: "team-a-custom-propose"
      plugin_type: "propose"
      priority: 4
      endpoint: "unix:///var/run/dynamo/team-a.sock"   # same-Pod sidecar
      auth_token: "${TEAM_A_PLUGIN_TOKEN}"             # template-injected from a Secret
      protocol_version: "1.0"
      hold_policy: "HOLD_LAST"
    - plugin_id: "team-b-budget-constrain"
      plugin_type: "constrain"
      priority: 2
      endpoint: "grpc://team-b-plugin.svc.cluster.local:9090"  # cross-Pod
      auth_token: "${TEAM_B_PLUGIN_TOKEN}"
```

The planner calls `await registry.register(...)` for each entry.
Per-entry failures (auth rejected, plugin Pod unreachable, protocol
mismatch, duplicate plugin_id) are **isolated**: they're logged with
the registry's reject reason and the next entry is still attempted.

The startup hook returns
`(num_accepted, [(plugin_id, reject_reason), ...])`; planners log a
one-liner summary so operators can grep:

```
plugin_bootstrap: accepted=2 failed=1 failures=[(team-c-old, "protocol_version_unsupported: requested=0.9, supported=[1.0,1.0]")]
```

### Model 2 — gRPC gateway self-register (W2)

The planner runs a gRPC server hosting the
`dynamo.planner.plugin.v1.PluginRegistry` service; plugin processes
open a gRPC channel and call `Register` themselves.

| When to use | Trade-offs |
|---|---|
| Plugins come and go without planner restart; CI pipelines that bring up plugins as part of a test fixture; plugins that need to declare their own endpoint dynamically (e.g. ephemeral Pod IP). | Requires the plugin author to ship a few lines of gRPC client code, and requires the planner to expose a port (or UDS socket) for the gateway. |

Planner-side wire-up (typically inside `NativePlannerBase` startup):

```python
from dynamo.planner.plugins.registry.gateway import start_gateway_server

grpc_server, listen = await start_gateway_server(
    server=orchestrator._registry,                    # the in-process registry
    listen=f"unix:{config.registry_gateway_socket}",  # same-Pod
    # OR
    # listen="0.0.0.0:9099",                          # cross-Pod
    # server_credentials=ssl_server_credentials_for_mtls(...),
)
# ...
await grpc_server.stop(grace=0.5)  # on planner shutdown
```

Plugin-side self-register (the relevant slice — see
`tests/integration/external_plugin_subprocess_runner.py` for the full
runnable example):

```python
import grpc
from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg

async with grpc.aio.insecure_channel("unix:/var/run/dynamo/registry.sock") as ch:
    stub = pbg.PluginRegistryStub(ch)
    resp = await stub.Register(pb.RegisterRequest(
        plugin_id="my-team-propose",
        plugin_type="propose",
        priority=5,
        endpoint="unix:/var/run/dynamo/my-team.sock",  # where my plugin's gRPC server listens
        auth_token=os.environ["MY_PLUGIN_TOKEN"],
        protocol_version="1.0",
        hold_policy=pb.HoldPolicy.HOLD_LAST,
        version="v1.2.3",
    ))
    if not resp.accepted:
        raise SystemExit(f"register rejected: {resp.reject_reason}")
```

Plugins are expected to send periodic `Heartbeat` calls (default
heartbeat interval is configured per-plugin via the registry's
`HeartbeatMonitor`). Missed heartbeats trigger automatic
`Unregister` after a configurable threshold; this is the K8s
"Pod went away" recovery path.

On graceful shutdown the plugin should call `Unregister` so the
HOLD_LAST cache for its `plugin_id` is invalidated immediately
(without waiting for heartbeat eviction).

## Choosing between W1 and W2

Both models are supported simultaneously — you can have static
entries from ConfigMap *and* dynamic self-registration via the
gateway. Two rules of thumb:

1. **Stable production plugins (vendor / platform team)** → W1.
   ConfigMap is the deployment artifact; restart cost is acceptable
   because the plugin set rarely changes.
2. **Experimental / per-team plugins (ML researchers shipping
   custom propose logic)** → W2. The plugin team ships its own
   image + Deployment + ServiceAccount; the planner picks them up
   automatically when they start.

## Crash isolation contract

A misbehaving external plugin must never take down the planner.
The architecture enforces this at three layers:

| Layer | What it does |
|---|---|
| `_GrpcTransportBase.call()` (per-plugin RPC timeout) | A hung plugin call raises `PluginTimeoutError` after `request_timeout_seconds` (default 5s); the stage continues with the remaining plugins. |
| `CircuitBreaker` (per-plugin failure budget) | After N consecutive failures (default 5), the plugin's circuit OPENs and it's silently skipped on subsequent ticks; `HeartbeatMonitor` HALF_OPENs after a cooldown so transient failures self-heal. |
| `tick_max_duration_seconds` (whole-tick deadline) | Even with all per-plugin timeouts elapsed, the orchestrator aborts the tick after this outer deadline (default 30s) and logs `skip_tick_timeout`; next tick runs from a clean state. |

The `tests/integration/test_external_plugin_subprocess_e2e.py`
suite covers the SIGKILL → circuit-OPEN path with a real subprocess
plugin to lock this behaviour against regressions.

## Securing the wire (mTLS)

For cross-Pod (`grpc://`) endpoints, mTLS is the default expected
configuration:

- Planner side: pass `ssl_server_credentials(...)` to
  `start_gateway_server(...)`.
- Plugin side: open a `grpc.aio.secure_channel(...)` with the matching
  trust bundle.
- Cert provisioning: cert-manager 3-key convention
  (`tls.crt` / `tls.key` / `ca.crt` mounted from a Secret) — see
  `plugins/transport/_mtls.py` for the loader.

Plain `grpc://` (no mTLS) is gated behind
`TransportConfig.allow_insecure_grpc=True` and emits a WARNING log
on transport construction; production deployments should fail-fast
on that warning.

For same-Pod sidecar (`unix://`) endpoints, the Pod boundary is the
trust boundary — mTLS over UDS adds no security and is silently
ignored if configured.

## See also

- [Planner Guide](planner-guide.md) — main config reference,
  `scheduling.use_orchestrator` flag
- [Orchestrator rollout runbook](orchestrator-rollout.md) — staged
  flag-flip guidance
- [Observability](observability.md) — `dynamo_planner_plugin_*`
  Prometheus metrics that surface external plugin health
- `components/src/dynamo/planner/plugins/registry/README.md` —
  decision tree for auth source (`static_secret` / `k8s_sa` /
  `spiffe_jwt` / `allow_unauthenticated`)
- `components/src/dynamo/planner/plugins/transport/README.md` —
  per-transport threat model (`inproc` / `unix` / `grpc` / `grpc+mTLS`)
