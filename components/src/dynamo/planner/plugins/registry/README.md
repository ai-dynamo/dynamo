# PluginRegistry

The registry tracks every plugin that can participate in a planner
pipeline, gates every Register through auth + protocol checks, evicts
stale plugins via heartbeat liveness, and coordinates with the circuit
breaker and scheduler so HOLD_LAST caches stay consistent with registry
state.

## Architecture

```
            +-------------------+
            |  register (RPC)   |<--- gateway.py (gRPC server)
            |  heartbeat (RPC)  |
            |  unregister (RPC) |
            |  list_plugins     |
            +---------+---------+
                      | method calls (single-threaded asyncio)
                      v
            +-------------------+
            | PluginRegistry    |  <-- register_internal (in-process)
            |    Server         |
            +---------+---------+
                      |
       +-- on_unregister events ---+
       |                            |
       v                            v
+---------------+          +------------------+
| CircuitBreaker|<-- open->| PluginScheduler  |---> cache_age lookup
|               |          |    - active set  |     (ListPlugins)
|               |          |    - HOLD_LAST   |
+---------------+          +------------------+
       ^
       | can_call()
       |
+---------------+
| Orchestrator  |   <-- composes all of the above
+---------------+
```

## Cache invalidation: the 6-row table

The PluginScheduler clears a plugin's HOLD_LAST cache when:

| # | Trigger | Entry point |
|---|---------|-------------|
| 1 | Client Unregister | `registry.on_unregister` → `scheduler._on_registry_unregister` |
| 2 | Heartbeat missed → auto-evict | Same as row 1 — registry-side `unregister(reason="heartbeat_missed")` is identical to client Unregister. The upstream monitor that calls it lands in a follow-up PR. |
| 3 | Circuit breaker OPEN | `circuit_breaker.on_open` → `scheduler._on_circuit_open` |
| 4 | Client-driven version upgrade (Unregister + Register) | Same as row 1 on the Unregister; fresh Register starts empty (Q6) |
| 5 | `config.reload()` | Explicit `scheduler.invalidate_cache(reason="config_reload")` |
| 6 | Planner process restart | Cache lives in memory; process exit discards it. No code required |

Each row has a dedicated must-pass test in
`tests/plugins/scheduler/test_cache_invalidation.py`.

## Auth source decision tree

```
 dev environment / single-tenant lab / pre-shared key OK?
    └── yes → static_secret  (PR #1 default)
        │
        └── share-a-secret-with-dynamo-planner K8s Secret; map the secret
            value → subject label in AuthConfig.static_secrets

 multi-cluster / mesh / zero-trust?
    └── yes → k8s_sa  (FOLLOW-UP PR — not in PR #1) — TokenReview against kube API
              or → spiffe_jwt (FOLLOW-UP PR — not in PR #1) — SPIRE JWT-SVID

 quick dev loop without real secrets?
    └── allow_unauthenticated  (emits WARNING on construction;
                               NEVER use in production)
```

`AuthConfig.trusted_sources=[]` is fail-closed — the registry refuses
every token at startup. You must opt in explicitly.

## Protocol versioning

`RegisterRequest.protocol_version` is checked against
`[protocol_version_min, protocol_version_max]` inclusive. v1 supports
`["1.0", "1.0"]` only; when introducing v1.1, set
`protocol_version_max="1.1"` to give the registry a window during which
both old and new plugins can register against the same server.

## In-process plugin registration

`register_internal(plugin_id, plugin_type, priority, instance, ...)`
skips auth and protocol checks and wraps `instance` in an
`InProcessTransport`. Two callers:

- The orchestrator (`OrchestratorEngineAdapter`) at startup, for builtin
  plugins (`is_builtin=True`).
- `NativePlannerBase`, for user plugins listed under
  `planner.plugin_registration.in_process_plugins` (`is_builtin=False`,
  loaded via `load_in_process_plugins`).

**Any future heartbeat monitor must skip every plugin whose
`transport_type` is `in_process`** — not just builtins. In-process
plugins live inside the planner process, so "heartbeat missed" is
meaningless and evicting them would drop correctly-registered user
plugins that don't emit heartbeats. (Monitor implementation deferred
to follow-up PR.)

## Single-threaded asyncio invariant

Every public method on `PluginRegistryServer`, `CircuitBreaker`, and
`PluginScheduler` MUST be called from the event loop's main task. In
particular:

- **Never** call `scheduler.record_result` or
  `scheduler.invalidate_cache` from inside an `asyncio.gather` plugin
  coroutine. Serialize after `await asyncio.gather(...)` completes.
- CircuitBreaker state and Scheduler cache are unlocked dicts — two
  concurrent mutations is undefined behaviour.

No runtime assert is shipped in PR #1 — the single-writer constraint
above is enforced only by code review. A defensive `current_task()`
check could be added in a follow-up.

## Deployment examples

### K8s Secret for `static_secret`

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: dynamo-planner-plugin-tokens
type: Opaque
stringData:
  # Each key is a shared secret; its value is the subject label surfaced
  # in audit logs and AuthIdentity.subject.
  "shared-with-team-a-SoMePHRaSe": "team-a"
  "shared-with-team-b-SoMEphRAsE": "team-b"
```

Then in `planner.plugin_registration.auth`:

```yaml
trusted_sources: [static_secret]
static_secrets:
  "shared-with-team-a-SoMePHRaSe": "team-a"
  "shared-with-team-b-SoMEphRAsE": "team-b"
```

(Real deployments should template-inject from the mounted Secret rather
than hard-code in values.yaml.)

### Dev-only `allow_unauthenticated`

```yaml
trusted_sources: [allow_unauthenticated]
```

Logs emit a WARNING on startup; production deployments SHOULD fail-fast
on seeing that warning.

## Pointers

| Topic | File |
|---|---|
| Data types + error hierarchy | `types.py` / `errors.py` |
| Auth validators | `auth/base.py` + `auth/static_secret.py` + `auth/multi.py` |
| Registry server (4 RPCs + `register_internal`) | `server.py` |
| Circuit breaker state machine | `circuit_breaker.py` |
| Scheduler + cache 6-row table | `../scheduler.py` |
| Config schema + factories | `config.py` |
| Integration tests | `../../tests/plugins/registry/test_integration.py` |
