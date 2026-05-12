# Plugin Transport

Plugin RPC transport abstractions for **DEP-XXXX Dynamo Planner Plugin
Architecture** (v11), implementing PR 2.

Three transports under one `PluginTransport` ABC; orchestrator pipeline
driver (PR 5) treats them uniformly via `await plugin.transport.call(method, request)`.

## Transports

| Transport | Endpoint scheme | Use case |
|---|---|---|
| `InProcessTransport` | `inproc://<plugin_id>` | Built-in plugins, in-process user plugins, replay/test; **first-class production transport, NOT a test fallback** |
| `UdsTransport` | `unix:///path/to/sock` | Same-Pod sidecar plugin (default recommended deployment) |
| `GrpcTransport` | `grpc://host:port` | Cross-Pod plugin; mTLS required by default |

### Choosing a transport — decision tree

```
plugin and orchestrator in same process?
  ├─ YES → InProcessTransport (zero RPC overhead, builtin plugins, in_process user plugins)
  └─ NO
      ├─ same K8s Pod (sidecar)?
      │   └─ YES → UdsTransport (Pod boundary is trust boundary; no TLS needed)
      └─ NO (cross-Pod / cross-node)?
          └─ YES → GrpcTransport with mTLS (mandatory unless allow_insecure=True for dev)
```

## `Clock` abstraction

All time access in orchestrator (PR 5) and PluginRegistry (PR 3) MUST go
through `Clock` — direct `time.time()` / `time.monotonic()` /
`asyncio.sleep` is forbidden (lint check enabled in PR 5 5-9).

| Implementation | Use |
|---|---|
| `WallClock` | Production |
| `VirtualClock` | Replay / test; `advance(N)` warps time forward |

**Production safety**: `make_clock()` rejects `clock.type=virtual` unless
`DYNAMO_PLANNER_TEST=1` is set in environment. Replay code paths set
this env var explicitly.

Two time sources:

- `now()`: epoch float — use for audit log timestamps, `decision_id`
- `monotonic()`: monotonic float — use for duration / scheduling
  (immune to NTP / clock skew)

## Configuration

```yaml
planner:
  plugin_registration:
    transport:
      allow_insecure_grpc: false      # default refuse plaintext grpc
      grpc_mtls:
        enabled: true
        secret_mount_path: /var/run/dynamo/planner-tls
        # K8s Secret with three keys: tls.crt / tls.key / ca.crt
        # (matches dynamo platform cert-manager / certificateSecret convention)
      request_timeout_seconds: 5
      keepalive_time_ms: 30000
      max_message_size_bytes: 10000000
  scheduling:
    clock:
      type: wall                       # virtual only allowed in test/replay
```

## Per-plugin timeout = M-7 OR option (no stage-level wait_for needed)

PR 2 implements **per-plugin RPC timeout** at the transport layer:

| Transport | Where | Code |
|---|---|---|
| `InProcessTransport` | `in_process.py` | `await asyncio.wait_for(coro, self.timeout_seconds)` |
| `UdsTransport` / `GrpcTransport` | `_grpc_base.py` | `await asyncio.wait_for(rpc(request), self.timeout_seconds)` |

This is the **OR option** in v11 M-7 (DEP main doc § "v11 实施细节注解"):

> **OR**: 保留整 tick budget 但 stage 内 `asyncio.wait_for(plugin_call, request_timeout_seconds)` 即可
> —— 单 plugin 不能拖死 stage. 推荐 OR 方案.

**Implication for PR 5 stage driver** (5-4 sub-task):

- The pipeline driver invokes plugins via `asyncio.gather(*[plugin.transport.call(...) for ...])`
- The driver **MUST NOT** add an additional stage-level `asyncio.wait_for` —
  per-plugin timeout already prevents any single plugin from dragging
  down the whole stage
- The whole-tick `tick_max_duration_seconds` is the outermost safety net
  (catches systemic deadlock); per-stage budget is intentionally NOT
  introduced in v1 (M-7 says "v1 不做，标 follow-up")

Default `request_timeout_seconds = 5.0` (configurable per-plugin via
`RegisterRequest.request_timeout_seconds`).

## Sync plugin red line (`InProcessTransport`)

`InProcessTransport` supports **both** `async def` and sync (`def`) plugin
methods; sync methods dispatch via `asyncio.to_thread` to avoid blocking
the orchestrator event loop.

**Hard rule**: sync plugin methods MUST NOT do blocking IO (HTTP, file,
`time.sleep > 100ms`). Default thread pool is small (~32 threads); a few
slow sync plugins doing blocking IO will exhaust the pool and stall the
orchestrator.

If your plugin needs IO, write it as `async def`.

PR 7 production config will additionally cap the executor with
`executor_max_workers <= 8` to bound damage from misbehaving sync plugins.

## Wire-message conversion (Pydantic ↔ proto)

The pipeline emits **Pydantic** stage requests (so it can keep using
attribute-style access on the way back); gRPC stubs need **proto**
messages. `_GrpcTransportBase.call()` handles the conversion at the
wire boundary using `_proto_bridge.pydantic_to_proto` /
`proto_to_pydantic`:

- **Pydantic in → Pydantic out** — pipeline path. Request gets converted
  to proto before send; response gets converted back to Pydantic before
  return.
- **Proto in → proto out** — passthrough. Used by the transport
  contract test which asserts byte-equal proto round-trip across all
  four transports.

The conversion was **missing in PR 2 ship** and only surfaced when the
external-plugin e2e test (`tests/integration/test_external_plugin_e2e.py`)
first drove a real gRPC plugin via the pipeline. Before the fix, every
external plugin call failed at `Message.SerializeToString` because the
gRPC stub received a Pydantic instance. The in-process transport
side-stepped this because Pydantic objects flow through unchanged.

If you add a new wire transport (TCP, QUIC, etc.), inherit from
`_GrpcTransportBase` so you get the bridge for free; if you must roll
your own, replicate the same Pydantic-vs-proto branch.

## Error contract

ALL `call()` failures raise a `PluginCallError` subclass — orchestrator
relies on this to never need a bare `except` clause.

| Subclass | When | Orchestrator response (PR 5) |
|---|---|---|
| `PluginTimeoutError` | `asyncio.wait_for` exceeded `timeout_seconds` | Increment circuit breaker failure count |
| `PluginConnectionError` | UDS socket missing, gRPC channel down, mTLS handshake failed | Mark plugin unreachable; on next tick attempt reconnect |
| `PluginUnknownMethodError` | Method name not registered on plugin | Log + treat as plugin contract violation |
| `PluginSerializationError` | (de)serialization failed; oneof empty | Log + circuit breaker increment |
| `PluginCallError` | Catch-all (plugin internal exception, etc.) | Log + circuit breaker increment |

## Threat Model

### `InProcessTransport`

Trust assumption: **plugin code shares the orchestrator process**. Any
Python module loaded as in_process plugin has full Python-level access
to orchestrator state (limited only by Python's lack of memory protection).

**Mitigation**:

- `in_process_plugins` discovery is **config-only** (no setuptools
  entrypoint auto-discovery) — operator must explicitly list each plugin
  module/class in YAML, preventing "pip install rogue-plugin" silent injection
- All in_process plugins go through the same `PluginRegistry` view
  (`ListPlugins` shows them with `transport=in_process`, `is_builtin=false`)
  for audit visibility
- Sync plugin red line above protects against blocking-IO denial of service

### `UdsTransport`

Trust assumption: **co-located in the same K8s Pod**. Pod boundary is the
trust boundary.

**Mitigation**:

- Socket file permission `0660` + shared GID — only the Pod's UID/GID can
  access the socket
- No TLS overhead (Pod boundary is sufficient; TLS would add CPU with no
  security benefit)
- Cross-namespace / cross-Pod cannot reach UDS path (mounted only inside
  the Pod's filesystem)

### `GrpcTransport` + mTLS

Trust assumption: **cross-Pod / cross-node** — anyone with network access
to the gRPC port could try to call.

**Mitigation**:

- mTLS **required by default** (`allow_insecure=True` is dev-only; logs
  WARNING on init)
- Both sides verify each other's certificate chain
- Reuses dynamo platform cert-manager / certificateSecret convention
  (`tls.crt` / `tls.key` / `ca.crt` three-key K8s Secret mount) — auto
  rotation handled by cert-manager triggering Pod restart
- gRPC `keepalive_time_ms=30000` detects dropped connections quickly

**Caveats**:

- v1 does not implement in-process cert hot reload (`cert_reload_inotify=false`).
  cert-manager-driven Pod restart handles rotation. v2 follow-up may add inotify.
- Plugin authentication via `RegisterRequest.auth_token` is enforced
  separately by `PluginRegistry` (PR 3); transport mTLS is the
  channel-level authentication only.

## Adding a new transport

1. Subclass `PluginTransport` in `transport/<name>.py`
2. Implement `call(method, request)` and `close()` per the contract
3. All failures must raise `PluginCallError` subclasses (no naked exceptions)
4. Add to `transport/__init__.py` exports
5. Update `transport/config.py` `make_transport_for_endpoint` factory + add
   endpoint scheme detection
6. Add a parametrized variant to
   `tests/plugins/transport/test_transport_contract.py` — your transport
   MUST pass `test_round_trip_equivalence` and
   `test_byte_equal_response_across_transports` (byte-equal with all other
   transports for the same input)

## References

- `dynamo/planner/plugins/proto/v1/` — plugin proto schema
- `tests/plugins/transport/test_transport_contract.py` — 50-case acceptance
- `tests/plugins/clock/test_clocks.py` — Clock unit tests
