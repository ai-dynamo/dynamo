# Plugin Transport

Plugin RPC transport abstractions for the Dynamo Planner plugin framework.

Two transports under one `PluginTransport` ABC; the orchestrator pipeline
driver treats them uniformly via `await plugin.transport.call(method, request)`.

A dedicated `UdsTransport` and mTLS for `GrpcTransport` are deferred to a
follow-up PR — see "Deferred" section below.

## Transports (shipped in PR #1)

| Transport | Endpoint scheme | Use case |
|---|---|---|
| `InProcessTransport` | `inproc://<plugin_id>` | Built-in plugins, in-process user plugins, replay/test; **first-class production transport, NOT a test fallback** |
| `GrpcTransport` | `grpc://host:port` | Out-of-process plugin (same Pod, cross-Pod, or cross-node). PR #1 supports plaintext only, gated behind `allow_insecure_grpc=true`; mTLS lands in a follow-up PR. |

### Choosing a transport — decision tree

```
plugin and orchestrator in same process?
  ├─ YES → InProcessTransport (zero RPC overhead, builtin plugins, in_process user plugins)
  └─ NO  → GrpcTransport (set allow_insecure_grpc=true for plaintext in
           PR #1; mTLS lands in a follow-up PR)
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
      allow_insecure_grpc: false      # default refuse plaintext grpc (PR #1
                                      # has no mTLS path yet — setting this to
                                      # true is the only way to use grpc:// in
                                      # PR #1; logs WARNING on startup)
      request_timeout_seconds: 5
      keepalive_time_ms: 30000
      max_message_size_bytes: 10000000
  scheduling:
    clock:
      type: wall                       # virtual only allowed in test/replay
```

The mTLS config block (`grpc_mtls.enabled` / `secret_mount_path` etc.)
documented in earlier drafts is **not** shipped in PR #1 and is not a
field on `TransportConfig`. It lands together with the cert-manager /
SPIFFE auth path in a follow-up PR.

## Per-plugin timeout (no stage-level wait_for needed)

Each transport implements a **per-plugin RPC timeout** at the call site:

| Transport | Where | Code |
|---|---|---|
| `InProcessTransport` | `in_process.py` | `await asyncio.wait_for(coro, self.timeout_seconds)` |
| `GrpcTransport` | `_grpc_base.py` | `await asyncio.wait_for(rpc(request), self.timeout_seconds)` |

**Implication for the pipeline driver**:

- The pipeline driver invokes plugins via `asyncio.gather(*[plugin.transport.call(...) for ...])`
- The driver **MUST NOT** add an additional stage-level `asyncio.wait_for` —
  per-plugin timeout already prevents any single plugin from dragging
  down the whole stage
- The whole-tick `tick_max_duration_seconds` is the outermost safety net
  (catches systemic deadlock); per-stage budget is intentionally NOT
  introduced in this version (left as a follow-up)

Default `request_timeout_seconds = 5.0`, applied uniformly to every
plugin in PR #1 (a per-plugin override field was prototyped on
`RegisterRequest` but not plumbed into `make_transport_for_endpoint`,
so it was removed before any client shipped — see `plugin.proto`
"reserved 11, 12"). A future PR may re-introduce a per-plugin timeout
at a new tag with the missing plumbing.

## Sync plugin red line (`InProcessTransport`)

`InProcessTransport` supports **both** `async def` and sync (`def`) plugin
methods; sync methods dispatch via `asyncio.to_thread` to avoid blocking
the orchestrator event loop.

**Hard rule**: sync plugin methods MUST NOT do blocking IO (HTTP, file,
`time.sleep > 100ms`). Default thread pool is small (~32 threads); a few
slow sync plugins doing blocking IO will exhaust the pool and stall the
orchestrator.

If your plugin needs IO, write it as `async def`.

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
| `PluginConnectionError` | gRPC channel down / unreachable | Mark plugin unreachable; on next tick attempt reconnect |
| `PluginUnknownMethodError` | Method name not registered on plugin | Log + treat as plugin contract violation |
| `PluginSerializationError` | bytes-level (de)serialization failed (proto schema mismatch / FpmData decode); empty oneof is NOT this error — see `plugins/proto/v1/README.md` "result oneof empty" | Log + circuit breaker increment |
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

### `GrpcTransport` (PR #1 state)

Trust assumption: **cross-Pod / cross-node** — anyone with network access
to the gRPC port could try to call.

**PR #1 ships plaintext gRPC only**, gated behind
`allow_insecure_grpc=true` on `TransportConfig`. `make_transport_for_endpoint`
refuses to build a `GrpcTransport` for a `grpc://` endpoint unless the
flag is set, and a WARNING is logged when it is. **mTLS is not shipped
in PR #1** — there is no certificate-loading code path, no `grpc_mtls`
config block, and no in-process cert hot reload.

Until the follow-up PR adds mTLS, the operational guidance is:

- Use `InProcessTransport` for builtin and in-process plugins (no
  network exposure, no flag needed).
- For out-of-process plugins, set `allow_insecure_grpc=true` and pair
  it with K8s NetworkPolicy / Pod-to-Pod identity to restrict who can
  reach the gRPC port. Plaintext on the wire means the channel layer
  contributes no authentication.
- Plugin authentication via `RegisterRequest.auth_token` (validated by
  `PluginRegistry` — see `plugins/registry/`) is the only auth in PR #1;
  transport-level mTLS will layer on top of it later, not replace it.
- gRPC `keepalive_time_ms=30000` detects dropped connections quickly.

## Deferred

The following are **not** shipped in PR #1 and will land in a follow-up:

- **`UdsTransport`** as a separate transport class for plugin endpoints —
  plugin endpoints are limited to `inproc://` and `grpc://`. (The gateway
  registration server's *listen address* can already be a UDS path via
  gRPC's URI scheme — see `gateway.start_gateway_server` — but that is a
  separate mechanism from plugin transport endpoints.)
- **mTLS for `GrpcTransport`** including cert-manager / certificateSecret
  convention (`tls.crt` / `tls.key` / `ca.crt`), in-process cert hot reload,
  and the matching `grpc_mtls.*` config block on `TransportConfig`.

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
- `tests/plugins/transport/test_transport_contract.py` — transport contract acceptance suite (round-trip equivalence + byte-equal cross-transport)
- `tests/plugins/clock/test_clocks.py` — Clock unit tests
