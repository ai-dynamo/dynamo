# SGLang sidecar bridge launch scripts

These mirror `examples/backends/sglang/launch/*.sh` but replace
`python -m dynamo.sglang` with the Rust sidecar bridge talking to a stock
upstream SGLang server in `--grpc-mode` over the `sglang.grpc.scheduler`
gRPC schema.

## Status

| Script | Status | Notes |
|---|---|---|
| `agg.sh` | ✅ tested 2026-05-04 | Single SGLang+bridge, agg flow. Inference verified end-to-end. |
| `agg_router.sh` | 🟡 untested, KV routing falls back to round-robin | Multi-worker dispatch path. KV-aware routing requires `SubscribeKvEvents` wiring (POC-2). |
| `disagg.sh` | 🟡 untested | Two GPUs, one prefill + one decode. Bridge already passes `bootstrap_info` via `DisaggregatedParams`. Needs `--disaggregation-mode` flag on bridge so frontend can distinguish prefill vs decode workers. |
| `disagg_same_gpu.sh` | 🟡 untested | Variant of disagg on single GPU. |
| `agg_embed.sh` | ❌ deferred | Bridge does not yet implement `Embed` RPC. |
| `agg_agent.sh` | ❌ deferred | Streaming-sessions / agent controller not yet upstream in SGLang. |
| `agg_vision.sh`, `multimodal_*.sh`, diffusion | ❌ out of POC scope | Multimodal + diffusion. |

## Defaults

```bash
MODEL=Qwen/Qwen3-0.6B
SGLANG_VENV=/ephemeral/sglang/.venv          # stock upstream sglang + smg-grpc-servicer
DYNAMO_VENV=/ephemeral/dynamo-sglang-grpc/.venv  # has dynamo.frontend
BRIDGE_BIN=/ephemeral/cargo-target/debug/dynamo-sglang-bridge
DYN_HTTP_PORT=8000                            # frontend
SGLANG_GRPC_PORT=30000                        # sglang gRPC (in --grpc-mode, single port, no HTTP)
DYN_SYSTEM_PORT=8082                          # bridge prometheus / readiness
```

## Build

```bash
cd /ephemeral/dynamo-sglang-smg-bridge
CARGO_TARGET_DIR=/ephemeral/cargo-target cargo build -p dynamo-sglang-bridge
```

## Notes on stock upstream compatibility

SGLang main currently ships with a `smg-grpc-servicer-0.5.1` whose
`request_manager.py` imports `get_zmq_socket` from `sglang.srt.utils` —
that symbol moved to `sglang.srt.utils.network` in current main, so the
import fails. One-line patch in the installed package:

```python
# /ephemeral/sglang/.venv/lib/python3.12/site-packages/smg_grpc_servicer/sglang/request_manager.py
- from sglang.srt.utils import get_or_create_event_loop, get_zmq_socket, kill_process_tree
+ from sglang.srt.utils import get_or_create_event_loop, kill_process_tree
+ from sglang.srt.utils.network import get_zmq_socket
```

This should be filed as either a smg-grpc-servicer release bump or fixed
upstream in sglang by re-exporting `get_zmq_socket` from `sglang.srt.utils`.
