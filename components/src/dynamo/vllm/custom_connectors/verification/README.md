# Verification of `vllm:nixl_num_pending_sends` metric

End-to-end checks for `dynamo.vllm.custom_connectors.nixl_with_pending_metrics`,
exercised against the **exact** versions the customer is running.

## Environment

```
Dynamo:  v1.1.1-rc1   ← matches "Dynamo 1.1.0 vllm runtime on 1.0.1 operator"
vLLM:    0.19.0        ← exact pin in Dynamo 1.1.1 pyproject.toml
NIXL:    1.1.0
Python:  3.12          (uv-managed venv at ~/repos/amz-ads/.venv)
GPU:     1× RTX 5880 Ada 48 GB
```

Runtime is the host venv (no container) — but the **code paths exercised are
identical** to what the customer's container runs, because we install the
same vLLM wheel and the same NIXL bindings and PYTHONPATH-import the same
Dynamo connector source.

## Files

| File | Purpose | Needs GPU |
|---|---|---|
| `e2e_real_vllm.py` | Unit-level: load real vLLM 0.19.0 base classes, drive the override logic with a fake worker | No |
| `factory_resolution.py` | Confirm `KVConnectorFactory` resolves our class via `kv_connector_module_path` | No |
| `gpu_smoke_test.py` | Start ONE vLLM process with our connector; confirm gauges appear at `/metrics` | Yes |
| `single_gpu_pd_test.sh` | Start P+D on the same GPU; send a real chat request through the proxy; kill decode; observe the gauges | Yes |
| `test_timeout_cliff.sh` | Same as above but with `VLLM_NIXL_ABORT_REQUEST_TIMEOUT=15` to walk through the leak window | Yes |
| `test_wake_engine.sh` | Confirm the timeout sweep fires when fresh traffic wakes the prefill engine | Yes |
| `toy_proxy_server.py` | Upstream-vLLM proxy that orchestrates a P→D handoff (single chat request through both servers) | No |

## How to run

```bash
cd ~/repos/amz-ads
. .venv/bin/activate

# Pure-Python checks (no GPU needed)
CUDA_VISIBLE_DEVICES="" VLLM_TARGET_DEVICE=cpu \
    python verification/e2e_real_vllm.py
CUDA_VISIBLE_DEVICES="" VLLM_TARGET_DEVICE=cpu \
    python verification/factory_resolution.py

# Single-process smoke (1 GPU)
python verification/gpu_smoke_test.py

# Full P+D round-trip with our connector (1 GPU, two vLLM processes sharing it)
bash verification/single_gpu_pd_test.sh
```

## Last observed result (host venv, GPU 0)

From `single_gpu_pd_test.sh`:

```
[T0] Before any traffic
  vllm:nixl_num_pending_sends{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 0.0
  vllm:nixl_num_in_process_reqs{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 0.0

[T2] After successful P→D round-trip
  vllm:nixl_num_pending_sends{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1.0
  vllm:nixl_num_in_process_reqs{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1.0

[T5] After decode killed mid-transfer + 5s
  vllm:nixl_num_pending_sends{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1.0   ← stays pinned (this is the leak signal!)
  vllm:nixl_num_in_process_reqs{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1.0
  vllm:nixl_num_kv_expired_reqs_total{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 0.0
```

The customer's previously invisible failure mode — *"100 % KV usage on
prefill while standard scheduler-state metrics read zero"* — is now
diagnosable via `vllm:nixl_num_pending_sends`.

## What's NOT covered by host-venv testing

- **Container image layering / file permissions** — would only matter if we
  baked the connector into an image; activation via `kv_connector_module_path`
  + volume mount avoids this entirely.
- **Multi-host NIXL transport** (EFA/libfabric) — single-host UCX loopback is
  what's exercised here.
- **Real production load** (RPS, ISL/OSL distributions) — would need the
  customer's load generator.

For the customer's deployment, the activation is one line in the prefill DGD:

```yaml
- --kv-transfer-config
- '{"kv_connector":"NixlConnectorWithPendingMetrics","kv_role":"kv_both","kv_connector_module_path":"dynamo.vllm.custom_connectors.nixl_with_pending_metrics"}'
```

…and the two files under `dynamo/components/src/dynamo/vllm/custom_connectors/`
need to land in their image (e.g. via the normal Dynamo source-build pipeline,
or as a layer overlay).
