<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shutdown validation

Run from a development environment with Rust, rebuilt editable Dynamo bindings, pytest, pytest-asyncio, and pytest-timeout:

```bash
cd lib/bindings/python
maturin develop --uv
cd ../../..
SHUTDOWN_PYTHON=.venv/bin/python bash tests/runtime/validate_shutdown.sh
```

The runner checks runtime teardown and worker stage contracts, then exercises shutdown end to end through real Rust bindings, TCP streams, OS signals, and subprocess exits. It needs permission to bind local sockets but no GPU, model download, etcd, or NATS service.

| Scenario | Required outcome |
| --- | --- |
| Concurrent runtime shutdown callers | All await the same teardown; cancelling one waiter does not cancel teardown. |
| SDK worker hosted by another application | Clean shutdown returns and the host survives beyond the former watchdog deadline. |
| SDK cleanup holds the GIL | The native watchdog still forces exit code 70. |
| Python pull and push streaming | SIGTERM removes discovery, completes admitted work, then cleans the engine and runtime. |
| Positive remaining budget below five seconds | Cleanup gets its five-second reserve instead of being truncated to the remainder. |
| Second SIGTERM | Immediate exit code 70. |
| Idle and slow embedding children | A process-group signal is forwarded once; idle event loops wake, children exit cleanly, and parent cleanup follows. |
| Stale router addresses a closed admission gate | Both response transports preserve the typed Unavailable rejection. |

`shutdown_probe.py` is the subprocess helper; use pytest or the runner rather than invoking its signal modes in a shared process group. Every subprocess has a timeout, and embedding children have their own bounded lifetime.

These checks use a CPU probe engine and the shared Python shutdown coordinator used by vLLM, SGLang, and TensorRT-LLM. The embedding scenarios import the production process supervisor while stubbing unused vLLM engine-construction imports. They do not instantiate WorkerFactory or validate GPU engine teardown, distributed KV transfer, external discovery propagation, or Kubernetes termination. Those require backend-specific serving environments and remain separate deployment validation.

## Real-engine container validation

`validate_gpu_shutdown.py` launches the actual Python backend and Dynamo HTTP frontend using the existing `ManagedProcess` harness. It sends SIGTERM after the first token of a 512-token GPU inference stream and requires a normal terminal response, exit code 0, completed engine/runtime cleanup, and no live engine descendants. File discovery and TCP keep the run isolated from existing etcd/NATS deployments. Logs and `result.json` are retained in a unique directory.

Rebuild the bindings first, then run from the checkout root. Mount a Hugging Face hub cache containing `Qwen/Qwen3-0.6B`, select a compatible backend image, and provide a results directory writable by the image's user:

```bash
docker run --rm --gpus all --shm-size=2g --entrypoint python3 \
  -v "$PWD:/shutdown:ro" \
  -v /path/to/huggingface/hub:/models:ro \
  -v /path/to/results:/results \
  -e PYTHONPATH=/shutdown/lib/bindings/python/src:/shutdown/components/src:/shutdown \
  -e HF_HUB_CACHE=/models -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  BACKEND_IMAGE /shutdown/tests/runtime/validate_gpu_shutdown.py BACKEND --output /results
```

`BACKEND` is `vllm`, `sglang`, or `trtllm`. Run sequentially on a single GPU. The bind mounts deliberately select both the checkout's Python source and its rebuilt `_core` library, not the image's released Dynamo code. This aggregated-serving check does not replace multi-node disaggregated KV-transfer or Kubernetes validation.

### Verified run: September 21, 2026

Validated source commit `4a44bca2e7` on one NVIDIA RTX 5880 Ada (48 GB), using Qwen3-0.6B and the container command above. All three runs completed the 512-token response, delivered 511 chunks after SIGTERM, exited with code 0, completed cleanup and runtime teardown, and left no live engine children.

| Backend | Container tag | Local image ID | SIGTERM to exit |
| --- | --- | --- | --- |
| vLLM 0.28.0 | `vllm/vllm-openai:latest` | `609a5b463503` | 4.38 s |
| SGLang 0.5.16 | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.1` | `c916cbde6c23` | 3.96 s |
| TensorRT-LLM 1.3.0rc22 | `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.4.1` | `88430fff8555` | 3.62 s |

The `latest` tag is mutable; the image ID identifies the image actually tested. The older cached `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1` image contains vLLM 0.26.0 and fails inference before shutdown because this source requires `vllm.exceptions.VLLMClientError`; that attempt is not counted as a passing validation. The successful vLLM run used the newer image without a compatibility shim or engine mock.
