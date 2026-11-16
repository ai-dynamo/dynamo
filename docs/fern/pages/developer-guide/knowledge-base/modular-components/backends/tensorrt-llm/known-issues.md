---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Known Issues and Mitigations
subtitle: Known issues for TensorRT-LLM in Dynamo, including disaggregated KV cache exhaustion and driver mismatch errors, with mitigations.
---

For general TensorRT-LLM features and configuration, see the [Reference Guide](reference-guide.md).

---

### KV Cache Exhaustion Causing Worker Deadlock (Disaggregated Serving)

**Issue:** In disaggregated serving mode, TensorRT-LLM workers can become stuck and unresponsive after sustained high-load traffic. Once in this state, workers require a pod/process restart to recover.

**Symptoms:**
- Workers function normally initially but hang after heavy load testing
- Inference requests get stuck and eventually timeout
- Logs show warnings: `num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache`
- Error logs may contain: `asyncio.exceptions.InvalidStateError: invalid state`

**Root Cause:** When `max_tokens_in_buffer` in the cache transceiver config is smaller than the maximum input sequence length (ISL) being processed, KV cache exhaustion can occur under heavy load. This causes context transfers to timeout, leaving workers stuck waiting for phantom transfers and entering an irrecoverable deadlock state.

**Mitigation:** Ensure `max_tokens_in_buffer` exceeds your maximum expected input sequence length. Update your engine configuration files (e.g., `prefill.yaml` and `decode.yaml`):

```yaml
cache_transceiver_config:
  backend: DEFAULT
  max_tokens_in_buffer: 65536  # Must exceed max ISL
```

For example, see `examples/backends/trtllm/engine_configs/gpt-oss-120b/prefill.yaml`.

**Related Issue:** [#4327](https://github.com/ai-dynamo/dynamo/issues/4327)

## Driver mismatch produces cryptic PyTorch errors

When the host NVIDIA driver is too old for the container's CUDA version, PyTorch surfaces the failure as:

```text
RuntimeError: The NVIDIA driver on your system is too old (found version 570). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx
```

This is the symptom, not the cause — the cause is that the container image you pulled needs a newer driver than the host ships.

**Fix:**

- Check the minimum driver for the tag you pulled in the [Container / driver matrix](overview.md#container--driver-matrix).
- Either upgrade the host driver, or pull a lower-CUDA variant (e.g. `vllm-runtime:1.0.2` on driver `575+` instead of `vllm-runtime:1.0.2-cuda13` on driver `580+`).

> The driver-mismatch error message itself is being improved — tracked as an engineering follow-up.

## Hang when running locally with `--network host`

**Issue:** Running the TensorRT-LLM container locally with `docker run --network host ...` can hang
before the server becomes ready. This was first reported on Amazon Linux 2023, but the failure has
not been shown to be specific to that distribution.

**Cause:** Not established. A comparable hang is tracked in an open
[mpi4py discussion](https://github.com/mpi4py/mpi4py/discussions/491#discussioncomment-12660609)
in which maintainers attribute it to the backend MPI implementation rather than to mpi4py itself.
That thread remains undiagnosed. Dynamo pins no mpi4py version — it arrives transitively in the
upstream TensorRT-LLM image.

**Fix:** Replace `--network host` with explicit port mappings, exposing only the ports you need —
e.g. `4222` (NATS), `2379`/`2380` (etcd), and `8000` (frontend).
