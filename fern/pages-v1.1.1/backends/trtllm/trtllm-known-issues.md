---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Known Issues and Mitigations
---

For general TensorRT-LLM features and configuration, see the [Reference Guide](trtllm-reference-guide.md).

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
