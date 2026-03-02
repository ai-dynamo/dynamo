---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reference Guide
subtitle: Features, configuration, and operational details for the TensorRT-LLM backend
---

## KV Cache Transfer

Dynamo with TensorRT-LLM supports two methods for transferring KV cache in disaggregated serving: UCX (default) and NIXL (experimental). For detailed information and configuration instructions for each method, see the [KV Cache Transfer Guide](./trtllm-kv-cache-transfer.md).

## Request Migration

Dynamo supports [request migration](../../fault-tolerance/request-migration.md) to handle worker failures gracefully. When enabled, requests can be automatically migrated to healthy workers if a worker fails mid-generation. See the [Request Migration Architecture](../../fault-tolerance/request-migration.md) documentation for configuration details.

## Request Cancellation

When a user cancels a request (e.g., by disconnecting from the frontend), the request is automatically cancelled across all workers, freeing compute resources for other requests.

### Cancellation Support Matrix

| | Prefill | Decode |
|-|---------|--------|
| **Aggregated** | ✅ | ✅ |
| **Disaggregated** | ✅ | ✅ |

For more details, see the [Request Cancellation Architecture](../../fault-tolerance/request-cancellation.md) documentation.

## Multimodal Support

Dynamo with the TensorRT-LLM backend supports multimodal models, enabling you to process both text and images (or pre-computed embeddings) in a single request. For detailed setup instructions, example requests, and best practices, see the [TensorRT-LLM Multimodal Guide](../../features/multimodal/multimodal-trtllm.md).

## Video Diffusion Support (Experimental)

Dynamo supports video generation using diffusion models through the `--modality video_diffusion` flag.

### Requirements

- **TensorRT-LLM with visual_gen**: The `visual_gen` module is part of TensorRT-LLM (`tensorrt_llm._torch.visual_gen`). Install TensorRT-LLM following the [official instructions](https://github.com/NVIDIA/TensorRT-LLM#installation).
- **imageio with ffmpeg**: Required for encoding generated frames to MP4 video:
  ```bash
  pip install imageio[ffmpeg]
  ```
- **dynamo-runtime with video API**: The Dynamo runtime must include `ModelType.Videos` support. Ensure you're using a compatible version.

### Supported Models

| Diffusers Pipeline | Description | Example Model |
|--------------------|-------------|---------------|
| `WanPipeline` | Wan 2.1/2.2 Text-to-Video | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |

The pipeline type is **auto-detected** from the model's `model_index.json` — no `--model-type` flag is needed.

### Quick Start

```bash
python -m dynamo.trtllm \
  --modality video_diffusion \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --media-output-fs-url file:///tmp/dynamo_media
```

### API Endpoint

Video generation uses the `/v1/videos` endpoint:

```bash
curl -X POST http://localhost:8000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano",
    "model": "wan_t2v",
    "seconds": 4,
    "size": "832x480",
    "nvext": {
      "fps": 24
    }
  }'
```

### Configuration Options

| Flag | Description | Default |
|------|-------------|---------|
| `--media-output-fs-url` | Filesystem URL for storing generated media | `file:///tmp/dynamo_media` |
| `--default-height` | Default video height | `480` |
| `--default-width` | Default video width | `832` |
| `--default-num-frames` | Default frame count | `81` |
| `--enable-teacache` | Enable TeaCache optimization | `False` |
| `--disable-torch-compile` | Disable torch.compile | `False` |

### Limitations

- Video diffusion is experimental and not recommended for production use
- Only text-to-video is supported in this release (image-to-video planned)
- Requires GPU with sufficient VRAM for the diffusion model

## Logits Processing

Logits processors let you modify the next-token logits at every decoding step (e.g., to apply custom constraints or sampling transforms). Dynamo provides a backend-agnostic interface and an adapter for TensorRT-LLM so you can plug in custom processors.

### How it works

- **Interface**: Implement `dynamo.logits_processing.BaseLogitsProcessor` which defines `__call__(input_ids, logits)` and modifies `logits` in-place.
- **TRT-LLM adapter**: Use `dynamo.trtllm.logits_processing.adapter.create_trtllm_adapters(...)` to convert Dynamo processors into TRT-LLM-compatible processors and assign them to `SamplingParams.logits_processor`.
- **Examples**: See example processors in `lib/bindings/python/src/dynamo/logits_processing/examples/` ([temperature](https://github.com/ai-dynamo/dynamo/tree/main/lib/bindings/python/src/dynamo/logits_processing/examples/temperature.py), [hello_world](https://github.com/ai-dynamo/dynamo/tree/main/lib/bindings/python/src/dynamo/logits_processing/examples/hello_world.py)).

### Quick test: HelloWorld processor

You can enable a test-only processor that forces the model to respond with "Hello world!". This is useful to verify the wiring without modifying your model or engine code.

```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR=1
./launch/agg.sh
```

<Note>
- When enabled, Dynamo initializes the tokenizer so the HelloWorld processor can map text to token IDs.
- Expected chat response contains "Hello world".
</Note>

### Bring your own processor

Implement a processor by conforming to `BaseLogitsProcessor` and modify logits in-place. For example, temperature scaling:

```python
from typing import Sequence
import torch
from dynamo.logits_processing import BaseLogitsProcessor

class TemperatureProcessor(BaseLogitsProcessor):
    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature

    def __call__(self, input_ids: Sequence[int], logits: torch.Tensor):
        if self.temperature == 1.0:
            return
        logits.div_(self.temperature)
```

Wire it into TRT-LLM by adapting and attaching to `SamplingParams`:

```python
from dynamo.trtllm.logits_processing.adapter import create_trtllm_adapters
from dynamo.logits_processing.examples import TemperatureProcessor

processors = [TemperatureProcessor(temperature=0.7)]
sampling_params.logits_processor = create_trtllm_adapters(processors)
```

### Current limitations

- Per-request processing only (batch size must be 1); beam width > 1 is not supported.
- Processors must modify logits in-place and not return a new tensor.
- If your processor needs tokenization, ensure the tokenizer is initialized (do not skip tokenizer init).

## DP Rank Routing (Attention Data Parallelism)

TensorRT-LLM supports [attention data parallelism](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models) (attention DP) for models like DeepSeek. When enabled, multiple attention DP ranks run within a single worker, each with its own KV cache. Dynamo can route requests to specific DP ranks based on KV cache state.

### Dynamo vs TRT-LLM Internal Routing

- **Dynamo DP Rank Routing**: The router selects the optimal DP rank based on KV cache overlap and instructs TRT-LLM to use that rank with strict routing (`attention_dp_relax=False`). Use this with `--router-mode kv` for cache-aware routing.
- **TRT-LLM Internal Routing**: TRT-LLM's scheduler assigns DP ranks internally. Use this with `--router-mode round-robin` or `random` when KV-aware routing isn't needed.

### Enabling DP Rank Routing

```bash
# Worker with attention DP
# (TP=2 acts as the "world size", in effect creating 2 attention DP ranks)
CUDA_VISIBLE_DEVICES=0,1 python3 -m dynamo.trtllm \
  --model-path <MODEL_PATH> \
  --tensor-parallel-size 2 \
  --enable-attention-dp \
  --publish-events-and-metrics

# Frontend with KV routing
python3 -m dynamo.frontend --router-mode kv
```

The `--enable-attention-dp` flag sets `attention_dp_size = tensor_parallel_size` and configures Dynamo to publish KV events per DP rank. The router automatically creates routing targets for each `(worker_id, dp_rank)` combination.

<Note>
Attention DP requires TRT-LLM's PyTorch backend. AutoDeploy does not support attention DP.
</Note>

## KVBM Integration

Dynamo with TensorRT-LLM currently supports integration with the Dynamo KV Block Manager. This integration can significantly reduce time-to-first-token (TTFT) latency, particularly in usage patterns such as multi-turn conversations and repeated long-context requests.

See the instructions here: [Running KVBM in TensorRT-LLM](../../components/kvbm/kvbm-guide.md#run-kvbm-in-dynamo-with-tensorrt-llm).

## Observability

TensorRT-LLM exposes Prometheus metrics for monitoring inference performance. For detailed metrics reference, collection setup, and Grafana integration, see the [Prometheus Metrics Guide](./trtllm-prometheus.md).

## Known Issues and Mitigations

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
