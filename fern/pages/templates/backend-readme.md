---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

{/* 2-3 sentence overview of this backend integration */}

## Feature Matrix

{/* Copy actual feature matrix from existing backend docs */}
{/* Example pattern (from vLLM README): */}

| Feature | Status | Notes |
|---------|--------|-------|
| Disaggregated Serving | ✅ | |
| KV-Aware Routing | ✅ | |
| SLA-Based Planner | ✅ | |
| Multimodal | ✅ | Vision models |
| LoRA | 🚧 | Experimental |

## Quick Start

### Prerequisites

- {/* List prerequisites */}

### Usage

```bash
# Add minimal usage example from existing backend docs
# Example pattern (vLLM):
# python -m dynamo.vllm --model <model-name>
# Example pattern (SGLang):
# python -m dynamo.sglang --model <model-name>
```

### Kubernetes

```yaml
# Add DGDR example - use apiVersion: nvidia.com/v1alpha1
# See recipes/ folder for production examples
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| {/* param */} | {/* default */} | {/* description */} |

{/* EXAMPLE: Filled-in Configuration for vLLM would look like:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | required | Model path or HuggingFace ID |
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism |
| `--max-model-len` | auto | Maximum sequence length | */}

## Next Steps

| Document | Path | Description |
|----------|------|-------------|
| `<Backend> Guide` | `<backend>_guide.md` | Advanced configuration |
| Backend Comparison | `../README.md` | Compare backends |

{/* Convert table rows to markdown links */}
