<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LoRA with vLLM Backend

For setup, usage, API reference, and troubleshooting, see the
[shared LoRA guide](../../../../common/lora.md).

## Quick Start

Load the adapter directly from Hugging Face Hub:

```bash
./agg_lora_hf.sh
```

Set `HF_TOKEN` before running the script when the base model or adapter is private. Override
`HF_LORA_REPO`, `LORA_NAME`, or `LORA_URI` to use another adapter or revision.

To use S3-compatible storage through MinIO:

```bash
./setup_minio.sh    # Start MinIO, download & upload LoRA
./agg_lora.sh       # Launch vLLM frontend + worker with LoRA
```

## vLLM-Specific Notes

- Default `--max-lora-rank 64` (same as SGLang)
- Override with environment variables: `MODEL`, `LORA_NAME`, `MAX_MODEL_LEN`, `MAX_CONCURRENT_SEQS`

### KV-Aware Routing (2 GPUs)

```bash
./agg_lora_router.sh
```

The script launches two vLLM workers behind a KV-aware router. Load the LoRA to both workers on
ports 8081 and 8082. The router then uses KV cache affinity to improve cache hit rates.
