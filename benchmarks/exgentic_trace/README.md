<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Exgentic Agent Trace Replay

Convert the [Exgentic agent-llm-traces](https://huggingface.co/datasets/Exgentic/agent-llm-traces)
Parquet shards to Mooncake JSONL:

```bash
uv run --with pyarrow python benchmarks/exgentic_trace/convert_to_mooncake.py \
  /path/to/agent-llm-traces/data \
  --output /tmp/exgentic.mooncake.jsonl \
  --block-size 512
```

Replay the result:

```bash
uv run --no-sync python -m dynamo.replay /tmp/exgentic.mooncake.jsonl \
  --trace-format mooncake \
  --trace-block-size 512 \
  --replay-concurrency 32 \
  --replay-mode offline \
  --router-mode round_robin \
  --num-workers 1 \
  --extra-engine-args '{"block_size":64}'
```

The converter emits one closed-loop session per dataset row. Failed and zero-token
spans are skipped. Overlapping client spans produce zero inter-turn delay. Input
length decreases reset the synthetic cache prefix.

The source has token counts but no token IDs or block hashes, so cache reuse is an
Applied-Compute-style approximation: hashes remain stable while a session prompt
grows, but cross-session prompt sharing is not inferred.
