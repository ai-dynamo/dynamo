<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multiturn Benchmark

`multiturn_bench` simulates concurrent multi-turn conversations against an
OpenAI-compatible chat endpoint and reports per-turn TTFT and total latency
statistics. It can optionally enable **speculative prefill** — a technique that
pre-warms the KV cache with the predicted next-turn prefix after each assistant
response, cutting TTFT on subsequent turns.

## Quick start

```bash
# Build
cargo build --release --package dynamo-bench --bin multiturn_bench

# Smoke test (1 user, 1 turn, ~50 tokens)
./target/release/multiturn_bench --ping
```

## Speculative prefill demo

Speculative prefill works best with multi-turn workloads where the conversation
grows incrementally (e.g. reasoning models in agentic loops). After each
assistant turn the frontend constructs the next-turn prompt prefix and sends a
`max_tokens=1` request to warm the KV cache, so the real follow-up hits a warm
cache and gets a much lower TTFT.

### 1. Launch the backend and frontend

```bash
# Terminal 1 — backend (vLLM example, any supported backend works)
python -m dynamo.vllm \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Terminal 2 — frontend with KV router
python -m dynamo.frontend \
  --router-mode kv \
  --http-port 8000
```

### 2. Run baseline (no speculative prefill)

```bash
./target/release/multiturn_bench \
  --url http://localhost:8000 \
  --num-users 10 \
  --num-turns 5 \
  --num-user-tokens 128 \
  --max-completion-tokens 256 \
  --mean-delay-ms 5000 \
  --output baseline.json \
  --verbose
```

### 3. Run with speculative prefill

```bash
./target/release/multiturn_bench \
  --url http://localhost:8000 \
  --num-users 10 \
  --num-turns 5 \
  --num-user-tokens 128 \
  --max-completion-tokens 256 \
  --mean-delay-ms 5000 \
  --speculative-prefill \
  --output specprefill.json \
  --verbose
```

Compare the per-turn TTFT columns: turns 2+ should show a significant TTFT
reduction (up to ~3x) because the KV cache is already warm when the real
request arrives.

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:8000` | Frontend HTTP endpoint |
| `--model` | auto-detected | Model name (queries `/v1/models` if omitted) |
| `--num-users` | `10` | Concurrent simulated users |
| `--num-turns` | `5` | Conversation turns per user |
| `--num-user-tokens` | `128` | Approximate user-prompt token count per turn |
| `--max-completion-tokens` | `1000` | Output sequence length cap |
| `--ignore-eos` | `true` | Force generation to max tokens |
| `--mean-delay-ms` | `5000` | Mean inter-turn delay (exponential distribution) |
| `--speculative-prefill` | `false` | Enable speculative prefill via `nvext.agent_hints` |
| `--output <path>` | none | Write results to JSON file |
| `--verbose` / `-v` | `false` | Print per-turn logging |
| `--seed` | `42` | Random seed |
| `--ping` | `false` | Smoke-test mode (1 user, 1 turn, ~50 tokens, no delay) |

## How speculative prefill works

1. The client sends `{"nvext": {"agent_hints": {"speculative_prefill": true}}}` in each request.
2. As the assistant response streams back, the frontend accumulates the full response text.
3. Once `finish_reason` is set, a background task constructs the next-turn prompt (conversation history + assistant response, thinking content stripped) and sends a `max_tokens=1` prefill-only request through the pipeline.
4. The KV router routes the speculative request to the same worker, warming its cache.
5. When the real next-turn request arrives, the KV router sees high cache overlap on that worker and routes there, yielding a much lower TTFT.

See also: [Agent Hints documentation](../../../../docs/pages/components/router/agent-hints.md)
