<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MM Router Worker (vLLM)

Multimodal-aware KV cache routing worker for vLLM backends.

## Overview

This worker sits between the Dynamo frontend and vLLM workers, providing MM-aware KV cache routing:

1. Receives OpenAI-format requests from the frontend
2. Downloads images and computes `mm_hash` (routing only)
3. Builds multimodal routing metadata (`mm_routing_info`)
4. Uses `KvRouter` to pick the best vLLM worker based on KV overlap
5. Forwards the request to the selected vLLM worker and streams responses back

## Architecture

```text
Frontend (standard)      MM Router Worker (this)         vLLM Worker (standard)
┌──────────────┐        ┌──────────────────────┐        ┌─────────────────────┐
│              │───────>│ 1. Download images   │───────>│ python -m dynamo.vllm│
│ round-robin  │        │ 2. Compute mm_hash   │        │ --enable-multimodal  │
│ to mm_router │<───────│ 3. Build routing     │<───────│ (publishes KV events)│
└──────────────┘        │ 4. KvRouter route    │        └─────────────────────┘
                        └──────────────────────┘
                                   │
                                   │ 
                                   v
                             ┌──────────┐
                             │   NATS   │
                             └──────────┘
```

## Prerequisites

- 1+ GPU with enough memory for your chosen multimodal model
- Docker (for local `etcd` + `nats`)
- Python environment with Dynamo installed (including vLLM backend support and Python bindings)

Throughout this README, assume:

```bash
export DYNAMO_ROOT=/path/to/dynamo
export MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
```

This guide assumes Dynamo is already installed in your current Python environment.

If the model is gated/private, also set `HF_TOKEN`.

### vLLM Patch Requirement (as of 2026-02-25)

This MM router depends on vLLM PR [`#33304`](https://github.com/vllm-project/vllm/pull/33304) (`[feat] Add per-block extra_keys to KV events`) so Dynamo can reconstruct MM-aware block hashes from KV events.

As of **2026-02-25**:
- PR [`#33304`](https://github.com/vllm-project/vllm/pull/33304) is merged into `vllm-project/vllm:main` (merged on **2026-02-21**)
- the latest vLLM release shown on GitHub releases is **v0.16.0 (2026-02-13)**, which predates this PR

If you are using a released vLLM package, you likely need to patch your installed `site-packages` manually.

Example (patch installed vLLM in-place, `site-packages` layout):

```bash
SITE_PACKAGES_ROOT="$(python - <<'PY'
import pathlib
import vllm
print(pathlib.Path(vllm.__file__).resolve().parent.parent)
PY
)"

cd "$SITE_PACKAGES_ROOT"

# Filter the PR diff to only files under vllm/, since site-packages does not
# contain the full vLLM repo layout (tests/, docs/, etc.).
curl -sL https://github.com/vllm-project/vllm/pull/33304.diff | python3 -c '
import sys
chunks = sys.stdin.read().split("diff --git ")
filtered = [c for c in chunks if c.startswith("a/vllm/")]
print("".join("diff --git " + c for c in filtered), end="")
' > /tmp/vllm_pr33304_vllm_only.diff

patch --dry-run -p1 < /tmp/vllm_pr33304_vllm_only.diff
patch -p1 < /tmp/vllm_pr33304_vllm_only.diff
```

After patching, restart the vLLM backend and MM router processes.

## Usage

### Quick Start

Start `etcd` + `NATS` first (separately), then run the launcher:

```bash
cd "$DYNAMO_ROOT"
docker compose -f deploy/docker-compose.yml up -d
```

```bash
cd "$DYNAMO_ROOT/examples/backends/vllm/mm_router_worker"
./launch.sh
```

Override defaults with environment variables, for example:

```bash
MODEL="$MODEL_NAME" HTTP_PORT=8001 ./launch.sh
```

### Quick Try (Manual, Step-by-Step)

Open 4 terminals.

### Terminal 1: Start `etcd` + `NATS`

```bash
cd "$DYNAMO_ROOT"
docker compose -f deploy/docker-compose.yml up -d
```

### Common Environment (all runtime terminals)

Use the same environment in terminals 2/3/4:

```bash
cd "$DYNAMO_ROOT"

export DYN_NAMESPACE=dynamo
export DYN_REQUEST_PLANE=nats
export NATS_SERVER=nats://127.0.0.1:4222
export ETCD_ENDPOINTS=http://127.0.0.1:2379
```

### Terminal 2: Start vLLM Worker (backend)

Use the same model string here and in the MM router.

```bash
cd "$DYNAMO_ROOT"

export DYN_NAMESPACE=dynamo
export DYN_REQUEST_PLANE=nats
export NATS_SERVER=nats://127.0.0.1:4222
export ETCD_ENDPOINTS=http://127.0.0.1:2379

python -m dynamo.vllm \
  --model "$MODEL_NAME" \
  --enable-multimodal \
  --enforce-eager \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

Notes:
- Current `dynamo.vllm` default component name is `backend` (used below by the MM router).
- MM-aware routing depends on KV events from the vLLM worker. In current Dynamo builds, KV events are auto-configured when prefix caching is enabled (you may see a deprecation warning suggesting explicit `--kv-events-config` in future versions).

### Terminal 3: Start MM Router Worker (vLLM)

Important:
- `--downstream-component backend` must match the vLLM worker component name.
- `--block-size` must match the vLLM KV cache block size (tested configs here use `16`).

```bash
cd "$DYNAMO_ROOT"

export DYN_NAMESPACE=dynamo
export DYN_REQUEST_PLANE=nats
export NATS_SERVER=nats://127.0.0.1:4222
export ETCD_ENDPOINTS=http://127.0.0.1:2379
export DYN_LOG=debug

python -m examples.backends.vllm.mm_router_worker \
  --model "$MODEL_NAME" \
  --namespace dynamo \
  --component mm_router \
  --endpoint generate \
  --downstream-component backend \
  --downstream-endpoint generate \
  --block-size 16
```

### Terminal 4: Start Frontend

Use `round-robin` here so the frontend routes to the MM router worker; the MM router itself performs KV-aware routing to vLLM backends.

```bash
cd "$DYNAMO_ROOT"

export DYN_NAMESPACE=dynamo
export DYN_REQUEST_PLANE=nats
export NATS_SERVER=nats://127.0.0.1:4222
export ETCD_ENDPOINTS=http://127.0.0.1:2379

python -m dynamo.frontend \
  --http-port 8000 \
  --router-mode round-robin
```

## Test Request

Send the same multimodal request twice. The second request should show much higher overlap in MM router logs.

```bash
MODEL="$MODEL_NAME"
IMAGE_URL="http://images.cocodataset.org/test2017/000000000001.jpg"

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data @- <<EOF
{
  "model": "${MODEL}",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image briefly."},
      {"type": "image_url", "image_url": {"url": "${IMAGE_URL}"}}
    ]
  }],
  "max_tokens": 100
}
EOF
```

Run the same `curl` command again. In the MM router worker logs (terminal 3), look for lines like:

```text
[ROUTING] Best: worker_... with X/Y blocks overlap
```

Expected behavior:
- First request: overlap near `0/Y`
- Repeated identical request: overlap jumps close to `Y`

### Example: Send the Same Request Twice

This is the exact shape you can use (same payload twice):

```bash
MODEL="$MODEL_NAME"

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data @- <<EOF | jq
{
    "model": "${MODEL}",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "http://images.cocodataset.org/test2017/000000000001.jpg"}}
      ]
    }],
    "max_tokens": 100
}
EOF
```

If MM-aware routing and prefix reuse are working, after sending the same request twice you should typically observe:

- MM router logs show a large overlap jump on the second request 
- Response metadata may show prompt cache reuse on the second request (for example `usage.prompt_tokens_details.cached_tokens`)
- End-to-end latency may drop on the second request (for example lower `nvext.timing.total_time_ms`)

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen3-VL-8B-Instruct` | Model path or HuggingFace ID |
| `--block-size` | `16` | KV cache block size used for routing (must match backend) |
| `--namespace` | `default` | Dynamo namespace |
| `--component` | `mm_router` | This worker's component name |
| `--endpoint` | `generate` | This worker's endpoint name |
| `--downstream-component` | `backend` | Downstream component name (use `backend` for current `dynamo.vllm` defaults) |
| `--downstream-endpoint` | `generate` | Downstream vLLM endpoint name |

## How It Works

### MM Hash Computation

The worker computes image hashes using the same image-UUID path Dynamo uses for vLLM multimodal inputs (`compute_mm_uuids_from_images`), then converts those UUIDs into integer `mm_hash` values for routing-only block hash computation.

This ensures:
- Same image -> same `mm_hash` -> same MM-aware block hashes -> cache reuse
- Different image -> different `mm_hash` -> different block hashes -> avoid false hits

### KV-Aware Routing

The worker calls `KvRouter.generate(...)` with:

- execution payload (`token_ids`, `multi_modal_data`, etc.)
- routing payload (`mm_routing_info`)

`mm_routing_info` contains:
- `routing_token_ids`: processor-expanded routing tokens (not frontend placeholder-only tokens)
- `block_mm_infos`: per-block MM metadata

This lets `KvRouter` compute MM-aware overlap and pick the best backend worker.

### Block MM Info Structure

Each routing block gets either `None` or a multimodal descriptor:

```python
block_mm_infos = [
    None,
    {"mm_objects": [{"mm_hash": 12345, "offsets": []}]},
    {"mm_objects": [{"mm_hash": 12345, "offsets": []}]},
]
```

For repeated identical images, multiple entries may appear in the same block when image boundaries overlap a block boundary. This matches vLLM's KV block hash boundary semantics.

## Files

| File | Description |
|------|-------------|
| `mm_router_worker.py` | Main worker (`@dynamo_worker`) and CLI |
| `handler.py` | `MMRouterHandler` routing logic |
| `mm_processor.py` | Image loading, token expansion, MM hash, block MM metadata |
| `__main__.py` | Module entry point |

## Dependencies

- `dynamo` (runtime + `KvRouter`)
- `transformers` (`AutoTokenizer`, `AutoProcessor`)
- `Pillow` (`PIL`) for image loading
- `requests` for `http(s)` image URLs
- vLLM-capable backend worker via `python -m dynamo.vllm`

## Known Limitations

- `mm_processor.py` currently only supports Qwen-style multimodal processors for per-image visual token counting (`Qwen2-VL`, `Qwen2.5-VL`, `Qwen3-VL` style processors). 
