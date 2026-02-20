# NeMo Switchyard (RouteLLM Proxy) — Testing Guide

## What This Is

A Python/FastAPI proxy (`dynamo.nemo_switchyard`) that sits in front of the Dynamo Frontend. It uses [RouteLLM](https://github.com/lm-sys/RouteLLM) to classify query complexity and route requests to either a "strong" or "weak" model.

```
Client (port 8080)
   │
   ▼
NeMo Switchyard (FastAPI proxy)
   │  - Extracts prompt from request
   │  - Calls RouteLLM Controller.route() → selects model
   │  - Rewrites "model" field in request body
   ▼
Dynamo Frontend (Rust HTTP, port 8000)
   │  - ModelManager dispatches to engine by model name
   ▼
Workers (strong model pool OR weak model pool)
```

## Files Created/Modified

| File | Purpose |
|------|---------|
| `components/src/dynamo/nemo_switchyard/__init__.py` | Package version boilerplate |
| `components/src/dynamo/nemo_switchyard/__main__.py` | Entry point: `python -m dynamo.nemo_switchyard` |
| `components/src/dynamo/nemo_switchyard/main.py` | Core proxy: RouteLLMProxy, FastAPI app, CLI |
| `components/backends/vllm/deploy/agg_routellm.yaml` | K8s deployment config |
| `components/backends/vllm/launch/agg_routellm.sh` | Local multi-GPU launch script |
| `pyproject.toml` | Added `routellm` optional dependency group |

## Pre-requisites on Linux Machine

### 1. Kill processes on GPUs 4-7 (GPU 0 is occupied, leave it alone)

```bash
# Check what's running
nvidia-smi

# Kill processes on GPUs 4,5,6,7
for gpu_id in 4 5 6 7; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $gpu_id 2>/dev/null | tr -d ' ')
    for pid in $pids; do
        [ -n "$pid" ] && echo "Killing PID $pid on GPU $gpu_id" && kill -9 $pid
    done
done

# Verify GPUs 1-7 are free
nvidia-smi
```

### 2. Install dependencies

```bash
# From the repo root
pip install -e ".[routellm]"
```

### 3. Verify model checkpoints exist

```bash
ls /data/models/gpt-oss-120b
ls /data/models/gpt-oss-20b
```

## Testing

### Quick Start (default config)

```bash
# Uses: strong=gpt-oss-120b on GPUs 1,2,3,4 (TP=4), weak=gpt-oss-20b on GPU 5
bash components/backends/vllm/launch/agg_routellm.sh
```

### Custom GPU/model layout

```bash
# Override any defaults via env vars:
STRONG_MODEL=/data/models/gpt-oss-120b \
WEAK_MODEL=/data/models/gpt-oss-20b \
STRONG_GPUS=1,2,3,4 \
STRONG_TP=4 \
WEAK_GPUS=5 \
THRESHOLD=0.3 \
bash components/backends/vllm/launch/agg_routellm.sh
```

### Test Requests

```bash
# Health check
curl http://localhost:8080/health

# List models
curl http://localhost:8080/v1/models

# Chat completion (should route based on complexity)
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}'

# Check routing stats
curl http://localhost:8080/metrics

# Verify which model was selected (check response header)
curl -v http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"Explain quantum entanglement in detail with mathematical formulations"}],"max_tokens":200}' \
    2>&1 | grep x-routellm-routed-model
```

### Streaming test

```bash
curl -N http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}],"max_tokens":50,"stream":true}'
```

## Troubleshooting

- **RouteLLM import error**: `pip install routellm`
- **httpx not found**: `pip install httpx>=0.27.0`
- **Model not found by Frontend**: The `--strong-model` and `--weak-model` values must exactly match the model names that workers register in etcd. Check with `curl http://localhost:8000/v1/models`.
- **GPU OOM**: The 120B model needs TP=4 across 4 GPUs. Make sure no other processes are on GPUs 1-4.
- **Proxy can't reach Frontend**: Ensure Frontend is running on port 8000 before the proxy starts. The launch script starts them in order but the Frontend may need a few seconds to be ready.

## Architecture Notes for Further Development

- The proxy does **byte-level SSE passthrough** — no SSE event parsing, just `aiter_bytes()` streaming from httpx to avoid re-serialization overhead.
- On routing failure or empty prompt, falls back to the strong model (configurable via `--fallback-model`).
- The `x-routellm-routed-model` response header shows which model was selected for each request.
- All RouteLLM router types are supported: `mf`, `causal_llm`, `bert`, `sw_ranking`.
