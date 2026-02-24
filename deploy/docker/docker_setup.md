# Docker Deployment Guide

Complete guide for running Dynamo with SGLang, TensorRT-LLM, or vLLM backends using Docker Compose.

## Prerequisites

- **NVIDIA GPU** with NVIDIA Container Toolkit installed
- **Docker** and **Docker Compose** installed
- A **HuggingFace account** (only required for gated/private models — see [Environment Setup](#1-environment-setup))

---

## One-Time Setup

### 1. Environment Setup (only required for gated/private models )

Create a `.env` file in `deploy/docker/`:

```bash
cd deploy/docker
cp env.example .env
```

> **For public models like `Qwen/Qwen3-0.6B`**: leave `HF_TOKEN` empty — no token required.
> **For gated/private models** (e.g. `meta-llama/Llama-3.1-8B-Instruct`): set your token from [HuggingFace Settings](https://huggingface.co/settings/tokens).

```env
# Required only for gated/private models (e.g. Llama, Gemma)
# Leave empty for public models like Qwen3
HF_TOKEN=
```

## How Local Development Works

### Why Two Containers Need to Find Each Other

Dynamo splits into two containers:

- **Frontend** — receives HTTP requests and routes them to a backend
- **Backend** — runs the actual LLM inference

For the frontend to route requests, both containers must share a **service discovery store** — a place where the backend registers itself and the frontend looks it up.

### The tmpfs Solution

```
┌──────────────────┐     ┌───────────────────┐
│  dynamo-frontend  │     │   dynamo-backend   │
│                   │     │                    │
│   "find backend"  │     │   "I am here"      │
└────────┬──────────┘     └──────────┬─────────┘
         │                           │
         └─────────────┬─────────────┘
                       ▼
            [Docker tmpfs volume]
            /tmp/dynamo_store_kv
            RAM-backed, Docker-managed
            Created on `up`, destroyed on `down`
```

**Why tmpfs instead of a host directory:**

Service discovery only needs to live as long as the containers are running — tmpfs was used for this. We can create also folder outside of docker and use it but then we need to give it proper permissions.

---

## Flag Reference

Every flag used in the compose files, explained in plain English.

### Frontend Flags

| Flag | Value | What it does |
|------|-------|--------------|
| `--http-port` | `8000` | Port the OpenAI-compatible HTTP API listens on |
| `--store-kv` | `file` | Backend for service discovery. `file` = shared directory (local dev). `etcd` = distributed (production). `mem` = process-local only — **will not work across containers** |
| `--request-plane` | `tcp` | Transport for requests between frontend and backend. `tcp` = direct socket, fastest. `nats` = message broker, required for KV-aware routing in production |

### Backend Flags

#### vLLM

| Flag | Value | What it does |
|------|-------|--------------|
| `--model` | `Qwen/Qwen3-0.6B` | HuggingFace model ID to load |
| `--store-kv` | `file` | Must match the frontend store |
| `--kv-events-config` | `{"enable_kv_cache_events": false}` | Disables KV cache event publishing. Required when not using NATS — KV events need a message broker to propagate |
| `--gpu-memory-utilization` | `0.8` | Fraction of GPU VRAM to allocate (80%). Leave headroom for system use |

#### SGLang

| Flag | Value | What it does |
|------|-------|--------------|
| `--model-path` | `Qwen/Qwen3-0.6B` | HuggingFace model ID or local path |
| `--store-kv` | `file` | Must match the frontend store |
| `--mem-fraction-static` | `0.8` | Fraction of GPU memory for static KV cache allocation |
| `--max-total-tokens` | `10000` | Maximum combined tokens (prompt + output) across all concurrent requests |
| `--attention-backend` | `flashinfer` | **Required for H100/A100 and most non-Blackwell GPUs.** TensorRT MHA backend only works on Blackwell (SM100) |

#### TensorRT-LLM

| Flag | Value | What it does |
|------|-------|--------------|
| `--model-path` | `Qwen/Qwen3-0.6B` | HuggingFace model ID or local path |
| `--store-kv` | `file` | Must match the frontend store |
| `--free-gpu-memory-fraction` | `0.8` | Fraction of free GPU memory to use for inference |
| `--max-seq-len` | `10000` | Maximum sequence length (prompt + output tokens) |

### Docker Compose Fields

| Field | What it does |
|-------|--------------|
| `ipc: host` | Shares host IPC namespace with the container. Required for CUDA shared memory between GPU processes |
| `gpus: all` | Passes all host GPUs into the container via NVIDIA Container Toolkit |
| `ulimit memlock: -1` | Removes the locked-memory limit. CUDA pins large memory regions — without this, allocations fail silently |
| `ulimit stack: 67108864` | Sets stack size to 64 MB. Large model inference uses deep call stacks |
| `ulimit nofile: 1048576` | Allows up to 1M open file descriptors. Large models open many memory-mapped files |
| `DYN_REQUEST_PLANE=tcp` | Sets request plane via environment variable (equivalent to `--request-plane tcp`) |

---

## Running the Containers

### Local Development Mode (Recommended for Testing)

No external services needed. Frontend and backend only.

```bash
cd deploy/docker/local-development

# vLLM
docker compose -f docker_compose_D_VLLM_Local_2.yaml up

# SGLang
docker compose -f docker_compose_D_SGLang_Local_2.yaml up

# TensorRT-LLM  (note the space in the filename — use quotes)
docker compose -f "docker_compose_ D_TRT_Local_2.yaml" up
```

### Distributed Mode (Production)

Adds NATS message broker + etcd for KV-aware routing and high availability.

```bash
cd deploy/docker/distributed

docker compose -f docker_compose_SGLang_Nats_Etcd.yaml up
docker compose -f docker_compose_TRT_Nats_Etcd.yaml up
docker compose -f docker_compose_VLLM_Nats_Etcd.yaml up
```

---

## Testing Your Deployment

Wait for the backend `depends_on` healthcheck to pass, then:

### Non-Streaming

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "stream": false
  }' | jq
```

### Streaming

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Write a short poem."}],
    "stream": true
  }'
```

---

## Stopping Services

```bash
# Stop containers, keep the discovery volume
docker compose down

# Stop and remove everything including the tmpfs volume
docker compose down -v
```

---

## Deployment Comparison

| Mode | Components | Best For |
|------|-----------|----------|
| **Local Development** | Frontend + Backend | Development, testing, single machine |
| **Distributed** | Frontend + Backend + NATS + etcd | Production, KV-aware routing, high availability |

## Backend Comparison

| Backend | GPU memory flag | Notes |
|---------|----------------|-------|
| **vLLM** | `--gpu-memory-utilization` | Needs `--kv-events-config '{"enable_kv_cache_events": false}'` in local mode |
| **SGLang** | `--mem-fraction-static` | Needs `--attention-backend flashinfer` on H100/A100 |
| **TensorRT-LLM** | `--free-gpu-memory-fraction` | Command must be prefixed with `trtllm-llmapi-launch` |

---

## Troubleshooting

### Frontend container exits with code 2

The frontend received an unknown flag. Check what flags the image actually supports:

```bash
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1 python3 -m dynamo.frontend --help
```

### 404 on `/v1/chat/completions`

The frontend started but cannot find the backend. Check that:

- Both services mount the same `discovery` volume at the same container path
- Neither service uses `--store-kv mem` — that is process-local and **cannot be shared across containers**
- Both services are on the same Docker network

### Check container logs

```bash
docker compose logs -f dynamo-frontend
docker compose logs -f dynamo-vllm-backend     # or dynamo-sglang-backend / dynamo-trtllm-backend
```

---

## Changing the Model

Edit the `--model` or `--model-path` argument in the compose file:

```yaml
# vLLM
command: python3 -m dynamo.vllm --model your-org/your-model ...

# SGLang
command: python3 -m dynamo.sglang --model-path your-org/your-model ...

# TensorRT-LLM
command: trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path your-org/your-model ...
```

For gated models, add `HF_TOKEN` to `deploy/docker/.env`.

---

## Next Steps

- [Kubernetes deployment](../../README.md)
- [Advanced examples](../../examples/)
- [Community Discord](https://discord.gg/D92uqZRjCZ)
