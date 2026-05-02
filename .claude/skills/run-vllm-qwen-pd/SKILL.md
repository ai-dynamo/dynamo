---
name: run-vllm-qwen-pd
description: Launch a single Qwen3-0.6B vLLM instance wired to KVBM DynamoConnector with a conditional-disagg role (prefill or decode). Sized small so two instances coexist on GPU 0.
---

# Skill: Run Qwen3-0.6B vLLM instance (P or D)

Print (and optionally execute) the invocation that starts ONE
Qwen3-0.6B vLLM in a conditional-disagg role. For the full P+D bring-up
including the hub and verification, prefer `/disagg-bringup` — this
skill is the single-instance building block.

## When to Use

When you need to launch (or re-launch) just one of the two participants
without tearing down the other. Typical case: prefill is healthy, you're
debugging decode startup. Run this with `role=decode` and watch its log.

## Arguments

`key=value` pairs. Defaults in parens.

| Key | Default | Notes |
|---|---|---|
| `role` | *required* | `prefill` or `decode` |
| `hub-url` | `http://127.0.0.1:1337` | kvbm-hub discovery URL — leader uses this to register |
| `gpu` | `0` | `CUDA_VISIBLE_DEVICES` |
| `mem-util` | `0.25` | `--gpu-memory-utilization` (GB10 unified memory: 0.25 fits two instances; 0.35 OOMs) |
| `port` | `8000` (prefill), `8001` (decode) | vLLM OpenAI API port |
| `host-cache-gb` | `2` | `DYN_KVBM_CPU_CACHE_GB` plus mirrored in `cache.host.cache_size_gb` |
| `max-model-len` | `2048` | smaller KV per slot so two instances fit |
| `max-num-seqs` | `16` | concurrent sequences |

## Prerequisites (don't skip)

1. **Canonical venv** at `/home/ryan/.venvs/dynamo-kvbm` (symlink to or
   replacement of `.sandbox/`). If missing → `/dynamo:kvbm:sandbox-venv`.

2. **Current-branch bindings.** Verify with:
   ```bash
   /home/ryan/.venvs/dynamo-kvbm/bin/python3 -c "import kvbm; print(kvbm.__file__)"
   ```
   The path must be under `/home/ryan/repos/dynamo/lib/bindings/kvbm/`.
   If it points elsewhere → `/dynamo:kvbm:maturin-dev`.

3. **`libkvbm_kernels.so`** at
   `/home/ryan/repos/dynamo/lib/bindings/kvbm/python/kvbm/libkvbm_kernels.so`.
   If missing, copy from
   `target/release/build/kvbm-kernels-*/out/libkvbm_kernels.so`.

4. **Hub running** at `hub-url` (or you'll get a registration error
   ~immediately).

## Lessons baked in (don't deviate without reason)

- `CUDA_PATH=/usr/local/cuda` was required at **build time** (in the
  maturin step). Without it, cudarc 0.19.3 picks `cuda-13010` and binds
  the non-existent `cuDevSmResourceSplit` symbol on a CUDA 13.0.x
  driver, panicking at the first cuda call from the leader.
- `LD_LIBRARY_PATH` must include the kvbm package dir at **runtime**
  because maturin couldn't set rpath without `patchelf`.
- vLLM's `--gpu-memory-utilization 0.35` OOMs the second instance on
  GB10 unified memory; use `0.25`.
- `kv_role: "kv_both"` is correct — the DynamoConnector handles produce
  and consume from one connector instance; the P/D distinction is
  carried in the nested `disagg` block, not in `kv_role`.
- The `disagg` block lives under `leader` profile only — the worker
  profile ignores it (worker doesn't talk to the hub).

## Workflow

1. Parse args. If `role` is missing → error.
2. If `port` unset → default by role.
3. Print the command in a fenced bash block.
4. If the user said "run it" or `run=true`, execute. Otherwise just
   print — don't silently launch a long-running inference server.

## Command template

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
DYN_KVBM_CPU_CACHE_GB=<host-cache-gb> \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
LD_LIBRARY_PATH=/home/ryan/repos/dynamo/lib/bindings/kvbm/python/kvbm${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
/home/ryan/.venvs/dynamo-kvbm/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --max-model-len <max-model-len> \
  --max-num-seqs <max-num-seqs> \
  --gpu-memory-utilization <mem-util> \
  --enable-chunked-prefill \
  --port <port> \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "kvbm.v2.vllm.schedulers.connector",
    "kv_connector_extra_config": {
      "leader": {
        "disagg": { "hub_url": "<hub-url>", "role": "<role>" },
        "cache":  { "host": { "cache_size_gb": <host-cache-gb> } },
        "tokio":  { "worker_threads": 2 }
      },
      "worker": {
        "nixl":  { "backends": { "UCX": {}, "POSIX": {} } },
        "tokio": { "worker_threads": 2 }
      }
    }
  }'
```

## See also

- `/disagg-bringup` — full P+D bring-up; embeds this template inline
- `/disagg-teardown` — kill processes started here
- `/disagg-hub-curl` — verify registration after start
- `/dynamo:kvbm:maturin-dev` — rebuild bindings
- `/dynamo:kvbm:diagnose` — triage failed runs (CUDA / nccl traps)
