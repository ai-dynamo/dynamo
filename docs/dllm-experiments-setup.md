# LLaDA 2 on Dynamo — Setup & Baselines

End-to-end reproducible setup for benchmarking the LLaDA 2.0 diffusion LM on Dynamo + SGLang. Two baselines covered:

1. **1 Dynamo worker** — single-GPU baseline (Dynamo frontend → 1 SGLang worker).
2. **2 Dynamo workers** — multi-worker baseline with round-robin routing (the configuration to beat).

Once these reproduce, scaling to 4 / 8 workers and switching the router mode is a one-flag change.

---

## 0. Prerequisites

| Item | Required |
|---|---|
| GPU | 1× (baseline 1) or 2× (baseline 2), any SM ≥ 8.0 with ≥ 80 GB. Tested on H100 80GB and RTX PRO 6000 Blackwell 97 GB. |
| CUDA toolkit | **≥ 12.8** for Blackwell (SM 12.0); 12.4+ for H100. Default Ubuntu `nvcc` is often 12.0 and will fail JIT for `sm_120`. |
| Python | 3.10 – 3.12 |
| Disk | ~80 GB for model weights + cache |
| `uv` | Any recent version (≥ 0.5) |
| Docker + Compose | for NATS / etcd (or run them however you prefer) |

If you're on a Blackwell box, set `PATH=/usr/local/cuda-12.9/bin:$PATH` before launching workers — otherwise SGLang's JIT-compiled fused-rope kernel dies with `nvcc fatal : Unsupported gpu architecture 'compute_120'` on first request.

---

## 1. Environment

### 1.1 Create the venv

```bash
# Pick a stable location outside the repo so the venv survives branch switches.
export VENV_ROOT="$HOME/Work/sglang-llada-env"
uv venv --python 3.12 "$VENV_ROOT/.venv"
source "$VENV_ROOT/.venv/bin/activate"
```

### 1.2 Install SGLang with the diffusion extra

```bash
uv pip install --prerelease=allow 'sglang[diffusion]==0.5.11'
```

This pulls torch, triton, flashinfer, and the SGLang scheduler. The `diffusion` extra is what enables the LLaDA DLLM path.

`--prerelease=allow` is required: sglang 0.5.11 depends on `flash-attn-4>=4.0.0b9`, which is a pre-release and uv refuses to resolve it otherwise.

### 1.3 Install Dynamo in dev mode

```bash
cd "$DYNAMO_REPO"        # path to your ai-dynamo/dynamo checkout
# maturin builds the Rust bindings; install it into the venv first:
uv pip install maturin
# Build the Rust bindings (one-time, ~5 min):
cd lib/bindings/python && maturin develop --uv && cd -
# Install the Python package editable:
uv pip install -e .
# NIXL Python bindings — dynamo.common.multimodal imports them transitively
# at worker startup, so the worker dies at import time without this.
uv pip install nixl
```

If you see `Command 'maturin' not found`, you skipped the `uv pip install maturin` line — ignore the apt suggestion, we want the PyPI version in the venv, not the system one.

If the worker dies with `ModuleNotFoundError: No module named 'nixl'` (or `ImportError: NIXL Python bindings must be installed`), you skipped the `uv pip install nixl` line.

After this, `python -m dynamo.frontend`, `python -m dynamo.sglang` should both be importable.

### 1.4 Bring up NATS + etcd

Dynamo's discovery layer needs both running locally:

```bash
docker compose -f "deploy/docker-compose.yml" up -d
# NATS on :4222, etcd on :2379. `docker compose ps` to confirm.
```

### 1.5 (Blackwell only) CUDA 12.9 PATH

```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export SGLANG_DISABLE_CUDNN_CHECK=1
```

H100 / Hopper boxes can skip this.

### 1.6 Smoke test the install

```bash
python -c "import sglang, dynamo; print(sglang.__version__, dynamo.__version__)"
# Expect: 0.5.11 <dynamo-version>
```

---

## 2. Baseline 1 — single Dynamo worker

Frontend + 1 SGLang DLLM worker on `CUDA_VISIBLE_DEVICES=0`. Frontend serves on `:8001`.

### 2.1 Launch

```bash
cd "$DYNAMO_REPO"
bash examples/backends/sglang/launch/diffusion_llada.sh
```

What this script does:

- Starts `python -m dynamo.frontend --http-port 8001`.
- Starts one `python -m dynamo.sglang` worker on GPU 0 with `inclusionAI/LLaDA2.0-mini-preview`, LowConfidence denoising algorithm, page-size 32, cudagraph disabled.

First start downloads the model (~32 GB bf16, ~44 GB if you re-launch with `QUANTIZATION=fp8` exported). Subsequent starts hit the HF cache.

### 2.2 Verify

```bash
curl -s localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"inclusionAI/LLaDA2.0-mini-preview",
    "messages":[{"role":"user","content":"What is the capital of France?"}],
    "max_tokens":64
  }' | jq .
```

Should return a standard OpenAI `chat.completion` object. If it returns a 500 with "Stream ended before generation completed", check that CUDA toolkit ≥ 12.8 is on `PATH` (Blackwell only).

### 2.3 Benchmark

We use `bench/llada-approx-kv/run_at_url.sh` — a thin wrapper around `aiperf profile` (NVIDIA's perf client, vendored at `dynamo/bin/aiperf`). The wrapper hard-codes the LLaDA model name and tokenizer, fixes the random seed to 42 for reproducibility, requests streaming responses, and writes raw aiperf artifacts under `bench/llada-approx-kv/results/<label>/` (CSV + JSON).

The workload it generates is a **prefix-heavy chat workload**: aiperf draws prompts from a pool of synthetic shared prefixes, then appends a small unique user turn — exactly the shape where KV-cache reuse matters (RAG / system-prompt-heavy traffic).

To compare 1-worker vs 2-worker honestly, **use the same params for both** — sweep concurrency so the saturation curve is visible end-to-end. Define a shared flag set once:

```bash
BENCH_FLAGS=(--reqs 200 --warmup 20 --prefix-pool 4 --prefix-length 2000 --osl 64)

for CONC in 4 8 16 32; do
  bash bench/llada-approx-kv/run_at_url.sh "baseline-1w-c${CONC}" http://localhost:8001 \
      --conc "$CONC" "${BENCH_FLAGS[@]}"
done
```

If you only want a single point first, conc=8 is the 1-worker sweet spot (saturated but not yet queueing):

```bash
bash bench/llada-approx-kv/run_at_url.sh baseline-1w-c8 http://localhost:8001 \
    --conc 8 "${BENCH_FLAGS[@]}"
```

| Arg | Maps to (aiperf) | Meaning |
|---|---|---|
| `baseline-1w` (positional) | artifact dir name | Label for the run; artifacts land under `bench/llada-approx-kv/results/baseline-1w/`. |
| `http://localhost:8001` (positional) | `--url` | OpenAI-compatible endpoint to hit. Frontend port from §2.1. |
| `--conc 8` | `--concurrency` | In-flight requests held by aiperf at all times (closed-loop). The 1-worker baseline saturates around conc=4–8. |
| `--reqs 200` | `--request-count` | Total measured requests. 200 is enough for stable p50/p95 at this concurrency. |
| `--warmup 20` | `--warmup-request-count` | Pre-measurement requests so kernel autotune / cache warmup don't pollute timings. |
| `--osl 64` | `--osl` | Output sequence length (tokens). LLaDA generates in fixed blocks; OSL controls block count (e.g. 64 ≈ 2 blocks at block_size=32). |
| `--prefix-pool 4` | `--prefix-prompt-pool-size` | Number of distinct shared prefixes. With pool=4, ~25% of requests hit each prefix → strong cross-request KV reuse opportunity. |
| `--prefix-length 2000` | `--prefix-prompt-length` | Tokens per shared prefix. Long prefixes (~2k) make the prefill cost dominant if not cached — the regime where KV routing matters. |
| (default) `--isl 64` | `--isl` | Per-request *unique* input tokens (the user turn appended to the shared prefix). |
| (default) `--quick 0` | — | If you pass `--quick`, sets `reqs=40`/`warmup=8` for a 30 s smoke run instead of the full bench. |
| (other flags) | `--osl-stddev` | Pass `--osl-stddev N` to vary output length across requests (exercises OSL-load routing). |

Key output files in the artifact dir:
- `profile_export_aiperf.csv` — per-request latencies, easy to grep / awk.
- `profile_export_aiperf.json` — full aiperf result, includes p50/p95/p99 percentiles and throughput.

Expected order of magnitude on a single H100 80GB (LLaDA2.0-mini, conc=8, OSL=64):
| Metric | Range |
|---|---|
| TTFT avg | 5–9 s |
| Throughput | ~0.9–1.1 req/s |
| Tokens/s | ~60–80 |

If your numbers are much worse, re-check `--disable-cuda-graph` (cuda graphs interact badly with DLLM scheduling) and that `--attention-backend triton` is set.

#### Metric interpretation for DLLMs

aiperf is an AR-flavored client; its metric definitions translate imperfectly to LLaDA. Read the numbers above with these caveats:

- **TTFT** for a DLLM is *not* "time to first decoded token" the way it is for AR. LLaDA emits in blocks (block_size=32 by default), so the first token only appears when the *first block* has finished denoising. TTFT for LLaDA ≈ prefill + (N denoise steps × 1 block); for OSL=64 (≈2 blocks) it's essentially "time to half the output."
- **Tokens/s** is the cleanest metric to compare across configs — total output tokens divided by wall time, block-vs-token granularity doesn't change the ratio.
- **ITL (inter-token latency)**, if you look at it, is *bimodal* for DLLM: near-zero gaps within a block-commit burst, then one long gap (a full denoise pass) between blocks. aiperf reports the *mean* across both regimes, which is roughly meaningless. Use p50 / p95 of TTFT and end-to-end latency instead.
- **Throughput (req/s)** is fine either way — completed-request count per second is well-defined regardless of how each request streamed internally.

When comparing 1w vs 2w vs router modes later, focus on **tok/s** (throughput at fixed concurrency) and **TTFT p95** (tail-latency under contention). Mean ITL deltas between configs are noise.

---

## 3. Baseline 2 — two Dynamo workers (round-robin)

Same model; one worker per GPU; one Dynamo frontend in front in round-robin mode. One command launches the whole fleet:

### 3.1 Launch the fleet

```bash
bash examples/backends/sglang/launch/diffusion_llada_multi.sh --gpu-ids 0,1
```

That single invocation starts:
- 1× Dynamo frontend on `:8001` in `--mode round-robin` (configurable via `--mode kv-approx|kv-events`).
- 1× SGLang DLLM worker per GPU listed in `--gpu-ids`, each with `CUDA_VISIBLE_DEVICES` pinned. Workers and frontend share a process group so `Ctrl-C` tears everything down.

Optional knobs:
- `QUANTIZATION=fp8 bash ...` — FP8 weights (cuts ~16 GB per worker; required if packing 2 workers per GPU).
- `bash ... --gpu-ids 0 --worker-only` — skip the frontend (use `frontend_router.sh` separately when you want to swap modes without restarting workers).

Wait for each worker to log `Engine ... ready` before benching (usually 30–60 s; the frontend prints `address="0.0.0.0:8001"` immediately).

### 3.2 Verify

Same curl as §2.2 against `:8001`. Repeat 4× — the frontend should alternate between the two workers (their logs show one request each).

### 3.3 Benchmark

**Reuse the exact `BENCH_FLAGS` from §2.3** so the 2-worker numbers are directly comparable point-by-point. Same sweep, same URL, only the label changes:

```bash
BENCH_FLAGS=(--reqs 200 --warmup 20 --prefix-pool 4 --prefix-length 2000 --osl 64)

for CONC in 4 8 16 32; do
  bash bench/llada-approx-kv/run_at_url.sh "baseline-2w-rr-c${CONC}" http://localhost:8001 \
      --conc "$CONC" "${BENCH_FLAGS[@]}"
done
```

Reading the sweep:
- **conc=4–8**: per-worker batch fits comfortably; 2w ≈ 2× the 1w throughput (parallel workers).
- **conc=16–32**: 1w saturates and queues (TTFT explodes, throughput plateaus around 0.9 req/s); 2w absorbs the load and pulls ahead nonlinearly. This is where the bigger-than-2× win shows up.

Expected on 2× H100 80GB at conc=16:
| Metric | Range |
|---|---|
| TTFT avg | 0.7–1.2 s |
| Throughput | ~7–9 req/s |
| Tokens/s | ~450–550 |

---

## 4. Tear-down

`Ctrl-C` in each launch shell. The `EXIT` trap propagates `kill 0` so frontend + worker shut down together. NATS / etcd containers stay up — `docker compose -f deploy/docker-compose.yml down` if you want them gone too.

---

## 5. Next experiments (one-line preview)

Each of these reuses the 2-worker setup above, changing only the frontend flag or the worker count:

```bash
# Same fleet, KV-aware routing (consumes radix events from SGLang)
bash examples/backends/sglang/launch/diffusion_llada_multi.sh --gpu-ids 0,1 --mode kv-events

# KV-aware routing with OSL-load weighting
OSL_LOAD_WEIGHT=1.0 bash examples/backends/sglang/launch/diffusion_llada_multi.sh --gpu-ids 0,1 --mode kv-events

# 4-worker fleet on 4 GPUs
bash examples/backends/sglang/launch/diffusion_llada_multi.sh --gpu-ids 0,1,2,3 --mode kv-events
```

Each frontend mode is one curl + one `run_at_url.sh` away — keep the worker shells running and only restart the frontend between modes for apples-to-apples comparison.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nvcc fatal : Unsupported gpu architecture 'compute_120'` (Blackwell) | Default `/usr/bin/nvcc` is 12.0 | Export `PATH=/usr/local/cuda-12.9/bin:$PATH` before launching workers |
| 500 "Stream ended before generation completed" | Worker JIT-compile failed mid-request | Same as above; check worker stdout for the real nvcc error |
| Worker hangs at startup with no GPU activity | NATS / etcd not running | `docker compose -f deploy/docker-compose.yml ps`; restart if missing |
| `ModuleNotFoundError: dynamo._core` | Rust bindings not built | `cd lib/bindings/python && maturin develop --uv` |
| Worker logs "kv_events_config missing 'endpoint'" but kv-events mode requested | Frontend mode is `kv-events` but worker has no `KV_EVENTS_CONFIG` | Either match the modes (RR ↔ RR, kv-events on both sides) or set the env var per `diffusion_llada_multi.sh` |
| Both workers grab the same GPU | `CUDA_VISIBLE_DEVICES` not isolated | The launch script sets it per-invocation; re-check you passed distinct `--gpu-id` values |

---

## 7. Benchmark matrix

The grid below is what to actually run on 8×H100 to answer "what does Dynamo-unique routing buy over plain RR." Each block builds on the launchers in §2–§3 and reuses the `BENCH_FLAGS` pattern from §2.3 so every row is apples-to-apples.

```bash
BENCH_FLAGS=(--reqs 200 --warmup 20 --prefix-length 2000 --osl 64)
```

### 7.1 Scaling baseline — RR only

**Question**: how far does naive round-robin scaling get you, and where does each fleet size saturate?

Same conc grid across all three fleet sizes — that's what makes it apples-to-apples (same total in-flight load, different parallelism).

| # | Workers | Conc grid | prefix-pool |
|---|---|---|---|
| 1 | 1 | 8, 16, 32, 64, 128 | 4 |
| 2 | 2 | 8, 16, 32, 64, 128 | 4 |
| 3 | 8 | 8, 16, 32, 64, 128 | 4 |

```bash
# Example: 8-worker RR scaling sweep
bash examples/backends/sglang/launch/diffusion_llada_multi.sh --gpu-ids 0,1,2,3,4,5,6,7 --mode round-robin &
for CONC in 8 16 32 64 128; do
  bash bench/llada-approx-kv/run_at_url.sh "8w-rr-c${CONC}" http://localhost:8001 \
      --conc "$CONC" --prefix-pool 4 "${BENCH_FLAGS[@]}"
done
```

**Reading the curves**: at low conc, 1w competes; at high conc, 1w queues (TTFT explodes, throughput plateaus around 0.9 req/s) while 8w absorbs. The crossover concs are the story.

### 7.2 KV routing — does it move the needle at 8 workers?

**Question**: at the conc where 8w-RR is in its sweet spot (probably 64), does KV-aware routing actually beat RR — and on which workload?

The dial that matters is **prefix-pool diversity**. Per-worker KV cache holds ~25–30 prefixes at `prefix-length=2000`; sweep across that boundary.

| # | Workers | Conc | prefix-pool | Modes | Expected |
|---|---|---|---|---|---|
| 4 | 8 | 64 | 4 | RR / kv-approx / kv-events | Cache-saturated — every worker holds every prefix → all modes tie. RR shouldn't lose. |
| 5 | 8 | 64 | **40** | RR / kv-approx / kv-events | **Sweet spot** — pool exceeds per-worker capacity, RR misses ~38% of prefixes, KV routes around it. **This is where the win shows up.** |
| 6 | 8 | 64 | 200 | RR / kv-approx / kv-events | High diversity — too much eviction churn, both modes degrade together. Confirms KV doesn't help when there's no reuse. |

```bash
# Workers stay up across mode swaps — only restart the frontend with --worker-only mode.
bash examples/backends/sglang/launch/diffusion_llada_multi.sh --gpu-ids 0,1,2,3,4,5,6,7 --worker-only &
sleep 60
for POOL in 4 40 200; do
  for MODE in round-robin kv-approx kv-events; do
    bash examples/backends/sglang/launch/frontend_router.sh --mode "$MODE" &
    FRONTEND_PID=$!
    sleep 5
    bash bench/llada-approx-kv/run_at_url.sh "8w-${MODE}-p${POOL}-c64" http://localhost:8001 \
        --conc 64 --prefix-pool "$POOL" "${BENCH_FLAGS[@]}"
    kill $FRONTEND_PID
    wait $FRONTEND_PID 2>/dev/null
  done
done
```

### 7.3 OSL-load weighting — second Dynamo-unique lever

**Question**: when output length varies across requests, does the router's OSL-aware load score (`--router-osl-load-weight`) beat plain KV routing?

| # | Workers | Conc | prefix-pool | OSL stddev | Mode | OSL weight |
|---|---|---|---|---|---|---|
| 7a | 8 | 64 | 40 | 64 | kv-events | 0.0 (off) |
| 7b | 8 | 64 | 40 | 64 | kv-events | 1.0 (on) |

```bash
# Add osl-stddev to BENCH_FLAGS for this block
for OSL_W in 0.0 1.0; do
  OSL_LOAD_WEIGHT=$OSL_W bash examples/backends/sglang/launch/frontend_router.sh --mode kv-events &
  FRONTEND_PID=$!
  sleep 5
  bash bench/llada-approx-kv/run_at_url.sh "8w-kvevents-p40-osl-w${OSL_W}" http://localhost:8001 \
      --conc 64 --prefix-pool 40 --osl-stddev 64 "${BENCH_FLAGS[@]}"
  kill $FRONTEND_PID
  wait $FRONTEND_PID 2>/dev/null
done
```

### 7.4 What to report

Focus on these per row — they're the metrics that mean what they say for a diffusion LM (§2.3 caveat block):

- **tok/s** — primary throughput number.
- **TTFT p95** — tail latency; this is what queuing looks like, and where 8w pulls away from 1w/2w.
- **req/s** — sanity check.

Skip **mean ITL** — bimodal for DLLM, meaningless without splitting by block-emit vs intra-block.

### 7.5 Time budget

- 7.1 (RR scaling, 3 fleets × 5 conc) = **15 runs ≈ 1.5 h** (includes one fleet restart between rows)
- 7.2 (KV modes × pool) = **9 runs ≈ 45 min** (workers stay up, only frontend restarts)
- 7.3 (OSL-load) = **2 runs ≈ 10 min**

≈ **2.5–3 hours** total. If time is tight, skip 7.1 #1 / #2 (those are already covered by the 2-GPU run summarized in `docs/llada-dynamo-1pager.md`) and start with **7.1 #3 + 7.2 #5** — those two answer the central question.
