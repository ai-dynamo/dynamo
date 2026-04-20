# Accuracy Benchmark Runbook — Qwen/Qwen3-VL-2B-Instruct

Stepwise guide to reproduce the accuracy benchmark against either the
**vLLM** or **TensorRT-LLM** runtime using NVIDIA's `vlmevalkit` eval harness.

Everything lives in `benchmarks/multimodal/accuracy/`.

---

## How It Works (End to End)

### Goal

Measure accuracy of **Qwen/Qwen3-VL-2B-Instruct** (a 2B-parameter
vision-language model) on two image-understanding benchmarks — **ChartQA**
and **OCRBench** — and compare scores between two serving runtimes to
confirm TRT-LLM is a drop-in replacement for vLLM with no accuracy
regression.

### Architecture

```
┌──────────────────────────┐         ┌──────────────────────────┐
│  Runtime Container       │  HTTP   │  vlmevalkit Container    │
│                          │◄────────│                          │
│  Serves the model as an  │  :8000  │  Sends image+question    │
│  OpenAI-compatible       │         │  pairs from benchmark    │
│  /v1/chat/completions    │         │  datasets, scores the    │
│  endpoint                │         │  model's answers         │
└──────────────────────────┘         └──────────────────────────┘
```

Both runtimes expose the **exact same API contract** — `/v1/chat/completions`
with multi-modal image content — so the eval script works unchanged for either.

### What happens at each stage

**1. Start the model server.**
A Docker container loads the model weights onto the GPU and exposes an HTTP
endpoint. For vLLM this is `vllm serve`; for TRT-LLM it is `trtllm-serve serve
--backend pytorch`. Both accept an HF model ID directly (no engine pre-build
needed). The serve script polls `/health` every few seconds until the server
reports ready.

**2. Run the eval harness** (`run_accuracy_test.sh`).
For each benchmark (`chartqa`, `ocrbench`):
  1. Launches the `vlmevalkit:26.01` container running `nemo-evaluator run_eval`.
  2. The harness downloads the benchmark dataset (e.g. ChartQA's 2500 test
     images + questions as a TSV file).
  3. For every sample, it sends a POST to `/v1/chat/completions` containing the
     base64-encoded image and the question text at `temperature=0`.
  4. The model returns an answer (e.g. `"No"`).
  5. The harness compares the model's answer to the ground truth and computes
     accuracy.

**3. Collect results.**
Each benchmark writes `results/<benchmark>/results.yml` with structured scores:
  - **ChartQA**: overall accuracy + per-split (test-human, test-augmented).
  - **OCRBench**: overall score out of 1000 + per-subtask breakdown (text
    recognition, scene VQA, document VQA, key info extraction, etc.).

**4. Compare runtimes.**
Run the same eval against both servers. Since the model weights are identical
and `temperature=0`, any accuracy delta is due to numerical differences in
the serving stack (FP precision, attention kernels, sampling paths). In
practice the scores are effectively identical — confirming that TRT-LLM is a
**drop-in replacement** for vLLM with no accuracy loss:

| Benchmark | vLLM 1.0.0 | TRT-LLM 1.0.0 |
|-----------|-----------|---------------|
| ChartQA — overall | 0.6612 | 0.6688 |
| OCRBench — final (/1000) | 754 | 754 |

---

## 0. Prerequisites

| Requirement | Notes |
|-------------|-------|
| Docker with `--gpus all` | nvidia-container-toolkit must be installed |
| GPU with ≥ 18 GiB free | Qwen3-VL-2B needs ~5 GiB weights + KV/activations |
| Disk | ~50 GiB for images + HF cache |
| Network | For pulling images, HF weights, and the ChartQA / OCRBench TSVs |

Pull the three images once:

```bash
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.0
docker pull nvcr.io/nvidia/eval-factory/vlmevalkit:26.01
```

---

## 1. Files in this directory

| File | Purpose |
|------|---------|
| `serve_model.sh` | Start the **vLLM** server for Qwen3-VL-2B |
| `serve_model_trtllm.sh` | Start the **TRT-LLM** server for Qwen3-VL-2B |
| `run_accuracy_test.sh` | Run vlmevalkit against any OpenAI-compatible endpoint |
| `run_test.sh` | End-to-end wrapper for the vLLM flow (pull + serve + eval + cleanup) |
| `trtllm_extra_config.yaml` | TRT-LLM extra LLM-API options (disables KV block reuse) |
| `strip_proxy.py` | Side-car proxy that strips nemo-evaluator's extra `"dataset"` field — required only for TRT-LLM |
| `accuracy_bench.md` | Setup notes + current baseline results |
| `results_vllm/` | Saved vLLM results |
| `results_trtllm/` | Saved TRT-LLM results |

The eval harness is **endpoint-agnostic** — once either server is up on
`http://localhost:8000/v1/chat/completions`, the same eval script drives both.

---

## 2. vLLM flow

### 2a. Fully automated

```bash
cd benchmarks/multimodal/accuracy
./run_test.sh
```

That script pulls, serves, evals, prints scores, and removes the server container.

### 2b. Manual (useful for debugging)

```bash
# 1. Start server (backgrounded; waits up to 5 min for /health)
./serve_model.sh

# 2. Run evals (server must be reachable on :8000)
RESULTS_DIR="$(pwd)/results_vllm" ./run_accuracy_test.sh

# 3. Stop server
docker rm -f vllm-accuracy-server
```

### vLLM serve flags that matter

| Flag | Value | Why |
|------|-------|-----|
| `--max-model-len` | `8192` | Qwen3-VL defaults to 262K → OOM |
| `--gpu-memory-utilization` | `0.40` | Another container on this host holds ~23 GiB of the 46 GiB L40S; raise to 0.9 if GPU is free |
| `--limit-mm-per-prompt` | `'{"image":4,"video":0}'` | **JSON string** — vLLM 1.0.0 rejects the older `image=4 video=0` form |
| `--trust-remote-code` | on | Qwen3-VL tokenizer |

### vlmevalkit flags that matter

| Flag | Value | Why |
|------|-------|-----|
| `--model_type` | `vlm` | `chartqa` and `ocrbench` reject `chat`; they require `vlm` |
| `--api_key_name` | `DUMMY_API_KEY` | vLLM has no auth; env var must exist |

---

## 3. TRT-LLM flow

Two host-environment fixes are required before this works. Both are already
baked into `serve_model_trtllm.sh`; they are documented here so you know
what to change if you port this elsewhere.

### 3a. Host-driver / CUDA-compat fix

The image is built for **CUDA 13** but many hosts still ship **driver 565.x
(CUDA 12.7)**. `--gpus all` injects the host's `libcuda.so.1` and
`libnvidia-ptxjitcompiler.so.1` into the container at
`/usr/lib/x86_64-linux-gnu/`, which PyTorch picks up first and rejects as
"driver too old".

Force-load the image's forward-compat libs:

```bash
-e LD_PRELOAD="/usr/local/cuda/compat/lib.real/libcuda.so.1 /usr/local/cuda/compat/lib.real/libnvidia-ptxjitcompiler.so.1"
```

**Do not** preload only `libcuda.so.1` — the mismatch with the
host-injected ptxjitcompiler segfaults on the first CUDA call.

### 3b. Request-shape fix (strip proxy)

`nemo-evaluator`'s `CachingInterceptor` adds `"dataset": "<benchmark>"` at the
top of every request body. vLLM ignores unknown fields; TRT-LLM's pydantic
validator returns `400 extra_forbidden`. Run the tiny Python proxy shipped
here on a free port (default `18001`) and point the eval at that port:

```bash
nohup python3 strip_proxy.py > strip_proxy.log 2>&1 &
```

`strip_proxy.py` also supports an override via env: `LISTEN_PORT=XXXXX UPSTREAM=http://127.0.0.1:8000`.

### 3c. TRT-LLM engine-config fix

`kv_cache_config.enable_block_reuse` causes **"Multimodal token count mismatch"**
errors on Qwen3-VL because the block-reuse path does not correctly re-project
vision embeddings. Disable it via `trtllm_extra_config.yaml`:

```yaml
kv_cache_config:
  enable_block_reuse: false
```

### 3d. Steps

```bash
cd benchmarks/multimodal/accuracy

# 1. Start TRT-LLM server (uses LD_PRELOAD + extra-config fixes)
./serve_model_trtllm.sh
#    → first request latency ~60-90s after /health reports ready

# 2. Start the strip-proxy
nohup python3 strip_proxy.py > strip_proxy.log 2>&1 &

# 3. Run evals through the proxy
MODEL_URL="http://localhost:18001/v1/chat/completions" \
RESULTS_DIR="$(pwd)/results_trtllm" \
./run_accuracy_test.sh

# 4. Cleanup
docker rm -f trtllm-accuracy-server
pkill -f strip_proxy.py
```

### TRT-LLM serve flags that matter

| Flag | Value | Why |
|------|-------|-----|
| `--backend` | `pytorch` | HF-direct VLM path; no `trtllm-build` step |
| `--trust_remote_code` | on | Qwen3-VL processor |
| `--max_seq_len` | `8192` | Matches vLLM run |
| `--max_batch_size` | `4` | Conservative on shared GPU |
| `--free_gpu_memory_fraction` | `0.55` | KV-cache fraction of *free* memory (different semantics from vLLM) |
| `--extra_llm_api_options` | `/cfg/trtllm_extra_config.yaml` | Disables KV block reuse |

---

## 4. Eval knobs common to both runtimes

All exposed as environment variables to `run_accuracy_test.sh`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_URL` | `http://localhost:8000/v1/chat/completions` | Where the eval sends requests |
| `RESULTS_DIR` | `$PWD/results` | Output directory (per-benchmark subdirs) |
| `BENCHMARKS` | `"chartqa ocrbench"` | Space-separated list |
| `LIMIT_SAMPLES` | empty (= full) | Set e.g. `10` for a smoke test |

Smoke test against either runtime:

```bash
LIMIT_SAMPLES=10 ./run_accuracy_test.sh
```

---

## 5. Reading results

After a run, each benchmark produces `results/<benchmark>/results.yml`. The
headline number is at `results.groups.<benchmark>.metrics.accuracy.scores.accuracy.value`.

Quick extract:

```bash
python3 -c "import yaml,sys; r=yaml.safe_load(open(sys.argv[1])); \
  print(r['results']['groups'])" results_vllm/chartqa/results.yml
```

OCRBench also writes a human-readable per-subtask file at:
`results_<runtime>/ocrbench/Qwen3-VL-2B-Instruct/T<ts>_G/Qwen3-VL-2B-Instruct_OCRBench_score.json`.

---

## 6. Baseline results (2026-04-19 / 2026-04-20, full test sets)

| Benchmark | vLLM 1.0.0 | TRT-LLM 1.0.0 |
|-----------|-----------|---------------|
| ChartQA — overall | 0.6612 | 0.6688 |
| ChartQA — test-human | 0.5432 | 0.5496 |
| ChartQA — test-augmented | 0.7792 | 0.7880 |
| OCRBench — final (/1000) | 754 | 754 |

Deltas are within run-to-run noise at `temperature=0`. Use this as a
regression floor — any run scoring ≥3 points below these should be
investigated.

See `accuracy_bench.md` for the full per-subtask OCRBench breakdown.

---

## 7. Troubleshooting

| Symptom | Root cause | Fix |
|---------|------------|-----|
| vLLM: `argument --limit-mm-per-prompt: Value image=4 cannot be converted` | Old `key=val` form | Use JSON: `'{"image":4,"video":0}'` |
| vLLM: `Free memory on device cuda:0 ... less than desired GPU memory utilization` | Another container holds the GPU | Lower `--gpu-memory-utilization` (e.g. `0.40`) or free the GPU |
| eval: `MisconfigurationError: 'chartqa' does not support model type 'chat'` | Wrong `--model_type` | Use `vlm` |
| TRT-LLM: `The NVIDIA driver on your system is too old (found version 12070)` | Host driver < image CUDA | Add the two `LD_PRELOAD` compat libs (see 3a) |
| TRT-LLM: Segfault in `libnvidia-ptxjitcompiler.so.1` | Only libcuda preloaded | Preload `libnvidia-ptxjitcompiler.so.1` too |
| TRT-LLM: `{"error":"[{'type':'extra_forbidden','loc':('body','dataset')...}]"}` | Strict pydantic validator | Route requests through `strip_proxy.py` |
| TRT-LLM: `Multimodal token count mismatch: found N image tokens ... received M image embeddings` | `enable_block_reuse` on | Set `kv_cache_config.enable_block_reuse: false` in `trtllm_extra_config.yaml` |
| `Address already in use` on 8001 | Another service owns 8001 | Pick a free port: `LISTEN_PORT=18001 python3 strip_proxy.py` |
