# Multimodal Benchmark Sweep

YAML-driven benchmark orchestrator that launches serving backends, runs
[aiperf](https://github.com/triton-inference-server/perf_analyzer) concurrency
sweeps, and optionally generates comparison plots.

## Quick Start

```bash
# from the repo root
python -m benchmarks.multimodal.sweep \
  --config benchmarks/multimodal/sweep/experiments/embedding_cache/vllm_serve.yaml
```

## How It Works

1. Parse the YAML experiment config.
2. For each **input file** × each **benchmark config**:
   - Launch the serving backend via the workflow script.
   - Run `aiperf profile` at every concurrency level.
   - Stop the server (by default the server restarts between concurrency
     levels to avoid warm-cache bias — controlled by
     `restart_server_every_benchmark`).
3. Generate comparison plots across configs for each input file.

## YAML Config Reference

```yaml
model: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
concurrencies: [16, 32, 64, 128, 256]
osl: 150                    # output sequence length
conversation_num: 10        # sessions per sweep value (optional; derived from
                            # input JSONL's unique session_id count if unset;
                            # flat JSONLs count each row as a 1-turn conversation)
warmup_count: 5
port: 8000
timeout: 900                # seconds to wait for server readiness
output_dir: benchmarks/multimodal/sweep/results/vllm_serve

# Optional env vars injected into the server process
env:
  ENABLE_ENCODER_CACHE: "0"

# JSONL files produced by benchmarks/multimodal/jsonl/
input_files:
  - benchmarks/multimodal/jsonl/1000req_1img_200pool_400word_http.jsonl
  - benchmarks/multimodal/jsonl/1000req_4img_200pool_400word_http.jsonl

# Each config launches the workflow with its own extra_args
configs:
  - label: cache-off
    workflow: benchmarks/multimodal/sweep/workflows/vllm_serve.sh
    extra_args: [--no-enable-prefix-caching, --multimodal-embedding-cache-capacity-gb, "0"]

  - label: cache-on
    workflow: benchmarks/multimodal/sweep/workflows/vllm_serve.sh
    extra_args: [--no-enable-prefix-caching, --multimodal-embedding-cache-capacity-gb, "10"]
```

## CLI Overrides

Any top-level YAML field can be overridden from the command line:

```bash
python -m benchmarks.multimodal.sweep \
  --config experiments/embedding_cache/vllm_serve.yaml \
  --concurrencies 1,2,4 \
  --osl 200 \
  --conversation-num 10 \
  --skip-plots
```

## Warmup semantics

`warmup_count: N` is a **request (turn) budget**, not a session budget. For a
10×10 JSONL with `warmup_count: 2`, warmup issues 2 total requests — both go
to `user_0` (turns 0 and 1) because aiperf's continuation-turn priority keeps
feeding the in-flight session until its budget runs out. Warmup does NOT
consume 2 full sessions (20 requests). Profiling then starts at `user_1`,
runs `user_1..user_9` to completion, and wraps to a fresh `user_0` instance
for the 10th session. Keep `warmup_count` small (≤ turns-per-session) so
warmup stays within a single session's prefix.

## Output Directory Structure

Given the config above with two input files and two configs (`cache-off`,
`cache-on`) at concurrencies `[16, 32]`, the output tree looks like:

```
<output_dir>/
├── 1000req_1img_200pool_400word_http/      # ← derived from input filename
│   ├── cache-off/                          # ← config label
│   │   ├── c16/                            # ← concurrency level
│   │   │   ├── profile_export.jsonl
│   │   │   ├── profile_export_aiperf.json
│   │   │   ├── profile_export_aiperf.csv
│   │   │   ├── gpu_telemetry_export.jsonl
│   │   │   ├── inputs.json
│   │   │   └── logs/
│   │   │       └── aiperf.log
│   │   └── c32/
│   │       └── ...
│   ├── cache-on/
│   │   ├── c16/
│   │   │   └── ...
│   │   └── c32/
│   │       └── ...
│   └── plots/                              # ← comparison plots across configs
│       └── ...
└── 1000req_4img_200pool_400word_http/
    ├── cache-off/
    │   └── ...
    ├── cache-on/
    │   └── ...
    └── plots/
        └── ...
```

## Existing Experiments

| Experiment | Config | Backend |
|---|---|---|
| Embedding cache (vLLM serve) | `experiments/embedding_cache/vllm_serve.yaml` | Single-node vLLM |
| Embedding cache (vLLM E+PD) | `experiments/embedding_cache/vllm_e_pd.yaml` | Disaggregated vLLM E+PD |
| Embedding cache (TRT-LLM E+PD) | `experiments/embedding_cache/trtllm_e_pd.yaml` | Disaggregated TRT-LLM E+PD |
