# Qwen3-Omni Agg vs Disagg vs `vllm-omni serve` (DYN-2581)

Mixed-modality (chat + audio) sweep that compares three deployment topologies
of Qwen3-Omni-MoE at the same hardware envelope:

| Topology     | Recipe                                          | Endpoint shape |
|--------------|-------------------------------------------------|----------------|
| `agg`        | [`recipes/qwen3-omni/vllm/agg/`](../../../recipes/qwen3-omni/vllm/agg/)               | One AsyncOmni instance, all stages colocated |
| `disagg`     | [`recipes/qwen3-omni/vllm/disagg/`](../../../recipes/qwen3-omni/vllm/disagg/)         | OmniRouter + Thinker + Talker + Code2Wav     |
| `vllm_serve` | [`recipes/qwen3-omni/vllm/vllm-serve/`](../../../recipes/qwen3-omni/vllm/vllm-serve/) | Upstream `vllm-omni serve`                   |

All three serve text and audio output through the same OpenAI-compatible
endpoints (`/v1/chat/completions`, `/v1/audio/speech`).

## Hardware

- 2× H100-80GB per topology (recipes target Hopper).
- The sweep itself runs from a CPU-only client pod / laptop — `aiperf` only
  needs the server URL.

## What the sweep measures

For each topology:

| Axis        | Values                       |
|-------------|------------------------------|
| Workload    | `chat`, `audio`              |
| Concurrency | `1, 4, 8, 16, 32`            |
| Prompt len  | short (~64 tok), long (~512) |

Per cell, AIPerf reports TTFT, ITL, request latency (p50/p90/p99) and request
+ output-token throughput. Artifacts land in
`benchmarks/omni/qwen3/results/<topology>/<workload>_c<c>_isl-<short|long>/`.

## Run

Stand up each deployment first (see the per-recipe README), then point the
sweep at it. From the sweep pod / box (Hopper cluster):

```bash
# 1. Smoke each topology (one cell per workload):
bash benchmarks/omni/qwen3/run_sweep.sh --topology agg        --url http://agg-qwen3-omni-frontend:8000        --quick
bash benchmarks/omni/qwen3/run_sweep.sh --topology disagg     --url http://disagg-qwen3-omni:8000              --quick
bash benchmarks/omni/qwen3/run_sweep.sh --topology vllm_serve --url http://vllm-serve-qwen3-omni:8000          --quick

# 2. Full grid:
bash benchmarks/omni/qwen3/run_sweep.sh --topology agg        --url http://agg-qwen3-omni-frontend:8000
bash benchmarks/omni/qwen3/run_sweep.sh --topology disagg     --url http://disagg-qwen3-omni:8000
bash benchmarks/omni/qwen3/run_sweep.sh --topology vllm_serve --url http://vllm-serve-qwen3-omni:8000

# 3. Aggregate to markdown:
python3 benchmarks/omni/qwen3/analyze.py
# -> benchmarks/omni/qwen3/RESULTS.md
```

## Mixed-Modality E2E

For the heaviest Qwen3-Omni request shape, use the mixed-modality driver. It
sends text, image, audio, and video inputs in a single `/v1/chat/completions`
request and asks for both text and audio output:

```bash
python3 benchmarks/omni/qwen3/run_mixed_modalities.py \
  --url http://vllm-serve-qwen3-omni:8000 \
  --concurrency 1 \
  --requests 3 \
  --warmup 1 \
  --output benchmarks/omni/qwen3/results/vllm_serve/mixed_c1/result.json
```

For local-port-forwarded URLs (laptop client against a single port-forwarded
service), substitute `http://localhost:<port>`.

The fresh vLLM 0.20 / `vllm-omni` 0.20.0rc1 Kubernetes deployment used for
DYN-2581 is captured in
[`k8s/vllm_serve_v020rc1.yaml`](k8s/vllm_serve_v020rc1.yaml).
It uses the Dynamo image only as a CUDA/Python runtime and starts the pure
`vllm-omni serve` frontend directly.

The 3-GPU variant is captured in
[`k8s/vllm_serve_v020rc1_3gpu.yaml`](k8s/vllm_serve_v020rc1_3gpu.yaml). It
uses the same pure `vllm-omni serve` stack with a mounted stage config that
places thinker, talker, and code2wav on GPUs 0, 1, and 2 respectively.

## Optional GPU + transfer telemetry

For one representative cell per topology layer
[`benchmarks/frontend/scripts/run_perf.sh`](../../frontend/scripts/run_perf.sh)
on top of `aiperf profile` to capture Prometheus / nsys / flamegraph data.
This is "good to have" per the Linear ticket — skip if the AIPerf grid alone
is conclusive.

## Output

`analyze.py` writes `RESULTS.md` next to itself. Cross-topology comparison
beyond simple percentile tables (Mann-Whitney U tests, regression detection)
is available via:

```bash
python3 benchmarks/frontend/scripts/analysis/frontend_perf_analysis.py compare \
  benchmarks/omni/qwen3/results/agg/chat_c8_isl-long \
  benchmarks/omni/qwen3/results/disagg/chat_c8_isl-long
```
