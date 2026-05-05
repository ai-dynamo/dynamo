# Qwen3-Omni Benchmark Results (DYN-2581)

Fresh pure `vllm-omni serve` baseline results only. Older AIPerf text-only
tables were removed to keep this file scoped to the current DYN-2581
mixed-modality experiment.

## Mixed-Modality E2E — `vllm-omni serve` v0.20.0rc1

Run date: 2026-05-04

Environment:

| Field | Value |
|---|---|
| Cluster context | `nv-prd-dgxc.teleport.sh-dynamo-nebius-2` |
| Namespace | `ptarasiewicz-test` |
| 2-GPU node | `computeinstance-e01sbamysmcch7djdn` |
| 3-GPU node | `computeinstance-e01dckvphr0q1xphw5` |
| GPU node type | Nebius `gpu-h200-sxm`, `nvidia.com/gpu.product=NVIDIA-H200` |
| 2-GPU deployment | `dyn2581-qwen3-omni-v020rc1` |
| 3-GPU deployment | `dyn2581-qwen3-omni-v020rc1-3gpu` |
| Image | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.2` |
| Image digest | `nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:ecab3c0b376bda94748c30ca980063580a0154647c4b0495f3d76f6667589815` |
| Runtime versions | `vllm 0.20.0`, `vllm-omni 0.20.0rc1`, `ai-dynamo 1.2.0` |
| `vllm-omni` source | `git+https://github.com/vllm-project/vllm-omni.git@v0.20.0rc1` (`704b6bb6e6315bb500f9418a8d53c079fc21017f`) |
| Model | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| Serving stack | Pure `vllm-omni serve`; Dynamo container is used only as the runtime image |
| 3-GPU stage placement | Stage 0 thinker on GPU 0, stage 1 talker on GPU 1, stage 2 code2wav on GPU 2 |

Request shape:

- Inputs: text + image URL + audio URL + video URL in one `/v1/chat/completions` request.
- Outputs: `modalities=["text","audio"]` for the sweep rows below.
- 2-GPU client: `benchmarks/omni/qwen3/run_mixed_modalities.py` through a local `kubectl port-forward`.
- 3-GPU client: same driver copied into the serving pod and run through `kubectl exec` against `http://127.0.0.1:8000` to remove local port-forward instability from the measurements.
- 2-GPU artifacts: `benchmarks/omni/qwen3/results/v020rc1_vllm_serve/`.
- 3-GPU artifacts: `benchmarks/omni/qwen3/results/v020rc1_vllm_serve_3gpu/`.

### Text + Audio Output — 2 GPUs

| Concurrency | Requests | HTTP 200 | Full text+audio | Req/s | Full req/s | Avg E2E (ms) | p50 E2E (ms) | p90 E2E (ms) | p99 E2E (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 3 | 0.264 | 0.264 | 3,789 | 4,035 | 4,203 | 4,240 |
| 4 | 8 | 8 | 8 | 0.712 | 0.712 | 5,507 | 5,556 | 6,433 | 6,647 |
| 8 | 16 | 16 | 16 | 0.941 | 0.941 | 8,335 | 8,351 | 9,054 | 9,312 |
| 16 | 32 | 32 | 32 | 1.107 | 1.107 | 13,628 | 12,739 | 20,298 | 21,097 |
| 32 | 32 | 32 | 32 | 1.545 | 1.545 | 18,559 | 19,588 | 20,253 | 20,674 |
| 64 | 64 | 64 | 64 | 1.786 | 1.786 | 33,260 | 34,377 | 35,421 | 35,810 |
| 128 | 128 | 128 | 124 | 2.277 | 2.205 | 45,729 | 47,215 | 55,018 | 55,856 |
| 256 | 256 | 256 | 252 | 2.516 | 2.476 | 72,986 | 75,667 | 98,729 | 100,866 |
| 512 | 512 | 512 | 511 | 2.460 | 2.455 | 135,625 | 143,882 | 201,828 | 207,431 |

### Text + Audio Output — 3 GPUs

The 3-GPU deployment uses
[`k8s/vllm_serve_v020rc1_3gpu.yaml`](k8s/vllm_serve_v020rc1_3gpu.yaml),
which passes a custom `--stage-configs-path` so the three Qwen3-Omni stages
are placed on separate GPUs.

| Concurrency | Requests | HTTP 200 | Full text+audio | Req/s | Full req/s | Avg E2E (ms) | p50 E2E (ms) | p90 E2E (ms) | p99 E2E (ms) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3 | 3 | 3 | 0.242 | 0.242 | 4,122 | 4,114 | 4,238 | 4,266 |  |
| 4 | 8 | 8 | 8 | 0.479 | 0.479 | 8,199 | 8,341 | 9,086 | 9,161 |  |
| 8 | 16 | 16 | 16 | 0.380 | 0.380 | 20,718 | 21,042 | 28,299 | 28,388 |  |
| 16 | 32 | 32 | 32 | 0.419 | 0.419 | 37,770 | 37,911 | 38,516 | 39,381 |  |
| 32 | 32 | 32 | 32 | 0.334 | 0.334 | 94,358 | 94,677 | 95,533 | 95,853 |  |
| 48 | 48 | 0 | 0 | 0.000 | 0.000 | - | - | - | - | Stalled after stage-0 admission; all GPUs idle; client killed |
| 64 | 64 | 0 | 0 | 0.000 | 0.000 | - | - | - | - | Stalled after stage-0 admission; all GPUs idle; client killed |

Smoke checks:

| Modalities | Concurrency | Requests | Successes | Req/s | p50 E2E (ms) | Notes |
|---|---:|---:|---:|---:|---:|---|
| `text` | 1 | 1 | 1 | 0.226 | 4,421 | Same mixed inputs, text-only output |
| `text,audio` | 1 | 1 | 1 | 0.192 | 5,202 | Returned 762,820 base64 audio chars |

Current read: on the 2-GPU pure `vllm-omni serve` baseline, raw HTTP
throughput peaks at concurrency 256 for this request shape, at 2.516 req/s.
The practical knee is lower: throughput gains flatten after c128/c256 while
p50 latency rises from ~47.2s at c128 to ~75.7s at c256 and ~143.9s at c512.
A few high-concurrency HTTP-200 responses returned audio without a text choice,
so `Full text+audio` is tracked separately from HTTP success.

The 3-GPU split did not improve this e2e workload. Its best completed point is
c4 at 0.479 req/s, and throughput falls at c8/c16/c32. Both c48 and c64 stalled
after requests were admitted to stage 0, with no stage progress and idle GPUs;
the benchmark client was killed and the server logged request aborts. Because
c48/c64 stall, c128+ was not run for this topology.
