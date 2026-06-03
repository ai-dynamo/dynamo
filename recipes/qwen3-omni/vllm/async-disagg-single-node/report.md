# Qwen3-Omni Async Disagg Single-Node H200 Report

Date: 2026-06-03

## Summary

Qwen3-Omni audio end-to-end works on Nebius H200 with the async disaggregated vLLM-Omni path built on top of `ai-dynamo/dynamo#10208` and the local stage-router/output-format fixes in this workspace.

The reproducible recipe comparison is:

- `deploy-baseline-1h1t1c.yaml`: 1 thinker, 1 talker, 1 code2wav on 3x H200.
- `deploy-scaled-1h2t1c.yaml`: 1 thinker, 2 talkers, 1 code2wav on 4x H200.

For the 512-token long-audio workload, `1h2t1c` reached 173.212 generated audio seconds/s at concurrency 256, or 43.303 audio s/s/GPU. The recorded `1h1t1c` baseline reached 96.578 generated audio seconds/s at concurrency 128, or 32.193 audio s/s/GPU. That is a 79.4% absolute throughput improvement and a 34.5% per-GPU throughput improvement for `1h2t1c`.

## Runtime

Image:

`nvcr.io/nvidian/dynamo-dev/dynamo:vllm-runtime-ptarasiewicz-dyn3068-qwen3-async-38584b4-r20-current-omni-20260602`

Cluster placement:

- Namespace: `ptarasiewicz-test`
- GPU type: `NVIDIA-H200`
- Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Model cache PVC: `model-cache`
- Pull secrets: `nvcr-dynamo-dev-pullsecret`, `nvcr-imagepullsecret`
- Hugging Face secret: `hf-token-secret`

## Workload

Benchmark script:

`benchmarks/omni/audio/chat_audio_benchmark.py`

The script sends streaming OpenAI-compatible chat completion requests with:

- `modalities=["audio"]`
- WAV output
- voice `Chelsie`
- `max_tokens=512`
- temperature `0.0`

Prompt:

```text
Read the following paragraph clearly and naturally: Independent stage scaling lets a disaggregated multimodal serving stack add capacity to the stage that is saturated. In this workload the audio talker must produce a longer spoken response, while the thinker and code to waveform stages should remain shared. This benchmark repeats the same paragraph to measure sustained audio generation throughput under concurrent requests.
```

## Results

All rows below completed with zero request failures.

| Recipe | GPUs | Artifact | Best concurrency | OK/fail | Req/s | Audio s/s | Req/s/GPU | Audio s/s/GPU |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `1h1t1c` | 3 | `results/baseline-1t1t1c-r20-retry-c80-c128-benchmark.json` | 128 | 256/0 | 3.015 | 96.578 | 1.005 | 32.193 |
| `1h2t1c` | 4 | `results/rate-match-1h2t1c-heavy-c256-c384.json` | 256 | 512/0 | 5.424 | 173.212 | 1.356 | 43.303 |

The `1h2t1c` run also stayed flat at higher concurrency, which confirms the c256 point is a plateau rather than a narrow spike:

| Recipe | Concurrency | OK/fail | Req/s | Audio s/s |
|---|---:|---:|---:|---:|
| `1h2t1c` | 256 | 512/0 | 5.424 | 173.212 |
| `1h2t1c` | 320 | 640/0 | 5.414 | 172.612 |
| `1h2t1c` | 384 | 768/0 | 5.418 | 172.185 |

## Stage Conclusions

- Adding a thinker alone did not materially move the long-audio workload (`2h1t1c` stayed near baseline).
- Adding code2wav alone did not materially move the workload while there was only one talker (`1h1t2c` stayed near baseline).
- Adding a second talker produced the first clear gain (`1h2t1c`), so talker was the first bottleneck for this payload.
- Later 8-GPU rate matching showed one code2wav becomes the next bottleneck at high talker counts, but the clean per-GPU recipe to present in this PR is `1h2t1c`.

## Reproduction

Use `README.md` in this directory for exact deployment, benchmark, artifact extraction, and cleanup commands.

Cleanup status from the validation run: all Qwen3-Omni benchmark deployments in `ptarasiewicz-test` were scaled to zero and benchmark jobs from the latest sweeps were deleted after logs were saved.
