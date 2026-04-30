# Dynamo Production Rules

1. Treat `deploy/production` as the canonical Kubernetes deployment path for the active REAP serving lane. Do not replace it with a minimal recipe-only deployment.
2. Use `ai-blaise/optimization-playground` as the SGLang inference engine under Dynamo for this lane.
3. The active target model is `BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1`.
4. The active A4 deployment is disaggregated P/D on one 8-GPU B200 node: 4 GPUs for prefill, 4 GPUs for decode, DP=4, TP=4, and decode-only SMC-SD. Do not remove SMC-SD when changing the target KV/cache path.
5. Current SGLang treats HiSparse, HiCache, IndexCache, and dense TurboQuant as separate KV-cache paths. For the production profile, prefer decode-side HiSparse over HiCache, IndexCache, and dense TurboQuant until a combined path is deliberately implemented and revalidated.
6. The model should run W4A4KV4 NVFP4 via `--quantization modelopt_fp4`; target KV uses BF16 in the HiSparse profile because HiSparse requires `--kv-cache-dtype bfloat16`.
7. Keep Dynamo event-backed KV-aware routing enabled through frontend `--router-mode kv --router-kv-events` and worker `--kv-events-config`.
8. Keep private Hugging Face access explicit through `HF_TOKEN` and do not rely on anonymous model access for the BlaiseAI checkpoint.
