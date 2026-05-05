# Dynamo Production Rules

1. Treat `deploy/production` as the canonical Kubernetes deployment path for the active REAP serving lane. Do not replace it with a minimal recipe-only deployment.
2. Use `ai-blaise/optimization-playground` as the SGLang inference engine under Dynamo for this lane.
3. The active target model is `BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1`.
4. The active A4 deployment is disaggregated P/D on one 8-GPU B200 node: 4 GPUs for prefill, 4 GPUs for decode, DP=4, TP=4, and decode-only SMC-SD. Do not remove SMC-SD when changing the target KV/cache path.
5. The production profile uses the combined SGLang HiSparse + IndexCache + dense TurboQuant path. Keep HiCache disabled unless the HiSparse no-radix contract is deliberately reworked and revalidated. Do not add LayerSplit flags to the 4+4 DP=4 profile until the prefill worker uses an effective attention CP size greater than 1.
6. The model should run W4A4KV4 NVFP4 through the checkpoint's `compressed-tensors` quantization metadata; target KV uses BF16 source dtype with dense TurboQuant 2.5-bit compressed MLA storage.
7. Keep Dynamo event-backed KV-aware routing enabled through frontend `--router-mode kv --router-kv-events` and worker `--kv-events-config`.
8. Keep private Hugging Face access explicit through `HF_TOKEN` and do not rely on anonymous model access for the BlaiseAI checkpoint.
