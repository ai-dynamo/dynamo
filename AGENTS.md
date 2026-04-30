# Dynamo Production Rules

1. Treat `deploy/production` as the canonical Kubernetes deployment path for the active REAP serving lane. Do not replace it with a minimal recipe-only deployment.
2. Use `ai-blaise/optimization-playground` as the SGLang inference engine under Dynamo for this lane.
3. The active target model is `BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1`.
4. The initial A4 deployment must keep the full IndexCache + dense TurboQuant + SMC-SD stack enabled. The model should run W4A4KV4 NVFP4 except target KV, which should use dense TurboQuant 2.5-bit.
5. Keep the first working deployment aggregated on `a4-us-001-rl9`; do not enable prefill/decode disaggregation until dense TurboQuant supports that path.
6. Keep private Hugging Face access explicit through `HF_TOKEN` and do not rely on anonymous model access for the BlaiseAI checkpoint.
