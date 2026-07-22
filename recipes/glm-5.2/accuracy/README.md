<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# glm-5.2 — Accuracy Check (GPQA-Diamond)

The checkpoints this recipe serves are already accuracy-validated:
[nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) publishes
GPQA-Diamond results for the quantized model alongside its FP8 baseline.
However, it can still be reassuring to perform a final accuracy check on the
deployed endpoint to ensure no bugs have crept in along the path to deployment
— a stale env var or misconfigured flag can silently corrupt outputs while
throughput looks healthy.

This check is an **A/B test**, not an absolute capability claim: A is the
score measured through this recipe's deployment; B is the checkpoint's
published model-card number. Dataset contamination, if any, affects both
sides equally and cancels out.

## Reference vs measured

| Benchmark | Model card (checkpoint)¹ | This recipe (aiperf)² | Config |
|---|---:|---:|---|
| GPQA-Diamond | 89.39 (NVFP4) | 90.91 (180/198, 0 unparsed) | disagg-b200 |
| GPQA-Diamond | 89.39 (NVFP4) | 88.89 (176/198, 0 unparsed) | agg-b200 |
| GPQA-Diamond | 89.52 (FP8) | 89.39 (177/198, 0 unparsed) | disagg-h200 |
| GPQA-Diamond | 89.52 (FP8) | 87.88 (174/198, 0 unparsed) | agg-h200 |

All four configs score close to their checkpoint's model-card baseline with no
errors or unparsed answers. (B200 recipes serve the NVFP4 checkpoint; H200
recipes serve FP8 — compare each row to its own baseline.) The small spread
between configs (~2-3%) is expected: at temperature=1.0 the model samples
randomly, so two runs on the same 198 questions will naturally differ by a few
answers. Statistically, each score over n=198 questions carries a 95% binomial
confidence interval of about ±4.0-4.6 points, and every model-card baseline
falls within its config's interval (the largest deviation is 0.75 standard
errors) — the differences are not a sign that one serving mode or SKU is more
accurate than another.

¹ Both baselines are from NVIDIA's
  [GLM-5.2-NVFP4 model card](https://huggingface.co/nvidia/GLM-5.2-NVFP4), measured
  at temperature 1.0, top_p 0.95, max_new_tokens 100k: NVFP4 89.39 and the FP8
  baseline 89.52. B200 recipes serve the NVFP4 checkpoint; H200 recipes serve
  [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8), whose own card
  reports 91.2 under Z.ai's harness — the 89.52 baseline is used here because it
  matches this recipe's sampling.
² aiperf accuracy mode: simple-evals prompt template (0-shot, deterministic
  SHA-256 choice shuffling), lighteval letter-extraction grading, same
  sampling as the card. **Different evaluation tools can produce slightly
  different scores even on the same model — compare the general range, not
  exact decimals.** A single run of 198 questions at temperature=1.0 can vary
  by a few percent between runs.

## How it works

1. `accuracy.yaml` waits for the target deployment, then runs
   `aiperf profile --accuracy-benchmark gpqa_diamond`: the 198 real
   GPQA-Diamond questions are sent as ordinary chat requests through the full
   serving path.
2. Each response is graded by extracting the chosen A/B/C/D letter
   (lighteval's extractive matcher) and exact-matching against ground truth.
   Responses with no extractable letter are counted wrong but flagged
   `unparsed` — a nonzero unparsed count means formatting/truncation problems,
   not wrong answers; investigate before trusting the score.
3. The scored summary prints to the job log and to
   `/model-cache/accuracy/<epoch>_<job-name>/accuracy_results.csv` on the
   model-cache PVC (same layout as the perf benchmark).

## Run

```bash
# glm-5-2-gpqa is a fixed-name Job; clear any prior run before re-applying
# (e.g. when switching ENDPOINT to a different target).
kubectl delete job glm-5-2-gpqa -n $NAMESPACE --ignore-not-found
kubectl apply -f accuracy.yaml -n $NAMESPACE     # edit ENDPOINT to pick a variant
kubectl logs -f -l app=glm-5-2-gpqa -n $NAMESPACE
```

## Requirements & caveats

- **Remove `SGLANG_SIMULATE_ACC_LEN`** (and related `SGLANG_SIMULATE_ACC_*`
  envs) from the deployment if present — it pins synthetic spec-decode
  acceptance for perf benchmarking and corrupts outputs. Accuracy must be
  measured with real verification.
- The HF token in `hf-token-secret` must have accepted the gated
  [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) terms.
- `MAX_TOKENS=100000`: reasoning chains average ~24k tokens and occasionally
  approach 100k; a smaller cap truncates chains mid-thought and deflates the
  score via unparsed answers.
- Replica counts don't affect accuracy (it's per-request); a minimal 1-replica
  deployment is sufficient and cheapest.
- aiperf prints an advisory recommending temperature 0 for reproducibility;
  temp 1.0 is used deliberately to match the model-card methodology.
