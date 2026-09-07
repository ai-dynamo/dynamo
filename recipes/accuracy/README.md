<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Accuracy Check

A model-agnostic accuracy check for any deployed recipe: `accuracy.yaml` runs a
public benchmark against a live Dynamo endpoint with [AIPerf](https://github.com/ai-dynamo/aiperf)
and grades the answers.

AIPerf ships graders for a range of benchmarks, among them MMLU and MMLU-Pro,
GSM8K, MATH-500, AIME, GPQA-Diamond, HellaSwag, BIG-Bench Hard, and
LiveCodeBench. The example defaults to GPQA-Diamond; see
[Accuracy Benchmarking](https://github.com/ai-dynamo/aiperf/blob/main/docs/accuracy/accuracy-benchmarking.md)
for the full list, each one's dataset and default few-shot setting, and the
benchmark-specific flags (chain-of-thought prompting, for instance).

Accuracy is usually evaluated alongside throughput and latency when validating a
deployment, to confirm the deployed configuration performs in line with the
checkpoint's published results.

Read the result as an **A/B test**, not an absolute capability claim: A is the
score measured through your deployment, B is the checkpoint's published
model-card number for the same benchmark.

## How it works

1. The Job waits for the target deployment to advertise the model on
   `/v1/models`, then runs `aiperf profile --accuracy-benchmark <benchmark>`:
   the benchmark's questions are sent as ordinary chat requests through the
   full serving path.
2. AIPerf grades each response against ground truth. Responses it cannot parse
   an answer out of are counted wrong but reported separately as `unparsed`. A
   nonzero unparsed count means formatting or truncation problems rather than
   wrong answers, and should be investigated before trusting the score.
3. The scored summary prints to the Job log and is written to
   `/model-cache/accuracy/<epoch>_<job-name>/accuracy_results.csv` on the
   model-cache PVC (the same layout the perf benchmarks use).

## Run

Point the Job at a deployed recipe by editing the `CONFIGURE` block at the top
of its env list, then apply it:

| Variable | Set it to |
|---|---|
| `ENDPOINT` | Frontend service of the deployed DGD, e.g. `<deployment>-frontend:8000` |
| `TARGET_MODEL` | Model ID exactly as the deployment advertises it on `/v1/models`; it also fetches the tokenizer, so it must resolve on Hugging Face |
| `BENCHMARK` | Benchmark to run. See [available benchmarks](https://github.com/ai-dynamo/aiperf/blob/main/docs/accuracy/accuracy-benchmarking.md#available-benchmarks) |
| `NUM_REQUESTS` | The benchmark's full dataset size, so the score is comparable to a published number |
| `TEMPERATURE`, `TOP_P`, `MAX_TOKENS` | Match the sampling of the model card being compared against |

```bash
# accuracy-check is a fixed-name Job; clear any prior run before re-applying
# (for example when switching ENDPOINT to a different target).
kubectl delete job accuracy-check -n $NAMESPACE --ignore-not-found
kubectl apply -f accuracy.yaml -n $NAMESPACE
kubectl logs -f job/accuracy-check -n $NAMESPACE
```

## Reading the result

Compare the score to the checkpoint's model card, not to a fixed threshold, and
compare the general range rather than exact decimals:

- **Match the sampling.** Scores are only comparable when temperature, top_p,
  and token limits match the methodology the baseline was measured under. If a
  card publishes numbers for several quantizations, compare against the one the
  recipe actually serves.
- **Harnesses differ.** Different evaluation tools produce slightly different
  scores for the same model on the same benchmark, so a small offset from a
  published number is not by itself a defect.
- **One run is not a measurement.** On a few-hundred-question benchmark at
  nonzero temperature, two runs of the same deployment will differ by a few
  answers, which is a few points of score. When a difference matters (comparing
  two serving modes, or deciding whether a gap from the model card is real), run
  it more than once and look at the spread rather than trusting a single number.
- **Check `unparsed` first.** A nonzero unparsed count usually means responses
  are being truncated. Raise `MAX_TOKENS` and re-run before reading anything
  into the score.

## Requirements and caveats

- **Remove perf-only synthetic settings from the deployment first.** Some
  recipes pin a simulated speculative-decode acceptance length for performance
  benchmarking (on SGLang, the `SGLANG_SIMULATE_ACC_*` environment variables).
  Those bypass real draft-token verification, so the outputs are not
  representative. Accuracy must be measured with verification on.
- The `hf-token-secret` is optional: the Job starts without it, and it is only
  needed for gated models or datasets. For a gated benchmark (GPQA-Diamond, for
  example) the token must have accepted the dataset's terms.
- The Job mounts a `model-cache` PVC for the tokenizer, dataset, and results.
  Most recipes create one under that name; if the recipe under test uses a
  different one, update `claimName` in the manifest, or the pod sits `Pending`
  with no other symptom.
- **Give reasoning models generous `MAX_TOKENS`.** Reasoning chains can run to
  tens of thousands of tokens; a low cap truncates them mid-thought and
  deflates the score through unparsed answers.
- **Check the time budget before a long run.** `activeDeadlineSeconds` bounds
  the entire Job, including the readiness wait and any retry. Runtime scales
  with the request count and per-request latency: a 198-question benchmark
  against a large reasoning model takes hours, and a larger benchmark scales
  from there. If the budget is exceeded the pod is killed mid-run and no score
  is printed, which looks like a failure rather than a timeout. Raising
  `CONCURRENCY` shortens the run — it sets how many requests are in flight, not
  what is asked — so it is the first lever to reach for before extending the
  deadline.
- Replica counts do not affect accuracy (it is per-request), so a minimal
  1-replica deployment is sufficient and cheapest.
- AIPerf prints an advisory recommending temperature 0 for reproducibility.
  Matching the model card's sampling takes precedence, since that is what makes
  the comparison valid.
- This is not a perf benchmark. Use the recipe's `perf.yaml` for
  throughput and latency.
