<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sweeper DGD comparison cases

This directory owns model, backend, and hardware inputs used by two complementary paths:

- automatic `aisim recommend --stack dynamo --output dgd` coverage through the default AIC
  generator; and
- manual comparison of the historical DGDR v1beta1 profiler, Sweeper AIC renderer, and Sweeper
  direct renderer.

See [DESIGN.md](DESIGN.md) for the file contracts and the distinction between cases, hardware
configurations, suites, goldens, and local diagnostics.

`testsuite-issue-8469.yaml` retains the rows from issue #8469. Every row has paired v1beta1 and
Sweeper inputs. Known rendering problems are recorded on the affected suite row with a reason and
optional evidence links. Only an explicit `match` regex accepts a known failure; reason-only
historical entries remain diagnostic and fail closed. `exceptions.recommend.aic` records deliberate
automatic CLI failures with an exact diagnostic regex. A missing generated variant has no
placeholder golden.

Every suite row runs as a parametrized Planner CPU integration test in
[`test_model_corpus.py`](../test_model_corpus.py) and
[`test_model_corpus_part2.py`](../test_model_corpus_part2.py). The split lets the two Planner CPU
workers share the full table instead of serializing it as one pytest module. Each test composes the
case and hardware inputs, executes one seeded public-CLI recommendation suggestion, and requires
either a successful default-AIC DGD with a checked-in parsed-YAML golden or a deliberate negative
expectation with an exact diagnostic. One suggestion is sufficient for model/backend/hardware and
output-adapter smoke coverage, avoids architecture-sensitive winner selection between candidates,
and keeps the full table inside the pre-merge CPU budget. Tests generate into pytest's temporary
directory. A golden mismatch writes an adjacent `.new.yaml` proposal and prints the unified diff and
acceptance command in the failing test output. The smaller
[DGD CLI E2E table](../e2e/test_dgd_cli.py) uses the same command and assertions with tiny fixed
searches. Neither path deploys generated manifests or recipes.

The portable runner and comparison-corpus direction build on Ashna Mehrotra's work in
[PR #14031](https://github.com/ai-dynamo/dynamo/pull/14031).

## Generate one combination

From the repository root:

```bash
python components/src/dynamo/aisimulate/tests/dgd/comparison/run_cases.py \
  --hardware h200-sxm-16gpu \
  qwen3-32b-vllm-disagg
```

## Generate a suite

```bash
python components/src/dynamo/aisimulate/tests/dgd/comparison/run_cases.py \
  --suite components/src/dynamo/aisimulate/tests/dgd/comparison/testsuite-issue-8469.yaml
```

The **AI Simulate DGD comparison** Actions workflow runs the same command on demand. It can generate
the full suite or selected cases on one hardware profile, including the `h100-sxm-2gpu` profile,
and uploads the generated DGDs and diagnostics without the local model caches. It runs in the
matching planner test image. On `main`, it walks first-parent history to the newest commit with a
published test image, covering documentation-only commits that intentionally skip image builds; on
another ref, that ref's exact image must already exist.

Use `--output-dir` to write into a temporary directory instead of updating the checked-in
`generated/<suite-name>/` tree. Individual `--hardware` runs write under the ignored
`generated/manual/` tree unless `--output-dir` is set.

Each Sweeper case performs one search. The selected Candidate is passed to both the AIC and direct
renderers, so renderer differences cannot be caused by separate searches.

All case inputs explicitly set the AISimulate search algorithm and seed. The
repository's exact AISimulate wheel pin supplies the model metadata, performance data, sampler, and
renderers used by the Planner test image.

Final DGDs use type-first names and are intended to be complete manifests:

```text
generated/<suite-name>/<hardware>/<case>/dgd-profiler-v1beta1.yaml
generated/<suite-name>/<hardware>/<case>/dgd-sweeper-aic.yaml
generated/<suite-name>/<hardware>/<case>/dgd-sweeper-direct.yaml
```

Only successfully generated variants are present in a case directory.

For example:

```bash
kubectl apply -f \
  components/src/dynamo/aisimulate/tests/dgd/comparison/generated/testsuite-issue-8469/h200-sxm-16gpu/qwen3-32b-vllm-disagg/dgd-sweeper-aic.yaml
```

Composed inputs, the selected Candidate, caches, and error files are local ignored diagnostics.
The suite intentionally defines no custom report format.

This corpus does not exercise DGDR v1beta2 Request, Run, Candidate, or Job orchestration. The
Kubernetes smoke coverage for that integration lives in `tests/deploy/test_dgdr_v1beta2.py`.
