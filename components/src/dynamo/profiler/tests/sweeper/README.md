<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sweeper DGD comparison cases

This directory compares DGDs generated for the same deployment intent by the existing DGDR
v1beta1 profiler and AI Simulate Sweeper. See [DESIGN.md](DESIGN.md) for the file contracts and
the distinction between cases, hardware configurations, suites, goldens, and local diagnostics.

`testsuite-issue-8469.yaml` retains all 29 rows from issue #8469. Every row has paired v1beta1 and
Sweeper inputs. Known render and deployment exceptions are recorded on the affected suite row with
a reason and optional evidence links. A missing generated variant has no placeholder golden.

The portable runner and live-validation direction build on Ashna Mehrotra's work in
[PR #14031](https://github.com/ai-dynamo/dynamo/pull/14031).

## Generate one combination

From the repository root:

```bash
python components/src/dynamo/profiler/tests/sweeper/run_cases.py \
  --hardware h200-sxm-16gpu \
  qwen3-32b-vllm-disagg
```

## Generate a suite

```bash
python components/src/dynamo/profiler/tests/sweeper/run_cases.py \
  --suite components/src/dynamo/profiler/tests/sweeper/testsuite-issue-8469.yaml
```

Use `--output-dir` to write into a temporary directory instead of updating the checked-in
`generated/<suite-name>/` tree. Individual `--hardware` runs write under the ignored
`generated/manual/` tree unless `--output-dir` is set.

Each Sweeper case performs one search. The selected Candidate is passed to both the AIC and direct
renderers, so renderer differences cannot be caused by separate searches.

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
  components/src/dynamo/profiler/tests/sweeper/generated/testsuite-issue-8469/h200-sxm-16gpu/qwen3-32b-vllm-disagg/dgd-sweeper-aic.yaml
```

Composed inputs, the selected Candidate, caches, and error files are local ignored diagnostics.
The suite intentionally defines no custom report format.

## Validate generated DGDs on a cluster

After generating the goldens, run the deployment test against the same suite:

```bash
pytest -q tests/deploy/test_sweeper_cases.py \
  --sweeper-suite components/src/dynamo/profiler/tests/sweeper/testsuite-issue-8469.yaml \
  --namespace <namespace>
```

The default variants are the v1beta1 profiler output, Sweeper-AIC output, and an eligible recipe.
Select a different set with `--sweeper-variants`, for example
`--sweeper-variants profiler-v1beta1,sweeper-aic,sweeper-direct,recipe`. The test rejects a suite
entry when its hardware family or total GPU budget is unavailable on the cluster.

To probe a recipe on hardware not yet listed in its `recipe.yaml`, add
`--sweeper-discover-recipe-hardware`. After successful deployment and inference, the test writes
the proposed requirement to the ignored adjacent `recipe.new.yaml`; it never edits `recipe.yaml`.
