<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Planner Manual Workflows

This directory keeps the old planner dry-run and scaling helpers that are still useful for
manual validation, but are intentionally not part of the automated `pytest -m planner`
suite.

The files here are for:
- local dry runs against profiling data and a synthetic dataset
- manual Kubernetes scaling checks
- example deployment manifests for planner perf experiments

The files here are not for:
- automated CI
- tests that require external backends or cluster state to be brought up by pytest

## Shared Data

The planner container already carries the shared profiling data used by the automated tests:

```text
/workspace/components/src/dynamo/planner/tests/data/profiling_results/H200_TP1P_TP1D
```

When the examples below refer to `profile_results_dir`, use that path inside the container.

## Dry Run

From the repo root:

```bash
python3 components/src/dynamo/planner/manual/unit/planner_sla_dryrun.py \
  --config '{"environment":"kubernetes","backend":"vllm","ttft":200,"itl":10,"profile_results_dir":"components/src/dynamo/planner/tests/data/profiling_results/H200_TP1P_TP1D","throughput_adjustment_interval":60,"no_correction":true}' \
  --dataset rr-5-45_i3000o300.jsonl \
  --start-num-p 1 \
  --start-num-d 1 \
  --output-plot dryrun_plot.png
```

## Manual Scaling

The manual scaling entrypoint is:

```bash
components/src/dynamo/planner/manual/scaling/run_scaling_test.sh --namespace <namespace>
```

Supported modes:
- `--mode throughput`
- `--mode load`

With `--save-results`, the script reuses the shared planner load-generator helper and writes
aiperf artifacts under:

```text
components/src/dynamo/planner/tests/e2e_scaling_results/
```

## Example Deployment Manifests

- `perf_test_configs/`: example aggregated and disaggregated perf test manifests
- `scaling/`: manual planner scaling manifests and runner scripts

The manifests in this directory assume the current image split:
- `dynamo-frontend` for the frontend service
- `dynamo-planner` for the planner service
- `vllm-runtime` for the backend workers

## Not Restored

The old `tests/planner/unit/controller.py` helper was not brought back here because it
depended on `LocalConnector`, which is no longer part of the planner package.
