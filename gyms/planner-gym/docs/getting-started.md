<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run Your First Planner Gym Comparison

Run Dynamo Planner and the KEDA/HPA policy port against 100 requests from the
public Mooncake trace bundled with Dynamo, then inspect an HTML report. This
CPU simulation requires no GPU or model weights.

## 1. Prepare a Dynamo checkout

Use Linux with Git, Python 3.12, Rust through `rustup`, and the compiler and
system libraries listed in Dynamo's
[source-build guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/advanced-customizations/building-from-source.md).
Internet access is needed to install dependencies and resolve performance data.
The pinned AISimulate package currently provides Linux wheels only.

For a new checkout:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo/gyms/planner-gym
```

If Dynamo is already checked out, change to its `gyms/planner-gym` directory.
Keep Gym, Dynamo's Python package, and the native runtime on the same revision.

## 2. Build the simulation environment

```bash
bash scripts/setup_simulation_env.sh
source .venv/bin/activate
```

The script locates the containing Dynamo checkout, creates `.venv`, builds its
native runtime, installs the Python packages, and checks the replay telemetry
API. The first native build can take several minutes. An existing `.venv` is
preserved: move it aside before requesting a fresh setup.

## 3. Validate the two-run matrix

```bash
python scripts/run_match_config.py configs/match.quickstart.yaml   --validate-only --print-matrix
```

The output reports `2 sim runs selected`: Planner and KEDA, each starting with
one prefill and one decode worker, one workload, and one relaxed SLO. The
example uses the same gpt-oss/H200/vLLM performance model for both policies.
Validation checks configuration and source paths; running the match also
checks native imports and performance-data availability.

## 4. Run the match

```bash
python scripts/run_match_config.py configs/match.quickstart.yaml
```

Expect two successful rows and output paths beneath `runs/quickstart/`:

```text
runs/quickstart/
├── artifacts/<session>/runs/<run-id>/
│   ├── telemetry.jsonl
│   └── trace-report.json
├── results.json
└── report.html
```

A failed row is not a benchmark result. Read its error in the terminal and
`results.json` before comparing policies.

## 5. Inspect the report

Open `runs/quickstart/report.html` in a browser. Compare goodput per GPU together
with the good-request rate, latency, allocation, and scaling timelines. The
short example verifies the workflow; it is not a representative policy study.

The HTML embeds its data and plotting code. Rerender it without another replay:

```bash
python scripts/render_saved_report.py runs/quickstart/results.json   --out runs/quickstart/report-rerendered.html
```

To rerun the match and replace the top-level JSON and HTML, pass `--overwrite`.
Each replay still gets a new artifact session directory.

## Next Steps

- Expand workloads and SLOs with the
  [Match Config guide](usage.md#user-guide-run-a-match-config).
- Build a [Golden Set](usage.md#build-a-golden-set-from-an-external-base-trace)
  from your own trace, then copy
  [the complete Golden Set config](../configs/match.golden-set.quickstart.yaml)
  beside `step-and-recovery.jsonl` and match `block_size` to the manifest.
- Use the [endpoint guide](usage.md#user-guide-benchmark-live-endpoints) to
  compare live deployments.

## Troubleshooting

For `ModuleNotFoundError` or missing replay parameters, rebuild with
`scripts/setup_simulation_env.sh` from the same checkout. An editable Python
install does not rebuild Rust.

For missing Mooncake data, verify
`lib/bench/testdata/mooncake_trace_1000.jsonl` exists in the containing Dynamo
checkout. `DYNAMO_DIR` can explicitly select another checkout.

For unsupported model/system/backend errors, check the installed AISimulate
performance data and the model settings in the config. Config validation alone
does not resolve that data.
