<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Planner Gym

**Experimental.** Compare LLM inference autoscaling policies with NVIDIA Dynamo's
CPU replay simulator, or benchmark existing OpenAI-compatible deployments with
[AIPerf](https://github.com/ai-dynamo/aiperf). The Python package is named
`autoscaling_arena`.

Offline matches run Dynamo Planner, a Python KEDA/HPA policy port, reactive
scaling, or fixed capacity against the same workload, model, engine, hardware
model, and service-level objectives (SLOs). The leaderboard ranks SLO-qualified
requests per second per average allocated GPU. It also reports latency, the
fraction of requests meeting the SLO, GPU-hours, and scaling activity.

The KEDA entry models controller behavior; it does not deploy Kubernetes or a
live KEDA controller. Simulation estimates performance and does not replace
validation on a serving deployment.

## Start Here

Use the [getting started guide](docs/getting-started.md) to run Planner and KEDA
against 100 requests from Dynamo's bundled public Mooncake trace. No separate
trace download or model weights are required.

From `dynamo/gyms/planner-gym` on Linux with Python 3.12, Rust, and Dynamo's
[source-build prerequisites](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/advanced-customizations/building-from-source.md):

```bash
bash scripts/setup_simulation_env.sh
source .venv/bin/activate
python scripts/run_match_config.py configs/match.quickstart.yaml --validate-only --print-matrix
python scripts/run_match_config.py configs/match.quickstart.yaml
```

Open `runs/quickstart/report.html` after the two runs succeed. Full metrics and
provenance are saved in `runs/quickstart/results.json`; replay diagnostics and
telemetry are under `runs/quickstart/artifacts/`.

> [!NOTE]
> The simulation setup requires Linux because Dynamo's pinned AISimulate wheel
> is Linux-only. Trace generation, config validation, scoring, saved reports,
> and endpoint benchmarking also work on macOS. Python 3.11–3.12 is supported
> for the standalone package; use 3.11 or 3.12 for simulation.

## Choose a Workflow

| Goal | Entry point | Requirements |
| --- | --- | --- |
| First offline comparison | [Getting started](docs/getting-started.md) | Linux and a native Dynamo build |
| Generate seven synthetic workloads | `python scripts/gen_traces.py --seed 0` | Python standard library |
| Configure a comparison matrix | [Match Config guide](docs/usage.md#user-guide-run-a-match-config) | Gym; Dynamo for sim, AIPerf for real |
| Build six schedule-shaped workloads from your data | [Golden Set builder](docs/usage.md#build-a-golden-set-from-an-external-base-trace) | Gym and a source trace |
| Replay recorded requests | [Recorded trace guide](docs/usage.md#bring-your-own-recorded-trace) | User-supplied Mooncake or Dynamo request traces |
| Compare running deployments | [Endpoint benchmarking](docs/usage.md#user-guide-benchmark-live-endpoints) | AIPerf and existing endpoints |
| Add an autoscaler | [Extension guide](docs/usage.md#add-an-offline-autoscaler) | Dynamo's engine interface |
| Use the hosted Jev decision engine | [Jev guide](docs/jev.md) | Simulation environment and a TypeSafe API key |

All commands run from `gyms/planner-gym`. Relative paths inside a Match Config
resolve from the YAML file's directory. The scripts are source-checkout tools;
the installed wheel provides the Python package.

For tools that do not need simulation, create a separate environment:

```bash
python3.12 -m venv .venv-tools
source .venv-tools/bin/activate
python -m pip install -e '.[test,jev]'
python scripts/gen_traces.py --seed 0
python scripts/run_match_config.py configs/match.quickstart.yaml --validate-only
```

The base dependencies are PyYAML and Plotly. The `test` and `jev` extras add
pytest and the optional HTTP client; `sim` adds the Planner's Python dependencies.
Dynamo's native runtime and AIPerf are installed separately.

## Metrics

A request meets an SLO only when it satisfies every configured threshold.
Dropped or failed requests count against the good-request rate.

| Metric | Definition |
| --- | --- |
| `goodput_rps` | SLO-qualified requests / benchmark seconds |
| `good_rate` | SLO-qualified requests / attempted requests |
| `gpu_hours` | Provisioned GPU allocation integrated over replay time by Dynamo |
| `average_gpus` | `gpu_hours / benchmark_hours` |
| `goodput_per_gpu` | `goodput_rps / average_gpus` |
| `oscillation_count` | Scaling direction reversals across worker pools |

Read efficiency together with good-request rate and latency. A policy can rank
well on efficiency while missing many SLOs. Simulated allocation includes
starting and draining workers, weighted by GPUs per worker. Live endpoint
comparisons lack deployment telemetry and therefore cannot rank by GPU cost.

| Built-in SLO | Time to First Token (TTFT) | Inter-token Latency (ITL) | End-to-end latency |
| --- | --- | --- | --- |
| `interactive` | 300 ms | 50 ms | Unconstrained |
| `agentic` | Unconstrained | 200 ms | 3,000 ms |
| `relaxed` | 2,000 ms | 50 ms | Unconstrained |

The [usage guide](docs/usage.md) describes the eight-workload catalog, matrix
expansion, report timelines, custom SLOs, and trace formats. The
[replay integration guide](docs/replay-integration.md) describes the boundary
with Dynamo and the telemetry contract.

## Development

```bash
python -m pip install -e '.[test,jev]'
python -m pytest -q
```

The standalone CI job runs on Python 3.11 and 3.12 and exercises config
validation, trace processing, scoring, report rendering, and mocked backend
contracts. It does not build Dynamo or run the native adapter tests in
`test_adapter_lifecycle.py`, `test_adapter_observations.py`, or `test_jev.py`;
these modules are skipped when Dynamo imports are unavailable. After the
[full setup](#start-here), run the same test command to include them, then run
the quickstart to verify actual native replay.

## Limitations

- The online runner targets existing deployments; it does not provision servers
  or collect their GPU cost or controller telemetry.
- Synthetic traces are open-loop arrival schedules. They do not model tool
  delays or completion-dependent conversation turns.
- Golden Set construction loads its source scope into memory. Bound it with
  `--max-source-requests` and `--max-source-hashes`, or narrow the input first.
- The legacy leaderboard scripts overwrite fixed diagnostic filenames. Use
  Match Configs for isolated session and run artifacts.
- Hosted Jev call latency is measured separately and does not advance simulated
  time. See its guide before interpreting results.

For setup, data, and endpoint failures, see
[troubleshooting](docs/usage.md#troubleshooting).
