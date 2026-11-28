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

Two further options, [cloudAI-MPC and Cognitive Agent](#cloudai-mpc-and-cognitive-agent),
add a model predictive controller and an agent around an open decision model.

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
| Run independent configurations with checkpoints | [Reproducible suites](docs/usage.md#reproducible-suites-and-bounded-execution) | Gym; native runtime for simulation |
| Build six schedule-shaped workloads from your data | [Golden Set builder](docs/usage.md#build-a-golden-set-from-an-external-base-trace) | Gym and a source trace |
| Replay recorded requests | [Recorded trace guide](docs/usage.md#bring-your-own-recorded-trace) | User-supplied Mooncake or Dynamo request traces |
| Compare running deployments | [Endpoint benchmarking](docs/usage.md#user-guide-benchmark-live-endpoints) | AIPerf and existing endpoints |
| Add an autoscaler | [Extension guide](docs/usage.md#add-an-offline-autoscaler) | Dynamo's engine interface |
| Use the hosted Jev decision engine | [Jev guide](docs/jev.md) | Simulation environment and a TypeSafe API key |
| Add the cloudAI-MPC controller | [cloudAI-MPC guide](docs/mpc.md) | Simulation environment with completion feedback telemetry |
| Add the Cognitive Agent | [Cognitive Agent guide](docs/cognitive-agent.md) | The same, plus a locally served KnowLine-31B model |

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

The base dependencies are PyYAML, Plotly, and psutil. The `test` and `jev` extras add
pytest and the optional HTTP client; `sim` adds the Planner's Python dependencies.
Dynamo's native runtime and AIPerf are installed separately.

## cloudAI-MPC and Cognitive Agent

Two more offline autoscalers can join a simulated Match Config:

| Planner | Select with | Guide |
| --- | --- | --- |
| cloudAI-MPC | `type: mpc` | [cloudAI-MPC](docs/mpc.md) |
| Cognitive Agent | `type: jev` with `provider: self_hosted` | [Cognitive Agent](docs/cognitive-agent.md) |

**cloudAI-MPC** is a model predictive controller. A capacity model built from
AISimulate prefill and decode predictions, an arrival forecast and a
cold-start order book evaluate each candidate fleet. The controller picks the
fleet expected to serve the most SLO-satisfying requests per GPU-second. It
needs no training.

**Cognitive Agent** is a scaling agent around the open decision model
KnowLine-31B, served locally. Code builds the measured observations and the
feasible joint prefill/decode fleets, the model chooses among them, and every
answer is validated before it is applied.

Both read measured SLO outcomes from replay telemetry (`completion_feedback`).
The AISimulate release pinned by Dynamo does not emit that block yet, so an
end-to-end run needs a replay runtime that does. Each guide states the
requirement.

### Results on 60 traces

Both planners were compared with Dynamo Planner, KEDA, Reactive and a fixed
Static fleet on 60 full-length cases: 30 aggregated and 30 disaggregated. The
cases come from 11 trace sources (Azure 2024 Code and Conversation, BurstGPT,
RAGPulse, ART-Chat, Qwen-Bailian and Mooncake), with offered windows from
17 minutes to 7 days and about 82 million requests in total. Each topology has
a high-throughput group (50 ms ITL target) and a latency-focused group (15 ms
ITL target), in small and large deployments. The simulated deployment is
DeepSeek-V4-Pro on B300 with vLLM 0.24.0. Every planner starts from the
case's Static fleet and shares one GPU ceiling: that fleet plus three more of
the largest workers.

Planners are ranked within each trace by **goodput per average simulated GPU**
(higher is better), then ordered by **mean rank** across traces (lower is
better). Each trace has equal weight; ties share rank and win credit, so wins
can be fractional. Mean good rate is the average per-trace fraction of
requests meeting the SLOs; it is not the ranking metric.

#### Aggregated serving, 30 traces

| Position | Planner | Mean rank ↓ | Wins ↑ | Mean good rate | Simulated GPU-hours |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **Cognitive Agent** | 2.467 | 5.00 | 58.26% | 20,011.3 |
| 2 | **cloudAI-MPC** | 2.700 | 7.50 | 57.43% | 18,821.9 |
| 3 | Static | 2.733 | 10.50 | 63.93% | 27,718.9 |
| 4 | Dynamo Planner | 3.733 | 5.00 | 81.57% | 30,488.8 |
| 5 | KEDA | 4.500 | 0.00 | 28.46% | 13,084.4 |
| 6 | Reactive | 4.867 | 2.00 | 21.26% | 10,952.0 |

#### Disaggregated serving, 30 traces

| Position | Planner | Mean rank ↓ | Wins ↑ | Mean good rate | Simulated GPU-hours |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **cloudAI-MPC** | 2.517 | 12.00 | 76.93% | 18,787.5 |
| 2 | Dynamo Planner | 3.017 | 6.50 | 62.30% | 15,490.7 |
| 3 | **Cognitive Agent** | 3.333 | 2.00 | 79.78% | 23,525.8 |
| 4 | Static | 3.433 | 5.83 | 65.53% | 23,897.4 |
| 5 | Reactive | 3.967 | 2.33 | 61.60% | 19,162.5 |
| 6 | KEDA | 4.733 | 1.33 | 64.09% | 21,337.6 |

cloudAI-MPC ranks first in disaggregated serving and second in aggregated.
Cognitive Agent ranks first in aggregated serving and third in disaggregated.
In aggregated serving the top three are close, and the fixed Static fleet
earns the most win credit.

Each cell is one CPU-simulated full-trace run, not a GPU-serving measurement.
Simulated GPU-hours exclude the physical GPUs that serve Cognitive Agent's
decision model. The runs used a replay runtime that emits `completion_feedback`
and holds every planner to the GPU ceiling while workers start and drain. The
60 cases, that harness and the full per-trace leaderboard are not part of this
package yet.

## Metrics

A request meets an SLO only when it satisfies every configured threshold.
Dropped or failed requests count against the good-request rate.
For simulation, `metrics.short_output_itl` explicitly selects whether one-token
outputs fail an ITL constraint (`fail`, the default) or skip that check (`skip`,
matching native replay). Incompatible summary-only scoring fails explicitly.
Scorecard schema 3 records the policy, measurement interval, and GPU cost basis.

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
An optional arrival measurement window includes eventual outcomes during drain
while integrating GPU allocation over the selected interval. Full-replay latency
percentiles remain context metrics; they are not relabeled as window percentiles.

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
`test_adapter_lifecycle.py`, `test_adapter_observations.py`, `test_jev.py`,
`test_mpc.py`, `test_knowline_efficiency.py`, or `test_outcome_feedback.py`;
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
- JSONL request capture avoids a Python request list, but the native runtime
  still buffers terminal rows before writing them. Summary capture uses less
  memory; explicit windows require memory or JSONL capture. RSS limits apply
  to the isolated worker and its descendants.
- Hosted Jev call latency is measured separately and does not advance simulated
  time. See its guide before interpreting results.

For setup, data, and endpoint failures, see
[troubleshooting](docs/usage.md#troubleshooting).
