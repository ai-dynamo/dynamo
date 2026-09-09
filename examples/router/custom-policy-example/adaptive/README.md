<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adaptive Worker Selection

**Experimental.** This NVIDIA Dynamo plugin adjusts cache-affinity and load-distribution scorer weights as traffic changes. It implements AIMD and sigmoid controllers through the existing `WorkerPicker` and linked catalog API. No host changes are needed. AIMD is the initial candidate because sigmoid regressed on the decode-heavy simulation; neither is a production recommendation.

The [evaluation](EVALUATION.md) compares both controllers with fixed weights, least-load selection, discounted UCB, and bucketed contextual discounted UCB. The bandits are simulation experiments, not selectable plugin algorithms: Dynamo's worker-selection API currently provides neither an outcome callback nor a public request-ID getter for credit assignment.

## Data Flow

```mermaid
flowchart LR
    R[Request] --> H[Dynamo eligibility and hard constraints]
    H --> I[Eligible device cache overlap and active requests]
    I --> P[Adaptive picker]
    I --> C[Rate-limited controller inside picker]
    C --> W[Cache and load weight budget]
    W --> P
    P --> D[Dynamo dispatch and accounting]
```

The factory constructs separate state for each model, routing group, and worker role. Dynamo serializes picker calls in its scheduler actor. The picker returns a row from the host's eligible table; hard pins, allowlists, and required taints remain host constraints. Soft affinity and preferred taints do not receive additional credit in this policy; device cache overlap provides affinity. Host/disk cache and heterogeneous worker capacity are not modeled by the scorers.

There are no background detectors. A request samples the current eligible pool at most once per `update_interval_ms`. Between updates the picker uses the last budget with current cache/load scores. Quiet periods do not synthesize measurements or replay missed ticks. This adds bounded work on update requests; it does not preserve the llm-d document's zero-per-request-overhead goal. Different constrained candidate sets share their partition's controller, so frequent changes in allowlists can perturb its signal.

## Scorers and Control Laws

For candidate `i`, `A_i` is device overlap divided by request blocks, clamped to `[0, 1]`; malformed cache input gets zero credit. Requests without prefill tracking get zero affinity credit. Let `L_i` be active requests and `s = load_scale`:

```text
distribution_cost_i = (L_i - min(L)) / (s + max(L) - min(L))
cost_i = (1 - w) * (1 - A_i) + w * distribution_cost_i
```

The picker chooses minimum cost and samples uniformly among exact ties. Both logical scorers run inside the picker because normalization needs the whole eligible pool; the per-row `WorkerScorer` callback has no request-wide preparation hook. This keeps one consistent weight budget per request without shared mutable state between callbacks. The strictly positive lower bound on `w` makes cold requests select least load even when the controller favors cache. Fixed weights in the evaluation use this exact scoring function, isolating the effect of adaptation; they are not Dynamo's built-in cost function.

At a control update, with `N` eligible rows:

```text
imbalance = CV(L) / sqrt(N - 1) * max(L) / (max(L) + s)
pressure = min(L) / (min(L) + s)
cap = distribution_max - EWMA(pressure) * (distribution_max - pressure_max)
```

Imbalance is zero for a singleton or all-zero load. Normalized coefficient of variation (CV) is bounded by one for nonnegative loads. The absolute-load factor prevents a single request from causing maximum distribution. Pressure is an active-count proxy, not engine queue depth or GPU utilization. Using the minimum requires every eligible candidate to be busy before reducing the distribution ceiling.

- **Sigmoid:** map smoothed imbalance through a logistic curve into `[distribution_min, cap]`. Endpoint normalization restores the exact baseline for balanced load. This follows the affinity/distribution category budgets in the [llm-d implementation](https://github.com/nirrozenbaum/gateway-api-inference-extension/blob/6c0f8d8496b56d68c30369520f3c4477adf8cd42/pkg/epp/scheduling/adaptive_configurator.go), using signals already available in Dynamo instead of async detectors.
- **AIMD:** above `midpoint`, add `max_step` to the distribution weight. Below `midpoint / 2`, multiply the excess over `distribution_min` by `0.9`. Keep the weight unchanged inside the hysteresis band. Clamp the target to the pressure-adjusted ceiling.
- **Both:** smooth signals with an exponentially weighted moving average (EWMA) and bound each budget change by `max_step`. A falling pressure ceiling is approached at that same bounded rate. These are reactive controllers, not reward learners.

The policy allocates no candidate copies and stores no per-worker map. Per-selection scoring is `O(N)`; variance calculation runs only on control ticks. Changing or removing workers therefore needs no plugin cleanup.

## Configure and Build

Follow the parent [catalog build guide](../README.md#6-build-and-test) to link `dynamo-custom-policy-example-catalog`. Stock images do not include this example. Then set:

```bash
export DYN_ROUTER_POLICY_CONFIG="$PWD/examples/router/custom-policy-example/adaptive/worker-selection.yaml"
python3 -m dynamo.frontend --router-mode kv
```

The supplied YAML selects `adaptive-aimd` for aggregated and prefill pools. To compare sigmoid, change those selections to `adaptive-sigmoid`. Decode and encode retain their configured default policies. The same catalog can be linked into the [example EPP](../epp/Cargo.toml).

| Parameter | Default | Meaning |
|---|---|---|
| `algorithm` | `aimd` | `aimd` or `sigmoid`; unknown values fail startup |
| `update_interval_ms` | `100` | Positive minimum interval between observed control updates |
| `smoothing` | `0.2` | EWMA coefficient in `(0, 1]`, applied once per observed update |
| `max_step` | `0.1` | Maximum absolute weight change per update, in `(0, 1]` |
| `distribution_min` | `0.1` | Baseline load weight; must be positive |
| `distribution_max` | `0.9` | Maximum load weight; must be less than one |
| `pressure_max` | `0.5` | Distribution ceiling approached when all candidates are busy |
| `midpoint` | `0.35` | Sigmoid midpoint or AIMD upper threshold, in `(0, 1)` |
| `slope` | `8.0` | Sigmoid steepness, in `[0.1, 100]` |
| `load_scale` | `8.0` | Active-request scale for load normalization and pressure, at least one |
| `seed` | omitted | Optional reproducible tie-break seed; omit for independently seeded replicas |

All numeric values must be finite and `distribution_min <= pressure_max <= distribution_max`. Unknown parameter keys fail startup. Adaptation still has bounds, time constants, and a load scale to validate for a deployment; it does not remove tuning. To inspect observed budgets, enable `RUST_LOG=dynamo_custom_policy_example_adaptive=trace`; one structured event per control update includes partition, worker role, algorithm, weight, imbalance, and pressure.

## Validation

From the repository root:

```bash
cargo test -p dynamo-custom-policy-example-adaptive --all-targets
cargo test -p dynamo-custom-policy-example-catalog
cargo bench -p dynamo-custom-policy-example-adaptive --bench selection
cargo run --release -p dynamo-custom-policy-example-adaptive --example phase_shift > phase-shift.csv
python3 examples/router/custom-policy-example/adaptive/summarize.py phase-shift.csv
```

The tests exercise real Dynamo YAML resolution and host selection, including hard DP pins, allowlists, independent partition state, and decode behavior. Controller tests use an injected monotonic clock to verify cadence, rate limits, recovery, finite bounds, topology changes, malformed cache input, and tie fairness. The simulation shares the exact controller and scoring code with the plugin; it does not execute a GPU engine, scheduler batching, or KV transfer.
