---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Spica
subtitle: Experimental replay-backed configuration search for Dynamo deployments
---

> [!WARNING]
> **Experimental.** Spica is intended for evaluation and feedback, not production capacity
> planning. Its Python API, configuration schema, search behavior, and output may change without a
> standard deprecation period. Spica does not guarantee service-level agreement (SLA) compliance,
> prediction accuracy, or globally optimal configurations.

Spica is a replay-backed **smart sweeper** for Dynamo deployments.
It searches engine / router / planner configuration with a black-box optimizer,
evaluates each candidate with Dynamo Replay, and returns a ranked candidate set
(or a Pareto front under a `pareto` goal).

## Documentation

- [Overview](overview.md) describes Spica and the end-to-end sweep flow
  (validate → filter policies → load-predictor sub-sweep → enumerate branches →
  per-branch Vizier study → merge by goal).
- [Optimization Goals](optimization-goal.md) defines the `OptimizationGoal` targets,
  the per-GPU metric, the SLA rule, and how **`pareto`** (multi-objective) works.
- [Traffic](traffic.md) describes the `Workload` load shapes (trace, request rate,
  fixed concurrency / KV load), candidate-relative `kv_load_ratio`, and request-count scaling.
- [Search Space](search-space.md) lists every knob (type, default, searched or pinned,
  choices), the composite presets, and how `parallel_configs` are derived.
- [Unrolled Samples](sample.md) explains the flat unrolled sample and the three ways to
  pin/override what it emits.

Spica's source lives in `aisimulate/spica` and is published by the standalone `aisimulate`
distribution in the Dynamo repository. Installing that distribution provides the canonical
`spica` Python package and its CPU Vizier and JAX dependencies. See the
[Spica examples](https://github.com/ai-dynamo/dynamo/tree/main/aisimulate/examples/spica) for runnable
configuration files and tools. Spica uses AI Configurator's lower-layer forward-pass and memory
provider, then evaluates candidates with Dynamo Replay.

## Spica and Replay Optimize

Use `dynamo.profiler.utils.replay_optimize`, documented in
[DynoSim Sweeps](../../../dynosim/sweeps.md), for a focused heuristic search over tensor-parallel
shapes, worker counts, router settings, SLA constraints, and a GPU budget. Use Spica for the broader
experimental search space, Planner policies, KVBM controls, black-box optimization, or Pareto
objectives.

## Status

- Input schema (`SmartSearchConfig`) is implemented. See [Search Space](search-space.md)
  for the full knob reference (what you can pin/search, composite-knob presets vs.
  raw-dict pins, and `parallel_configs`).
- Planner load-predictor independent grid sweep (`sweep_load_predictor`) reuses the Dynamo Planner
  predictors and the Planner's trace-to-window tool.
- `run_smart_search` implements the Vizier and Replay sweep: enumerate, sample, deploy, replay,
  score, and rank.

Spica source, documentation, and examples were migrated from
[AIConfigurator commit `111b093a2a516d6cb2eabac5ad601c95c14ebdbe`](https://github.com/ai-dynamo/aiconfigurator/commit/111b093a2a516d6cb2eabac5ad601c95c14ebdbe).

## Current Limitations

The `dynamo-planner` image currently keeps AI Configurator 0.9 for compatibility with the existing
Profiler. That release does not provide `aiconfigurator.sdk.memory`:

- Spica emits a warning and skips the pre-search KV-capacity shape filter when the memory estimator
  is unavailable. Trace workloads and synthetic workloads with fixed `concurrency` remain usable;
  Replay and the GPU-budget checks still evaluate their candidates.
- `kv_load_ratio` needs the compatible AI Configurator memory estimator to convert a relative load
  into candidate-specific concurrency. Spica fails fast before starting the search for this
  workload mode in the current default `dynamo-planner` image instead of evaluating an unverified
  load or returning an empty candidate set after the sweep.

Treat workloads and search modes not covered by image smoke tests as unsupported experimental paths.
See [Traffic](traffic.md#kv-load-ratio-candidate-relative-concurrency) for the KV-load contract.

## Develop

```bash
uv venv .venv
source .venv/bin/activate
uv pip install pip "maturin[patchelf]"
cd lib/bindings/python
maturin develop --uv --release --features aic-forward-pass,mocker-kvbm-offload
cd ../../..
uv pip install --no-deps -e .
uv pip install -e "./aisimulate[test]"
pytest aisimulate/tests/spica
```

These steps install the Dynamo runtime and AI Simulate from the same checkout. Do not combine a
source checkout of AI Simulate with an older released `ai-dynamo` runtime.

The `aisimulate` distribution installs **CPU** JAX. It resolves on every supported platform, but the Vizier
multi-objective GP suggest is slow on CPU (and can stall on larger sweeps). On a
**Linux x86-64 host with an NVIDIA GPU**, add the matching CUDA plugin to run the
optimizer on CUDA (XLA), which removes that bottleneck:

```bash
uv pip install --python .venv/bin/python "jax[cuda12]==0.4.38"
```

The JAX CUDA wheels exist **only for Linux x86-64**. macOS, Windows, and Arm64 use CPU-only
JAX. With no GPU present,
JAX just warns and falls back to CPU, so there's no reason to install them without
one. When a Spica sampler is constructed, it detects the installed CUDA plugin and
leaves JAX's platform selection enabled; an explicit `JAX_PLATFORMS=cpu` or
`JAX_PLATFORMS=cuda` still overrides that behavior. Importing
`spica` alone does not modify JAX, logging, or warning settings.

### Real Replay

The replay-backed evaluator (`spica.evaluator.ReplayEvaluator`) drives the Dynamo
mocker's AI Configurator performance model. Dynamo's `dynamo-planner` image includes the required
`aic-forward-pass` Cargo feature. The development setup above builds both required features. If
the bindings are already installed without them, rebuild from the repository root:

```bash
cd lib/bindings/python
maturin develop --uv --release --features aic-forward-pass,mocker-kvbm-offload
```

`RustEnginePerfModel` is importable from `dynamo._core` only when the feature is compiled in. The
real-replay integration tests skip when it is absent.

After replacing the example's placeholder `workload.trace_path`, run the search:

```bash
python -m spica --config aisimulate/examples/spica/configs/smart_sweep.yaml
```
