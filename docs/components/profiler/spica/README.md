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

Spica is the Replay-backed **smart sweeper** for Dynamo deployments (Profiler V2).
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

Spica's source lives in `components/src/dynamo/profiler/spica` and ships as part of Dynamo. Install
the `spica` optional dependency group to use its CPU Vizier and JAX stack. See the
[Spica examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/profiler/spica) for runnable
configuration files and tools. Spica uses AI Configurator's lower-layer forward-pass and memory
provider, then evaluates candidates with Dynamo Replay.

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

The Planner/Profiler image currently keeps AI Configurator 0.9 for compatibility with the existing
Profiler. That release does not provide `aiconfigurator.sdk.memory`:

- Spica emits a warning and skips the pre-search KV-capacity shape filter when the memory estimator
  is unavailable. Trace workloads and synthetic workloads with fixed `concurrency` remain usable;
  Replay and the GPU-budget checks still evaluate their candidates.
- `kv_load_ratio` needs the compatible AI Configurator memory estimator to convert a relative load
  into candidate-specific concurrency. Spica fails closed for this workload mode in the current
  default Planner/Profiler image instead of evaluating an unverified load.

Treat workloads and search modes not covered by image smoke tests as unsupported experimental paths.
See [Traffic](traffic.md#kv-load-ratio-candidate-relative-concurrency) for the KV-load contract.

## Develop

```bash
uv sync --extra dev --extra spica
uv run --extra dev --extra spica pytest components/src/dynamo/profiler/spica/tests
```

The root `spica` extra installs the CPU Vizier and JAX stack used by `run_smart_search`.

The `spica` extra installs **CPU** JAX. It resolves on every platform, but the Vizier
multi-objective GP suggest is slow on CPU (and can stall on larger sweeps). On a
**Linux x86-64 host with an NVIDIA GPU**, add the matching CUDA plugin to run the
optimizer on CUDA (XLA), which removes that bottleneck:

```bash
uv pip install --python .venv/bin/python "jax[cuda12]==0.4.38"
```

The JAX CUDA wheels exist **only for Linux x86-64**. macOS, Windows, and Arm64 use the CPU-only
`spica` extra. With no GPU present,
JAX just warns and falls back to CPU, so there's no reason to install them without
one. Spica detects the installed CUDA plugin and leaves JAX's
platform selection enabled; an explicit `JAX_PLATFORMS=cpu` or `JAX_PLATFORMS=cuda`
still overrides that behavior.

### Real Replay

The replay-backed evaluator (`dynamo.profiler.spica.evaluator.ReplayEvaluator`) drives the Dynamo
mocker's AI Configurator performance model. Dynamo's Planner/Profiler image includes the required
`aic-forward-pass` Cargo feature. For a local source checkout, rebuild the bindings with that
feature:

```bash
cd lib/bindings/python
maturin develop --uv --release --features aic-forward-pass,mocker-kvbm-offload
```

`RustEnginePerfModel` is importable from `dynamo._core` only when the feature is compiled in. The
real-replay integration tests skip when it is absent.

After replacing the example's placeholder `workload.trace_path`, run the search:

```bash
python -m dynamo.profiler.spica --config examples/profiler/spica/configs/smart_sweep.yaml
```
