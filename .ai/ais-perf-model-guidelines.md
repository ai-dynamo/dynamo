<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate performance-model integration

All new performance-model features must enter through AISimulate's canonical
`ForwardPassPerfModelConfig` and `ForwardPassPerfModel::best_available(config)`.
Python uses `aisimulate_core.sdk.RustForwardPassPerfModel.best_available(config)`.
Do not add parallel constructors, revive removed `from_native` / `from_regression`
APIs, or construct a private native engine to bypass this interface.

- Use `ais` / `AIS` for Dynamo integration names. Accept old `--aic-*` spellings
  only in CLI parsers and immediately lower them to the canonical configuration.
  Do not restore retired SDK aliases, modules, environment variables or config
  formats. Actual AIConfigurator task-v2/interpolation APIs and upstream-owned
  wire protocols retain their upstream names at their boundaries.
- AISimulate owns schema, defaults, validation, selection, tuning and prediction.
  Pass complete configurations, including ordered roots and nested estimator
  controls. Reject unknown fields and conflicting old/new input names.
- New configs default to `estimation_mode: auto`, `fallback_policy: deny`.
  Auto still searches `op_level`, `fpm_interpolation`, then `fpm_regression`.
  Explicit selections obey their fallback policy; do not silently relax it.
- Bind `worker_type` to the deployed worker role, not the current query phase.
  Preserve model, backend/version, parallelism, quantization and speculation
  identity across CLI, SDK, Rust, replay, diagnostics, and saved configuration.
- Planner organizes full configurations under `ais_perf_model.roles` for
  `prefill`, `decode`, or `aggregated`. Omitted configuration explicitly creates
  cold regression with traceable model identity. Consumers without a training
  source must reject an unready regression model.
- Use upstream migration helpers for legacy controls: 16 total buckets means a
  `[4, 4]` grid, not `[16, 16]`. Preserve independent regression/correction controls.
- Include complete configuration and limits in model reuse decisions. Keep
  online training state local to the worker group/consumer. Rebuild and replay
  retained observations when identity or limits change.
- Upgrade Python wheel and Rust crate together from the same released source and
  rebuild Dynamo bindings. Validate the installed wheel and actual Rust callback;
  permissive test fakes cannot detect removed APIs or EngineSpec schema mismatches.
