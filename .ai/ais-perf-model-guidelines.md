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

- Use `ais` / `AIS` for Dynamo integration names. Actual AIConfigurator
  task-v2/interpolation APIs and upstream-owned wire protocols retain their
  upstream names at their boundaries.
- Each component takes one AIS perf model input: `--ais-perf-model VALUE` or
  `DYN_AIS_PERF_MODEL` on a command line, `ais_perf_model` in a configuration,
  `router_ais_perf_model=` / `ais_perf_model=` in the replay API, and
  `router.ais_perf_model` in a simulation config. The value is one perf model
  config or a list of configs with one `worker_type` each; on a command line it
  is a file path or inline JSON. Do not add a second input for the same data.
- Only `dynamo.common.ais_perf_model` calls the AISimulate loader. Read and
  resolve a value through its functions; select an entry with
  `select_router_entry` or with the document that `load_document` returns.
- Old inputs keep their behavior until `REMOVE_IN` of that module. A component
  collects the old inputs in use and calls `warn_old_inputs` once, as the last
  statement of the check that finds them, so that no error of that check follows
  the warning. An old input together with the new one is an error.
- AISimulate owns schema, defaults, validation, selection, tuning and prediction.
  Pass complete configurations, including ordered roots and nested estimator
  controls. Reject unknown fields.
- New configs default to `estimation_mode: auto`, `fallback_policy: deny`.
  Auto still searches `op_level`, `fpm_interpolation`, then `fpm_regression`.
  Explicit selections obey their fallback policy; do not silently relax it.
- Bind `worker_type` to the deployed worker role, not the current query phase.
  Preserve model, backend/version, parallelism, quantization and speculation
  identity across CLI, SDK, Rust, replay, diagnostics, and saved configuration.
- The Planner value of `ais_perf_model` is one full configuration, or a list
  with one per worker type (`prefill`, `decode`, `aggregated`); it has no flag
  and no environment variable. Omitted configuration explicitly creates
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
