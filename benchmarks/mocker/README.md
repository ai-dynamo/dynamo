<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mocker AIS performance-model benchmark

`bench_ais_concurrency.py` measures query throughput through the canonical Python
`AisSession` as the number of caller threads changes. It includes Python facade
and FPM serialization costs; it does not measure Dynamo's pure Rust Router/Mocker
callback directly.

Build Dynamo bindings with `--features ais-forward-pass` and install a matching
AISimulate wheel, then run:

```bash
python benchmarks/mocker/bench_ais_concurrency.py --osl 256 --calls 20000
```

The benchmark creates one model through `ForwardPassPerfModel.best_available`
and reports calls per second and scaling relative to one caller thread. Model,
system and backend settings are defined at the top of the script. The retired
Python op-walk and callback fallback paths are no longer benchmark variants.

## Historical measurements

The following retained measurements predate the canonical API and do not measure
its current implementation. The retired comparison scripts are available in Git
history.

Offline replay: Qwen3-32B / h200_sxm / vLLM 0.19.0, Apple M-series, release build.
Values are steady-state minimum replay seconds.

| workload (isl/osl/reqs/conc/workers) | rust | opwalk (pure-py) | compiled | BIG WIN | this-PR |
|--------------------------------------|-----:|-----------------:|---------:|--------:|--------:|
| decode_heavy_1w  1024/1024/256/64/1  | 0.076 | 0.089 | 0.082 | 1.17x | 1.07x |
| balanced_1w      2048/256/256/64/1   | 0.021 | 0.026 | 0.023 | 1.23x | 1.11x |
| decode_heavy_4w  1024/1024/512/128/4 | 0.239 | 0.289 | 0.257 | 1.21x | 1.07x |

Historical concurrency: the same model/system/backend, 12 physical cores,
output length 256. Values are calls per second.

| variant (osl=256), calls/s | 1 thr | 4 thr | 8 thr | 12 thr | 16 thr | 24 thr |
|----------------------------|------:|------:|------:|-------:|-------:|-------:|
| `engine_pyo3` (Rust core)        | 12.4k | 47.5k | 81.9k | **100k** | 97k | 99k |
| `compiled_shell` (Python → Rust) | 12.7k | 48.0k | 80.5k | 97.6k | 99.6k | 100k |
| `opwalk` (pre-#1200 pure Python) | 4.2k | 10.9k | 10.9k | 11.0k | 10.9k | 10.9k |
| **ratio (engine / opwalk)**      | 3.0x | 4.4x | 7.5x | **9.1x** | 8.9x | 9.1x |
