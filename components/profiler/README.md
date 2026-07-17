<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiler

The profiler is distributed as the standalone `ai-dynamo-profiler` wheel. It
keeps the existing `dynamo.profiler` import package and runs in the existing
`dynamo-planner` image.

## Install Built Wheels

Install the matching Dynamo, profiler, and split AIC artifacts from one local
wheelhouse:

```bash
uv pip install \
  /path/to/wheelhouse/aiconfigurator_core-0.10.0-*.whl \
  /path/to/wheelhouse/aiconfigurator-0.10.0-*.whl \
  /path/to/wheelhouse/ai_dynamo-1.3.0-*.whl \
  /path/to/wheelhouse/ai_dynamo_profiler-1.3.0-*.whl
```

The planner image builds the two AIC wheels from one pinned source revision and
verifies that the upper and core distributions do not own overlapping files.

## Editable Development

Complete the repository's
[source build](../../docs/getting-started/building-from-source.md) first. Then
install the profiler and both AIC layers from a single AIC checkout:

```bash
AICONFIGURATOR_DIR=/path/to/aiconfigurator
git -C "$AICONFIGURATOR_DIR" lfs pull
uv pip install \
  "$AICONFIGURATOR_DIR/aic-core" \
  "$AICONFIGURATOR_DIR" \
  --editable components/profiler
```

## Test

From the Dynamo repository root:

```bash
PYTHONPATH=components/profiler/src:components/src:lib/bindings/python/src \
  pytest -m unit components/profiler/src/dynamo/profiler/tests/unit
```

## Documentation

- [Profiler Overview](../../docs/components/profiler/README.md)
- [Profiler Guide](../../docs/components/profiler/profiler-guide.md)
- [Profiler Examples](../../docs/components/profiler/profiler-examples.md)
