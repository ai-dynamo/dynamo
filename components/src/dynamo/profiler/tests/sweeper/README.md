<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sweeper DGD comparison cases

This directory contains inputs for learning how the existing DGDR v1beta1 profiler and the new
AI Simulate Sweeper render the same deployment intent. It is not a golden test suite: the runner
does not compare manifests or decide which output is correct.

Each case is a directory with conventional file names:

```text
<case>/
├── dgdr-v1beta1.yaml  # profiler input: the DGDR v1beta1 spec, without the resource wrapper
├── sweeper.yaml       # native AI Simulate SmartSearchConfig
└── recipe-dgd.yaml    # optional hand-tuned reference copied from recipes/
```

Run one or more cases from an environment containing Dynamo, AI Simulate, and the Planner
dependencies:

```bash
python components/src/dynamo/profiler/tests/sweeper/run_cases.py qwen3-32b-vllm-disagg
```

With no case names, the runner executes every directory under `cases/`. One scalar sweep is
executed per case. Its best candidate is rendered through both renderers, so renderer differences
cannot be caused by different searches.

The runner writes ignored files under `<case>/generated/`:

```text
dgdr-v1beta1-dgd.yaml
sweeper-candidate.yaml
sweeper-aic-dgd.yaml
sweeper-direct-dgd.yaml
```

If a renderer cannot consume the candidate, the runner continues with the other renderer, writes
`sweeper-<renderer>-error.txt`, and exits non-zero after all renderers have been attempted. This is
comparison evidence too: it identifies candidate shapes that a renderer does not yet support.

Compare them manually, for example:

```bash
diff -u \
  components/src/dynamo/profiler/tests/sweeper/cases/qwen3-32b-vllm-disagg/generated/dgdr-v1beta1-dgd.yaml \
  components/src/dynamo/profiler/tests/sweeper/cases/qwen3-32b-vllm-disagg/generated/sweeper-aic-dgd.yaml
```

`dgdr-v1beta1.yaml` intentionally matches the spec payload consumed by `python -m
dynamo.profiler`; the Kubernetes `apiVersion`, `kind`, and metadata are not profiler inputs.

Sweeper runs replay simulations and may take minutes or longer as cases grow. Keep small learning
cases bounded with their native `sweep` settings rather than adding runner-specific sweep knobs.
