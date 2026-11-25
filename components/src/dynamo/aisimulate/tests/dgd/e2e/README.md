<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate CLI golden tests

This directory contains one automatic CPU suite for AISimulate's public CLI.
Feature, model/backend/hardware, and expected-error scenarios use the same case
format and the same parametrized test entrypoint.

Each directory under `cases/` owns:

- `case.yaml`: the command, expected outcome, declared output files, and optional
  immutable model-metadata mirrors.
- `config.yaml`: a complete native AISim configuration that can be passed to the
  CLI without translation.
- `expected/`: every reviewed public artifact for a successful case.

The runner stages a case in a private temporary workspace, substitutes only
`{python}` and `{output}`, executes the command without a shell, and compares
declared YAML and JSON files structurally. It does not parse AISim
configurations, Candidates, backends, hardware, or DGD roles.

Model metadata mirrors are downloaded from immutable revision URLs into the
gitignored `.cache/aisimulate-golden-tests/` directory and copied into each
staged case. Authored fixtures, native configurations, and expected outputs
remain committed.

When a golden is missing or differs, the test fails with a unified diff and
writes the actual output beside the expected file as `<stem>.new.<suffix>`.
Reviewed goldens are never overwritten. Replace a golden only after inspecting
its corresponding proposal.

Run the full table from the repository root:

```bash
python3 -m pytest \
  components/src/dynamo/aisimulate/tests/dgd/e2e/test_cases.py -v
```

Run one group or case with pytest's normal selectors:

```bash
python3 -m pytest \
  components/src/dynamo/aisimulate/tests/dgd/e2e/test_cases.py \
  -k 'features or deepseek-r1-vllm' -v
```
