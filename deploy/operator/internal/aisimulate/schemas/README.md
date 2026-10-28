<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate input schemas

Each JSON file is generated from the exact pinned wheel's
`aisimulate.config.CoreRecommendationConfig.model_json_schema(mode="validation")`.
The operator embeds the schema selected by the generated `SchemaVersion` constant.
Keep historical snapshots when updating the pin; Requests cannot select an older version.

The generator preserves JSON Schema 2020-12, including local references and unions.
It extends only the root envelope with object-valued sections registered by Dynamo's
config adapters and the selected `dgdr_run` output adapter, matching the CLI's
section extraction before core validation. These adapters own their detailed rules.
Python `model_validator` and `field_validator` rules absent from the JSON Schema
remain runtime checks; do not recreate them in Go.

After updating both AISimulate Python pins, install the generator dependencies and
regenerate from the repository root:

```bash
python3 -m pip install pydantic==2.13.5 PyYAML
python3 -m pip install --no-deps -r container/deps/requirements.aisimulate.txt
make -C deploy/operator generate-aisimulate-schema
make -C deploy/operator check-aisimulate-schema
```

The check compares a fresh export with the committed schema and version registration.
Missing snapshots, changed upstream schemas, inconsistent pins, and an installed
version different from the pin fail the check. Operator CI runs it for pin-only
changes too. `make check` verifies rather than regenerating these external artifacts,
so dependency bumpers must review and commit the schema change explicitly.
