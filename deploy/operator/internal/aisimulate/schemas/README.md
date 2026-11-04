<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate input schemas

Each JSON file is generated from the release's exact pinned AISim source:
`aisimulate.config.CoreRecommendationConfig.model_json_schema(mode="validation")`.
The operator embeds only `latest.json`, which the generator refreshes together with
the versioned snapshot and the `SchemaVersion` constant. Webhook setup compiles it
before registering handlers, so an invalid schema fails startup rather than admission.
Keep historical snapshots when updating the pin; they are not embedded, and Requests
cannot select an older version.

The generator preserves JSON Schema 2020-12, including local references and unions.
It extends only the root envelope with object-valued sections registered by Dynamo's
config adapters and the selected `dgdr_run` output adapter, matching the CLI's
section extraction before core validation. These adapters own their detailed rules.
Python `model_validator` and `field_validator` rules absent from the JSON Schema
remain runtime checks; do not recreate them in Go.

The Python distribution is `aisim`; its import namespace remains `aisimulate`.
Release candidates can share the same Python version, so the source SHA in
`Cargo.toml` identifies the exported schema. The generator checks that the workspace
and bindings source pins agree, verifies the checkout and Python version, and records
the SHA in the snapshot filename and schema comment.

After updating the pins, prepare a clean source checkout and regenerate from the
repository root. Schema extraction imports only Python models and needs no Rust build:

```bash
python3 -m pip install pydantic==2.13.5 PyYAML
export AISIMULATE_SOURCE_DIR="$(mktemp -d)"
git -C "$AISIMULATE_SOURCE_DIR" init
git -C "$AISIMULATE_SOURCE_DIR" fetch --depth=1 https://github.com/ai-dynamo/aisimulate.git \
  "$(python3 -c 'import tomllib; from pathlib import Path; print(tomllib.loads(Path("Cargo.toml").read_text())["patch"]["crates-io"]["aisim-core"]["rev"])')"
git -C "$AISIMULATE_SOURCE_DIR" checkout --detach FETCH_HEAD
make -C deploy/operator generate-aisimulate-schema
make -C deploy/operator check-aisimulate-schema
```

The check compares a fresh export with `latest.json`, the versioned snapshot, and the
version registration. Missing snapshots, changed upstream schemas, inconsistent pins,
and a source checkout different from the pin fail the check. Operator CI runs it for
Python-version and source-pin changes too. `make check` verifies rather than regenerating these external artifacts,
so dependency bumpers must review and commit the schema change explicitly.
