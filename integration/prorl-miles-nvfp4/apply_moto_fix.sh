#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

workspace=/polar/session/workspace
target=${workspace}/moto/dynamodb/models/dynamo_type.py

cd "${workspace}"
test -z "$(git status --porcelain)"

python - "${target}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
assert source.count("float(self.value)") == 3
assert source.count("float(other.value)") == 2
updated = source.replace(
    "float(self.value)", "decimal.Decimal(self.value)"
).replace(
    "float(other.value)", "decimal.Decimal(other.value)"
)
assert updated.count("float(self.value)") == 0
assert updated.count("float(other.value)") == 0
assert updated.count("decimal.Decimal(self.value)") == (
    source.count("decimal.Decimal(self.value)") + 3
)
assert updated.count("decimal.Decimal(other.value)") == (
    source.count("decimal.Decimal(other.value)") + 2
)
path.write_text(updated)
PY

python -m py_compile moto/dynamodb/models/dynamo_type.py
pytest -q -k test_update_item
git diff --check
test "$(git diff --name-only)" = moto/dynamodb/models/dynamo_type.py
test -n "$(git status --porcelain)"
printf '__M6_BASH_VALIDATION_PASS__\n'
