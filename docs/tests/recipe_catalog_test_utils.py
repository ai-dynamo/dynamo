# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG = REPO_ROOT / "docs/fern/pages/recipes/_catalog"
VALIDATOR_PATH = CATALOG / "validate.py"


def load_catalog_validator(module_name: str) -> ModuleType:
    if not VALIDATOR_PATH.is_file():
        pytest.skip(
            "recipe catalog sources are not present in this runtime image",
            allow_module_level=True,
        )
    spec = importlib.util.spec_from_file_location(module_name, VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    return validator
