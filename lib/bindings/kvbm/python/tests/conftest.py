# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for testing the optional SGLang integration in isolation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "kvbm" / "sglang_integration"


@pytest.fixture
def install_module(monkeypatch):
    """Install an isolated module hierarchy for optional runtime dependencies."""

    def install(name: str, **attributes) -> ModuleType:
        parts = name.split(".")
        for index in range(1, len(parts)):
            package_name = ".".join(parts[:index])
            if package_name not in sys.modules:
                package = ModuleType(package_name)
                package.__path__ = []
                monkeypatch.setitem(sys.modules, package_name, package)
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    return install


@pytest.fixture
def load_source(monkeypatch):
    """Load one integration source file without importing package ``__init__``."""

    def load(name: str, filename: str):
        spec = importlib.util.spec_from_file_location(name, SOURCE_ROOT / filename)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load integration source {filename!r}.")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    return load
