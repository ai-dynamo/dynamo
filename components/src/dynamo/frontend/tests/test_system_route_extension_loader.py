# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest

from dynamo.frontend.main import (
    _load_system_route_extension_factories,
    _split_system_route_extension_paths,
)


def test_split_system_route_extension_paths_accepts_commas_and_whitespace():
    assert _split_system_route_extension_paths("a:f, b:g\n c:h") == [
        "a:f",
        "b:g",
        "c:h",
    ]


def test_load_system_route_extension_factories_from_import_paths(monkeypatch):
    module = types.ModuleType("test_route_extension_module")

    def factory():
        return []

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)

    assert _load_system_route_extension_factories(f"{module.__name__}:factory") == [
        factory
    ]


def test_load_system_route_extension_factories_rejects_invalid_path():
    with pytest.raises(ValueError, match="module:factory"):
        _load_system_route_extension_factories("missing_separator")
