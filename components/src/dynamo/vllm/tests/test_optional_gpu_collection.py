# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure optional-framework collection boundary; never import serving libraries."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


@pytest.mark.parametrize(
    ("filename", "available", "expected", "checks"),
    [
        ("test_gpu_offloading.py", False, True, 1),
        ("test_vllm_args.py", False, True, 1),
        ("test_kv_hints.py", False, None, 0),
        ("test_gpu_offloading.py", True, None, 1),
    ],
)
def test_optional_framework_collection_boundary(
    monkeypatch, filename, available, expected, checks
):
    path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("pure_vllm_collection_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def can_import():
        calls.append("canonical_vllm_check")
        return available

    monkeypatch.setattr(module, "_can_import_vllm", can_import)
    monkeypatch.setattr(module, "_can_import_gpu_tiering", lambda: True)
    result = module.pytest_ignore_collect(path.with_name(filename), None)
    assert result is expected
    assert len(calls) == checks


@pytest.mark.parametrize(
    "missing",
    [
        "vllm",
        "vllm.v1",
        "vllm.v1.kv_offload",
        "vllm.v1.kv_offload.tiering",
        "vllm.v1.kv_offload.tiering.kvcr",
        "vllm.v1.kv_offload.tiering.kvcr.manager",
    ],
)
def test_exact_optional_upstream_tiering_absence(monkeypatch, missing):
    path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("pure_gpu_tiering_absence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def absent(name):
        assert name == "vllm.v1.kv_offload.tiering.kvcr.manager"
        raise ModuleNotFoundError(name=missing)

    monkeypatch.setattr(module.importlib, "import_module", absent)
    assert module._can_import_gpu_tiering() is False


def test_available_optional_upstream_tiering_is_collected(monkeypatch):
    path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("pure_gpu_tiering_available", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def available(name):
        calls.append(name)
        return object()

    monkeypatch.setattr(module.importlib, "import_module", available)
    assert module._can_import_gpu_tiering() is True
    assert calls == ["vllm.v1.kv_offload.tiering.kvcr.manager"]


def test_gpu_file_with_absent_optional_upstream_is_ignored(monkeypatch):
    path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("pure_gpu_optional_file", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_can_import_vllm", lambda: True)
    monkeypatch.setattr(module, "_can_import_gpu_tiering", lambda: False)
    assert (
        module.pytest_ignore_collect(path.with_name("test_gpu_offloading.py"), None)
        is True
    )


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError(name="kvcr"),
        ImportError("broken adapter ABI"),
        RuntimeError("bad API"),
    ],
)
def test_gpu_tiering_errors_are_not_optional_absence(monkeypatch, error):
    path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("pure_gpu_tiering_errors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def broken(_):
        raise error

    monkeypatch.setattr(module.importlib, "import_module", broken)
    with pytest.raises(type(error)):
        module._can_import_gpu_tiering()
