# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for early SGLang cache-plugin registration."""

import pytest

# isort: split

import dynamo.sglang.cache_plugin as cache_plugin

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.kvbm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "argv",
    [
        ["--radix-cache-backend", "dynamo_kvbm"],
        ["--radix-cache-backend=dynamo_kvbm"],
        ["--radix-cache-backend", "unified", "--radix-cache-backend=dynamo_kvbm"],
    ],
)
def test_kvbm_cache_plugin_is_loaded_for_explicit_backend(monkeypatch, argv):
    imports = []
    monkeypatch.setattr(cache_plugin.importlib, "import_module", imports.append)

    assert cache_plugin.load_sglang_cache_plugin(argv) is True
    assert imports == ["kvbm.sglang_integration"]


def test_kvbm_cache_plugin_is_not_loaded_for_other_backends(monkeypatch):
    imports = []
    monkeypatch.setattr(cache_plugin.importlib, "import_module", imports.append)

    assert cache_plugin.load_sglang_cache_plugin([]) is False
    assert (
        cache_plugin.load_sglang_cache_plugin(["--radix-cache-backend", "unified"])
        is False
    )
    assert imports == []
