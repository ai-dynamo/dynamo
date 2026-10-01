# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm.v1.core.kv_cache_manager")

from vllm import envs  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.engine.core import EngineCore  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

from dynamo.vllm import kv_cache_metadata_compat as compat  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def missing_getter(monkeypatch):
    name = "get_kv_cache_group_metadata"
    monkeypatch.setattr(
        EngineCore, name, getattr(EngineCore, name, None), raising=False
    )
    monkeypatch.delattr(EngineCore, name)


@pytest.mark.parametrize(
    "is_active, installed_version, has_native, has_getter",
    [
        (False, "0.30.0", False, False),
        (True, "0.29.0", False, False),
        (True, "0.30.0rc1", False, False),
        (True, "0.30.1", False, False),
        (True, "0.30.0+cu129", True, True),
        (True, "0.30.0+cu129", False, True),
    ],
)
def test_register_guards(
    monkeypatch, missing_getter, is_active, installed_version, has_native, has_getter
):
    monkeypatch.setenv(compat._ACTIVATION_ENV, "1" if is_active else "0")
    monkeypatch.setattr(compat, "version", lambda _: installed_version)
    native_getter = Mock()
    if has_native:
        monkeypatch.setattr(
            EngineCore, "get_kv_cache_group_metadata", native_getter, raising=False
        )

    compat.register()
    assert hasattr(EngineCore, "get_kv_cache_group_metadata") is has_getter
    if has_native:
        assert EngineCore.get_kv_cache_group_metadata is native_getter


@pytest.mark.parametrize(
    "allowlist, ray_vars",
    [(None, None), ("", ""), ("modelexpress,omni", "CUSTOM_CACHE,CUSTOM_MODEL")],
)
def test_enable_preserves_plugin_selection(monkeypatch, allowlist, ray_vars):
    monkeypatch.setenv(compat._ACTIVATION_ENV, "0")
    monkeypatch.setenv(compat._RAY_COPY_ENV, ray_vars or "")
    if ray_vars is None:
        monkeypatch.delenv(compat._RAY_COPY_ENV)
    if allowlist is None:
        monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    else:
        monkeypatch.setenv("VLLM_PLUGINS", allowlist)
    register = Mock()
    monkeypatch.setattr(compat, "register", register)

    compat.enable_kv_cache_metadata_compat()
    compat.enable_kv_cache_metadata_compat()

    assert os.environ[compat._ACTIVATION_ENV] == "1"
    assert register.call_count == 2
    assert os.environ[compat._RAY_COPY_ENV].split(",") == [
        *(ray_vars.split(",") if ray_vars else []),
        compat._ACTIVATION_ENV,
    ]
    envs.validate_environ(hard_fail=True)
    if allowlist is None:
        assert "VLLM_PLUGINS" not in os.environ
    else:
        assert os.environ["VLLM_PLUGINS"].split(",") == [
            *allowlist.split(","),
            compat._PLUGIN_NAME,
        ]


@pytest.mark.parametrize("is_hybrid, dcp", [(False, 2), (True, 1)])
def test_metadata_uses_initialized_cache_managers(
    monkeypatch, missing_getter, is_hybrid, dcp
):
    monkeypatch.setenv(compat._ACTIVATION_ENV, "1")
    monkeypatch.setattr(compat, "version", lambda _: "0.30.0")
    compat.register()
    specs = [
        FullAttentionSpec(
            block_size=32 if is_hybrid else 16,
            num_kv_heads=1,
            head_size=32 if is_hybrid else 64,
            dtype=torch.float16,
        )
    ]
    if is_hybrid:
        specs.insert(
            0,
            SlidingWindowSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.float16,
                sliding_window=64,
            ),
        )
    cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[f"layer.{i}"], kv_cache_spec=spec)
            for i, spec in enumerate(specs)
        ],
    )
    manager = KVCacheManager(
        cache_config,
        max_model_len=128,
        scheduler_block_size=32,
        hash_block_size=16 if is_hybrid else 32,
        dcp_world_size=dcp,
    )
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = SimpleNamespace(
        kv_cache_config=cache_config, kv_cache_manager=manager
    )

    expected = [
        {
            "group_idx": 0,
            "kind": "full_attention",
            "block_size": 32,
            "sliding_window": None,
        }
    ]
    if is_hybrid:
        expected = [
            {
                "group_idx": 0,
                "kind": "sliding_window",
                "block_size": 16,
                "sliding_window": 64,
            },
            {
                "group_idx": 1,
                "kind": "full_attention",
                "block_size": 32,
                "sliding_window": None,
            },
        ]
    assert engine.get_kv_cache_group_metadata() == expected
    engine.scheduler.kv_cache_config = None
    assert engine.get_kv_cache_group_metadata() == []
