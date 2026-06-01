# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.common.utils.worker_id import make_fpm_worker_id

pytestmark = [
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.pre_merge,
]


_PLACEMENT_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "ZE_AFFINITY_MASK",
    "HABANA_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
)


def _clear_worker_id_env(monkeypatch):
    for key in _PLACEMENT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_make_fpm_worker_id_is_stable_for_same_config(monkeypatch):
    _clear_worker_id_env(monkeypatch)
    config = SimpleNamespace(namespace="dynamo", component="backend", model="model-a")

    assert make_fpm_worker_id(config) == make_fpm_worker_id(config)


def test_make_fpm_worker_id_changes_for_different_config(monkeypatch):
    _clear_worker_id_env(monkeypatch)

    first = make_fpm_worker_id(
        SimpleNamespace(namespace="dynamo", component="backend", model="model-a")
    )
    second = make_fpm_worker_id(
        SimpleNamespace(namespace="dynamo", component="backend", model="model-b")
    )

    assert first != second


def test_make_fpm_worker_id_changes_for_placement(monkeypatch):
    _clear_worker_id_env(monkeypatch)
    config = SimpleNamespace(namespace="dynamo", component="backend", model="model-a")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    first = make_fpm_worker_id(config)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    second = make_fpm_worker_id(config)

    assert first != second


def test_make_fpm_worker_id_ignores_transient_config_fields(monkeypatch):
    _clear_worker_id_env(monkeypatch)

    first = make_fpm_worker_id(
        SimpleNamespace(
            namespace="dynamo",
            component="backend",
            model="model-a",
            runtime_object=object(),
        )
    )
    second = make_fpm_worker_id(
        SimpleNamespace(
            namespace="dynamo",
            component="backend",
            model="model-a",
            runtime_object=object(),
        )
    )

    assert first == second


