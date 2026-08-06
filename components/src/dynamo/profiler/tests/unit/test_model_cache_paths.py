# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.profiler.utils.model_cache_paths import (
    model_cache_path_in_pvc,
    normalize_model_cache_path,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.parametrize(
    "mount,model_path,expected",
    [
        (
            "/opt/models",
            "hub/models--Qwen--Qwen3-32B",
            "/opt/models/hub/models--Qwen--Qwen3-32B",
        ),
        ("/opt/model-cache", "/model/qwen", "/opt/model-cache/model/qwen"),
        ("/opt/models", "/opt/models", "/opt/models"),
        (
            "/opt/models",
            "/opt/models/hub/models--Qwen--Qwen3-32B/snapshots/abc",
            "/opt/models/hub/models--Qwen--Qwen3-32B/snapshots/abc",
        ),
        ("/opt/models", "/opt/models/checkpoint", "/opt/models/checkpoint"),
        (
            "/opt/models",
            "opt/models/checkpoint",
            "/opt/models/opt/models/checkpoint",
        ),
    ],
)
def test_normalize_model_cache_path(mount, model_path, expected):
    assert normalize_model_cache_path(mount, model_path) == expected


@pytest.mark.parametrize(
    "mount,model_path,expected",
    [
        ("/opt/models", "hub/models--Qwen--Qwen3-32B", "hub/models--Qwen--Qwen3-32B"),
        ("/opt/model-cache", "/model/qwen", "/model/qwen"),
        ("/opt/models", "/opt/models", "."),
        (
            "/opt/models",
            "/opt/models/hub/models--Qwen--Qwen3-32B/snapshots/abc",
            "hub/models--Qwen--Qwen3-32B/snapshots/abc",
        ),
        ("/opt/models", "/opt/models/checkpoint", "checkpoint"),
        ("/opt/models", "opt/models/checkpoint", "opt/models/checkpoint"),
        ("/opt/models", None, None),
        ("/opt/models", "", None),
    ],
)
def test_model_cache_path_in_pvc(mount, model_path, expected):
    assert model_cache_path_in_pvc(mount, model_path) == expected
