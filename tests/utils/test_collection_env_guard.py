# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils.collection_env_guard import (
    ALLOWED_COLLECTION_ENV_MUTATIONS,
    COLLECTION_ENV_GUARD_DISABLE_ENV,
    WATCHED_ENV_PREFIXES,
    collection_env_guard_disabled,
    diff_collection_env,
    format_collection_env_changes,
    snapshot_collection_env,
)

pytestmark = [
    pytest.mark.parallel,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.pre_merge,
]


def test_snapshot_collection_env_filters_to_watched_prefixes():
    env = {
        "DYNAMO_TEST_FRAMEWORK": "vllm",
        "DYN_SYSTEM_PORT": "9090",
        "SGLANG_LOGGING_LEVEL": "debug",
        "TRTLLM_LOG_LEVEL": "info",
        "VLLM_NO_USAGE_STATS": "1",
        "NCCL_DEBUG": "INFO",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": "/tmp/hf",
        "TORCH_LOGS": "dynamic",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PATH": "/usr/bin",
    }

    assert snapshot_collection_env(env) == {
        "DYN_SYSTEM_PORT": "9090",
        "SGLANG_LOGGING_LEVEL": "debug",
        "TRTLLM_LOG_LEVEL": "info",
        "VLLM_NO_USAGE_STATS": "1",
        "NCCL_DEBUG": "INFO",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": "/tmp/hf",
        "TORCH_LOGS": "dynamic",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def test_watched_prefixes_cover_backend_env_without_long_dynamo_prefix():
    assert "DYN_" in WATCHED_ENV_PREFIXES
    assert "SGLANG_" in WATCHED_ENV_PREFIXES
    assert "TRTLLM_" in WATCHED_ENV_PREFIXES
    assert "DYNAMO_" not in WATCHED_ENV_PREFIXES


def test_diff_collection_env_reports_added_changed_and_removed_values():
    before = {
        "DYN_SKIP_PYTHON_LOG_INIT": "1",
        "SGLANG_LOGGING_CONFIG_PATH": "/tmp/old.json",
        "VLLM_NO_USAGE_STATS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    after = {
        "SGLANG_LOGGING_CONFIG_PATH": "/tmp/new.json",
        "VLLM_NO_USAGE_STATS": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "NCCL_DEBUG": "INFO",
    }

    assert diff_collection_env(before, after) == {
        "DYN_SKIP_PYTHON_LOG_INIT": ("1", None),
        "NCCL_DEBUG": (None, "INFO"),
        "VLLM_NO_USAGE_STATS": ("1", "0"),
    }


def test_diff_collection_env_ignores_narrow_logging_allowlist():
    assert "SGLANG_LOGGING_CONFIG_PATH" in ALLOWED_COLLECTION_ENV_MUTATIONS
    assert "VLLM_CONFIGURE_LOGGING" in ALLOWED_COLLECTION_ENV_MUTATIONS
    assert (
        diff_collection_env(
            {},
            {
                "SGLANG_LOGGING_CONFIG_PATH": "/tmp/sglang.json",
                "VLLM_CONFIGURE_LOGGING": "1",
            },
        )
        == {}
    )


def test_format_collection_env_changes_redacts_sensitive_values():
    message = format_collection_env_changes(
        {
            "DYN_API_KEY": (None, "secret-value"),
            "DYN_SKIP_PYTHON_LOG_INIT": (None, "1"),
        }
    )

    assert "DYN_API_KEY: <unset> -> <redacted>" in message
    assert "DYN_SKIP_PYTHON_LOG_INIT: <unset> -> '1'" in message
    assert COLLECTION_ENV_GUARD_DISABLE_ENV in message
    assert "secret-value" not in message


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_collection_env_guard_disabled_accepts_truthy_values(value):
    assert collection_env_guard_disabled({COLLECTION_ENV_GUARD_DISABLE_ENV: value})


def test_collection_env_guard_disabled_rejects_default_and_falsey_values():
    assert not collection_env_guard_disabled({})
    assert not collection_env_guard_disabled({COLLECTION_ENV_GUARD_DISABLE_ENV: "0"})
