# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.utils.collection_env_guard import (
    ALLOWED_COLLECTION_ENV_MUTATIONS,
    COLLECTION_ENV_GUARD_DISABLE_ENV,
    WATCHED_ENV_KEYS,
    WATCHED_ENV_PREFIXES,
    collection_env_guard_disabled,
    diff_collection_env,
    format_collection_env_changes,
    snapshot_collection_env,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.parallel,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.pre_merge,
]


def test_snapshot_collection_env_filters_to_watched_prefixes():
    env = {
        "DYNAMO_TEST_FRAMEWORK": "vllm",
        "DYNAMO_SKIP_PYTHON_LOG_INIT": "1",
        "DYN_SYSTEM_PORT": "9090",
        "SGLANG_LOGGING_LEVEL": "debug",
        "TRTLLM_LOG_LEVEL": "info",
        "VLLM_NO_USAGE_STATS": "1",
        "NCCL_DEBUG": "INFO",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": "/example/hf-home",
        "TORCH_LOGS": "dynamic",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PATH": "/usr/bin",
    }

    # The broad "DYNAMO_" prefix is watched, so any DYNAMO_* var (including
    # DYNAMO_TEST_FRAMEWORK) is captured -- nothing should set these during
    # collection, which is exactly what the guard is meant to catch.
    assert snapshot_collection_env(env) == {
        "DYNAMO_TEST_FRAMEWORK": "vllm",
        "DYNAMO_SKIP_PYTHON_LOG_INIT": "1",
        "DYN_SYSTEM_PORT": "9090",
        "SGLANG_LOGGING_LEVEL": "debug",
        "TRTLLM_LOG_LEVEL": "info",
        "VLLM_NO_USAGE_STATS": "1",
        "NCCL_DEBUG": "INFO",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": "/example/hf-home",
        "TORCH_LOGS": "dynamic",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def test_watched_prefixes_cover_backend_and_dynamo_env():
    assert "DYN_" in WATCHED_ENV_PREFIXES
    assert "DYNAMO_" in WATCHED_ENV_PREFIXES
    assert "SGLANG_" in WATCHED_ENV_PREFIXES
    assert "TRTLLM_" in WATCHED_ENV_PREFIXES
    # The #9724 leak key is now covered by the broad "DYNAMO_" prefix, so the
    # exact-key set is empty.
    assert WATCHED_ENV_KEYS == frozenset()
    assert "DYNAMO_SKIP_PYTHON_LOG_INIT".startswith(tuple(WATCHED_ENV_PREFIXES))


def test_diff_collection_env_reports_added_changed_and_removed_values():
    before = {
        "DYNAMO_SKIP_PYTHON_LOG_INIT": "1",
        "SGLANG_LOGGING_CONFIG_PATH": "/example/old.json",
        "VLLM_NO_USAGE_STATS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    after = {
        "SGLANG_LOGGING_CONFIG_PATH": "/example/new.json",
        "VLLM_NO_USAGE_STATS": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "NCCL_DEBUG": "INFO",
    }

    assert diff_collection_env(before, after) == {
        "DYNAMO_SKIP_PYTHON_LOG_INIT": ("1", None),
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
                "SGLANG_LOGGING_CONFIG_PATH": "/example/sglang.json",
                "VLLM_CONFIGURE_LOGGING": "1",
            },
        )
        == {}
    )


def test_diff_collection_env_ignores_tensorrt_llm_import_side_effects():
    # `import tensorrt_llm` (pulled in by trtllm test modules at collection)
    # sets these as import-time side effects; they are benign and allowlisted.
    assert "TLLM_LOG_LEVEL" in ALLOWED_COLLECTION_ENV_MUTATIONS
    assert "OMPI_MCA_coll_ucc_enable" in ALLOWED_COLLECTION_ENV_MUTATIONS
    assert (
        diff_collection_env(
            {},
            {
                "TLLM_LOG_LEVEL": "INFO",
                "OMPI_MCA_coll_ucc_enable": "0",
            },
        )
        == {}
    )


def test_format_collection_env_changes_redacts_sensitive_values():
    message = format_collection_env_changes(
        {
            "DYN_API_KEY": (None, "secret-value"),
            "DYNAMO_SKIP_PYTHON_LOG_INIT": (None, "1"),
        }
    )

    assert "DYN_API_KEY: <unset> -> <redacted>" in message
    assert "DYNAMO_SKIP_PYTHON_LOG_INIT: <unset> -> '1'" in message
    assert COLLECTION_ENV_GUARD_DISABLE_ENV in message
    assert "secret-value" not in message


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_collection_env_guard_disabled_accepts_truthy_values(value):
    assert collection_env_guard_disabled({COLLECTION_ENV_GUARD_DISABLE_ENV: value})


def test_collection_env_guard_disabled_rejects_default_and_falsey_values():
    assert not collection_env_guard_disabled({})
    assert not collection_env_guard_disabled({COLLECTION_ENV_GUARD_DISABLE_ENV: "0"})


# Watched var the wider suite never sets, so the regression below is the only
# thing mutating it during collection.
_REGRESSION_ENV_VAR = "SGLANG_COLLECTION_ENV_GUARD_REGRESSION"


def _run_collect_only_with_import_mutation(
    tmp_path: Path, *, guard_disabled: bool, parallel: bool = False
) -> subprocess.CompletedProcess[str]:
    """Collect a dummy module that mutates a watched env var at import time.

    The real ``tests/conftest.py`` is loaded as a plugin (``-p tests.conftest``)
    so this exercises the wired hooks end-to-end, not just the helpers. If the
    hooks are ever removed from conftest, the failure assertion below breaks.

    With ``parallel=True`` the run uses pytest-xdist workers. The import (and
    therefore the mutation) happens on a worker, so this covers the worker ->
    controller reporting path rather than the in-process raise.
    """
    dummy = tmp_path / "test_collection_env_guard_regression_dummy.py"
    dummy.write_text(
        textwrap.dedent(
            f"""
            import os

            os.environ["{_REGRESSION_ENV_VAR}"] = "leaked-at-import"


            def test_placeholder():
                pass
            """
        )
    )

    env = os.environ.copy()
    env.pop(_REGRESSION_ENV_VAR, None)
    if guard_disabled:
        env[COLLECTION_ENV_GUARD_DISABLE_ENV] = "1"
    else:
        env.pop(COLLECTION_ENV_GUARD_DISABLE_ENV, None)

    parallel_args = ["-n", "2"] if parallel else []
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            *parallel_args,
            "--collect-only",
            "-q",
            str(dummy),
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_collection_guard_fails_on_import_time_env_mutation(tmp_path):
    result = _run_collect_only_with_import_mutation(tmp_path, guard_disabled=False)

    # pytest.UsageError raised in pytest_collection_finish -> exit code 4.
    assert result.returncode == 4, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "pytest collection mutated watched environment variables" in combined
    assert _REGRESSION_ENV_VAR in combined


def test_collection_guard_bypass_allows_import_time_env_mutation(tmp_path):
    result = _run_collect_only_with_import_mutation(tmp_path, guard_disabled=True)

    assert result.returncode == 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "pytest collection mutated watched environment variables" not in combined


def test_collection_guard_fails_cleanly_under_xdist(tmp_path):
    # Under xdist the mutation happens on a worker. Raising there used to crash
    # the worker and surface as an opaque "INTERNALERROR: assert not crashitem"
    # instead of our message. The worker now reports back to the controller,
    # which fails the session cleanly with the same USAGE_ERROR exit code.
    result = _run_collect_only_with_import_mutation(
        tmp_path, guard_disabled=False, parallel=True
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 4, combined
    assert "pytest collection mutated watched environment variables" in combined
    assert _REGRESSION_ENV_VAR in combined
    assert "INTERNALERROR" not in combined
