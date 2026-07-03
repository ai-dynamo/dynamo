# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend admission gate knob parsing and validation (DIS-2186)."""

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

_GATE_ENV_VARS = (
    "DYN_REJECTION_FRONTEND_REQUEST_CONCURRENCY_LIMIT",
    "DYN_REJECTION_FRONTEND_RUNTIME_TASK_LIMIT",
    "DYN_REJECTION_FRONTEND_REQUEST_PLANE_CONNECTION_LIMIT",
)


def _parse_config(argv: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    args = parser.parse_args(argv)
    config = FrontendConfig.from_cli_args(args)
    config.validate()
    return config


@pytest.fixture(autouse=True)
def _clear_gate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _GATE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_admission_gates_disabled_by_default() -> None:
    config = _parse_config([])

    assert config.rejection_frontend_request_concurrency_limit is None
    assert config.rejection_frontend_runtime_task_limit is None
    assert config.rejection_frontend_request_plane_connection_limit is None


def test_admission_gate_flags_parse() -> None:
    config = _parse_config(
        [
            "--rejection-frontend-request-concurrency-limit",
            "8",
            "--rejection-frontend-runtime-task-limit",
            "10000",
            "--rejection-frontend-request-plane-connection-limit",
            "512",
        ]
    )

    assert config.rejection_frontend_request_concurrency_limit == 8
    assert config.rejection_frontend_runtime_task_limit == 10000
    assert config.rejection_frontend_request_plane_connection_limit == 512


@pytest.mark.parametrize(
    "flag",
    [
        "--rejection-frontend-request-concurrency-limit",
        "--rejection-frontend-runtime-task-limit",
        "--rejection-frontend-request-plane-connection-limit",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_admission_gate_rejects_non_positive_values(flag: str, value: str) -> None:
    with pytest.raises(ValueError, match="must be between 1 and"):
        _parse_config([flag, value])


def test_admission_gate_env_vars_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DYN_REJECTION_FRONTEND_REQUEST_CONCURRENCY_LIMIT", "4")
    monkeypatch.setenv("DYN_REJECTION_FRONTEND_RUNTIME_TASK_LIMIT", "20000")
    monkeypatch.setenv(
        "DYN_REJECTION_FRONTEND_REQUEST_PLANE_CONNECTION_LIMIT", "1024"
    )

    config = _parse_config([])

    assert config.rejection_frontend_request_concurrency_limit == 4
    assert config.rejection_frontend_runtime_task_limit == 20000
    assert config.rejection_frontend_request_plane_connection_limit == 1024


def test_admission_gate_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DYN_REJECTION_FRONTEND_REQUEST_CONCURRENCY_LIMIT", "4")

    config = _parse_config(
        ["--rejection-frontend-request-concurrency-limit", "16"]
    )

    assert config.rejection_frontend_request_concurrency_limit == 16
