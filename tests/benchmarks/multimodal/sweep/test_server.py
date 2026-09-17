# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.multimodal.sweep.server import ServerManager

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"DYN_DISABLE_NSYS": "1"}, 15.0),
        ({"DYN_DISABLE_NSYS": "0"}, 300.0),
        ({"DYN_SERVER_TERMINATE_TIMEOUT": "42"}, 42.0),
    ],
)
def test_start_resolves_termination_timeout_and_port(
    tmp_path, env: dict[str, str], expected: float
) -> None:
    workflow = tmp_path / "workflow.sh"
    workflow.write_text("#!/bin/bash\n")
    process = MagicMock()

    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "benchmarks.multimodal.sweep.server.subprocess.Popen",
            return_value=process,
        ) as popen,
        patch.object(ServerManager, "wait_for_ready"),
    ):
        manager = ServerManager(port=8123)
        manager.start(str(workflow), "model", env_overrides=env)

    assert manager.terminate_timeout == expected
    assert popen.call_args.kwargs["env"]["DYN_HTTP_PORT"] == "8123"


def test_start_rejects_non_positive_termination_timeout(tmp_path) -> None:
    workflow = tmp_path / "workflow.sh"
    workflow.write_text("#!/bin/bash\n")
    manager = ServerManager()

    with pytest.raises(ValueError, match="must be positive"):
        with patch.dict("os.environ", {}, clear=True):
            manager.start(
                str(workflow),
                "model",
                env_overrides={"DYN_SERVER_TERMINATE_TIMEOUT": "0"},
            )


def test_start_rejects_timeout_shorter_than_wrapper_grace(tmp_path) -> None:
    workflow = tmp_path / "workflow.sh"
    workflow.write_text("#!/bin/bash\n")
    manager = ServerManager()

    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(ValueError, match="must exceed"),
    ):
        manager.start(
            str(workflow),
            "model",
            env_overrides={
                "DYN_SERVER_TERMINATE_TIMEOUT": "15",
                "DYN_SERVER_SHUTDOWN_GRACE_SECONDS": "20",
            },
        )


def test_start_treats_empty_timeout_override_as_unset(tmp_path) -> None:
    workflow = tmp_path / "workflow.sh"
    workflow.write_text("#!/bin/bash\n")
    process = MagicMock()

    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "benchmarks.multimodal.sweep.server.subprocess.Popen",
            return_value=process,
        ),
        patch.object(ServerManager, "wait_for_ready"),
    ):
        manager = ServerManager()
        manager.start(
            str(workflow),
            "model",
            env_overrides={"DYN_SERVER_TERMINATE_TIMEOUT": ""},
        )

    assert manager.terminate_timeout == 15.0


def test_validate_prefix_cache_persists_successful_probe(tmp_path) -> None:
    responses = [
        {"usage": {"prompt_tokens": 8016, "prompt_tokens_details": {}}},
        {
            "usage": {
                "prompt_tokens": 8016,
                "prompt_tokens_details": {"cached_tokens": 8000},
            }
        },
    ]

    class FakeResponse:
        def __init__(self, value: dict) -> None:
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self.value).encode()

    output = tmp_path / "probe.json"
    with patch(
        "benchmarks.multimodal.sweep.server.urllib.request.urlopen",
        side_effect=[FakeResponse(value) for value in responses],
    ) as urlopen:
        result = ServerManager(port=8123).validate_prefix_cache(
            model="model",
            user_text="question",
            min_cached_tokens=7936,
            output_path=output,
        )

    assert urlopen.call_count == 2
    assert result["passed"] is True
    assert json.loads(output.read_text()) == result


def test_validate_prefix_cache_rejects_uncached_prompt(tmp_path) -> None:
    response = {
        "usage": {
            "prompt_tokens": 8016,
            "prompt_tokens_details": {"cached_tokens": 128},
        }
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response).encode()

    with (
        patch(
            "benchmarks.multimodal.sweep.server.urllib.request.urlopen",
            side_effect=[FakeResponse(), FakeResponse()],
        ),
        pytest.raises(RuntimeError, match="cached 128 tokens"),
    ):
        ServerManager().validate_prefix_cache(
            model="model",
            user_text="question",
            min_cached_tokens=7936,
            output_path=tmp_path / "probe.json",
        )
