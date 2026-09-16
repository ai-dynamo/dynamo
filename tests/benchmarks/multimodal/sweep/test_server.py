# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        manager.start(
            str(workflow),
            "model",
            env_overrides={"DYN_SERVER_TERMINATE_TIMEOUT": "0"},
        )
