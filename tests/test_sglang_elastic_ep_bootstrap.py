# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract checks for the narrow Grove/SGLang bootstrap entrypoint."""

from unittest.mock import Mock

import pytest
from dynamo.sglang import elastic_ep_bootstrap

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
]


@pytest.fixture
def bootstrap_arguments() -> list[str]:
    return [
        "--model-path",
        "model",
        "--tp",
        "1",
        "--dp",
        "1",
        "--nnodes",
        "1",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--elastic-ep-initial-size",
        "1",
        "--max-ep-size",
        "2",
        "--dist-init-addr",
        "0.0.0.0:23456",
        "--elastic-ep-backend",
        "mooncake",
        "--moe-a2a-backend",
        "nixl",
    ]


@pytest.mark.parametrize(
    ("index", "module", "nodes"),
    [("0", "dynamo.sglang", "1"), ("1", "sglang.launch_server", "2")],
)
def test_shared_template_selects_primary_or_joiner(
    bootstrap_arguments: list[str], index: str, module: str, nodes: str
) -> None:
    environment = {
        "GROVE_PCLQ_POD_INDEX": index,
        "GROVE_PCLQ_NAME": "world-0-members",
        "GROVE_HEADLESS_SERVICE": "workload-0.namespace.svc.cluster.local",
    }
    original = bootstrap_arguments.copy()

    command = elastic_ep_bootstrap.resolve_command(
        bootstrap_arguments, environment, "/python"
    )

    assert bootstrap_arguments == original
    assert command[:3] == ["/python", "-m", module]
    assert command[command.index("--nnodes") + 1] == nodes
    assert command[command.index("--dist-init-addr") + 1] == (
        "world-0-members-0.workload-0.namespace.svc.cluster.local:23456"
    )
    assert command[command.index("--elastic-ep-initial-size") + 1] == "1"
    assert command[command.index("--max-ep-size") + 1] == "2"
    assert command[command.index("--elastic-ep-backend") + 1] == "mooncake"
    if index == "1":
        assert command[command.index("--node-rank") + 1] == "1"
        assert command[command.index("--elastic-ep-join-mode") + 1] == "scale"
        assert command[command.index("--elastic-ep-join-rank-offset") + 1] == "1"
    else:
        assert "--elastic-ep-join-mode" not in command
        assert "--node-rank" not in command


@pytest.mark.parametrize(
    "extra",
    [
        ["--tp=2"],
        ["--dp-size", "2"],
        ["--ep-size", "2"],
        ["--nnodes", "2"],
        ["--elastic-ep-initial-size", "2"],
        ["--max-ep-size", "3"],
        ["--node-rank", "0"],
        ["--elastic-ep-join-mode", "recover"],
        ["--elastic-ep-join-rank-offset", "1"],
        ["--dist-init-addr", "host:0"],
    ],
)
def test_bootstrap_rejects_conflicting_geometry(
    bootstrap_arguments: list[str], extra: list[str]
) -> None:
    environment = {
        "GROVE_PCLQ_POD_INDEX": "1",
        "GROVE_PCLQ_NAME": "world-0-members",
        "GROVE_HEADLESS_SERVICE": "workload-0.namespace.svc.cluster.local",
    }
    with pytest.raises(ValueError):
        elastic_ep_bootstrap.resolve_command(
            bootstrap_arguments + extra, environment, "/python"
        )


@pytest.mark.parametrize("index", ["", "01", "-1", "2"])
def test_bootstrap_rejects_invalid_or_unsupported_slots(
    bootstrap_arguments: list[str], index: str
) -> None:
    environment = {
        "GROVE_PCLQ_POD_INDEX": index,
        "GROVE_PCLQ_NAME": "world-0-members",
        "GROVE_HEADLESS_SERVICE": "workload-0.namespace.svc.cluster.local",
    }
    with pytest.raises(ValueError, match="allocation index"):
        elastic_ep_bootstrap.resolve_command(
            bootstrap_arguments, environment, "/python"
        )


def test_bootstrap_requires_dp_attention(bootstrap_arguments: list[str]) -> None:
    environment = {
        "GROVE_PCLQ_POD_INDEX": "0",
        "GROVE_PCLQ_NAME": "world-0-members",
        "GROVE_HEADLESS_SERVICE": "workload-0.namespace.svc.cluster.local",
    }
    bootstrap_arguments.remove("--enable-dp-attention")
    with pytest.raises(ValueError, match="one-GPU"):
        elastic_ep_bootstrap.resolve_command(
            bootstrap_arguments, environment, "/python"
        )


def test_launcher_execs_instead_of_supervising_children(
    bootstrap_arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROVE_PCLQ_POD_INDEX", "1")
    monkeypatch.setenv("GROVE_PCLQ_NAME", "world-0-members")
    monkeypatch.setenv(
        "GROVE_HEADLESS_SERVICE", "workload-0.namespace.svc.cluster.local"
    )
    monkeypatch.setattr(
        elastic_ep_bootstrap.sys, "argv", ["bootstrap", *bootstrap_arguments]
    )
    execute = Mock()
    monkeypatch.setattr(elastic_ep_bootstrap.os, "execv", execute)

    elastic_ep_bootstrap.main()

    execute.assert_called_once()
    executable, command = execute.call_args.args
    assert command[:3] == [executable, "-m", "sglang.launch_server"]
