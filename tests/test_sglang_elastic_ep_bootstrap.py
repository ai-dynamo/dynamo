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
    # The launcher is stdlib-only; these checks must run without an installed engine.
    pytest.mark.core,
]


@pytest.fixture(params=[(2, 3), (4, 8), (8, 16)])
def bootstrap_arguments(
    request: pytest.FixtureRequest, dynamo_dynamic_ports
) -> list[str]:
    initial, maximum = request.param
    return [
        "--model-path",
        "model",
        "--tp",
        str(initial),
        "--dp",
        str(initial),
        "--nnodes",
        str(initial),
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--elastic-ep-initial-size",
        str(initial),
        "--max-ep-size",
        str(maximum),
        "--dist-init-addr",
        f"0.0.0.0:{dynamo_dynamic_ports.frontend_port}",
        "--elastic-ep-backend",
        "mooncake",
        "--moe-a2a-backend",
        "nixl",
    ]


@pytest.mark.parametrize(
    "role", ["primary", "initial-peer", "first-joiner", "last-joiner"]
)
def test_shared_template_selects_primary_or_joiner(
    bootstrap_arguments: list[str], role: str
) -> None:
    initial = int(
        bootstrap_arguments[bootstrap_arguments.index("--elastic-ep-initial-size") + 1]
    )
    maximum = int(bootstrap_arguments[bootstrap_arguments.index("--max-ep-size") + 1])
    indices = {
        "primary": 0,
        "initial-peer": initial - 1,
        "first-joiner": initial,
        "last-joiner": maximum - 1,
    }
    index = indices[role]
    initial_participant = index < initial
    module = "dynamo.sglang" if initial_participant else "sglang.launch_server"
    width = str(initial) if initial_participant else "1"
    environment = {
        "GROVE_PCLQ_POD_INDEX": str(index),
        "GROVE_PCLQ_NAME": "world-0-members",
        "GROVE_HEADLESS_SERVICE": "workload-0.namespace.svc.cluster.local",
    }
    original = bootstrap_arguments.copy()

    command = elastic_ep_bootstrap.resolve_command(
        bootstrap_arguments, environment, "/python"
    )

    assert bootstrap_arguments == original
    assert command[:3] == ["/python", "-m", module]
    assert command[command.index("--nnodes") + 1] == (
        str(initial) if initial_participant else "2"
    )
    for option in ("--tp", "--dp", "--ep"):
        assert command[command.index(option) + 1] == width
    port = bootstrap_arguments[bootstrap_arguments.index("--dist-init-addr") + 1].split(
        ":"
    )[-1]
    assert command[command.index("--dist-init-addr") + 1] == (
        f"world-0-members-0.workload-0.namespace.svc.cluster.local:{port}"
    )
    assert command[command.index("--elastic-ep-initial-size") + 1] == str(initial)
    assert command[command.index("--max-ep-size") + 1] == str(maximum)
    assert command[command.index("--pp-size") + 1] == "1"
    assert command[command.index("--moe-dense-tp-size") + 1] == "1"
    assert "--pp" not in command
    assert command[command.index("--elastic-ep-backend") + 1] == "mooncake"
    if not initial_participant:
        assert command[command.index("--node-rank") + 1] == "1"
        assert command[command.index("--elastic-ep-join-mode") + 1] == "scale"
        assert command[command.index("--elastic-ep-join-rank-offset") + 1] == str(index)
    else:
        assert "--elastic-ep-join-mode" not in command
        assert command[command.index("--node-rank") + 1] == str(index)


@pytest.mark.parametrize(
    "extra",
    [
        ["--tp=1"],
        ["--dp-size", "1"],
        ["--ep-size", "1"],
        ["--nnodes", "1"],
        ["--elastic-ep-initial-size", "1"],
        ["--max-ep-size", "1"],
        ["--moe-dense-tp-size", "2"],
        ["--moe-dp-size", "2"],
        ["--pp-size", "2"],
        ["--attn-cp-size", "2"],
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


@pytest.mark.parametrize("index", ["", "01", "-1", "max", "past-max", "invalid"])
def test_bootstrap_rejects_invalid_or_unsupported_slots(
    bootstrap_arguments: list[str], index: str
) -> None:
    maximum = int(bootstrap_arguments[bootstrap_arguments.index("--max-ep-size") + 1])
    if index == "max":
        index = str(maximum)
    elif index == "past-max":
        index = str(maximum + 1)
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


def test_bootstrap_derives_initial_size_from_launch_tp(
    bootstrap_arguments: list[str],
) -> None:
    option_index = bootstrap_arguments.index("--elastic-ep-initial-size")
    initial = bootstrap_arguments[option_index + 1]
    del bootstrap_arguments[option_index : option_index + 2]
    environment = {
        "GROVE_PCLQ_POD_INDEX": "0",
        "GROVE_PCLQ_NAME": "world-0-members",
        "GROVE_HEADLESS_SERVICE": "workload-0.namespace.svc.cluster.local",
    }

    command = elastic_ep_bootstrap.resolve_command(
        bootstrap_arguments, environment, "/python"
    )

    assert command[command.index("--elastic-ep-initial-size") + 1] == initial
    assert command[command.index("--nnodes") + 1] == initial


def test_launcher_execs_instead_of_supervising_children(
    bootstrap_arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = bootstrap_arguments[
        bootstrap_arguments.index("--elastic-ep-initial-size") + 1
    ]
    monkeypatch.setenv("GROVE_PCLQ_POD_INDEX", initial)
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
