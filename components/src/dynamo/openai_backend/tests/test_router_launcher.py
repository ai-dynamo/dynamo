# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def _launcher_args(**overrides):
    args = SimpleNamespace(
        model="org/model",
        served_model_name=None,
        engine_host="127.0.0.1",
        engine_port=30010,
        api_prefix="/v1",
        health_path="/health",
        engine_args=[],
        router_port=None,
        router_args="",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestRouterCommand:
    def test_disabled_without_port(self):
        from dynamo.openai_backend.sglang import _router_command

        assert _router_command(_launcher_args()) is None

    def test_builds_command_with_port_and_args(self, monkeypatch):
        from dynamo.openai_backend import sglang as sglang_launcher

        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: object() if name == "sglang_router" else real_find_spec(name),
        )

        command = sglang_launcher._router_command(
            _launcher_args(
                router_port=30000,
                router_args="--dp-aware --policy manual --assignment-mode min_load",
            )
        )

        assert command is not None
        assert command[1:3] == ["-m", "sglang_router.launch_router"]
        assert command[command.index("--port") + 1] == "30000"
        assert (
            command[command.index("--worker-urls") + 1] == "http://127.0.0.1:30010"
        )
        assert command[-5:] == [
            "--dp-aware",
            "--policy",
            "manual",
            "--assignment-mode",
            "min_load",
        ]

    def test_missing_package_raises(self, monkeypatch):
        from dynamo.openai_backend import sglang as sglang_launcher

        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: None if name == "sglang_router" else real_find_spec(name),
        )

        with pytest.raises(SystemExit):
            sglang_launcher._router_command(_launcher_args(router_port=30000))


class TestWorkerUpstream:
    def test_worker_targets_engine_without_router(self):
        from dynamo.openai_backend.launcher_common import build_worker_command

        command = build_worker_command(_launcher_args())
        url = command[command.index("--upstream-base-url") + 1]
        assert url == "http://127.0.0.1:30010/v1"

    def test_worker_targets_router_when_enabled(self):
        from dynamo.openai_backend.launcher_common import build_worker_command

        command = build_worker_command(_launcher_args(), upstream_port=30000)
        url = command[command.index("--upstream-base-url") + 1]
        assert url == "http://127.0.0.1:30000/v1"


class TestUpstreamHeaders:
    def test_no_user_field(self):
        from dynamo.openai_backend.worker import _upstream_headers

        assert _upstream_headers({"model": "m"}) == {
            "Content-Type": "application/json"
        }

    def test_user_field_sets_routing_key(self):
        from dynamo.openai_backend.worker import ROUTING_KEY_HEADER, _upstream_headers

        headers = _upstream_headers({"model": "m", "user": "glmload03-17"})
        assert headers[ROUTING_KEY_HEADER] == "glmload03-17"
        assert headers["Content-Type"] == "application/json"

    def test_non_string_user_ignored(self):
        from dynamo.openai_backend.worker import ROUTING_KEY_HEADER, _upstream_headers

        for value in (None, "", 7, {"id": "x"}):
            headers = _upstream_headers({"user": value})
            assert ROUTING_KEY_HEADER not in headers
