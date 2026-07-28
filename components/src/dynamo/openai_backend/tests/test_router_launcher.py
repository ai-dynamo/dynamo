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

    def test_router_health_gate_polls_router_health(self):
        from dynamo.openai_backend.sglang import _router_health_url

        assert _router_health_url(_launcher_args()) is None
        assert (
            _router_health_url(_launcher_args(router_port=30000))
            == "http://127.0.0.1:30000/health"
        )

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

    def test_abort_always_targets_engine(self):
        from dynamo.openai_backend.launcher_common import build_worker_command

        for upstream_port in (None, 30000):
            command = build_worker_command(
                _launcher_args(), upstream_port=upstream_port
            )
            abort_url = command[command.index("--abort-base-url") + 1]
            assert abort_url == "http://127.0.0.1:30010"


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


class TestEnsureRid:
    def test_generates_and_injects_rid(self):
        from dynamo.openai_backend.worker import _ensure_rid

        request = {"model": "m"}
        rid = _ensure_rid(request)
        assert rid.startswith("dyn-")
        assert request["rid"] == rid

    def test_preserves_existing_rid(self):
        from dynamo.openai_backend.worker import _ensure_rid

        request = {"model": "m", "rid": "client-supplied"}
        assert _ensure_rid(request) == "client-supplied"
        assert request["rid"] == "client-supplied"

    def test_replaces_non_string_rid(self):
        from dynamo.openai_backend.worker import _ensure_rid

        request = {"rid": 42}
        rid = _ensure_rid(request)
        assert rid.startswith("dyn-")
        assert request["rid"] == rid


class TestAbortBaseUrl:
    def test_defaults_to_upstream_origin(self):
        from dynamo.openai_backend.worker import Config, UpstreamClient

        client = UpstreamClient(
            Config(
                model="m",
                served_model_name=None,
                upstream_base_url="http://127.0.0.1:30000/v1",
                upstream_health_path="/health",
                connect_timeout_seconds=1.0,
                write_timeout_seconds=1.0,
            )
        )
        assert client._abort_base_url == "http://127.0.0.1:30000"

    def test_explicit_override_wins(self):
        from dynamo.openai_backend.worker import Config, UpstreamClient

        client = UpstreamClient(
            Config(
                model="m",
                served_model_name=None,
                upstream_base_url="http://127.0.0.1:30000/v1",
                upstream_health_path="/health",
                connect_timeout_seconds=1.0,
                write_timeout_seconds=1.0,
                abort_base_url="http://127.0.0.1:30010",
            )
        )
        assert client._abort_base_url == "http://127.0.0.1:30010"
