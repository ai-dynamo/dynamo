# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the OpenAI backend launcher and worker helpers."""

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
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestWorkerUpstream:
    def test_worker_targets_engine(self):
        from dynamo.openai_backend.launcher_common import build_worker_command

        command = build_worker_command(_launcher_args())
        url = command[command.index("--upstream-base-url") + 1]
        assert url == "http://127.0.0.1:30010/v1"

    def test_abort_targets_engine(self):
        from dynamo.openai_backend.launcher_common import build_worker_command

        command = build_worker_command(_launcher_args())
        abort_url = command[command.index("--abort-base-url") + 1]
        assert abort_url == "http://127.0.0.1:30010"


class TestUpstreamHeaders:
    def test_user_field_is_not_forwarded_as_a_header(self):
        from dynamo.openai_backend.worker import _upstream_headers

        assert _upstream_headers({"model": "m", "user": "session-1"}) == {
            "Content-Type": "application/json"
        }

    def test_nvext_dp_rank_sets_sglang_header(self):
        from dynamo.openai_backend.worker import DP_RANK_HEADER, _upstream_headers

        headers = _upstream_headers({"model": "m", "nvext": {"dp_rank": 7}})
        assert headers[DP_RANK_HEADER] == "7"

    def test_invalid_nvext_dp_rank_is_ignored(self):
        from dynamo.openai_backend.worker import DP_RANK_HEADER, _upstream_headers

        for value in (None, True, -1, 2**32, "7", 1.5):
            headers = _upstream_headers({"nvext": {"dp_rank": value}})
            assert DP_RANK_HEADER not in headers

        assert DP_RANK_HEADER not in _upstream_headers({"nvext": "not-an-object"})


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
