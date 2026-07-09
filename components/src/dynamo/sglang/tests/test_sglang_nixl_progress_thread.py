# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the downstream SGLang NIXL progress-thread switch."""

from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

nixl_api = pytest.importorskip("nixl._api", reason="NIXL is not installed")
common_conn = pytest.importorskip(
    "sglang.srt.disaggregation.common.conn",
    reason="SGLang NIXL disaggregation support is not installed",
)
nixl_conn = pytest.importorskip(
    "sglang.srt.disaggregation.nixl.conn",
    reason="SGLang NIXL disaggregation support is not installed",
)
disaggregation_utils = pytest.importorskip(
    "sglang.srt.disaggregation.utils",
    reason="SGLang disaggregation support is not installed",
)

CommonKVManager = common_conn.CommonKVManager
NixlKVManager = nixl_conn.NixlKVManager
DisaggregationMode = disaggregation_utils.DisaggregationMode


class _FakeNixlAgent:
    def create_backend(self, backend, backend_params):
        assert backend == "LIBFABRIC"
        assert backend_params == {}

    def get_plugin_list(self):
        return ["LIBFABRIC"]


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, True), ("false", False), ("true", True)],
)
def test_nixl_progress_thread_env_reaches_agent_config(
    monkeypatch, env_value, expected
):
    env_name = "SGLANG_DISAGGREGATION_NIXL_ENABLE_PROG_THREAD"
    if env_value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, env_value)
    monkeypatch.setenv("SGLANG_DISAGGREGATION_NIXL_BACKEND", "LIBFABRIC")
    monkeypatch.setenv("SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS", "{}")
    monkeypatch.setenv("SGLANG_DISAGGREGATION_QUEUE_SIZE", "0")
    monkeypatch.setenv("SGLANG_DISAGG_STAGING_BUFFER", "false")

    captured = {}
    real_agent_config = nixl_api.nixl_agent_config

    def capture_agent_config(**kwargs):
        captured["kwargs"] = kwargs
        config = real_agent_config(**kwargs)
        captured["config"] = config
        return config

    monkeypatch.setattr(nixl_api, "nixl_agent_config", capture_agent_config)
    monkeypatch.setattr(nixl_api, "nixl_agent", lambda *_args: _FakeNixlAgent())

    def init_common_manager(self, args, disaggregation_mode, *_args, **_kwargs):
        self.kv_args = args
        self.disaggregation_mode = disaggregation_mode

    monkeypatch.setattr(CommonKVManager, "__init__", init_common_manager)
    monkeypatch.setattr(NixlKVManager, "register_buffer_to_engine", lambda _self: None)
    monkeypatch.setattr(NixlKVManager, "_start_bootstrap_thread", lambda _self: None)

    NixlKVManager(
        SimpleNamespace(kv_data_lens=[1], kv_item_lens=[1]),
        DisaggregationMode.PREFILL,
        SimpleNamespace(),
    )

    config = captured["config"]
    assert captured["kwargs"]["enable_prog_thread"] is expected
    assert config.backends == []
    assert config.num_threads == 8
    assert config.enable_pthread is expected
