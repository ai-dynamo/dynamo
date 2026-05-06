# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TokenSpeed disaggregated request plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm.exceptions import InvalidArgument
from dynamo.tokenspeed.disagg import (
    BOOTSTRAP_HOST_ENV,
    resolve_bootstrap_host,
    runtime_disaggregated_endpoint,
    validate_disagg_compatibility,
)
from dynamo.tokenspeed.llm_engine import _bootstrap_kwargs_for_request

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


# ---------------------------------------------------------------------------
# _bootstrap_kwargs_for_request
# ---------------------------------------------------------------------------


def test_bootstrap_kwargs_aggregated_returns_empty_dict():
    request = {"token_ids": [1, 2, 3]}
    assert _bootstrap_kwargs_for_request(DisaggregationMode.AGGREGATED, request) == {}


def test_bootstrap_kwargs_aggregated_ignores_disaggregated_state():
    # Even if state is somehow present, agg workers don't pass bootstrap kwargs.
    request = {
        "token_ids": [1],
        "disaggregated_state": {
            "bootstrap_host": "h",
            "bootstrap_port": 1,
            "bootstrap_room": 2,
        },
    }
    assert _bootstrap_kwargs_for_request(DisaggregationMode.AGGREGATED, request) == {}


@pytest.mark.parametrize(
    "mode", [DisaggregationMode.PREFILL, DisaggregationMode.DECODE]
)
def test_bootstrap_kwargs_disagg_requires_state(mode):
    request = {"token_ids": [1, 2]}
    with pytest.raises(InvalidArgument, match="disaggregated_state"):
        _bootstrap_kwargs_for_request(mode, request)


@pytest.mark.parametrize(
    "mode", [DisaggregationMode.PREFILL, DisaggregationMode.DECODE]
)
def test_bootstrap_kwargs_disagg_extracts_all_three_keys(mode):
    request = {
        "token_ids": [1],
        "disaggregated_state": {
            "bootstrap_host": "prefill-host",
            "bootstrap_port": 8998,
            "bootstrap_room": 12345,
        },
    }
    kwargs = _bootstrap_kwargs_for_request(mode, request)
    assert kwargs == {
        "bootstrap_host": "prefill-host",
        "bootstrap_port": 8998,
        "bootstrap_room": 12345,
    }


@pytest.mark.parametrize(
    "missing", ["bootstrap_host", "bootstrap_port", "bootstrap_room"]
)
def test_bootstrap_kwargs_disagg_rejects_partial_state(missing):
    state = {
        "bootstrap_host": "h",
        "bootstrap_port": 1,
        "bootstrap_room": 2,
    }
    state.pop(missing)
    request = {"token_ids": [1], "disaggregated_state": state}
    with pytest.raises(InvalidArgument, match=missing):
        _bootstrap_kwargs_for_request(DisaggregationMode.DECODE, request)


# ---------------------------------------------------------------------------
# disagg.py helpers
# ---------------------------------------------------------------------------


def test_resolve_bootstrap_host_env_overrides(monkeypatch):
    monkeypatch.setenv(BOOTSTRAP_HOST_ENV, "env-host")
    server_args = SimpleNamespace(host="cli-host")
    assert resolve_bootstrap_host(server_args) == "env-host"


def test_resolve_bootstrap_host_falls_back_to_server_args(monkeypatch):
    monkeypatch.delenv(BOOTSTRAP_HOST_ENV, raising=False)
    server_args = SimpleNamespace(host="cli-host")
    assert resolve_bootstrap_host(server_args) == "cli-host"


def test_resolve_bootstrap_host_falls_back_to_hostname(monkeypatch):
    monkeypatch.delenv(BOOTSTRAP_HOST_ENV, raising=False)
    monkeypatch.setattr("socket.gethostname", lambda: "from-socket")
    server_args = SimpleNamespace(host=None)
    assert resolve_bootstrap_host(server_args) == "from-socket"


def test_runtime_disaggregated_endpoint_requires_port(monkeypatch):
    monkeypatch.delenv(BOOTSTRAP_HOST_ENV, raising=False)
    server_args = SimpleNamespace(host="h", disaggregation_bootstrap_port=None)
    with pytest.raises(ValueError, match="--disaggregation-bootstrap-port"):
        runtime_disaggregated_endpoint(server_args)


def test_runtime_disaggregated_endpoint_returns_host_port(monkeypatch):
    monkeypatch.delenv(BOOTSTRAP_HOST_ENV, raising=False)
    server_args = SimpleNamespace(host="h", disaggregation_bootstrap_port=8998)
    assert runtime_disaggregated_endpoint(server_args) == ("h", 8998)


# ---------------------------------------------------------------------------
# validate_disagg_compatibility
# ---------------------------------------------------------------------------


def test_validate_disagg_compatibility_aggregated_skips_checks():
    # Even with bad config, aggregated mode is exempt.
    server_args = SimpleNamespace(dp_size=4, block_size=None)
    validate_disagg_compatibility(DisaggregationMode.AGGREGATED, server_args)


@pytest.mark.parametrize(
    "mode", [DisaggregationMode.PREFILL, DisaggregationMode.DECODE]
)
def test_validate_disagg_compatibility_rejects_dp_gt_one(mode):
    server_args = SimpleNamespace(dp_size=2, block_size=64)
    with pytest.raises(ValueError, match="DP > 1"):
        validate_disagg_compatibility(mode, server_args)


@pytest.mark.parametrize(
    "mode", [DisaggregationMode.PREFILL, DisaggregationMode.DECODE]
)
def test_validate_disagg_compatibility_requires_block_size(mode):
    server_args = SimpleNamespace(dp_size=1, block_size=None)
    with pytest.raises(ValueError, match="block-size"):
        validate_disagg_compatibility(mode, server_args)


@pytest.mark.parametrize(
    "mode", [DisaggregationMode.PREFILL, DisaggregationMode.DECODE]
)
def test_validate_disagg_compatibility_passes_with_valid_config(mode):
    server_args = SimpleNamespace(dp_size=1, block_size=64)
    validate_disagg_compatibility(mode, server_args)
