# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TokenSpeed disaggregation argument plumbing.

These tests exercise the pure-Python helpers added on top of PR #9212 to
support prefill/decode disaggregated serving. They intentionally avoid the
full ``parse_args`` flow (which imports TokenSpeed's ``ServerArgs``) and
instead pin down the contract of the small helpers used inside it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.tokenspeed.args import (
    _resolve_disaggregation_mode,
    _validate_disagg_transfer_backend,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, DisaggregationMode.AGGREGATED),
        ("null", DisaggregationMode.AGGREGATED),
        ("prefill", DisaggregationMode.PREFILL),
        ("decode", DisaggregationMode.DECODE),
    ],
)
def test_resolve_disaggregation_mode_maps_tokenspeed_strings(raw, expected):
    server_args = SimpleNamespace(disaggregation_mode=raw)
    assert _resolve_disaggregation_mode(server_args) == expected


def test_resolve_disaggregation_mode_unknown_string_falls_back_to_aggregated():
    # Forward-compat: an unrecognized future TokenSpeed mode should not crash
    # arg parsing; it just won't be promoted to PREFILL/DECODE here.
    server_args = SimpleNamespace(disaggregation_mode="something-new")
    assert _resolve_disaggregation_mode(server_args) == DisaggregationMode.AGGREGATED


def test_validate_disagg_transfer_backend_skips_aggregated():
    server_args = SimpleNamespace(disaggregation_transfer_backend=None)
    # Should not raise — aggregated mode doesn't care about transfer backend.
    _validate_disagg_transfer_backend(DisaggregationMode.AGGREGATED, server_args)


@pytest.mark.parametrize("backend", ["mooncake", "mooncake_async"])
@pytest.mark.parametrize(
    "mode", [DisaggregationMode.PREFILL, DisaggregationMode.DECODE]
)
def test_validate_disagg_transfer_backend_accepts_mooncake_family(mode, backend):
    server_args = SimpleNamespace(disaggregation_transfer_backend=backend)
    _validate_disagg_transfer_backend(mode, server_args)


@pytest.mark.parametrize(
    "mode", [DisaggregationMode.PREFILL, DisaggregationMode.DECODE]
)
@pytest.mark.parametrize("backend", [None, "nixl", "ucx", "tcp"])
def test_validate_disagg_transfer_backend_rejects_unsupported(mode, backend):
    server_args = SimpleNamespace(disaggregation_transfer_backend=backend)
    with pytest.raises(ValueError, match="disaggregation-transfer-backend"):
        _validate_disagg_transfer_backend(mode, server_args)
