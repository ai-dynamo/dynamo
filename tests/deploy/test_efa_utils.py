# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the EFA deploy-test helpers (tests/deploy/efa_utils.py)."""

import pytest

from tests.deploy.efa_utils import parse_efa_device_metrics

# Real exporter output, trimmed. Device names are nested on purpose: rdmap16s2
# is a prefix of rdmap16s27, and rdmap16s27 of rdmap16s270. A substring match on
# the device name -- which is what this parser used to do -- attributes the wrong
# NIC's counters to the worker, and because the result is keyed only by metric
# name the last match silently wins.
_COLLIDING_METRICS = """\
# HELP node_amazonefa_rdma_read_bytes The number of bytes read with RDMA
# TYPE node_amazonefa_rdma_read_bytes gauge
node_amazonefa_rdma_read_bytes{device="rdmap16s2",port="1"} 100
node_amazonefa_rdma_read_bytes{device="rdmap16s27",port="1"} 200
node_amazonefa_rdma_read_bytes{device="rdmap16s270",port="1"} 300
node_amazonefa_rdma_read_resp_bytes{device="rdmap16s27",port="1"} 400
node_amazonefa_tx_bytes{device="rdmap16s270",port="1"} 500
node_cpu_seconds_total{cpu="0",mode="idle"} 999
""".splitlines()


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_parse_efa_device_metrics_matches_device_exactly() -> None:
    """Each device gets only its own samples, never a longer name's."""
    assert parse_efa_device_metrics(_COLLIDING_METRICS, "rdmap16s2") == {
        "node_amazonefa_rdma_read_bytes": 100.0
    }
    assert parse_efa_device_metrics(_COLLIDING_METRICS, "rdmap16s27") == {
        "node_amazonefa_rdma_read_bytes": 200.0,
        "node_amazonefa_rdma_read_resp_bytes": 400.0,
    }
    assert parse_efa_device_metrics(_COLLIDING_METRICS, "rdmap16s270") == {
        "node_amazonefa_rdma_read_bytes": 300.0,
        "node_amazonefa_tx_bytes": 500.0,
    }
    # An absent device yields nothing rather than borrowing a neighbour's series,
    # so assert_efa_device_traffic fails closed on its "counter missing" branch.
    assert parse_efa_device_metrics(_COLLIDING_METRICS, "rdmap99s0") == {}


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_parse_efa_device_metrics_ignores_unattributable_samples() -> None:
    """Non-EFA series, unlabelled samples and junk values are skipped."""
    lines = [
        'node_cpu_seconds_total{cpu="0"} 1',  # not an EFA metric
        "node_amazonefa_rdma_read_bytes 7",  # no label block -> no device
        'node_amazonefa_rdma_read_bytes{port="1"} 8',  # labelled, but no device
        'node_amazonefa_tx_bytes{device="rdmap0s0",port="1"} not_a_number',
        'node_amazonefa_rx_bytes{device="rdmap0s0",port="1"} 9',
    ]
    assert parse_efa_device_metrics(lines, "rdmap0s0") == {
        "node_amazonefa_rx_bytes": 9.0
    }
