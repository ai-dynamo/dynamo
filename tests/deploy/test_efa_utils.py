# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the EFA deploy-test helpers (tests/deploy/efa_utils.py)."""

import re
from unittest.mock import MagicMock

import httpx
import kr8s
import pytest

from tests.deploy.efa_utils import (
    _read_pod_logs,
    parse_efa_device_metrics,
    runtime_version_override,
    sum_nixl_agent_samples,
)

# Real exporter output, trimmed. Device names are nested on purpose: rdmap16s2
# is a prefix of rdmap16s27, and rdmap16s27 of rdmap16s270. A substring match on
# the device name -- which is what this parser used to do -- attributes the wrong
# NIC's counters to the worker, and because the result is keyed only by metric
# name the last match silently wins.
_COLLIDING_METRICS = tuple(
    """\
# HELP node_amazonefa_rdma_read_bytes The number of bytes read with RDMA
# TYPE node_amazonefa_rdma_read_bytes gauge
node_amazonefa_rdma_read_bytes{device="rdmap16s2",port="1"} 100
node_amazonefa_rdma_read_bytes{device="rdmap16s27",port="1"} 200
node_amazonefa_rdma_read_bytes{device="rdmap16s270",port="1"} 300
node_amazonefa_rdma_read_resp_bytes{device="rdmap16s27",port="1"} 400
node_amazonefa_tx_bytes{device="rdmap16s270",port="1"} 500
node_cpu_seconds_total{cpu="0",mode="idle"} 999
""".splitlines()
)

# NIXL publishes one sample per agent, and the gate that consumes this only asks
# whether the total grew. A sibling series that grows on its own -- a per-backend
# split, or a histogram's _bucket/_sum -- would therefore be enough to satisfy it
# on evidence from somewhere other than the counter under test.
_SIBLING_SERIES = """\
# HELP agent_tx_bytes Bytes sent by this agent
# TYPE agent_tx_bytes counter
agent_tx_bytes{agent="prefill-0"} 100
agent_tx_bytes{agent="prefill-1"} 200
agent_tx_bytes_by_backend{agent="prefill-0",backend="UCX"} 900
agent_tx_bytes_bucket{agent="prefill-0",le="1024"} 700
agent_rx_bytes{agent="prefill-0"} 50
"""


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


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_sum_nixl_agent_samples_matches_metric_name_exactly() -> None:
    """Every agent's sample is summed, and only that counter's own series."""
    assert sum_nixl_agent_samples(_SIBLING_SERIES, "agent_tx_bytes") == ("ok", 300.0)
    assert sum_nixl_agent_samples(_SIBLING_SERIES, "agent_rx_bytes") == ("ok", 50.0)


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_sum_nixl_agent_samples_accepts_total_suffix() -> None:
    """NIXL spells these counters <name>_total in some versions."""
    renamed = 'agent_tx_bytes_total{agent="prefill-0"} 42\n'
    assert sum_nixl_agent_samples(renamed, "agent_tx_bytes") == ("ok", 42.0)


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_sum_nixl_agent_samples_reports_absent() -> None:
    """A counter that is missing or unparseable fails closed in the caller."""
    assert sum_nixl_agent_samples(_SIBLING_SERIES, "agent_read_bytes") == (
        "absent",
        None,
    )
    junk = 'agent_tx_bytes{agent="prefill-0"} not_a_number\n'
    assert sum_nixl_agent_samples(junk, "agent_tx_bytes") == ("absent", None)


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "image",
    [
        "registry.example/ai-dynamo/dynamo:1.6.0-ci-0123abc-vllm-runtime-nightly",
        "registry.example/ai-dynamo/vllm-runtime:1.2.0-efa",
        "registry.example/ai-dynamo/dynamo:v1.5.1",
        "registry.example/ai-dynamo/dynamo:V1.5.1",
        "registry.example/ai-dynamo/dynamo:0.0.0-0.alpha-1.2+build.01",
        "registry.example:5000/ai-dynamo/dynamo:1.6.0+build.01",
        " registry.example/ai-dynamo/dynamo: v1.6.0 ",
        "registry.example/ai-dynamo/dynamo:18446744073709551615.0.0",
        "registry.example/ai-dynamo/dynamo:1.5.1@sha256:" + "0" * 64,
    ],
)
def test_runtime_version_override_leaves_versioned_tags_alone(image: str) -> None:
    assert runtime_version_override(image) is None


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "image",
    [
        "registry.example/ai-dynamo/dynamo:0123abc-vllm-runtime-efa-nightly",
        "registry.example:5000/ai-dynamo/dynamo",
        "registry.example/ai-dynamo/dynamo@sha256:" + "0" * 64,
        "",
        "1.6.0",
        "registry.example/ai-dynamo/dynamo:01.2.3",
        "registry.example/ai-dynamo/dynamo:1.02.3",
        "registry.example/ai-dynamo/dynamo:1.2.03",
        "registry.example/ai-dynamo/dynamo:1.6.0-bad_suffix",
        "registry.example/ai-dynamo/dynamo:1.6.0-01",
        "registry.example/ai-dynamo/dynamo:1.6.0-alpha.01",
        "registry.example/ai-dynamo/dynamo:1.6.0-alpha..1",
        "registry.example/ai-dynamo/dynamo:1.6.0-",
        "registry.example/ai-dynamo/dynamo:1.6.0+",
        "registry.example/ai-dynamo/dynamo:1.6.0+bad_suffix",
        "registry.example/ai-dynamo/dynamo:1.6.0+build..1",
        "registry.example/ai-dynamo/dynamo:1.6.0junk",
        "registry.example/ai-dynamo/dynamo:1.6.0-alpha\njunk",
        "registry.example/ai-dynamo/dynamo:１.6.0",
        "registry.example/ai-dynamo/dynamo:18446744073709551616.0.0",
        "registry.example:1.6.0/ai-dynamo/dynamo",
        "registry.example/ai-dynamo/dynamo:01.2.3@sha256:" + "0" * 64,
    ],
)
def test_runtime_version_override_for_unversioned_tags(image: str) -> None:
    version = runtime_version_override(image)
    assert version is not None
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)


_LIBFABRIC_LOG = ["nixl: Backend LIBFABRIC was instantiated", "efa_mr_reg ok"]


def _fake_pod(current_reads: list, previous_read: BaseException) -> MagicMock:
    """A one-container pod whose previous instance is missing.

    Each current-log read takes the next item of ``current_reads``: an exception
    is raised, a list is returned as the log lines.
    """
    pod = MagicMock()
    pod.name = "sglang-disagg-efa-0-decode-abcde"
    pod.raw = {"spec": {"containers": [{"name": "main"}]}}
    reads = iter(current_reads)

    def logs(*, previous: bool, **_kwargs):
        result = previous_read if previous else next(reads)
        if isinstance(result, BaseException):
            raise result
        return iter(result)

    pod.logs = MagicMock(side_effect=logs)
    return pod


_NO_PREVIOUS = kr8s.ServerError("previous terminated container not found")


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_read_pod_logs_recovers_from_transient_tunnel_drop(monkeypatch) -> None:
    """A log readable after a tunnel recovery is scored, not replaced by an empty log."""
    sleep = MagicMock()
    monkeypatch.setattr("tests.deploy.vcluster_utils.time.sleep", sleep)
    pod = _fake_pod(
        [httpx.ConnectError("All connection attempts failed"), _LIBFABRIC_LOG],
        _NO_PREVIOUS,
    )

    assert "Backend LIBFABRIC was instantiated" in _read_pod_logs(pod)
    sleep.assert_called_once_with(5)


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_read_pod_logs_raises_when_tunnel_stays_down(monkeypatch) -> None:
    """A log that cannot be read fails the read, not the LIBFABRIC check."""
    monkeypatch.setattr("tests.deploy.vcluster_utils.time.sleep", MagicMock())
    pod = _fake_pod(
        [httpx.ConnectError("All connection attempts failed")] * 4, _NO_PREVIOUS
    )

    with pytest.raises(httpx.ConnectError):
        _read_pod_logs(pod)
