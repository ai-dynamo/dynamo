# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`dynamo.vllm.nixl_push`.

Covers which engines advertise NIXL push coordinates and what they publish.
The address computation is pinned against vLLM's own derivation
(``NixlBaseConnectorScheduler.__init__``): host straight from the env var,
port offset by the data-parallel index. No engine is started -- ``vllm_config``
is a ``SimpleNamespace`` and the env vars are monkeypatched.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamo.llm import WorkerType
from dynamo.vllm.nixl_push import publish_nixl_push_endpoint

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _vllm_config(
    connector="NixlPushConnector",
    engine_id="prefill-engine-001",
    extra_config=None,
    data_parallel_index=0,
    data_parallel_size=1,
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
):
    kv_transfer_config = (
        None
        if connector is None
        else SimpleNamespace(
            kv_connector=connector,
            engine_id=engine_id,
            kv_connector_extra_config=extra_config,
        )
    )
    return SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        parallel_config=SimpleNamespace(
            data_parallel_index=data_parallel_index,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
    )


@pytest.fixture
def side_channel(monkeypatch):
    """Set the NIXL side-channel env vars vLLM reads lazily."""
    monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_HOST", "10.0.0.1")
    monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600")


def test_publishes_engine_id_and_side_channel_for_push_prefill(side_channel):
    runtime_config = MagicMock()

    assert (
        publish_nixl_push_endpoint(
            runtime_config, _vllm_config(), WorkerType.Prefill, (0, 1)
        )
        is True
    )
    runtime_config.set_nixl_push_endpoint.assert_called_once_with(
        "prefill-engine-001", "10.0.0.1", 5600, 4, 2
    )


def test_port_is_offset_by_data_parallel_index(side_channel):
    """vLLM gives each DP rank its own side channel at base + index; publishing
    the base for a non-zero rank would point decode at the wrong engine."""
    runtime_config = MagicMock()

    publish_nixl_push_endpoint(
        runtime_config,
        _vllm_config(data_parallel_index=3),
        WorkerType.Prefill,
        (3, 1),
    )

    _, _, port, _, _ = runtime_config.set_nixl_push_endpoint.call_args.args
    assert port == 5603


def test_resolves_push_connector_nested_in_pd_connector(side_channel):
    """KVBM composition wraps the PD connector; the child's engine_id is the
    one the peer must name, not the wrapper's."""
    runtime_config = MagicMock()

    publish_nixl_push_endpoint(
        runtime_config,
        _vllm_config(
            connector="PdConnector",
            engine_id="wrapper-engine",
            extra_config={
                "connectors": [
                    {"kv_connector": "DynamoConnector"},
                    {
                        "kv_connector": "NixlPushConnector",
                        "engine_id": "child-engine",
                    },
                ]
            },
        ),
        WorkerType.Prefill,
        (0, 1),
    )

    engine_id = runtime_config.set_nixl_push_endpoint.call_args.args[0]
    assert engine_id == "child-engine"


def test_nested_child_engine_id_falls_back_to_wrapper(side_channel):
    """vLLM's MultiConnector hands children the wrapper's engine_id when the
    child entry does not override it."""
    runtime_config = MagicMock()

    publish_nixl_push_endpoint(
        runtime_config,
        _vllm_config(
            connector="PdConnector",
            engine_id="wrapper-engine",
            extra_config={"connectors": [{"kv_connector": "NixlPushConnector"}]},
        ),
        WorkerType.Prefill,
        (0, 1),
    )

    engine_id = runtime_config.set_nixl_push_endpoint.call_args.args[0]
    assert engine_id == "wrapper-engine"


@pytest.mark.parametrize(
    "worker_type",
    [WorkerType.Decode, WorkerType.Aggregated],
)
def test_only_prefill_workers_advertise(side_channel, worker_type):
    """Only the prefill side of a push transfer gets named by its peer."""
    runtime_config = MagicMock()

    assert (
        publish_nixl_push_endpoint(runtime_config, _vllm_config(), worker_type, (0, 1))
        is False
    )
    runtime_config.set_nixl_push_endpoint.assert_not_called()


@pytest.mark.parametrize(
    "connector",
    [None, "NixlConnector", "LMCacheConnectorV1", "DynamoConnector"],
)
def test_non_push_engines_do_not_advertise(side_channel, connector):
    """Pull mode keeps these coordinates private, and connectors with no PD
    protocol at all must not trip an error during registration."""
    runtime_config = MagicMock()

    assert (
        publish_nixl_push_endpoint(
            runtime_config,
            _vllm_config(connector=connector),
            WorkerType.Prefill,
            (0, 1),
        )
        is False
    )
    runtime_config.set_nixl_push_endpoint.assert_not_called()


def test_declines_when_worker_fronts_multiple_dp_ranks(side_channel, caplog):
    """Each DP rank has its own side channel, so one advertised port would be
    wrong for every rank but the first. Sequential handoff still works."""
    runtime_config = MagicMock()

    assert (
        publish_nixl_push_endpoint(
            runtime_config, _vllm_config(), WorkerType.Prefill, (0, 4)
        )
        is False
    )
    runtime_config.set_nixl_push_endpoint.assert_not_called()
    assert "data-parallel" in caplog.text


def test_declines_when_side_channel_host_is_unset(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_HOST", "")
    runtime_config = MagicMock()

    assert (
        publish_nixl_push_endpoint(
            runtime_config, _vllm_config(), WorkerType.Prefill, (0, 1)
        )
        is False
    )
    runtime_config.set_nixl_push_endpoint.assert_not_called()


def test_declines_when_engine_id_is_missing(side_channel, caplog):
    """Without an engine_id the decode side has nothing to address the
    registration to, so publishing coordinates would be worse than silence."""
    runtime_config = MagicMock()

    assert (
        publish_nixl_push_endpoint(
            runtime_config,
            _vllm_config(engine_id=None),
            WorkerType.Prefill,
            (0, 1),
        )
        is False
    )
    runtime_config.set_nixl_push_endpoint.assert_not_called()
    assert "engine_id" in caplog.text


def test_dense_engine_advertises_the_unsuffixed_engine_id(side_channel):
    """TP/TEP with a single DP rank keeps vLLM's base engine ID.

    vLLM only rewrites ``engine_id`` when ``data_parallel_size > 1 or
    dp_rank > 0`` (``EngineCoreProc.run_engine_core``). Advertising ``_dp0``
    for a dense TEP engine names an agent that does not exist, and every push
    handshake is rejected with "Remote NIXL agent engine ID mismatch".
    """
    runtime_config = MagicMock()

    publish_nixl_push_endpoint(
        runtime_config,
        _vllm_config(data_parallel_size=1, data_parallel_index=0),
        WorkerType.Prefill,
        (0, 1),
    )

    engine_id = runtime_config.set_nixl_push_endpoint.call_args.args[0]
    assert engine_id == "prefill-engine-001"


def test_data_parallel_engine_advertises_the_dp_suffixed_engine_id(side_channel):
    """Each DP rank gets its own NIXL agent named ``<base>_dp<global_rank>``."""
    runtime_config = MagicMock()

    publish_nixl_push_endpoint(
        runtime_config,
        _vllm_config(data_parallel_size=4, data_parallel_index=2),
        WorkerType.Prefill,
        (2, 1),
    )

    engine_id, _, port, _, _ = runtime_config.set_nixl_push_endpoint.call_args.args
    assert engine_id == "prefill-engine-001_dp2"
    assert port == 5602


def test_nonzero_dp_index_is_suffixed_even_when_size_is_one(side_channel):
    """External load balancing can hand a worker rank N with a local size of
    one; vLLM still suffixes, because its guard is ``size > 1 or rank > 0``."""
    runtime_config = MagicMock()

    publish_nixl_push_endpoint(
        runtime_config,
        _vllm_config(data_parallel_size=1, data_parallel_index=3),
        WorkerType.Prefill,
        (3, 1),
    )

    engine_id = runtime_config.set_nixl_push_endpoint.call_args.args[0]
    assert engine_id == "prefill-engine-001_dp3"


def test_declines_when_dynamo_range_disagrees_with_vllm_rank(side_channel, caplog):
    """The published identity is derived from vLLM's rank, so if Dynamo's
    assigned range starts somewhere else one of the two is wrong. Publishing
    anyway would name a nonexistent agent and fail the handshake silently."""
    runtime_config = MagicMock()

    assert (
        publish_nixl_push_endpoint(
            runtime_config,
            _vllm_config(data_parallel_size=4, data_parallel_index=2),
            WorkerType.Prefill,
            (1, 1),
        )
        is False
    )
    runtime_config.set_nixl_push_endpoint.assert_not_called()
    assert "data_parallel_index" in caplog.text
