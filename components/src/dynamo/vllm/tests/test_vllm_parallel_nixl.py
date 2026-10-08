# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Backend imports follow importorskip so the CPU-only marker reporter can collect.

"""Real vLLM scheduling/serialization/notifications with device I/O replaced.

These CPU regressions do not replace a disaggregated GPU inference run.
"""

import pickle
from collections import defaultdict
from queue import Queue
from threading import Lock
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.factory")

import torch
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineTransferInfo,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    compute_tp_mapping,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from dynamo.vllm.kv_connector_protocols import (
    PARALLEL_CONSUMERS_KEY,
    NixlConnectorProtocol,
    configure_parallel_nixl_connector,
    make_kv_connector_protocol,
    parallel_decode_kv_params,
)
from dynamo.vllm.parallel_nixl_connector import NixlConnector, ParallelNixlWorker

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.fixture
def cache_config():
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=4, head_size=8, dtype=torch.float16
                ),
            )
        ],
    )


@pytest.fixture
def scheduler(cache_config):
    config = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="NixlConnector", kv_role="kv_consumer", engine_id="decode"
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            data_parallel_index=0,
        ),
        cache_config=SimpleNamespace(block_size=16, mamba_cache_mode="none"),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )
    configure_parallel_nixl_connector(config.kv_transfer_config)
    connector = KVConnectorFactory.create_connector(
        config, KVConnectorRole.SCHEDULER, cache_config
    )
    yield connector.connector_scheduler
    connector.shutdown()


def _children(n):
    params = SamplingParams(n=n, max_tokens=2)
    params.extra_args = {
        "kv_transfer_params": parallel_decode_kv_params(
            {
                "do_remote_prefill": True,
                "do_remote_decode": False,
                "remote_request_id": "shared-prefill",
                "remote_engine_id": "prefill",
                "remote_host": "localhost",
                "remote_port": 1234,
                "remote_block_ids": ([1],),
                "tp_size": 1,
            },
            n,
        )
    }
    core_request = EngineCoreRequest(
        request_id="decode-request",
        external_req_id="request",
        prompt_token_ids=list(range(16)),
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    parent = ParentRequest(core_request)
    children = []
    for index in range(n):
        child_id, child_params = parent.get_child_info(index)
        child = pickle.loads(pickle.dumps(core_request))
        child.request_id, child.sampling_params = child_id, child_params
        # AsyncLLM sends each child over this codec independently, so cached
        # child SamplingParams must not alias mutable KV params in EngineCore.
        child = MsgpackDecoder(EngineCoreRequest).decode(MsgpackEncoder().encode(child))
        children.append(Request.from_engine_core_request(child, block_hasher=None))
    return children


def _metadata(scheduler, request, mode):
    if mode == "cancel":
        request.status = RequestStatus.FINISHED_ABORTED
        scheduler.request_finished(request, [])
    else:
        blocks = KVCacheBlocks(blocks=((KVCacheBlock(block_id=2),),))
        scheduler.update_state_after_alloc(request, blocks, 16 if mode == "read" else 0)
    # Multiproc executor pickles SchedulerOutput, including connector metadata.
    return pickle.loads(pickle.dumps(scheduler.build_connector_meta(None)))


class _NixlIO:
    """Hold READ notifications until explicitly completing device I/O."""

    def __init__(self):
        self.pending = {}
        self.notifications = []
        self.next_handle = 0

    def make_prepped_xfer(self, *args, notif_msg):
        self.next_handle += 1
        self.pending[self.next_handle] = notif_msg
        return self.next_handle

    def transfer(self, handle):
        assert handle in self.pending

    def send_notif(self, agent, notif_msg):
        self.notifications.append(notif_msg)

    def complete(self):
        self.notifications.extend(self.pending.values())
        self.pending.clear()

    def get_new_notifs(self):
        notifications, self.notifications = self.notifications, []
        return {"decode": notifications}


def _worker(cache_config, io, *, tp_size=1, dcp_size=1, mla=False, remote_tp=1):
    worker = object.__new__(ParallelNixlWorker)
    worker.engine_id = "decode"
    worker.kv_cache_config = cache_config
    worker._physical_blocks_per_logical_kv_block = 1
    worker._bidirectional_kv_xfer_enabled = False
    worker._engine_last_active = {}
    worker._engine_ttl = 0
    worker.dcp_size, worker.dcp_rank = dcp_size, 0
    worker.use_mla, worker._has_mamba = mla, False
    worker._mixed_mem_types = False
    worker._uses_region_group_mapping = False
    worker._transfer_layer_group_ids = []
    worker._group_spec_types = (FullAttentionSpec,)
    worker.num_regions = 1
    worker.block_len_per_layer = [128]
    worker.region_group_ids = [0]
    worker.dst_region_group_ids = {"prefill": [0]}
    worker.dst_region_num_blocks = {}
    worker.dst_uses_region_group_mapping = {"prefill": False}
    worker.dst_num_blocks = {"prefill": 16, "decode": 16}
    worker.src_xfer_handles_by_block_size = {16: 1}
    worker.dst_xfer_side_handles = {"prefill": {0: 2}}
    worker._remote_agents = {
        "prefill": {(0, rank): f"p{rank}" for rank in range(remote_tp)}
    }
    worker._hb_handshake_notif_only = False
    worker._handshake_lock = Lock()
    worker._recving_transfers = {}
    worker._recving_metadata = {}
    worker._ready_requests = Queue()
    worker._reqs_to_process = set()
    worker._reqs_to_send = {}
    worker.pcp_rank, worker.pcp_dcp_sharded = 0, False
    worker.nixl_wrapper = io
    worker.transfer_topo = TransferTopology(
        tp_rank=0,
        tp_size=tp_size,
        block_size=16,
        engine_id="decode",
        is_mla=mla,
        is_mamba=False,
        total_num_kv_heads=4,
        attn_backends=[],
        dcp_size=dcp_size,
    )
    worker.transfer_topo.register_remote_engine(
        "prefill",
        EngineTransferInfo(
            remote_tp_size=remote_tp,
            remote_block_len=128,
            remote_block_size=16,
            remote_physical_blocks_per_logical=1,
        ),
    )
    worker.tp_mappings = {
        "prefill": compute_tp_mapping(
            worker.transfer_topo, remote_tp, (FullAttentionSpec,)
        )
    }
    return worker


def _producer(io):
    producer = object.__new__(NixlPullConnectorWorker)
    producer.transfer_topo = object()
    producer.nixl_wrapper = io
    producer._reqs_to_send = {"shared-prefill": 30.0}
    producer._reqs_to_process = {"shared-prefill"}
    producer.consumer_notification_counts_by_req = defaultdict(int)
    producer.expected_consumer_notifications_by_req = {}
    producer._lease_extension = 30
    producer.xfer_stats = MagicMock()
    return producer


@pytest.mark.parametrize("mode", ["read", "cache_hit", "cancel"])
@pytest.mark.parametrize("n", [1, 3])
def test_shared_lease_waits_for_every_choice(scheduler, cache_config, mode, n):
    children = _children(n)
    for child in children:
        scheduler.on_new_request(child)
    io = _NixlIO()
    worker, producer = _worker(cache_config, io), _producer(io)
    original_plan = worker.tp_mappings["prefill"]
    for index, child in enumerate(children):
        metadata = _metadata(scheduler, child, mode)
        metadata.heartbeat_by_engine = {}  # Separate test controls renewal time.
        worker.start_load_kv(metadata)
        assert worker.tp_mappings["prefill"] is original_plan
        if mode == "read":
            assert producer._get_new_notifs() == set()
            assert len(io.pending) == 1
            io.complete()
        released = producer._get_new_notifs()
        assert released == ({"shared-prefill"} if index == n - 1 else set())
        assert ("shared-prefill" in producer._reqs_to_send) == (index < n - 1)


def test_queued_siblings_keep_lease_alive_after_first_completion(
    scheduler, cache_config, monkeypatch
):
    children = _children(3)
    for child in children:
        scheduler.on_new_request(child)
    # Only the first child received KV; siblings have no allocation yet.
    scheduler.update_connector_output(
        KVConnectorOutput(finished_recving={children[0].request_id})
    )
    clock = [20.0]
    monkeypatch.setattr(base_worker.time, "perf_counter", lambda: clock[0])
    producer = _producer(_NixlIO())
    worker = _worker(cache_config, producer.nixl_wrapper)
    for now in (20.0, 40.0, 60.0):
        clock[0] = now
        metadata = pickle.loads(pickle.dumps(scheduler.build_connector_meta(None)))
        worker.start_load_kv(metadata)
        assert producer._get_new_notifs() == set()
        expired = set()
        producer._reap_expired_send_leases(expired)
        assert expired == set()
        assert producer._reqs_to_send["shared-prefill"] > now
    for child in children[1:]:
        child.status = RequestStatus.FINISHED_ABORTED
        scheduler.request_finished(child, [])
    assert not scheduler._heartbeat_by_engine
    assert not scheduler._heartbeat_req_engine


@pytest.mark.parametrize(
    "tp_size,dcp_size,mla,remote_tp",
    [(4, 1, False, 1), (4, 4, True, 1), (1, 1, True, 4)],
)
def test_topology_and_mla_notifications_include_logical_choices(
    scheduler, cache_config, tp_size, dcp_size, mla, remote_tp
):
    child = _children(3)[0]
    scheduler.on_new_request(child)
    metadata = _metadata(scheduler, child, "cancel")
    metadata.heartbeat_by_engine = {}
    io = _NixlIO()
    worker = _worker(
        cache_config,
        io,
        tp_size=tp_size,
        dcp_size=dcp_size,
        mla=mla,
        remote_tp=remote_tp,
    )
    plan = worker.tp_mappings["prefill"]
    worker.start_load_kv(metadata)
    assert io.notifications
    assert set(io.notifications) == {
        f"shared-prefill:{3 * plan.local_consumers}".encode()
    }
    assert len(io.notifications) == remote_tp
    assert worker.tp_mappings["prefill"] is plan


@pytest.mark.parametrize("wrapper", [None, "MultiConnector", "PdConnector"])
def test_parallel_nixl_config_reaches_vllm_factory(wrapper):
    child = {"kv_connector": "NixlConnector", "kv_role": "kv_consumer"}
    kv_config = KVTransferConfig(
        **(
            child
            if wrapper is None
            else {
                "kv_connector": wrapper,
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {"connectors": [child]},
            }
        )
    )
    configure_parallel_nixl_connector(kv_config)
    resolved = (
        kv_config
        if wrapper is None
        else KVTransferConfig(**kv_config.kv_connector_extra_config["connectors"][0])
    )
    assert KVConnectorFactory.get_connector_class(resolved) is NixlConnector
    assert isinstance(
        make_kv_connector_protocol(SimpleNamespace(kv_transfer_config=kv_config)),
        NixlConnectorProtocol,
    )


def test_parallel_nixl_config_preserves_custom_connector():
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="NixlConnector", kv_connector_module_path="user.connector"
        )
    )
    configure_parallel_nixl_connector(config.kv_transfer_config)
    assert config.kv_transfer_config.kv_connector_module_path == "user.connector"


def test_parallel_decode_params_keep_original_handoff():
    handoff = {"do_remote_prefill": True, "remote_request_id": "shared"}
    decoded = parallel_decode_kv_params(handoff, 3)
    assert decoded[PARALLEL_CONSUMERS_KEY] == 3
    assert handoff == {"do_remote_prefill": True, "remote_request_id": "shared"}
