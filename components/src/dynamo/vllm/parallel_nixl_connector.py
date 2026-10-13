# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep a shared prefill lease until every parallel decode choice has read it."""

from dataclasses import dataclass, replace

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqMeta
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)

from dynamo.vllm.kv_connector_protocols import PARALLEL_CONSUMERS_KEY


@dataclass
class ParallelReqMeta(ReqMeta):
    """The logical fan-out survives scheduler-to-worker serialization."""

    parallel_consumers: int = 1


class ParallelNixlScheduler(NixlPullConnectorScheduler):
    def build_connector_meta(self, scheduler_output):
        # Capture before the upstream builder drains _reqs_need_recv. Each
        # child has n=1, but retains the parent's original count in KV params.
        consumers = {
            req_id: pending[0].kv_transfer_params.get(PARALLEL_CONSUMERS_KEY, 1)
            for req_id, pending in self._reqs_need_recv.items()
        }
        metadata = super().build_connector_meta(scheduler_output)
        for req_id, count in consumers.items():
            metadata.reqs_to_recv[req_id] = ParallelReqMeta(
                **vars(metadata.reqs_to_recv[req_id]), parallel_consumers=count
            )
        return metadata

    def _stop_heartbeat(self, req_id):
        key = self._heartbeat_req_engine.get(req_id)
        if key is not None and any(
            other_id != req_id and other_key == key
            for other_id, other_key in self._heartbeat_req_engine.items()
        ):
            # on_new_request registers *all admitted children*, including
            # those still queued and not yet allocated any blocks. Completing
            # or cancelling one child must not stop its siblings' lease.
            del self._heartbeat_req_engine[req_id]
            return
        super()._stop_heartbeat(req_id)


class ParallelNixlWorker(NixlPullConnectorWorker):
    def _read_blocks_for_req(self, req_id, meta):
        count = getattr(meta, "parallel_consumers", 1)
        if count == 1:
            return super()._read_blocks_for_req(req_id, meta)
        assert meta.remote is not None
        engine_id = meta.remote.engine_id
        plan = self.tp_mappings[engine_id]
        # vLLM dispatches this synchronous method on the worker's main thread,
        # after handshake completion. Scope the immutable plan to this call:
        # both READ completion notifications and notify-only paths (cache
        # hits, aborted children, MLA replicas) use the same total count.
        self.tp_mappings[engine_id] = replace(
            plan, local_consumers=plan.local_consumers * count
        )
        try:
            return super()._read_blocks_for_req(req_id, meta)
        finally:
            self.tp_mappings[engine_id] = plan


class NixlConnector(NixlPullConnector):
    """Dynamo's pull connector with logical-choice lifetime accounting."""

    def __init__(self, vllm_config, role, kv_cache_config):
        # Initialize the facade once, then instantiate the specialized roles.
        # Calling NixlPullConnector.__init__ would allocate two NIXL workers.
        NixlBaseConnector.__init__(self, vllm_config, role, kv_cache_config)
        self.connector_scheduler = None
        self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = ParallelNixlScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = ParallelNixlWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
