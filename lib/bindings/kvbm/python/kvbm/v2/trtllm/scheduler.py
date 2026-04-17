# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TRT-LLM KV-connector scheduler wired to the Rust ConnectorLeader.

Adapts TRT-LLM's ``KvCacheConnectorScheduler`` ABC to the Rust
``ConnectorLeader`` via ``KvbmConnectorLeaderBase``.

Bring-up flow:
1. ``__init__`` builds a ``KvbmRuntime`` (Velo messenger + tokio) and calls
   ``_init_connector(runtime, block_size)`` from the base class.
2. TRT-LLM calls ``set_xfer_handshake_metadata`` with each worker's
   ``VeloPeerMetadata``; the base class registers them as Velo peers
   (rank-ordered) and then calls ``initialize_workers()`` to drive the async
   layout-config gather and compute G2/G3 block counts from the config.

Key differences from the vLLM leader:
- ``request_id`` is ``int`` in TRT-LLM -> ``str()`` for Rust.
- ``resumed=False`` always (TRT-LLM re-admits preempted requests as new
  context requests).
- Decode offload is disabled for stage 1.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import kvbm
from kvbm.v2.connector.base import KvbmConnectorLeaderBase

from .sched_output import process_trtllm_scheduler_output

if kvbm.v2.is_available():
    KvbmRuntime = kvbm.v2.KvbmRuntime
    KvbmRequest = kvbm.v2.KvbmRequest
else:
    KvbmRuntime = None
    KvbmRequest = None

from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
    KvCacheConnectorScheduler,
    KVConnectorOutput,
    SchedulerOutput,
)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

logger = logging.getLogger(__name__)


class TrtllmConnectorScheduler(KvCacheConnectorScheduler, KvbmConnectorLeaderBase):
    """
    TRT-LLM KV-connector scheduler backed by the Rust ConnectorLeader.

    Implements the ``KvCacheConnectorScheduler`` ABC from TRT-LLM while
    delegating to ``KvbmConnectorLeaderBase`` for the Rust connector.
    """

    def __init__(self, llm_args: TorchLlmArgs):
        KvCacheConnectorScheduler.__init__(self, llm_args)
        KvbmConnectorLeaderBase.__init__(self)

        if not kvbm.v2.is_available():
            raise ImportError(
                "TrtllmConnectorScheduler requires the 'v2' feature. "
                "Rebuild kvbm with: maturin develop --features v2"
            )

        block_size = llm_args.kv_cache_config.tokens_per_block
        runtime = KvbmRuntime.build_leader(None)
        self._init_connector(runtime, block_size)

        instance_id = self._runtime.instance_id()
        logger.info(
            "TrtllmConnectorScheduler initialized with Velo instance: %s",
            instance_id.hex()[:8],
        )

    # ------------------------------------------------------------------
    # KvCacheConnectorScheduler ABC implementations
    # ------------------------------------------------------------------

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> bytes:
        self._iteration += 1
        # Decode offload: extend Rust slot token sequences with new decode tokens.
        #
        # TRT-LLM's SchedulerOutputManager computes new_tokens as the delta
        # since last iteration for each cached request. We apply these BEFORE
        # build_connector_metadata so the Rust slot can evaluate complete
        # decode blocks for offload.
        #
        # This is the TRT-LLM equivalent of vLLM's update_slot() which calls
        # extend_slot_tokens() before build_connector_metadata(). Unlike vLLM
        # where new_token_ids is empty during decode, TRT-LLM reliably
        # populates new_tokens via KvCacheConnectorSchedulerOutputRequest.
        if self.enable_decode_offload:
            for req in scheduler_output.cached_requests:
                if req.new_tokens:
                    self.extend_slot_tokens(str(req.request_id), req.new_tokens)

        output = process_trtllm_scheduler_output(self._iteration, scheduler_output)
        return bytes(self._connector.build_connector_metadata(output))

    def get_num_new_matched_tokens(
        self, request: LlmRequest, num_computed_tokens: int
    ) -> Tuple[int, bool]:
        self._create_slot(request)
        result = self._connector.get_num_new_matched_tokens(
            str(request.request_id), num_computed_tokens
        )
        # Rust returns (Optional[int], bool). TRT-LLM expects (int, bool).
        num_tokens = result[0] if result[0] is not None else 0
        return (num_tokens, result[1])

    def update_state_after_alloc(
        self,
        request: LlmRequest,
        block_ids: List[int],
        num_external_tokens: int,
    ) -> None:
        self._connector.update_state_after_alloc(
            str(request.request_id), block_ids, num_external_tokens
        )

    def request_finished(self, request: LlmRequest, cache_block_ids: List[int]) -> bool:
        return KvbmConnectorLeaderBase.request_finished(self, str(request.request_id))

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        finished_sending = {
            str(req_id) for req_id in (connector_output.finished_sending or [])
        }
        finished_recving = {
            str(req_id) for req_id in (connector_output.finished_recving or [])
        }
        KvbmConnectorLeaderBase.update_connector_output(
            self, finished_sending, finished_recving
        )

    def set_xfer_handshake_metadata(self, metadata: Dict[int, object]) -> None:
        KvbmConnectorLeaderBase.set_xfer_handshake_metadata(self, metadata)

    # ------------------------------------------------------------------
    # Abstract implementation from KvbmConnectorLeaderBase
    # ------------------------------------------------------------------

    def _create_slot(self, request: LlmRequest) -> None:
        req_id = str(request.request_id)

        if self._connector.has_slot(req_id):
            tokens = list(request.get_tokens(0))
            self._sync_slot_tokens(req_id, tokens)
            return

        all_token_ids = list(request.get_tokens(0))

        kv_request = KvbmRequest(
            request_id=req_id,
            tokens=all_token_ids,
            lora_name=None,
            salt_hash=None,
            max_tokens=request.max_new_tokens,
        )
        self._connector.create_slot(kv_request)
        self._inflight_requests[req_id] = request
