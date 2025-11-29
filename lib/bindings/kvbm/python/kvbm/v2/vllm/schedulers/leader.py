# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal scheduler connector leader implementation for testing.

This is a barebones implementation that returns minimal/no-op responses,
used specifically for scheduler integration testing without actual KV transfer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from kvbm._core.v2 import KvbmRequest
from kvbm._core.v2 import KvConnectorLeader as RustKvConnectorLeader
from kvbm._core.v2 import RustSchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class SchedulerConnectorLeader:
    """
    Minimal scheduler connector leader that returns no-op responses.

    This connector is used for scheduler integration where no actual
    KV transfer is needed. All methods return minimal valid responses.
    """

    def __init__(self, vllm_config: "VllmConfig", engine_id: str, **kwargs):
        """Initialize the scheduler connector leader."""
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self._connector = RustKvConnectorLeader(engine_id)
        print(f"SchedulerConnectorLeader initialized with engine_id: {engine_id}")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """
        Always returns (0, False) indicating no external tokens available.

        Returns:
            (0, False): No external tokens, no async loading
        """
        self._create_slot(request)
        total_tokens = getattr(request, "num_tokens", 0)
        return self._connector.get_num_new_matched_tokens(
            request.request_id,
            total_tokens,
            num_computed_tokens,
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """
        No-op since we never have external tokens.

        This should never be called with num_external_tokens > 0.
        """
        self._create_slot(request)
        block_ids = [int(block_id) for block_id in blocks.get_block_ids()[0]]
        self._connector.update_state_after_alloc(
            request.request_id,
            block_ids,
            num_external_tokens,
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> bytes:
        """
        Build minimal connector metadata.

        Returns:
            Empty bytes object
        """
        output = RustSchedulerOutput()

        for req in scheduler_output.scheduled_new_reqs:
            prompt_ids = [int(token) for token in req.prompt_token_ids]
            block_ids = [int(block_id) for block_id in req.block_ids[0]]
            output.add_new_request(
                req.req_id,
                prompt_ids,
                block_ids,
                int(req.num_computed_tokens),
            )

        cached = scheduler_output.scheduled_cached_reqs
        if cached is not None:
            for (
                req_id,
                resumed_from_preemption,
                new_token_ids,
                new_block_ids,
                num_computed_tokens,
            ) in zip(
                cached.req_ids,
                cached.resumed_from_preemption,
                cached.new_token_ids,
                cached.new_block_ids,
                cached.num_computed_tokens,
            ):
                tokens = (
                    [int(token) for token in new_token_ids] if new_token_ids else []
                )
                block_ids = (
                    [int(block_id) for block_id in new_block_ids[0]]
                    if new_block_ids is not None and len(new_block_ids) > 0
                    else []
                )
                output.add_cached_request(
                    req_id,
                    bool(resumed_from_preemption),
                    tokens,
                    block_ids,
                    int(num_computed_tokens),
                )

        counts_source = getattr(scheduler_output, "num_scheduled_tokens", None)
        if counts_source:
            counts = {
                str(req_id): int(value) for req_id, value in counts_source.items()
            }
            output.set_num_scheduled_tokens(counts)

        return bytes(self._connector.build_connector_metadata(output))

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Never delays block freeing.

        Returns:
            (False, None): Don't delay block freeing, no KV transfer params
        """
        assert self._connector.has_slot(
            request.request_id
        ), f"Slot not found for request {request.request_id}"
        delay = self._connector.request_finished(request.request_id)
        return (delay, None)

    def update_connector_output(self, connector_output) -> None:
        _ = connector_output

    def get_finished_count(self) -> Optional[int]:
        return None

    # Utility functions

    def _create_slot(self, request: "Request") -> None:
        if self._connector.has_slot(request.request_id):
            return

        if bool(getattr(request, "mm_features", None)) or bool(
            getattr(request, "mm_positions", None)
        ):
            raise ValueError("Unsupported request - requires mm extra keys")

        all_token_ids = [
            [int(token) for token in tokens] for tokens in request.all_token_ids
        ]

        kv_request = KvbmRequest(
            request_id=request.request_id,
            lora_name=request.lora_request.lora_name()
            if request.lora_request
            else None,
            salt_hash=str(getattr(request, "cache_salt", None))
            if getattr(request, "cache_salt", None) is not None
            else None,
        )

        self._connector.create_slot(kv_request, all_token_ids)
