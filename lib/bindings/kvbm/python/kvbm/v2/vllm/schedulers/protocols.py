# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Protocol definitions for vLLM scheduler output types.

These Protocols define the interface that vLLM expects from scheduler outputs.
By defining them as Protocols, we can:
1. Use vLLM's dataclasses directly when convenient
2. Implement our own classes (Python or Rust PyO3) that conform to the same interface
3. See version differences as explicit Protocol changes

Based on vLLM v0.11+ SchedulerOutput structure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Set, Tuple, runtime_checkable


@runtime_checkable
class NewRequestDataProtocol(Protocol):
    """
    Protocol matching vLLM's NewRequestData.

    Represents a request being scheduled for the first time.
    The worker processes will cache this data.
    """

    req_id: str
    prompt_token_ids: List[int] | None
    mm_features: List[Any]  # List[MultiModalFeatureSpec]
    sampling_params: Any | None  # SamplingParams | None
    pooling_params: Any | None  # PoolingParams | None
    block_ids: Tuple[List[int], ...]
    num_computed_tokens: int
    lora_request: Any | None  # LoRARequest | None
    prompt_embeds: Any | None  # torch.Tensor | None


@runtime_checkable
class CachedRequestDataProtocol(Protocol):
    """
    Protocol matching vLLM's CachedRequestData.

    Represents requests that have been scheduled before.
    Only the diff is sent to minimize communication cost.
    """

    req_ids: List[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended.
    # For those in the set, new_block_ids replaces the existing block IDs.
    resumed_req_ids: Set[str]
    # Only used for pipeline parallelism; empty when PP is not used.
    new_token_ids: List[List[int]]
    # For requests not scheduled in the last step, propagate token ids.
    all_token_ids: Dict[str, List[int]]
    new_block_ids: List[Tuple[List[int], ...] | None]
    num_computed_tokens: List[int]
    num_output_tokens: List[int]

    @property
    def num_reqs(self) -> int:
        """Number of cached requests."""
        ...


@runtime_checkable
class SchedulerOutputProtocol(Protocol):
    """
    Protocol matching vLLM's SchedulerOutput.

    Contains all scheduling decisions for a single step.
    """

    # Requests being scheduled for the first time
    scheduled_new_reqs: List[NewRequestDataProtocol]
    # Requests that have been scheduled before (only diff sent)
    scheduled_cached_reqs: CachedRequestDataProtocol

    # req_id -> num_scheduled_tokens
    num_scheduled_tokens: Dict[str, int]
    # Total tokens scheduled (sum of num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids (only for requests with spec decode tokens)
    scheduled_spec_decode_tokens: Dict[str, List[int]]
    # req_id -> encoder input indices to process
    scheduled_encoder_inputs: Dict[str, List[int]]
    # Common prefix blocks per KV cache group (for cascade attention)
    num_common_prefix_blocks: List[int]

    # Finished request IDs (to notify workers to free cached states)
    finished_req_ids: Set[str]
    # mm_hash strings for encoder outputs to free from cache
    free_encoder_mm_hashes: List[str]

    # Whether scheduled requests have all output tokens for grammar bitmask
    pending_structured_output_tokens: bool

    # KV Cache Connector metadata
    kv_connector_metadata: Any | None
    # EC Cache Connector metadata
    ec_connector_metadata: Any | None
