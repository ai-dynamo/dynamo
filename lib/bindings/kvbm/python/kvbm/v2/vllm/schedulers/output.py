# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler output implementations conforming to Protocol definitions.

These dataclasses implement the SchedulerOutputProtocol and related protocols,
allowing us to construct scheduler outputs from Rust scheduler results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


@dataclass
class RustNewRequestData:
    """
    Our implementation of NewRequestDataProtocol.

    Conforms to the same interface as vLLM's NewRequestData.
    """

    req_id: str
    prompt_token_ids: List[int] | None
    block_ids: Tuple[List[int], ...]
    num_computed_tokens: int
    # Fields we get from stored Request objects
    mm_features: List[Any] = field(default_factory=list)
    sampling_params: Any | None = None
    pooling_params: Any | None = None
    lora_request: Any | None = None
    prompt_embeds: Any | None = None


@dataclass
class RustCachedRequestData:
    """
    Our implementation of CachedRequestDataProtocol.

    Conforms to the same interface as vLLM's CachedRequestData.
    """

    req_ids: List[str] = field(default_factory=list)
    resumed_req_ids: Set[str] = field(default_factory=set)
    resumed_from_preemption: List[bool] = field(default_factory=list)
    new_token_ids: List[List[int]] = field(default_factory=list)
    all_token_ids: Dict[str, List[int]] = field(default_factory=dict)
    new_block_ids: List[Tuple[List[int], ...] | None] = field(default_factory=list)
    num_computed_tokens: List[int] = field(default_factory=list)
    num_output_tokens: List[int] = field(default_factory=list)

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @classmethod
    def make_empty(cls) -> "RustCachedRequestData":
        """Create an empty cached request data."""
        return cls()


@dataclass
class RustSchedulerOutput:
    """
    Our implementation of SchedulerOutputProtocol.

    Conforms to the same interface as vLLM's SchedulerOutput.
    """

    scheduled_new_reqs: List[RustNewRequestData]
    scheduled_cached_reqs: RustCachedRequestData
    num_scheduled_tokens: Dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: Dict[str, List[int]] = field(default_factory=dict)
    scheduled_encoder_inputs: Dict[str, List[int]] = field(default_factory=dict)
    num_common_prefix_blocks: List[int] = field(default_factory=list)
    finished_req_ids: Set[str] = field(default_factory=set)
    free_encoder_mm_hashes: List[str] = field(default_factory=list)
    pending_structured_output_tokens: bool = False
    kv_connector_metadata: Any | None = None
    ec_connector_metadata: Any | None = None

    @classmethod
    def make_empty(cls) -> "RustSchedulerOutput":
        """Create an empty scheduler output."""
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=RustCachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
        )
