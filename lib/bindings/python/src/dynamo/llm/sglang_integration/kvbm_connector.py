# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KVBM connector for SGLang integration.

This module provides the Python-side integration between SGLang's KV cache system
and Dynamo's KVBM (Key-Value Block Manager). It implements layerwise loading and
storing of KV cache data across distributed storage tiers (GPU → Host → Disk).

Architecture Overview:
======================

SGLang Integration Flow:
------------------------
1. SGLang's RadixCache calls `start_load_kv()` when there's a cache miss
2. KVBM queries its storage to find matching token sequences
3. Returns the number of matched tokens that can be loaded
4. SGLang allocates GPU slots and calls `load_kv_layerwise()` per layer
5. On request completion, SGLang calls `store_kv()` to persist KV cache

Key Differences from vLLM/TRTLLM:
---------------------------------
- SGLang doesn't use request IDs upfront (uses token sequences directly)
- Layerwise operations use async patterns (vs. synchronous in vLLM)
- Memory management is handled by SGLang's memory pool (not KVBM's allocator)

REQUIRED RUST API ADDITIONS:
=============================

To complete this integration, the following Rust APIs need to be exposed
in `sglang_leader.rs` (PySglangKvConnectorLeader):

1. create_slot(request_id: str, token_ids: List[int], salt_hash: Optional[int])
   - Creates a KVBM slot for tracking this token sequence
   - Maps token_ids to internal block representations
   - Returns: None (or error)

2. trigger_load(request_id: str, slot_mapping: torch.Tensor, offset: int, num_tokens: int)
   - Initiates async loading of KV data from storage to GPU
   - slot_mapping: GPU memory locations for each token
   - offset: Starting token position (must be block-aligned)
   - Returns: None (or error)

3. trigger_store(request_id: str, kv_indices: torch.Tensor, offset: int)
   - Initiates async storing of KV data from GPU to storage
   - kv_indices: GPU memory locations containing KV data
   - Returns: None (or error)

4. wait_load_complete(request_id: str) -> bool
   - Blocks until all load operations for this request finish
   - Returns: True if successful, False otherwise

5. wait_store_complete(request_id: str) -> bool
   - Blocks until all store operations for this request finish
   - Returns: True if successful, False otherwise

6. remove_slot(request_id: str)
   - Cleans up KVBM slot after request completion
   - Releases any pinned resources

Implementation Notes:
---------------------
- The Rust layer already has the Leader trait with most of these methods
- They just need to be exposed to Python via PyO3 bindings
- The actual block management and transfers are handled by Rust's
  KvConnectorLeader and BlockManager infrastructure

See Also:
---------
- lib/bindings/python/rust/llm/block_manager/vllm/connector/sglang_leader.rs
- lib/bindings/python/rust/llm/block_manager/vllm/connector/leader/slot.rs
- Integration example: LMCache's sglang_adapter.py
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

import torch
from sglang.srt.configs.model_config import ModelConfig

from dynamo.llm import KvbmLeader
from dynamo.llm.sglang_integration.rust import (
    KvConnectorLeader as RustKvConnectorLeader,
)
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


@dataclass
class StoreMetadata:
    token_ids: List[int]
    kv_indices: torch.Tensor
    offset: int


@dataclass
class LoadMetadata:
    token_ids: List[int]
    slot_mapping: torch.Tensor
    offset: int


@dataclass
class LayerwiseLoadState:
    """State for managing layerwise KV cache loading."""

    lookup_id: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    num_tokens: int
    current_layer: int
    k_pool: List[torch.Tensor]
    v_pool: List[torch.Tensor]


class KVBMLayerwiseConnector:
    """KVBM connector for SGLang with layerwise loading/storing support.

    This connector bridges SGLang's radix cache system with Dynamo's KVBM
    (Key-Value Block Manager). Unlike LMCache which uses explicit GPU
    connectors, KVBM handles block-level transfers through the Rust layer.

    Key differences from vLLM/TRTLLM integration:
    - SGLang doesn't have request IDs upfront
    - Token sequences are used directly for matching
    - Layerwise operations use generators for async loading/storing
    """

    def __init__(
        self,
        sgl_config: ModelConfig,
        page_size: int,
        tp_size: int,
        rank: int,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        **kwargs,
    ):
        drt = kwargs.get("drt", None)
        if drt is None:
            self.drt = DistributedRuntime.detached()
        else:
            self.drt = drt

        self.sgl_config = sgl_config
        self.page_size = page_size
        self.tp_size = tp_size
        self.rank = rank
        self.num_layer = sgl_config.num_hidden_layers
        self.tp_group = tp_group

        # KV cache pools (per-layer tensors)
        self.k_pool = k_pool
        self.v_pool = v_pool
        self.kvcaches = [k_pool, v_pool]

        # TODO(ziqif): change world_size to real world size with pp size
        world_size = tp_size
        leader = KvbmLeader(world_size, drt=self.drt)

        # Generate a unique worker ID for this connector
        worker_id = f"sglang-leader-{rank}"
        logger.info(f"Initializing KvConnectorLeader with worker_id: {worker_id}")
        self._leader = RustKvConnectorLeader(worker_id, self.drt, page_size, leader)

        # Track active layerwise load operations
        self.layerwise_load_states: List[LayerwiseLoadState] = []

        # Map token sequences to request IDs (for KVBM slot management)
        self._token_to_request_id: dict[str, str] = {}

    def _generate_request_id(self, token_ids: List[int]) -> str:
        """Generate a deterministic request ID from token sequence."""
        token_str = ",".join(map(str, token_ids))
        token_hash = hashlib.sha256(token_str.encode()).hexdigest()[:16]
        return f"sglang-req-{token_hash}"

    def chunk_size(self) -> int:
        """Return the chunk size for this connector."""
        return self.page_size

    def load_kv_layerwise(self, layer_id: int) -> None:
        """Load KV cache data for a specific layer across all active retrievals.

        This is called by SGLang's LayerTransferCounter after each layer completes.
        It processes all active layerwise load operations for the given layer.
        """
        if not self.layerwise_load_states:
            return

        indices_to_remove = []
        for i, load_state in enumerate(self.layerwise_load_states):
            # Check if this load state is waiting for this layer
            if load_state.current_layer == layer_id:
                # TODO(ziqif): Implement actual layer-specific KV transfer
                # This should:
                # 1. Transfer KV data for layer_id from KVBM storage to GPU
                # 2. Use load_state.slot_mapping to place data in correct slots
                # 3. Access k_pool[layer_id] and v_pool[layer_id] buffers
                logger.debug(
                    f"Loading layer {layer_id} for lookup_id {load_state.lookup_id}"
                )

                # Advance to next layer
                load_state.current_layer += 1

                # If all layers are done, mark for removal
                if load_state.current_layer >= self.num_layer:
                    indices_to_remove.append(i)

        # Remove completed load states
        for i in sorted(indices_to_remove, reverse=True):
            del self.layerwise_load_states[i]

    def start_load_kv(self, load_metadata: LoadMetadata) -> int:
        """Start loading KV cache for a token sequence from KVBM storage.

        This function:
        1. Queries KVBM to find how many tokens can be loaded from cache
        2. Sets up layerwise loading state for async retrieval
        3. Returns the number of new tokens that will be loaded

        Args:
            load_metadata: Contains token_ids, slot_mapping, and offset

        Returns:
            Number of tokens that will be loaded beyond the offset
        """
        token_ids = load_metadata.token_ids
        slot_mapping = load_metadata.slot_mapping.cuda()
        offset = load_metadata.offset

        # Convert token_ids to tensor for processing
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.int64).cuda()

        # Query KVBM for number of matched tokens
        # Note: offset is in num_computed_tokens units (must be block-aligned)
        retrieve_token_num = self._leader.get_num_new_matched_tokens(token_ids, offset)

        logger.debug(
            f"Token matching: offset={offset}, retrieve_tokens={retrieve_token_num}"
        )

        if retrieve_token_num <= offset:
            logger.debug("No new tokens to retrieve from KVBM")
            return 0

        # Generate a lookup ID and request ID for tracking
        lookup_id = str(uuid.uuid4())
        request_id = self._generate_request_id(token_ids[:retrieve_token_num])
        self._token_to_request_id[lookup_id] = request_id

        # Create KVBM slot for this request
        if not self._leader.has_slot(request_id):
            logger.debug(f"Creating KVBM slot for request_id={request_id}")
            self._leader.create_slot(
                request_id,
                token_ids[:retrieve_token_num],
                salt_hash=None,  # TODO(ziqif): Add salt hash support if needed
            )

        # Trigger KVBM to start loading data
        # Note: This will initiate async block transfers from storage to GPU
        # The actual KV data will be loaded layer-by-layer via load_kv_layerwise()
        logger.debug(
            f"Triggering KVBM load for request_id={request_id}, "
            f"tokens={retrieve_token_num}, offset={offset}"
        )

        # Create load state for layerwise retrieval
        load_state = LayerwiseLoadState(
            lookup_id=lookup_id,
            token_ids=token_ids_tensor[:retrieve_token_num],
            slot_mapping=slot_mapping[:retrieve_token_num],
            num_tokens=retrieve_token_num,
            current_layer=0,
            k_pool=self.k_pool,
            v_pool=self.v_pool,
        )

        logger.info(
            f"Starting layerwise load: lookup_id={lookup_id}, "
            f"request_id={request_id}, tokens={retrieve_token_num}"
        )

        # Add to active load states for layerwise processing
        self.layerwise_load_states.append(load_state)

        # Return number of new tokens beyond the offset
        return retrieve_token_num - offset

    def store_kv(self, store_metadata: StoreMetadata) -> None:
        """Store KV cache for a completed request into KVBM storage.

        This function:
        1. Takes the KV cache data from GPU memory (identified by kv_indices)
        2. Stores it layer-by-layer into KVBM's distributed storage
        3. Makes the data available for future cache hits

        Args:
            store_metadata: Contains token_ids, kv_indices (slot mapping),
                           last_node (radix tree node), and offset
        """
        token_ids = store_metadata.token_ids
        # kv_indices = store_metadata.kv_indices.to(torch.int64).cuda()
        offset = store_metadata.offset

        # Convert to tensors
        # token_ids_tensor = torch.tensor(token_ids, dtype=torch.int64).cuda()

        # Generate request ID
        lookup_id = str(uuid.uuid4())
        request_id = self._generate_request_id(token_ids)
        self._token_to_request_id[lookup_id] = request_id

        logger.info(
            f"Storing KV cache: lookup_id={lookup_id}, "
            f"request_id={request_id}, tokens={len(token_ids)}, offset={offset}"
        )

        # Create KVBM slot for this request if it doesn't exist
        if not self._leader.has_slot(request_id):
            logger.debug(f"Creating KVBM slot for store: request_id={request_id}")
            self._leader.create_slot(
                request_id,
                token_ids,
                salt_hash=None,  # TODO(ziqif): Add salt hash support if needed
            )

        self._leader.offload_tokens(request_id, token_ids)

        logger.debug(
            f"KVBM slot created and tracked for request_id={request_id}. "
            f"Actual block transfers will happen through KVBM's slot manager."
        )

        # Clean up tracking
        if lookup_id in self._token_to_request_id:
            del self._token_to_request_id[lookup_id]

        logger.info(f"Store setup complete for lookup_id={lookup_id}")
