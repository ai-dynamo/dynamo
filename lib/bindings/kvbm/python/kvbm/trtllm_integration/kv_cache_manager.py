# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dependency-light TensorRT-LLM KV cache manager shell for KVBM integration.

This module intentionally avoids importing TensorRT-LLM at import time so it can
be exercised with stubbed request objects in unit tests and in environments
where the full TRT-LLM runtime is not installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Optional


BAD_PAGE_INDEX = -1


@dataclass
class KvCacheStats:
    allocated_bytes: int = 0


@dataclass
class _RequestState:
    block_ids: list[int] = field(default_factory=list)
    capacity: int = 0
    history_length: int = 0
    active: bool = True
    committed_tokens: int = 0


class KvbmKVCacheManager:
    """
    Thin TensorRT-LLM-facing manager shell.

    This class implements the v2-shaped scheduling surface and keeps enough
    request state to exercise the contract without depending on the native
    TensorRT-LLM allocator. Buffer export remains explicit and must be wired to
    a KVBM-owned backend in a later milestone.
    """

    def __init__(
        self,
        *,
        tokens_per_block: int,
        dtype: Any,
        head_dim: int,
        pp_layers: Iterable[int],
        total_num_kv_heads_per_layer: Iterable[int],
        max_seq_len: int,
        num_blocks: int,
        num_pools: int = 1,
        max_num_sequences: int = 32,
        max_beam_width: int = 1,
        max_blocks_per_seq: Optional[int] = None,
        kv_layout: str = "NHD",
        primary_pool: Any = None,
        layer_buffers: Optional[dict[int, Any]] = None,
        layer_offsets: Optional[dict[int, int]] = None,
        kv_cache_pool_mapping: Optional[list[int]] = None,
        kv_cache_pool_pointers: Optional[list[int]] = None,
        host_kv_cache_block_offsets: Optional[Any] = None,
        num_extra_kv_tokens: int = 0,
        is_draft: bool = False,
    ) -> None:
        if tokens_per_block <= 0:
            raise ValueError("tokens_per_block must be greater than 0")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if kv_layout not in {"NHD", "HND"}:
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")

        self.tokens_per_block = tokens_per_block
        self.dtype = dtype
        self.head_dim = head_dim
        self.pp_layers = list(pp_layers)
        self.total_num_kv_heads_per_layer = list(total_num_kv_heads_per_layer)
        self.max_seq_len = max_seq_len
        self.num_blocks = num_blocks
        self.num_pools = num_pools
        self.max_num_sequences = max_num_sequences
        self.max_beam_width = max_beam_width
        self.max_blocks_per_seq = max_blocks_per_seq or num_blocks
        self.kv_layout = kv_layout
        self.num_extra_kv_tokens = num_extra_kv_tokens
        self.is_draft = is_draft

        self.primary_pool = primary_pool
        self.layer_buffers = dict(layer_buffers or {})
        self.layer_offsets = layer_offsets or {
            layer_idx: layer_idx for layer_idx in range(len(self.pp_layers))
        }
        self.kv_cache_pool_mapping = kv_cache_pool_mapping or [0] * len(
            self.layer_offsets
        )
        self.kv_cache_pool_pointers = kv_cache_pool_pointers or []
        self.host_kv_cache_block_offsets = host_kv_cache_block_offsets or [
            [
                [
                    [BAD_PAGE_INDEX for _ in range(self.max_blocks_per_seq)]
                    for _ in range(2)
                ]
                for _ in range((self.max_num_sequences + 1) * self.max_beam_width)
            ]
            for _ in range(self.num_pools)
        ]

        self.impl = None
        self._request_state: dict[int, _RequestState] = {}
        self._free_block_ids = list(range(num_blocks))
        self._iteration_events: list[Any] = []
        self._request_slots: dict[int, int] = {}
        self._free_slots = list(range(self.max_num_sequences + 1))

    def _request_id(self, request: Any) -> int:
        if hasattr(request, "py_request_id"):
            return int(request.py_request_id)
        if hasattr(request, "request_id"):
            return int(request.request_id)
        raise AttributeError("request object is missing py_request_id/request_id")

    def _request_state_for(self, request_id: int) -> _RequestState:
        state = self._request_state.get(request_id)
        if state is None:
            raise KeyError(f"request {request_id} is not active")
        return state

    def _slot_for(self, request_id: int) -> int:
        if request_id in self._request_slots:
            return self._request_slots[request_id]
        if not self._free_slots:
            raise RuntimeError("no free TRTLLM cache slots available")
        slot = self._free_slots.pop(0)
        self._request_slots[request_id] = slot
        return slot

    def _clear_slot(self, slot: int) -> None:
        for pool_idx in range(self.num_pools):
            for beam_idx in range(self.max_beam_width):
                row = slot * self.max_beam_width + beam_idx
                self.host_kv_cache_block_offsets[pool_idx][row][0] = [
                    BAD_PAGE_INDEX for _ in range(self.max_blocks_per_seq)
                ]
                self.host_kv_cache_block_offsets[pool_idx][row][1] = [
                    BAD_PAGE_INDEX for _ in range(self.max_blocks_per_seq)
                ]

    def _write_host_block_offsets(self, request_id: int) -> None:
        state = self._request_state_for(request_id)
        slot = self._slot_for(request_id)
        padded = list(state.block_ids[: self.max_blocks_per_seq])
        padded.extend([BAD_PAGE_INDEX] * (self.max_blocks_per_seq - len(padded)))

        for pool_idx in range(self.num_pools):
            for beam_idx in range(self.max_beam_width):
                row = slot * self.max_beam_width + beam_idx
                self.host_kv_cache_block_offsets[pool_idx][row][0] = list(padded)
                self.host_kv_cache_block_offsets[pool_idx][row][1] = list(padded)

    def _required_blocks(self, token_capacity: int) -> int:
        if token_capacity <= 0:
            return 0
        return min(
            math.ceil(token_capacity / self.tokens_per_block),
            self.max_blocks_per_seq,
        )

    def _resize_state(self, state: _RequestState, target_capacity: int) -> bool:
        target_capacity = min(target_capacity, self.max_seq_len)
        required_blocks = self._required_blocks(target_capacity)
        current_blocks = len(state.block_ids)

        if required_blocks > current_blocks:
            needed = required_blocks - current_blocks
            if needed > len(self._free_block_ids):
                return False
            state.block_ids.extend(self._free_block_ids[:needed])
            del self._free_block_ids[:needed]
        elif required_blocks < current_blocks:
            released = state.block_ids[required_blocks:]
            self._free_block_ids.extend(released)
            state.block_ids = state.block_ids[:required_blocks]

        state.capacity = min(target_capacity, required_blocks * self.tokens_per_block)
        return True

    def _get_draft_token_length(self, request: Any) -> int:
        return len(getattr(request, "py_draft_tokens", ()) or ())

    def is_request_active(self, request_id: int) -> bool:
        state = self._request_state.get(request_id)
        return state is not None and state.active

    def prepare_context(self, req: Any) -> bool:
        request_id = self._request_id(req)
        state = self._request_state.setdefault(request_id, _RequestState())
        self._slot_for(request_id)
        state.active = True
        self._write_host_block_offsets(request_id)
        return True

    def resize_context(self, req: Any, num_tokens: int) -> bool:
        request_id = self._request_id(req)
        state = self._request_state_for(request_id)
        target = getattr(req, "context_current_position", 0) + num_tokens
        target += self.num_extra_kv_tokens
        resized = self._resize_state(state, target)
        if resized:
            self._write_host_block_offsets(request_id)
        return resized

    def try_allocate_generation(self, req: Any) -> bool:
        request_id = self._request_id(req)
        state = self._request_state_for(request_id)
        state.active = True
        target = state.capacity + 1 + self._get_draft_token_length(req)
        resized = self._resize_state(state, target)
        if resized:
            self._write_host_block_offsets(request_id)
        return resized

    def suspend_request(self, req: Any) -> None:
        self._request_state_for(self._request_id(req)).active = False

    def prepare_resources(self, scheduled_batch: Any) -> None:
        for request in getattr(scheduled_batch, "context_requests", ()):
            if not self.prepare_context(request):
                raise RuntimeError(
                    f"failed to prepare context for request {self._request_id(request)}"
                )
            chunk_size = getattr(
                request,
                "context_chunk_size",
                getattr(request, "context_remaining_length", 0),
            )
            if not self.resize_context(request, chunk_size):
                raise RuntimeError(
                    f"failed to resize context for request {self._request_id(request)}"
                )

        for request in getattr(scheduled_batch, "generation_requests", ()):
            if not self.try_allocate_generation(request):
                raise RuntimeError(
                    f"failed to allocate generation for request {self._request_id(request)}"
                )

    def update_resources(
        self,
        scheduled_batch: Any,
        attn_metadata: Any = None,
        kv_cache_dtype_byte_size: float = None,
    ) -> None:
        del attn_metadata
        del kv_cache_dtype_byte_size

        for request in getattr(scheduled_batch, "context_requests", ()):
            state = self._request_state.get(self._request_id(request))
            if state is None:
                continue
            state.history_length = getattr(request, "context_current_position", 0)
            state.committed_tokens = state.history_length
            self._write_host_block_offsets(self._request_id(request))

        for request in getattr(scheduled_batch, "generation_requests", ()):
            request_id = self._request_id(request)
            state = self._request_state.get(request_id)
            if state is None:
                continue
            rewind_len = getattr(request, "py_rewind_len", 0)
            target = max(state.capacity - rewind_len, state.history_length)
            self._resize_state(state, target)
            state.history_length = max(getattr(request, "max_beam_num_tokens", 1) - 1, 0)
            state.committed_tokens = state.history_length
            self._write_host_block_offsets(request_id)

    def free_resources(self, request: Any, pin_on_release: bool = False) -> None:
        del pin_on_release

        request_id = self._request_id(request)
        state = self._request_state.pop(request_id, None)
        if state is None:
            return
        self._free_block_ids.extend(state.block_ids)
        slot = self._request_slots.pop(request_id, None)
        if slot is not None:
            self._clear_slot(slot)
            self._free_slots.append(slot)
            self._free_slots.sort()

    def get_cache_indices(self, request_id: int) -> list[int]:
        return list(self._request_state_for(request_id).block_ids)

    def get_batch_cache_indices(
        self,
        request_ids: list[int],
        layer_id: int = 0,
    ) -> list[list[int]]:
        del layer_id
        return [self.get_cache_indices(request_id) for request_id in request_ids]

    def get_block_ids_per_seq(self, request_ids: list[int]) -> list[list[int]]:
        return self.get_batch_cache_indices(request_ids)

    def _missing_export(self, export_name: str) -> NotImplementedError:
        return NotImplementedError(
            f"{export_name} is not wired yet; KVBM-backed TRTLLM tensor export "
            "lands in the next milestone"
        )

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> Any:
        del kv_layout
        if layer_idx in self.layer_buffers:
            return self.layer_buffers[layer_idx]
        raise self._missing_export("get_buffers")

    def get_unique_primary_pool(self) -> Any:
        if self.primary_pool is not None:
            return self.primary_pool
        raise self._missing_export("get_unique_primary_pool")

    def get_num_free_blocks(self) -> int:
        return len(self._free_block_ids)

    def get_num_available_tokens(
        self,
        *,
        token_num_upper_bound: int,
        batch_size: int = 1,
        max_num_draft_tokens: int = 0,
    ) -> int:
        del batch_size

        free_tokens = len(self._free_block_ids) * self.tokens_per_block
        available = max(free_tokens - self.num_extra_kv_tokens - max_num_draft_tokens, 0)
        return min(token_num_upper_bound, available)

    def get_num_kv_blocks(self) -> int:
        return self.num_blocks

    def add_dummy_requests(
        self,
        request_ids: list[int],
        token_nums: Optional[list[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        num_extra_decoding_steps: int = 0,
        draft_kv_cache_manager: Optional["KvbmKVCacheManager"] = None,
    ) -> list[Any]:
        del use_mrope
        del max_beam_width

        requests = []
        for index, request_id in enumerate(request_ids):
            token_num = token_nums[index] if token_nums is not None else 1
            request = type("DummyRequest", (), {})()
            request.request_id = request_id
            request.py_request_id = request_id
            request.context_current_position = token_num
            request.context_chunk_size = token_num
            request.context_remaining_length = token_num
            request.max_beam_num_tokens = token_num
            request.py_rewind_len = 0
            request.py_draft_tokens = [1] * max_num_draft_tokens if is_gen else []
            requests.append(request)

            if not prepare_resource:
                continue

            self.prepare_context(request)
            target = token_num + self.num_extra_kv_tokens + num_extra_decoding_steps
            if is_gen:
                target += max_num_draft_tokens + 1
            if not self._resize_state(self._request_state_for(request_id), target):
                raise RuntimeError(f"failed to allocate dummy request {request_id}")
            if draft_kv_cache_manager is not None:
                draft_kv_cache_manager.prepare_context(request)
                if not draft_kv_cache_manager._resize_state(
                    draft_kv_cache_manager._request_state_for(request_id), target
                ):
                    raise RuntimeError(
                        f"failed to allocate draft dummy request {request_id}"
                    )

        return requests

    def get_kv_cache_stats(self) -> KvCacheStats:
        allocated_blocks = self.num_blocks - len(self._free_block_ids)
        allocated_bytes = allocated_blocks * self.tokens_per_block * self.head_dim
        return KvCacheStats(allocated_bytes=allocated_bytes)

    def copy_batch_block_offsets(
        self,
        dst_tensor: Any,
        request_ids: list[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        del beam_width
        del num_contexts

        block_ids = self.get_batch_cache_indices(request_ids)
        if len(block_ids) != num_seqs:
            raise ValueError(
                f"num_seqs={num_seqs} does not match request_ids={len(block_ids)}"
            )

        for index, value in enumerate(block_ids):
            slot = self._slot_for(request_ids[index]) * self.max_beam_width
            dst_tensor[index] = list(self.host_kv_cache_block_offsets[0][slot][0])

    def flush_iteration_events(self) -> None:
        self._iteration_events.clear()

    def get_latest_events(self, timeout_ms: Optional[float] = 0) -> list[Any]:
        del timeout_ms
        return list(self._iteration_events)

    def store_blocks_for_reuse(
        self,
        request: Any,
        block_ids: Optional[list[int]] = None,
        pin_on_release: bool = False,
    ) -> Optional[int]:
        del request
        del pin_on_release

        if not block_ids:
            return None
        return block_ids[-1]

    def unpin_blocks_by_id(self, kv_cache_block_id: int) -> None:
        del kv_cache_block_id

    def shutdown(self) -> None:
        self._request_state.clear()
        self._free_block_ids = list(range(self.num_blocks))
        self._request_slots.clear()
        self._free_slots = list(range(self.max_num_sequences + 1))
        for slot in self._free_slots:
            self._clear_slot(slot)
