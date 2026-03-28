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

from . import rust as rust_bindings


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


class _ImplCompat:
    """Small compatibility surface for aggregated TensorRT-LLM call sites."""

    def __init__(self, manager: "KvbmKVCacheManager") -> None:
        self._manager = manager

    def get_primary_pool_data(self, layer_idx: int) -> Any:
        return self._manager.get_buffers(layer_idx)

    def get_unique_primary_pool(self) -> Any:
        return self._manager.get_unique_primary_pool()

    def clear_reusable_blocks(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class KvbmKVCacheManager:
    """
    Thin TensorRT-LLM-facing manager shell.

    This class implements the v2-shaped scheduling surface and keeps enough
    request state to exercise the contract without depending on the native
    TensorRT-LLM allocator. Buffer export uses a KVBM-owned local pool when the
    Rust helper is available and falls back to explicit injection otherwise.
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
        self.num_local_layers = len(self.pp_layers)
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

        if self.num_local_layers == 0:
            raise ValueError("pp_layers must contain at least one layer")
        if len(self.total_num_kv_heads_per_layer) <= max(self.pp_layers):
            raise ValueError(
                "total_num_kv_heads_per_layer must cover all local layers"
            )

        self.kv_factor = 2
        self.layer_buffers = dict(layer_buffers or {})
        self.layer_offsets = layer_offsets or {
            layer_idx: layer_idx for layer_idx in range(len(self.pp_layers))
        }
        self.local_layers = {offset: layer_idx for layer_idx, offset in self.layer_offsets.items()}
        self.num_kv_heads_per_layer = [
            self.total_num_kv_heads_per_layer[layer_idx] for layer_idx in self.pp_layers
        ]
        self.primary_pool = primary_pool or self._maybe_create_native_primary_pool()
        self._native_state = self._maybe_create_native_state()
        self.kv_cache_pool_mapping = kv_cache_pool_mapping or [
            [0, local_layer_idx] for local_layer_idx in range(self.num_local_layers)
        ]
        self.kv_cache_pool_pointers = kv_cache_pool_pointers or [
            [0, 0] for _ in range(self.num_pools)
        ]
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

        self.impl = _ImplCompat(self)
        self._request_state: dict[int, _RequestState] = {}
        self._free_block_ids = list(range(num_blocks))
        self._iteration_events: list[Any] = []
        self._request_slots: dict[int, int] = {}
        self._free_slots = list(range(self.max_num_sequences + 1))

    def _uniform_num_kv_heads(self) -> Optional[int]:
        if not self.num_kv_heads_per_layer:
            return None
        first = self.num_kv_heads_per_layer[0]
        if any(num_heads != first for num_heads in self.num_kv_heads_per_layer[1:]):
            return None
        return first

    def _normalize_dtype_name(self) -> Optional[str]:
        value = str(self.dtype).lower()
        if value in {"float16", "torch.float16", "fp16"}:
            return "float16"
        if value in {"bfloat16", "torch.bfloat16", "bf16"}:
            return "bfloat16"
        if value in {"float32", "torch.float32", "fp32"}:
            return "float32"
        return None

    def _maybe_create_native_primary_pool(self) -> Any:
        create_primary_pool = getattr(rust_bindings, "create_primary_pool", None)
        if create_primary_pool is None or self.num_blocks == 0:
            return None

        num_kv_heads = self._uniform_num_kv_heads()
        dtype_name = self._normalize_dtype_name()
        if num_kv_heads is None or dtype_name is None:
            return None

        try:
            return create_primary_pool(
                num_blocks=self.num_blocks,
                num_layers=self.num_local_layers,
                kv_factor=self.kv_factor,
                page_size=self.tokens_per_block,
                inner_dim=num_kv_heads * self.head_dim,
                dtype=dtype_name,
            )
        except Exception:
            return None

    def _maybe_create_native_state(self) -> Any:
        state_cls = getattr(rust_bindings, "TrtllmStateManager", None)
        if state_cls is None:
            return None
        try:
            return state_cls(
                tokens_per_block=self.tokens_per_block,
                max_seq_len=self.max_seq_len,
                num_blocks=self.num_blocks,
                max_blocks_per_seq=self.max_blocks_per_seq,
                max_num_sequences=self.max_num_sequences,
                max_beam_width=self.max_beam_width,
            )
        except Exception:
            return None

    def _torch_from_dlpack(self, value: Any) -> Any:
        if not hasattr(value, "__dlpack__"):
            return value
        try:
            import torch
        except ImportError:
            return value
        return torch.utils.dlpack.from_dlpack(value)

    def _reshape_layer_export(self, value: Any, layer_offset: int, kv_layout: str) -> Any:
        tensor = self._torch_from_dlpack(value)
        if tensor is value:
            return value

        num_kv_heads = self.num_kv_heads_per_layer[layer_offset]
        expected_inner_dim = num_kv_heads * self.head_dim
        if tensor.dim() != 5 or tensor.shape[1] != 1 or tensor.shape[-1] != expected_inner_dim:
            return tensor

        tensor = tensor.squeeze(1).unflatten(-1, (num_kv_heads, self.head_dim))
        if kv_layout == "HND":
            tensor = tensor.permute(0, 1, 3, 2, 4)
        return tensor

    def _reshape_primary_pool_export(self, value: Any) -> Any:
        tensor = self._torch_from_dlpack(value)
        if tensor is value:
            return value

        num_kv_heads = self._uniform_num_kv_heads()
        if num_kv_heads is None:
            return tensor

        expected_inner_dim = num_kv_heads * self.head_dim
        if tensor.dim() != 5 or tensor.shape[-1] != expected_inner_dim:
            return tensor

        return tensor.unflatten(-1, (num_kv_heads, self.head_dim))

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

    def _write_host_block_offsets_row(self, request_id: int, padded: list[int]) -> None:
        if self._native_state is not None:
            slot = self._native_state.get_slot(request_id)
        else:
            slot = self._slot_for(request_id)
        padded = list(padded[: self.max_blocks_per_seq])
        padded.extend([BAD_PAGE_INDEX] * (self.max_blocks_per_seq - len(padded)))

        for pool_idx in range(self.num_pools):
            for beam_idx in range(self.max_beam_width):
                row = slot * self.max_beam_width + beam_idx
                self.host_kv_cache_block_offsets[pool_idx][row][0] = list(padded)
                self.host_kv_cache_block_offsets[pool_idx][row][1] = list(padded)

    def _resolve_layer_offset(self, layer_idx: int) -> int:
        if layer_idx in self.layer_offsets:
            return self.layer_offsets[layer_idx]
        if layer_idx in self.local_layers:
            return layer_idx
        raise KeyError(f"unknown TRTLLM layer index: {layer_idx}")

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
        if self._native_state is not None:
            return self._native_state.is_request_active(request_id)
        state = self._request_state.get(request_id)
        return state is not None and state.active

    def prepare_context(self, req: Any) -> bool:
        request_id = self._request_id(req)
        is_first_chunk = getattr(req, "is_first_context_chunk", True)
        if self._native_state is not None:
            prepared = self._native_state.prepare_context(request_id, is_first_chunk)
            if prepared:
                self._write_host_block_offsets_row(
                    request_id, self._native_state.get_padded_block_row(request_id, BAD_PAGE_INDEX)
                )
            return prepared
        if is_first_chunk:
            state = self._request_state.setdefault(request_id, _RequestState())
        else:
            state = self._request_state_for(request_id)
        self._slot_for(request_id)
        state.active = True
        self._write_host_block_offsets(request_id)
        return True

    def resize_context(self, req: Any, num_tokens: int) -> bool:
        request_id = self._request_id(req)
        if self._native_state is not None:
            resized = self._native_state.resize_context(
                request_id,
                getattr(req, "context_current_position", 0),
                num_tokens,
                self.num_extra_kv_tokens,
                getattr(req, "is_first_context_chunk", True),
            )
            if resized:
                self._write_host_block_offsets_row(
                    request_id, self._native_state.get_padded_block_row(request_id, BAD_PAGE_INDEX)
                )
            return resized
        state = self._request_state_for(request_id)
        target = getattr(req, "context_current_position", 0) + num_tokens
        target += self.num_extra_kv_tokens
        target = max(state.capacity, target)
        resized = self._resize_state(state, target)
        if resized:
            self._write_host_block_offsets(request_id)
        elif getattr(req, "is_first_context_chunk", True):
            state.active = False
        return resized

    def try_allocate_generation(self, req: Any) -> bool:
        request_id = self._request_id(req)
        if self._native_state is not None:
            resized = self._native_state.try_allocate_generation(
                request_id, self._get_draft_token_length(req)
            )
            if resized:
                self._write_host_block_offsets_row(
                    request_id, self._native_state.get_padded_block_row(request_id, BAD_PAGE_INDEX)
                )
            return resized
        state = self._request_state.get(request_id)
        if state is None:
            return False
        state.active = True
        target = state.capacity + 1 + self._get_draft_token_length(req)
        resized = self._resize_state(state, target)
        if resized:
            self._write_host_block_offsets(request_id)
        return resized

    def suspend_request(self, req: Any) -> None:
        if self._native_state is not None:
            self._native_state.suspend_request(self._request_id(req))
            return
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
            request_id = self._request_id(request)
            if self._native_state is not None:
                updated = self._native_state.update_context(
                    request_id,
                    getattr(request, "context_current_position", 0),
                    self.num_extra_kv_tokens,
                )
                if not updated:
                    raise ValueError(
                        f"failed to resize context history for request {request_id}"
                    )
                if self.is_request_active(request_id):
                    self._write_host_block_offsets_row(
                        request_id,
                        self._native_state.get_padded_block_row(
                            request_id, BAD_PAGE_INDEX
                        ),
                    )
                continue
            state = self._request_state.get(request_id)
            if state is None:
                continue
            if not state.active:
                continue
            if state.capacity < getattr(request, "context_current_position", 0):
                if not self._resize_state(
                    state,
                    getattr(request, "context_current_position", 0)
                    + self.num_extra_kv_tokens,
                ):
                    raise ValueError(
                        f"failed to resize context history for request {request_id}"
                    )
            state.history_length = getattr(request, "context_current_position", 0)
            state.committed_tokens = state.history_length
            self._write_host_block_offsets(request_id)

        for request in getattr(scheduled_batch, "generation_requests", ()):
            request_id = self._request_id(request)
            if self._native_state is not None:
                updated = self._native_state.update_generation(
                    request_id,
                    max(getattr(request, "max_beam_num_tokens", 1), 0),
                    getattr(request, "py_rewind_len", 0),
                )
                if not updated:
                    raise ValueError(
                        f"failed to update generation state for request {request_id}"
                    )
                if self.is_request_active(request_id):
                    self._write_host_block_offsets_row(
                        request_id,
                        self._native_state.get_padded_block_row(
                            request_id, BAD_PAGE_INDEX
                        ),
                    )
                continue
            state = self._request_state.get(request_id)
            if state is None:
                continue
            if not state.active:
                continue
            rewind_len = getattr(request, "py_rewind_len", 0)
            history_length = max(getattr(request, "max_beam_num_tokens", 1) - 1, 0)
            target = max(state.capacity - rewind_len, history_length)
            if not self._resize_state(state, target):
                raise ValueError(
                    f"failed to update generation state for request {request_id}"
                )
            state.history_length = history_length
            state.committed_tokens = state.history_length
            self._write_host_block_offsets(request_id)

    def free_resources(self, request: Any, pin_on_release: bool = False) -> None:
        del pin_on_release

        request_id = self._request_id(request)
        if self._native_state is not None:
            slot = self._native_state.free_resources(request_id)
            if slot is not None:
                self._clear_slot(slot)
            return
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
        if self._native_state is not None:
            return list(self._native_state.get_cache_indices(request_id))
        return list(self._request_state_for(request_id).block_ids)

    def get_batch_cache_indices(
        self,
        request_ids: list[int],
        layer_id: int = 0,
    ) -> list[list[int]]:
        del layer_id
        if self._native_state is not None:
            return [list(row) for row in self._native_state.get_batch_cache_indices(request_ids)]
        return [self.get_cache_indices(request_id) for request_id in request_ids]

    def get_block_ids_per_seq(self, request_ids: list[int]) -> list[list[int]]:
        rows = self.get_batch_cache_indices(request_ids)
        padded_len = max((len(row) for row in rows), default=0)
        result = []
        for row in rows:
            padded = list(row)
            padded.extend([0] * (padded_len - len(padded)))
            result.append(padded)
        return result

    def _missing_export(self, export_name: str) -> NotImplementedError:
        return NotImplementedError(
            f"{export_name} is not wired yet; KVBM-backed TRTLLM tensor export "
            "lands in the next milestone"
        )

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> Any:
        if kv_layout not in {"NHD", "HND"}:
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        layer_offset = self._resolve_layer_offset(layer_idx)
        if layer_idx in self.layer_buffers:
            return self.layer_buffers[layer_idx]
        if layer_offset in self.layer_buffers:
            return self.layer_buffers[layer_offset]
        if self.primary_pool is not None:
            if hasattr(self.primary_pool, "get_layer_view"):
                return self.primary_pool.get_layer_view(layer_offset, kv_layout=kv_layout)
            if hasattr(self.primary_pool, "layer_view"):
                try:
                    export = self.primary_pool.layer_view(layer_offset, kv_layout=kv_layout)
                except TypeError:
                    export = self.primary_pool.layer_view(layer_offset)
                return self._reshape_layer_export(export, layer_offset, kv_layout)
        raise self._missing_export("get_buffers")

    def get_unique_primary_pool(self) -> Any:
        if self.primary_pool is not None:
            return self._reshape_primary_pool_export(self.primary_pool)
        raise self._missing_export("get_unique_primary_pool")

    def get_num_free_blocks(self) -> int:
        if self._native_state is not None:
            return int(self._native_state.get_num_free_blocks())
        return len(self._free_block_ids)

    def get_num_available_tokens(
        self,
        *,
        token_num_upper_bound: int,
        batch_size: int = 1,
        max_num_draft_tokens: int = 0,
    ) -> int:
        del batch_size
        if self._native_state is not None:
            return int(
                self._native_state.get_num_available_tokens(
                    token_num_upper_bound,
                    max_num_draft_tokens,
                    self.num_extra_kv_tokens,
                )
            )

        free_tokens = len(self._free_block_ids) * self.tokens_per_block
        available = max(free_tokens - self.num_extra_kv_tokens - max_num_draft_tokens, 0)
        return min(token_num_upper_bound, available)

    def get_num_kv_blocks(self) -> int:
        if self._native_state is not None:
            return int(self._native_state.get_num_kv_blocks())
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
            request.is_first_context_chunk = True
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
            if self._native_state is not None:
                if not self._native_state.resize_context(
                    request_id, 0, target, 0, True
                ):
                    raise RuntimeError(f"failed to allocate dummy request {request_id}")
                self._write_host_block_offsets_row(
                    request_id,
                    self._native_state.get_padded_block_row(request_id, BAD_PAGE_INDEX),
                )
            else:
                if not self._resize_state(self._request_state_for(request_id), target):
                    raise RuntimeError(f"failed to allocate dummy request {request_id}")
            if draft_kv_cache_manager is not None:
                draft_kv_cache_manager.prepare_context(request)
                if draft_kv_cache_manager._native_state is not None:
                    if not draft_kv_cache_manager._native_state.resize_context(
                        request_id, 0, target, 0, True
                    ):
                        raise RuntimeError(
                            f"failed to allocate draft dummy request {request_id}"
                        )
                    draft_kv_cache_manager._write_host_block_offsets_row(
                        request_id,
                        draft_kv_cache_manager._native_state.get_padded_block_row(
                            request_id, BAD_PAGE_INDEX
                        ),
                    )
                elif not draft_kv_cache_manager._resize_state(
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

        multi_pool_dst = (
            self.num_pools > 1
            and isinstance(dst_tensor, list)
            and len(dst_tensor) == self.num_pools
        )

        for index, value in enumerate(block_ids):
            del value
            if self._native_state is not None:
                row = self._native_state.get_padded_block_row(
                    request_ids[index], BAD_PAGE_INDEX
                )
                self._write_host_block_offsets_row(request_ids[index], row)
                slot = self._native_state.get_slot(request_ids[index]) * self.max_beam_width
            else:
                slot = self._slot_for(request_ids[index]) * self.max_beam_width
            if multi_pool_dst:
                for pool_idx in range(self.num_pools):
                    dst_tensor[pool_idx][index] = list(
                        self.host_kv_cache_block_offsets[pool_idx][slot][0]
                    )
            else:
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
        if self._native_state is not None:
            self._native_state.shutdown()
        self._request_state.clear()
        self._free_block_ids = list(range(self.num_blocks))
        self._request_slots.clear()
        self._free_slots = list(range(self.max_num_sequences + 1))
        for slot in self._free_slots:
            self._clear_slot(slot)
