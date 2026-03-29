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


@dataclass(frozen=True)
class _MappingShim:
    world_size: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    rank: int
    device_id: int
    dp_size: int = 1
    dp_rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    gpus_per_node: int = 1
    enable_attention_dp: bool = False
    cp_config: dict[str, Any] = field(default_factory=dict)

    def is_first_pp_rank(self) -> bool:
        return self.pp_rank == 0

    def is_last_pp_rank(self) -> bool:
        return self.pp_rank == self.pp_size - 1


@dataclass(frozen=True)
class _FakeBufferAttr:
    life_cycle_id: int
    pool_index: int
    offset: int
    size: int


@dataclass(frozen=True)
class _FakeStoragePool:
    base_address: int
    slot_size: int
    num_slots: int

    def slot_address(self, slot_id: int) -> int:
        return self.base_address + slot_id * self.slot_size


@dataclass(frozen=True)
class _FakeStoragePoolGroup:
    _pools: list[_FakeStoragePool]

    @property
    def num_pools(self) -> int:
        return len(self._pools)


@dataclass(frozen=True)
class _FakeStorageLevel:
    storage: Any


@dataclass(frozen=True)
class _FakePoolGroups:
    _pool_groups: list[_FakeStoragePoolGroup]


@dataclass(frozen=True)
class _FakeLifeCycle:
    window_size: Optional[int]


@dataclass(frozen=True)
class _FakeCacheTierConfig:
    tier: Any


@dataclass(frozen=True)
class _FakeInitConfig:
    tokens_per_block: int
    cache_tiers: list[_FakeCacheTierConfig]


@dataclass(frozen=True)
class _RequestSnapshot:
    request_id: int
    is_first_context_chunk: bool = True
    context_current_position: int = 0
    context_chunk_size: int = 0
    max_beam_num_tokens: int = 1
    py_rewind_len: int = 0
    draft_token_length: int = 0

    @classmethod
    def from_request(cls, request: Any) -> "_RequestSnapshot":
        try:
            context_chunk_size = request.context_chunk_size
        except AttributeError:
            context_chunk_size = request.context_remaining_length

        draft_tokens = request.py_draft_tokens
        return cls(
            request_id=int(request.py_request_id),
            is_first_context_chunk=bool(request.is_first_context_chunk),
            context_current_position=int(request.context_current_position),
            context_chunk_size=int(context_chunk_size),
            max_beam_num_tokens=int(request.max_beam_num_tokens),
            py_rewind_len=int(request.py_rewind_len),
            draft_token_length=len(draft_tokens or ()),
        )


@dataclass
class _DummyRequest:
    request_id: int
    py_request_id: int
    is_first_context_chunk: bool
    context_current_position: int
    context_chunk_size: int
    context_remaining_length: int
    max_beam_num_tokens: int
    py_rewind_len: int
    py_draft_tokens: list[int]


class _FakeStorage:
    def __init__(self, pool_group: _FakeStoragePoolGroup, buffer_attr: dict[Any, Any]) -> None:
        self._buffer_attr = buffer_attr
        self._levels = [_FakeStorageLevel(storage=_FakePoolGroups(_pool_groups=[pool_group]))]
        self.num_life_cycles = 1

    def get_pool_group_index(self, life_cycle_id: int) -> int:
        del life_cycle_id
        return 0


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
        self._manager.shutdown()

    @property
    def layer_grouping(self) -> list[list[int]]:
        return self._manager.get_layer_grouping()

    @property
    def _storage(self) -> Any:
        return self._manager.get_disagg_storage_metadata()

    @property
    def _init_config(self) -> Any:
        return self._manager.get_disagg_init_config()

    @property
    def _life_cycles(self) -> list[Any]:
        return self._manager.get_disagg_life_cycles()

    def get_indexer_k_cache_pool(self) -> Any:
        raise NotImplementedError(
            "Indexer K-cache export is not implemented for the current KVBM TRTLLM path"
        )


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
        device_id: int = 0,
        world_size: int = 1,
        tp_size: int = 1,
        tp_rank: int = 0,
        pp_size: int = 1,
        pp_rank: int = 0,
        cache_mode: str = "standard",
    ) -> None:
        if tokens_per_block <= 0:
            raise ValueError("tokens_per_block must be greater than 0")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if kv_layout not in {"NHD", "HND"}:
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        if world_size <= 0:
            raise ValueError("world_size must be greater than 0")
        if tp_size <= 0:
            raise ValueError("tp_size must be greater than 0")
        if pp_size <= 0:
            raise ValueError("pp_size must be greater than 0")
        if not 0 <= tp_rank < tp_size:
            raise ValueError("tp_rank must be within [0, tp_size)")
        if not 0 <= pp_rank < pp_size:
            raise ValueError("pp_rank must be within [0, pp_size)")
        if world_size != tp_size * pp_size:
            raise ValueError("world_size must equal tp_size * pp_size")
        if cache_mode not in {"standard", "mla"}:
            raise ValueError(f"Unsupported cache_mode: {cache_mode}")

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
        self.device_id = device_id
        self.world_size = world_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.cache_mode = cache_mode
        self.is_mla = cache_mode == "mla"
        self.is_mla_enable = self.is_mla
        self.max_batch_size = max(1, math.ceil(self.max_num_sequences / self.pp_size))
        self.is_vswa = False
        self.max_draft_len = 0
        self.enable_indexer_k_cache = False
        self.mapping = _MappingShim(
            world_size=self.world_size,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            pp_size=self.pp_size,
            pp_rank=self.pp_rank,
            rank=self.pp_rank * self.tp_size + self.tp_rank,
            device_id=self.device_id,
            gpus_per_node=max(self.world_size, 1),
        )

        if self.num_local_layers == 0:
            raise ValueError("pp_layers must contain at least one layer")
        if len(self.total_num_kv_heads_per_layer) <= max(self.pp_layers):
            raise ValueError(
                "total_num_kv_heads_per_layer must cover all local layers"
            )

        self.kv_factor = 1 if self.is_mla else 2
        self.layer_buffers = dict(layer_buffers or {})
        self.layer_offsets = layer_offsets or {
            layer_idx: offset for offset, layer_idx in enumerate(self.pp_layers)
        }
        self.local_layers = {offset: layer_idx for layer_idx, offset in self.layer_offsets.items()}
        self.num_kv_heads_per_layer = [
            self._local_num_kv_heads(self.total_num_kv_heads_per_layer[layer_idx])
            for layer_idx in self.pp_layers
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
        self._disagg_metadata: Optional[dict[str, Any]] = None

    def _local_num_kv_heads(self, total_num_kv_heads: int) -> int:
        if total_num_kv_heads < 0:
            raise ValueError("total_num_kv_heads_per_layer values must be non-negative")
        if self.is_mla:
            return total_num_kv_heads
        return math.ceil(total_num_kv_heads / self.tp_size)

    def get_worker_identity(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "pp_size": self.pp_size,
            "pp_rank": self.pp_rank,
            "cache_mode": self.cache_mode,
            "kv_factor": self.kv_factor,
        }

    def get_layer_grouping(self) -> list[list[int]]:
        return [list(range(self.num_local_layers))]

    def _get_window_size_to_layers(self) -> dict[Optional[int], list[int]]:
        window_size_to_layers: dict[Optional[int], list[int]] = {}
        for life_cycle, layers in zip(self.get_disagg_life_cycles(), self.get_layer_grouping()):
            window_size_to_layers.setdefault(life_cycle.window_size, []).extend(layers)
        return window_size_to_layers

    def _build_disagg_metadata(self) -> dict[str, Any]:
        pool = self.get_unique_primary_pool()
        required_attrs = ("shape", "data_ptr", "element_size")
        if any(not hasattr(pool, attr) for attr in required_attrs):
            raise NotImplementedError(
                "Primary-pool export does not expose tensor metadata needed for TRTLLM "
                "disaggregation page-table construction"
            )
        if not hasattr(pool, "stride"):
            raise NotImplementedError(
                "Primary-pool export is missing stride metadata needed for TRTLLM "
                "disaggregation page-table construction"
            )
        shape = tuple(int(dim) for dim in pool.shape)
        if len(shape) != 6:
            raise NotImplementedError(
                "Disaggregation metadata expects primary pool layout "
                "[blocks, layers, kv_factor, page_size, num_heads, head_dim]"
            )

        element_size = int(pool.element_size())
        slot_bytes = int(pool.stride(0)) * element_size
        layer_stride = int(pool.stride(1)) * element_size
        role_stride = int(pool.stride(2)) * element_size
        buffer_size = role_stride
        base_address = int(pool.data_ptr())

        role_key = "key"
        role_value = "value"
        try:
            from tensorrt_llm._torch.pyexecutor.resource_manager import Role
        except ImportError:
            pass
        else:
            role_key = Role.KEY
            role_value = Role.VALUE

        buffer_attr = {}
        for local_layer_id in range(self.num_local_layers):
            layer_offset = local_layer_id * layer_stride
            buffer_attr[(local_layer_id, role_key)] = _FakeBufferAttr(
                life_cycle_id=0,
                pool_index=0,
                offset=layer_offset,
                size=buffer_size,
            )
            if self.kv_factor > 1:
                buffer_attr[(local_layer_id, role_value)] = _FakeBufferAttr(
                    life_cycle_id=0,
                    pool_index=0,
                    offset=layer_offset + role_stride,
                    size=buffer_size,
                )

        pool_group = _FakeStoragePoolGroup(
            _pools=[
                _FakeStoragePool(
                    base_address=base_address,
                    slot_size=slot_bytes,
                    num_slots=shape[0],
                )
            ]
        )

        cache_tier = "gpu_mem"
        try:
            from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier
        except ImportError:
            pass
        else:
            cache_tier = CacheTier.GPU_MEM

        return {
            "storage": _FakeStorage(pool_group=pool_group, buffer_attr=buffer_attr),
            "init_config": _FakeInitConfig(
                tokens_per_block=self.tokens_per_block,
                cache_tiers=[_FakeCacheTierConfig(tier=cache_tier)],
            ),
            "life_cycles": [_FakeLifeCycle(window_size=None)],
        }

    def _get_disagg_metadata(self) -> dict[str, Any]:
        if self._disagg_metadata is None:
            self._disagg_metadata = self._build_disagg_metadata()
        return self._disagg_metadata

    def get_disagg_storage_metadata(self) -> Any:
        return self._get_disagg_metadata()["storage"]

    def get_disagg_init_config(self) -> Any:
        return self._get_disagg_metadata()["init_config"]

    def get_disagg_life_cycles(self) -> list[Any]:
        return list(self._get_disagg_metadata()["life_cycles"])

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

    def _dtype_size_bytes(self) -> int:
        dtype_name = self._normalize_dtype_name()
        if dtype_name == "float16":
            return 2
        if dtype_name == "bfloat16":
            return 2
        if dtype_name == "float32":
            return 4
        return 0

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
                device_id=self.device_id,
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
                device_id=self.device_id,
                world_size=self.world_size,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                pp_size=self.pp_size,
                pp_rank=self.pp_rank,
                kv_factor=self.kv_factor,
                cache_mode=self.cache_mode,
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

    def _reset_host_block_offsets(self) -> None:
        for slot in range(self.max_num_sequences + 1):
            self._clear_slot(slot)

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

    def is_request_active(self, request_id: int) -> bool:
        if self._native_state is not None:
            return self._native_state.is_request_active(request_id)
        state = self._request_state.get(request_id)
        return state is not None and state.active

    def prepare_context(self, req: Any) -> bool:
        request = _RequestSnapshot.from_request(req)
        if self._native_state is not None:
            prepared = self._native_state.prepare_context(
                request.request_id, request.is_first_context_chunk
            )
            if prepared:
                self._write_host_block_offsets_row(
                    request.request_id,
                    self._native_state.get_padded_block_row(
                        request.request_id, BAD_PAGE_INDEX
                    ),
                )
            return prepared
        if request.is_first_context_chunk:
            state = self._request_state.setdefault(request.request_id, _RequestState())
        else:
            state = self._request_state_for(request.request_id)
        self._slot_for(request.request_id)
        state.active = True
        self._write_host_block_offsets(request.request_id)
        return True

    def resize_context(self, req: Any, num_tokens: int) -> bool:
        request = _RequestSnapshot.from_request(req)
        if self._native_state is not None:
            resized = self._native_state.resize_context(
                request.request_id,
                request.context_current_position,
                num_tokens,
                self.num_extra_kv_tokens,
                request.is_first_context_chunk,
            )
            if resized:
                self._write_host_block_offsets_row(
                    request.request_id,
                    self._native_state.get_padded_block_row(
                        request.request_id, BAD_PAGE_INDEX
                    ),
                )
            return resized
        state = self._request_state_for(request.request_id)
        target = request.context_current_position + num_tokens
        target += self.num_extra_kv_tokens
        target = max(state.capacity, target)
        resized = self._resize_state(state, target)
        if resized:
            self._write_host_block_offsets(request.request_id)
        elif request.is_first_context_chunk:
            state.active = False
        return resized

    def try_allocate_generation(self, req: Any) -> bool:
        request = _RequestSnapshot.from_request(req)
        if self._native_state is not None:
            resized = self._native_state.try_allocate_generation(
                request.request_id, request.draft_token_length
            )
            if resized:
                self._write_host_block_offsets_row(
                    request.request_id,
                    self._native_state.get_padded_block_row(
                        request.request_id, BAD_PAGE_INDEX
                    ),
                )
            return resized
        state = self._request_state.get(request.request_id)
        if state is None:
            return False
        state.active = True
        target = state.capacity + 1 + request.draft_token_length
        resized = self._resize_state(state, target)
        if resized:
            self._write_host_block_offsets(request.request_id)
        return resized

    def suspend_request(self, req: Any) -> None:
        request = _RequestSnapshot.from_request(req)
        if self._native_state is not None:
            self._native_state.suspend_request(request.request_id)
            return
        self._request_state_for(request.request_id).active = False

    def prepare_resources(self, scheduled_batch: Any) -> None:
        for request in getattr(scheduled_batch, "context_requests", ()):
            request_state = _RequestSnapshot.from_request(request)
            if not self.prepare_context(request):
                raise RuntimeError(
                    f"failed to prepare context for request {request_state.request_id}"
                )
            if not self.resize_context(request, request_state.context_chunk_size):
                raise RuntimeError(
                    f"failed to resize context for request {request_state.request_id}"
                )

        for request in getattr(scheduled_batch, "generation_requests", ()):
            request_state = _RequestSnapshot.from_request(request)
            if not self.try_allocate_generation(request):
                raise RuntimeError(
                    f"failed to allocate generation for request {request_state.request_id}"
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
            request_state = _RequestSnapshot.from_request(request)
            request_id = request_state.request_id
            if self._native_state is not None:
                updated = self._native_state.update_context(
                    request_id,
                    request_state.context_current_position,
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
            if state.capacity < request_state.context_current_position:
                if not self._resize_state(
                    state,
                    request_state.context_current_position + self.num_extra_kv_tokens,
                ):
                    raise ValueError(
                        f"failed to resize context history for request {request_id}"
                    )
            state.history_length = request_state.context_current_position
            state.committed_tokens = state.history_length
            self._write_host_block_offsets(request_id)

        for request in getattr(scheduled_batch, "generation_requests", ()):
            request_state = _RequestSnapshot.from_request(request)
            request_id = request_state.request_id
            if self._native_state is not None:
                updated = self._native_state.update_generation(
                    request_id,
                    max(request_state.max_beam_num_tokens, 0),
                    request_state.py_rewind_len,
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
            rewind_len = request_state.py_rewind_len
            history_length = max(request_state.max_beam_num_tokens - 1, 0)
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

        request_id = _RequestSnapshot.from_request(request).request_id
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
            request = _DummyRequest(
                request_id=request_id,
                py_request_id=request_id,
                is_first_context_chunk=True,
                context_current_position=token_num,
                context_chunk_size=token_num,
                context_remaining_length=token_num,
                max_beam_num_tokens=token_num,
                py_rewind_len=0,
                py_draft_tokens=[1] * max_num_draft_tokens if is_gen else [],
            )
            requests.append(request)

            if not prepare_resource:
                continue

            target = token_num + self.num_extra_kv_tokens + num_extra_decoding_steps
            if is_gen:
                target += max_num_draft_tokens + 1
            if self._native_state is not None:
                if not self._native_state.add_dummy_request(request_id, target):
                    raise RuntimeError(f"failed to allocate dummy request {request_id}")
                self._write_host_block_offsets_row(
                    request_id,
                    self._native_state.get_padded_block_row(request_id, BAD_PAGE_INDEX),
                )
            else:
                self.prepare_context(request)
                if not self._resize_state(self._request_state_for(request_id), target):
                    raise RuntimeError(f"failed to allocate dummy request {request_id}")
            if draft_kv_cache_manager is not None:
                if draft_kv_cache_manager._native_state is not None:
                    if not draft_kv_cache_manager._native_state.add_dummy_request(
                        request_id, target
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
                else:
                    draft_kv_cache_manager.prepare_context(request)
                    if not draft_kv_cache_manager._resize_state(
                        draft_kv_cache_manager._request_state_for(request_id), target
                    ):
                        raise RuntimeError(
                            f"failed to allocate draft dummy request {request_id}"
                        )

        return requests

    def get_kv_cache_stats(self) -> KvCacheStats:
        if self._native_state is not None:
            allocated_blocks = self.num_blocks - int(self._native_state.get_num_free_blocks())
        else:
            allocated_blocks = self.num_blocks - len(self._free_block_ids)
        block_bytes = (
            self.tokens_per_block
            * self.kv_factor
            * self.head_dim
            * sum(self.num_kv_heads_per_layer)
            * self._dtype_size_bytes()
        )
        allocated_bytes = allocated_blocks * block_bytes
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
        if self._native_state is not None and hasattr(self._native_state, "shutdown"):
            self._native_state.shutdown()
        self._request_state.clear()
        self._free_block_ids = list(range(self.num_blocks))
        self._iteration_events.clear()
        self._request_slots.clear()
        self._free_slots = list(range(self.max_num_sequences + 1))
        self._reset_host_block_offsets()
        self.layer_buffers.clear()
        self.primary_pool = None
        self._disagg_metadata = None
