# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Python shim around the Rust :class:`RustKvCacheManager`.

This module hosts :class:`RustKvCacheManager`, the kvbm v2
implementation of vLLM's ``KVCacheManager`` contract. It:

1. Implements :class:`VllmKvCacheManagerProtocol` — the drift test in
   ``tests/test_v2_vllm_kv_cache_manager_protocol.py`` guarantees this
   stays in sync with vLLM.
2. Owns a single PyO3 ``_RustKvCacheManager`` under the hood (see
   ``src/v2/vllm/kv_cache_manager.rs``). Block handles stay inside
   Rust via RAII; only ``block_id``\\ s and small metadata cross the
   FFI boundary.
3. Wraps the ``block_id`` lists handed back by Rust in vLLM's own
   ``KVCacheBlocks`` / ``KVCacheBlock`` dataclasses so vLLM's
   scheduler and KV connector code see exactly the types they expect.

The intended use site is the scheduler swap-in helper in
``kvbm.v2.vllm.schedulers.dynamo_kv_cache_manager.make_dynamo_scheduler``
— build a stock :class:`vllm.v1.core.sched.scheduler.Scheduler` and
then reassign ``scheduler.kv_cache_manager`` to one of these.

Features this shim **does not** implement (assert on construction):

* ``use_eagle``
* ``dcp_world_size > 1``
* ``num_encoder_tokens > 0`` on ``allocate_slots``
* ``num_lookahead_tokens > 0`` on ``allocate_slots``

These all raise ``NotImplementedError`` at the relevant call sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import PrefixCacheStats
    from vllm.v1.request import Request

from kvbm.v2.vllm.kv_cache_manager_protocol import VllmKvCacheManagerProtocol
from kvbm.v2.vllm.version_check import version_check

try:
    from kvbm._core import v2 as _kvbm_v2_core

    _RustKvCacheManagerCore: Any = _kvbm_v2_core.RustKvCacheManager
    _RUST_KV_CACHE_MANAGER_AVAILABLE = True
except (ImportError, AttributeError):
    _RustKvCacheManagerCore = None
    _RUST_KV_CACHE_MANAGER_AVAILABLE = False


class RustKvCacheManager(VllmKvCacheManagerProtocol):
    """vLLM-compatible KV cache manager backed by kvbm-logical.

    See the module docstring for scope.
    """

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        connector_leader: Any = None,
    ) -> None:
        # Enforce the kvbm.v2 vLLM version policy lazily, the same way
        # ``kvbm.v2.vllm.config`` does — this module is the gateway
        # for the Rust cache manager and should not let an unsupported
        # vllm slip through.
        version_check()

        if use_eagle:
            raise NotImplementedError(
                "RustKvCacheManager does not yet support eagle spec decode "
                "(use_eagle=True)"
            )
        if dcp_world_size > 1:
            raise NotImplementedError(
                "RustKvCacheManager does not yet support decode-context "
                "parallel (dcp_world_size > 1)"
            )
        if _RustKvCacheManagerCore is None:
            raise RuntimeError(
                "kvbm._core.v2.RustKvCacheManager is unavailable — "
                "kvbm was built without the v2 feature. Rebuild with "
                "`--features v2`."
            )

        # kvbm-logical only supports a single KV cache group today.
        # vLLM's KVCacheConfig lays groups out as
        # `kv_cache_groups: list[KVCacheGroupSpec]`; each group has a
        # `kv_cache_spec.block_size` and the whole config shares
        # `num_blocks`.
        num_groups = len(kv_cache_config.kv_cache_groups)
        if num_groups != 1:
            raise NotImplementedError(
                "RustKvCacheManager currently supports exactly one KV "
                f"cache group; vLLM config has {num_groups}"
            )
        group = kv_cache_config.kv_cache_groups[0]
        block_size = int(group.kv_cache_spec.block_size)
        total_blocks = int(kv_cache_config.num_blocks)

        self._kv_cache_config = kv_cache_config
        self._max_model_len = int(max_model_len)
        self._enable_caching = bool(enable_caching)
        self._log_stats = bool(log_stats)
        self._block_size = block_size
        self._total_blocks = total_blocks
        self._num_kv_cache_groups = num_groups

        # When a connector leader is provided, defer core construction
        # until first use so that initialize_workers() has time to run.
        self._connector_leader = connector_leader
        if connector_leader is not None:
            self._core = None
        else:
            self._core = _RustKvCacheManagerCore(
                total_blocks=total_blocks,
                block_size=block_size,
                enable_caching=self._enable_caching,
                log_stats=self._log_stats,
            )

        # Pre-built empty KVCacheBlocks, returned by `get_computed_blocks`
        # (and used by vLLM's scheduler as a shortcut in the
        # no-cached-blocks case via `scheduler.kv_cache_manager.empty_kv_cache_blocks`).
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks

        self.empty_kv_cache_blocks: "KVCacheBlocks" = KVCacheBlocks(
            tuple(() for _ in range(self._num_kv_cache_groups))
        )

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _ensure_core(self) -> None:
        """Lazily construct the Rust core when a connector leader is present."""
        if self._core is not None:
            return
        if self._connector_leader is not None:
            handle = self._connector_leader.ensure_g1_block_manager(
                self._total_blocks, self._block_size
            )
            self._core = _RustKvCacheManagerCore.from_g1_handle(
                handle, self._enable_caching, self._log_stats
            )
        else:
            self._core = _RustKvCacheManagerCore(
                total_blocks=self._total_blocks,
                block_size=self._block_size,
                enable_caching=self._enable_caching,
                log_stats=self._log_stats,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def usage(self) -> float:
        self._ensure_core()
        return float(self._core.usage())

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def make_prefix_cache_stats(self) -> "PrefixCacheStats | None":
        if not self._log_stats:
            return None
        self._ensure_core()
        from vllm.v1.metrics.stats import PrefixCacheStats

        hits, queries = self._core.take_prefix_cache_stats()
        stats = PrefixCacheStats()
        stats.requests = int(queries)
        stats.queries = int(queries) * self._block_size
        stats.hits = int(hits) * self._block_size
        return stats

    def get_computed_blocks(
        self, request: "Request"
    ) -> "tuple[KVCacheBlocks, int]":
        self._ensure_slot(request)
        block_ids, num_tokens = self._core.get_computed_blocks(
            request.request_id
        )
        return self._wrap_block_ids(block_ids), int(num_tokens)

    def allocate_slots(
        self,
        request: "Request",
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: "KVCacheBlocks | None" = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> "KVCacheBlocks | None":
        if num_lookahead_tokens:
            raise NotImplementedError(
                "RustKvCacheManager does not yet support "
                "num_lookahead_tokens > 0"
            )
        if num_encoder_tokens:
            raise NotImplementedError(
                "RustKvCacheManager does not yet support "
                "num_encoder_tokens > 0"
            )
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be > 0")

        self._ensure_slot(request)

        # Pick out the slice of tokens that is *newly* visible to the
        # cache manager this step. vLLM hands us `request.all_token_ids`
        # as an append-only list; everything beyond
        # `request.num_computed_tokens - num_new_computed_tokens` is
        # considered "about to be computed".
        #
        # We trust the scheduler's accounting here rather than holding
        # our own copy of the tokens in the Rust slot beyond the
        # initial prefill: the slot already has the full token stream
        # from `create_slot`, so the token slice only needs to cover
        # brand-new tokens the scheduler appended since the last call.
        prev_committed = max(
            0, request.num_computed_tokens - num_new_computed_tokens
        )
        new_committed = request.num_computed_tokens + num_new_tokens
        if new_committed > len(request.all_token_ids):
            new_committed = len(request.all_token_ids)
        new_token_ids = list(request.all_token_ids[prev_committed:new_committed])

        new_block_ids = self._core.allocate_slots(
            request_id=request.request_id,
            new_token_ids=new_token_ids,
            num_new_tokens=int(num_new_tokens),
            num_new_computed_tokens=int(num_new_computed_tokens),
            delay_cache_blocks=bool(delay_cache_blocks),
        )
        if new_block_ids is None:
            return None
        return self._wrap_block_ids(new_block_ids)

    def free(self, request: "Request") -> None:
        self._ensure_core()
        self._core.free(request.request_id)

    def reset_prefix_cache(self) -> bool:
        self._ensure_core()
        return bool(self._core.reset_prefix_cache())

    def get_num_common_prefix_blocks(
        self, running_request_id: str
    ) -> list[int]:
        # Mirrors vLLM's ``FullAttentionManager.get_num_common_prefix_blocks``
        # (vllm/v1/core/single_type_kv_cache_manager.py): walk the
        # running request's assigned blocks in order, count blocks
        # whose ref_cnt equals the number of tracked requests, stop
        # on the first non-match.
        #
        # kvbm-logical only ever hands the *same* block id back to
        # two different slots when they prefix-matched the same
        # registry entry, so "shared by every request" collapses to
        # "present in every slot's owned-id list" — see the Rust
        # implementation for details. Returned as a list with one
        # entry per kv cache group (kvbm currently supports exactly
        # one group).
        self._ensure_core()
        count = int(self._core.get_num_common_prefix_blocks(running_request_id))
        return [count] * self._num_kv_cache_groups

    def take_events(self) -> "list[KVCacheEvent]":
        self._ensure_core()
        return list(self._core.take_events())

    def get_blocks(self, request_id: str) -> "KVCacheBlocks":
        self._ensure_core()
        return self._wrap_block_ids(self._core.get_block_ids(request_id))

    def get_block_ids(self, request_id: str) -> "tuple[list[int], ...]":
        self._ensure_core()
        ids = list(self._core.get_block_ids(request_id))
        # vLLM returns one list per kv cache group. We only support one.
        return (ids,)

    def cache_blocks(
        self, request: "Request", num_computed_tokens: int
    ) -> None:
        self._ensure_core()
        self._core.cache_blocks(request.request_id, int(num_computed_tokens))

    def create_kv_cache_blocks(
        self, blocks: "tuple[list[Any], ...]"
    ) -> "KVCacheBlocks":
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks

        return KVCacheBlocks(blocks=blocks)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_slot(self, request: "Request") -> None:
        """Create a Rust-side slot for this request if we haven't seen
        it before. Idempotent."""
        self._ensure_core()
        if self._core.has_slot(request.request_id):
            return
        salt_hash = 0  # v1 computes this from lora + cache salt; TODO
        self._core.create_slot(
            request_id=request.request_id,
            tokens=list(request.all_token_ids),
            salt_hash=salt_hash,
            max_output_tokens=max(
                1, self._max_model_len - len(request.all_token_ids)
            ),
        )

    def _wrap_block_ids(self, block_ids: "list[int]") -> "KVCacheBlocks":
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks
        from vllm.v1.core.kv_cache_utils import KVCacheBlock

        blocks = tuple(KVCacheBlock(block_id=int(bid)) for bid in block_ids)
        # One group only.
        return KVCacheBlocks(blocks=(blocks,))


__all__ = ["RustKvCacheManager", "_RUST_KV_CACHE_MANAGER_AVAILABLE"]
