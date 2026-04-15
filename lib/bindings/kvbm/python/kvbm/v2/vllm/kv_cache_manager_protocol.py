# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pinned typing.Protocol mirror of vLLM's ``KVCacheManager`` surface.

This file is the single source of truth for the kvbm v2 understanding of
vLLM's cache manager contract. Every public method and attribute is
transcribed verbatim from vLLM ``v0.11.1``'s
``vllm/v1/core/kv_cache_manager.py``, and the companion drift test
(``tests/test_vllm_kv_cache_manager_protocol.py``) asserts full signature
equality against the real class at runtime so that any upstream
add/rename/retype produces a loud CI failure instead of silent semantic
drift.

When bumping the pin:

1. Update ``PINNED_VLLM_VERSION`` below.
2. Re-run the drift test against the new vLLM release.
3. Apply any name/type/default changes here and in
   ``kv_cache_manager.py`` (the Python shim that implements the
   Protocol) and in the Rust binding (``src/v2/vllm/kv_cache_manager.rs``).

**Unsupported options.** The current kvbm Rust backing does not honor
``use_eagle``, ``dcp_world_size > 1``, ``num_encoder_tokens > 0``, or
``num_lookahead_tokens > 0``; the shim asserts on these and raises
``NotImplementedError``. The Protocol itself still accepts them — it is
meant to match vLLM, not to advertise kvbm capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import PrefixCacheStats
    from vllm.v1.request import Request


PINNED_VLLM_VERSION = "0.11.1"
"""The vLLM release this Protocol was transcribed from. Drift test
compares against the *installed* vLLM at runtime; this constant is used
only for human-readable error messages."""


@runtime_checkable
class VllmKvCacheManagerProtocol(Protocol):
    """Mirror of ``vllm.v1.core.kv_cache_manager.KVCacheManager``.

    Method signatures are transcribed verbatim. The drift test enforces
    exact parameter names, order, defaults, type annotations, and return
    types.
    """

    empty_kv_cache_blocks: "KVCacheBlocks"
    """Pre-built empty ``KVCacheBlocks`` with one empty group per kv
    cache group. Scheduler reads this as a shortcut for the "no cached
    blocks" case."""

    # -- Constructor ---------------------------------------------------

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
    ) -> None: ...

    # -- Core scheduling surface ---------------------------------------

    @property
    def usage(self) -> float:
        """Fraction of GPU KV cache blocks currently in use in ``[0, 1]``."""
        ...

    def make_prefix_cache_stats(self) -> "PrefixCacheStats | None":
        """Return-and-reset the accumulated prefix cache stats, or
        ``None`` if ``log_stats=False``."""
        ...

    def get_computed_blocks(
        self, request: "Request"
    ) -> "tuple[KVCacheBlocks, int]":
        """Prefix-match the request's token ids against the cache. Returns
        ``(matched_blocks, num_local_computed_tokens)``."""
        ...

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
        """Allocate the KV cache slots required to grow ``request`` by
        ``num_new_tokens`` tokens. Returns ``None`` if there is not
        enough free KV cache to satisfy the allocation.

        ``delay_cache_blocks=True`` is vLLM's "don't register yet"
        signal used during P/D disaggregation — newly filled blocks
        stay staged and are registered only when the KV transfer for
        them completes. This maps directly onto kvbm-logical's
        ``LogicalBlockAssignments::stage`` / ``register`` split: we
        stage the blocks into the slot's assignments but defer the
        ``register_staged`` call until ``cache_blocks`` is invoked.
        """
        ...

    def free(self, request: "Request") -> None:
        """Release every block currently owned by ``request``. Safe to
        call on unknown request ids."""
        ...

    def reset_prefix_cache(self) -> bool:
        """Drop the entire prefix cache. Used by RLHF workflows after
        weight updates. Returns ``True`` on success."""
        ...

    def get_num_common_prefix_blocks(
        self, running_request_id: str
    ) -> list[int]:
        """Return the number of blocks shared by every running request,
        one count per KV cache group."""
        ...

    def take_events(self) -> "list[KVCacheEvent]":
        """Drain queued KV cache events (only populated when
        ``enable_kv_cache_events=True``)."""
        ...

    def get_blocks(self, request_id: str) -> "KVCacheBlocks":
        """Return the ``KVCacheBlocks`` currently owned by ``request_id``."""
        ...

    def get_block_ids(self, request_id: str) -> "tuple[list[int], ...]":
        """Return the block id lists owned by ``request_id``, one list
        per KV cache group."""
        ...

    def cache_blocks(
        self, request: "Request", num_computed_tokens: int
    ) -> None:
        """Register the blocks holding the first ``num_computed_tokens``
        tokens of ``request`` into the prefix cache. Called after a KV
        transfer finishes for a request that was allocated with
        ``delay_cache_blocks=True``."""
        ...

    def create_kv_cache_blocks(
        self, blocks: "tuple[list[KVCacheBlock], ...]"
    ) -> "KVCacheBlocks":
        """Wrap the given lists of ``KVCacheBlock`` (one per kv cache
        group) in a ``KVCacheBlocks`` dataclass."""
        ...
