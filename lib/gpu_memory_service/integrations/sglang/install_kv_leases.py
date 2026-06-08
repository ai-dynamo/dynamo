# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang token/page allocator integration for GMS KV block leases."""

from __future__ import annotations

import logging
from typing import Callable

from gpu_memory_service.integrations.common.kv_lease_client import (
    GMSKVLeaseClient,
    KVLease,
    KVLeaseClient,
    kv_leases_enabled,
    log_lease_pressure,
    resolve_lease_device,
)

logger = logging.getLogger(__name__)

_patched = False
_factory: Callable[[object, int], KVLeaseClient] | None = None
_STATE: dict[int, dict[str, object]] = {}


def install(factory: Callable[[object, int], KVLeaseClient] | None = None) -> bool:
    global _patched, _factory
    if factory is not None:
        _factory = factory
    if _patched:
        return False
    if _factory is None and not kv_leases_enabled("sglang"):
        return False

    try:
        import torch
        from sglang.srt.mem_cache import allocator as alloc_mod
        from sglang.srt.utils import get_num_new_pages
    except Exception:  # noqa: BLE001
        logger.debug("[GMS-KVLease] SGLang allocator not importable", exc_info=True)
        return False

    Base = alloc_mod.BaseTokenToKVPoolAllocator
    Token = alloc_mod.TokenToKVPoolAllocator
    Paged = alloc_mod.PagedTokenToKVPoolAllocator

    orig_base_init = Base.__init__
    orig_token_alloc = Token.alloc
    orig_token_free = Token.free
    orig_token_clear = Token.clear
    orig_token_available = Token.available_size
    orig_paged_alloc = Paged.alloc
    orig_paged_alloc_extend = Paged.alloc_extend
    orig_paged_alloc_decode = Paged.alloc_decode
    orig_paged_free = Paged.free
    orig_paged_clear = Paged.clear
    orig_base_available = Base.available_size

    def _make_client(self, total_pages: int) -> KVLeaseClient:
        if _factory is not None:
            return _factory(self, total_pages)
        device_idx = resolve_lease_device("GMS_SGLANG_KV_LEASE_DEVICE")
        suffix = (
            f"{self.__class__.__name__}:size{int(self.size)}:page{int(self.page_size)}"
        )
        return GMSKVLeaseClient.from_env(
            "sglang",
            device_idx,
            total_blocks=total_pages + 1,
            namespace_suffix=suffix,
            reserved_blocks=[0],
        )

    def _state(self) -> dict[str, object] | None:
        return _STATE.get(id(self))

    def _safe_free_count(client: KVLeaseClient) -> int:
        try:
            return int(client.free_count())
        except Exception:  # noqa: BLE001
            logger.debug("[GMS-KVLease] SGLang free-count read failed", exc_info=True)
            return -1

    def _pages_to_list(pages) -> list[int]:
        if pages is None:
            return []
        if hasattr(pages, "numel") and int(pages.numel()) == 0:
            return []
        return [int(x) for x in pages.detach().cpu().tolist()]

    def _page_tensor(self, pages: list[int]):
        return torch.tensor(
            pages,
            dtype=self.free_pages.dtype,
            device=self.free_pages.device,
        )

    def _prepend_pages(self, pages: list[int]) -> None:
        if not pages:
            return
        self.free_pages = torch.cat((_page_tensor(self, pages), self.free_pages))

    def _deprioritize_pages(self, pages: list[int]) -> None:
        if not pages:
            return
        page_tensor = _page_tensor(self, pages)
        move_mask = torch.isin(self.free_pages, page_tensor)
        moved = self.free_pages[move_mask]
        if moved.numel() == 0:
            return
        self.free_pages = torch.cat((self.free_pages[~move_mask], moved))

    def _max_lease_attempts(num_pages: int, local_free: int) -> int:
        if num_pages <= 0:
            return 1
        return max(1, min(8, (int(local_free) + int(num_pages) - 1) // int(num_pages)))

    def _record_leases(st: dict[str, object], leases: list[KVLease]) -> None:
        lease_map = st["leases_by_page"]
        assert isinstance(lease_map, dict)
        for lease in leases:
            lease_map[int(lease.block_id)] = lease

    def _lease_pages(
        self,
        pages: list[int],
        *,
        local_free: int,
        operation: str,
    ) -> bool:
        st = _state(self)
        if st is None:
            return False
        client = st["client"]
        assert isinstance(client, GMSKVLeaseClient) or hasattr(client, "acquire")
        if not pages:
            return True
        try:
            leases = client.acquire(
                len(pages),
                preferred_blocks=pages,
                strict_preferred=True,
            )
        except Exception as exc:  # noqa: BLE001
            log_lease_pressure(
                logger,
                f"sglang:{getattr(client, 'namespace', '?')}:acquire-error",
                "[GMS-KVLease] SGLang lease acquire failed",
                namespace=getattr(client, "namespace", "?"),
                owner_id=getattr(client, "owner_id", "?"),
                operation=operation,
                requested=len(pages),
                local_free=local_free,
                shared_free=_safe_free_count(client),
                preferred_count=len(pages),
                error=type(exc).__name__,
            )
            logger.debug("[GMS-KVLease] SGLang lease acquire failed", exc_info=True)
            return False
        if len(leases) != len(pages):
            log_lease_pressure(
                logger,
                f"sglang:{getattr(client, 'namespace', '?')}:acquire-short",
                "[GMS-KVLease] SGLang lease acquire returned fewer pages than requested",
                namespace=getattr(client, "namespace", "?"),
                owner_id=getattr(client, "owner_id", "?"),
                operation=operation,
                requested=len(pages),
                returned=len(leases),
                shared_free=_safe_free_count(client),
                preferred_count=len(pages),
            )
            client.release(leases)
            return False
        _record_leases(st, leases)
        return True

    def _release_indices(self, free_index) -> None:
        st = _state(self)
        if st is None or free_index.numel() == 0:
            return
        if int(self.page_size) == 1:
            pages_tensor = free_index
        else:
            pages_tensor = torch.unique(free_index // int(self.page_size))
        pages = [int(x) for x in pages_tensor.detach().cpu().tolist() if int(x) > 0]
        lease_map = st["leases_by_page"]
        client = st["client"]
        assert isinstance(lease_map, dict)
        missing_pages = [page for page in pages if page not in lease_map]
        leases = [lease_map.pop(page) for page in pages if page in lease_map]
        if missing_pages:
            log_lease_pressure(
                logger,
                f"sglang:{getattr(client, 'namespace', '?')}:missing-release",
                "[GMS-KVLease] SGLang releasing pages without matching leases",
                namespace=getattr(client, "namespace", "?"),
                owner_id=getattr(client, "owner_id", "?"),
                missing_count=len(missing_pages),
                first_missing_page=missing_pages[0],
                active_leases=len(lease_map),
            )
        client.release(leases)

    def patched_base_init(self, *args, **kwargs):
        orig_base_init(self, *args, **kwargs)
        total_pages = int(self.size // self.page_size)
        client = _make_client(self, total_pages)
        _STATE[id(self)] = {"client": client, "leases_by_page": {}}
        logger.info(
            "[GMS-KVLease] SGLang allocator leases enabled namespace=%s owner=%s pages=%d",
            getattr(client, "namespace", "?"),
            getattr(client, "owner_id", "?"),
            total_pages,
        )

    def patched_token_available(self):
        return orig_token_available(self)

    def patched_base_available(self):
        return orig_base_available(self)

    def patched_token_alloc(self, need_size: int):
        st = _state(self)
        if st is None:
            return orig_token_alloc(self, need_size)
        local_free = len(self.free_pages)
        for _attempt in range(_max_lease_attempts(int(need_size), local_free)):
            out = orig_token_alloc(self, need_size)
            if out is None:
                return None
            pages = _pages_to_list(out)
            if _lease_pages(
                self,
                pages,
                local_free=local_free,
                operation="token_alloc",
            ):
                return out
            _prepend_pages(self, pages)
            _deprioritize_pages(self, pages)
        return None

    def patched_token_free(self, free_index):
        _release_indices(self, free_index)
        return orig_token_free(self, free_index)

    def patched_token_clear(self):
        st = _state(self)
        if st is not None:
            lease_map = st["leases_by_page"]
            client = st["client"]
            assert isinstance(lease_map, dict)
            outstanding = list(lease_map.values())
            if outstanding:
                logger.info(
                    "[GMS-KVLease] SGLang allocator clear releases %d outstanding leases namespace=%s owner=%s",
                    len(outstanding),
                    getattr(client, "namespace", "?"),
                    getattr(client, "owner_id", "?"),
                )
            client.release(outstanding)
            lease_map.clear()
        return orig_token_clear(self)

    def patched_paged_alloc(self, need_size: int):
        st = _state(self)
        if st is None:
            return orig_paged_alloc(self, need_size)
        num_pages = int(need_size) // int(self.page_size)
        local_free = len(self.free_pages)
        for _attempt in range(_max_lease_attempts(num_pages, local_free)):
            out = orig_paged_alloc(self, need_size)
            if out is None:
                return None
            pages = _pages_to_list(torch.unique(out // int(self.page_size)))
            if _lease_pages(
                self,
                pages,
                local_free=local_free,
                operation="paged_alloc",
            ):
                return out
            _prepend_pages(self, pages)
            _deprioritize_pages(self, pages)
        return None

    def patched_paged_alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens: int,
        num_new_pages: int | None = None,
    ):
        st = _state(self)
        if st is None:
            return orig_paged_alloc_extend(
                self,
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages=num_new_pages,
            )
        premerge_pages = extend_num_tokens // int(self.page_size) + len(prefix_lens) + 1
        if self.need_sort and premerge_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=int(self.page_size),
                prefix_lens=prefix_lens_cpu,
            )
        local_free = len(self.free_pages)
        for _attempt in range(_max_lease_attempts(int(num_new_pages), local_free)):
            new_pages = self.free_pages[: int(num_new_pages)].clone()
            out = orig_paged_alloc_extend(
                self,
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages=int(num_new_pages),
            )
            if out is None:
                return None
            pages = _pages_to_list(new_pages)
            if _lease_pages(
                self,
                pages,
                local_free=local_free,
                operation="paged_alloc_extend",
            ):
                return out
            _prepend_pages(self, pages)
            _deprioritize_pages(self, pages)
        return None

    def patched_paged_alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        st = _state(self)
        if st is None:
            return orig_paged_alloc_decode(self, seq_lens, seq_lens_cpu, last_loc)
        if self.need_sort and len(seq_lens) > len(self.free_pages):
            self.merge_and_sort_free()
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=int(self.page_size),
            decode=True,
        )
        local_free = len(self.free_pages)
        for _attempt in range(_max_lease_attempts(int(num_new_pages), local_free)):
            new_pages = self.free_pages[: int(num_new_pages)].clone()
            out = orig_paged_alloc_decode(self, seq_lens, seq_lens_cpu, last_loc)
            if out is None:
                return None
            pages = _pages_to_list(new_pages)
            if _lease_pages(
                self,
                pages,
                local_free=local_free,
                operation="paged_alloc_decode",
            ):
                return out
            _prepend_pages(self, pages)
            _deprioritize_pages(self, pages)
        return None

    def patched_paged_free(self, free_index):
        _release_indices(self, free_index)
        return orig_paged_free(self, free_index)

    def patched_paged_clear(self):
        st = _state(self)
        if st is not None:
            lease_map = st["leases_by_page"]
            client = st["client"]
            assert isinstance(lease_map, dict)
            outstanding = list(lease_map.values())
            if outstanding:
                logger.info(
                    "[GMS-KVLease] SGLang allocator clear releases %d outstanding leases namespace=%s owner=%s",
                    len(outstanding),
                    getattr(client, "namespace", "?"),
                    getattr(client, "owner_id", "?"),
                )
            client.release(outstanding)
            lease_map.clear()
        return orig_paged_clear(self)

    Base.__init__ = patched_base_init  # type: ignore[method-assign]
    Base.available_size = patched_base_available  # type: ignore[method-assign]
    Token.alloc = patched_token_alloc  # type: ignore[method-assign]
    Token.free = patched_token_free  # type: ignore[method-assign]
    Token.clear = patched_token_clear  # type: ignore[method-assign]
    Token.available_size = patched_token_available  # type: ignore[method-assign]
    Paged.alloc = patched_paged_alloc  # type: ignore[method-assign]
    Paged.alloc_extend = patched_paged_alloc_extend  # type: ignore[method-assign]
    Paged.alloc_decode = patched_paged_alloc_decode  # type: ignore[method-assign]
    Paged.free = patched_paged_free  # type: ignore[method-assign]
    Paged.clear = patched_paged_clear  # type: ignore[method-assign]

    _patched = True
    logger.info("[GMS-KVLease] patched SGLang token/page allocators")
    return True


if kv_leases_enabled("sglang"):
    try:
        install()
    except Exception:  # noqa: BLE001
        logger.exception("[GMS-KVLease] SGLang auto-install failed")
