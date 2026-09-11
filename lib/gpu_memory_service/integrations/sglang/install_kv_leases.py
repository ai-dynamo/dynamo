# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang token/page allocator integration for GMS KV block leases."""

from __future__ import annotations

import logging
from collections.abc import Callable

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


def retain_hbm_indices(allocator, indices) -> list[KVLease]:
    """Keep allocator pages leased after SGLang returns them locally."""
    import torch

    st = _STATE.get(id(allocator))
    if st is None or indices is None or int(indices.numel()) == 0:
        return []
    page_size = int(allocator.page_size)
    pages = torch.unique(indices // page_size)
    page_ids = [int(value) for value in pages.detach().cpu().tolist()]
    lease_map = st["leases_by_page"]
    retained = st["retained_pages"]
    client = st["client"]
    assert isinstance(lease_map, dict) and isinstance(retained, set)
    leases = [lease_map[page] for page in page_ids if page in lease_map]
    if len(leases) != len(page_ids):
        return []
    client.seal(leases)
    retained.update(page_ids)
    return leases


def adopt_hbm_pages(allocator, pages: list[int], generations: list[int]):
    """Atomically transfer exact preserved pages into SGLang native state."""
    import torch

    st = _STATE.get(id(allocator))
    if st is None or len(pages) != len(generations) or len(set(pages)) != len(pages):
        return None, []
    page_size = int(allocator.page_size)
    if allocator.need_sort:
        allocator.merge_and_sort_free()
    page_tensor = torch.tensor(
        pages, dtype=allocator.free_pages.dtype, device=allocator.free_pages.device
    )
    total_pages = getattr(allocator, "num_pages", None)
    if total_pages is None and hasattr(allocator, "size"):
        total_pages = int(allocator.size) // page_size
    if total_pages is None:
        # Test doubles and older adapters may not expose the pool size. The
        # production paged allocator exposes ``num_pages``, so this compatibility
        # fallback is not on the failover path.
        known_pages = allocator.free_pages.detach().cpu().tolist()
        total_pages = max((*map(int, known_pages), *pages), default=0)
    total_pages = int(total_pages)
    if any(page < 0 or page > total_pages for page in pages):
        return None, []

    # Page IDs are dense in [0, num_pages]. Indexing a bitmap preserves the
    # allocator's existing free-page order and avoids ``torch.isin``'s costly
    # first-use CUDA path. ``_gms_paged_clear`` pre-warms these operators so
    # takeover pays only the small steady-state selection cost.
    requested = torch.zeros(
        total_pages + 1, dtype=torch.bool, device=allocator.free_pages.device
    )
    requested[page_tensor] = True
    selected = requested[allocator.free_pages]
    if int(selected.sum().item()) != len(pages):
        return None, []
    client = st["client"]
    old = [KVLease(page, generation) for page, generation in zip(pages, generations)]
    acquired = client.adopt(old)
    if [int(lease.block_id) for lease in acquired] != pages:
        client.release(acquired)
        return None, []
    lease_map = st["leases_by_page"]
    retained = st["retained_pages"]
    assert isinstance(lease_map, dict) and isinstance(retained, set)
    original_free_pages = allocator.free_pages
    try:
        allocator.free_pages = original_free_pages[~selected]
        for lease in acquired:
            lease_map[int(lease.block_id)] = lease
            retained.discard(int(lease.block_id))
        offsets = torch.arange(page_size, device=allocator.free_pages.device)
        indices = (page_tensor[:, None] * page_size + offsets).reshape(-1)
        return indices, acquired
    except Exception:
        # The caller cannot roll back leases if this helper raises before
        # returning them. Restore the native allocator and shared ownership
        # together so a transient tensor failure cannot strand adopted pages.
        allocator.free_pages = original_free_pages
        _release_tracked_leases(st, acquired)
        for lease in acquired:
            retained.discard(int(lease.block_id))
        raise


def rollback_adopted_hbm_pages(allocator, leases: list[KVLease]) -> None:
    if not leases:
        return
    import torch

    st = _STATE.get(id(allocator))
    if st is None:
        return
    lease_map = st["leases_by_page"]
    retained = st["retained_pages"]
    assert isinstance(lease_map, dict) and isinstance(retained, set)
    pages = [int(lease.block_id) for lease in leases]
    _release_tracked_leases(st, leases)
    for page in pages:
        retained.discard(page)
    page_tensor = torch.tensor(
        pages, dtype=allocator.free_pages.dtype, device=allocator.free_pages.device
    )
    allocator.free_pages = torch.cat((page_tensor, allocator.free_pages))
    if allocator.need_sort:
        allocator.free_pages, _ = torch.sort(allocator.free_pages)


def _append_free_pages(allocator, pages: list[int]) -> None:
    """Return rejected local pages at lowest allocation priority."""
    if not pages:
        return
    import torch

    page_tensor = torch.tensor(
        pages,
        dtype=allocator.free_pages.dtype,
        device=allocator.free_pages.device,
    )
    allocator.free_pages = torch.cat((allocator.free_pages, page_tensor))


# Resolved by `install()` from the running SGLang build. Module globals rather
# than closure cells so every patched allocator method below can live at module
# scope, where it is importable, greppable and unit-testable.
torch = None
get_num_new_pages = None
_gms_token_allocator_class = None
_gms_paged_allocator_class = None
_native_token_allocator_class = None
_native_paged_allocator_class = None
orig_token_init = None
orig_token_alloc = None
orig_token_free = None
orig_token_clear = None
orig_paged_init = None
orig_paged_alloc = None
orig_paged_alloc_extend = None
orig_paged_alloc_decode = None
orig_paged_release_page_ids = None
orig_paged_clear = None


def _make_client(self, total_pages: int) -> KVLeaseClient:
    if _factory is not None:
        return _factory(self, total_pages)
    device_idx = resolve_lease_device("GMS_SGLANG_KV_LEASE_DEVICE")
    from gpu_memory_service.integrations.common.kv_lease_client import (
        default_kv_lease_namespace_suffix,
    )

    return GMSKVLeaseClient.from_env(
        "sglang",
        device_idx,
        total_blocks=total_pages + 1,
        namespace_suffix=default_kv_lease_namespace_suffix("sglang"),
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


def _restore_directory_victims(directory, victims: list[dict]) -> None:
    """Compensate a known destructive selection before leases are touched."""
    if not victims:
        return
    items = [
        {
            "content_hash": bytes(victim["content_hash"]),
            "engine_id": str(victim["engine_id"]),
            "slot_ids": [int(page) for page in victim["slot_ids"]],
            "generations": [int(generation) for generation in victim["generations"]],
            "tier": "hbm",
            "active": False,
        }
        for victim in victims
    ]
    if directory.publish(items) != len(items):
        raise RuntimeError("could not restore retired SGLang HBM directory entries")
    entries = directory.lookup_authoritative([item["content_hash"] for item in items])
    expected = [
        (item["engine_id"], item["slot_ids"], item["generations"]) for item in items
    ]
    observed = [
        None
        if entry is None
        else (entry.get("engine_id"), entry.get("slot_ids"), entry.get("generations"))
        for entry in entries
    ]
    if observed != expected:
        raise RuntimeError("restored SGLang HBM directory entries did not verify")


def _ensure_directory_capacity(self, required_pages: int) -> int:
    st = _state(self)
    directory = getattr(self, "_gms_kv_directory", None)
    if st is None or directory is None or not directory.authoritative:
        return 0
    client = st["client"]
    available = _safe_free_count(client)
    # The directory RPC is destructive: it retires otherwise reusable HBM
    # entries. If the ring cannot report its capacity, or reports enough free
    # leases, the acquire failure was not confirmed as capacity pressure.
    if available < 0 or available >= int(required_pages):
        return 0
    shortage = int(required_pages) - available
    # Retained means sealed, not evicted from the engine's native prefix tree.
    # Only native-free pages may be retired and handed to a future allocation.
    eligible = _pages_to_list(self.free_pages)
    if not eligible:
        return 0
    lease_map = st["leases_by_page"]
    retained = st["retained_pages"]
    assert isinstance(lease_map, dict) and isinstance(retained, set)
    victims = []
    victims_leases = []
    try:
        victims = directory.ensure_hbm_capacity(shortage, eligible_slot_ids=eligible)
        seen = set()
        for victim in victims:
            pages = victim["slot_ids"]
            generations = victim["generations"]
            if len(pages) != len(generations):
                raise RuntimeError("malformed GMS pressure victim")
            for page, generation in zip(pages, generations):
                lease = KVLease(int(page), int(generation))
                current = lease_map.get(lease.block_id)
                if lease.block_id not in eligible or lease.block_id in seen:
                    raise RuntimeError(
                        "GMS pressure victim is not uniquely native-free"
                    )
                if current is not None and (
                    current != lease or lease.block_id not in retained
                ):
                    raise RuntimeError(
                        "GMS pressure victim has divergent retained generation"
                    )
                seen.add(lease.block_id)
                victims_leases.append(lease)
    except Exception:
        # No lease moved yet. Republish every known removed directory record.
        try:
            _restore_directory_victims(directory, victims)
        except Exception as restore_error:
            raise RuntimeError(
                "could not compensate SGLang HBM capacity selection; "
                "retaining leases and failing closed"
            ) from restore_error
        raise
    if not victims_leases:
        return 0
    # Exact-generation adoption validates foreign preserved pages too. It never
    # makes bytes FREE; failure leaves undiscoverable leases for recovery fencing.
    releases = client.adopt(victims_leases)
    if [lease.block_id for lease in releases] != [
        lease.block_id for lease in victims_leases
    ]:
        raise RuntimeError("GMS pressure victim generation validation failed")
    client.release(releases)
    for lease in releases:
        lease_map.pop(lease.block_id, None)
        retained.discard(lease.block_id)
    return len(releases)


def _pages_to_list(pages) -> list[int]:
    if pages is None:
        return []
    if hasattr(pages, "numel") and int(pages.numel()) == 0:
        return []
    return [int(x) for x in pages.detach().cpu().tolist()]


def _record_leases(st: dict[str, object], leases: list[KVLease]) -> None:
    lease_map = st["leases_by_page"]
    assert isinstance(lease_map, dict)
    for lease in leases:
        lease_map[int(lease.block_id)] = lease


def _release_tracked_leases(st: dict[str, object], leases: list[KVLease]) -> None:
    if not leases:
        return
    lease_map = st["leases_by_page"]
    assert isinstance(lease_map, dict)

    st["client"].release(leases)
    for lease in leases:
        page = int(lease.block_id)
        if lease_map.get(page) == lease:
            lease_map.pop(page)


def _rollback_reserved_pages(st: dict[str, object], leases: list[KVLease]) -> None:
    _release_tracked_leases(st, leases)


def _reserve_pages(
    self,
    pages: list[int],
    *,
    local_free: int,
    operation: str,
) -> list[KVLease] | None:
    """Lease pages before mutating SGLang's allocator.

    The ring prefers SGLang's next pages but may return another shared-free
    page when a preserved directory entry occupies that slot. Only that rare
    fallback reorders ``free_pages``; the normal path remains one local CAS.
    """

    st = _state(self)
    if st is None:
        return None
    client = st["client"]
    assert isinstance(client, GMSKVLeaseClient) or hasattr(client, "acquire")
    if not pages:
        return []
    try:
        leases = client.acquire(
            len(pages),
            preferred_blocks=pages,
            strict_preferred=False,
        )
    except Exception as initial_error:  # noqa: BLE001
        error: Exception | None = initial_error
        # The lease ring is the allocation authority. Reading its shared free
        # count and consulting the directory before every successful native
        # allocation adds work to SGLang's scheduler hot path. Reclaim
        # directory-owned capacity only after the optimistic CAS fails.
        if _ensure_directory_capacity(self, len(pages)):
            try:
                leases = client.acquire(
                    len(pages),
                    preferred_blocks=pages,
                    strict_preferred=False,
                )
            except Exception as retry_error:  # noqa: BLE001
                error = retry_error
            else:
                error = None
        if error is not None:
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
                error=type(error).__name__,
            )
            logger.debug("[GMS-KVLease] SGLang lease acquire failed: %s", error)
            return None
    if len(leases) != len(pages):
        client.release(leases)
        return None

    leased_pages = [int(lease.block_id) for lease in leases]
    if leased_pages != pages:
        page_tensor = torch.tensor(
            leased_pages,
            dtype=self.free_pages.dtype,
            device=self.free_pages.device,
        )
        selected = torch.isin(self.free_pages, page_tensor)
        if int(selected.sum().item()) != len(leased_pages):
            log_lease_pressure(
                logger,
                f"sglang:{getattr(client, 'namespace', '?')}:native-mismatch",
                "[GMS-KVLease] shared-free page absent from SGLang free list",
                namespace=getattr(client, "namespace", "?"),
                owner_id=getattr(client, "owner_id", "?"),
                operation=operation,
                requested=len(pages),
                leased=len(leased_pages),
            )
            client.release(leases)
            return None
        self.free_pages = torch.cat((page_tensor, self.free_pages[~selected]))

    _record_leases(st, leases)
    return leases


def _release_pages(self, *page_ids) -> None:
    st = _state(self)
    if st is None:
        return
    pages = [
        int(page)
        for ids in page_ids
        for page in ids.detach().cpu().tolist()
        if int(page) > 0
    ]
    lease_map = st["leases_by_page"]
    client = st["client"]
    retained = st["retained_pages"]
    assert isinstance(lease_map, dict) and isinstance(retained, set)
    released_pages = list(dict.fromkeys(page for page in pages if page not in retained))
    missing_pages = [page for page in released_pages if page not in lease_map]
    leases = [lease_map[page] for page in released_pages if page in lease_map]
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
    _release_tracked_leases(st, leases)


def _release_indices(self, free_index) -> None:
    if free_index.numel() == 0:
        return
    pages = (
        free_index
        if int(self.page_size) == 1
        else torch.unique(free_index // int(self.page_size))
    )
    _release_pages(self, pages)


def _initialize_allocator(self) -> None:
    total_pages = int(self.size // self.page_size)
    client = _make_client(self, total_pages)
    lease_map: dict[int, KVLease] = {}
    retained_pages: set[int] = set()
    _STATE[id(self)] = {
        "client": client,
        "leases_by_page": lease_map,
        "retained_pages": retained_pages,
    }
    self._gms_kv_lease_client = client
    self._gms_kv_leases_by_page = lease_map
    self._gms_retained_pages = retained_pages
    logger.info(
        "[GMS-KVLease] SGLang allocator leases enabled namespace=%s owner=%s pages=%d",
        getattr(client, "namespace", "?"),
        getattr(client, "owner_id", "?"),
        total_pages,
    )


def _gms_token_init(self, *args, **kwargs):
    orig_token_init(self, *args, **kwargs)
    _initialize_allocator(self)


def _gms_paged_init(self, *args, **kwargs):
    orig_paged_init(self, *args, **kwargs)
    _initialize_allocator(self)


def _gms_token_alloc(self, need_size: int):
    st = _state(self)
    if st is None:
        return orig_token_alloc(self, need_size)
    if self.need_sort and int(need_size) > len(self.free_pages):
        self.merge_and_sort_free()
    local_free = len(self.free_pages)
    if int(need_size) > local_free:
        return None
    pages = _pages_to_list(self.free_pages[: int(need_size)])
    leases = _reserve_pages(self, pages, local_free=local_free, operation="token_alloc")
    if leases is None:
        return None
    try:
        out = orig_token_alloc(self, need_size)
    except Exception:
        _rollback_reserved_pages(st, leases)
        raise
    if out is None:
        _rollback_reserved_pages(st, leases)
    return out


def _gms_token_free(self, free_index):
    # SGLang may defer the native free until ``free_group_end``. Do not make
    # the shared lease reusable before the native allocator has retired its
    # last reference to the page. Base.free_group_end sets ``free_group`` to
    # None and calls ``self.free`` again, which releases the lease exactly once.
    result = orig_token_free(self, free_index)
    if self.free_group is None:
        _release_indices(self, free_index)
    return result


def _gms_token_clear(self):
    st = _state(self)
    outstanding = []
    if st is not None:
        lease_map = st["leases_by_page"]
        assert isinstance(lease_map, dict)
        outstanding = list(lease_map.values())
        if outstanding:
            logger.info(
                "[GMS-KVLease] SGLang allocator clear releases %d outstanding leases namespace=%s owner=%s",
                len(outstanding),
                getattr(st["client"], "namespace", "?"),
                getattr(st["client"], "owner_id", "?"),
            )
    result = orig_token_clear(self)
    if st is not None:
        client = st["client"]
        retained = st["retained_pages"]
        assert isinstance(retained, set)
        # Publish the shared release only after the native reset has made the
        # old allocation unreachable. A failed release therefore fails the
        # clear closed instead of exposing a page while SGLang still owns it.
        client.release(outstanding)
        lease_map.clear()
        retained.clear()
    return result


def _gms_paged_alloc(self, need_size: int):
    st = _state(self)
    if st is None:
        return orig_paged_alloc(self, need_size)
    num_pages = int(need_size) // int(self.page_size)
    if self.need_sort and num_pages > len(self.free_pages):
        self.merge_and_sort_free()
    local_free = len(self.free_pages)
    if num_pages > local_free:
        return None
    pages = _pages_to_list(self.free_pages[:num_pages])
    leases = _reserve_pages(self, pages, local_free=local_free, operation="paged_alloc")
    if leases is None:
        return None
    try:
        out = orig_paged_alloc(self, need_size)
    except Exception:
        _rollback_reserved_pages(st, leases)
        raise
    if out is None:
        _rollback_reserved_pages(st, leases)
    return out


def _gms_paged_alloc_extend(
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
    num_new_pages = int(num_new_pages)
    local_free = len(self.free_pages)
    if num_new_pages > local_free:
        return None
    pages = _pages_to_list(self.free_pages[:num_new_pages])
    leases = _reserve_pages(
        self, pages, local_free=local_free, operation="paged_alloc_extend"
    )
    if leases is None:
        return None
    try:
        out = orig_paged_alloc_extend(
            self,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
    except Exception:
        _rollback_reserved_pages(st, leases)
        raise
    if out is None:
        _rollback_reserved_pages(st, leases)
    return out


def _gms_paged_alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
    st = _state(self)
    if st is None:
        return orig_paged_alloc_decode(self, seq_lens, seq_lens_cpu, last_loc)
    if self.need_sort and len(seq_lens) > len(self.free_pages):
        self.merge_and_sort_free()
    num_new_pages = int(
        get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=int(self.page_size),
            decode=True,
        )
    )
    local_free = len(self.free_pages)
    if num_new_pages > local_free:
        return None
    pages = _pages_to_list(self.free_pages[:num_new_pages])
    leases = _reserve_pages(
        self, pages, local_free=local_free, operation="paged_alloc_decode"
    )
    if leases is None:
        return None
    try:
        out = orig_paged_alloc_decode(self, seq_lens, seq_lens_cpu, last_loc)
    except Exception:
        _rollback_reserved_pages(st, leases)
        raise
    if out is None:
        _rollback_reserved_pages(st, leases)
    return out


def _gms_paged_release_page_ids(self, *page_ids):
    # SGLang funnels free(), free_segment(), and grouped frees through this
    # primitive. Publish the shared release only after native state owns the
    # pages again, so another engine can never lease a page still in use here.
    result = orig_paged_release_page_ids(self, *page_ids)
    _release_pages(self, *page_ids)
    return result


def _gms_paged_clear(self):
    st = _state(self)
    outstanding = []
    if st is not None:
        lease_map = st["leases_by_page"]
        assert isinstance(lease_map, dict)
        outstanding = list(lease_map.values())
        if outstanding:
            logger.info(
                "[GMS-KVLease] SGLang allocator clear releases %d outstanding leases namespace=%s owner=%s",
                len(outstanding),
                getattr(st["client"], "namespace", "?"),
                getattr(st["client"], "owner_id", "?"),
            )
    result = orig_paged_clear(self)
    if st is not None:
        client = st["client"]
        retained = st["retained_pages"]
        assert isinstance(retained, set)
        client.release(outstanding)
        lease_map.clear()
        retained.clear()

    # Base.__init__ has not created ``free_pages`` yet, while Paged.clear
    # has. Warm the exact-page adoption operators here, once per allocator,
    # so the first replayed request does not pay CUDA kernel setup latency.
    free_pages = self.free_pages
    if (
        not getattr(self, "_gms_adoption_warmed", False)
        and int(free_pages.numel()) > 0
        and getattr(free_pages, "is_cuda", False)
    ):
        sample = free_pages[: min(2, int(free_pages.numel()))]
        requested = torch.zeros(
            int(self.num_pages) + 1,
            dtype=torch.bool,
            device=free_pages.device,
        )
        requested[sample] = True
        selected = requested[free_pages]
        _ = int(selected.sum().item())
        _ = free_pages[~selected]
        offsets = torch.arange(int(self.page_size), device=free_pages.device)
        _ = (sample[:, None] * int(self.page_size) + offsets).reshape(-1)
        torch.cuda.synchronize(free_pages.device)
        self._gms_adoption_warmed = True
    return result


def _build_allocator_classes(token_class, paged_class):
    class GMSTokenToKVPoolAllocator(token_class):
        __init__ = _gms_token_init
        alloc = _gms_token_alloc
        free = _gms_token_free
        clear = _gms_token_clear

    class GMSPagedTokenToKVPoolAllocator(paged_class):
        __init__ = _gms_paged_init
        alloc = _gms_paged_alloc
        alloc_extend = _gms_paged_alloc_extend
        alloc_decode = _gms_paged_alloc_decode
        _release_page_ids = _gms_paged_release_page_ids
        clear = _gms_paged_clear

    GMSTokenToKVPoolAllocator.__name__ = "GMSTokenToKVPoolAllocator"
    GMSTokenToKVPoolAllocator.__qualname__ = "GMSTokenToKVPoolAllocator"
    GMSPagedTokenToKVPoolAllocator.__name__ = "GMSPagedTokenToKVPoolAllocator"
    GMSPagedTokenToKVPoolAllocator.__qualname__ = "GMSPagedTokenToKVPoolAllocator"
    return GMSTokenToKVPoolAllocator, GMSPagedTokenToKVPoolAllocator


def install(factory: Callable[[object, int], KVLeaseClient] | None = None) -> bool:
    global _patched, _factory, torch, get_num_new_pages
    global _gms_token_allocator_class, _gms_paged_allocator_class
    global _native_token_allocator_class, _native_paged_allocator_class
    global orig_token_init, orig_token_alloc, orig_token_free, orig_token_clear
    global orig_paged_init, orig_paged_alloc, orig_paged_alloc_extend
    global orig_paged_alloc_decode, orig_paged_release_page_ids, orig_paged_clear
    if factory is not None:
        _factory = factory
    if _patched:
        return False
    if _factory is None and not kv_leases_enabled("sglang"):
        return False

    try:
        import torch
        from sglang.srt.mem_cache import allocator as alloc_mod
        from sglang.srt.mem_cache import kv_cache_configurator
        from sglang.srt.utils import get_num_new_pages
    except Exception:  # noqa: BLE001
        logger.debug("[GMS-KVLease] SGLang allocator not importable", exc_info=True)
        return False

    Token = alloc_mod.TokenToKVPoolAllocator
    Paged = alloc_mod.PagedTokenToKVPoolAllocator
    _native_token_allocator_class = Token
    _native_paged_allocator_class = Paged
    orig_token_init = Token.__init__
    orig_token_alloc = Token.alloc
    orig_token_free = Token.free
    orig_token_clear = Token.clear
    orig_paged_init = Paged.__init__
    orig_paged_alloc = Paged.alloc
    orig_paged_alloc_extend = Paged.alloc_extend
    orig_paged_alloc_decode = Paged.alloc_decode
    orig_paged_release_page_ids = Paged._release_page_ids
    orig_paged_clear = Paged.clear

    _gms_token_allocator_class, _gms_paged_allocator_class = _build_allocator_classes(
        Token, Paged
    )
    # The configurator imported these names directly. Rebinding its two
    # construction references installs lease-aware subclasses without
    # modifying any SGLang class or method globally.
    kv_cache_configurator.TokenToKVPoolAllocator = _gms_token_allocator_class
    kv_cache_configurator.PagedTokenToKVPoolAllocator = _gms_paged_allocator_class

    _patched = True
    logger.info("[GMS-KVLease] installed SGLang lease-aware allocator subclasses")
    return True


def lease_hooks_installed() -> bool:
    """Verify the live SGLang allocator construction bindings."""
    try:
        from sglang.srt.mem_cache import kv_cache_configurator
    except Exception:  # noqa: BLE001
        return False

    return bool(
        _gms_token_allocator_class is not None
        and _gms_paged_allocator_class is not None
        and _native_token_allocator_class is not None
        and _native_paged_allocator_class is not None
        and kv_cache_configurator.TokenToKVPoolAllocator is _gms_token_allocator_class
        and kv_cache_configurator.PagedTokenToKVPoolAllocator
        is _gms_paged_allocator_class
        and issubclass(_gms_token_allocator_class, _native_token_allocator_class)
        and issubclass(_gms_paged_allocator_class, _native_paged_allocator_class)
        and _native_token_allocator_class.alloc is orig_token_alloc
        and _native_paged_allocator_class.alloc is orig_paged_alloc
    )


if kv_leases_enabled("sglang"):
    try:
        install()
    except Exception:  # noqa: BLE001
        logger.exception("[GMS-KVLease] SGLang auto-install failed")
