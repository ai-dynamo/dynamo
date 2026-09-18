# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang token/page allocator integration for GMS KV block leases."""

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Callable

from gpu_memory_service.integrations.common.kv_lease_client import (
    GMSKVLeaseClient,
    KVLease,
    KVLeaseClient,
    kv_leases_enabled,
    log_lease_pressure,
    resolve_lease_device,
)
from gpu_memory_service.integrations.common.process_lifecycle import (
    arm_parent_death_signal,
)

logger = logging.getLogger(__name__)

_patched = False
_factory: Callable[[object, int], KVLeaseClient] | None = None
_STATE: dict[int, dict[str, object]] = {}
_TP_RESERVATION_ATTEMPTS = 4


def retain_hbm_pages(allocator, page_ids: list[int]) -> list[KVLease]:
    """Seal validated CPU page IDs without another GPU index round trip."""
    st = _STATE.get(id(allocator))
    if st is None or not page_ids:
        return []
    if len(set(page_ids)) != len(page_ids) or any(page <= 0 for page in page_ids):
        return []
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


def demote_hbm_pages_local(allocator, page_ids: list[int]) -> list[KVLease]:
    """Make exact sealed pages locally writable without a directory RPC.

    The caller must publish generation-conditional directory tombstones for
    the returned old generations before making any client-visible promise. A
    reader racing that publication still rejects the stale directory
    generation against the already-advanced lease ring.
    """
    st = _STATE.get(id(allocator))
    if st is None or not page_ids:
        return []
    pages = [int(page) for page in page_ids]
    if len(set(pages)) != len(pages):
        raise RuntimeError("duplicate SGLang HBM demotion page")
    lease_map = st["leases_by_page"]
    retained = st["retained_pages"]
    client = st["client"]
    assert isinstance(lease_map, dict) and isinstance(retained, set)
    old = [lease_map[page] for page in pages if page in retained and page in lease_map]
    if len(old) != len(pages):
        raise RuntimeError("SGLang HBM demotion page is not retained")
    successors = client.adopt(old)
    if [int(lease.block_id) for lease in successors] != pages:
        raise RuntimeError("SGLang HBM local demotion failed")
    for lease in successors:
        lease_map[int(lease.block_id)] = lease
        retained.discard(int(lease.block_id))
    return old


def adopt_hbm_pages(allocator, pages: list[int], generations: list[int]):
    """Atomically transfer exact preserved pages into SGLang native state."""
    import torch

    st = _STATE.get(id(allocator))
    if st is None or len(pages) != len(generations) or len(set(pages)) != len(pages):
        return None, []
    page_size = int(allocator.page_size)
    if allocator.need_sort:
        allocator.merge_and_sort_free()
        st["tp_reservation_aligned"] = False
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
    if any(page <= 0 or page > total_pages for page in pages):
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
    hidden = st.get("exclusive_hidden_pages")
    hidden_pages = hidden if isinstance(hidden, set) else set()
    selected_free = {
        int(page) for page in allocator.free_pages[selected].detach().cpu().tolist()
    }
    selected_hidden = set(pages).intersection(hidden_pages)
    if selected_free.union(selected_hidden) != set(pages):
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
    original_hidden_pages = set(selected_hidden)
    try:
        allocator.free_pages = original_free_pages[~selected]
        hidden_pages.difference_update(selected_hidden)
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
        hidden_pages.update(original_hidden_pages)
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
    exclusive = bool(st.get("exclusive_steady_state", False))
    cpu_free = st.get("cpu_free_pages")
    if exclusive and not isinstance(cpu_free, deque):
        raise RuntimeError("SGLang CPU free-page mirror is unavailable")
    if not exclusive:
        _release_tracked_leases(st, leases)
    for page in pages:
        retained.discard(page)
    page_tensor = torch.tensor(
        pages, dtype=allocator.free_pages.dtype, device=allocator.free_pages.device
    )
    allocator.free_pages = torch.cat((page_tensor, allocator.free_pages))
    st["tp_reservation_aligned"] = False
    if allocator.need_sort:
        allocator.free_pages, _ = torch.sort(allocator.free_pages)
    if exclusive:
        if allocator.need_sort:
            cpu_free.extend(pages)
            ordered = sorted(cpu_free)
            cpu_free.clear()
            cpu_free.extend(ordered)
        else:
            cpu_free.extendleft(reversed(pages))


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
orig_paged_merge_and_sort_free = None
orig_paged_clear = None
orig_schedule_alloc_for_extend = None
orig_schedule_alloc_for_decode = None


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


def hidden_recoverable_tokens(allocator) -> int:
    """Return externally recoverable capacity withheld from native allocation."""
    st = _STATE.get(id(allocator))
    if st is None:
        return 0
    hidden = st.get("exclusive_hidden_pages")
    if not isinstance(hidden, set):
        return 0
    return len(hidden) * int(getattr(allocator, "page_size", 1))


def activate_hidden_recovery_capacity(allocator, required_tokens: int) -> int:
    """Make a bounded batch of predecessor pages natively writable.

    A promoted SGLang process initially keeps exact-generation predecessor KV
    outside its native free list so matching requests can adopt it lazily. When
    native allocation reports a shortfall, retire cold directory candidates in
    one TP-agreed batch, adopt their leases for this writer, and expose the
    pages to SGLang. This confines reduced capacity to the recovery transition;
    the steady allocation path remains entirely native afterwards.
    """
    st = _state(allocator)
    directory = getattr(allocator, "_gms_kv_directory", None)
    cohort = getattr(allocator, "_gms_tp_consistency", None)
    hidden = None if st is None else st.get("exclusive_hidden_pages")
    if (
        st is None
        or int(required_tokens) <= 0
        or not st.get("exclusive_steady_state", False)
        or st.get("recovery_capacity_exhausted", False)
        or not isinstance(hidden, set)
        or not hidden
        or directory is None
        or not directory.authoritative
        or cohort is None
    ):
        return 0

    page_size = max(1, int(getattr(allocator, "page_size", 1)))
    required_pages = (int(required_tokens) + page_size - 1) // page_size
    total_pages = int(getattr(allocator, "size", 0)) // page_size
    batch_pages = min(len(hidden), max(required_pages, max(1, total_pages // 4)))
    eligible = sorted(int(page) for page in hidden)
    lease_map = st["leases_by_page"]
    retained = st["retained_pages"]
    client = st["client"]
    assert isinstance(lease_map, dict) and isinstance(retained, set)
    local_victims: list[dict] = []

    try:

        def select_and_validate():
            local_victims.extend(
                directory.ensure_hbm_capacity(batch_pages, eligible_slot_ids=eligible)
            )
            victims = _canonical_victims(local_victims)
            leases = []
            seen = set()
            for _content_hash, _engine_id, pages, generations in victims:
                if len(pages) != len(generations):
                    raise RuntimeError("malformed GMS recovery-capacity victim")
                for page, generation in zip(pages, generations):
                    lease = KVLease(int(page), int(generation))
                    if lease.block_id in seen or lease.block_id not in hidden:
                        raise RuntimeError(
                            "GMS recovery-capacity victim is not uniquely hidden"
                        )
                    current = lease_map.get(lease.block_id)
                    if current is not None and (
                        current != lease or lease.block_id not in retained
                    ):
                        raise RuntimeError("GMS recovery-capacity generation diverged")
                    seen.add(lease.block_id)
                    leases.append(lease)
            return leases, _victim_identity_digest(victims)

        victims = cohort.run_agreed_digest(
            "recovery-capacity:select-validate", select_and_validate
        )
    except Exception:
        try:
            _restore_directory_victims(directory, local_victims)
        except Exception as restore_error:
            raise RuntimeError(
                "could not compensate SGLang recovery-capacity selection; "
                "failing closed"
            ) from restore_error
        raise

    if not victims:
        # A page can remain temporarily unretirable while its directory entry
        # is claimed or ACTIVE. Retrying this RPC and its TP agreement from
        # every subsequent native eviction permanently taxes the recovered
        # engine. Once recovery pressure has reduced the tail to one page,
        # abandon that candidate for this process generation. The page stays
        # sealed and outside native free_pages, so this trades 1/N capacity for
        # a terminal, race-free transition back to SGLang's native hot path.
        if len(hidden) == 1:
            st["recovery_capacity_exhausted"] = True
            candidates = getattr(allocator, "_gms_recovery_candidates", None)
            if isinstance(candidates, set):
                candidates.clear()
            logger.info(
                "[GMS-KVLease] SGLang recovery transition complete with "
                "one sealed page withheld"
            )
        return 0

    def adopt_victims():
        successors = client.adopt(victims)
        if [lease.block_id for lease in successors] != [
            lease.block_id for lease in victims
        ]:
            raise RuntimeError("GMS recovery-capacity adoption failed")
        return successors

    cpu_free = st.get("cpu_free_pages")
    if not isinstance(cpu_free, deque):
        raise RuntimeError("SGLang CPU free-page mirror is unavailable")
    successors = cohort.run("recovery-capacity:adopt", adopt_victims)
    pages = [int(lease.block_id) for lease in successors]
    page_tensor = torch.tensor(
        pages, dtype=allocator.free_pages.dtype, device=allocator.free_pages.device
    )
    if torch.isin(page_tensor, allocator.get_all_free_pages()).any().item():
        raise RuntimeError("GMS recovery capacity overlaps native-free pages")
    allocator.free_pages = torch.cat((allocator.free_pages, page_tensor))
    cpu_free.extend(pages)
    _record_leases(st, successors)
    hidden.difference_update(pages)
    retained.difference_update(pages)
    candidates = getattr(allocator, "_gms_recovery_candidates", None)
    if isinstance(candidates, set):
        candidates.difference_update(
            bytes(victim["content_hash"]) for victim in local_victims
        )
    logger.info(
        "[GMS-KVLease] SGLang activated recovery capacity pages=%d "
        "remaining_candidates=%d",
        len(pages),
        len(hidden),
    )
    return len(pages) * page_size


def _agree_native_capacity(self, operation: str, required: int, available: int) -> None:
    """Coordinate only the exceptional native-capacity path.

    Successful allocations consume pages from an already-agreed reservation
    window. Repeating a Python-object collective for their identical local
    capacity values put network synchronization on every scheduler step.
    """
    cohort = getattr(self, "_gms_tp_consistency", None)
    if cohort is not None and cohort.enabled and int(required) > int(available):
        cohort.agree(f"{operation}:capacity", (int(required), int(available)))


def _safe_free_count(client: KVLeaseClient) -> int:
    try:
        return int(client.free_count())
    except Exception:
        logger.debug("[GMS-KVLease] SGLang free-count read failed", exc_info=True)
        return -1


def _canonical_victims(victims: list[dict]) -> list[tuple]:
    """Return an order-independent representation for TP agreement."""
    return sorted(
        (
            bytes(victim["content_hash"]),
            str(victim["engine_id"]),
            tuple(int(page) for page in victim["slot_ids"]),
            tuple(int(generation) for generation in victim["generations"]),
        )
        for victim in victims
    )


def _victim_identity_digest(victims: list[tuple]) -> bytes:
    """Digest generation-independent victim identity for TP agreement."""
    from hashlib import sha256

    digest = sha256(b"sglang-gms-victims-v1\0")
    for content_hash, engine_id, pages, _generations in victims:
        engine = str(engine_id).encode("utf-8")
        digest.update(len(content_hash).to_bytes(4, "little"))
        digest.update(bytes(content_hash))
        digest.update(len(engine).to_bytes(4, "little"))
        digest.update(engine)
        digest.update(len(pages).to_bytes(4, "little"))
        for page in pages:
            digest.update(int(page).to_bytes(8, "little", signed=False))
    return digest.digest()


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
        (
            item["engine_id"],
            item["slot_ids"],
            item["generations"],
        )
        for item in items
    ]
    observed = [
        (
            None
            if entry is None
            else (
                entry.get("engine_id"),
                entry.get("slot_ids"),
                entry.get("generations"),
            )
        )
        for entry in entries
    ]
    if observed != expected:
        raise RuntimeError("restored SGLang HBM directory entries did not verify")


def _ensure_directory_capacity(self, required_pages: int) -> int:
    from gpu_memory_service.integrations.sglang.tp_consistency import TPConsistency

    cohort = getattr(self, "_gms_tp_consistency", None) or TPConsistency()
    st = _state(self)
    directory = getattr(self, "_gms_kv_directory", None)
    if st is None or directory is None or not directory.authoritative:
        return 0
    client = st["client"]
    available = _safe_free_count(client)
    if not cohort.common_prefix(
        "pressure:capacity", [(int(required_pages), available)]
    ):
        # A competing standby can temporarily skew rank-local free counts.
        # Contention is not permission to retire persistent bytes.
        return 0
    # The directory RPC is destructive: it retires otherwise reusable HBM
    # entries. If the ring cannot report its capacity, or reports enough free
    # leases, the acquire failure was not confirmed as capacity pressure.
    if available < 0 or available >= int(required_pages):
        return 0
    shortage = int(required_pages) - available
    # Retained means sealed, not necessarily evicted from SGLang's native tree.
    # Only pages native-free on every rank may be retired. This also protects
    # a live TP1 prefix against a lease-protected standby's warmup writes.
    _native_free, eligible = cohort.run_intersection(
        "pressure:native-free", lambda: _pages_to_list(self.free_pages)
    )
    if not eligible:
        return 0
    # These pages have ALREADY left every rank's native cache. Retire a small
    # batch so subsequent allocations consume a reservation window instead of
    # repeating the directory/TP transaction for every decode-page boundary.
    # Never evict a native-live page or reserve the whole persistent pool.
    total_pages = int(getattr(self, "size", 0)) // max(
        1, int(getattr(self, "page_size", 1))
    )
    batch = min(
        _tp_reservation_window_pages(), max(1, total_pages // 16), len(eligible)
    )
    shortage = max(shortage, batch)
    lease_map = st["leases_by_page"]
    retained = st["retained_pages"]
    assert isinstance(lease_map, dict) and isinstance(retained, set)
    local_victims = []
    eligible_set = set(eligible)

    try:

        def select_and_validate_victims():
            local_victims.extend(
                directory.ensure_hbm_capacity(shortage, eligible_slot_ids=eligible)
            )
            victims = _canonical_victims(local_victims)
            leases = []
            seen = set()
            for _content_hash, _engine_id, pages, generations in victims:
                if len(pages) != len(generations):
                    raise RuntimeError("malformed GMS pressure victim")
                for page, generation in zip(pages, generations):
                    lease = KVLease(int(page), int(generation))
                    current = lease_map.get(lease.block_id)
                    if lease.block_id not in eligible_set or lease.block_id in seen:
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
                    leases.append(lease)
            return leases, _victim_identity_digest(victims)

        # Selection is reversible until this vote. Agree local validation and
        # the logical victim set together before any rank advances a lease.
        # Rank-local generations are checked locally, not compared across TP.
        victims_leases = cohort.run_agreed_digest(
            "pressure:select-validate", select_and_validate_victims
        )
    except Exception:
        try:
            _restore_directory_victims(directory, local_victims)
        except Exception as restore_error:
            raise RuntimeError(
                "could not compensate SGLang HBM capacity selection; "
                "retaining leases and failing closed"
            ) from restore_error
        raise

    if not victims_leases:
        return 0

    def pin_victims():
        # Atomic exact-generation adoption verifies the rank-local ring too,
        # including foreign preserved pages absent from this engine's map.
        # It never makes bytes FREE. A peer failure therefore strands only
        # retired, undiscoverable pages until the cohort is fenced.
        leases = client.adopt(victims_leases)
        if [lease.block_id for lease in leases] != [
            lease.block_id for lease in victims_leases
        ]:
            raise RuntimeError("GMS pressure victim generation validation failed")
        return leases

    releases = cohort.run("pressure:pin", pin_victims)
    # Every rank has validated and pinned the identical retired victim set.
    # Never retry an ambiguous release or proceed to allocation before the vote.
    cohort.run("pressure:release", lambda: client.release(releases))
    for lease in releases:
        lease_map.pop(lease.block_id, None)
        retained.discard(lease.block_id)
    candidates = getattr(self, "_gms_recovery_candidates", None)
    if isinstance(candidates, set):
        candidates.difference_update(
            bytes(victim["content_hash"]) for victim in local_victims
        )
    return len(releases)


def maintain_steady_state_headroom(self) -> int:
    """Keep a bounded recoverable set and a local recyclable working set.

    Completed prefixes are sealed for crash recovery. If every page stays
    sealed until native allocation pressure, SGLang must synchronously retire
    directory entries before it can reuse those pages. Instead, periodically
    demote one cold batch from recoverable SEALED state to the current writer's
    LEASED state. The bytes remain in SGLang's native radix cache; only their
    cross-process recoverability is dropped. Native eviction can then recycle
    them through the already-owned TP reservation window without an RPC or TP
    vote on the allocation path.
    """
    st = _state(self)
    directory = getattr(self, "_gms_kv_directory", None)
    cohort = getattr(self, "_gms_tp_consistency", None)
    if (
        st is None
        or not st.get("steady_state", False)
        or directory is None
        or not directory.authoritative
        or cohort is None
    ):
        return 0

    lease_map = st["leases_by_page"]
    retained = st["retained_pages"]
    client = st["client"]
    assert isinstance(lease_map, dict) and isinstance(retained, set)
    total_pages = int(getattr(self, "size", 0)) // max(
        1, int(getattr(self, "page_size", 1))
    )
    if total_pages <= 2:
        return 0
    hidden = st.get("exclusive_hidden_pages")
    hidden_count = len(hidden) if isinstance(hidden, set) else 0
    high = max(1, total_pages * 3 // 4)
    low = max(1, total_pages // 2)
    recoverable = hidden_count + len(retained)
    if recoverable <= high:
        return 0
    # A promoted writer's predecessor pages are deliberately absent from the
    # native tree, but still consume the same recovery budget. Ignoring them
    # postpones demotion until native eviction has already selected a newly
    # sealed page, forcing one directory RPC and TP transaction per page on the
    # scheduler thread. Count both sets, while retiring only current-writer
    # pages that the native tree can eventually recycle.
    retire = min(len(retained), recoverable - low)
    eligible = sorted(int(page) for page in retained)
    local_victims: list[dict] = []

    try:

        def select_and_validate():
            # A promoted directory also contains the predecessor's sealed
            # records. Those are not writable until this engine explicitly
            # adopts them, and therefore are not candidates for local cache
            # maintenance. Restrict retirement to generations already
            # tracked by this allocator.
            local_victims.extend(
                directory.ensure_hbm_capacity(retire, eligible_slot_ids=eligible)
            )
            victims = _canonical_victims(local_victims)
            leases = []
            seen = set()
            for _content_hash, _engine_id, pages, generations in victims:
                if len(pages) != len(generations):
                    raise RuntimeError("malformed GMS headroom victim")
                for page, generation in zip(pages, generations):
                    lease = KVLease(int(page), int(generation))
                    if lease.block_id in seen:
                        raise RuntimeError("duplicate GMS headroom victim")
                    if (
                        lease.block_id not in retained
                        or lease_map.get(lease.block_id) != lease
                    ):
                        raise RuntimeError(
                            "GMS headroom victim has divergent retained generation"
                        )
                    seen.add(lease.block_id)
                    leases.append(lease)
            return leases, _victim_identity_digest(victims)

        victims = cohort.run_agreed_digest(
            "headroom:select-validate", select_and_validate
        )
    except Exception:
        try:
            _restore_directory_victims(directory, local_victims)
        except Exception as restore_error:
            raise RuntimeError(
                "could not compensate SGLang recovery-window demotion; "
                "retaining leases and failing closed"
            ) from restore_error
        raise

    if not victims:
        return 0

    def demote():
        successors = client.adopt(victims)
        if [lease.block_id for lease in successors] != [
            lease.block_id for lease in victims
        ]:
            raise RuntimeError("GMS recovery-window demotion failed")
        return successors

    successors = cohort.run("headroom:demote", demote)
    for lease in successors:
        lease_map[int(lease.block_id)] = lease
        retained.discard(int(lease.block_id))
    candidates = getattr(self, "_gms_recovery_candidates", None)
    if isinstance(candidates, set):
        candidates.difference_update(
            bytes(victim["content_hash"]) for victim in local_victims
        )
    return len(successors)


def enter_exclusive_steady_state(self) -> int:
    """Make native free_pages the sole writable-page authority.

    Before this transition, the reservation queue fences every allocation while
    the primary and standby establish ownership. Afterwards every page exposed
    through ``free_pages`` already has a current-writer LEASED generation, so
    SGLang can use its native allocator without repeating page-count work or TP
    coordination on every decode step. SEALED recovery pages are deliberately
    excluded and are made writable only by ordered directory retirement.
    """
    st = _state(self)
    if st is None:
        return 0
    if st.get("exclusive_steady_state", False):
        return len(self.free_pages)
    cohort = getattr(self, "_gms_tp_consistency", None)
    if cohort is None:
        from gpu_memory_service.integrations.sglang.tp_consistency import TPConsistency

        cohort = TPConsistency()
    if getattr(self, "need_sort", False):
        self.merge_and_sort_free()
    lease_map = st["leases_by_page"]
    retained = st["retained_pages"]
    assert isinstance(lease_map, dict) and isinstance(retained, set)

    # The reservation window may be empty before the first request. Acquire the
    # common native-free set once here instead of mistaking an empty window for
    # an empty writable pool. ``allow_partial`` deliberately respects any
    # headroom reserved for a standby and skips SEALED predecessor pages.
    _local_free, common_free = cohort.run_intersection(
        "steady:native-free",
        lambda: [page for page in _pages_to_list(self.free_pages) if page > 0],
    )
    candidates = [
        page for page in common_free if page not in lease_map and page not in retained
    ]
    acquired: list[KVLease] = []
    try:

        def acquire_common_pages():
            acquired.extend(
                st["client"].acquire(
                    len(candidates),
                    preferred_blocks=candidates,
                    allow_partial=True,
                    strict_preferred=True,
                )
            )
            pages = [int(lease.block_id) for lease in acquired]
            return acquired, pages

        new_leases = cohort.run_agreed("steady:acquire-free", acquire_common_pages)
    except Exception:
        # No native mutation has happened yet. Return a locally acquired batch
        # before failing the cohort closed; an ambiguous release still raises.
        st["client"].release(acquired)
        raise
    _record_leases(st, new_leases)

    local, common = cohort.run_intersection(
        "steady:writable-pages",
        lambda: [
            page
            for page in _pages_to_list(self.free_pages)
            if page in lease_map and page not in retained
        ],
    )
    if len(local) != len(common) or set(local) != set(common):
        raise RuntimeError("SGLang TP writable-page ownership diverged")
    st["exclusive_hidden_pages"] = set(common_free).difference(local)
    self.free_pages = torch.tensor(
        local, dtype=self.free_pages.dtype, device=self.free_pages.device
    )
    # Every page left in native free_pages is already leased by this writer.
    # SGLang's GPU list remains authoritative. Mirror only page IDs on the CPU
    # so completion can publish exact identities without synchronizing CUDA.
    st["cpu_free_pages"] = deque(local)
    cpu_staged = st.get("cpu_staged_pages")
    if isinstance(cpu_staged, list):
        cpu_staged.clear()
    queue = st.get("tp_reserved_pages")
    if isinstance(queue, list):
        queue.clear()
    st["tp_reservation_aligned"] = True
    st["exclusive_steady_state"] = True
    st["steady_state"] = True
    return len(local)


def _demote_exact_retained_pages(self, pages: list[int]) -> None:
    """Turn exact SEALED pages into current-writer LEASED pages before free."""
    st = _state(self)
    if st is None:
        return
    retained = st["retained_pages"]
    lease_map = st["leases_by_page"]
    assert isinstance(retained, set) and isinstance(lease_map, dict)
    targets = sorted(set(int(page) for page in pages).intersection(retained))
    if not targets:
        return
    directory = getattr(self, "_gms_kv_directory", None)
    cohort = getattr(self, "_gms_tp_consistency", None)
    if directory is None or not directory.authoritative or cohort is None:
        raise RuntimeError("cannot retire sealed SGLang pages without writer authority")
    local_victims: list[dict] = []

    try:

        def select_and_validate():
            local_victims.extend(
                directory.ensure_hbm_capacity(len(targets), eligible_slot_ids=targets)
            )
            victims = _canonical_victims(local_victims)
            leases = []
            selected = set()
            for _content_hash, _engine_id, slot_ids, generations in victims:
                if len(slot_ids) != len(generations):
                    raise RuntimeError("malformed GMS sealed-page retirement")
                for page, generation in zip(slot_ids, generations):
                    lease = KVLease(int(page), int(generation))
                    if page in selected or page not in targets:
                        raise RuntimeError("GMS retired an unexpected sealed page")
                    if lease_map.get(page) != lease or page not in retained:
                        raise RuntimeError("GMS sealed-page generation diverged")
                    selected.add(page)
                    leases.append(lease)
            if selected != set(targets):
                raise RuntimeError("GMS could not retire every sealed page")
            return leases, [victim[:3] for victim in victims]

        victims = cohort.run_agreed("free:retire-sealed", select_and_validate)
    except Exception:
        try:
            _restore_directory_victims(directory, local_victims)
        except Exception as restore_error:
            raise RuntimeError(
                "could not compensate sealed-page retirement; failing closed"
            ) from restore_error
        raise

    def demote():
        successors = st["client"].adopt(victims)
        successor_pages = [int(lease.block_id) for lease in successors]
        if len(successor_pages) != len(targets) or set(successor_pages) != set(targets):
            raise RuntimeError("GMS sealed-page demotion failed")
        return successors

    successors = cohort.run("free:demote-sealed", demote)
    for lease in successors:
        lease_map[int(lease.block_id)] = lease
        retained.discard(int(lease.block_id))
    candidates = getattr(self, "_gms_recovery_candidates", None)
    if isinstance(candidates, set):
        candidates.difference_update(
            bytes(victim["content_hash"]) for victim in local_victims
        )


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
    released_pages = {int(lease.block_id) for lease in leases}
    queue = st.get("tp_reserved_pages")
    if isinstance(queue, list):
        remaining = [page for page in queue if int(page) not in released_pages]
        if len(remaining) != len(queue):
            st["tp_reservation_aligned"] = False
        queue[:] = remaining
    for lease in leases:
        page = int(lease.block_id)
        if lease_map.get(page) == lease:
            lease_map.pop(page)


def _rollback_reserved_pages(st: dict[str, object], leases: list[KVLease]) -> None:
    if st.get("steady_state", False):
        queue = st.get("tp_reserved_pages")
        if isinstance(queue, list):
            returned = [int(lease.block_id) for lease in leases]
            queue[:0] = [page for page in returned if page not in queue]
        st["tp_reservation_aligned"] = False
        return
    st["tp_reservation_aligned"] = False
    _release_tracked_leases(st, leases)


def _reserve_tp_pages(
    self,
    pages: list[int],
    operation: str,
    *,
    count: int | None = None,
    reclaim: bool = True,
) -> list[KVLease] | None:
    """Hold leader-selected pages until every rank reserves the same layout.

    Contention with a standby produces a reversible NACK. All successful
    reservations are released before a bounded retry; native state changes
    only after unanimous acceptance. Generations fence each rank's own ring
    and directory and deliberately need not match across ranks.
    """
    st = _state(self)
    client = st["client"]
    cohort = self._gms_tp_consistency
    requested = len(pages) if count is None else int(count)
    if requested < 0 or requested > len(pages):
        raise ValueError("invalid SGLang TP lease reservation count")
    # Native allocator entries agreed on required capacity before arriving
    # here, so every rank takes this zero-page fast path together.
    if requested == 0:
        return []
    tried = set()
    leases = []
    reclaimed = False

    def release_candidate():
        nonlocal leases
        # Forget the batch before its release CAS or peer vote can fail.
        # An ambiguous release must never be retried: retain any stranded
        # ownership until this failed cohort is fenced and reclaimed.
        pending, leases = leases, []
        if pending:
            client.release(pending)

    for attempt in range(_TP_RESERVATION_ATTEMPTS):
        stage = f"{operation}:reserve:{attempt}"

        def choose_pages():
            nonlocal leases
            candidates = (
                pages
                if not tried
                else [
                    page
                    for page in _pages_to_list(self.free_pages)
                    if page not in tried
                ]
            )
            try:
                leases = client.acquire(
                    requested,
                    preferred_blocks=candidates,
                    strict_preferred=True,
                )
            except RuntimeError:
                return []
            return [int(lease.block_id) for lease in leases]

        try:
            chosen = cohort.leader_call(f"{stage}:choose", choose_pages)

            prepared_free_pages = None

            def reserve_peer(chosen=chosen):
                nonlocal leases, prepared_free_pages
                if len(chosen) != requested or len(set(chosen)) != len(chosen):
                    raise RuntimeError("SGLang GMS TP candidate unavailable")
                native_order = _pages_to_list(self.free_pages)
                native_free = set(native_order)
                if len(native_free) != len(native_order) or not set(chosen).issubset(
                    native_free
                ):
                    raise RuntimeError(
                        "SGLang GMS TP candidate is not uniquely native-free"
                    )
                if cohort._rank() != 0:
                    leases = client.acquire(
                        len(chosen), preferred_blocks=chosen, strict_preferred=True
                    )
                if [int(lease.block_id) for lease in leases] != chosen:
                    raise RuntimeError("SGLang GMS TP exact-page reservation failed")

                # Prepare without changing native state. The reservation vote
                # also covers allocation/validation failure of this tensor; no
                # second collective is needed to install an already prepared
                # value after unanimous acceptance.
                desired = list(st.get("tp_reserved_pages", [])) + chosen
                selected = set(desired)
                if len(selected) != len(desired) or not selected.issubset(native_free):
                    raise RuntimeError(
                        "SGLang TP reservation window is not native-free"
                    )
                prepared_free_pages = self.free_pages
                if native_order[: len(desired)] != desired:
                    prepared_free_pages = torch.tensor(
                        desired
                        + [page for page in native_order if page not in selected],
                        dtype=self.free_pages.dtype,
                        device=self.free_pages.device,
                    )

            accepted, _ = cohort.attempt(f"{stage}:vote", reserve_peer)
            if accepted:
                self.free_pages = prepared_free_pages
                _record_leases(st, leases)
                st["tp_reservation_aligned"] = True
                return leases
            cohort.run(f"{stage}:rollback", release_candidate)
        except Exception:
            # A failed collective or rollback is not ordinary contention.
            # No native write began; release our reservation and fail closed.
            release_candidate()
            raise
        # A failed peer vote only disproves the leader-selected pages. Keep the
        # rest of a candidate superset eligible for the bounded retry. Marking
        # the whole superset tried can manufacture OOM while many common free
        # pages remain.
        tried.update(chosen)
        if reclaim and not reclaimed:
            reclaimed = True
            if _ensure_directory_capacity(self, requested):
                tried.clear()
    return None


def _tp_reservation_window_pages() -> int:
    raw = os.environ.get("GMS_SGLANG_TP_LEASE_WINDOW_PAGES", "4096")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid GMS_SGLANG_TP_LEASE_WINDOW_PAGES=%r", raw)
        return 4096


def _consume_tp_reservation(self, count: int) -> list[KVLease] | None:
    """Take the next agreed pages without another cross-rank collective."""
    st = _state(self)
    if st is None:
        return None
    queue = st.setdefault("tp_reserved_pages", [])
    lease_map = st["leases_by_page"]
    assert isinstance(queue, list) and isinstance(lease_map, dict)
    if len(queue) < count:
        return None
    pages = [int(page) for page in queue[:count]]
    leases: list[KVLease] = []
    for page in pages:
        lease = lease_map.get(page)
        if lease is None:
            raise RuntimeError("SGLang TP reservation window lost a lease")
        leases.append(lease)

    if not st.get("tp_reservation_aligned", False):
        # Re-establish the invariant for the entire remaining window, not only
        # this allocation. Token frees and sorted merges can displace later
        # reservations too; marking the window aligned after moving only its
        # first entry would let a later allocation consume a different page.
        reserved = [int(page) for page in queue]
        current = _pages_to_list(self.free_pages[: len(reserved)])
        if current != reserved:
            page_tensor = torch.tensor(
                reserved, dtype=self.free_pages.dtype, device=self.free_pages.device
            )
            selected = torch.isin(self.free_pages, page_tensor)
            if int(selected.sum().item()) != len(reserved):
                raise RuntimeError(
                    "SGLang TP reservation window diverged from native free pages"
                )
            self.free_pages = torch.cat((page_tensor, self.free_pages[~selected]))
        st["tp_reservation_aligned"] = True
    del queue[:count]
    return leases


def _refill_tp_reservation(self, count: int, operation: str) -> list[KVLease] | None:
    """Amortize TP agreement over a deterministic window of free pages."""
    st = _state(self)
    if st is None:
        return None
    queue = st.setdefault("tp_reserved_pages", [])
    lease_map = st["leases_by_page"]
    retained = st["retained_pages"]
    assert isinstance(queue, list)
    assert isinstance(lease_map, dict) and isinstance(retained, set)

    cohort = self._gms_tp_consistency
    cohort.agree(f"{operation}:window-state", (int(count), tuple(queue)))
    needed = int(count) - len(queue)
    target = max(int(count), _tp_reservation_window_pages()) - len(queue)

    def available_candidates():
        # Rank-local retained maps may differ after a failed publication. Agree
        # the eligible set before taking branches or choosing a batch size.
        candidates, common = cohort.run_intersection(
            f"{operation}:window-free",
            lambda: [
                page
                for page in _pages_to_list(self.free_pages)
                if page > 0 and page not in lease_map and page not in retained
            ],
        )
        common = set(common)
        return [page for page in candidates if page in common]

    # A promoted shadow can have many pages that are native-free but still
    # SEALED in the shared ring. Confirm and retire bounded native-free capacity
    # before attempting a speculative window; otherwise four doomed TP votes
    # precede every capacity reclaim during the post-takeover transition.
    _ensure_directory_capacity(self, needed)
    candidates = available_candidates()
    if len(candidates) < needed:
        # A full retained cache still has evictable native-free pages. Reclaim
        # a bounded native-free batch, never the whole reservation window.
        if not _ensure_directory_capacity(self, needed):
            return None
        candidates = available_candidates()
        if len(candidates) < needed:
            return None
    # Speculation must not monopolize a small pool before a standby can warm
    # up. Leave half the eligible pages unreserved unless actual request demand
    # needs them; this is a batching limit, not a permanent capacity partition.
    target = max(needed, min(target, len(candidates) // 2))
    # Give the ring the whole common native-free candidate set. It can then
    # select ``target`` pages that are actually shared-free, rather than being
    # forced to use an arbitrary prefix that may consist of preserved pages.
    leases = _reserve_tp_pages(
        self,
        candidates,
        f"{operation}:window",
        count=target,
        reclaim=False,
    )
    if leases is None and target > needed:
        # One competing owner must not turn a large speculative window into
        # backpressure when the request itself needs only a few pages.
        leases = _reserve_tp_pages(self, candidates, operation, count=needed)
    elif leases is None and _ensure_directory_capacity(self, needed):
        candidates = available_candidates()
        leases = _reserve_tp_pages(self, candidates, operation, count=needed)
    if leases is None:
        return None
    queue.extend(int(lease.block_id) for lease in leases)
    # The reservation transaction agreed and aligned both the remaining window
    # and these new pages before reporting success.
    return _consume_tp_reservation(self, count)


def _reserve_pages(
    self,
    pages: list[int] | None,
    *,
    count: int | None = None,
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
    cohort = getattr(self, "_gms_tp_consistency", None)
    requested = len(pages) if count is None else int(count)
    if cohort is not None and cohort.enabled:
        leases = _consume_tp_reservation(self, requested)
        if leases is not None:
            return leases
        return _refill_tp_reservation(self, requested, operation)
    if pages is None:
        raise RuntimeError("single-rank SGLang reservation requires page IDs")
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
    if st.get("steady_state", False):
        # Recycle already-owned idle pages without a release/acquire round
        # trip. They remain LEASED, so promotion can still reclaim them after
        # a crash. SEALED retained pages stay excluded until ordered directory
        # retirement makes them writable again.
        queue = st.get("tp_reserved_pages")
        if isinstance(queue, list):
            queue.extend(
                page
                for page in released_pages
                if page in lease_map and page not in queue
            )
        return
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
    arm_parent_death_signal()
    total_pages = int(self.size // self.page_size)
    client = _make_client(self, total_pages)
    lease_map: dict[int, KVLease] = {}
    retained_pages: set[int] = set()
    _STATE[id(self)] = {
        "client": client,
        "leases_by_page": lease_map,
        "retained_pages": retained_pages,
        "tp_reserved_pages": [],
        "tp_reservation_aligned": False,
        "steady_state": False,
        "exclusive_steady_state": False,
        "exclusive_hidden_pages": set(),
        # Populated once the writer owns one stable native free-page set. The
        # mirror lets request-aware allocation record physical page ownership
        # without synchronizing SGLang's GPU allocator tensor back to Python.
        "cpu_free_pages": None,
        "cpu_staged_pages": [],
        # Finished-request frees can be derived from the request's CPU page
        # record.  Cache adapters append those page IDs here; the allocator
        # consumes them only when their count exactly matches the native
        # release.  Any disagreement falls back to reading the device tensor.
        "cpu_release_pages": [],
        "active_batch": None,
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
    if st is None or int(need_size) == 0:
        return orig_token_alloc(self, need_size)
    if self.need_sort and int(need_size) > len(self.free_pages):
        self.merge_and_sort_free()
        st["tp_reservation_aligned"] = False
    local_free = len(self.free_pages)
    _agree_native_capacity(self, "token_alloc", need_size, local_free)
    if int(need_size) > local_free:
        return None
    cohort = getattr(self, "_gms_tp_consistency", None)
    pages = (
        None
        if cohort is not None and cohort.enabled
        else _pages_to_list(self.free_pages[: int(need_size)])
    )
    leases = _reserve_pages(
        self,
        pages,
        count=int(need_size),
        local_free=local_free,
        operation="token_alloc",
    )
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
    st = _state(self)
    if st is not None:
        st["tp_reservation_aligned"] = False
    if self.free_group is None:
        _release_indices(self, free_index)
    return result


def _revoke_allocator_fast_path(st) -> None:
    """Invalidate every derived allocator view before native clear mutates pages."""
    st["exclusive_steady_state"] = False
    st["steady_state"] = False
    st["tp_reservation_aligned"] = False
    st["exclusive_hidden_pages"] = set()
    st["cpu_free_pages"] = None
    for name in ("cpu_staged_pages", "cpu_release_pages", "tp_reserved_pages"):
        values = st.get(name)
        if isinstance(values, list):
            values.clear()
    st["active_batch"] = None


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
    if st is not None:
        _revoke_allocator_fast_path(st)
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
        queue = st.get("tp_reserved_pages")
        if isinstance(queue, list):
            queue.clear()
        st["tp_reservation_aligned"] = False
    return result


def _gms_paged_alloc(self, need_size: int):
    st = _state(self)
    if st is None:
        return orig_paged_alloc(self, need_size)
    if st.get("exclusive_steady_state", False):
        num_pages = int(need_size) // int(self.page_size)
        if num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        result = orig_paged_alloc(self, need_size)
        if result is not None and num_pages:
            cpu_free = st.get("cpu_free_pages")
            if not isinstance(cpu_free, deque) or len(cpu_free) < num_pages:
                raise RuntimeError("SGLang CPU free-page mirror diverged")
            for _ in range(num_pages):
                cpu_free.popleft()
        return result
    num_pages = int(need_size) // int(self.page_size)
    if num_pages == 0:
        return orig_paged_alloc(self, need_size)
    if self.need_sort and num_pages > len(self.free_pages):
        self.merge_and_sort_free()
        st["tp_reservation_aligned"] = False
    local_free = len(self.free_pages)
    _agree_native_capacity(self, "paged_alloc", num_pages, local_free)
    if num_pages > local_free:
        return None
    cohort = getattr(self, "_gms_tp_consistency", None)
    pages = (
        None
        if cohort is not None and cohort.enabled
        else _pages_to_list(self.free_pages[:num_pages])
    )
    leases = _reserve_pages(
        self,
        pages,
        count=num_pages,
        local_free=local_free,
        operation="paged_alloc",
    )
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
    if st.get("exclusive_steady_state", False):
        premerge_pages = extend_num_tokens // int(self.page_size) + len(prefix_lens) + 1
        if self.need_sort and premerge_pages > len(self.free_pages):
            self.merge_and_sort_free()
        free_before = len(self.free_pages)
        result = orig_paged_alloc_extend(
            self,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        num_new_pages = free_before - len(self.free_pages)
        if result is not None and num_new_pages:
            cpu_free = st.get("cpu_free_pages")
            if not isinstance(cpu_free, deque) or len(cpu_free) < num_new_pages:
                raise RuntimeError("SGLang CPU free-page mirror diverged")
            selected = [cpu_free.popleft() for _ in range(num_new_pages)]
            _record_extend_pages(
                st, selected, prefix_lens_cpu, seq_lens_cpu, int(self.page_size)
            )
        return result
    premerge_pages = extend_num_tokens // int(self.page_size) + len(prefix_lens) + 1
    if self.need_sort and premerge_pages > len(self.free_pages):
        self.merge_and_sort_free()
        st["tp_reservation_aligned"] = False
    if num_new_pages is None:
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=int(self.page_size),
            prefix_lens=prefix_lens_cpu,
        )
    num_new_pages = int(num_new_pages)
    if num_new_pages == 0:
        return orig_paged_alloc_extend(
            self,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=0,
        )
    local_free = len(self.free_pages)
    _agree_native_capacity(self, "paged_alloc_extend", num_new_pages, local_free)
    if num_new_pages > local_free:
        return None
    cohort = getattr(self, "_gms_tp_consistency", None)
    pages = (
        None
        if cohort is not None and cohort.enabled
        else _pages_to_list(self.free_pages[:num_new_pages])
    )
    leases = _reserve_pages(
        self,
        pages,
        count=num_new_pages,
        local_free=local_free,
        operation="paged_alloc_extend",
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
    if st.get("exclusive_steady_state", False):
        if len(seq_lens) > len(self.free_pages):
            self.merge_and_sort_free()
        free_before = len(self.free_pages)
        result = orig_paged_alloc_decode(self, seq_lens, seq_lens_cpu, last_loc)
        num_new_pages = free_before - len(self.free_pages)
        if result is not None and num_new_pages:
            cpu_free = st.get("cpu_free_pages")
            if not isinstance(cpu_free, deque) or len(cpu_free) < num_new_pages:
                raise RuntimeError("SGLang CPU free-page mirror diverged")
            selected = [cpu_free.popleft() for _ in range(num_new_pages)]
            _record_decode_pages(st, selected, seq_lens_cpu, int(self.page_size))
        return result
    if self.need_sort and len(seq_lens) > len(self.free_pages):
        self.merge_and_sort_free()
        st["tp_reservation_aligned"] = False
    num_new_pages = int(
        get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=int(self.page_size),
            decode=True,
        )
    )
    if num_new_pages == 0:
        return orig_paged_alloc_decode(self, seq_lens, seq_lens_cpu, last_loc)
    local_free = len(self.free_pages)
    _agree_native_capacity(self, "paged_alloc_decode", num_new_pages, local_free)
    if num_new_pages > local_free:
        return None
    cohort = getattr(self, "_gms_tp_consistency", None)
    pages = (
        None
        if cohort is not None and cohort.enabled
        else _pages_to_list(self.free_pages[:num_new_pages])
    )
    leases = _reserve_pages(
        self,
        pages,
        count=num_new_pages,
        local_free=local_free,
        operation="paged_alloc_decode",
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
    st = _state(self)
    if st is not None and st.get("exclusive_steady_state", False):
        released = sum(int(values.numel()) for values in page_ids)
        hinted = st.get("cpu_release_pages")
        pages = []
        consumed = 0
        if isinstance(hinted, list):
            for group in hinted:
                if len(pages) + len(group) > released:
                    break
                pages.extend(int(page) for page in group if int(page) > 0)
                consumed += 1
                if len(pages) == released:
                    break
        if len(pages) == released:
            del hinted[:consumed]
        else:
            pages = [
                int(page)
                for values in page_ids
                for page in values.detach().cpu().tolist()
                if int(page) > 0
            ]
            if isinstance(hinted, list):
                hinted.clear()
        _demote_exact_retained_pages(self, pages)
        result = orig_paged_release_page_ids(self, *page_ids)
        _forget_local_page_mappings(self, pages)
        cpu_free = st.get("cpu_free_pages")
        if isinstance(cpu_free, (list, deque)):
            if self.need_sort:
                cpu_staged = st.get("cpu_staged_pages")
                if isinstance(cpu_staged, list):
                    cpu_staged.extend(pages)
            else:
                if isinstance(cpu_free, deque):
                    cpu_free.extendleft(reversed(pages))
                else:
                    cpu_free[:0] = pages
        return result
    result = orig_paged_release_page_ids(self, *page_ids)
    if (
        st is not None
        and st.get("tp_reservation_aligned", False)
        and not self.need_sort
    ):
        queue = st.get("tp_reserved_pages")
        released = sum(int(values.numel()) for values in page_ids)
        if isinstance(queue, list) and queue and released:
            queued = len(queue)
            self.free_pages = torch.cat(
                (
                    self.free_pages[released : released + queued],
                    self.free_pages[:released],
                    self.free_pages[released + queued :],
                )
            )
    _release_pages(self, *page_ids)
    return result


def hint_hbm_page_release(self, pages: list[int]) -> bool:
    """Provide exact CPU page IDs for the next native paged release.

    The hint is an optimization, not an authority: ``_release_page_ids`` uses
    it only when its cardinality matches the native device-tensor release.
    """
    st = _state(self)
    if st is None or not st.get("exclusive_steady_state", False):
        return False
    hinted = st.get("cpu_release_pages")
    if not isinstance(hinted, list):
        return False
    pages = [int(page) for page in pages]
    if not pages:
        return True
    hinted.append(pages)
    return True


def _gms_paged_merge_and_sort_free(self):
    # Native PagedTokenToKVPoolAllocator is an exact no-op with no staged
    # pages. Preserve that ordering: sorting only the CPU mirror would make
    # subsequent exact-page reservations disagree with the device free list.
    had_staged_pages = bool(getattr(self, "staged_pages", ()))
    result = orig_paged_merge_and_sort_free(self)
    st = _state(self)
    if had_staged_pages and st is not None and st.get("exclusive_steady_state", False):
        cpu_free = st.get("cpu_free_pages")
        cpu_staged = st.get("cpu_staged_pages")
        if isinstance(cpu_free, (list, deque)) and isinstance(cpu_staged, list):
            merged = sorted((*cpu_free, *cpu_staged))
            cpu_free.clear()
            cpu_free.extend(merged)
            cpu_staged.clear()
    return result


def _request_pages(req) -> list[int] | None:
    pages = getattr(req, "_gms_kv_page_ids", None)
    if pages is None:
        return None
    return [int(page) for page in pages]


def _record_extend_pages(st, selected, prefix_lens_cpu, seq_lens_cpu, page_size):
    batch = st.get("active_batch")
    if batch is None:
        return
    prefix_lens = [int(value) for value in prefix_lens_cpu.tolist()]
    seq_lens = [int(value) for value in seq_lens_cpu.tolist()]
    cursor = 0
    for req, prefix_len, seq_len in zip(batch.reqs, prefix_lens, seq_lens):
        old_pages = (prefix_len + page_size - 1) // page_size
        new_pages = (seq_len + page_size - 1) // page_size - old_pages
        allocated = selected[cursor : cursor + new_pages]
        cursor += new_pages
        known = _request_pages(req)
        if old_pages and (known is None or len(known) < old_pages):
            setattr(req, "_gms_kv_page_ids", None)
            continue
        setattr(req, "_gms_kv_page_ids", (known or [])[:old_pages] + allocated)
    if cursor != len(selected):
        raise RuntimeError("SGLang extend page accounting diverged")


def _record_decode_pages(st, selected, seq_lens_cpu, page_size):
    batch = st.get("active_batch")
    if batch is None:
        return
    cursor = 0
    for req, seq_len in zip(batch.reqs, seq_lens_cpu.tolist()):
        seq_len = int(seq_len)
        old_pages = (max(0, seq_len - 1) + page_size - 1) // page_size
        new_pages = (seq_len + page_size - 1) // page_size - old_pages
        known = _request_pages(req)
        if old_pages and (known is None or len(known) < old_pages):
            setattr(req, "_gms_kv_page_ids", None)
            cursor += new_pages
            continue
        setattr(
            req,
            "_gms_kv_page_ids",
            (known or [])[:old_pages] + selected[cursor : cursor + new_pages],
        )
        cursor += new_pages
    if cursor != len(selected):
        raise RuntimeError("SGLang decode page accounting diverged")


def _with_active_batch(original, batch, *args, **kwargs):
    allocator = batch.tree_cache.token_to_kv_pool_allocator
    st = _state(allocator)
    if st is None:
        return original(batch, *args, **kwargs)
    previous = st.get("active_batch")
    st["active_batch"] = batch
    try:
        return original(batch, *args, **kwargs)
    finally:
        st["active_batch"] = previous


def _gms_schedule_alloc_for_extend(batch, *args, **kwargs):
    return _with_active_batch(orig_schedule_alloc_for_extend, batch, *args, **kwargs)


def _gms_schedule_alloc_for_decode(batch, *args, **kwargs):
    return _with_active_batch(orig_schedule_alloc_for_decode, batch, *args, **kwargs)


def _forget_local_page_mappings(self, pages: list[int]) -> None:
    reverse = getattr(self, "_gms_local_hashes_by_page", None)
    forward = getattr(self, "_gms_local_pages_by_hash", None)
    if not isinstance(reverse, dict) or not isinstance(forward, dict):
        return
    for page in pages:
        for content_hash in reverse.pop(int(page), ()):
            if forward.get(content_hash) == int(page):
                forward.pop(content_hash, None)


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
    if st is not None:
        _revoke_allocator_fast_path(st)
    result = orig_paged_clear(self)
    if st is not None:
        client = st["client"]
        retained = st["retained_pages"]
        assert isinstance(retained, set)
        client.release(outstanding)
        lease_map.clear()
        retained.clear()
        queue = st.get("tp_reserved_pages")
        if isinstance(queue, list):
            queue.clear()
        st["tp_reservation_aligned"] = False

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
        merge_and_sort_free = _gms_paged_merge_and_sort_free
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
    global orig_paged_alloc_decode, orig_paged_release_page_ids
    global orig_paged_merge_and_sort_free, orig_paged_clear
    global orig_schedule_alloc_for_extend, orig_schedule_alloc_for_decode
    if factory is not None:
        _factory = factory
    if _patched:
        return False
    if _factory is None and not kv_leases_enabled("sglang"):
        return False

    try:
        import torch
        from sglang.srt.managers import schedule_batch
        from sglang.srt.mem_cache import allocator as alloc_mod
        from sglang.srt.mem_cache import kv_cache_configurator
        from sglang.srt.utils import get_num_new_pages
    except Exception:
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
    orig_paged_merge_and_sort_free = Paged.merge_and_sort_free
    orig_paged_clear = Paged.clear
    orig_schedule_alloc_for_extend = schedule_batch.alloc_for_extend
    orig_schedule_alloc_for_decode = schedule_batch.alloc_for_decode

    _gms_token_allocator_class, _gms_paged_allocator_class = _build_allocator_classes(
        Token, Paged
    )
    # The configurator imported these names directly. Rebinding its two
    # construction references installs lease-aware subclasses without
    # modifying SGLang allocator classes globally. The two scheduler functions
    # only scope request-local CPU page bookkeeping around native allocation.
    kv_cache_configurator.TokenToKVPoolAllocator = _gms_token_allocator_class
    kv_cache_configurator.PagedTokenToKVPoolAllocator = _gms_paged_allocator_class
    schedule_batch.alloc_for_extend = _gms_schedule_alloc_for_extend
    schedule_batch.alloc_for_decode = _gms_schedule_alloc_for_decode

    _patched = True
    logger.info("[GMS-KVLease] installed SGLang lease-aware allocator subclasses")
    return True


def lease_hooks_installed() -> bool:
    """Verify the live SGLang allocator construction bindings."""
    try:
        from sglang.srt.managers import schedule_batch
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
        and schedule_batch.alloc_for_extend is _gms_schedule_alloc_for_extend
        and schedule_batch.alloc_for_decode is _gms_schedule_alloc_for_decode
    )


if kv_leases_enabled("sglang"):
    try:
        install()
    except Exception:
        logger.exception("[GMS-KVLease] SGLang auto-install failed")
