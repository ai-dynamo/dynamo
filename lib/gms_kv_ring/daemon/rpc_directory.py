# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-neutral persistent-HBM content-directory RPC domain."""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING, Optional

from gms_kv_ring.daemon.rpc_types import Handler, Message, Response

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gms_kv_ring.daemon.kv_cache_manager import GmsKvCacheManager


def _directory_record_change_locked(
    daemon: "GmsKvCacheManager",
    key: tuple[str, bytes],
    entry: Optional[dict],
    *,
    scope: str = "",
) -> int:
    """Append one public upsert/delete after its mutation is committed.

    The caller holds the content-hash lock. Revisions are global to the
    daemon, while consumers filter by manifest and optional engine scope.
    """
    daemon._content_directory_revision += 1
    revision = int(daemon._content_directory_revision)
    daemon._content_directory_changes.append(
        {
            "revision": revision,
            "manifest_id": key[0],
            "content_hash": key[1].hex(),
            "scope": str(scope or (entry or {}).get("_scope", "")),
            "entry": None if entry is None else _directory_public_entry(entry),
        }
    )
    daemon._content_hash_lock.notify_all()
    return revision


def _directory_remove_locked(
    daemon: "GmsKvCacheManager",
    key: tuple[str, bytes],
) -> bool:
    """Remove one directory entry and its reverse slot mappings."""
    entry = daemon._content_directory.pop(key, None)
    if entry is None:
        return False
    manifest_id, content_hash = key
    engine_id = str(entry["engine_id"])
    for slot_id in entry["slot_ids"]:
        slot_key = (manifest_id, engine_id, int(slot_id))
        if daemon._content_directory_by_slot.get(slot_key) == content_hash:
            daemon._content_directory_by_slot.pop(slot_key, None)
    _directory_record_change_locked(
        daemon,
        key,
        None,
        scope=str(entry.get("_scope", "")),
    )
    return True


def _directory_public_entry(entry: dict) -> dict:
    """Return a wire-safe residency without daemon bookkeeping fields."""
    return {name: value for name, value in entry.items() if not name.startswith("_")}


def _directory_touch_locked(daemon: "GmsKvCacheManager", entry: dict) -> None:
    daemon._content_directory_access_seq += 1
    entry["_last_access_seq"] = int(daemon._content_directory_access_seq)


def _directory_writer_matches(
    daemon: "GmsKvCacheManager",
    writer_id: str,
    expected_epoch: int,
) -> bool:
    return daemon._content_directory_writer_id == writer_id and int(
        daemon._content_directory_epoch
    ) == int(expected_epoch)


def _directory_release_claim_locked(
    daemon: "GmsKvCacheManager",
    claim_token: str,
) -> bool:
    claim = daemon._content_directory_claims.pop(claim_token, None)
    if claim is None:
        return False
    # Decrement based on CLAIM IDENTITY, not generation equality: adopt_claim
    # overwrites an entry's generations, so a generation-gated decrement would
    # skip and leak _claim_count, pinning the entry (unevictable/unreplaceable)
    # for the daemon's lifetime. Each claim token incremented the count once, so
    # release it once regardless of any subsequent generation change.
    for key, _generations in claim["entries"]:
        entry = daemon._content_directory.get(key)
        if entry is None:
            continue
        entry["_claim_count"] = max(0, int(entry.get("_claim_count", 0)) - 1)
    return True


def _directory_release_writer_claims_locked(
    daemon: "GmsKvCacheManager",
    writer_id: Optional[str],
) -> int:
    if writer_id is None:
        return 0
    tokens = [
        token
        for token, claim in daemon._content_directory_claims.items()
        if claim.get("writer_id") == writer_id
    ]
    for token in tokens:
        _directory_release_claim_locked(daemon, token)
    return len(tokens)


def _directory_entry_ready(daemon: "GmsKvCacheManager", entry: dict) -> bool:
    """Validate physical readiness for entries published by engine adapters."""
    tier = str(entry.get("tier", ""))
    if tier not in ("host", "storage"):
        return True
    engine_id = str(entry["engine_id"])
    ranges = entry.get("ranges") or []
    slot_ids = entry.get("slot_ids") or []
    generations = entry.get("generations") or []
    if (
        not ranges
        or not slot_ids
        or len(slot_ids) != len(generations)
        or len(ranges) % len(slot_ids) != 0
    ):
        return False
    tier_store = getattr(
        daemon, "host_tier" if tier == "host" else "storage_tier", None
    )
    if tier_store is None:
        # A minimal directory server configured without this storage tier must
        # not raise here: an AttributeError would propagate out of every later
        # snapshot/lookup and permanently brick the directory (the entry could
        # never be pruned). Treat the entry as not-ready (and thus prunable).
        return False
    getter = tier_store.get
    ranges_per_slot = len(ranges) // len(slot_ids)
    for index, (slot_id, generation) in enumerate(zip(slot_ids, generations)):
        begin = index * ranges_per_slot
        end = begin + ranges_per_slot
        for layer, offset, _size in ranges[begin:end]:
            physical = getter(engine_id, int(layer), int(offset))
            if physical is None:
                return False
            if (
                tier == "host"
                and int(generation)
                and int(physical.generation) != int(generation)
            ):
                return False
        if tier == "storage" and int(generation):
            with daemon._lock:
                current = daemon._storage_slot_generations.get(engine_id, {}).get(
                    int(slot_id), 0
                )
            if int(current) != int(generation):
                return False
    return True


def handle_directory_promote(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    """CAS-promote a writer after the external failover fence is held.

    Repeating the call for the active writer is idempotent so TP ranks can
    safely execute the same post-lock hook. A different writer must present
    the current epoch; a successful promotion increments it and immediately
    fences publications from the former writer.
    """
    writer_id = str(msg.get("writer_id", "")).strip()
    if not writer_id:
        return {"ok": False, "error": "writer_id is required"}
    try:
        expected_epoch = int(msg["expected_epoch"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "expected_epoch is required"}
    with daemon._content_hash_lock:
        current = int(daemon._content_directory_epoch)
        active = daemon._content_directory_writer_id
        if active == writer_id:
            # Idempotent self-promote (e.g. TP ranks repeating the post-lock
            # hook, or the same writer restarting with a stable writer_id).
            # Release any claims this writer left pinned before a crash-restart
            # so they cannot pin HBM forever.
            _directory_release_writer_claims_locked(daemon, writer_id)
            return {
                "ok": True,
                "promoted": True,
                "directory_epoch": current,
                "writer_id": active,
            }
        if expected_epoch != current:
            return {
                "ok": True,
                "promoted": False,
                "directory_epoch": current,
                "writer_id": active,
            }
        # The external failover lock already fenced the former writer. Its HBM
        # entries still in "active" state were published at SCHEDULING time
        # (before the forward pass wrote the KV bytes), so their write
        # completion was never confirmed. Making them adoptable would let the
        # replacement serve tokens computed from unwritten/garbage KV. Fail
        # closed: DROP them (the replacement recomputes those blocks). Entries
        # the former writer advanced past "active" are durable and survive for
        # adoption.
        stale_active = [
            key
            for key, entry in daemon._content_directory.items()
            if (
                entry.get("tier") == "hbm"
                and entry.get("state") == "active"
                and entry.get("_owner_writer") == active
            )
        ]
        for key in stale_active:
            _directory_remove_locked(daemon, key)

        # Drop abandoned lookup claims so a crashed engine cannot pin HBM
        # forever.
        _directory_release_writer_claims_locked(daemon, active)
        daemon._content_directory_epoch = current + 1
        daemon._content_directory_writer_id = writer_id
        return {
            "ok": True,
            "promoted": True,
            "directory_epoch": current + 1,
            "writer_id": writer_id,
        }


def handle_directory_snapshot(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    """Return one atomic public inventory snapshot and its delta cursor."""
    manifest_id = str(msg.get("manifest_id", "")).strip()
    scope = str(msg.get("scope", ""))
    if not manifest_id:
        return {"ok": False, "error": "manifest_id is required"}
    with daemon._content_hash_lock:
        items = []
        for key, entry in list(daemon._content_directory.items()):
            if key[0] != manifest_id:
                continue
            if scope and entry.get("_scope") != scope:
                continue
            if not _directory_entry_ready(daemon, entry):
                _directory_remove_locked(daemon, key)
                continue
            items.append(
                {
                    "content_hash": key[1].hex(),
                    "entry": _directory_public_entry(entry),
                }
            )
        return {
            "ok": True,
            "items": items,
            "directory_epoch": int(daemon._content_directory_epoch),
            "directory_revision": int(daemon._content_directory_revision),
            "writer_id": daemon._content_directory_writer_id,
        }


def handle_directory_changes(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    """Return committed changes after a cursor or request a fresh snapshot."""
    manifest_id = str(msg.get("manifest_id", "")).strip()
    scope = str(msg.get("scope", ""))
    try:
        after = max(0, int(msg.get("after_revision", 0)))
        limit = min(16_384, max(1, int(msg.get("limit", 4096))))
        wait_ms = min(1_000, max(0, int(msg.get("wait_ms", 0))))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed change cursor: {exc}"}
    if not manifest_id:
        return {"ok": False, "error": "manifest_id is required"}
    with daemon._content_hash_lock:
        if wait_ms and int(daemon._content_directory_revision) <= after:
            daemon._content_hash_lock.wait(wait_ms / 1_000)
        current = int(daemon._content_directory_revision)
        changes_log = daemon._content_directory_changes
        oldest = int(changes_log[0]["revision"]) if changes_log else current + 1
        reset_required = after > current or after < oldest - 1
        if reset_required:
            return {
                "ok": True,
                "changes": [],
                "next_revision": current,
                "directory_revision": current,
                "directory_epoch": int(daemon._content_directory_epoch),
                "writer_id": daemon._content_directory_writer_id,
                "has_more": False,
                "reset_required": True,
            }

        selected = []
        next_revision = after
        exhausted = True
        for change in changes_log:
            revision = int(change["revision"])
            if revision <= after:
                continue
            next_revision = revision
            if change["manifest_id"] == manifest_id and (
                not scope or change.get("scope") == scope
            ):
                selected.append(
                    {
                        "revision": revision,
                        "content_hash": change["content_hash"],
                        "entry": change["entry"],
                    }
                )
                if len(selected) >= limit:
                    exhausted = False
                    break
        if exhausted:
            # Writer promotion can change the epoch without an entry delta.
            next_revision = current
        return {
            "ok": True,
            "changes": selected,
            "next_revision": int(next_revision),
            "directory_revision": current,
            "directory_epoch": int(daemon._content_directory_epoch),
            "writer_id": daemon._content_directory_writer_id,
            "has_more": int(next_revision) < current,
            "reset_required": False,
        }


def handle_directory_lookup(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    """Return manifest-scoped READY entries aligned with requested hashes."""
    manifest_id = str(msg.get("manifest_id", "")).strip()
    if not manifest_id:
        return {"ok": False, "error": "manifest_id is required"}
    try:
        content_hashes = [bytes.fromhex(str(h)) for h in msg.get("hashes", [])]
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed hash: {exc}"}
    with daemon._content_hash_lock:
        entries = []
        for content_hash in content_hashes:
            key = (manifest_id, content_hash)
            entry = daemon._content_directory.get(key)
            if entry is not None and not _directory_entry_ready(daemon, entry):
                _directory_remove_locked(daemon, key)
                entry = None
            if entry is not None and entry.get("state") != "ready":
                entry = None
            if entry is not None:
                _directory_touch_locked(daemon, entry)
            entries.append(None if entry is None else _directory_public_entry(entry))
        epoch = int(daemon._content_directory_epoch)
        writer_id = daemon._content_directory_writer_id
    return {
        "ok": True,
        "entries": entries,
        "directory_epoch": epoch,
        "writer_id": writer_id,
    }


def handle_directory_lookup_claim(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    """Lookup READY entries and pin every hit under one opaque claim."""
    manifest_id = str(msg.get("manifest_id", "")).strip()
    writer_id = str(msg.get("writer_id", "")).strip()
    try:
        expected_epoch = int(msg["expected_epoch"])
        content_hashes = [bytes.fromhex(str(h)) for h in msg.get("hashes", [])]
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed lookup claim: {exc}"}
    if not manifest_id or not writer_id:
        return {"ok": False, "error": "manifest_id and writer_id are required"}

    with daemon._content_hash_lock:
        if not _directory_writer_matches(daemon, writer_id, expected_epoch):
            return {
                "ok": True,
                "entries": [None] * len(content_hashes),
                "claim_token": None,
                "rejected_stale_writer": True,
                "directory_epoch": int(daemon._content_directory_epoch),
                "writer_id": daemon._content_directory_writer_id,
            }
        entries: list[Optional[dict]] = []
        claimed = []
        for content_hash in content_hashes:
            key = (manifest_id, content_hash)
            entry = daemon._content_directory.get(key)
            if entry is not None and not _directory_entry_ready(daemon, entry):
                _directory_remove_locked(daemon, key)
                entry = None
            if entry is None or entry.get("state") != "ready":
                entries.append(None)
                continue
            entry["_claim_count"] = int(entry.get("_claim_count", 0)) + 1
            _directory_touch_locked(daemon, entry)
            claimed.append((key, tuple(entry.get("generations") or ())))
            entries.append(_directory_public_entry(entry))
        claim_token = uuid.uuid4().hex if claimed else None
        if claim_token is not None:
            daemon._content_directory_claims[claim_token] = {
                "writer_id": writer_id,
                "epoch": expected_epoch,
                "entries": claimed,
            }
        if os.environ.get("GMS_KV_DIRECTORY_DIAGNOSTICS"):
            count = int(getattr(daemon, "_content_directory_diag_claims", 0))
            if count < 16:
                daemon._content_directory_diag_claims = count + 1
                logger.warning(
                    "[GMS-KVDiag] claim writer=%s manifest=%s keys=%s matched=%s",
                    writer_id,
                    manifest_id,
                    [value.hex()[:16] for value in content_hashes],
                    [entry is not None for entry in entries],
                )
        return {
            "ok": True,
            "entries": entries,
            "claim_token": claim_token,
            "rejected_stale_writer": False,
            "directory_epoch": int(daemon._content_directory_epoch),
            "writer_id": daemon._content_directory_writer_id,
        }


def handle_directory_release_claim(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    token = str(msg.get("claim_token", "")).strip()
    if not token:
        return {"ok": False, "error": "claim_token is required"}
    with daemon._content_hash_lock:
        released = _directory_release_claim_locked(daemon, token)
    return {"ok": True, "released": released}


def handle_directory_adopt_claim(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    """Commit new lease generations for HBM entries held by a claim."""
    token = str(msg.get("claim_token", "")).strip()
    writer_id = str(msg.get("writer_id", "")).strip()
    manifest_id = str(msg.get("manifest_id", "")).strip()
    try:
        expected_epoch = int(msg["expected_epoch"])
        updates = [
            (
                bytes.fromhex(str(item["content_hash"])),
                [int(value) for value in item["generations"]],
            )
            for item in (msg.get("items") or [])
        ]
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed adopt: {exc}"}
    with daemon._content_hash_lock:
        claim = daemon._content_directory_claims.get(token)
        if (
            claim is None
            or claim.get("writer_id") != writer_id
            or not _directory_writer_matches(daemon, writer_id, expected_epoch)
        ):
            return {
                "ok": True,
                "adopted": 0,
                "rejected_stale_writer": True,
            }
        claim_keys = {key for key, _generations in claim["entries"]}
        parsed = []
        for content_hash, generations in updates:
            key = (manifest_id, content_hash)
            entry = daemon._content_directory.get(key)
            if (
                key not in claim_keys
                or entry is None
                or entry.get("tier") != "hbm"
                or len(generations) != len(entry.get("slot_ids") or [])
            ):
                return {"ok": False, "error": "adopt entry is not a claimed HBM hit"}
            parsed.append((key, entry, generations))
        _directory_release_claim_locked(daemon, token)
        for key, entry, generations in parsed:
            entry["generations"] = generations
            entry["state"] = "active"
            entry["_owner_writer"] = writer_id
            _directory_touch_locked(daemon, entry)
            _directory_record_change_locked(daemon, key, entry)
        return {
            "ok": True,
            "adopted": len(parsed),
            "rejected_stale_writer": False,
        }


def handle_directory_mark_hbm_dormant(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    """Move writer-owned active HBM records into the evictable READY set."""
    writer_id = str(msg.get("writer_id", "")).strip()
    manifest_id = str(msg.get("manifest_id", "")).strip()
    try:
        expected_epoch = int(msg["expected_epoch"])
        hashes = [bytes.fromhex(str(value)) for value in msg.get("hashes", [])]
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed dormant update: {exc}"}
    with daemon._content_hash_lock:
        if not _directory_writer_matches(daemon, writer_id, expected_epoch):
            return {
                "ok": True,
                "updated": 0,
                "rejected_stale_writer": True,
            }
        updated = 0
        for content_hash in hashes:
            key = (manifest_id, content_hash)
            entry = daemon._content_directory.get(key)
            if (
                entry is None
                or entry.get("tier") != "hbm"
                or entry.get("state") != "active"
                or entry.get("_owner_writer") != writer_id
            ):
                continue
            entry["state"] = "ready"
            entry.pop("_owner_writer", None)
            _directory_touch_locked(daemon, entry)
            _directory_record_change_locked(daemon, key, entry)
            updated += 1
        return {
            "ok": True,
            "updated": updated,
            "rejected_stale_writer": False,
        }


def handle_directory_ensure_hbm_capacity(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    """Retire the coldest unclaimed dormant HBM records.

    Exact lease generations are returned to the active engine, which releases
    the corresponding shared-memory leases only after the directory stops
    advertising them. This is a generic LRU over dormant records, not a prefix
    or request policy.
    """
    manifest_id = str(msg.get("manifest_id", "")).strip()
    writer_id = str(msg.get("writer_id", "")).strip()
    try:
        expected_epoch = int(msg["expected_epoch"])
        required = max(0, int(msg.get("required_blocks", 0)))
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"malformed capacity request: {exc}"}
    with daemon._content_hash_lock:
        if not _directory_writer_matches(daemon, writer_id, expected_epoch):
            return {
                "ok": True,
                "victims": [],
                "rejected_stale_writer": True,
            }
        candidates = [
            (key, entry)
            for key, entry in daemon._content_directory.items()
            if key[0] == manifest_id
            and entry.get("tier") == "hbm"
            and entry.get("state") == "ready"
            and int(entry.get("_claim_count", 0)) == 0
        ]
        candidates.sort(key=lambda pair: int(pair[1].get("_last_access_seq", 0)))
        victims = []
        freed = 0
        for key, entry in candidates:
            victims.append(
                {
                    "content_hash": key[1].hex(),
                    "engine_id": str(entry["engine_id"]),
                    "slot_ids": [int(value) for value in entry["slot_ids"]],
                    "generations": [
                        int(value) for value in entry.get("generations") or []
                    ],
                }
            )
            freed += len(entry["slot_ids"])
            _directory_remove_locked(daemon, key)
            if freed >= required:
                break
        return {
            "ok": True,
            "victims": victims,
            "freed_blocks": freed,
            "rejected_stale_writer": False,
        }


def handle_directory_hbm_inventory(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    """Return HBM slot ids that selective post-fence reclaim must preserve."""
    writer_id = str(msg.get("writer_id", "")).strip()
    scope = str(msg.get("scope", ""))
    try:
        expected_epoch = int(msg["expected_epoch"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "expected_epoch is required"}
    with daemon._content_hash_lock:
        if not _directory_writer_matches(daemon, writer_id, expected_epoch):
            return {
                "ok": True,
                "protected": {},
                "rejected_stale_writer": True,
            }
        protected: dict[str, set[int]] = {}
        for entry in daemon._content_directory.values():
            if scope and entry.get("_scope") != scope:
                continue
            if entry.get("tier") != "hbm" or entry.get("state") not in (
                "ready",
                "active",
            ):
                continue
            protected.setdefault(str(entry["engine_id"]), set()).update(
                int(slot_id) for slot_id in entry["slot_ids"]
            )
        return {
            "ok": True,
            "protected": {
                engine_id: sorted(slot_ids) for engine_id, slot_ids in protected.items()
            },
            "rejected_stale_writer": False,
        }


def handle_directory_publish_batch(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    """Atomically publish sealed GMS residencies for one active writer."""
    writer_id = str(msg.get("writer_id", "")).strip()
    manifest_id = str(msg.get("manifest_id", "")).strip()
    scope = str(msg.get("scope", ""))
    try:
        expected_epoch = int(msg["expected_epoch"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "expected_epoch is required"}
    if not writer_id or not manifest_id:
        return {
            "ok": False,
            "error": "writer_id and manifest_id are required",
        }
    parsed = []
    for item in msg.get("items", []) or []:
        try:
            content_hash = bytes.fromhex(str(item["content_hash"]))
            engine_id = str(item["engine_id"])
            raw_slots = item.get("slot_ids")
            if raw_slots is None:
                raw_slots = [item["slot_id"]]
            slot_ids = [int(slot) for slot in raw_slots]
            if not slot_ids:
                raise ValueError("slot_ids is empty")
            raw_generations = item.get("generations")
            if raw_generations is None:
                raw_generations = [item.get("generation", 0)] * len(slot_ids)
            generations = [int(generation) for generation in raw_generations]
            if len(generations) != len(slot_ids):
                raise ValueError("generations length differs from slot_ids")
            ranges = [
                (int(r["layer"]), int(r["offset"]), int(r["size"]))
                for r in (item.get("ranges") or [])
            ]
            tier = str(item.get("tier", ""))
            sealed = bool(item.get("sealed", True))
            active_hbm = bool(item.get("active", False)) and tier == "hbm"
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "error": f"malformed item: {exc}"}
        parsed.append(
            (
                content_hash,
                engine_id,
                slot_ids,
                generations,
                ranges,
                tier,
                sealed,
                active_hbm,
            )
        )
    if os.environ.get("GMS_KV_DIRECTORY_DIAGNOSTICS"):
        count = int(getattr(daemon, "_content_directory_diag_publishes", 0))
        if count < 16:
            daemon._content_directory_diag_publishes = count + 1
            logger.warning(
                "[GMS-KVDiag] publish writer=%s manifest=%s keys=%s active=%s",
                writer_id,
                manifest_id,
                [value[0].hex()[:16] for value in parsed[:4]],
                [value[7] for value in parsed[:4]],
            )

    with daemon._content_hash_lock:
        active = daemon._content_directory_writer_id
        if not _directory_writer_matches(daemon, writer_id, expected_epoch):
            return {
                "ok": True,
                "published": 0,
                "rejected_stale_writer": True,
                "directory_epoch": int(daemon._content_directory_epoch),
                "writer_id": active,
            }
        # Validate every conflict before mutating any record so the batch is
        # actually atomic on malformed reuse or concurrent claims.
        seen_keys = set()
        seen_slots = {}
        for (
            content_hash,
            engine_id,
            slot_ids,
            generations,
            _ranges,
            _tier,
            sealed,
            _active_hbm,
        ) in parsed:
            key = (manifest_id, content_hash)
            if key in seen_keys:
                return {"ok": False, "error": "duplicate directory content hash"}
            seen_keys.add(key)
            existing = daemon._content_directory.get(key)
            if (
                not sealed
                and existing is not None
                and any(generations)
                and tuple(existing.get("generations") or ()) != tuple(generations)
            ):
                # A delayed invalidation for an older lease generation must
                # not retire a newer publication for the same content hash.
                continue
            if existing is not None and int(existing.get("_claim_count", 0)):
                action = "remove" if not sealed else "replace"
                return {
                    "ok": False,
                    "error": f"cannot {action} a claimed directory entry",
                }
            if not sealed:
                continue
            for slot_id in slot_ids:
                slot_key = (manifest_id, engine_id, slot_id)
                pending_hash = seen_slots.get(slot_key)
                if pending_hash is not None and pending_hash != content_hash:
                    return {"ok": False, "error": "duplicate directory slot reuse"}
                seen_slots[slot_key] = content_hash
                old_hash = daemon._content_directory_by_slot.get(slot_key)
                if old_hash is None or old_hash == content_hash:
                    continue
                old_entry = daemon._content_directory.get((manifest_id, old_hash))
                if old_entry is not None and int(old_entry.get("_claim_count", 0)):
                    return {
                        "ok": False,
                        "error": "cannot reuse a claimed directory slot",
                    }

        published = 0
        removed = 0
        for (
            content_hash,
            engine_id,
            slot_ids,
            generations,
            ranges,
            tier,
            sealed,
            active_hbm,
        ) in parsed:
            key = (manifest_id, content_hash)
            if not sealed:
                existing = daemon._content_directory.get(key)
                if (
                    existing is not None
                    and any(generations)
                    and tuple(existing.get("generations") or ()) != tuple(generations)
                ):
                    continue
                removed += int(_directory_remove_locked(daemon, key))
                continue
            # Reusing any physical slot retires the entire previous logical
            # entry before the new entry becomes discoverable.
            for slot_id in slot_ids:
                old_hash = daemon._content_directory_by_slot.get(
                    (manifest_id, engine_id, slot_id)
                )
                if old_hash is not None and old_hash != content_hash:
                    removed += int(
                        _directory_remove_locked(daemon, (manifest_id, old_hash))
                    )
            entry = {
                "engine_id": engine_id,
                "slot_ids": slot_ids,
                "generations": generations,
                "ranges": ranges,
                "state": "active" if active_hbm else "ready",
                "_claim_count": 0,
                "_scope": scope,
            }
            if active_hbm:
                entry["_owner_writer"] = writer_id
            if tier:
                entry["tier"] = tier
            # If we are overwriting an existing entry for the SAME (manifest,
            # hash) key with a different slot set, drop the old entry's stale
            # reverse slot-mappings for slots the new entry does not reuse.
            # Otherwise a later reuse of an orphaned old slot resolves back to
            # this hash and deletes the current, valid entry (C2).
            prior = daemon._content_directory.get(key)
            if prior is not None:
                prior_engine = prior.get("engine_id", engine_id)
                new_slot_set = {int(s) for s in slot_ids}
                for old_slot in prior.get("slot_ids") or []:
                    if prior_engine == engine_id and int(old_slot) in new_slot_set:
                        continue
                    slot_key = (manifest_id, prior_engine, int(old_slot))
                    if daemon._content_directory_by_slot.get(slot_key) == content_hash:
                        daemon._content_directory_by_slot.pop(slot_key, None)
            daemon._content_directory[key] = entry
            _directory_touch_locked(daemon, entry)
            for slot_id in slot_ids:
                daemon._content_directory_by_slot[
                    (manifest_id, engine_id, slot_id)
                ] = content_hash
            _directory_record_change_locked(daemon, key, entry)
            published += 1
        epoch = int(daemon._content_directory_epoch)
    return {
        "ok": True,
        "published": published,
        "removed": removed,
        "rejected_stale_writer": False,
        "directory_epoch": epoch,
        "writer_id": writer_id,
    }


DIRECTORY_HANDLERS: dict[str, Handler] = {
    "directory_snapshot": handle_directory_snapshot,
    "directory_changes": handle_directory_changes,
    "directory_lookup": handle_directory_lookup,
    "directory_lookup_claim": handle_directory_lookup_claim,
    "directory_promote": handle_directory_promote,
    "directory_publish_batch": handle_directory_publish_batch,
    "directory_release_claim": handle_directory_release_claim,
    "directory_adopt_claim": handle_directory_adopt_claim,
    "directory_mark_hbm_dormant": handle_directory_mark_hbm_dormant,
    "directory_ensure_hbm_capacity": handle_directory_ensure_hbm_capacity,
    "directory_hbm_inventory": handle_directory_hbm_inventory,
}
