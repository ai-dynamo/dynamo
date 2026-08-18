# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether a woken engine may reuse its predecessor's index.

One question decides it: did *every* rank inherit the previous engine's KV
pages, or did this engine build its own? Inherited means the entries describe
real bytes. Fresh pages mean the old store describes memory that no longer
exists -- and replacing it is what makes a stale store inert for whoever comes
next.

    inherit(scheduler)             decide, then restore and attach
    compatibility_digest(cfg, n)   what makes two engines' block ids comparable
    adopted_everywhere(path, n)    the per-rank answers, consumed on read
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

from gpu_memory_service.integrations.vllm.kv_cache import block_pool
from gpu_memory_service.kv_cache import MmapBlockIndexStore, Refusal, store_path

logger = logging.getLogger(__name__)


def compatibility_digest(cfg, num_blocks: int) -> bytes:
    """What must match for two engines to mean the same thing by "block 457".

    A mismatch is refused rather than migrated: same block ids under a different
    geometry is the one failure here that would be a wrong answer, not a miss.
    """
    import vllm
    from vllm.v1.core import kv_cache_utils

    fields = {
        "vllm": vllm.__version__,
        "model": cfg.model_config.model,
        "block_size": cfg.cache_config.block_size,
        "cache_dtype": str(cfg.cache_config.cache_dtype),
        "tp": cfg.parallel_config.tensor_parallel_size,
        "num_blocks": num_blocks,
        # Root of vLLM's block-hash chain: os.urandom per process unless
        # PYTHONHASHSEED is pinned, so this catches an unpinned seed too.
        "none_hash": bytes(kv_cache_utils.NONE_HASH).hex(),
        "hash_algo": str(cfg.cache_config.prefix_caching_hash_algo),
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).digest()


def adopted_everywhere(path: str, world_size: int) -> bool:
    """Did every rank inherit its KV pages?

    A scheduler has no handle on the executor, so it cannot ask the workers over
    ``collective_rpc``. Each rank publishes a one-byte answer beside the store at
    wake (``_publish_kv_adoption`` in ``worker.py``).

    Consumed, not merely read. A file left behind is a *stale* answer, and a
    stale "yes" is the one input that turns a miss into a wrong answer -- it
    would restore a predecessor's entries onto pages this engine allocated
    itself. Deleting after each read makes absence the resting state.

    Anything short of unanimity is a no: the index is engine-wide but the bytes
    are per-rank.
    """
    answers = []
    for rank in range(world_size):
        rank_path = f"{path}.rank{rank}"
        try:
            with open(rank_path) as f:
                answers.append(f.read(1) == "1")
        except OSError:
            answers.append(False)
        try:
            os.unlink(rank_path)
        except OSError:
            logger.warning("[kv_cache] could not consume %s; refusing", rank_path)
            answers.append(False)
    return bool(answers) and all(answers)


def inherit(scheduler) -> None:
    """Bring this engine's block index up, reusing a predecessor's if we may.

    Called from ``set_pause_state(UNPAUSED)``, which vLLM reaches only after
    ``model_executor.wake_up()`` has returned -- a blocking collective, so every
    rank has re-attached -- and before the scheduler can hand out a block.
    """
    path = store_path()
    if not path:
        return

    pool = scheduler.kv_cache_manager.block_pool
    num_blocks = len(pool.blocks)
    identity = compatibility_digest(scheduler.vllm_config, num_blocks)
    # vLLM's own index as we find it. A sleep clears it, so this is normally 0;
    # it distinguishes "restore rebuilt the index" from "entries happened to
    # survive in process memory".
    held_before = sum(1 for b in pool.blocks if b.block_hash is not None)

    world_size = getattr(scheduler.parallel_config, "world_size", 1)
    adopted = adopted_everywhere(path, world_size)

    store, reason, installed = None, Refusal.ABSENT, []
    if adopted:
        store, reason = MmapBlockIndexStore.open(
            path, identity=identity, num_blocks=num_blocks
        )
        if store is None:
            logger.info("[kv_cache] not restoring (%s); starting fresh", reason.value)
        else:
            installed = block_pool.restore(pool, store)
            store.retain({b.block_id for b in installed})

    _log_handover(
        path,
        adopted=adopted,
        reason=reason.value,
        held_before=held_before,
        restored=len(installed),
    )

    if store is None:
        # Either we built a fresh pool -- so whatever the old store describes no
        # longer exists -- or it was refused. Same action either way, and it is
        # what makes a stale store inert for everyone after us.
        store = MmapBlockIndexStore.create(path, num_blocks, identity)

    block_pool.attach(pool, store, skip_reset=scheduler.skipping_index_reset)
    scheduler.block_index = store


def discard(path: str | None) -> None:
    """Remove the store so no successor can trust it."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError as e:
        logger.warning("[kv_cache] could not discard store %s: %s", path, e)


def _log_handover(path: str, **fields) -> None:
    """One line per wake in ``<store>.status.jsonl``.

    A design that fails closed *silently* decays into failing open the first time
    someone inverts a condition, and "refused" has to be distinguishable from
    "never ran". The engine process's logging config is not reliably ours.
    """
    try:
        with open(f"{path}.status.jsonl", "a") as f:
            f.write(json.dumps(fields) + "\n")
    except OSError as e:
        logger.debug("[kv_cache] could not log handover: %s", e)
