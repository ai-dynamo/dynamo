/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_index.h"
#include "dynkv_state.h"

bool worker_forget_node(WorkerState *worker, IndexNode *node) {
    uint8_t key[8];
    encode_u64_be(key, node->external_hash);
    void *encoded =
        ValkeyModule_DictGetC(worker->node_members, key, sizeof(key), NULL);
    if (encoded == NULL) {
        return false;
    }
    size_t position = (size_t)(uintptr_t)encoded - 1;
    if (position >= worker->node_count || worker->nodes[position] != node) {
        return false;
    }
    size_t last = worker->node_count - 1;
    IndexNode *moved = worker->nodes[last];
    size_t moved_owner_position = worker->node_owner_indices[last];
    if (position != last &&
        !worker_replace_node_position(worker, moved, last, position)) {
        return false;
    }
    if (ValkeyModule_DictDelC(
            worker->node_members, key, sizeof(key), NULL) != VALKEYMODULE_OK) {
        if (position != last) {
            bool restored = worker_replace_node_position(
                worker, moved, position, last);
            ValkeyModule_Assert(restored);
        }
        return false;
    }
    worker->node_count = last;
    if (position != last) {
        worker->nodes[position] = moved;
        worker->node_owner_indices[position] = moved_owner_position;
    }
    if (worker->epoch != NULL && worker->epoch->node_membership_count != 0) {
        --worker->epoch->node_membership_count;
    }
    return true;
}

bool node_remove_owner(IndexNode *node, WorkerState *worker) {
    Owner *owner = node_owner(node, worker);
    if (owner == NULL) {
        return false;
    }
    size_t position = (size_t)(owner - node->owners);
    size_t last = node->owner_count - 1;
    Owner moved = node->owners[last];
    size_t moved_worker_position = 0;
    if (position != last &&
        (!worker_node_position(
             moved.worker, node, &moved_worker_position) ||
         moved.worker->node_owner_indices[moved_worker_position] != last)) {
        return false;
    }
    if (!worker_forget_node(worker, node)) {
        return false;
    }
    node->owner_count = last;
    if (position != last) {
        node->owners[position] = moved;
        moved.worker->node_owner_indices[moved_worker_position] = position;
    }
    return true;
}

bool router_index_delete_ownerless_leaf(RouterIndex *index, IndexNode *node) {
    if (node == NULL || node->owner_count != 0 || node->child_count != 0) {
        return false;
    }
    uint8_t external_key[8];
    uint8_t edge_key[16];
    encode_u64_be(external_key, node->external_hash);
    child_key(edge_key, node->parent_external_hash, node->local_hash);
    if (ValkeyModule_DictDelC(
            index->nodes_by_external, external_key, sizeof(external_key), NULL) !=
            VALKEYMODULE_OK ||
        ValkeyModule_DictDelC(
            index->children_by_parent_and_local, edge_key, sizeof(edge_key), NULL) !=
            VALKEYMODULE_OK) {
        return false;
    }
    if (node->parent_external_hash != DYNKV_ROOT_PARENT) {
        IndexNode *parent =
            router_index_node_by_external(index, node->parent_external_hash);
        if (parent != NULL && parent->child_count != 0) {
            --parent->child_count;
        }
    }
    index_node_free(node);
    return true;
}

void router_index_prune_ownerless_ancestors(
    RouterIndex *index,
    uint64_t external_hash) {
    while (external_hash != DYNKV_ROOT_PARENT) {
        IndexNode *node = router_index_node_by_external(index, external_hash);
        if (node == NULL || node->owner_count != 0 || node->child_count != 0) {
            return;
        }
        uint64_t parent_hash = node->parent_external_hash;
        if (!router_index_delete_ownerless_leaf(index, node)) {
            return;
        }
        external_hash = parent_hash;
    }
}

/*
 * Owner-fenced writers provide a single ordered stream per rank. Once their
 * REMOVE/CLEAR transition is accepted, the inactive per-block tombstone is no
 * longer needed; the rank/worker generation remains the recovery fence.
 */
void router_index_compact_direct_worker_owners(
    RouterIndex *index,
    WorkerState *worker) {
    if (worker == NULL || !worker->lifecycle_managed || worker->legacy_tainted ||
        worker->node_count == 0) {
        return;
    }
    uint64_t *removed =
        ValkeyModule_Alloc(worker->node_count * sizeof(*removed));
    size_t removed_count = 0;
    size_t i = 0;
    while (i < worker->node_count) {
        IndexNode *node = worker->nodes[i];
        Owner *owner = node_owner(node, worker);
        if (owner == NULL || owner->active) {
            ++i;
            continue;
        }
        removed[removed_count++] = node->external_hash;
        if (!node_remove_owner(node, worker)) {
            ++i;
        }
    }
    for (size_t removed_index = 0; removed_index < removed_count; ++removed_index) {
        router_index_prune_ownerless_ancestors(index, removed[removed_index]);
    }
    ValkeyModule_Free(removed);
}

void router_index_compact_direct_worker_all_ranks(
    RouterIndex *index,
    uint64_t worker_id) {
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    if (epoch != NULL) {
        for (size_t i = 0; i < epoch->worker_state_count; ++i) {
            router_index_compact_direct_worker_owners(
                index, epoch->worker_states[i]);
        }
        return;
    }
    /* Legacy state can lack an epoch; it is tainted and will be retained. */
}

void router_index_compact_direct_remove(
    RouterIndex *index,
    WorkerState *worker,
    const uint8_t *payload,
    size_t payload_length) {
    if (worker == NULL || !worker->lifecycle_managed || worker->legacy_tainted) {
        return;
    }
    Reader reader = {.data = payload, .length = payload_length, .offset = 0};
    uint8_t version = 0;
    uint8_t kind = 0;
    uint64_t ignored_worker = 0;
    uint32_t ignored_rank = 0;
    uint64_t ignored_event = 0;
    uint32_t count = 0;
    if (!reader_u8(&reader, &version) || !reader_u8(&reader, &kind) ||
        !reader_u64(&reader, &ignored_worker) ||
        !reader_u32(&reader, &ignored_rank) ||
        !reader_u64(&reader, &ignored_event) || !reader_u32(&reader, &count) ||
        version != DYNKV_WIRE_VERSION || kind != DYNKV_EVENT_REMOVE) {
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t external_hash = 0;
        if (!reader_u64(&reader, &external_hash)) {
            return;
        }
        IndexNode *node = router_index_node_by_external(index, external_hash);
        Owner *owner = node == NULL ? NULL : node_owner(node, worker);
        if (owner == NULL || owner->active || !node_remove_owner(node, worker)) {
            continue;
        }
        router_index_prune_ownerless_ancestors(index, external_hash);
    }
}

bool node_set_owner(
    IndexNode *node,
    WorkerState *worker,
    uint64_t event_id,
    bool active) {
    if (event_id < worker->last_clear_event_id) {
        return false;
    }
    Owner *owner = node_owner(node, worker);
    if (owner != NULL && event_id < owner->event_id) {
        return false;
    }
    if (owner != NULL && owner->event_id == event_id && owner->active == active) {
        return false;
    }
    owner = owner == NULL ? node_owner_create(node, worker) : owner;
    owner->event_id = event_id;
    owner->active = active;
    return true;
}

bool router_index_clear_worker(WorkerState *worker, uint64_t event_id, bool track_clear) {
    if (track_clear && worker->has_clear_dedupe_event_id &&
        event_id <= worker->last_clear_dedupe_event_id) {
        return false;
    }
    bool changed = false;
    if (track_clear) {
        worker->last_clear_event_id = event_id;
        worker->last_clear_dedupe_event_id = event_id;
        worker->has_clear_dedupe_event_id = true;
        changed = true;
    }
    for (size_t i = 0; i < worker->node_count; ++i) {
        Owner *owner = node_owner(worker->nodes[i], worker);
        if (owner != NULL && event_id >= owner->event_id &&
            (owner->event_id != event_id || owner->active)) {
            owner->event_id = event_id;
            owner->active = false;
            changed = true;
        }
    }
    return changed;
}

/*
 * A removal is also used to replace one DP rank from a worker tree dump.
 * Do not leave an UINT64_MAX tombstone behind: replayed dump events can have
 * IDs lower than the events previously applied to this in-memory index.
 */
bool router_index_deactivate_worker(WorkerState *worker) {
    bool changed = worker->last_clear_event_id != 0;
    worker->last_clear_event_id = 0;
    for (size_t i = 0; i < worker->node_count; ++i) {
        Owner *owner = node_owner(worker->nodes[i], worker);
        if (owner != NULL) {
            changed |= owner->event_id != 0 || owner->active;
            owner->event_id = 0;
            owner->active = false;
        }
    }
    return changed;
}

/* Reset is used only before replaying a current worker tree dump. */
bool router_index_reset_worker(WorkerState *worker) {
    bool changed = worker->retired;
    worker->retired = false;
    return router_index_deactivate_worker(worker) || changed;
}

/*
 * A Cleared event is emitted for the worker, not an individual DP rank.
 * Event IDs are rank-local, so only the emitting rank can use the clear ID as
 * an ordering barrier. Other ranks must be invalidated unconditionally and
 * then recovered from their own worker dump.
 */
bool router_index_clear_all_worker(
    RouterIndex *index,
    uint64_t worker_id,
    uint32_t clear_dp_rank,
    uint64_t event_id) {
    /*
     * Direct worker publishing and frontend recovery can deliver the same
     * clear. Never let a duplicate/stale clear deactivate freshly recovered
     * sibling ranks. A new worker-wide incarnation fence is still required
     * before pure-direct mode can replace recovery, but this preserves the
     * current mirrored path's idempotency.
     */
    WorkerState *emitter = router_index_worker(index, worker_id, clear_dp_rank, false);
    bool changed = false;
    if (emitter == NULL) {
        emitter = router_index_worker(index, worker_id, clear_dp_rank, true);
        changed = true;
    }
    if (emitter->has_clear_dedupe_event_id &&
        event_id <= emitter->last_clear_dedupe_event_id) {
        return false;
    }
    ValkeyModuleDictIter *workers =
        ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
        WorkerState *worker = data;
        if (worker->worker_id == worker_id) {
            if (worker->dp_rank == clear_dp_rank) {
                changed |= router_index_clear_worker(worker, event_id, true);
            } else {
                changed |= router_index_deactivate_worker(worker);
            }
        }
    }
    ValkeyModule_DictIteratorStop(workers);
    return changed;
}

/* Validate every byte before opening or mutating the persistent module key. */
int router_index_validate_event(const uint8_t *data, size_t length) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint8_t kind = 0;
    uint64_t worker_id = 0;
    uint32_t dp_rank = 0;
    uint64_t event_id = 0;
    if (!reader_u8(&reader, &version) || !reader_u8(&reader, &kind) ||
        !reader_u64(&reader, &worker_id) || !reader_u32(&reader, &dp_rank) ||
        !reader_u64(&reader, &event_id) || version != DYNKV_WIRE_VERSION) {
        return VALKEYMODULE_ERR;
    }
    (void)worker_id;
    (void)dp_rank;
    (void)event_id;

    uint32_t count = 0;
    switch (kind) {
        case DYNKV_EVENT_STORE: {
            uint64_t parent_hash = 0;
            if (!reader_u64(&reader, &parent_hash) || !reader_u32(&reader, &count) ||
                count > DYNKV_MAX_EVENT_BLOCKS ||
                reader.length - reader.offset != (size_t)count * 16) {
                return VALKEYMODULE_ERR;
            }
            (void)parent_hash;
            return VALKEYMODULE_OK;
        }
        case DYNKV_EVENT_REMOVE:
            if (!reader_u32(&reader, &count) || count > DYNKV_MAX_EVENT_BLOCKS ||
                reader.length - reader.offset != (size_t)count * sizeof(uint64_t)) {
                return VALKEYMODULE_ERR;
            }
            return VALKEYMODULE_OK;
        case DYNKV_EVENT_CLEAR:
            return reader.offset == reader.length ? VALKEYMODULE_OK : VALKEYMODULE_ERR;
        default:
            return VALKEYMODULE_ERR;
    }
}

int router_index_apply_store(
    RouterIndex *index,
    Reader *reader,
    WorkerState *worker,
    uint64_t worker_id,
    uint32_t dp_rank,
    uint64_t event_id,
    bool *changed_out) {
    *changed_out = false;
    uint64_t parent_hash = 0;
    uint32_t count = 0;
    if (!reader_u64(reader, &parent_hash) || !reader_u32(reader, &count)) {
        return VALKEYMODULE_ERR;
    }
    if (count > DYNKV_MAX_EVENT_BLOCKS) {
        return VALKEYMODULE_ERR;
    }
    if (reader->length - reader->offset != (size_t)count * 16) {
        return VALKEYMODULE_ERR;
    }
    if (worker != NULL && worker->retired) {
        return VALKEYMODULE_OK;
    }
    if (parent_hash != DYNKV_ROOT_PARENT) {
        IndexNode *parent = router_index_node_by_external(index, parent_hash);
        Owner *owner = parent == NULL || worker == NULL ? NULL : node_owner(parent, worker);
        if (owner == NULL || !owner->active) {
            return 2;
        }
    }

    StoreBlock *blocks = ValkeyModule_Alloc((size_t)count * sizeof(*blocks));
    uint64_t current_parent = parent_hash;
    for (uint32_t i = 0; i < count; ++i) {
        StoreBlock *block = &blocks[i];
        block->parent_hash = current_parent;
        if (!reader_u64(reader, &block->external_hash) || !reader_u64(reader, &block->local_hash)) {
            ValkeyModule_Free(blocks);
            return VALKEYMODULE_ERR;
        }
        current_parent = block->external_hash;
    }

    /* Validate the whole batch before publishing any mutation. */
    ValkeyModuleDict *seen_external = ValkeyModule_CreateDict(NULL);
    for (uint32_t i = 0; i < count; ++i) {
        StoreBlock *block = &blocks[i];
        uint8_t external_key[8];
        encode_u64_be(external_key, block->external_hash);
        if (ValkeyModule_DictGetC(seen_external, external_key, sizeof(external_key), NULL) != NULL) {
            ValkeyModule_FreeDict(NULL, seen_external);
            ValkeyModule_Free(blocks);
            return 3;
        }
        ValkeyModule_DictSetC(seen_external, external_key, sizeof(external_key), block);

        IndexNode *by_external = router_index_node_by_external(index, block->external_hash);
        if (by_external != NULL &&
            (by_external->parent_external_hash != block->parent_hash ||
             by_external->local_hash != block->local_hash)) {
            ValkeyModule_FreeDict(NULL, seen_external);
            ValkeyModule_Free(blocks);
            return 3;
        }
        IndexNode *by_edge = router_index_child(index, block->parent_hash, block->local_hash);
        if (by_edge != NULL && by_edge->external_hash != block->external_hash) {
            ValkeyModule_FreeDict(NULL, seen_external);
            ValkeyModule_Free(blocks);
            return 3;
        }
    }
    ValkeyModule_FreeDict(NULL, seen_external);

    if (worker == NULL) {
        worker = router_index_worker(index, worker_id, dp_rank, true);
        *changed_out = true;
    }

    for (uint32_t i = 0; i < count; ++i) {
        StoreBlock *block = &blocks[i];
        bool node_was_absent =
            router_index_node_by_external(index, block->external_hash) == NULL;
        IndexNode *node = router_index_add_node(
            index, block->external_hash, block->parent_hash, block->local_hash);
        if (node == NULL) {
            /* This cannot happen after validation unless allocation failed. */
            ValkeyModule_Free(blocks);
            return 3;
        }
        bool owner_changed = node_set_owner(node, worker, event_id, true);
        *changed_out |= node_was_absent || owner_changed;
    }
    ValkeyModule_Free(blocks);
    return VALKEYMODULE_OK;
}

int router_index_apply_remove(
    RouterIndex *index,
    Reader *reader,
    WorkerState *worker,
    uint64_t event_id,
    bool *changed_out) {
    *changed_out = false;
    uint32_t count = 0;
    if (!reader_u32(reader, &count) || count > DYNKV_MAX_EVENT_BLOCKS) {
        return VALKEYMODULE_ERR;
    }
    if (reader->length - reader->offset != (size_t)count * sizeof(uint64_t)) {
        return VALKEYMODULE_ERR;
    }
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t external_hash = 0;
        if (!reader_u64(reader, &external_hash)) {
            return VALKEYMODULE_ERR;
        }
        IndexNode *node = router_index_node_by_external(index, external_hash);
        if (node != NULL && worker != NULL) {
            *changed_out |= node_set_owner(node, worker, event_id, false);
        }
    }
    return VALKEYMODULE_OK;
}

int router_index_apply_event(RouterIndex *index, const uint8_t *data, size_t length) {
    if (router_index_validate_event(data, length) != VALKEYMODULE_OK) {
        return VALKEYMODULE_ERR;
    }
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint8_t kind = 0;
    uint64_t worker_id = 0;
    uint32_t dp_rank = 0;
    uint64_t event_id = 0;
    if (!reader_u8(&reader, &version) || !reader_u8(&reader, &kind) ||
        !reader_u64(&reader, &worker_id) || !reader_u32(&reader, &dp_rank) ||
        !reader_u64(&reader, &event_id) || version != DYNKV_WIRE_VERSION) {
        return VALKEYMODULE_ERR;
    }

    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    bool changed = false;
    int result = VALKEYMODULE_ERR;
    switch (kind) {
        case DYNKV_EVENT_STORE:
            result = router_index_apply_store(
                index, &reader, worker, worker_id, dp_rank, event_id, &changed);
            break;
        case DYNKV_EVENT_REMOVE:
            result = router_index_apply_remove(index, &reader, worker, event_id, &changed);
            break;
        case DYNKV_EVENT_CLEAR:
            changed = router_index_clear_all_worker(index, worker_id, dp_rank, event_id);
            result = VALKEYMODULE_OK;
            break;
        default:
            return VALKEYMODULE_ERR;
    }

    if (result == VALKEYMODULE_OK && reader.offset == reader.length) {
        if (changed) {
            ++index->mutation_count;
            return VALKEYMODULE_OK;
        }
        return DYNKV_NOOP;
    }
    return result;
}

/*
 * A tree dump contains only STORE events.  Validate its full target-rank
 * topology before resetting the live rank so malformed snapshots cannot leave
 * a partially replaced worker behind.  The plan begins from the post-reset
 * state: every old owner is inactive with event ID zero.
 */
