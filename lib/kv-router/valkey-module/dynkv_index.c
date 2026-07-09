/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_index.h"
#include "dynkv_state.h"

bool router_event_header(const uint8_t *data, size_t length, EventHeader *header) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    return reader_u8(&reader, &version) && reader_u8(&reader, &header->kind) &&
           reader_u64(&reader, &header->worker_id) && reader_u32(&reader, &header->dp_rank) &&
           reader_u64(&reader, &header->event_id) && version == DYNKV_WIRE_VERSION;
}

bool router_event_remove_has_blocks(const uint8_t *data, size_t length) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint8_t kind = 0;
    uint64_t worker_id = 0;
    uint32_t dp_rank = 0;
    uint64_t event_id = 0;
    uint32_t count = 0;
    return reader_u8(&reader, &version) && reader_u8(&reader, &kind) &&
           reader_u64(&reader, &worker_id) && reader_u32(&reader, &dp_rank) &&
           reader_u64(&reader, &event_id) && reader_u32(&reader, &count) &&
           version == DYNKV_WIRE_VERSION && kind == DYNKV_EVENT_REMOVE && count != 0;
}

/*
 * Generations are opaque, globally ordered tickets.  A rank keeps its last
 * direct-mutation ticket, while a worker epoch supplies a lower bound after a
 * worker-wide clear/removal, including for ranks not yet materialized here.
 */
uint64_t router_index_rank_generation(
    RouterIndex *index,
    uint64_t worker_id,
    uint32_t dp_rank) {
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, false);
    if (worker == NULL && epoch == NULL) {
        return index->generation_counter;
    }
    uint64_t rank_generation = worker == NULL ? 0 : worker->mutation_generation;
    uint64_t worker_generation = epoch == NULL ? 0 : epoch->generation;
    return rank_generation > worker_generation ? rank_generation : worker_generation;
}

bool router_index_generation_can_advance(const RouterIndex *index) {
    return index->generation_counter != UINT64_MAX;
}

bool router_index_worker_epochs_can_advance(
    const RouterIndex *index,
    uint32_t count) {
    return (uint64_t)count <= UINT64_MAX - index->generation_counter;
}

bool router_index_advance_rank_generation(RouterIndex *index, WorkerState *worker) {
    if (!router_index_generation_can_advance(index)) {
        return false;
    }
    worker->mutation_generation = ++index->generation_counter;
    return true;
}

bool router_index_advance_worker_epoch(RouterIndex *index, uint64_t worker_id) {
    if (!router_index_generation_can_advance(index)) {
        return false;
    }
    WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, true);
    epoch->generation = ++index->generation_counter;
    return true;
}

bool router_index_advance_lifecycle_generation(
    RouterIndex *index,
    WorkerEpoch *epoch) {
    if (epoch == NULL || index->lifecycle_generation_counter == UINT64_MAX) {
        return false;
    }
    epoch->lifecycle_generation = ++index->lifecycle_generation_counter;
    return true;
}

bool router_index_mark_legacy_rank(
    RouterIndex *index,
    uint64_t worker_id,
    uint32_t dp_rank) {
    bool changed = false;
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    if (worker != NULL && !worker->legacy_tainted) {
        worker->legacy_tainted = true;
        changed = true;
    }
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    if (epoch != NULL && !epoch->legacy_tainted) {
        epoch->legacy_tainted = true;
        changed = true;
    }
    return changed;
}

bool router_index_mark_legacy_worker(RouterIndex *index, uint64_t worker_id) {
    bool changed = false;
    ValkeyModuleDictIter *workers =
        ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
        WorkerState *worker = data;
        if (worker->worker_id == worker_id && !worker->legacy_tainted) {
            worker->legacy_tainted = true;
            changed = true;
        }
    }
    ValkeyModule_DictIteratorStop(workers);
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    if (epoch != NULL && !epoch->legacy_tainted) {
        epoch->legacy_tainted = true;
        changed = true;
    }
    return changed;
}

bool worker_registration_is_live(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t now_ms) {
    WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, false);
    return epoch == NULL || !epoch->registration_owner_set ||
           (!epoch->lease_cleanup_pending &&
            epoch->registration_expires_at_ms > now_ms);
}

bool worker_registration_owner_is_live(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t now_ms) {
    WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, false);
    return epoch != NULL && epoch->registration_owner_set &&
           !epoch->lease_cleanup_pending &&
           epoch->registration_owner_nonce == owner_nonce &&
           epoch->registration_expires_at_ms > now_ms;
}

bool router_index_worker_cleanup_pending(
    RouterIndex *index,
    uint64_t worker_id) {
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    return epoch != NULL && epoch->lease_cleanup_pending;
}

bool router_index_worker_lease_can_end(RouterIndex *index, uint64_t worker_id) {
    if (!router_index_generation_can_advance(index) ||
        index->lifecycle_generation_counter == UINT64_MAX) {
        return false;
    }
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    if (epoch == NULL || epoch->lease_cleanup_pending) {
        return false;
    }
    for (size_t i = 0; i < epoch->admission_rank_count; ++i) {
        if (epoch->admission_rank_states[i]->incarnation == UINT64_MAX) {
            return false;
        }
    }
    return true;
}

/*
 * End an owner lease without installing a permanent retirement fence. Worker
 * states remain as counted tombstones because inactive node owners point at
 * them and delayed legacy APPLY remains wire-compatible.
 */
bool router_index_end_worker_lease(RouterIndex *index, WorkerEpoch *epoch) {
    if (epoch == NULL || !epoch->registration_owner_set ||
        epoch->lease_cleanup_pending ||
        !router_index_worker_lease_can_end(index, epoch->worker_id) ||
        !router_index_worker_lease_expiry_heap_remove(index, epoch)) {
        return false;
    }
    for (size_t i = 0; i < epoch->worker_state_count; ++i) {
        WorkerState *worker = epoch->worker_states[i];
        (void)router_index_deactivate_worker(worker);
        /*
         * Event IDs restart with a new owner process. The owner nonce and
         * advanced worker generation fence the old incarnation, so its
         * CLEAR dedupe watermark must not reject the successor's IDs.
         */
        worker->last_clear_dedupe_event_id = 0;
        worker->has_clear_dedupe_event_id = false;
        worker->admission_registered = false;
        worker->lifecycle_tombstone = true;
    }
    for (size_t i = 0; i < epoch->admission_rank_count; ++i) {
        (void)admission_rank_advance_incarnation(
            epoch->admission_rank_states[i]);
    }
    while (epoch->reservation_count != 0) {
        if (!router_index_remove_reservation(
                index, epoch->reservation_states[epoch->reservation_count - 1])) {
            return false;
        }
    }
    if (!router_index_advance_worker_epoch(index, epoch->worker_id)) {
        return false;
    }
    if (!router_index_advance_lifecycle_generation(index, epoch)) {
        return false;
    }
    for (size_t i = 0; i < epoch->worker_state_count; ++i) {
        WorkerState *worker = epoch->worker_states[i];
        if (worker->lifecycle_tombstone) {
            worker->lifecycle_tombstone_generation = epoch->generation;
        }
    }
    epoch->registration_owner_set = false;
    epoch->registration_owner_nonce = 0;
    epoch->registration_expires_at_ms = 0;
    return true;
}

bool router_index_can_advance_for_event(
    const RouterIndex *index,
    const EventHeader *header) {
    (void)header;
    return router_index_generation_can_advance(index);
}

bool router_index_advance_for_event(RouterIndex *index, const EventHeader *header) {
    if (header->kind == DYNKV_EVENT_CLEAR) {
        return router_index_advance_worker_epoch(index, header->worker_id);
    }
    WorkerState *worker =
        router_index_worker(index, header->worker_id, header->dp_rank, true);
    return router_index_advance_rank_generation(index, worker);
}

IndexNode *router_index_node_by_external(RouterIndex *index, uint64_t external_hash) {
    uint8_t key[8];
    encode_u64_be(key, external_hash);
    return ValkeyModule_DictGetC(index->nodes_by_external, key, sizeof(key), NULL);
}

IndexNode *router_index_child(
    RouterIndex *index,
    uint64_t parent_external_hash,
    uint64_t local_hash) {
    uint8_t key[16];
    child_key(key, parent_external_hash, local_hash);
    return ValkeyModule_DictGetC(index->children_by_parent_and_local, key, sizeof(key), NULL);
}

IndexNode *router_index_add_node(
    RouterIndex *index,
    uint64_t external_hash,
    uint64_t parent_external_hash,
    uint64_t local_hash) {
    IndexNode *existing = router_index_node_by_external(index, external_hash);
    if (existing != NULL) {
        if (existing->parent_external_hash != parent_external_hash || existing->local_hash != local_hash) {
            return NULL;
        }
        return existing;
    }

    IndexNode *node = ValkeyModule_Calloc(1, sizeof(*node));
    node->external_hash = external_hash;
    node->parent_external_hash = parent_external_hash;
    node->local_hash = local_hash;

    uint8_t external_key[8];
    uint8_t edge_key[16];
    encode_u64_be(external_key, external_hash);
    child_key(edge_key, parent_external_hash, local_hash);

    IndexNode *child = ValkeyModule_DictGetC(
        index->children_by_parent_and_local, edge_key, sizeof(edge_key), NULL);
    if (child != NULL && child->external_hash != external_hash) {
        index_node_free(node);
        return NULL;
    }

    ValkeyModule_DictSetC(index->nodes_by_external, external_key, sizeof(external_key), node);
    ValkeyModule_DictSetC(index->children_by_parent_and_local, edge_key, sizeof(edge_key), node);
    if (parent_external_hash != DYNKV_ROOT_PARENT) {
        IndexNode *parent = router_index_node_by_external(index, parent_external_hash);
        if (parent != NULL) {
            ++parent->child_count;
        }
    }
    return node;
}

void router_index_rebuild_child_counts(RouterIndex *index) {
    ValkeyModuleDictIter *nodes =
        ValkeyModule_DictIteratorStartC(index->nodes_by_external, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(nodes, NULL, &data) != NULL) {
        ((IndexNode *)data)->child_count = 0;
    }
    ValkeyModule_DictIteratorStop(nodes);
    nodes = ValkeyModule_DictIteratorStartC(index->nodes_by_external, "^", NULL, 0);
    while (ValkeyModule_DictNextC(nodes, NULL, &data) != NULL) {
        IndexNode *node = data;
        if (node->parent_external_hash == DYNKV_ROOT_PARENT) {
            continue;
        }
        IndexNode *parent =
            router_index_node_by_external(index, node->parent_external_hash);
        if (parent != NULL) {
            ++parent->child_count;
        }
    }
    ValkeyModule_DictIteratorStop(nodes);
}

Owner *node_owner(IndexNode *node, WorkerState *worker) {
    uint8_t key[8];
    encode_u64_be(key, node->external_hash);
    void *encoded =
        ValkeyModule_DictGetC(worker->node_members, key, sizeof(key), NULL);
    if (encoded == NULL) {
        return NULL;
    }
    size_t worker_position = (size_t)(uintptr_t)encoded - 1;
    if (worker_position >= worker->node_count ||
        worker->nodes[worker_position] != node) {
        return NULL;
    }
    size_t owner_position = worker->node_owner_indices[worker_position];
    return owner_position < node->owner_count &&
                   node->owners[owner_position].worker == worker
               ? &node->owners[owner_position]
               : NULL;
}

bool worker_node_position(
    WorkerState *worker,
    IndexNode *node,
    size_t *position_out) {
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
    *position_out = position;
    return true;
}

bool worker_record_node(
    WorkerState *worker,
    IndexNode *node,
    size_t owner_position) {
    uint8_t key[8];
    encode_u64_be(key, node->external_hash);
    if (ValkeyModule_DictGetC(worker->node_members, key, sizeof(key), NULL) != NULL) {
        return true;
    }
    if (worker->node_count == worker->node_capacity) {
        size_t capacity = worker->node_capacity == 0 ? 8 : worker->node_capacity * 2;
        worker->nodes = ValkeyModule_Realloc(worker->nodes, capacity * sizeof(*worker->nodes));
        worker->node_owner_indices = ValkeyModule_Realloc(
            worker->node_owner_indices,
            capacity * sizeof(*worker->node_owner_indices));
        worker->node_capacity = capacity;
    }
    size_t worker_position = worker->node_count++;
    worker->nodes[worker_position] = node;
    worker->node_owner_indices[worker_position] = owner_position;
    ValkeyModule_DictSetC(
        worker->node_members,
        key,
        sizeof(key),
        (void *)(uintptr_t)worker->node_count);
    if (worker->epoch != NULL) {
        ++worker->epoch->node_membership_count;
    }
    return true;
}

Owner *node_owner_create(IndexNode *node, WorkerState *worker) {
    Owner *owner = node_owner(node, worker);
    if (owner != NULL) {
        return owner;
    }
    if (node->owner_count == node->owner_capacity) {
        size_t capacity = node->owner_capacity == 0 ? 4 : node->owner_capacity * 2;
        node->owners = ValkeyModule_Realloc(node->owners, capacity * sizeof(*node->owners));
        node->owner_capacity = capacity;
    }
    size_t owner_position = node->owner_count++;
    owner = &node->owners[owner_position];
    owner->worker = worker;
    owner->event_id = 0;
    owner->active = false;
    owner->lease_cleanup_generation = 0;
    worker_record_node(worker, node, owner_position);
    return owner;
}

bool worker_replace_node_position(
    WorkerState *worker,
    IndexNode *node,
    size_t expected_position,
    size_t new_position) {
    uint8_t key[8];
    encode_u64_be(key, node->external_hash);
    void *encoded =
        ValkeyModule_DictGetC(worker->node_members, key, sizeof(key), NULL);
    if (encoded != (void *)(uintptr_t)(expected_position + 1)) {
        return false;
    }
    /*
     * DictReplaceC overwrites an existing rax value but reports ERR for the
     * replacement case. Verify the stored value instead of interpreting that
     * return as mutation failure.
     */
    (void)ValkeyModule_DictReplaceC(
        worker->node_members,
        key,
        sizeof(key),
        (void *)(uintptr_t)(new_position + 1));
    return ValkeyModule_DictGetC(
               worker->node_members, key, sizeof(key), NULL) ==
           (void *)(uintptr_t)(new_position + 1);
}

bool worker_swap_node_positions(
    WorkerState *worker,
    size_t left,
    size_t right) {
    if (left == right) {
        return true;
    }
    IndexNode *left_node = worker->nodes[left];
    IndexNode *right_node = worker->nodes[right];
    size_t left_owner = worker->node_owner_indices[left];
    size_t right_owner = worker->node_owner_indices[right];
    if (!worker_replace_node_position(
            worker, right_node, right, left)) {
        return false;
    }
    if (!worker_replace_node_position(
            worker, left_node, left, right)) {
        bool restored = worker_replace_node_position(
            worker, right_node, left, right);
        ValkeyModule_Assert(restored);
        return false;
    }
    worker->nodes[left] = right_node;
    worker->node_owner_indices[left] = right_owner;
    worker->nodes[right] = left_node;
    worker->node_owner_indices[right] = left_owner;
    return true;
}

/*
 * Rebuild a semantic-marker partition after RDB/AOF/full-sync load. Runtime
 * array positions are deliberately not persisted as cursors.
 */
bool worker_rebuild_cleanup_node_partition(
    WorkerState *worker,
    uint64_t cleanup_generation) {
    size_t remaining = 0;
    for (size_t i = 0; i < worker->node_count; ++i) {
        Owner *owner = node_owner(worker->nodes[i], worker);
        if (owner == NULL) {
            return false;
        }
        if (owner->lease_cleanup_generation != cleanup_generation) {
            if (!worker_swap_node_positions(worker, remaining, i)) {
                return false;
            }
            ++remaining;
        }
    }
    worker->lease_cleanup_node_runtime_generation = cleanup_generation;
    worker->lease_cleanup_node_remaining = remaining;
    return true;
}

bool router_index_rebuild_lease_cleanup_state(RouterIndex *index) {
    ValkeyModuleDictIter *epochs =
        ValkeyModule_DictIteratorStartC(index->worker_epochs, "^", NULL, 0);
    void *data = NULL;
    bool valid = true;
    bool pending_seen = false;
    while (ValkeyModule_DictNextC(epochs, NULL, &data) != NULL && valid) {
        WorkerEpoch *epoch = data;
        if (!epoch->lease_cleanup_pending) {
            valid = epoch->lease_cleanup_generation == 0;
            continue;
        }
        uint64_t generation = epoch->lease_cleanup_generation;
        pending_seen = true;
        if (!epoch->registration_owner_set || generation == 0 ||
            generation != epoch->generation ||
            epoch->registration_expiry_heap_index != SIZE_MAX) {
            valid = false;
            break;
        }

        size_t worker_remaining = 0;
        for (size_t i = 0; i < epoch->worker_state_count; ++i) {
            WorkerState *worker = epoch->worker_states[i];
            bool pending = worker->lease_cleanup_generation != generation;
            if (pending) {
                WorkerState *moved = epoch->worker_states[worker_remaining];
                epoch->worker_states[worker_remaining] = worker;
                epoch->worker_states[i] = moved;
                worker->worker_epoch_index = worker_remaining;
                moved->worker_epoch_index = i;
                ++worker_remaining;
            }
        }
        epoch->lease_cleanup_worker_remaining = worker_remaining;
        for (size_t i = 0; i < epoch->worker_state_count; ++i) {
            WorkerState *worker = epoch->worker_states[i];
            if (!worker_rebuild_cleanup_node_partition(worker, generation) ||
                (worker->lease_cleanup_generation == generation &&
                 worker->lease_cleanup_node_remaining != 0)) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            break;
        }

        size_t admission_remaining = 0;
        for (size_t i = 0; i < epoch->admission_rank_count; ++i) {
            AdmissionRankState *rank = epoch->admission_rank_states[i];
            if (rank->lease_cleanup_generation != generation) {
                AdmissionRankState *moved =
                    epoch->admission_rank_states[admission_remaining];
                epoch->admission_rank_states[admission_remaining] = rank;
                epoch->admission_rank_states[i] = moved;
                rank->worker_epoch_index = admission_remaining;
                moved->worker_epoch_index = i;
                ++admission_remaining;
            }
        }
        epoch->lease_cleanup_admission_remaining = admission_remaining;
    }
    ValkeyModule_DictIteratorStop(epochs);
    if (valid && pending_seen) {
        index->gc_phase = DYNKV_GC_PHASE_WORKER_EPOCHS;
        index->gc_cursor_length = 0;
    }
    return valid;
}
