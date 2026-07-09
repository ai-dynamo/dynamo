/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_gc.h"
#include "dynkv_index.h"
#include "dynkv_lease.h"
#include "dynkv_state.h"

void gc_prioritize_pending_cleanup(RouterIndex *index) {
    index->gc_phase = DYNKV_GC_PHASE_WORKER_EPOCHS;
    index->gc_cursor_length = 0;
}

bool gc_can_begin_lease_cleanup(
    RouterIndex *index,
    uint64_t now_ms,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation,
    uint64_t lifecycle_generation) {
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    return epoch != NULL && epoch->registration_owner_set &&
           !epoch->lease_cleanup_pending &&
           epoch->registration_owner_nonce == owner_nonce &&
           epoch->registration_expires_at_ms == expires_at_ms &&
           expires_at_ms <= now_ms && index->generation_counter != UINT64_MAX &&
           cleanup_generation == index->generation_counter + 1 &&
           index->lifecycle_generation_counter != UINT64_MAX &&
           lifecycle_generation == index->lifecycle_generation_counter + 1 &&
           epoch->registration_expiry_heap_index <
               index->worker_lease_expiry_heap_length &&
           index->worker_lease_expiry_heap[epoch->registration_expiry_heap_index] == epoch;
}

static bool gc_begin_lease_cleanup(
    RouterIndex *index,
    WorkerEpoch *epoch,
    uint64_t cleanup_generation,
    uint64_t lifecycle_generation) {
    if (!router_index_worker_lease_expiry_heap_remove(index, epoch)) {
        return false;
    }
    index->generation_counter = cleanup_generation;
    epoch->generation = cleanup_generation;
    index->lifecycle_generation_counter = lifecycle_generation;
    epoch->lifecycle_generation = lifecycle_generation;
    epoch->lease_cleanup_pending = true;
    epoch->lease_cleanup_generation = cleanup_generation;
    epoch->lease_cleanup_worker_remaining = epoch->worker_state_count;
    epoch->lease_cleanup_admission_remaining = epoch->admission_rank_count;
    epoch->last_registration_expected_set = false;
    epoch->last_registration_expected_generation = 0;
    gc_prioritize_pending_cleanup(index);
    return true;
}

bool gc_apply_begin_lease_cleanup(
    RouterIndex *index,
    uint64_t now_ms,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation,
    uint64_t lifecycle_generation) {
    if (!gc_can_begin_lease_cleanup(
            index,
            now_ms,
            worker_id,
            owner_nonce,
            expires_at_ms,
            cleanup_generation,
            lifecycle_generation)) {
        return false;
    }
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    return gc_begin_lease_cleanup(
        index, epoch, cleanup_generation, lifecycle_generation);
}

bool gc_apply_begin_unregister_cleanup(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce) {
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    if (epoch == NULL || !epoch->registration_owner_set ||
        epoch->registration_owner_nonce != owner_nonce ||
        !router_index_worker_lease_can_end(index, worker_id)) {
        return false;
    }
    return gc_begin_lease_cleanup(
        index,
        epoch,
        index->generation_counter + 1,
        index->lifecycle_generation_counter + 1);
}

bool gc_can_cleanup_reservation(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation,
    const uint8_t *domain,
    uint32_t domain_length,
    uint64_t client_nonce,
    uint64_t request_nonce,
    uint32_t dp_rank,
    uint64_t reservation_expires_at_ms) {
    WorkerEpoch *epoch = gc_cleanup_epoch(
        index, worker_id, owner_nonce, expires_at_ms, cleanup_generation);
    Reservation *reservation = router_index_reservation(
        index, domain, domain_length, client_nonce, request_nonce);
    return epoch != NULL && reservation != NULL &&
           reservation->worker_id == worker_id && reservation->dp_rank == dp_rank &&
           reservation->expires_at_ms == reservation_expires_at_ms;
}

bool gc_apply_cleanup_reservation(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation,
    const uint8_t *domain,
    uint32_t domain_length,
    uint64_t client_nonce,
    uint64_t request_nonce,
    uint32_t dp_rank,
    uint64_t reservation_expires_at_ms) {
    if (!gc_can_cleanup_reservation(
            index,
            worker_id,
            owner_nonce,
            expires_at_ms,
            cleanup_generation,
            domain,
            domain_length,
            client_nonce,
            request_nonce,
            dp_rank,
            reservation_expires_at_ms)) {
        return false;
    }
    bool removed = router_index_remove_reservation(
        index,
        router_index_reservation(
            index, domain, domain_length, client_nonce, request_nonce));
    if (removed) {
        gc_prioritize_pending_cleanup(index);
    }
    return removed;
}

bool gc_can_cleanup_admission_rank(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation,
    const uint8_t *domain,
    uint32_t domain_length,
    uint32_t dp_rank,
    uint64_t incarnation) {
    WorkerEpoch *epoch = gc_cleanup_epoch(
        index, worker_id, owner_nonce, expires_at_ms, cleanup_generation);
    AdmissionRankState *rank = router_index_admission_rank(
        index, domain, domain_length, worker_id, dp_rank, false);
    return epoch != NULL && rank != NULL && rank->incarnation == incarnation &&
           rank->active_reservations == 0 &&
           rank->lease_cleanup_generation != cleanup_generation &&
           epoch->lease_cleanup_admission_remaining != 0 &&
           rank->worker_epoch_index <
               epoch->lease_cleanup_admission_remaining;
}

bool gc_apply_cleanup_admission_rank(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation,
    const uint8_t *domain,
    uint32_t domain_length,
    uint32_t dp_rank,
    uint64_t incarnation) {
    if (!gc_can_cleanup_admission_rank(
            index,
            worker_id,
            owner_nonce,
            expires_at_ms,
            cleanup_generation,
            domain,
            domain_length,
            dp_rank,
            incarnation)) {
        return false;
    }
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    AdmissionRankState *rank = router_index_admission_rank(
        index, domain, domain_length, worker_id, dp_rank, false);
    worker_epoch_swap_admission_positions(
        epoch,
        rank->worker_epoch_index,
        epoch->lease_cleanup_admission_remaining - 1);
    if (rank->incarnation == UINT64_MAX) {
        /* Saturation cannot strand a pending cleanup; recreate on successor. */
        router_index_delete_admission_rank(index, rank);
    } else {
        ++rank->incarnation;
        rank->lease_cleanup_generation = cleanup_generation;
    }
    --epoch->lease_cleanup_admission_remaining;
    gc_prioritize_pending_cleanup(index);
    return true;
}

void worker_prepare_cleanup_nodes(
    WorkerState *worker,
    uint64_t cleanup_generation) {
    if (worker->lease_cleanup_node_runtime_generation != cleanup_generation) {
        worker->lease_cleanup_node_runtime_generation = cleanup_generation;
        worker->lease_cleanup_node_remaining = worker->node_count;
    }
}

bool gc_can_cleanup_owner(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation,
    uint32_t dp_rank,
    uint64_t external_hash,
    uint64_t event_id,
    bool active) {
    WorkerEpoch *epoch = gc_cleanup_epoch(
        index, worker_id, owner_nonce, expires_at_ms, cleanup_generation);
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    IndexNode *node = router_index_node_by_external(index, external_hash);
    if (epoch == NULL || worker == NULL || node == NULL ||
        worker->lease_cleanup_generation == cleanup_generation ||
        epoch->lease_cleanup_worker_remaining == 0 ||
        worker->worker_epoch_index >= epoch->lease_cleanup_worker_remaining) {
        return false;
    }
    worker_prepare_cleanup_nodes(worker, cleanup_generation);
    Owner *owner = node_owner(node, worker);
    size_t node_position = 0;
    return owner != NULL && owner->event_id == event_id && owner->active == active &&
           owner->lease_cleanup_generation != cleanup_generation &&
           worker->lease_cleanup_node_remaining != 0 &&
           worker_node_position(worker, node, &node_position) &&
           node_position < worker->lease_cleanup_node_remaining;
}

bool gc_apply_cleanup_owner(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation,
    uint32_t dp_rank,
    uint64_t external_hash,
    uint64_t event_id,
    bool active) {
    if (!gc_can_cleanup_owner(
            index,
            worker_id,
            owner_nonce,
            expires_at_ms,
            cleanup_generation,
            dp_rank,
            external_hash,
            event_id,
            active)) {
        return false;
    }
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    IndexNode *node = router_index_node_by_external(index, external_hash);
    size_t node_position = 0;
    if (!worker_node_position(worker, node, &node_position)) {
        return false;
    }
    if (!worker_swap_node_positions(
            worker,
            node_position,
            worker->lease_cleanup_node_remaining - 1)) {
        return false;
    }
    Owner *owner = node_owner(node, worker);
    if (owner == NULL) {
        return false;
    }
    if (epoch->lifecycle_managed && !epoch->legacy_tainted &&
        worker->lifecycle_managed && !worker->legacy_tainted) {
        if (!node_remove_owner(node, worker)) {
            return false;
        }
    } else {
        owner->event_id = 0;
        owner->active = false;
        owner->lease_cleanup_generation = cleanup_generation;
    }
    worker_epoch_swap_worker_positions(
        epoch,
        worker->worker_epoch_index,
        epoch->lease_cleanup_worker_remaining - 1);
    --worker->lease_cleanup_node_remaining;
    gc_prioritize_pending_cleanup(index);
    return true;
}

bool gc_can_cleanup_worker(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation,
    uint32_t dp_rank) {
    WorkerEpoch *epoch = gc_cleanup_epoch(
        index, worker_id, owner_nonce, expires_at_ms, cleanup_generation);
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    if (epoch == NULL || worker == NULL ||
        worker->lease_cleanup_generation == cleanup_generation ||
        epoch->lease_cleanup_worker_remaining == 0 ||
        worker->worker_epoch_index >= epoch->lease_cleanup_worker_remaining ||
        epoch->lease_cleanup_admission_remaining != 0 ||
        worker->reservation_count != 0) {
        return false;
    }
    worker_prepare_cleanup_nodes(worker, cleanup_generation);
    return worker->lease_cleanup_node_remaining == 0 &&
           epoch->lease_cleanup_admission_remaining == 0;
}

bool gc_apply_cleanup_worker(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation,
    uint32_t dp_rank) {
    if (!gc_can_cleanup_worker(
            index,
            worker_id,
            owner_nonce,
            expires_at_ms,
            cleanup_generation,
            dp_rank)) {
        return false;
    }
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    worker_epoch_swap_worker_positions(
        epoch,
        worker->worker_epoch_index,
        epoch->lease_cleanup_worker_remaining - 1);
    worker->last_clear_event_id = 0;
    worker->last_clear_dedupe_event_id = 0;
    worker->has_clear_dedupe_event_id = false;
    worker->admission_registered = false;
    worker->lifecycle_tombstone = true;
    worker->lifecycle_tombstone_generation = cleanup_generation;
    worker->lease_cleanup_generation = cleanup_generation;
    --epoch->lease_cleanup_worker_remaining;
    gc_prioritize_pending_cleanup(index);
    return true;
}

bool gc_can_finalize_lease_cleanup(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation) {
    WorkerEpoch *epoch = gc_cleanup_epoch(
        index, worker_id, owner_nonce, expires_at_ms, cleanup_generation);
    return epoch != NULL && epoch->lease_cleanup_worker_remaining == 0 &&
           epoch->lease_cleanup_admission_remaining == 0 &&
           epoch->reservation_count == 0 &&
           epoch->registration_expiry_heap_index == SIZE_MAX;
}

bool gc_apply_finalize_lease_cleanup(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation) {
    if (!gc_can_finalize_lease_cleanup(
            index,
            worker_id,
            owner_nonce,
            expires_at_ms,
            cleanup_generation)) {
        return false;
    }
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    epoch->lease_cleanup_pending = false;
    epoch->lease_cleanup_generation = 0;
    epoch->registration_owner_set = false;
    epoch->registration_owner_nonce = 0;
    epoch->registration_expires_at_ms = 0;
    gc_prioritize_pending_cleanup(index);
    return true;
}

bool gc_can_remove_owner(
    RouterIndex *index,
    uint64_t external_hash,
    uint64_t worker_id,
    uint32_t dp_rank,
    uint64_t tombstone_generation) {
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    IndexNode *node = router_index_node_by_external(index, external_hash);
    Owner *owner = node == NULL || worker == NULL ? NULL : node_owner(node, worker);
    return worker != NULL && worker->lifecycle_tombstone &&
           worker->lifecycle_tombstone_generation == tombstone_generation &&
           owner != NULL && !owner->active;
}

bool gc_apply_remove_owner(
    RouterIndex *index,
    uint64_t external_hash,
    uint64_t worker_id,
    uint32_t dp_rank,
    uint64_t tombstone_generation) {
    if (!gc_can_remove_owner(
            index,
            external_hash,
            worker_id,
            dp_rank,
            tombstone_generation)) {
        return false;
    }
    return node_remove_owner(
        router_index_node_by_external(index, external_hash),
        router_index_worker(index, worker_id, dp_rank, false));
}

bool gc_can_remove_admission_rank(
    RouterIndex *index,
    const uint8_t *domain,
    uint32_t domain_length,
    uint64_t worker_id,
    uint32_t dp_rank,
    uint64_t incarnation,
    uint64_t tombstone_generation) {
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    AdmissionRankState *rank = router_index_admission_rank(
        index, domain, domain_length, worker_id, dp_rank, false);
    return worker != NULL && worker->lifecycle_tombstone &&
           worker->lifecycle_tombstone_generation == tombstone_generation &&
           rank != NULL && rank->incarnation == incarnation &&
           rank->active_reservations == 0;
}

bool gc_apply_remove_admission_rank(
    RouterIndex *index,
    const uint8_t *domain,
    uint32_t domain_length,
    uint64_t worker_id,
    uint32_t dp_rank,
    uint64_t incarnation,
    uint64_t tombstone_generation) {
    AdmissionRankState *rank = router_index_admission_rank(
        index, domain, domain_length, worker_id, dp_rank, false);
    if (!gc_can_remove_admission_rank(
            index,
            domain,
            domain_length,
            worker_id,
            dp_rank,
            incarnation,
            tombstone_generation)) {
        return false;
    }
    router_index_delete_admission_rank(index, rank);
    return true;
}

bool gc_apply_remove_node(
    RouterIndex *index,
    uint64_t external_hash,
    uint64_t parent_hash,
    uint64_t local_hash) {
    IndexNode *node = router_index_node_by_external(index, external_hash);
    return node != NULL && node->parent_external_hash == parent_hash &&
           node->local_hash == local_hash &&
           router_index_delete_ownerless_leaf(index, node);
}

bool gc_can_remove_node(
    RouterIndex *index,
    uint64_t external_hash,
    uint64_t parent_hash,
    uint64_t local_hash) {
    IndexNode *node = router_index_node_by_external(index, external_hash);
    return node != NULL && node->parent_external_hash == parent_hash &&
           node->local_hash == local_hash && node->owner_count == 0 &&
           node->child_count == 0;
}

bool gc_apply_remove_worker(
    RouterIndex *index,
    uint64_t worker_id,
    uint32_t dp_rank,
    uint64_t tombstone_generation) {
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    return worker != NULL && worker->lifecycle_tombstone &&
           worker->lifecycle_managed && !worker->legacy_tainted &&
           worker->lifecycle_tombstone_generation == tombstone_generation &&
           router_index_delete_worker_state(index, worker);
}

bool gc_can_remove_worker(
    RouterIndex *index,
    uint64_t worker_id,
    uint32_t dp_rank,
    uint64_t tombstone_generation) {
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    return worker != NULL && worker->lifecycle_tombstone &&
           worker->lifecycle_managed && !worker->legacy_tainted &&
           worker->lifecycle_tombstone_generation == tombstone_generation &&
           worker->node_count == 0 && worker->admission_rank_count == 0 &&
           worker->reservation_count == 0 && epoch != NULL &&
           !epoch->registration_owner_set;
}

bool gc_apply_remove_worker_epoch(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t generation) {
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    if (index->generation_counter == UINT64_MAX ||
        !(epoch != NULL && epoch->lifecycle_managed &&
           !epoch->legacy_tainted && !epoch->admission_retired_all &&
           epoch->generation == generation &&
           router_index_delete_worker_epoch(index, epoch))) {
        return false;
    }
    if (index->registration_gc_floor < index->lifecycle_generation_counter) {
        index->registration_gc_floor = index->lifecycle_generation_counter;
    }
    ++index->generation_counter;
    return true;
}

bool gc_can_remove_worker_epoch(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t generation) {
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    return epoch != NULL && epoch->lifecycle_managed &&
           !epoch->legacy_tainted && !epoch->admission_retired_all &&
           !epoch->registration_owner_set && epoch->generation == generation &&
           epoch->worker_state_count == 0 && epoch->admission_rank_count == 0 &&
           epoch->reservation_count == 0 &&
           epoch->registration_expiry_heap_index == SIZE_MAX &&
           index->generation_counter != UINT64_MAX;
}

bool gc_can_expire_worker_lease(
    RouterIndex *index,
    uint64_t now_ms,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms) {
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    return epoch != NULL && epoch->registration_owner_set &&
           epoch->registration_owner_nonce == owner_nonce &&
           epoch->registration_expires_at_ms == expires_at_ms &&
           expires_at_ms <= now_ms &&
           router_index_worker_lease_can_end(index, worker_id);
}

bool gc_apply_expire_worker_lease(
    RouterIndex *index,
    uint64_t now_ms,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms) {
    if (!gc_can_expire_worker_lease(
            index, now_ms, worker_id, owner_nonce, expires_at_ms)) {
        return false;
    }
    WorkerLeaseExpiry expired = {
        .worker_id = worker_id,
        .owner_nonce = owner_nonce,
        .expires_at_ms = expires_at_ms,
    };
    bool changed = false;
    return router_index_worker_lease_apply_expire(
               index, now_ms, &expired, 1, &changed) == VALKEYMODULE_OK &&
           changed;
}

ValkeyModuleDict *gc_phase_dict(RouterIndex *index) {
    switch (index->gc_phase) {
        case DYNKV_GC_PHASE_WORKERS:
            return index->workers;
        case DYNKV_GC_PHASE_ADMISSION_RANKS:
            return index->admission_ranks;
        case DYNKV_GC_PHASE_NODES:
            return index->nodes_by_external;
        case DYNKV_GC_PHASE_WORKER_EPOCHS:
            return index->worker_epochs;
        default:
            index->gc_phase = DYNKV_GC_PHASE_WORKERS;
            index->gc_cursor_length = 0;
            return index->workers;
    }
}

/* Return one record and advance a runtime-only cursor, never scanning ahead. */
bool gc_next_record(
    RouterIndex *index,
    void **data_out,
    uint8_t *phase_out) {
    for (uint8_t advanced = 0; advanced < DYNKV_GC_PHASE_COUNT; ++advanced) {
        ValkeyModuleDict *dict = gc_phase_dict(index);
        ValkeyModuleDictIter *iter = ValkeyModule_DictIteratorStartC(
            dict,
            index->gc_cursor_length == 0 ? "^" : ">",
            index->gc_cursor_length == 0 ? NULL : index->gc_cursor,
            index->gc_cursor_length);
        size_t key_length = 0;
        void *data = NULL;
        void *key = ValkeyModule_DictNextC(iter, &key_length, &data);
        if (key != NULL && key_length <= sizeof(index->gc_cursor)) {
            memcpy(index->gc_cursor, key, key_length);
            index->gc_cursor_length = key_length;
            *data_out = data;
            *phase_out = index->gc_phase;
            ValkeyModule_DictIteratorStop(iter);
            return true;
        }
        ValkeyModule_DictIteratorStop(iter);
        index->gc_phase = (uint8_t)((index->gc_phase + 1) % DYNKV_GC_PHASE_COUNT);
        index->gc_cursor_length = 0;
    }
    return false;
}
