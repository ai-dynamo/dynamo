/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_gc.h"
#include "dynkv_state.h"

bool gc_worker_is_reclaimable(
    RouterIndex *index,
    const WorkerState *worker,
    uint64_t watermark) {
    WorkerEpoch *epoch =
        router_index_worker_epoch_lookup(index, worker->worker_id);
    return worker->lifecycle_tombstone && worker->lifecycle_managed &&
           !worker->legacy_tainted && !worker->retired &&
           !worker->admission_registered &&
           worker->lifecycle_tombstone_generation != 0 &&
           worker->lifecycle_tombstone_generation <= watermark &&
           epoch != NULL && epoch->lifecycle_managed &&
           !epoch->legacy_tainted && !epoch->registration_owner_set;
}

bool router_index_delete_worker_state(
    RouterIndex *index,
    WorkerState *worker) {
    if (worker == NULL || worker->node_count != 0 ||
        worker->admission_rank_count != 0 || worker->reservation_count != 0) {
        return false;
    }
    uint8_t key[12];
    worker_key(key, worker->worker_id, worker->dp_rank);
    if (ValkeyModule_DictDelC(index->workers, key, sizeof(key), NULL) !=
        VALKEYMODULE_OK) {
        return false;
    }
    WorkerEpoch *epoch =
        router_index_worker_epoch_lookup(index, worker->worker_id);
    if (epoch != NULL) {
        worker_epoch_remove_worker_state(epoch, worker);
    }
    worker_state_free(worker);
    return true;
}

bool router_index_delete_worker_epoch(
    RouterIndex *index,
    WorkerEpoch *epoch) {
    if (epoch == NULL || epoch->registration_owner_set ||
        epoch->worker_state_count != 0 || epoch->admission_rank_count != 0 ||
        epoch->reservation_count != 0 ||
        epoch->registration_expiry_heap_index != SIZE_MAX) {
        return false;
    }
    uint8_t key[8];
    encode_u64_be(key, epoch->worker_id);
    if (ValkeyModule_DictDelC(index->worker_epochs, key, sizeof(key), NULL) !=
        VALKEYMODULE_OK) {
        return false;
    }
    worker_epoch_free(epoch);
    return true;
}

bool gc_append_remove_owner(
    Buffer *payload,
    const IndexNode *node,
    const WorkerState *worker) {
    return buffer_reserve(payload, 1 + 8 + 8 + 4 + 8) &&
           buffer_u8(payload, DYNKV_GC_REMOVE_OWNER) &&
           buffer_u64(payload, node->external_hash) &&
           buffer_u64(payload, worker->worker_id) &&
           buffer_u32(payload, worker->dp_rank) &&
           buffer_u64(payload, worker->lifecycle_tombstone_generation);
}

bool gc_append_remove_admission_rank(
    Buffer *payload,
    const AdmissionRankState *rank,
    const WorkerState *worker) {
    return buffer_reserve(payload, 1 + 4 + rank->domain_length + 8 + 4 + 8 + 8) &&
           buffer_u8(payload, DYNKV_GC_REMOVE_ADMISSION_RANK) &&
           buffer_u32(payload, rank->domain_length) &&
           buffer_bytes(payload, rank->domain, rank->domain_length) &&
           buffer_u64(payload, rank->worker_id) &&
           buffer_u32(payload, rank->dp_rank) &&
           buffer_u64(payload, rank->incarnation) &&
           buffer_u64(payload, worker->lifecycle_tombstone_generation);
}

bool gc_append_remove_node(Buffer *payload, const IndexNode *node) {
    return buffer_reserve(payload, 1 + 8 + 8 + 8) &&
           buffer_u8(payload, DYNKV_GC_REMOVE_NODE) &&
           buffer_u64(payload, node->external_hash) &&
           buffer_u64(payload, node->parent_external_hash) &&
           buffer_u64(payload, node->local_hash);
}

bool gc_append_remove_worker(Buffer *payload, const WorkerState *worker) {
    return buffer_reserve(payload, 1 + 8 + 4 + 8) &&
           buffer_u8(payload, DYNKV_GC_REMOVE_WORKER) &&
           buffer_u64(payload, worker->worker_id) &&
           buffer_u32(payload, worker->dp_rank) &&
           buffer_u64(payload, worker->lifecycle_tombstone_generation);
}

bool gc_append_remove_worker_epoch(
    Buffer *payload,
    const WorkerEpoch *epoch) {
    return buffer_reserve(payload, 1 + 8 + 8) &&
           buffer_u8(payload, DYNKV_GC_REMOVE_WORKER_EPOCH) &&
           buffer_u64(payload, epoch->worker_id) &&
           buffer_u64(payload, epoch->generation);
}

bool gc_append_cleanup_epoch_identity(
    Buffer *payload,
    const WorkerEpoch *epoch) {
    return buffer_u64(payload, epoch->worker_id) &&
           buffer_u64(payload, epoch->registration_owner_nonce) &&
           buffer_u64(payload, epoch->registration_expires_at_ms) &&
           buffer_u64(payload, epoch->lease_cleanup_generation);
}

bool gc_append_begin_lease_cleanup(
    Buffer *payload,
    const RouterIndex *index,
    const WorkerEpoch *epoch,
    uint64_t now_ms) {
    return index->generation_counter != UINT64_MAX &&
           index->lifecycle_generation_counter != UINT64_MAX &&
           buffer_reserve(payload, 1 + 7 * 8) &&
           buffer_u8(payload, DYNKV_GC_BEGIN_LEASE_CLEANUP) &&
           buffer_u64(payload, now_ms) &&
           buffer_u64(payload, epoch->worker_id) &&
           buffer_u64(payload, epoch->registration_owner_nonce) &&
           buffer_u64(payload, epoch->registration_expires_at_ms) &&
           buffer_u64(payload, index->generation_counter + 1) &&
           buffer_u64(payload, index->lifecycle_generation_counter + 1);
}

bool gc_append_cleanup_reservation(
    Buffer *payload,
    const WorkerEpoch *epoch,
    const Reservation *reservation) {
    return buffer_reserve(
               payload, 1 + 4 * 8 + 4 + reservation->domain_length + 8 + 8 + 4 + 8) &&
           buffer_u8(payload, DYNKV_GC_CLEANUP_RESERVATION) &&
           gc_append_cleanup_epoch_identity(payload, epoch) &&
           buffer_u32(payload, reservation->domain_length) &&
           buffer_bytes(payload, reservation->domain, reservation->domain_length) &&
           buffer_u64(payload, reservation->client_nonce) &&
           buffer_u64(payload, reservation->request_nonce) &&
           buffer_u32(payload, reservation->dp_rank) &&
           buffer_u64(payload, reservation->expires_at_ms);
}

bool gc_append_cleanup_admission_rank(
    Buffer *payload,
    const WorkerEpoch *epoch,
    const AdmissionRankState *rank) {
    return buffer_reserve(
               payload, 1 + 4 * 8 + 4 + rank->domain_length + 4 + 8) &&
           buffer_u8(payload, DYNKV_GC_CLEANUP_ADMISSION_RANK) &&
           gc_append_cleanup_epoch_identity(payload, epoch) &&
           buffer_u32(payload, rank->domain_length) &&
           buffer_bytes(payload, rank->domain, rank->domain_length) &&
           buffer_u32(payload, rank->dp_rank) &&
           buffer_u64(payload, rank->incarnation);
}

bool gc_append_cleanup_owner(
    Buffer *payload,
    const WorkerEpoch *epoch,
    const WorkerState *worker,
    const IndexNode *node,
    const Owner *owner) {
    return buffer_reserve(payload, 1 + 4 * 8 + 4 + 8 + 8 + 1) &&
           buffer_u8(payload, DYNKV_GC_CLEANUP_OWNER) &&
           gc_append_cleanup_epoch_identity(payload, epoch) &&
           buffer_u32(payload, worker->dp_rank) &&
           buffer_u64(payload, node->external_hash) &&
           buffer_u64(payload, owner->event_id) &&
           buffer_u8(payload, owner->active ? 1 : 0);
}

bool gc_append_cleanup_worker(
    Buffer *payload,
    const WorkerEpoch *epoch,
    const WorkerState *worker) {
    return buffer_reserve(payload, 1 + 4 * 8 + 4) &&
           buffer_u8(payload, DYNKV_GC_CLEANUP_WORKER) &&
           gc_append_cleanup_epoch_identity(payload, epoch) &&
           buffer_u32(payload, worker->dp_rank);
}

bool gc_append_finalize_lease_cleanup(
    Buffer *payload,
    const WorkerEpoch *epoch) {
    return buffer_reserve(payload, 1 + 4 * 8) &&
           buffer_u8(payload, DYNKV_GC_FINALIZE_LEASE_CLEANUP) &&
           gc_append_cleanup_epoch_identity(payload, epoch);
}

WorkerEpoch *gc_cleanup_epoch(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    uint64_t cleanup_generation) {
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    return epoch != NULL && epoch->registration_owner_set &&
                   epoch->lease_cleanup_pending &&
                   epoch->registration_owner_nonce == owner_nonce &&
                   epoch->registration_expires_at_ms == expires_at_ms &&
                   epoch->lease_cleanup_generation == cleanup_generation &&
                   epoch->generation == cleanup_generation
               ? epoch
               : NULL;
}
