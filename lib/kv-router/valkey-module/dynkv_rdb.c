/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_index.h"
#include "dynkv_persistence.h"
#include "dynkv_state.h"

void *router_index_rdb_load(ValkeyModuleIO *io, int encoding_version) {
    if (encoding_version < 1 || encoding_version > DYNKV_SNAPSHOT_VERSION) {
        return NULL;
    }
    RouterIndex *index = router_index_create();
    uint64_t worker_count = ValkeyModule_LoadUnsigned(io);
    if (worker_count > 1048576 || ValkeyModule_IsIOError(io)) {
        router_index_free(index);
        return NULL;
    }
    for (uint64_t i = 0; i < worker_count; ++i) {
        uint64_t worker_id = ValkeyModule_LoadUnsigned(io);
        uint32_t dp_rank = (uint32_t)ValkeyModule_LoadUnsigned(io);
        uint64_t last_clear = ValkeyModule_LoadUnsigned(io);
        uint64_t retired = encoding_version >= 2 ? ValkeyModule_LoadUnsigned(io) : 0;
        uint64_t mutation_generation =
            encoding_version >= 3 ? ValkeyModule_LoadUnsigned(io) : 0;
        uint64_t last_clear_dedupe =
            encoding_version >= 4 ? ValkeyModule_LoadUnsigned(io) : last_clear;
        uint64_t has_clear_dedupe =
            encoding_version >= 5 ? ValkeyModule_LoadUnsigned(io) : last_clear_dedupe != 0;
        uint64_t admission_capacity_set =
            encoding_version >= 7 ? ValkeyModule_LoadUnsigned(io) : 0;
        uint64_t admission_capacity =
            encoding_version >= 7 ? ValkeyModule_LoadUnsigned(io) : 0;
        uint64_t admission_incarnation =
            encoding_version >= 7 ? ValkeyModule_LoadUnsigned(io) : 0;
        uint64_t admission_registered =
            encoding_version >= 10 ? ValkeyModule_LoadUnsigned(io) : 1;
        uint64_t lifecycle_tombstone =
            encoding_version >= 11 ? ValkeyModule_LoadUnsigned(io) : 0;
        uint64_t lifecycle_managed =
            encoding_version >= 13 ? ValkeyModule_LoadUnsigned(io) : 0;
        uint64_t legacy_tainted =
            encoding_version >= 13 ? ValkeyModule_LoadUnsigned(io) : 1;
        uint64_t lifecycle_tombstone_generation =
            encoding_version >= 13 ? ValkeyModule_LoadUnsigned(io) : 0;
        uint64_t lease_cleanup_generation =
            encoding_version >= 14 ? ValkeyModule_LoadUnsigned(io) : 0;
        if (retired > 1 || has_clear_dedupe > 1 || admission_capacity_set > 1 ||
            admission_registered > 1 || lifecycle_tombstone > 1 ||
            lifecycle_managed > 1 || legacy_tainted > 1 ||
            (lifecycle_tombstone == 0 && lifecycle_tombstone_generation != 0) ||
            admission_capacity > UINT32_MAX ||
            (admission_capacity_set == 0 && admission_capacity != 0) ||
            (admission_capacity_set == 1 && admission_capacity == 0) ||
            ValkeyModule_IsIOError(io)) {
            router_index_free(index);
            return NULL;
        }
        WorkerState *worker = router_index_worker(index, worker_id, dp_rank, true);
        worker->last_clear_event_id = last_clear;
        worker->retired = retired == 1;
        worker->mutation_generation = mutation_generation;
        worker->last_clear_dedupe_event_id = last_clear_dedupe;
        worker->has_clear_dedupe_event_id = has_clear_dedupe == 1;
        worker->admission_capacity_set = admission_capacity_set == 1;
        worker->admission_capacity = (uint32_t)admission_capacity;
        worker->admission_incarnation = admission_incarnation;
        worker->admission_registered = admission_registered == 1;
        worker->lifecycle_tombstone = lifecycle_tombstone == 1;
        worker->lifecycle_managed = lifecycle_managed == 1;
        worker->legacy_tainted = legacy_tainted == 1;
        worker->lifecycle_tombstone_generation = lifecycle_tombstone_generation;
        worker->lease_cleanup_generation = lease_cleanup_generation;
        if (mutation_generation > index->generation_counter) {
            index->generation_counter = mutation_generation;
        }
    }
    if (encoding_version >= 4) {
        uint64_t worker_epoch_count = ValkeyModule_LoadUnsigned(io);
        if (worker_epoch_count > 1048576 || ValkeyModule_IsIOError(io)) {
            router_index_free(index);
            return NULL;
        }
        for (uint64_t i = 0; i < worker_epoch_count; ++i) {
            uint64_t worker_id = ValkeyModule_LoadUnsigned(io);
            uint64_t generation = ValkeyModule_LoadUnsigned(io);
            uint64_t lifecycle_generation =
                encoding_version >= 13 ? ValkeyModule_LoadUnsigned(io) : 0;
            uint64_t admission_retired_all =
                encoding_version >= 8 ? ValkeyModule_LoadUnsigned(io) : 0;
            uint64_t registration_owner_set =
                encoding_version >= 11 ? ValkeyModule_LoadUnsigned(io) : 0;
            uint64_t registration_owner_nonce =
                encoding_version >= 11 ? ValkeyModule_LoadUnsigned(io) : 0;
            uint64_t registration_expires_at_ms =
                encoding_version >= 11 ? ValkeyModule_LoadUnsigned(io) : 0;
            uint64_t lifecycle_managed =
                encoding_version >= 13 ? ValkeyModule_LoadUnsigned(io) : 0;
            uint64_t legacy_tainted =
                encoding_version >= 13 ? ValkeyModule_LoadUnsigned(io) : 1;
            uint64_t last_registration_expected_set =
                encoding_version >= 13 ? ValkeyModule_LoadUnsigned(io) : 0;
            uint64_t last_registration_expected_generation =
                encoding_version >= 13 ? ValkeyModule_LoadUnsigned(io) : 0;
            uint64_t lease_cleanup_pending =
                encoding_version >= 14 ? ValkeyModule_LoadUnsigned(io) : 0;
            uint64_t lease_cleanup_generation =
                encoding_version >= 14 ? ValkeyModule_LoadUnsigned(io) : 0;
            if (admission_retired_all > 1 || registration_owner_set > 1 ||
                lifecycle_managed > 1 || legacy_tainted > 1 ||
                last_registration_expected_set > 1 || lease_cleanup_pending > 1 ||
                (last_registration_expected_set == 0 &&
                 last_registration_expected_generation != 0) ||
                (registration_owner_set == 0 &&
                 (registration_owner_nonce != 0 || registration_expires_at_ms != 0)) ||
                (registration_owner_set == 1 &&
                 (registration_owner_nonce == 0 || registration_expires_at_ms == 0)) ||
                (lease_cleanup_pending == 0 && lease_cleanup_generation != 0) ||
                (lease_cleanup_pending == 1 &&
                 (registration_owner_set == 0 || lease_cleanup_generation == 0 ||
                  lease_cleanup_generation != generation)) ||
                ValkeyModule_IsIOError(io)) {
                router_index_free(index);
                return NULL;
            }
            WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, true);
            epoch->generation = generation;
            epoch->lifecycle_generation = lifecycle_generation;
            epoch->admission_retired_all = admission_retired_all == 1;
            epoch->registration_owner_set = registration_owner_set == 1;
            epoch->registration_owner_nonce = registration_owner_nonce;
            epoch->registration_expires_at_ms = registration_expires_at_ms;
            epoch->lifecycle_managed = lifecycle_managed == 1;
            epoch->legacy_tainted = legacy_tainted == 1;
            epoch->last_registration_expected_set =
                last_registration_expected_set == 1;
            epoch->last_registration_expected_generation =
                last_registration_expected_generation;
            epoch->lease_cleanup_pending = lease_cleanup_pending == 1;
            epoch->lease_cleanup_generation = lease_cleanup_generation;
            if (epoch->registration_owner_set && !epoch->lease_cleanup_pending &&
                !router_index_worker_lease_expiry_heap_insert(index, epoch)) {
                router_index_free(index);
                return NULL;
            }
            if (generation > index->generation_counter) {
                index->generation_counter = generation;
            }
            if (lifecycle_generation > index->lifecycle_generation_counter) {
                index->lifecycle_generation_counter = lifecycle_generation;
            }
        }
    }
    if (encoding_version >= 9) {
        uint64_t admission_rank_count = ValkeyModule_LoadUnsigned(io);
        if (admission_rank_count > 1048576 || ValkeyModule_IsIOError(io)) {
            router_index_free(index);
            return NULL;
        }
        for (uint64_t i = 0; i < admission_rank_count; ++i) {
            uint64_t worker_id = ValkeyModule_LoadUnsigned(io);
            uint64_t dp_rank_value = ValkeyModule_LoadUnsigned(io);
            uint64_t capacity_value = ValkeyModule_LoadUnsigned(io);
            uint64_t incarnation = ValkeyModule_LoadUnsigned(io);
            uint64_t lease_cleanup_generation =
                encoding_version >= 14 ? ValkeyModule_LoadUnsigned(io) : 0;
            size_t domain_length = 0;
            char *domain = ValkeyModule_LoadStringBuffer(io, &domain_length);
            bool valid = !ValkeyModule_IsIOError(io) && domain != NULL &&
                         dp_rank_value <= UINT32_MAX && capacity_value > 0 &&
                         capacity_value <= UINT32_MAX && incarnation != 0 &&
                         domain_length > 0 &&
                         domain_length <= DYNKV_MAX_ADMISSION_DOMAIN_LENGTH &&
                         router_index_worker(
                             index, worker_id, (uint32_t)dp_rank_value, false) != NULL &&
                         router_index_admission_rank(
                             index,
                             (const uint8_t *)domain,
                             (uint32_t)domain_length,
                             worker_id,
                             (uint32_t)dp_rank_value,
                             false) == NULL;
            if (!valid) {
                ValkeyModule_Free(domain);
                router_index_free(index);
                return NULL;
            }
            AdmissionRankState *rank = router_index_admission_rank(
                index,
                (const uint8_t *)domain,
                (uint32_t)domain_length,
                worker_id,
                (uint32_t)dp_rank_value,
                true);
            ValkeyModule_Free(domain);
            if (rank == NULL) {
                router_index_free(index);
                return NULL;
            }
            rank->capacity = (uint32_t)capacity_value;
            rank->incarnation = incarnation;
            rank->lease_cleanup_generation = lease_cleanup_generation;
        }
    }
    uint64_t node_count = ValkeyModule_LoadUnsigned(io);
    if (node_count > 10485760 || ValkeyModule_IsIOError(io)) {
        router_index_free(index);
        return NULL;
    }
    for (uint64_t i = 0; i < node_count; ++i) {
        uint64_t external_hash = ValkeyModule_LoadUnsigned(io);
        uint64_t parent_hash = ValkeyModule_LoadUnsigned(io);
        uint64_t local_hash = ValkeyModule_LoadUnsigned(io);
        uint64_t owner_count = ValkeyModule_LoadUnsigned(io);
        if (owner_count > 1048576 || ValkeyModule_IsIOError(io)) {
            router_index_free(index);
            return NULL;
        }
        IndexNode *node = router_index_add_node(index, external_hash, parent_hash, local_hash);
        if (node == NULL) {
            router_index_free(index);
            return NULL;
        }
        for (uint64_t owner_idx = 0; owner_idx < owner_count; ++owner_idx) {
            uint64_t worker_id = ValkeyModule_LoadUnsigned(io);
            uint32_t dp_rank = (uint32_t)ValkeyModule_LoadUnsigned(io);
            uint64_t event_id = ValkeyModule_LoadUnsigned(io);
            uint64_t active = ValkeyModule_LoadUnsigned(io);
            uint64_t lease_cleanup_generation =
                encoding_version >= 14 ? ValkeyModule_LoadUnsigned(io) : 0;
            if (active > 1 || ValkeyModule_IsIOError(io)) {
                router_index_free(index);
                return NULL;
            }
            WorkerState *worker = router_index_worker(index, worker_id, dp_rank, true);
            Owner *owner = node_owner_create(node, worker);
            owner->event_id = event_id;
            owner->active = active == 1;
            owner->lease_cleanup_generation = lease_cleanup_generation;
        }
    }
    if (encoding_version >= 6) {
        uint64_t reservation_count = ValkeyModule_LoadUnsigned(io);
        if (reservation_count > 1048576 || ValkeyModule_IsIOError(io)) {
            router_index_free(index);
            return NULL;
        }
        for (uint64_t i = 0; i < reservation_count; ++i) {
            uint64_t client_nonce = ValkeyModule_LoadUnsigned(io);
            uint64_t request_nonce = ValkeyModule_LoadUnsigned(io);
            uint64_t worker_id = ValkeyModule_LoadUnsigned(io);
            uint64_t dp_rank_value = ValkeyModule_LoadUnsigned(io);
            uint64_t worker_admission_incarnation =
                encoding_version >= 7 ? ValkeyModule_LoadUnsigned(io) : 0;
            uint64_t expires_at_ms = ValkeyModule_LoadUnsigned(io);
            uint64_t matched_blocks_value = ValkeyModule_LoadUnsigned(io);
            uint64_t active_reservations_at_grant_value = ValkeyModule_LoadUnsigned(io);
            size_t domain_length = 0;
            size_t request_length = 0;
            char *domain = ValkeyModule_LoadStringBuffer(io, &domain_length);
            char *request_bytes = ValkeyModule_LoadStringBuffer(io, &request_length);
            bool valid = !ValkeyModule_IsIOError(io) && domain != NULL && request_bytes != NULL &&
                         dp_rank_value <= UINT32_MAX && matched_blocks_value <= UINT32_MAX &&
                         active_reservations_at_grant_value <= UINT32_MAX &&
                         domain_length > 0 && domain_length <= DYNKV_MAX_ADMISSION_DOMAIN_LENGTH &&
                         request_length > 0 &&
                         request_length <= DYNKV_MAX_ADMISSION_REQUEST_BYTES;
            uint32_t dp_rank = (uint32_t)dp_rank_value;
            uint64_t effective_incarnation = worker_admission_incarnation;
            if (valid && encoding_version < 9) {
                WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
                if (worker == NULL) {
                    valid = false;
                } else {
                    if (effective_incarnation == 0) {
                        effective_incarnation =
                            worker->admission_incarnation == 0 ? 1 : worker->admission_incarnation;
                    }
                    valid = router_index_restore_legacy_admission_rank(
                        index,
                        (const uint8_t *)domain,
                        (uint32_t)domain_length,
                        worker_id,
                        dp_rank,
                        effective_incarnation);
                }
            }
            if (valid) {
                valid = router_index_add_reservation(
                    index,
                    (const uint8_t *)domain,
                    (uint32_t)domain_length,
                    client_nonce,
                    request_nonce,
                    worker_id,
                    dp_rank,
                    effective_incarnation,
                    expires_at_ms,
                    (uint32_t)matched_blocks_value,
                    (uint32_t)active_reservations_at_grant_value,
                    (const uint8_t *)request_bytes,
                    (uint32_t)request_length);
            }
            ValkeyModule_Free(domain);
            ValkeyModule_Free(request_bytes);
            if (!valid) {
                router_index_free(index);
                return NULL;
            }
        }
    }
    index->mutation_count = ValkeyModule_LoadUnsigned(io);
    if (encoding_version >= 4) {
        uint64_t stored_generation_counter = ValkeyModule_LoadUnsigned(io);
        if (stored_generation_counter > index->generation_counter) {
            index->generation_counter = stored_generation_counter;
        }
    }
    if (encoding_version >= 13) {
        uint64_t stored_lifecycle_generation_counter =
            ValkeyModule_LoadUnsigned(io);
        if (stored_lifecycle_generation_counter >
            index->lifecycle_generation_counter) {
            index->lifecycle_generation_counter =
                stored_lifecycle_generation_counter;
        }
        index->registration_gc_floor = ValkeyModule_LoadUnsigned(io);
        if (index->registration_gc_floor >
            index->lifecycle_generation_counter) {
            router_index_free(index);
            return NULL;
        }
    }
    if (ValkeyModule_IsIOError(io)) {
        router_index_free(index);
        return NULL;
    }
    router_index_rebuild_child_counts(index);
    if (!router_index_rebuild_lease_cleanup_state(index)) {
        router_index_free(index);
        return NULL;
    }
    return index;
}

void router_index_rdb_save(ValkeyModuleIO *io, void *value) {
    RouterIndex *index = value;
    ValkeyModule_SaveUnsigned(io, ValkeyModule_DictSize(index->workers));
    ValkeyModuleDictIter *workers =
        ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
        WorkerState *worker = data;
        ValkeyModule_SaveUnsigned(io, worker->worker_id);
        ValkeyModule_SaveUnsigned(io, worker->dp_rank);
        ValkeyModule_SaveUnsigned(io, worker->last_clear_event_id);
        ValkeyModule_SaveUnsigned(io, worker->retired ? 1 : 0);
        ValkeyModule_SaveUnsigned(io, worker->mutation_generation);
        ValkeyModule_SaveUnsigned(io, worker->last_clear_dedupe_event_id);
        ValkeyModule_SaveUnsigned(io, worker->has_clear_dedupe_event_id ? 1 : 0);
        ValkeyModule_SaveUnsigned(io, worker->admission_capacity_set ? 1 : 0);
        ValkeyModule_SaveUnsigned(io, worker->admission_capacity);
        ValkeyModule_SaveUnsigned(io, worker->admission_incarnation);
        ValkeyModule_SaveUnsigned(io, worker->admission_registered ? 1 : 0);
        ValkeyModule_SaveUnsigned(io, worker->lifecycle_tombstone ? 1 : 0);
        ValkeyModule_SaveUnsigned(io, worker->lifecycle_managed ? 1 : 0);
        ValkeyModule_SaveUnsigned(io, worker->legacy_tainted ? 1 : 0);
        ValkeyModule_SaveUnsigned(io, worker->lifecycle_tombstone_generation);
        ValkeyModule_SaveUnsigned(io, worker->lease_cleanup_generation);
    }
    ValkeyModule_DictIteratorStop(workers);

    ValkeyModule_SaveUnsigned(io, ValkeyModule_DictSize(index->worker_epochs));
    ValkeyModuleDictIter *epochs =
        ValkeyModule_DictIteratorStartC(index->worker_epochs, "^", NULL, 0);
    while (ValkeyModule_DictNextC(epochs, NULL, &data) != NULL) {
        WorkerEpoch *epoch = data;
        ValkeyModule_SaveUnsigned(io, epoch->worker_id);
        ValkeyModule_SaveUnsigned(io, epoch->generation);
        ValkeyModule_SaveUnsigned(io, epoch->lifecycle_generation);
        ValkeyModule_SaveUnsigned(io, epoch->admission_retired_all ? 1 : 0);
        ValkeyModule_SaveUnsigned(io, epoch->registration_owner_set ? 1 : 0);
        ValkeyModule_SaveUnsigned(io, epoch->registration_owner_nonce);
        ValkeyModule_SaveUnsigned(io, epoch->registration_expires_at_ms);
        ValkeyModule_SaveUnsigned(io, epoch->lifecycle_managed ? 1 : 0);
        ValkeyModule_SaveUnsigned(io, epoch->legacy_tainted ? 1 : 0);
        ValkeyModule_SaveUnsigned(
            io, epoch->last_registration_expected_set ? 1 : 0);
        ValkeyModule_SaveUnsigned(
            io, epoch->last_registration_expected_generation);
        ValkeyModule_SaveUnsigned(io, epoch->lease_cleanup_pending ? 1 : 0);
        ValkeyModule_SaveUnsigned(io, epoch->lease_cleanup_generation);
    }
    ValkeyModule_DictIteratorStop(epochs);

    ValkeyModule_SaveUnsigned(io, ValkeyModule_DictSize(index->admission_ranks));
    ValkeyModuleDictIter *admission_ranks =
        ValkeyModule_DictIteratorStartC(index->admission_ranks, "^", NULL, 0);
    while (ValkeyModule_DictNextC(admission_ranks, NULL, &data) != NULL) {
        AdmissionRankState *rank = data;
        ValkeyModule_SaveUnsigned(io, rank->worker_id);
        ValkeyModule_SaveUnsigned(io, rank->dp_rank);
        ValkeyModule_SaveUnsigned(io, rank->capacity);
        ValkeyModule_SaveUnsigned(io, rank->incarnation);
        ValkeyModule_SaveUnsigned(io, rank->lease_cleanup_generation);
        ValkeyModule_SaveStringBuffer(io, (const char *)rank->domain, rank->domain_length);
    }
    ValkeyModule_DictIteratorStop(admission_ranks);

    ValkeyModule_SaveUnsigned(io, ValkeyModule_DictSize(index->nodes_by_external));
    ValkeyModuleDictIter *nodes =
        ValkeyModule_DictIteratorStartC(index->nodes_by_external, "^", NULL, 0);
    while (ValkeyModule_DictNextC(nodes, NULL, &data) != NULL) {
        IndexNode *node = data;
        ValkeyModule_SaveUnsigned(io, node->external_hash);
        ValkeyModule_SaveUnsigned(io, node->parent_external_hash);
        ValkeyModule_SaveUnsigned(io, node->local_hash);
        ValkeyModule_SaveUnsigned(io, node->owner_count);
        for (size_t i = 0; i < node->owner_count; ++i) {
            Owner *owner = &node->owners[i];
            ValkeyModule_SaveUnsigned(io, owner->worker->worker_id);
            ValkeyModule_SaveUnsigned(io, owner->worker->dp_rank);
            ValkeyModule_SaveUnsigned(io, owner->event_id);
            ValkeyModule_SaveUnsigned(io, owner->active ? 1 : 0);
            ValkeyModule_SaveUnsigned(io, owner->lease_cleanup_generation);
        }
    }
    ValkeyModule_DictIteratorStop(nodes);

    ValkeyModule_SaveUnsigned(io, ValkeyModule_DictSize(index->reservations));
    ValkeyModuleDictIter *reservations =
        ValkeyModule_DictIteratorStartC(index->reservations, "^", NULL, 0);
    while (ValkeyModule_DictNextC(reservations, NULL, &data) != NULL) {
        Reservation *reservation = data;
        ValkeyModule_SaveUnsigned(io, reservation->client_nonce);
        ValkeyModule_SaveUnsigned(io, reservation->request_nonce);
        ValkeyModule_SaveUnsigned(io, reservation->worker_id);
        ValkeyModule_SaveUnsigned(io, reservation->dp_rank);
        ValkeyModule_SaveUnsigned(io, reservation->worker_admission_incarnation);
        ValkeyModule_SaveUnsigned(io, reservation->expires_at_ms);
        ValkeyModule_SaveUnsigned(io, reservation->matched_blocks);
        ValkeyModule_SaveUnsigned(io, reservation->active_reservations_at_grant);
        ValkeyModule_SaveStringBuffer(
            io, (const char *)reservation->domain, reservation->domain_length);
        ValkeyModule_SaveStringBuffer(
            io, (const char *)reservation->request_bytes, reservation->request_length);
    }
    ValkeyModule_DictIteratorStop(reservations);
    ValkeyModule_SaveUnsigned(io, index->mutation_count);
    ValkeyModule_SaveUnsigned(io, index->generation_counter);
    ValkeyModule_SaveUnsigned(io, index->lifecycle_generation_counter);
    ValkeyModule_SaveUnsigned(io, index->registration_gc_floor);
}

void router_index_aof_rewrite(ValkeyModuleIO *aof, ValkeyModuleString *key, void *value) {
    Buffer snapshot = {0};
    if (router_index_snapshot(value, &snapshot)) {
        ValkeyModule_EmitAOF(aof, "DYNKV.RESTORE", "sb", key, snapshot.data, snapshot.length);
    }
    buffer_free(&snapshot);
}

size_t router_index_mem_usage(const void *value) {
    const RouterIndex *index = value;
    size_t size = sizeof(*index) +
                  index->reservation_expiry_heap_capacity *
                      sizeof(*index->reservation_expiry_heap) +
                  index->worker_lease_expiry_heap_capacity *
                      sizeof(*index->worker_lease_expiry_heap);
    ValkeyModuleDictIter *nodes =
        ValkeyModule_DictIteratorStartC(index->nodes_by_external, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(nodes, NULL, &data) != NULL) {
        const IndexNode *node = data;
        size += sizeof(*node) + node->owner_capacity * sizeof(*node->owners);
    }
    ValkeyModule_DictIteratorStop(nodes);
    ValkeyModuleDictIter *workers =
        ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
    while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
        const WorkerState *worker = data;
        size += sizeof(*worker) + worker->node_capacity * sizeof(*worker->nodes) +
                worker->node_capacity * sizeof(*worker->node_owner_indices) +
                ValkeyModule_DictSize(worker->node_members) * sizeof(uint64_t);
    }
    ValkeyModule_DictIteratorStop(workers);
    ValkeyModuleDictIter *epochs =
        ValkeyModule_DictIteratorStartC(index->worker_epochs, "^", NULL, 0);
    while (ValkeyModule_DictNextC(epochs, NULL, &data) != NULL) {
        const WorkerEpoch *epoch = data;
        size += sizeof(*epoch) +
                epoch->worker_state_capacity * sizeof(*epoch->worker_states) +
                epoch->admission_rank_capacity *
                    sizeof(*epoch->admission_rank_states) +
                epoch->reservation_capacity * sizeof(*epoch->reservation_states);
    }
    ValkeyModule_DictIteratorStop(epochs);
    ValkeyModuleDictIter *admission_ranks =
        ValkeyModule_DictIteratorStartC(index->admission_ranks, "^", NULL, 0);
    while (ValkeyModule_DictNextC(admission_ranks, NULL, &data) != NULL) {
        const AdmissionRankState *rank = data;
        size += sizeof(*rank) + rank->domain_length;
    }
    ValkeyModule_DictIteratorStop(admission_ranks);
    ValkeyModuleDictIter *reservations =
        ValkeyModule_DictIteratorStartC(index->reservations, "^", NULL, 0);
    while (ValkeyModule_DictNextC(reservations, NULL, &data) != NULL) {
        const Reservation *reservation = data;
        size += sizeof(*reservation) + reservation->domain_length + reservation->request_length;
    }
    ValkeyModule_DictIteratorStop(reservations);
    return size;
}

RouterIndex *router_index_for_write(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString *key_name,
    ValkeyModuleKey **key_out) {
    return router_index_for_write_tracking_creation(
        ctx, key_name, key_out, NULL);
}

RouterIndex *router_index_for_write_tracking_creation(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString *key_name,
    ValkeyModuleKey **key_out,
    bool *created_out) {
    ValkeyModuleKey *key = ValkeyModule_OpenKey(ctx, key_name, VALKEYMODULE_READ | VALKEYMODULE_WRITE);
    int type = ValkeyModule_KeyType(key);
    if (type != VALKEYMODULE_KEYTYPE_EMPTY && ValkeyModule_ModuleTypeGetType(key) != RouterIndexType) {
        ValkeyModule_ReplyWithError(ctx, VALKEYMODULE_ERRORMSG_WRONGTYPE);
        return NULL;
    }
    RouterIndex *index = ValkeyModule_ModuleTypeGetValue(key);
    bool created = index == NULL;
    if (index == NULL) {
        index = router_index_create();
        ValkeyModule_ModuleTypeSetValue(key, RouterIndexType, index);
    }
    if (created_out != NULL) {
        *created_out = created;
    }
    *key_out = key;
    return index;
}

RouterIndex *router_index_for_read(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString *key_name,
    ValkeyModuleKey **key_out) {
    ValkeyModuleKey *key = ValkeyModule_OpenKey(ctx, key_name, VALKEYMODULE_READ);
    int type = ValkeyModule_KeyType(key);
    if (type == VALKEYMODULE_KEYTYPE_EMPTY) {
        *key_out = key;
        return NULL;
    }
    if (ValkeyModule_ModuleTypeGetType(key) != RouterIndexType) {
        ValkeyModule_ReplyWithError(ctx, VALKEYMODULE_ERRORMSG_WRONGTYPE);
        return (RouterIndex *)(uintptr_t)1;
    }
    *key_out = key;
    return ValkeyModule_ModuleTypeGetValue(key);
}
