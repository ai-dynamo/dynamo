/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_index.h"
#include "dynkv_persistence.h"
#include "dynkv_state.h"

void admission_request_free(AdmissionRequest *request) {
    ValkeyModule_Free(request->candidates);
    memset(request, 0, sizeof(*request));
}

/*
 * Reserve payload:
 *   u8 version, u32 domain-length, domain bytes, u64 client nonce,
 *   u64 request nonce, u64 lease ms, u32 local-hash-count, local hashes,
 *   u32 candidate-count,
 *   candidate-count * (u64 worker, u32 rank, u32 capacity).
 */
int admission_request_parse(
    RouterIndex *index,
    const uint8_t *data,
    size_t length,
    AdmissionRequest *request,
    MatchScores *scores,
    uint64_t now_ms) {
    memset(request, 0, sizeof(*request));
    if (length > UINT32_MAX) {
        return VALKEYMODULE_ERR;
    }
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint32_t hash_count = 0;
    if (!reader_u8(&reader, &version) || !reader_u32(&reader, &request->domain_length) ||
        version != DYNKV_ADMISSION_WIRE_VERSION || request->domain_length == 0 ||
        request->domain_length > DYNKV_MAX_ADMISSION_DOMAIN_LENGTH ||
        reader.length - reader.offset < request->domain_length) {
        return VALKEYMODULE_ERR;
    }
    request->domain = reader.data + reader.offset;
    reader.offset += request->domain_length;
    if (!reader_u64(&reader, &request->client_nonce) ||
        !reader_u64(&reader, &request->request_nonce) || !reader_u64(&reader, &request->lease_ms) ||
        !reader_u32(&reader, &hash_count) || request->lease_ms == 0 ||
        request->lease_ms > DYNKV_MAX_ADMISSION_LEASE_MS ||
        hash_count > DYNKV_MAX_MATCH_HASHES ||
        reader.length - reader.offset < (size_t)hash_count * sizeof(uint64_t) ||
        router_index_collect_matches_from_reader(index, &reader, hash_count, scores, now_ms) !=
            VALKEYMODULE_OK ||
        !reader_u32(&reader, &request->candidate_count) ||
        request->candidate_count > DYNKV_MAX_ADMISSION_CANDIDATES ||
        reader.length - reader.offset != (size_t)request->candidate_count * 16) {
        match_scores_free(scores);
        return VALKEYMODULE_ERR;
    }
    if (request->candidate_count != 0) {
        request->candidates = ValkeyModule_Alloc(
            (size_t)request->candidate_count * sizeof(*request->candidates));
    }
    for (uint32_t i = 0; i < request->candidate_count; ++i) {
        AdmissionCandidate *candidate = &request->candidates[i];
        if (!reader_u64(&reader, &candidate->worker_id) ||
            !reader_u32(&reader, &candidate->dp_rank) ||
            !reader_u32(&reader, &candidate->capacity)) {
            admission_request_free(request);
            match_scores_free(scores);
            return VALKEYMODULE_ERR;
        }
    }
    request->raw_bytes = data;
    request->raw_length = (uint32_t)length;
    return VALKEYMODULE_OK;
}

AdmissionSelection router_index_admission_select(
    RouterIndex *index,
    const AdmissionRequest *request,
    const MatchScores *scores,
    uint64_t now_ms) {
    AdmissionSelection selected = {0};
    uint32_t selected_load = 0;
    for (uint32_t i = 0; i < request->candidate_count; ++i) {
        const AdmissionCandidate *candidate = &request->candidates[i];
        WorkerState *worker =
            router_index_worker(index, candidate->worker_id, candidate->dp_rank, false);
        WorkerEpoch *epoch = router_index_worker_epoch(index, candidate->worker_id, false);
        AdmissionRankState *rank = router_index_admission_rank(
            index,
            request->domain,
            request->domain_length,
            candidate->worker_id,
            candidate->dp_rank,
            false);
        if (worker == NULL || worker->retired || !worker->admission_registered ||
            candidate->capacity == 0 ||
            candidate->capacity == DYNKV_UNKNOWN_ADMISSION_CAPACITY ||
            (epoch != NULL && epoch->admission_retired_all) ||
            !worker_registration_is_live(index, candidate->worker_id, now_ms) ||
            !admission_rank_capacity_matches(rank, candidate->capacity)) {
            continue;
        }
        uint32_t capacity = rank == NULL ? candidate->capacity : rank->capacity;
        uint32_t active_reservations = rank == NULL ? 0 : rank->active_reservations;
        if (active_reservations >= capacity || active_reservations == UINT32_MAX) {
            continue;
        }
        MatchScore *score = worker == NULL ? NULL : match_scores_find(scores, worker);
        uint32_t matched_blocks = score == NULL ? 0 : score->matched_blocks;
        uint32_t effective_load = active_reservations;
        bool better = !selected.selected || matched_blocks > selected.matched_blocks ||
                      (matched_blocks == selected.matched_blocks &&
                       effective_load < selected_load) ||
                      (matched_blocks == selected.matched_blocks &&
                       effective_load == selected_load &&
                       (candidate->worker_id < selected.worker_id ||
                        (candidate->worker_id == selected.worker_id &&
                         candidate->dp_rank < selected.dp_rank)));
        if (better) {
            selected.selected = true;
            selected.worker_id = candidate->worker_id;
            selected.dp_rank = candidate->dp_rank;
            selected.worker_admission_incarnation =
                rank == NULL || rank->incarnation == 0 ? 1 : rank->incarnation;
            selected.expires_at_ms = now_ms + request->lease_ms;
            selected.matched_blocks = matched_blocks;
            selected.active_reservations_at_grant = active_reservations + 1;
            selected.capacity = capacity;
            selected_load = effective_load;
        }
    }
    return selected;
}

bool admission_response_append(
    Buffer *response,
    uint8_t status,
    const Reservation *reservation) {
    if (!buffer_u8(response, DYNKV_ADMISSION_WIRE_VERSION) || !buffer_u8(response, status)) {
        return false;
    }
    if (status != DYNKV_ADMISSION_RESERVED) {
        return true;
    }
    return reservation != NULL && buffer_u64(response, reservation->client_nonce) &&
           buffer_u64(response, reservation->request_nonce) &&
           buffer_u64(response, reservation->worker_id) &&
           buffer_u32(response, reservation->dp_rank) &&
           buffer_u64(response, reservation->expires_at_ms) &&
           buffer_u32(response, reservation->matched_blocks) &&
           buffer_u32(response, reservation->active_reservations_at_grant);
}

bool reservation_matches_request(
    const Reservation *reservation,
    const AdmissionRequest *request) {
    return reservation->request_length == request->raw_length &&
           memcmp(reservation->request_bytes, request->raw_bytes, request->raw_length) == 0;
}

bool snapshot_append_worker(Buffer *snapshot, WorkerState *worker) {
    return buffer_u64(snapshot, worker->worker_id) && buffer_u32(snapshot, worker->dp_rank) &&
           buffer_u64(snapshot, worker->last_clear_event_id) &&
           buffer_u8(snapshot, worker->retired ? 1 : 0) &&
           buffer_u64(snapshot, worker->mutation_generation) &&
           buffer_u64(snapshot, worker->last_clear_dedupe_event_id) &&
           buffer_u8(snapshot, worker->has_clear_dedupe_event_id ? 1 : 0) &&
           buffer_u8(snapshot, worker->admission_capacity_set ? 1 : 0) &&
           buffer_u32(snapshot, worker->admission_capacity) &&
           buffer_u64(snapshot, worker->admission_incarnation) &&
           buffer_u8(snapshot, worker->admission_registered ? 1 : 0) &&
           buffer_u8(snapshot, worker->lifecycle_tombstone ? 1 : 0) &&
           buffer_u8(snapshot, worker->lifecycle_managed ? 1 : 0) &&
           buffer_u8(snapshot, worker->legacy_tainted ? 1 : 0) &&
           buffer_u64(snapshot, worker->lifecycle_tombstone_generation) &&
           buffer_u64(snapshot, worker->lease_cleanup_generation);
}

bool snapshot_append_worker_epoch(Buffer *snapshot, WorkerEpoch *epoch) {
    return buffer_u64(snapshot, epoch->worker_id) && buffer_u64(snapshot, epoch->generation) &&
           buffer_u64(snapshot, epoch->lifecycle_generation) &&
           buffer_u8(snapshot, epoch->admission_retired_all ? 1 : 0) &&
           buffer_u8(snapshot, epoch->registration_owner_set ? 1 : 0) &&
           buffer_u64(snapshot, epoch->registration_owner_nonce) &&
           buffer_u64(snapshot, epoch->registration_expires_at_ms) &&
           buffer_u8(snapshot, epoch->lifecycle_managed ? 1 : 0) &&
           buffer_u8(snapshot, epoch->legacy_tainted ? 1 : 0) &&
           buffer_u8(snapshot, epoch->last_registration_expected_set ? 1 : 0) &&
           buffer_u64(snapshot, epoch->last_registration_expected_generation) &&
           buffer_u8(snapshot, epoch->lease_cleanup_pending ? 1 : 0) &&
           buffer_u64(snapshot, epoch->lease_cleanup_generation);
}

bool snapshot_append_admission_rank(Buffer *snapshot, AdmissionRankState *rank) {
    return buffer_u64(snapshot, rank->worker_id) && buffer_u32(snapshot, rank->dp_rank) &&
           buffer_u32(snapshot, rank->capacity) && buffer_u64(snapshot, rank->incarnation) &&
           buffer_u64(snapshot, rank->lease_cleanup_generation) &&
           buffer_u32(snapshot, rank->domain_length) &&
           buffer_bytes(snapshot, rank->domain, rank->domain_length);
}

bool snapshot_append_reservation(Buffer *snapshot, Reservation *reservation) {
    return buffer_u64(snapshot, reservation->client_nonce) &&
           buffer_u64(snapshot, reservation->request_nonce) &&
           buffer_u64(snapshot, reservation->worker_id) &&
           buffer_u32(snapshot, reservation->dp_rank) &&
           buffer_u64(snapshot, reservation->worker_admission_incarnation) &&
           buffer_u64(snapshot, reservation->expires_at_ms) &&
           buffer_u32(snapshot, reservation->matched_blocks) &&
           buffer_u32(snapshot, reservation->active_reservations_at_grant) &&
           buffer_u32(snapshot, reservation->domain_length) &&
           buffer_bytes(snapshot, reservation->domain, reservation->domain_length) &&
           buffer_u32(snapshot, reservation->request_length) &&
           buffer_bytes(snapshot, reservation->request_bytes, reservation->request_length);
}

bool snapshot_append_node(Buffer *snapshot, IndexNode *node) {
    if (!buffer_u64(snapshot, node->external_hash) ||
        !buffer_u64(snapshot, node->parent_external_hash) ||
        !buffer_u64(snapshot, node->local_hash) ||
        !buffer_u32(snapshot, (uint32_t)node->owner_count)) {
        return false;
    }
    for (size_t i = 0; i < node->owner_count; ++i) {
        Owner *owner = &node->owners[i];
        if (!buffer_u64(snapshot, owner->worker->worker_id) ||
            !buffer_u32(snapshot, owner->worker->dp_rank) ||
            !buffer_u64(snapshot, owner->event_id) ||
            !buffer_u8(snapshot, owner->active ? 1 : 0) ||
            !buffer_u64(snapshot, owner->lease_cleanup_generation)) {
            return false;
        }
    }
    return true;
}

bool router_index_snapshot(RouterIndex *index, Buffer *snapshot) {
    uint64_t worker_count = ValkeyModule_DictSize(index->workers);
    uint64_t worker_epoch_count = ValkeyModule_DictSize(index->worker_epochs);
    uint64_t admission_rank_count = ValkeyModule_DictSize(index->admission_ranks);
    uint64_t node_count = ValkeyModule_DictSize(index->nodes_by_external);
    uint64_t reservation_count = ValkeyModule_DictSize(index->reservations);
    if (worker_count > UINT32_MAX || worker_epoch_count > UINT32_MAX || node_count > UINT32_MAX ||
        reservation_count > UINT32_MAX || admission_rank_count > UINT32_MAX ||
        !buffer_u8(snapshot, DYNKV_SNAPSHOT_VERSION) ||
        !buffer_u32(snapshot, (uint32_t)worker_count)) {
        return false;
    }
    ValkeyModuleDictIter *workers =
        ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
        if (!snapshot_append_worker(snapshot, data)) {
            ValkeyModule_DictIteratorStop(workers);
            return false;
        }
    }
    ValkeyModule_DictIteratorStop(workers);

    if (!buffer_u32(snapshot, (uint32_t)worker_epoch_count)) {
        return false;
    }
    ValkeyModuleDictIter *epochs =
        ValkeyModule_DictIteratorStartC(index->worker_epochs, "^", NULL, 0);
    while (ValkeyModule_DictNextC(epochs, NULL, &data) != NULL) {
        if (!snapshot_append_worker_epoch(snapshot, data)) {
            ValkeyModule_DictIteratorStop(epochs);
            return false;
        }
    }
    ValkeyModule_DictIteratorStop(epochs);

    if (!buffer_u32(snapshot, (uint32_t)admission_rank_count)) {
        return false;
    }
    ValkeyModuleDictIter *admission_ranks =
        ValkeyModule_DictIteratorStartC(index->admission_ranks, "^", NULL, 0);
    while (ValkeyModule_DictNextC(admission_ranks, NULL, &data) != NULL) {
        if (!snapshot_append_admission_rank(snapshot, data)) {
            ValkeyModule_DictIteratorStop(admission_ranks);
            return false;
        }
    }
    ValkeyModule_DictIteratorStop(admission_ranks);

    if (!buffer_u32(snapshot, (uint32_t)node_count)) {
        return false;
    }
    ValkeyModuleDictIter *nodes =
        ValkeyModule_DictIteratorStartC(index->nodes_by_external, "^", NULL, 0);
    while (ValkeyModule_DictNextC(nodes, NULL, &data) != NULL) {
        if (!snapshot_append_node(snapshot, data)) {
            ValkeyModule_DictIteratorStop(nodes);
            return false;
        }
    }
    ValkeyModule_DictIteratorStop(nodes);
    if (!buffer_u32(snapshot, (uint32_t)reservation_count)) {
        return false;
    }
    ValkeyModuleDictIter *reservations =
        ValkeyModule_DictIteratorStartC(index->reservations, "^", NULL, 0);
    while (ValkeyModule_DictNextC(reservations, NULL, &data) != NULL) {
        if (!snapshot_append_reservation(snapshot, data)) {
            ValkeyModule_DictIteratorStop(reservations);
            return false;
        }
    }
    ValkeyModule_DictIteratorStop(reservations);
    return buffer_u64(snapshot, index->mutation_count) &&
           buffer_u64(snapshot, index->generation_counter) &&
           buffer_u64(snapshot, index->lifecycle_generation_counter) &&
           buffer_u64(snapshot, index->registration_gc_floor);
}

RouterIndex *router_index_from_snapshot(const uint8_t *data, size_t length) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint32_t worker_count = 0;
    if (!reader_u8(&reader, &version) || !reader_u32(&reader, &worker_count) ||
        (version < 1 || version > DYNKV_SNAPSHOT_VERSION) ||
        worker_count > 1048576) {
        return NULL;
    }
    RouterIndex *index = router_index_create();
    for (uint32_t i = 0; i < worker_count; ++i) {
        uint64_t worker_id = 0;
        uint32_t dp_rank = 0;
        uint64_t last_clear = 0;
        uint8_t retired = 0;
        uint64_t mutation_generation = 0;
        uint64_t last_clear_dedupe = 0;
        uint8_t has_clear_dedupe = 0;
        uint8_t admission_capacity_set = 0;
        uint32_t admission_capacity = 0;
        uint64_t admission_incarnation = 0;
        uint8_t admission_registered = 1;
        uint8_t lifecycle_tombstone = 0;
        uint8_t lifecycle_managed = 0;
        uint8_t legacy_tainted = version < 13 ? 1 : 0;
        uint64_t lifecycle_tombstone_generation = 0;
        uint64_t lease_cleanup_generation = 0;
        if (!reader_u64(&reader, &worker_id) || !reader_u32(&reader, &dp_rank) ||
            !reader_u64(&reader, &last_clear) ||
            (version >= 2 && !reader_u8(&reader, &retired)) || retired > 1 ||
            (version >= 3 && !reader_u64(&reader, &mutation_generation)) ||
            (version >= 4 && !reader_u64(&reader, &last_clear_dedupe)) ||
            (version >= 5 &&
             !reader_u8(&reader, &has_clear_dedupe)) ||
            (version >= 7 &&
             (!reader_u8(&reader, &admission_capacity_set) ||
              !reader_u32(&reader, &admission_capacity) ||
              !reader_u64(&reader, &admission_incarnation))) ||
            (version >= 10 && !reader_u8(&reader, &admission_registered)) ||
            (version >= 11 && !reader_u8(&reader, &lifecycle_tombstone)) ||
            (version >= 13 &&
             (!reader_u8(&reader, &lifecycle_managed) ||
              !reader_u8(&reader, &legacy_tainted) ||
              !reader_u64(&reader, &lifecycle_tombstone_generation))) ||
            (version >= 14 &&
             !reader_u64(&reader, &lease_cleanup_generation)) ||
            has_clear_dedupe > 1 || admission_capacity_set > 1 ||
            admission_registered > 1 || lifecycle_tombstone > 1 ||
            lifecycle_managed > 1 || legacy_tainted > 1 ||
            (lifecycle_tombstone == 0 && lifecycle_tombstone_generation != 0) ||
            (admission_capacity_set == 0 && admission_capacity != 0) ||
            (admission_capacity_set == 1 && admission_capacity == 0)) {
            router_index_free(index);
            return NULL;
        }
        WorkerState *worker = router_index_worker(index, worker_id, dp_rank, true);
        worker->last_clear_event_id = last_clear;
        worker->retired = retired == 1;
        worker->mutation_generation = mutation_generation;
        worker->last_clear_dedupe_event_id =
            version >= 4 ? last_clear_dedupe : last_clear;
        worker->has_clear_dedupe_event_id =
            version >= 5 ? has_clear_dedupe == 1 : worker->last_clear_dedupe_event_id != 0;
        worker->admission_capacity_set = admission_capacity_set == 1;
        worker->admission_capacity = admission_capacity;
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
    if (version >= 4) {
        uint32_t worker_epoch_count = 0;
        if (!reader_u32(&reader, &worker_epoch_count) || worker_epoch_count > 1048576) {
            router_index_free(index);
            return NULL;
        }
        for (uint32_t i = 0; i < worker_epoch_count; ++i) {
            uint64_t worker_id = 0;
            uint64_t generation = 0;
            uint64_t lifecycle_generation = 0;
            uint8_t admission_retired_all = 0;
            uint8_t registration_owner_set = 0;
            uint64_t registration_owner_nonce = 0;
            uint64_t registration_expires_at_ms = 0;
            uint8_t lifecycle_managed = 0;
            uint8_t legacy_tainted = version < 13 ? 1 : 0;
            uint8_t last_registration_expected_set = 0;
            uint64_t last_registration_expected_generation = 0;
            uint8_t lease_cleanup_pending = 0;
            uint64_t lease_cleanup_generation = 0;
            if (!reader_u64(&reader, &worker_id) || !reader_u64(&reader, &generation) ||
                (version >= 13 &&
                 !reader_u64(&reader, &lifecycle_generation)) ||
                (version >= 8 &&
                 !reader_u8(&reader, &admission_retired_all)) ||
                (version >= 11 &&
                 (!reader_u8(&reader, &registration_owner_set) ||
                  !reader_u64(&reader, &registration_owner_nonce) ||
                  !reader_u64(&reader, &registration_expires_at_ms))) ||
                (version >= 13 &&
                 (!reader_u8(&reader, &lifecycle_managed) ||
                  !reader_u8(&reader, &legacy_tainted) ||
                  !reader_u8(&reader, &last_registration_expected_set) ||
                  !reader_u64(
                      &reader, &last_registration_expected_generation))) ||
                (version >= 14 &&
                 (!reader_u8(&reader, &lease_cleanup_pending) ||
                  !reader_u64(&reader, &lease_cleanup_generation))) ||
                admission_retired_all > 1 || registration_owner_set > 1 ||
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
                  lease_cleanup_generation != generation))) {
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
    if (version >= 9) {
        uint32_t admission_rank_count = 0;
        if (!reader_u32(&reader, &admission_rank_count) || admission_rank_count > 1048576) {
            router_index_free(index);
            return NULL;
        }
        for (uint32_t i = 0; i < admission_rank_count; ++i) {
            uint64_t worker_id = 0;
            uint32_t dp_rank = 0;
            uint32_t capacity = 0;
            uint64_t incarnation = 0;
            uint64_t lease_cleanup_generation = 0;
            uint32_t domain_length = 0;
            if (!reader_u64(&reader, &worker_id) || !reader_u32(&reader, &dp_rank) ||
                !reader_u32(&reader, &capacity) || !reader_u64(&reader, &incarnation) ||
                (version >= 14 &&
                 !reader_u64(&reader, &lease_cleanup_generation)) ||
                !reader_u32(&reader, &domain_length) || capacity == 0 || incarnation == 0 ||
                domain_length == 0 || domain_length > DYNKV_MAX_ADMISSION_DOMAIN_LENGTH ||
                reader.length - reader.offset < domain_length ||
                router_index_worker(index, worker_id, dp_rank, false) == NULL ||
                router_index_admission_rank(
                    index,
                    reader.data + reader.offset,
                    domain_length,
                    worker_id,
                    dp_rank,
                    false) != NULL) {
                router_index_free(index);
                return NULL;
            }
            AdmissionRankState *rank = router_index_admission_rank(
                index,
                reader.data + reader.offset,
                domain_length,
                worker_id,
                dp_rank,
                true);
            if (rank == NULL) {
                router_index_free(index);
                return NULL;
            }
            rank->capacity = capacity;
            rank->incarnation = incarnation;
            rank->lease_cleanup_generation = lease_cleanup_generation;
            reader.offset += domain_length;
        }
    }
    uint32_t node_count = 0;
    if (!reader_u32(&reader, &node_count) || node_count > 10485760) {
        router_index_free(index);
        return NULL;
    }
    for (uint32_t i = 0; i < node_count; ++i) {
        uint64_t external_hash = 0;
        uint64_t parent_hash = 0;
        uint64_t local_hash = 0;
        uint32_t owner_count = 0;
        if (!reader_u64(&reader, &external_hash) || !reader_u64(&reader, &parent_hash) ||
            !reader_u64(&reader, &local_hash) || !reader_u32(&reader, &owner_count) ||
            owner_count > 1048576) {
            router_index_free(index);
            return NULL;
        }
        IndexNode *node = router_index_add_node(index, external_hash, parent_hash, local_hash);
        if (node == NULL) {
            router_index_free(index);
            return NULL;
        }
        for (uint32_t owner_idx = 0; owner_idx < owner_count; ++owner_idx) {
            uint64_t worker_id = 0;
            uint32_t dp_rank = 0;
            uint64_t event_id = 0;
            uint8_t active = 0;
            uint64_t lease_cleanup_generation = 0;
            if (!reader_u64(&reader, &worker_id) || !reader_u32(&reader, &dp_rank) ||
                !reader_u64(&reader, &event_id) || !reader_u8(&reader, &active) ||
                (version >= 14 &&
                 !reader_u64(&reader, &lease_cleanup_generation)) ||
                active > 1) {
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
    if (version >= 6) {
        uint32_t reservation_count = 0;
        if (!reader_u32(&reader, &reservation_count) || reservation_count > 1048576) {
            router_index_free(index);
            return NULL;
        }
        for (uint32_t i = 0; i < reservation_count; ++i) {
            uint64_t client_nonce = 0;
            uint64_t request_nonce = 0;
            uint64_t worker_id = 0;
            uint32_t dp_rank = 0;
            uint64_t worker_admission_incarnation = 0;
            uint64_t expires_at_ms = 0;
            uint32_t matched_blocks = 0;
            uint32_t active_reservations_at_grant = 0;
            uint32_t domain_length = 0;
            uint32_t request_length = 0;
            if (!reader_u64(&reader, &client_nonce) || !reader_u64(&reader, &request_nonce) ||
                !reader_u64(&reader, &worker_id) || !reader_u32(&reader, &dp_rank) ||
                (version >= 7 &&
                 !reader_u64(&reader, &worker_admission_incarnation)) ||
                !reader_u64(&reader, &expires_at_ms) ||
                !reader_u32(&reader, &matched_blocks) ||
                !reader_u32(&reader, &active_reservations_at_grant) ||
                !reader_u32(&reader, &domain_length) || domain_length == 0 ||
                domain_length > DYNKV_MAX_ADMISSION_DOMAIN_LENGTH ||
                reader.length - reader.offset < domain_length) {
                router_index_free(index);
                return NULL;
            }
            const uint8_t *domain = reader.data + reader.offset;
            reader.offset += domain_length;
            if (!reader_u32(&reader, &request_length) || request_length == 0 ||
                request_length > DYNKV_MAX_ADMISSION_REQUEST_BYTES ||
                reader.length - reader.offset < request_length) {
                router_index_free(index);
                return NULL;
            }
            const uint8_t *request_bytes = reader.data + reader.offset;
            reader.offset += request_length;
            if (version < 9) {
                WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
                if (worker == NULL) {
                    router_index_free(index);
                    return NULL;
                }
                if (worker_admission_incarnation == 0) {
                    worker_admission_incarnation =
                        worker->admission_incarnation == 0 ? 1 : worker->admission_incarnation;
                }
                if (!router_index_restore_legacy_admission_rank(
                        index,
                        domain,
                        domain_length,
                        worker_id,
                        dp_rank,
                        worker_admission_incarnation)) {
                    router_index_free(index);
                    return NULL;
                }
            }
            if (!router_index_add_reservation(
                    index,
                    domain,
                    domain_length,
                    client_nonce,
                    request_nonce,
                    worker_id,
                    dp_rank,
                    worker_admission_incarnation,
                    expires_at_ms,
                    matched_blocks,
                    active_reservations_at_grant,
                    request_bytes,
                    request_length)) {
                router_index_free(index);
                return NULL;
            }
        }
    }
    if (version >= 12 && !reader_u64(&reader, &index->mutation_count)) {
        router_index_free(index);
        return NULL;
    }
    if (version >= 4) {
        uint64_t stored_generation_counter = 0;
        if (!reader_u64(&reader, &stored_generation_counter)) {
            router_index_free(index);
            return NULL;
        }
        if (stored_generation_counter > index->generation_counter) {
            index->generation_counter = stored_generation_counter;
        }
    }
    if (version >= 13 &&
        !reader_u64(&reader, &index->lifecycle_generation_counter)) {
        router_index_free(index);
        return NULL;
    }
    if (version >= 13 &&
        (!reader_u64(&reader, &index->registration_gc_floor) ||
         index->registration_gc_floor > index->lifecycle_generation_counter)) {
        router_index_free(index);
        return NULL;
    }
    if (reader.offset != reader.length) {
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
