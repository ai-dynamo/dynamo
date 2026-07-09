/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_index.h"
#include "dynkv_state.h"

void worker_epoch_add_worker_state(WorkerEpoch *epoch, WorkerState *worker) {
    if (epoch->worker_state_count == epoch->worker_state_capacity) {
        size_t capacity = epoch->worker_state_capacity == 0
                              ? 4
                              : epoch->worker_state_capacity * 2;
        epoch->worker_states = ValkeyModule_Realloc(
            epoch->worker_states, capacity * sizeof(*epoch->worker_states));
        epoch->worker_state_capacity = capacity;
    }
    worker->worker_epoch_index = epoch->worker_state_count;
    epoch->worker_states[epoch->worker_state_count++] = worker;
}

void worker_epoch_remove_worker_state(WorkerEpoch *epoch, WorkerState *worker) {
    size_t position = worker->worker_epoch_index;
    if (position >= epoch->worker_state_count ||
        epoch->worker_states[position] != worker) {
        return;
    }
    WorkerState *moved = epoch->worker_states[--epoch->worker_state_count];
    epoch->worker_states[position] = moved;
    moved->worker_epoch_index = position;
    worker->worker_epoch_index = SIZE_MAX;
}

void worker_epoch_swap_worker_positions(
    WorkerEpoch *epoch,
    size_t left,
    size_t right) {
    if (left == right) {
        return;
    }
    WorkerState *left_worker = epoch->worker_states[left];
    WorkerState *right_worker = epoch->worker_states[right];
    epoch->worker_states[left] = right_worker;
    epoch->worker_states[right] = left_worker;
    right_worker->worker_epoch_index = left;
    left_worker->worker_epoch_index = right;
}

void worker_epoch_add_admission_rank(
    WorkerEpoch *epoch,
    AdmissionRankState *rank) {
    if (epoch->admission_rank_count == epoch->admission_rank_capacity) {
        size_t capacity = epoch->admission_rank_capacity == 0
                              ? 4
                              : epoch->admission_rank_capacity * 2;
        epoch->admission_rank_states = ValkeyModule_Realloc(
            epoch->admission_rank_states,
            capacity * sizeof(*epoch->admission_rank_states));
        epoch->admission_rank_capacity = capacity;
    }
    rank->worker_epoch_index = epoch->admission_rank_count;
    epoch->admission_rank_states[epoch->admission_rank_count++] = rank;
}

void worker_epoch_remove_admission_rank(
    WorkerEpoch *epoch,
    AdmissionRankState *rank) {
    size_t position = rank->worker_epoch_index;
    if (position >= epoch->admission_rank_count ||
        epoch->admission_rank_states[position] != rank) {
        return;
    }
    AdmissionRankState *moved =
        epoch->admission_rank_states[--epoch->admission_rank_count];
    epoch->admission_rank_states[position] = moved;
    moved->worker_epoch_index = position;
    rank->worker_epoch_index = SIZE_MAX;
}

void worker_epoch_swap_admission_positions(
    WorkerEpoch *epoch,
    size_t left,
    size_t right) {
    if (left == right) {
        return;
    }
    AdmissionRankState *left_rank = epoch->admission_rank_states[left];
    AdmissionRankState *right_rank = epoch->admission_rank_states[right];
    epoch->admission_rank_states[left] = right_rank;
    epoch->admission_rank_states[right] = left_rank;
    right_rank->worker_epoch_index = left;
    left_rank->worker_epoch_index = right;
}

void worker_epoch_add_reservation(
    WorkerEpoch *epoch,
    Reservation *reservation) {
    if (epoch->reservation_count == epoch->reservation_capacity) {
        size_t capacity = epoch->reservation_capacity == 0
                              ? 4
                              : epoch->reservation_capacity * 2;
        epoch->reservation_states = ValkeyModule_Realloc(
            epoch->reservation_states,
            capacity * sizeof(*epoch->reservation_states));
        epoch->reservation_capacity = capacity;
    }
    reservation->worker_epoch_index = epoch->reservation_count;
    epoch->reservation_states[epoch->reservation_count++] = reservation;
}

void worker_epoch_remove_reservation(
    WorkerEpoch *epoch,
    Reservation *reservation) {
    size_t position = reservation->worker_epoch_index;
    if (position >= epoch->reservation_count ||
        epoch->reservation_states[position] != reservation) {
        return;
    }
    Reservation *moved = epoch->reservation_states[--epoch->reservation_count];
    epoch->reservation_states[position] = moved;
    moved->worker_epoch_index = position;
    reservation->worker_epoch_index = SIZE_MAX;
}

WorkerState *router_index_worker(
    RouterIndex *index,
    uint64_t worker_id,
    uint32_t dp_rank,
    bool create) {
    uint8_t key[12];
    worker_key(key, worker_id, dp_rank);
    WorkerState *worker = ValkeyModule_DictGetC(index->workers, key, sizeof(key), NULL);
    if (worker != NULL || !create) {
        return worker;
    }

    worker = ValkeyModule_Calloc(1, sizeof(*worker));
    worker->worker_id = worker_id;
    worker->dp_rank = dp_rank;
    worker->worker_epoch_index = SIZE_MAX;
    worker->node_members = ValkeyModule_CreateDict(NULL);
    ValkeyModule_DictSetC(index->workers, key, sizeof(key), worker);
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    if (epoch != NULL) {
        worker->epoch = epoch;
        worker_epoch_add_worker_state(epoch, worker);
    }
    return worker;
}

WorkerEpoch *router_index_worker_epoch(
    RouterIndex *index,
    uint64_t worker_id,
    bool create) {
    uint8_t key[8];
    encode_u64_be(key, worker_id);
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    if (epoch != NULL || !create) {
        return epoch;
    }

    epoch = ValkeyModule_Calloc(1, sizeof(*epoch));
    epoch->worker_id = worker_id;
    epoch->registration_expiry_heap_index = SIZE_MAX;
    /*
     * Epochs can be materialized after legacy rank state. Derive all existing
     * references once here; subsequent creates/deletes maintain the counters.
     */
    ValkeyModuleDictIter *workers =
        ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
        WorkerState *worker = data;
        if (worker->worker_id == worker_id) {
            worker->epoch = epoch;
            worker_epoch_add_worker_state(epoch, worker);
            epoch->node_membership_count += worker->node_count;
        }
    }
    ValkeyModule_DictIteratorStop(workers);
    ValkeyModuleDictIter *admission_ranks =
        ValkeyModule_DictIteratorStartC(index->admission_ranks, "^", NULL, 0);
    while (ValkeyModule_DictNextC(admission_ranks, NULL, &data) != NULL) {
        AdmissionRankState *rank = data;
        if (rank->worker_id == worker_id) {
            worker_epoch_add_admission_rank(epoch, rank);
        }
    }
    ValkeyModule_DictIteratorStop(admission_ranks);
    ValkeyModuleDictIter *reservations =
        ValkeyModule_DictIteratorStartC(index->reservations, "^", NULL, 0);
    while (ValkeyModule_DictNextC(reservations, NULL, &data) != NULL) {
        Reservation *reservation = data;
        if (reservation->worker_id == worker_id) {
            worker_epoch_add_reservation(epoch, reservation);
        }
    }
    ValkeyModule_DictIteratorStop(reservations);
    ValkeyModule_DictSetC(index->worker_epochs, key, sizeof(key), epoch);
    return epoch;
}

AdmissionRankState *router_index_admission_rank(
    RouterIndex *index,
    const uint8_t *domain,
    uint32_t domain_length,
    uint64_t worker_id,
    uint32_t dp_rank,
    bool create) {
    uint8_t key[4 + DYNKV_MAX_ADMISSION_DOMAIN_LENGTH + 12];
    size_t key_length = 0;
    if (!admission_rank_key(
            key,
            sizeof(key),
            domain,
            domain_length,
            worker_id,
            dp_rank,
            &key_length)) {
        return NULL;
    }
    AdmissionRankState *rank =
        ValkeyModule_DictGetC(index->admission_ranks, key, key_length, NULL);
    if (rank != NULL || !create) {
        return rank;
    }
    rank = ValkeyModule_Calloc(1, sizeof(*rank));
    rank->domain = ValkeyModule_Alloc(domain_length);
    memcpy(rank->domain, domain, domain_length);
    rank->domain_length = domain_length;
    rank->worker_id = worker_id;
    rank->dp_rank = dp_rank;
    rank->worker_epoch_index = SIZE_MAX;
    ValkeyModule_DictSetC(index->admission_ranks, key, key_length, rank);
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    if (worker != NULL) {
        ++worker->admission_rank_count;
    }
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    if (epoch != NULL) {
        worker_epoch_add_admission_rank(epoch, rank);
    }
    return rank;
}

bool admission_rank_capacity_matches(const AdmissionRankState *rank, uint32_t capacity) {
    return rank == NULL || rank->capacity == 0 ||
           (rank->capacity != DYNKV_UNKNOWN_ADMISSION_CAPACITY && rank->capacity == capacity);
}

bool admission_rank_configure(AdmissionRankState *rank, uint32_t capacity) {
    if (rank == NULL || capacity == 0 || capacity == DYNKV_UNKNOWN_ADMISSION_CAPACITY ||
        !admission_rank_capacity_matches(rank, capacity)) {
        return false;
    }
    rank->capacity = capacity;
    if (rank->incarnation == 0) {
        rank->incarnation = 1;
    }
    return true;
}

bool admission_rank_advance_incarnation(AdmissionRankState *rank) {
    if (rank == NULL || rank->incarnation == UINT64_MAX) {
        return false;
    }
    ++rank->incarnation;
    if (rank->incarnation == 0) {
        rank->incarnation = 1;
    }
    return true;
}

bool router_index_admission_ranks_can_advance(
    RouterIndex *index,
    uint64_t worker_id,
    bool all_ranks,
    uint32_t dp_rank) {
    ValkeyModuleDictIter *iter =
        ValkeyModule_DictIteratorStartC(index->admission_ranks, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(iter, NULL, &data) != NULL) {
        AdmissionRankState *rank = data;
        if (rank->worker_id == worker_id && (all_ranks || rank->dp_rank == dp_rank) &&
            rank->incarnation == UINT64_MAX) {
            ValkeyModule_DictIteratorStop(iter);
            return false;
        }
    }
    ValkeyModule_DictIteratorStop(iter);
    return true;
}

void router_index_advance_admission_rank_incarnations(
    RouterIndex *index,
    uint64_t worker_id,
    bool all_ranks,
    uint32_t dp_rank) {
    ValkeyModuleDictIter *iter =
        ValkeyModule_DictIteratorStartC(index->admission_ranks, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(iter, NULL, &data) != NULL) {
        AdmissionRankState *rank = data;
        if (rank->worker_id == worker_id && (all_ranks || rank->dp_rank == dp_rank)) {
            (void)admission_rank_advance_incarnation(rank);
        }
    }
    ValkeyModule_DictIteratorStop(iter);
}

/*
 * Versions 6-8 stored admission configuration on WorkerState.  On load,
 * split that legacy configuration into the domain-specific authority used by
 * version 9.  A missing legacy capacity is deliberately restored as an
 * unmatchable capacity so a new client cannot over-admit before the old lease
 * expires or is released.
 */
bool router_index_restore_legacy_admission_rank(
    RouterIndex *index,
    const uint8_t *domain,
    uint32_t domain_length,
    uint64_t worker_id,
    uint32_t dp_rank,
    uint64_t reservation_incarnation) {
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    if (worker == NULL) {
        return false;
    }
    uint32_t capacity = worker->admission_capacity_set ? worker->admission_capacity
                                                        : DYNKV_UNKNOWN_ADMISSION_CAPACITY;
    uint64_t incarnation = reservation_incarnation;
    if (incarnation == 0) {
        incarnation = worker->admission_incarnation == 0 ? 1 : worker->admission_incarnation;
    }
    AdmissionRankState *rank = router_index_admission_rank(
        index, domain, domain_length, worker_id, dp_rank, true);
    if (rank == NULL) {
        return false;
    }
    if (rank->capacity == 0 && rank->incarnation == 0) {
        rank->capacity = capacity;
        rank->incarnation = incarnation;
        return true;
    }
    return rank->capacity == capacity && rank->incarnation == incarnation;
}

Reservation *router_index_reservation(
    RouterIndex *index,
    const uint8_t *domain,
    uint32_t domain_length,
    uint64_t client_nonce,
    uint64_t request_nonce) {
    uint8_t key[4 + DYNKV_MAX_ADMISSION_DOMAIN_LENGTH + 16];
    size_t key_length = 0;
    if (!reservation_key(
            key,
            sizeof(key),
            domain,
            domain_length,
            client_nonce,
            request_nonce,
            &key_length)) {
        return NULL;
    }
    return ValkeyModule_DictGetC(index->reservations, key, key_length, NULL);
}

bool router_index_add_reservation(
    RouterIndex *index,
    const uint8_t *domain,
    uint32_t domain_length,
    uint64_t client_nonce,
    uint64_t request_nonce,
    uint64_t worker_id,
    uint32_t dp_rank,
    uint64_t worker_admission_incarnation,
    uint64_t expires_at_ms,
    uint32_t matched_blocks,
    uint32_t active_reservations_at_grant,
    const uint8_t *request_bytes,
    uint32_t request_length) {
    uint8_t key[4 + DYNKV_MAX_ADMISSION_DOMAIN_LENGTH + 16];
    size_t key_length = 0;
    if (!reservation_key(
            key,
            sizeof(key),
            domain,
            domain_length,
            client_nonce,
            request_nonce,
            &key_length) ||
        router_index_reservation(index, domain, domain_length, client_nonce, request_nonce) != NULL) {
        return false;
    }
    AdmissionRankState *rank = router_index_admission_rank(
        index, domain, domain_length, worker_id, dp_rank, false);
    if (rank == NULL || rank->incarnation != worker_admission_incarnation ||
        rank->active_reservations == UINT32_MAX) {
        return false;
    }

    Reservation *reservation = ValkeyModule_Calloc(1, sizeof(*reservation));
    reservation->domain = ValkeyModule_Alloc(domain_length);
    reservation->request_bytes = ValkeyModule_Alloc(request_length);
    memcpy(reservation->domain, domain, domain_length);
    memcpy(reservation->request_bytes, request_bytes, request_length);
    reservation->client_nonce = client_nonce;
    reservation->request_nonce = request_nonce;
    reservation->worker_id = worker_id;
    reservation->dp_rank = dp_rank;
    reservation->worker_admission_incarnation = worker_admission_incarnation;
    reservation->expires_at_ms = expires_at_ms;
    reservation->matched_blocks = matched_blocks;
    reservation->active_reservations_at_grant = active_reservations_at_grant;
    reservation->domain_length = domain_length;
    reservation->request_length = request_length;
    reservation->expiry_heap_index = SIZE_MAX;
    reservation->worker_epoch_index = SIZE_MAX;
    if (!router_index_reservation_expiry_heap_insert(index, reservation)) {
        reservation_free(reservation);
        return false;
    }
    if (ValkeyModule_DictSetC(index->reservations, key, key_length, reservation) !=
        VALKEYMODULE_OK) {
        (void)router_index_reservation_expiry_heap_remove(index, reservation);
        reservation_free(reservation);
        return false;
    }
    ++rank->active_reservations;
    WorkerState *worker =
        router_index_worker(index, worker_id, dp_rank, false);
    if (worker != NULL) {
        ++worker->reservation_count;
    }
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, worker_id);
    if (epoch != NULL) {
        worker_epoch_add_reservation(epoch, reservation);
    }
    return true;
}

void router_index_delete_admission_rank(RouterIndex *index, AdmissionRankState *rank) {
    uint8_t key[4 + DYNKV_MAX_ADMISSION_DOMAIN_LENGTH + 12];
    size_t key_length = 0;
    if (rank == NULL ||
        !admission_rank_key(
            key,
            sizeof(key),
            rank->domain,
            rank->domain_length,
            rank->worker_id,
            rank->dp_rank,
            &key_length) ||
        ValkeyModule_DictDelC(index->admission_ranks, key, key_length, NULL) != VALKEYMODULE_OK) {
        return;
    }
    WorkerState *worker =
        router_index_worker(index, rank->worker_id, rank->dp_rank, false);
    if (worker != NULL && worker->admission_rank_count != 0) {
        --worker->admission_rank_count;
    }
    WorkerEpoch *epoch = router_index_worker_epoch_lookup(index, rank->worker_id);
    if (epoch != NULL) {
        worker_epoch_remove_admission_rank(epoch, rank);
    }
    admission_rank_state_free(rank);
}

bool router_index_remove_reservation(RouterIndex *index, Reservation *reservation) {
    uint8_t key[4 + DYNKV_MAX_ADMISSION_DOMAIN_LENGTH + 16];
    size_t key_length = 0;
    if (!reservation_key(
            key,
            sizeof(key),
            reservation->domain,
            reservation->domain_length,
            reservation->client_nonce,
            reservation->request_nonce,
            &key_length) ||
        !router_index_reservation_expiry_heap_remove(index, reservation)) {
        return false;
    }
    if (ValkeyModule_DictDelC(index->reservations, key, key_length, NULL) != VALKEYMODULE_OK) {
        (void)router_index_reservation_expiry_heap_insert(index, reservation);
        return false;
    }
    AdmissionRankState *rank = router_index_admission_rank(
        index,
        reservation->domain,
        reservation->domain_length,
        reservation->worker_id,
        reservation->dp_rank,
        false);
    if (rank != NULL && rank->active_reservations > 0) {
        --rank->active_reservations;
        if (rank->active_reservations == 0 &&
            rank->capacity == DYNKV_UNKNOWN_ADMISSION_CAPACITY &&
            !router_index_worker_cleanup_pending(index, reservation->worker_id)) {
            router_index_delete_admission_rank(index, rank);
        }
    }
    WorkerState *worker = router_index_worker(
        index, reservation->worker_id, reservation->dp_rank, false);
    if (worker != NULL && worker->reservation_count != 0) {
        --worker->reservation_count;
    }
    WorkerEpoch *epoch =
        router_index_worker_epoch_lookup(index, reservation->worker_id);
    if (epoch != NULL) {
        worker_epoch_remove_reservation(epoch, reservation);
    }
    reservation_free(reservation);
    return true;
}

size_t router_index_cleanup_expired_reservations(
    RouterIndex *index,
    uint64_t now_ms,
    size_t budget) {
    size_t count = 0;
    while (count < budget && index->reservation_expiry_heap_length != 0) {
        Reservation *reservation = index->reservation_expiry_heap[0];
        if (reservation->expires_at_ms > now_ms ||
            !router_index_remove_reservation(index, reservation)) {
            break;
        }
        ++count;
    }
    return count;
}

size_t router_index_revoke_worker_reservations(
    RouterIndex *index,
    uint64_t worker_id,
    bool all_ranks,
    uint32_t dp_rank) {
    Reservation **revoked = NULL;
    size_t count = 0;
    size_t capacity = 0;
    ValkeyModuleDictIter *iter =
        ValkeyModule_DictIteratorStartC(index->reservations, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(iter, NULL, &data) != NULL) {
        Reservation *reservation = data;
        if (reservation->worker_id != worker_id ||
            (!all_ranks && reservation->dp_rank != dp_rank)) {
            continue;
        }
        if (count == capacity) {
            capacity = capacity == 0 ? 8 : capacity * 2;
            revoked = ValkeyModule_Realloc(revoked, capacity * sizeof(*revoked));
        }
        revoked[count++] = reservation;
    }
    ValkeyModule_DictIteratorStop(iter);
    for (size_t i = 0; i < count; ++i) {
        router_index_remove_reservation(index, revoked[i]);
    }
    ValkeyModule_Free(revoked);
    return count;
}
