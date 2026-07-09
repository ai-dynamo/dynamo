/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_admission.h"
#include "dynkv_index.h"
#include "dynkv_state.h"

bool admission_identity_read(Reader *reader, AdmissionIdentity *identity) {
    if (!reader_u32(reader, &identity->domain_length) || identity->domain_length == 0 ||
        identity->domain_length > DYNKV_MAX_ADMISSION_DOMAIN_LENGTH ||
        reader->length - reader->offset < identity->domain_length) {
        return false;
    }
    identity->domain = reader->data + reader->offset;
    reader->offset += identity->domain_length;
    return reader_u64(reader, &identity->client_nonce) &&
           reader_u64(reader, &identity->request_nonce);
}

bool admission_identity_append(Buffer *payload, const AdmissionIdentity *identity) {
    return buffer_u32(payload, identity->domain_length) &&
           buffer_bytes(payload, identity->domain, identity->domain_length) &&
           buffer_u64(payload, identity->client_nonce) &&
           buffer_u64(payload, identity->request_nonce);
}

bool admission_identity_parse_payload(
    const uint8_t *data,
    size_t length,
    AdmissionIdentity *identity,
    Reader *reader_out) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    if (!reader_u8(&reader, &version) || version != DYNKV_ADMISSION_WIRE_VERSION ||
        !admission_identity_read(&reader, identity)) {
        return false;
    }
    *reader_out = reader;
    return true;
}

bool admission_apply_prefix_append(Buffer *payload, uint8_t op, uint64_t now_ms) {
    return buffer_u8(payload, DYNKV_ADMISSION_WIRE_VERSION) && buffer_u8(payload, op) &&
           buffer_u64(payload, now_ms);
}

bool admission_reservation_matches_raw(
    const Reservation *reservation,
    const uint8_t *request_bytes,
    uint32_t request_length) {
    return reservation->request_length == request_length &&
           memcmp(reservation->request_bytes, request_bytes, request_length) == 0;
}

bool admission_candidates_valid(
    RouterIndex *index,
    const AdmissionRequest *request,
    uint64_t now_ms) {
    for (uint32_t i = 0; i < request->candidate_count; ++i) {
        const AdmissionCandidate *candidate = &request->candidates[i];
        if (candidate->capacity == 0 ||
            candidate->capacity == DYNKV_UNKNOWN_ADMISSION_CAPACITY) {
            return false;
        }
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
        bool eligible = worker != NULL && !worker->retired && worker->admission_registered &&
                        (epoch == NULL || !epoch->admission_retired_all) &&
                        worker_registration_is_live(index, candidate->worker_id, now_ms);
        /*
         * Frontends discover workers independently. A lagging frontend may
         * still submit a rank whose lease expired or whose discovery record
         * was removed; selection already skips those ranks, so they must not
         * poison otherwise healthy candidates in the same atomic request.
         * A capacity conflict on a live rank remains a caller/configuration
         * error rather than silently changing the module-owned capacity.
         */
        if (eligible && !admission_rank_capacity_matches(rank, candidate->capacity)) {
            return false;
        }
    }
    return true;
}

int router_index_admission_apply_reserve(
    RouterIndex *index,
    uint64_t now_ms,
    const AdmissionIdentity *identity,
    uint64_t worker_id,
    uint32_t dp_rank,
    uint64_t worker_admission_incarnation,
    uint32_t capacity,
    uint64_t expires_at_ms,
    uint32_t matched_blocks,
    uint32_t active_reservations_at_grant,
    const uint8_t *request_bytes,
    uint32_t request_length,
    Reservation **reservation_out,
    bool *changed_out) {
    /* SELECT_RESERVE already committed its bounded expiry pass before it
     * computed active_reservations_at_grant. Repeating cleanup here would
     * change that count between planning and apply. Replicas receive the same
     * cleanup record immediately before this reserve record. */
    *changed_out = false;
    *reservation_out = NULL;
    Reservation *existing = router_index_reservation(
        index,
        identity->domain,
        identity->domain_length,
        identity->client_nonce,
        identity->request_nonce);
    if (existing != NULL) {
        if (!admission_reservation_matches_raw(existing, request_bytes, request_length) ||
            existing->worker_id != worker_id || existing->dp_rank != dp_rank ||
            existing->worker_admission_incarnation != worker_admission_incarnation ||
            existing->expires_at_ms != expires_at_ms || existing->matched_blocks != matched_blocks ||
            existing->active_reservations_at_grant != active_reservations_at_grant) {
            return DYNKV_ADMISSION_CONFLICT;
        }
        if (!worker_registration_is_live(index, existing->worker_id, now_ms)) {
            return DYNKV_ADMISSION_EXPIRED;
        }
        *reservation_out = existing;
        return VALKEYMODULE_OK;
    }
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, false);
    AdmissionRankState *rank = router_index_admission_rank(
        index, identity->domain, identity->domain_length, worker_id, dp_rank, false);
    if (worker == NULL || worker->retired || !worker->admission_registered || capacity == 0 ||
        capacity == DYNKV_UNKNOWN_ADMISSION_CAPACITY || expires_at_ms <= now_ms ||
        (epoch != NULL && epoch->admission_retired_all) ||
        !worker_registration_is_live(index, worker_id, now_ms)) {
        return DYNKV_ADMISSION_INVALID;
    }
    uint32_t active_reservations = rank == NULL ? 0 : rank->active_reservations;
    if (!admission_rank_capacity_matches(rank, capacity) ||
        (rank != NULL && rank->incarnation != 0 &&
         rank->incarnation != worker_admission_incarnation) ||
        active_reservations >= capacity || active_reservations == UINT32_MAX ||
        active_reservations_at_grant != active_reservations + 1) {
        return DYNKV_ADMISSION_INVALID;
    }

    existing = router_index_reservation(
        index,
        identity->domain,
        identity->domain_length,
        identity->client_nonce,
        identity->request_nonce);
    if (existing != NULL) {
        return DYNKV_ADMISSION_CONFLICT;
    }
    rank = router_index_admission_rank(
        index, identity->domain, identity->domain_length, worker_id, dp_rank, false);
    if (rank == NULL) {
        rank = router_index_admission_rank(
            index, identity->domain, identity->domain_length, worker_id, dp_rank, true);
    }
    if (rank == NULL || !admission_rank_configure(rank, capacity) ||
        rank->incarnation != worker_admission_incarnation ||
        rank->active_reservations >= capacity ||
        active_reservations_at_grant != rank->active_reservations + 1 ||
        !router_index_add_reservation(
            index,
            identity->domain,
            identity->domain_length,
            identity->client_nonce,
            identity->request_nonce,
            worker_id,
            dp_rank,
            worker_admission_incarnation,
            expires_at_ms,
            matched_blocks,
            active_reservations_at_grant,
            request_bytes,
            request_length)) {
        return DYNKV_ADMISSION_INVALID;
    }
    *changed_out = true;
    *reservation_out = router_index_reservation(
        index,
        identity->domain,
        identity->domain_length,
        identity->client_nonce,
        identity->request_nonce);
    return *reservation_out == NULL ? DYNKV_ADMISSION_INVALID : VALKEYMODULE_OK;
}

int router_index_admission_apply_release(
    RouterIndex *index,
    uint64_t now_ms,
    const AdmissionIdentity *identity,
    uint64_t expected_expires_at_ms,
    bool *released_out,
    bool *changed_out) {
    *changed_out = router_index_cleanup_expired_reservations(
                       index,
                       now_ms,
                       DYNKV_ADMISSION_EXPIRY_CLEANUP_BUDGET) != 0;
    *released_out = false;
    Reservation *reservation = router_index_reservation(
        index,
        identity->domain,
        identity->domain_length,
        identity->client_nonce,
        identity->request_nonce);
    if (reservation == NULL) {
        return VALKEYMODULE_OK;
    }
    if (reservation->expires_at_ms != expected_expires_at_ms) {
        return DYNKV_ADMISSION_EXPIRED;
    }
    *released_out = router_index_remove_reservation(index, reservation);
    *changed_out |= *released_out;
    return VALKEYMODULE_OK;
}

int router_index_admission_apply_renew(
    RouterIndex *index,
    uint64_t now_ms,
    const AdmissionIdentity *identity,
    uint64_t expected_expires_at_ms,
    uint64_t expires_at_ms,
    Reservation **reservation_out,
    bool *changed_out) {
    *changed_out = router_index_cleanup_expired_reservations(
                       index,
                       now_ms,
                       DYNKV_ADMISSION_EXPIRY_CLEANUP_BUDGET) != 0;
    Reservation *reservation = router_index_reservation(
        index,
        identity->domain,
        identity->domain_length,
        identity->client_nonce,
        identity->request_nonce);
    if (reservation == NULL || expires_at_ms <= now_ms) {
        return DYNKV_ADMISSION_EXPIRED;
    }
    WorkerState *worker = router_index_worker(
        index, reservation->worker_id, reservation->dp_rank, false);
    AdmissionRankState *rank = router_index_admission_rank(
        index,
        reservation->domain,
        reservation->domain_length,
        reservation->worker_id,
        reservation->dp_rank,
        false);
    if (worker == NULL || worker->retired || !worker->admission_registered ||
        rank == NULL || rank->incarnation != reservation->worker_admission_incarnation ||
        !worker_registration_is_live(index, reservation->worker_id, now_ms)) {
        return DYNKV_ADMISSION_EXPIRED;
    }
    if (reservation->expires_at_ms != expected_expires_at_ms) {
        if (expected_expires_at_ms < reservation->expires_at_ms) {
            *reservation_out = reservation;
            return VALKEYMODULE_OK;
        }
        return DYNKV_ADMISSION_EXPIRED;
    }
    if (expires_at_ms < reservation->expires_at_ms) {
        return DYNKV_ADMISSION_EXPIRED;
    }
    if (expires_at_ms == reservation->expires_at_ms) {
        *reservation_out = reservation;
        return VALKEYMODULE_OK;
    }
    uint64_t previous_expires_at_ms = reservation->expires_at_ms;
    reservation->expires_at_ms = expires_at_ms;
    if (!router_index_reservation_expiry_heap_reposition(index, reservation)) {
        reservation->expires_at_ms = previous_expires_at_ms;
        return DYNKV_ADMISSION_INVALID;
    }
    *changed_out = true;
    *reservation_out = reservation;
    return VALKEYMODULE_OK;
}
