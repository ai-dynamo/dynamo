/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_admission.h"
#include "dynkv_index.h"
#include "dynkv_lease.h"
#include "dynkv_persistence.h"
#include "dynkv_state.h"

bool admission_replicate_payload(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString *key,
    const Buffer *payload) {
    return ValkeyModule_Replicate(
               ctx,
               "DYNKV.ADMIT_APPLY",
               "sb",
               key,
               (char *)payload->data,
               payload->length) == VALKEYMODULE_OK;
}

bool admission_commit_payload(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString *key,
    RouterIndex *index,
    const Buffer *payload) {
    ++index->mutation_count;
    ValkeyModule_SignalModifiedKey(ctx, key);
    return admission_replicate_payload(ctx, key, payload);
}

bool admission_cleanup_payload(Buffer *payload, uint64_t now_ms) {
    return admission_apply_prefix_append(payload, DYNKV_ADMISSION_CLEANUP, now_ms);
}

bool admission_reserve_apply_payload(
    Buffer *payload,
    uint64_t now_ms,
    const AdmissionIdentity *identity,
    const AdmissionSelection *selected,
    const AdmissionRequest *request) {
    return admission_apply_prefix_append(payload, DYNKV_ADMISSION_RESERVE, now_ms) &&
           admission_identity_append(payload, identity) &&
           buffer_u64(payload, selected->worker_id) && buffer_u32(payload, selected->dp_rank) &&
           buffer_u64(payload, selected->worker_admission_incarnation) &&
           buffer_u32(payload, selected->capacity) &&
           buffer_u64(payload, selected->expires_at_ms) &&
           buffer_u32(payload, selected->matched_blocks) &&
           buffer_u32(payload, selected->active_reservations_at_grant) &&
           buffer_u32(payload, request->raw_length) &&
           buffer_bytes(payload, request->raw_bytes, request->raw_length);
}

bool admission_release_apply_payload(
    Buffer *payload,
    uint64_t now_ms,
    const AdmissionIdentity *identity,
    uint64_t expected_expires_at_ms) {
    return admission_apply_prefix_append(payload, DYNKV_ADMISSION_RELEASE, now_ms) &&
           admission_identity_append(payload, identity) &&
           buffer_u64(payload, expected_expires_at_ms);
}

bool admission_renew_apply_payload(
    Buffer *payload,
    uint64_t now_ms,
    const AdmissionIdentity *identity,
    uint64_t expected_expires_at_ms,
    uint64_t expires_at_ms) {
    return admission_apply_prefix_append(payload, DYNKV_ADMISSION_RENEW, now_ms) &&
           admission_identity_append(payload, identity) &&
           buffer_u64(payload, expected_expires_at_ms) && buffer_u64(payload, expires_at_ms);
}

bool admission_commit_cleanup(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString *key,
    RouterIndex *index,
    uint64_t now_ms) {
    Buffer payload = {0};
    bool valid = admission_cleanup_payload(&payload, now_ms);
    bool committed = valid && admission_commit_payload(ctx, key, index, &payload);
    buffer_free(&payload);
    return committed;
}

int dynkv_admit_apply_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    if (!ValkeyModule_MustObeyClient(ctx)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_ADMIT_APPLY_INTERNAL_ONLY");
    }
    size_t payload_length = 0;
    const char *payload_data = ValkeyModule_StringPtrLen(argv[2], &payload_length);
    Reader reader = {
        .data = (const uint8_t *)payload_data,
        .length = payload_length,
        .offset = 0,
    };
    uint8_t version = 0;
    uint8_t op = 0;
    uint64_t now_ms = 0;
    if (!reader_u8(&reader, &version) || !reader_u8(&reader, &op) ||
        !reader_u64(&reader, &now_ms) || version != DYNKV_ADMISSION_WIRE_VERSION ||
        op < DYNKV_ADMISSION_RESERVE || op > DYNKV_ADMISSION_CLEANUP) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_ADMIT_APPLY");
    }

    AdmissionIdentity identity = {0};
    uint64_t expected_expires_at_ms = 0;
    uint64_t expires_at_ms = 0;
    uint64_t worker_id = 0;
    uint32_t dp_rank = 0;
    uint64_t worker_admission_incarnation = 0;
    uint32_t capacity = 0;
    uint32_t matched_blocks = 0;
    uint32_t active_reservations_at_grant = 0;
    uint32_t request_length = 0;
    const uint8_t *request_bytes = NULL;
    if (op != DYNKV_ADMISSION_CLEANUP && !admission_identity_read(&reader, &identity)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_ADMIT_APPLY");
    }
    if (op == DYNKV_ADMISSION_RESERVE) {
        if (!reader_u64(&reader, &worker_id) || !reader_u32(&reader, &dp_rank) ||
            !reader_u64(&reader, &worker_admission_incarnation) || !reader_u32(&reader, &capacity) ||
            !reader_u64(&reader, &expires_at_ms) || !reader_u32(&reader, &matched_blocks) ||
            !reader_u32(&reader, &active_reservations_at_grant) ||
            !reader_u32(&reader, &request_length) || request_length == 0 ||
            request_length > DYNKV_MAX_ADMISSION_REQUEST_BYTES ||
            reader.length - reader.offset != request_length) {
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_ADMIT_APPLY");
        }
        request_bytes = reader.data + reader.offset;
        reader.offset += request_length;
    } else if (op == DYNKV_ADMISSION_RELEASE) {
        if (!reader_u64(&reader, &expected_expires_at_ms) || reader.offset != reader.length) {
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_ADMIT_APPLY");
        }
    } else if (op == DYNKV_ADMISSION_RENEW) {
        if (!reader_u64(&reader, &expected_expires_at_ms) ||
            !reader_u64(&reader, &expires_at_ms) || reader.offset != reader.length) {
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_ADMIT_APPLY");
        }
    } else if (reader.offset != reader.length) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_ADMIT_APPLY");
    }

    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    bool changed = false;
    int result = VALKEYMODULE_OK;
    if (op == DYNKV_ADMISSION_CLEANUP) {
        changed = router_index_cleanup_expired_reservations(
                      index,
                      now_ms,
                      DYNKV_ADMISSION_EXPIRY_CLEANUP_BUDGET) != 0;
    } else if (op == DYNKV_ADMISSION_RESERVE) {
        Reservation *reservation = NULL;
        result = router_index_admission_apply_reserve(
            index,
            now_ms,
            &identity,
            worker_id,
            dp_rank,
            worker_admission_incarnation,
            capacity,
            expires_at_ms,
            matched_blocks,
            active_reservations_at_grant,
            request_bytes,
            request_length,
            &reservation,
            &changed);
    } else if (op == DYNKV_ADMISSION_RELEASE) {
        bool released = false;
        result = router_index_admission_apply_release(
            index,
            now_ms,
            &identity,
            expected_expires_at_ms,
            &released,
            &changed);
        if (result == DYNKV_ADMISSION_EXPIRED) {
            result = VALKEYMODULE_OK;
        }
    } else {
        Reservation *reservation = NULL;
        result = router_index_admission_apply_renew(
            index,
            now_ms,
            &identity,
            expected_expires_at_ms,
            expires_at_ms,
            &reservation,
            &changed);
        if (result == DYNKV_ADMISSION_EXPIRED) {
            result = VALKEYMODULE_OK;
        }
    }
    if (result != VALKEYMODULE_OK) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_ADMIT_APPLY");
    }
    if (changed) {
        ++index->mutation_count;
        ValkeyModule_SignalModifiedKey(ctx, argv[1]);
    }
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

int dynkv_select_reserve_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t payload_length = 0;
    const char *payload = ValkeyModule_StringPtrLen(argv[2], &payload_length);
    ValkeyModuleKey *key = NULL;
    bool created = false;
    RouterIndex *index = router_index_for_write_tracking_creation(
        ctx, argv[1], &key, &created);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    uint64_t now_ms = 0;
    if (!admission_now_ms(&now_ms)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RESERVE");
    }
    if (!worker_lease_commit_one_bounded_expiry(
            ctx, argv[1], index, now_ms)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
    }
    AdmissionRequest request = {0};
    MatchScores scores = {0};
    if (admission_request_parse(
            index, (const uint8_t *)payload, payload_length, &request, &scores, now_ms) !=
        VALKEYMODULE_OK) {
        admission_request_free(&request);
        if (created) {
            ValkeyModule_DeleteKey(key);
        }
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RESERVE");
    }
    if (now_ms > UINT64_MAX - request.lease_ms) {
        admission_request_free(&request);
        match_scores_free(&scores);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RESERVE");
    }
    AdmissionIdentity identity = {
        .domain = request.domain,
        .domain_length = request.domain_length,
        .client_nonce = request.client_nonce,
        .request_nonce = request.request_nonce,
    };
    bool cleanup_changed = router_index_cleanup_expired_reservations(
                               index,
                               now_ms,
                               DYNKV_ADMISSION_EXPIRY_CLEANUP_BUDGET) != 0;
    Reservation *existing = router_index_reservation(
        index,
        identity.domain,
        identity.domain_length,
        identity.client_nonce,
        identity.request_nonce);
    if (existing != NULL) {
        if (cleanup_changed && !admission_commit_cleanup(ctx, argv[1], index, now_ms)) {
            admission_request_free(&request);
            match_scores_free(&scores);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
        }
        if (!reservation_matches_request(existing, &request)) {
            admission_request_free(&request);
            match_scores_free(&scores);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_REQUEST_CONFLICT");
        }
        if (!worker_registration_is_live(index, existing->worker_id, now_ms)) {
            uint64_t expected_expires_at_ms = existing->expires_at_ms;
            Buffer release_payload = {0};
            bool released = false;
            bool changed = false;
            bool valid = admission_release_apply_payload(
                             &release_payload,
                             now_ms,
                             &identity,
                             expected_expires_at_ms) &&
                         router_index_admission_apply_release(
                             index,
                             now_ms,
                             &identity,
                             expected_expires_at_ms,
                             &released,
                             &changed) == VALKEYMODULE_OK &&
                         released && changed &&
                         admission_commit_payload(
                             ctx, argv[1], index, &release_payload);
            buffer_free(&release_payload);
            admission_request_free(&request);
            match_scores_free(&scores);
            if (!valid) {
                return ValkeyModule_ReplyWithError(
                    ctx, "DYNKV_REPLICATION_FAILED");
            }
            Buffer response = {0};
            if (!admission_response_append(
                    &response, DYNKV_ADMISSION_NO_CAPACITY, NULL)) {
                buffer_free(&response);
                return ValkeyModule_ReplyWithError(
                    ctx, "DYNKV_INVALID_RESERVE");
            }
            int reply = ValkeyModule_ReplyWithStringBuffer(
                ctx, (const char *)response.data, response.length);
            buffer_free(&response);
            return reply;
        }
        if (!cleanup_changed) {
            AdmissionRankState *rank = router_index_admission_rank(
                index,
                existing->domain,
                existing->domain_length,
                existing->worker_id,
                existing->dp_rank,
                false);
            AdmissionSelection replay = {
                .selected = true,
                .worker_id = existing->worker_id,
                .dp_rank = existing->dp_rank,
                .worker_admission_incarnation = existing->worker_admission_incarnation,
                .expires_at_ms = existing->expires_at_ms,
                .matched_blocks = existing->matched_blocks,
                .active_reservations_at_grant = existing->active_reservations_at_grant,
                .capacity = rank == NULL ? 0 : rank->capacity,
            };
            Buffer replay_payload = {0};
            bool replicated = replay.capacity != 0 && admission_reserve_apply_payload(
                                  &replay_payload, now_ms, &identity, &replay, &request) &&
                              admission_replicate_payload(ctx, argv[1], &replay_payload);
            buffer_free(&replay_payload);
            if (!replicated) {
                admission_request_free(&request);
                match_scores_free(&scores);
                return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
            }
        }
        Buffer response = {0};
        bool valid = admission_response_append(&response, DYNKV_ADMISSION_RESERVED, existing);
        admission_request_free(&request);
        match_scores_free(&scores);
        if (!valid) {
            buffer_free(&response);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RESERVE");
        }
        int reply = ValkeyModule_ReplyWithStringBuffer(
            ctx, (const char *)response.data, response.length);
        buffer_free(&response);
        return reply;
    }

    if (!admission_candidates_valid(index, &request, now_ms)) {
        admission_request_free(&request);
        match_scores_free(&scores);
        if (cleanup_changed && !admission_commit_cleanup(ctx, argv[1], index, now_ms)) {
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
        }
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_CANDIDATE");
    }

    AdmissionSelection selected = router_index_admission_select(index, &request, &scores, now_ms);
    match_scores_free(&scores);
    if (!selected.selected) {
        if (cleanup_changed && !admission_commit_cleanup(ctx, argv[1], index, now_ms)) {
            admission_request_free(&request);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
        }
        Buffer response = {0};
        bool valid = admission_response_append(&response, DYNKV_ADMISSION_NO_CAPACITY, NULL);
        admission_request_free(&request);
        if (!valid) {
            buffer_free(&response);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RESERVE");
        }
        int reply = ValkeyModule_ReplyWithStringBuffer(
            ctx, (const char *)response.data, response.length);
        buffer_free(&response);
        return reply;
    }

    Buffer apply_payload = {0};
    bool valid = admission_reserve_apply_payload(
        &apply_payload, now_ms, &identity, &selected, &request);
    Reservation *reservation = NULL;
    bool changed = false;
    int result = valid ? router_index_admission_apply_reserve(
                            index,
                            now_ms,
                            &identity,
                            selected.worker_id,
                            selected.dp_rank,
                            selected.worker_admission_incarnation,
                            selected.capacity,
                            selected.expires_at_ms,
                            selected.matched_blocks,
                            selected.active_reservations_at_grant,
                            request.raw_bytes,
                            request.raw_length,
                            &reservation,
                            &changed)
                       : DYNKV_ADMISSION_INVALID;
    admission_request_free(&request);
    if (result != VALKEYMODULE_OK || !changed || reservation == NULL) {
        buffer_free(&apply_payload);
        if (cleanup_changed && !admission_commit_cleanup(ctx, argv[1], index, now_ms)) {
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
        }
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RESERVE");
    }
    /* The reserve apply record carries active_reservations_at_grant computed
     * after the bounded expiry pass. Replicas must observe that same cleanup
     * first or they reject the active-count fence and retain stale state. */
    if (cleanup_changed && !admission_commit_cleanup(ctx, argv[1], index, now_ms)) {
        buffer_free(&apply_payload);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
    }
    if (!admission_commit_payload(ctx, argv[1], index, &apply_payload)) {
        buffer_free(&apply_payload);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
    }
    buffer_free(&apply_payload);
    Buffer response = {0};
    if (!admission_response_append(&response, DYNKV_ADMISSION_RESERVED, reservation)) {
        buffer_free(&response);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RESERVE");
    }
    int reply = ValkeyModule_ReplyWithStringBuffer(ctx, (const char *)response.data, response.length);
    buffer_free(&response);
    return reply;
}

int dynkv_release_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t payload_length = 0;
    const char *payload = ValkeyModule_StringPtrLen(argv[2], &payload_length);
    AdmissionIdentity identity = {0};
    Reader reader = {0};
    uint64_t expected_expires_at_ms = 0;
    if (!admission_identity_parse_payload(
            (const uint8_t *)payload, payload_length, &identity, &reader) ||
        !reader_u64(&reader, &expected_expires_at_ms) || reader.offset != reader.length) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RELEASE");
    }
    uint64_t now_ms = 0;
    if (!admission_now_ms(&now_ms)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RELEASE");
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    Buffer apply_payload = {0};
    bool valid = admission_release_apply_payload(
        &apply_payload, now_ms, &identity, expected_expires_at_ms);
    bool released = false;
    bool changed = false;
    int result = valid ? router_index_admission_apply_release(
                            index,
                            now_ms,
                            &identity,
                            expected_expires_at_ms,
                            &released,
                            &changed)
                       : DYNKV_ADMISSION_INVALID;
    if (changed && !admission_commit_payload(ctx, argv[1], index, &apply_payload)) {
        buffer_free(&apply_payload);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
    }
    buffer_free(&apply_payload);
    if (result == DYNKV_ADMISSION_EXPIRED) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_RESERVATION_EXPIRED");
    }
    if (result != VALKEYMODULE_OK) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RELEASE");
    }
    uint8_t response[2] = {DYNKV_ADMISSION_WIRE_VERSION, released ? 1 : 0};
    return ValkeyModule_ReplyWithStringBuffer(ctx, (const char *)response, sizeof(response));
}

int dynkv_renew_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t payload_length = 0;
    const char *payload = ValkeyModule_StringPtrLen(argv[2], &payload_length);
    AdmissionIdentity identity = {0};
    Reader reader = {0};
    uint64_t expected_expires_at_ms = 0;
    uint64_t lease_ms = 0;
    if (!admission_identity_parse_payload(
            (const uint8_t *)payload, payload_length, &identity, &reader) ||
        !reader_u64(&reader, &expected_expires_at_ms) || !reader_u64(&reader, &lease_ms) ||
        reader.offset != reader.length || lease_ms == 0 ||
        lease_ms > DYNKV_MAX_ADMISSION_LEASE_MS) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RENEW");
    }
    uint64_t now_ms = 0;
    if (!admission_now_ms(&now_ms) || now_ms > UINT64_MAX - lease_ms) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RENEW");
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    uint64_t expires_at_ms = now_ms + lease_ms;
    Reservation *existing = router_index_reservation(
        index,
        identity.domain,
        identity.domain_length,
        identity.client_nonce,
        identity.request_nonce);
    if (existing != NULL && existing->expires_at_ms > expires_at_ms) {
        expires_at_ms = existing->expires_at_ms;
    }
    Buffer apply_payload = {0};
    bool valid = admission_renew_apply_payload(
        &apply_payload, now_ms, &identity, expected_expires_at_ms, expires_at_ms);
    Reservation *reservation = NULL;
    bool changed = false;
    int result = valid ? router_index_admission_apply_renew(
                            index,
                            now_ms,
                            &identity,
                            expected_expires_at_ms,
                            expires_at_ms,
                            &reservation,
                            &changed)
                       : DYNKV_ADMISSION_INVALID;
    if (changed && !admission_commit_payload(ctx, argv[1], index, &apply_payload)) {
        buffer_free(&apply_payload);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
    }
    buffer_free(&apply_payload);
    if (result == DYNKV_ADMISSION_EXPIRED) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_RESERVATION_EXPIRED");
    }
    if (result != VALKEYMODULE_OK || reservation == NULL) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RENEW");
    }
    Buffer response = {0};
    if (!admission_response_append(&response, DYNKV_ADMISSION_RESERVED, reservation)) {
        buffer_free(&response);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_RENEW");
    }
    int reply = ValkeyModule_ReplyWithStringBuffer(ctx, (const char *)response.data, response.length);
    buffer_free(&response);
    return reply;
}

/*
 * Return the current per-rank replacement fence as an unsigned big-endian
 * u64.  A recovery client reads it before fetching a worker tree dump and
 * passes it back to DYNKV.REPLACE_RANK_IF_GENERATION.
 */
