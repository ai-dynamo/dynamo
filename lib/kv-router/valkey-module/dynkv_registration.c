/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_index.h"
#include "dynkv_lease.h"
#include "dynkv_persistence.h"
#include "dynkv_state.h"

int compare_u32(const void *left, const void *right) {
    uint32_t left_value = *(const uint32_t *)left;
    uint32_t right_value = *(const uint32_t *)right;
    return (left_value > right_value) - (left_value < right_value);
}

/*
 * Read and canonicalize a complete rank batch before any module key is
 * opened for write. The payload is:
 *
 *   u8 version, u32 rank_count, rank_count * u32 dp_rank
 *
 * Duplicate ranks are rejected rather than silently normalized so every
 * accepted request has one unambiguous rank set.
 */
bool registration_ranks_read(
    const uint8_t *data,
    size_t length,
    uint32_t **ranks_out,
    uint32_t *rank_count_out) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint32_t rank_count = 0;
    if (!reader_u8(&reader, &version) || version != DYNKV_REGISTRATION_WIRE_VERSION ||
        !reader_u32(&reader, &rank_count) || rank_count == 0 ||
        rank_count > DYNKV_MAX_REGISTRATION_RANKS ||
        reader.length - reader.offset != (size_t)rank_count * sizeof(uint32_t)) {
        return false;
    }

    uint32_t *ranks = ValkeyModule_Alloc((size_t)rank_count * sizeof(*ranks));
    for (uint32_t i = 0; i < rank_count; ++i) {
        if (!reader_u32(&reader, &ranks[i])) {
            ValkeyModule_Free(ranks);
            return false;
        }
    }
    qsort(ranks, rank_count, sizeof(*ranks), compare_u32);
    for (uint32_t i = 1; i < rank_count; ++i) {
        if (ranks[i] == ranks[i - 1]) {
            ValkeyModule_Free(ranks);
            return false;
        }
    }
    *ranks_out = ranks;
    *rank_count_out = rank_count;
    return true;
}

bool leased_registration_read(
    const uint8_t *data,
    size_t length,
    uint64_t *owner_nonce_out,
    uint64_t *lease_ms_out,
    uint64_t *expected_generation_out,
    bool *replay_safe_out,
    uint32_t **ranks_out,
    uint32_t *rank_count_out) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint64_t owner_nonce = 0;
    uint64_t lease_ms = 0;
    uint64_t expected_generation = 0;
    uint32_t rank_count = 0;
    if (!reader_u8(&reader, &version) ||
        (version != DYNKV_LEASED_REGISTRATION_WIRE_VERSION_LEGACY &&
         version != DYNKV_LEASED_REGISTRATION_WIRE_VERSION) ||
        !reader_u64(&reader, &owner_nonce) || owner_nonce == 0 ||
        !reader_u64(&reader, &lease_ms) || lease_ms == 0 ||
        lease_ms > DYNKV_MAX_WORKER_LEASE_MS ||
        (version == DYNKV_LEASED_REGISTRATION_WIRE_VERSION &&
         !reader_u64(&reader, &expected_generation)) ||
        !reader_u32(&reader, &rank_count) ||
        rank_count == 0 || rank_count > DYNKV_MAX_REGISTRATION_RANKS ||
        reader.length - reader.offset != (size_t)rank_count * sizeof(uint32_t)) {
        return false;
    }
    uint32_t *ranks = ValkeyModule_Alloc((size_t)rank_count * sizeof(*ranks));
    for (uint32_t i = 0; i < rank_count; ++i) {
        if (!reader_u32(&reader, &ranks[i])) {
            ValkeyModule_Free(ranks);
            return false;
        }
    }
    qsort(ranks, rank_count, sizeof(*ranks), compare_u32);
    for (uint32_t i = 1; i < rank_count; ++i) {
        if (ranks[i] == ranks[i - 1]) {
            ValkeyModule_Free(ranks);
            return false;
        }
    }
    *owner_nonce_out = owner_nonce;
    *lease_ms_out = lease_ms;
    *expected_generation_out = expected_generation;
    *replay_safe_out = version == DYNKV_LEASED_REGISTRATION_WIRE_VERSION;
    *ranks_out = ranks;
    *rank_count_out = rank_count;
    return true;
}

bool worker_lease_register_apply_payload(
    Buffer *payload,
    uint64_t now_ms,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    bool replay_safe,
    uint64_t expected_registration_generation,
    const uint32_t *ranks,
    uint32_t rank_count) {
    if (!worker_lease_apply_prefix_append(payload, DYNKV_WORKER_LEASE_REGISTER, now_ms) ||
        !buffer_u64(payload, worker_id) || !buffer_u64(payload, owner_nonce) ||
        !buffer_u64(payload, expires_at_ms) ||
        !buffer_u8(payload, replay_safe ? 1 : 0) ||
        !buffer_u64(payload, expected_registration_generation) ||
        !buffer_u32(payload, rank_count)) {
        return false;
    }
    for (uint32_t i = 0; i < rank_count; ++i) {
        if (!buffer_u32(payload, ranks[i])) {
            return false;
        }
    }
    return true;
}

int dynkv_register_worker_ranks_leased(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString *key_name,
    uint64_t worker_id,
    const uint8_t *payload_data,
    size_t payload_length) {
    uint64_t owner_nonce = 0;
    uint64_t lease_ms = 0;
    uint64_t expected_generation = 0;
    bool replay_safe = false;
    uint32_t *ranks = NULL;
    uint32_t rank_count = 0;
    if (!leased_registration_read(
            payload_data,
            payload_length,
            &owner_nonce,
            &lease_ms,
            &expected_generation,
            &replay_safe,
            &ranks,
            &rank_count)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_REGISTRATION");
    }
    uint64_t now_ms = 0;
    if (!admission_now_ms(&now_ms) || now_ms > UINT64_MAX - lease_ms) {
        ValkeyModule_Free(ranks);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_REGISTRATION");
    }
    uint64_t expires_at_ms = now_ms + lease_ms;

    /* Preflight the complete rank set before opening an absent key for write. */
    ValkeyModuleKey *read_key = NULL;
    RouterIndex *existing = router_index_for_read(ctx, key_name, &read_key);
    if (existing == (RouterIndex *)(uintptr_t)1) {
        ValkeyModule_Free(ranks);
        return VALKEYMODULE_OK;
    }
    if (existing != NULL) {
        WorkerEpoch *epoch = router_index_worker_epoch(existing, worker_id, false);
        if (epoch != NULL && epoch->registration_owner_set &&
            (epoch->lease_cleanup_pending ||
             epoch->registration_expires_at_ms <= now_ms)) {
            ValkeyModule_Free(ranks);
            return ValkeyModule_ReplyWithError(
                ctx, "DYNKV_WORKER_CLEANUP_PENDING");
        }
        if (replay_safe && epoch == NULL &&
            existing->lifecycle_generation_counter == UINT64_MAX) {
            ValkeyModule_Free(ranks);
            return ValkeyModule_ReplyWithError(
                ctx, "DYNKV_LIFECYCLE_GENERATION_EXHAUSTED");
        }
        uint64_t current_generation =
            epoch == NULL ? existing->lifecycle_generation_counter + 1
                          : epoch->lifecycle_generation;
        if (epoch != NULL && epoch->registration_owner_set &&
            epoch->registration_expires_at_ms <= now_ms) {
            if (existing->lifecycle_generation_counter == UINT64_MAX) {
                ValkeyModule_Free(ranks);
                return ValkeyModule_ReplyWithError(
                    ctx, "DYNKV_LIFECYCLE_GENERATION_EXHAUSTED");
            }
            current_generation = existing->lifecycle_generation_counter + 1;
        }
        bool exact_replay =
            replay_safe && epoch != NULL && epoch->registration_owner_set &&
            epoch->registration_expires_at_ms > now_ms &&
            epoch->registration_owner_nonce == owner_nonce &&
            epoch->last_registration_expected_set &&
            epoch->last_registration_expected_generation == expected_generation &&
            router_index_registered_ranks_equal(
                existing, worker_id, ranks, rank_count);
        bool absent_token_valid =
            replay_safe && epoch == NULL &&
            expected_generation > existing->registration_gc_floor &&
            expected_generation <= current_generation;
        if (replay_safe && expected_generation != current_generation &&
            !absent_token_valid && !exact_replay) {
            ValkeyModule_Free(ranks);
            return ValkeyModule_ReplyWithError(
                ctx, "DYNKV_STALE_REGISTRATION_GENERATION");
        }
        if (epoch != NULL && epoch->admission_retired_all) {
            ValkeyModule_Free(ranks);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_RETIRED");
        }
        if (epoch != NULL && epoch->registration_owner_set &&
            epoch->registration_expires_at_ms > now_ms &&
            epoch->registration_owner_nonce != owner_nonce) {
            ValkeyModule_Free(ranks);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_OWNED");
        }
        if (epoch != NULL && epoch->registration_owner_set &&
            epoch->registration_owner_nonce == owner_nonce &&
            epoch->registration_expires_at_ms > expires_at_ms) {
            expires_at_ms = epoch->registration_expires_at_ms;
        }
        if (epoch != NULL && epoch->registration_owner_set &&
            epoch->registration_expires_at_ms <= now_ms &&
            !router_index_worker_lease_can_end(existing, worker_id)) {
            ValkeyModule_Free(ranks);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_REGISTRATION");
        }
        for (uint32_t i = 0; i < rank_count; ++i) {
            WorkerState *worker = router_index_worker(existing, worker_id, ranks[i], false);
            if (worker != NULL && worker->retired) {
                ValkeyModule_Free(ranks);
                return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_RETIRED");
            }
        }
    } else if (replay_safe && expected_generation != 1) {
        ValkeyModule_Free(ranks);
        return ValkeyModule_ReplyWithError(
            ctx, "DYNKV_STALE_REGISTRATION_GENERATION");
    }

    Buffer apply_payload = {0};
    bool valid = worker_lease_register_apply_payload(
        &apply_payload,
        now_ms,
        worker_id,
        owner_nonce,
        expires_at_ms,
        replay_safe,
        expected_generation,
        ranks,
        rank_count);
    ValkeyModuleKey *write_key = NULL;
    RouterIndex *index = valid ? router_index_for_write(ctx, key_name, &write_key) : NULL;
    bool changed = false;
    int result = index == NULL
                     ? DYNKV_WORKER_LEASE_INVALID
                     : router_index_worker_lease_apply_register(
                           index,
                           now_ms,
                           worker_id,
                           owner_nonce,
                           expires_at_ms,
                           ranks,
                           rank_count,
                           replay_safe,
                           expected_generation,
                           &changed);
    ValkeyModule_Free(ranks);
    if (result == DYNKV_WORKER_LEASE_OWNED) {
        buffer_free(&apply_payload);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_OWNED");
    }
    if (result != VALKEYMODULE_OK ||
        !worker_lease_commit_payload(ctx, key_name, index, &apply_payload, changed)) {
        buffer_free(&apply_payload);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
    }
    buffer_free(&apply_payload);
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

bool worker_lease_control_read(
    const uint8_t *data,
    size_t length,
    bool includes_lease_ms,
    uint64_t *worker_id_out,
    uint64_t *owner_nonce_out,
    uint64_t *lease_ms_out) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint64_t worker_id = 0;
    uint64_t owner_nonce = 0;
    uint64_t lease_ms = 0;
    if (!reader_u8(&reader, &version) || version != DYNKV_WORKER_LEASE_CONTROL_VERSION ||
        !reader_u64(&reader, &worker_id) || !reader_u64(&reader, &owner_nonce) ||
        owner_nonce == 0 ||
        (includes_lease_ms &&
         (!reader_u64(&reader, &lease_ms) || lease_ms == 0 ||
          lease_ms > DYNKV_MAX_WORKER_LEASE_MS)) ||
        reader.offset != reader.length) {
        return false;
    }
    *worker_id_out = worker_id;
    *owner_nonce_out = owner_nonce;
    *lease_ms_out = lease_ms;
    return true;
}

bool worker_lease_renew_apply_payload(
    Buffer *payload,
    uint64_t now_ms,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms) {
    return worker_lease_apply_prefix_append(payload, DYNKV_WORKER_LEASE_RENEW, now_ms) &&
           buffer_u64(payload, worker_id) && buffer_u64(payload, owner_nonce) &&
           buffer_u64(payload, expires_at_ms);
}

bool worker_lease_unregister_apply_payload(
    Buffer *payload,
    uint64_t now_ms,
    uint64_t worker_id,
    uint64_t owner_nonce) {
    return worker_lease_apply_prefix_append(payload, DYNKV_WORKER_LEASE_UNREGISTER, now_ms) &&
           buffer_u64(payload, worker_id) && buffer_u64(payload, owner_nonce);
}

int dynkv_renew_worker_lease_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t payload_length = 0;
    const char *payload = ValkeyModule_StringPtrLen(argv[2], &payload_length);
    uint64_t worker_id = 0;
    uint64_t owner_nonce = 0;
    uint64_t lease_ms = 0;
    if (!worker_lease_control_read(
            (const uint8_t *)payload,
            payload_length,
            true,
            &worker_id,
            &owner_nonce,
            &lease_ms)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE");
    }
    uint64_t now_ms = 0;
    if (!admission_now_ms(&now_ms) || now_ms > UINT64_MAX - lease_ms) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE");
    }
    ValkeyModuleKey *read_key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &read_key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    if (index == NULL) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_STALE_WORKER_OWNER");
    }
    ValkeyModuleKey *write_key = NULL;
    index = router_index_for_write(ctx, argv[1], &write_key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    uint64_t expires_at_ms = now_ms + lease_ms;
    WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, false);
    if (epoch != NULL && epoch->registration_owner_set &&
        epoch->registration_owner_nonce == owner_nonce &&
        epoch->registration_expires_at_ms > expires_at_ms) {
        expires_at_ms = epoch->registration_expires_at_ms;
    }
    Buffer apply_payload = {0};
    bool valid = worker_lease_renew_apply_payload(
        &apply_payload, now_ms, worker_id, owner_nonce, expires_at_ms);
    bool changed = false;
    int result = valid ? router_index_worker_lease_apply_renew(
                             index,
                             now_ms,
                             worker_id,
                             owner_nonce,
                             expires_at_ms,
                             &changed)
                       : DYNKV_WORKER_LEASE_INVALID;
    bool committed = false;
    if (result == VALKEYMODULE_OK || changed) {
        committed = worker_lease_commit_payload(
            ctx, argv[1], index, &apply_payload, changed);
    }
    buffer_free(&apply_payload);
    if ((result == VALKEYMODULE_OK || changed) && !committed) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
    }
    if (result == DYNKV_WORKER_LEASE_STALE) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_STALE_WORKER_OWNER");
    }
    if (result != VALKEYMODULE_OK) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE");
    }
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

int dynkv_unregister_worker_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t payload_length = 0;
    const char *payload = ValkeyModule_StringPtrLen(argv[2], &payload_length);
    uint64_t worker_id = 0;
    uint64_t owner_nonce = 0;
    uint64_t ignored_lease_ms = 0;
    if (!worker_lease_control_read(
            (const uint8_t *)payload,
            payload_length,
            false,
            &worker_id,
            &owner_nonce,
            &ignored_lease_ms)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE");
    }
    uint64_t now_ms = 0;
    if (!admission_now_ms(&now_ms)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE");
    }
    ValkeyModuleKey *read_key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &read_key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    if (index == NULL) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_STALE_WORKER_OWNER");
    }
    ValkeyModuleKey *write_key = NULL;
    index = router_index_for_write(ctx, argv[1], &write_key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    Buffer apply_payload = {0};
    bool valid = worker_lease_unregister_apply_payload(
        &apply_payload, now_ms, worker_id, owner_nonce);
    bool changed = false;
    int result = valid ? router_index_worker_lease_apply_unregister(
                             index, worker_id, owner_nonce, &changed)
                       : DYNKV_WORKER_LEASE_INVALID;
    bool committed = result == VALKEYMODULE_OK &&
                     worker_lease_commit_payload(
                         ctx, argv[1], index, &apply_payload, changed);
    buffer_free(&apply_payload);
    if (result == DYNKV_WORKER_LEASE_STALE) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_STALE_WORKER_OWNER");
    }
    if (result != VALKEYMODULE_OK) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE");
    }
    if (!committed) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
    }
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

/*
 * Atomically register all listed DP ranks for one worker. Structural input,
 * the worker-wide retirement fence, and every rank retirement fence are
 * checked before creating a WorkerState or changing admission eligibility.
 */
int dynkv_register_worker_ranks_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 4) {
        return ValkeyModule_WrongArity(ctx);
    }

    size_t worker_length = 0;
    size_t payload_length = 0;
    const char *worker_data = ValkeyModule_StringPtrLen(argv[2], &worker_length);
    const char *payload = ValkeyModule_StringPtrLen(argv[3], &payload_length);
    Reader worker_reader = {.data = (const uint8_t *)worker_data, .length = worker_length};
    uint64_t worker_id = 0;
    if (!reader_u64(&worker_reader, &worker_id) || worker_reader.offset != worker_reader.length) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_REGISTRATION");
    }
    if (payload_length != 0 &&
        ((uint8_t)payload[0] == DYNKV_LEASED_REGISTRATION_WIRE_VERSION_LEGACY ||
         (uint8_t)payload[0] == DYNKV_LEASED_REGISTRATION_WIRE_VERSION)) {
        return dynkv_register_worker_ranks_leased(
            ctx, argv[1], worker_id, (const uint8_t *)payload, payload_length);
    }
    uint32_t *ranks = NULL;
    uint32_t rank_count = 0;
    if (!registration_ranks_read(
            (const uint8_t *)payload,
            payload_length,
            &ranks,
            &rank_count)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_REGISTRATION");
    }

    /*
     * Preflight against a read-only view. This also avoids materializing an
     * empty module key for malformed, fenced, or fully idempotent requests.
     */
    ValkeyModuleKey *read_key = NULL;
    RouterIndex *existing = router_index_for_read(ctx, argv[1], &read_key);
    if (existing == (RouterIndex *)(uintptr_t)1) {
        ValkeyModule_Free(ranks);
        return VALKEYMODULE_OK;
    }
    bool membership_changed = existing == NULL;
    bool provenance_changed = false;
    if (existing != NULL) {
        WorkerEpoch *epoch = router_index_worker_epoch(existing, worker_id, false);
        if (epoch != NULL && epoch->lease_cleanup_pending) {
            ValkeyModule_Free(ranks);
            return ValkeyModule_ReplyWithError(
                ctx, "DYNKV_WORKER_CLEANUP_PENDING");
        }
        if (epoch != NULL && epoch->admission_retired_all) {
            ValkeyModule_Free(ranks);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_RETIRED");
        }
        for (uint32_t i = 0; i < rank_count; ++i) {
            WorkerState *worker = router_index_worker(existing, worker_id, ranks[i], false);
            if (worker != NULL && worker->retired) {
                ValkeyModule_Free(ranks);
                return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_RETIRED");
            }
            membership_changed |= worker == NULL || !worker->admission_registered;
            provenance_changed |= worker != NULL && !worker->legacy_tainted;
        }
        provenance_changed |= epoch != NULL && !epoch->legacy_tainted;
        if (epoch != NULL && epoch->registration_owner_set && membership_changed) {
            ValkeyModule_Free(ranks);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_OWNED");
        }
    }
    if (!membership_changed && !provenance_changed) {
        ValkeyModule_Free(ranks);
        return ValkeyModule_ReplyWithSimpleString(ctx, "NOOP");
    }

    ValkeyModuleKey *write_key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &write_key);
    if (index == NULL) {
        ValkeyModule_Free(ranks);
        return VALKEYMODULE_OK;
    }
    if (router_index_worker_cleanup_pending(index, worker_id)) {
        ValkeyModule_Free(ranks);
        return ValkeyModule_ReplyWithError(
            ctx, "DYNKV_WORKER_CLEANUP_PENDING");
    }
    for (uint32_t i = 0; i < rank_count; ++i) {
        WorkerState *worker = router_index_worker(index, worker_id, ranks[i], true);
        worker->admission_registered = true;
        worker->lifecycle_tombstone = false;
        worker->lifecycle_tombstone_generation = 0;
        router_index_mark_legacy_rank(index, worker_id, ranks[i]);
    }
    ValkeyModule_Free(ranks);

    /* One logical batch is one persisted mutation and one replication record. */
    ++index->mutation_count;
    ValkeyModule_SignalModifiedKey(ctx, argv[1]);
    ValkeyModule_ReplicateVerbatim(ctx);
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}
