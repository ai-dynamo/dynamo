/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_index.h"
#include "dynkv_lease.h"
#include "dynkv_persistence.h"
#include "dynkv_state.h"

int dynkv_rank_generation_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 4) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t worker_length = 0;
    size_t dp_length = 0;
    const char *worker_data = ValkeyModule_StringPtrLen(argv[2], &worker_length);
    const char *dp_data = ValkeyModule_StringPtrLen(argv[3], &dp_length);
    Reader worker_reader = {.data = (const uint8_t *)worker_data, .length = worker_length};
    Reader dp_reader = {.data = (const uint8_t *)dp_data, .length = dp_length};
    uint64_t worker_id = 0;
    uint32_t dp_rank = 0;
    if (!reader_u64(&worker_reader, &worker_id) || worker_reader.offset != worker_reader.length ||
        !reader_u32(&dp_reader, &dp_rank) || dp_reader.offset != dp_reader.length) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER");
    }

    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    uint8_t response[8];
    encode_u64_be(
        response,
        index == NULL ? 0 : router_index_rank_generation(index, worker_id, dp_rank));
    return ValkeyModule_ReplyWithStringBuffer(ctx, (const char *)response, sizeof(response));
}

/* O(1) CAS token for replay-safe worker registration wire version 3. */
int dynkv_registration_generation_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t worker_length = 0;
    const char *worker_data =
        ValkeyModule_StringPtrLen(argv[2], &worker_length);
    Reader reader = {
        .data = (const uint8_t *)worker_data, .length = worker_length};
    uint64_t worker_id = 0;
    if (!reader_u64(&reader, &worker_id) || reader.offset != reader.length) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER");
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    uint64_t generation = 1;
    if (index != NULL) {
        WorkerEpoch *epoch =
            router_index_worker_epoch_lookup(index, worker_id);
        if (epoch == NULL) {
            if (index->lifecycle_generation_counter == UINT64_MAX) {
                return ValkeyModule_ReplyWithError(
                    ctx, "DYNKV_LIFECYCLE_GENERATION_EXHAUSTED");
            }
            generation = index->lifecycle_generation_counter + 1;
        } else {
            generation = epoch->lifecycle_generation;
        }
        uint64_t now_ms = 0;
        if (epoch != NULL && epoch->registration_owner_set &&
            !epoch->lease_cleanup_pending &&
            admission_now_ms(&now_ms) &&
            epoch->registration_expires_at_ms <= now_ms &&
            index->lifecycle_generation_counter != UINT64_MAX) {
            generation = index->lifecycle_generation_counter + 1;
        }
    }
    uint8_t response[8];
    encode_u64_be(response, generation);
    return ValkeyModule_ReplyWithStringBuffer(
        ctx, (const char *)response, sizeof(response));
}

/*
 * Atomically replace one worker/rank from a full tree dump only when its
 * generation is unchanged.  The dump wire format is:
 *
 *   u8 version, u32 event_count,
 *   event_count * (u32 event_length, event_length bytes of DYNKV.APPLY wire event)
 *
 * Every embedded event must be a STORE for this exact worker/rank.  Compressed
 * Dynamo tree dumps have this form.  STORE-only input lets the module validate
 * the complete post-reset prefix topology before it clears any live owner.
 */
int dynkv_replace_rank_if_generation_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 6) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t worker_length = 0;
    size_t dp_length = 0;
    size_t generation_length = 0;
    size_t snapshot_length = 0;
    const char *worker_data = ValkeyModule_StringPtrLen(argv[2], &worker_length);
    const char *dp_data = ValkeyModule_StringPtrLen(argv[3], &dp_length);
    const char *generation_data = ValkeyModule_StringPtrLen(argv[4], &generation_length);
    const char *snapshot = ValkeyModule_StringPtrLen(argv[5], &snapshot_length);
    Reader worker_reader = {.data = (const uint8_t *)worker_data, .length = worker_length};
    Reader dp_reader = {.data = (const uint8_t *)dp_data, .length = dp_length};
    Reader generation_reader = {
        .data = (const uint8_t *)generation_data,
        .length = generation_length,
    };
    uint64_t worker_id = 0;
    uint32_t dp_rank = 0;
    uint64_t expected_generation = 0;
    if (!reader_u64(&worker_reader, &worker_id) || worker_reader.offset != worker_reader.length ||
        !reader_u32(&dp_reader, &dp_rank) || dp_reader.offset != dp_reader.length ||
        !reader_u64(&generation_reader, &expected_generation) ||
        generation_reader.offset != generation_reader.length) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_REPLACE");
    }

    /* Read first so invalid snapshots cannot create an empty module key. */
    ValkeyModuleKey *read_key = NULL;
    RouterIndex *existing = router_index_for_read(ctx, argv[1], &read_key);
    if (existing == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    uint64_t current_generation =
        existing == NULL ? 0 : router_index_rank_generation(existing, worker_id, dp_rank);
    if (current_generation != expected_generation) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_STALE_GENERATION");
    }
    if (existing != NULL && !router_index_generation_can_advance(existing)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_GENERATION_EXHAUSTED");
    }

    RouterIndex *empty_index = NULL;
    RouterIndex *validation_index = existing;
    if (validation_index == NULL) {
        empty_index = router_index_create();
        validation_index = empty_index;
    }
    bool valid = replace_snapshot_validate(
        validation_index,
        (const uint8_t *)snapshot,
        snapshot_length,
        worker_id,
        dp_rank);
    router_index_free(empty_index);
    if (!valid) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_REPLACE");
    }

    ValkeyModuleKey *write_key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &write_key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    if (router_index_worker_cleanup_pending(index, worker_id)) {
        return ValkeyModule_ReplyWithError(
            ctx, "DYNKV_WORKER_CLEANUP_PENDING");
    }
    /* Commands are serialized by Valkey, so this is a defensive assertion only. */
    if (router_index_rank_generation(index, worker_id, dp_rank) != expected_generation) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_STALE_GENERATION");
    }
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, true);
    if (!router_index_admission_ranks_can_advance(index, worker_id, false, dp_rank)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_ADMISSION_INCARNATION_EXHAUSTED");
    }
    router_index_advance_admission_rank_incarnations(index, worker_id, false, dp_rank);
    router_index_revoke_worker_reservations(index, worker_id, false, dp_rank);

    bool was_retired = worker->retired;
    router_index_reset_worker(worker);
    if (was_retired) {
        worker->admission_registered = false;
    }
    if (replace_snapshot_apply(index, (const uint8_t *)snapshot, snapshot_length) !=
        VALKEYMODULE_OK) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_REPLACE");
    }
    if (!router_index_advance_rank_generation(index, worker)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_GENERATION_EXHAUSTED");
    }
    router_index_mark_legacy_rank(index, worker_id, dp_rank);
    ++index->mutation_count;
    ValkeyModule_SignalModifiedKey(ctx, argv[1]);
    ValkeyModule_ReplicateVerbatim(ctx);
    uint8_t response[8];
    encode_u64_be(response, router_index_rank_generation(index, worker_id, dp_rank));
    return ValkeyModule_ReplyWithStringBuffer(ctx, (const char *)response, sizeof(response));
}

int dynkv_restore_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    /* AOF rewrite/load helper, not a client-visible rollback primitive. */
    if ((ValkeyModule_GetContextFlags(ctx) & VALKEYMODULE_CTX_FLAGS_LOADING) == 0) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_RESTORE_LOADING_ONLY");
    }
    size_t snapshot_length = 0;
    const char *snapshot = ValkeyModule_StringPtrLen(argv[2], &snapshot_length);
    RouterIndex *index = router_index_from_snapshot((const uint8_t *)snapshot, snapshot_length);
    if (index == NULL) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_SNAPSHOT");
    }
    ValkeyModuleKey *key = ValkeyModule_OpenKey(ctx, argv[1], VALKEYMODULE_READ | VALKEYMODULE_WRITE);
    int type = ValkeyModule_KeyType(key);
    if (type != VALKEYMODULE_KEYTYPE_EMPTY && ValkeyModule_ModuleTypeGetType(key) != RouterIndexType) {
        router_index_free(index);
        return ValkeyModule_ReplyWithError(ctx, VALKEYMODULE_ERRORMSG_WRONGTYPE);
    }
    void *old_value = NULL;
    if (type == VALKEYMODULE_KEYTYPE_EMPTY) {
        ValkeyModule_ModuleTypeSetValue(key, RouterIndexType, index);
    } else {
        ValkeyModule_ModuleTypeReplaceValue(key, RouterIndexType, index, &old_value);
        router_index_free(old_value);
    }
    ValkeyModule_SignalModifiedKey(ctx, argv[1]);
    ValkeyModule_ReplicateVerbatim(ctx);
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

int dynkv_remove_worker_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 4) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t worker_length = 0;
    const char *worker_data = ValkeyModule_StringPtrLen(argv[2], &worker_length);
    size_t dp_length = 0;
    const char *dp_data = ValkeyModule_StringPtrLen(argv[3], &dp_length);
    Reader worker_reader = {.data = (const uint8_t *)worker_data, .length = worker_length};
    Reader dp_reader = {.data = (const uint8_t *)dp_data, .length = dp_length};
    uint64_t worker_id = 0;
    uint32_t dp_rank = 0;
    if (!reader_u64(&worker_reader, &worker_id) || worker_reader.offset != worker_reader.length ||
        !reader_u32(&dp_reader, &dp_rank) || dp_reader.offset != dp_reader.length) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER");
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    if (router_index_worker_cleanup_pending(index, worker_id)) {
        return ValkeyModule_ReplyWithError(
            ctx, "DYNKV_WORKER_CLEANUP_PENDING");
    }
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, true);
    if (!router_index_generation_can_advance(index)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_GENERATION_EXHAUSTED");
    }
    if (!router_index_admission_ranks_can_advance(index, worker_id, false, dp_rank)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_ADMISSION_INCARNATION_EXHAUSTED");
    }
    router_index_advance_admission_rank_incarnations(index, worker_id, false, dp_rank);
    router_index_revoke_worker_reservations(index, worker_id, false, dp_rank);
    router_index_deactivate_worker(worker);
    worker->retired = true;
    worker->admission_registered = false;
    router_index_advance_rank_generation(index, worker);
    router_index_mark_legacy_rank(index, worker_id, dp_rank);
    ++index->mutation_count;
    ValkeyModule_SignalModifiedKey(ctx, argv[1]);
    ValkeyModule_ReplicateVerbatim(ctx);
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

int dynkv_reset_worker_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 4) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t worker_length = 0;
    const char *worker_data = ValkeyModule_StringPtrLen(argv[2], &worker_length);
    size_t dp_length = 0;
    const char *dp_data = ValkeyModule_StringPtrLen(argv[3], &dp_length);
    Reader worker_reader = {.data = (const uint8_t *)worker_data, .length = worker_length};
    Reader dp_reader = {.data = (const uint8_t *)dp_data, .length = dp_length};
    uint64_t worker_id = 0;
    uint32_t dp_rank = 0;
    if (!reader_u64(&worker_reader, &worker_id) || worker_reader.offset != worker_reader.length ||
        !reader_u32(&dp_reader, &dp_rank) || dp_reader.offset != dp_reader.length) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER");
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    if (router_index_worker_cleanup_pending(index, worker_id)) {
        return ValkeyModule_ReplyWithError(
            ctx, "DYNKV_WORKER_CLEANUP_PENDING");
    }
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, true);
    if (!router_index_generation_can_advance(index)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_GENERATION_EXHAUSTED");
    }
    if (!router_index_admission_ranks_can_advance(index, worker_id, false, dp_rank)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_ADMISSION_INCARNATION_EXHAUSTED");
    }
    router_index_advance_admission_rank_incarnations(index, worker_id, false, dp_rank);
    router_index_revoke_worker_reservations(index, worker_id, false, dp_rank);
    bool was_retired = worker->retired;
    router_index_reset_worker(worker);
    if (was_retired) {
        worker->admission_registered = false;
    }
    router_index_advance_rank_generation(index, worker);
    router_index_mark_legacy_rank(index, worker_id, dp_rank);
    ++index->mutation_count;
    ValkeyModule_SignalModifiedKey(ctx, argv[1]);
    ValkeyModule_ReplicateVerbatim(ctx);
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

int dynkv_remove_worker_all_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t worker_length = 0;
    const char *worker_data = ValkeyModule_StringPtrLen(argv[2], &worker_length);
    Reader reader = {.data = (const uint8_t *)worker_data, .length = worker_length};
    uint64_t worker_id = 0;
    if (!reader_u64(&reader, &worker_id) || reader.offset != reader.length) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER");
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    if (router_index_worker_cleanup_pending(index, worker_id)) {
        return ValkeyModule_ReplyWithError(
            ctx, "DYNKV_WORKER_CLEANUP_PENDING");
    }
    if (!router_index_generation_can_advance(index)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_GENERATION_EXHAUSTED");
    }
    if (!router_index_admission_ranks_can_advance(index, worker_id, true, 0)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_ADMISSION_INCARNATION_EXHAUSTED");
    }
    router_index_advance_admission_rank_incarnations(index, worker_id, true, 0);
    ValkeyModuleDictIter *workers = ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
        WorkerState *worker = data;
        if (worker->worker_id == worker_id) {
            router_index_deactivate_worker(worker);
            worker->retired = true;
            worker->admission_registered = false;
        }
    }
    ValkeyModule_DictIteratorStop(workers);
    router_index_revoke_worker_reservations(index, worker_id, true, 0);
    router_index_advance_worker_epoch(index, worker_id);
    router_index_mark_legacy_worker(index, worker_id);
    WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, false);
    if (epoch->registration_owner_set) {
        (void)router_index_worker_lease_expiry_heap_remove(index, epoch);
        epoch->registration_owner_set = false;
        epoch->registration_owner_nonce = 0;
        epoch->registration_expires_at_ms = 0;
    }
    epoch->admission_retired_all = true;
    ++index->mutation_count;
    ValkeyModule_SignalModifiedKey(ctx, argv[1]);
    ValkeyModule_ReplicateVerbatim(ctx);
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

int dynkv_register_worker_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 4) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t worker_length = 0;
    size_t dp_length = 0;
    const char *worker_data = ValkeyModule_StringPtrLen(argv[2], &worker_length);
    const char *dp_data = ValkeyModule_StringPtrLen(argv[3], &dp_length);
    Reader worker_reader = {.data = (const uint8_t *)worker_data, .length = worker_length};
    Reader dp_reader = {.data = (const uint8_t *)dp_data, .length = dp_length};
    uint64_t worker_id = 0;
    uint32_t dp_rank = 0;
    if (!reader_u64(&worker_reader, &worker_id) || worker_reader.offset != worker_reader.length ||
        !reader_u32(&dp_reader, &dp_rank) || dp_reader.offset != dp_reader.length) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER");
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, false);
    if (epoch != NULL && epoch->lease_cleanup_pending) {
        return ValkeyModule_ReplyWithError(
            ctx, "DYNKV_WORKER_CLEANUP_PENDING");
    }
    if (epoch != NULL && epoch->admission_retired_all) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_RETIRED");
    }
    WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
    if (epoch != NULL && epoch->registration_owner_set) {
        if (worker != NULL && worker->retired) {
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_RETIRED");
        }
        if (worker != NULL && worker->admission_registered) {
            if (router_index_mark_legacy_rank(index, worker_id, dp_rank)) {
                ++index->mutation_count;
                ValkeyModule_SignalModifiedKey(ctx, argv[1]);
                ValkeyModule_ReplicateVerbatim(ctx);
            }
            return ValkeyModule_ReplyWithSimpleString(ctx, "NOOP");
        }
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_OWNED");
    }
    bool changed = false;
    if (worker == NULL) {
        worker = router_index_worker(index, worker_id, dp_rank, true);
        changed = true;
    }
    if (worker->retired) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_RETIRED");
    }
    if (!worker->admission_registered) {
        worker->admission_registered = true;
        worker->lifecycle_tombstone = false;
        worker->lifecycle_tombstone_generation = 0;
        changed = true;
    }
    if (!changed) {
        if (router_index_mark_legacy_rank(index, worker_id, dp_rank)) {
            ++index->mutation_count;
            ValkeyModule_SignalModifiedKey(ctx, argv[1]);
            ValkeyModule_ReplicateVerbatim(ctx);
        }
        return ValkeyModule_ReplyWithSimpleString(ctx, "NOOP");
    }
    router_index_mark_legacy_rank(index, worker_id, dp_rank);
    ++index->mutation_count;
    ValkeyModule_SignalModifiedKey(ctx, argv[1]);
    ValkeyModule_ReplicateVerbatim(ctx);
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}
