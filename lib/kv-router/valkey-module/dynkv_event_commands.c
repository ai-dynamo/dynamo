/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_commands.h"
#include "dynkv_index.h"
#include "dynkv_lease.h"
#include "dynkv_persistence.h"
#include "dynkv_state.h"

int dynkv_apply_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t payload_length = 0;
    const char *payload = ValkeyModule_StringPtrLen(argv[2], &payload_length);
    if (router_index_validate_event((const uint8_t *)payload, payload_length) != VALKEYMODULE_OK) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_EVENT");
    }
    EventHeader header = {0};
    if (!router_event_header((const uint8_t *)payload, payload_length, &header)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_EVENT");
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    if (router_index_worker_cleanup_pending(index, header.worker_id)) {
        return ValkeyModule_ReplyWithError(
            ctx, "DYNKV_WORKER_CLEANUP_PENDING");
    }
    if (!router_index_can_advance_for_event(index, &header)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_GENERATION_EXHAUSTED");
    }
    int result = router_index_apply_event(index, (const uint8_t *)payload, payload_length);
    if (result == 2) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_MISSING_PARENT");
    }
    if (result == 3) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_CONFLICTING_BLOCK");
    }
    /*
     * A remove can arrive before this index has seen the matching store.  It
     * still fences a tree dump captured before the remove: otherwise that old
     * dump could resurrect the removed block.  Persist a rank ticket even
     * though no owner changed locally.
     */
    bool fence_noop_remove =
        result == DYNKV_NOOP && header.kind == DYNKV_EVENT_REMOVE &&
        router_event_remove_has_blocks((const uint8_t *)payload, payload_length);
    if (result != VALKEYMODULE_OK && !fence_noop_remove) {
        if (result == DYNKV_NOOP) {
            bool provenance_changed = header.kind == DYNKV_EVENT_CLEAR
                                          ? router_index_mark_legacy_worker(
                                                index, header.worker_id)
                                          : router_index_mark_legacy_rank(
                                                index,
                                                header.worker_id,
                                                header.dp_rank);
            if (provenance_changed) {
                ++index->mutation_count;
                ValkeyModule_SignalModifiedKey(ctx, argv[1]);
                ValkeyModule_ReplicateVerbatim(ctx);
            }
            return ValkeyModule_ReplyWithSimpleString(ctx, "NOOP");
        }
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_EVENT");
    }
    if (!router_index_advance_for_event(index, &header)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_GENERATION_EXHAUSTED");
    }
    if (header.kind == DYNKV_EVENT_CLEAR) {
        router_index_mark_legacy_worker(index, header.worker_id);
    } else {
        router_index_mark_legacy_rank(index, header.worker_id, header.dp_rank);
    }
    if (fence_noop_remove) {
        ++index->mutation_count;
    }
    ValkeyModule_SignalModifiedKey(ctx, argv[1]);
    ValkeyModule_ReplicateVerbatim(ctx);
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

int dynkv_apply_owned_at(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString *key_name,
    uint64_t owner_nonce,
    uint64_t now_ms,
    const uint8_t *payload,
    size_t payload_length,
    bool replicate) {
    if (router_index_validate_event(payload, payload_length) != VALKEYMODULE_OK) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_EVENT");
    }
    EventHeader header = {0};
    if (!router_event_header(payload, payload_length, &header)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_EVENT");
    }
    ValkeyModuleKey *read_key = NULL;
    RouterIndex *existing = router_index_for_read(ctx, key_name, &read_key);
    if (existing == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    if (existing == NULL ||
        !worker_registration_owner_is_live(existing, header.worker_id, owner_nonce, now_ms)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_STALE_WORKER_OWNER");
    }
    WorkerState *owned_rank =
        router_index_worker(existing, header.worker_id, header.dp_rank, false);
    if (owned_rank == NULL || owned_rank->retired || !owned_rank->admission_registered) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_UNREGISTERED_WORKER_RANK");
    }
    ValkeyModuleKey *write_key = NULL;
    RouterIndex *index = router_index_for_write(ctx, key_name, &write_key);
    if (index == NULL) {
        return VALKEYMODULE_OK;
    }
    if (!router_index_can_advance_for_event(index, &header)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_GENERATION_EXHAUSTED");
    }
    int result = router_index_apply_event(index, payload, payload_length);
    if (result == 2) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_MISSING_PARENT");
    }
    if (result == 3) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_CONFLICTING_BLOCK");
    }
    bool fence_noop_remove =
        result == DYNKV_NOOP && header.kind == DYNKV_EVENT_REMOVE &&
        router_event_remove_has_blocks(payload, payload_length);
    if (result != VALKEYMODULE_OK && !fence_noop_remove) {
        if (result == DYNKV_NOOP) {
            return ValkeyModule_ReplyWithSimpleString(ctx, "NOOP");
        }
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_EVENT");
    }
    if (result == VALKEYMODULE_OK) {
        if (header.kind == DYNKV_EVENT_CLEAR) {
            router_index_compact_direct_worker_all_ranks(
                index, header.worker_id);
        } else if (header.kind == DYNKV_EVENT_REMOVE) {
            router_index_compact_direct_remove(
                index,
                router_index_worker(
                    index, header.worker_id, header.dp_rank, false),
                payload,
                payload_length);
        }
    }
    if (!router_index_advance_for_event(index, &header)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_GENERATION_EXHAUSTED");
    }
    if (fence_noop_remove) {
        ++index->mutation_count;
    }
    ValkeyModule_SignalModifiedKey(ctx, key_name);
    if (replicate) {
        uint8_t owner_data[8];
        uint8_t now_data[8];
        encode_u64_be(owner_data, owner_nonce);
        encode_u64_be(now_data, now_ms);
        if (ValkeyModule_Replicate(
                ctx,
                "DYNKV.APPLY_OWNED_AT",
                "sbbb",
                key_name,
                (char *)owner_data,
                sizeof(owner_data),
                (char *)now_data,
                sizeof(now_data),
                (char *)payload,
                payload_length) != VALKEYMODULE_OK) {
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
        }
    }
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

int dynkv_apply_owned_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 4) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t owner_length = 0;
    size_t payload_length = 0;
    const char *owner_data = ValkeyModule_StringPtrLen(argv[2], &owner_length);
    const char *payload = ValkeyModule_StringPtrLen(argv[3], &payload_length);
    Reader owner_reader = {.data = (const uint8_t *)owner_data, .length = owner_length};
    uint64_t owner_nonce = 0;
    uint64_t now_ms = 0;
    if (!reader_u64(&owner_reader, &owner_nonce) || owner_reader.offset != owner_reader.length ||
        owner_nonce == 0 || !admission_now_ms(&now_ms)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_OWNER");
    }
    return dynkv_apply_owned_at(
        ctx,
        argv[1],
        owner_nonce,
        now_ms,
        (const uint8_t *)payload,
        payload_length,
        true);
}

int dynkv_apply_owned_at_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 5) {
        return ValkeyModule_WrongArity(ctx);
    }
    if (!ValkeyModule_MustObeyClient(ctx)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_APPLY_OWNED_AT_INTERNAL_ONLY");
    }
    size_t owner_length = 0;
    size_t now_length = 0;
    size_t payload_length = 0;
    const char *owner_data = ValkeyModule_StringPtrLen(argv[2], &owner_length);
    const char *now_data = ValkeyModule_StringPtrLen(argv[3], &now_length);
    const char *payload = ValkeyModule_StringPtrLen(argv[4], &payload_length);
    Reader owner_reader = {.data = (const uint8_t *)owner_data, .length = owner_length};
    Reader now_reader = {.data = (const uint8_t *)now_data, .length = now_length};
    uint64_t owner_nonce = 0;
    uint64_t now_ms = 0;
    if (!reader_u64(&owner_reader, &owner_nonce) || owner_reader.offset != owner_reader.length ||
        owner_nonce == 0 || !reader_u64(&now_reader, &now_ms) ||
        now_reader.offset != now_reader.length) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_OWNER");
    }
    return dynkv_apply_owned_at(
        ctx,
        argv[1],
        owner_nonce,
        now_ms,
        (const uint8_t *)payload,
        payload_length,
        false);
}

/*
 * Propagate a no-state-change replication barrier. This is used only after an
 * ambiguous APPLY/WAIT failure: a retry can legitimately return NOOP because
 * the primary already committed the original event, so a new replicated
 * command plus WAIT is needed to prove the preceding primary stream reached
 * the standby. Keep ordinary NOOP APPLYs unreplicated on the hot path.
 */
int dynkv_barrier_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 2) {
        return ValkeyModule_WrongArity(ctx);
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    ValkeyModule_ReplicateVerbatim(ctx);
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

int dynkv_match_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t payload_length = 0;
    const char *payload = ValkeyModule_StringPtrLen(argv[2], &payload_length);
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    uint64_t now_ms = 0;
    if (!admission_now_ms(&now_ms)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_MATCH");
    }
    Buffer response = {0};
    int result = VALKEYMODULE_ERR;
    if (index == NULL) {
        result = router_match_request_valid((const uint8_t *)payload, payload_length) &&
                         buffer_u8(&response, DYNKV_WIRE_VERSION) && buffer_u32(&response, 0)
                     ? VALKEYMODULE_OK
                     : VALKEYMODULE_ERR;
    } else {
        result = router_index_match(
            index, (const uint8_t *)payload, payload_length, &response, now_ms);
    }
    if (result != VALKEYMODULE_OK) {
        buffer_free(&response);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_MATCH");
    }
    int reply = ValkeyModule_ReplyWithStringBuffer(ctx, (const char *)response.data, response.length);
    buffer_free(&response);
    return reply;
}

int dynkv_match_primary_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    if ((ValkeyModule_GetContextFlags(ctx) & VALKEYMODULE_CTX_FLAGS_PRIMARY) == 0) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_NOT_PRIMARY");
    }
    return dynkv_match_command(ctx, argv, argc);
}

int dynkv_select_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t payload_length = 0;
    const char *payload = ValkeyModule_StringPtrLen(argv[2], &payload_length);
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    uint64_t now_ms = 0;
    if (!admission_now_ms(&now_ms)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_SELECT");
    }
    RouterIndex *empty_index = NULL;
    if (index == NULL) {
        empty_index = router_index_create();
        index = empty_index;
    }
    Buffer response = {0};
    int result = router_index_select(
        index, (const uint8_t *)payload, payload_length, &response, now_ms);
    router_index_free(empty_index);
    if (result != VALKEYMODULE_OK) {
        buffer_free(&response);
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_SELECT");
    }
    int reply = ValkeyModule_ReplyWithStringBuffer(ctx, (const char *)response.data, response.length);
    buffer_free(&response);
    return reply;
}
