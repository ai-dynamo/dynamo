/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_index.h"
#include "dynkv_gc.h"
#include "dynkv_lease.h"
#include "dynkv_persistence.h"
#include "dynkv_state.h"

bool admission_now_ms(uint64_t *now_ms) {
    mstime_t now = ValkeyModule_Milliseconds();
    if (now < 0) {
        return false;
    }
    *now_ms = (uint64_t)now;
    return true;
}

bool worker_lease_apply_prefix_append(Buffer *payload, uint8_t op, uint64_t now_ms) {
    return buffer_u8(payload, DYNKV_WORKER_LEASE_APPLY_VERSION) && buffer_u8(payload, op) &&
           buffer_u64(payload, now_ms);
}

bool worker_lease_replicate_payload(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString *key,
    const Buffer *payload) {
    return ValkeyModule_Replicate(
               ctx,
               "DYNKV.WORKER_LEASE_APPLY",
               "sb",
               key,
               (char *)payload->data,
               payload->length) == VALKEYMODULE_OK;
}

bool worker_lease_commit_payload(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString *key,
    RouterIndex *index,
    const Buffer *payload,
    bool changed) {
    if (changed) {
        ++index->mutation_count;
        ValkeyModule_SignalModifiedKey(ctx, key);
    }
    return worker_lease_replicate_payload(ctx, key, payload);
}

bool sorted_ranks_contains(
    const uint32_t *ranks,
    uint32_t rank_count,
    uint32_t dp_rank) {
    uint32_t left = 0;
    uint32_t right = rank_count;
    while (left < right) {
        uint32_t middle = left + (right - left) / 2;
        if (ranks[middle] < dp_rank) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left < rank_count && ranks[left] == dp_rank;
}

bool router_index_registered_ranks_equal(
    RouterIndex *index,
    uint64_t worker_id,
    const uint32_t *ranks,
    uint32_t rank_count) {
    uint32_t found = 0;
    ValkeyModuleDictIter *workers =
        ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
    void *data = NULL;
    bool equal = true;
    while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
        WorkerState *worker = data;
        if (worker->worker_id != worker_id || !worker->admission_registered) {
            continue;
        }
        if (!sorted_ranks_contains(ranks, rank_count, worker->dp_rank)) {
            equal = false;
            break;
        }
        ++found;
    }
    ValkeyModule_DictIteratorStop(workers);
    return equal && found == rank_count;
}

int router_index_worker_lease_apply_register(
    RouterIndex *index,
    uint64_t now_ms,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    const uint32_t *ranks,
    uint32_t rank_count,
    bool replay_safe,
    uint64_t expected_registration_generation,
    bool *changed_out) {
    *changed_out = false;
    if (owner_nonce == 0 || expires_at_ms <= now_ms || rank_count == 0 ||
        rank_count > DYNKV_MAX_REGISTRATION_RANKS) {
        return DYNKV_WORKER_LEASE_INVALID;
    }
    WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, false);
    bool epoch_was_absent = epoch == NULL;
    if (epoch != NULL && epoch->admission_retired_all) {
        return DYNKV_WORKER_LEASE_INVALID;
    }
    for (uint32_t i = 0; i < rank_count; ++i) {
        WorkerState *worker = router_index_worker(index, worker_id, ranks[i], false);
        if (worker != NULL && worker->retired) {
            return DYNKV_WORKER_LEASE_INVALID;
        }
    }

    bool rank_set_changed =
        !router_index_registered_ranks_equal(index, worker_id, ranks, rank_count);
    bool removes_rank = false;
    ValkeyModuleDictIter *existing_workers =
        ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
    void *existing_data = NULL;
    while (ValkeyModule_DictNextC(existing_workers, NULL, &existing_data) != NULL) {
        WorkerState *worker = existing_data;
        if (worker->worker_id == worker_id && worker->admission_registered &&
            !sorted_ranks_contains(ranks, rank_count, worker->dp_rank)) {
            removes_rank = true;
            break;
        }
    }
    ValkeyModule_DictIteratorStop(existing_workers);
    bool expired = epoch != NULL && epoch->registration_owner_set &&
                   epoch->registration_expires_at_ms <= now_ms;
    /* Expired ownership is drained only by bounded GC, never on this path. */
    if (expired || (epoch != NULL && epoch->lease_cleanup_pending)) {
        return DYNKV_WORKER_LEASE_INVALID;
    }
    bool owner_acquisition =
        epoch == NULL || !epoch->registration_owner_set || expired;
    bool advance_registration =
        epoch_was_absent || (!owner_acquisition && rank_set_changed);
    uint64_t lifecycle_advances =
        (expired ? 1 : 0) + (advance_registration ? 1 : 0);
    if (lifecycle_advances >
        UINT64_MAX - index->lifecycle_generation_counter) {
        return DYNKV_WORKER_LEASE_INVALID;
    }
    if (removes_rank &&
        (!router_index_generation_can_advance(index) ||
         !router_index_admission_ranks_can_advance(index, worker_id, true, 0))) {
        return DYNKV_WORKER_LEASE_INVALID;
    }

    if (epoch != NULL && epoch->registration_owner_set && !expired &&
        epoch->registration_owner_nonce != owner_nonce) {
        return DYNKV_WORKER_LEASE_OWNED;
    }
    if (expired && !router_index_worker_lease_can_end(index, worker_id)) {
        return DYNKV_WORKER_LEASE_INVALID;
    }
    if (!router_index_worker_lease_expiry_heap_reserve(
            index, index->worker_lease_expiry_heap_length + (expired ? 0 : 1))) {
        return DYNKV_WORKER_LEASE_INVALID;
    }
    if (expired) {
        if (!router_index_end_worker_lease(index, epoch)) {
            return DYNKV_WORKER_LEASE_INVALID;
        }
        *changed_out = true;
        /* The expired lease already fenced and removed every prior rank. */
        removes_rank = false;
        epoch = router_index_worker_epoch(index, worker_id, false);
    }
    if (epoch == NULL) {
        epoch = router_index_worker_epoch(index, worker_id, true);
    }
    epoch->lifecycle_managed = true;
    if (!replay_safe && !epoch->legacy_tainted) {
        epoch->legacy_tainted = true;
        *changed_out = true;
    }

    if (!epoch->registration_owner_set) {
        epoch->registration_owner_set = true;
        epoch->registration_owner_nonce = owner_nonce;
        epoch->registration_expires_at_ms = expires_at_ms;
        if (!router_index_worker_lease_expiry_heap_insert(index, epoch)) {
            epoch->registration_owner_set = false;
            epoch->registration_owner_nonce = 0;
            epoch->registration_expires_at_ms = 0;
            return DYNKV_WORKER_LEASE_INVALID;
        }
        *changed_out = true;
    } else {
        if (epoch->registration_owner_nonce != owner_nonce) {
            return DYNKV_WORKER_LEASE_OWNED;
        }
        if (expires_at_ms > epoch->registration_expires_at_ms) {
            uint64_t previous_expiry = epoch->registration_expires_at_ms;
            epoch->registration_expires_at_ms = expires_at_ms;
            if (!router_index_worker_lease_expiry_heap_reposition(index, epoch)) {
                epoch->registration_expires_at_ms = previous_expiry;
                return DYNKV_WORKER_LEASE_INVALID;
            }
            *changed_out = true;
        }
    }

    for (uint32_t i = 0; i < rank_count; ++i) {
        WorkerState *worker = router_index_worker(index, worker_id, ranks[i], true);
        worker->lifecycle_managed = true;
        if (!replay_safe && !worker->legacy_tainted) {
            worker->legacy_tainted = true;
            *changed_out = true;
        }
        if (!worker->admission_registered || worker->lifecycle_tombstone) {
            worker->admission_registered = true;
            worker->lifecycle_tombstone = false;
            worker->lifecycle_tombstone_generation = 0;
            *changed_out = true;
        }
    }
    if (removes_rank) {
        ValkeyModuleDictIter *workers =
            ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
        void *data = NULL;
        while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
            WorkerState *worker = data;
            if (worker->worker_id != worker_id || !worker->admission_registered ||
                sorted_ranks_contains(ranks, rank_count, worker->dp_rank)) {
                continue;
            }
            (void)router_index_deactivate_worker(worker);
            worker->admission_registered = false;
            worker->lifecycle_tombstone = true;
            router_index_advance_admission_rank_incarnations(
                index, worker_id, false, worker->dp_rank);
            (void)router_index_revoke_worker_reservations(
                index, worker_id, false, worker->dp_rank);
        }
        ValkeyModule_DictIteratorStop(workers);
        (void)router_index_advance_worker_epoch(index, worker_id);
        WorkerEpoch *advanced_epoch =
            router_index_worker_epoch(index, worker_id, false);
        ValkeyModuleDictIter *tombstones =
            ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
        while (ValkeyModule_DictNextC(tombstones, NULL, &data) != NULL) {
            WorkerState *worker = data;
            if (worker->worker_id == worker_id && worker->lifecycle_tombstone) {
                worker->lifecycle_tombstone_generation = advanced_epoch->generation;
            }
        }
        ValkeyModule_DictIteratorStop(tombstones);
        *changed_out = true;
    }
    if (advance_registration) {
        if (!router_index_advance_lifecycle_generation(index, epoch)) {
            return DYNKV_WORKER_LEASE_INVALID;
        }
        if (replay_safe) {
            epoch->last_registration_expected_set = true;
            epoch->last_registration_expected_generation =
                expected_registration_generation;
        } else {
            epoch->last_registration_expected_set = false;
            epoch->last_registration_expected_generation = 0;
        }
        *changed_out = true;
    }
    return VALKEYMODULE_OK;
}

int router_index_worker_lease_apply_renew(
    RouterIndex *index,
    uint64_t now_ms,
    uint64_t worker_id,
    uint64_t owner_nonce,
    uint64_t expires_at_ms,
    bool *changed_out) {
    *changed_out = false;
    WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, false);
    if (epoch == NULL || !epoch->registration_owner_set ||
        epoch->lease_cleanup_pending ||
        epoch->registration_owner_nonce != owner_nonce) {
        return DYNKV_WORKER_LEASE_STALE;
    }
    if (epoch->registration_expires_at_ms <= now_ms) {
        return DYNKV_WORKER_LEASE_STALE;
    }
    if (expires_at_ms <= now_ms) {
        return DYNKV_WORKER_LEASE_INVALID;
    }
    if (expires_at_ms <= epoch->registration_expires_at_ms) {
        return VALKEYMODULE_OK;
    }
    uint64_t previous_expiry = epoch->registration_expires_at_ms;
    epoch->registration_expires_at_ms = expires_at_ms;
    if (!router_index_worker_lease_expiry_heap_reposition(index, epoch)) {
        epoch->registration_expires_at_ms = previous_expiry;
        return DYNKV_WORKER_LEASE_INVALID;
    }
    *changed_out = true;
    return VALKEYMODULE_OK;
}

int router_index_worker_lease_apply_unregister(
    RouterIndex *index,
    uint64_t worker_id,
    uint64_t owner_nonce,
    bool *changed_out) {
    *changed_out = false;
    WorkerEpoch *epoch = router_index_worker_epoch(index, worker_id, false);
    if (epoch == NULL || !epoch->registration_owner_set ||
        epoch->lease_cleanup_pending ||
        epoch->registration_owner_nonce != owner_nonce) {
        return DYNKV_WORKER_LEASE_STALE;
    }
    if (!gc_apply_begin_unregister_cleanup(index, worker_id, owner_nonce)) {
        return DYNKV_WORKER_LEASE_INVALID;
    }
    *changed_out = true;
    return VALKEYMODULE_OK;
}

int router_index_worker_lease_apply_expire(
    RouterIndex *index,
    uint64_t now_ms,
    const WorkerLeaseExpiry *expired,
    uint32_t expired_count,
    bool *changed_out) {
    *changed_out = false;
    uint32_t leases_to_end = 0;
    for (uint32_t i = 0; i < expired_count; ++i) {
        const WorkerLeaseExpiry *entry = &expired[i];
        WorkerEpoch *epoch = router_index_worker_epoch(index, entry->worker_id, false);
        if (epoch != NULL && epoch->registration_owner_set &&
            epoch->registration_owner_nonce == entry->owner_nonce &&
            epoch->registration_expires_at_ms == entry->expires_at_ms &&
            epoch->registration_expires_at_ms <= now_ms) {
            if (!router_index_worker_lease_can_end(index, entry->worker_id)) {
                return DYNKV_WORKER_LEASE_INVALID;
            }
            ++leases_to_end;
        }
    }
    if (!router_index_worker_epochs_can_advance(index, leases_to_end) ||
        (uint64_t)leases_to_end >
            UINT64_MAX - index->lifecycle_generation_counter) {
        return DYNKV_WORKER_LEASE_INVALID;
    }
    for (uint32_t i = 0; i < expired_count; ++i) {
        const WorkerLeaseExpiry *entry = &expired[i];
        WorkerEpoch *epoch = router_index_worker_epoch(index, entry->worker_id, false);
        if (epoch == NULL || !epoch->registration_owner_set ||
            epoch->registration_owner_nonce != entry->owner_nonce ||
            epoch->registration_expires_at_ms != entry->expires_at_ms ||
            epoch->registration_expires_at_ms > now_ms) {
            continue;
        }
        if (!router_index_end_worker_lease(index, epoch)) {
            return DYNKV_WORKER_LEASE_INVALID;
        }
        *changed_out = true;
    }
    return VALKEYMODULE_OK;
}

bool worker_lease_commit_one_bounded_expiry(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString *key,
    RouterIndex *index,
    uint64_t now_ms) {
    if (index->worker_lease_expiry_heap_length == 0) {
        return true;
    }
    WorkerEpoch *epoch = index->worker_lease_expiry_heap[0];
    if (epoch->registration_expires_at_ms > now_ms) {
        return true;
    }
    uint64_t cost = 1 + epoch->worker_state_count +
                    epoch->admission_rank_count +
                    epoch->reservation_count * 32 +
                    epoch->node_membership_count;
    if (cost > DYNKV_INLINE_EXPIRY_WORK_BUDGET) {
        return true;
    }
    WorkerLeaseExpiry expired = {
        .worker_id = epoch->worker_id,
        .owner_nonce = epoch->registration_owner_nonce,
        .expires_at_ms = epoch->registration_expires_at_ms,
    };
    Buffer payload = {0};
    bool valid = worker_lease_apply_prefix_append(
                     &payload, DYNKV_WORKER_LEASE_EXPIRE, now_ms) &&
                 buffer_u32(&payload, 1) &&
                 buffer_u64(&payload, expired.worker_id) &&
                 buffer_u64(&payload, expired.owner_nonce) &&
                 buffer_u64(&payload, expired.expires_at_ms);
    bool changed = false;
    int result = valid ? router_index_worker_lease_apply_expire(
                             index, now_ms, &expired, 1, &changed)
                       : DYNKV_WORKER_LEASE_INVALID;
    bool committed = result == VALKEYMODULE_OK &&
                     worker_lease_commit_payload(ctx, key, index, &payload, changed);
    buffer_free(&payload);
    return committed;
}

int dynkv_worker_lease_apply_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    if (!ValkeyModule_MustObeyClient(ctx)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_WORKER_LEASE_APPLY_INTERNAL_ONLY");
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
        !reader_u64(&reader, &now_ms) ||
        (version != DYNKV_WORKER_LEASE_APPLY_VERSION_LEGACY &&
         version != DYNKV_WORKER_LEASE_APPLY_VERSION) ||
        op < DYNKV_WORKER_LEASE_REGISTER || op > DYNKV_WORKER_LEASE_EXPIRE) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE_APPLY");
    }

    uint64_t worker_id = 0;
    uint64_t owner_nonce = 0;
    uint64_t expires_at_ms = 0;
    uint32_t rank_count = 0;
    uint32_t *ranks = NULL;
    uint8_t replay_safe_value = 0;
    uint64_t expected_registration_generation = 0;
    uint32_t expired_count = 0;
    WorkerLeaseExpiry *expired = NULL;
    if (op == DYNKV_WORKER_LEASE_REGISTER) {
        if (!reader_u64(&reader, &worker_id) || !reader_u64(&reader, &owner_nonce) ||
            !reader_u64(&reader, &expires_at_ms) ||
            (version >= DYNKV_WORKER_LEASE_APPLY_VERSION &&
             (!reader_u8(&reader, &replay_safe_value) || replay_safe_value > 1 ||
              !reader_u64(&reader, &expected_registration_generation))) ||
            !reader_u32(&reader, &rank_count) ||
            rank_count == 0 || rank_count > DYNKV_MAX_REGISTRATION_RANKS ||
            reader.length - reader.offset != (size_t)rank_count * sizeof(uint32_t)) {
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE_APPLY");
        }
        ranks = ValkeyModule_Alloc((size_t)rank_count * sizeof(*ranks));
        for (uint32_t i = 0; i < rank_count; ++i) {
            if (!reader_u32(&reader, &ranks[i])) {
                ValkeyModule_Free(ranks);
                return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE_APPLY");
            }
        }
        qsort(ranks, rank_count, sizeof(*ranks), compare_u32);
        for (uint32_t i = 1; i < rank_count; ++i) {
            if (ranks[i] == ranks[i - 1]) {
                ValkeyModule_Free(ranks);
                return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE_APPLY");
            }
        }
    } else if (op == DYNKV_WORKER_LEASE_RENEW) {
        if (!reader_u64(&reader, &worker_id) || !reader_u64(&reader, &owner_nonce) ||
            !reader_u64(&reader, &expires_at_ms) || reader.offset != reader.length) {
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE_APPLY");
        }
    } else if (op == DYNKV_WORKER_LEASE_UNREGISTER) {
        if (!reader_u64(&reader, &worker_id) || !reader_u64(&reader, &owner_nonce) ||
            reader.offset != reader.length) {
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE_APPLY");
        }
    } else {
        if (!reader_u32(&reader, &expired_count) ||
            reader.length - reader.offset != (size_t)expired_count * 24) {
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE_APPLY");
        }
        if (expired_count != 0) {
            expired = ValkeyModule_Alloc((size_t)expired_count * sizeof(*expired));
        }
        for (uint32_t i = 0; i < expired_count; ++i) {
            if (!reader_u64(&reader, &expired[i].worker_id) ||
                !reader_u64(&reader, &expired[i].owner_nonce) ||
                !reader_u64(&reader, &expired[i].expires_at_ms)) {
                ValkeyModule_Free(expired);
                return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE_APPLY");
            }
        }
    }

    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &key);
    if (index == NULL) {
        ValkeyModule_Free(ranks);
        ValkeyModule_Free(expired);
        return VALKEYMODULE_OK;
    }
    bool changed = false;
    int result = VALKEYMODULE_OK;
    if (op == DYNKV_WORKER_LEASE_REGISTER) {
        result = router_index_worker_lease_apply_register(
            index,
            now_ms,
            worker_id,
            owner_nonce,
            expires_at_ms,
            ranks,
            rank_count,
            replay_safe_value == 1,
            expected_registration_generation,
            &changed);
    } else if (op == DYNKV_WORKER_LEASE_RENEW) {
        result = router_index_worker_lease_apply_renew(
            index, now_ms, worker_id, owner_nonce, expires_at_ms, &changed);
    } else if (op == DYNKV_WORKER_LEASE_UNREGISTER) {
        result = router_index_worker_lease_apply_unregister(
            index, worker_id, owner_nonce, &changed);
    } else {
        result = router_index_worker_lease_apply_expire(
            index, now_ms, expired, expired_count, &changed);
    }
    ValkeyModule_Free(ranks);
    ValkeyModule_Free(expired);
    if (result != VALKEYMODULE_OK && result != DYNKV_WORKER_LEASE_STALE &&
        result != DYNKV_WORKER_LEASE_OWNED) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_WORKER_LEASE_APPLY");
    }
    if (changed) {
        ++index->mutation_count;
        ValkeyModule_SignalModifiedKey(ctx, argv[1]);
    }
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}
