/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_commands.h"
#include "dynkv_lease.h"
#include "dynkv_persistence.h"

int dynkv_admission_stats_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 2) {
        return ValkeyModule_WrongArity(ctx);
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    return ValkeyModule_ReplyWithLongLong(
        ctx, index == NULL ? 0 : (long long)ValkeyModule_DictSize(index->reservations));
}

int dynkv_lifecycle_stats_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 2) {
        return ValkeyModule_WrongArity(ctx);
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    uint64_t now_ms = 0;
    if (!admission_now_ms(&now_ms)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_LIFECYCLE_STATS");
    }
    uint64_t active_owner_leases = 0;
    uint64_t retained_rank_tombstones = 0;
    uint64_t ownerless_worker_epochs = 0;
    if (index != NULL) {
        ValkeyModuleDictIter *workers =
            ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
        void *data = NULL;
        while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
            WorkerState *worker = data;
            retained_rank_tombstones += worker->lifecycle_tombstone ? 1 : 0;
        }
        ValkeyModule_DictIteratorStop(workers);
        ValkeyModuleDictIter *epochs =
            ValkeyModule_DictIteratorStartC(index->worker_epochs, "^", NULL, 0);
        while (ValkeyModule_DictNextC(epochs, NULL, &data) != NULL) {
            WorkerEpoch *epoch = data;
            bool owner_is_live = epoch->registration_owner_set &&
                                 !epoch->lease_cleanup_pending &&
                                 epoch->registration_expires_at_ms > now_ms;
            active_owner_leases += owner_is_live ? 1 : 0;
            ownerless_worker_epochs += owner_is_live ? 0 : 1;
        }
        ValkeyModule_DictIteratorStop(epochs);
    }
    ValkeyModule_ReplyWithArray(ctx, 3);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)active_owner_leases);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)retained_rank_tombstones);
    return ValkeyModule_ReplyWithLongLong(ctx, (long long)ownerless_worker_epochs);
}

int dynkv_stats_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 2) {
        return ValkeyModule_WrongArity(ctx);
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    uint64_t nodes = index == NULL ? 0 : ValkeyModule_DictSize(index->nodes_by_external);
    uint64_t workers = index == NULL ? 0 : ValkeyModule_DictSize(index->workers);
    uint64_t mutations = index == NULL ? 0 : index->mutation_count;
    ValkeyModule_ReplyWithArray(ctx, 3);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)nodes);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)workers);
    return ValkeyModule_ReplyWithLongLong(ctx, (long long)mutations);
}
