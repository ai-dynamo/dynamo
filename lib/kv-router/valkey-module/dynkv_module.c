/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_admission.h"
#include "dynkv_commands.h"
#include "dynkv_gc.h"
#include "dynkv_lease.h"
#include "dynkv_persistence.h"
#include "dynkv_state.h"

int ValkeyModule_OnLoad(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    VALKEYMODULE_NOT_USED(argv);
    VALKEYMODULE_NOT_USED(argc);

    if (ValkeyModule_Init(ctx, "dynkv", 1, VALKEYMODULE_APIVER_1) == VALKEYMODULE_ERR) {
        return VALKEYMODULE_ERR;
    }
    ValkeyModule_SetModuleOptions(ctx, VALKEYMODULE_OPTIONS_HANDLE_IO_ERRORS);

    ValkeyModuleTypeMethods type_methods = {
        .version = VALKEYMODULE_TYPE_METHOD_VERSION,
        .rdb_load = router_index_rdb_load,
        .rdb_save = router_index_rdb_save,
        .aof_rewrite = router_index_aof_rewrite,
        .mem_usage = router_index_mem_usage,
        .free = router_index_free,
    };
    RouterIndexType = ValkeyModule_CreateDataType(
        ctx, "dynkvidx1", DYNKV_SNAPSHOT_VERSION, &type_methods);
    if (RouterIndexType == NULL) {
        return VALKEYMODULE_ERR;
    }
    if (ValkeyModule_CreateCommand(ctx, "dynkv.apply", dynkv_apply_command, "write deny-oom", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.apply_owned", dynkv_apply_owned_command, "write deny-oom", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx,
            "dynkv.apply_owned_at",
            dynkv_apply_owned_at_command,
            "write deny-oom allow-loading",
            1,
            1,
            1) == VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(ctx, "dynkv.barrier", dynkv_barrier_command, "write", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(ctx, "dynkv.match", dynkv_match_command, "readonly", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx,
            "dynkv.match_primary",
            dynkv_match_primary_command,
            "readonly",
            1,
            1,
            1) == VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(ctx, "dynkv.select", dynkv_select_command, "readonly", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx,
            "dynkv.select_reserve",
            dynkv_select_reserve_command,
            "write deny-oom",
            1,
            1,
            1) == VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.release", dynkv_release_command, "write deny-oom", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(ctx, "dynkv.renew", dynkv_renew_command, "write deny-oom", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx,
            "dynkv.admit_apply",
            dynkv_admit_apply_command,
            "write deny-oom allow-loading",
            1,
            1,
            1) == VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx,
            "dynkv.worker_lease_apply",
            dynkv_worker_lease_apply_command,
            "write deny-oom allow-loading",
            1,
            1,
            1) == VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx,
            "dynkv.gc_apply",
            dynkv_gc_apply_command,
            "write deny-oom allow-loading",
            1,
            1,
            1) == VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.rank_generation", dynkv_rank_generation_command, "readonly", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx,
            "dynkv.registration_generation",
            dynkv_registration_generation_command,
            "readonly",
            1,
            1,
            1) == VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx,
            "dynkv.replace_rank_if_generation",
            dynkv_replace_rank_if_generation_command,
            "write deny-oom",
            1,
            1,
            1) == VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.restore", dynkv_restore_command, "write deny-oom allow-loading", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.remove_worker", dynkv_remove_worker_command, "write deny-oom", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.reset_worker", dynkv_reset_worker_command, "write deny-oom", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.remove_worker_all", dynkv_remove_worker_all_command, "write deny-oom", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.register_worker", dynkv_register_worker_command, "write deny-oom", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx,
            "dynkv.register_worker_ranks",
            dynkv_register_worker_ranks_command,
            "write deny-oom",
            1,
            1,
            1) == VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx,
            "dynkv.renew_worker_lease",
            dynkv_renew_worker_lease_command,
            "write deny-oom",
            1,
            1,
            1) == VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx,
            "dynkv.unregister_worker",
            dynkv_unregister_worker_command,
            "write deny-oom",
            1,
            1,
            1) == VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.admission_stats", dynkv_admission_stats_command, "readonly", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.lifecycle_stats", dynkv_lifecycle_stats_command, "readonly", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.gc", dynkv_gc_command, "write deny-oom", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(
            ctx, "dynkv.gc_stats", dynkv_gc_stats_command, "readonly", 1, 1, 1) ==
            VALKEYMODULE_ERR ||
        ValkeyModule_CreateCommand(ctx, "dynkv.stats", dynkv_stats_command, "readonly", 1, 1, 1) ==
            VALKEYMODULE_ERR) {
        return VALKEYMODULE_ERR;
    }
    return VALKEYMODULE_OK;
}
