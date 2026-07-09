/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYNKV_COMMANDS_H
#define DYNKV_COMMANDS_H

#include "dynkv_types.h"

int dynkv_apply_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_apply_owned_at(ValkeyModuleCtx *, ValkeyModuleString *, uint64_t, uint64_t, const uint8_t *, size_t, bool);
int dynkv_apply_owned_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_apply_owned_at_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_barrier_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_match_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_match_primary_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_select_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_admission_stats_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_lifecycle_stats_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_stats_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int ValkeyModule_OnLoad(ValkeyModuleCtx *, ValkeyModuleString **, int);

#endif /* DYNKV_COMMANDS_H */
