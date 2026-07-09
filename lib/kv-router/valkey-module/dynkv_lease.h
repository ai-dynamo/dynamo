/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYNKV_LEASE_H
#define DYNKV_LEASE_H

#include "dynkv_types.h"

bool admission_now_ms(uint64_t *);
bool worker_lease_apply_prefix_append(Buffer *, uint8_t, uint64_t);
bool worker_lease_replicate_payload(ValkeyModuleCtx *, ValkeyModuleString *, const Buffer *);
bool worker_lease_commit_payload(ValkeyModuleCtx *, ValkeyModuleString *, RouterIndex *, const Buffer *, bool);
bool sorted_ranks_contains(const uint32_t *, uint32_t, uint32_t);
bool router_index_registered_ranks_equal(RouterIndex *, uint64_t, const uint32_t *, uint32_t);
int router_index_worker_lease_apply_register(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, const uint32_t *, uint32_t, bool, uint64_t, bool *);
int router_index_worker_lease_apply_renew(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, bool *);
int router_index_worker_lease_apply_unregister(RouterIndex *, uint64_t, uint64_t, bool *);
int router_index_worker_lease_apply_expire(RouterIndex *, uint64_t, const WorkerLeaseExpiry *, uint32_t, bool *);
bool worker_lease_commit_one_bounded_expiry(ValkeyModuleCtx *, ValkeyModuleString *, RouterIndex *, uint64_t);
int dynkv_worker_lease_apply_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_rank_generation_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_registration_generation_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_replace_rank_if_generation_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_restore_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_remove_worker_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_reset_worker_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_remove_worker_all_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_register_worker_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int compare_u32(const void *, const void *);
bool registration_ranks_read(const uint8_t *, size_t, uint32_t **, uint32_t *);
bool leased_registration_read(const uint8_t *, size_t, uint64_t *, uint64_t *, uint64_t *, bool *, uint32_t **, uint32_t *);
bool worker_lease_register_apply_payload(Buffer *, uint64_t, uint64_t, uint64_t, uint64_t, bool, uint64_t, const uint32_t *, uint32_t);
int dynkv_register_worker_ranks_leased(ValkeyModuleCtx *, ValkeyModuleString *, uint64_t, const uint8_t *, size_t);
bool worker_lease_control_read(const uint8_t *, size_t, bool, uint64_t *, uint64_t *, uint64_t *);
bool worker_lease_renew_apply_payload(Buffer *, uint64_t, uint64_t, uint64_t, uint64_t);
bool worker_lease_unregister_apply_payload(Buffer *, uint64_t, uint64_t, uint64_t);
int dynkv_renew_worker_lease_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_unregister_worker_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_register_worker_ranks_command(ValkeyModuleCtx *, ValkeyModuleString **, int);

#endif /* DYNKV_LEASE_H */
