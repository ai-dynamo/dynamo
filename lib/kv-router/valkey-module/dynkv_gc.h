/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYNKV_GC_H
#define DYNKV_GC_H

#include "dynkv_types.h"

bool gc_worker_is_reclaimable(RouterIndex *, const WorkerState *, uint64_t);
bool router_index_delete_worker_state(RouterIndex *, WorkerState *);
bool router_index_delete_worker_epoch(RouterIndex *, WorkerEpoch *);
bool gc_append_remove_owner(Buffer *, const IndexNode *, const WorkerState *);
bool gc_append_remove_admission_rank(Buffer *, const AdmissionRankState *, const WorkerState *);
bool gc_append_remove_node(Buffer *, const IndexNode *);
bool gc_append_remove_worker(Buffer *, const WorkerState *);
bool gc_append_remove_worker_epoch(Buffer *, const WorkerEpoch *);
bool gc_append_cleanup_epoch_identity(Buffer *, const WorkerEpoch *);
bool gc_append_begin_lease_cleanup(Buffer *, const RouterIndex *, const WorkerEpoch *, uint64_t);
bool gc_append_cleanup_reservation(Buffer *, const WorkerEpoch *, const Reservation *);
bool gc_append_cleanup_admission_rank(Buffer *, const WorkerEpoch *, const AdmissionRankState *);
bool gc_append_cleanup_owner(Buffer *, const WorkerEpoch *, const WorkerState *, const IndexNode *, const Owner *);
bool gc_append_cleanup_worker(Buffer *, const WorkerEpoch *, const WorkerState *);
bool gc_append_finalize_lease_cleanup(Buffer *, const WorkerEpoch *);
WorkerEpoch *gc_cleanup_epoch(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t);
void gc_prioritize_pending_cleanup(RouterIndex *);
bool gc_can_begin_lease_cleanup(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
bool gc_apply_begin_lease_cleanup(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
bool gc_apply_begin_unregister_cleanup(RouterIndex *, uint64_t, uint64_t);
bool gc_can_cleanup_reservation(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, const uint8_t *, uint32_t, uint64_t, uint64_t, uint32_t, uint64_t);
bool gc_apply_cleanup_reservation(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, const uint8_t *, uint32_t, uint64_t, uint64_t, uint32_t, uint64_t);
bool gc_can_cleanup_admission_rank(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, const uint8_t *, uint32_t, uint32_t, uint64_t);
bool gc_apply_cleanup_admission_rank(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, const uint8_t *, uint32_t, uint32_t, uint64_t);
void worker_prepare_cleanup_nodes(WorkerState *, uint64_t);
bool gc_can_cleanup_owner(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint64_t, uint64_t, bool);
bool gc_apply_cleanup_owner(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint64_t, uint64_t, bool);
bool gc_can_cleanup_worker(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t);
bool gc_apply_cleanup_worker(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t);
bool gc_can_finalize_lease_cleanup(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t);
bool gc_apply_finalize_lease_cleanup(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t);
bool gc_can_remove_owner(RouterIndex *, uint64_t, uint64_t, uint32_t, uint64_t);
bool gc_apply_remove_owner(RouterIndex *, uint64_t, uint64_t, uint32_t, uint64_t);
bool gc_can_remove_admission_rank(RouterIndex *, const uint8_t *, uint32_t, uint64_t, uint32_t, uint64_t, uint64_t);
bool gc_apply_remove_admission_rank(RouterIndex *, const uint8_t *, uint32_t, uint64_t, uint32_t, uint64_t, uint64_t);
bool gc_apply_remove_node(RouterIndex *, uint64_t, uint64_t, uint64_t);
bool gc_can_remove_node(RouterIndex *, uint64_t, uint64_t, uint64_t);
bool gc_apply_remove_worker(RouterIndex *, uint64_t, uint32_t, uint64_t);
bool gc_can_remove_worker(RouterIndex *, uint64_t, uint32_t, uint64_t);
bool gc_apply_remove_worker_epoch(RouterIndex *, uint64_t, uint64_t);
bool gc_can_remove_worker_epoch(RouterIndex *, uint64_t, uint64_t);
bool gc_can_expire_worker_lease(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t);
bool gc_apply_expire_worker_lease(RouterIndex *, uint64_t, uint64_t, uint64_t, uint64_t);
ValkeyModuleDict *gc_phase_dict(RouterIndex *);
bool gc_next_record(RouterIndex *, void **, uint8_t *);
bool gc_plan_pending_lease_cleanup(WorkerEpoch *, uint64_t, Buffer *, GcResult *);
bool gc_plan_record(RouterIndex *, uint8_t, void *, uint64_t, uint64_t, uint64_t, Buffer *, GcResult *);
bool gc_read_cleanup_epoch_identity(Reader *, uint64_t *, uint64_t *, uint64_t *, uint64_t *);
bool gc_payload_identities_unique(const uint8_t *, size_t);
bool gc_process_payload(RouterIndex *, const uint8_t *, size_t, bool);
bool gc_apply_payload(RouterIndex *, const uint8_t *, size_t);
int dynkv_gc_apply_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_gc_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_gc_stats_command(ValkeyModuleCtx *, ValkeyModuleString **, int);

#endif /* DYNKV_GC_H */
