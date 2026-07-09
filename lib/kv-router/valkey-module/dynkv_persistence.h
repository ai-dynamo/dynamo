/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYNKV_PERSISTENCE_H
#define DYNKV_PERSISTENCE_H

#include "dynkv_types.h"

void admission_request_free(AdmissionRequest *);
int admission_request_parse(RouterIndex *, const uint8_t *, size_t, AdmissionRequest *, MatchScores *, uint64_t);
AdmissionSelection router_index_admission_select(RouterIndex *, const AdmissionRequest *, const MatchScores *, uint64_t);
bool admission_response_append(Buffer *, uint8_t, const Reservation *);
bool reservation_matches_request(const Reservation *, const AdmissionRequest *);
bool snapshot_append_worker(Buffer *, WorkerState *);
bool snapshot_append_worker_epoch(Buffer *, WorkerEpoch *);
bool snapshot_append_admission_rank(Buffer *, AdmissionRankState *);
bool snapshot_append_reservation(Buffer *, Reservation *);
bool snapshot_append_node(Buffer *, IndexNode *);
bool router_index_snapshot(RouterIndex *, Buffer *);
RouterIndex *router_index_from_snapshot(const uint8_t *, size_t);
void *router_index_rdb_load(ValkeyModuleIO *, int);
void router_index_rdb_save(ValkeyModuleIO *, void *);
void router_index_aof_rewrite(ValkeyModuleIO *, ValkeyModuleString *, void *);
size_t router_index_mem_usage(const void *);
RouterIndex *router_index_for_write(ValkeyModuleCtx *, ValkeyModuleString *, ValkeyModuleKey **);
RouterIndex *router_index_for_write_tracking_creation(
    ValkeyModuleCtx *,
    ValkeyModuleString *,
    ValkeyModuleKey **,
    bool *);
RouterIndex *router_index_for_read(ValkeyModuleCtx *, ValkeyModuleString *, ValkeyModuleKey **);

#endif /* DYNKV_PERSISTENCE_H */
