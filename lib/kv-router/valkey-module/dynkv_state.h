/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYNKV_STATE_H
#define DYNKV_STATE_H

#include "dynkv_types.h"

/* Internal cross-translation-unit API. LTO restores whole-module inlining. */
void encode_u32_be(uint8_t[4], uint32_t);
void encode_u64_be(uint8_t[8], uint64_t);
void worker_key(uint8_t[12], uint64_t, uint32_t);
bool reservation_key(uint8_t *, size_t, const uint8_t *, uint32_t, uint64_t, uint64_t, size_t *);
bool admission_rank_key(uint8_t *, size_t, const uint8_t *, uint32_t, uint64_t, uint32_t, size_t *);
void child_key(uint8_t[16], uint64_t, uint64_t);
bool reader_u8(Reader *, uint8_t *);
bool reader_u32(Reader *, uint32_t *);
bool reader_u64(Reader *, uint64_t *);
bool buffer_reserve(Buffer *, size_t);
bool buffer_u8(Buffer *, uint8_t);
bool buffer_u32(Buffer *, uint32_t);
bool buffer_u64(Buffer *, uint64_t);
bool buffer_bytes(Buffer *, const uint8_t *, size_t);
void buffer_free(Buffer *);
RouterIndex *router_index_create(void);
WorkerEpoch *router_index_worker_epoch_lookup(RouterIndex *, uint64_t);
void index_node_free(IndexNode *);
void worker_state_free(WorkerState *);
void worker_epoch_free(WorkerEpoch *);
void admission_rank_state_free(AdmissionRankState *);
void reservation_free(Reservation *);
bool reservation_expires_before(const Reservation *, const Reservation *);
void router_index_reservation_expiry_heap_swap(RouterIndex *, size_t, size_t);
void router_index_reservation_expiry_heap_sift_up(RouterIndex *, size_t);
void router_index_reservation_expiry_heap_sift_down(RouterIndex *, size_t);
bool router_index_reservation_expiry_heap_reserve(RouterIndex *, size_t);
bool router_index_reservation_expiry_heap_insert(RouterIndex *, Reservation *);
bool router_index_reservation_expiry_heap_remove(RouterIndex *, Reservation *);
bool router_index_reservation_expiry_heap_reposition(RouterIndex *, Reservation *);
bool worker_lease_expires_before(const WorkerEpoch *, const WorkerEpoch *);
void router_index_worker_lease_expiry_heap_swap(RouterIndex *, size_t, size_t);
void router_index_worker_lease_expiry_heap_sift_up(RouterIndex *, size_t);
void router_index_worker_lease_expiry_heap_sift_down(RouterIndex *, size_t);
bool router_index_worker_lease_expiry_heap_reserve(RouterIndex *, size_t);
bool router_index_worker_lease_expiry_heap_insert(RouterIndex *, WorkerEpoch *);
bool router_index_worker_lease_expiry_heap_remove(RouterIndex *, WorkerEpoch *);
bool router_index_worker_lease_expiry_heap_reposition(RouterIndex *, WorkerEpoch *);
void router_index_free(void *);
void worker_epoch_add_worker_state(WorkerEpoch *, WorkerState *);
void worker_epoch_remove_worker_state(WorkerEpoch *, WorkerState *);
void worker_epoch_swap_worker_positions(WorkerEpoch *, size_t, size_t);
void worker_epoch_add_admission_rank(WorkerEpoch *, AdmissionRankState *);
void worker_epoch_remove_admission_rank(WorkerEpoch *, AdmissionRankState *);
void worker_epoch_swap_admission_positions(WorkerEpoch *, size_t, size_t);
void worker_epoch_add_reservation(WorkerEpoch *, Reservation *);
void worker_epoch_remove_reservation(WorkerEpoch *, Reservation *);
WorkerState *router_index_worker(RouterIndex *, uint64_t, uint32_t, bool);
WorkerEpoch *router_index_worker_epoch(RouterIndex *, uint64_t, bool);
AdmissionRankState *router_index_admission_rank(RouterIndex *, const uint8_t *, uint32_t, uint64_t, uint32_t, bool);
bool admission_rank_capacity_matches(const AdmissionRankState *, uint32_t);
bool admission_rank_configure(AdmissionRankState *, uint32_t);
bool admission_rank_advance_incarnation(AdmissionRankState *);
bool router_index_admission_ranks_can_advance(RouterIndex *, uint64_t, bool, uint32_t);
void router_index_advance_admission_rank_incarnations(RouterIndex *, uint64_t, bool, uint32_t);
bool router_index_restore_legacy_admission_rank(RouterIndex *, const uint8_t *, uint32_t, uint64_t, uint32_t, uint64_t);
Reservation *router_index_reservation(RouterIndex *, const uint8_t *, uint32_t, uint64_t, uint64_t);
bool router_index_add_reservation(RouterIndex *, const uint8_t *, uint32_t, uint64_t, uint64_t, uint64_t, uint32_t, uint64_t, uint64_t, uint32_t, uint32_t, const uint8_t *, uint32_t);
void router_index_delete_admission_rank(RouterIndex *, AdmissionRankState *);
bool router_index_remove_reservation(RouterIndex *, Reservation *);
size_t router_index_cleanup_expired_reservations(
    RouterIndex *, uint64_t, size_t);
size_t router_index_revoke_worker_reservations(RouterIndex *, uint64_t, bool, uint32_t);

#endif /* DYNKV_STATE_H */
