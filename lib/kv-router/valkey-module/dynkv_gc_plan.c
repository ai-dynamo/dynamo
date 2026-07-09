/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_gc.h"
#include "dynkv_index.h"
#include "dynkv_state.h"

bool gc_plan_pending_lease_cleanup(
    WorkerEpoch *epoch,
    uint64_t budget,
    Buffer *payload,
    GcResult *result) {
    uint64_t available = budget - (result->examined - 1);
    uint64_t planned = 0;
    if (epoch->reservation_count != 0) {
        uint64_t max_items = available / DYNKV_GC_HEAP_WORK_COST;
        size_t count = epoch->reservation_count;
        if ((uint64_t)count > max_items) {
            count = (size_t)max_items;
        }
        for (size_t i = 0; i < count; ++i) {
            Reservation *reservation =
                epoch->reservation_states[epoch->reservation_count - 1 - i];
            if (!gc_append_cleanup_reservation(payload, epoch, reservation)) {
                return false;
            }
        }
        planned = count;
        result->examined += planned * DYNKV_GC_HEAP_WORK_COST -
                            (planned == 0 ? 0 : 1);
        result->reclaimed += planned;
        return true;
    }
    if (epoch->lease_cleanup_admission_remaining != 0) {
        size_t count = epoch->lease_cleanup_admission_remaining;
        if ((uint64_t)count > available) {
            count = (size_t)available;
        }
        for (size_t i = 0; i < count; ++i) {
            AdmissionRankState *rank = epoch->admission_rank_states[
                epoch->lease_cleanup_admission_remaining - 1 - i];
            if (!gc_append_cleanup_admission_rank(payload, epoch, rank)) {
                return false;
            }
        }
        result->examined += count - (count == 0 ? 0 : 1);
        result->reclaimed += count;
        result->admission_ranks += count;
        return true;
    }
    if (epoch->lease_cleanup_worker_remaining != 0) {
        WorkerState *worker = epoch->worker_states[
            epoch->lease_cleanup_worker_remaining - 1];
        worker_prepare_cleanup_nodes(worker, epoch->lease_cleanup_generation);
        if (worker->lease_cleanup_node_remaining != 0) {
            size_t count = worker->lease_cleanup_node_remaining;
            if ((uint64_t)count > available) {
                count = (size_t)available;
            }
            for (size_t i = 0; i < count; ++i) {
                IndexNode *node = worker->nodes[
                    worker->lease_cleanup_node_remaining - 1 - i];
                Owner *owner = node_owner(node, worker);
                if (owner == NULL ||
                    !gc_append_cleanup_owner(
                        payload, epoch, worker, node, owner)) {
                    return false;
                }
            }
            result->examined += count - (count == 0 ? 0 : 1);
            result->reclaimed += count;
            result->owners += count;
            return true;
        }
        if (available != 0 &&
            !gc_append_cleanup_worker(payload, epoch, worker)) {
            return false;
        }
        if (available != 0) {
            ++result->reclaimed;
            ++result->workers;
        }
        return true;
    }
    if (available != 0 && !gc_append_finalize_lease_cleanup(payload, epoch)) {
        return false;
    }
    if (available != 0) {
        ++result->reclaimed;
        ++result->worker_epochs;
    }
    return true;
}

bool gc_plan_record(
    RouterIndex *index,
    uint8_t phase,
    void *data,
    uint64_t watermark,
    uint64_t now_ms,
    uint64_t budget,
    Buffer *payload,
    GcResult *result) {
    switch (phase) {
        case DYNKV_GC_PHASE_WORKERS: {
            WorkerState *worker = data;
            if (!gc_worker_is_reclaimable(index, worker, watermark)) {
                return true;
            }
            if (worker->node_count != 0) {
                IndexNode *node = worker->nodes[worker->node_count - 1];
                Owner *owner = node_owner(node, worker);
                if (owner == NULL || owner->active) {
                    return true;
                }
                if (!gc_append_remove_owner(payload, node, worker)) {
                    return false;
                }
                ++result->reclaimed;
                ++result->owners;
                return true;
            }
            if (worker->admission_rank_count == 0 && worker->reservation_count == 0) {
                if (!gc_append_remove_worker(payload, worker)) {
                    return false;
                }
                ++result->reclaimed;
                ++result->workers;
            }
            return true;
        }
        case DYNKV_GC_PHASE_ADMISSION_RANKS: {
            AdmissionRankState *rank = data;
            WorkerState *worker = router_index_worker(
                index, rank->worker_id, rank->dp_rank, false);
            if (worker == NULL || !gc_worker_is_reclaimable(index, worker, watermark) ||
                rank->active_reservations != 0) {
                return true;
            }
            if (!gc_append_remove_admission_rank(payload, rank, worker)) {
                return false;
            }
            ++result->reclaimed;
            ++result->admission_ranks;
            return true;
        }
        case DYNKV_GC_PHASE_NODES: {
            IndexNode *node = data;
            if (node->owner_count != 0 || node->child_count != 0) {
                return true;
            }
            if (!gc_append_remove_node(payload, node)) {
                return false;
            }
            ++result->reclaimed;
            ++result->nodes;
            return true;
        }
        case DYNKV_GC_PHASE_WORKER_EPOCHS: {
            WorkerEpoch *epoch = data;
            if (epoch->registration_owner_set) {
                if (epoch->lease_cleanup_pending) {
                    return gc_plan_pending_lease_cleanup(
                        epoch, budget, payload, result);
                }
                if (epoch->registration_expires_at_ms <= now_ms) {
                    /* BEGIN is the only counter-advancing op in this plan. */
                    uint64_t available = budget - (result->examined - 1);
                    if (available < DYNKV_GC_HEAP_WORK_COST ||
                        index->generation_counter == UINT64_MAX ||
                        index->lifecycle_generation_counter == UINT64_MAX) {
                        return true;
                    }
                    if (!gc_append_begin_lease_cleanup(
                            payload, index, epoch, now_ms)) {
                        return false;
                    }
                    result->examined += DYNKV_GC_HEAP_WORK_COST - 1;
                    ++result->reclaimed;
                    ++result->worker_epochs;
                    ++result->lease_expiries;
                    result->stop_planning = true;
                }
                return true;
            }
            if (!epoch->lifecycle_managed || epoch->legacy_tainted ||
                epoch->admission_retired_all ||
                epoch->generation == 0 || epoch->generation > watermark ||
                epoch->worker_state_count != 0 || epoch->admission_rank_count != 0 ||
                epoch->reservation_count != 0) {
                return true;
            }
            if (result->epoch_deletions >=
                UINT64_MAX - index->generation_counter) {
                return true;
            }
            if (!gc_append_remove_worker_epoch(payload, epoch)) {
                return false;
            }
            ++result->reclaimed;
            ++result->worker_epochs;
            ++result->epoch_deletions;
            result->stop_planning = true;
            return true;
        }
        default:
            return false;
    }
}
