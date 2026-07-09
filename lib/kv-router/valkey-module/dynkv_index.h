/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYNKV_INDEX_H
#define DYNKV_INDEX_H

#include "dynkv_types.h"

bool router_event_header(const uint8_t *, size_t, EventHeader *);
bool router_event_remove_has_blocks(const uint8_t *, size_t);
uint64_t router_index_rank_generation(RouterIndex *, uint64_t, uint32_t);
bool router_index_generation_can_advance(const RouterIndex *);
bool router_index_worker_epochs_can_advance(const RouterIndex *, uint32_t);
bool router_index_advance_rank_generation(RouterIndex *, WorkerState *);
bool router_index_advance_worker_epoch(RouterIndex *, uint64_t);
bool router_index_advance_lifecycle_generation(RouterIndex *, WorkerEpoch *);
bool router_index_mark_legacy_rank(RouterIndex *, uint64_t, uint32_t);
bool router_index_mark_legacy_worker(RouterIndex *, uint64_t);
bool worker_registration_is_live(RouterIndex *, uint64_t, uint64_t);
bool worker_registration_owner_is_live(RouterIndex *, uint64_t, uint64_t, uint64_t);
bool router_index_worker_cleanup_pending(RouterIndex *, uint64_t);
bool router_index_worker_lease_can_end(RouterIndex *, uint64_t);
bool router_index_end_worker_lease(RouterIndex *, WorkerEpoch *);
bool router_index_can_advance_for_event(const RouterIndex *, const EventHeader *);
bool router_index_advance_for_event(RouterIndex *, const EventHeader *);
IndexNode *router_index_node_by_external(RouterIndex *, uint64_t);
IndexNode *router_index_child(RouterIndex *, uint64_t, uint64_t);
IndexNode *router_index_add_node(RouterIndex *, uint64_t, uint64_t, uint64_t);
void router_index_rebuild_child_counts(RouterIndex *);
Owner *node_owner(IndexNode *, WorkerState *);
bool worker_node_position(WorkerState *, IndexNode *, size_t *);
bool worker_record_node(WorkerState *, IndexNode *, size_t);
Owner *node_owner_create(IndexNode *, WorkerState *);
bool worker_replace_node_position(WorkerState *, IndexNode *, size_t, size_t);
bool worker_swap_node_positions(WorkerState *, size_t, size_t);
bool worker_rebuild_cleanup_node_partition(WorkerState *, uint64_t);
bool router_index_rebuild_lease_cleanup_state(RouterIndex *);
bool worker_forget_node(WorkerState *, IndexNode *);
bool node_remove_owner(IndexNode *, WorkerState *);
bool router_index_delete_ownerless_leaf(RouterIndex *, IndexNode *);
void router_index_prune_ownerless_ancestors(RouterIndex *, uint64_t);
void router_index_compact_direct_worker_owners(RouterIndex *, WorkerState *);
void router_index_compact_direct_worker_all_ranks(RouterIndex *, uint64_t);
void router_index_compact_direct_remove(RouterIndex *, WorkerState *, const uint8_t *, size_t);
bool node_set_owner(IndexNode *, WorkerState *, uint64_t, bool);
bool router_index_clear_worker(WorkerState *, uint64_t, bool);
bool router_index_deactivate_worker(WorkerState *);
bool router_index_reset_worker(WorkerState *);
bool router_index_clear_all_worker(RouterIndex *, uint64_t, uint32_t, uint64_t);
int router_index_validate_event(const uint8_t *, size_t);
int router_index_apply_store(RouterIndex *, Reader *, WorkerState *, uint64_t, uint32_t, uint64_t, bool *);
int router_index_apply_remove(RouterIndex *, Reader *, WorkerState *, uint64_t, bool *);
int router_index_apply_event(RouterIndex *, const uint8_t *, size_t);
void replace_plan_free_dict(ValkeyModuleDict *);
void replace_plan_free(ReplacePlan *);
bool replace_plan_init(ReplacePlan *);
ReplaceOwnerPlan *replace_plan_owner(ReplacePlan *, uint64_t);
bool replace_plan_set_owner(ReplacePlan *, uint64_t, uint64_t, bool);
bool replace_plan_validate_node(RouterIndex *, ReplacePlan *, uint64_t, uint64_t, uint64_t);
bool replace_snapshot_validate_store(RouterIndex *, ReplacePlan *, const uint8_t *, size_t, uint64_t, uint32_t);
bool replace_snapshot_validate(RouterIndex *, const uint8_t *, size_t, uint64_t, uint32_t);
int replace_snapshot_apply(RouterIndex *, const uint8_t *, size_t);
MatchScore *match_scores_find(const MatchScores *, WorkerState *);
MatchScore *match_scores_add(MatchScores *, WorkerState *);
void match_scores_free(MatchScores *);
int router_index_collect_matches_from_reader(RouterIndex *, Reader *, uint32_t, MatchScores *, uint64_t);
bool router_match_request_valid(const uint8_t *, size_t);
int router_index_collect_matches(RouterIndex *, const uint8_t *, size_t, MatchScores *, uint64_t);
int router_index_match(RouterIndex *, const uint8_t *, size_t, Buffer *, uint64_t);
int router_index_select(RouterIndex *, const uint8_t *, size_t, Buffer *, uint64_t);

#endif /* DYNKV_INDEX_H */
