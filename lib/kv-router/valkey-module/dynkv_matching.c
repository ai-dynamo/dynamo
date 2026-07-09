/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_index.h"
#include "dynkv_state.h"

void replace_plan_free_dict(ValkeyModuleDict *dict) {
    if (dict == NULL) {
        return;
    }
    ValkeyModuleDictIter *iter = ValkeyModule_DictIteratorStartC(dict, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(iter, NULL, &data) != NULL) {
        ValkeyModule_Free(data);
    }
    ValkeyModule_DictIteratorStop(iter);
    ValkeyModule_FreeDict(NULL, dict);
}

void replace_plan_free(ReplacePlan *plan) {
    replace_plan_free_dict(plan->nodes_by_external);
    /* nodes_by_edge aliases values in nodes_by_external. */
    if (plan->nodes_by_edge != NULL) {
        ValkeyModule_FreeDict(NULL, plan->nodes_by_edge);
    }
    replace_plan_free_dict(plan->owners_by_external);
    memset(plan, 0, sizeof(*plan));
}

bool replace_plan_init(ReplacePlan *plan) {
    plan->nodes_by_external = ValkeyModule_CreateDict(NULL);
    plan->nodes_by_edge = ValkeyModule_CreateDict(NULL);
    plan->owners_by_external = ValkeyModule_CreateDict(NULL);
    if (plan->nodes_by_external == NULL || plan->nodes_by_edge == NULL ||
        plan->owners_by_external == NULL) {
        replace_plan_free(plan);
        return false;
    }
    return true;
}

ReplaceOwnerPlan *replace_plan_owner(ReplacePlan *plan, uint64_t external_hash) {
    uint8_t key[8];
    encode_u64_be(key, external_hash);
    return ValkeyModule_DictGetC(plan->owners_by_external, key, sizeof(key), NULL);
}

bool replace_plan_set_owner(
    ReplacePlan *plan,
    uint64_t external_hash,
    uint64_t event_id,
    bool active) {
    uint8_t key[8];
    encode_u64_be(key, external_hash);
    ReplaceOwnerPlan *owner = ValkeyModule_DictGetC(plan->owners_by_external, key, sizeof(key), NULL);
    if (owner == NULL) {
        owner = ValkeyModule_Calloc(1, sizeof(*owner));
        owner->event_id = event_id;
        owner->active = active;
        ValkeyModule_DictSetC(plan->owners_by_external, key, sizeof(key), owner);
        return true;
    }
    if (event_id < owner->event_id || (event_id == owner->event_id && owner->active == active)) {
        return true;
    }
    owner->event_id = event_id;
    owner->active = active;
    return true;
}

bool replace_plan_validate_node(
    RouterIndex *index,
    ReplacePlan *plan,
    uint64_t external_hash,
    uint64_t parent_hash,
    uint64_t local_hash) {
    uint8_t external_key[8];
    uint8_t edge_key[16];
    encode_u64_be(external_key, external_hash);
    child_key(edge_key, parent_hash, local_hash);

    IndexNode *existing = router_index_node_by_external(index, external_hash);
    if (existing != NULL &&
        (existing->parent_external_hash != parent_hash || existing->local_hash != local_hash)) {
        return false;
    }
    IndexNode *existing_edge = router_index_child(index, parent_hash, local_hash);
    if (existing_edge != NULL && existing_edge->external_hash != external_hash) {
        return false;
    }

    ReplaceNodePlan *planned =
        ValkeyModule_DictGetC(plan->nodes_by_external, external_key, sizeof(external_key), NULL);
    if (planned != NULL) {
        return planned->parent_hash == parent_hash && planned->local_hash == local_hash;
    }
    ReplaceNodePlan *planned_edge =
        ValkeyModule_DictGetC(plan->nodes_by_edge, edge_key, sizeof(edge_key), NULL);
    if (planned_edge != NULL && planned_edge->external_hash != external_hash) {
        return false;
    }

    planned = ValkeyModule_Alloc(sizeof(*planned));
    planned->external_hash = external_hash;
    planned->parent_hash = parent_hash;
    planned->local_hash = local_hash;
    ValkeyModule_DictSetC(plan->nodes_by_external, external_key, sizeof(external_key), planned);
    ValkeyModule_DictSetC(plan->nodes_by_edge, edge_key, sizeof(edge_key), planned);
    return true;
}

bool replace_snapshot_validate_store(
    RouterIndex *index,
    ReplacePlan *plan,
    const uint8_t *data,
    size_t length,
    uint64_t expected_worker_id,
    uint32_t expected_dp_rank) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint8_t kind = 0;
    uint64_t worker_id = 0;
    uint32_t dp_rank = 0;
    uint64_t event_id = 0;
    uint64_t parent_hash = 0;
    uint32_t count = 0;
    if (!reader_u8(&reader, &version) || !reader_u8(&reader, &kind) ||
        !reader_u64(&reader, &worker_id) || !reader_u32(&reader, &dp_rank) ||
        !reader_u64(&reader, &event_id) || !reader_u64(&reader, &parent_hash) ||
        !reader_u32(&reader, &count) || version != DYNKV_WIRE_VERSION ||
        kind != DYNKV_EVENT_STORE || worker_id != expected_worker_id ||
        dp_rank != expected_dp_rank || count > DYNKV_MAX_EVENT_BLOCKS ||
        reader.length - reader.offset != (size_t)count * 16) {
        return false;
    }
    if (parent_hash != DYNKV_ROOT_PARENT) {
        ReplaceOwnerPlan *parent = replace_plan_owner(plan, parent_hash);
        if (parent == NULL || !parent->active) {
            return false;
        }
    }

    uint64_t current_parent = parent_hash;
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t external_hash = 0;
        uint64_t local_hash = 0;
        if (!reader_u64(&reader, &external_hash) || !reader_u64(&reader, &local_hash) ||
            !replace_plan_validate_node(index, plan, external_hash, current_parent, local_hash) ||
            !replace_plan_set_owner(plan, external_hash, event_id, true)) {
            return false;
        }
        current_parent = external_hash;
    }
    return reader.offset == reader.length;
}

bool replace_snapshot_validate(
    RouterIndex *index,
    const uint8_t *data,
    size_t length,
    uint64_t expected_worker_id,
    uint32_t expected_dp_rank) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint32_t count = 0;
    if (!reader_u8(&reader, &version) || !reader_u32(&reader, &count) ||
        version != DYNKV_WIRE_VERSION || count > DYNKV_MAX_REPLACE_EVENTS) {
        return false;
    }

    ReplacePlan plan = {0};
    if (!replace_plan_init(&plan)) {
        return false;
    }
    bool valid = true;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t event_length = 0;
        if (!reader_u32(&reader, &event_length) || event_length == 0 ||
            reader.length - reader.offset < event_length) {
            valid = false;
            break;
        }
        const uint8_t *event = reader.data + reader.offset;
        reader.offset += event_length;
        if (router_index_validate_event(event, event_length) != VALKEYMODULE_OK ||
            !replace_snapshot_validate_store(
                index, &plan, event, event_length, expected_worker_id, expected_dp_rank)) {
            valid = false;
            break;
        }
    }
    valid &= reader.offset == reader.length;
    replace_plan_free(&plan);
    return valid;
}

int replace_snapshot_apply(
    RouterIndex *index,
    const uint8_t *data,
    size_t length) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint32_t count = 0;
    if (!reader_u8(&reader, &version) || !reader_u32(&reader, &count) ||
        version != DYNKV_WIRE_VERSION || count > DYNKV_MAX_REPLACE_EVENTS) {
        return VALKEYMODULE_ERR;
    }
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t event_length = 0;
        if (!reader_u32(&reader, &event_length) || event_length == 0 ||
            reader.length - reader.offset < event_length) {
            return VALKEYMODULE_ERR;
        }
        int result = router_index_apply_event(index, reader.data + reader.offset, event_length);
        if (result != VALKEYMODULE_OK && result != DYNKV_NOOP) {
            return VALKEYMODULE_ERR;
        }
        reader.offset += event_length;
    }
    return reader.offset == reader.length ? VALKEYMODULE_OK : VALKEYMODULE_ERR;
}

MatchScore *match_scores_find(const MatchScores *scores, WorkerState *worker) {
    if (scores->positions == NULL) {
        return NULL;
    }
    uint8_t key[12];
    worker_key(key, worker->worker_id, worker->dp_rank);
    void *encoded = ValkeyModule_DictGetC(scores->positions, key, sizeof(key), NULL);
    return encoded == NULL ? NULL : &scores->scores[(size_t)(uintptr_t)encoded - 1];
}

MatchScore *match_scores_add(MatchScores *scores, WorkerState *worker) {
    if (scores->positions == NULL) {
        scores->positions = ValkeyModule_CreateDict(NULL);
        if (scores->positions == NULL) {
            return NULL;
        }
    }
    MatchScore *score = match_scores_find(scores, worker);
    if (score != NULL) {
        return score;
    }
    if (scores->count == scores->capacity) {
        size_t capacity = scores->capacity == 0 ? 8 : scores->capacity * 2;
        scores->scores = ValkeyModule_Realloc(scores->scores, capacity * sizeof(*scores->scores));
        scores->capacity = capacity;
    }
    size_t position = scores->count++;
    score = &scores->scores[position];
    score->worker = worker;
    score->matched_blocks = 0;
    score->last_external_hash = DYNKV_ROOT_PARENT;
    uint8_t key[12];
    worker_key(key, worker->worker_id, worker->dp_rank);
    if (ValkeyModule_DictSetC(
            scores->positions,
            key,
            sizeof(key),
            (void *)(uintptr_t)(position + 1)) != VALKEYMODULE_OK) {
        --scores->count;
        return NULL;
    }
    return score;
}

void match_scores_free(MatchScores *scores) {
    if (scores->positions != NULL) {
        ValkeyModule_FreeDict(NULL, scores->positions);
    }
    ValkeyModule_Free(scores->scores);
    scores->positions = NULL;
    scores->scores = NULL;
    scores->count = 0;
    scores->capacity = 0;
}

int router_index_collect_matches_from_reader(
    RouterIndex *index,
    Reader *reader,
    uint32_t count,
    MatchScores *scores,
    uint64_t now_ms) {
    uint64_t parent_hash = DYNKV_ROOT_PARENT;
    bool prefix_present = true;
    for (uint32_t depth = 0; depth < count; ++depth) {
        uint64_t local_hash = 0;
        if (!reader_u64(reader, &local_hash)) {
            return VALKEYMODULE_ERR;
        }
        if (!prefix_present) {
            continue;
        }
        IndexNode *node = router_index_child(index, parent_hash, local_hash);
        if (node == NULL) {
            prefix_present = false;
            continue;
        }
        for (size_t owner_idx = 0; owner_idx < node->owner_count; ++owner_idx) {
            Owner *owner = &node->owners[owner_idx];
            if (!owner->active ||
                !worker_registration_is_live(index, owner->worker->worker_id, now_ms)) {
                continue;
            }
            MatchScore *score = match_scores_find(scores, owner->worker);
            if (depth == 0) {
                score = score == NULL ? match_scores_add(scores, owner->worker) : score;
                if (score == NULL) {
                    return VALKEYMODULE_ERR;
                }
                score->matched_blocks = 1;
                score->last_external_hash = node->external_hash;
            } else if (score != NULL && score->matched_blocks == depth) {
                ++score->matched_blocks;
                score->last_external_hash = node->external_hash;
            }
        }
        parent_hash = node->external_hash;
    }

    return VALKEYMODULE_OK;
}

static bool match_request_reader(
    const uint8_t *data,
    size_t length,
    Reader *reader,
    uint32_t *count) {
    *reader = (Reader){.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    return reader_u8(reader, &version) && reader_u32(reader, count) &&
           version == DYNKV_WIRE_VERSION && *count <= DYNKV_MAX_MATCH_HASHES &&
           reader->length - reader->offset == (size_t)*count * sizeof(uint64_t);
}

bool router_match_request_valid(const uint8_t *data, size_t length) {
    Reader reader = {0};
    uint32_t count = 0;
    return match_request_reader(data, length, &reader, &count);
}

int router_index_collect_matches(
    RouterIndex *index,
    const uint8_t *data,
    size_t length,
    MatchScores *scores,
    uint64_t now_ms) {
    Reader reader = {0};
    uint32_t count = 0;
    if (!match_request_reader(data, length, &reader, &count)) {
        return VALKEYMODULE_ERR;
    }
    return router_index_collect_matches_from_reader(index, &reader, count, scores, now_ms);
}

int router_index_match(
    RouterIndex *index,
    const uint8_t *data,
    size_t length,
    Buffer *response,
    uint64_t now_ms) {
    MatchScores scores = {0};
    if (router_index_collect_matches(index, data, length, &scores, now_ms) != VALKEYMODULE_OK) {
        match_scores_free(&scores);
        return VALKEYMODULE_ERR;
    }

    if (!buffer_u8(response, DYNKV_WIRE_VERSION) ||
        !buffer_u32(response, (uint32_t)scores.count)) {
        match_scores_free(&scores);
        return VALKEYMODULE_ERR;
    }
    for (size_t i = 0; i < scores.count; ++i) {
        MatchScore *score = &scores.scores[i];
        if (!buffer_u64(response, score->worker->worker_id) ||
            !buffer_u32(response, score->worker->dp_rank) ||
            !buffer_u32(response, score->matched_blocks) ||
            !buffer_u64(response, score->last_external_hash)) {
            match_scores_free(&scores);
            return VALKEYMODULE_ERR;
        }
    }
    match_scores_free(&scores);
    return VALKEYMODULE_OK;
}

/*
 * Select payload:
 *   u8 version, u32 local-hash-count, local-hash-count * u64 local hashes,
 *   u32 candidate-count, candidate-count * (u64 worker, u32 rank, u64 load).
 *
 * Candidate membership is supplied by the frontend after it applies model,
 * affinity, and topology constraints. The module deterministically ranks the
 * remaining candidates by longest device-prefix match, then lowest supplied
 * load, then worker/rank. Load is an input in this first stateless selector;
 * authoritative global reservations are a separate command family.
 */
int router_index_select(
    RouterIndex *index,
    const uint8_t *data,
    size_t length,
    Buffer *response,
    uint64_t now_ms) {
    if (length < 5) {
        return VALKEYMODULE_ERR;
    }
    Reader header = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint32_t hash_count = 0;
    if (!reader_u8(&header, &version) || !reader_u32(&header, &hash_count) ||
        version != DYNKV_WIRE_VERSION || hash_count > DYNKV_MAX_MATCH_HASHES ||
        header.length - header.offset < (size_t)hash_count * sizeof(uint64_t)) {
        return VALKEYMODULE_ERR;
    }
    size_t match_length = header.offset + (size_t)hash_count * sizeof(uint64_t);
    Reader candidates = {.data = data + match_length, .length = length - match_length, .offset = 0};
    uint32_t candidate_count = 0;
    if (!reader_u32(&candidates, &candidate_count) ||
        candidate_count > DYNKV_MAX_SELECT_CANDIDATES ||
        candidates.length - candidates.offset != (size_t)candidate_count * 20) {
        return VALKEYMODULE_ERR;
    }

    MatchScores scores = {0};
    if (router_index_collect_matches(index, data, match_length, &scores, now_ms) !=
        VALKEYMODULE_OK) {
        match_scores_free(&scores);
        return VALKEYMODULE_ERR;
    }

    bool selected = false;
    uint64_t selected_worker = 0;
    uint32_t selected_rank = 0;
    uint64_t selected_load = 0;
    uint32_t selected_matched_blocks = 0;
    uint64_t selected_last_hash = DYNKV_ROOT_PARENT;
    for (uint32_t i = 0; i < candidate_count; ++i) {
        uint64_t worker_id = 0;
        uint32_t dp_rank = 0;
        uint64_t load = 0;
        if (!reader_u64(&candidates, &worker_id) || !reader_u32(&candidates, &dp_rank) ||
            !reader_u64(&candidates, &load)) {
            match_scores_free(&scores);
            return VALKEYMODULE_ERR;
        }
        if (!worker_registration_is_live(index, worker_id, now_ms)) {
            continue;
        }
        WorkerState *worker = router_index_worker(index, worker_id, dp_rank, false);
        MatchScore *score = worker == NULL ? NULL : match_scores_find(&scores, worker);
        uint32_t matched_blocks = score == NULL ? 0 : score->matched_blocks;
        uint64_t last_hash = score == NULL ? DYNKV_ROOT_PARENT : score->last_external_hash;
        bool better = !selected || matched_blocks > selected_matched_blocks ||
                      (matched_blocks == selected_matched_blocks && load < selected_load) ||
                      (matched_blocks == selected_matched_blocks && load == selected_load &&
                       (worker_id < selected_worker ||
                        (worker_id == selected_worker && dp_rank < selected_rank)));
        if (better) {
            selected = true;
            selected_worker = worker_id;
            selected_rank = dp_rank;
            selected_load = load;
            selected_matched_blocks = matched_blocks;
            selected_last_hash = last_hash;
        }
    }
    match_scores_free(&scores);

    if (!buffer_u8(response, DYNKV_WIRE_VERSION) ||
        !buffer_u8(response, selected ? 1 : 0) ||
        (selected &&
         (!buffer_u64(response, selected_worker) || !buffer_u32(response, selected_rank) ||
          !buffer_u32(response, selected_matched_blocks) ||
          !buffer_u64(response, selected_last_hash)))) {
        return VALKEYMODULE_ERR;
    }
    return VALKEYMODULE_OK;
}
