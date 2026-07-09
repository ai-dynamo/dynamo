/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYNKV_TYPES_H
#define DYNKV_TYPES_H

/*
 * Persistent Valkey index for Dynamo KV routing.
 *
 * The module stores one index under each Dynamo routing-scope key. A MATCH
 * traverses that index in one Valkey round trip.
 * Values are native module types and are therefore covered by RDB snapshots;
 * command replication and the AOF rewrite callback preserve the same state for
 * append-only deployments.
 */

#include "valkeymodule.h"
#include "dynkv_limits.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DYNKV_WIRE_VERSION 1
#define DYNKV_EVENT_STORE 1
#define DYNKV_EVENT_REMOVE 2
#define DYNKV_EVENT_CLEAR 3
#define DYNKV_ROOT_PARENT UINT64_MAX
#define DYNKV_SNAPSHOT_VERSION 14
#define DYNKV_NOOP 4
#define DYNKV_MAX_REPLACE_EVENTS 1048576
#define DYNKV_MAX_EVENT_BLOCKS 1048576
#define DYNKV_ADMISSION_WIRE_VERSION 1
#define DYNKV_ADMISSION_RESERVE 1
#define DYNKV_ADMISSION_RELEASE 2
#define DYNKV_ADMISSION_RENEW 3
#define DYNKV_ADMISSION_CLEANUP 4
#define DYNKV_ADMISSION_NO_CAPACITY 0
#define DYNKV_ADMISSION_RESERVED 1
#define DYNKV_ADMISSION_CONFLICT 5
#define DYNKV_ADMISSION_EXPIRED 6
#define DYNKV_ADMISSION_INVALID 7
/* One admission mutation performs at most one such cleanup pass. */
#define DYNKV_ADMISSION_EXPIRY_CLEANUP_BUDGET 64
#define DYNKV_REGISTRATION_WIRE_VERSION 1
#define DYNKV_LEASED_REGISTRATION_WIRE_VERSION_LEGACY 2
#define DYNKV_LEASED_REGISTRATION_WIRE_VERSION 3
#define DYNKV_WORKER_LEASE_CONTROL_VERSION 1
#define DYNKV_WORKER_LEASE_APPLY_VERSION_LEGACY 1
#define DYNKV_WORKER_LEASE_APPLY_VERSION 2
#define DYNKV_WORKER_LEASE_REGISTER 1
#define DYNKV_WORKER_LEASE_RENEW 2
#define DYNKV_WORKER_LEASE_UNREGISTER 3
#define DYNKV_WORKER_LEASE_EXPIRE 4
#define DYNKV_MAX_WORKER_LEASE_MS 600000
#define DYNKV_INLINE_EXPIRY_WORK_BUDGET 256
#define DYNKV_WORKER_LEASE_STALE 8
#define DYNKV_WORKER_LEASE_OWNED 9
#define DYNKV_WORKER_LEASE_INVALID 10
#define DYNKV_GC_WIRE_VERSION_LEGACY 1
#define DYNKV_GC_WIRE_VERSION 2
#define DYNKV_GC_REMOVE_OWNER 1
#define DYNKV_GC_REMOVE_ADMISSION_RANK 2
#define DYNKV_GC_REMOVE_NODE 3
#define DYNKV_GC_REMOVE_WORKER 4
#define DYNKV_GC_REMOVE_WORKER_EPOCH 5
#define DYNKV_GC_EXPIRE_WORKER_LEASE 6
#define DYNKV_GC_BEGIN_LEASE_CLEANUP 7
#define DYNKV_GC_CLEANUP_RESERVATION 8
#define DYNKV_GC_CLEANUP_ADMISSION_RANK 9
#define DYNKV_GC_CLEANUP_OWNER 10
#define DYNKV_GC_CLEANUP_WORKER 11
#define DYNKV_GC_FINALIZE_LEASE_CLEANUP 12
#define DYNKV_GC_PHASE_WORKERS 0
#define DYNKV_GC_PHASE_ADMISSION_RANKS 1
#define DYNKV_GC_PHASE_NODES 2
#define DYNKV_GC_PHASE_WORKER_EPOCHS 3
#define DYNKV_GC_PHASE_COUNT 4
#define DYNKV_MAX_GC_ITEMS 1048576
/* Budget units are object transitions; each heap mutation is one transition. */
#define DYNKV_GC_HEAP_WORK_COST 1
#define DYNKV_MAX_GC_CURSOR_BYTES (4 + DYNKV_MAX_ADMISSION_DOMAIN_LENGTH + 12)
/* Reserved while loading v6 state whose active leases lacked a capacity. */
#define DYNKV_UNKNOWN_ADMISSION_CAPACITY UINT32_MAX

typedef struct WorkerState WorkerState;
typedef struct IndexNode IndexNode;
typedef struct WorkerEpoch WorkerEpoch;
typedef struct Reservation Reservation;
typedef struct AdmissionRankState AdmissionRankState;

typedef struct {
    WorkerState *worker;
    uint64_t event_id;
    bool active;
    /* Persistent semantic marker; runtime array order is never a cursor. */
    uint64_t lease_cleanup_generation;
} Owner;

struct IndexNode {
    uint64_t external_hash;
    uint64_t parent_external_hash;
    uint64_t local_hash;
    Owner *owners;
    size_t owner_count;
    size_t owner_capacity;
    /* Runtime-derived topology reference count; snapshots persist the edges. */
    size_t child_count;
};

struct WorkerState {
    uint64_t worker_id;
    uint32_t dp_rank;
    /* Resettable event-ordering barrier for this rank. */
    uint64_t last_clear_event_id;
    /* Persistent duplicate-CLEAR barrier; recovery resets must not erase it. */
    uint64_t last_clear_dedupe_event_id;
    bool has_clear_dedupe_event_id;
    /*
     * Monotonic per-rank mutation fence.  A worker tree dump captures this
     * value before it is fetched, then can replace the rank only if no live
     * GPU event changed the rank in the meantime.
     */
    uint64_t mutation_generation;
    bool retired;
    /* Explicit admission eligibility; STORE alone must not make a rank routable. */
    bool admission_registered;
    /* Retained after an owner lease ends; see DYNKV.LIFECYCLE_STATS. */
    bool lifecycle_tombstone;
    /* Only leased, owner-fenced ranks are automatically reclaimable. */
    bool lifecycle_managed;
    /* Legacy APPLY/recovery can arrive without an owner nonce, so retain it. */
    bool legacy_tainted;
    /* Generation installed when the owner lease stopped admitting this rank. */
    uint64_t lifecycle_tombstone_generation;
    /* Last chunked lease cleanup that completed this rank's scalar state. */
    uint64_t lease_cleanup_generation;
    /* Runtime-derived references used to make rank reclamation O(1). */
    size_t admission_rank_count;
    size_t reservation_count;
    WorkerEpoch *epoch;
    size_t worker_epoch_index;
    /* Version 7-9 compatibility fields; AdmissionRankState is authoritative. */
    uint32_t admission_capacity;
    bool admission_capacity_set;
    uint64_t admission_incarnation;
    ValkeyModuleDict *node_members;
    IndexNode **nodes;
    size_t *node_owner_indices;
    size_t node_count;
    size_t node_capacity;
    /* Runtime-derived pending-node partition for the active cleanup. */
    uint64_t lease_cleanup_node_runtime_generation;
    size_t lease_cleanup_node_remaining;
};

/* A worker-wide fence covers ranks that have not yet reached this index. */
struct WorkerEpoch {
    uint64_t worker_id;
    uint64_t generation;
    /* Owner-control CAS token, independent of the high-rate event counter. */
    uint64_t lifecycle_generation;
    bool admission_retired_all;
    bool registration_owner_set;
    uint64_t registration_owner_nonce;
    uint64_t registration_expires_at_ms;
    bool lifecycle_managed;
    bool legacy_tainted;
    bool last_registration_expected_set;
    uint64_t last_registration_expected_generation;
    /*
     * An expired lease is fenced first, then drained by exact bounded chunks.
     * The original registration tuple remains installed until FINALIZE so a
     * successor cannot race partial cleanup.
     */
    bool lease_cleanup_pending;
    uint64_t lease_cleanup_generation;
    /* Runtime-derived references used to make epoch reclamation O(1). */
    WorkerState **worker_states;
    size_t worker_state_count;
    size_t worker_state_capacity;
    AdmissionRankState **admission_rank_states;
    size_t admission_rank_count;
    size_t admission_rank_capacity;
    Reservation **reservation_states;
    size_t reservation_count;
    size_t reservation_capacity;
    size_t node_membership_count;
    /* Runtime-derived partitions rebuilt from per-object semantic markers. */
    size_t lease_cleanup_worker_remaining;
    size_t lease_cleanup_admission_remaining;
    /* Runtime-only derived position in RouterIndex::worker_lease_expiry_heap. */
    size_t registration_expiry_heap_index;
};

struct Reservation {
    uint64_t client_nonce;
    uint64_t request_nonce;
    uint64_t worker_id;
    uint32_t dp_rank;
    uint64_t worker_admission_incarnation;
    uint64_t expires_at_ms;
    uint32_t matched_blocks;
    uint32_t active_reservations_at_grant;
    uint8_t *domain;
    uint32_t domain_length;
    uint8_t *request_bytes;
    uint32_t request_length;
    /*
     * Index into RouterIndex::reservation_expiry_heap. This is runtime-only
     * derived state: snapshots persist the deadline, then rebuild the heap
     * through router_index_add_reservation on load.
     */
    size_t expiry_heap_index;
    /* Runtime-derived position in WorkerEpoch::reservation_states. */
    size_t worker_epoch_index;
};

struct AdmissionRankState {
    uint64_t worker_id;
    uint32_t dp_rank;
    uint32_t capacity;
    uint64_t incarnation;
    uint32_t active_reservations;
    uint8_t *domain;
    uint32_t domain_length;
    size_t worker_epoch_index;
    uint64_t lease_cleanup_generation;
};

typedef struct {
    ValkeyModuleDict *nodes_by_external;
    ValkeyModuleDict *children_by_parent_and_local;
    ValkeyModuleDict *workers;
    ValkeyModuleDict *worker_epochs;
    ValkeyModuleDict *admission_ranks;
    ValkeyModuleDict *reservations;
    /* Min-heap ordered by Reservation::expires_at_ms. */
    Reservation **reservation_expiry_heap;
    size_t reservation_expiry_heap_length;
    size_t reservation_expiry_heap_capacity;
    /* Min-heap of worker-owned registration leases by absolute expiry. */
    WorkerEpoch **worker_lease_expiry_heap;
    size_t worker_lease_expiry_heap_length;
    size_t worker_lease_expiry_heap_capacity;
    uint64_t mutation_count;
    uint64_t generation_counter;
    uint64_t lifecycle_generation_counter;
    /* Reject absent-worker registration tokens issued before epoch GC. */
    uint64_t registration_gc_floor;
    /* Runtime-only bounded-scan cursor. Exact mutations are replicated. */
    uint8_t gc_phase;
    uint8_t gc_cursor[DYNKV_MAX_GC_CURSOR_BYTES];
    size_t gc_cursor_length;
} RouterIndex;

typedef struct {
    const uint8_t *data;
    size_t length;
    size_t offset;
} Reader;

typedef struct {
    uint8_t *data;
    size_t length;
    size_t capacity;
} Buffer;

typedef struct {
    WorkerState *worker;
    uint32_t matched_blocks;
    uint64_t last_external_hash;
} MatchScore;

typedef struct {
    uint64_t external_hash;
    uint64_t parent_hash;
    uint64_t local_hash;
} ReplaceNodePlan;

typedef struct {
    uint64_t event_id;
    bool active;
} ReplaceOwnerPlan;

typedef struct {
    uint64_t external_hash;
    uint64_t parent_hash;
    uint64_t local_hash;
} StoreBlock;

typedef struct {
    MatchScore *scores;
    size_t count;
    size_t capacity;
    ValkeyModuleDict *positions;
} MatchScores;

typedef struct {
    uint8_t kind;
    uint64_t worker_id;
    uint32_t dp_rank;
    uint64_t event_id;
} EventHeader;

typedef struct {
    ValkeyModuleDict *nodes_by_external;
    ValkeyModuleDict *nodes_by_edge;
    ValkeyModuleDict *owners_by_external;
} ReplacePlan;

typedef struct {
    uint64_t worker_id;
    uint32_t dp_rank;
    uint32_t capacity;
} AdmissionCandidate;

typedef struct {
    const uint8_t *domain;
    uint32_t domain_length;
    uint64_t client_nonce;
    uint64_t request_nonce;
    uint64_t lease_ms;
    const uint8_t *raw_bytes;
    uint32_t raw_length;
    AdmissionCandidate *candidates;
    uint32_t candidate_count;
} AdmissionRequest;

typedef struct {
    bool selected;
    uint64_t worker_id;
    uint32_t dp_rank;
    uint64_t worker_admission_incarnation;
    uint64_t expires_at_ms;
    uint32_t matched_blocks;
    uint32_t active_reservations_at_grant;
    uint32_t capacity;
} AdmissionSelection;

typedef struct {
    const uint8_t *domain;
    uint32_t domain_length;
    uint64_t client_nonce;
    uint64_t request_nonce;
} AdmissionIdentity;

typedef struct {
    uint64_t worker_id;
    uint64_t owner_nonce;
    uint64_t expires_at_ms;
} WorkerLeaseExpiry;

typedef struct {
    uint64_t examined;
    uint64_t reclaimed;
    uint64_t owners;
    uint64_t admission_ranks;
    uint64_t nodes;
    uint64_t workers;
    uint64_t worker_epochs;
    uint64_t lease_expiries;
    uint64_t epoch_deletions;
    bool stop_planning;
} GcResult;

extern ValkeyModuleType *RouterIndexType;


#endif /* DYNKV_TYPES_H */
