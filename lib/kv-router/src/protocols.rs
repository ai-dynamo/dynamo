// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use dynamo_kv_events::event::{
    KV_EVENT_SUBJECT, KvCacheEvent, KvCacheEventData, KvCacheEventError, KvCacheEvents,
    KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData, RouterEvent,
};
pub use dynamo_kv_events::placement::{Placement, PlacementEvent, PlacementOwner};

pub use dynamo_kv_hashing::routing::{
    BlockHashOptions, MM_PAD_HASH_MASK, MM_PAD_SHIFT_VALUE, TokensWithHashes, XXH3_SEED,
    complete_block_count, compute_block_hash, compute_block_hash_for_seq,
    compute_block_hash_for_seq_with_seed, compute_next_seq_hash, compute_seq_hash_for_block,
    compute_seq_hash_for_tokens_with_seeds, hash_mm_identifier, pad_value_for_mm_hash,
};
pub use dynamo_router_protocol::mm::{
    BlockExtraInfo, BlockMmObjectInfo, RequestExtraInfo, RequestMmObjectInfo,
};

pub use dynamo_kv_hints::{KvHintTransferWorkerMetadata, KvTransferEnforcement};
pub use dynamo_scheduler::protocols::{
    PotentialLoad, RouterEventSink, RouterRequest, RouterResponse, RoutingConstraints,
    SelectionLogFields, WorkerConfigLike, WorkerSelectionResult,
};

pub use dynamo_router_protocol::block_hash::{ExternalSequenceBlockHash, LocalBlockHash};

pub use dynamo_router_protocol::metadata::RouterHintSourceMetadata;

pub use dynamo_kv_events::residency::{
    ResetScope, ResidencyDomain, ResidencyOwner, StorageTier, UnsupportedResidencyDomain,
    WireResidencyDomain,
};

pub use dynamo_router_protocol::worker::{
    ActiveLoad, DpRank, WorkerAffinityTarget, WorkerId, WorkerWithDpRank,
};

pub use dynamo_kv_indexer::residency::{
    ResidencyOwnerKey, ResidencyOwnerKeyExt, ResidencyProjection, ResidencyProjectionError,
    ResidencyRoutingSnapshot, ResolvedRouterHintSource,
};

pub use crate::plugins::worker_selection::{
    SelectionTrace, SelectionTraceRow, SelectionTraceSpec, SharedCacheHits,
};

pub use dynamo_slot_tracker::wire::{
    ActiveBlockEvent, ActiveBlockEventData, ActiveSequenceEvent, ActiveSequenceEventBatch,
    ActiveSequenceEventData, MAX_REPLICA_BATCH_DURATION, MAX_REPLICA_BATCH_EVENTS, PrefillEvent,
    PrefillEventData, PrefillLoadHint, SchedulerGroup, SchedulerLoad,
};

pub use dynamo_kv_indexer::overlap::OverlapScores;
