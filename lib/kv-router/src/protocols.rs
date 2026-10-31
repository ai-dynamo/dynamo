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

/// TRANSFER hint metadata exposed by a worker config for one global DP rank.
///
/// This is borrowed from the underlying worker config so candidate filtering can
/// check capability, role compatibility, and source endpoint presence without
/// allocating. `source_control_endpoint` is optional because targets only need
/// to consume hints, while sources must provide an endpoint.
pub use crate::scheduling::protocols::{
    KvHintTransferWorkerMetadata, KvTransferEnforcement, PotentialLoad, RouterEventSink,
    RouterRequest, RouterResponse, RoutingConstraints, SelectionLogFields, WorkerConfigLike,
    WorkerSelectionResult,
};

pub use dynamo_router_protocol::block_hash::{ExternalSequenceBlockHash, LocalBlockHash};

pub use dynamo_router_protocol::load::{ActiveLoad, SchedulerGroup, SchedulerLoad};

pub use dynamo_router_protocol::metadata::RouterHintSourceMetadata;

pub use dynamo_router_protocol::residency::{
    ResetScope, ResidencyDomain, ResidencyOwner, ResidencyOwnerKey, StorageTier,
    UnsupportedResidencyDomain, WireResidencyDomain,
};

pub use dynamo_router_protocol::worker::{
    DpRank, WorkerAffinityTarget, WorkerId, WorkerWithDpRank,
};

pub use dynamo_kv_indexer::residency::{
    ResidencyOwnerKeyExt, ResidencyProjection, ResidencyProjectionError, ResidencyRoutingSnapshot,
    ResolvedRouterHintSource,
};

pub use crate::plugins::worker_selection::{
    SelectionTrace, SelectionTraceRow, SelectionTraceSpec, SharedCacheHits,
};

pub use dynamo_slot_tracker::wire::{
    ActiveBlockEvent, ActiveBlockEventData, ActiveSequenceEvent, ActiveSequenceEventBatch,
    ActiveSequenceEventData, MAX_REPLICA_BATCH_DURATION, MAX_REPLICA_BATCH_EVENTS, PrefillEvent,
    PrefillEventData, PrefillLoadHint,
};

pub use dynamo_kv_indexer::overlap::OverlapScores;

// ------
// TokensWithHashes
// ------

// ------
// Tests
// ------

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json;

    // Temporary N/N-1 fixtures. Remove this module when domainless RouterEvent
    // is outside the supported mixed-version window.

    /// Pin the sglang pad_value constants and formula against upstream
    /// `MultimodalItem._compute_pad_value`. If sglang bumps a constant, this
    /// fails — otherwise routing-side pad_value would silently diverge from
    /// sglang's `BlockStored` bytes and MM-routing would degrade to text-prefix.

    #[test]
    fn test_local_block_hash_serialization() {
        let hash = LocalBlockHash(12345);
        let serialized = serde_json::to_string(&hash).unwrap();
        assert_eq!(serialized, "12345");

        let deserialized: LocalBlockHash = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, hash);
    }

    #[test]
    fn test_external_sequence_block_hash_serialization() {
        let hash = ExternalSequenceBlockHash(67890);
        let serialized = serde_json::to_string(&hash).unwrap();
        assert_eq!(serialized, "67890");

        let deserialized: ExternalSequenceBlockHash = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, hash);
    }

    #[test]
    fn test_shared_cache_hits_from_hits() {
        // All hits contiguous
        let hits = SharedCacheHits::from_hits(&[true, true, true, true]);
        assert_eq!(hits.ranges, vec![0..4]);
        assert_eq!(hits.total_hits, 4);

        // Sparse hits
        let hits = SharedCacheHits::from_hits(&[true, false, true, true, false, true]);
        assert_eq!(hits.ranges, vec![0..1, 2..4, 5..6]);
        assert_eq!(hits.total_hits, 4);

        // No hits
        let hits = SharedCacheHits::from_hits(&[false, false, false]);
        assert!(hits.ranges.is_empty());
        assert_eq!(hits.total_hits, 0);

        // Empty
        let hits = SharedCacheHits::from_hits(&[]);
        assert!(hits.ranges.is_empty());
        assert_eq!(hits.total_hits, 0);
    }

    #[test]
    fn test_shared_cache_hits_beyond() {
        // Shared has [A, B, C, D] => range 0..4
        #[allow(clippy::single_range_in_vec_init)]
        let hits = SharedCacheHits::from_ranges(vec![0..4]);

        // Device has overlap=2 (positions 0,1 on device) => shared_beyond should count positions 2,3
        assert_eq!(hits.hits_beyond(2), 2);

        // Device has overlap=0 => all 4 shared hits count
        assert_eq!(hits.hits_beyond(0), 4);

        // Device has overlap=4 => nothing beyond
        assert_eq!(hits.hits_beyond(4), 0);

        // Device overlap exceeds range
        assert_eq!(hits.hits_beyond(10), 0);
    }

    #[test]
    fn test_shared_cache_hits_beyond_sparse() {
        // Ranges: [1..3, 5..8] => positions 1,2,5,6,7
        let hits = SharedCacheHits::from_ranges(vec![1..3, 5..8]);
        assert_eq!(hits.total_hits, 5);

        // from_position=0 => all 5 hits
        assert_eq!(hits.hits_beyond(0), 5);
        // from_position=2 => pos 2 (from first range) + 5,6,7 (from second) = 4
        assert_eq!(hits.hits_beyond(2), 4);
        // from_position=3 => only second range: 3 hits
        assert_eq!(hits.hits_beyond(3), 3);
        // from_position=6 => positions 6,7 from second range = 2
        assert_eq!(hits.hits_beyond(6), 2);
        // from_position=8 => nothing
        assert_eq!(hits.hits_beyond(8), 0);
    }
}
