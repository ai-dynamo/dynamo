// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bench")]
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::kv_hints::KvTransferCandidates;
use crate::prelude::*;
use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;

#[cfg(feature = "bench")]
use super::{EventCompletionBuffer, EventCompletionWriter, ObservationSeal};

/// Errors that can occur in the KV Router.
#[derive(Debug, thiserror::Error)]
pub enum KvRouterError {
    #[error("Block not found")]
    BlockNotFound,

    #[error("Indexer is offline")]
    IndexerOffline,

    #[error("Indexer dropped the request")]
    IndexerDroppedRequest,

    #[error("Prune operation failed: {0}")]
    PruneFailed(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

/// Shared structural anchor used by branch-sharded routing when a routed
/// subtree starts on a different shard from its parent prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AnchorRef {
    pub anchor_id: ExternalSequenceBlockHash,
    pub anchor_local_hash: LocalBlockHash,
    pub anchor_depth: usize,
}

/// Worker task payload that installs an [`AnchorRef`] into a shard-local
/// backend before dependent suffix events are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AnchorTask {
    pub anchor_id: ExternalSequenceBlockHash,
    pub anchor_local_hash: LocalBlockHash,
    pub anchor_depth: usize,
}

/// Wire-friendly overlap scores for JSON serialization.
/// `OverlapScores` uses `FxHashMap<WorkerWithDpRank, _>` which can't be
/// serialized as JSON (struct keys aren't valid JSON map keys), so we flatten
/// to vecs of tuples for the wire protocol.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WireOverlapScores {
    pub scores: Vec<(WorkerWithDpRank, u32)>,
    pub frequencies: Vec<usize>,
}

impl From<OverlapScores> for WireOverlapScores {
    fn from(s: OverlapScores) -> Self {
        Self {
            scores: s.scores.into_iter().collect(),
            frequencies: s.frequencies,
        }
    }
}

impl From<WireOverlapScores> for OverlapScores {
    fn from(w: WireOverlapScores) -> Self {
        Self {
            scores: w.scores.into_iter().collect(),
            frequencies: w.frequencies,
        }
    }
}

/// Wire-friendly lower-tier match payload for JSON serialization.
///
/// Mirrors `LowerTierMatchDetails.hits`. `next_continuations` and KV transfer
/// candidates are server-side intermediate state and are not carried over the wire.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WireLowerTierMatchDetails {
    pub hits: Vec<(WorkerWithDpRank, usize)>,
}

/// Wire-friendly tiered match payload: device overlap plus per-tier hits.
///
/// Lower tiers are a `Vec<(StorageTier, _)>` rather than a map so we never
/// depend on `StorageTier` being a JSON-legal map key. Each `StorageTier` is
/// expected to appear at most once; the inbound conversion warns and keeps the
/// last entry if duplicates are observed.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WireTieredMatchDetails {
    pub device: WireOverlapScores,
    pub lower_tier: Vec<(StorageTier, WireLowerTierMatchDetails)>,
}

/// Precomputed hashes for recording a route-time indexer update.
#[derive(Debug, Clone)]
pub struct RoutingDecisionHashes {
    pub local_hashes: Vec<LocalBlockHash>,
    pub sequence_hashes: Vec<SequenceHash>,
}

impl RoutingDecisionHashes {
    pub fn from_local_hashes(local_hashes: Vec<LocalBlockHash>) -> Self {
        let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
        Self {
            local_hashes,
            sequence_hashes,
        }
    }
}

/// Rich non-wire query result for router-local device tier lookups.
#[derive(Debug, Clone, Default)]
pub struct MatchDetails {
    /// Existing overlap scores used by scheduling.
    pub overlap_scores: OverlapScores,
    /// Last matched device sequence hash per worker, used to seed lower-tier queries.
    pub last_matched_hashes: FxHashMap<WorkerWithDpRank, ExternalSequenceBlockHash>,
    /// Optional root-aligned device candidates used to build compact KV transfer hints.
    pub kv_transfer_candidates: Option<KvTransferCandidates>,
}

impl MatchDetails {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn retain_kv_transfer_candidates(
        &mut self,
        mut block_hashes: Vec<ExternalSequenceBlockHash>,
    ) {
        let mut owner_prefix_blocks: Vec<_> = self
            .overlap_scores
            .scores
            .iter()
            .filter_map(|(worker, blocks)| {
                let blocks = usize::try_from(*blocks).ok()?;
                (blocks > 0 && blocks <= block_hashes.len()).then_some(((*worker).into(), blocks))
            })
            .collect();
        if block_hashes.is_empty() || owner_prefix_blocks.is_empty() {
            return;
        }
        let max_owner_prefix_blocks = owner_prefix_blocks
            .iter()
            .map(|(_, blocks)| *blocks)
            .max()
            .unwrap_or(0);
        block_hashes.truncate(max_owner_prefix_blocks);
        owner_prefix_blocks.sort_unstable_by_key(|(worker, _)| *worker);
        self.kv_transfer_candidates = Some(KvTransferCandidates {
            block_hashes,
            owner_prefix_blocks,
            routing_snapshot: None,
        });
    }
}

/// A request to find matches in the Radix Tree.
pub struct MatchRequest {
    /// A vector of `LocalBlockHash` representing the sequence to match.
    pub sequence: Vec<LocalBlockHash>,
    /// A boolean indicating whether to exit early if a single match is found.
    pub early_exit: bool,
    /// A channel sender to send the `OverlapScores` response.
    pub resp: oneshot::Sender<OverlapScores>,
    /// Timestamp when the request was created (for queue wait time measurement)
    #[cfg(feature = "bench")]
    pub created_at: Instant,
}

impl MatchRequest {
    pub(super) fn new(
        sequence: Vec<LocalBlockHash>,
        early_exit: bool,
        resp: oneshot::Sender<OverlapScores>,
    ) -> Self {
        Self {
            sequence,
            early_exit,
            resp,
            #[cfg(feature = "bench")]
            created_at: Instant::now(),
        }
    }
}

/// A request to find matches while also returning continuation metadata.
pub struct MatchDetailsRequest {
    /// A vector of `LocalBlockHash` representing the sequence to match.
    pub sequence: Vec<LocalBlockHash>,
    /// A boolean indicating whether to exit early if a single match is found.
    pub early_exit: bool,
    /// When true, retain the matched root-aligned external hash chain for KV transfer hints.
    pub retain_kv_transfer_chain: bool,
    /// A channel sender to send the `MatchDetails` response.
    pub resp: oneshot::Sender<MatchDetails>,
}

/// A request to test one worker's ownership of an external block hash.
pub struct ContainsWorkerBlockRequest {
    pub worker: WorkerWithDpRank,
    pub block_hash: ExternalSequenceBlockHash,
    pub resp: oneshot::Sender<bool>,
}

impl MatchDetailsRequest {
    pub(super) fn new(
        sequence: Vec<LocalBlockHash>,
        early_exit: bool,
        retain_kv_transfer_chain: bool,
        resp: oneshot::Sender<MatchDetails>,
    ) -> Self {
        Self {
            sequence,
            early_exit,
            retain_kv_transfer_chain,
            resp,
        }
    }
}

/// A request to dump the tree as events
pub struct DumpRequest {
    /// Channel to send the dumped events
    pub resp: oneshot::Sender<Vec<RouterEvent>>,
}

/// A request to wait until all previously submitted work is applied.
pub struct FlushRequest {
    /// Channel to acknowledge completion.
    pub resp: oneshot::Sender<()>,
}

/// A request to get all workers currently tracked
pub struct GetWorkersRequest {
    /// Channel to send the worker IDs
    pub resp: oneshot::Sender<Vec<WorkerId>>,
}

#[derive(Debug, Default)]
pub struct WorkerLookupStats {
    pub worker_blocks: Vec<(WorkerWithDpRank, usize)>,
}

impl WorkerLookupStats {
    pub fn from_worker_block_counts(
        counts: impl IntoIterator<Item = (WorkerWithDpRank, usize)>,
    ) -> Self {
        Self {
            worker_blocks: counts
                .into_iter()
                .filter(|(_, block_count)| *block_count > 0)
                .collect(),
        }
    }

    pub fn worker_count(&self) -> usize {
        self.worker_blocks.len()
    }

    pub fn block_count(&self) -> usize {
        self.worker_blocks
            .iter()
            .map(|(_, block_count)| *block_count)
            .sum()
    }

    pub fn block_count_for_worker(&self, worker: WorkerWithDpRank) -> Option<usize> {
        self.worker_blocks
            .iter()
            .find_map(|(candidate, block_count)| (*candidate == worker).then_some(*block_count))
    }
}

pub enum WorkerTask {
    Event(RouterEvent),
    EventWithAck {
        event: RouterEvent,
        resp: oneshot::Sender<bool>,
    },
    ApproximateLru(super::ApproximateLruTask),
    #[cfg(feature = "bench")]
    InstallObservation {
        writer: EventCompletionWriter,
        resp: oneshot::Sender<bool>,
    },
    #[cfg(feature = "bench")]
    ObservedEvent {
        event: RouterEvent,
        correlation_id: u32,
    },
    #[cfg(feature = "bench")]
    SealObservation(oneshot::Sender<Option<ObservationSeal>>),
    #[cfg(feature = "bench")]
    HarvestObservation(oneshot::Sender<EventCompletionBuffer>),
    Anchor {
        worker: WorkerWithDpRank,
        anchor: AnchorTask,
    },
    /// Permanently remove a worker from tracking.
    RemoveWorker {
        worker_id: WorkerId,
        /// True for the one shared-state backend task that owns structural cleanup.
        sweep_tree: bool,
        /// Acknowledges completion of this lane's cold-path removal phase.
        resp: oneshot::Sender<()>,
    },
    /// Remove a single dp_rank for a worker.
    RemoveWorkerDpRank {
        worker_id: WorkerId,
        dp_rank: DpRank,
        /// True for the one shared-state backend task that owns structural cleanup.
        sweep_tree: bool,
    },
    DumpEvents(oneshot::Sender<anyhow::Result<Vec<RouterEvent>>>),
    Stats(oneshot::Sender<WorkerLookupStats>),
    ContainsWorkerBlock {
        worker: WorkerWithDpRank,
        block_hash: ExternalSequenceBlockHash,
        resp: oneshot::Sender<bool>,
    },
    Flush(oneshot::Sender<()>),
    Terminate,
}

/// A request to process a routing decision.
pub(super) struct RoutingDecisionRequest {
    pub(super) worker: WorkerWithDpRank,
    pub(super) local_hashes: Vec<LocalBlockHash>,
    pub(super) sequence_hashes: Vec<SequenceHash>,
}
