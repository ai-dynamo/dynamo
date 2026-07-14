// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bench")]
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::protocols::*;
use crate::router_hint::RouterHintRootCandidates;
use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;

#[cfg(feature = "bench")]
use super::{EventCompletionBuffer, EventCompletionWriter, ObservationSeal};

/// Trait for types that may represent an error response.
/// Used for RPC-style responses that can indicate success or failure.
pub trait MaybeError {
    /// Construct an instance from an error.
    fn from_err(err: impl std::error::Error + 'static) -> Self;
    /// Convert to an error instance if this represents an error.
    fn err(&self) -> Option<Box<dyn std::error::Error + Send + Sync>>;
}

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

// -------
// Distributed router - Worker KV Query types
// -------

/// Request to query a worker's local KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkerKvQueryRequest {
    /// The worker ID of the worker to query.
    pub worker_id: WorkerId,
    /// Data-parallel rank owned by this worker query endpoint.
    pub dp_rank: DpRank,

    /// Start event ID (inclusive). If `None`, dumps entire tree.
    pub start_event_id: Option<u64>,
    /// End event ID (inclusive). Used for validation and `TooNew` responses.
    /// Successful buffer-backed recovery may still return through the current
    /// newest buffered event.
    pub end_event_id: Option<u64>,
}

/// Response from a worker's local KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WorkerKvQueryResponse {
    /// Events served from the circular buffer with original event IDs. The batch
    /// is recovery-equivalent to replaying the requested `start_event_id` through
    /// the current buffered tail. If the range contains one or more `Cleared`
    /// barriers, the worker may omit events before the last clear while preserving
    /// that clear event and all following events. `last_event_id` is taken from the
    /// same buffer snapshot and should be used as the recovery watermark after
    /// applying the batch.
    Events {
        events: Vec<RouterEvent>,
        last_event_id: u64,
    },
    /// Full tree dump (with synthetic 0-indexed event IDs).
    /// Includes `last_event_id`: the newest real event ID in the worker's buffer
    /// at the time of the dump, so the caller can set its tracking cursor correctly.
    TreeDump {
        events: Vec<RouterEvent>,
        last_event_id: u64,
    },
    /// Requested range is newer than available data
    TooNew {
        requested_start: Option<u64>,
        requested_end: Option<u64>,
        newest_available: u64,
    },
    /// Invalid range: end_id < start_id
    InvalidRange { start_id: u64, end_id: u64 },
    /// Query failed on worker (serialized error)
    Error(String),
}

impl MaybeError for WorkerKvQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        WorkerKvQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<Box<dyn std::error::Error + Send + Sync>> {
        match self {
            WorkerKvQueryResponse::Error(msg) => Some(Box::new(std::io::Error::other(msg.clone()))),
            _ => None,
        }
    }
}

#[cfg(feature = "runtime-protocols")]
impl dynamo_runtime::protocols::maybe_error::MaybeError for WorkerKvQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        WorkerKvQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<dynamo_runtime::error::DynamoError> {
        match self {
            WorkerKvQueryResponse::Error(msg) => {
                Some(dynamo_runtime::error::DynamoError::msg(msg.clone()))
            }
            _ => None,
        }
    }
}

// -------
// Standalone indexer query types (request plane)
// -------

/// Endpoint name for the standalone KV indexer query service.
pub const KV_INDEXER_QUERY_ENDPOINT: &str = "kv_indexer_query";
/// Endpoint name for recording approximate-mode routing decisions on a remote indexer.
pub const KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT: &str = "kv_indexer_record_routing_decision";

/// Request to query a served KV indexer for overlap scores.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IndexerQueryRequest {
    /// Model name to query the indexer for.
    pub model_name: String,
    /// Block hashes to find matches for in the radix tree.
    pub block_hashes: Vec<LocalBlockHash>,
    /// When true, the server skips the lower-tier walk and returns only the
    /// device-tier overlap. Older clients that omit this field default to
    /// `false`, preserving the full tiered response.
    #[serde(default)]
    pub device_only: bool,
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
/// Mirrors `LowerTierMatchDetails.hits`. `next_continuations` and router hint
/// candidates are server-side intermediate state and are not carried over the wire.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WireLowerTierMatchDetails {
    pub hits: Vec<(WorkerWithDpRank, usize)>,
}

impl From<&super::lower_tier::LowerTierMatchDetails> for WireLowerTierMatchDetails {
    fn from(d: &super::lower_tier::LowerTierMatchDetails) -> Self {
        Self {
            hits: d.hits.iter().map(|(w, h)| (*w, *h)).collect(),
        }
    }
}

impl From<WireLowerTierMatchDetails> for super::lower_tier::LowerTierMatchDetails {
    fn from(w: WireLowerTierMatchDetails) -> Self {
        // `next_continuations` is server-side intermediate state; consumers of
        // the tiered result never read it, so we reconstruct an empty map on
        // the wire-inbound path.
        Self {
            hits: w.hits.into_iter().collect(),
            next_continuations: Default::default(),
            router_hint_root_candidates: None,
            router_hint_extensions: None,
        }
    }
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

/// Response from a served KV indexer query.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum IndexerQueryResponse {
    /// Tiered match details: device overlap plus per-tier hits.
    TieredScores(WireTieredMatchDetails),
    /// An error occurred processing the query.
    Error(String),
}

impl MaybeError for IndexerQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        IndexerQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<Box<dyn std::error::Error + Send + Sync>> {
        match self {
            IndexerQueryResponse::Error(msg) => Some(Box::new(std::io::Error::other(msg.clone()))),
            _ => None,
        }
    }
}

#[cfg(feature = "runtime-protocols")]
impl dynamo_runtime::protocols::maybe_error::MaybeError for IndexerQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        IndexerQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<dynamo_runtime::error::DynamoError> {
        match self {
            IndexerQueryResponse::Error(msg) => {
                Some(dynamo_runtime::error::DynamoError::msg(msg.clone()))
            }
            _ => None,
        }
    }
}

/// Request to record a routing decision on a served approximate-mode indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IndexerRecordRoutingDecisionRequest {
    /// Model name to update.
    pub model_name: String,
    /// Selected worker for this routing decision.
    pub worker: WorkerWithDpRank,
    /// Locally-computed block hashes for the routed request.
    pub local_hashes: Vec<LocalBlockHash>,
    /// Locally-computed rolling sequence hashes for the routed request.
    pub sequence_hashes: Vec<SequenceHash>,
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

/// Response from a served approximate-mode routing-decision endpoint.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum IndexerRecordRoutingDecisionResponse {
    Recorded,
    Error(String),
}

impl MaybeError for IndexerRecordRoutingDecisionResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        IndexerRecordRoutingDecisionResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<Box<dyn std::error::Error + Send + Sync>> {
        match self {
            IndexerRecordRoutingDecisionResponse::Error(msg) => {
                Some(Box::new(std::io::Error::other(msg.clone())))
            }
            _ => None,
        }
    }
}

#[cfg(feature = "runtime-protocols")]
impl dynamo_runtime::protocols::maybe_error::MaybeError for IndexerRecordRoutingDecisionResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        IndexerRecordRoutingDecisionResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<dynamo_runtime::error::DynamoError> {
        match self {
            IndexerRecordRoutingDecisionResponse::Error(msg) => {
                Some(dynamo_runtime::error::DynamoError::msg(msg.clone()))
            }
            _ => None,
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
    /// Optional root-aligned device candidates used to build compact router hints.
    pub router_hint_root_candidates: Option<RouterHintRootCandidates>,
}

impl MatchDetails {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn retain_router_hint_root_candidates(
        &mut self,
        block_hashes: Vec<ExternalSequenceBlockHash>,
    ) {
        let mut owner_prefix_blocks: Vec<_> = self
            .overlap_scores
            .scores
            .iter()
            .filter_map(|(worker, blocks)| {
                let blocks = usize::try_from(*blocks).ok()?;
                (blocks > 0 && blocks <= block_hashes.len()).then_some((*worker, blocks))
            })
            .collect();
        if block_hashes.is_empty() || owner_prefix_blocks.is_empty() {
            return;
        }
        owner_prefix_blocks.sort_unstable_by_key(|(worker, _)| *worker);
        self.router_hint_root_candidates = Some(RouterHintRootCandidates {
            block_hashes,
            owner_prefix_blocks,
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
    /// When true, retain the matched root-aligned external hash chain for router hints.
    pub retain_router_hint_chain: bool,
    /// A channel sender to send the `MatchDetails` response.
    pub resp: oneshot::Sender<MatchDetails>,
}

impl MatchDetailsRequest {
    pub(super) fn new(
        sequence: Vec<LocalBlockHash>,
        early_exit: bool,
        retain_router_hint_chain: bool,
        resp: oneshot::Sender<MatchDetails>,
    ) -> Self {
        Self {
            sequence,
            early_exit,
            retain_router_hint_chain,
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
    },
    /// Remove a single dp_rank for a worker.
    RemoveWorkerDpRank {
        worker_id: WorkerId,
        dp_rank: DpRank,
        /// True for the one shared-state backend task that owns structural cleanup.
        sweep_tree: bool,
    },
    /// Best-effort maintenance task for shared-state backends.
    CleanupStaleChildren,
    DumpEvents(oneshot::Sender<anyhow::Result<Vec<RouterEvent>>>),
    Stats(oneshot::Sender<WorkerLookupStats>),
    Flush(oneshot::Sender<()>),
    Terminate,
}

/// A request to process a routing decision.
pub(super) struct RoutingDecisionRequest {
    pub(super) worker: WorkerWithDpRank,
    pub(super) local_hashes: Vec<LocalBlockHash>,
    pub(super) sequence_hashes: Vec<SequenceHash>,
}
