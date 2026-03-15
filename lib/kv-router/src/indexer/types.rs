// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bench")]
use std::time::Instant;

#[cfg(feature = "metrics")]
use dynamo_runtime::error::DynamoError;
#[cfg(feature = "metrics")]
pub use dynamo_runtime::protocols::maybe_error::MaybeError;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use crate::protocols::*;
use dynamo_tokens::SequenceHash;

/// Trait for types that may represent an error response.
/// Used for RPC-style responses that can indicate success or failure.
#[cfg(not(feature = "metrics"))]
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
}

// -------
// Distributed router - Worker KV Query types
// -------

/// Request to query a worker's local KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkerKvQueryRequest {
    /// The worker ID of the worker to query.
    pub worker_id: WorkerId,

    /// Start event ID (inclusive). If `None`, dumps entire tree.
    pub start_event_id: Option<u64>,
    /// End event ID (inclusive). If `None`, returns up to newest available.
    pub end_event_id: Option<u64>,
}

/// Response from a worker's local KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WorkerKvQueryResponse {
    /// Events served from the circular buffer (with original event IDs)
    Events(Vec<RouterEvent>),
    /// Full tree dump (with synthetic 0-indexed event IDs)
    TreeDump(Vec<RouterEvent>),
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

#[cfg(feature = "metrics")]
impl MaybeError for WorkerKvQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        WorkerKvQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<DynamoError> {
        match self {
            WorkerKvQueryResponse::Error(msg) => Some(DynamoError::msg(msg.clone())),
            _ => None,
        }
    }
}

// -------
// Standalone indexer query types (request plane)
// -------

/// Endpoint name for the standalone KV indexer query service.
pub const KV_INDEXER_QUERY_ENDPOINT: &str = "kv_indexer_query";

/// Request to query the standalone KV indexer for overlap scores.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IndexerQueryRequest {
    /// Model name to query the indexer for.
    pub model_name: String,
    /// Dynamo namespace (used as tenant_id for indexer lookup).
    pub namespace: String,
    /// Block hashes to find matches for in the radix tree.
    pub block_hashes: Vec<LocalBlockHash>,
}

/// Wire-friendly overlap scores for JSON serialization.
/// `OverlapScores` uses `FxHashMap<WorkerWithDpRank, _>` which can't be
/// serialized as JSON (struct keys aren't valid JSON map keys), so we flatten
/// to vecs of tuples for the wire protocol.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WireOverlapScores {
    pub scores: Vec<(WorkerWithDpRank, u32)>,
    pub frequencies: Vec<usize>,
    pub tree_sizes: Vec<(WorkerWithDpRank, usize)>,
}

impl From<OverlapScores> for WireOverlapScores {
    fn from(s: OverlapScores) -> Self {
        Self {
            scores: s.scores.into_iter().collect(),
            frequencies: s.frequencies,
            tree_sizes: s.tree_sizes.into_iter().collect(),
        }
    }
}

impl From<WireOverlapScores> for OverlapScores {
    fn from(w: WireOverlapScores) -> Self {
        Self {
            scores: w.scores.into_iter().collect(),
            frequencies: w.frequencies,
            tree_sizes: w.tree_sizes.into_iter().collect(),
        }
    }
}

/// Response from the standalone KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum IndexerQueryResponse {
    /// Overlap scores per worker.
    Scores(WireOverlapScores),
    /// An error occurred processing the query.
    Error(String),
}

#[cfg(feature = "metrics")]
impl MaybeError for IndexerQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        IndexerQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<DynamoError> {
        match self {
            IndexerQueryResponse::Error(msg) => Some(DynamoError::msg(msg.clone())),
            _ => None,
        }
    }
}

#[cfg(not(feature = "metrics"))]
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

/// A request to dump the tree as events
pub struct DumpRequest {
    /// Channel to send the dumped events
    pub resp: oneshot::Sender<Vec<RouterEvent>>,
}

/// A request to get all workers currently tracked
pub struct GetWorkersRequest {
    /// Channel to send the worker IDs
    pub resp: oneshot::Sender<Vec<WorkerId>>,
}

pub enum WorkerTask {
    Event(RouterEvent),
    /// Permanently remove a worker from tracking (keep_worker: false).
    RemoveWorker(WorkerId),
    /// Remove a single dp_rank for a worker.
    RemoveWorkerDpRank(WorkerId, DpRank),
    DumpEvents(oneshot::Sender<anyhow::Result<Vec<RouterEvent>>>),
    Terminate,
}

/// A request to process a routing decision.
pub(super) struct RoutingDecisionRequest {
    pub(super) worker: WorkerWithDpRank,
    pub(super) local_hashes: Vec<LocalBlockHash>,
    pub(super) sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone)]
pub struct ShardedMatchRequest {
    pub(super) sequence: Vec<LocalBlockHash>,
    pub(super) early_exit: bool,
    pub(super) resp: mpsc::Sender<OverlapScores>,
    #[cfg(feature = "bench")]
    pub(super) created_at: Instant,
}

impl ShardedMatchRequest {
    pub(super) fn new(
        sequence: Vec<LocalBlockHash>,
        early_exit: bool,
        resp: mpsc::Sender<OverlapScores>,
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
