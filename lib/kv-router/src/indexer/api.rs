// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-plane DTOs for the worker KV query endpoint and the served indexer endpoints.
//! They stay in `dynamo-kv-router` because `lib/llm` implements the runtime `MaybeError` on
//! them; the `MaybeError` newtypes move that out before the DTOs join the stage crate.

use serde::{Deserialize, Serialize};

use super::WireTieredMatchDetails;
use crate::identity::CacheOwnerId;
use crate::protocols::*;
use dynamo_tokens::SequenceHash;

/// Trait for types that may represent an error response.
/// Used for RPC-style responses that can indicate success or failure.
pub trait MaybeError {
    /// Construct an instance from an error.
    fn from_err(err: impl std::error::Error + 'static) -> Self;
    /// Convert to an error instance if this represents an error.
    fn err(&self) -> Option<Box<dyn std::error::Error + Send + Sync>>;
}

// -------
// Distributed router - Worker KV Query types
// -------

/// Immutable protocol selected for one state-agent lifecycle.
///
/// A live worker/handler never switches this value. Upgrading the protocol
/// requires a fresh publisher incarnation and discovery advertisement.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum KvStateProtocolVersion {
    V2,
}

/// Exact identity returned by the state agent's callable status endpoint.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct KvStateAgentIdentity {
    pub cache_owner_id: CacheOwnerId,
    pub publisher_id: u64,
    pub protocol_version: KvStateProtocolVersion,
}

/// Current engine attachment observed by a state agent.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct KvStateAttachmentStatus {
    pub generation: u64,
    pub worker: WorkerWithDpRank,
    pub ready: bool,
    pub ready_at_outbound_cursor: u64,
}

/// Lightweight liveness response served without entering the dump queue.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct KvStateAgentStatus {
    pub identity: KvStateAgentIdentity,
    pub attachment: Option<KvStateAttachmentStatus>,
    /// False after a CacheOwner local/recovery transaction becomes uncertain.
    pub cache_owner_ready: bool,
    pub outbound_cursor: u64,
}

/// Proof that recovery completed against one exact state-agent incarnation.
///
/// The attachment generation is present when the response contains ephemeral
/// Worker ownership. CacheOwner-only recovery is fenced by the source identity.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct KvStateRecoveryReceipt {
    pub identity: KvStateAgentIdentity,
    pub attachment_generation: Option<u64>,
    pub recovered_through_cursor: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "snake_case")]
pub enum WorkerKvQueryKind {
    #[default]
    Recovery,
    StateAgentRecovery {
        expected: KvStateAgentIdentity,
        expected_attachment_generation: Option<u64>,
    },
    Status {
        expected: KvStateAgentIdentity,
        expected_attachment_generation: Option<u64>,
    },
}

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

    /// Opt in to an explicit [`WorkerKvQueryResponse::TreeDumpFailed`] result.
    /// Named MessagePack clients that predate this field deserialize it as false.
    #[serde(default)]
    pub supports_tree_dump_failed: bool,

    /// Recovery remains the default for old clients. Status is an explicit v2
    /// control-path request and never queues behind a full tree dump.
    #[serde(default)]
    pub kind: WorkerKvQueryKind,
}

/// Response from a worker's local KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[non_exhaustive]
pub enum WorkerKvQueryResponse {
    /// Callable liveness and identity fence for a v2 state agent.
    Status(KvStateAgentStatus),
    /// Recovery payload and proof returned only for an explicit state-agent
    /// recovery request after identity/attachment revalidation.
    StateAgentRecovery {
        response: Box<WorkerKvQueryResponse>,
        receipt: KvStateRecoveryReceipt,
    },
    /// Events served from the circular buffer with original event IDs. The batch
    /// is recovery-equivalent to replaying the requested `start_event_id` through
    /// the current buffered tail. If the rank stream contains one or more `Cleared`
    /// all-domain `Cleared` events, the source may omit events before the latest such
    /// clear while preserving that clear event and all following events. Domain-scoped
    /// clears remain ordinary ordered events. `last_event_id` is taken from the same
    /// buffer snapshot and should be used as the recovery watermark after applying the
    /// batch.
    Events {
        events: Vec<RouterEvent>,
        last_event_id: u64,
    },
    /// Full replay-ordered tree dump (with synthetic 0-indexed event IDs).
    ///
    /// Parent-addressed stored events appear after the event that introduces their parent.
    /// Consumers may therefore rebuild exact source state in one pass. Indexers that describe
    /// state positionally may use `start_position`; consumers that require parent-addressed
    /// replay must reject unsupported non-zero orphan positions rather than treating them as
    /// independent roots.
    /// Includes `last_event_id`: the newest real event ID in the worker's buffer
    /// at the time of the dump, so the caller can set its tracking cursor correctly.
    TreeDump {
        events: Vec<RouterEvent>,
        last_event_id: u64,
        /// Scope authoritatively represented by this snapshot. Legacy responses
        /// omitted this field and always represented the whole rank.
        #[serde(default)]
        reset_scope: ResetScope,
    },
    /// The exact tree dump could not be produced. This is distinct from an
    /// authoritative empty tree; recovery must leave indexed state and its
    /// admission cursor unchanged.
    TreeDumpFailed { last_event_id: u64, message: String },
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
