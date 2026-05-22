// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Feature-owned wire protocol for the KV indexer.
//!
//! All paths are **relative** — the server nests them under
//! `/v1/features/{ROUTE_PREFIX}` (see
//! [`FeatureManager::route_prefix`](crate::features::FeatureManager::route_prefix)).
//! Nothing here lives in the central [`crate::protocol::paths`]; the feature
//! owns its whole namespace.

use kvbm_logical::SequenceHash;
use serde::{Deserialize, Serialize};
use velo_ext::InstanceId;

/// URL segment the server nests this feature's routers under
/// (`/v1/features/indexer/...`).
pub const ROUTE_PREFIX: &str = "indexer";

/// Velo active-message handler name for the client → hub block lookup. The hub
/// installs this handler on its own velo in
/// [`IndexerManager::attach`](super::manager::IndexerManager); a
/// [`IndexerLookupClient`](super::client::IndexerLookupClient) calls it. Follows
/// the `kvbm_hub_*` convention shared with the heartbeat handler.
pub const QUERY_HANDLER: &str = "kvbm_hub_indexer_query";

/// Relative route paths (mounted under `/v1/features/indexer`).
pub mod paths {
    /// `GET /config` — indexer configuration + ZMQ ingest endpoint. A `200`
    /// also serves as the capability probe used by connectors.
    pub const CONFIG: &str = "/config";

    /// `GET /instances` — the set of instances that declared `Feature::Indexer`
    /// at registration. Not every registered instance necessarily emits KV
    /// events, so this is the *registered* (participating) set.
    pub const INSTANCES: &str = "/instances";

    /// `GET /hashes/by_position/{pos}` — dump the index bucket at `pos`.
    pub const BY_POSITION: &str = "/hashes/by_position/{pos}";

    /// `POST /query` — resolve a block-hash sequence to the holding instances.
    pub const QUERY: &str = "/query";
}

/// Response for `GET /config`. Doubles as the capability probe: a successful
/// `200` tells a connector the indexer is present and where to publish.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexerConfigResponse {
    /// Maximum sequence length (tokens) the index is sized for.
    pub max_seq_len: usize,
    /// Block size (tokens per block). Must match the publisher's page size.
    pub block_size: usize,
    /// Number of position buckets (`max_seq_len / block_size`).
    pub num_positions: usize,
    /// ZMQ endpoint a publisher connects its `PUB` socket to
    /// (e.g. `tcp://127.0.0.1:54231`). Empty when ingest is not yet bound.
    pub zmq_endpoint: String,
}

/// Response for `GET /instances`. The set of instances that declared
/// `Feature::Indexer` at registration, as decimal `u128` strings (matching the
/// holder ids in [`IndexEntry::instances`]). Lets an operator distinguish
/// "registered to participate" from "actually holding indexed blocks".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct InstancesResponse {
    /// Registered (participating) instance ids, decimal `u128`, sorted.
    pub instances: Vec<String>,
}

/// One indexed block: a positional-lineage hash and the instances holding it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexEntry {
    /// Human-readable PLH (`position:current[:parent]`, base58).
    pub hash: String,
    /// Raw 128-bit PLH as a decimal string (jq-safe; avoids JSON number
    /// precision loss for values above 2^53).
    pub hash_u128: String,
    /// Block position decoded from the PLH.
    pub position: u64,
    /// Instance ids (decimal `u128` strings) currently holding this block.
    pub instances: Vec<String>,
}

/// Response for `GET /hashes/by_position/{pos}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ByPositionResponse {
    /// The queried position.
    pub position: usize,
    /// Entries indexed at that position.
    pub entries: Vec<IndexEntry>,
}

/// Request body for `POST /query`.
///
/// `hashes` are the block-sequence PLHs in position order (low → high). The
/// indexer walks them high → low and returns the deepest one present.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueryRequest {
    /// Positional-lineage hashes of the candidate block sequence.
    pub hashes: Vec<SequenceHash>,
}

/// Response body for `POST /query`. `hit` is `None` when no supplied hash is
/// indexed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueryResponse {
    /// The deepest matching block, or `None` if nothing matched.
    pub hit: Option<IndexEntry>,
}

/// Typed result of the velo [`QUERY_HANDLER`] lookup.
///
/// Unlike the HTTP [`IndexEntry`] (which stringifies ids for jq-safety), this
/// stays typed end-to-end over the velo plane: the matched [`SequenceHash`] and
/// the holder [`InstanceId`]s feed straight into peer discovery. Paired so the
/// matched hash and its holders cannot drift — a full miss is the `None` arm of
/// the `Option<FindBlocksHit>` the handler and
/// [`IndexerLookupClient`](super::client::IndexerLookupClient) return.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FindBlocksHit {
    /// Deepest candidate hash present in the index.
    pub matched: SequenceHash,
    /// Instances currently holding `matched`. Always non-empty.
    pub candidates: Vec<InstanceId>,
}
