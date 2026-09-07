// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dependency-light metadata shared by the isolated CKF benchmark targets.

use std::path::PathBuf;

use dynamo_kv_router::protocols::WorkerWithDpRank;
use serde::{Deserialize, Serialize};

pub const DEFAULT_CKF_MOONCAKE_BLOCK_SIZE: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DcCkfCorpusSpec {
    pub trace_path: PathBuf,
    pub expected_sha256: Option<String>,
    pub trace_block_size: usize,
    pub prefix_depth_factor: usize,
    pub trace_duplication_factor: usize,
    pub dc_count: usize,
    pub workers_per_dc: usize,
    pub endpoint_ordinal: usize,
    pub default_dp_rank: u32,
}

impl DcCkfCorpusSpec {
    pub fn new(trace_path: impl Into<PathBuf>, dc_count: usize, workers_per_dc: usize) -> Self {
        Self {
            trace_path: trace_path.into(),
            expected_sha256: None,
            trace_block_size: DEFAULT_CKF_MOONCAKE_BLOCK_SIZE,
            prefix_depth_factor: 1,
            trace_duplication_factor: 1,
            dc_count,
            workers_per_dc,
            endpoint_ordinal: 0,
            default_dp_rank: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DcCkfSourceTopology {
    pub source_index: usize,
    pub dc_ordinal: usize,
    pub endpoint_ordinal: usize,
    pub worker_ordinal: usize,
    pub member: WorkerWithDpRank,
    pub session_count: usize,
    pub turn_count: usize,
    pub trace_distinct_hash_upper_bound: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DcCkfPoolCorpusMetadata {
    pub dc_ordinal: usize,
    pub endpoint_ordinal: usize,
    pub session_count: usize,
    pub turn_count: usize,
    /// Conservative trace-row bound, not a measured live CKF peak.
    pub trace_distinct_hash_upper_bound: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DcCkfCorpusMetadata {
    pub trace_path: PathBuf,
    pub trace_sha256: String,
    pub trace_block_size: usize,
    pub prefix_depth_factor: usize,
    pub trace_duplication_factor: usize,
    pub original_session_count: usize,
    pub prepared_session_count: usize,
    pub turn_count: usize,
    pub hash_reference_count: usize,
    /// Conservative trace-row bound, not a measured live CKF peak.
    pub trace_distinct_hash_upper_bound: usize,
    pub dc_count: usize,
    pub workers_per_dc: usize,
    pub endpoint_ordinal: usize,
    pub sources: Vec<DcCkfSourceTopology>,
    pub pools: Vec<DcCkfPoolCorpusMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DcCkfPoolCapacity {
    pub dc_ordinal: usize,
    pub endpoint_ordinal: usize,
    pub measured_peak_active_distinct_hashes: usize,
    pub final_active_distinct_hashes: usize,
    pub recommended_distinct_hash_capacity: usize,
    pub event_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DcCkfCapacityMetadata {
    pub headroom_percent: usize,
    pub pools: Vec<DcCkfPoolCapacity>,
}
