// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

use crate::mocker::perf_model::PerfModel;
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{BlockHash, SequenceHash, Token};

pub type NumBlocks = usize;

/// Represents different block movement operations in the cache
/// For Use and Promote variants, block hashes are included for KV event publishing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveBlock {
    Use(Vec<UniqueBlock>, Vec<BlockHash>),
    Destroy(Vec<UniqueBlock>),
    Deref(Vec<UniqueBlock>),
    Promote(Uuid, SequenceHash, Option<u64>, BlockHash),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveBlockResponse {
    Store(Vec<SequenceHash>, Option<u64>),
    Remove(Vec<SequenceHash>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectRequest {
    pub tokens: Vec<Token>,
    pub max_output_tokens: usize,
    pub uuid: Option<Uuid>,
    pub dp_rank: u32,
}

/// Represents the cost of prefilling content in the cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillCost {
    pub new_blocks: usize,
    pub new_tokens: usize,
}

impl PrefillCost {
    pub fn predict_prefill_compute(
        &self,
        new_tokens: Option<usize>,
        perf_model: &PerfModel,
    ) -> f64 {
        let tokens = new_tokens.unwrap_or(self.new_tokens);
        perf_model.predict_prefill_time(tokens)
    }
}

/// Signal for output token generation with completion status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSignal {
    pub uuid: Uuid,
    pub completed: bool,
}

/// Worker type for disaggregated serving configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WorkerType {
    /// Standard aggregated worker handling both prefill and decode
    #[default]
    Aggregated,
    /// Dedicated prefill worker in disaggregated mode
    Prefill,
    /// Dedicated decode worker in disaggregated mode
    Decode,
}

/// Configuration arguments for MockVllmEngine
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "owned", build_fn(public))]
pub struct MockEngineArgs {
    #[builder(default = "16384")]
    pub num_gpu_blocks: usize,

    #[builder(default = "64")]
    pub block_size: usize,

    // This was 1024 in the past but reverted back to 256
    #[builder(default = Some(256))]
    pub max_num_seqs: Option<usize>,

    // default for open api server, for llm class it's 16384
    #[builder(default = Some(8192))]
    pub max_num_batched_tokens: Option<usize>,

    #[builder(default = true)]
    pub enable_prefix_caching: bool,

    #[builder(default = true)]
    pub enable_chunked_prefill: bool,

    #[builder(default = "0.01")]
    pub watermark: f64,

    #[builder(default = "1.0")]
    pub speedup_ratio: f64,

    #[builder(default = "1")]
    pub dp_size: u32,

    /// Optional startup time in seconds to simulate engine initialization delay
    #[builder(default = "None")]
    pub startup_time: Option<f64>,

    /// Worker type for disaggregated serving (Aggregated, Prefill, or Decode)
    #[builder(default = "WorkerType::Aggregated")]
    pub worker_type: WorkerType,

    /// Performance model for timing predictions (not serialized, loaded from planner_profile_data)
    #[serde(skip)]
    #[builder(default = "Arc::new(PerfModel::default())")]
    pub perf_model: Arc<PerfModel>,

    /// Enable worker-local KV indexer for tracking this worker's own KV cache state
    #[builder(default = "false")]
    pub enable_local_indexer: bool,

    /// Bootstrap port for disaggregated serving rendezvous.
    /// Prefill workers listen on this port; decode workers connect to it.
    /// If None, bootstrap rendezvous is disabled.
    #[builder(default = "None")]
    pub bootstrap_port: Option<u16>,
}

impl Default for MockEngineArgs {
    fn default() -> MockEngineArgs {
        MockEngineArgsBuilder::default()
            .build()
            .expect("Failed to build default MockEngineArgs")
    }
}

impl MockEngineArgs {
    pub fn builder() -> MockEngineArgsBuilder {
        MockEngineArgsBuilder::default()
    }

    /// Create MockEngineArgs from a JSON file containing extra engine arguments
    pub fn from_json_file(path: &Path) -> anyhow::Result<Self> {
        let mut builder = Self::builder();

        // Load and parse the JSON file
        let file_content = std::fs::read_to_string(path)?;
        let extra_args: HashMap<String, serde_json::Value> = serde_json::from_str(&file_content)?;

        // Define valid field names
        let valid_fields: HashSet<&str> = [
            "num_gpu_blocks",
            "block_size",
            "max_num_seqs",
            "max_num_batched_tokens",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "watermark",
            "speedup_ratio",
            "dp_size",
            "startup_time",
            "is_prefill",
            "is_decode",
            "planner_profile_data",
            "enable_local_indexer",
            "bootstrap_port",
        ]
        .iter()
        .cloned()
        .collect();

        // Check for invalid arguments
        let invalid_args: Vec<String> = extra_args
            .keys()
            .filter(|key| !valid_fields.contains(key.as_str()))
            .cloned()
            .collect();

        if !invalid_args.is_empty() {
            return Err(anyhow::anyhow!(
                "Invalid arguments found in JSON file: {}. Valid arguments are: {:?}",
                invalid_args.join(", "),
                valid_fields
            ));
        }

        // Apply each extra argument to the builder
        if let Some(value) = extra_args.get("num_gpu_blocks")
            && let Some(num) = value.as_u64()
        {
            builder = builder.num_gpu_blocks(num as usize);
        }

        if let Some(value) = extra_args.get("block_size")
            && let Some(num) = value.as_u64()
        {
            builder = builder.block_size(num as usize);
        }

        if let Some(value) = extra_args.get("max_num_seqs")
            && let Some(num) = value.as_u64()
        {
            builder = builder.max_num_seqs(Some(num as usize));
        }

        if let Some(value) = extra_args.get("max_num_batched_tokens")
            && let Some(num) = value.as_u64()
        {
            builder = builder.max_num_batched_tokens(Some(num as usize));
        }

        if let Some(value) = extra_args.get("enable_prefix_caching")
            && let Some(enabled) = value.as_bool()
        {
            builder = builder.enable_prefix_caching(enabled);
        }

        if let Some(value) = extra_args.get("enable_chunked_prefill")
            && let Some(enabled) = value.as_bool()
        {
            builder = builder.enable_chunked_prefill(enabled);
        }

        if let Some(value) = extra_args.get("watermark")
            && let Some(num) = value.as_f64()
        {
            builder = builder.watermark(num);
        }

        if let Some(value) = extra_args.get("speedup_ratio")
            && let Some(num) = value.as_f64()
        {
            builder = builder.speedup_ratio(num);
        }

        if let Some(value) = extra_args.get("dp_size")
            && let Some(num) = value.as_u64()
        {
            builder = builder.dp_size(num as u32);
        }

        if let Some(value) = extra_args.get("startup_time")
            && let Some(num) = value.as_f64()
        {
            builder = builder.startup_time(Some(num));
        }

        if let Some(value) = extra_args.get("enable_local_indexer")
            && let Some(enabled) = value.as_bool()
        {
            builder = builder.enable_local_indexer(enabled);
        }

        if let Some(value) = extra_args.get("bootstrap_port")
            && let Some(port) = value.as_u64()
        {
            builder = builder.bootstrap_port(Some(port as u16));
        }

        // Parse worker type from is_prefill and is_decode flags
        let is_prefill = extra_args
            .get("is_prefill")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let is_decode = extra_args
            .get("is_decode")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Determine worker type based on flags
        let worker_type = match (is_prefill, is_decode) {
            (false, false) => WorkerType::Aggregated,
            (true, false) => WorkerType::Prefill,
            (false, true) => WorkerType::Decode,
            (true, true) => panic!(
                "Invalid worker configuration: is_prefill and is_decode cannot both be true. \
                 Worker must be either Aggregated (both false), Prefill (is_prefill=true), or Decode (is_decode=true)."
            ),
        };
        builder = builder.worker_type(worker_type);

        // Load performance model from NPZ file if provided
        let perf_model = if let Some(path_str) = extra_args.get("planner_profile_data")
            && let Some(path_str) = path_str.as_str()
        {
            let npz_path = PathBuf::from(path_str);
            match PerfModel::from_npz(&npz_path) {
                Ok(model) => {
                    tracing::info!("Successfully loaded performance model from: {:?}", npz_path);
                    Arc::new(model)
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to load performance model from {:?}: {}. Falling back to polynomial model.",
                        npz_path,
                        e
                    );
                    Arc::new(PerfModel::default())
                }
            }
        } else {
            Arc::new(PerfModel::default())
        };
        builder = builder.perf_model(perf_model);

        // Build the MockEngineArgs with either defaults or overridden values
        builder
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build MockEngineArgs: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_block_default_uniqueness() {
        // Create 10 default UniqueBlock instances
        let blocks: Vec<UniqueBlock> = (0..10).map(|_| UniqueBlock::default()).collect();

        // Extract UUIDs from each block
        let mut uuids = Vec::new();
        for block in blocks {
            match block {
                UniqueBlock::PartialBlock(uuid) => uuids.push(uuid),
                _ => panic!("Expected UuidIdentifier variant"),
            }
        }

        // Check that all UUIDs are unique by comparing each with every other
        for i in 0..uuids.len() {
            for j in i + 1..uuids.len() {
                assert_ne!(
                    uuids[i], uuids[j],
                    "UUID at index {} and {} are identical",
                    i, j
                );
            }
        }
    }
}
