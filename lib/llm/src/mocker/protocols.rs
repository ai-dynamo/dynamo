// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use uuid::Uuid;

use crate::tokens::blocks::UniqueBlock;
use crate::tokens::{BlockHash, SequenceHash, Token};

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
    pub fn predict_prefill_compute(&self, new_tokens: Option<usize>) -> f64 {
        let tokens = new_tokens.unwrap_or(self.new_tokens);
        4.209989e-07 * (tokens as f64).powi(2) + 1.518344e-02 * (tokens as f64) + 1.650142e+01
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
