// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for the G1↔G2 offload simulation.
//!
//! [`KvbmOffloadConfig`] holds the parameters that shape both the PS
//! bandwidth model ([`super::bandwidth_sharing_model::BandwidthSharingModel`]) and the
//! kvbm-engine pipeline topology (G2 capacity, offload batch size).

use crate::common::protocols::MockEngineArgs;

#[derive(Debug, Clone)]
pub struct KvbmOffloadConfig {
    /// Number of G2 blocks to simulate. Sets the capacity of the
    /// `BlockManager<G2>` owned by the engine.
    pub num_g2_blocks: usize,

    /// Tokens per logical KV block. Used as the kvbm-logical block size for
    /// G2 so its block identity matches the engine/G1 block geometry.
    pub block_size_tokens: usize,

    /// Batch size for the G1→G2 pipeline. Offloads are grouped into
    /// batches of this size before being handed to the worker.
    pub offload_batch_size: usize,

    /// Bytes per block — used by `MockWorker` to compute transfer size
    /// from block counts. Typically `block_size_tokens * kv_bytes_per_token`.
    /// `None` means "unknown" and the worker will use `0` as a sentinel
    /// (every transfer completes instantly).
    pub block_size_bytes: Option<usize>,

    /// Throughput of the G1→G2 offload link in GB/s. Non-positive values
    /// mean "infinite bandwidth" (transfers complete instantly).
    pub bandwidth_g1_to_g2_gbps: f64,

    /// Throughput of the G2→G1 onboard link in GB/s. Non-positive values
    /// mean "infinite bandwidth" (transfers complete instantly).
    pub bandwidth_g2_to_g1_gbps: f64,
}

impl Default for KvbmOffloadConfig {
    fn default() -> Self {
        Self {
            num_g2_blocks: 100_000,
            block_size_tokens: 64,
            offload_batch_size: 32,
            block_size_bytes: None,
            bandwidth_g1_to_g2_gbps: 14.0,
            bandwidth_g2_to_g1_gbps: 14.0,
        }
    }
}

impl KvbmOffloadConfig {
    /// Derive an offload config from scheduler-level [`MockEngineArgs`].
    ///
    /// Returns `None` unless both `num_g2_blocks` and `kv_bytes_per_token`
    /// are set. `num_g2_blocks` is the explicit opt-in for the G2 tier;
    /// `kv_bytes_per_token` is required to compute `block_size_bytes`.
    /// Caller should interpret `None` as "don't attach an offload engine
    /// for this run".
    pub fn from_args(args: &MockEngineArgs) -> Option<Self> {
        let num_g2_blocks = args.num_g2_blocks?;
        let bpt = args.kv_bytes_per_token?;
        let defaults = Self::default();
        Some(Self {
            num_g2_blocks,
            block_size_tokens: args.block_size,
            offload_batch_size: args
                .offload_batch_size
                .unwrap_or(defaults.offload_batch_size),
            block_size_bytes: Some(args.block_size * bpt),
            bandwidth_g1_to_g2_gbps: args
                .bandwidth_g1_to_g2_gbps
                .unwrap_or(defaults.bandwidth_g1_to_g2_gbps),
            bandwidth_g2_to_g1_gbps: args
                .bandwidth_g2_to_g1_gbps
                .unwrap_or(defaults.bandwidth_g2_to_g1_gbps),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_args_none_when_kv_bytes_per_token_missing() {
        let args = MockEngineArgs::builder()
            .num_g2_blocks(Some(10_000))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        assert!(args.kv_bytes_per_token.is_none());
        assert!(KvbmOffloadConfig::from_args(&args).is_none());
    }

    #[test]
    fn from_args_none_when_num_g2_blocks_missing() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        assert!(args.num_g2_blocks.is_none());
        assert!(KvbmOffloadConfig::from_args(&args).is_none());
    }

    #[test]
    fn from_args_computes_block_size_bytes() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .num_g2_blocks(Some(10_000))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let cfg = KvbmOffloadConfig::from_args(&args).expect("bpt set");
        assert_eq!(cfg.block_size_bytes, Some(64 * 131_072));
        // Defaults preserved.
        assert_eq!(cfg.num_g2_blocks, 10_000);
        assert_eq!(cfg.block_size_tokens, 64);
        assert_eq!(cfg.offload_batch_size, 32);
        assert_eq!(cfg.bandwidth_g1_to_g2_gbps, 14.0);
        assert_eq!(cfg.bandwidth_g2_to_g1_gbps, 14.0);
    }

    #[test]
    fn from_args_threads_optional_knobs_when_set() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .num_g2_blocks(Some(10_000))
            .offload_batch_size(Some(16))
            .bandwidth_g1_to_g2_gbps(Some(8.0))
            .bandwidth_g2_to_g1_gbps(Some(12.0))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let cfg = KvbmOffloadConfig::from_args(&args).expect("bpt set");
        assert_eq!(cfg.num_g2_blocks, 10_000);
        assert_eq!(cfg.block_size_tokens, 64);
        assert_eq!(cfg.offload_batch_size, 16);
        assert_eq!(cfg.bandwidth_g1_to_g2_gbps, 8.0);
        assert_eq!(cfg.bandwidth_g2_to_g1_gbps, 12.0);
    }
}
