// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for the G1↔G2/G2↔G3 offload simulation.
//!
//! [`KvbmOffloadConfig`] holds the parameters that shape both the PS
//! bandwidth model ([`super::bandwidth_sharing_model::BandwidthSharingModel`]) and the
//! kvbm-engine pipeline topology (G2/G3 capacity, offload batch sizes).

use crate::common::protocols::MockEngineArgs;

const DEFAULT_G1_G2_BANDWIDTH_GBPS: f64 = 14.0;
const DEFAULT_G2_G3_BANDWIDTH_GBPS: f64 = 7.0;

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

    /// Optional number of shared G3 blocks to simulate. When set, the mocker
    /// wires a G2→G3 presence pipeline chained after G1→G2.
    pub num_g3_blocks: Option<usize>,

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

    /// Throughput of the G2→G3 offload link in GB/s. Defaults to a
    /// conservative single local NVMe/SSD bandwidth; non-positive values
    /// mean "infinite bandwidth" (transfers complete instantly).
    pub bandwidth_g2_to_g3_gbps: f64,

    /// Throughput of the G3→G2 staging link in GB/s. Defaults to the same
    /// conservative single local NVMe/SSD bandwidth as G2→G3; non-positive
    /// values mean "infinite bandwidth" (transfers complete instantly).
    pub bandwidth_g3_to_g2_gbps: f64,
}

impl Default for KvbmOffloadConfig {
    fn default() -> Self {
        Self {
            num_g2_blocks: 100_000,
            block_size_tokens: 64,
            offload_batch_size: 32,
            num_g3_blocks: None,
            block_size_bytes: None,
            bandwidth_g1_to_g2_gbps: DEFAULT_G1_G2_BANDWIDTH_GBPS,
            bandwidth_g2_to_g1_gbps: DEFAULT_G1_G2_BANDWIDTH_GBPS,
            bandwidth_g2_to_g3_gbps: DEFAULT_G2_G3_BANDWIDTH_GBPS,
            bandwidth_g3_to_g2_gbps: DEFAULT_G2_G3_BANDWIDTH_GBPS,
        }
    }
}

impl KvbmOffloadConfig {
    /// Derive an offload config from scheduler-level [`MockEngineArgs`].
    ///
    /// Returns `Ok(None)` unless both `num_g2_blocks` and `kv_bytes_per_token`
    /// are set. Positive `num_g2_blocks` is the explicit opt-in for the G2
    /// tier; `kv_bytes_per_token` is required to compute `block_size_bytes`.
    /// Caller should interpret `Ok(None)` as "don't attach an offload engine
    /// for this run".
    pub fn from_args(args: &MockEngineArgs) -> anyhow::Result<Option<Self>> {
        let num_g3_blocks = args
            .num_g3_blocks
            .and_then(|block_count| (block_count > 0).then_some(block_count));
        let Some(num_g2_blocks) = args.num_g2_blocks else {
            if num_g3_blocks.is_some() {
                anyhow::bail!(
                    "num_g3_blocks requires num_g2_blocks because mocker stages G3 through G2"
                );
            }
            return Ok(None);
        };
        if num_g2_blocks == 0 {
            return Ok(None);
        }
        let Some(bpt) = args.kv_bytes_per_token else {
            if num_g3_blocks.is_some() {
                anyhow::bail!(
                    "num_g3_blocks requires kv_bytes_per_token so mocker can size G2/G3 transfers"
                );
            }
            return Ok(None);
        };
        let defaults = Self::default();
        let offload_batch_size = args
            .offload_batch_size
            .filter(|batch_size| *batch_size > 0)
            .unwrap_or(defaults.offload_batch_size);
        Ok(Some(Self {
            num_g2_blocks,
            block_size_tokens: args.block_size,
            offload_batch_size,
            num_g3_blocks,
            block_size_bytes: Some(args.block_size * bpt),
            bandwidth_g1_to_g2_gbps: args
                .bandwidth_g1_to_g2_gbps
                .unwrap_or(defaults.bandwidth_g1_to_g2_gbps),
            bandwidth_g2_to_g1_gbps: args
                .bandwidth_g2_to_g1_gbps
                .unwrap_or(defaults.bandwidth_g2_to_g1_gbps),
            bandwidth_g2_to_g3_gbps: args
                .bandwidth_g2_to_g3_gbps
                .unwrap_or(defaults.bandwidth_g2_to_g3_gbps),
            bandwidth_g3_to_g2_gbps: args
                .bandwidth_g3_to_g2_gbps
                .unwrap_or(defaults.bandwidth_g3_to_g2_gbps),
        }))
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
        assert!(KvbmOffloadConfig::from_args(&args).unwrap().is_none());
    }

    #[test]
    fn from_args_errors_when_g3_kv_bytes_per_token_missing() {
        let args = MockEngineArgs::builder()
            .num_g2_blocks(Some(10_000))
            .num_g3_blocks(Some(20_000))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let error = KvbmOffloadConfig::from_args(&args).unwrap_err();
        assert!(
            error.to_string().contains("requires kv_bytes_per_token"),
            "unexpected error: {error}",
        );
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
        assert!(KvbmOffloadConfig::from_args(&args).unwrap().is_none());
    }

    #[test]
    fn from_args_none_when_num_g2_blocks_zero() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .num_g2_blocks(Some(0))
            .build()
            .unwrap();

        assert!(KvbmOffloadConfig::from_args(&args).unwrap().is_none());
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
        let cfg = KvbmOffloadConfig::from_args(&args)
            .unwrap()
            .expect("bpt set");
        assert_eq!(cfg.block_size_bytes, Some(64 * 131_072));
        // Defaults preserved.
        assert_eq!(cfg.num_g2_blocks, 10_000);
        assert_eq!(cfg.block_size_tokens, 64);
        assert_eq!(cfg.offload_batch_size, 32);
        assert_eq!(cfg.num_g3_blocks, None);
        assert_eq!(cfg.bandwidth_g1_to_g2_gbps, DEFAULT_G1_G2_BANDWIDTH_GBPS);
        assert_eq!(cfg.bandwidth_g2_to_g1_gbps, DEFAULT_G1_G2_BANDWIDTH_GBPS);
        assert_eq!(cfg.bandwidth_g2_to_g3_gbps, DEFAULT_G2_G3_BANDWIDTH_GBPS);
        assert_eq!(cfg.bandwidth_g3_to_g2_gbps, DEFAULT_G2_G3_BANDWIDTH_GBPS);
    }

    #[test]
    fn from_args_threads_optional_knobs_when_set() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .num_g2_blocks(Some(10_000))
            .offload_batch_size(Some(16))
            .num_g3_blocks(Some(20_000))
            .bandwidth_g1_to_g2_gbps(Some(8.0))
            .bandwidth_g2_to_g1_gbps(Some(12.0))
            .bandwidth_g2_to_g3_gbps(Some(3.0))
            .bandwidth_g3_to_g2_gbps(Some(4.0))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let cfg = KvbmOffloadConfig::from_args(&args)
            .unwrap()
            .expect("bpt set");
        assert_eq!(cfg.num_g2_blocks, 10_000);
        assert_eq!(cfg.block_size_tokens, 64);
        assert_eq!(cfg.offload_batch_size, 16);
        assert_eq!(cfg.num_g3_blocks, Some(20_000));
        assert_eq!(cfg.bandwidth_g1_to_g2_gbps, 8.0);
        assert_eq!(cfg.bandwidth_g2_to_g1_gbps, 12.0);
        assert_eq!(cfg.bandwidth_g2_to_g3_gbps, 3.0);
        assert_eq!(cfg.bandwidth_g3_to_g2_gbps, 4.0);
    }

    #[test]
    fn from_args_treats_zero_optional_g3_and_batch_size_as_disabled() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .num_g2_blocks(Some(10_000))
            .num_g3_blocks(Some(0))
            .offload_batch_size(Some(0))
            .build()
            .unwrap();

        let cfg = KvbmOffloadConfig::from_args(&args)
            .unwrap()
            .expect("G2 remains enabled");
        assert_eq!(cfg.num_g2_blocks, 10_000);
        assert_eq!(cfg.num_g3_blocks, None);
        assert_eq!(
            cfg.offload_batch_size,
            KvbmOffloadConfig::default().offload_batch_size
        );
    }
}
