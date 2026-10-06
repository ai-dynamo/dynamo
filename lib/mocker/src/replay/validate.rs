// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};

use super::{OfflineDisaggReplayConfig, ReplayArgsMode};
use crate::common::protocols::{MockerConfig, WorkerType};

pub fn validate_replay_args_mode(
    aggregated_args: Option<&MockerConfig>,
    prefill_args: Option<&MockerConfig>,
    decode_args: Option<&MockerConfig>,
    num_workers: usize,
    num_prefill_workers: usize,
    num_decode_workers: usize,
) -> Result<ReplayArgsMode> {
    if aggregated_args.is_some() && (prefill_args.is_some() || decode_args.is_some()) {
        bail!("extra_engine_args cannot be combined with prefill_engine_args/decode_engine_args");
    }

    match (aggregated_args, prefill_args, decode_args) {
        (Some(_), None, None) | (None, None, None) => {
            if num_prefill_workers != 1 || num_decode_workers != 1 {
                bail!(
                    "num_prefill_workers and num_decode_workers are only used for disagg replay; use num_workers for aggregated replay"
                );
            }
            Ok(ReplayArgsMode::Aggregated)
        }
        (None, Some(_), Some(_)) => {
            if num_workers != 1 {
                bail!(
                    "num_workers is only used for aggregated replay; use num_prefill_workers and num_decode_workers for disagg replay"
                );
            }
            Ok(ReplayArgsMode::Disagg)
        }
        (None, Some(_), None) | (None, None, Some(_)) => {
            bail!("prefill_engine_args and decode_engine_args must be provided together")
        }
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => unreachable!(),
    }
}

fn validate_aggregated_worker(args: &MockerConfig, mode: &str) -> Result<()> {
    if args.worker_type != WorkerType::Aggregated {
        bail!(
            "{mode} only supports aggregated workers, got {:?}",
            args.worker_type,
        );
    }
    Ok(())
}

// Engine and topology validation belong to AISimulate's ReplaySpec/engine factory.
// Keep only the argument roles that Dynamo lowers into those contracts here.
pub(super) fn validate_offline_replay_args(args: &MockerConfig) -> Result<()> {
    validate_aggregated_worker(args, "offline replay")
}

pub(super) fn validate_online_replay_args(args: &MockerConfig, num_workers: usize) -> Result<()> {
    if num_workers == 0 {
        bail!("online replay requires num_workers >= 1");
    }
    validate_aggregated_worker(args, "online replay")
}

pub(super) fn validate_online_concurrency_args(
    args: &MockerConfig,
    num_workers: usize,
    max_in_flight: usize,
) -> Result<()> {
    if max_in_flight == 0 {
        bail!("online concurrency replay requires max_in_flight >= 1");
    }
    validate_online_replay_args(args, num_workers)
}

pub(super) fn validate_offline_disagg_replay_args(
    config: &OfflineDisaggReplayConfig,
) -> Result<()> {
    let mode = "offline disaggregated replay";
    if config.prefill_args.worker_type != WorkerType::Prefill {
        bail!(
            "{mode} requires prefill_engine_args.worker_type=prefill, got {:?}",
            config.prefill_args.worker_type,
        );
    }
    if config.decode_args.worker_type != WorkerType::Decode {
        bail!(
            "{mode} requires decode_engine_args.worker_type=decode, got {:?}",
            config.decode_args.worker_type,
        );
    }
    // TODO(aisimulate): validate per-role block geometry and reblock workload
    // hashes/handoff metadata when prefill and decode use different sizes.
    // The adapter creates one workload/hash stream at the prefill block size
    // and shares it with the decode router. Unequal block sizes need reblocking.
    if config.prefill_args.block_size != config.decode_args.block_size {
        bail!(
            "{mode} requires matching prefill/decode block_size, got {} and {}",
            config.prefill_args.block_size,
            config.decode_args.block_size,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> OfflineDisaggReplayConfig {
        OfflineDisaggReplayConfig {
            prefill_args: MockerConfig::from_value(serde_json::json!({
                "engine": {
                    "worker_type": WorkerType::Prefill
                }
            }))
            .unwrap(),
            decode_args: MockerConfig::from_value(serde_json::json!({
                "engine": {
                    "worker_type": WorkerType::Decode
                }
            }))
            .unwrap(),
            num_prefill_workers: 1,
            num_decode_workers: 1,
        }
    }

    #[test]
    fn offline_replay_rejects_inconsistent_worker_roles() {
        let mut config = config();
        assert!(validate_offline_replay_args(&config.prefill_args).is_err());
        config.prefill_args.worker_type = WorkerType::Aggregated;
        assert!(validate_offline_disagg_replay_args(&config).is_err());
        config.prefill_args.worker_type = WorkerType::Prefill;
        config.decode_args.worker_type = WorkerType::Aggregated;
        assert!(validate_offline_disagg_replay_args(&config).is_err());
    }

    #[test]
    fn disagg_requires_one_block_size_for_the_shared_workload() {
        let mut config = config();
        config.decode_args.block_size *= 2;
        let error = validate_offline_disagg_replay_args(&config).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("matching prefill/decode block_size")
        );
    }

    #[test]
    fn online_replay_accepts_attention_dp() {
        let args = MockerConfig::from_value(serde_json::json!({"dp_size":2})).unwrap();
        validate_online_replay_args(&args, 1).unwrap();
        validate_online_concurrency_args(&args, 1, 1).unwrap();
    }
}
