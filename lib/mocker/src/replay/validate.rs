// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};

use super::{OfflineDisaggReplayConfig, ReplayArgsMode, ReplayRouterMode};
use crate::common::protocols::{MockEngineArgs, WorkerType};

pub fn validate_replay_args_mode(
    aggregated_args: Option<&MockEngineArgs>,
    prefill_args: Option<&MockEngineArgs>,
    decode_args: Option<&MockEngineArgs>,
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

fn validate_replay_args(args: &MockEngineArgs, num_workers: usize, mode: &str) -> Result<()> {
    if num_workers == 0 {
        bail!("{mode} requires num_workers >= 1");
    }
    if args.worker_type != WorkerType::Aggregated {
        bail!(
            "{mode} only supports aggregated workers, got {:?}",
            args.worker_type,
        );
    }
    if args.dp_size != 1 {
        bail!(
            "{mode} only supports data_parallel_size=1, got {}",
            args.dp_size,
        );
    }

    Ok(())
}

fn validate_offline_router_mode(router_mode: ReplayRouterMode, num_workers: usize) -> Result<()> {
    if router_mode != ReplayRouterMode::KvRouter {
        return Ok(());
    }
    if num_workers > 1 {
        return Ok(());
    }

    bail!("offline replay only supports router_mode=kv_router when num_workers > 1");
}

pub(super) fn validate_offline_replay_args(
    args: &MockEngineArgs,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<()> {
    validate_offline_router_mode(router_mode, num_workers)?;
    validate_replay_args(args, num_workers, "trace replay")
}

pub(super) fn validate_offline_concurrency_args(
    args: &MockEngineArgs,
    num_workers: usize,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> Result<()> {
    if max_in_flight == 0 {
        bail!("concurrency replay requires max_in_flight >= 1");
    }

    validate_offline_router_mode(router_mode, num_workers)?;
    validate_replay_args(args, num_workers, "concurrency replay")
}

pub(super) fn validate_online_replay_args(args: &MockEngineArgs, num_workers: usize) -> Result<()> {
    validate_replay_args(args, num_workers, "online replay")
}

pub(super) fn validate_online_concurrency_args(
    args: &MockEngineArgs,
    num_workers: usize,
    max_in_flight: usize,
) -> Result<()> {
    if max_in_flight == 0 {
        bail!("online concurrency replay requires max_in_flight >= 1");
    }

    validate_replay_args(args, num_workers, "online replay")
}

fn validate_disagg_args(config: &OfflineDisaggReplayConfig, mode: &str) -> Result<()> {
    if config.num_prefill_workers == 0 {
        bail!("{mode} requires num_prefill_workers >= 1");
    }
    if config.num_decode_workers == 0 {
        bail!("{mode} requires num_decode_workers >= 1");
    }
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
    if config.prefill_args.dp_size != 1 {
        bail!(
            "{mode} only supports prefill data_parallel_size=1, got {}",
            config.prefill_args.dp_size,
        );
    }
    if config.decode_args.dp_size != 1 {
        bail!(
            "{mode} only supports decode data_parallel_size=1, got {}",
            config.decode_args.dp_size,
        );
    }
    if config.prefill_args.block_size != config.decode_args.block_size {
        bail!(
            "{mode} requires matching prefill/decode block_size, got {} and {}",
            config.prefill_args.block_size,
            config.decode_args.block_size,
        );
    }

    Ok(())
}

pub(super) fn validate_offline_disagg_replay_args(
    config: &OfflineDisaggReplayConfig,
    _router_mode: ReplayRouterMode,
) -> Result<()> {
    validate_disagg_args(config, "trace replay")
}

pub(super) fn validate_offline_disagg_concurrency_args(
    config: &OfflineDisaggReplayConfig,
    max_in_flight: usize,
    _router_mode: ReplayRouterMode,
) -> Result<()> {
    if max_in_flight == 0 {
        bail!("concurrency replay requires max_in_flight >= 1");
    }
    validate_disagg_args(config, "concurrency replay")
}
