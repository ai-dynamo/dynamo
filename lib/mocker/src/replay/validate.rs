// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};

use super::ReplayRouterMode;
use crate::common::protocols::{EngineType, MockEngineArgs, WorkerType};

fn validate_replay_args(args: &MockEngineArgs, num_workers: usize, mode: &str) -> Result<()> {
    if num_workers == 0 {
        bail!("{mode} requires num_workers >= 1");
    }
    if args.engine_type != EngineType::Vllm {
        bail!(
            "{mode} only supports engine_type=vllm, got {:?}",
            args.engine_type,
        );
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

fn validate_router_mode(router_mode: ReplayRouterMode, mode: &str) -> Result<()> {
    if router_mode == ReplayRouterMode::KvRouter {
        bail!("{mode} only supports router_mode=round_robin");
    }
    Ok(())
}

pub(super) fn validate_offline_replay_args(
    args: &MockEngineArgs,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<()> {
    validate_router_mode(router_mode, "offline replay")?;
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

    validate_router_mode(router_mode, "offline replay")?;
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
