// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};

use super::ReplayRouterMode;
use crate::common::protocols::{MockEngineArgs, WorkerType};

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
