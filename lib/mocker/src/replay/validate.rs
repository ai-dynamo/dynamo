// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};

use crate::common::protocols::{EngineType, MockEngineArgs, WorkerType};

pub(super) fn validate_offline_replay_args(
    args: &MockEngineArgs,
    num_workers: usize,
) -> Result<()> {
    if num_workers == 0 {
        bail!("trace replay requires num_workers >= 1");
    }
    if args.engine_type != EngineType::Vllm {
        bail!(
            "trace replay only supports engine_type=vllm, got {:?}",
            args.engine_type
        );
    }
    if args.worker_type != WorkerType::Aggregated {
        bail!(
            "trace replay only supports aggregated workers, got {:?}",
            args.worker_type
        );
    }
    if args.dp_size != 1 {
        bail!(
            "trace replay only supports data_parallel_size=1, got {}",
            args.dp_size
        );
    }

    Ok(())
}

pub(super) fn validate_offline_concurrency_args(
    args: &MockEngineArgs,
    num_workers: usize,
    max_in_flight: usize,
) -> Result<()> {
    if max_in_flight == 0 {
        bail!("concurrency replay requires max_in_flight >= 1");
    }

    validate_offline_replay_args(args, num_workers)
}
