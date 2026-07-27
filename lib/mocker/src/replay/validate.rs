// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};

use super::{OfflineDisaggReplayConfig, ReplayArgsMode, ReplayRouterMode};
use crate::common::protocols::{EngineType, MockEngineArgs, WorkerType};

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

fn validate_replay_args(
    args: &MockEngineArgs,
    num_workers: usize,
    mode: &str,
    allow_dp_replication: bool,
) -> Result<()> {
    if num_workers == 0 {
        bail!("{mode} requires num_workers >= 1");
    }
    if args.worker_type != WorkerType::Aggregated {
        bail!(
            "{mode} only supports aggregated workers, got {:?}",
            args.worker_type,
        );
    }
    // Offline replay treats dp_size>1 as rank topology: each mocker worker gets
    // that many independent scheduler/KV-pool states, mirroring the live path.
    // Online replay does not support it.
    if args.dp_size != 1 && !allow_dp_replication {
        bail!(
            "{mode} only supports data_parallel_size=1, got {}",
            args.dp_size,
        );
    }

    Ok(())
}

fn validate_offline_router_mode(
    router_mode: ReplayRouterMode,
    num_workers: usize,
    dp_size: u32,
) -> Result<()> {
    if router_mode != ReplayRouterMode::KvRouter {
        return Ok(());
    }
    if num_workers.saturating_mul(dp_size.max(1) as usize) > 1 {
        return Ok(());
    }

    bail!(
        "offline replay only supports router_mode=kv_router with more than one worker/DP-rank target"
    );
}

pub(super) fn validate_offline_replay_args(
    args: &MockEngineArgs,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<()> {
    validate_offline_router_mode(router_mode, num_workers, args.dp_size)?;
    validate_replay_args(args, num_workers, "trace replay", true)
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

    validate_offline_router_mode(router_mode, num_workers, args.dp_size)?;
    validate_replay_args(args, num_workers, "concurrency replay", true)
}

pub(super) fn validate_online_replay_args(args: &MockEngineArgs, num_workers: usize) -> Result<()> {
    validate_replay_args(args, num_workers, "online replay", false)
}

pub(super) fn validate_online_concurrency_args(
    args: &MockEngineArgs,
    num_workers: usize,
    max_in_flight: usize,
) -> Result<()> {
    if max_in_flight == 0 {
        bail!("online concurrency replay requires max_in_flight >= 1");
    }

    validate_replay_args(args, num_workers, "online replay", false)
}

fn validate_disagg_args(config: &OfflineDisaggReplayConfig, mode: &str) -> Result<()> {
    if config.prefill_args.engine_type == EngineType::Trtllm
        || config.decode_args.engine_type == EngineType::Trtllm
    {
        bail!("{mode} disaggregation does not support TRT-LLM");
    }
    if config.prefill_args.engine_type != config.decode_args.engine_type {
        bail!(
            "{mode} disaggregation requires matching prefill/decode engine_type, got {:?} and {:?}",
            config.prefill_args.engine_type,
            config.decode_args.engine_type,
        );
    }
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
    if config.prefill_args.dp_size != 1 || config.decode_args.dp_size != 1 {
        bail!(
            "offline disaggregated replay does not support attention DP; prefill/decode dp_size \
             must both be 1 (got prefill_dp_size={}, decode_dp_size={})",
            config.prefill_args.dp_size,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args(engine_type: EngineType, worker_type: WorkerType) -> MockEngineArgs {
        MockEngineArgs {
            engine_type,
            worker_type,
            block_size: 64,
            ..MockEngineArgs::default()
        }
    }

    fn config(
        prefill_engine_type: EngineType,
        decode_engine_type: EngineType,
    ) -> OfflineDisaggReplayConfig {
        OfflineDisaggReplayConfig {
            prefill_args: args(prefill_engine_type, WorkerType::Prefill),
            decode_args: args(decode_engine_type, WorkerType::Decode),
            num_prefill_workers: 1,
            num_decode_workers: 1,
        }
    }

    #[test]
    fn disagg_rejects_trtllm_on_either_side() {
        for config in [
            config(EngineType::Trtllm, EngineType::Vllm),
            config(EngineType::Vllm, EngineType::Trtllm),
        ] {
            let error = validate_offline_disagg_replay_args(&config, ReplayRouterMode::RoundRobin)
                .unwrap_err();
            assert!(error.to_string().contains("does not support TRT-LLM"));
        }
    }

    #[test]
    fn disagg_rejects_mixed_supported_backends() {
        for config in [
            config(EngineType::Vllm, EngineType::Sglang),
            config(EngineType::Sglang, EngineType::Vllm),
        ] {
            let error = validate_offline_disagg_replay_args(&config, ReplayRouterMode::RoundRobin)
                .unwrap_err();
            assert!(error.to_string().contains("matching prefill/decode"));
        }
    }

    #[test]
    fn disagg_rejects_attention_dp_with_clear_error() {
        for (prefill_dp_size, decode_dp_size) in [(2, 1), (1, 2), (2, 4)] {
            let mut config = config(EngineType::Vllm, EngineType::Vllm);
            config.prefill_args.dp_size = prefill_dp_size;
            config.decode_args.dp_size = decode_dp_size;
            let expected = format!(
                "offline disaggregated replay does not support attention DP; prefill/decode \
                 dp_size must both be 1 (got prefill_dp_size={prefill_dp_size}, \
                 decode_dp_size={decode_dp_size})"
            );

            let trace_error =
                validate_offline_disagg_replay_args(&config, ReplayRouterMode::RoundRobin)
                    .unwrap_err();
            assert_eq!(trace_error.to_string(), expected);

            let concurrency_error =
                validate_offline_disagg_concurrency_args(&config, 1, ReplayRouterMode::RoundRobin)
                    .unwrap_err();
            assert_eq!(concurrency_error.to_string(), expected);
        }
    }

    #[test]
    fn offline_kv_router_accepts_one_worker_with_multiple_dp_ranks() {
        let mut args = args(EngineType::Vllm, WorkerType::Aggregated);
        args.dp_size = 2;
        validate_offline_replay_args(&args, 1, ReplayRouterMode::KvRouter).unwrap();
    }
}
