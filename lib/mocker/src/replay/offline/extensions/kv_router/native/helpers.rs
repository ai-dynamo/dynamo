// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for AISimulate: native Router integration with AISim-owned types.
// Derived from https://github.com/ai-dynamo/dynamo/blob/6d6d36d7bfa63d80a0302df637c08f85523aa5b6/lib/mocker/src/replay/router_shared.rs

use dynamo_custom_policy_builtin::DefaultWorkerSelector;
use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerId};
use dynamo_kv_router::scheduling::queue::DEFAULT_MAX_BATCHED_TOKENS;
use dynamo_kv_router::{ActiveSequencesMultiWorker, NoopSequencePublisher};

/// The engine capacity/topology fields needed by the native Router.
#[derive(Clone, Copy, Debug)]
pub(super) struct NativeRouterEngineArgs {
    pub(super) dp_size: u32,
    pub(super) block_size: u32,
    pub(super) num_gpu_blocks: usize,
    pub(super) max_num_batched_tokens: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ReplayWorkerConfig {
    pub(super) max_num_batched_tokens: u64,
    pub(super) total_kv_blocks: u64,
    pub(super) data_parallel_start_rank: u32,
    pub(super) data_parallel_size: u32,
}

impl WorkerConfigLike for ReplayWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        self.data_parallel_start_rank
    }

    fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        Some(self.max_num_batched_tokens)
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        Some(self.total_kv_blocks)
    }
}

pub(super) fn replay_worker_config(args: &NativeRouterEngineArgs) -> ReplayWorkerConfig {
    ReplayWorkerConfig {
        max_num_batched_tokens: args
            .max_num_batched_tokens
            .map(|tokens| tokens as u64)
            .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS),
        total_kv_blocks: args.num_gpu_blocks as u64,
        data_parallel_start_rank: 0,
        data_parallel_size: args.dp_size.max(1),
    }
}

pub(super) fn replay_workers_with_configs(
    args: &NativeRouterEngineArgs,
    num_workers: usize,
) -> HashMap<WorkerId, ReplayWorkerConfig> {
    let worker_config = replay_worker_config(args);
    (0..num_workers)
        .map(|worker_idx| (worker_idx as WorkerId, worker_config.clone()))
        .collect()
}

pub(super) fn replay_slots(
    args: &NativeRouterEngineArgs,
    workers_with_configs: &HashMap<WorkerId, ReplayWorkerConfig>,
) -> Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>> {
    let dp_range = workers_with_configs
        .iter()
        .map(|(&worker_id, config)| {
            (
                worker_id,
                (config.data_parallel_start_rank, config.data_parallel_size),
            )
        })
        .collect();
    // NOTE: Offline replay must retire requests through explicit lifecycle events. Wall-clock
    // expiry is a live-router cleanup heuristic and must not observe simulator CPU time: a
    // healthy replay may spend minutes of wall time advancing seconds of virtual time. Keep
    // expiry disabled here until replay has a liveness-aware definition of a stale request; do
    // not mask replay dead ends by expiring requests that are still live in virtual time.
    Arc::new(ActiveSequencesMultiWorker::new_without_expiry(
        NoopSequencePublisher,
        args.block_size as usize,
        dp_range,
        false,
        0,
        "replay",
    ))
}

#[cfg(test)]
fn replay_selector(config: &KvRouterConfig) -> anyhow::Result<DefaultWorkerSelector> {
    replay_selector_with_seed(config, None)
}

pub(super) fn replay_selector_with_seed(
    config: &KvRouterConfig,
    selector_seed: Option<u64>,
) -> anyhow::Result<DefaultWorkerSelector> {
    if config.request_classifier_config()?.is_some() {
        anyhow::bail!("offline replay does not support request_classifier plugins");
    }
    if let Some(instance) = config
        .selected_worker_selection_policy_instance()
        .map_err(anyhow::Error::from)?
    {
        anyhow::bail!("custom worker-selection policy {instance:?} is not supported by replay");
    }

    Ok(match selector_seed {
        #[cfg(feature = "replay-bench")]
        Some(seed) => DefaultWorkerSelector::new_seeded(Some(config.clone()), "replay", seed),
        #[cfg(not(feature = "replay-bench"))]
        Some(_) => unreachable!("canonical KV Router replay requires the replay-bench feature"),
        None => DefaultWorkerSelector::new(Some(config.clone()), "replay"),
    })
}

pub(super) fn replay_router_config(router_config: Option<KvRouterConfig>) -> KvRouterConfig {
    router_config.unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_selector_rejects_request_classifier() {
        let policy = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(policy.path(), "request_classifier: {type: test}").unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy.path().display().to_string()),
            ..Default::default()
        };
        let Err(error) = replay_selector(&config) else {
            panic!("classifier ignored")
        };
        assert!(
            error
                .to_string()
                .contains("offline replay does not support request_classifier")
        );
    }

    #[test]
    fn replay_selector_rejects_custom_worker_selection() {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            policy_file.path(),
            r#"
worker_selection:
  aggregated: custom
  instances:
    - name: custom
      type: test
      parameters: {}
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };

        let error = match replay_selector(&config) {
            Err(error) => error,
            Ok(_) => panic!("replay must reject a custom worker selector it cannot execute"),
        };
        assert!(
            error
                .to_string()
                .contains("custom worker-selection policy \"custom\" is not supported by replay")
        );
    }
}
