// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_custom_policy_builtin::DefaultWorkerSelector;
use std::collections::HashMap;
use std::sync::Arc;

use crate::common::protocols::{MockerConfig, WorkerType};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{
    ActiveSequenceEvent, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::queue::DEFAULT_MAX_BATCHED_TOKENS;
use dynamo_kv_router::sequences::{LocalWorkerLoad, SchedulerLoadSnapshot};
use dynamo_kv_router::{ActiveSequencesMultiWorker, LocalScheduler, SequencePublisher};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ReplayNoopPublisher;

impl SequencePublisher for ReplayNoopPublisher {
    fn enqueue_event(&self, _event: ActiveSequenceEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn publish_scheduler_load(&self, _load: SchedulerLoadSnapshot) {}

    fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: LocalWorkerLoad) {}
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

pub(super) type ReplayScheduler =
    LocalScheduler<ReplayNoopPublisher, ReplayWorkerConfig, DefaultWorkerSelector>;

pub(in crate::replay) fn replay_worker_config(args: &MockerConfig) -> ReplayWorkerConfig {
    ReplayWorkerConfig {
        max_num_batched_tokens: if args.max_num_batched_tokens == usize::MAX {
            DEFAULT_MAX_BATCHED_TOKENS
        } else {
            args.max_num_batched_tokens as u64
        },
        total_kv_blocks: args.num_gpu_blocks as u64,
        data_parallel_start_rank: 0,
        data_parallel_size: args.dp_size.max(1),
    }
}

pub(super) fn replay_workers_with_configs(
    args: &MockerConfig,
    num_workers: usize,
) -> HashMap<WorkerId, ReplayWorkerConfig> {
    let worker_config = replay_worker_config(args);
    (0..num_workers)
        .map(|worker_idx| (worker_idx as WorkerId, worker_config.clone()))
        .collect()
}

pub(super) fn replay_slots(
    args: &MockerConfig,
    workers_with_configs: &HashMap<WorkerId, ReplayWorkerConfig>,
) -> Arc<ActiveSequencesMultiWorker<ReplayNoopPublisher>> {
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
        ReplayNoopPublisher,
        args.block_size,
        dp_range,
        false,
        0,
        "replay",
    ))
}

/// Map the engine's role onto the router's worker type so per-role `prefill:` / `decode:`
/// policy selections apply to the matching replay pool.
pub(super) fn replay_role(worker_type: WorkerType) -> dynamo_kv_router::WorkerType {
    match worker_type {
        WorkerType::Prefill => dynamo_kv_router::WorkerType::Prefill,
        WorkerType::Decode => dynamo_kv_router::WorkerType::Decode,
        _ => dynamo_kv_router::WorkerType::Aggregated,
    }
}

pub(super) fn replay_selector(
    config: &KvRouterConfig,
    role: dynamo_kv_router::WorkerType,
) -> anyhow::Result<DefaultWorkerSelector> {
    replay_selector_with_seed(config, None, role)
}

pub(super) fn replay_selector_with_seed(
    config: &KvRouterConfig,
    selector_seed: Option<u64>,
    role: dynamo_kv_router::WorkerType,
) -> anyhow::Result<DefaultWorkerSelector> {
    if config.request_classifier_config()?.is_some() {
        anyhow::bail!("offline replay does not support request_classifier plugins");
    }
    if let Some(instance) = config
        .selected_worker_selection_policy_instance_for(role)
        .map_err(anyhow::Error::from)?
    {
        let mut registry = dynamo_custom_policy_builtin::default_registry();
        dynamo_custom_policy_builtin::register(&mut registry)?;
        let factory = registry
            .resolve_for_worker_type(config, role)
            .map_err(anyhow::Error::from)?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "custom worker-selection policy {instance:?} is not supported by replay"
                )
            })?;
        return Ok(DefaultWorkerSelector::from_custom_factory(
            config.clone(),
            "replay",
            role,
            factory,
        ));
    }

    Ok(match selector_seed {
        #[cfg(feature = "replay-bench")]
        Some(seed) => DefaultWorkerSelector::new_seeded(Some(config.clone()), "replay", seed),
        #[cfg(not(feature = "replay-bench"))]
        Some(_) => unreachable!("canonical KV Router replay requires the replay-bench feature"),
        None => DefaultWorkerSelector::new(Some(config.clone()), "replay"),
    })
}

pub(crate) fn replay_router_config(
    args: &MockerConfig,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = router_config.unwrap_or_default();
    if let Some(policy) = args.runtime.router_queue_policy {
        config.router_queue_policy = policy;
    }
    config
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
        let Err(error) = replay_selector(&config, dynamo_kv_router::WorkerType::Aggregated) else {
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

        let error = match replay_selector(&config, dynamo_kv_router::WorkerType::Aggregated) {
            Err(error) => error,
            Ok(_) => panic!("replay must reject a custom worker selector it cannot execute"),
        };
        assert!(error.to_string().contains("test"), "{error:#}");
    }

    #[test]
    fn replay_selector_accepts_shipped_custom_worker_selection() {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            policy_file.path(),
            r#"
worker_selection:
  aggregated: ep
  instances:
    - name: ep
      type: dynamo-sglang-ep-balance
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };
        replay_selector(&config, dynamo_kv_router::WorkerType::Aggregated).unwrap();
    }
}
