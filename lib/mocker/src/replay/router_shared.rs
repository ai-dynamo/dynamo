// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_custom_policy_builtin::DefaultWorkerSelector;
use std::collections::HashMap;
use std::sync::Arc;

use crate::common::protocols::{MockEngineArgs, WorkerType as EngineWorkerType};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{WorkerInputs, WorkerSelectionPolicy};
use dynamo_kv_router::protocols::{
    ActiveSequenceEvent, WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::queue::DEFAULT_MAX_BATCHED_TOKENS;
use dynamo_kv_router::sequences::SchedulerLoadSnapshot;
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, KvSchedulerError, LocalScheduler, RoutingPartitionRef,
    SequencePublisher, WorkerSelectionInput, WorkerSelector, WorkerType,
};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ReplayNoopPublisher;

impl SequencePublisher for ReplayNoopPublisher {
    fn enqueue_event(&self, _event: ActiveSequenceEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn publish_scheduler_load(&self, _load: SchedulerLoadSnapshot) {}

    fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: usize, _: usize) {}
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
    LocalScheduler<ReplayNoopPublisher, ReplayWorkerConfig, ReplaySelector>;

/// Dynamo's default selector, or a worker-selection policy from the builtin catalog.
// Each replay router holds one long-lived selector, so the variant size gap costs nothing.
#[allow(clippy::large_enum_variant)]
pub(super) enum ReplaySelector {
    Default(DefaultWorkerSelector),
    Policy(WorkerSelectionPolicy),
}

impl<C: WorkerConfigLike + Sync> WorkerSelector<C> for ReplaySelector {
    fn required_worker_inputs(&self) -> WorkerInputs {
        match self {
            Self::Default(selector) => WorkerSelector::<C>::required_worker_inputs(selector),
            Self::Policy(policy) => WorkerSelector::<C>::required_worker_inputs(policy),
        }
    }

    fn uses_exclusive_affinity_target(&self) -> bool {
        match self {
            Self::Default(selector) => {
                WorkerSelector::<C>::uses_exclusive_affinity_target(selector)
            }
            Self::Policy(policy) => WorkerSelector::<C>::uses_exclusive_affinity_target(policy),
        }
    }

    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        match self {
            Self::Default(selector) => selector.select_worker(input),
            Self::Policy(policy) => policy.select_worker(input),
        }
    }
}

pub(in crate::replay) fn replay_worker_config(args: &MockEngineArgs) -> ReplayWorkerConfig {
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

/// Replay validation guarantees each pool's engine role, so it also names the router's role.
pub(super) fn replay_router_role(args: &MockEngineArgs) -> WorkerType {
    match args.worker_type {
        EngineWorkerType::Aggregated => WorkerType::Aggregated,
        EngineWorkerType::Prefill => WorkerType::Prefill,
        EngineWorkerType::Decode => WorkerType::Decode,
    }
}

pub(super) fn replay_workers_with_configs(
    args: &MockEngineArgs,
    num_workers: usize,
) -> HashMap<WorkerId, ReplayWorkerConfig> {
    let worker_config = replay_worker_config(args);
    (0..num_workers)
        .map(|worker_idx| (worker_idx as WorkerId, worker_config.clone()))
        .collect()
}

pub(super) fn replay_slots(
    args: &MockEngineArgs,
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

pub(super) fn replay_selector(
    config: &KvRouterConfig,
    worker_type: WorkerType,
) -> anyhow::Result<ReplaySelector> {
    replay_selector_with_seed(config, None, worker_type)
}

/// The seed applies only to Dynamo's default selector; a catalog policy owns its randomness.
pub(super) fn replay_selector_with_seed(
    config: &KvRouterConfig,
    selector_seed: Option<u64>,
    worker_type: WorkerType,
) -> anyhow::Result<ReplaySelector> {
    if config.request_classifier_config()?.is_some() {
        anyhow::bail!("offline replay does not support request_classifier plugins");
    }
    if let Some(instance) = config
        .selected_worker_selection_policy_instance_for(worker_type)
        .map_err(anyhow::Error::from)?
        .filter(|instance| instance != "default")
    {
        let mut registry = dynamo_custom_policy_builtin::default_registry();
        dynamo_custom_policy_builtin::register(&mut registry)?;
        // One self-contained message: Python bindings display only the outermost error.
        let factory = registry
            .resolve_for_worker_type(config, worker_type)
            .map_err(|error| {
                anyhow::anyhow!("resolving worker-selection policy {instance:?}: {error}")
            })?
            .ok_or_else(|| {
                anyhow::anyhow!("worker-selection policy {instance:?} did not resolve")
            })?;
        return Ok(ReplaySelector::Policy(factory(
            config,
            worker_type,
            RoutingPartitionRef::new("replay", "default"),
        )));
    }

    Ok(ReplaySelector::Default(match selector_seed {
        #[cfg(feature = "replay-bench")]
        Some(seed) => DefaultWorkerSelector::new_seeded(Some(config.clone()), "replay", seed),
        #[cfg(not(feature = "replay-bench"))]
        Some(_) => unreachable!("canonical KV Router replay requires the replay-bench feature"),
        None => DefaultWorkerSelector::new(Some(config.clone()), "replay"),
    }))
}

pub(crate) fn replay_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = router_config.unwrap_or_default();
    if let Some(policy) = args.router_queue_policy {
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
        let Err(error) = replay_selector(&config, WorkerType::Aggregated) else {
            panic!("classifier ignored")
        };
        assert!(
            error
                .to_string()
                .contains("offline replay does not support request_classifier")
        );
    }

    fn policy_config(yaml: &str) -> (tempfile::NamedTempFile, KvRouterConfig) {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(policy_file.path(), yaml).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };
        (policy_file, config)
    }

    #[test]
    fn replay_selector_resolves_catalog_policies_per_role() {
        let (_file, config) = policy_config(
            r#"
worker_selection:
  prefill: two-tier
  instances:
    - name: two-tier
      type: dynamo-two-tier-cost-fn
"#,
        );

        assert!(matches!(
            replay_selector(&config, WorkerType::Prefill).unwrap(),
            ReplaySelector::Policy(_)
        ));
        assert!(matches!(
            replay_selector(&config, WorkerType::Aggregated).unwrap(),
            ReplaySelector::Default(_)
        ));
    }

    #[test]
    fn replay_selector_rejects_a_policy_type_outside_the_builtin_catalog() {
        let (_file, config) = policy_config(
            r#"
worker_selection:
  aggregated: custom
  instances:
    - name: custom
      type: test
      parameters: {}
"#,
        );

        let Err(error) = replay_selector(&config, WorkerType::Aggregated) else {
            panic!("replay must reject a worker-selection policy it cannot build");
        };
        assert!(
            format!("{error:#}").contains("test"),
            "the error should name the unknown policy type: {error:#}"
        );
    }
}
