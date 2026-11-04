// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::replay::default_selector::DefaultWorkerSelector;
use dynamo_worker_selection::host::SelectionConfig;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::common::protocols::MockerConfig;
use dynamo_kv_hints::{
    KvHintTransferWorkerMetadata, WorkerKvTransferKeys, WorkerKvTransferKeysCache,
};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_router_protocol::worker::DEFAULT_MAX_BATCHED_TOKENS;
use dynamo_router_protocol::worker::{DpRank, WorkerId, WorkerWithDpRank};
use dynamo_scheduler::host::{LocalScheduler, WorkerConfigLike, WorkerKvTransferKeysExt};
use dynamo_slot_tracker::host::{
    ActiveSequenceEvent, ActiveSequencesMultiWorker, LocalWorkerLoad, SchedulerLoadSnapshot,
    SequencePublisher,
};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ReplayNoopPublisher;

impl SequencePublisher for ReplayNoopPublisher {
    fn enqueue_event(&self, _event: ActiveSequenceEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn publish_scheduler_load(&self, _load: SchedulerLoadSnapshot) {}

    fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: LocalWorkerLoad) {}
}

/// KV fetch hints pair only workers of one type; a replay router serves one pool.
/// Replay workers report no `kv_transfer_domain` or `topology_domains`, so
/// production's island rules never apply offline: replay is one transfer island.
const ROUTER_HINT_WORKER_TYPE: &str = "replay";
/// Marks a rank as a KV fetch source; replay never dials it.
const ROUTER_HINT_ENDPOINT: &str = "aisim-replay://worker";

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ReplayWorkerConfig {
    pub(super) max_num_batched_tokens: u64,
    pub(super) total_kv_blocks: u64,
    pub(super) data_parallel_start_rank: u32,
    pub(super) data_parallel_size: u32,
    hint_capable: bool,
    /// Derived from the fields above on first use.
    kv_transfer_keys: WorkerKvTransferKeysCache,
}

impl ReplayWorkerConfig {
    /// A worker with these limits that does or does not receive KV fetch hints
    /// and serve as their source.
    pub(super) fn with_router_hint(&self, router_hint: bool) -> Self {
        Self {
            hint_capable: router_hint,
            kv_transfer_keys: WorkerKvTransferKeysCache::default(),
            ..*self
        }
    }

    pub(super) fn is_hint_capable(&self) -> bool {
        self.hint_capable
    }
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

    fn kv_hint_transfer_metadata_for_dp_rank(
        &self,
        dp_rank: DpRank,
    ) -> Option<KvHintTransferWorkerMetadata<'_>> {
        let rank = dp_rank.checked_sub(self.data_parallel_start_rank)?;
        if !self.hint_capable || rank >= self.data_parallel_size {
            return None;
        }
        Some(KvHintTransferWorkerMetadata {
            worker_type: ROUTER_HINT_WORKER_TYPE,
            source_control_endpoint: Some(ROUTER_HINT_ENDPOINT),
        })
    }

    fn kv_transfer_keys(&self) -> Cow<'_, WorkerKvTransferKeys> {
        Cow::Borrowed(
            self.kv_transfer_keys
                .get_or_init(|| WorkerKvTransferKeys::of(self)),
        )
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
        hint_capable: false,
        kv_transfer_keys: WorkerKvTransferKeysCache::default(),
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

pub(super) fn replay_selector(config: &KvRouterConfig) -> anyhow::Result<DefaultWorkerSelector> {
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
        Some(seed) => {
            DefaultWorkerSelector::new_seeded(Some(SelectionConfig::from(config)), "replay", seed)
        }
        None => DefaultWorkerSelector::new(Some(SelectionConfig::from(config)), "replay"),
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
    fn router_hint_metadata_covers_each_rank_and_keys_are_built_once() {
        let template = ReplayWorkerConfig {
            data_parallel_start_rank: 2,
            data_parallel_size: 2,
            ..replay_worker_config(&MockerConfig::default())
        };
        assert_eq!(template.kv_hint_transfer_metadata_for_dp_rank(2), None);
        assert!(template.kv_transfer_keys().target().is_none());
        let config = template.with_router_hint(true);
        assert!(config.is_hint_capable());
        let endpoint = |rank| {
            config
                .kv_hint_transfer_metadata_for_dp_rank(rank)
                .map(|metadata| (metadata.worker_type, metadata.source_control_endpoint))
        };
        assert_eq!(endpoint(2), Some(("replay", Some(ROUTER_HINT_ENDPOINT))));
        assert_eq!(endpoint(3), Some(("replay", Some(ROUTER_HINT_ENDPOINT))));
        // Ranks outside the worker's range get no metadata.
        assert_eq!(endpoint(1), None);
        assert_eq!(endpoint(4), None);
        // The hint config derives its own keys, not the template's cached ones,
        // and selections borrow them.
        let keys = config.kv_transfer_keys();
        assert!(matches!(keys, Cow::Borrowed(_)));
        assert_eq!(*keys, WorkerKvTransferKeys::of(&config));
        assert!(keys.target().is_some());
    }

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
