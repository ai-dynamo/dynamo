// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities. The indexer event builders live in
//! `dynamo_kv_indexer::testing` and are re-exported here by name.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use dynamo_admission::PolicyProfile;
use dynamo_router_protocol::worker::WorkerId;
use dynamo_slot_tracker::ActiveSequencesMultiWorker;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::protocols::WorkerConfigLike;
use crate::selector::DefaultWorkerSelector;
pub use dynamo_kv_hashing::routing::{
    BlockHashOptions, compute_block_hash_for_seq, compute_seq_hash_for_block,
};
pub use dynamo_kv_indexer::testing::{
    add_blocks, add_blocks_with_start_position, create_remove_event, create_store_event,
    create_store_event_with_start_position, make_blocks, make_clear_event,
    make_clear_event_with_dp_rank, make_remove_event, make_remove_event_with_dp_rank,
    make_remove_event_with_parent, make_store_event, make_store_event_full,
    make_store_event_with_dp_rank, make_store_event_with_parent,
    make_store_event_with_start_position, remove_event, router_event, snapshot_events,
    stored_blocks_with_sequence_hashes,
};
pub use dynamo_slot_tracker::NoopSequencePublisher;

/// Minimal [`WorkerConfigLike`] for scheduler/queue tests and benchmarks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleWorkerConfig {
    pub data_parallel_start_rank: u32,
    pub data_parallel_size: u32,
    pub max_num_batched_tokens: Option<u64>,
    pub total_kv_blocks: Option<u64>,
    pub taints: HashSet<String>,
}

impl Default for SimpleWorkerConfig {
    fn default() -> Self {
        Self {
            data_parallel_start_rank: 0,
            data_parallel_size: 1,
            max_num_batched_tokens: None,
            total_kv_blocks: None,
            taints: HashSet::new(),
        }
    }
}

impl WorkerConfigLike for SimpleWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        self.data_parallel_start_rank
    }

    fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }

    fn taints(&self) -> &HashSet<String> {
        &self.taints
    }
}

/// The kernel configuration the selection-core tests run under: stage defaults, a synthetic
/// policy profile (optionally queueing at `queue_threshold`), the flat configuration's default
/// `assume_kv_reuse` (true) and `track_prefill_tokens` (true), a 60 s recheck.
pub fn test_router_config(
    use_kv_events: bool,
    queue_threshold: Option<f64>,
) -> crate::config::RouterConfig {
    crate::config::RouterConfig {
        selection: dynamo_worker_selection::SelectionConfig::default(),
        conditional_disagg: dynamo_worker_selection::ConditionalDisaggConfig::default(),
        use_kv_events,
        track_prefill_tokens: true,
        assume_kv_reuse: true,
        queue_recheck_interval: std::time::Duration::from_secs(60),
        profiles: std::sync::Arc::new(move |_model: Option<&str>| {
            Ok(dynamo_admission::PolicyProfile::synthetic(
                queue_threshold,
                dynamo_admission::RouterQueuePolicy::Fcfs,
            ))
        }),
    }
}

/// The reference worker-selection policy as a factory.
pub fn reference_policy_factory() -> crate::types::WorkerSelectionPolicyFactory {
    std::sync::Arc::new(|context| {
        dynamo_worker_selection::WorkerSelectionPolicy::reference(
            context.selection_config().clone(),
            context.worker_type().default_selector_label(),
        )
    })
}

/// A selection core over `config` with an explicit host, the reference policy unless given,
/// a public tracking hash, and the indexer policy `use_kv_events` implies (an event-driven or
/// an approximate primary, no side indexer).
pub fn test_core_with(
    config: crate::config::RouterConfig,
    indexer_threads: usize,
    cancel_token: tokio_util::sync::CancellationToken,
    host: crate::selection::SelectionHost,
    policy_factory: Option<crate::types::WorkerSelectionPolicyFactory>,
    worker_type: dynamo_router_protocol::worker_type::WorkerType,
    affinity: Option<dynamo_session_affinity::SessionAffinityConfig>,
) -> crate::selection::SelectionCore {
    let indexer_policy =
        dynamo_kv_indexer::IndexerPolicy::from_config(&dynamo_kv_indexer::IndexerConfig {
            use_kv_events: config.use_kv_events,
            ..Default::default()
        })
        .expect("indexer policy");
    crate::selection::SelectionCore::new(
        config.validate().expect("valid router config"),
        indexer_threads,
        cancel_token,
        None,
        policy_factory.unwrap_or_else(reference_policy_factory),
        host,
        worker_type,
        true,
        crate::selection::SelectionCacheConfig::default(),
        std::sync::Arc::new(crate::tracking_hash::TrackingHashContext::new(
            crate::tracking_hash::TrackingHashConfig::default(),
        )),
        indexer_policy,
        affinity,
    )
}

/// A local core over the passive index ingress (no listeners), as hosts without KV events run.
pub fn local_test_core(
    config: crate::config::RouterConfig,
    indexer_threads: usize,
    cancel_token: tokio_util::sync::CancellationToken,
) -> crate::selection::SelectionCore {
    test_core_with(
        config,
        indexer_threads,
        cancel_token,
        crate::selection::SelectionHost::default(),
        None,
        dynamo_router_protocol::worker_type::WorkerType::Aggregated,
        None,
    )
}

/// The scheduler the kernel's tests and benchmarks drive: no sequence publisher,
/// [`SimpleWorkerConfig`] workers, and the default selector.
pub type TestScheduler =
    crate::LocalScheduler<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>;

/// A [`TestScheduler`] over `workers` and `profile`: no load estimator, overlap refresh, or
/// host providers, prefill tokens tracked, a 60 s recheck, and no config monitor. Keep the
/// returned sender for as long as the scheduler runs; publish worker changes through it.
pub fn local_test_scheduler(
    workers: HashMap<WorkerId, SimpleWorkerConfig>,
    profile: PolicyProfile,
    block_size: u32,
    cancel_token: CancellationToken,
) -> (
    Arc<TestScheduler>,
    watch::Sender<HashMap<WorkerId, SimpleWorkerConfig>>,
) {
    let (scheduler, _, configs) = local_test_scheduler_with(
        workers,
        profile,
        block_size,
        cancel_token,
        false,
        None,
        false,
    );
    (scheduler, configs)
}

/// As [`local_test_scheduler`], for the kernel's own tests: it also returns the slot tracker,
/// and takes what that function fixes, which is replica sync, the prefill load estimator, and
/// the config monitor.
#[allow(clippy::type_complexity)]
pub(crate) fn local_test_scheduler_with(
    workers: HashMap<WorkerId, SimpleWorkerConfig>,
    profile: PolicyProfile,
    block_size: u32,
    cancel_token: CancellationToken,
    replica_sync: bool,
    prefill_load_estimator: Option<Arc<dyn crate::PrefillLoadEstimator>>,
    monitor_worker_configs: bool,
) -> (
    Arc<TestScheduler>,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    watch::Sender<HashMap<WorkerId, SimpleWorkerConfig>>,
) {
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        block_size as usize,
        workers
            .iter()
            .map(|(&id, config)| {
                (
                    id,
                    (config.data_parallel_start_rank, config.data_parallel_size),
                )
            })
            .collect(),
        replica_sync,
        0,
        "test",
    ));
    let (configs, config_rx) = watch::channel(workers);
    let scheduler = Arc::new(crate::LocalScheduler::new(
        Arc::clone(&slots),
        config_rx,
        profile,
        block_size,
        DefaultWorkerSelector::new(None, "test"),
        prefill_load_estimator,
        None::<Arc<crate::NoopOverlapScoresRefresh>>,
        None,
        None,
        None,
        Duration::from_secs(60),
        true,
        cancel_token,
        "test",
        monitor_worker_configs,
    ));
    (scheduler, slots, configs)
}

/// A request of `isl_tokens` in `mode` with nothing else set: no tracking hashes, session,
/// pin, class, or overlap.
pub fn test_schedule_request(
    mode: crate::ScheduleMode,
    isl_tokens: usize,
) -> crate::ScheduleRequest {
    crate::ScheduleRequest {
        accounting_attribution: Default::default(),
        mode,
        token_seq: None,
        block_hashes: None,
        isl_tokens,
        lora_name: None,
        expected_output_tokens: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: crate::RoutingConstraints::default(),
        router_config_override: None,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        overlap: crate::OverlapSignals::default(),
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        shared_cache_hits: None,
    }
}
