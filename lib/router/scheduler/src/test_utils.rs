// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities. The indexer event builders and assertions live in
//! `dynamo_kv_indexer::testing` and are re-exported here at their old paths.

use std::collections::HashSet;

use crate::protocols::WorkerConfigLike;
pub use dynamo_kv_indexer::testing::*;
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
        conditional_disagg:
            dynamo_worker_selection::conditional_disagg::ConditionalDisaggConfig::default(),
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
    std::sync::Arc::new(|config, role, _| {
        dynamo_worker_selection::WorkerSelectionPolicy::reference(
            config.clone(),
            role.default_selector_label(),
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
    let indexer_policy = dynamo_kv_indexer::backend::IndexerPolicy::from_config(
        &dynamo_kv_indexer::config::IndexerConfig {
            use_kv_events: config.use_kv_events,
            ..Default::default()
        },
    )
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
