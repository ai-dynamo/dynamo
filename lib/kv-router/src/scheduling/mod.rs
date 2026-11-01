// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod admission_config;
#[deprecated(
    since = "1.7.0",
    note = "the kernel owns it: dynamo_scheduler (hosts import dynamo_scheduler::host); see lib/kv-router/README.md, Deprecated paths"
)]
pub mod classifier_rows {
    pub use dynamo_scheduler::classifier_rows::*;
}
pub mod config;
mod lora_filter;
#[deprecated(
    since = "1.7.0",
    note = "the kernel owns it: dynamo_scheduler (hosts import dynamo_scheduler::host); see lib/kv-router/README.md, Deprecated paths"
)]
pub mod overlap {
    pub use dynamo_scheduler::overlap::*;
}
#[deprecated(
    since = "1.7.0",
    note = "the kernel owns it: dynamo_scheduler (hosts import dynamo_scheduler::host); see lib/kv-router/README.md, Deprecated paths"
)]
pub mod overlap_refresh {
    pub use dynamo_scheduler::overlap_refresh::*;
}
pub mod policy_config;
pub mod policy_queue;
#[cfg(test)]
mod policy_queue_tests;
#[cfg(test)]
mod queue_yaml_tests;
#[deprecated(
    since = "1.7.0",
    note = "the kernel owns it: dynamo_scheduler (hosts import dynamo_scheduler::host); see lib/kv-router/README.md, Deprecated paths"
)]
pub mod prefill_load {
    pub use dynamo_scheduler::prefill_load::*;
}
#[deprecated(
    since = "1.7.0",
    note = "the kernel owns it: dynamo_scheduler (hosts import dynamo_scheduler::host); see lib/kv-router/README.md, Deprecated paths"
)]
pub mod protocols {
    pub use dynamo_scheduler::protocols::*;
}
#[deprecated(
    since = "1.7.0",
    note = "the kernel owns it: dynamo_scheduler (hosts import dynamo_scheduler::host); see lib/kv-router/README.md, Deprecated paths"
)]
pub mod queue {
    pub use dynamo_scheduler::queue::*;
}
mod queue_admission;
pub mod request_classifier;
#[deprecated(
    since = "1.7.0",
    note = "the kernel owns it: dynamo_scheduler (hosts import dynamo_scheduler::host); see lib/kv-router/README.md, Deprecated paths"
)]
pub mod router_config {
    pub use dynamo_scheduler::config::*;
}
#[deprecated(
    since = "1.7.0",
    note = "the kernel owns it: dynamo_scheduler (hosts import dynamo_scheduler::host); see lib/kv-router/README.md, Deprecated paths"
)]
pub mod selector {
    pub use dynamo_scheduler::selector::*;
}
#[deprecated(
    since = "1.7.0",
    note = "the kernel owns it: dynamo_scheduler (hosts import dynamo_scheduler::host); see lib/kv-router/README.md, Deprecated paths"
)]
pub mod shared_cache {
    pub use dynamo_scheduler::shared_cache::*;
}
#[deprecated(
    since = "1.7.0",
    note = "the kernel owns it: dynamo_scheduler (hosts import dynamo_scheduler::host); see lib/kv-router/README.md, Deprecated paths"
)]
pub mod tracking_hash {
    pub use dynamo_scheduler::tracking_hash::*;
}

#[deprecated(
    since = "1.7.0",
    note = "dynamo_scheduler::host::LocalScheduler; see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_scheduler::LocalScheduler;
#[deprecated(
    since = "1.7.0",
    note = "the owner's host module or plugins::prelude; see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_scheduler::config::{
    CacheHitWeights, PolicyProfileSource, RouterConfig, RouterConfigError, Validated,
};
#[deprecated(
    since = "1.7.0",
    note = "dynamo_scheduler::host; see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_scheduler::filter::*;
#[deprecated(
    since = "1.7.0",
    note = "the owner's host module or plugins::prelude; see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_scheduler::overlap::{
    CacheHitEstimates, OverlapAnalysis, OverlapScoresResponse, OverlapSignals,
    SelectedWorkerTierSnapshot, SharedCacheOverlapScore, WorkerOverlapScore,
};
#[deprecated(
    since = "1.7.0",
    note = "the owner's host module or plugins::prelude; see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_scheduler::overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, RefreshedOverlap, TieredOverlapRefresher,
};
#[deprecated(
    since = "1.7.0",
    note = "the owner's host module or plugins::prelude; see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_scheduler::prefill_load::{
    InvalidEffectivePrefillTokens, PrefillLoadEstimator, effective_prefill_tokens,
    prefill_load_hint_from_effective_tokens,
};
#[deprecated(
    since = "1.7.0",
    note = "dynamo_worker_selection::host::LoraWorkerFilter; see lib/kv-router/README.md, Deprecated paths"
)]
pub use lora_filter::LoraWorkerFilter;
pub use policy_config::{
    BusyRanksExt, BusyThresholds, BusyThresholdsExt, PolicyClassConfig, PolicyProfile,
    RejectThresholdsProvider, RouterPolicyConfig, RouterPolicyConfigError,
};
#[deprecated(
    since = "1.7.0",
    note = "the owner's host module or plugins::prelude; see lib/kv-router/README.md, Deprecated paths"
)]
pub use policy_queue::{
    PolicyQueue, PolicyQueueEntry, QueueLimitKind, QueueRejection, QueueSnapshot,
};
#[deprecated(
    since = "1.7.0",
    note = "dynamo_admission::host::WorkerPlacement; see lib/kv-router/README.md, Deprecated paths"
)]
pub use queue_admission::WorkerPlacement;
#[deprecated(
    since = "1.7.0",
    note = "dynamo_admission::host; see lib/kv-router/README.md, Deprecated paths"
)]
pub use request_classifier::{ClassificationError, RequestLifecycle};
// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
#[deprecated(
    since = "1.7.0",
    note = "the owner's host module or plugins::prelude; see lib/kv-router/README.md, Deprecated paths"
)]
pub use crate::plugins::request_classifier::{
    AbortCause, ClassifierError, ClassifyEvent, ClassifyFuture, ClassifyRequest, RequestClassifier,
    RequestClassifierConfig, RequestClassifierContext, RequestClassifierFactory,
    RequestClassifierParameters, RequestClassifierProvider, RequestClassifierProviderError,
    RequestClassifierRegistryError, RequestProgress, RequestProgressUpdater,
};
// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
#[deprecated(
    since = "1.7.0",
    note = "the owner's host module or plugins::prelude; see lib/kv-router/README.md, Deprecated paths"
)]
pub use crate::plugins::worker_selection::{WorkerSelectionConfig, WorkerSelectionInstance};
#[deprecated(
    since = "1.7.0",
    note = "dynamo_scheduler::host; see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_scheduler::types::*;
