// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod config;
pub mod policy_config;

// Pre-split paths kept through 1.9 and removed in 1.10; the named list is the inventory in compat.rs.
#[allow(deprecated)]
pub use crate::compat::scheduling::{
    AccountingAttribution, AdmissionAttempt, AdmittedSchedulingResponse,
    AdvisorySchedulingResponse, AdvisoryWorkerLoad, AttemptId, CacheHitEstimates,
    InvalidEffectivePrefillTokens, KvSchedulerError, LocalScheduler, LoraWorkerFilter,
    NonMaxOverlapSelection, NonMaxOverlapSelectionObserver, NoopOverlapScoresRefresh,
    OverlapAnalysis, OverlapScoresRefresh, OverlapScoresResponse, OverlapSignals,
    OverloadedWorkerProvider, PolicyQueue, PolicyQueueEntry, PotentialLoad, PrefillLoadEstimator,
    QueueLimitKind, QueueMetadata, QueueRejection, QueueSnapshot, RefreshedOverlap,
    RequestLifecycle, RoutingEligibility, ScheduleMode, ScheduleRequest, SchedulingContext,
    SchedulingRequest, SchedulingResponse, SelectedWorkerTierSnapshot, SessionContext, SessionLane,
    SharedCacheOverlapScore, TierOverlapBlocks, TieredOverlapRefresher, WorkerAvailabilityProvider,
    WorkerEligibilityError, WorkerIdSet, WorkerOverlapScore, WorkerSelectionInputTrigger,
    WorkerSelectionPolicyError, effective_prefill_tokens, overlap, overlap_refresh, policy,
    policy_queue, prefill_load, prefill_load_hint_from_effective_tokens, queue, request_classifier,
    selector,
};
#[cfg(test)]
mod policy_queue_tests;
#[cfg(test)]
mod queue_yaml_tests;

pub use policy_config::{
    BusyRanksExt, BusyThresholds, BusyThresholdsExt, PolicyClassConfig, PolicyProfile,
    RejectThresholdsProvider, RouterPolicyConfig, RouterPolicyConfigError,
};
