// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolved framework behavior for host-offload simulation.
//!
//! Policy dimensions are independent so later framework adapters can select a
//! new combination without replacing the shared residency and transfer core.
//! The PoC intentionally exposes only the default vLLM `OffloadingConnector`
//! resolver; custom and additional framework resolvers can be added once their
//! behavior is implemented and covered by parity tests.

use serde::Serialize;

/// Scheduler point at which the host tier is consulted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LookupTiming {
    /// Consult G2 before reserving or allocating the G1 destination.
    BeforeG1Allocation,
    /// Consult G2 only after G1 capacity has been allocated.
    AfterG1Allocation,
}

/// Order in which cache tiers contribute to a request's reusable prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LookupOrder {
    /// Match the local G1 prefix first, then scan its contiguous suffix in G2.
    G1ThenG2,
    /// Match the host-resident G2 prefix before consulting G1.
    G2ThenG1,
}

/// G2 recency updates performed by a framework lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LookupTouch {
    None,
    /// Touch only the contiguous G2 prefix returned by the lookup.
    MatchedPrefix,
    /// Touch every candidate in reverse logical request order. This includes
    /// resident candidates after a prefix gap and applies to misses and
    /// deferred lookups as well as hits.
    AllCandidatesReverseLogical,
}

/// How a waiting request is retried when the current pass cannot admit it.
///
/// Both choices retain the request for a later scheduler pass. They differ in
/// whether unrelated waiting requests may still be considered in this pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerRetry {
    RetryNextPassAndContinue,
    RetryNextPassAndStop,
}

/// Logical G1 headroom applied before a new asynchronous host load starts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadAdmissionHeadroom {
    None,
    /// Subtract the unfinished block footprint of every prefill already in
    /// flight. This prevents multiple non-preemptible loads from committing
    /// more eventual G1 residency than the worker can satisfy.
    InflightPrefillRemainders,
}

/// The scheduler event that creates a store opportunity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreTrigger {
    BlocksCompleted,
    RequestCompleted,
    G1Eviction,
    RequestSuspended,
}

/// How often eligible store work is collected from its trigger.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreCadence {
    Immediate,
    OncePerEngineStep,
    OncePerRequest,
}

/// Capacity-admission cohort formed from eligible store blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreCohort {
    /// Admit each eligible block independently.
    IndividualBlocks,
    /// Admit a request's eligible blocks as one cohort.
    PerRequest,
    /// Admit all eligible blocks observed in an engine step as one cohort.
    EngineStep,
}

/// The request-owned KV eligible for a store.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreScope {
    PromptOnly,
    PromptAndDecode,
    SuspendedRequest,
}

/// The filter applied before a store is admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum StoreAdmission {
    /// Store only blocks that are not already resident in G2.
    MissingFromG2,
    /// Require a minimum number of observations in addition to presence.
    MinimumFrequency { observations: u32 },
    /// Require a minimum framework retention priority in addition to presence.
    MinimumRetentionPriority { priority: i32 },
}

/// What to do when a store cannot immediately reserve G2 capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityHandling {
    /// Reclaim an evictable G2 block, or leave the store cursor in place so a
    /// later scheduler pass retries the same block.
    EvictOrRetry,
    Skip,
}

/// Whether transfer completion is on the current scheduler critical path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferExecution {
    Async,
    Sync,
}

/// Scheduler point at which a prepared store enters the transfer lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreSubmitTiming {
    Immediate,
    NextEngineStep,
}

/// Dependency that may wait for an asynchronous store.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreFence {
    SourceReuseOrPreemption,
}

/// Resolved store execution behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct StoreExecution {
    mode: TransferExecution,
    submit_timing: StoreSubmitTiming,
    fence: Option<StoreFence>,
}

impl StoreExecution {
    pub fn mode(self) -> TransferExecution {
        self.mode
    }

    pub fn submit_timing(self) -> StoreSubmitTiming {
        self.submit_timing
    }

    pub fn fence(self) -> Option<StoreFence> {
        self.fence
    }
}

/// Dependency that may wait for an asynchronous load.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadFence {
    DependentRequest,
}

/// Resolved load execution behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct LoadExecution {
    mode: TransferExecution,
    fence: Option<LoadFence>,
}

impl LoadExecution {
    pub fn mode(self) -> TransferExecution {
        self.mode
    }

    pub fn fence(self) -> Option<LoadFence> {
        self.fence
    }
}

/// Victim selection in the host-resident tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum G2EvictionPolicy {
    Lru,
    Arc,
    RetentionPriorityLru,
}

/// Residency of a host block after a successful G2-to-G1 load.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PostLoadResidency {
    RetainG2Copy,
    ReleaseG2Copy,
}

/// Fully resolved target framework behavior consumed by the scheduler adapter
/// and shared offload core.
///
/// Fields are private so unsupported policy combinations cannot be presented
/// as native framework behavior. Add a resolver only with the corresponding
/// implementation and parity coverage. A transition gap must be called out at
/// the adapter boundary rather than silently ignored.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ResolvedHostOffloadPolicy {
    lookup_timing: LookupTiming,
    lookup_order: LookupOrder,
    lookup_touch: LookupTouch,
    lookup_deferred_retry: SchedulerRetry,
    g1_capacity_retry: SchedulerRetry,
    load_admission_headroom: LoadAdmissionHeadroom,
    store_trigger: StoreTrigger,
    store_cadence: StoreCadence,
    store_cohort: StoreCohort,
    store_scope: StoreScope,
    store_admission: StoreAdmission,
    capacity_handling: CapacityHandling,
    store_execution: StoreExecution,
    load_execution: LoadExecution,
    g2_eviction: G2EvictionPolicy,
    post_load_residency: PostLoadResidency,
}

impl ResolvedHostOffloadPolicy {
    /// Resolve the default CPU path of vLLM's `OffloadingConnector`.
    pub fn vllm_offloading_connector_defaults() -> Self {
        Self {
            // vLLM first resolves the local prefix, then asks the connector
            // for a contiguous G2 suffix before allocating destination slots.
            lookup_timing: LookupTiming::BeforeG1Allocation,
            lookup_order: LookupOrder::G1ThenG2,
            lookup_touch: LookupTouch::AllCandidatesReverseLogical,
            // A pending store or an already in-flight load defers only this
            // request; the waiting queue continues. G1 allocation failure
            // stops the waiting loop. Both cases retry on a later pass.
            lookup_deferred_retry: SchedulerRetry::RetryNextPassAndContinue,
            g1_capacity_retry: SchedulerRetry::RetryNextPassAndStop,
            load_admission_headroom: LoadAdmissionHeadroom::InflightPrefillRemainders,
            store_trigger: StoreTrigger::BlocksCompleted,
            store_cadence: StoreCadence::OncePerEngineStep,
            store_cohort: StoreCohort::PerRequest,
            store_scope: StoreScope::PromptOnly,
            store_admission: StoreAdmission::MissingFromG2,
            capacity_handling: CapacityHandling::EvictOrRetry,
            store_execution: StoreExecution {
                mode: TransferExecution::Async,
                submit_timing: StoreSubmitTiming::NextEngineStep,
                fence: Some(StoreFence::SourceReuseOrPreemption),
            },
            load_execution: LoadExecution {
                mode: TransferExecution::Async,
                fence: Some(LoadFence::DependentRequest),
            },
            g2_eviction: G2EvictionPolicy::Lru,
            post_load_residency: PostLoadResidency::RetainG2Copy,
        }
    }

    pub fn lookup_timing(self) -> LookupTiming {
        self.lookup_timing
    }

    pub fn lookup_order(self) -> LookupOrder {
        self.lookup_order
    }

    pub fn lookup_touch(self) -> LookupTouch {
        self.lookup_touch
    }

    pub fn lookup_deferred_retry(self) -> SchedulerRetry {
        self.lookup_deferred_retry
    }

    pub fn g1_capacity_retry(self) -> SchedulerRetry {
        self.g1_capacity_retry
    }

    pub fn load_admission_headroom(self) -> LoadAdmissionHeadroom {
        self.load_admission_headroom
    }

    pub fn store_trigger(self) -> StoreTrigger {
        self.store_trigger
    }

    pub fn store_cadence(self) -> StoreCadence {
        self.store_cadence
    }

    pub fn store_cohort(self) -> StoreCohort {
        self.store_cohort
    }

    pub fn store_scope(self) -> StoreScope {
        self.store_scope
    }

    pub fn store_admission(self) -> StoreAdmission {
        self.store_admission
    }

    pub fn capacity_handling(self) -> CapacityHandling {
        self.capacity_handling
    }

    pub fn store_execution(self) -> StoreExecution {
        self.store_execution
    }

    pub fn load_execution(self) -> LoadExecution {
        self.load_execution
    }

    pub fn g2_eviction(self) -> G2EvictionPolicy {
        self.g2_eviction
    }

    pub fn post_load_residency(self) -> PostLoadResidency {
        self.post_load_residency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_defaults_resolve_every_policy_dimension() {
        let policy = ResolvedHostOffloadPolicy::vllm_offloading_connector_defaults();

        assert_eq!(policy.lookup_timing(), LookupTiming::BeforeG1Allocation);
        assert_eq!(policy.lookup_order(), LookupOrder::G1ThenG2);
        assert_eq!(
            policy.lookup_touch(),
            LookupTouch::AllCandidatesReverseLogical
        );
        assert_eq!(
            policy.lookup_deferred_retry(),
            SchedulerRetry::RetryNextPassAndContinue
        );
        assert_eq!(
            policy.g1_capacity_retry(),
            SchedulerRetry::RetryNextPassAndStop
        );
        assert_eq!(
            policy.load_admission_headroom(),
            LoadAdmissionHeadroom::InflightPrefillRemainders
        );
        assert_eq!(policy.store_trigger(), StoreTrigger::BlocksCompleted);
        assert_eq!(policy.store_cadence(), StoreCadence::OncePerEngineStep);
        assert_eq!(policy.store_cohort(), StoreCohort::PerRequest);
        assert_eq!(policy.store_scope(), StoreScope::PromptOnly);
        assert_eq!(policy.store_admission(), StoreAdmission::MissingFromG2);
        assert_eq!(policy.capacity_handling(), CapacityHandling::EvictOrRetry);
        assert_eq!(policy.store_execution().mode(), TransferExecution::Async);
        assert_eq!(
            policy.store_execution().submit_timing(),
            StoreSubmitTiming::NextEngineStep
        );
        assert_eq!(
            policy.store_execution().fence(),
            Some(StoreFence::SourceReuseOrPreemption)
        );
        assert_eq!(policy.load_execution().mode(), TransferExecution::Async);
        assert_eq!(
            policy.load_execution().fence(),
            Some(LoadFence::DependentRequest)
        );
        assert_eq!(policy.g2_eviction(), G2EvictionPolicy::Lru);
        assert_eq!(
            policy.post_load_residency(),
            PostLoadResidency::RetainG2Copy
        );
    }

    #[test]
    fn resolved_policy_serializes_with_named_dimensions() {
        let policy = ResolvedHostOffloadPolicy::vllm_offloading_connector_defaults();
        let value = serde_json::to_value(policy).unwrap();

        assert_eq!(value["lookup_timing"], "before_g1_allocation");
        assert_eq!(value["lookup_order"], "g1_then_g2");
        assert_eq!(value["lookup_touch"], "all_candidates_reverse_logical");
        assert_eq!(
            value["lookup_deferred_retry"],
            "retry_next_pass_and_continue"
        );
        assert_eq!(value["g1_capacity_retry"], "retry_next_pass_and_stop");
        assert_eq!(
            value["load_admission_headroom"],
            "inflight_prefill_remainders"
        );
        assert_eq!(value["store_trigger"], "blocks_completed");
        assert_eq!(value["store_cadence"], "once_per_engine_step");
        assert_eq!(value["store_cohort"], "per_request");
        assert_eq!(value["store_scope"], "prompt_only");
        assert_eq!(value["store_admission"]["kind"], "missing_from_g2");
        assert_eq!(value["capacity_handling"], "evict_or_retry");
        assert_eq!(value["store_execution"]["mode"], "async");
        assert_eq!(
            value["store_execution"]["submit_timing"],
            "next_engine_step"
        );
        assert_eq!(
            value["store_execution"]["fence"],
            "source_reuse_or_preemption"
        );
        assert_eq!(value["load_execution"]["mode"], "async");
        assert_eq!(value["load_execution"]["fence"], "dependent_request");
        assert!(value.get("g1_eviction").is_none());
        assert_eq!(value["g2_eviction"], "lru");
        assert_eq!(value["post_load_residency"], "retain_g2_copy");
    }
}
