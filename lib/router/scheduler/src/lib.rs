// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The scheduler kernel: run the stages atomically for one request. Owns the request and
//! response vocabulary, eligibility, the queue actor and `LocalScheduler`, the overlap and
//! prefill-load analysis, the selection kernel over the worker-selection stage, the validated
//! kernel configuration, and the tracking hash. Hosts and the composed router
//! (`dynamo-kv-router`) sit above it; the five stages, protocol, kv-hashing, and kv-hints sit
//! below.

mod classifier_rows;
mod config;
mod filter;
pub mod host;
mod identity;
mod kv_hints;
mod local;
mod overlap;
mod overlap_refresh;
mod overlap_summary;
mod policy;
mod prefill_load;
mod protocols;
mod queue;
mod replication;
mod selection;
mod selector;
mod shared_cache;
mod tracking_hash;
mod types;

#[cfg(any(test, feature = "testing", feature = "bench"))]
pub mod test_utils;

pub use config::{
    CacheHitWeights, PolicyProfileSource, RouterConfig, RouterConfigError, RouterConfigOverride,
    RouterConfigOverrideBuilder, RouterConfigOverrideBuilderError, RouterPrefillLoadModel,
    Validated,
};
pub use filter::{RoutingEligibility, WorkerEligibilityError};
pub use identity::default_routing_group;
pub use local::LocalScheduler;
pub use overlap::{
    CacheHitEstimates, OverlapAnalysis, OverlapScoresResponse, OverlapSignals,
    SelectedWorkerTierSnapshot, SharedCacheOverlapScore, WorkerOverlapScore,
    build_overlap_scores_response, cache_hit_estimates_from_tiered_matches,
    tier_overlap_blocks_from_tiered_matches,
};
pub use overlap_refresh::{
    DEFAULT_OVERLAP_REFRESH_AFTER_SECS, NoopOverlapScoresRefresh, OVERLAP_REFRESH_TIMEOUT,
    OverlapScoresRefresh, RefreshedOverlap, TieredOverlapRefresher, read_overlap_refresh_after,
    refresh_overlap, should_refresh_overlap,
};
pub use overlap_summary::{MooncakeOverlapSummary, build_mooncake_overlap_summaries};
pub use policy::{FcfsPolicy, LcfsPolicy, RouterSchedulingPolicy, SchedulingPolicy, WsptPolicy};
pub use prefill_load::{
    InvalidEffectivePrefillTokens, PrefillLoadEstimator, effective_prefill_tokens,
    prefill_load_hint_from_effective_tokens,
};
pub use protocols::{
    PotentialLoad, RouterEventSink, RouterRequest, RouterResponse, RoutingConstraints,
    SelectionLogFields, WorkerConfigLike, WorkerSelectionResult,
};
pub use queue::{
    BookingHandle, ClassQueueStats, RequestLifecycleLease, RuntimePolicyControl,
    SchedulerBookingCleanup, SchedulerCleanupAck, SchedulerQueue,
};
pub use replication::{
    AFFINITY_EVENT_CHANNEL_CAPACITY, ChannelSequenceSubscriber, HostReplicaChannels,
    HostReplicaSyncFactory, REPLICA_EVENT_CHANNEL_CAPACITY, ReplicaEventSender, ReplicaInbox,
    ReplicaIngressObserver, ReplicaSyncConfig, SchedulerLoadSink, ScopedReplicaEvent,
    ScopedReplicaSync, ScopedSequencePublisher, setup_scoped_replica_sync,
};
pub use selection::core::LookupTimings;
pub use selection::input::{
    MmRoutingInfoRequest, NormalizedPrompt, NormalizedReservation, TrackingHashInput,
};
pub use selection::types::REQUEST_BODY_LIMIT_BYTES;
pub use selection::{
    CatalogObserver, CatalogReconciler, DEFAULT_MODEL_NAME, HostCache, HostEligibility, HostLoad,
    HostReplication, HostTelemetry, KvEventIngress, KvIndexSource, ModelLoadResponse,
    OutputBlockRequest, OverlapScoresRequest, PotentialLoadsRequest, PromptRequest, PromptView,
    ReadyResponse, ReservationRequest, ReservationResponse, SelectAndReserveRequest, SelectRequest,
    SelectResponse, SelectionCacheConfig, SelectionCore, SelectionError, SelectionHost,
    SelectionInputTrigger, SelectionPartition, SelectionScheduler, SelectionSessionContext,
    SelectionWorkerConfig, SelectionWorkerLoad, WorkerCatalogRecord, WorkerCatalogSource,
    WorkerLifecycle, WorkerPatchRequest, WorkerRequest,
};
#[cfg(any(test, feature = "testing", feature = "bench"))]
pub use selector::DefaultWorkerSelector;
pub use selector::{
    ConfiguredSelectionInputs, PreparedCandidates, WorkerCapacityExt, WorkerInfoExt,
    WorkerLoadInputExt, WorkerSelectionInput, WorkerSelector,
};
pub use shared_cache::SharedKvCache;
pub use tracking_hash::{
    KEY_SIZE, TrackingHashAlgorithm, TrackingHashConfig, TrackingHashContext, TrackingHashScope,
};
pub use types::{
    AccountingAttribution, AdmissionAttempt, AdmittedSchedulingResponse,
    AdvisorySchedulingResponse, AdvisoryWorkerLoad, KvSchedulerError, NonMaxOverlapSelection,
    NonMaxOverlapSelectionObserver, OverloadedWorkerProvider, RejectThresholdsProvider,
    ScheduleMode, ScheduleRequest, SchedulingContext, SchedulingRequest, SchedulingResponse,
    TierOverlapBlocks, WorkerAvailabilityProvider, WorkerIdSet, WorkerSelectionPolicyFactory,
};
