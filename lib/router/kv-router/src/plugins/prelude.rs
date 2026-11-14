// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Everything a statically linked router plugin names, in one import.
//!
//! `use dynamo_kv_router::plugins::prelude::*;` gives a policy or classifier crate the plugin
//! contracts, the signal views they read, the wire vocabulary those views carry, the
//! configuration they are built from, and the registry they register into, so the crate
//! depends on `dynamo-kv-router` alone. Every item is listed by name; nothing here is a glob,
//! so a plugin's imports do not change when the crates behind this one do.

// Worker-selection contracts and the candidate rows they read.
pub use crate::plugins::worker_selection::{
    ActivePrefillSums, HostedSelectionRequest, HostedWorkerPicker, RuntimePolicyBuilder,
    RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate, ScoredWorkerCandidate,
    SelectionConfig, SelectionFailure, WorkerCacheInput, WorkerCacheInputs, WorkerCandidate,
    WorkerCandidates, WorkerCapacity, WorkerCopyInput, WorkerFilter, WorkerInfo, WorkerInputView,
    WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerScorer, WorkerSelectionConfig,
    WorkerSelectionContext, WorkerSelectionInstance, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelectionPolicyFactory, WorkerSelectionPolicySelection,
};

// Request-classifier contracts and the request they classify.
pub use crate::plugins::request_classifier::{
    AbortCause, ClassifierError, ClassifyEvent, ClassifyFuture, ClassifyRequest, RequestClassifier,
    RequestClassifierContext, RequestClassifierFactory, RequestClassifierParameters,
    RequestClassifierProvider, RequestClassifierProviderError, RequestClassifierRegistryError,
    RequestProgress, WorkerView,
};

// Experimental hint-policy contracts, their registration, and the KV copies they send.
pub use crate::plugins::hint_policy::{
    Commitments, HintContext, HintHost, HintPolicy, HintPolicyConstructor, HintPolicyError,
    HintPolicyParameters, HintPolicyProvider, HintPolicyProviderError, HintPolicyRegistryError,
    KvHintAction,
};
pub use crate::plugins::worker_selection::{KvCopyLocations, KvCopySource};

// Registration and the configuration the registry resolves plugins from.
pub use crate::config::{KvRouterConfig, SharedCacheType};
pub use crate::plugins::{
    RouterPluginRegistry, RouterPluginRegistryError, RouterPlugins,
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};

// Wire vocabulary the signal views carry.
pub use dynamo_kv_hashing::ExternalSequenceBlockHash;
pub use dynamo_kv_hints::{
    KvHintTransferWorkerMetadata, KvSourceLocationsPayload, KvTransferEnforcement,
    WorkerKvTransferKeys, WorkerKvTransferKeysCache,
};
pub use dynamo_kv_indexer::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};
pub use dynamo_router_protocol::WorkerType;
pub use dynamo_router_protocol::identity::{RoutingPartitionId, RoutingPartitionRef};
pub use dynamo_router_protocol::session::{SessionContext, WorkerSelectionInputTrigger};
pub use dynamo_router_protocol::worker::{DpRank, WorkerId, WorkerWithDpRank};
pub use dynamo_scheduler::overlap::OverlapSignals;
pub use dynamo_scheduler::protocols::{
    RoutingConstraints, WorkerConfigLike, WorkerSelectionResult,
};
pub use dynamo_scheduler::{
    AccountingAttribution, KvSchedulerError, ScheduleMode, SchedulingRequest, WorkerIdSet,
};
pub use dynamo_session_affinity::{
    MAX_SESSION_AFFINITY_ENTRIES, MAX_SESSION_AFFINITY_ID_BYTES, WorkerAffinityTarget,
};
pub use dynamo_slot_tracker::WorkerLoadProjection;
pub use dynamo_worker_selection::{SelectionTraceSpec, SharedCacheHits};
