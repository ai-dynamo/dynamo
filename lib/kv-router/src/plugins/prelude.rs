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
    RuntimePolicyBuilder, RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
    ScoredWorkerCandidate, SelectionConfig, WorkerCacheInput, WorkerCacheInputs, WorkerCandidate,
    WorkerCandidates, WorkerCapacity, WorkerFilter, WorkerInfo, WorkerInputView, WorkerInputs,
    WorkerLoadInput, WorkerPicker, WorkerScorer, WorkerSelectionConfig, WorkerSelectionContext,
    WorkerSelectionInstance, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    WorkerSelectionPolicyFactory, WorkerTransferInput,
};

// Request-classifier contracts and the request they classify.
pub use crate::plugins::request_classifier::{
    AbortCause, ClassifierError, ClassifyEvent, ClassifyFuture, ClassifyRequest, RequestClassifier,
    RequestClassifierContext, RequestClassifierFactory, RequestClassifierParameters,
    RequestClassifierProvider, RequestClassifierProviderError, RequestClassifierRegistryError,
    RequestProgress, WorkerView,
};

// Registration and the configuration the registry resolves plugins from.
pub use crate::config::{KvRouterConfig, SharedCacheType};
pub use crate::plugins::{
    RouterPluginRegistry, RouterPluginRegistryError, RouterPlugins,
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};

// Wire vocabulary the signal views carry.
pub use crate::WorkerType;
pub use crate::identity::{RoutingPartitionId, RoutingPartitionRef};
pub use crate::kv_hints::{
    KvTransferCandidateSource, KvTransferCandidates, WorkerKvTransferKeys,
    WorkerKvTransferKeysCache,
};
pub use crate::protocols::{
    DpRank, ExternalSequenceBlockHash, KvHintTransferWorkerMetadata, KvTransferEnforcement,
    RoutingConstraints, SelectionTraceSpec, SharedCacheHits, WorkerAffinityTarget,
    WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank,
};
pub use crate::scheduling::{
    KvSchedulerError, OverlapSignals, ScheduleMode, SchedulingRequest, SessionContext,
    WorkerSelectionInputTrigger,
};
pub use crate::sequences::WorkerLoadProjection;
