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
pub use dynamo_kv_indexer::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};
pub use dynamo_router_protocol::WorkerType;
pub use dynamo_router_protocol::block_hash::ExternalSequenceBlockHash;
pub use dynamo_router_protocol::identity::{RoutingPartitionId, RoutingPartitionRef};
pub use dynamo_router_protocol::session::{SessionContext, WorkerSelectionInputTrigger};
pub use dynamo_router_protocol::worker::{
    DpRank, WorkerAffinityTarget, WorkerId, WorkerWithDpRank,
};
pub use dynamo_scheduler::kv_hints::WorkerKvTransferKeysCache;
pub use dynamo_scheduler::overlap::OverlapSignals;
pub use dynamo_scheduler::protocols::{
    KvHintTransferWorkerMetadata, KvTransferEnforcement, RoutingConstraints, WorkerConfigLike,
    WorkerSelectionResult,
};
pub use dynamo_scheduler::{KvSchedulerError, ScheduleMode, SchedulingRequest};
pub use dynamo_slot_tracker::WorkerLoadProjection;
pub use dynamo_worker_selection::kv_transfer::WorkerKvTransferKeys;
pub use dynamo_worker_selection::{SelectionTraceSpec, SharedCacheHits};
