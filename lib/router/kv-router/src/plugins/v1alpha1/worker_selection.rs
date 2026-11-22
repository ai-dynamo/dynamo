// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The worker-selection plugin API: everything a filter, scorer, or picker names, in one
//! import.
//!
//! ```
//! use dynamo_kv_router::plugins::v1alpha1::worker_selection::*;
//! ```
//!
//! Stability and scope are those of [`super`].

// The three traits a policy is built from, and the policy they assemble into.
pub use crate::plugins::worker_selection::{
    SelectionConfig, SharedCacheType, WorkerFilter, WorkerPicker, WorkerScorer,
    WorkerSelectionPolicy, WorkerSelectionPolicyContext, WorkerSelectionPolicyError,
    WorkerSelectionPolicyFactory,
};

// Parameters a policy lets an operator change while it runs. The update and the snapshot are
// here, although a host applies them, so that a policy's tests can exercise its builder
// through `WorkerSelectionPolicy::update_runtime_config`.
pub use crate::plugins::worker_selection::{
    RuntimePolicyBuilder, RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
};

// The request a policy reads.
pub use crate::plugins::worker_selection::{SharedCacheHits, WorkerSelectionRequest};
pub use dynamo_session_affinity::{
    MAX_SESSION_AFFINITY_ENTRIES, MAX_SESSION_AFFINITY_ID_BYTES, WorkerAffinityTarget,
};

// The table a picker reads: the surviving candidates with their costs.
pub use crate::plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerCacheInputs, WorkerInputView,
};

// Registration: the parameters a provider is built from, and its errors.
pub use crate::plugins::{
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};

// What both contracts share, so that this module is the one import a plugin needs.
pub use super::{
    BusyThresholds, DpRank, RouterPluginRegistry, RouterPluginRegistryError, RoutingPartitionId,
    RoutingPartitionRef, SessionContext, WorkerCacheInput, WorkerCandidate, WorkerCandidates,
    WorkerCapacity, WorkerCopyInput, WorkerId, WorkerInfo, WorkerInputs, WorkerLoadInput,
    WorkerSelectionInputTrigger, WorkerType, WorkerWithDpRank,
};
