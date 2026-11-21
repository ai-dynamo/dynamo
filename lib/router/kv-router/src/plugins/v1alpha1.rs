// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The router plugin API, version `v1alpha1`: everything a statically linked worker-selection
//! policy or request classifier names, in one import.
//!
//! ```
//! use dynamo_kv_router::plugins::v1alpha1::*;
//! ```
//!
//! # Stability
//!
//! `v1alpha1` is an alpha API. Its shape is the one Dynamo intends to keep, but an item can
//! change or move in a minor release; each such change is listed in the release's deprecation
//! notes, and the previous path keeps resolving for one release where that is possible. When
//! the API is frozen it becomes `v1`, and `v1alpha1` stays as an alias of it for one release.
//!
//! An API that Dynamo may still reshape or remove without notice is in
//! [`super::experimental`], not here, and is imported by name.
//!
//! # What belongs here
//!
//! - The traits a plugin implements, the read-only views the router hands them, the errors
//!   they return, and the registry and parameter types a plugin registers with.
//! - Every item is listed by name; nothing here is a glob, so a plugin's imports do not
//!   change when the crates behind this one do.
//! - No type that only a routing host or the scheduler names: those are in [`super::host`] and
//!   in each owning crate's `host` module. A plugin's tests drive a policy through
//!   [`super::testing`].

// Worker selection: the three traits a policy is built from, and the policy they assemble into.
pub use crate::plugins::worker_selection::{
    SelectionConfig, SharedCacheType, WorkerFilter, WorkerPicker, WorkerScorer,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};

// Worker selection: parameters a policy lets an operator change while it runs.
pub use crate::plugins::worker_selection::{
    RuntimePolicyBuilder, RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
};

// Worker selection: the request a policy reads.
pub use crate::plugins::worker_selection::{
    SessionContext, SharedCacheHits, WorkerSelectionContext, WorkerSelectionInputTrigger,
};
pub use dynamo_session_affinity::{
    MAX_SESSION_AFFINITY_ENTRIES, MAX_SESSION_AFFINITY_ID_BYTES, WorkerAffinityTarget,
};

// Worker selection: the candidate workers a policy reads, and the input groups it declares.
pub use crate::plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerCacheInput, WorkerCacheInputs, WorkerCandidate, WorkerCandidates,
    WorkerCapacity, WorkerCopyInput, WorkerInfo, WorkerInputView, WorkerInputs, WorkerLoadInput,
};

// Request classification: the trait, the request it classifies, the lifecycle events it
// receives, and the workers it reads.
pub use crate::plugins::request_classifier::{
    AbortCause, ClassifierError, ClassifyEvent, ClassifyRequest, RequestClassifier,
    RequestClassifierContext, RequestClassifierFactory, RequestProgress, WorkerCacheView,
    WorkerLoadView, WorkerView,
};

// Registration: the registry a plugin crate's `register` fills, and the parameters a provider
// is built from.
pub use crate::plugins::request_classifier::{
    RequestClassifierParameters, RequestClassifierProvider, RequestClassifierProviderError,
    RequestClassifierRegistryError,
};
pub use crate::plugins::{
    RouterPluginRegistry, RouterPluginRegistryError, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProvider, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

// Identity vocabulary the views carry.
pub use dynamo_router_protocol::WorkerType;
pub use dynamo_router_protocol::identity::{RoutingPartitionId, RoutingPartitionRef};
pub use dynamo_router_protocol::worker::{DpRank, WorkerId, WorkerWithDpRank};
