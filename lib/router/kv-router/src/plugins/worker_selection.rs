// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The worker-selection plugin API: everything a filter, scorer, or picker names, in one
//! import.
//!
//! ```
//! use dynamo_kv_router::plugins::worker_selection::*;
//! ```
//!
//! Stability and scope are those of [`super`]. The stage lives in `dynamo-worker-selection`;
//! what only a routing host or the scheduler names from it is in [`super::host`] and in that
//! crate's `host` module, not here.

pub(crate) mod config;

// The three traits a policy is built from, the policy they assemble into, and the factory
// that creates one policy per routing partition.
pub use dynamo_scheduler::WorkerSelectionPolicyFactory;
pub use dynamo_worker_selection::{
    SelectionConfig, SharedCacheType, WorkerFilter, WorkerPicker, WorkerScorer,
    WorkerSelectionPolicy, WorkerSelectionPolicyContext, WorkerSelectionPolicyError,
};

// Parameters a policy lets an operator change while it runs. The update and the snapshot are
// here, although a host applies them, so that a policy's tests can exercise its builder
// through `WorkerSelectionPolicy::update_runtime_config`.
pub use dynamo_worker_selection::{
    RuntimePolicyBuilder, RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
};

// The request a policy reads.
pub use dynamo_session_affinity::{
    MAX_SESSION_AFFINITY_ENTRIES, MAX_SESSION_AFFINITY_ID_BYTES, WorkerAffinityTarget,
};
pub use dynamo_worker_selection::{SharedCacheHits, WorkerSelectionRequest};

// The table a picker reads: the surviving candidates with their costs.
pub use dynamo_worker_selection::{ScoredWorkerCandidate, WorkerCacheInputs, WorkerInputView};

// Registration: the parameters a provider is built from, and its errors.
pub use super::registry::{
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};

// What both kinds share, so that this module is the one import a plugin needs. The names are
// listed once, in `plugins::shared`.
pub use super::shared::*;
