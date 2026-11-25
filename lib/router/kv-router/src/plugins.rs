// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The router plugin API: the contracts and registration for statically linked router plugins.
//!
//! Plugin authors implement a worker-selection policy or a request classifier and register
//! their providers with [`RouterPluginRegistry`]. Scheduling owns execution, eligibility,
//! lifecycle synchronization, and capacity accounting.
//!
//! A plugin imports the module for the kind of plugin it is. Each is one import and holds that
//! kind's API together with the names both kinds share:
//!
//! ```
//! use dynamo_kv_router::plugins::worker_selection::*;
//! ```
//!
//! ```
//! use dynamo_kv_router::plugins::request_classifier::*;
//! ```
//!
//! - [`worker_selection`]: filters, scorers, pickers, the policy they assemble into, the
//!   request view and the candidate rows they read.
//! - [`request_classifier`]: the classifier, the request it classifies, the lifecycle events
//!   it receives, and the worker rows it reads.
//! - This module: only what both share, which is the registry a plugin registers into, the
//!   worker input groups a component declares and the rows it reads them from, the request
//!   facts every stage reads ([`RequestInfo`]), the session, and the identity types. A crate
//!   that only registers plugins imports this.
//!
//! Both kinds read workers through the same rows: a classifier's `classify` receives the
//! [`WorkerCandidates`] a filter or scorer reads, with the input groups it declared. A value
//! added to an input group reaches both. Both also read a request through the same
//! [`RequestInfo`], which `info()` returns on the request each kind receives.
//!
//! # Stability
//!
//! Stability has two levels, and the path says which one a name is at:
//!
//! - The plugin API, which is this module, [`worker_selection`], and [`request_classifier`],
//!   can change in a minor release, and each change is listed in that release's deprecation
//!   notes.
//! - [`experimental`] has no compatibility promise. Each API there has its own submodule and
//!   is imported by name. An experimental API's names move into a kind's module when it is
//!   ready.
//!
//! # What belongs here
//!
//! - The traits a plugin implements, the read-only views the router hands them, the errors
//!   they return, and the registry and parameter types a plugin registers with.
//! - Every item is listed by name and no other crate is re-exported by glob, so a plugin's
//!   imports do not change when the crates behind this one do. The names both kinds share
//!   are listed once, in the private `shared` module, which this module and the two kind
//!   modules re-export whole.
//! - No type that only a routing host or the scheduler names: those are in [`host`] and in
//!   each owning crate's `host` module. Plugin crates do not import [`host`].
//! - A plugin's tests and benches drive a policy through [`testing`], a harness that runs a
//!   policy over a described request and workers.

pub mod experimental;
pub(crate) mod hint_policy;
pub mod host;
mod registry;
pub mod request_classifier;
pub mod testing;
pub mod worker_selection;

/// The names both kinds of plugin share, each listed once. This module and the two kind
/// modules re-export all of it, so a name added here reaches the three together.
mod shared {
    // Registration: the registry a plugin crate's `register` fills, and its error.
    pub use super::RouterPluginRegistryError;
    pub use super::registry::RouterPluginRegistry;

    // The worker input groups a component declares, the rows it reads them from, and the
    // facts every component reads about a worker and about a request without declaring a
    // group.
    pub use dynamo_worker_selection::{
        RequestInfo, WorkerCacheInput, WorkerCandidate, WorkerCandidates, WorkerCapacity,
        WorkerCopyInput, WorkerInfo, WorkerInputs, WorkerLoadInput,
    };

    // The session a request belongs to.
    pub use dynamo_router_protocol::session::{SessionContext, WorkerSelectionInputTrigger};

    // Identity vocabulary the views carry.
    pub use dynamo_router_protocol::WorkerType;
    pub use dynamo_router_protocol::identity::{RoutingPartitionId, RoutingPartitionRef};
    pub use dynamo_router_protocol::worker::{DpRank, WorkerId, WorkerWithDpRank};
}

pub use shared::*;

use hint_policy::HintPolicyRegistryError;
use request_classifier::RequestClassifierRegistryError;
use worker_selection::WorkerSelectionPolicyRegistryError;

/// Non-exhaustive: the `HintPolicy` variant carries an experimental type and can change.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RouterPluginRegistryError {
    #[error(transparent)]
    WorkerSelection(#[from] WorkerSelectionPolicyRegistryError),
    #[error(transparent)]
    RequestClassifier(#[from] RequestClassifierRegistryError),
    #[error(transparent)]
    HintPolicy(#[from] HintPolicyRegistryError),
}
