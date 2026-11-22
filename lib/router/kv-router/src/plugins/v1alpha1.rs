// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The router plugin API, version `v1alpha1`: everything a statically linked worker-selection
//! policy or request classifier names.
//!
//! A plugin imports the module for the kind of plugin it is. Each is one import and holds that
//! contract together with the names both contracts share:
//!
//! ```
//! use dynamo_kv_router::plugins::v1alpha1::worker_selection::*;
//! ```
//!
//! ```
//! use dynamo_kv_router::plugins::v1alpha1::request_classifier::*;
//! ```
//!
//! - [`worker_selection`]: filters, scorers, pickers, the policy they assemble into, the
//!   request view and the candidate rows they read.
//! - [`request_classifier`]: the classifier, the request it classifies, the lifecycle events
//!   it receives, and the worker views it reads.
//! - This module: what both share, which is the registry a plugin registers into, the worker
//!   input groups a component declares and the rows it reads them from, the session, and the
//!   identity types. A crate that only registers plugins imports this.
//!
//! Both contracts read workers through the same rows: a classifier's `classify` receives the
//! [`WorkerCandidates`] a filter or scorer reads, with the input groups it declared. A value
//! added to an input group reaches both.
//!
//! # Stability
//!
//! `v1alpha1` is an alpha API. Its shape is the one Dynamo intends to keep, but an item can
//! change or move in a minor release; each such change is listed in the release's deprecation
//! notes, and the previous path keeps resolving for one release where that is possible. When
//! a contract is frozen it becomes `v1`, and its `v1alpha1` module stays as an alias of it for
//! one release. The two contracts can be frozen separately.
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

pub mod request_classifier;
pub mod worker_selection;

// Registration: the registry a plugin crate's `register` fills.
pub use crate::plugins::{RouterPluginRegistry, RouterPluginRegistryError};

// The worker input groups a component declares, the rows it reads them from, and the facts
// every component reads about a worker without declaring a group.
pub use crate::plugins::worker_selection::{
    WorkerCacheInput, WorkerCandidate, WorkerCandidates, WorkerCapacity, WorkerCopyInput,
    WorkerInfo, WorkerInputs, WorkerLoadInput,
};

// The session a request belongs to.
pub use crate::plugins::worker_selection::{SessionContext, WorkerSelectionInputTrigger};

// Identity vocabulary the views carry.
pub use dynamo_router_protocol::WorkerType;
pub use dynamo_router_protocol::identity::{RoutingPartitionId, RoutingPartitionRef};
pub use dynamo_router_protocol::worker::{DpRank, WorkerId, WorkerWithDpRank};
