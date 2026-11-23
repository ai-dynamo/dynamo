// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request-classifier plugin API: everything a classifier names, in one import.
//!
//! ```
//! use dynamo_kv_router::plugins::request_classifier::*;
//! ```
//!
//! Stability and scope are those of [`super`]. The stage lives in `dynamo-admission`; what
//! only a routing host or the scheduler names from it is in [`super::host`] and in that
//! crate's `host` module, not here. The dispatch gate a classifier can return is experimental
//! and lives in `plugins::experimental::dispatch_gate`.

pub(crate) mod config;
pub(crate) mod registry;

// The trait, the request it classifies, and the lifecycle events it receives. The workers it
// reads are the shared `WorkerCandidates`, below.
pub use dynamo_admission::{
    AbortCause, ClassifierError, ClassifyEvent, ClassifyRequest, RequestClassifier,
    RequestClassifierContext, RequestProgress,
};

// Registration: the factory a provider returns, the parameters it is built from, and its
// errors.
pub use registry::{
    RequestClassifierFactory, RequestClassifierParameters, RequestClassifierProvider,
    RequestClassifierProviderError, RequestClassifierRegistryError,
};

// What both kinds share, so that this module is the one import a plugin needs.
pub use super::{
    BusyThresholds, DpRank, RequestInfo, RouterPluginRegistry, RouterPluginRegistryError,
    RoutingPartitionId, RoutingPartitionRef, SessionContext, WorkerCacheInput, WorkerCandidate,
    WorkerCandidates, WorkerCapacity, WorkerCopyInput, WorkerId, WorkerInfo, WorkerInputs,
    WorkerLoadInput, WorkerSelectionInputTrigger, WorkerType, WorkerWithDpRank,
};
