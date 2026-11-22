// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request-classifier plugin API: everything a classifier names, in one import.
//!
//! ```
//! use dynamo_kv_router::plugins::v1alpha1::request_classifier::*;
//! ```
//!
//! Stability and scope are those of [`super`]. The dispatch gate a classifier can return is
//! experimental and lives in `plugins::experimental::dispatch_gate`.

// The trait, the request it classifies, and the lifecycle events it receives. The workers it
// reads are the shared `WorkerCandidates`, below.
pub use crate::plugins::request_classifier::{
    AbortCause, ClassifierError, ClassifyEvent, ClassifyRequest, RequestClassifier,
    RequestClassifierContext, RequestClassifierFactory, RequestProgress,
};

// The router-wide busy thresholds `ClassifyRequest::reject_thresholds` returns; `has_room`
// on them is the fullness check the router runs on a rank's load.
pub use dynamo_slot_tracker::BusyThresholds;

// Registration: the parameters a provider is built from, and its errors.
pub use crate::plugins::request_classifier::{
    RequestClassifierParameters, RequestClassifierProvider, RequestClassifierProviderError,
    RequestClassifierRegistryError,
};

// What both contracts share, so that this module is the one import a plugin needs.
pub use super::{
    DpRank, RouterPluginRegistry, RouterPluginRegistryError, RoutingPartitionId,
    RoutingPartitionRef, SessionContext, WorkerCacheInput, WorkerCandidate, WorkerCandidates,
    WorkerCapacity, WorkerCopyInput, WorkerId, WorkerInfo, WorkerInputs, WorkerLoadInput,
    WorkerSelectionInputTrigger, WorkerType, WorkerWithDpRank,
};
