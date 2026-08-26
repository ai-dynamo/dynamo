// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-free worker selection service.
//!
//! The service owns worker selection and reservation state, but never forwards
//! model requests and never owns model responses.

mod catalog;
mod core;
mod error;
mod input;
mod pending;
mod policy_registry;
mod server;
mod service;
mod types;

#[cfg(test)]
mod tests;

pub use crate::WorkerSelectionPolicyFactory;
pub use crate::services::common::replica_sync::ReplicaPeerError;
pub use core::{SelectionCore, SelectionServiceConfig};
pub use error::SelectionError;
pub use input::PromptRequest;
pub use pending::SelectionCacheConfig;
pub use policy_registry::{
    DYN_ROUTER_DECODE_POLICY, DYN_ROUTER_PREFILL_POLICY, DYN_ROUTER_WORKER_SELECTION_POLICY,
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
pub use server::{AppState, run_server, run_server_with_service};
pub use service::{
    SelectionService, SelectionServiceBuilder, warn_for_unserved_worker_selection_policies,
};
pub use types::{
    ModelLoadResponse, OutputBlockRequest, OverlapScoresRequest, OverlapScoresResponse,
    PotentialLoadsRequest, ReadyResponse, ReservationRequest, ReservationResponse,
    SelectAndReserveRequest, SelectRequest, SelectResponse, SelectionWorkerConfig,
    SharedCacheOverlapScore, WorkerCatalogRecord, WorkerLifecycle, WorkerOverlapScore,
    WorkerPatchRequest, WorkerRequest,
};
