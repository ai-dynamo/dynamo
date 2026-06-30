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
mod server;
mod service;
mod types;

#[cfg(test)]
mod tests;

pub use crate::services::common::replica_sync::ReplicaPeerError;
pub use core::{SelectionCore, SelectionServiceConfig};
pub use error::SelectionError;
pub use input::PromptRequest;
pub use server::{AppState, run_server};
pub use service::{SelectionService, SelectionServiceBuilder};
pub use types::{
    ModelLoadResponse, OutputBlockRequest, OverlapScoresRequest, OverlapScoresResponse,
    PotentialLoadsRequest, ReadyResponse, ReservationRequest, ReservationResponse,
    SelectAndReserveRequest, SelectRequest, SelectResponse, SelectionKey, SelectionWorkerConfig,
    SharedCacheOverlapScore, WorkerCatalogRecord, WorkerLifecycle, WorkerOverlapScore,
    WorkerPatchRequest, WorkerRequest,
};
