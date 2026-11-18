// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The selection core: the kernel's host-facing entry. `SelectionCore` owns the partitions,
//! their schedulers, the reservation index, and the catalog; the operation types describe one
//! selection; the DTOs in `types` are what a host or the composed router's HTTP server hands in
//! and gets back.

pub mod catalog;
pub mod core;
pub mod error;
pub mod ingress;
pub mod input;
pub mod membership;
pub mod pending;
pub mod types;

pub use core::{
    HostCache, HostEligibility, HostLoad, HostReplication, HostTelemetry, KvIndexSource,
    SelectionCore, SelectionHost, SelectionPartition, SelectionScheduler,
};
pub use error::SelectionError;
pub use ingress::KvEventIngress;
pub use input::{PromptRequest, PromptView};
pub use membership::{CatalogObserver, CatalogReconciler, WorkerCatalogSource};
pub use pending::SelectionCacheConfig;
pub use types::{
    DEFAULT_MODEL_NAME, ModelLoadResponse, OutputBlockRequest, OverlapScoresRequest,
    PotentialLoadsRequest, ReadyResponse, ReservationRequest, ReservationResponse,
    SelectAndReserveRequest, SelectRequest, SelectResponse, SelectionInputTrigger,
    SelectionSessionContext, SelectionWorkerConfig, SelectionWorkerLoad, WorkerCatalogRecord,
    WorkerLifecycle, WorkerPatchRequest, WorkerRequest,
};
