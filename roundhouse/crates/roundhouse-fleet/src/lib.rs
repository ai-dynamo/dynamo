// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Execution backends: the local Dynamo fleet and frontier providers.
//!
//! This crate owns every transport, so `roundhouse-core` stays testable without
//! a network. Its job is to turn each available target into a priced
//! [`Candidate`](roundhouse_core::routing::Candidate) and then execute whatever
//! the policy picks.
//!
//! [`LocalFleet`] is the seam that keeps the embedded and out-of-process
//! selection services interchangeable. The embedded implementation calls
//! Dynamo's `SelectionService` directly — every HTTP endpoint of
//! `dynamo.select_service` exists as a plain async method, so embedding removes
//! both the TCP round trip and the JSON serialization of the prompt. That
//! serialization is the dominant per-call cost for long agentic contexts, which
//! is precisely the case this design targets.

pub mod frontier;
pub mod local;

pub use frontier::{
    EchoFrontierClient, FrontierChunk, FrontierClient, FrontierError, FrontierModelSpec,
    FrontierQuote, StaticFrontierCatalog,
};
pub use local::{
    EmbeddedFleet, FleetError, FleetQuery, LocalFleet, LocalQuote, Reservation, WorkerRegistration,
};

/// Re-exported so callers can configure and own the embedded service without
/// depending on `dynamo-kv-router` directly.
pub use dynamo_kv_router::config::KvRouterConfig;
pub use dynamo_kv_router::services::selection::{SelectionService, SelectionServiceBuilder};
