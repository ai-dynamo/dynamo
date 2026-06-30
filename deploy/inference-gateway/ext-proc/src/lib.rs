// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Envoy ext_proc gRPC server for Dynamo inference routing.
//!
//! Mirrors the Go LW-EPP architecture from GAIE (issue #2834 / PR #2842):
//! - `StreamingServer` handles the ext-proc bidirectional streaming protocol
//! - `EndpointPicker` trait abstracts endpoint selection
//! - The Dynamo `epp::Router` implements `EndpointPicker` using the KV-aware router
//!
//! ```text
//! Envoy ──ext-proc──▶ ExtProcServer<epp::Router> ──EndpointPicker──▶ Dynamo KV Router
//! ```

pub mod envoy_helpers;
pub mod epp;
pub mod epp_config;
pub mod epp_router;
pub mod inference_pool;
pub mod offline_preprocessor;
pub mod picker;
pub mod pod_discovery;
pub mod proto;
pub mod selection_backend;
pub mod selector_fleet;
pub mod server;
pub mod topology_adapter;

pub use epp::Router;
pub use epp_config::EppConfig;
pub use epp_router::EppRouter;
pub use inference_pool::PoolState;
pub use offline_preprocessor::build_offline_preprocessor;
pub use picker::{Endpoint, EndpointPicker, PickResult, RequestInfo};
pub use pod_discovery::{PodDiscovery, RawWorker};
pub use selection_backend::{
    SelectRequest, SelectResponse, SelectionBackend, WorkerPatch, WorkerRegistration,
};
pub use selector_fleet::SelectorFleet;
pub use server::ExtProcServer;
pub use topology_adapter::{RegistrationDefaults, TopologyAdapter};
