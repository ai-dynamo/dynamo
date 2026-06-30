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

#[cfg(feature = "selector-embedded")]
pub mod embedded_selector;
pub mod envoy_helpers;
pub mod epp;
pub mod inference_pool;
pub mod offline_preprocessor;
pub mod picker;
pub mod proto;
pub mod selection_backend;
#[cfg(feature = "selector-http")]
pub mod selector_client;
pub mod selector_config;
pub mod selector_reflector;
pub mod selector_router;
pub mod server;
pub mod topology_adapter;

pub use epp::Router;
pub use inference_pool::PoolState;
pub use offline_preprocessor::build_offline_preprocessor;
pub use picker::{Endpoint, EndpointPicker, PickResult, RequestInfo};
pub use selection_backend::{
    SelectRequest, SelectResponse, SelectionBackend, WorkerPatch, WorkerRegistration,
};
#[cfg(feature = "selector-http")]
pub use selector_client::{HttpSelectionBackend, SelectorClient};
pub use selector_config::{SelectorConfig, SelectorMode};
pub use selector_reflector::{RawWorker, SelectorReflector};
pub use selector_router::SelectorRouter;
pub use server::ExtProcServer;
pub use topology_adapter::{RegistrationDefaults, TopologyAdapter};
