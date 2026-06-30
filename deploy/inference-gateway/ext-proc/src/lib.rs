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
pub mod inference_pool;
pub mod offline_preprocessor;
pub mod picker;
pub mod proto;
pub mod selector_client;
pub mod selector_config;
pub mod selector_reflector;
pub mod server;

pub use epp::Router;
pub use inference_pool::PoolState;
pub use offline_preprocessor::build_offline_preprocessor;
pub use picker::{Endpoint, EndpointPicker, PickResult, RequestInfo};
pub use selector_client::{
    SelectRequest, SelectResponse, SelectorClient, WorkerPatch, WorkerRegistration,
};
pub use selector_config::SelectorConfig;
pub use selector_reflector::{RawWorker, SelectorReflector};
pub use server::ExtProcServer;
