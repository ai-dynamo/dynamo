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
pub mod epp_standalone_config;
pub mod inference_pool;
pub mod picker;
pub mod pod_discovery;
pub mod proto;
pub mod server;
pub mod vllm_render_client;

pub use epp::Router;
pub use epp_standalone_config::{EppMode, EppStandaloneConfig, TokenizerProtocol};
pub use inference_pool::PoolState;
pub use picker::{Endpoint, EndpointPicker, PickResult, RequestInfo};
pub use pod_discovery::{PodDiscovery, RawWorker};
pub use server::ExtProcServer;
pub use vllm_render_client::{VllmRenderClient, VllmRenderError};
