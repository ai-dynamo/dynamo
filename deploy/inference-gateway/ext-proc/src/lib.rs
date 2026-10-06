// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Envoy ext_proc gRPC server for Dynamo inference routing.
//!
//! - `StreamingServer` handles the ext-proc bidirectional streaming protocol
//! - `EndpointPicker` trait abstracts endpoint selection
//! - The Dynamo `epp::Router` implements `EndpointPicker` using the KV-aware router
//!
//! ```text
//! Envoy ──ext-proc──▶ ExtProcServer ──EndpointPicker──▶ Dynamo KV Router
//! ```

pub mod envoy_helpers;
#[cfg(feature = "epp")]
pub mod epp;
#[cfg(feature = "epp")]
pub mod epp_router;
#[cfg(feature = "epp")]
pub mod epp_standalone_config;
#[cfg(feature = "epp")]
pub mod inference_pool;
#[cfg(feature = "epp")]
pub mod metrics;
#[cfg(feature = "epp")]
pub mod peer_discovery;
pub mod picker;
#[cfg(feature = "epp")]
pub mod pod_discovery;
pub mod preprocess;
pub mod proto;
#[cfg(feature = "epp")]
pub mod render_http;
#[cfg(feature = "epp")]
mod runner;
#[cfg(feature = "epp")]
pub mod selector;
pub mod server;
#[cfg(feature = "epp")]
pub mod sglang_renderer_client;
#[cfg(feature = "epp")]
pub mod topology_adapter;
#[cfg(feature = "epp")]
pub mod vllm_render_client;

#[cfg(feature = "epp")]
pub use epp::Router;
#[cfg(feature = "epp")]
pub use epp_router::EppRouter;
#[cfg(feature = "epp")]
pub use epp_standalone_config::{
    EppMode, EppStandaloneConfig, PeerReplicationConfig, RendererProtocol,
};
#[cfg(feature = "epp")]
pub use inference_pool::PoolState;
pub use picker::{Endpoint, EndpointPicker, PickResult, RequestInfo, ResponseUsage};
#[cfg(feature = "epp")]
pub use pod_discovery::{PodDiscovery, RawWorker};
pub use preprocess::{PreprocessError, PreprocessLimits, RequestMutation, RequestPreprocessor};
#[cfg(feature = "epp")]
pub use render_http::RenderError;
#[cfg(feature = "epp")]
pub use runner::run;
#[cfg(feature = "epp")]
pub use selector::{OverlapSummary, SelectRequest, SelectResponse, Selector};
pub use server::ExtProcServer;
#[cfg(feature = "epp")]
pub use sglang_renderer_client::SglangRendererClient;
#[cfg(feature = "epp")]
pub use topology_adapter::{RegistrationDefaults, TopologyAdapter};
#[cfg(feature = "epp")]
pub use vllm_render_client::VllmRenderClient;
