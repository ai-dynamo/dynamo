// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Envoy ext_proc gRPC server for Dynamo inference routing.
//!
//! - `StreamingServer` handles the ext-proc bidirectional streaming protocol
//! - `EndpointPicker` trait abstracts endpoint selection
//! - `EppRouter` implements `EndpointPicker` using the in-process KV-aware selector
//!
//! ```text
//! Envoy ──ext-proc──▶ ExtProcServer<EppRouter> ──EndpointPicker──▶ KV SelectionService
//! ```

pub mod envoy_helpers;
pub mod epp_router;
pub mod epp_standalone_config;
pub mod inference_pool;
pub mod metrics;
pub mod peer_discovery;
pub mod picker;
pub mod pod_discovery;
pub mod proto;
mod runner;
pub mod selector;
pub mod server;
pub mod topology_adapter;
pub mod vllm_render_client;

pub use epp_router::EppRouter;
pub use epp_standalone_config::{EppStandaloneConfig, PeerReplicationConfig, TokenizerProtocol};
pub use inference_pool::PoolState;
pub use picker::{Endpoint, EndpointPicker, PickResult, RequestInfo, ResponseUsage};
pub use pod_discovery::{PodDiscovery, RawWorker};
pub use runner::run;
pub use selector::{OverlapSummary, SelectRequest, SelectResponse, Selector, WorkerRegistration};
pub use server::ExtProcServer;
pub use topology_adapter::{RegistrationDefaults, TopologyAdapter};
pub use vllm_render_client::{VllmRenderClient, VllmRenderError};
