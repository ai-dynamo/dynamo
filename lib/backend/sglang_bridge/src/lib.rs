// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo Rust bridge to upstream SGLang's gRPC scheduler
//! (`sglang.grpc.scheduler`).
//!
//! Two consumers:
//!
//! - **Sidecar binary** (`bin/bridge.rs`, B2): registers as a Dynamo worker
//!   and serves requests over the runtime IPC layer.
//! - **In-process library** (B1): linked directly into the Dynamo frontend
//!   via [`build_in_process_engine`], no Dynamo IPC hop.

use std::sync::Arc;

pub mod engine;
pub mod proto;

pub use engine::{Args, DisaggMode, SglangBridge};

use dynamo_backend_common::{DynamoError, LLMEngine};

/// Build the bridge as an in-process engine, returning `Arc<dyn LLMEngine>`
/// suitable for wrapping with `EngineAdapter` and stuffing into
/// `EngineConfig::InProcessTokens`.
///
/// Performs the gRPC handshake (`HealthCheck`, `GetModelInfo`, optional
/// `GetServerInfo` for prefill workers) before returning, so the engine is
/// ready to serve once the launcher attaches it.
pub async fn build_in_process_engine(
    grpc_endpoint: String,
    served_model_name: String,
    connect_timeout_secs: u64,
    disaggregation_mode: DisaggMode,
    bootstrap_host_override: Option<String>,
) -> Result<Arc<dyn LLMEngine>, DynamoError> {
    let bridge = SglangBridge::new(
        grpc_endpoint,
        served_model_name,
        connect_timeout_secs,
        disaggregation_mode,
        bootstrap_host_override,
    );
    // In-process pre-flight handshake. worker_id is opaque to the bridge
    // (SGLang has no use for it), so pass 0.
    bridge.start(0).await?;
    Ok(Arc::new(bridge))
}
