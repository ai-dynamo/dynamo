// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo bridge to SGLang's native gRPC server (`sglang.runtime.v1`).
//! Deployable as the `dynamo-sglang-bridge` sidecar binary or embedded
//! into `dynamo.frontend` via [`build_in_process_engine_from_env`].

use std::sync::Arc;

pub mod engine;
pub mod proto;

pub use engine::{Args, SglangBridge};

use dynamo_backend_common::{DisaggregationMode, DynamoError, InProcessEngine, LLMEngine};

/// Build the bridge as an `InProcessTokens` engine for `dynamo.frontend`.
///
/// Env vars (all optional):
/// - `SGLANG_GRPC_ENDPOINT` — default `http://127.0.0.1:30000`
/// - `DYN_DISAGGREGATION_MODE` — `agg` (default) / `prefill` / `decode`
/// - `DYN_BOOTSTRAP_HOST` — override for SGLang's `dist_init_addr`
/// - `DYN_SGLANG_CONNECT_TIMEOUT_SECS` — default 30
pub async fn build_in_process_engine_from_env(
    served_model_name: String,
) -> Result<InProcessEngine, DynamoError> {
    let grpc_endpoint = std::env::var("SGLANG_GRPC_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:30000".to_string());
    let connect_timeout_secs: u64 = std::env::var("DYN_SGLANG_CONNECT_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let mode = std::env::var("DYN_DISAGGREGATION_MODE")
        .ok()
        .and_then(|s| s.parse::<DisaggregationMode>().ok())
        .unwrap_or(DisaggregationMode::Aggregated);
    let bootstrap_host = std::env::var("DYN_BOOTSTRAP_HOST").ok();

    let bridge = SglangBridge::new(
        grpc_endpoint,
        served_model_name,
        connect_timeout_secs,
        mode,
        bootstrap_host,
    );
    bridge.start(0).await?;

    Ok(dynamo_backend_common::wrap_in_process(Arc::new(bridge), mode))
}
