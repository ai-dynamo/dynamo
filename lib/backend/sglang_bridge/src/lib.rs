// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo bridge to SGLang's native gRPC server (`sglang.runtime.v1`).
//! Deployable as the `dynamo-sglang-bridge` sidecar binary or embedded
//! into `dynamo.frontend` via [`build_in_process_engine_from_env`].

use std::str::FromStr;
use std::sync::Arc;

pub mod engine;
pub mod proto;

pub use engine::{Args, SglangBridge};

use dynamo_backend_common::{
    BackendError, DisaggregationMode, DynamoError, ErrorType, InProcessEngine, LLMEngine,
};

/// Build the bridge as an `InProcessTokens` engine for `dynamo.frontend`.
/// Returns the engine and the resolved disaggregation mode so the caller
/// can wire `is_prefill` consistently.
///
/// Env vars (all optional):
/// - `SGLANG_GRPC_ENDPOINT` — default `http://127.0.0.1:30000`
/// - `DYN_DISAGGREGATION_MODE` — `agg` (default) / `prefill` / `decode`
/// - `DYN_BOOTSTRAP_HOST` — override for SGLang's `dist_init_addr`
/// - `DYN_SGLANG_CONNECT_TIMEOUT_SECS` — default 30
pub async fn build_in_process_engine_from_env(
    served_model_name: String,
) -> Result<(InProcessEngine, DisaggregationMode), DynamoError> {
    let grpc_endpoint = std::env::var("SGLANG_GRPC_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:30000".to_string());
    let connect_timeout_secs = env_parse("DYN_SGLANG_CONNECT_TIMEOUT_SECS")?.unwrap_or(30u64);
    let mode = env_parse("DYN_DISAGGREGATION_MODE")?.unwrap_or(DisaggregationMode::Aggregated);
    let bootstrap_host = std::env::var("DYN_BOOTSTRAP_HOST").ok();

    let bridge = SglangBridge::new(
        grpc_endpoint,
        served_model_name,
        connect_timeout_secs,
        mode,
        bootstrap_host,
    );
    bridge.start(0).await?;

    Ok((
        dynamo_backend_common::wrap_in_process(Arc::new(bridge), mode),
        mode,
    ))
}

/// `None` when the env var is unset; `Err` when set but unparsable.
fn env_parse<T>(name: &str) -> Result<Option<T>, DynamoError>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(s) => s.parse::<T>().map(Some).map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::InvalidArgument))
                .message(format!("invalid {name}={s:?}: {e}"))
                .build()
        }),
        Err(_) => Ok(None),
    }
}
