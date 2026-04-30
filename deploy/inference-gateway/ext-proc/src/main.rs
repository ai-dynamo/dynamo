// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone Rust EPP binary.
//!
//! Replaces the Go EPP + CGO bridge with a single native Rust binary that
//! implements the Envoy ext_proc gRPC service and uses Dynamo's KV-aware
//! router for endpoint selection.
//!
//! ```text
//! Envoy ──ext-proc gRPC──▶ this binary ──▶ Dynamo KV Router
//! ```

use std::sync::Arc;

use anyhow::Result;
use dynamo_ext_proc::{ExtProcServer, Router};

/// CLI / environment configuration.
///
/// Namespace resolution matches the Go EPP plugin behavior:
///   DYN_NAMESPACE_PREFIX > DYN_NAMESPACE > "vllm-agg"
/// Component is hardcoded to "backend" (same as Go EPP's ffiComponent).
const GRPC_PORT: u16 = 9002;

struct Config {
    namespace: String,
    component: String,
    enforce_disagg: bool,
}

impl Config {
    fn from_env() -> Self {
        let namespace = env_or("DYN_NAMESPACE_PREFIX", "")
            .or_else(|| env_or("DYN_NAMESPACE", ""))
            .unwrap_or_else(|| "vllm-agg".to_string());

        Self {
            namespace,
            component: env_or("DYN_COMPONENT_NAME", "").unwrap_or_else(|| "backend".to_string()),
            enforce_disagg: parse_env("DYN_ENFORCE_DISAGG", false),
        }
    }
}

fn env_or(key: &str, empty_means_unset: &str) -> Option<String> {
    std::env::var(key).ok().and_then(|v| {
        let trimmed = v.trim();
        if trimmed.is_empty() || trimmed == empty_means_unset {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn parse_env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env();

    tracing::info!(
        port = GRPC_PORT,
        namespace = %config.namespace,
        component = %config.component,
        enforce_disagg = config.enforce_disagg,
        "Starting Dynamo Rust EPP"
    );

    tracing::info!("Initializing KV-aware router from discovery...");
    let router =
        Router::from_discovery(&config.namespace, &config.component, config.enforce_disagg).await?;

    tracing::info!("Router initialized, starting ext_proc gRPC server");
    let picker = Arc::new(router);

    let addr = format!("0.0.0.0:{GRPC_PORT}").parse()?;
    let server = ExtProcServer::new(picker);

    tracing::info!(%addr, "Listening for ext_proc connections");

    tonic::transport::Server::builder()
        .add_service(server.into_service())
        .serve(addr)
        .await?;

    Ok(())
}
