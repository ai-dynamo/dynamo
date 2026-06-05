// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone HTTP slot tracker.
//!
//! Hosts an Axum HTTP server that exposes manual worker registration,
//! request-lifecycle updates, and advisory load reads. The service intentionally
//! stays independent of Dynamo runtime and LLM-layer dependencies.

pub mod registry;
pub mod server;

use std::sync::Arc;

use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use registry::SlotTrackerRegistry;
use server::{AppState, create_router};

pub struct SlotTrackerConfig {
    pub port: u16,
}

pub async fn run_server(config: SlotTrackerConfig) -> anyhow::Result<()> {
    let cancel_token = CancellationToken::new();
    let shutdown_token = cancel_token.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Received shutdown signal");
        shutdown_token.cancel();
    });

    tracing::info!(
        port = config.port,
        "Starting standalone slot tracker (HTTP-only mode)"
    );

    let registry = Arc::new(SlotTrackerRegistry::new(cancel_token.clone()));
    let app = create_router(Arc::new(AppState { registry }));
    let listener = TcpListener::bind(("0.0.0.0", config.port)).await?;
    tracing::info!("HTTP server listening on 0.0.0.0:{}", config.port);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            cancel_token.cancelled().await;
            tracing::info!("Received shutdown signal, stopping HTTP server");
        })
        .await?;

    Ok(())
}
