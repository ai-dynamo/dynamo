// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone dummy shared KV cache server.
//!
//! This is a minimal implementation intended for development and testing.
//! It stores block hashes in a simple in-memory `HashSet` and responds to
//! `check_blocks` queries with the positions that are present.

pub mod server;

use std::sync::Arc;

use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use server::{AppState, SharedCacheStore, create_router};

pub struct SharedCacheConfig {
    pub port: u16,
}

pub async fn run_server(config: SharedCacheConfig) -> anyhow::Result<()> {
    let cancel_token = CancellationToken::new();
    let shutdown_token = cancel_token.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Received shutdown signal");
        shutdown_token.cancel();
    });

    tracing::info!(
        port = config.port,
        "Starting standalone shared KV cache server"
    );

    let store = Arc::new(SharedCacheStore::new());
    let state = Arc::new(AppState { store });

    let app = create_router(state);
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
