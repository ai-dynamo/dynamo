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

#[cfg(feature = "indexer-runtime")]
pub async fn run_with_runtime(
    runtime: dynamo_runtime::Runtime,
    config: SharedCacheConfig,
    namespace: String,
    component_name: String,
) -> anyhow::Result<()> {
    use dynamo_runtime::{
        DistributedRuntime,
        pipeline::{ManyOut, SingleIn, network::Ingress},
    };

    tracing::info!(
        namespace,
        component_name,
        port = config.port,
        "Starting standalone shared KV cache server (Dynamo runtime mode)"
    );

    let distributed_runtime = DistributedRuntime::from_settings(runtime).await?;
    let cancel_token = distributed_runtime.primary_token();
    let component = distributed_runtime
        .namespace(namespace)?
        .component(component_name)?;

    let store = Arc::new(SharedCacheStore::new());

    // Register a request-plane endpoint so routers can query via SharedKvCacheRequestPlaneClient.
    let engine = Arc::new(server::SharedCacheQueryEngine {
        store: store.clone(),
    });
    let ingress = Ingress::<
        SingleIn<server::SharedCacheQueryRequest>,
        ManyOut<server::SharedCacheQueryResponse>,
    >::for_engine(engine)?;
    let query_endpoint = component
        .endpoint(server::SHARED_KV_CACHE_QUERY_ENDPOINT)
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true);

    distributed_runtime.runtime().secondary().spawn(async move {
        if let Err(err) = query_endpoint.start().await {
            tracing::error!(error = %err, "Shared cache query endpoint failed");
        }
    });

    tracing::info!(
        endpoint = server::SHARED_KV_CACHE_QUERY_ENDPOINT,
        "Query endpoint registered"
    );

    let state = Arc::new(AppState {
        store: store.clone(),
    });
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
