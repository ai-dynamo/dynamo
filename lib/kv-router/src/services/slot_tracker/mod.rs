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

use crate::services::common::replica_sync::{
    PeerManager, generate_process_id, replica_sync_bind_endpoint, start_replica_publisher,
};
use registry::SlotTrackerRegistry;
use server::{AppState, create_router};

pub struct SlotTrackerConfig {
    pub port: u16,
    pub replica_sync_port: Option<u16>,
    pub replica_sync_peers: Vec<String>,
}

pub async fn run_server(config: SlotTrackerConfig) -> anyhow::Result<()> {
    let cancel_token = CancellationToken::new();
    let shutdown_token = cancel_token.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Received shutdown signal");
        shutdown_token.cancel();
    });

    let (registry, peer_manager) = if let Some(replica_sync_port) = config.replica_sync_port {
        let bind_endpoint = replica_sync_bind_endpoint(replica_sync_port)?;
        let process_id = generate_process_id();
        let outbound_tx = start_replica_publisher(&bind_endpoint, cancel_token.child_token())?;
        let registry = Arc::new(SlotTrackerRegistry::new_with_replica_sync(
            cancel_token.clone(),
            process_id,
            outbound_tx,
        ));
        let dispatch_registry = Arc::clone(&registry);
        let peer_manager = PeerManager::start(
            config.replica_sync_peers,
            cancel_token.child_token(),
            move |event| dispatch_registry.dispatch_replica_event(event),
        )?;
        tracing::info!(
            port = config.port,
            replica_sync_port,
            process_id,
            "Starting standalone slot tracker with replica sync"
        );
        (registry, Some(peer_manager))
    } else {
        if !config.replica_sync_peers.is_empty() {
            anyhow::bail!("--replica-sync-peers requires --replica-sync-port");
        }
        tracing::info!(
            port = config.port,
            "Starting standalone slot tracker (HTTP-only mode)"
        );
        (
            Arc::new(SlotTrackerRegistry::new(cancel_token.clone())),
            None,
        )
    };

    let app = create_router(Arc::new(AppState {
        registry,
        peer_manager,
    }));
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
