// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod config;
pub mod error;
pub mod metadata;
mod proxy;
pub mod server;

pub use config::Config;
pub use error::SidecarError;
pub use metadata::{PREFILLER_HOST_PORT, PrefillEndpoint};
pub use server::{PdAdapter, SidecarState, UnavailablePdAdapter, router};

use std::future::IntoFuture;
use std::sync::Arc;

use anyhow::Result;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

pub async fn run(config: Config, adapter: Arc<dyn PdAdapter>) -> Result<()> {
    let listener = TcpListener::bind(config.listen_addr).await?;
    tracing::info!(listen_addr = %config.listen_addr, "Starting EPP decode sidecar");
    let draining = CancellationToken::new();
    let graceful_shutdown = CancellationToken::new();
    let force_shutdown = CancellationToken::new();
    let drain_timeout = config.drain_timeout;
    let server = axum::serve(
        listener,
        router(SidecarState::new(
            config.decode_engine_url,
            config.connect_timeout,
            config.read_timeout,
            adapter,
            draining.clone(),
            force_shutdown.clone(),
        )?),
    )
    .with_graceful_shutdown(graceful_shutdown.clone().cancelled_owned())
    .into_future();
    tokio::pin!(server);

    tokio::select! {
        result = &mut server => result?,
        () = shutdown_signal() => {
            tracing::info!(?drain_timeout, "Shutdown signal received; draining requests");
            draining.cancel();
            graceful_shutdown.cancel();
            match tokio::time::timeout(drain_timeout, &mut server).await {
                Ok(result) => result?,
                Err(_) => {
                    tracing::warn!(?drain_timeout, "Drain deadline reached; closing active streams");
                    force_shutdown.cancel();
                    server.await?;
                }
            }
        }
    }
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "Failed to install Ctrl+C handler");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => tracing::error!(%error, "Failed to install SIGTERM handler"),
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}
