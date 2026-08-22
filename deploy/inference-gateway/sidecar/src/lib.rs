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

use std::sync::Arc;

use anyhow::Result;
use tokio::net::TcpListener;

pub async fn run(config: Config, adapter: Arc<dyn PdAdapter>) -> Result<()> {
    let listener = TcpListener::bind(config.listen_addr).await?;
    tracing::info!(listen_addr = %config.listen_addr, "Starting EPP decode sidecar");
    axum::serve(
        listener,
        router(SidecarState::new(
            config.decode_engine_url,
            config.connect_timeout,
            config.read_timeout,
            adapter,
        )?),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;
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
