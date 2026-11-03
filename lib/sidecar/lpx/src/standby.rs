// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standby for the Pods of a multi-Pod engine that do not serve, so that each
//! becomes Ready with the engine without registering a second worker.

use std::num::NonZeroU32;

use anyhow::Context as _;
use axum::{Json, Router, http::header::CONTENT_TYPE, response::IntoResponse, routing::get};
use dynamo_runtime::config::RuntimeConfig;
use dynamo_sidecar_common::run_task;
use serde_json::json;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

/// Stands by as `rank` of a multi-Pod engine, in which only rank 0 serves.
///
/// A standby answers the system health and live paths with 200, `/metrics`
/// with an empty Prometheus exposition, and every other path with 404, on the
/// system server's address (`DYN_SYSTEM_*`). It builds no `DistributedRuntime`,
/// so it joins no discovery, request plane or event plane, and it never
/// connects to the engine. It returns `Ok` on SIGTERM or SIGINT.
pub fn run_standby(rank: NonZeroU32) -> anyhow::Result<()> {
    run_task(|_runtime, shutdown| async move {
        let config = RuntimeConfig::from_settings()?;
        let listener = bind(&config).await?;
        stand_by(rank.get(), listener, &config, shutdown).await
    })
}

/// Binds the system server's address, or returns `None` when the system
/// server is disabled.
async fn bind(config: &RuntimeConfig) -> anyhow::Result<Option<TcpListener>> {
    if !config.system_server_enabled() {
        return Ok(None);
    }
    let host = config.system_host.as_str();
    let port = u16::try_from(config.system_port)?;
    let listener = TcpListener::bind((host, port))
        .await
        .with_context(|| format!("bind the standby system server to {host}:{port}"))?;
    Ok(Some(listener))
}

async fn stand_by(
    rank: u32,
    listener: Option<TcpListener>,
    config: &RuntimeConfig,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let Some(listener) = listener else {
        tracing::info!(
            rank,
            "Sidecar standing by without a system server; only rank 0 serves"
        );
        shutdown.cancelled().await;
        return Ok(());
    };
    tracing::info!(
        rank,
        address = %listener.local_addr()?,
        "Sidecar standing by; only rank 0 serves"
    );
    axum::serve(listener, probes(rank, config))
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await?;
    tracing::info!(rank, "Sidecar standby stopped");
    Ok(())
}

/// Answers 200 on the health, live and metrics paths; axum answers 404
/// elsewhere.
fn probes(rank: u32, config: &RuntimeConfig) -> Router {
    let ready = get(move || async move {
        Json(json!({"status": "ready", "role": "standby", "rank": rank}))
    });
    Router::new()
        .route(&config.system_health_path, ready.clone())
        .route(&config.system_live_path, ready)
        // The runtime's system server serves its metrics on this fixed path.
        .route("/metrics", get(metrics))
}

/// A standby has no series of its own, so it answers a scrape with an empty
/// exposition rather than showing up as a down target. The content type is
/// the one the serving rank's system server sends.
async fn metrics() -> impl IntoResponse {
    ([(CONTENT_TYPE, "text/plain; charset=utf-8")], String::new())
}

// The executable's tests cover a standby with a system server.
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn without_a_system_server_stands_by_until_shutdown() {
        let config = RuntimeConfig {
            system_port: -1,
            ..RuntimeConfig::default()
        };
        let listener = bind(&config).await.expect("bind");
        assert!(listener.is_none(), "the system server is disabled");
        let shutdown = CancellationToken::new();
        let mut standby = tokio::spawn({
            let shutdown = shutdown.clone();
            async move { stand_by(1, listener, &config, shutdown).await }
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut standby)
                .await
                .is_err(),
            "a standby runs until shutdown"
        );
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(5), standby)
            .await
            .expect("the standby stops on shutdown")
            .expect("the standby task")
            .expect("a standby stops cleanly");
    }
}
