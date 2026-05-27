// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context as _;
use dynamo_llm::http::service::stateful_responses::{StatefulResponsesConfig, StoreConfig, router};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let config = StatefulResponsesConfig::from_env()
        .context("failed to load stateful Responses proxy configuration")?;
    let store_config =
        StoreConfig::from_env().context("failed to load stateful Responses store config")?;
    let store = store_config
        .open()
        .await
        .with_context(|| format!("failed to open state store: {store_config:?}"))?;
    let app = router(store, config.clone()).context("failed to build proxy router")?;

    tracing::info!(
        listen_addr = %config.listen_addr,
        route_path = %config.route_path,
        upstream = %config.upstream_responses_url,
        ?store_config,
        "starting stateful Responses proxy"
    );

    let listener = tokio::net::TcpListener::bind(config.listen_addr)
        .await
        .with_context(|| format!("failed to bind {}", config.listen_addr))?;

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            if let Err(err) = tokio::signal::ctrl_c().await {
                tracing::warn!(%err, "failed to listen for shutdown signal");
            }
        })
        .await
        .context("stateful Responses proxy failed")?;

    Ok(())
}
