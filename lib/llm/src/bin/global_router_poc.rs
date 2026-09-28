// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run one regional Global Router replica from a JSON POC configuration file.

use std::env;

use anyhow::{Context, Result, bail};
use dynamo_llm::global_view::poc_config::PocRouterConfig;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let mut args = env::args_os().skip(1);
    let Some(config_path) = args.next() else {
        bail!("usage: global-router-poc CONFIG.json");
    };
    if args.next().is_some() {
        bail!("usage: global-router-poc CONFIG.json");
    }
    let (address, service) = PocRouterConfig::load(config_path)?.build()?;
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("bind Global Router listener at {address}"))?;
    tracing::info!(%address, "Global Router listening");
    let cancel = CancellationToken::new();
    let run = service.run(listener, cancel.clone());
    tokio::pin!(run);
    tokio::select! {
        result = &mut run => result,
        signal = shutdown_signal() => {
            signal?;
            cancel.cancel();
            run.await
        }
    }
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate()).context("register SIGTERM handler")?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.context("wait for Ctrl-C")?,
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await.context("wait for Ctrl-C")?;
    Ok(())
}
