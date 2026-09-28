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
    let signal_cancel = cancel.clone();
    let signal_task = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal_cancel.cancel();
        }
    });
    let result = service.run(listener, cancel).await;
    signal_task.abort();
    result
}
