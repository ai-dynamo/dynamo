// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::Context;
use tokio::time::{Instant, timeout_at};
use tonic::transport::{Channel, Endpoint};

use crate::config::ConnectionConfig;

pub async fn connect(config: &ConnectionConfig) -> anyhow::Result<Channel> {
    let endpoint = Endpoint::from_shared(format!("http://127.0.0.1:{}", config.nim_port))?
        .connect_timeout(Duration::from_secs(5));
    let deadline = Instant::now()
        .checked_add(config.nim_startup_timeout_s)
        .context("NIM startup timeout is out of range")?;
    timeout_at(deadline, async {
        loop {
            match endpoint.connect().await {
                Ok(channel) => return channel,
                Err(error) => {
                    tracing::debug!(%error, "Waiting for the Speech NIM gRPC channel");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    })
    .await
    .context("Speech NIM gRPC channel did not become ready before the startup deadline")
}
