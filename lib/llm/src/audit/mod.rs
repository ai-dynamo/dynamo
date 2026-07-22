// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod bus;
pub mod config;
pub mod handle;
pub mod sink;
pub mod stream;

use tokio_util::sync::CancellationToken;

pub use config::{AuditPolicy, policy};

pub async fn init_from_env() -> anyhow::Result<()> {
    init_from_env_with_shutdown(CancellationToken::new()).await
}

pub async fn init_from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        return Ok(());
    }

    bus::init(policy.capacity);
    sink::spawn_workers_from_env(shutdown).await?;

    tracing::info!(capacity = policy.capacity, "Audit initialized");
    Ok(())
}
