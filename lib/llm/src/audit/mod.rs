// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod bus;
pub mod config;
pub mod handle;
pub mod otel_sink;
pub mod sink;
pub mod stream;

use tokio_util::sync::CancellationToken;

pub use config::AuditPolicy;
pub use config::policy;

pub async fn init_from_env() -> anyhow::Result<()> {
    init_from_env_with_shutdown(CancellationToken::new()).await
}

pub async fn init_from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<()> {
    crate::request_trace::init_from_env_with_shutdown(shutdown).await
}
