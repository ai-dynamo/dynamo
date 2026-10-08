// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared sidecar test infrastructure; intended for development dependencies only.

pub mod assert;
pub mod control;
pub mod fixtures;
pub mod server;

pub async fn bounded<T>(label: &str, future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(std::time::Duration::from_secs(10), future)
        .await
        .unwrap_or_else(|_| panic!("timed out: {label}"))
}
