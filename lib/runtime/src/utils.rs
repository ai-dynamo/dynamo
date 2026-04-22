// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use tokio::time::{Duration, Instant};

pub mod graceful_shutdown;
pub mod ip_resolver;
pub mod pool;
pub mod stream;
pub mod task;
pub mod tasks;
pub mod typed_prefix_watcher;

pub use graceful_shutdown::GracefulShutdownTracker;
pub use ip_resolver::{
    get_http_rpc_bind_host_from_env, get_http_rpc_host_from_env,
    get_tcp_rpc_advertise_host_from_env, get_tcp_rpc_bind_host_from_env, get_tcp_rpc_host_from_env,
};

#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(()));
