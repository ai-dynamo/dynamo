// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, sync::Arc};

use anyhow::Result;

/// Default blocking-pool size for the auxiliary runtimes built here.
/// Tokio's own default is 512, which is far more than these runtimes need.
const DEFAULT_AUX_MAX_BLOCKING_THREADS: usize = 8;

pub async fn build_in_runtime<
    T: Send + Sync + 'static,
    F: Future<Output = Result<T>> + Send + 'static,
>(
    f: F,
    num_threads: usize,
) -> Result<(T, Arc<tokio::runtime::Runtime>)> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    // Cap the blocking pool. Tokio's default is 512 blocking threads PER RUNTIME, and
    // this helper is used to build two long-lived auxiliary runtimes (the etcd lease
    // runtime with 1 worker, and the NATS client runtime with NATS_WORKER_THREADS=4).
    // Left at the default they contribute ~1024 threads to the process, dwarfing the
    // main runtime (which IS configurable via DYN_RUNTIME_MAX_BLOCKING_THREADS) and
    // dominating the process thread count. Neither of these runtimes does meaningful
    // blocking work - they hold a lease and drive a NATS client - so a small pool is
    // ample. Override with DYN_RUNTIME_AUX_MAX_BLOCKING_THREADS if a workload proves
    // otherwise.
    let max_blocking = std::env::var(
        crate::config::environment_names::runtime::DYN_RUNTIME_AUX_MAX_BLOCKING_THREADS,
    )
    .ok()
    .and_then(|v| v.parse::<usize>().ok())
    .filter(|v| *v > 0)
    .unwrap_or(DEFAULT_AUX_MAX_BLOCKING_THREADS);

    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_threads)
            .max_blocking_threads(max_blocking)
            .enable_all()
            .build()?,
    );

    let runtime_clone = runtime.clone();
    std::thread::spawn(move || {
        runtime_clone.block_on(async move {
            let result = f.await;
            tx.send(result)
                .unwrap_or_else(|_| panic!("This should never happen!"));

            std::future::pending::<()>().await;
        })
    });

    let result = rx.await??;

    Ok((result, runtime))
}
