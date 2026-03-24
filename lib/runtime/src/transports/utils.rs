// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, sync::Arc};

use anyhow::Result;

pub async fn build_in_runtime<
    T: Send + Sync + 'static,
    F: Future<Output = Result<T>> + Send + 'static,
>(
    f: F,
    num_threads: usize,
) -> Result<(T, Arc<tokio::runtime::Runtime>)> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_threads)
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
