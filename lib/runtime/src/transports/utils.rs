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
    let (mut tx, rx) = tokio::sync::oneshot::channel();

    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_threads)
            .enable_all()
            .build()?,
    );

    std::thread::spawn(move || {
        runtime.block_on(async {
            let result = tokio::select! {
                biased;

                _ = tx.closed() => return,
                result = f => result,
            };

            let value = match result {
                Ok(value) => value,
                Err(error) => {
                    let _ = tx.send(Err(error));
                    return;
                }
            };
            if tx.send(Ok((value, runtime.clone()))).is_err() {
                return;
            }

            std::future::pending::<()>().await;
        })
    });

    rx.await?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn dropping_caller_stops_pending_initialization() {
        let started = CancellationToken::new();
        let stopped = CancellationToken::new();
        let work_started = started.clone();
        let stop_on_drop = stopped.clone().drop_guard();
        let caller = tokio::spawn(build_in_runtime(
            async move {
                let _stop_on_drop = stop_on_drop;
                work_started.cancel();
                std::future::pending::<Result<()>>().await
            },
            1,
        ));

        time_out(started.cancelled()).await;
        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        time_out(stopped.cancelled()).await;
    }

    async fn time_out(future: impl Future<Output = ()>) {
        tokio::time::timeout(Duration::from_secs(2), future)
            .await
            .expect("transport initialization did not respond");
    }

    #[tokio::test]
    async fn failed_initialization_stops_runtime_tasks() {
        let stopped = CancellationToken::new();
        let stop_on_drop = stopped.clone().drop_guard();
        let result = build_in_runtime(
            async move {
                tokio::spawn(async move {
                    let _stop_on_drop = stop_on_drop;
                    std::future::pending::<()>().await;
                });
                Err::<(), _>(anyhow::anyhow!("initialization failed"))
            },
            1,
        )
        .await;

        assert_eq!(result.unwrap_err().to_string(), "initialization failed");
        time_out(stopped.cancelled()).await;
    }
}
