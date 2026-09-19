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
            match result {
                Ok(value) => {
                    // A successful send may still be dropped unread when the
                    // caller is cancelled. Keep the runtime only after receipt.
                    // Tuple drop order releases the runtime Arc before closing
                    // the receipt channel, so its last owner stays on this thread.
                    let (receipt, received) = tokio::sync::oneshot::channel();
                    if tx.send(Ok((value, runtime.clone(), receipt))).is_ok()
                        && received.await.is_ok()
                    {
                        std::future::pending::<()>().await;
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(error));
                }
            }
        })
    });

    let (value, runtime, receipt) = rx.await??;
    let _ = receipt.send(());
    Ok((value, runtime))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression: cancelling startup must drop pending connection work on the
    // dedicated transport thread instead of leaving it running indefinitely.
    #[tokio::test]
    async fn cancelled_build_drops_pending_connection() {
        let dropped = tokio_util::sync::CancellationToken::new();
        let guard = dropped.clone().drop_guard();
        let (started, waiting) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(build_in_runtime(
            async move {
                let _guard = guard;
                started.send(()).unwrap();
                std::future::pending::<Result<()>>().await
            },
            1,
        ));
        waiting.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(std::time::Duration::from_secs(5), dropped.cancelled())
            .await
            .expect("pending transport initialization must be dropped");
    }
}
