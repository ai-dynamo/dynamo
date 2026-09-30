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
    let (accepted_tx, accepted_rx) = tokio::sync::oneshot::channel();

    // The thread owns the runtime until the caller accepts the constructed
    // transport. Cancellation must drop the runtime on this synchronous thread,
    // not inside the caller's async context.
    std::thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_threads)
            .enable_all()
            .build()
        {
            Ok(runtime) => Arc::new(runtime),
            Err(error) => {
                let _ = tx.send(Err(error.into()));
                return;
            }
        };
        runtime.block_on(async {
            let result = tokio::select! {
                biased;
                _ = tx.closed() => return,
                result = f => result,
            };
            match result {
                Ok(value) => {
                    if tx.send(Ok((value, Arc::downgrade(&runtime)))).is_err() {
                        return;
                    }
                    if accepted_rx.await.is_err() {
                        return;
                    }
                    // Preserve the established lifetime of a successfully
                    // handed-off transport's dedicated executor.
                    std::future::pending::<()>().await;
                }
                Err(error) => {
                    let _ = tx.send(Err(error));
                }
            }
        });
    });

    let (value, runtime) = rx.await??;
    let runtime = runtime
        .upgrade()
        .expect("constructor thread awaits acceptance");
    let _ = accepted_tx.send(());
    Ok((value, runtime))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression: a cancelled connection attempt kept running forever on its
    // dedicated executor, or panicked when it sent to the abandoned caller.
    #[tokio::test]
    async fn cancelled_transport_construction_drops_inflight_work() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (dropped_tx, dropped_rx) = tokio::sync::oneshot::channel();
        struct OnDrop(Option<tokio::sync::oneshot::Sender<()>>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                let _ = self.0.take().unwrap().send(());
            }
        }
        let task = tokio::spawn(build_in_runtime(
            async move {
                let _drop = OnDrop(Some(dropped_tx));
                started_tx.send(()).unwrap();
                std::future::pending::<Result<()>>().await
            },
            1,
        ));
        started_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(std::time::Duration::from_secs(3), dropped_rx)
            .await
            .unwrap()
            .unwrap();
    }
}
