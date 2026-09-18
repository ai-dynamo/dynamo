// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(all(target_os = "linux", target_env = "gnu"))]
use std::sync::{Arc, LazyLock, mpsc};
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use std::time::Duration;

#[cfg(all(target_os = "linux", target_env = "gnu"))]
struct AllocatorTrimScheduler {
    sender: mpsc::SyncSender<()>,
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
impl AllocatorTrimScheduler {
    fn new(quiet_period: Duration, trim: Arc<dyn Fn() + Send + Sync>) -> Self {
        let (sender, receiver) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            while receiver.recv().is_ok() {
                loop {
                    match receiver.recv_timeout(quiet_period) {
                        Ok(()) => {}
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            trim();
                            break;
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            trim();
                            return;
                        }
                    }
                }
            }
        });
        Self { sender }
    }

    fn schedule(&self) {
        match self.sender.try_send(()) {
            Ok(()) | Err(mpsc::TrySendError::Full(())) => {}
            Err(mpsc::TrySendError::Disconnected(())) => {
                tracing::warn!("Allocator trim worker stopped unexpectedly");
            }
        }
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
static ALLOCATOR_TRIM_SCHEDULER: LazyLock<Arc<AllocatorTrimScheduler>> = LazyLock::new(|| {
    Arc::new(AllocatorTrimScheduler::new(
        Duration::from_millis(100),
        Arc::new(|| {
            // SAFETY: malloc_trim is process-wide and internally synchronizes glibc arenas.
            let released_pages = unsafe { libc::malloc_trim(0) != 0 };
            tracing::debug!(released_pages, "Trimmed allocator after model teardown");
        }),
    ))
});

/// Schedules allocator trimming after the last WorkerSet, active request, and teardown task exits.
pub(crate) struct AllocatorTrimOnDrop {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    scheduler: Arc<AllocatorTrimScheduler>,
}

impl AllocatorTrimOnDrop {
    pub(crate) fn new() -> Self {
        Self {
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            scheduler: ALLOCATOR_TRIM_SCHEDULER.clone(),
        }
    }
}

impl Drop for AllocatorTrimOnDrop {
    fn drop(&mut self) {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        self.scheduler.schedule();
    }
}

#[cfg(all(test, target_os = "linux", target_env = "gnu"))]
mod tests {
    use super::*;
    use dynamo_runtime::{
        engine::{AsyncEngineContext, EngineContextGuard},
        pipeline::Context,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test(flavor = "current_thread")]
    async fn trim_waits_for_request_teardown_and_runs_off_async_worker() {
        let async_thread = std::thread::current().id();
        let calls = Arc::new(AtomicUsize::new(0));
        let trim_thread = Arc::new(std::sync::Mutex::new(None));
        let scheduler = Arc::new(AllocatorTrimScheduler::new(
            Duration::from_millis(10),
            Arc::new({
                let calls = calls.clone();
                let trim_thread = trim_thread.clone();
                move || {
                    *trim_thread.lock().unwrap() = Some(std::thread::current().id());
                    calls.fetch_add(1, Ordering::SeqCst);
                }
            }),
        ));
        let teardown: EngineContextGuard = Arc::new(AllocatorTrimOnDrop { scheduler });
        let context = Context::new(());
        context.controller().retain(teardown.clone());
        drop(teardown);
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        std::thread::spawn(move || drop(context)).join().unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("allocator trim did not run after request teardown");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_ne!(trim_thread.lock().unwrap().unwrap(), async_thread);
    }

    #[tokio::test]
    async fn trim_waits_for_background_teardown_task() {
        let calls = Arc::new(AtomicUsize::new(0));
        let scheduler = Arc::new(AllocatorTrimScheduler::new(
            Duration::from_millis(10),
            Arc::new({
                let calls = calls.clone();
                move || {
                    calls.fetch_add(1, Ordering::SeqCst);
                }
            }),
        ));
        let teardown = Arc::new(AllocatorTrimOnDrop { scheduler });
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn({
            let teardown = teardown.clone();
            async move {
                let _teardown = teardown;
                let _ = release_rx.await;
            }
        });

        drop(teardown);
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        release_tx.send(()).unwrap();
        task.await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("allocator trim did not run after task teardown");
    }
}
