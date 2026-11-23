// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coalescing of concurrent `/metrics` renders.

use std::sync::Arc;
use std::time::Instant;

/// Runs one `/metrics` render at a time and shares it with every scrape that
/// arrived before it started, so N concurrent scrapers cost one gather rather
/// than N blocking-pool tasks. Clones share the same state.
#[derive(Clone, Default)]
pub struct ScrapeCoalescer {
    last: Arc<tokio::sync::Mutex<LastScrape>>,
}

/// The most recent render and when it started collecting.
#[derive(Default)]
struct LastScrape {
    started_at: Option<Instant>,
    text: bytes::Bytes,
}

impl ScrapeCoalescer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Render for a scrape arriving now. A render that started collecting
    /// after the scrape arrived is reused; otherwise `render` runs on the
    /// blocking pool. Waiting scrapes queue on the runtime, not in the pool.
    /// The owned guard moves into the blocking task, so a cancelled request
    /// cannot release the gate before its render finishes.
    pub async fn render<F>(&self, render: F) -> anyhow::Result<bytes::Bytes>
    where
        F: FnOnce() -> anyhow::Result<String> + Send + 'static,
    {
        let requested_at = Instant::now();
        let mut last = self.last.clone().lock_owned().await;
        if last.started_at.is_some_and(|at| at >= requested_at) {
            return Ok(last.text.clone());
        }
        let rendered = tokio::task::spawn_blocking(move || {
            let started_at = Instant::now();
            let text = bytes::Bytes::from(render()?);
            *last = LastScrape {
                started_at: Some(started_at),
                text: text.clone(),
            };
            Ok(text)
        })
        .await;
        rendered.unwrap_or_else(|error| {
            tracing::error!(%error, "metrics render panicked");
            Err(anyhow::anyhow!("metrics render panicked: {error}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::MetricsRegistry;
    use std::time::Duration;

    /// Engine callback whose renders block until the test releases them, and
    /// which reports each render start, so the tests never sleep-and-hope.
    fn held_registry() -> (
        MetricsRegistry,
        tokio::sync::mpsc::UnboundedReceiver<()>,
        std::sync::mpsc::Sender<()>,
    ) {
        let registry = MetricsRegistry::new();
        let (started_tx, started_rx) = tokio::sync::mpsc::unbounded_channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = std::sync::Mutex::new(release_rx);
        registry.add_expfmt_callback(Arc::new(move || {
            let _ = started_tx.send(());
            release_rx.lock().unwrap().recv().expect("release");
            Ok("engine_metric 1\n".to_string())
        }));
        (registry, started_rx, release_tx)
    }

    /// A scrape that arrives while a render is in flight waits on the runtime,
    /// not in the blocking pool, and then renders once for every later arrival
    /// to share; cancelling the active request does not release the gate.
    #[test]
    fn scrapes_queue_on_the_gate_and_share_the_next_render() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .max_blocking_threads(2)
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async {
            let (registry, mut started, release) = held_registry();
            let scrapes = ScrapeCoalescer::new();
            let scrape = |scrapes: &ScrapeCoalescer| {
                let (scrapes, registry) = (scrapes.clone(), registry.clone());
                tokio::spawn(async move {
                    scrapes
                        .render(move || registry.prometheus_expfmt_combined())
                        .await
                })
            };

            // First scrape starts a render, then its request is cancelled.
            let first = scrape(&scrapes);
            started.recv().await.expect("first render started");
            first.abort();
            let _ = first.await;

            // Two scrapes arrive during that render. They must wait on the gate,
            // leaving the pool's other thread free for unrelated work.
            let second = scrape(&scrapes);
            let third = scrape(&scrapes);
            tokio::time::timeout(Duration::from_secs(5), tokio::task::spawn_blocking(|| ()))
                .await
                .expect("a waiting scrape occupied a blocking thread")
                .expect("probe");

            // Releasing the cancelled render lets the second scrape render; the
            // third arrived before that render started and shares it.
            release.send(()).unwrap();
            started.recv().await.expect("second render started");
            release.send(()).unwrap();
            let second = second.await.expect("task").expect("render");
            let third = third.await.expect("task").expect("render");
            assert_eq!(second, third);
            assert!(
                std::str::from_utf8(&second)
                    .unwrap()
                    .contains("engine_metric 1")
            );
            assert!(
                started.try_recv().is_err(),
                "the third scrape must share the second render, not start its own"
            );
        });
    }

    /// A render that panics is reported as an error, and the next scrape
    /// renders afresh.
    #[tokio::test]
    async fn a_panicking_render_is_an_error_and_the_next_scrape_renders() {
        let scrapes = ScrapeCoalescer::new();
        let error = scrapes
            .render(|| panic!("render exploded"))
            .await
            .expect_err("a panicking render must fail the scrape");
        assert!(error.to_string().contains("panicked"), "{error}");
        let text = scrapes
            .render(|| Ok("dynamo_probe 1\n".to_string()))
            .await
            .expect("the next scrape renders");
        assert_eq!(text, "dynamo_probe 1\n");
    }
}
