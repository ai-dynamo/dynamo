// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use futures::Stream;
use llm_rs::protocols::common::llm_backend::LLMEngineOutput;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

struct TrackerFinishGuard {
    tracker: Option<Arc<RequestTracker>>,
}

impl TrackerFinishGuard {
    fn new(tracker: Option<Arc<RequestTracker>>) -> Self {
        Self { tracker }
    }

    fn observe(&mut self) {
        if let Some(tracker) = self.tracker.take() {
            tracker.observe_finish_gauges();
        }
    }
}

impl Drop for TrackerFinishGuard {
    fn drop(&mut self) {
        self.observe();
    }
}

struct KvDemandStream {
    stream: Option<rs::pipeline::EngineStream<RsAnnotated<LLMEngineOutput>>>,
    tracker: Option<Arc<RequestTracker>>,
    finish_guard: TrackerFinishGuard,
    first_item: bool,
    first_token_gauges_observed: bool,
    finished: bool,
}

impl KvDemandStream {
    fn new(
        stream: rs::pipeline::EngineStream<RsAnnotated<LLMEngineOutput>>,
        tracker: Option<Arc<RequestTracker>>,
    ) -> Self {
        Self {
            stream: Some(stream),
            finish_guard: TrackerFinishGuard::new(tracker.clone()),
            tracker,
            first_item: true,
            first_token_gauges_observed: false,
            finished: false,
        }
    }

    fn process_response(&mut self, mut response: RsAnnotated<LLMEngineOutput>) -> Option<PyObject> {
        if self.first_item {
            self.first_item = false;
            if let (Some(tracker), Some(data)) = (&self.tracker, &mut response.data) {
                inject_worker_id_from_tracker(data, tracker);
            }
        }

        if !self.first_token_gauges_observed {
            let has_tokens = response
                .data
                .as_ref()
                .map(|data| !data.token_ids.is_empty())
                .unwrap_or(false);
            if has_tokens {
                if let Some(ref tracker) = self.tracker {
                    tracker.observe_first_token_gauges();
                }
                self.first_token_gauges_observed = true;
            }
        }

        let terminal = response
            .data
            .as_ref()
            .is_some_and(|data| data.finish_reason.is_some());
        if terminal && let (Some(tracker), Some(data)) = (&self.tracker, &mut response.data) {
            tracker.record_finish();
            inject_timing_from_tracker(data, tracker);
        }
        if terminal {
            self.finished = true;
            self.stream.take();
            self.finish_guard.observe();
        }

        let py_response = Python::with_gil(|py| {
            pythonize(py, &response.data)
                .map(|obj| obj.unbind())
                .map_err(|error| error.to_string())
        });

        match py_response {
            Ok(response) => Some(response),
            Err(error) => {
                tracing::error!("Failed to pythonize response: {}", error);
                self.finished = true;
                self.stream.take();
                self.finish_guard.observe();
                None
            }
        }
    }
}

impl Stream for KvDemandStream {
    type Item = PyObject;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        let this = self.as_mut().get_mut();
        let Some(stream) = this.stream.as_mut() else {
            this.finished = true;
            return Poll::Ready(None);
        };

        match stream.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(response)) => match this.process_response(response) {
                Some(response) => Poll::Ready(Some(response)),
                None => {
                    this.finished = true;
                    Poll::Ready(None)
                }
            },
            Poll::Ready(None) => {
                this.finished = true;
                this.finish_guard.observe();
                Poll::Ready(None)
            }
        }
    }
}

struct DemandDrivenState<T> {
    stream: Mutex<Pin<Box<dyn Stream<Item = T> + Send>>>,
    cancelled: CancellationToken,
}

impl<T> DemandDrivenState<T> {
    async fn next(&self) -> Option<T> {
        tokio::select! {
            biased;
            _ = self.cancelled.cancelled() => None,
            item = async { self.stream.lock().await.next().await } => item,
        }
    }
}

struct DemandDrivenOwner<T: 'static> {
    state: Option<Arc<DemandDrivenState<T>>>,
}

impl<T: 'static> DemandDrivenOwner<T> {
    fn new(stream: Pin<Box<dyn Stream<Item = T> + Send>>) -> Self {
        Self {
            state: Some(Arc::new(DemandDrivenState {
                stream: Mutex::new(stream),
                cancelled: CancellationToken::new(),
            })),
        }
    }

    fn state(&self) -> Arc<DemandDrivenState<T>> {
        self.state
            .as_ref()
            .expect("demand-driven source already dropped")
            .clone()
    }

    #[cfg(test)]
    async fn next(&self) -> Option<T> {
        self.state().next().await
    }
}

impl<T: 'static> Drop for DemandDrivenOwner<T> {
    fn drop(&mut self) {
        let Some(state) = self.state.take() else {
            return;
        };
        state.cancelled.cancel();

        if tokio::runtime::Handle::try_current().is_ok() {
            drop(state);
            return;
        }

        // Python can release the stream outside Tokio. Keep the final Arc alive
        // until the PyO3 runtime drops it so nested router guards can spawn cleanup.
        drop(pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            drop(state);
        }));
    }
}

#[pyclass]
struct DemandDrivenResponseStream {
    source: DemandDrivenOwner<PyObject>,
}

impl DemandDrivenResponseStream {
    fn new(stream: Pin<Box<dyn Stream<Item = PyObject> + Send>>) -> Self {
        Self {
            source: DemandDrivenOwner::new(stream),
        }
    }
}

#[pymethods]
impl DemandDrivenResponseStream {
    #[pyo3(name = "__aiter__")]
    fn aiter(slf: PyRef<Self>, py: Python) -> PyResult<Py<PyAny>> {
        slf.into_py_any(py)
    }

    #[pyo3(name = "__anext__")]
    fn next<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let source = self.source.state();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            source
                .next()
                .await
                .ok_or_else(|| PyStopAsyncIteration::new_err("Stream exhausted"))
        })
    }
}

pub(super) fn process_request_to_stream<'p>(
    py: Python<'p>,
    inner: Arc<RsKvPushRouter>,
    request: llm_rs::protocols::common::preprocessor::PreprocessedRequest,
    tracker: Option<Arc<RequestTracker>>,
) -> PyResult<Bound<'p, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let single_in = SingleIn::new(request);
        let stream = inner.generate(single_in).await.map_err(to_pyerr)?;
        // Zero capacity is genuinely demand-driven only for direct Python
        // consumption of this KvRouter iterator: each __anext__ polls the
        // upstream stream once. PythonServerStreamingEngine adds its own
        // eager response buffer, so an end client does not control this
        // boundary when the iterator is re-served through an endpoint.
        Ok(DemandDrivenResponseStream::new(Box::pin(
            KvDemandStream::new(stream, tracker),
        )))
    })
}

#[cfg(test)]
mod tests {
    use super::DemandDrivenOwner;
    use futures::Stream;
    use std::collections::VecDeque;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};
    use std::time::Duration;

    struct CountingStream {
        items: VecDeque<usize>,
        polls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        runtime_drops: Arc<AtomicUsize>,
        pending_when_empty: bool,
    }

    impl Stream for CountingStream {
        type Item = usize;

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            match self.items.pop_front() {
                Some(item) => Poll::Ready(Some(item)),
                None if self.pending_when_empty => Poll::Pending,
                None => Poll::Ready(None),
            }
        }
    }

    impl Drop for CountingStream {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
            if tokio::runtime::Handle::try_current().is_ok() {
                self.runtime_drops.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    #[tokio::test]
    async fn source_polls_once_per_request() {
        let polls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let runtime_drops = Arc::new(AtomicUsize::new(0));
        let owner = DemandDrivenOwner::new(Box::pin(CountingStream {
            items: VecDeque::from([1, 2]),
            polls: polls.clone(),
            drops: drops.clone(),
            runtime_drops: runtime_drops.clone(),
            pending_when_empty: false,
        }));

        assert_eq!(polls.load(Ordering::SeqCst), 0);
        tokio::task::yield_now().await;
        assert_eq!(polls.load(Ordering::SeqCst), 0);

        assert_eq!(owner.next().await, Some(1));
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        tokio::task::yield_now().await;
        assert_eq!(polls.load(Ordering::SeqCst), 1);

        assert_eq!(owner.next().await, Some(2));
        assert_eq!(polls.load(Ordering::SeqCst), 2);

        std::thread::spawn(move || drop(owner))
            .join()
            .expect("owner drop thread should not panic");
        tokio::time::timeout(Duration::from_secs(1), async {
            while drops.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("source should be dropped");
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(runtime_drops.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dropping_owner_cancels_pending_request_and_drops_source() {
        let polls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let runtime_drops = Arc::new(AtomicUsize::new(0));
        let owner = DemandDrivenOwner::new(Box::pin(CountingStream {
            items: VecDeque::new(),
            polls: polls.clone(),
            drops: drops.clone(),
            runtime_drops: runtime_drops.clone(),
            pending_when_empty: true,
        }));
        let state = owner.state();
        let pending = tokio::spawn(async move { state.next().await });

        while polls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }

        drop(owner);

        let result = tokio::time::timeout(Duration::from_secs(1), pending)
            .await
            .expect("pending response request should be cancelled")
            .expect("pending response task should not panic");
        assert_eq!(result, None);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(runtime_drops.load(Ordering::SeqCst), 1);
    }
}
