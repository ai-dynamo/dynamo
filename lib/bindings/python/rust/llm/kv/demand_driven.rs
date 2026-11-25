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

struct KvDemandStream {
    stream: Option<rs::pipeline::EngineStream<RsAnnotated<LLMEngineOutput>>>,
    tracker: Option<Arc<RequestTracker>>,
    first_item: bool,
    finished: bool,
}

impl KvDemandStream {
    fn new(
        stream: rs::pipeline::EngineStream<RsAnnotated<LLMEngineOutput>>,
        tracker: Option<Arc<RequestTracker>>,
    ) -> Self {
        Self {
            stream: Some(stream),
            tracker,
            first_item: true,
            finished: false,
        }
    }

    fn process_response(
        &mut self,
        mut response: RsAnnotated<LLMEngineOutput>,
    ) -> RsAnnotated<Py<PyAny>> {
        if self.first_item {
            self.first_item = false;
            if let (Some(tracker), Some(data)) = (&self.tracker, &mut response.data) {
                inject_worker_id_from_tracker(data, tracker);
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
        let response = response.map_data(|data| {
            Python::attach(|py| {
                pythonize(py, &data)
                    .map(|obj| obj.unbind())
                    .map_err(|error| error.to_string())
            })
        });

        if terminal || response.is_error() {
            self.finished = true;
            self.stream.take();
        }

        response
    }
}

impl Stream for KvDemandStream {
    type Item = RsAnnotated<Py<PyAny>>;

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
            Poll::Ready(Some(response)) => Poll::Ready(Some(this.process_response(response))),
            Poll::Ready(None) => {
                this.finished = true;
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
        drop(crate::bridge_runtime().spawn(async move {
            drop(state);
        }));
    }
}

#[pyclass]
struct DemandDrivenResponseStream {
    source: DemandDrivenOwner<RsAnnotated<Py<PyAny>>>,
}

impl DemandDrivenResponseStream {
    fn new(stream: Pin<Box<dyn Stream<Item = RsAnnotated<Py<PyAny>>> + Send>>) -> Self {
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
        crate::future_into_py(py, async move {
            loop {
                let response = source
                    .next()
                    .await
                    .ok_or_else(|| PyStopAsyncIteration::new_err("Stream exhausted"))?;
                let response = response.ok().map_err(PyValueError::new_err)?;
                if let Some(data) = response.data {
                    return Ok(data);
                }
            }
        })
    }
}

pub(super) fn process_request_to_stream<'p>(
    py: Python<'p>,
    inner: Arc<RsRoutingHost>,
    request: SingleIn<llm_rs::protocols::common::preprocessor::PreprocessedRequest>,
    tracker: Option<Arc<RequestTracker>>,
    dispatch_span: tracing::Span,
) -> PyResult<Bound<'p, PyAny>> {
    crate::future_into_py(py, async move {
        let stream = inner
            .generate(request)
            .instrument(dispatch_span)
            .await
            .map_err(to_pyerr)?;
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
