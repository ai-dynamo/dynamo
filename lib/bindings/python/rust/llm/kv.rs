// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;
use std::sync::atomic::AtomicU32;

use super::*;
use llm_rs::kv_router::indexer::compute_block_hash_for_seq;
use llm_rs::kv_router::indexer::KvIndexerInterface;
use rs::traits::events::EventSubscriber;
use tracing;

use llm_rs::kv_router::protocols::*;
use llm_rs::kv_router::publisher::{create_stored_blocks, KvEventSourceConfig};

#[pyclass]
pub(crate) struct KvRouter {
    inner: Arc<llm_rs::kv_router::KvRouter>,
}

#[pymethods]
impl KvRouter {
    #[new]
    fn new(component: Component, kv_block_size: usize) -> PyResult<Self> {
        if kv_block_size == 0 {
            return Err(to_pyerr(anyhow::anyhow!("kv_block_size cannot be 0")));
        };

        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async {
            let inner =
                llm_rs::kv_router::KvRouter::new(component.inner.clone(), kv_block_size, None)
                    .await
                    .map_err(to_pyerr)?;
            Ok(Self {
                inner: Arc::new(inner),
            })
        })
    }

    fn schedule<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
        lora_id: u64,
    ) -> PyResult<Bound<'p, PyAny>> {
        let router = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let worker_id = router
                .schedule(&token_ids, lora_id)
                .await
                .map_err(to_pyerr)?;
            Ok(worker_id)
        })
    }
}

#[pyfunction]
pub fn compute_block_hash_for_seq_py(tokens: Vec<u32>, kv_block_size: usize) -> PyResult<Vec<u64>> {
    if kv_block_size == 0 {
        return Err(to_pyerr(anyhow::anyhow!("kv_block_size cannot be 0")));
    }

    let hashes = compute_block_hash_for_seq(&tokens, kv_block_size);
    Ok(hashes.into_iter().map(|h| h.0).collect())
}

#[pyclass]
pub(crate) struct WorkerMetricsPublisher {
    inner: Arc<llm_rs::kv_router::publisher::WorkerMetricsPublisher>,
}

#[pymethods]
impl WorkerMetricsPublisher {
    #[new]
    fn new() -> PyResult<Self> {
        let inner =
            llm_rs::kv_router::publisher::WorkerMetricsPublisher::new().map_err(to_pyerr)?;
        Ok(Self {
            inner: inner.into(),
        })
    }

    #[pyo3(signature = (component))]
    fn create_endpoint<'p>(
        &self,
        py: Python<'p>,
        component: Component,
    ) -> PyResult<Bound<'p, PyAny>> {
        let rs_publisher = self.inner.clone();
        let rs_component = component.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            rs_publisher
                .create_endpoint(rs_component)
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (request_active_slots, request_total_slots, kv_active_blocks, kv_total_blocks, num_requests_waiting, gpu_cache_usage_perc, gpu_prefix_cache_hit_rate, data_parallel_rank = 0))]
    fn publish(
        &self,
        _py: Python,
        request_active_slots: u64,
        request_total_slots: u64,
        kv_active_blocks: u64,
        kv_total_blocks: u64,
        num_requests_waiting: u64,
        gpu_cache_usage_perc: f32,
        gpu_prefix_cache_hit_rate: f32,
        data_parallel_rank: u32,
    ) -> PyResult<()> {
        self.inner
            .publish(
                llm_rs::kv_router::protocols::ForwardPassMetrics {
                    data_parallel_rank: Some(data_parallel_rank),
                    request_active_slots,
                    request_total_slots,
                    kv_active_blocks,
                    kv_total_blocks,
                    num_requests_waiting,
                    gpu_cache_usage_perc,
                    gpu_prefix_cache_hit_rate,
                }
                .into(),
            )
            .map_err(to_pyerr)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ZmqKvEventPublisherConfig {
    #[pyo3(get, set)]
    pub worker_id: i64,
    #[pyo3(get, set)]
    pub kv_block_size: usize,
    #[pyo3(get, set)]
    pub zmq_endpoint: String,
    #[pyo3(get, set)]
    pub zmq_topic: String,
}

#[pymethods]
impl ZmqKvEventPublisherConfig {
    #[new]
    #[pyo3(signature = (
        worker_id,
        kv_block_size,
        zmq_endpoint = "tcp://127.0.0.1:5557".to_string(),
        zmq_topic = "".to_string()
    ))]
    pub fn new(
        worker_id: i64,
        kv_block_size: usize,
        zmq_endpoint: String,
        zmq_topic: String,
    ) -> Self {
        Self {
            worker_id,
            kv_block_size,
            zmq_endpoint,
            zmq_topic,
        }
    }
}

#[pyclass]
pub(crate) struct ZmqKvEventPublisher {
    inner: llm_rs::kv_router::publisher::KvEventPublisher,
}

#[pymethods]
impl ZmqKvEventPublisher {
    #[new]
    fn new(component: Component, config: ZmqKvEventPublisherConfig) -> PyResult<Self> {
        let inner = llm_rs::kv_router::publisher::KvEventPublisher::new(
            component.inner,
            config.worker_id,
            config.kv_block_size,
            Some(KvEventSourceConfig::Zmq {
                endpoint: config.zmq_endpoint,
                topic: config.zmq_topic,
            }),
        )
        .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn shutdown(&mut self) {
        self.inner.shutdown()
    }
}

/// A ZMQ-based key-value cache event listener that operates independently
/// of the dynamo runtime or event plane infrastructure.
#[pyclass]
pub(crate) struct ZmqKvEventListener {
    event_receiver: Arc<tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<KvCacheEvent>>>,
    shutdown_token: tokio_util::sync::CancellationToken,
}

#[pymethods]
impl ZmqKvEventListener {
    #[new]
    fn new(zmq_endpoint: String, zmq_topic: String, kv_block_size: usize) -> PyResult<Self> {
        if kv_block_size == 0 {
            return Err(to_pyerr(anyhow::anyhow!("kv_block_size cannot be 0")));
        }

        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<KvCacheEvent>();
            let shutdown_token = tokio_util::sync::CancellationToken::new();

            tokio::spawn(llm_rs::kv_router::publisher::start_zmq_listener(
                zmq_endpoint,
                zmq_topic,
                tx,
                shutdown_token.clone(),
                kv_block_size,
            ));

            Ok(Self {
                event_receiver: Arc::new(tokio::sync::Mutex::new(rx)),
                shutdown_token,
            })
        })
    }

    fn get_events<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let receiver = self.event_receiver.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut rx = receiver.lock().await;
            let mut events = Vec::new();

            // Drain all available events
            while let Ok(event) = rx.try_recv() {
                events.push(event);
            }

            // Convert events to JSON strings
            let json_events: Result<Vec<String>, _> =
                events.iter().map(serde_json::to_string).collect();

            match json_events {
                Ok(json_strings) => Ok(json_strings),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to serialize events to JSON: {}",
                    e
                ))),
            }
        })
    }
}

// manual shutdown needed as it's not tied to the dynamo DRT
impl Drop for ZmqKvEventListener {
    fn drop(&mut self) {
        self.shutdown_token.cancel();
    }
}

#[pyclass]
pub(crate) struct KvEventPublisher {
    inner: Arc<llm_rs::kv_router::publisher::KvEventPublisher>,
    kv_block_size: usize,
    warning_count: Arc<AtomicU32>,
}

#[pymethods]
impl KvEventPublisher {
    #[new]
    fn new(component: Component, worker_id: i64, kv_block_size: usize) -> PyResult<Self> {
        if kv_block_size == 0 {
            return Err(to_pyerr(anyhow::anyhow!("kv_block_size cannot be 0")));
        }

        let inner = llm_rs::kv_router::publisher::KvEventPublisher::new(
            component.inner,
            worker_id,
            kv_block_size,
            None,
        )
        .map_err(to_pyerr)?;

        Ok(Self {
            inner: inner.into(),
            kv_block_size,
            warning_count: Arc::new(AtomicU32::new(0)),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (event_id, token_ids, num_block_tokens, block_hashes, lora_id, parent_hash=None))]
    fn publish_stored(
        &mut self,
        _py: Python,
        event_id: u64,
        token_ids: Vec<u32>,
        num_block_tokens: Vec<u64>,
        block_hashes: Vec<i64>,
        lora_id: u64,
        parent_hash: Option<i64>,
    ) -> PyResult<()> {
        let event = KvCacheEvent {
            event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash::from),
                blocks: create_stored_blocks(
                    self.kv_block_size,
                    &token_ids,
                    &num_block_tokens,
                    &block_hashes,
                    lora_id,
                    &self.warning_count,
                ),
            }),
        };

        self.inner.publish(event).map_err(to_pyerr)
    }

    fn publish_removed(&self, _py: Python, event_id: u64, block_hashes: Vec<i64>) -> PyResult<()> {
        let block_hashes: Vec<ExternalSequenceBlockHash> = block_hashes
            .iter()
            .map(|&h| ExternalSequenceBlockHash::from(h))
            .collect();
        let event = KvCacheEvent {
            event_id,
            data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
        };

        self.inner.publish(event).map_err(to_pyerr)
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct OverlapScores {
    inner: llm_rs::kv_router::indexer::OverlapScores,
}

#[pymethods]
impl OverlapScores {
    #[getter]
    fn scores(&self) -> HashMap<llm_rs::kv_router::indexer::WorkerId, u32> {
        self.inner.scores.clone()
    }

    #[getter]
    fn frequencies(&self) -> Vec<usize> {
        self.inner.frequencies.clone()
    }
}

// NOTE: the user needs to guarantee that this stays single threaded in Python land
#[pyclass(unsendable)]
pub(crate) struct RadixTree {
    inner: llm_rs::kv_router::indexer::RadixTree,
}

#[pymethods]
impl RadixTree {
    #[new]
    #[pyo3(signature = (expiration_duration_secs=None))]
    fn new(expiration_duration_secs: Option<f64>) -> PyResult<Self> {
        let expiration_duration = expiration_duration_secs.map(std::time::Duration::from_secs_f64);
        let inner = llm_rs::kv_router::indexer::RadixTree::new_with_frequency(expiration_duration);
        Ok(Self { inner })
    }

    #[pyo3(signature = (sequence, early_exit=false))]
    fn find_matches(
        &self,
        _py: Python,
        sequence: Vec<u64>,
        early_exit: bool,
    ) -> PyResult<OverlapScores> {
        let local_block_hashes: Vec<llm_rs::kv_router::protocols::LocalBlockHash> = sequence
            .into_iter()
            .map(llm_rs::kv_router::protocols::LocalBlockHash)
            .collect();

        let rs_overlap_scores = self.inner.find_matches(local_block_hashes, early_exit);
        Ok(OverlapScores {
            inner: rs_overlap_scores,
        })
    }

    fn apply_event(
        &mut self,
        _py: Python,
        worker_id: i64,
        kv_cache_event_bytes: &[u8],
    ) -> PyResult<()> {
        let kv_cache_event: llm_rs::kv_router::protocols::KvCacheEvent =
            serde_json::from_slice(kv_cache_event_bytes).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to deserialize KvCacheEvent: {}",
                    e
                ))
            })?;

        let router_event = llm_rs::kv_router::indexer::RouterEvent::new(worker_id, kv_cache_event);
        self.inner.apply_event(router_event);
        Ok(())
    }

    fn remove_worker(&mut self, _py: Python, worker_id: i64) -> PyResult<()> {
        self.inner.remove_worker(worker_id);
        Ok(())
    }

    fn clear_all_blocks(&mut self, _py: Python, worker_id: i64) -> PyResult<()> {
        self.inner.clear_all_blocks(worker_id);
        Ok(())
    }
}

#[pyclass]
pub(crate) struct KvIndexer {
    inner: Arc<llm_rs::kv_router::indexer::KvIndexer>,
}

#[pymethods]
impl KvIndexer {
    #[new]
    fn new(component: Component, kv_block_size: usize) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async {
            let inner: Arc<llm_rs::kv_router::indexer::KvIndexer> =
                llm_rs::kv_router::indexer::KvIndexer::new(
                    component.inner.drt().runtime().child_token(),
                    kv_block_size,
                )
                .into();
            // [gluo TODO] try subscribe_with_type::<RouterEvent>,
            // error checking below will be different.
            let mut kv_events_rx = component
                .inner
                .subscribe(llm_rs::kv_router::KV_EVENT_SUBJECT)
                .await
                .map_err(to_pyerr)?;
            let kv_events_tx = inner.event_sender();

            // [FIXME] this is the added functionality to the indexer to subscribe to kv events,
            // should have been made to a trait and implemented here? i.e. AsyncEngine style
            tokio::spawn(async move {
                while let Some(event) = kv_events_rx.next().await {
                    let event: llm_rs::kv_router::indexer::RouterEvent =
                        serde_json::from_slice(&event.payload).unwrap();
                    tracing::debug!("received kv event: {:?}", event);
                    if let Err(e) = kv_events_tx.send(event).await {
                        tracing::trace!(
                            "failed to send kv event to indexer; shutting down: {:?}",
                            e
                        );
                    }
                }
            });
            Ok(Self { inner })
        })
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn find_matches_for_request<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
        _lora_id: u64,
    ) -> PyResult<Bound<'p, PyAny>> {
        let indexer = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let rs_overlap_scores = indexer
                .find_matches_for_request(token_ids.as_slice())
                .await
                .map_err(to_pyerr)?;
            Ok(OverlapScores {
                inner: rs_overlap_scores,
            })
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct EndpointKvMetrics {
    #[pyo3(get, set)]
    pub worker_id: i64,
    #[pyo3(get, set)]
    pub request_active_slots: u64,
    #[pyo3(get, set)]
    pub request_total_slots: u64,
    #[pyo3(get, set)]
    pub kv_active_blocks: u64,
    #[pyo3(get, set)]
    pub kv_total_blocks: u64,
    #[pyo3(get, set)]
    pub num_requests_waiting: u64,
    #[pyo3(get, set)]
    pub gpu_cache_usage_perc: f32,
    #[pyo3(get, set)]
    pub gpu_prefix_cache_hit_rate: f32,
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct AggregatedMetrics {
    #[pyo3(get, set)]
    pub endpoints: Vec<EndpointKvMetrics>,
    #[pyo3(get, set)]
    pub load_avg: f64,
    #[pyo3(get, set)]
    pub load_std: f64,
}

#[pyclass]
pub(crate) struct KvMetricsAggregator {
    inner: Arc<llm_rs::kv_router::metrics_aggregator::KvMetricsAggregator>,
}

#[pymethods]
impl KvMetricsAggregator {
    #[new]
    fn new(component: Component) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async {
            let inner = llm_rs::kv_router::metrics_aggregator::KvMetricsAggregator::new(
                component.inner.clone(),
                component.inner.drt().runtime().child_token(),
            )
            .await;
            Ok(Self {
                inner: inner.into(),
            })
        })
    }

    fn get_metrics<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let endpoints = self.inner.get_endpoints();
        let endpoint_kv_metrics = endpoints
            .endpoints
            .iter()
            .map(|(worker_id, x)| EndpointKvMetrics {
                worker_id: *worker_id,
                request_active_slots: x.data.request_active_slots,
                request_total_slots: x.data.request_total_slots,
                kv_active_blocks: x.data.kv_active_blocks,
                kv_total_blocks: x.data.kv_total_blocks,
                num_requests_waiting: x.data.num_requests_waiting,
                gpu_cache_usage_perc: x.data.gpu_cache_usage_perc,
                gpu_prefix_cache_hit_rate: x.data.gpu_prefix_cache_hit_rate,
            })
            .collect();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(AggregatedMetrics {
                endpoints: endpoint_kv_metrics,
                load_avg: endpoints.load_avg,
                load_std: endpoints.load_std,
            })
        })
    }
}

#[pyclass]
pub(crate) struct KvRecorder {
    inner: Arc<llm_rs::kv_router::recorder::KvRecorder>,
}

#[pymethods]
impl KvRecorder {
    #[new]
    #[pyo3(signature = (component, output_path=None, max_lines_per_file=None, max_count=None, max_time=None))]
    fn new(
        component: Component,
        output_path: Option<String>,
        max_lines_per_file: Option<usize>,
        max_count: Option<usize>,
        max_time: Option<f64>,
    ) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async {
            let token = component.inner.drt().runtime().child_token();

            // Create a temp path if none provided
            let path = match output_path {
                Some(p) => p,
                None => {
                    let temp_dir = std::env::temp_dir();
                    temp_dir
                        .join("kv_events.jsonl")
                        .to_string_lossy()
                        .to_string()
                }
            };

            let inner = llm_rs::kv_router::recorder::KvRecorder::new(
                token.clone(),
                path,
                max_lines_per_file,
                max_count,
                max_time,
            )
            .await
            .map_err(to_pyerr)?;

            // Subscribe to KV events
            let mut kv_events_rx = component
                .inner
                .subscribe(llm_rs::kv_router::KV_EVENT_SUBJECT)
                .await
                .map_err(to_pyerr)?;
            let event_tx = inner.event_sender();

            // Spawn a task to forward events to the recorder
            tokio::spawn(async move {
                while let Some(event) = kv_events_rx.next().await {
                    let event: llm_rs::kv_router::indexer::RouterEvent =
                        serde_json::from_slice(&event.payload).unwrap();
                    tracing::debug!("KvRecorder received kv event: {:?}", event);
                    if let Err(e) = event_tx.send(event).await {
                        tracing::trace!(
                            "KvRecorder failed to send kv event; shutting down: {:?}",
                            e
                        );
                    }
                }
            });

            Ok(Self {
                inner: Arc::new(inner),
            })
        })
    }

    fn event_count<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let recorder = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let count = recorder.event_count().await;
            Ok(count)
        })
    }

    fn elapsed_time<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let recorder = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match recorder.elapsed_time().await {
                Ok(elapsed) => Ok(elapsed.as_secs_f64()),
                Err(_) => Ok(0.0), // Return 0.0 when no events have been received yet
            }
        })
    }

    #[pyo3(signature = (indexer, timed=false, max_count=None, max_time=None))]
    fn replay_events<'py>(
        &self,
        py: Python<'py>,
        indexer: &KvIndexer,
        timed: bool,
        max_count: Option<usize>,
        max_time: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let event_tx = indexer.inner.event_sender();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let count = llm_rs::kv_router::recorder::KvRecorder::send_events(
                "dummy_path", // This doesn't matter as we'll use the provided event_tx
                &event_tx,
                timed,
                max_count,
                max_time,
            )
            .await
            .map_err(to_pyerr)?;
            Ok(count)
        })
    }

    fn shutdown(&self) -> PyResult<()> {
        self.inner.shutdown();
        Ok(())
    }
}
