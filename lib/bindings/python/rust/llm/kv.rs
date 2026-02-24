// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pythonize::{depythonize, pythonize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::mpsc;
use tokio_stream::StreamExt;

use super::*;
use crate::Endpoint;
use llm_rs::kv_router::protocols::compute_block_hash_for_seq;
use rs::pipeline::{AsyncEngine, SingleIn};
use rs::protocols::annotated::Annotated as RsAnnotated;
use tracing;

use llm_rs::kv_router::KvPushRouter as RsKvPushRouter;
use llm_rs::kv_router::protocols::*;
use llm_rs::kv_router::publisher::{KvEventSourceConfig, create_stored_blocks};
use llm_rs::protocols::common::timing::RequestTracker;
use llm_rs::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use serde_json::json;

fn depythonize_block_mm_infos(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Option<BlockExtraInfo>>> {
    depythonize(obj).map_err(to_pyerr)
}

#[pyfunction]
#[pyo3(name = "start_kv_block_indexer", signature = (endpoint, block_size, kv_router_config))]
pub fn start_kv_block_indexer_py<'p>(
    py: Python<'p>,
    endpoint: &Endpoint,
    block_size: u32,
    kv_router_config: &super::entrypoint::KvRouterConfig,
) -> PyResult<Bound<'p, PyAny>> {
    let component = endpoint.inner.component().clone();
    let config = kv_router_config.inner();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        llm_rs::kv_router::indexer_standalone::start_kv_block_indexer(
            &component, &config, block_size,
        )
        .await
        .map_err(to_pyerr)?;
        Ok(())
    })
}

#[pyfunction]
#[pyo3(name = "compute_block_hash_for_seq", signature = (tokens, kv_block_size, block_mm_infos=None))]
pub fn compute_block_hash_for_seq_py(
    _py: Python,
    tokens: Vec<u32>,
    kv_block_size: usize,
    block_mm_infos: Option<Bound<PyAny>>,
) -> PyResult<Vec<u64>> {
    if kv_block_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "kv_block_size cannot be 0",
        ));
    }

    let mm_infos = block_mm_infos
        .as_ref()
        .map(depythonize_block_mm_infos)
        .transpose()?;

    let hashes = compute_block_hash_for_seq(&tokens, kv_block_size as u32, mm_infos.as_deref());

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

    #[pyo3(signature = (endpoint))]
    fn create_endpoint<'p>(
        &self,
        py: Python<'p>,
        endpoint: Endpoint,
    ) -> PyResult<Bound<'p, PyAny>> {
        let rs_publisher = self.inner.clone();
        let rs_component = endpoint.inner.component().clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            rs_publisher
                .create_endpoint(rs_component)
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Publish worker metrics for load monitoring.
    ///
    /// # Arguments
    /// * `dp_rank` - Data parallel rank of the worker (None defaults to 0)
    /// * `active_decode_blocks` - Number of active KV cache blocks
    #[pyo3(signature = (dp_rank, active_decode_blocks))]
    fn publish(&self, dp_rank: Option<u32>, active_decode_blocks: u64) -> PyResult<()> {
        self.inner
            .publish(dp_rank, active_decode_blocks)
            .map_err(to_pyerr)
    }
}

#[pyclass]
pub(crate) struct KvEventPublisher {
    inner: Arc<llm_rs::kv_router::publisher::KvEventPublisher>,
    kv_block_size: usize,
    dp_rank: DpRank,
    warning_count: Arc<AtomicU32>,
}

#[pymethods]
impl KvEventPublisher {
    #[new]
    #[pyo3(signature = (endpoint, worker_id=0, kv_block_size=0, dp_rank=0, enable_local_indexer=false, zmq_endpoint=None, zmq_topic=None))]
    fn new(
        endpoint: Endpoint,
        worker_id: WorkerId,
        kv_block_size: usize,
        dp_rank: DpRank,
        enable_local_indexer: bool,
        zmq_endpoint: Option<String>,
        zmq_topic: Option<String>,
    ) -> PyResult<Self> {
        let _ = worker_id;

        let source_config = zmq_endpoint.map(|ep| KvEventSourceConfig::Zmq {
            endpoint: ep,
            topic: zmq_topic.unwrap_or_default(),
        });

        if kv_block_size == 0 {
            return Err(to_pyerr(anyhow::anyhow!("kv_block_size cannot be 0")));
        }

        // Extract component from endpoint
        let component = endpoint.inner.component().clone();

        let inner = llm_rs::kv_router::publisher::KvEventPublisher::new_with_local_indexer(
            component,
            kv_block_size as u32,
            source_config,
            enable_local_indexer,
            dp_rank,
        )
        .map_err(to_pyerr)?;

        Ok(Self {
            inner: inner.into(),
            kv_block_size,
            dp_rank,
            warning_count: Arc::new(AtomicU32::new(0)),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (token_ids, num_block_tokens, block_hashes, lora_id, parent_hash=None, block_mm_infos=None))]
    fn publish_stored(
        &self,
        py: Python,
        token_ids: Vec<u32>,
        num_block_tokens: Vec<u64>,
        block_hashes: Vec<i64>,
        lora_id: u64,
        parent_hash: Option<i64>,
        block_mm_infos: Option<Bound<PyAny>>,
    ) -> PyResult<()> {
        let kv_block_size = self.kv_block_size as u32;
        let dp_rank = self.dp_rank;
        let warning_count = self.warning_count.clone();
        let inner = self.inner.clone();

        // Use shared monotonic event_id counter from the inner publisher
        let event_id = inner.next_event_id();

        let mm_infos = block_mm_infos
            .as_ref()
            .map(depythonize_block_mm_infos)
            .transpose()?;

        py.allow_threads(|| {
            let block_hashes_u64: Vec<u64> = block_hashes.iter().map(|&h| h as u64).collect();
            let event = KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_hash.map(ExternalSequenceBlockHash::from),
                    blocks: create_stored_blocks(
                        kv_block_size,
                        &token_ids,
                        &num_block_tokens,
                        &block_hashes_u64,
                        lora_id,
                        &warning_count,
                        mm_infos.as_deref(),
                    ),
                }),
                dp_rank,
            };

            inner.publish(event).map_err(to_pyerr)
        })
    }

    fn publish_removed(&self, py: Python, block_hashes: Vec<i64>) -> PyResult<()> {
        let dp_rank = self.dp_rank;
        let inner = self.inner.clone();

        // Use shared monotonic event_id counter from the inner publisher
        let event_id = inner.next_event_id();

        py.allow_threads(|| {
            let block_hashes: Vec<ExternalSequenceBlockHash> = block_hashes
                .into_iter()
                .map(ExternalSequenceBlockHash::from)
                .collect();
            let event = KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
                dp_rank,
            };

            inner.publish(event).map_err(to_pyerr)
        })
    }

    fn shutdown(&mut self) {
        // If no other Arc clones exist, shut down eagerly.
        // Otherwise the Drop impl handles cleanup when the last reference is freed.
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.shutdown();
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct OverlapScores {
    inner: llm_rs::kv_router::protocols::OverlapScores,
}

#[pymethods]
impl OverlapScores {
    #[getter]
    fn scores(&self) -> HashMap<(u64, u32), u32> {
        // Return scores with full WorkerWithDpRank granularity as (worker_id, dp_rank) tuples
        self.inner
            .scores
            .iter()
            .map(|(worker, score)| ((worker.worker_id, worker.dp_rank), *score))
            .collect()
    }

    #[getter]
    fn frequencies(&self) -> Vec<usize> {
        self.inner.frequencies.clone()
    }
}

#[derive(Debug)]
enum RadixTreeRequest {
    FindMatches {
        local_block_hashes: Vec<llm_rs::kv_router::protocols::LocalBlockHash>,
        early_exit: bool,
        response_tx: mpsc::SyncSender<llm_rs::kv_router::protocols::OverlapScores>,
    },
    ApplyEvent {
        worker_id: WorkerId,
        kv_cache_event_bytes: Vec<u8>,
        response_tx: mpsc::SyncSender<PyResult<()>>,
    },
    RemoveWorker {
        worker_id: WorkerId,
        response_tx: mpsc::SyncSender<()>,
    },
    ClearAllBlocks {
        worker_id: WorkerId,
        response_tx: mpsc::SyncSender<()>,
    },
    DumpTreeAsEvents {
        response_tx: mpsc::SyncSender<Vec<llm_rs::kv_router::protocols::RouterEvent>>,
    },
    Shutdown,
}

// NOTE: RadixTree is now thread-safe with pure sync patterns
#[pyclass]
pub(crate) struct RadixTree {
    request_tx: mpsc::Sender<RadixTreeRequest>,
}

#[pymethods]
impl RadixTree {
    #[new]
    #[pyo3(signature = (expiration_duration_secs=None))]
    fn new(expiration_duration_secs: Option<f64>) -> PyResult<Self> {
        let expiration_duration = expiration_duration_secs.map(std::time::Duration::from_secs_f64);

        let (request_tx, request_rx) = mpsc::channel::<RadixTreeRequest>();

        // Spawn dedicated thread with simplified sync processing
        std::thread::spawn(move || {
            let mut radix_tree =
                llm_rs::kv_router::indexer::RadixTree::new_with_frequency(expiration_duration);

            loop {
                match request_rx.recv() {
                    Ok(RadixTreeRequest::Shutdown) => {
                        tracing::debug!("RadixTree thread received shutdown request");
                        break;
                    }
                    Ok(request) => {
                        Self::handle_request(&mut radix_tree, request);
                    }
                    Err(mpsc::RecvError) => {
                        tracing::debug!("RadixTree request channel disconnected");
                        break;
                    }
                }
            }
        });

        Ok(Self { request_tx })
    }

    #[pyo3(signature = (sequence, early_exit=false))]
    fn find_matches(
        &self,
        py: Python,
        sequence: Vec<u64>,
        early_exit: bool,
    ) -> PyResult<OverlapScores> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let local_block_hashes = py.allow_threads(|| {
            sequence
                .into_iter()
                .map(llm_rs::kv_router::protocols::LocalBlockHash)
                .collect()
        });

        let request = RadixTreeRequest::FindMatches {
            local_block_hashes,
            early_exit,
            response_tx,
        };

        self.request_tx.send(request).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "RadixTree background task has shut down",
            )
        })?;

        // Release GIL while waiting for response
        let result = py.allow_threads(move || {
            response_rx.recv().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("RadixTree request was cancelled")
            })
        })?;

        Ok(OverlapScores { inner: result })
    }

    fn apply_event(
        &self,
        py: Python,
        worker_id: WorkerId,
        kv_cache_event_bytes: &[u8],
    ) -> PyResult<()> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let request = RadixTreeRequest::ApplyEvent {
            worker_id,
            kv_cache_event_bytes: kv_cache_event_bytes.to_vec(),
            response_tx,
        };

        self.request_tx.send(request).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "RadixTree background task has shut down",
            )
        })?;

        // Release GIL while waiting for response
        let result = py.allow_threads(move || response_rx.recv());

        result.map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("RadixTree request was cancelled")
        })?
    }

    fn remove_worker(&self, py: Python, worker_id: WorkerId) -> PyResult<()> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let request = RadixTreeRequest::RemoveWorker {
            worker_id,
            response_tx,
        };

        self.request_tx.send(request).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "RadixTree background task has shut down",
            )
        })?;

        // Release GIL while waiting for response
        py.allow_threads(move || {
            response_rx.recv().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("RadixTree request was cancelled")
            })
        })
    }

    fn clear_all_blocks(&self, py: Python, worker_id: WorkerId) -> PyResult<()> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let request = RadixTreeRequest::ClearAllBlocks {
            worker_id,
            response_tx,
        };

        self.request_tx.send(request).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "RadixTree background task has shut down",
            )
        })?;

        // Release GIL while waiting for response
        py.allow_threads(move || {
            response_rx.recv().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("RadixTree request was cancelled")
            })
        })
    }

    fn dump_tree_as_events(&self, py: Python) -> PyResult<Vec<String>> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let request = RadixTreeRequest::DumpTreeAsEvents { response_tx };

        self.request_tx.send(request).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send dump tree request")
        })?;

        // Release GIL while waiting for response from dedicated thread
        let events = py.allow_threads(move || {
            response_rx.recv().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Failed to receive dump tree response",
                )
            })
        })?;

        // Serialize RouterEvent structs to JSON strings with GIL released
        py.allow_threads(move || {
            events
                .into_iter()
                .map(|event| {
                    serde_json::to_string(&event).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Failed to serialize event to JSON: {}",
                            e
                        ))
                    })
                })
                .collect::<Result<Vec<String>, PyErr>>()
        })
    }
}

impl RadixTree {
    fn handle_request(
        radix_tree: &mut llm_rs::kv_router::indexer::RadixTree,
        request: RadixTreeRequest,
    ) {
        match request {
            RadixTreeRequest::FindMatches {
                local_block_hashes,
                early_exit,
                response_tx,
            } => {
                let result = radix_tree.find_matches(local_block_hashes, early_exit);
                let _ = response_tx.send(result);
            }
            RadixTreeRequest::ApplyEvent {
                worker_id,
                kv_cache_event_bytes,
                response_tx,
            } => {
                let result = match serde_json::from_slice::<
                    llm_rs::kv_router::protocols::KvCacheEvent,
                >(&kv_cache_event_bytes)
                {
                    Ok(kv_cache_event) => {
                        let router_event = llm_rs::kv_router::protocols::RouterEvent::new(
                            worker_id,
                            kv_cache_event,
                        );
                        match radix_tree.apply_event(router_event) {
                            Ok(_) => Ok(()),
                            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                                format!("Failed to apply event: {}", e),
                            )),
                        }
                    }
                    Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Failed to deserialize KvCacheEvent: {}",
                        e
                    ))),
                };
                let _ = response_tx.send(result);
            }
            RadixTreeRequest::RemoveWorker {
                worker_id,
                response_tx,
            } => {
                radix_tree.remove_worker(worker_id);
                let _ = response_tx.send(());
            }
            RadixTreeRequest::ClearAllBlocks {
                worker_id,
                response_tx,
            } => {
                radix_tree.clear_all_blocks(worker_id);
                let _ = response_tx.send(());
            }
            RadixTreeRequest::DumpTreeAsEvents { response_tx } => {
                let events = radix_tree.dump_tree_as_events();
                let _ = response_tx.send(events);
            }
            RadixTreeRequest::Shutdown => {
                // This is handled in the main loop
            }
        }
    }
}

// Cleanup when RadixTree is dropped
impl Drop for RadixTree {
    fn drop(&mut self) {
        // Only need graceful shutdown via RadixTreeRequest::Shutdown
        let _ = self.request_tx.send(RadixTreeRequest::Shutdown);
    }
}

/// Helper function to create a KV router from an endpoint using the ModelManager
/// to ensure proper etcd registration.
/// Infers worker type using endpoint naming and router config:
/// - If endpoint name/component contains "prefill", treat as prefill
/// - If router_track_active_blocks is disabled, treat as prefill
/// - Otherwise, default to decode
async fn create_kv_router_from_endpoint(
    endpoint: &Endpoint,
    block_size: usize,
    kv_router_config: Option<llm_rs::kv_router::KvRouterConfig>,
) -> Result<Arc<llm_rs::kv_router::KvRouter>, PyErr> {
    // Create ModelManager and use it to create KvRouter (ensures registration)
    let model_manager = Arc::new(llm_rs::discovery::ModelManager::new());
    let endpoint_id = endpoint.inner.id();
    let namespace = endpoint_id.namespace.to_lowercase();
    let component = endpoint_id.component.to_lowercase();
    let name = endpoint_id.name.to_lowercase();
    let endpoint_is_prefill =
        namespace.contains("prefill") || component.contains("prefill") || name.contains("prefill");
    let track_active_blocks = kv_router_config
        .as_ref()
        .map(|cfg| cfg.router_track_active_blocks)
        .unwrap_or(true);
    let worker_type = if endpoint_is_prefill || !track_active_blocks {
        llm_rs::discovery::WORKER_TYPE_PREFILL
    } else {
        llm_rs::discovery::WORKER_TYPE_DECODE
    };
    let kv_router = model_manager
        .kv_chooser_for(
            &endpoint.inner,
            block_size as u32,
            kv_router_config,
            worker_type,
        )
        .await
        .map_err(to_pyerr)?;

    Ok(kv_router)
}

#[pyclass]
pub(crate) struct KvRouter {
    inner: Arc<RsKvPushRouter>,
}

/// Inject worker_id info from tracker into response's disaggregated_params.
/// This is needed for Python bindings to expose worker routing info since
/// the raw LLMEngineOutput doesn't go through DeltaGenerator (which adds nvext).
fn inject_worker_id_from_tracker(
    data: &mut llm_rs::protocols::common::llm_backend::LLMEngineOutput,
    tracker: &RequestTracker,
) {
    let Some(worker_info) = tracker.get_worker_info() else {
        return;
    };

    let worker_id_json =
        serde_json::to_value(&worker_info).expect("WorkerIdInfo serialization should not fail");

    if let Some(obj) = data
        .disaggregated_params
        .as_mut()
        .and_then(|p| p.as_object_mut())
    {
        obj.insert("worker_id".to_string(), worker_id_json);
    } else {
        data.disaggregated_params = Some(json!({"worker_id": worker_id_json}));
    }
}

// TODO: can this reuse the stream conversion method in Client bindings?
impl KvRouter {
    /// Helper method to process a request and create a Python async generator
    fn process_request_to_stream<'p>(
        py: Python<'p>,
        inner: Arc<RsKvPushRouter>,
        request: llm_rs::protocols::common::preprocessor::PreprocessedRequest,
        tracker: Option<Arc<RequestTracker>>,
    ) -> PyResult<Bound<'p, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let single_in = SingleIn::new(request);
            let stream = inner.generate(single_in).await.map_err(to_pyerr)?;
            let (tx, rx) = tokio::sync::mpsc::channel::<RsAnnotated<PyObject>>(100);

            tokio::spawn(async move {
                let mut stream = stream;
                let mut first_item = true;
                let mut first_token_gauges_observed = false;

                while let Some(mut response) = stream.next().await {
                    if first_item {
                        first_item = false;
                        if let (Some(tracker), Some(data)) = (&tracker, &mut response.data) {
                            inject_worker_id_from_tracker(data, tracker);
                        }
                    }

                    if !first_token_gauges_observed {
                        let has_tokens = response
                            .data
                            .as_ref()
                            .map(|d| !d.token_ids.is_empty())
                            .unwrap_or(false);
                        if has_tokens {
                            if let Some(ref tracker) = tracker {
                                tracker.observe_first_token_gauges();
                            }
                            first_token_gauges_observed = true;
                        }
                    }

                    let py_response = Python::with_gil(|py| {
                        pythonize(py, &response.data)
                            .map(|obj| obj.unbind())
                            .map_err(|e| e.to_string())
                    });

                    match py_response {
                        Ok(obj) => {
                            if tx.send(RsAnnotated::from_data(obj)).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to pythonize response: {}", e);
                            break;
                        }
                    }
                }

                if let Some(ref tracker) = tracker {
                    tracker.observe_finish_gauges();
                }
            });

            Ok(crate::AsyncResponseStream::new(rx, false))
        })
    }
}

#[pymethods]
impl KvRouter {
    /// Create a new KvRouter for KV-aware routing to workers.
    ///
    /// # Arguments
    /// * `endpoint` - The endpoint to route requests to
    /// * `block_size` - KV cache block size for routing decisions
    /// * `kv_router_config` - Configuration for the KV router
    ///
    /// Note: Worker type for Prometheus metrics is inferred from the endpoint name/component
    /// (contains "prefill") or by `router_track_active_blocks` being disabled.
    #[new]
    #[pyo3(signature = (endpoint, block_size, kv_router_config))]
    fn new(
        endpoint: &Endpoint,
        block_size: usize,
        kv_router_config: &super::entrypoint::KvRouterConfig,
    ) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async move {
            let client = endpoint.inner.client().await.map_err(to_pyerr)?;

            // Create PushRouter with KV router mode
            let push_router = rs::pipeline::PushRouter::<
                llm_rs::protocols::common::preprocessor::PreprocessedRequest,
                rs::protocols::annotated::Annotated<
                    llm_rs::protocols::common::llm_backend::LLMEngineOutput,
                >,
            >::from_client(
                client,
                rs::pipeline::network::egress::push_router::RouterMode::KV,
            )
            .await
            .map_err(to_pyerr)?;

            // Create KvRouter using helper function (ensures etcd registration)
            let kv_router = create_kv_router_from_endpoint(
                endpoint,
                block_size,
                Some(kv_router_config.inner()),
            )
            .await?;

            let kv_push_router = RsKvPushRouter::new(push_router, kv_router);

            Ok(Self {
                inner: Arc::new(kv_push_router),
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (token_ids, model, stop_conditions=None, sampling_options=None, output_options=None, router_config_override=None, worker_id=None, dp_rank=None, extra_args=None, block_mm_infos=None, multi_modal_data=None, mm_routing_info=None))]
    fn generate<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
        model: String,
        stop_conditions: Option<PyObject>,
        sampling_options: Option<PyObject>,
        output_options: Option<PyObject>,
        router_config_override: Option<PyObject>,
        worker_id: Option<WorkerId>,
        dp_rank: Option<DpRank>,
        extra_args: Option<PyObject>,
        block_mm_infos: Option<PyObject>,
        multi_modal_data: Option<PyObject>,
        mm_routing_info: Option<PyObject>,
    ) -> PyResult<Bound<'p, PyAny>> {
        // Depythonize the options with defaults
        let stop_conditions: StopConditions = if let Some(obj) = stop_conditions {
            depythonize(obj.bind(py)).map_err(to_pyerr)?
        } else {
            StopConditions::default()
        };

        let sampling_options: SamplingOptions = if let Some(obj) = sampling_options {
            depythonize(obj.bind(py)).map_err(to_pyerr)?
        } else {
            SamplingOptions::default()
        };

        let output_options: OutputOptions = if let Some(obj) = output_options {
            depythonize(obj.bind(py)).map_err(to_pyerr)?
        } else {
            OutputOptions::default()
        };

        let router_config_override: Option<llm_rs::kv_router::RouterConfigOverride> =
            if let Some(obj) = router_config_override {
                Some(depythonize(obj.bind(py)).map_err(to_pyerr)?)
            } else {
                None
            };

        let extra_args: Option<serde_json::Value> = if let Some(obj) = extra_args {
            Some(depythonize(obj.bind(py)).map_err(to_pyerr)?)
        } else {
            None
        };

        let block_mm_infos = block_mm_infos
            .map(|obj| depythonize_block_mm_infos(obj.bind(py)))
            .transpose()?;

        let multi_modal_data: Option<llm_rs::protocols::common::preprocessor::MultimodalDataMap> =
            if let Some(obj) = multi_modal_data {
                Some(depythonize(obj.bind(py)).map_err(to_pyerr)?)
            } else {
                None
            };

        let mm_routing_info: Option<llm_rs::protocols::common::preprocessor::MmRoutingInfo> =
            if let Some(obj) = mm_routing_info {
                Some(depythonize(obj.bind(py)).map_err(to_pyerr)?)
            } else {
                block_mm_infos.map(
                    |infos| llm_rs::protocols::common::preprocessor::MmRoutingInfo {
                        routing_token_ids: token_ids.clone(),
                        block_mm_infos: infos,
                    },
                )
            };

        // Create tracker to capture worker routing info from KvRouter
        let tracker = Arc::new(RequestTracker::new());

        // Build the PreprocessedRequest
        let mut request_builder =
            llm_rs::protocols::common::preprocessor::PreprocessedRequest::builder();
        request_builder
            .model(model)
            .token_ids(token_ids)
            .stop_conditions(stop_conditions)
            .sampling_options(sampling_options)
            .output_options(output_options)
            .router_config_override(router_config_override)
            .multi_modal_data(multi_modal_data)
            .mm_routing_info(mm_routing_info)
            .extra_args(extra_args)
            .tracker(Some(tracker.clone()));

        // Set routing hints if worker_id or dp_rank is provided
        if worker_id.is_some() || dp_rank.is_some() {
            let routing = llm_rs::protocols::common::preprocessor::RoutingHints {
                backend_instance_id: worker_id,
                dp_rank,
                ..Default::default()
            };
            request_builder.routing(Some(routing));
        }

        let request = request_builder.build().map_err(to_pyerr)?;

        // Use the helper method to process the request
        Self::process_request_to_stream(py, self.inner.clone(), request, Some(tracker))
    }

    fn generate_from_request<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
    ) -> PyResult<Bound<'p, PyAny>> {
        // Depythonize the request directly into PreprocessedRequest
        let mut request: llm_rs::protocols::common::preprocessor::PreprocessedRequest =
            depythonize(request.bind(py)).map_err(to_pyerr)?;

        // Create tracker if not already set, to capture worker routing info
        let tracker = match request.tracker {
            Some(ref t) => t.clone(),
            None => {
                let t = Arc::new(RequestTracker::new());
                request.tracker = Some(t.clone());
                t
            }
        };

        // Use the helper method to process the request
        Self::process_request_to_stream(py, self.inner.clone(), request, Some(tracker))
    }

    #[pyo3(signature = (token_ids, router_config_override=None, request_id=None, block_mm_infos=None))]
    fn best_worker<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
        router_config_override: Option<PyObject>,
        request_id: Option<String>,
        block_mm_infos: Option<PyObject>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let router_config_override = if let Some(obj) = router_config_override {
            let override_config: llm_rs::kv_router::RouterConfigOverride =
                depythonize(obj.bind(py)).map_err(to_pyerr)?;
            Some(override_config)
        } else {
            None
        };

        let block_mm_infos = block_mm_infos
            .map(|obj| depythonize_block_mm_infos(obj.bind(py)))
            .transpose()?;

        let chooser = self.inner.chooser.clone();
        let update_states = request_id.is_some();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (best_worker, overlap_blocks) = chooser
                .find_best_match(
                    request_id.as_deref(),
                    &token_ids,
                    block_mm_infos.as_deref(),
                    router_config_override.as_ref(),
                    update_states,
                    None, // lora_name not exposed in Python API yet
                    0.0,
                )
                .await
                .map_err(to_pyerr)?;

            Ok((best_worker.worker_id, best_worker.dp_rank, overlap_blocks))
        })
    }

    /// Mark prefill as completed for a request
    fn mark_prefill_complete<'p>(
        &self,
        py: Python<'p>,
        request_id: String,
    ) -> PyResult<Bound<'p, PyAny>> {
        let chooser = self.inner.chooser.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            chooser
                .mark_prefill_completed(&request_id)
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Free a request by its ID, signaling the router to release resources
    fn free<'p>(&self, py: Python<'p>, request_id: String) -> PyResult<Bound<'p, PyAny>> {
        let chooser = self.inner.chooser.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            chooser.free(&request_id).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn get_potential_loads<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let chooser = self.inner.chooser.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let loads = chooser
                .get_potential_loads(&token_ids, None)
                .await
                .map_err(to_pyerr)?;

            // Return loads without aggregation - each (worker_id, dp_rank) pair is a separate entry
            // Use pythonize to convert Vec<PotentialLoad> to Python list of dicts
            Python::with_gil(|py| {
                pythonize(py, &loads)
                    .map(|obj| obj.unbind())
                    .map_err(to_pyerr)
            })
        })
    }

    /// Dump all events from the KV router's indexer as a JSON string
    fn dump_events<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let chooser = self.inner.chooser.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let events = chooser.dump_events().await.map_err(to_pyerr)?;
            // Serialize to JSON string
            let json_str = serde_json::to_string(&events).map_err(to_pyerr)?;
            Ok(json_str)
        })
    }
}
