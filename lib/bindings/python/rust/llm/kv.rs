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

use super::*;
use llm_rs::kv_router::indexer::KvIndexerInterface;
use rs::traits::events::EventSubscriber;
use tracing;

use llm_rs::kv_router::{indexer::compute_block_hash_for_seq, protocols::*};

#[pyclass]
pub(crate) struct KvRouter {
    inner: Arc<llm_rs::kv_router::KvRouter>,
}

#[pymethods]
impl KvRouter {
    #[new]
    fn new(component: Component, kv_block_size: usize) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async {
            let inner =
                llm_rs::kv_router::KvRouter::new(component.inner.clone(), kv_block_size, None)
                    .await
                    .map_err(to_pyerr)?;
            Ok(Self { inner })
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

#[pyclass]
pub(crate) struct KvMetricsPublisher {
    inner: Arc<llm_rs::kv_router::publisher::KvMetricsPublisher>,
}

#[pymethods]
impl KvMetricsPublisher {
    #[new]
    fn new() -> PyResult<Self> {
        let inner = llm_rs::kv_router::publisher::KvMetricsPublisher::new().map_err(to_pyerr)?;
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
    ) -> PyResult<()> {
        self.inner
            .publish(
                llm_rs::kv_router::protocols::ForwardPassMetrics {
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
pub(crate) struct KvEventPublisher {
    inner: Arc<llm_rs::kv_router::publisher::KvEventPublisher>,
    warning_count: u32,
}

#[pymethods]
impl KvEventPublisher {
    #[new]
    fn new(component: Component, worker_id: i64, kv_block_size: usize) -> PyResult<Self> {
        let inner = llm_rs::kv_router::publisher::KvEventPublisher::new(
            component.inner.clone(),
            worker_id,
            kv_block_size,
        )
        .map_err(to_pyerr)?;
        Ok(Self {
            inner: inner.into(),
            warning_count: 0,
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
        block_hashes: Vec<u64>,
        lora_id: u64,
        parent_hash: Option<u64>,
    ) -> PyResult<()> {
        let event = KvCacheEvent {
            event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                blocks: self.create_stored_blocks(
                    &token_ids,
                    &num_block_tokens,
                    &block_hashes,
                    lora_id,
                ),
            }),
        };

        self.inner.publish(event).map_err(to_pyerr)
    }

    fn publish_removed(&self, _py: Python, event_id: u64, block_hashes: Vec<u64>) -> PyResult<()> {
        let block_hashes: Vec<ExternalSequenceBlockHash> = block_hashes
            .iter()
            .map(|&v| ExternalSequenceBlockHash(v))
            .collect();
        let event = KvCacheEvent {
            event_id,
            data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
        };

        self.inner.publish(event).map_err(to_pyerr)
    }
}

impl KvEventPublisher {
    fn create_stored_block_from_parts(
        &self,
        block_hash: u64,
        token_ids: &[u32],
        _lora_id: u64,
    ) -> KvCacheStoredBlockData {
        let tokens_hash = compute_block_hash_for_seq(token_ids, self.inner.kv_block_size())[0];
        KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(block_hash),
            tokens_hash,
        }
    }

    fn create_stored_blocks(
        &mut self,
        token_ids: &[u32],
        num_block_tokens: &[u64],
        block_hashes: &[u64],
        lora_id: u64,
    ) -> Vec<KvCacheStoredBlockData> {
        let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

        let mut token_offset: usize = 0;
        for (num_tokens_it, block_hash_it) in num_block_tokens.iter().zip(block_hashes.iter()) {
            if (self.warning_count < 3) && (*num_tokens_it != self.inner.kv_block_size() as u64) {
                tracing::warn!(
                    "Block not published. Block size must be {} tokens to be published. Block size is: {}",
                    self.inner.kv_block_size(),
                    *num_tokens_it
                );
                self.warning_count += 1;
                break;
            }

            let tokens = &token_ids[token_offset..(token_offset + *num_tokens_it as usize)];
            blocks.push(self.create_stored_block_from_parts(*block_hash_it, tokens, lora_id));
            token_offset += *num_tokens_it as usize;
        }

        blocks
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
