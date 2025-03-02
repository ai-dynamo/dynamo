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
use tracing;
use llm_rs::kv_router::indexer::KvIndexerInterface;

#[pyclass]
pub(crate) struct KvRouter {
    inner: Arc<llm_rs::kv_router::KvRouter>,
}

#[pymethods]
impl KvRouter {
    #[new]
    // [FXIME] 'drt' can be obtained from 'component'
    fn new(drt: DistributedRuntime, component: Component) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async {
            let inner = llm_rs::kv_router::KvRouter::from_runtime(
                drt.inner.clone(),
                component.inner.clone(),
            )
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

    fn publish(
        &self,
        _py: Python,
        request_active_slots: u64,
        request_total_slots: u64,
        kv_active_blocks: u64,
        kv_total_blocks: u64,
    ) -> PyResult<()> {
        self.inner
            .publish(
                llm_rs::kv_router::protocols::ForwardPassMetrics {
                    request_active_slots,
                    request_total_slots,
                    kv_active_blocks,
                    kv_total_blocks,
                }
                .into(),
            )
            .map_err(to_pyerr)
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct OverlapScores(pub llm_rs::kv_router::indexer::OverlapScores);

#[pymethods]
impl OverlapScores {
    fn scores(&self) -> HashMap<llm_rs::kv_router::indexer::WorkerId, u32> {
        self.0.scores.clone()
    }

    fn frequencies(&self) -> Vec<usize> {
        self.0.frequencies.clone()
    }
}

#[pyclass]
pub(crate) struct KvIndexer {
    inner: Arc<llm_rs::kv_router::indexer::KvIndexer>,
}

#[pymethods]
impl KvIndexer {
    #[new]
    fn new(component: Component, token: CancellationToken) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async {
            let kv_subject = component.inner.event_subject(llm_rs::kv_router::KV_EVENT_SUBJECT);
            let inner: Arc<llm_rs::kv_router::indexer::KvIndexer> = llm_rs::kv_router::indexer::KvIndexer::new(token.inner).into();
            let mut kv_events_rx = component.inner.drt().nats_client().client().subscribe(kv_subject).await.map_err(to_pyerr)?;
            let kv_events_tx = inner.event_sender();
        
            // [FIXME] this is the added functionality to the indexer to subscribe to kv events,
            // should have been made to a trait and implemented here? i.e. AsyncEngine style
            tokio::spawn(async move {
                while let Some(event) = kv_events_rx.next().await {
                    let event: llm_rs::kv_router::indexer::RouterEvent = serde_json::from_slice(&event.payload).unwrap();
                    tracing::debug!("received kv event: {:?}", event);
                    if let Err(e) = kv_events_tx.send(event).await {
                        tracing::trace!("failed to send kv event to indexer; shutting down: {:?}", e);
                    }
                }
            });
            Ok(Self { inner })
        })
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
            Ok(OverlapScores(rs_overlap_scores))
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct EndpiontKvMetrics
{
    pub worker_ids: i64,
    pub request_active_slots: u64,
    pub request_total_slots: u64,
    pub kv_active_blocks: u64,
    pub kv_total_blocks: u64,
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct AggregatedMetrics {
    pub endpoints: Vec<EndpiontKvMetrics>,
    pub load_avg: f64,
    pub load_std: f64,
}


#[pyclass]
pub(crate) struct KvMetricsAggregator {
    inner: Arc<llm_rs::kv_router::metrics_aggregator::KvMetricsAggregator>,
}

#[pymethods]
impl KvMetricsAggregator {
    #[new]
    fn new(component: Component, token: CancellationToken) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async {
            let inner = llm_rs::kv_router::metrics_aggregator::KvMetricsAggregator::new(
                component.inner.clone(),
                token.inner,
            )
            .await;
            Ok(Self { inner: inner.into() })
        })
    }
    
    fn get_metrics<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let endpoints = self.inner.get_endpoints();
        let endpiont_kv_metrics = endpoints
            .endpoints
            .iter()
            .map(|x| EndpiontKvMetrics {
                worker_ids: x.worker_id(),
                request_active_slots: x.data.request_active_slots,
                request_total_slots: x.data.request_total_slots,
                kv_active_blocks: x.data.kv_active_blocks,
                kv_total_blocks: x.data.kv_total_blocks,
            })
            .collect();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(AggregatedMetrics {
                endpoints: endpiont_kv_metrics,
                load_avg: endpoints.load_avg,
                load_std: endpoints.load_std,
            })})
    }
}
