// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::Deserialize;
use tokio::sync::OnceCell;

use super::*;
use crate::{Endpoint, to_pyerr};

#[pyclass]
pub struct KvStateAgentHost {
    endpoint: dynamo_runtime::component::Endpoint,
    max_slots: usize,
    inner: Arc<OnceCell<Arc<llm_rs::kv_router::publisher::KvStateAgentHost>>>,
}

#[pymethods]
impl KvStateAgentHost {
    #[new]
    #[pyo3(signature = (endpoint, max_slots=8))]
    fn new(endpoint: Endpoint, max_slots: usize) -> Self {
        Self {
            endpoint: endpoint.inner,
            max_slots,
            inner: Arc::new(OnceCell::new()),
        }
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let endpoint = self.endpoint.clone();
        let max_slots = self.max_slots;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .get_or_try_init(|| async move {
                    llm_rs::kv_router::publisher::KvStateAgentHost::start(
                        llm_rs::kv_router::publisher::KvStateAgentHostConfig {
                            endpoint,
                            max_slots,
                        },
                    )
                    .await
                })
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Python::with_gil(|py| {
                pythonize::pythonize(py, inner.status().as_ref())
                    .map(|value| value.unbind())
                    .map_err(to_pyerr)
            })
        })
    }

    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.shutdown().await.map_err(to_pyerr)
        })
    }

    fn wait_terminated<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.wait_terminated().await;
            Ok(())
        })
    }
}

impl KvStateAgentHost {
    fn started(&self) -> PyResult<Arc<llm_rs::kv_router::publisher::KvStateAgentHost>> {
        self.inner
            .get()
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("KvStateAgentHost.start() must complete first"))
    }
}

#[derive(Deserialize)]
struct AttachmentDescriptorInput {
    cache_owner_id: String,
    global_dp_rank: u32,
    kv_state_endpoint: String,
    indexer_domain_id: String,
    kv_block_size: u32,
    raw_zmq_endpoint: String,
    #[serde(default)]
    raw_topic: String,
    #[serde(default)]
    image_token_id: Option<u32>,
    #[serde(default)]
    router_hint_source: Option<dynamo_kv_router::protocols::RouterHintSourceMetadata>,
}

#[pyclass]
pub struct KvStateAttachmentOwner {
    endpoint: dynamo_runtime::component::Endpoint,
    worker_id: u64,
    descriptors: Arc<Vec<llm_rs::kv_router::publisher::KvStateAttachmentDescriptor>>,
    inner: Arc<OnceCell<Arc<llm_rs::kv_router::publisher::KvStateAttachmentOwner>>>,
}

#[pymethods]
impl KvStateAttachmentOwner {
    #[new]
    fn new(endpoint: Endpoint, worker_id: u64, descriptors: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inputs: Vec<AttachmentDescriptorInput> =
            pythonize::depythonize(descriptors).map_err(to_pyerr)?;
        let descriptors = inputs
            .into_iter()
            .map(|input| {
                Ok(llm_rs::kv_router::publisher::KvStateAttachmentDescriptor {
                    cache_owner_id: input.cache_owner_id.parse()?,
                    worker: dynamo_kv_router::protocols::WorkerWithDpRank {
                        worker_id,
                        dp_rank: input.global_dp_rank,
                    },
                    kv_state_endpoint: input.kv_state_endpoint.parse()?,
                    indexer_domain_id: input.indexer_domain_id.parse()?,
                    kv_block_size: input.kv_block_size,
                    ingress_protocol:
                        llm_rs::discovery::kv_state_agent::KvStateIngressProtocol::VllmResidencyV1,
                    raw_zmq_endpoint: input.raw_zmq_endpoint,
                    raw_topic: input.raw_topic,
                    image_token_id: input.image_token_id,
                    router_hint_source: input.router_hint_source,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(to_pyerr)?;
        Ok(Self {
            endpoint: endpoint.inner,
            worker_id,
            descriptors: Arc::new(descriptors),
            inner: Arc::new(OnceCell::new()),
        })
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let endpoint = self.endpoint.clone();
        let worker_id = self.worker_id;
        let descriptors = self.descriptors.clone();
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .get_or_try_init(|| async move {
                    llm_rs::kv_router::publisher::KvStateAttachmentOwner::start(
                        endpoint,
                        worker_id,
                        descriptors.as_ref().clone(),
                    )
                    .await
                })
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.close().await.map_err(to_pyerr)
        })
    }
}

impl KvStateAttachmentOwner {
    fn started(&self) -> PyResult<Arc<llm_rs::kv_router::publisher::KvStateAttachmentOwner>> {
        self.inner.get().cloned().ok_or_else(|| {
            PyRuntimeError::new_err("KvStateAttachmentOwner.start() must complete first")
        })
    }
}
