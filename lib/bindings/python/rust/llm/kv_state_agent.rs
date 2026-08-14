// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
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
}

impl KvStateAgentHost {
    fn started(&self) -> PyResult<Arc<llm_rs::kv_router::publisher::KvStateAgentHost>> {
        self.inner
            .get()
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("KvStateAgentHost.start() must complete first"))
    }
}
