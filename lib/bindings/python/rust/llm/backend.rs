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

use super::*;
use crate::llm::model_card::ModelDeploymentCard;

use llm_rs::protocols::common::llm_backend::{BackendInput, BackendOutput};
use llm_rs::types::Annotated;

use triton_distributed_runtime::pipeline::{Operator, ServiceBackend, ServiceFrontend, Source};

use crate::engine::PythonAsyncEngine;

#[pyclass]
pub(crate) struct Backend {
    inner: Arc<llm_rs::backend::Backend>,
    endpoint: Endpoint,
}

#[pymethods]
impl Backend {
    #[new]
    fn new(mdc: ModelDeploymentCard, endpoint: Endpoint) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let backend = runtime
            .block_on(llm_rs::backend::Backend::from_mdc(mdc.inner))
            .map_err(to_pyerr)?;
        Ok(Self {
            inner: backend,
            endpoint,
        })
    }

    fn start<'p>(&self, py: Python<'p>, generator: PyObject) -> PyResult<Bound<'p, PyAny>> {
        let frontend = ServiceFrontend::<
            SingleIn<BackendInput>,
            ManyOut<Annotated<BackendOutput>>,
        >::new();

        let backend = self.inner.into_operator();
        let engine = Arc::new(PythonAsyncEngine::new(
            generator,
            self.endpoint.event_loop.clone(),
        )?);
        let engine = ServiceBackend::from_engine(engine);
        let pipeline = frontend
            .link(backend.forward_edge())
            .map_err(to_pyerr)?
            .link(engine)
            .map_err(to_pyerr)?
            .link(backend.backward_edge())
            .map_err(to_pyerr)?
            .link(frontend)
            .map_err(to_pyerr)?;
        let ingress = Ingress::for_engine(pipeline).map_err(to_pyerr)?;
        let builder = self.endpoint.inner.endpoint_builder().handler(ingress);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            builder.start().await.map_err(to_pyerr)?;
            Ok(())
        })
    }
}
