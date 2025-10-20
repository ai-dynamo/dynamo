// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::PyErr;

use crate::{CancellationToken, engine::*, to_pyerr};

pub use dynamo_llm::grpc::service::kserve;
pub use dynamo_runtime::{
    Error, Result, error,
    pipeline::{AsyncEngine, Data, ManyOut, SingleIn, async_trait},
    protocols::annotated::Annotated,
};

#[pyclass]
pub struct KserveGrpcService {
    inner: kserve::KserveService,
}

#[pymethods]
impl KserveGrpcService {
    #[new]
    #[pyo3(signature = (port=None, host=None))]
    pub fn new(port: Option<u16>, host: Option<String>) -> PyResult<Self> {
        let mut builder = kserve::KserveService::builder();
        if let Some(port) = port {
            builder = builder.port(port);
        }
        if let Some(host) = host {
            builder = builder.host(host);
        }
        let inner = builder.build().map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    pub fn add_completions_model(
        &self,
        model: String,
        checksum: String,
        engine: KserveGrpcAsyncEngine,
    ) -> PyResult<()> {
        let engine = Arc::new(engine);
        self.inner
            .model_manager()
            .add_completions_model(&model, &checksum, engine)
            .map_err(to_pyerr)
    }

    pub fn add_chat_completions_model(
        &self,
        model: String,
        checksum: String,
        engine: KserveGrpcAsyncEngine,
    ) -> PyResult<()> {
        let engine = Arc::new(engine);
        self.inner
            .model_manager()
            .add_chat_completions_model(&model, &checksum, engine)
            .map_err(to_pyerr)
    }

    pub fn add_tensor_model(
        &self,
        model: String,
        checksum: String,
        engine: KserveGrpcAsyncEngine,
    ) -> PyResult<()> {
        let engine = Arc::new(engine);
        self.inner
            .model_manager()
            .add_tensor_model(&model, &checksum, engine)
            .map_err(to_pyerr)
    }

    pub fn remove_completions_model(&self, model: String) -> PyResult<()> {
        self.inner
            .model_manager()
            .remove_completions_model(&model)
            .map_err(to_pyerr)
    }

    pub fn remove_chat_completions_model(&self, model: String) -> PyResult<()> {
        self.inner
            .model_manager()
            .remove_chat_completions_model(&model)
            .map_err(to_pyerr)
    }

    pub fn remove_tensor_model(&self, model: String) -> PyResult<()> {
        self.inner
            .model_manager()
            .remove_tensor_model(&model)
            .map_err(to_pyerr)
    }

    pub fn list_chat_completions_models(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.model_manager().list_chat_completions_models())
    }

    pub fn list_completions_models(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.model_manager().list_completions_models())
    }

    pub fn list_tensor_models(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.model_manager().list_tensor_models())
    }

    fn run<'p>(&self, py: Python<'p>, token: CancellationToken) -> PyResult<Bound<'p, PyAny>> {
        let service = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            service.run(token.inner).await.map_err(to_pyerr)?;
            Ok(())
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct KserveGrpcAsyncEngine(pub PythonAsyncEngine);

impl From<PythonAsyncEngine> for KserveGrpcAsyncEngine {
    fn from(engine: PythonAsyncEngine) -> Self {
        Self(engine)
    }
}

#[pymethods]
impl KserveGrpcAsyncEngine {
    #[new]
    pub fn new(generator: PyObject, event_loop: PyObject) -> PyResult<Self> {
        Ok(PythonAsyncEngine::new(generator, event_loop)?.into())
    }
}

#[async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error> for KserveGrpcAsyncEngine
where
    Req: Data + Serialize,
    Resp: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
        match self.0.generate(request).await {
            Ok(res) => Ok(res),
            Err(e) => {
                if let Some(py_err) = e.downcast_ref::<PyErr>() {
                    Python::with_gil(|_py| Err(error!("Python Error: {}", py_err.to_string())))
                } else {
                    Err(e)
                }
            }
        }
    }
}
