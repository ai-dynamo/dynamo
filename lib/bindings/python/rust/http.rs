// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;

use crate::{CancellationToken, engine::*, to_pyerr};

pub use dynamo_llm::endpoint_type::EndpointType;
pub use dynamo_llm::http::service::{error as http_error, service_v2};
pub use dynamo_runtime::{
    Error, Result, error,
    pipeline::{AsyncEngine, Data, ManyOut, SingleIn, async_trait},
    protocols::annotated::Annotated,
};

#[pyclass]
pub struct HttpService {
    inner: service_v2::HttpService,
}

#[pymethods]
impl HttpService {
    #[new]
    #[pyo3(signature = (port=None))]
    pub fn new(port: Option<u16>) -> PyResult<Self> {
        let builder = service_v2::HttpService::builder().port(port.unwrap_or(8080));
        let inner = builder.build().map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    pub fn add_completions_model(&self, model: String, engine: HttpAsyncEngine) -> PyResult<()> {
        let engine = Arc::new(engine);
        self.inner
            .model_manager()
            .add_completions_model(&model, engine)
            .map_err(to_pyerr)
    }

    pub fn add_chat_completions_model(
        &self,
        model: String,
        engine: HttpAsyncEngine,
    ) -> PyResult<()> {
        let engine = Arc::new(engine);
        self.inner
            .model_manager()
            .add_chat_completions_model(&model, engine)
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

    pub fn list_chat_completions_models(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.model_manager().list_chat_completions_models())
    }

    pub fn list_completions_models(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.model_manager().list_completions_models())
    }

    fn run<'p>(&self, py: Python<'p>, token: CancellationToken) -> PyResult<Bound<'p, PyAny>> {
        let service = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            service.run(token.inner).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn enable_endpoint(&self, endpoint_type: String, enabled: bool) -> PyResult<()> {
        let endpoint_type = EndpointType::all()
            .iter()
            .find(|&&ep_type| ep_type.as_str().to_lowercase() == endpoint_type.to_lowercase())
            .copied()
            .ok_or_else(|| {
                let valid_types = EndpointType::all()
                    .iter()
                    .map(|&ep_type| ep_type.as_str().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                to_pyerr(format!(
                    "Invalid endpoint type: '{}'. Valid types are: {}",
                    endpoint_type, valid_types
                ))
            })?;

        self.inner.enable_model_endpoint(endpoint_type, enabled);
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct HttpAsyncEngine(pub PythonAsyncEngine);

impl From<PythonAsyncEngine> for HttpAsyncEngine {
    fn from(engine: PythonAsyncEngine) -> Self {
        Self(engine)
    }
}

#[pymethods]
impl HttpAsyncEngine {
    /// Create a new instance of the HttpAsyncEngine
    /// This is a simple extension of the PythonAsyncEngine that handles HttpError
    /// exceptions from Python and converts them to the Rust version of HttpError
    ///
    /// # Arguments
    /// - `generator`: a Python async generator that will be used to generate responses
    /// - `event_loop`: the Python event loop that will be used to run the generator
    ///
    /// Note: In Rust land, the request and the response are both concrete; however, in
    /// Python land, the request and response are not strongly typed, meaning the generator
    /// could accept a different type of request or return a different type of response
    /// and we would not know until runtime.
    #[new]
    pub fn new(generator: PyObject, event_loop: PyObject) -> PyResult<Self> {
        Ok(PythonAsyncEngine::new(generator, event_loop)?.into())
    }
}

#[async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error> for HttpAsyncEngine
where
    Req: Data + Serialize,
    Resp: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
        match self.0.generate(request).await {
            Ok(res) => Ok(res),

            // Inspect the error - if it was an HttpError from Python, extract the code and message
            // and return the rust version of HttpError
            Err(e) => {
                if let Some(py_err) = e.downcast_ref::<PyErr>() {
                    Python::with_gil(|py| {
                        let err_val = py_err.clone_ref(py).into_value(py);
                        let bound_err = err_val.bind(py);

                        // check: Py03 exceptions cannot be cross-compiled, so we duck-type by name
                        // and fields.
                        if let Ok(type_name) = bound_err.get_type().name()
                            && type_name.to_string().contains("HttpError")
                            && let (Ok(code), Ok(message)) =
                                (bound_err.getattr("code"), bound_err.getattr("message"))
                            && let (Ok(code), Ok(message)) =
                                (code.extract::<u16>(), message.extract::<String>())
                        {
                            // SSE panics if there are carriage returns or newlines
                            let message = message.replace(['\r', '\n'], "");
                            return Err(http_error::HttpError { code, message })?;
                        }
                        Err(error!("Python Error: {}", py_err.to_string()))
                    })
                } else {
                    Err(e)
                }
            }
        }
    }
}
