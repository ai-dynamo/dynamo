// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use anyhow::{Error, Result, anyhow as error};
use pyo3::prelude::*;

use crate::{CancellationToken, DistributedRuntime, engine::*, to_pyerr};

pub use dynamo_llm::endpoint_type::EndpointType;
pub use dynamo_llm::http::service::{error as http_error, service_v2};
pub use dynamo_runtime::{
    pipeline::{AsyncEngine, Data, ManyOut, SingleIn, async_trait},
    protocols::annotated::Annotated,
};

use dynamo_llm::discovery::ModelWatcher;
use dynamo_llm::entrypoint::RouterConfig;
use dynamo_llm::namespace::NamespaceFilter;

#[pyclass]
pub struct HttpService {
    inner: service_v2::HttpService,
    // CancellationToken is already Send + Sync + Clone, no Mutex needed
    cancel_token: Arc<OnceLock<CancellationToken>>,
}

#[pymethods]
impl HttpService {
    #[new]
    #[pyo3(signature = (port=None))]
    pub fn new(port: Option<u16>) -> PyResult<Self> {
        let builder = service_v2::HttpService::builder().port(port.unwrap_or(8080));
        let inner = builder.build().map_err(to_pyerr)?;
        Ok(Self {
            inner,
            cancel_token: Arc::new(OnceLock::new()),
        })
    }

    pub fn add_completions_model(
        &self,
        model: String,
        checksum: String,
        engine: HttpAsyncEngine,
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
        engine: HttpAsyncEngine,
    ) -> PyResult<()> {
        let engine = Arc::new(engine);
        self.inner
            .model_manager()
            .add_chat_completions_model(&model, &checksum, engine)
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

    fn run<'p>(&self, py: Python<'p>, runtime: &DistributedRuntime) -> PyResult<Bound<'p, PyAny>> {
        // Check if run() was already called to avoid creating unnecessary token
        if self.cancel_token.get().is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "HttpService.run() has already been called on this instance",
            ));
        }

        let service = self.inner.clone();
        // Only create token if we passed the check above
        let token = runtime.inner().child_token();

        // Store the token for shutdown - should always succeed after the check above
        self.cancel_token
            .set(CancellationToken {
                inner: token.clone(),
            })
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Race condition detected in HttpService.run()",
                )
            })?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            service.run(token).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn shutdown(&self) {
        // CancellationToken.cancel() is thread-safe, no lock needed
        if let Some(token) = self.cancel_token.get() {
            token.inner.cancel();
        }
    }

    /// Spawn a background discovery watcher that pulls model deployment
    /// cards out of the runtime's discovery store and registers the
    /// matching typed engines on this service's model manager. Returns
    /// immediately after spawning; the watcher runs until the runtime's
    /// primary token is cancelled.
    ///
    /// Tests use this in combination with `Endpoint.serve_bidirectional_endpoint`
    /// and `register_model(ModelType::Realtime + ModelInput::Text, ...)` to
    /// exercise the discovery → typed-PushRouter → realtime engine path
    /// without bringing up a full LLM frontend.
    fn start_discovery_watcher<'p>(
        &self,
        py: Python<'p>,
        runtime: &DistributedRuntime,
    ) -> PyResult<Bound<'p, PyAny>> {
        let drt = runtime.inner().clone();
        let model_manager = self.inner.state_clone().manager_clone();
        let metrics = self.inner.state_clone().metrics_clone();
        let service_arc = Arc::new(self.inner.clone());

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut watch_obj = ModelWatcher::new(
                drt.clone(),
                model_manager,
                RouterConfig::default(),
                0,
                None,
                None,
                None,
                metrics.clone(),
            );

            let discovery = drt.discovery();
            let discovery_stream = discovery
                .list_and_watch(
                    dynamo_runtime::discovery::DiscoveryQuery::AllModels,
                    Some(drt.primary_token()),
                )
                .await
                .map_err(to_pyerr)?;

            let (tx, mut rx) = tokio::sync::mpsc::channel(32);
            watch_obj.set_notify_on_model_update(tx);
            let watch_obj = Arc::new(watch_obj);

            // Endpoint enabler task: when a card lands, flip on the
            // matching HTTP endpoint flag (e.g. EndpointType::Realtime
            // for ModelType::Realtime), so the frontend accepts traffic
            // without the test having to call enable_endpoint explicitly.
            let service_for_enabler = service_arc.clone();
            tokio::spawn(async move {
                while let Some(update) = rx.recv().await {
                    let card = match &update {
                        dynamo_llm::discovery::ModelUpdate::Added(c) => c,
                        dynamo_llm::discovery::ModelUpdate::Removed(c) => c,
                    };
                    let enabled = matches!(update, dynamo_llm::discovery::ModelUpdate::Added(_));
                    for endpoint_type in card.model_type.as_endpoint_types() {
                        service_for_enabler.enable_model_endpoint(endpoint_type, enabled);
                    }
                }
            });

            tokio::spawn(async move {
                watch_obj.watch(discovery_stream, NamespaceFilter::Global).await;
            });

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

#[derive(FromPyObject)]
struct HttpError {
    code: u16,
    message: String,
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
                        // With the Stable ABI, we can't subclass Python's built-in exceptions in PyO3, so instead we
                        // implement the exception in Python and assume that it's an HttpError if the code and message
                        // are present.
                        if let Ok(HttpError { code, message }) = py_err.value(py).extract() {
                            // SSE panics if there are carriage returns or newlines
                            let message = message.replace(['\r', '\n'], "");
                            return Err(http_error::HttpError { code, message })?;
                        }
                        Err(error!("Python Error: {}", py_err))
                    })
                } else {
                    Err(e)
                }
            }
        }
    }
}
