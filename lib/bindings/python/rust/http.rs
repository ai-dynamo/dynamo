// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use anyhow::{Error, Result};
use pyo3::prelude::*;

use crate::{CancellationToken, DistributedRuntime, engine::*, to_pyerr};

pub use dynamo_llm::endpoint_type::EndpointType;
pub use dynamo_llm::http::service::service_v2;
pub use dynamo_runtime::{
    pipeline::{AsyncEngine, Data, ManyOut, SingleIn, async_trait},
    protocols::annotated::Annotated,
};

#[pyclass]
pub struct HttpService {
    inner: service_v2::HttpService,
    // CancellationToken is already Send + Sync + Clone, no Mutex needed
    cancel_token: Arc<OnceLock<CancellationToken>>,
}

#[pymethods]
impl HttpService {
    /// # Arguments
    /// - `port`: listen port (default 8080)
    /// - `wait_for_first_item`: hold the HTTP status of a streaming request
    ///   until the engine's first item, so an error raised by a Python
    ///   generator before its first `yield` maps to the same HTTP response as
    ///   it does for a non-streaming request
    /// - `batch_gateway_url`: internal Batch Gateway URL for files and batch APIs
    #[new]
    #[pyo3(signature = (port=None, *, wait_for_first_item=false, batch_gateway_url=None))]
    pub fn new(
        port: Option<u16>,
        wait_for_first_item: bool,
        batch_gateway_url: Option<String>,
    ) -> PyResult<Self> {
        let mut builder = service_v2::HttpService::builder()
            .port(port.unwrap_or(8080))
            .batch_gateway_url(batch_gateway_url);
        if wait_for_first_item {
            builder = builder
                .streaming_backend_error_check(service_v2::BackendErrorCheck::UntilFirstEvent);
        }
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

        // Hold Phase 2 of Runtime::shutdown until axum finishes draining
        // in-flight requests, so discovery watches are not torn down while
        // multi-minute streams are still draining.
        let guard = runtime.inner().register_graceful_task();

        crate::future_into_py(py, async move {
            let _guard = guard;
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

        self.inner
            .enable_model_endpoint(endpoint_type, enabled)
            .map_err(to_pyerr)
    }
}

#[pyclass(from_py_object)]
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
    pub fn new(generator: Py<PyAny>, event_loop: Py<PyAny>) -> PyResult<Self> {
        Ok(PythonAsyncEngine::new(generator, event_loop)?.into())
    }
}

#[derive(FromPyObject)]
struct HttpError {
    code: u16,
    message: String,
}

/// Keep raw HTTP status ownership while carrying disclosure facts outside the source chain.
fn map_http_async_error(error: Error) -> Error {
    let Some(exception) = error.downcast_ref::<PyErr>() else {
        return error;
    };
    Python::attach(|py| {
        if let Ok(HttpError { code, message }) = exception.value(py).extract() {
            let captured = crate::errors::preserve_python_error_content(
                py,
                exception,
                crate::errors::build_http_like_error(code, message.clone()),
            );
            // An anyhow context remains downcastable without becoming a typed
            // source node that could override main's raw HttpError status path.
            return Error::new(dynamo_error::http::HttpError {
                code,
                message: message.replace(['\r', '\n'], ""),
            })
            .context(captured);
        }
        let mapped = dynamo_runtime::error::DynamoError::msg(format!("Python Error: {exception}"));
        crate::errors::preserve_python_error_content(py, exception, mapped).into()
    })
}

#[async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error> for HttpAsyncEngine
where
    Req: Data + Serialize,
    Resp: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
        self.0.generate(request).await.map_err(map_http_async_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::error::{DynamoError, ErrorClass, UpstreamError};
    use pyo3::{ffi::c_str, types::PyModule};

    #[test]
    fn http_adapter_keeps_raw_status_separate_from_disclosure_context() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c_str!(
                    r#"
class HttpError(Exception):
    __module__ = 'dynamo.common.http.base'
    def __init__(self, code):
        self.code = code
        self.status_code = 503
        self.message = 'PRIVATE_HTTP_SOURCE\r\nprivate continuation'
        super().__init__(self.message)
def fail(code):
    raise HttpError(code)
"#
                ),
                c_str!("http_adapter_test.py"),
                c_str!("http_adapter_test"),
            )
            .unwrap();
            for status in [7, 422, 502, 504] {
                let exception = module
                    .getattr("fail")
                    .unwrap()
                    .call1((status,))
                    .unwrap_err();
                let mapped = map_http_async_error(exception.into());
                let raw = mapped
                    .downcast_ref::<dynamo_error::http::HttpError>()
                    .unwrap();
                assert_eq!(raw.code, status);
                assert!(!raw.message.contains('\r') && !raw.message.contains('\n'));
                assert!(
                    dynamo_error::http::find_canonical_error_in_chain(mapped.as_ref()).is_none()
                );
                let captured = mapped.downcast_ref::<DynamoError>().unwrap();
                let upstream = captured.upstream_error().unwrap();
                assert_eq!(upstream.family_name(), Some("media_fetch"));
                assert_eq!(upstream.http_status_value(), Some(503));
                assert!(upstream.message().contains("PRIVATE_HTTP_SOURCE"));
                assert_eq!(
                    mapped
                        .downcast::<dynamo_error::http::HttpError>()
                        .unwrap()
                        .code,
                    status
                );
            }
        });
    }

    #[test]
    fn http_adapter_leaves_existing_semantic_errors_unchanged() {
        let source = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .public_message("Authored validation replacement")
            .upstream_error(UpstreamError::new("received explanation").http_status(503))
            .build();
        let mapped = map_http_async_error(source.clone().into());
        let captured = mapped.downcast_ref::<DynamoError>().unwrap();
        assert_eq!(captured.class(), source.class());
        assert_eq!(captured.reason(), source.reason());
        assert_eq!(captured.public_details(), source.public_details());
        assert_eq!(captured.upstream_error(), source.upstream_error());
        assert!(
            mapped
                .downcast_ref::<dynamo_error::http::HttpError>()
                .is_none()
        );
    }
}
