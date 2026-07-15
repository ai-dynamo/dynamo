// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use dynamo_llm::http::service::custom::{
    CustomHttpError, CustomHttpRequest, CustomHttpResponse, CustomHttpRoute,
    CustomHttpRouteCallback,
};
use dynamo_llm::http::service::axum;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pyo3_async_runtimes::TaskLocals;

use crate::context::Context;
use crate::errors::extract_http_like_error;

fn header_map_to_dict<'py>(
    py: Python<'py>,
    headers: &axum::http::HeaderMap,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let mut grouped = BTreeMap::<String, Vec<String>>::new();
    for (name, value) in headers {
        let Ok(value) = value.to_str() else {
            continue;
        };
        grouped
            .entry(name.as_str().to_string())
            .or_default()
            .push(value.to_string());
    }
    for (name, values) in grouped {
        dict.set_item(name, values)?;
    }
    Ok(dict)
}

fn multimap_to_dict<'py>(
    py: Python<'py>,
    values: &BTreeMap<String, Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, values) in values {
        dict.set_item(name, values)?;
    }
    Ok(dict)
}

fn extract_response_headers(
    headers: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<(String, Vec<String>)>> {
    let Some(headers) = headers else {
        return Ok(Vec::new());
    };
    headers
        .iter()
        .map(|(name, value)| {
            let name = name.extract::<String>().map_err(|_| {
                PyTypeError::new_err("response header names must be strings")
            })?;
            let values = if let Ok(value) = value.extract::<String>() {
                vec![value]
            } else {
                value.extract::<Vec<String>>().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "response header {name:?} must be a string or sequence of strings"
                    ))
                })?
            };
            if values.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "response header {name:?} must have at least one value"
                )));
            }
            Ok((name, values))
        })
        .collect()
}

fn extract_body(body: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(body) = body.extract::<String>() {
        return Ok(body.into_bytes());
    }
    if let Ok(body) = body.downcast::<PyBytes>() {
        return Ok(body.as_bytes().to_vec());
    }
    Err(PyTypeError::new_err(
        "response body must be str or bytes",
    ))
}

#[pyclass(name = "_CustomHttpRequest")]
pub struct PyCustomHttpRequest {
    method: String,
    path: String,
    path_params: BTreeMap<String, String>,
    query_string: String,
    query_params: BTreeMap<String, Vec<String>>,
    headers: axum::http::HeaderMap,
    body: Vec<u8>,
    context: Context,
}

impl PyCustomHttpRequest {
    fn from_rust(request: CustomHttpRequest, span: tracing::Span) -> Self {
        Self {
            method: request.method.to_string(),
            path: request.path,
            path_params: request.path_params.into_iter().collect(),
            query_string: request.query_string,
            query_params: request.query_params,
            headers: request.headers,
            body: request.body.to_vec(),
            context: Context::new(
                request.context,
                request.trace_context,
                None,
                request.metadata,
            )
            .with_span(span),
        }
    }
}

#[pymethods]
impl PyCustomHttpRequest {
    #[getter]
    fn method(&self) -> &str {
        &self.method
    }

    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    #[getter]
    fn path_params(&self) -> BTreeMap<String, String> {
        self.path_params.clone()
    }

    #[getter]
    fn query_string(&self) -> &str {
        &self.query_string
    }

    #[getter]
    fn query_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        multimap_to_dict(py, &self.query_params)
    }

    #[getter]
    fn headers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        header_map_to_dict(py, &self.headers)
    }

    #[getter]
    fn body<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.body)
    }

    #[getter]
    fn context(&self) -> Context {
        self.context.clone()
    }

    fn json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value: serde_json::Value = serde_json::from_slice(&self.body)
            .map_err(|err| PyValueError::new_err(format!("invalid JSON request body: {err}")))?;
        Ok(pythonize::pythonize(py, &value)?)
    }
}

#[pyclass(name = "_CustomHttpResponse")]
#[derive(Clone)]
pub struct PyCustomHttpResponse {
    inner: CustomHttpResponse,
}

impl PyCustomHttpResponse {
    fn build(
        body: Vec<u8>,
        status: u16,
        headers: Vec<(String, Vec<String>)>,
    ) -> PyResult<Self> {
        if !(100..=599).contains(&status) {
            return Err(PyValueError::new_err(
                "response status must be between 100 and 599",
            ));
        }
        let inner = CustomHttpResponse::new(status, headers, body)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyCustomHttpResponse {
    #[new]
    #[pyo3(signature = (body, status=200, headers=None))]
    fn new(
        body: &Bound<'_, PyAny>,
        status: u16,
        headers: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Self::build(extract_body(body)?, status, extract_response_headers(headers)?)
    }

    #[staticmethod]
    #[pyo3(signature = (value, status=200, headers=None))]
    fn json(
        value: &Bound<'_, PyAny>,
        status: u16,
        headers: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let value = pythonize::depythonize::<serde_json::Value>(value)
            .map_err(|err| PyTypeError::new_err(format!("value is not JSON serializable: {err}")))?;
        let body = serde_json::to_vec(&value)
            .map_err(|err| PyValueError::new_err(format!("failed to serialize JSON: {err}")))?;
        let mut headers = extract_response_headers(headers)?;
        if !headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        {
            headers.push((
                "content-type".to_string(),
                vec!["application/json".to_string()],
            ));
        }
        Self::build(body, status, headers)
    }

    #[staticmethod]
    #[pyo3(signature = (value, status=200, headers=None))]
    fn text(
        value: String,
        status: u16,
        headers: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut headers = extract_response_headers(headers)?;
        if !headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        {
            headers.push((
                "content-type".to_string(),
                vec!["text/plain; charset=utf-8".to_string()],
            ));
        }
        Self::build(value.into_bytes(), status, headers)
    }

    #[getter]
    fn status(&self) -> u16 {
        self.inner.status.as_u16()
    }

    #[getter]
    fn headers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        header_map_to_dict(py, &self.inner.headers)
    }

    #[getter]
    fn body<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.body)
    }
}

fn python_error(err: PyErr) -> CustomHttpError {
    Python::with_gil(|py| {
        extract_http_like_error(py, &err)
            .map(|(status, message)| CustomHttpError::Http { status, message })
            .unwrap_or_else(|| CustomHttpError::Internal(err.to_string()))
    })
}

#[pyclass(name = "_CustomHttpRoute")]
#[derive(Clone)]
pub struct PyCustomHttpRoute {
    pub(crate) inner: CustomHttpRoute,
}

#[pymethods]
impl PyCustomHttpRoute {
    #[new]
    fn new(
        py: Python<'_>,
        source: String,
        method: String,
        path: String,
        callback: PyObject,
    ) -> PyResult<Self> {
        let method = method
            .parse::<axum::http::Method>()
            .map_err(|err| PyValueError::new_err(format!("invalid HTTP method: {err}")))?;
        let callback = Arc::new(callback);
        let locals = Arc::new(
            pyo3_async_runtimes::tokio::get_current_locals(py).map_err(|err| {
                PyRuntimeError::new_err(format!(
                    "custom routes must be registered from a running asyncio loop: {err}"
                ))
            })?,
        );
        let rust_callback: CustomHttpRouteCallback = Arc::new(move |request| {
            let callback = callback.clone();
            let locals: Arc<TaskLocals> = locals.clone();
            let span = tracing::Span::current();
            Box::pin(async move {
                let future = Python::with_gil(|py| {
                    let request = Py::new(py, PyCustomHttpRequest::from_rust(request, span))?;
                    let coroutine = callback.call1(py, (request,))?;
                    pyo3_async_runtimes::into_future_with_locals(
                        &locals,
                        coroutine.into_bound(py),
                    )
                })
                .map_err(python_error)?;
                let result = future.await.map_err(python_error)?;
                Python::with_gil(|py| {
                    result
                        .bind(py)
                        .extract::<PyRef<'_, PyCustomHttpResponse>>()
                        .map(|response| response.inner.clone())
                        .map_err(|_| {
                            CustomHttpError::Internal(
                                "custom route handler must return Response".to_string(),
                            )
                        })
                })
            })
        });
        Ok(Self {
            inner: CustomHttpRoute {
                method,
                path,
                source,
                callback: rust_callback,
            },
        })
    }

    #[getter]
    fn method(&self) -> &str {
        self.inner.method.as_str()
    }

    #[getter]
    fn path(&self) -> &str {
        &self.inner.path
    }

    #[getter]
    fn source(&self) -> &str {
        &self.inner.source
    }
}
