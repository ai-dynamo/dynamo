// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_llm::http::service::axum::{
    Json, Router,
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, patch, post, put},
};
use dynamo_llm::http::service::{
    FrontendRouteContext as RsFrontendRouteContext, FrontendRouteExtension, FrontendRouteSet,
    RouteDoc,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyIterator, PyTuple},
};
use pythonize::depythonize;
use serde_json::{Value, json};

/// A trusted Python-provided system route hosted by the Dynamo HTTP frontend.
///
/// Handlers are synchronous and return JSON-serializable values. Returning
/// `(status_code, body)` overrides the default 200 status code.
#[pyclass(name = "FrontendRoute")]
#[derive(Clone)]
pub(crate) struct PyFrontendRoute {
    method: Method,
    path: String,
    handler: PyObject,
}

#[pymethods]
impl PyFrontendRoute {
    #[new]
    pub fn new(py: Python<'_>, method: String, path: String, handler: PyObject) -> PyResult<Self> {
        if !handler.bind(py).is_callable() {
            return Err(PyTypeError::new_err(
                "FrontendRoute handler must be callable",
            ));
        }
        if !path.starts_with('/') {
            return Err(PyValueError::new_err(
                "FrontendRoute path must start with '/'",
            ));
        }
        let method = parse_route_method(&method)?;
        Ok(Self {
            method,
            path,
            handler,
        })
    }

    #[getter]
    pub fn method(&self) -> String {
        self.method.to_string()
    }

    #[getter]
    pub fn path(&self) -> String {
        self.path.clone()
    }
}

/// Read-only live frontend state exposed to Python system route handlers.
#[pyclass(name = "FrontendRouteContext")]
#[derive(Clone)]
pub(crate) struct PyFrontendRouteContext {
    inner: RsFrontendRouteContext,
}

#[pymethods]
impl PyFrontendRouteContext {
    pub fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    pub fn has_any_ready_model(&self) -> bool {
        self.inner.manager().has_any_ready_model()
    }

    pub fn is_model_ready_to_serve(&self, model: &str) -> bool {
        self.inner.manager().is_model_ready_to_serve(model)
    }

    pub fn model_display_names(&self) -> Vec<String> {
        sorted_strings(self.inner.manager().model_display_names().into_iter())
    }

    pub fn serving_ready_display_names(&self) -> Vec<String> {
        sorted_strings(
            self.inner
                .manager()
                .serving_ready_display_names()
                .into_iter(),
        )
    }
}

pub(crate) fn frontend_route_extensions_from_py(
    py: Python<'_>,
    routes: Option<PyObject>,
) -> PyResult<Vec<FrontendRouteExtension>> {
    let Some(routes) = routes else {
        return Ok(Vec::new());
    };
    if routes.is_none(py) {
        return Ok(Vec::new());
    }

    let bound = routes.bind(py);
    let iter = PyIterator::from_object(bound)?;
    let mut route_specs = Vec::new();
    for item in iter {
        let item = item?;
        let route = item.extract::<PyRef<'_, PyFrontendRoute>>()?;
        route_specs.push(route.clone());
    }

    if route_specs.is_empty() {
        Ok(Vec::new())
    } else {
        Ok(vec![frontend_route_extension_from_routes(route_specs)])
    }
}

fn frontend_route_extension_from_routes(routes: Vec<PyFrontendRoute>) -> FrontendRouteExtension {
    let routes = Arc::new(routes);
    Arc::new(move |context: RsFrontendRouteContext| {
        let mut docs = Vec::with_capacity(routes.len());
        let mut router = Router::new();

        for route in routes.iter() {
            docs.push(RouteDoc::new(route.method.clone(), route.path.clone()));
            let path = route.path.clone();
            let handler = route.handler.clone();
            let route_context = context.clone();
            router = match route.method {
                Method::GET => router.route(
                    &path,
                    get(move || call_python_frontend_route(handler.clone(), route_context.clone())),
                ),
                Method::POST => router.route(
                    &path,
                    post(move || {
                        call_python_frontend_route(handler.clone(), route_context.clone())
                    }),
                ),
                Method::PUT => router.route(
                    &path,
                    put(move || call_python_frontend_route(handler.clone(), route_context.clone())),
                ),
                Method::PATCH => router.route(
                    &path,
                    patch(move || {
                        call_python_frontend_route(handler.clone(), route_context.clone())
                    }),
                ),
                Method::DELETE => router.route(
                    &path,
                    delete(move || {
                        call_python_frontend_route(handler.clone(), route_context.clone())
                    }),
                ),
                _ => unreachable!("FrontendRoute constructor restricts methods"),
            };
        }

        FrontendRouteSet::new(docs, router)
    })
}

async fn call_python_frontend_route(
    handler: PyObject,
    context: RsFrontendRouteContext,
) -> Response {
    match Python::with_gil(|py| call_python_frontend_route_inner(py, handler, context)) {
        Ok((status, body)) => (status, Json(body)).into_response(),
        Err(err) => {
            tracing::error!(error = %err, "Python system route extension failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "system route extension failed"})),
            )
                .into_response()
        }
    }
}

fn call_python_frontend_route_inner(
    py: Python<'_>,
    handler: PyObject,
    context: RsFrontendRouteContext,
) -> PyResult<(StatusCode, Value)> {
    let py_context = Py::new(py, PyFrontendRouteContext { inner: context })?;
    let result = handler.call1(py, (py_context,))?;
    normalize_route_response(py, result)
}

fn normalize_route_response(py: Python<'_>, result: PyObject) -> PyResult<(StatusCode, Value)> {
    let bound = result.bind(py);
    if bound.hasattr("__await__")? {
        return Err(PyTypeError::new_err(
            "async FrontendRoute handlers are not supported; return JSON synchronously",
        ));
    }

    if let Ok(tuple) = bound.downcast::<PyTuple>()
        && tuple.len() == 2
    {
        let status_code: u16 = tuple.get_item(0)?.extract()?;
        let status = StatusCode::from_u16(status_code)
            .map_err(|e| PyValueError::new_err(format!("invalid status code: {e}")))?;
        let body: Value = depythonize(&tuple.get_item(1)?).map_err(|e| {
            PyValueError::new_err(format!("response body must be JSON-serializable: {e}"))
        })?;
        return Ok((status, body));
    }

    let body: Value = depythonize(bound).map_err(|e| {
        PyValueError::new_err(format!("response body must be JSON-serializable: {e}"))
    })?;
    Ok((StatusCode::OK, body))
}

fn parse_route_method(method: &str) -> PyResult<Method> {
    match method.to_ascii_uppercase().as_str() {
        "GET" => Ok(Method::GET),
        "POST" => Ok(Method::POST),
        "PUT" => Ok(Method::PUT),
        "PATCH" => Ok(Method::PATCH),
        "DELETE" => Ok(Method::DELETE),
        other => Err(PyValueError::new_err(format!(
            "unsupported FrontendRoute method '{other}'; supported methods: GET, POST, PUT, PATCH, DELETE"
        ))),
    }
}

fn sorted_strings(values: impl Iterator<Item = String>) -> Vec<String> {
    let mut values: Vec<String> = values.collect();
    values.sort();
    values
}
