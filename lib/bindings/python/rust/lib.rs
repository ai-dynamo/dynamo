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

use futures::StreamExt;
use once_cell::sync::OnceCell;
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::types::PyBytes;
use pyo3::types::{PyDict, PyList, PyString};
use pyo3::IntoPyObjectExt;
use pyo3::{exceptions::PyException, prelude::*};
use rs::pipeline::network::Ingress;
use std::{fmt::Display, sync::Arc};
use tokio::sync::Mutex;

use dynamo_runtime::{
    self as rs, logging,
    pipeline::{EngineStream, ManyOut, SingleIn},
    protocols::annotated::Annotated as RsAnnotated,
    traits::DistributedRuntimeProvider,
};

use dynamo_llm::{self as llm_rs};

mod engine;
mod http;
mod llm;

type JsonServerStreamingIngress =
    Ingress<SingleIn<serde_json::Value>, ManyOut<RsAnnotated<serde_json::Value>>>;

static INIT: OnceCell<()> = OnceCell::new();

const DEFAULT_ANNOTATED_SETTING: Option<bool> = Some(true);

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    logging::init();
    m.add_function(wrap_pyfunction!(log_message, m)?)?;
    m.add_function(wrap_pyfunction!(register_llm, m)?)?;

    m.add_class::<DistributedRuntime>()?;
    m.add_class::<CancellationToken>()?;
    m.add_class::<Namespace>()?;
    m.add_class::<Component>()?;
    m.add_class::<Endpoint>()?;
    m.add_class::<Client>()?;
    m.add_class::<EtcdClient>()?;
    m.add_class::<AsyncResponseStream>()?;
    m.add_class::<llm::kv::KvRouter>()?;
    m.add_class::<llm::disagg_router::DisaggregatedRouter>()?;
    m.add_class::<llm::kv::KvMetricsPublisher>()?;
    m.add_class::<llm::model_card::ModelDeploymentCard>()?;
    m.add_class::<llm::preprocessor::OAIChatPreprocessor>()?;
    m.add_class::<llm::backend::Backend>()?;
    m.add_class::<llm::kv::OverlapScores>()?;
    m.add_class::<llm::kv::KvIndexer>()?;
    m.add_class::<llm::kv::EndpointKvMetrics>()?;
    m.add_class::<llm::kv::AggregatedMetrics>()?;
    m.add_class::<llm::kv::KvMetricsAggregator>()?;
    m.add_class::<llm::kv::KvEventPublisher>()?;
    m.add_class::<llm::kv::KvRecorder>()?;
    m.add_class::<llm::nats::NatsQueue>()?;
    m.add_class::<http::HttpService>()?;
    m.add_class::<http::HttpError>()?;
    m.add_class::<http::HttpAsyncEngine>()?;
    m.add_class::<EtcdKvCache>()?;
    m.add_class::<ModelType>()?;

    engine::add_to_module(m)?;

    #[cfg(feature = "block-manager")]
    llm::block_manager::add_to_module(m)?;

    Ok(())
}

pub fn to_pyerr<E>(err: E) -> PyErr
where
    E: Display,
{
    PyException::new_err(format!("{}", err))
}

/// Log a message from Python with file and line info
#[pyfunction]
#[pyo3(text_signature = "(level, message, module, file, line)")]
fn log_message(level: &str, message: &str, module: &str, file: &str, line: u32) {
    logging::log_message(level, message, module, file, line);
}

#[pyfunction]
#[pyo3(signature = (model_type, endpoint, model_path, model_name=None))]
fn register_llm<'p>(
    py: Python<'p>,
    model_type: ModelType,
    endpoint: Endpoint,
    model_path: &str,
    model_name: Option<&str>,
) -> PyResult<Bound<'p, PyAny>> {
    let model_type_obj = match model_type {
        ModelType::Chat => llm_rs::model_type::ModelType::Chat,
        ModelType::Completion => llm_rs::model_type::ModelType::Completion,
        ModelType::Backend => llm_rs::model_type::ModelType::Backend,
    };

    let inner_path = model_path.to_string();
    let model_name = model_name.map(|n| n.to_string());
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        // Download from HF, load the ModelDeploymentCard
        let mut local_model = llm_rs::LocalModel::prepare(&inner_path, None, model_name)
            .await
            .map_err(to_pyerr)?;

        // Advertise ourself on etcd so ingress can find us
        local_model
            .attach(&endpoint.inner, model_type_obj)
            .await
            .map_err(to_pyerr)?;

        Ok(())
    })
}

#[pyclass]
#[derive(Clone)]
struct EtcdKvCache {
    inner: Arc<rs::transports::etcd::KvCache>,
}

#[pyclass]
#[derive(Clone)]
struct DistributedRuntime {
    inner: rs::DistributedRuntime,
    event_loop: PyObject,
}

#[pyclass]
#[derive(Clone)]
struct EtcdClient {
    inner: rs::transports::etcd::Client,
}

#[pyclass]
#[derive(Clone)]
struct CancellationToken {
    inner: rs::CancellationToken,
}

#[pyclass]
#[derive(Clone)]
struct Namespace {
    inner: rs::component::Namespace,
    event_loop: PyObject,
}

#[pyclass]
#[derive(Clone)]
struct Component {
    inner: rs::component::Component,
    event_loop: PyObject,
}

#[pyclass]
#[derive(Clone)]
struct Endpoint {
    inner: rs::component::Endpoint,
    event_loop: PyObject,
}

#[pyclass]
#[derive(Clone)]
struct Client {
    router: rs::pipeline::PushRouter<serde_json::Value, serde_json::Value>,
}

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq)]
#[repr(i32)]
enum ModelType {
    Chat = 1,
    Completion = 2,
    Backend = 3,
}

#[pymethods]
impl DistributedRuntime {
    #[new]
    fn new(event_loop: PyObject, is_static: bool) -> PyResult<Self> {
        let worker = rs::Worker::from_settings().map_err(to_pyerr)?;
        INIT.get_or_try_init(|| {
            let primary = worker.tokio_runtime()?;
            pyo3_async_runtimes::tokio::init_with_runtime(primary)
                .map_err(|e| rs::error!("failed to initialize pyo3 static runtime: {:?}", e))?;
            rs::OK(())
        })
        .map_err(to_pyerr)?;

        let runtime = worker.runtime().clone();

        let inner =
            if is_static {
                runtime.secondary().block_on(
                    rs::DistributedRuntime::from_settings_without_discovery(runtime),
                )
            } else {
                runtime
                    .secondary()
                    .block_on(rs::DistributedRuntime::from_settings(runtime))
            };
        let inner = inner.map_err(to_pyerr)?;

        Ok(DistributedRuntime { inner, event_loop })
    }

    fn namespace(&self, name: String) -> PyResult<Namespace> {
        Ok(Namespace {
            inner: self.inner.namespace(name).map_err(to_pyerr)?,
            event_loop: self.event_loop.clone(),
        })
    }

    fn etcd_client(&self) -> PyResult<Option<EtcdClient>> {
        match self.inner.etcd_client().clone() {
            Some(etcd_client) => Ok(Some(EtcdClient { inner: etcd_client })),
            None => Ok(None),
        }
    }

    fn shutdown(&self) {
        self.inner.runtime().shutdown();
    }

    fn event_loop(&self) -> PyObject {
        self.event_loop.clone()
    }
}

#[pymethods]
impl EtcdKvCache {
    #[new]
    fn py_new(
        _etcd_client: &EtcdClient,
        _prefix: String,
        _initial_values: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        // We can't create the KvCache here because it's async, so we'll return an error
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "EtcdKvCache must be created using the 'new' class method",
        ))
    }

    #[staticmethod]
    #[allow(clippy::new_ret_no_self)]
    fn create<'p>(
        py: Python<'p>,
        etcd_client: &EtcdClient,
        prefix: String,
        initial_values: &Bound<'p, PyDict>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let client = etcd_client.inner.clone();

        // Convert Python dict to Rust HashMap
        let mut rust_initial_values = std::collections::HashMap::new();
        for (key, value) in initial_values.iter() {
            let key_str = key.extract::<String>()?;

            // Handle both string and bytes values
            let value_bytes = if let Ok(bytes) = value.extract::<Vec<u8>>() {
                bytes
            } else if let Ok(string) = value.extract::<String>() {
                string.into_bytes()
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Values must be either strings or bytes",
                ));
            };

            rust_initial_values.insert(key_str, value_bytes);
        }

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kv_cache = rs::transports::etcd::KvCache::new(client, prefix, rust_initial_values)
                .await
                .map_err(to_pyerr)?;

            Ok(EtcdKvCache {
                inner: Arc::new(kv_cache),
            })
        })
    }

    fn get<'p>(&self, py: Python<'p>, key: String) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            if let Some(value) = inner.get(&key).await {
                match Python::with_gil(|py| {
                    let py_obj = PyBytes::new(py, &value).into_pyobject(py)?;
                    Ok(py_obj.unbind().into_any())
                }) {
                    Ok(result) => Ok(result),
                    Err(e) => Err(e),
                }
            } else {
                Ok(Python::with_gil(|py| py.None()))
            }
        })
    }

    fn get_all<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let all_values = inner.get_all().await;

            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                for (key, value) in all_values {
                    // Strip the prefix from the key
                    let stripped_key = if let Some(stripped) = key.strip_prefix(&inner.prefix) {
                        stripped.to_string()
                    } else {
                        key
                    };
                    dict.set_item(stripped_key, PyBytes::new(py, &value))?;
                }
                let py_obj = dict.into_pyobject(py)?;
                Ok(py_obj.unbind().into_any())
            })
        })
    }

    #[pyo3(signature = (key, value, lease_id=None))]
    fn put<'p>(
        &self,
        py: Python<'p>,
        key: String,
        value: Vec<u8>,
        lease_id: Option<i64>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.put(&key, value, lease_id).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn delete<'p>(&self, py: Python<'p>, key: String) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.delete(&key).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn clear_all<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Get all keys with the prefix
            let all_keys = inner
                .get_all()
                .await
                .keys()
                .cloned()
                .collect::<Vec<String>>();

            // Delete each key
            for key in all_keys {
                // Strip the prefix from the key before deleting
                if let Some(stripped_key) = key.strip_prefix(&inner.prefix) {
                    inner.delete(stripped_key).await.map_err(to_pyerr)?;
                } else {
                    inner.delete(&key).await.map_err(to_pyerr)?;
                }
            }

            Ok(())
        })
    }
}

#[pymethods]
impl CancellationToken {
    fn cancel(&self) {
        self.inner.cancel();
    }

    fn cancelled<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let token = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            token.cancelled().await;
            Ok(())
        })
    }
}

#[pymethods]
impl Component {
    fn endpoint(&self, name: String) -> PyResult<Endpoint> {
        let inner = self.inner.endpoint(name);
        Ok(Endpoint {
            inner,
            event_loop: self.event_loop.clone(),
        })
    }

    fn create_service<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let builder = self.inner.service_builder();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let _ = builder.create().await.map_err(to_pyerr)?;
            Ok(())
        })
    }
}

#[pymethods]
impl Endpoint {
    #[pyo3(signature = (generator))]
    fn serve_endpoint<'p>(
        &self,
        py: Python<'p>,
        generator: PyObject,
    ) -> PyResult<Bound<'p, PyAny>> {
        let engine = Arc::new(engine::PythonAsyncEngine::new(
            generator,
            self.event_loop.clone(),
        )?);
        let ingress = JsonServerStreamingIngress::for_engine(engine).map_err(to_pyerr)?;
        let builder = self.inner.endpoint_builder().handler(ingress);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            builder.start().await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn client<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let client = inner.client().await.map_err(to_pyerr)?;
            let push_router =
                rs::pipeline::PushRouter::<serde_json::Value, serde_json::Value>::from_client(
                    client,
                    Default::default(),
                )
                .await
                .map_err(to_pyerr)?;
            Ok(Client {
                router: push_router,
            })
        })
    }

    fn lease_id(&self) -> i64 {
        self.inner
            .drt()
            .primary_lease()
            .map(|l| l.id())
            .unwrap_or(0)
    }
}

#[pymethods]
impl Namespace {
    fn component(&self, name: String) -> PyResult<Component> {
        let inner = self.inner.component(name).map_err(to_pyerr)?;
        Ok(Component {
            inner,
            event_loop: self.event_loop.clone(),
        })
    }
}

#[pymethods]
impl EtcdClient {
    #[pyo3(signature = (key, value, lease_id=None))]
    fn kv_create<'p>(
        &self,
        py: Python<'p>,
        key: String,
        value: Vec<u8>,
        lease_id: Option<i64>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client
                .kv_create(key, value, lease_id)
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    #[pyo3(signature = (key, value, lease_id=None))]
    fn kv_create_or_validate<'p>(
        &self,
        py: Python<'p>,
        key: String,
        value: Vec<u8>,
        lease_id: Option<i64>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client
                .kv_create_or_validate(key, value, lease_id)
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn primary_lease_id(&self) -> i64 {
        self.inner.lease_id()
    }

    #[pyo3(signature = (key, value, lease_id=None))]
    fn kv_put<'p>(
        &self,
        py: Python<'p>,
        key: String,
        value: Vec<u8>,
        lease_id: Option<i64>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client
                .kv_put(key, value, lease_id)
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn kv_get_prefix<'p>(&self, py: Python<'p>, prefix: String) -> PyResult<Bound<'p, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = client
                .kv_get_prefix(prefix)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // Convert Vec<KeyValue> to a list of dictionaries
            let py_list = Python::with_gil(|py| {
                let list = PyList::empty(py);
                for kv in result {
                    let dict = PyDict::new(py);
                    dict.set_item("key", String::from_utf8_lossy(kv.key()).to_string())?;
                    dict.set_item("value", PyBytes::new(py, kv.value()))?;
                    dict.set_item("create_revision", kv.create_revision())?;
                    dict.set_item("mod_revision", kv.mod_revision())?;
                    dict.set_item("version", kv.version())?;
                    dict.set_item("lease", kv.lease())?;
                    list.append(dict)?;
                }
                Ok::<Py<PyList>, PyErr>(list.into())
            })?;

            Ok(py_list)
        })
    }

    fn revoke_lease<'p>(&self, py: Python<'p>, lease_id: i64) -> PyResult<Bound<'p, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client.revoke_lease(lease_id).await.map_err(to_pyerr)?;
            Ok(())
        })
    }
}

#[pymethods]
impl Client {
    /// Get list of current endpoints
    fn endpoint_ids(&self) -> Vec<i64> {
        self.router.client.endpoint_ids()
    }

    fn wait_for_endpoints<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.router.client.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .wait_for_endpoints()
                .await
                .map(|v| v.into_iter().map(|cei| cei.id()).collect::<Vec<i64>>())
                .map_err(to_pyerr)
        })
    }

    /// Issue a request to the endpoint using the default routing strategy.
    #[pyo3(signature = (request, annotated=DEFAULT_ANNOTATED_SETTING))]
    fn generate<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
        annotated: Option<bool>,
    ) -> PyResult<Bound<'p, PyAny>> {
        if self.router.client.is_static() {
            self.r#static(py, request, annotated)
        } else {
            self.random(py, request, annotated)
        }
    }

    /// Send a request to the next endpoint in a round-robin fashion.
    #[pyo3(signature = (request, annotated=DEFAULT_ANNOTATED_SETTING))]
    fn round_robin<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
        annotated: Option<bool>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let request: serde_json::Value = pythonize::depythonize(&request.into_bound(py))?;
        let annotated = annotated.unwrap_or(false);

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let client = self.router.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = client.round_robin(request.into()).await.map_err(to_pyerr)?;
            tokio::spawn(process_stream(stream, tx));
            Ok(AsyncResponseStream {
                rx: Arc::new(Mutex::new(rx)),
                annotated,
            })
        })
    }

    /// Send a request to a random endpoint.
    #[pyo3(signature = (request, annotated=DEFAULT_ANNOTATED_SETTING))]
    fn random<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
        annotated: Option<bool>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let request: serde_json::Value = pythonize::depythonize(&request.into_bound(py))?;
        let annotated = annotated.unwrap_or(false);

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let client = self.router.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = client.random(request.into()).await.map_err(to_pyerr)?;
            tokio::spawn(process_stream(stream, tx));
            Ok(AsyncResponseStream {
                rx: Arc::new(Mutex::new(rx)),
                annotated,
            })
        })
    }

    /// Directly send a request to a specific endpoint.
    #[pyo3(signature = (request, endpoint_id, annotated=DEFAULT_ANNOTATED_SETTING))]
    fn direct<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
        endpoint_id: i64,
        annotated: Option<bool>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let request: serde_json::Value = pythonize::depythonize(&request.into_bound(py))?;
        let annotated = annotated.unwrap_or(false);

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let client = self.router.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = client
                .direct(request.into(), endpoint_id)
                .await
                .map_err(to_pyerr)?;

            tokio::spawn(process_stream(stream, tx));

            Ok(AsyncResponseStream {
                rx: Arc::new(Mutex::new(rx)),
                annotated,
            })
        })
    }

    /// Directly send a request to a pre-defined static worker
    #[pyo3(signature = (request, annotated=DEFAULT_ANNOTATED_SETTING))]
    fn r#static<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
        annotated: Option<bool>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let request: serde_json::Value = pythonize::depythonize(&request.into_bound(py))?;
        let annotated = annotated.unwrap_or(false);

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let client = self.router.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = client.r#static(request.into()).await.map_err(to_pyerr)?;

            tokio::spawn(process_stream(stream, tx));

            Ok(AsyncResponseStream {
                rx: Arc::new(Mutex::new(rx)),
                annotated,
            })
        })
    }
}

async fn process_stream(
    stream: EngineStream<serde_json::Value>,
    tx: tokio::sync::mpsc::Sender<RsAnnotated<PyObject>>,
) {
    let mut stream = stream;
    while let Some(response) = stream.next().await {
        // Convert the response to a PyObject using Python's GIL
        // TODO: Remove the clone, but still log the full JSON string on error. But how?
        let annotated: RsAnnotated<serde_json::Value> = match serde_json::from_value(
            response.clone(),
        ) {
            Ok(a) => a,
            Err(err) => {
                tracing::error!(%err, %response, "process_stream: Failed de-serializing JSON into RsAnnotated");
                break;
            }
        };

        let annotated: RsAnnotated<PyObject> = annotated.map_data(|data| {
            let result = Python::with_gil(|py| match pythonize::pythonize(py, &data) {
                Ok(pyobj) => Ok(pyobj.into()),
                Err(e) => Err(e.to_string()),
            });
            result
        });

        let is_error = annotated.is_error();

        // Send the PyObject through the channel or log an error
        if let Err(e) = tx.send(annotated).await {
            tracing::error!("Failed to send response: {:?}", e);
            break;
        }

        if is_error {
            break;
        }
    }
}

#[pyclass]
struct AsyncResponseStream {
    rx: Arc<Mutex<tokio::sync::mpsc::Receiver<RsAnnotated<PyObject>>>>,
    annotated: bool,
}

#[pymethods]
impl AsyncResponseStream {
    /// This method is required to implement the `AsyncIterator` protocol.
    #[pyo3(name = "__aiter__")]
    fn aiter(slf: PyRef<Self>, py: Python) -> PyResult<Py<PyAny>> {
        slf.into_py_any(py)
    }
    /// This method is required to implement the `AsyncIterator` protocol.
    #[pyo3(name = "__anext__")]
    fn next<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let rx = self.rx.clone();
        let annotated = self.annotated;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            loop {
                let value = rx.lock().await.recv().await;
                match value {
                    Some(pyobj) => {
                        let pyobj = match pyobj.ok() {
                            Ok(pyobj) => pyobj,
                            Err(e) => {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e));
                            }
                        };

                        if annotated {
                            let object = Annotated { inner: pyobj };
                            #[allow(deprecated)]
                            let object = Python::with_gil(|py| object.into_py(py));
                            return Ok(object);
                        } else {
                            match pyobj.data {
                                Some(data) => return Ok(data),
                                None => continue,
                            }
                        }
                    }
                    None => return Err(PyStopAsyncIteration::new_err("Stream exhausted")),
                }
            }
        })
    }
}

#[pyclass]
struct Annotated {
    inner: RsAnnotated<PyObject>,
}

#[pymethods]
impl Annotated {
    #[new]
    fn new(data: PyObject) -> Self {
        Annotated {
            inner: RsAnnotated::from_data(data),
        }
    }

    fn is_error(&self) -> bool {
        self.inner.is_error()
    }

    fn data(&self) -> Option<PyObject> {
        self.inner.data.clone()
    }

    fn event(&self) -> Option<String> {
        self.inner.event.clone()
    }

    fn comments(&self) -> Option<Vec<String>> {
        self.inner.comment.clone()
    }

    fn id(&self) -> Option<String> {
        self.inner.id.clone()
    }

    #[pyo3(name = "__repr__")]
    fn _repr(&self, py: Python) -> String {
        let data = self.inner.data.clone().map(|obj| {
            obj.call_method0(py, "__repr__")
                .and_then(|repr_obj| repr_obj.extract::<Py<PyString>>(py))
                .map(|py_str| py_str.to_string_lossy(py).into_owned())
                .unwrap_or_else(|_| "<failed_repr>".to_string())
        });

        format!(
            "Annotated(data={}, event={}, comment={:?}, id={})",
            data.unwrap_or_else(|| "<no_data>".to_string()),
            self.inner.event.as_deref().unwrap_or("None"),
            self.inner.comment.as_deref().unwrap_or(&[]),
            self.inner.id.as_deref().unwrap_or("None")
        )
    }
}
