// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio::sync::OnceCell;

use super::*;
use crate::{Endpoint, to_pyerr};

#[pyclass]
pub struct KvDcRelay {
    endpoint: dynamo_runtime::component::Endpoint,
    dc_id: String,
    config: llm_rs::kv_dc_relay::KvDcRelayConfig,
    inner: Arc<OnceCell<Arc<llm_rs::kv_dc_relay::KvDcRelay>>>,
}

#[pymethods]
impl KvDcRelay {
    #[new]
    #[pyo3(signature = (
        endpoint,
        dc_id,
        namespace_filter=None,
        endpoint_prefix=None,
        publication_threshold=16,
        publication_delay_ms=1,
        recovery_attempt_timeout_ms=30_000,
        *,
        namespaces=None,
        endpoint_prefixes=None,
        watch_all=None,
        expected_unique_blocks=1_048_576,
        bind=None,
        tuning=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        endpoint: Endpoint,
        dc_id: String,
        namespace_filter: Option<String>,
        endpoint_prefix: Option<String>,
        publication_threshold: usize,
        publication_delay_ms: u64,
        recovery_attempt_timeout_ms: u64,
        namespaces: Option<Vec<String>>,
        endpoint_prefixes: Option<Vec<String>>,
        watch_all: Option<bool>,
        expected_unique_blocks: usize,
        bind: Option<String>,
        tuning: Option<std::collections::HashMap<String, u64>>,
    ) -> PyResult<Self> {
        if namespace_filter.is_some() && namespaces.is_some() {
            return Err(PyValueError::new_err(
                "namespace_filter cannot be combined with namespaces",
            ));
        }
        if endpoint_prefix.is_some() && endpoint_prefixes.is_some() {
            return Err(PyValueError::new_err(
                "endpoint_prefix cannot be combined with endpoint_prefixes",
            ));
        }

        let namespaces = namespaces.unwrap_or_else(|| namespace_filter.into_iter().collect());
        let endpoint_prefixes =
            endpoint_prefixes.unwrap_or_else(|| endpoint_prefix.into_iter().collect());
        let watch_all = watch_all.unwrap_or(namespaces.is_empty());
        if watch_all && !namespaces.is_empty() {
            return Err(PyValueError::new_err(
                "watch_all cannot be combined with discovery namespaces",
            ));
        }
        if !watch_all && namespaces.is_empty() {
            return Err(PyValueError::new_err(
                "at least one discovery namespace or watch_all=True is required",
            ));
        }

        #[cfg(not(feature = "kv-dc-relay-wan"))]
        if bind.is_some() {
            return Err(PyRuntimeError::new_err(
                "this Python extension was built without the kv-dc-relay-wan feature",
            ));
        }

        let mut producer = llm_rs::kv_dc_relay::KvDcRelayProducerConfig {
            publication_threshold,
            publication_delay_ms,
            recovery_attempt_timeout_ms,
            expected_unique_blocks,
            ..Default::default()
        };

        #[cfg(feature = "kv-dc-relay-wan")]
        let mut transport = match bind {
            Some(bind) => {
                let bind = bind.parse().map_err(|error| {
                    PyValueError::new_err(format!("invalid KV DC Relay bind address: {error}"))
                })?;
                Some(llm_rs::kv_dc_relay::KvDcRelayTransportConfig::new(bind))
            }
            None => None,
        };

        for (key, &value) in tuning.iter().flatten() {
            let to_usize = |value: u64| {
                usize::try_from(value).map_err(|_| {
                    PyValueError::new_err(format!("tuning value for {key} is out of range"))
                })
            };
            match key.as_str() {
                "publication_threshold" => producer.publication_threshold = to_usize(value)?,
                "publication_delay_ms" => producer.publication_delay_ms = value,
                "recovery_attempt_timeout_ms" => producer.recovery_attempt_timeout_ms = value,
                #[cfg(feature = "kv-dc-relay-wan")]
                "max_message_bytes"
                | "keepalive_interval_ms"
                | "keepalive_timeout_ms"
                | "pool_heartbeat_interval_ms"
                | "readiness_heartbeat_interval_ms"
                | "snapshot_progress_timeout_ms"
                | "load_window_ms"
                | "load_fanout_capacity"
                | "publication_queue_capacity"
                | "publication_queue_bytes"
                | "publication_encoding_concurrency"
                | "max_catalog_subscribers"
                | "max_pool_streams_total"
                | "max_subscribers_per_pool"
                | "max_initialized_pool_hubs"
                | "max_readiness_subscribers"
                | "max_load_subscribers" => {
                    let Some(transport) = transport.as_mut() else {
                        return Err(PyValueError::new_err(format!(
                            "tuning key {key} requires bind to enable the WAN server"
                        )));
                    };
                    match key.as_str() {
                        "max_message_bytes" => transport.max_message_bytes = to_usize(value)?,
                        "keepalive_interval_ms" => transport.keepalive_interval_ms = value,
                        "keepalive_timeout_ms" => transport.keepalive_timeout_ms = value,
                        "pool_heartbeat_interval_ms" => {
                            transport.pool_heartbeat_interval_ms = value
                        }
                        "readiness_heartbeat_interval_ms" => {
                            transport.readiness_heartbeat_interval_ms = value
                        }
                        "snapshot_progress_timeout_ms" => {
                            transport.snapshot_progress_timeout_ms = value
                        }
                        "load_window_ms" => transport.load_window_ms = value,
                        "load_fanout_capacity" => transport.load_fanout_capacity = to_usize(value)?,
                        "publication_queue_capacity" => {
                            transport.publication_queue_capacity = to_usize(value)?
                        }
                        "publication_queue_bytes" => {
                            transport.publication_queue_bytes = to_usize(value)?
                        }
                        "publication_encoding_concurrency" => {
                            transport.publication_encoding_concurrency = to_usize(value)?
                        }
                        "max_catalog_subscribers" => {
                            transport.max_catalog_subscribers = to_usize(value)?
                        }
                        "max_pool_streams_total" => {
                            transport.max_pool_streams_total = to_usize(value)?
                        }
                        "max_subscribers_per_pool" => {
                            transport.max_subscribers_per_pool = to_usize(value)?
                        }
                        "max_initialized_pool_hubs" => {
                            transport.max_initialized_pool_hubs = to_usize(value)?
                        }
                        "max_readiness_subscribers" => {
                            transport.max_readiness_subscribers = to_usize(value)?
                        }
                        "max_load_subscribers" => transport.max_load_subscribers = to_usize(value)?,
                        unexpected => {
                            return Err(PyRuntimeError::new_err(format!(
                                "internal KV DC Relay tuning dispatch error for key {unexpected}"
                            )));
                        }
                    }
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown tuning key {key}; producer keys: publication_threshold, \
                         publication_delay_ms, recovery_attempt_timeout_ms; WAN transport keys \
                         mirror the KvDcRelayTransportConfig field names"
                    )));
                }
            }
        }

        Ok(Self {
            endpoint: endpoint.inner,
            dc_id,
            config: llm_rs::kv_dc_relay::KvDcRelayConfig {
                discovery: llm_rs::kv_dc_relay::KvDcRelayDiscoveryConfig {
                    namespaces,
                    endpoint_prefixes,
                    watch_all,
                },
                producer,
                #[cfg(feature = "kv-dc-relay-wan")]
                transport,
            },
            inner: Arc::new(OnceCell::new()),
        })
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let endpoint = self.endpoint.clone();
        let dc_id = self.dc_id.clone();
        let config = self.config.clone();
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .get_or_try_init(|| async move {
                    llm_rs::kv_dc_relay::KvDcRelay::start(
                        endpoint.component().clone(),
                        dc_id,
                        config,
                    )
                    .await
                    .map(Arc::new)
                })
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    #[cfg(feature = "ckf-diagnostics")]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stats = inner.stats().await.map_err(to_pyerr)?;
            Python::with_gil(|py| {
                pythonize::pythonize(py, &stats)
                    .map(|value| value.unbind())
                    .map_err(to_pyerr)
            })
        })
    }

    fn health<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let health = inner.health().await;
            Python::with_gil(|py| {
                pythonize::pythonize(py, &health)
                    .map(|value| value.unbind())
                    .map_err(to_pyerr)
            })
        })
    }

    fn flush<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.flush().await.map_err(to_pyerr)
        })
    }

    #[cfg(feature = "ckf-diagnostics")]
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        serving_endpoint: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let endpoint = dynamo_runtime::protocols::EndpointId::from(serving_endpoint.as_str());
            let diagnostic = inner
                .diagnostic_snapshot(&endpoint)
                .await
                .map_err(to_pyerr)?;
            Python::with_gil(|py| {
                pythonize::pythonize(py, &diagnostic)
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

    fn wait_for_shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.wait_for_shutdown().await;
            Ok(())
        })
    }
}

impl KvDcRelay {
    fn started(&self) -> PyResult<Arc<llm_rs::kv_dc_relay::KvDcRelay>> {
        self.inner
            .get()
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("KvDcRelay.start() must complete first"))
    }
}
