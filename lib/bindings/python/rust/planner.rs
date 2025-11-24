// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TODO: This was ported directly from Python so some changes may be beneficial.
//! - Do we really want to convert to/from string before writing to etcd? It takes Vec<U8>
//! - We can probably replace wrap the whole InnerConnector in a Mutex, it should be uncontended.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use parking_lot::Mutex;
use pyo3::{exceptions::PyException, prelude::*};

use super::to_pyerr;
use dynamo_runtime::transports::etcd::{self, Client, KvCache};
use tokio_util::sync::CancellationToken;

// All three AI's I asked agreed, this is the way
const NONE_SENTINEL: usize = usize::MAX;

struct InnerConnector {
    check_interval: Duration,
    max_wait_time: Duration,
    max_retries: usize,
    namespace: String,
    etcd_client: Client,
    // We need a mutex because we are `async`, but it should never be contended, planner should
    // be calling it from max one thread at once.
    kv_cache: Mutex<Option<Arc<KvCache>>>,

    // On x86 AtomicUsize at Relaxed compiles to usize, it's free
    num_prefill_workers: AtomicUsize,
    num_decode_workers: AtomicUsize,
    decision_id: AtomicUsize,          // NONE_SENTINEL means not set
    first_skip_timestamp: AtomicUsize, // In seconds since epoch, with NONE_SENTINEL
}

#[pyclass]
#[derive(Clone)]
pub struct VirtualConnectorCoordinator(Arc<InnerConnector>);

#[pymethods]
impl VirtualConnectorCoordinator {
    #[new]
    pub fn new(
        drt: super::DistributedRuntime,
        dynamo_namespace: &str,
        check_interval_secs: usize,
        max_wait_time_secs: usize,
        max_retries: usize,
    ) -> PyResult<Self> {
        let check_interval = Duration::from_secs(check_interval_secs as u64);
        let max_wait_time = Duration::from_secs(max_wait_time_secs as u64);
        // default reads from environment variables
        let etcd_config = etcd::ClientOptions::default();
        // etcd client construction is async, but async python constructors are not allowed
        let etcd_client = drt
            .inner
            .runtime()
            .secondary()
            .block_on(
                async move { etcd::Client::new(etcd_config, drt.inner.runtime().clone()).await },
            )
            .map_err(to_pyerr)?;

        let c = InnerConnector {
            check_interval,
            max_wait_time,
            max_retries,
            namespace: dynamo_namespace.to_string(),
            etcd_client,
            kv_cache: Mutex::new(None),
            num_prefill_workers: AtomicUsize::new(NONE_SENTINEL),
            num_decode_workers: AtomicUsize::new(NONE_SENTINEL),
            decision_id: AtomicUsize::new(NONE_SENTINEL),
            first_skip_timestamp: AtomicUsize::new(NONE_SENTINEL),
        };
        Ok(Self(Arc::new(c)))
    }

    #[pyo3(signature = ())]
    pub fn read_state(&self) -> PlannerDecision {
        let current_prefill = load(&self.0.num_prefill_workers);
        let current_decode = load(&self.0.num_decode_workers);
        let current_decision_id = load(&self.0.decision_id);
        PlannerDecision {
            num_prefill_workers: if current_prefill != NONE_SENTINEL {
                current_prefill as isize
            } else {
                -1
            },
            num_decode_workers: if current_decode != NONE_SENTINEL {
                current_decode as isize
            } else {
                -1
            },
            decision_id: if current_decision_id != NONE_SENTINEL {
                current_decision_id as isize
            } else {
                -1
            },
        }
    }

    #[pyo3(signature = ())]
    pub fn async_init<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let prefix = root_key(&self.0.namespace);
        let inner = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kv_cache = KvCache::new(inner.etcd_client.clone(), prefix, HashMap::new())
                .await
                .map_err(to_pyerr)?;
            *inner.kv_cache.lock() = Some(Arc::new(kv_cache));
            inner.load_current_state().await.map_err(to_pyerr)
        })
    }

    #[pyo3(signature = (num_prefill, num_decode))]
    pub fn update_scaling_decision<'p>(
        &self,
        py: Python<'p>,
        num_prefill: Option<usize>,
        num_decode: Option<usize>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let current_prefill = load(&inner.num_prefill_workers);
            let has_prefill_changed = num_prefill.is_some_and(|n| n != current_prefill);

            let current_decode = load(&inner.num_decode_workers);
            let has_decode_changed = num_decode.is_some_and(|n| n != current_decode);

            if !(has_prefill_changed || has_decode_changed) {
                tracing::info!(
                    current_prefill,
                    current_decode,
                    "No scaling needed, skipping update"
                );
                return Ok(());
            }

            // Check if previous scaling is ready
            let is_ready = inner.is_scaling_ready().await;

            if !is_ready {
                let current_time = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map_err(to_pyerr)?
                    .as_secs() as usize;

                // If this is the first time we're skipping, record the timestamp
                if load(&inner.first_skip_timestamp) == NONE_SENTINEL {
                    inner
                        .first_skip_timestamp
                        .store(current_time, Ordering::Relaxed);
                    tracing::info!(
                        decision_id = load(&inner.decision_id),
                        "Previous scaling decision not ready, starting to track skip time"
                    )
                }

                // Check if we've been waiting too long
                let time_waited = current_time - load(&inner.first_skip_timestamp);
                if time_waited < inner.max_wait_time.as_secs() as usize {
                    tracing::warn!(
                        decision_id = load(&inner.decision_id),
                        time_waited,
                        "Previous scaling decision not ready, skipping new decision",
                    );
                    return Ok(());
                } else {
                    tracing::warn!(
                        decision_id = load(&inner.decision_id),
                        scaling_max_wait_time = inner.max_wait_time.as_secs(),
                        "Previous scaling decision not ready, proceeding with new decision anyway"
                    )
                }
            }

            // Reset the skip timestamp since we're making a decision
            inner
                .first_skip_timestamp
                .store(NONE_SENTINEL, Ordering::Relaxed);

            let Some(kv_cache) = inner.kv_cache.lock().as_ref().cloned() else {
                return Err(PyErr::new::<PyException, _>(
                    "Call async_init before using this object",
                ));
            };
            if let Some(new_prefill) = num_prefill {
                inner
                    .num_prefill_workers
                    .store(new_prefill, Ordering::Relaxed);
                kv_cache
                    .put(
                        "num_prefill_workers",
                        new_prefill.to_string().into_bytes(),
                        None,
                    )
                    .await
                    .map_err(to_pyerr)?;
            }
            if let Some(new_decode) = num_decode {
                inner
                    .num_decode_workers
                    .store(new_decode, Ordering::Relaxed);
                kv_cache
                    .put(
                        "num_decode_workers",
                        new_decode.to_string().into_bytes(),
                        None,
                    )
                    .await
                    .map_err(to_pyerr)?;
            }
            let new_decision_id = match load(&inner.decision_id) {
                NONE_SENTINEL => {
                    inner.decision_id.store(0, Ordering::Relaxed);
                    0
                }
                _ => {
                    inner.decision_id.fetch_add(1, Ordering::Relaxed);
                    load(&inner.decision_id)
                }
            };
            kv_cache
                .put(
                    "decision_id",
                    new_decision_id.to_string().into_bytes(),
                    None,
                )
                .await
                .map_err(to_pyerr)?;

            tracing::info!(
                decision_id = new_decision_id,
                ?num_prefill,
                ?num_decode,
                "Updated scaling decision"
            );
            Ok(())
        })
    }

    #[pyo3(signature = ())]
    pub fn wait_for_scaling_completion<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let Some(kv_cache) = inner.kv_cache.lock().as_ref().cloned() else {
                return Err(PyErr::new::<PyException, _>(
                    "Call async_init before using this object",
                ));
            };
            for _ in 0..inner.max_retries {
                match kv_cache.get("scaled_decision_id").await {
                    None => {
                        tokio::time::sleep(inner.check_interval).await;
                    }
                    Some(scaled_decision_id_bytes) => {
                        match String::from_utf8_lossy(&scaled_decision_id_bytes).parse::<usize>() {
                            Ok(scaled_decision_id) => {
                                let current = load(&inner.decision_id);
                                if scaled_decision_id >= current || current == NONE_SENTINEL {
                                    tracing::info!(
                                        decision_id = current,
                                        "Scaling decision completed"
                                    );
                                    return Ok(());
                                }
                            }
                            Err(err) => {
                                tracing::warn!(%err, "Failed to parse scaled_decision_id");
                            }
                        }
                    }
                }
            }
            tracing::warn!(
                decision_id = load(&inner.decision_id),
                scaling_max_wait_time = inner.max_wait_time.as_secs(),
                "Timeout waiting for scaling decision to complete"
            );
            Ok(())
        })
    }
}

impl InnerConnector {
    async fn load_current_state(&self) -> PyResult<()> {
        let Some(kv_cache) = self.kv_cache.lock().as_ref().cloned() else {
            return Err(PyErr::new::<PyException, _>(
                "Call async_init before using this object",
            ));
        };
        let all_values = kv_cache.get_all().await;

        if let Some(v) = all_values.get("num_prefill_workers") {
            match String::from_utf8_lossy(v).parse() {
                Ok(vv) => self.num_prefill_workers.store(vv, Ordering::Relaxed),
                Err(err) => {
                    tracing::error!(
                        "Failed to parse num_prefill_workers from ETCD, using default 0: {err}"
                    );
                    self.num_prefill_workers.store(0, Ordering::Relaxed);
                }
            }
        }

        if let Some(v) = all_values.get("num_decode_workers") {
            match String::from_utf8_lossy(v).parse() {
                Ok(vv) => self.num_decode_workers.store(vv, Ordering::Relaxed),
                Err(err) => {
                    tracing::error!(
                        "Failed to parse num_decode_workers from ETCD, using default 0: {err}"
                    );
                    self.num_decode_workers.store(0, Ordering::Relaxed);
                }
            }
        }

        if let Some(v) = all_values.get("decision_id") {
            match String::from_utf8_lossy(v).parse() {
                Ok(vv) => self.decision_id.store(vv, Ordering::Relaxed),
                Err(err) => {
                    tracing::error!(
                        "Failed to parse decision_id from ETCD, using default None: {err}"
                    );
                    self.decision_id.store(NONE_SENTINEL, Ordering::Relaxed);
                }
            }
        }

        Ok(())
    }

    /// Check if the previous scaling decision has been completed"""
    async fn is_scaling_ready(&self) -> bool {
        let current = load(&self.decision_id);
        // If this is the first decision, it's always ready
        if current == NONE_SENTINEL {
            return true;
        }
        let Some(kv_cache) = self.kv_cache.lock().as_ref().cloned() else {
            tracing::warn!("Call async_init before using this object");
            return false;
        };

        // Check if scaled_decision_id matches current decision_id
        if let Some(scaled_decision_id_bytes) = kv_cache.get("scaled_decision_id").await {
            match String::from_utf8_lossy(&scaled_decision_id_bytes).parse::<usize>() {
                Ok(scaled_decision_id) => {
                    // Success case
                    // We checked for NONE_SENTINEL earlier
                    return scaled_decision_id >= current;
                }
                Err(err) => {
                    tracing::warn!(%err, "Failed to parse scaled_decision_id");
                }
            }
        }
        // If no scaled_decision_id exists, assume not ready
        false
    }
}

#[pyclass]
#[derive(Clone)]
pub struct VirtualConnectorClient(Arc<InnerClient>);

#[pymethods]
impl VirtualConnectorClient {
    #[new]
    pub fn new(drt: super::DistributedRuntime, dynamo_namespace: &str) -> PyResult<Self> {
        let runtime = drt.inner.runtime();
        let cancellation_token = runtime.child_token();
        // default reads from environment variables
        let etcd_config = etcd::ClientOptions::default();
        // etcd client construction is async, but async python constructors are not allowed
        let etcd_client = runtime
            .secondary()
            .block_on(
                async move { etcd::Client::new(etcd_config, drt.inner.runtime().clone()).await },
            )
            .map_err(to_pyerr)?;
        let c = InnerClient {
            etcd_client,
            key: root_key(dynamo_namespace),
            cancellation_token,
        };
        Ok(Self(Arc::new(c)))
    }

    /// Get the current values as a PlannerDecision
    #[pyo3(signature = ())]
    pub fn get<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.get().await.map_err(to_pyerr)
        })
    }

    /// Mark this scaling decision complete
    #[pyo3(signature = (event))]
    pub fn complete<'p>(
        &self,
        py: Python<'p>,
        event: PlannerDecision,
    ) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.complete(event).await.map_err(to_pyerr)
        })
    }

    /// Wait until a new PlannerDecision appears. Will block until there is one to fetch.
    /// Use `get` to fetch the decision.
    #[pyo3(signature = ())]
    pub fn wait<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.wait().await.map_err(to_pyerr)
        })
    }
}

#[pyclass]
#[derive(Clone, Copy)]
/// The decision Planner made. The client should make necessary changes to the environment to make
/// this true, and then call `complete` on the VirtualConnectorClient.
pub struct PlannerDecision {
    #[pyo3(get)]
    pub num_prefill_workers: isize,
    #[pyo3(get)]
    pub num_decode_workers: isize,
    #[pyo3(get)]
    pub decision_id: isize,
}

struct InnerClient {
    key: String,
    etcd_client: Client,
    cancellation_token: CancellationToken,
}

impl InnerClient {
    /// Fetch the latest scaling decision
    async fn get(&self) -> anyhow::Result<PlannerDecision> {
        let mut num_prefill_workers = -1;
        let mut num_decode_workers = -1;
        let mut decision_id = -1;
        for kv in self.etcd_client.kv_get_prefix(&self.key).await? {
            match kv.key_str()? {
                x if x.ends_with("/num_prefill_workers") => {
                    num_prefill_workers = kv.value_str()?.parse()?;
                }
                x if x.ends_with("/num_decode_workers") => {
                    num_decode_workers = kv.value_str()?.parse()?;
                }
                x if x.ends_with("/decision_id") => {
                    decision_id = kv.value_str()?.parse()?;
                }
                x if x.ends_with("/scaled_decision_id") => {
                    // This is the client's response, it doesn't go in PlannerDecision
                }
                x => {
                    tracing::warn!(
                        unexpected_key = x,
                        root = self.key,
                        "Unexpected key in planner etcd"
                    );
                }
            }
        }
        Ok(PlannerDecision {
            num_prefill_workers,
            num_decode_workers,
            decision_id,
        })
    }

    /// Mark this decision as having been handled.
    async fn complete(&self, event: PlannerDecision) -> anyhow::Result<()> {
        self.etcd_client
            .kv_put(
                format!("{}scaled_decision_id", self.key),
                event.decision_id.to_string().as_bytes(),
                None,
            )
            .await
    }

    /// Wait for a new scaling decision. Use `get` when this returns to fetch the values.
    async fn wait(&self) -> anyhow::Result<()> {
        let watcher = self.etcd_client.kv_watch_prefix(&self.key).await?;
        let (_prefix, mut receiver) = watcher.dissolve();
        tokio::select! {
            _ = receiver.recv() => {
                Ok(())
            }
            _ = self.cancellation_token.cancelled() => {
                anyhow::bail!("VirtualConnectorClient.wait: Runtime shutdown");
            },
        }
    }
}

// This compiles to a `mov`, it's basically free
fn load(a: &AtomicUsize) -> usize {
    a.load(Ordering::Relaxed)
}

fn root_key(namespace: &str) -> String {
    format!("v1/{namespace}/planner/")
}
