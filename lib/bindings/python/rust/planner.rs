// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TODO: This was ported directly from Python so some changes may be beneficial.
//! - Do we really want to convert to/from string before writing to etcd? It takes Vec<U8>

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

    decision: Mutex<PlannerDecision>,
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
            decision: Mutex::new(PlannerDecision {
                num_prefill_workers: -1,
                num_decode_workers: -1,
                decision_id: -1,
            }),
            first_skip_timestamp: AtomicUsize::new(NONE_SENTINEL),
        };
        Ok(Self(Arc::new(c)))
    }

    #[pyo3(signature = ())]
    pub fn read_state(&self) -> PlannerDecision {
        *self.0.decision.lock()
    }

    #[pyo3(signature = ())]
    pub fn async_init<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let prefix = root_key(&self.0.namespace);
        let inner = self.0.clone();
        crate::future_into_py(py, async move {
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
        crate::future_into_py(py, async move {
            let current = *inner.decision.lock();
            let current_prefill = current.num_prefill_workers;
            let has_prefill_changed = num_prefill.is_some_and(|n| n as isize != current_prefill);

            let current_decode = current.num_decode_workers;
            let has_decode_changed = num_decode.is_some_and(|n| n as isize != current_decode);

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
                        decision_id = current.decision_id,
                        "Previous scaling decision not ready, starting to track skip time"
                    )
                }

                // Check if we've been waiting too long
                let time_waited = current_time - load(&inner.first_skip_timestamp);
                if time_waited < inner.max_wait_time.as_secs() as usize {
                    tracing::warn!(
                        decision_id = current.decision_id,
                        time_waited,
                        "Previous scaling decision not ready, skipping new decision",
                    );
                    return Ok(());
                } else {
                    tracing::warn!(
                        decision_id = current.decision_id,
                        scaling_max_wait_time = inner.max_wait_time.as_secs(),
                        "Previous scaling decision not ready, proceeding with new decision anyway"
                    )
                }
            }

            let Some(kv_cache) = inner.kv_cache.lock().as_ref().cloned() else {
                return Err(PyErr::new::<PyException, _>(
                    "Call async_init before using this object",
                ));
            };
            let new_prefill = num_prefill.map(|n| n as isize).unwrap_or(current_prefill);
            let new_decode = num_decode.map(|n| n as isize).unwrap_or(current_decode);
            let new_decision_id = current.decision_id + 1;
            let decision = PlannerDecision {
                num_prefill_workers: new_prefill,
                num_decode_workers: new_decode,
                decision_id: new_decision_id,
            };

            // Preserve the existing keys and absent counts, but commit the decision at one revision.
            let entries = [
                ("num_prefill_workers", decision.num_prefill_workers),
                ("num_decode_workers", decision.num_decode_workers),
                ("decision_id", decision.decision_id),
            ]
            .into_iter()
            .filter(|(_, value)| *value != -1)
            .map(|(key, value)| {
                (
                    format!("{}{key}", kv_cache.prefix),
                    value.to_string().into_bytes(),
                )
            })
            .collect();
            inner
                .etcd_client
                .kv_put_many(entries, None)
                .await
                .map_err(to_pyerr)?;

            // A failed publication must leave the previous decision available for retry.
            *inner.decision.lock() = decision;
            inner
                .first_skip_timestamp
                .store(NONE_SENTINEL, Ordering::Relaxed);

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
        crate::future_into_py(py, async move {
            let Some(kv_cache) = inner.kv_cache.lock().as_ref().cloned() else {
                return Err(PyErr::new::<PyException, _>(
                    "Call async_init before using this object",
                ));
            };
            for _ in 0..inner.max_retries {
                // If no scaling decision has been made yet, return immediately
                // rather than waiting for a scaled_decision_id that will never
                // appear (the client only writes it after handling a decision).
                let current = inner.decision.lock().decision_id;
                if current == -1 {
                    tracing::info!(
                        decision_id = current,
                        "No scaling decision pending, skipping wait"
                    );
                    return Ok(());
                }
                match kv_cache.get("scaled_decision_id").await {
                    None => {
                        tokio::time::sleep(inner.check_interval).await;
                    }
                    Some(scaled_decision_id_bytes) => {
                        match String::from_utf8_lossy(&scaled_decision_id_bytes).parse::<usize>() {
                            Ok(scaled_decision_id) => {
                                if scaled_decision_id >= current as usize {
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
                decision_id = inner.decision.lock().decision_id,
                scaling_max_wait_time = inner.max_wait_time.as_secs(),
                "Timeout waiting for scaling decision to complete"
            );
            Ok(())
        })
    }

    /// Return whether the client has acknowledged the current scaling decision.
    #[pyo3(signature = ())]
    pub fn is_scaling_ready<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.0.clone();
        crate::future_into_py(py, async move { Ok(inner.is_scaling_ready().await) })
    }
}

impl InnerConnector {
    async fn load_current_state(&self) -> PyResult<()> {
        let Some(kv_cache) = self.kv_cache.lock().as_ref().cloned() else {
            return Err(PyErr::new::<PyException, _>(
                "Call async_init before using this object",
            ));
        };
        // Read etcd directly: the cache's initial watch snapshot is applied asynchronously.
        let (decision, _) = read_scaling_state(&self.etcd_client, &kv_cache.prefix)
            .await
            .map_err(to_pyerr)?;
        *self.decision.lock() = decision;

        Ok(())
    }

    /// Check if the previous scaling decision has been completed"""
    async fn is_scaling_ready(&self) -> bool {
        let current = self.decision.lock().decision_id;
        // If this is the first decision, it's always ready
        if scaling_decision_is_ready(current, None) {
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
                    return scaling_decision_is_ready(current, Some(scaled_decision_id));
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

fn scaling_decision_is_ready(current: isize, scaled: Option<usize>) -> bool {
    current == -1 || scaled.is_some_and(|scaled| scaled >= current as usize)
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
        crate::future_into_py(py, async move { inner.get().await.map_err(to_pyerr) })
    }

    /// Mark this scaling decision complete
    #[pyo3(signature = (event))]
    pub fn complete<'p>(
        &self,
        py: Python<'p>,
        event: PlannerDecision,
    ) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.0.clone();
        crate::future_into_py(
            py,
            async move { inner.complete(event).await.map_err(to_pyerr) },
        )
    }

    /// Wait for an unacknowledged PlannerDecision, including one already published.
    /// Use `get` to fetch the decision.
    #[pyo3(signature = ())]
    pub fn wait<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.0.clone();
        crate::future_into_py(py, async move { inner.wait().await.map_err(to_pyerr) })
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
        read_scaling_state(&self.etcd_client, &self.key)
            .await
            .map(|(decision, _)| decision)
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

    async fn wait(&self) -> anyhow::Result<()> {
        tokio::select! {
            _ = self.cancellation_token.cancelled() => {
                anyhow::bail!("VirtualConnectorClient.wait: Runtime shutdown");
            },
            result = async {
                // Subscribe before reading so a decision published during the read is not lost.
                let watcher = self.etcd_client.kv_watch_prefix(&self.key).await?;
                let (_, mut receiver) = watcher.dissolve();
                loop {
                    let (decision, scaled_decision_id) =
                        read_scaling_state(&self.etcd_client, &self.key).await?;
                    if !scaling_decision_is_ready(decision.decision_id, scaled_decision_id) {
                        return Ok(());
                    }
                    // ACKs, deletions and resyncs also wake the watch; recheck persisted state.
                    if receiver.recv().await.is_none() {
                        anyhow::bail!("VirtualConnectorClient.wait: Watch closed");
                    }
                }
            } => result,
        }
    }
}

async fn read_scaling_state(
    client: &Client,
    prefix: &str,
) -> anyhow::Result<(PlannerDecision, Option<usize>)> {
    let mut decision = PlannerDecision {
        num_prefill_workers: -1,
        num_decode_workers: -1,
        decision_id: -1,
    };
    let mut scaled_decision_id = None;
    // One prefix read observes the decision and its acknowledgement at the same revision.
    for kv in client.kv_get_prefix(prefix).await? {
        let key = kv.key_str()?;
        match key.strip_prefix(prefix) {
            Some("num_prefill_workers") => {
                decision.num_prefill_workers = kv.value_str()?.parse()?
            }
            Some("num_decode_workers") => decision.num_decode_workers = kv.value_str()?.parse()?,
            Some("decision_id") => decision.decision_id = kv.value_str()?.parse()?,
            Some("scaled_decision_id") => {
                scaled_decision_id = String::from_utf8_lossy(kv.value()).parse().ok();
            }
            _ => tracing::warn!(
                unexpected_key = key,
                root = prefix,
                "Unexpected key in planner etcd"
            ),
        }
    }
    Ok((decision, scaled_decision_id))
}

// This compiles to a `mov`, it's basically free
fn load(a: &AtomicUsize) -> usize {
    a.load(Ordering::Relaxed)
}

fn root_key(namespace: &str) -> String {
    format!("v1/{namespace}/planner/")
}

#[cfg(test)]
mod tests {
    use super::scaling_decision_is_ready;

    #[test]
    fn scaling_is_ready_before_first_decision() {
        assert!(scaling_decision_is_ready(-1, None));
    }

    #[test]
    fn scaling_is_not_ready_without_acknowledgement() {
        assert!(!scaling_decision_is_ready(3, None));
    }

    #[test]
    fn scaling_is_not_ready_with_stale_acknowledgement() {
        assert!(!scaling_decision_is_ready(3, Some(2)));
    }

    #[test]
    fn scaling_is_ready_with_matching_acknowledgement() {
        assert!(scaling_decision_is_ready(3, Some(3)));
    }

    #[test]
    fn scaling_is_ready_with_newer_acknowledgement() {
        assert!(scaling_decision_is_ready(3, Some(4)));
    }
}
