// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pythonize::{depythonize, pythonize};
use std::collections::HashMap;
use std::ffi::OsString;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::mpsc;
use std::time::Duration;
use tokio_stream::StreamExt;

use super::local_model::RoutingConstraints;
use super::*;
use crate::Endpoint;
#[cfg(any(
    feature = "kv-indexer",
    feature = "slot-tracker",
    feature = "select-service"
))]
use clap::Parser;
#[cfg(feature = "select-service")]
use dynamo_kv_router::config::kv_router_config_from_dynamo_env;
use dynamo_kv_router::config::{KvRouterConfig, RouterConfigOverride};
use dynamo_kv_router::protocols::compute_block_hash_for_seq;
use dynamo_kv_router::protocols::*;
#[cfg(feature = "kv-indexer")]
use dynamo_kv_router::services::indexer::{self, IndexerConfig};
#[cfg(feature = "select-service")]
use dynamo_kv_router::services::selection::{
    self, OverlapScoresRequest, PotentialLoadsRequest, ReservationRequest, SelectAndReserveRequest,
    SelectRequest, SelectionCore, SelectionError, SelectionServiceConfig, WorkerRequest,
};
#[cfg(feature = "slot-tracker")]
use dynamo_kv_router::services::slot_tracker::{self, SlotTrackerConfig};
use rs::pipeline::{AsyncEngine, SingleIn};
use rs::protocols::annotated::Annotated as RsAnnotated;
use tracing;

use llm_rs::kv_router::KvPushRouter as RsKvPushRouter;
use llm_rs::kv_router::publisher::{KvEventSourceConfig, create_stored_blocks};
use llm_rs::protocols::common::timing::RequestTracker;
use llm_rs::protocols::common::{OutputOptions, SamplingOptions, StopConditions};

use super::aic_callback::create_aic_prefill_load_estimator;
use super::entrypoint::AicPerfConfig;

mod demand_driven;

const MAX_RESPONSE_BUFFER_SIZE: usize = tokio::sync::Semaphore::MAX_PERMITS;

#[cfg(any(feature = "slot-tracker", feature = "select-service"))]
fn parse_nonzero_port(value: &str) -> Result<u16, String> {
    let port = value
        .parse::<u16>()
        .map_err(|error| format!("invalid port `{value}`: {error}"))?;
    if port == 0 {
        return Err("port must be greater than zero".to_string());
    }
    Ok(port)
}

#[cfg(all(test, any(feature = "slot-tracker", feature = "select-service")))]
mod replica_sync_cli_tests {
    use super::*;

    #[test]
    fn parses_only_nonzero_ports() {
        assert_eq!(parse_nonzero_port("9000"), Ok(9000));
        assert!(parse_nonzero_port("0").is_err());
    }
}

#[derive(Clone, Copy)]
enum ResponseBufferMode {
    Rendezvous,
    Buffered(usize),
}

fn validate_response_buffer_size(response_buffer_size: isize) -> PyResult<ResponseBufferMode> {
    let capacity = usize::try_from(response_buffer_size)
        .map_err(|_| PyValueError::new_err("response_buffer_size must be non-negative"))?;

    if capacity == 0 {
        return Ok(ResponseBufferMode::Rendezvous);
    }

    if capacity > MAX_RESPONSE_BUFFER_SIZE {
        return Err(PyValueError::new_err(format!(
            "response_buffer_size must not exceed {MAX_RESPONSE_BUFFER_SIZE}"
        )));
    }

    Ok(ResponseBufferMode::Buffered(capacity))
}

fn depythonize_block_mm_infos(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Option<BlockExtraInfo>>> {
    depythonize(obj).map_err(to_pyerr)
}

#[cfg(feature = "kv-indexer")]
#[derive(Parser)]
#[command(
    name = "python -m dynamo.indexer",
    about = "Standalone KV cache indexer"
)]
struct KvIndexerCli {
    /// KV cache block size for initial workers registered via --workers
    #[arg(long)]
    block_size: Option<u32>,

    /// HTTP server port
    #[arg(long, default_value_t = 8090)]
    port: u16,

    /// Number of indexer threads (1 = single-threaded KvIndexer, >1 = ThreadPoolIndexer)
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Initial workers as "worker_id[:dp_rank]=zmq_address,..." (e.g. "1=tcp://host:5557,1:1=tcp://host:5558")
    #[arg(long)]
    workers: Option<String>,

    /// Model name for initial workers registered via --workers
    #[arg(long, default_value = "default")]
    model_name: String,

    /// Tenant ID for initial workers registered via --workers
    #[arg(long, default_value = "default")]
    tenant_id: String,

    /// Comma-separated peer URLs for P2P recovery (e.g. "http://host1:8090,http://host2:8091")
    #[arg(long)]
    peers: Option<String>,

    /// Write access log (JSON lines) to this file
    #[arg(long)]
    access_log: Option<std::path::PathBuf>,

    /// HTTP header name to extract trace-id from
    #[arg(long, default_value = "x-trace-id")]
    trace_id_header: String,

    /// Use local timezone for access log timestamps (default: UTC)
    #[arg(long)]
    access_log_local_time: bool,
}

pub fn run_kv_indexer_cli<I, T>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    #[cfg(feature = "kv-indexer")]
    {
        let cli = KvIndexerCli::try_parse_from(
            std::iter::once(OsString::from("python -m dynamo.indexer"))
                .chain(args.into_iter().map(Into::into)),
        )?;

        let trace_id_header =
            dynamo_kv_router::services::indexer::logging::parse_header_name(&cli.trace_id_header)
                .map_err(|e| {
                anyhow::anyhow!("invalid --trace-id-header '{}': {e}", cli.trace_id_header)
            })?;

        init_standalone_logging();

        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(indexer::run_server(IndexerConfig {
            block_size: cli.block_size,
            port: cli.port,
            threads: cli.threads,
            workers: cli.workers,
            model_name: cli.model_name,
            tenant_id: cli.tenant_id,
            peers: cli.peers,
            access_log: cli.access_log,
            trace_id_header,
            access_log_local_time: cli.access_log_local_time,
        }))
    }

    #[cfg(not(feature = "kv-indexer"))]
    {
        let _ = args;
        anyhow::bail!(
            "dynamo.indexer is not available in this build; reinstall with --features kv-indexer"
        )
    }
}

#[cfg(feature = "slot-tracker")]
#[derive(Debug, Parser)]
#[command(
    name = "python -m dynamo.slot_tracker",
    about = "Standalone KV router slot tracker"
)]
struct SlotTrackerCli {
    /// HTTP server port
    #[arg(long, default_value_t = 8091)]
    port: u16,

    /// Local ZMQ PUB port for replica-sync events
    #[arg(long, value_parser = parse_nonzero_port)]
    replica_sync_port: Option<u16>,

    /// Comma-separated ZMQ PUB endpoints for peer slot trackers
    #[arg(long, value_delimiter = ',', requires = "replica_sync_port")]
    replica_sync_peers: Vec<String>,
}

#[cfg(all(test, feature = "slot-tracker"))]
mod slot_tracker_cli_tests {
    use super::*;

    #[test]
    fn replica_peers_require_port() {
        let error = SlotTrackerCli::try_parse_from([
            "dynamo.slot_tracker",
            "--replica-sync-peers",
            "tcp://slot-a:9000",
        ])
        .unwrap_err();

        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }
}

pub fn run_slot_tracker_cli<I, T>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    #[cfg(feature = "slot-tracker")]
    {
        let cli = SlotTrackerCli::try_parse_from(
            std::iter::once(OsString::from("python -m dynamo.slot_tracker"))
                .chain(args.into_iter().map(Into::into)),
        )?;

        init_standalone_logging();

        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(slot_tracker::run_server(SlotTrackerConfig {
            port: cli.port,
            replica_sync_port: cli.replica_sync_port,
            replica_sync_peers: cli.replica_sync_peers,
        }))
    }

    #[cfg(not(feature = "slot-tracker"))]
    {
        let _ = args;
        anyhow::bail!(
            "dynamo.slot_tracker is not available in this build; reinstall with --features slot-tracker"
        )
    }
}

#[cfg(feature = "select-service")]
#[derive(Debug, Parser)]
#[command(
    name = "python -m dynamo.select_service",
    about = "Runtime-free Dynamo worker selection service"
)]
struct SelectServiceCli {
    /// HTTP server port
    #[arg(long, default_value_t = 8092)]
    port: u16,

    /// Number of KV indexer worker threads
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Comma-separated selector/indexer HTTP URLs used for startup KV recovery
    #[arg(long, value_delimiter = ',')]
    indexer_peers: Vec<String>,

    /// Local ZMQ PUB port for active-load replica events
    #[arg(long, value_parser = parse_nonzero_port)]
    replica_sync_port: Option<u16>,

    /// Comma-separated ZMQ PUB endpoints for peer selectors
    #[arg(long, value_delimiter = ',', requires = "replica_sync_port")]
    replica_sync_peers: Vec<String>,
}

#[cfg(all(test, feature = "select-service"))]
mod select_service_cli_tests {
    use super::*;

    #[test]
    fn parses_selector_peer_planes() {
        let cli = SelectServiceCli::try_parse_from([
            "dynamo.select_service",
            "--indexer-peers",
            "http://indexer-a:8092,http://indexer-b:8092",
            "--replica-sync-port",
            "9000",
            "--replica-sync-peers",
            "tcp://selector-b:9000,tcp://selector-c:9000",
        ])
        .unwrap();

        assert_eq!(cli.indexer_peers.len(), 2);
        assert_eq!(cli.replica_sync_port, Some(9000));
        assert_eq!(cli.replica_sync_peers.len(), 2);
    }

    #[test]
    fn replica_peers_require_port() {
        let error = SelectServiceCli::try_parse_from([
            "dynamo.select_service",
            "--replica-sync-peers",
            "tcp://selector-b:9000",
        ])
        .unwrap_err();

        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }
}

pub fn run_select_service_cli<I, T>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    #[cfg(feature = "select-service")]
    {
        let cli = SelectServiceCli::try_parse_from(
            std::iter::once(OsString::from("python -m dynamo.select_service"))
                .chain(args.into_iter().map(Into::into)),
        )?;

        init_standalone_logging();

        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(selection::run_server(SelectionServiceConfig {
            port: cli.port,
            threads: cli.threads,
            indexer_peers: cli.indexer_peers,
            replica_sync_port: cli.replica_sync_port,
            replica_sync_peers: cli.replica_sync_peers,
            kv_router_config: kv_router_config_from_dynamo_env(),
        }))
    }

    #[cfg(not(feature = "select-service"))]
    {
        let _ = args;
        anyhow::bail!(
            "dynamo.select_service is not available in this build; reinstall with --features select-service"
        )
    }
}

/// Map a [`SelectionError`] to a Python exception: invalid input becomes a
/// `ValueError`, anything else a `SelectionServiceError` carrying `kind` and
/// `status_code`.
#[cfg(feature = "select-service")]
fn selection_to_pyerr(err: SelectionError) -> PyErr {
    if let SelectionError::BadRequest(message) = &err {
        return PyValueError::new_err(message.clone());
    }
    let message = err.to_string();
    let kind = err.kind();
    let status_code = err.status_code();
    Python::with_gil(|py| {
        let pyerr = crate::errors::SelectionServiceError::new_err(message);
        let value = pyerr.value(py);
        let _ = value.setattr("kind", kind);
        let _ = value.setattr("status_code", status_code);
        pyerr
    })
}

/// In-process handle to a runtime-free Dynamo `SelectionCore`.
#[cfg(feature = "select-service")]
#[pyclass]
pub(crate) struct SelectionService {
    inner: Arc<SelectionCore>,
}

#[cfg(feature = "select-service")]
#[pymethods]
impl SelectionService {
    /// Create a selection service. `indexer_threads` sizes the KV indexer pool.
    #[new]
    #[pyo3(signature = (*, indexer_threads = 4))]
    fn new(indexer_threads: usize) -> Self {
        let inner = Arc::new(SelectionCore::new(
            kv_router_config_from_dynamo_env(),
            indexer_threads,
            tokio_util::sync::CancellationToken::new(),
        ));
        Self { inner }
    }

    /// Stop the service: cancel KV-event listeners and scheduling so that
    /// in-flight and queued selections fail fast.
    ///
    /// The KV indexer thread pool is released when the last Python handle is
    /// dropped. Idempotent, and also run automatically on drop.
    fn shutdown(&self) {
        self.inner.shutdown();
    }

    /// Upsert a worker and subscribe to its live KV events via each `kv_events_endpoints`.
    fn upsert_worker<'p>(&self, py: Python<'p>, worker: PyObject) -> PyResult<Bound<'p, PyAny>> {
        let req: WorkerRequest =
            depythonize(worker.bind(py)).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let core = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let record = core.upsert_worker(req).await.map_err(selection_to_pyerr)?;
            Python::with_gil(|py| pythonize(py, &record).map(|o| o.unbind()).map_err(to_pyerr))
        })
    }

    /// Remove a worker and tear down its KV-event listener.
    fn delete_worker<'p>(&self, py: Python<'p>, worker_id: u64) -> PyResult<Bound<'p, PyAny>> {
        let core = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let record = core
                .delete_worker(worker_id)
                .await
                .map_err(selection_to_pyerr)?;
            Python::with_gil(|py| pythonize(py, &record).map(|o| o.unbind()).map_err(to_pyerr))
        })
    }

    /// List catalog records, optionally filtered by model and tenant.
    #[pyo3(signature = (*, model_name = None, tenant_id = None))]
    fn list_workers(
        &self,
        py: Python<'_>,
        model_name: Option<String>,
        tenant_id: Option<String>,
    ) -> PyResult<PyObject> {
        let workers = self
            .inner
            .list_workers(model_name.as_deref(), tenant_id.as_deref());
        pythonize(py, &workers)
            .map(|o| o.unbind())
            .map_err(to_pyerr)
    }

    /// Readiness: whether at least one worker is schedulable, plus catalog state.
    fn ready(&self, py: Python<'_>) -> PyResult<PyObject> {
        pythonize(py, &self.inner.ready())
            .map(|o| o.unbind())
            .map_err(to_pyerr)
    }

    /// Per-worker KV-overlap scores for a prompt (dict, see `OverlapScoresRequest`).
    fn overlap_scores<'p>(&self, py: Python<'p>, request: PyObject) -> PyResult<Bound<'p, PyAny>> {
        let req: OverlapScoresRequest =
            depythonize(request.bind(py)).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let core = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = core.overlap_scores(req).await.map_err(selection_to_pyerr)?;
            Python::with_gil(|py| pythonize(py, &resp).map(|o| o.unbind()).map_err(to_pyerr))
        })
    }

    /// Select the best worker by KV-overlap + load, no booking (dict, see `SelectRequest`).
    fn select<'p>(&self, py: Python<'p>, request: PyObject) -> PyResult<Bound<'p, PyAny>> {
        let req: SelectRequest =
            depythonize(request.bind(py)).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let core = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = core.select(req).await.map_err(selection_to_pyerr)?;
            Python::with_gil(|py| pythonize(py, &resp).map(|o| o.unbind()).map_err(to_pyerr))
        })
    }

    /// Select the best worker and book its load (dict, see `SelectAndReserveRequest`).
    fn select_and_reserve<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
    ) -> PyResult<Bound<'p, PyAny>> {
        let req: SelectAndReserveRequest =
            depythonize(request.bind(py)).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let core = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = core
                .select_and_reserve(req)
                .await
                .map_err(selection_to_pyerr)?;
            Python::with_gil(|py| pythonize(py, &resp).map(|o| o.unbind()).map_err(to_pyerr))
        })
    }

    /// Book a request's load against a chosen worker (dict, see `ReservationRequest`).
    fn create_reservation<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
    ) -> PyResult<Bound<'p, PyAny>> {
        let req: ReservationRequest =
            depythonize(request.bind(py)).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let core = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = core
                .create_reservation(req)
                .await
                .map_err(selection_to_pyerr)?;
            Python::with_gil(|py| pythonize(py, &resp).map(|o| o.unbind()).map_err(to_pyerr))
        })
    }

    /// Mark a reservation's prefill complete; its load shifts prefill -> decode.
    fn prefill_complete<'p>(
        &self,
        py: Python<'p>,
        reservation_id: String,
    ) -> PyResult<Bound<'p, PyAny>> {
        let core = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            core.prefill_complete(&reservation_id)
                .await
                .map_err(selection_to_pyerr)?;
            Ok(())
        })
    }

    /// Record one decode output block for a reservation, advancing its decode load.
    #[pyo3(signature = (reservation_id, *, decay_fraction = None))]
    fn add_output_block(
        &self,
        reservation_id: String,
        decay_fraction: Option<f64>,
    ) -> PyResult<()> {
        self.inner
            .add_output_block(&reservation_id, decay_fraction)
            .map_err(selection_to_pyerr)
    }

    /// Free a finished reservation, releasing its tracked load.
    fn free_reservation<'p>(
        &self,
        py: Python<'p>,
        reservation_id: String,
    ) -> PyResult<Bound<'p, PyAny>> {
        let core = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            core.free_reservation(&reservation_id)
                .await
                .map_err(selection_to_pyerr)?;
            Ok(())
        })
    }

    /// Current per-model active load (pending counts + per-worker potential loads).
    #[pyo3(signature = (*, model_name = None, tenant_id = None))]
    fn loads(
        &self,
        py: Python<'_>,
        model_name: Option<String>,
        tenant_id: Option<String>,
    ) -> PyResult<PyObject> {
        let loads = self
            .inner
            .loads(model_name.as_deref(), tenant_id.as_deref());
        pythonize(py, &loads).map(|o| o.unbind()).map_err(to_pyerr)
    }

    /// Per-worker potential loads for a prompt, without booking (dict, see `PotentialLoadsRequest`).
    fn potential_loads<'p>(&self, py: Python<'p>, request: PyObject) -> PyResult<Bound<'p, PyAny>> {
        let req: PotentialLoadsRequest =
            depythonize(request.bind(py)).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let core = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let resp = core
                .potential_loads(req)
                .await
                .map_err(selection_to_pyerr)?;
            Python::with_gil(|py| pythonize(py, &resp).map(|o| o.unbind()).map_err(to_pyerr))
        })
    }
}

#[cfg(all(test, feature = "select-service"))]
mod selection_service_lifecycle_tests {
    use super::*;

    #[test]
    fn idempotent_shutdown() {
        let service = SelectionService::new(1);
        service.shutdown();

        // Idempotent: a second shutdown and the final drop must not panic.
        service.shutdown();
        drop(service);
    }
}

#[cfg(feature = "select-service")]
impl Drop for SelectionService {
    fn drop(&mut self) {
        self.inner.shutdown();
    }
}

#[cfg(any(
    feature = "kv-indexer",
    feature = "slot-tracker",
    feature = "select-service"
))]
fn init_standalone_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
}

#[pyfunction]
#[pyo3(name = "compute_block_hash_for_seq", signature = (tokens, kv_block_size, block_mm_infos=None, lora_name=None, is_eagle=None))]
pub fn compute_block_hash_for_seq_py(
    _py: Python,
    tokens: Vec<u32>,
    kv_block_size: usize,
    block_mm_infos: Option<Bound<PyAny>>,
    lora_name: Option<String>,
    is_eagle: Option<bool>,
) -> PyResult<Vec<u64>> {
    if kv_block_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "kv_block_size cannot be 0",
        ));
    }

    let mm_infos = block_mm_infos
        .as_ref()
        .map(depythonize_block_mm_infos)
        .transpose()?;

    let hashes = compute_block_hash_for_seq(
        &tokens,
        kv_block_size as u32,
        BlockHashOptions {
            block_mm_infos: mm_infos.as_deref(),
            lora_name: lora_name.as_deref(),
            is_eagle,
        },
    );

    Ok(hashes.into_iter().map(|h| h.0).collect())
}

#[pyclass]
pub(crate) struct WorkerMetricsPublisher {
    inner: Arc<llm_rs::kv_router::publisher::WorkerMetricsPublisher>,
}

#[pymethods]
impl WorkerMetricsPublisher {
    #[new]
    fn new() -> PyResult<Self> {
        let inner =
            llm_rs::kv_router::publisher::WorkerMetricsPublisher::new().map_err(to_pyerr)?;
        Ok(Self {
            inner: inner.into(),
        })
    }

    #[pyo3(signature = (endpoint))]
    fn create_endpoint<'p>(
        &self,
        py: Python<'p>,
        endpoint: Endpoint,
    ) -> PyResult<Bound<'p, PyAny>> {
        let rs_publisher = self.inner.clone();
        let rs_component = endpoint.inner.component().clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            rs_publisher
                .create_endpoint(rs_component)
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Publish worker metrics for load monitoring.
    ///
    /// # Arguments
    /// * `dp_rank` - Data parallel rank of the worker (None defaults to 0)
    /// * `active_decode_blocks` - Scheduler-compatible active decode block count
    /// * `kv_used_blocks` - Authoritative total KV blocks currently in use
    #[pyo3(signature = (dp_rank=None, active_decode_blocks=None, kv_used_blocks=None))]
    fn publish(
        &self,
        dp_rank: Option<u32>,
        active_decode_blocks: Option<u64>,
        kv_used_blocks: Option<u64>,
    ) -> PyResult<()> {
        self.inner
            .publish(dp_rank, active_decode_blocks, kv_used_blocks)
            .map_err(to_pyerr)
    }
}

#[pyclass]
pub(crate) struct MultimodalEmbeddingCachePublisher {
    inner: Arc<llm_rs::kv_router::publisher::MultimodalEmbeddingCachePublisher>,
}

#[pymethods]
impl MultimodalEmbeddingCachePublisher {
    #[new]
    fn new() -> Self {
        let inner = llm_rs::kv_router::publisher::MultimodalEmbeddingCachePublisher::new();
        Self {
            inner: inner.into(),
        }
    }

    #[pyo3(signature = (endpoint))]
    fn create_endpoint<'p>(
        &self,
        py: Python<'p>,
        endpoint: Endpoint,
    ) -> PyResult<Bound<'p, PyAny>> {
        let rs_publisher = self.inner.clone();
        let rs_component = endpoint.inner.component().clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            rs_publisher
                .create_endpoint(rs_component)
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    #[pyo3(signature = (added_keys, removed_keys))]
    fn publish_delta(&self, added_keys: Vec<String>, removed_keys: Vec<String>) -> PyResult<()> {
        self.inner
            .publish_delta(added_keys, removed_keys)
            .map_err(to_pyerr)
    }
}

#[pyclass]
pub(crate) struct KvEventPublisher {
    inner: Arc<llm_rs::kv_router::publisher::KvEventPublisher>,
    kv_block_size: usize,
    dp_rank: DpRank,
    warning_count: Arc<AtomicU32>,
}

impl KvEventPublisher {
    /// Wrap an already-constructed Rust publisher as the Python pyclass.
    ///
    /// Used by the unified-backend bridge (`crate::backend`) so the Worker
    /// can hand a publisher built from a [`PushSource`] back to the Python
    /// engine without going through the Python-side `__init__` (which
    /// requires an `Endpoint` and rebuilds the publisher from scratch).
    pub(crate) fn from_arc(
        inner: Arc<llm_rs::kv_router::publisher::KvEventPublisher>,
        dp_rank: DpRank,
    ) -> Self {
        let kv_block_size = inner.kv_block_size() as usize;
        Self {
            inner,
            kv_block_size,
            dp_rank,
            warning_count: Arc::new(AtomicU32::new(0)),
        }
    }
}

#[pymethods]
impl KvEventPublisher {
    /// Create a KV event publisher that batches raw engine events before forwarding
    /// them to NATS / the event plane.
    ///
    /// Args:
    ///     endpoint: The Dynamo component endpoint for this worker.
    ///     worker_id: Optional worker identity override. When None, the publisher
    ///         uses the endpoint's local connection id.
    ///     kv_block_size: KV cache block size in tokens; must be > 0.
    ///     dp_rank: Data-parallel rank of this worker (default 0).
    ///     enable_local_indexer: When True, a local KV indexer is kept in-process
    ///         so that routers can recover events directly from this worker.
    ///     zmq_endpoint: Optional ZMQ SUB endpoint to read raw engine events from.
    ///     zmq_topic: ZMQ topic filter (default "").
    ///     batching_timeout_ms: Maximum time (in **milliseconds**) to accumulate
    ///         events into a single batch before flushing.
    ///         ``None`` disables batching: every event is published immediately.
    ///         ``50`` to enable batching with a 50 ms window.
    ///         ``0`` is treated as ``None`` (also disables batching).
    ///         Maximum allowed is 15_000 (15 seconds); larger values are capped.
    #[new]
    #[pyo3(signature = (endpoint, worker_id=None, kv_block_size=0, dp_rank=0, enable_local_indexer=false, zmq_endpoint=None, zmq_topic=None, batching_timeout_ms=llm_rs::kv_router::publisher::DEFAULT_BATCHING_TIMEOUT_MS, image_token_id=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        endpoint: Endpoint,
        worker_id: Option<WorkerId>,
        kv_block_size: usize,
        dp_rank: DpRank,
        enable_local_indexer: bool,
        zmq_endpoint: Option<String>,
        zmq_topic: Option<String>,
        batching_timeout_ms: Option<u64>,
        image_token_id: Option<u32>,
    ) -> PyResult<Self> {
        let source_config = zmq_endpoint.map(|ep| KvEventSourceConfig::Zmq {
            endpoint: ep,
            topic: zmq_topic.unwrap_or_default(),
            image_token_id,
        });

        if kv_block_size == 0 {
            return Err(to_pyerr(anyhow::anyhow!("kv_block_size cannot be 0")));
        }

        // Extract component from endpoint
        let component = endpoint.inner.component().clone();

        let inner =
            llm_rs::kv_router::publisher::KvEventPublisher::new_with_local_indexer_and_worker_id(
                component,
                worker_id,
                kv_block_size as u32,
                source_config,
                enable_local_indexer,
                dp_rank,
                batching_timeout_ms,
            )
            .map_err(to_pyerr)?;

        Ok(Self {
            inner: inner.into(),
            kv_block_size,
            dp_rank,
            warning_count: Arc::new(AtomicU32::new(0)),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (token_ids, num_block_tokens, block_hashes, parent_hash=None, block_mm_infos=None, lora_name=None, is_eagle=None))]
    fn publish_stored(
        &self,
        py: Python,
        token_ids: Vec<u32>,
        num_block_tokens: Vec<u64>,
        block_hashes: Vec<i64>,
        parent_hash: Option<i64>,
        block_mm_infos: Option<Bound<PyAny>>,
        lora_name: Option<String>,
        is_eagle: Option<bool>,
    ) -> PyResult<()> {
        let kv_block_size = self.kv_block_size as u32;
        let dp_rank = self.dp_rank;
        let warning_count = self.warning_count.clone();
        let inner = self.inner.clone();

        let event_id = inner.next_event_id();

        let mm_infos = block_mm_infos
            .as_ref()
            .map(depythonize_block_mm_infos)
            .transpose()?;

        py.allow_threads(|| {
            let block_hashes_u64: Vec<u64> = block_hashes.iter().map(|&h| h as u64).collect();
            let event = KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_hash.map(ExternalSequenceBlockHash::from),
                    start_position: None,
                    blocks: create_stored_blocks(
                        kv_block_size,
                        &token_ids,
                        &num_block_tokens,
                        &block_hashes_u64,
                        lora_name.as_deref(),
                        &warning_count,
                        mm_infos.as_deref(),
                        is_eagle,
                        None, // image_token_id: publish path keeps caller-supplied mm_infos
                    ),
                }),
                dp_rank,
            };

            inner.publish(event).map_err(to_pyerr)
        })
    }

    fn publish_removed(&self, py: Python, block_hashes: Vec<i64>) -> PyResult<()> {
        let dp_rank = self.dp_rank;
        let inner = self.inner.clone();

        // Use shared monotonic event_id counter from the inner publisher
        let event_id = inner.next_event_id();

        py.allow_threads(|| {
            let block_hashes: Vec<ExternalSequenceBlockHash> = block_hashes
                .into_iter()
                .map(ExternalSequenceBlockHash::from)
                .collect();
            let event = KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
                dp_rank,
            };

            inner.publish(event).map_err(to_pyerr)
        })
    }

    fn shutdown(&mut self) {
        // If no other Arc clones exist, shut down eagerly.
        // Otherwise the Drop impl handles cleanup when the last reference is freed.
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.shutdown();
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct OverlapScores {
    inner: dynamo_kv_router::protocols::OverlapScores,
}

#[pymethods]
impl OverlapScores {
    #[getter]
    fn scores(&self) -> HashMap<(u64, u32), u32> {
        // Return scores with full WorkerWithDpRank granularity as (worker_id, dp_rank) tuples
        self.inner
            .scores
            .iter()
            .map(|(worker, score)| ((worker.worker_id, worker.dp_rank), *score))
            .collect()
    }

    #[getter]
    fn frequencies(&self) -> Vec<usize> {
        self.inner.frequencies.clone()
    }
}

#[derive(Debug)]
enum RadixTreeRequest {
    FindMatches {
        local_block_hashes: Vec<LocalBlockHash>,
        early_exit: bool,
        response_tx: mpsc::SyncSender<dynamo_kv_router::protocols::OverlapScores>,
    },
    ApplyEvent {
        worker_id: WorkerId,
        kv_cache_event_bytes: Vec<u8>,
        response_tx: mpsc::SyncSender<PyResult<()>>,
    },
    RemoveWorker {
        worker_id: WorkerId,
        response_tx: mpsc::SyncSender<()>,
    },
    ClearAllBlocks {
        worker_id: WorkerId,
        response_tx: mpsc::SyncSender<()>,
    },
    DumpTreeAsEvents {
        response_tx: mpsc::SyncSender<Vec<RouterEvent>>,
    },
    Shutdown,
}

// NOTE: RadixTree is now thread-safe with pure sync patterns
#[pyclass]
pub(crate) struct RadixTree {
    request_tx: mpsc::Sender<RadixTreeRequest>,
}

#[pymethods]
impl RadixTree {
    #[new]
    fn new() -> PyResult<Self> {
        let (request_tx, request_rx) = mpsc::channel::<RadixTreeRequest>();

        // Spawn dedicated thread with simplified sync processing
        std::thread::spawn(move || {
            let mut radix_tree = dynamo_kv_router::indexer::RadixTree::new();

            loop {
                match request_rx.recv() {
                    Ok(RadixTreeRequest::Shutdown) => {
                        tracing::debug!("RadixTree thread received shutdown request");
                        break;
                    }
                    Ok(request) => {
                        Self::handle_request(&mut radix_tree, request);
                    }
                    Err(mpsc::RecvError) => {
                        tracing::debug!("RadixTree request channel disconnected");
                        break;
                    }
                }
            }
        });

        Ok(Self { request_tx })
    }

    #[pyo3(signature = (sequence, early_exit=false))]
    fn find_matches(
        &self,
        py: Python,
        sequence: Vec<u64>,
        early_exit: bool,
    ) -> PyResult<OverlapScores> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let local_block_hashes =
            py.allow_threads(|| sequence.into_iter().map(LocalBlockHash).collect());

        let request = RadixTreeRequest::FindMatches {
            local_block_hashes,
            early_exit,
            response_tx,
        };

        self.request_tx.send(request).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "RadixTree background task has shut down",
            )
        })?;

        // Release GIL while waiting for response
        let result = py.allow_threads(move || {
            response_rx.recv().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("RadixTree request was cancelled")
            })
        })?;

        Ok(OverlapScores { inner: result })
    }

    fn apply_event(
        &self,
        py: Python,
        worker_id: WorkerId,
        kv_cache_event_bytes: &[u8],
    ) -> PyResult<()> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let request = RadixTreeRequest::ApplyEvent {
            worker_id,
            kv_cache_event_bytes: kv_cache_event_bytes.to_vec(),
            response_tx,
        };

        self.request_tx.send(request).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "RadixTree background task has shut down",
            )
        })?;

        // Release GIL while waiting for response
        let result = py.allow_threads(move || response_rx.recv());

        result.map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("RadixTree request was cancelled")
        })?
    }

    fn remove_worker(&self, py: Python, worker_id: WorkerId) -> PyResult<()> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let request = RadixTreeRequest::RemoveWorker {
            worker_id,
            response_tx,
        };

        self.request_tx.send(request).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "RadixTree background task has shut down",
            )
        })?;

        // Release GIL while waiting for response
        py.allow_threads(move || {
            response_rx.recv().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("RadixTree request was cancelled")
            })
        })
    }

    fn clear_all_blocks(&self, py: Python, worker_id: WorkerId) -> PyResult<()> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let request = RadixTreeRequest::ClearAllBlocks {
            worker_id,
            response_tx,
        };

        self.request_tx.send(request).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "RadixTree background task has shut down",
            )
        })?;

        // Release GIL while waiting for response
        py.allow_threads(move || {
            response_rx.recv().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("RadixTree request was cancelled")
            })
        })
    }

    fn dump_tree_as_events(&self, py: Python) -> PyResult<Vec<String>> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let request = RadixTreeRequest::DumpTreeAsEvents { response_tx };

        self.request_tx.send(request).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send dump tree request")
        })?;

        // Release GIL while waiting for response from dedicated thread
        let events = py.allow_threads(move || {
            response_rx.recv().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Failed to receive dump tree response",
                )
            })
        })?;

        // Serialize RouterEvent structs to JSON strings with GIL released
        py.allow_threads(move || {
            events
                .into_iter()
                .map(|event| {
                    serde_json::to_string(&event).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Failed to serialize event to JSON: {}",
                            e
                        ))
                    })
                })
                .collect::<Result<Vec<String>, PyErr>>()
        })
    }
}

impl RadixTree {
    fn handle_request(
        radix_tree: &mut dynamo_kv_router::indexer::RadixTree,
        request: RadixTreeRequest,
    ) {
        match request {
            RadixTreeRequest::FindMatches {
                local_block_hashes,
                early_exit,
                response_tx,
            } => {
                let result = radix_tree.find_matches(local_block_hashes, early_exit);
                let _ = response_tx.send(result);
            }
            RadixTreeRequest::ApplyEvent {
                worker_id,
                kv_cache_event_bytes,
                response_tx,
            } => {
                let result = match serde_json::from_slice::<KvCacheEvent>(&kv_cache_event_bytes) {
                    Ok(kv_cache_event) => {
                        let router_event = RouterEvent::new(worker_id, kv_cache_event);
                        match radix_tree.apply_event(router_event) {
                            Ok(_) => Ok(()),
                            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                                format!("Failed to apply event: {}", e),
                            )),
                        }
                    }
                    Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Failed to deserialize KvCacheEvent: {}",
                        e
                    ))),
                };
                let _ = response_tx.send(result);
            }
            RadixTreeRequest::RemoveWorker {
                worker_id,
                response_tx,
            } => {
                radix_tree.remove_worker(worker_id);
                let _ = response_tx.send(());
            }
            RadixTreeRequest::ClearAllBlocks {
                worker_id,
                response_tx,
            } => {
                radix_tree.clear_all_blocks(worker_id);
                let _ = response_tx.send(());
            }
            RadixTreeRequest::DumpTreeAsEvents { response_tx } => {
                let events = radix_tree.dump_tree_as_events();
                let _ = response_tx.send(events);
            }
            RadixTreeRequest::Shutdown => {
                // This is handled in the main loop
            }
        }
    }
}

// Cleanup when RadixTree is dropped
impl Drop for RadixTree {
    fn drop(&mut self) {
        // Only need graceful shutdown via RadixTreeRequest::Shutdown
        let _ = self.request_tx.send(RadixTreeRequest::Shutdown);
    }
}

/// Helper function to create a KV router from an endpoint using the ModelManager
/// to ensure proper etcd registration.
/// Infers worker type using endpoint naming and router config:
/// - If endpoint name/component contains "prefill", treat as prefill
/// - If router_track_active_blocks is disabled, treat as prefill
/// - Otherwise, default to decode
async fn create_kv_router_from_endpoint(
    endpoint: &Endpoint,
    block_size: usize,
    kv_router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<Arc<dyn dynamo_kv_router::PrefillLoadEstimator>>,
) -> Result<Arc<llm_rs::kv_router::KvRouter>, PyErr> {
    // Create ModelManager and use it to create KvRouter (ensures registration)
    let model_manager = Arc::new(llm_rs::discovery::ModelManager::new());
    let endpoint_id = endpoint.inner.id();
    let namespace = endpoint_id.namespace.to_lowercase();
    let component = endpoint_id.component.to_lowercase();
    let name = endpoint_id.name.to_lowercase();
    let endpoint_is_prefill =
        namespace.contains("prefill") || component.contains("prefill") || name.contains("prefill");
    let track_active_blocks = kv_router_config
        .as_ref()
        .map(|cfg| cfg.router_track_active_blocks)
        .unwrap_or(true);
    let worker_type = if endpoint_is_prefill || !track_active_blocks {
        llm_rs::discovery::WORKER_TYPE_PREFILL
    } else {
        llm_rs::discovery::WORKER_TYPE_DECODE
    };

    // Look up the worker's model card so we can derive both model_name (required
    // for remote/served indexer) and Eagle routing semantics. When the model_name
    // is required but no worker has registered yet, wait via the discovery watch
    // stream until one appears so we don't race worker startup. Bounded by
    // `DYN_ROUTER_MODEL_CARD_WAIT_SECS` (default 600s).
    let needs_model_name = kv_router_config
        .as_ref()
        .map(|cfg| cfg.use_remote_indexer || cfg.serve_indexer)
        .unwrap_or(false);
    let (model_name, enable_eagle) = {
        let maybe_card = if needs_model_name {
            let wait_secs: u64 = std::env::var("DYN_ROUTER_MODEL_CARD_WAIT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600);
            tracing::info!(
                namespace = %endpoint_id.namespace,
                component = %endpoint_id.component,
                endpoint = %endpoint_id.name,
                wait_secs,
                "Waiting for worker model card in discovery (required for remote/served indexer)"
            );
            llm_rs::discovery::wait_for_endpoint_model_card(
                &endpoint.inner,
                std::time::Duration::from_secs(wait_secs),
                None,
            )
            .await
            .map_err(to_pyerr)?
        } else {
            // Non-blocking snapshot — used only to detect Eagle routing semantics
            // when a card happens to already be registered.
            let discovery = endpoint.inner.component().drt().discovery();
            let instances = discovery
                .list(rs::discovery::DiscoveryQuery::EndpointModels {
                    namespace: endpoint_id.namespace.clone(),
                    component: endpoint_id.component.clone(),
                    endpoint: endpoint_id.name.clone(),
                })
                .await
                .map_err(to_pyerr)?;
            instances.into_iter().find_map(|inst| {
                inst.deserialize_model::<llm_rs::model_card::ModelDeploymentCard>()
                    .ok()
            })
        };

        match maybe_card {
            Some(card) => {
                let model_name = needs_model_name.then(|| card.display_name.clone());
                (model_name, card.runtime_config.enable_eagle)
            }
            None => {
                tracing::warn!(
                    namespace = %endpoint_id.namespace,
                    component = %endpoint_id.component,
                    endpoint = %endpoint_id.name,
                    "No model card found in discovery; defaulting to non-Eagle routing semantics"
                );
                (None, false)
            }
        }
    };

    let kv_router = model_manager
        .kv_chooser_for(
            &endpoint.inner,
            block_size as u32,
            kv_router_config,
            prefill_load_estimator,
            worker_type,
            model_name,
            enable_eagle,
        )
        .await
        .map_err(to_pyerr)?;

    Ok(kv_router)
}

#[pyclass]
pub(crate) struct KvRouter {
    inner: Arc<RsKvPushRouter>,
}

/// Attach worker_id info from the tracker to `routing_data` so it survives the
/// Rust->Python->Rust router path to the frontend (data survives, annotations don't).
fn inject_worker_id_from_tracker(
    data: &mut llm_rs::protocols::common::llm_backend::LLMEngineOutput,
    tracker: &RequestTracker,
) {
    let Some(worker_info) = tracker.get_worker_info() else {
        return;
    };

    data.routing_data
        .get_or_insert_with(Default::default)
        .worker_id = Some(worker_info);
}

/// Attach the request's timing to `routing_data` so it survives the
/// Rust->Python->Rust router path to the frontend (data survives, annotations don't).
fn inject_timing_from_tracker(
    data: &mut llm_rs::protocols::common::llm_backend::LLMEngineOutput,
    tracker: &RequestTracker,
) {
    data.routing_data
        .get_or_insert_with(Default::default)
        .timing = Some(tracker.get_timing_info());
}

// TODO: can this reuse the stream conversion method in Client bindings?
impl KvRouter {
    /// Helper method to process a request and create a Python async generator
    fn process_request_to_stream<'p>(
        py: Python<'p>,
        inner: Arc<RsKvPushRouter>,
        request: llm_rs::protocols::common::preprocessor::PreprocessedRequest,
        tracker: Option<Arc<RequestTracker>>,
        response_buffer_size: usize,
    ) -> PyResult<Bound<'p, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let single_in = SingleIn::new(request);
            let stream = inner.generate(single_in).await.map_err(to_pyerr)?;
            let (tx, rx) =
                tokio::sync::mpsc::channel::<RsAnnotated<PyObject>>(response_buffer_size);

            tokio::spawn(async move {
                let mut stream = stream;
                let mut first_item = true;
                let mut first_token_gauges_observed = false;

                while let Some(mut response) = stream.next().await {
                    if first_item {
                        first_item = false;
                        if let (Some(tracker), Some(data)) = (&tracker, &mut response.data) {
                            inject_worker_id_from_tracker(data, tracker);
                        }
                    }

                    if !first_token_gauges_observed {
                        let has_tokens = response
                            .data
                            .as_ref()
                            .map(|d| !d.token_ids.is_empty())
                            .unwrap_or(false);
                        if has_tokens {
                            if let Some(ref tracker) = tracker {
                                tracker.observe_first_token_gauges();
                            }
                            first_token_gauges_observed = true;
                        }
                    }

                    // On the terminal chunk, finalize timing and attach it for the frontend.
                    if let (Some(tracker), Some(data)) = (&tracker, &mut response.data)
                        && data.finish_reason.is_some()
                    {
                        tracker.record_finish();
                        inject_timing_from_tracker(data, tracker);
                    }

                    let py_response = Python::with_gil(|py| {
                        pythonize(py, &response.data)
                            .map(|obj| obj.unbind())
                            .map_err(|e| e.to_string())
                    });

                    match py_response {
                        Ok(obj) => {
                            if tx.send(RsAnnotated::from_data(obj)).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to pythonize response: {}", e);
                            break;
                        }
                    }
                }

                if let Some(ref tracker) = tracker {
                    tracker.observe_finish_gauges();
                }
            });

            Ok(crate::AsyncResponseStream::new(rx, false))
        })
    }

    fn dispatch_request_to_stream<'p>(
        py: Python<'p>,
        inner: Arc<RsKvPushRouter>,
        request: llm_rs::protocols::common::preprocessor::PreprocessedRequest,
        tracker: Option<Arc<RequestTracker>>,
        response_buffer_mode: ResponseBufferMode,
    ) -> PyResult<Bound<'p, PyAny>> {
        match response_buffer_mode {
            ResponseBufferMode::Rendezvous => {
                demand_driven::process_request_to_stream(py, inner, request, tracker)
            }
            ResponseBufferMode::Buffered(capacity) => {
                Self::process_request_to_stream(py, inner, request, tracker, capacity)
            }
        }
    }
}

#[pymethods]
impl KvRouter {
    /// Create a new KvRouter for KV-aware routing to workers.
    ///
    /// # Arguments
    /// * `endpoint` - The endpoint to route requests to
    /// * `block_size` - KV cache block size for routing decisions
    /// * `kv_router_config` - Configuration for the KV router
    ///
    /// Note: Worker type for Prometheus metrics is inferred from the endpoint name/component
    /// (contains "prefill") or by `router_track_active_blocks` being disabled.
    #[new]
    #[pyo3(signature = (endpoint, block_size, kv_router_config, aic_perf_config=None, session_affinity_ttl_secs=None))]
    fn new(
        endpoint: &Endpoint,
        block_size: usize,
        kv_router_config: &super::entrypoint::KvRouterConfig,
        aic_perf_config: Option<&AicPerfConfig>,
        session_affinity_ttl_secs: Option<u64>,
    ) -> PyResult<Self> {
        if session_affinity_ttl_secs.is_some_and(|ttl| !(1..=31_536_000).contains(&ttl)) {
            return Err(PyValueError::new_err(
                "session_affinity_ttl_secs must be between 1 and 31536000",
            ));
        }
        let prefill_load_estimator = aic_perf_config
            .map(|config| {
                Python::with_gil(|py| {
                    create_aic_prefill_load_estimator(
                        py,
                        config.backend_name(),
                        config.system(),
                        config.model_path(),
                        config.tp_size(),
                        config.backend_version(),
                        config.moe_tp_size(),
                        config.moe_ep_size(),
                        config.attention_dp_size(),
                        config.gemm_dtype(),
                        config.moe_dtype(),
                        config.fmha_dtype(),
                        config.kv_cache_dtype(),
                        config.comm_dtype(),
                        config.nextn(),
                        config.nextn_accept_rates(),
                    )
                })
            })
            .transpose()
            .map_err(to_pyerr)?;

        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        runtime.block_on(async move {
            let client = endpoint.inner.client().await.map_err(to_pyerr)?;

            // Create PushRouter with KV router mode
            let push_router = rs::pipeline::PushRouter::<
                llm_rs::protocols::common::preprocessor::PreprocessedRequest,
                rs::protocols::annotated::Annotated<
                    llm_rs::protocols::common::llm_backend::LLMEngineOutput,
                >,
            >::from_client(
                client,
                rs::pipeline::network::egress::push_router::RouterMode::KV,
            )
            .await
            .map_err(to_pyerr)?;

            // Create KvRouter using helper function (ensures etcd registration)
            let kv_router = create_kv_router_from_endpoint(
                endpoint,
                block_size,
                Some(kv_router_config.inner()),
                prefill_load_estimator,
            )
            .await?;

            let kv_push_router = RsKvPushRouter::new(
                push_router,
                kv_router,
                session_affinity_ttl_secs.map(Duration::from_secs),
            )
            .map_err(to_pyerr)?;

            Ok(Self {
                inner: Arc::new(kv_push_router),
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (token_ids, model, stop_conditions=None, sampling_options=None, output_options=None, router_config_override=None, worker_id=None, dp_rank=None, extra_args=None, block_mm_infos=None, multi_modal_data=None, mm_routing_info=None, routing_constraints=None, response_buffer_size=100))]
    fn generate<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
        model: String,
        stop_conditions: Option<PyObject>,
        sampling_options: Option<PyObject>,
        output_options: Option<PyObject>,
        router_config_override: Option<PyObject>,
        worker_id: Option<WorkerId>,
        dp_rank: Option<DpRank>,
        extra_args: Option<PyObject>,
        block_mm_infos: Option<PyObject>,
        multi_modal_data: Option<PyObject>,
        mm_routing_info: Option<PyObject>,
        routing_constraints: Option<RoutingConstraints>,
        response_buffer_size: isize,
    ) -> PyResult<Bound<'p, PyAny>> {
        let response_buffer_size = validate_response_buffer_size(response_buffer_size)?;
        // Depythonize the options with defaults
        let stop_conditions: StopConditions = if let Some(obj) = stop_conditions {
            depythonize(obj.bind(py)).map_err(to_pyerr)?
        } else {
            StopConditions::default()
        };

        let sampling_options: SamplingOptions = if let Some(obj) = sampling_options {
            depythonize(obj.bind(py)).map_err(to_pyerr)?
        } else {
            SamplingOptions::default()
        };

        let output_options: OutputOptions = if let Some(obj) = output_options {
            depythonize(obj.bind(py)).map_err(to_pyerr)?
        } else {
            OutputOptions::default()
        };

        let router_config_override: Option<RouterConfigOverride> =
            if let Some(obj) = router_config_override {
                Some(depythonize(obj.bind(py)).map_err(to_pyerr)?)
            } else {
                None
            };

        let extra_args: Option<serde_json::Value> = if let Some(obj) = extra_args {
            Some(depythonize(obj.bind(py)).map_err(to_pyerr)?)
        } else {
            None
        };

        let block_mm_infos = block_mm_infos
            .map(|obj| depythonize_block_mm_infos(obj.bind(py)))
            .transpose()?;

        let multi_modal_data: Option<llm_rs::protocols::common::preprocessor::MultimodalDataMap> =
            if let Some(obj) = multi_modal_data {
                Some(depythonize(obj.bind(py)).map_err(to_pyerr)?)
            } else {
                None
            };

        let mm_routing_info: Option<llm_rs::protocols::common::preprocessor::MmRoutingInfo> =
            if let Some(obj) = mm_routing_info {
                Some(depythonize(obj.bind(py)).map_err(to_pyerr)?)
            } else {
                block_mm_infos.map(
                    |infos| llm_rs::protocols::common::preprocessor::MmRoutingInfo {
                        routing_token_ids: token_ids.clone(),
                        block_mm_infos: infos,
                        expanded_prompt_len: token_ids.len(),
                    },
                )
            };

        // Create tracker to capture worker routing info from KvRouter
        let tracker = Arc::new(RequestTracker::new());

        // Build the PreprocessedRequest
        let mut request_builder =
            llm_rs::protocols::common::preprocessor::PreprocessedRequest::builder();
        request_builder
            .model(model)
            .token_ids(token_ids)
            .stop_conditions(stop_conditions)
            .sampling_options(sampling_options)
            .output_options(output_options)
            .router_config_override(router_config_override)
            .multi_modal_data(multi_modal_data)
            .mm_routing_info(mm_routing_info)
            .extra_args(extra_args)
            .tracker(Some(tracker.clone()));

        // Set routing hints if worker_id or dp_rank is provided
        if worker_id.is_some() || dp_rank.is_some() || routing_constraints.is_some() {
            let routing = llm_rs::protocols::common::preprocessor::RoutingHints {
                backend_instance_id: worker_id,
                dp_rank,
                routing_constraints: routing_constraints.map(Into::into),
                ..Default::default()
            };
            request_builder.routing(Some(routing));
        }

        let request = request_builder.build().map_err(to_pyerr)?;

        // Use the helper method to process the request
        Self::dispatch_request_to_stream(
            py,
            self.inner.clone(),
            request,
            Some(tracker),
            response_buffer_size,
        )
    }

    #[pyo3(signature = (request, response_buffer_size=100))]
    fn generate_from_request<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
        response_buffer_size: isize,
    ) -> PyResult<Bound<'p, PyAny>> {
        let response_buffer_size = validate_response_buffer_size(response_buffer_size)?;
        // Depythonize the request directly into PreprocessedRequest
        let mut request: llm_rs::protocols::common::preprocessor::PreprocessedRequest =
            depythonize(request.bind(py)).map_err(to_pyerr)?;

        // Create tracker if not already set, to capture worker routing info
        let tracker = match request.tracker {
            Some(ref t) => t.clone(),
            None => {
                let t = Arc::new(RequestTracker::new());
                request.tracker = Some(t.clone());
                t
            }
        };

        // Use the helper method to process the request
        Self::dispatch_request_to_stream(
            py,
            self.inner.clone(),
            request,
            Some(tracker),
            response_buffer_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (token_ids, router_config_override=None, request_id=None, update_indexer=false, block_mm_infos=None, lora_name=None, routing_constraints=None, strict_priority=0, policy_class=None))]
    fn best_worker<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
        router_config_override: Option<PyObject>,
        request_id: Option<String>,
        update_indexer: bool,
        block_mm_infos: Option<PyObject>,
        lora_name: Option<String>,
        routing_constraints: Option<RoutingConstraints>,
        strict_priority: u32,
        policy_class: Option<String>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let router_config_override = if let Some(obj) = router_config_override {
            let override_config: RouterConfigOverride =
                depythonize(obj.bind(py)).map_err(to_pyerr)?;
            Some(override_config)
        } else {
            None
        };

        let block_mm_infos = block_mm_infos
            .map(|obj| depythonize_block_mm_infos(obj.bind(py)))
            .transpose()?;

        let chooser = self.inner.chooser.clone();
        let update_states = request_id.is_some();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let outcome = chooser
                .find_best_match_details_with_policy_class(
                    request_id.as_deref(),
                    &token_ids,
                    block_mm_infos.as_deref(),
                    router_config_override.as_ref(),
                    update_states,
                    false,
                    lora_name.clone(),
                    0.0,
                    strict_priority,
                    policy_class,
                    None,
                    None,
                    None, // allowed_worker_ids: pass via RoutingHints in PreprocessedRequest path
                    routing_constraints.map(Into::into).unwrap_or_default(),
                )
                .await
                .map_err(to_pyerr)?;
            let (best_worker, overlap_blocks) = match outcome {
                llm_rs::kv_router::FindBestMatchOutcome::Routed {
                    worker,
                    overlap_blocks,
                    ..
                } => (worker, overlap_blocks),
                llm_rs::kv_router::FindBestMatchOutcome::QueueRejected { rejection } => {
                    return Err(crate::errors::queue_rejection_to_pyerr(rejection));
                }
            };

            if update_indexer {
                let cfg = chooser.kv_router_config();
                if !cfg.use_kv_events || cfg.predict_on_route_enabled() {
                    let mut tokens_with_hashes =
                        TokensWithHashes::new(token_ids.clone(), chooser.block_size())
                            .with_is_eagle(chooser.is_eagle());
                    if let Some(infos) = block_mm_infos.as_ref() {
                        tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.clone());
                    }
                    if let Some(lora_name) = lora_name.as_ref() {
                        tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name.clone());
                    }
                    chooser
                        .record_routing_decision(tokens_with_hashes, best_worker)
                        .await
                        .map_err(to_pyerr)?;
                }
            }

            Ok((best_worker.worker_id, best_worker.dp_rank, overlap_blocks))
        })
    }

    /// Mark prefill as completed for a request
    fn mark_prefill_complete<'p>(
        &self,
        py: Python<'p>,
        request_id: String,
    ) -> PyResult<Bound<'p, PyAny>> {
        let chooser = self.inner.chooser.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            chooser
                .mark_prefill_completed(&request_id)
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Free a request by its ID, signaling the router to release resources
    fn free<'p>(&self, py: Python<'p>, request_id: String) -> PyResult<Bound<'p, PyAny>> {
        let chooser = self.inner.chooser.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            chooser.free(&request_id).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    #[pyo3(signature = (token_ids, block_mm_infos=None, lora_name=None))]
    fn get_potential_loads<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
        block_mm_infos: Option<PyObject>,
        lora_name: Option<String>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let block_mm_infos = block_mm_infos
            .map(|obj| depythonize_block_mm_infos(obj.bind(py)))
            .transpose()?;
        let chooser = self.inner.chooser.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let loads = chooser
                .get_potential_loads(
                    &token_ids,
                    None,
                    block_mm_infos.as_deref(),
                    lora_name.as_deref(),
                )
                .await
                .map_err(to_pyerr)?;

            // Return loads without aggregation - each (worker_id, dp_rank) pair is a separate entry
            // Use pythonize to convert Vec<PotentialLoad> to Python list of dicts
            Python::with_gil(|py| {
                pythonize(py, &loads)
                    .map(|obj| obj.unbind())
                    .map_err(to_pyerr)
            })
        })
    }

    #[pyo3(signature = (token_ids, router_config_override=None, block_mm_infos=None, lora_name=None, include_shared=true))]
    fn get_overlap_scores<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
        router_config_override: Option<PyObject>,
        block_mm_infos: Option<PyObject>,
        lora_name: Option<String>,
        include_shared: bool,
    ) -> PyResult<Bound<'p, PyAny>> {
        let router_config_override = if let Some(obj) = router_config_override {
            let override_config: RouterConfigOverride =
                depythonize(obj.bind(py)).map_err(to_pyerr)?;
            Some(override_config)
        } else {
            None
        };
        let block_mm_infos = block_mm_infos
            .map(|obj| depythonize_block_mm_infos(obj.bind(py)))
            .transpose()?;
        let chooser = self.inner.chooser.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let scores = chooser
                .get_overlap_scores(
                    &token_ids,
                    router_config_override.as_ref(),
                    block_mm_infos.as_deref(),
                    lora_name.as_deref(),
                    include_shared,
                )
                .await
                .map_err(to_pyerr)?;

            Python::with_gil(|py| {
                pythonize(py, &scores)
                    .map(|obj| obj.unbind())
                    .map_err(to_pyerr)
            })
        })
    }

    /// Dump all events from the KV router's indexer as a JSON string
    fn dump_events<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let chooser = self.inner.chooser.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let events = chooser.dump_events().await.map_err(to_pyerr)?;
            // Serialize to JSON string
            let json_str = serde_json::to_string(&events).map_err(to_pyerr)?;
            Ok(json_str)
        })
    }
}
