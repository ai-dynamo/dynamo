// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Display;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use pyo3::{exceptions::PyException, prelude::*};
use pyo3_async_runtimes::TaskLocals;
use pythonize::pythonize;
use uuid::Uuid;

use dynamo_kv_router::config::KvRouterConfig as RsKvRouterConfig;
use dynamo_llm::discovery::LoadThresholdConfig as RsLoadThresholdConfig;
use dynamo_llm::entrypoint::ChatEngineFactoryCallback;
use dynamo_llm::entrypoint::EngineConfig as RsEngineConfig;
use dynamo_llm::entrypoint::RouterConfig as RsRouterConfig;
use dynamo_llm::entrypoint::input::Input;
use dynamo_llm::local_model::DEFAULT_HTTP_PORT;
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder};
use dynamo_llm::mocker::make_mocker_engine;
use dynamo_llm::model_card::ModelDeploymentCard as RsModelDeploymentCard;
use dynamo_llm::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;
use dynamo_mocker::common::perf_model::PerfModel;

use super::aic_callback::create_aic_callback;
use dynamo_mocker::common::protocols::{DirectRequest, MockEngineArgs};
use dynamo_runtime::discovery::ModelCardInstanceId as RsModelCardInstanceId;
use dynamo_runtime::protocols::EndpointId;

use super::local_model::ModelRuntimeConfig;
use super::model_card::ModelDeploymentCard;
use crate::RouterMode;
use crate::engine::PythonAsyncEngine;

#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
#[repr(i32)]
pub enum EngineType {
    Echo = 1,
    Dynamic = 2,
    Mocker = 3,
}

#[pyclass]
#[derive(Default, Clone, Debug)]
pub struct KvRouterConfig {
    inner: RsKvRouterConfig,
}

impl KvRouterConfig {
    pub fn inner(&self) -> RsKvRouterConfig {
        self.inner.clone()
    }
}

#[pymethods]
impl KvRouterConfig {
    #[new]
    #[pyo3(signature = (overlap_score_weight=1.0, router_temperature=0.0, use_kv_events=true, durable_kv_events=false, router_replica_sync=false, router_track_active_blocks=true, router_track_output_blocks=false, router_assume_kv_reuse=true, router_snapshot_threshold=1000000, router_reset_states=false, router_ttl_secs=120.0, router_max_tree_size=1048576, router_prune_target_ratio=0.8, router_queue_threshold=Some(4.0), router_event_threads=4, router_enable_cache_control=false, min_initial_workers=1, router_queue_policy="fcfs", remote_indexer_component=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        overlap_score_weight: f64,
        router_temperature: f64,
        use_kv_events: bool,
        durable_kv_events: bool,
        router_replica_sync: bool,
        router_track_active_blocks: bool,
        router_track_output_blocks: bool,
        router_assume_kv_reuse: bool,
        router_snapshot_threshold: Option<u32>,
        router_reset_states: bool,
        router_ttl_secs: f64,
        router_max_tree_size: usize,
        router_prune_target_ratio: f64,
        router_queue_threshold: Option<f64>,
        router_event_threads: u32,
        router_enable_cache_control: bool,
        min_initial_workers: usize,
        router_queue_policy: &str,
        remote_indexer_component: Option<String>,
    ) -> Self {
        KvRouterConfig {
            inner: RsKvRouterConfig {
                overlap_score_weight,
                router_temperature,
                use_kv_events,
                durable_kv_events,
                router_replica_sync,
                router_track_active_blocks,
                router_track_output_blocks,
                router_assume_kv_reuse,
                router_snapshot_threshold,
                router_reset_states,
                router_ttl_secs,
                router_max_tree_size,
                router_prune_target_ratio,
                router_queue_threshold,
                router_event_threads,
                router_enable_cache_control,
                skip_initial_worker_wait: false,
                min_initial_workers,
                router_queue_policy: router_queue_policy.parse().unwrap_or_else(|_| {
                    panic!("invalid router_queue_policy: {router_queue_policy:?}")
                }),
                remote_indexer_component,
            },
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct RouterConfig {
    #[pyo3(get, set)]
    pub router_mode: RouterMode,

    #[pyo3(get, set)]
    pub kv_router_config: KvRouterConfig,

    /// Threshold for active decode blocks utilization (0.0-1.0)
    active_decode_blocks_threshold: Option<f64>,
    /// Threshold for active prefill tokens utilization (literal token count)
    active_prefill_tokens_threshold: Option<u64>,
    /// Threshold for active prefill tokens as fraction of max_num_batched_tokens
    active_prefill_tokens_threshold_frac: Option<f64>,
    enforce_disagg: bool,
}

#[pymethods]
impl RouterConfig {
    #[new]
    #[pyo3(signature = (mode, config=None, active_decode_blocks_threshold=None, active_prefill_tokens_threshold=None, active_prefill_tokens_threshold_frac=None, enforce_disagg=false))]
    pub fn new(
        mode: RouterMode,
        config: Option<KvRouterConfig>,
        active_decode_blocks_threshold: Option<f64>,
        active_prefill_tokens_threshold: Option<u64>,
        active_prefill_tokens_threshold_frac: Option<f64>,
        enforce_disagg: bool,
    ) -> Self {
        Self {
            router_mode: mode,
            kv_router_config: config.unwrap_or_default(),
            active_decode_blocks_threshold,
            active_prefill_tokens_threshold,
            active_prefill_tokens_threshold_frac,
            enforce_disagg,
        }
    }
}

impl From<RouterConfig> for RsRouterConfig {
    fn from(rc: RouterConfig) -> RsRouterConfig {
        RsRouterConfig {
            router_mode: rc.router_mode.into(),
            kv_router_config: rc.kv_router_config.inner,
            load_threshold_config: RsLoadThresholdConfig {
                active_decode_blocks_threshold: rc.active_decode_blocks_threshold,
                active_prefill_tokens_threshold: rc.active_prefill_tokens_threshold,
                active_prefill_tokens_threshold_frac: rc.active_prefill_tokens_threshold_frac,
            },
            enforce_disagg: rc.enforce_disagg,
        }
    }
}

/// Wrapper to hold Python callback and its TaskLocals for async execution
#[derive(Clone)]
struct PyEngineFactory {
    callback: Arc<PyObject>,
    locals: Arc<TaskLocals>,
}

impl std::fmt::Debug for PyEngineFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyEngineFactory")
            .field("callback", &"<PyObject>")
            .finish()
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct EntrypointArgs {
    engine_type: EngineType,
    model_path: Option<PathBuf>,
    model_name: Option<String>,
    endpoint_id: Option<EndpointId>,
    context_length: Option<u32>,
    template_file: Option<PathBuf>,
    router_config: Option<RouterConfig>,
    kv_cache_block_size: Option<u32>,
    http_host: Option<String>,
    http_port: u16,
    http_metrics_port: Option<u16>,
    tls_cert_path: Option<PathBuf>,
    tls_key_path: Option<PathBuf>,
    extra_engine_args: Option<PathBuf>,
    runtime_config: Option<ModelRuntimeConfig>,
    namespace: Option<String>,
    namespace_prefix: Option<String>,
    is_prefill: bool,
    migration_limit: u32,
    chat_engine_factory: Option<PyEngineFactory>,
}

#[pymethods]
impl EntrypointArgs {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (engine_type, model_path=None, model_name=None, endpoint_id=None, context_length=None, template_file=None, router_config=None, kv_cache_block_size=None, http_host=None, http_port=None, http_metrics_port=None, tls_cert_path=None, tls_key_path=None, extra_engine_args=None, runtime_config=None, namespace=None, namespace_prefix=None, is_prefill=false, migration_limit=0, chat_engine_factory=None))]
    pub fn new(
        py: Python<'_>,
        engine_type: EngineType,
        model_path: Option<PathBuf>,
        model_name: Option<String>, // e.g. "dyn://namespace.component.endpoint"
        endpoint_id: Option<String>,
        context_length: Option<u32>,
        template_file: Option<PathBuf>,
        router_config: Option<RouterConfig>,
        kv_cache_block_size: Option<u32>,
        http_host: Option<String>,
        http_port: Option<u16>,
        http_metrics_port: Option<u16>,
        tls_cert_path: Option<PathBuf>,
        tls_key_path: Option<PathBuf>,
        extra_engine_args: Option<PathBuf>,
        runtime_config: Option<ModelRuntimeConfig>,
        namespace: Option<String>,
        namespace_prefix: Option<String>,
        is_prefill: bool,
        migration_limit: u32,
        chat_engine_factory: Option<PyObject>,
    ) -> PyResult<Self> {
        let endpoint_id_obj: Option<EndpointId> = endpoint_id.as_deref().map(EndpointId::from);
        if (tls_cert_path.is_some() && tls_key_path.is_none())
            || (tls_cert_path.is_none() && tls_key_path.is_some())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tls_cert_path and tls_key_path must be provided together",
            ));
        }

        // Capture TaskLocals at registration time for the chat engine factory callback
        let chat_engine_factory = chat_engine_factory
            .map(|callback| {
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to get TaskLocals for chat_engine_factory: {}",
                        e
                    ))
                })?;
                Ok::<_, PyErr>(PyEngineFactory {
                    callback: Arc::new(callback),
                    locals: Arc::new(locals),
                })
            })
            .transpose()?;

        Ok(EntrypointArgs {
            engine_type,
            model_path,
            model_name,
            endpoint_id: endpoint_id_obj,
            context_length,
            template_file,
            router_config,
            kv_cache_block_size,
            http_host,
            http_port: http_port.unwrap_or(DEFAULT_HTTP_PORT),
            http_metrics_port,
            tls_cert_path,
            tls_key_path,
            extra_engine_args,
            runtime_config,
            namespace,
            namespace_prefix,
            is_prefill,
            migration_limit,
            chat_engine_factory,
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct EngineConfig {
    inner: RsEngineConfig,
}

/// Create the backend engine wrapper to run the model.
/// Download the model if necessary.
#[pyfunction]
#[pyo3(signature = (distributed_runtime, args))]
pub fn make_engine<'p>(
    py: Python<'p>,
    distributed_runtime: super::DistributedRuntime,
    args: EntrypointArgs,
) -> PyResult<Bound<'p, PyAny>> {
    let mut builder = LocalModelBuilder::default();
    builder
        .model_name(
            args.model_name
                .clone()
                .or_else(|| args.model_path.clone().map(|p| p.display().to_string())),
        )
        .endpoint_id(args.endpoint_id.clone())
        .context_length(args.context_length)
        .request_template(args.template_file.clone())
        .kv_cache_block_size(args.kv_cache_block_size)
        .router_config(args.router_config.clone().map(|rc| rc.into()))
        .migration_limit(Some(args.migration_limit))
        .http_host(args.http_host.clone())
        .http_port(args.http_port)
        .http_metrics_port(args.http_metrics_port)
        .tls_cert_path(args.tls_cert_path.clone())
        .tls_key_path(args.tls_key_path.clone())
        .is_mocker(matches!(args.engine_type, EngineType::Mocker))
        .extra_engine_args(args.extra_engine_args.clone())
        .runtime_config(args.runtime_config.clone().unwrap_or_default().inner)
        .namespace(args.namespace.clone())
        .namespace_prefix(args.namespace_prefix.clone());
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        if let Some(model_path) = args.model_path.clone() {
            let local_path = if model_path.exists() {
                model_path
            } else {
                // Mocker only needs tokenizer, not weights
                let ignore_weights = matches!(args.engine_type, EngineType::Mocker);
                LocalModel::fetch(&model_path.display().to_string(), ignore_weights)
                    .await
                    .map_err(to_pyerr)?
            };
            builder.model_path(local_path);
        }

        let local_model = builder.build().await.map_err(to_pyerr)?;
        let inner = select_engine(distributed_runtime, args, local_model)
            .await
            .map_err(to_pyerr)?;
        Ok(EngineConfig { inner })
    })
}

/// Convert a PyEngineFactory to a Rust ChatEngineFactoryCallback
fn py_engine_factory_to_callback(factory: PyEngineFactory) -> ChatEngineFactoryCallback {
    let callback = factory.callback;
    let locals = factory.locals;

    Arc::new(
        move |instance_id: RsModelCardInstanceId,
              card: RsModelDeploymentCard|
              -> Pin<
            Box<dyn Future<Output = anyhow::Result<OpenAIChatCompletionsStreamingEngine>> + Send>,
        > {
            let callback = callback.clone();
            let locals = locals.clone();

            Box::pin(async move {
                // Acquire GIL to call Python callback and convert coroutine to future
                let py_future = Python::with_gil(|py| {
                    let py_instance_id =
                        Py::new(py, crate::ModelCardInstanceId { inner: instance_id }).map_err(
                            |e| anyhow::anyhow!("Failed to create Python ModelCardInstanceId: {e}"),
                        )?;
                    // Create Python ModelDeploymentCard wrapper
                    let py_card = ModelDeploymentCard { inner: card };
                    let py_card_obj = Py::new(py, py_card)
                        .map_err(|e| anyhow::anyhow!("Failed to create Python MDC: {e}"))?;

                    // Call Python async function to get a coroutine
                    let coroutine = callback
                        .call1(py, (py_instance_id, py_card_obj))
                        .map_err(|e| anyhow::anyhow!("Failed to call chat_engine_factory: {e}"))?;

                    // Use the TaskLocals captured at registration time
                    pyo3_async_runtimes::into_future_with_locals(&locals, coroutine.into_bound(py))
                        .map_err(|e| anyhow::anyhow!("Failed to convert coroutine to future: {e}"))
                })?;

                // Await the Python coroutine (GIL is released during await)
                let py_result = py_future
                    .await
                    .map_err(|e| anyhow::anyhow!("chat_engine_factory callback failed: {}", e))?;

                // Extract PythonAsyncEngine from the Python result and wrap in Arc
                let engine: OpenAIChatCompletionsStreamingEngine = Python::with_gil(|py| {
                    let engine: PythonAsyncEngine = py_result.extract(py).map_err(|e| {
                        anyhow::anyhow!("Failed to extract PythonAsyncEngine: {}", e)
                    })?;
                    Ok::<_, anyhow::Error>(Arc::new(engine))
                })?;

                Ok(engine)
            })
        },
    )
}

async fn select_engine(
    #[allow(unused_variables)] distributed_runtime: super::DistributedRuntime,
    args: EntrypointArgs,
    local_model: LocalModel,
) -> anyhow::Result<RsEngineConfig> {
    let inner = match args.engine_type {
        EngineType::Echo => {
            // There is no validation for the echo engine
            RsEngineConfig::InProcessText {
                model: Box::new(local_model),
                engine: dynamo_llm::engines::make_echo_engine(),
            }
        }
        EngineType::Dynamic => {
            //  Convert Python chat engine factory to Rust callback
            let chat_engine_factory = args.chat_engine_factory.map(py_engine_factory_to_callback);
            RsEngineConfig::Dynamic {
                model: Box::new(local_model),
                chat_engine_factory,
            }
        }
        EngineType::Mocker => {
            let mut mocker_args = if let Some(extra_args_path) = args.extra_engine_args {
                MockEngineArgs::from_json_file(&extra_args_path).map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to load mocker args from {:?}: {}",
                        extra_args_path,
                        e
                    )
                })?
            } else {
                tracing::warn!(
                    "No extra_engine_args specified for mocker engine. Using default mocker args."
                );
                MockEngineArgs::default()
            };

            // If aic_backend is set, create Python AIC callback and override perf_model
            if let Some(ref backend_name) = mocker_args.aic_backend {
                let backend = backend_name.clone();
                let system = mocker_args.aic_system.as_deref().unwrap_or("h200_sxm");
                let model_name = mocker_args
                    .aic_model_path
                    .as_deref()
                    .unwrap_or_else(|| local_model.card().source_path());
                let backend_version = mocker_args.aic_backend_version.as_deref();
                let tp_size = mocker_args.aic_tp_size.unwrap_or(1);
                match Python::with_gil(|py| {
                    create_aic_callback(py, &backend, system, model_name, tp_size, backend_version)
                }) {
                    Ok(callback) => {
                        tracing::info!(
                            "AIC perf model: backend={}, gpu={}, model={}, version={:?}",
                            backend,
                            system,
                            model_name,
                            backend_version
                        );
                        mocker_args.perf_model = Arc::new(PerfModel::from_aic_callback(callback));
                    }
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Failed to create AIC callback (--aic-perf-model was requested): {}",
                            e
                        ));
                    }
                }
            }

            let endpoint = local_model.endpoint_id().clone();

            let engine =
                make_mocker_engine(distributed_runtime.inner, endpoint, mocker_args).await?;

            RsEngineConfig::InProcessTokens {
                engine,
                model: Box::new(local_model),
                is_prefill: args.is_prefill,
            }
        }
    };

    Ok(inner)
}

#[pyfunction]
#[pyo3(signature = (distributed_runtime, input, engine_config))]
pub fn run_input<'p>(
    py: Python<'p>,
    distributed_runtime: super::DistributedRuntime,
    input: &str,
    engine_config: EngineConfig,
) -> PyResult<Bound<'p, PyAny>> {
    let input_enum: Input = input.parse().map_err(to_pyerr)?;
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        dynamo_llm::entrypoint::input::run_input(
            distributed_runtime.inner.clone(),
            input_enum,
            engine_config.inner,
        )
        .await
        .map_err(to_pyerr)?;
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (trace_file, extra_engine_args=None, extra_engine_args_json=None, router_config=None, router_config_json=None, num_workers=1, replay_concurrency=None, replay_mode="offline", router_mode="round_robin", arrival_speedup_ratio=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn run_mocker_trace_replay(
    py: Python<'_>,
    trace_file: PathBuf,
    extra_engine_args: Option<PathBuf>,
    extra_engine_args_json: Option<&str>,
    router_config: Option<PathBuf>,
    router_config_json: Option<&str>,
    num_workers: usize,
    replay_concurrency: Option<isize>,
    replay_mode: &str,
    router_mode: &str,
    arrival_speedup_ratio: f64,
) -> PyResult<PyObject> {
    let args = load_replay_mocker_args(py, extra_engine_args, extra_engine_args_json)?;
    let router_config = load_replay_router_config(router_config, router_config_json)?;
    let replay_mode = replay_mode.to_owned();
    let router_mode = parse_replay_router_mode(router_mode)?;
    let report = py.allow_threads(move || {
        let replay_concurrency = parse_replay_concurrency(replay_concurrency)?;

        match (replay_mode.as_str(), replay_concurrency) {
            ("offline", Some(max_in_flight)) => {
                dynamo_mocker::replay::simulate_concurrency_file_with_router_mode(
                    args,
                    router_config.clone(),
                    &trace_file,
                    max_in_flight,
                    num_workers,
                    router_mode,
                )
            }
            ("offline", None) => dynamo_mocker::replay::simulate_trace_file_with_router_mode(
                args,
                router_config.clone(),
                &trace_file,
                num_workers,
                arrival_speedup_ratio,
                router_mode,
            ),
            ("online", Some(max_in_flight)) => {
                dynamo_mocker::replay::simulate_concurrency_live_file_with_router_mode(
                    args,
                    router_config.clone(),
                    &trace_file,
                    max_in_flight,
                    num_workers,
                    router_mode,
                )
            }
            ("online", None) => dynamo_mocker::replay::simulate_trace_live_file_with_router_mode(
                args,
                router_config.clone(),
                &trace_file,
                num_workers,
                arrival_speedup_ratio,
                router_mode,
            ),
            (other, _) => anyhow::bail!(
                "replay_mode must be either 'offline' or 'online', got '{}'",
                other
            ),
        }
    });
    let report = report.map_err(to_pyerr)?;
    pythonize(py, &report)
        .map_err(to_pyerr)
        .map(|obj| obj.unbind())
}

#[pyfunction]
#[pyo3(signature = (input_tokens, output_tokens, request_count, extra_engine_args=None, extra_engine_args_json=None, router_config=None, router_config_json=None, num_workers=1, replay_concurrency=None, replay_mode="offline", router_mode="round_robin", arrival_speedup_ratio=1.0, arrival_interval_ms=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn run_mocker_synthetic_trace_replay(
    py: Python<'_>,
    input_tokens: usize,
    output_tokens: usize,
    request_count: usize,
    extra_engine_args: Option<PathBuf>,
    extra_engine_args_json: Option<&str>,
    router_config: Option<PathBuf>,
    router_config_json: Option<&str>,
    num_workers: usize,
    replay_concurrency: Option<isize>,
    replay_mode: &str,
    router_mode: &str,
    arrival_speedup_ratio: f64,
    arrival_interval_ms: f64,
) -> PyResult<PyObject> {
    let args = load_replay_mocker_args(py, extra_engine_args, extra_engine_args_json)?;
    let router_config = load_replay_router_config(router_config, router_config_json)?;
    let replay_mode = replay_mode.to_owned();
    let router_mode = parse_replay_router_mode(router_mode)?;
    let report = py.allow_threads(move || {
        let replay_concurrency = parse_replay_concurrency(replay_concurrency)?;
        let requests = build_synthetic_requests(
            input_tokens,
            output_tokens,
            request_count,
            arrival_interval_ms,
            replay_concurrency.is_none(),
        )?;

        match (replay_mode.as_str(), replay_concurrency) {
            ("offline", Some(max_in_flight)) => {
                dynamo_mocker::replay::simulate_concurrency_requests_with_router_mode(
                    args,
                    router_config.clone(),
                    requests,
                    max_in_flight,
                    num_workers,
                    router_mode,
                )
            }
            ("offline", None) => dynamo_mocker::replay::simulate_trace_requests_with_router_mode(
                args,
                router_config.clone(),
                requests,
                num_workers,
                arrival_speedup_ratio,
                router_mode,
            ),
            ("online", Some(max_in_flight)) => {
                dynamo_mocker::replay::simulate_concurrency_live_requests_with_router_mode(
                    args,
                    router_config.clone(),
                    requests,
                    max_in_flight,
                    num_workers,
                    router_mode,
                )
            }
            ("online", None) => {
                dynamo_mocker::replay::simulate_trace_live_requests_with_router_mode(
                    args,
                    router_config.clone(),
                    requests,
                    num_workers,
                    arrival_speedup_ratio,
                    router_mode,
                )
            }
            (other, _) => anyhow::bail!(
                "replay_mode must be either 'offline' or 'online', got '{}'",
                other
            ),
        }
    });
    let report = report.map_err(to_pyerr)?;
    pythonize(py, &report)
        .map_err(to_pyerr)
        .map(|obj| obj.unbind())
}

fn load_replay_mocker_args(
    py: Python<'_>,
    extra_engine_args: Option<PathBuf>,
    extra_engine_args_json: Option<&str>,
) -> PyResult<MockEngineArgs> {
    if extra_engine_args.is_some() && extra_engine_args_json.is_some() {
        return Err(PyException::new_err(
            "extra_engine_args and extra_engine_args_json are mutually exclusive",
        ));
    }

    let mut args =
        match (extra_engine_args.as_ref(), extra_engine_args_json) {
            (Some(extra_args_path), None) => MockEngineArgs::from_json_file(extra_args_path)
                .map_err(|e| {
                    PyException::new_err(format!(
                        "Failed to load mocker args from {:?}: {}",
                        extra_args_path, e
                    ))
                })?,
            (None, Some(extra_args_json)) => MockEngineArgs::from_json_str(extra_args_json)
                .map_err(|e| {
                    PyException::new_err(format!("Failed to parse mocker args JSON: {}", e))
                })?,
            (None, None) => MockEngineArgs::default(),
            (Some(_), Some(_)) => unreachable!(),
        };

    if let Some(ref backend_name) = args.aic_backend.clone() {
        let backend = backend_name.clone();
        let system = args.aic_system.as_deref().unwrap_or("h200_sxm").to_string();
        let model_name = args
            .aic_model_path
            .clone()
            .ok_or_else(|| PyException::new_err("--aic-perf-model requires --model-path"))?;
        let backend_version = args.aic_backend_version.clone();
        let tp_size = args.aic_tp_size.unwrap_or(1);
        let callback = create_aic_callback(
            py,
            &backend,
            &system,
            &model_name,
            tp_size,
            backend_version.as_deref(),
        )
        .map_err(|e| {
            PyException::new_err(format!(
                "Failed to create AIC callback (--aic-perf-model was requested): {}",
                e
            ))
        })?;
        tracing::info!(
            "AIC perf model: backend={}, gpu={}, model={}, version={:?}",
            backend,
            system,
            model_name,
            backend_version
        );
        args.perf_model = Arc::new(PerfModel::from_aic_callback(callback));
    }

    Ok(args)
}

fn load_replay_router_config(
    router_config: Option<PathBuf>,
    router_config_json: Option<&str>,
) -> PyResult<Option<RsKvRouterConfig>> {
    if router_config.is_some() && router_config_json.is_some() {
        return Err(PyException::new_err(
            "router_config and router_config_json are mutually exclusive",
        ));
    }

    match (router_config, router_config_json) {
        (Some(path), None) => {
            let file_content = std::fs::read_to_string(&path).map_err(|error| {
                PyException::new_err(format!(
                    "Failed to read replay router config from {:?}: {}",
                    path, error
                ))
            })?;
            serde_json::from_str(&file_content)
                .map(Some)
                .map_err(|error| {
                    PyException::new_err(format!(
                        "Failed to parse replay router config from {:?}: {}",
                        path, error
                    ))
                })
        }
        (None, Some(config_json)) => serde_json::from_str(config_json)
            .map(Some)
            .map_err(|error| {
                PyException::new_err(format!(
                    "Failed to parse replay router config JSON: {}",
                    error
                ))
            }),
        (None, None) => Ok(None),
        (Some(_), Some(_)) => unreachable!(),
    }
}

fn parse_replay_router_mode(
    router_mode: &str,
) -> PyResult<dynamo_mocker::replay::ReplayRouterMode> {
    match router_mode {
        "round_robin" => Ok(dynamo_mocker::replay::ReplayRouterMode::RoundRobin),
        "kv_router" => Ok(dynamo_mocker::replay::ReplayRouterMode::KvRouter),
        other => Err(PyException::new_err(format!(
            "router_mode must be either 'round_robin' or 'kv_router', got '{}'",
            other
        ))),
    }
}

fn parse_replay_concurrency(replay_concurrency: Option<isize>) -> anyhow::Result<Option<usize>> {
    match replay_concurrency {
        Some(value) if value < 1 => anyhow::bail!("replay_concurrency must be at least 1"),
        Some(value) => Ok(Some(value as usize)),
        None => Ok(None),
    }
}

fn build_synthetic_requests(
    input_tokens: usize,
    output_tokens: usize,
    request_count: usize,
    arrival_interval_ms: f64,
    include_arrival_timestamps: bool,
) -> anyhow::Result<Vec<DirectRequest>> {
    if input_tokens == 0 {
        anyhow::bail!("input_tokens must be at least 1");
    }
    if output_tokens == 0 {
        anyhow::bail!("output_tokens must be at least 1");
    }
    if request_count == 0 {
        anyhow::bail!("request_count must be at least 1");
    }
    if !arrival_interval_ms.is_finite() || arrival_interval_ms < 0.0 {
        anyhow::bail!(
            "arrival_interval_ms must be a finite non-negative number, got {}",
            arrival_interval_ms
        );
    }

    let mut requests = Vec::with_capacity(request_count);
    for request_idx in 0..request_count {
        let tokens = (0..input_tokens)
            .map(|token_idx| synthetic_token_id(request_idx, token_idx))
            .collect();
        requests.push(DirectRequest {
            tokens,
            max_output_tokens: output_tokens,
            uuid: Some(Uuid::from_u128((request_idx as u128) + 1)),
            dp_rank: 0,
            arrival_timestamp_ms: include_arrival_timestamps
                .then_some(request_idx as f64 * arrival_interval_ms),
        });
    }

    Ok(requests)
}

fn synthetic_token_id(request_idx: usize, token_idx: usize) -> u32 {
    let mut value =
        (((request_idx as u64) << 32) ^ (token_idx as u64)).wrapping_add(0x9E37_79B9_7F4A_7C15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    let token = value as u32;
    if token == 0 { 1 } else { token }
}

pub fn to_pyerr<E>(err: E) -> PyErr
where
    E: Display,
{
    PyException::new_err(format!("{}", err))
}
