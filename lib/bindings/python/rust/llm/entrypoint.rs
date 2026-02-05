// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Display;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use pyo3::{exceptions::PyException, prelude::*};
use pyo3_async_runtimes::TaskLocals;

use dynamo_llm::discovery::LoadThresholdConfig as RsLoadThresholdConfig;
use dynamo_llm::entrypoint::EngineConfig as RsEngineConfig;
use dynamo_llm::entrypoint::EngineFactoryCallback;
use dynamo_llm::entrypoint::RouterConfig as RsRouterConfig;
use dynamo_llm::entrypoint::input::Input;
use dynamo_llm::kv_router::KvRouterConfig as RsKvRouterConfig;
use dynamo_llm::local_model::DEFAULT_HTTP_PORT;
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder};
use dynamo_llm::mocker::protocols::MockEngineArgs;
use dynamo_llm::model_card::ModelDeploymentCard as RsModelDeploymentCard;
use dynamo_llm::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;
use dynamo_runtime::protocols::EndpointId;

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
#[derive(Default, Clone, Debug, Copy)]
pub struct KvRouterConfig {
    inner: RsKvRouterConfig,
}

impl KvRouterConfig {
    pub fn inner(&self) -> RsKvRouterConfig {
        self.inner
    }
}

#[pymethods]
impl KvRouterConfig {
    #[new]
    #[pyo3(signature = (overlap_score_weight=1.0, router_temperature=0.0, use_kv_events=true, router_replica_sync=false, router_track_active_blocks=true, router_track_output_blocks=false, router_assume_kv_reuse=true, router_snapshot_threshold=1000000, router_reset_states=false, router_ttl_secs=120.0, router_max_tree_size=1048576, router_prune_target_ratio=0.8))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        overlap_score_weight: f64,
        router_temperature: f64,
        use_kv_events: bool,
        router_replica_sync: bool,
        router_track_active_blocks: bool,
        router_track_output_blocks: bool,
        router_assume_kv_reuse: bool,
        router_snapshot_threshold: Option<u32>,
        router_reset_states: bool,
        router_ttl_secs: f64,
        router_max_tree_size: usize,
        router_prune_target_ratio: f64,
    ) -> Self {
        KvRouterConfig {
            inner: RsKvRouterConfig {
                overlap_score_weight,
                router_temperature,
                use_kv_events,
                router_replica_sync,
                router_track_active_blocks,
                router_track_output_blocks,
                router_assume_kv_reuse,
                router_snapshot_threshold,
                router_reset_states,
                router_ttl_secs,
                router_max_tree_size,
                router_prune_target_ratio,
            },
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct RouterConfig {
    router_mode: RouterMode,
    kv_router_config: KvRouterConfig,
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
    namespace: Option<String>,
    custom_backend_metrics_endpoint: Option<String>,
    custom_backend_metrics_polling_interval: Option<f64>,
    is_prefill: bool,
    engine_factory: Option<PyEngineFactory>,
}

#[pymethods]
impl EntrypointArgs {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (engine_type, model_path=None, model_name=None, endpoint_id=None, context_length=None, template_file=None, router_config=None, kv_cache_block_size=None, http_host=None, http_port=None, http_metrics_port=None, tls_cert_path=None, tls_key_path=None, extra_engine_args=None, namespace=None, custom_backend_metrics_endpoint=None, custom_backend_metrics_polling_interval=None, is_prefill=false, engine_factory=None))]
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
        namespace: Option<String>,
        custom_backend_metrics_endpoint: Option<String>,
        custom_backend_metrics_polling_interval: Option<f64>,
        is_prefill: bool,
        engine_factory: Option<PyObject>,
    ) -> PyResult<Self> {
        let endpoint_id_obj: Option<EndpointId> = endpoint_id.as_deref().map(EndpointId::from);
        if (tls_cert_path.is_some() && tls_key_path.is_none())
            || (tls_cert_path.is_none() && tls_key_path.is_some())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tls_cert_path and tls_key_path must be provided together",
            ));
        }

        // Capture TaskLocals at registration time for the engine factory callback
        let engine_factory = engine_factory
            .map(|callback| {
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to get TaskLocals for engine_factory: {}",
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
            namespace,
            custom_backend_metrics_endpoint,
            custom_backend_metrics_polling_interval,
            is_prefill,
            engine_factory,
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
        .http_host(args.http_host.clone())
        .http_port(args.http_port)
        .http_metrics_port(args.http_metrics_port)
        .tls_cert_path(args.tls_cert_path.clone())
        .tls_key_path(args.tls_key_path.clone())
        .is_mocker(matches!(args.engine_type, EngineType::Mocker))
        .extra_engine_args(args.extra_engine_args.clone())
        .namespace(args.namespace.clone())
        .custom_backend_metrics_endpoint(args.custom_backend_metrics_endpoint.clone())
        .custom_backend_metrics_polling_interval(args.custom_backend_metrics_polling_interval);
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

/// Convert a PyEngineFactory to a Rust EngineFactoryCallback
fn py_engine_factory_to_callback(factory: PyEngineFactory) -> EngineFactoryCallback {
    let callback = factory.callback;
    let locals = factory.locals;

    Arc::new(
        move |card: RsModelDeploymentCard| -> Pin<
            Box<dyn Future<Output = anyhow::Result<OpenAIChatCompletionsStreamingEngine>> + Send>,
        > {
            let callback = callback.clone();
            let locals = locals.clone();

            Box::pin(async move {
                // Acquire GIL to call Python callback and convert coroutine to future
                let py_future = Python::with_gil(|py| {
                    // Create Python ModelDeploymentCard wrapper
                    let py_card = ModelDeploymentCard { inner: card };
                    let py_card_obj = Py::new(py, py_card)
                        .map_err(|e| anyhow::anyhow!("Failed to create Python MDC: {}", e))?;

                    // Call Python async function to get a coroutine
                    let coroutine = callback
                        .call1(py, (py_card_obj,))
                        .map_err(|e| anyhow::anyhow!("Failed to call engine_factory: {}", e))?;

                    // Use the TaskLocals captured at registration time
                    pyo3_async_runtimes::into_future_with_locals(&locals, coroutine.into_bound(py))
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to convert coroutine to future: {}", e)
                        })
                })?;

                // Await the Python coroutine (GIL is released during await)
                let py_result = py_future
                    .await
                    .map_err(|e| anyhow::anyhow!("engine_factory callback failed: {}", e))?;

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
            //  Convert Python engine factory to Rust callback
            let engine_factory = args.engine_factory.map(py_engine_factory_to_callback);
            RsEngineConfig::Dynamic {
                model: Box::new(local_model),
                engine_factory,
            }
        }
        EngineType::Mocker => {
            let mocker_args = if let Some(extra_args_path) = args.extra_engine_args {
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

            let endpoint = local_model.endpoint_id().clone();

            let engine = dynamo_llm::mocker::make_mocker_engine(
                distributed_runtime.inner,
                endpoint,
                mocker_args,
            )
            .await?;

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

pub fn to_pyerr<E>(err: E) -> PyErr
where
    E: Display,
{
    PyException::new_err(format!("{}", err))
}
