// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Display;
use std::path::PathBuf;

use pyo3::{exceptions::PyException, prelude::*};

use dynamo_llm::entrypoint::EngineConfig as RsEngineConfig;
use dynamo_llm::entrypoint::RouterConfig as RsRouterConfig;
use dynamo_llm::entrypoint::input::Input;
use dynamo_llm::kv_router::KvRouterConfig as RsKvRouterConfig;
use dynamo_llm::local_model::DEFAULT_HTTP_PORT;
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder};
use dynamo_llm::mocker::protocols::MockEngineArgs;
use dynamo_runtime::protocols::EndpointId;

use crate::RouterMode;

#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
#[repr(i32)]
pub enum EngineType {
    Echo = 1,
    Dynamic = 2,
    Mocker = 3,
    Static = 4,
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
    #[pyo3(signature = (overlap_score_weight=1.0, router_temperature=0.0, use_kv_events=true, router_replica_sync=false, router_track_active_blocks=true, router_snapshot_threshold=1000000, router_reset_states=false))]
    fn new(
        overlap_score_weight: f64,
        router_temperature: f64,
        use_kv_events: bool,
        router_replica_sync: bool,
        router_track_active_blocks: bool,
        router_snapshot_threshold: Option<u32>,
        router_reset_states: bool,
    ) -> Self {
        KvRouterConfig {
            inner: RsKvRouterConfig {
                overlap_score_weight,
                router_temperature,
                use_kv_events,
                router_replica_sync,
                router_track_active_blocks,
                router_snapshot_threshold,
                router_reset_states,
            },
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct RouterConfig {
    router_mode: RouterMode,
    kv_router_config: KvRouterConfig,
    busy_threshold: Option<f64>,
}

#[pymethods]
impl RouterConfig {
    #[new]
    #[pyo3(signature = (mode, config=None, busy_threshold=None))]
    pub fn new(
        mode: RouterMode,
        config: Option<KvRouterConfig>,
        busy_threshold: Option<f64>,
    ) -> Self {
        Self {
            router_mode: mode,
            kv_router_config: config.unwrap_or_default(),
            busy_threshold,
        }
    }
}

impl From<RouterConfig> for RsRouterConfig {
    fn from(rc: RouterConfig) -> RsRouterConfig {
        RsRouterConfig {
            router_mode: rc.router_mode.into(),
            kv_router_config: rc.kv_router_config.inner,
            busy_threshold: rc.busy_threshold,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct EntrypointArgs {
    engine_type: EngineType,
    model_path: Option<PathBuf>,
    model_name: Option<String>,
    model_config: Option<PathBuf>,
    endpoint_id: Option<EndpointId>,
    context_length: Option<u32>,
    template_file: Option<PathBuf>,
    router_config: Option<RouterConfig>,
    kv_cache_block_size: Option<u32>,
    http_host: Option<String>,
    http_port: u16,
    tls_cert_path: Option<PathBuf>,
    tls_key_path: Option<PathBuf>,
    extra_engine_args: Option<PathBuf>,
    namespace: Option<String>,
}

#[pymethods]
impl EntrypointArgs {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (engine_type, model_path=None, model_name=None, model_config=None, endpoint_id=None, context_length=None, template_file=None, router_config=None, kv_cache_block_size=None, http_host=None, http_port=None, tls_cert_path=None, tls_key_path=None, extra_engine_args=None, namespace=None))]
    pub fn new(
        engine_type: EngineType,
        model_path: Option<PathBuf>,
        model_name: Option<String>, // e.g. "dyn://namespace.component.endpoint"
        model_config: Option<PathBuf>,
        endpoint_id: Option<String>,
        context_length: Option<u32>,
        template_file: Option<PathBuf>,
        router_config: Option<RouterConfig>,
        kv_cache_block_size: Option<u32>,
        http_host: Option<String>,
        http_port: Option<u16>,
        tls_cert_path: Option<PathBuf>,
        tls_key_path: Option<PathBuf>,
        extra_engine_args: Option<PathBuf>,
        namespace: Option<String>,
    ) -> PyResult<Self> {
        let endpoint_id_obj: Option<EndpointId> = endpoint_id.as_deref().map(EndpointId::from);
        if (tls_cert_path.is_some() && tls_key_path.is_none())
            || (tls_cert_path.is_none() && tls_key_path.is_some())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tls_cert_path and tls_key_path must be provided together",
            ));
        }
        Ok(EntrypointArgs {
            engine_type,
            model_path,
            model_name,
            model_config,
            endpoint_id: endpoint_id_obj,
            context_length,
            template_file,
            router_config,
            kv_cache_block_size,
            http_host,
            http_port: http_port.unwrap_or(DEFAULT_HTTP_PORT),
            tls_cert_path,
            tls_key_path,
            extra_engine_args,
            namespace,
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct EngineConfig {
    inner: RsEngineConfig,
}

#[pyfunction]
#[pyo3(signature = (distributed_runtime, args))]
pub fn make_engine<'p>(
    py: Python<'p>,
    distributed_runtime: super::DistributedRuntime,
    args: EntrypointArgs,
) -> PyResult<Bound<'p, PyAny>> {
    let mut builder = LocalModelBuilder::default();
    builder
        .model_path(args.model_path.clone())
        .model_name(args.model_name.clone())
        .model_config(args.model_config.clone())
        .endpoint_id(args.endpoint_id.clone())
        .context_length(args.context_length)
        .request_template(args.template_file.clone())
        .kv_cache_block_size(args.kv_cache_block_size)
        .router_config(args.router_config.clone().map(|rc| rc.into()))
        .http_host(args.http_host.clone())
        .http_port(args.http_port)
        .tls_cert_path(args.tls_cert_path.clone())
        .tls_key_path(args.tls_key_path.clone())
        .is_mocker(matches!(args.engine_type, EngineType::Mocker))
        .extra_engine_args(args.extra_engine_args.clone())
        .namespace(args.namespace.clone());
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let local_model = builder.build().await.map_err(to_pyerr)?;
        let inner = select_engine(distributed_runtime, args, local_model)
            .await
            .map_err(to_pyerr)?;
        Ok(EngineConfig { inner })
    })
}

async fn select_engine(
    #[allow(unused_variables)] distributed_runtime: super::DistributedRuntime,
    args: EntrypointArgs,
    local_model: LocalModel,
) -> anyhow::Result<RsEngineConfig> {
    let inner = match args.engine_type {
        EngineType::Echo => {
            // There is no validation for the echo engine
            RsEngineConfig::StaticFull {
                model: Box::new(local_model),
                engine: dynamo_llm::engines::make_echo_engine(),
                is_static: false,
            }
        }
        EngineType::Dynamic => RsEngineConfig::Dynamic(Box::new(local_model)),
        EngineType::Static => RsEngineConfig::StaticRemote(Box::new(local_model)),
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

            let engine = dynamo_llm::mocker::engine::make_mocker_engine(
                distributed_runtime.inner,
                endpoint,
                mocker_args,
            )
            .await?;

            RsEngineConfig::StaticCore {
                engine,
                model: Box::new(local_model),
                is_static: false,
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
            either::Either::Right(distributed_runtime.inner.clone()),
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
