// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Display;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use pyo3::{
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
};
use pyo3_async_runtimes::TaskLocals;

use dynamo_kv_router::config::{
    KvRouterConfig as RsKvRouterConfig, RouterPrefillLoadModel as RsRouterPrefillLoadModel,
    apply_deprecated_overlap_score_weight_override,
};
use dynamo_llm::discovery::LoadThresholdConfig as RsLoadThresholdConfig;
use dynamo_llm::entrypoint::EngineConfig as RsEngineConfig;
use dynamo_llm::entrypoint::RouterConfig as RsRouterConfig;
use dynamo_llm::entrypoint::input::Input;
use dynamo_llm::entrypoint::{ChatEngineFactoryCallback, PrefillRoutedEngine};
use dynamo_llm::frontend_config::{FrontendApiConfig, MetricsConfig};
use dynamo_llm::local_model::DEFAULT_HTTP_PORT;
use dynamo_llm::local_model::runtime_config::TokenizerBackend;
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder};
use dynamo_llm::mocker::make_mocker_engine;
use dynamo_llm::model_card::ModelDeploymentCard as RsModelDeploymentCard;
use dynamo_llm::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;
use dynamo_mocker::common::perf_model::PerfModel;

use super::aic_callback::{create_aic_callback, create_aic_prefill_load_estimator};
use super::replay::MockEngineArgs as PyMockEngineArgs;
use dynamo_mocker::common::protocols::MockEngineArgs as RsMockEngineArgs;
use dynamo_runtime::discovery::ModelCardInstanceId as RsModelCardInstanceId;
use dynamo_runtime::protocols::EndpointId;

use super::local_model::ModelRuntimeConfig;
use super::model_card::ModelDeploymentCard;
use crate::RouterMode;
use crate::engine::PythonAsyncEngine;

fn validate_kv_router_config(config: &RsKvRouterConfig) -> PyResult<()> {
    config.validate_config().map_err(PyValueError::new_err)
}

fn warn_overlap_score_weight_deprecated() {
    tracing::warn!("overlap_score_weight is deprecated; use prefill_load_scale");
}

fn apply_deprecated_overlap_score_weight(
    value: f64,
    overlap_score_credit: &mut f64,
    prefill_load_scale: &mut f64,
) {
    warn_overlap_score_weight_deprecated();
    apply_deprecated_overlap_score_weight_override(value, overlap_score_credit, prefill_load_scale);
}

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

#[pyclass]
#[derive(Clone, Debug)]
pub struct AicPerfConfig {
    aic_backend: String,
    aic_system: String,
    aic_backend_version: Option<String>,
    aic_tp_size: usize,
    aic_model_path: String,
    aic_moe_tp_size: Option<usize>,
    aic_moe_ep_size: Option<usize>,
    aic_attention_dp_size: Option<usize>,
    aic_gemm_dtype: Option<String>,
    aic_moe_dtype: Option<String>,
    aic_fmha_dtype: Option<String>,
    aic_kv_cache_dtype: Option<String>,
    aic_comm_dtype: Option<String>,
    aic_nextn: Option<usize>,
    aic_nextn_accept_rates: Option<String>,
}

impl AicPerfConfig {
    pub(crate) fn backend_name(&self) -> &str {
        &self.aic_backend
    }

    pub(crate) fn system(&self) -> &str {
        &self.aic_system
    }

    pub(crate) fn backend_version(&self) -> Option<&str> {
        self.aic_backend_version.as_deref()
    }

    pub(crate) fn tp_size(&self) -> usize {
        self.aic_tp_size
    }

    pub(crate) fn model_path(&self) -> &str {
        &self.aic_model_path
    }

    pub(crate) fn moe_tp_size(&self) -> Option<usize> {
        self.aic_moe_tp_size
    }

    pub(crate) fn moe_ep_size(&self) -> Option<usize> {
        self.aic_moe_ep_size
    }

    pub(crate) fn attention_dp_size(&self) -> Option<usize> {
        self.aic_attention_dp_size
    }

    pub(crate) fn gemm_dtype(&self) -> Option<&str> {
        self.aic_gemm_dtype.as_deref()
    }

    pub(crate) fn moe_dtype(&self) -> Option<&str> {
        self.aic_moe_dtype.as_deref()
    }

    pub(crate) fn fmha_dtype(&self) -> Option<&str> {
        self.aic_fmha_dtype.as_deref()
    }

    pub(crate) fn kv_cache_dtype(&self) -> Option<&str> {
        self.aic_kv_cache_dtype.as_deref()
    }

    pub(crate) fn comm_dtype(&self) -> Option<&str> {
        self.aic_comm_dtype.as_deref()
    }

    pub(crate) fn nextn(&self) -> Option<usize> {
        self.aic_nextn
    }

    pub(crate) fn nextn_accept_rates(&self) -> Option<&str> {
        self.aic_nextn_accept_rates.as_deref()
    }
}

#[pymethods]
impl AicPerfConfig {
    #[new]
    #[pyo3(signature = (aic_backend, aic_system, aic_model_path, aic_tp_size=1, aic_backend_version=None, aic_moe_tp_size=None, aic_moe_ep_size=None, aic_attention_dp_size=None, aic_nextn=None, aic_nextn_accept_rates=None, aic_gemm_dtype=None, aic_moe_dtype=None, aic_fmha_dtype=None, aic_kv_cache_dtype=None, aic_comm_dtype=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        aic_backend: String,
        aic_system: String,
        aic_model_path: String,
        aic_tp_size: usize,
        aic_backend_version: Option<String>,
        aic_moe_tp_size: Option<usize>,
        aic_moe_ep_size: Option<usize>,
        aic_attention_dp_size: Option<usize>,
        aic_nextn: Option<usize>,
        aic_nextn_accept_rates: Option<String>,
        aic_gemm_dtype: Option<String>,
        aic_moe_dtype: Option<String>,
        aic_fmha_dtype: Option<String>,
        aic_kv_cache_dtype: Option<String>,
        aic_comm_dtype: Option<String>,
    ) -> PyResult<Self> {
        if aic_backend.is_empty() {
            return Err(PyValueError::new_err("aic_backend must be non-empty"));
        }
        if aic_system.is_empty() {
            return Err(PyValueError::new_err("aic_system must be non-empty"));
        }
        if aic_model_path.is_empty() {
            return Err(PyValueError::new_err("aic_model_path must be non-empty"));
        }
        if aic_tp_size == 0 {
            return Err(PyValueError::new_err("aic_tp_size must be >= 1"));
        }
        for (name, value) in [
            ("aic_moe_tp_size", aic_moe_tp_size),
            ("aic_moe_ep_size", aic_moe_ep_size),
            ("aic_attention_dp_size", aic_attention_dp_size),
        ] {
            if matches!(value, Some(0)) {
                return Err(PyValueError::new_err(format!("{name} must be >= 1")));
            }
        }
        // AIC caps MTP draft tokens at 5; >5 would IndexError in calc_expectation.
        if let Some(nextn) = aic_nextn
            && !(1..=5).contains(&nextn)
        {
            return Err(PyValueError::new_err(
                "aic_nextn must be 1..=5 when set (omit to disable spec dec)",
            ));
        }

        Ok(Self {
            aic_backend,
            aic_system,
            aic_backend_version,
            aic_tp_size,
            aic_model_path,
            aic_moe_tp_size,
            aic_moe_ep_size,
            aic_attention_dp_size,
            aic_gemm_dtype,
            aic_moe_dtype,
            aic_fmha_dtype,
            aic_kv_cache_dtype,
            aic_comm_dtype,
            aic_nextn,
            aic_nextn_accept_rates,
        })
    }
}

#[pymethods]
impl KvRouterConfig {
    #[new]
    #[pyo3(signature = (overlap_score_weight=None, host_cache_hit_weight=0.75, disk_cache_hit_weight=0.25, router_temperature=0.0, use_kv_events=true, durable_kv_events=false, router_replica_sync=false, router_track_active_blocks=true, router_track_output_blocks=false, router_assume_kv_reuse=true, router_track_prefill_tokens=true, router_prefill_load_model="none", router_snapshot_threshold=1000000, router_reset_states=false, router_ttl_secs=120.0, router_queue_threshold=None, router_event_threads=4, router_queue_policy="fcfs", use_remote_indexer=false, serve_indexer=false, shared_cache_multiplier=0.0, shared_cache_type="none", router_predicted_ttl_secs=None, *, overlap_score_credit=1.0, overlap_score_credit_decay=0.0, prefill_load_scale=1.0, router_policy_config=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        overlap_score_weight: Option<f64>,
        host_cache_hit_weight: f64,
        disk_cache_hit_weight: f64,
        router_temperature: f64,
        use_kv_events: bool,
        durable_kv_events: bool,
        router_replica_sync: bool,
        router_track_active_blocks: bool,
        router_track_output_blocks: bool,
        router_assume_kv_reuse: bool,
        router_track_prefill_tokens: bool,
        router_prefill_load_model: &str,
        router_snapshot_threshold: Option<u32>,
        router_reset_states: bool,
        router_ttl_secs: f64,
        router_queue_threshold: Option<f64>,
        router_event_threads: u32,
        router_queue_policy: &str,
        use_remote_indexer: bool,
        serve_indexer: bool,
        shared_cache_multiplier: f64,
        shared_cache_type: &str,
        router_predicted_ttl_secs: Option<f64>,
        mut overlap_score_credit: f64,
        overlap_score_credit_decay: f64,
        mut prefill_load_scale: f64,
        router_policy_config: Option<String>,
    ) -> PyResult<Self> {
        if let Some(value) = overlap_score_weight {
            apply_deprecated_overlap_score_weight(
                value,
                &mut overlap_score_credit,
                &mut prefill_load_scale,
            );
        }

        let inner = RsKvRouterConfig {
            overlap_score_credit,
            overlap_score_credit_decay,
            prefill_load_scale,
            host_cache_hit_weight,
            disk_cache_hit_weight,
            router_temperature,
            use_kv_events,
            durable_kv_events,
            router_replica_sync,
            router_track_active_blocks,
            router_track_output_blocks,
            router_assume_kv_reuse,
            router_track_prefill_tokens,
            router_prefill_load_model: router_prefill_load_model
                .parse::<RsRouterPrefillLoadModel>()
                .map_err(PyValueError::new_err)?,
            router_snapshot_threshold,
            router_reset_states,
            router_ttl_secs,
            router_queue_threshold,
            router_policy_config,
            policy_model_name: None,
            policy_config_cache: Default::default(),
            router_event_threads,
            skip_initial_worker_wait: false,
            router_queue_policy: router_queue_policy.parse().map_err(PyValueError::new_err)?,
            use_remote_indexer,
            serve_indexer,
            shared_cache_multiplier,
            shared_cache_type: shared_cache_type.parse().map_err(PyValueError::new_err)?,
            router_predicted_ttl_secs,
        };
        validate_kv_router_config(&inner)?;
        Ok(KvRouterConfig { inner })
    }

    #[staticmethod]
    fn from_json(config_json: &str) -> PyResult<Self> {
        let inner = serde_json::from_str::<RsKvRouterConfig>(config_json).map_err(|e| {
            PyValueError::new_err(format!("Failed to parse KvRouterConfig JSON: {e}"))
        })?;
        validate_kv_router_config(&inner)?;
        Ok(KvRouterConfig { inner })
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    #[getter]
    fn overlap_score_credit(&self) -> f64 {
        self.inner.overlap_score_credit
    }

    #[setter]
    fn set_overlap_score_credit(&mut self, value: f64) -> PyResult<()> {
        let mut inner = self.inner.clone();
        inner.overlap_score_credit = value;
        validate_kv_router_config(&inner)?;
        self.inner = inner;
        Ok(())
    }

    #[getter]
    fn overlap_score_credit_decay(&self) -> f64 {
        self.inner.overlap_score_credit_decay
    }

    #[setter]
    fn set_overlap_score_credit_decay(&mut self, value: f64) -> PyResult<()> {
        let mut inner = self.inner.clone();
        inner.overlap_score_credit_decay = value;
        validate_kv_router_config(&inner)?;
        self.inner = inner;
        Ok(())
    }

    #[getter]
    fn overlap_score_weight(&self) -> f64 {
        self.inner.prefill_load_scale
    }

    #[setter]
    fn set_overlap_score_weight(&mut self, value: f64) -> PyResult<()> {
        let mut inner = self.inner.clone();
        apply_deprecated_overlap_score_weight(
            value,
            &mut inner.overlap_score_credit,
            &mut inner.prefill_load_scale,
        );
        validate_kv_router_config(&inner)?;
        self.inner = inner;
        Ok(())
    }

    #[getter]
    fn prefill_load_scale(&self) -> f64 {
        self.inner.prefill_load_scale
    }

    #[setter]
    fn set_prefill_load_scale(&mut self, value: f64) -> PyResult<()> {
        let mut inner = self.inner.clone();
        inner.prefill_load_scale = value;
        validate_kv_router_config(&inner)?;
        self.inner = inner;
        Ok(())
    }

    #[pyo3(signature = (overlap_score_weight=None, *, overlap_score_credit=None, overlap_score_credit_decay=None, prefill_load_scale=None))]
    fn with_overrides(
        &self,
        overlap_score_weight: Option<f64>,
        overlap_score_credit: Option<f64>,
        overlap_score_credit_decay: Option<f64>,
        prefill_load_scale: Option<f64>,
    ) -> PyResult<Self> {
        let mut inner = self.inner.clone();
        if let Some(credit) = overlap_score_credit {
            inner.overlap_score_credit = credit;
        }
        if let Some(decay) = overlap_score_credit_decay {
            inner.overlap_score_credit_decay = decay;
        }
        if let Some(scale) = prefill_load_scale {
            inner.prefill_load_scale = scale;
        }
        if let Some(weight) = overlap_score_weight {
            apply_deprecated_overlap_score_weight(
                weight,
                &mut inner.overlap_score_credit,
                &mut inner.prefill_load_scale,
            );
        }
        validate_kv_router_config(&inner)?;
        Ok(Self { inner })
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
    session_affinity_ttl_secs: Option<u64>,
}

#[pymethods]
impl RouterConfig {
    #[new]
    #[pyo3(signature = (mode, config=None, active_decode_blocks_threshold=None, active_prefill_tokens_threshold=None, active_prefill_tokens_threshold_frac=None, enforce_disagg=false, session_affinity_ttl_secs=None))]
    pub fn new(
        mode: RouterMode,
        config: Option<KvRouterConfig>,
        active_decode_blocks_threshold: Option<f64>,
        active_prefill_tokens_threshold: Option<u64>,
        active_prefill_tokens_threshold_frac: Option<f64>,
        enforce_disagg: bool,
        session_affinity_ttl_secs: Option<u64>,
    ) -> PyResult<Self> {
        if enforce_disagg {
            static WARN_ONCE: std::sync::Once = std::sync::Once::new();
            WARN_ONCE.call_once(|| {
                tracing::warn!(
                    "enforce_disagg is deprecated and ignored; routing topology and readiness are determined from registered worker types"
                );
            });
        }
        if session_affinity_ttl_secs.is_some_and(|ttl| !(1..=31_536_000).contains(&ttl)) {
            return Err(PyValueError::new_err(
                "session_affinity_ttl_secs must be between 1 and 31536000",
            ));
        }
        RsLoadThresholdConfig {
            active_decode_blocks_threshold,
            active_prefill_tokens_threshold,
            active_prefill_tokens_threshold_frac,
        }
        .validate()
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            router_mode: mode,
            kv_router_config: config.unwrap_or_default(),
            active_decode_blocks_threshold,
            active_prefill_tokens_threshold,
            active_prefill_tokens_threshold_frac,
            session_affinity_ttl_secs,
        })
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
            enforce_disagg: false,
            session_affinity_ttl_secs: rc.session_affinity_ttl_secs,
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
    template_file: Option<PathBuf>,
    router_config: Option<RouterConfig>,
    kv_cache_block_size: Option<u32>,
    http_host: Option<String>,
    http_port: u16,
    http_metrics_port: Option<u16>,
    metrics_config: Option<MetricsConfig>,
    frontend_api_config: Option<FrontendApiConfig>,
    tls_cert_path: Option<PathBuf>,
    tls_key_path: Option<PathBuf>,
    extra_engine_args: Option<PathBuf>,
    mocker_engine_args: Option<PyMockEngineArgs>,
    runtime_config: ModelRuntimeConfig,
    namespace: Option<String>,
    namespace_prefix: Option<String>,
    is_prefill: bool,
    is_decode: bool,
    migration_limit: u32,
    migration_max_seq_len: Option<u32>,
    chat_engine_factory: Option<PyEngineFactory>,
    aic_perf_config: Option<AicPerfConfig>,
}

#[pymethods]
impl EntrypointArgs {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (engine_type, model_path=None, model_name=None, endpoint_id=None, template_file=None, router_config=None, kv_cache_block_size=None, http_host=None, http_port=None, http_metrics_port=None, tls_cert_path=None, tls_key_path=None, extra_engine_args=None, mocker_engine_args=None, runtime_config=None, namespace=None, namespace_prefix=None, is_prefill=false, is_decode=false, migration_limit=0, migration_max_seq_len=None, chat_engine_factory=None, aic_perf_config=None, *, metrics_prefix=None, enable_anthropic_api=None, strip_anthropic_preamble=None, enable_streaming_tool_dispatch=None, enable_streaming_reasoning_dispatch=None, tokenizer_backend=None))]
    pub fn new(
        py: Python<'_>,
        engine_type: EngineType,
        model_path: Option<PathBuf>,
        model_name: Option<String>, // e.g. "dyn://namespace.component.endpoint"
        endpoint_id: Option<String>,
        template_file: Option<PathBuf>,
        router_config: Option<RouterConfig>,
        kv_cache_block_size: Option<u32>,
        http_host: Option<String>,
        http_port: Option<u16>,
        http_metrics_port: Option<u16>,
        tls_cert_path: Option<PathBuf>,
        tls_key_path: Option<PathBuf>,
        extra_engine_args: Option<PathBuf>,
        mocker_engine_args: Option<PyMockEngineArgs>,
        runtime_config: Option<ModelRuntimeConfig>,
        namespace: Option<String>,
        namespace_prefix: Option<String>,
        is_prefill: bool,
        is_decode: bool,
        migration_limit: u32,
        migration_max_seq_len: Option<u32>,
        chat_engine_factory: Option<PyObject>,
        aic_perf_config: Option<AicPerfConfig>,
        metrics_prefix: Option<String>,
        enable_anthropic_api: Option<bool>,
        strip_anthropic_preamble: Option<bool>,
        enable_streaming_tool_dispatch: Option<bool>,
        enable_streaming_reasoning_dispatch: Option<bool>,
        tokenizer_backend: Option<String>,
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

        let tokenizer_backend = tokenizer_backend
            .map(|backend| {
                backend
                    .parse::<TokenizerBackend>()
                    .map_err(PyValueError::new_err)
            })
            .transpose()?;

        let mut runtime_config = runtime_config.unwrap_or_default();
        if let Some(tokenizer_backend) = tokenizer_backend {
            runtime_config
                .inner
                .set_tokenizer_backend(Some(tokenizer_backend));
        }
        runtime_config.validate_config()?;

        Ok(EntrypointArgs {
            engine_type,
            model_path,
            model_name,
            endpoint_id: endpoint_id_obj,
            template_file,
            router_config,
            kv_cache_block_size,
            http_host,
            http_port: http_port.unwrap_or(DEFAULT_HTTP_PORT),
            http_metrics_port,
            metrics_config: metrics_prefix.map(|prefix| MetricsConfig::new(Some(prefix))),
            frontend_api_config: FrontendApiConfig::from_optional_flags(
                enable_anthropic_api,
                strip_anthropic_preamble,
                enable_streaming_tool_dispatch,
                enable_streaming_reasoning_dispatch,
            ),
            tls_cert_path,
            tls_key_path,
            extra_engine_args,
            mocker_engine_args,
            runtime_config,
            namespace,
            namespace_prefix,
            is_prefill,
            is_decode,
            migration_limit,
            migration_max_seq_len,
            chat_engine_factory,
            aic_perf_config,
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
        .request_template(args.template_file.clone())
        .kv_cache_block_size(args.kv_cache_block_size)
        .router_config(args.router_config.clone().map(|rc| rc.into()))
        .migration_limit(Some(args.migration_limit))
        .migration_max_seq_len(args.migration_max_seq_len)
        .http_host(args.http_host.clone())
        .http_port(args.http_port)
        .http_metrics_port(args.http_metrics_port);
    if let Some(metrics_config) = args.metrics_config.clone() {
        builder.metrics_config(metrics_config);
    }
    if let Some(frontend_api_config) = args.frontend_api_config.clone() {
        builder.frontend_api_config(frontend_api_config);
    }
    builder
        .tls_cert_path(args.tls_cert_path.clone())
        .tls_key_path(args.tls_key_path.clone())
        .is_mocker(matches!(args.engine_type, EngineType::Mocker))
        .extra_engine_args(args.extra_engine_args.clone())
        .runtime_config(args.runtime_config.clone().inner)
        .namespace(args.namespace.clone())
        .namespace_prefix(args.namespace_prefix.clone());
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        if let Some(model_path) = args.model_path.clone() {
            let local_path = if model_path.exists() {
                model_path
            } else {
                // Mocker only needs tokenizer, not weights
                let ignore_weights = matches!(args.engine_type, EngineType::Mocker);
                // Preserve the original HF model ID as source_path so the
                // frontend can resolve model metadata even when the served
                // model name differs (e.g., --model-name model-1 --model-path
                // Qwen/Qwen3-0.6B).
                builder.source_path(model_path.clone());
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
              card: RsModelDeploymentCard,
              routed_engine: PrefillRoutedEngine|
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
                    let py_routed = Py::new(
                        py,
                        crate::llm::routed_engine::RoutedEngine::new(routed_engine),
                    )
                    .map_err(|e| anyhow::anyhow!("Failed to create Python RoutedEngine: {e}"))?;

                    // Call Python async function to get a coroutine
                    let coroutine = callback
                        .call1(py, (py_instance_id, py_card_obj, py_routed))
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
            let prefill_load_estimator = args
                .aic_perf_config
                .as_ref()
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
                .transpose()?;
            RsEngineConfig::Dynamic {
                model: Box::new(local_model),
                chat_engine_factory,
                prefill_load_estimator,
            }
        }
        EngineType::Mocker => {
            let mut mocker_args = if let Some(mocker_engine_args) = args.mocker_engine_args {
                mocker_engine_args.inner()
            } else if let Some(extra_args_path) = args.extra_engine_args {
                RsMockEngineArgs::from_json_file(&extra_args_path).map_err(|e| {
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
                RsMockEngineArgs::default()
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
                let moe_tp_size = mocker_args.aic_moe_tp_size;
                let moe_ep_size = mocker_args.aic_moe_ep_size;
                let attention_dp_size = mocker_args.aic_attention_dp_size;
                let gemm_dtype = mocker_args.aic_gemm_dtype.as_deref();
                let moe_dtype = mocker_args.aic_moe_dtype.as_deref();
                let fmha_dtype = mocker_args.aic_fmha_dtype.as_deref();
                let kv_cache_dtype = mocker_args.aic_kv_cache_dtype.as_deref();
                let comm_dtype = mocker_args.aic_comm_dtype.as_deref();
                let nextn = mocker_args.aic_nextn;
                let undiscounted_accept_rates = mocker_args.undiscounted_aic_accept_rates();
                match Python::with_gil(|py| {
                    create_aic_callback(
                        py,
                        &backend,
                        system,
                        model_name,
                        tp_size,
                        backend_version,
                        moe_tp_size,
                        moe_ep_size,
                        attention_dp_size,
                        gemm_dtype,
                        moe_dtype,
                        fmha_dtype,
                        kv_cache_dtype,
                        comm_dtype,
                        nextn,
                        undiscounted_accept_rates.as_deref(),
                    )
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
                is_decode: args.is_decode,
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
pub fn run_sglang_sidecar(py: Python<'_>, args: Vec<String>) -> PyResult<()> {
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("python -m dynamo.sglang_sidecar".to_string());
    argv.extend(args);

    let (engine, config) = py
        .allow_threads(|| dynamo_sglang_sidecar::SglangSidecarEngine::from_args(Some(argv)))
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    py.allow_threads(|| {
        dynamo_backend_common::run(Arc::new(engine), config)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    })
}

pub fn to_pyerr<E>(err: E) -> PyErr
where
    E: Display,
{
    PyException::new_err(format!("{}", err))
}
