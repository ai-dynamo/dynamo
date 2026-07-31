// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::time::Duration;

use dynamo_renderer::PromptFormatter;

use crate::{
    backend::{Backend, ExecutionContext},
    discovery::{KvWorkerMonitor, ModelManager, ModelWatcher},
    engines::StreamingEngineAdapter,
    entrypoint::EngineConfig,
    http::service::metrics::Metrics,
    kv_router::indexer::try_build_cache_indexer,
    kv_router::{
        EncoderRouter, KvPushRouter, KvRouter, PrefillRouter, metrics::RouterRequestMetrics,
    },
    lora::LoraFilteredRouter,
    migration::Migration,
    model_card::ModelDeploymentCard,
    namespace::NamespaceFilter,
    preprocessor::{OpenAIPreprocessor, prompt::prompt_formatter_from_mdc},
    protocols::common::{
        llm_backend::{BackendOutput, LLMEngineOutput, PreprocessedRequest},
        preprocessor::MultimodalData,
    },
    request_template::RequestTemplate,
    session_affinity::{
        AffinityCoordinator, SessionAffinityPushRouter, create_affinity_coordinator,
    },
    types::{
        Annotated,
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            OpenAIChatCompletionsStreamingEngine,
        },
    },
};

use dynamo_kv_router::config::min_initial_workers_from_env;
use dynamo_runtime::{
    DistributedRuntime,
    component::Client,
    engine::{AsyncEngineStream, Data},
    pipeline::{
        Context, ManyOut, MultimodalCacheKeyExtractor, Operator, PushRouter, RouterMode,
        SegmentSource, ServiceBackend, ServiceEngine, ServiceFrontend, SingleIn, Source,
    },
};
use std::sync::Arc;

fn multimodal_cache_key_from_url(url: &str) -> String {
    blake3::hash(url.as_bytes()).to_hex().to_string()
}

fn preprocessed_multimodal_cache_keys(request: &PreprocessedRequest) -> Vec<String> {
    let Some(items) = request
        .multi_modal_data
        .as_ref()
        .and_then(|media| media.get("image_url"))
    else {
        return Vec::new();
    };

    let mut keys = Vec::with_capacity(items.len());
    for item in items {
        match item {
            MultimodalData::Url(url) => keys.push(multimodal_cache_key_from_url(url.as_str())),
            MultimodalData::RawUrl(url) => keys.push(multimodal_cache_key_from_url(url)),
            MultimodalData::Decoded(_) => {}
            // Opaque UUIDs are not content-derived routing keys. UUID-only
            // reuse intentionally relies on text-prefix routing and affinity
            // to the worker that owns the processor/embedding cache entry.
            MultimodalData::UuidOnly(_) => {}
        }
    }
    keys.sort();
    keys.dedup();
    keys
}

type LlmPushRouter = PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>;

#[derive(Clone)]
pub struct PreprocessedRouting {
    backend_engine:
        ServiceEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>>,
    prefill_router: Arc<PrefillRouter>,
    encoder_router: Arc<EncoderRouter>,
}

pub struct PreparedEngine {
    pub service_name: String,
    pub engine: OpenAIChatCompletionsStreamingEngine,
    pub inspect_template: bool,
    pub request_template: Option<RequestTemplate>,
}

async fn wait_for_min_initial_workers(
    client: &Client,
    min_initial_workers: usize,
) -> anyhow::Result<()> {
    if min_initial_workers == 0 {
        return Ok(());
    }

    if min_initial_workers == 1 {
        client.wait_for_instances().await?;
        return Ok(());
    }

    let mut watcher = client.instance_avail_watcher();
    loop {
        let available = watcher.borrow_and_update().len();
        if available >= min_initial_workers {
            return Ok(());
        }

        tokio::time::timeout(Duration::from_secs(120), watcher.changed())
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "timed out waiting for {} initial workers for endpoint {}",
                    min_initial_workers,
                    client.endpoint.id()
                )
            })?
            .map_err(|_| {
                anyhow::anyhow!(
                    "instance watcher closed before {} workers appeared for endpoint {}",
                    min_initial_workers,
                    client.endpoint.id()
                )
            })?;
    }
}

fn router_client(
    client: &Client,
    router_mode: RouterMode,
    chooser: Option<&Arc<KvRouter>>,
) -> anyhow::Result<Client> {
    if router_mode == RouterMode::KV {
        let Some(chooser) = chooser else {
            anyhow::bail!("RouterMode::KV requires KVRouter to not be null");
        };
        Ok(chooser.client().clone())
    } else {
        Ok(client.clone())
    }
}

/// LoRA-aware routing is only implemented for the KV, Random, and RoundRobin modes. `Direct`
/// dispatches to a caller-chosen worker and bypasses both the LoRA filter and non-KV load
/// tracking; the advanced load-based modes (`PowerOfTwoChoices`, `LeastLoaded`,
/// `DeviceAwareWeighted`) have no 2-stage LoRA filtering. Reject those combinations so a
/// misconfiguration fails fast — at startup, before the initial-worker wait — instead of
/// silently mis-routing adapter traffic to a worker without the adapter.
fn validate_router_mode_for_lora(
    router_mode: RouterMode,
    lora_enabled: bool,
    session_affinity_enabled: bool,
) -> anyhow::Result<()> {
    if !lora_enabled {
        return Ok(());
    }
    if session_affinity_enabled
        && matches!(router_mode, RouterMode::Random | RouterMode::RoundRobin)
    {
        anyhow::bail!(
            "session affinity is unsupported with DYN_LORA_ENABLED and {router_mode:?} routing"
        );
    }
    match router_mode {
        RouterMode::KV | RouterMode::Random | RouterMode::RoundRobin => Ok(()),
        RouterMode::Direct
        | RouterMode::PowerOfTwoChoices
        | RouterMode::LeastLoaded
        | RouterMode::DeviceAwareWeighted => anyhow::bail!(
            "LoRA serving (DYN_LORA_ENABLED) is not supported with router mode {router_mode:?}; \
             use KV, Random, or RoundRobin for LoRA-aware routing, or disable LoRA serving."
        ),
    }
}

fn preprocessed_backend_engine(
    router: LlmPushRouter,
    router_mode: RouterMode,
    chooser: Option<Arc<KvRouter>>,
    model_manager: &Arc<crate::discovery::ModelManager>,
    endpoint_id: &dynamo_runtime::protocols::EndpointId,
    affinity: Option<AffinityCoordinator>,
) -> anyhow::Result<ServiceEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>>>
{
    // Reject LoRA + unsupported-mode combinations up front (single source of truth, shared with
    // the fail-fast check in `build_preprocessed_routing`). After this, the Direct and advanced
    // arms below are only reached with LoRA serving disabled.
    validate_router_mode_for_lora(
        router_mode,
        model_manager.lora_enabled(),
        affinity.is_some(),
    )?;

    let engine: ServiceEngine<_, _> = match router_mode {
        RouterMode::Direct => Arc::new(SessionAffinityPushRouter::new_with_coordinator(
            router, affinity, true,
        )),
        RouterMode::Random | RouterMode::RoundRobin => {
            match model_manager.lora_filter_for(endpoint_id) {
                Some(lora_filter) => Arc::new(LoraFilteredRouter::new(
                    router,
                    lora_filter,
                    model_manager.lora_load_estimator_for(endpoint_id),
                    router_mode,
                )),
                None => Arc::new(SessionAffinityPushRouter::new_with_coordinator(
                    router, affinity, false,
                )),
            }
        }
        RouterMode::PowerOfTwoChoices
        | RouterMode::LeastLoaded
        | RouterMode::DeviceAwareWeighted => {
            Arc::new(SessionAffinityPushRouter::new_with_coordinator(
                router,
                affinity,
                router_mode.is_direct_routing(),
            ))
        }
        RouterMode::KV => {
            let Some(chooser) = chooser else {
                anyhow::bail!("RouterMode::KV requires KVRouter to not be null");
            };
            Arc::new(KvPushRouter::new_with_coordinator(
                router, chooser, affinity,
            ))
        }
    };

    Ok(engine)
}

#[allow(clippy::too_many_arguments)]
pub async fn build_preprocessed_routing(
    client: &Client,
    model_manager: Arc<crate::discovery::ModelManager>,
    router_mode: RouterMode,
    worker_monitor: Option<KvWorkerMonitor>,
    chooser: Option<Arc<KvRouter>>,
    prefill_chooser: Option<Arc<PrefillRouter>>,
    encoder_chooser: Option<Arc<EncoderRouter>>,
    enable_multimodal_cache_indexer: bool,
    session_affinity_ttl_secs: Option<u64>,
    non_cpu_to_cpu_ratio: usize,
) -> anyhow::Result<PreprocessedRouting> {
    // Fail fast on an unsupported LoRA + router-mode combination BEFORE waiting for the initial
    // worker set, so a misconfiguration surfaces immediately at startup rather than after the
    // (possibly long) DYN_ROUTER_MIN_INITIAL_WORKERS wait.
    validate_router_mode_for_lora(
        router_mode,
        model_manager.lora_enabled(),
        session_affinity_ttl_secs.is_some(),
    )?;
    let min_initial_workers = min_initial_workers_from_env()?;
    let router_client = router_client(client, router_mode, chooser.as_ref())?;

    wait_for_min_initial_workers(&router_client, min_initial_workers).await?;
    let endpoint_id = router_client.endpoint.id();

    let affinity = create_affinity_coordinator(
        session_affinity_ttl_secs.map(Duration::from_secs),
        router_client.clone(),
    )
    .await?;

    let embedding_cache_indexer = if enable_multimodal_cache_indexer
        && matches!(router_mode, RouterMode::DeviceAwareWeighted)
    {
        try_build_cache_indexer(&router_client.endpoint).await
    } else {
        None
    };
    let cache_key_extractor = embedding_cache_indexer.as_ref().map(|_| {
        Arc::new(preprocessed_multimodal_cache_keys)
            as MultimodalCacheKeyExtractor<PreprocessedRequest>
    });

    let monitor_arc =
        worker_monitor.map(|m| Arc::new(m) as Arc<dyn dynamo_runtime::pipeline::WorkerLoadMonitor>);

    let router = LlmPushRouter::from_client_with_state_and_ratio(
        router_client,
        router_mode,
        monitor_arc,
        embedding_cache_indexer,
        cache_key_extractor,
        non_cpu_to_cpu_ratio,
    )
    .await?;

    // Eagerly register router request metrics so they appear as zeros even in
    // non-KV modes (Direct, Random, RoundRobin) where KvPushRouter is never created.
    // In KV mode, KvPushRouter::new() also calls from_component() (idempotent via
    // OnceLock), which covers the standalone router path as well.
    RouterRequestMetrics::from_component(client.endpoint.component());

    let prefill_router = prefill_chooser.unwrap_or_else(|| {
        PrefillRouter::disabled(
            model_manager.clone(),
            router_mode,
            session_affinity_ttl_secs,
        )
    });
    let encoder_router = encoder_chooser.unwrap_or_else(EncoderRouter::disabled);

    let backend_engine = preprocessed_backend_engine(
        router,
        router_mode,
        chooser,
        &model_manager,
        &endpoint_id,
        affinity,
    )?;
    Ok(PreprocessedRouting {
        backend_engine,
        prefill_router,
        encoder_router,
    })
}

/// Turns an EngineConfig into an OpenAI chat-completions and completions supported StreamingEngine.
pub async fn prepare_engine(
    distributed_runtime: DistributedRuntime,
    engine_config: EngineConfig,
) -> anyhow::Result<PreparedEngine> {
    match engine_config {
        EngineConfig::Dynamic {
            model: local_model,
            prefill_load_estimator,
            ..
        } => {
            let model_manager = Arc::new(ModelManager::new());
            // Create metrics for migration tracking (not exposed via /metrics in Dynamic engine mode)
            let metrics = Arc::new(Metrics::new());
            let mut watcher = ModelWatcher::new(
                distributed_runtime.clone(),
                model_manager.clone(),
                local_model.router_config().clone(),
                local_model.migration_limit(),
                local_model.migration_max_seq_len(),
                None,
                prefill_load_estimator,
                metrics,
            );
            if !local_model.path().as_os_str().is_empty() {
                watcher.set_local_model_path(Some(local_model.path().to_path_buf()));
            }
            watcher.set_tokenizer_backend(local_model.runtime_config().tokenizer_backend);
            let watch_obj = Arc::new(watcher);
            let discovery = distributed_runtime.discovery();
            let discovery_stream = discovery
                .list_and_watch(
                    dynamo_runtime::discovery::DiscoveryQuery::AllModels,
                    Some(distributed_runtime.primary_token().clone()),
                )
                .await?;
            let inner_watch_obj = watch_obj.clone();
            let namespace_filter = NamespaceFilter::from_namespace_and_prefix(
                local_model.namespace(),
                local_model.namespace_prefix(),
            );
            let _watcher_task = tokio::spawn(async move {
                inner_watch_obj
                    .watch(discovery_stream, namespace_filter)
                    .await;
            });
            tracing::info!("Waiting for remote model..");

            // TODO: We use the first model to appear, usually we have only one
            // We should add slash commands to text input `/model <name>` to choose,
            // '/models` to list, and notifications when models are added / removed.

            let model_service_name = watch_obj.wait_for_chat_model().await;
            tracing::info!("Connected to {model_service_name}");
            // In disaggregated deployments the model may be listed before the prefill
            // router is fully activated, causing a transient ModelUnavailable. Retry
            // with a timeout so the startup path doesn't fail during this cold-start
            // window, but also doesn't hang indefinitely on misconfiguration.
            let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
            let engine = loop {
                match model_manager.get_chat_completions_engine(&model_service_name) {
                    Ok(engine) => break engine,
                    Err(crate::discovery::ModelManagerError::ModelUnavailable(_))
                        if tokio::time::Instant::now() < deadline =>
                    {
                        tracing::debug!(
                            model = %model_service_name,
                            "Model listed but not yet servable, waiting for prefill activation"
                        );
                        tokio::time::sleep(Duration::from_millis(500)).await;
                    }
                    Err(e) => return Err(e.into()),
                }
            };
            Ok(PreparedEngine {
                service_name: model_service_name,
                engine,
                inspect_template: false,
                request_template: local_model.request_template(),
            })
        }
        EngineConfig::InProcessText { engine, model, .. } => {
            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with engine pre-processing");
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            Ok(PreparedEngine {
                service_name,
                engine,
                inspect_template: false,
                request_template: model.request_template(),
            })
        }
        EngineConfig::InProcessTokens {
            engine: inner_engine,
            model,
            ..
        } => {
            let pipeline = build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(model.card(), inner_engine, model.card().tokenizer()?)
            .await?;

            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with Dynamo pre-processing");
            Ok(PreparedEngine {
                service_name,
                engine: pipeline,
                inspect_template: true,
                request_template: model.request_template(),
            })
        }
    }
}

pub async fn build_pipeline<Req, Resp>(
    card: &ModelDeploymentCard,
    engine: ExecutionContext,
    tokenizer: crate::tokenizers::Tokenizer,
) -> anyhow::Result<Arc<ServiceFrontend<SingleIn<Req>, ManyOut<Annotated<Resp>>>>>
where
    Req: Data,
    Resp: Data,
    OpenAIPreprocessor: Operator<
            Context<Req>,
            Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
            Context<PreprocessedRequest>,
            Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>,
        >,
{
    let frontend = ServiceFrontend::<SingleIn<Req>, ManyOut<Annotated<Resp>>>::new();
    let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(card)?;
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone())?
            .into_operator();
    let backend = Backend::from_tokenizer(tokenizer).into_operator();
    let engine = ServiceBackend::from_engine(engine);

    Ok(frontend
        .link(preprocessor.forward_edge())?
        .link(backend.forward_edge())?
        .link(engine)?
        .link(backend.backward_edge())?
        .link(preprocessor.backward_edge())?
        .link(frontend)?)
}

impl PreprocessedRouting {
    /// The normal way to build an inference pipeline. Connect this directly to HTTP layer.
    pub fn build_pipeline<Req, Resp>(
        &self,
        card: &ModelDeploymentCard,
        preprocessor: Arc<OpenAIPreprocessor>,
        tokenizer: crate::tokenizers::Tokenizer,
        migration_limit: u32,
        migration_max_seq_len: Option<u32>,
        metrics: Arc<Metrics>,
    ) -> anyhow::Result<ServiceEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>>>
    where
        Req: Data,
        Resp: Data,
        OpenAIPreprocessor: Operator<
                Context<Req>,
                Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
                Context<PreprocessedRequest>,
                Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>,
            >,
    {
        let frontend = SegmentSource::<SingleIn<Req>, ManyOut<Annotated<Resp>>>::new();
        let preprocessor_op = preprocessor.into_operator();
        let token_backend = Backend::from_tokenizer(tokenizer).into_operator();
        let migration = Migration::from_mdc(card, migration_limit, migration_max_seq_len, metrics)
            .into_operator_for::<BackendOutput>();
        let prefill_op = self.prefill_router.into_operator();
        let encoder_op = self.encoder_router.into_operator();
        let backend = ServiceBackend::from_engine(self.backend_engine.clone());

        let engine = frontend
            .link(preprocessor_op.forward_edge())?
            .link(migration.forward_edge())?
            .link(token_backend.forward_edge())?
            .link(encoder_op.forward_edge())?
            .link(prefill_op.forward_edge())?
            .link(backend)?
            .link(prefill_op.backward_edge())?
            .link(encoder_op.backward_edge())?
            .link(token_backend.backward_edge())?
            .link(migration.backward_edge())?
            .link(preprocessor_op.backward_edge())?
            .link(frontend)?;

        Ok(engine)
    }

    /// Bring your own pre/post processor. Used when frontend has `--dyn-chat-processor
    /// vllm|sglang`.
    pub fn build_preprocessed_pipeline(
        &self,
        card: &ModelDeploymentCard,
        migration_limit: u32,
        migration_max_seq_len: Option<u32>,
        metrics: Arc<Metrics>,
    ) -> anyhow::Result<
        ServiceEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>>,
    > {
        let frontend = SegmentSource::<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
        >::new();
        let migration = Migration::from_mdc(card, migration_limit, migration_max_seq_len, metrics)
            .into_operator_for::<LLMEngineOutput>();
        let prefill_op = self.prefill_router.into_operator();
        let encoder_op = self.encoder_router.into_operator();
        let backend = ServiceBackend::from_engine(self.backend_engine.clone());

        let engine = frontend
            .link(migration.forward_edge())?
            .link(encoder_op.forward_edge())?
            .link(prefill_op.forward_edge())?
            .link(backend)?
            .link(prefill_op.backward_edge())?
            .link(encoder_op.backward_edge())?
            .link(migration.backward_edge())?
            .link(frontend)?;

        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_router_mode_for_lora() {
        use RouterMode::*;
        let all = [
            Direct,
            KV,
            Random,
            RoundRobin,
            PowerOfTwoChoices,
            LeastLoaded,
            DeviceAwareWeighted,
        ];

        // LoRA disabled: every mode is allowed (the unmodified routing path).
        for m in all {
            assert!(
                validate_router_mode_for_lora(m, false, false).is_ok(),
                "{m:?} must be allowed when LoRA serving is disabled"
            );
        }

        // LoRA enabled: only the LoRA-aware modes are accepted.
        for m in [KV, Random, RoundRobin] {
            assert!(
                validate_router_mode_for_lora(m, true, false).is_ok(),
                "{m:?} must be supported for LoRA-aware routing"
            );
        }

        // LoRA enabled: Direct + the advanced load-based modes are rejected with a clear error.
        for m in [Direct, PowerOfTwoChoices, LeastLoaded, DeviceAwareWeighted] {
            let err = validate_router_mode_for_lora(m, true, false)
                .expect_err("must reject unsupported LoRA router mode")
                .to_string();
            assert!(
                err.contains("not supported with router mode"),
                "{m:?} rejection must explain the unsupported mode, got: {err}"
            );
        }

        let err = validate_router_mode_for_lora(Random, true, true)
            .expect_err("must reject session affinity with LoRA-aware non-KV routing");
        assert!(err.to_string().contains("session affinity"));
    }
}
