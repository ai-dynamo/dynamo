// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::time::Duration;

use anyhow::Context as _;
use dynamo_renderer::PromptFormatter;

use crate::{
    backend::{Backend, ExecutionContext},
    discovery::{ModelManager, ModelWatcher},
    engines::StreamingEngineAdapter,
    entrypoint::EngineConfig,
    http::service::metrics::Metrics,
    kv_router::indexer::{preprocessed_multimodal_cache_keys, try_build_cache_indexer},
    kv_router::{EncoderRouter, KvRouter, PrefillRouter, RoutingHost, RoutingLoadContext},
    migration::Migration,
    model_card::ModelDeploymentCard,
    namespace::NamespaceFilter,
    preprocessor::{OpenAIPreprocessor, prompt::prompt_formatter_from_mdc},
    protocols::common::llm_backend::{BackendOutput, LLMEngineOutput, PreprocessedRequest},
    request_template::RequestTemplate,
    session_affinity::{SessionAffinityMode, create_affinity_coordinator},
    types::{
        Annotated,
        openai::{
            chat_completions::{
                NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
                OpenAIChatCompletionsStreamingEngine,
            },
            completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
        },
    },
};

use dynamo_kv_router::config::min_initial_workers_from_env;
use dynamo_lora::policy::validate_router_mode_for_lora;
use dynamo_runtime::{
    DistributedRuntime,
    component::Client,
    engine::{AsyncEngineStream, Data},
    pipeline::{
        Context, ManyOut, MultimodalCacheKeyExtractor, Operator, RouterMode, SegmentSource,
        ServiceBackend, ServiceEngine, ServiceFrontend, SingleIn, Source,
    },
};
use std::sync::Arc;

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

#[allow(clippy::too_many_arguments)]
pub async fn build_preprocessed_routing(
    client: &Client,
    model_manager: Arc<crate::discovery::ModelManager>,
    router_mode: RouterMode,
    load_context: Arc<RoutingLoadContext>,
    chooser: Option<Arc<KvRouter>>,
    prefill_chooser: Option<Arc<PrefillRouter>>,
    encoder_chooser: Option<Arc<EncoderRouter>>,
    enable_multimodal_cache_indexer: bool,
    session_affinity_ttl_secs: Option<u64>,
) -> anyhow::Result<PreprocessedRouting> {
    build_preprocessed_routing_with_session_affinity_mode(
        client,
        model_manager,
        router_mode,
        load_context,
        chooser,
        prefill_chooser,
        encoder_chooser,
        enable_multimodal_cache_indexer,
        session_affinity_ttl_secs,
        SessionAffinityMode::Hard,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_preprocessed_routing_with_session_affinity_mode(
    client: &Client,
    model_manager: Arc<crate::discovery::ModelManager>,
    router_mode: RouterMode,
    load_context: Arc<RoutingLoadContext>,
    chooser: Option<Arc<KvRouter>>,
    prefill_chooser: Option<Arc<PrefillRouter>>,
    encoder_chooser: Option<Arc<EncoderRouter>>,
    enable_multimodal_cache_indexer: bool,
    session_affinity_ttl_secs: Option<u64>,
    session_affinity_mode: SessionAffinityMode,
) -> anyhow::Result<PreprocessedRouting> {
    // Fail fast on an unsupported LoRA + router-mode combination BEFORE waiting for the initial
    // worker set, so a misconfiguration surfaces immediately at startup rather than after the
    // (possibly long) DYN_ROUTER_MIN_INITIAL_WORKERS wait.
    validate_router_mode_for_lora(
        router_mode,
        model_manager.lora_enabled(),
        session_affinity_ttl_secs.is_some(),
    )?;
    let runtime_lora_enabled = crate::lora::runtime::RuntimeLoraConfig::from_env()?.enabled;
    let min_initial_workers = min_initial_workers_from_env()?;
    let router_client = router_client(client, router_mode, chooser.as_ref())?;

    wait_for_min_initial_workers(&router_client, min_initial_workers).await?;
    let endpoint_id = router_client.endpoint.id();
    let runtime_configs = if runtime_lora_enabled && !router_mode.is_kv_routing() {
        Some(
            model_manager
                .get_or_create_runtime_config_watcher(&router_client.endpoint)
                .await?,
        )
    } else {
        None
    };

    let ttl = session_affinity_ttl_secs.map(Duration::from_secs);
    let affinity = match (ttl, chooser.as_ref()) {
        (Some(ttl), Some(chooser)) => {
            let affinity = chooser.affinity_coordinator(ttl, session_affinity_mode)?;
            affinity.enable_replica_sync(router_client.clone()).await?;
            Some(affinity)
        }
        _ => create_affinity_coordinator(ttl, session_affinity_mode, router_client.clone()).await?,
    };

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

    let prefill_router = prefill_chooser.unwrap_or_else(|| {
        PrefillRouter::disabled_with_session_affinity_mode(
            model_manager.clone(),
            router_mode,
            session_affinity_ttl_secs,
            session_affinity_mode,
        )
    });
    let encoder_router = encoder_chooser.unwrap_or_else(EncoderRouter::disabled);

    // LoRA-aware selection lives in the host only for the built-in modes; under
    // KV routing the KV router applies the adapter filter itself.
    let candidate_policy = (router_mode != RouterMode::KV)
        .then(|| {
            model_manager
                .lora_filter_for(&endpoint_id)
                .map(|filter| (filter, model_manager.lora_load_estimator_for(&endpoint_id)))
        })
        .flatten()
        .map(|(filter, estimator)| {
            crate::lora_routing::LoraHostedPolicy::hosted(
                filter,
                estimator,
                router_mode,
                runtime_configs,
            )
        })
        .transpose()?;
    let routing_host = Arc::new(
        RoutingHost::builder(router_client, router_mode)
            .kv_router(chooser)
            .load_context(load_context)
            .affinity_coordinator(affinity)
            .candidate_policy(candidate_policy)
            .multimodal_cache(embedding_cache_indexer.zip(cache_key_extractor))
            .build()
            .await?,
    );
    if router_mode.is_kv_routing() && prefill_router.conditional_disagg_enabled() {
        prefill_router
            .set_decode_routing_host(routing_host.clone())
            .context("install conditional-disagg decode RoutingHost")?;
    }
    let backend_engine: ServiceEngine<_, _> = routing_host;
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
            watcher.set_tokenizer_fallback_enabled(
                local_model.runtime_config().tokenizer_fallback_enabled,
            );
            let watch_obj = Arc::new(watcher);
            let namespace_filter = NamespaceFilter::from_namespace_and_prefix(
                local_model.namespace(),
                local_model.namespace_prefix(),
            );
            let discovery = distributed_runtime.discovery();
            let discovery_stream = discovery
                .list_and_watch(
                    namespace_filter.model_query(),
                    Some(distributed_runtime.primary_token().clone()),
                )
                .await?;
            let inner_watch_obj = watch_obj.clone();
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
    let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(card)?;
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone())?;
    link_pipeline(engine, tokenizer, preprocessor)
}

type InputFrontend<Req, Resp> = Arc<ServiceFrontend<SingleIn<Req>, ManyOut<Annotated<Resp>>>>;

pub(crate) fn build_completions_pipeline(
    card: &ModelDeploymentCard,
    engine: ExecutionContext,
    tokenizer: crate::tokenizers::Tokenizer,
) -> anyhow::Result<InputFrontend<NvCreateCompletionRequest, NvCreateCompletionResponse>> {
    let preprocessor = OpenAIPreprocessor::new_for_completions(card.clone(), tokenizer.clone())?;
    link_pipeline(engine, tokenizer, preprocessor)
}

fn link_pipeline<Req, Resp>(
    engine: ExecutionContext,
    tokenizer: crate::tokenizers::Tokenizer,
    preprocessor: Arc<OpenAIPreprocessor>,
) -> anyhow::Result<InputFrontend<Req, Resp>>
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
    let preprocessor = preprocessor.into_operator();
    let backend = Backend::from_tokenizer(tokenizer).into_operator();
    let engine = ServiceBackend::from_engine(engine);

    Ok(frontend
        .link(preprocessor.forward_edge())?
        .link(backend.forward_edge())?
        .link(engine)?
        .link(backend.backward_edge())?
        .link(preprocessor.backward_edge())?
        .link_terminal(frontend)?)
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
            .link_terminal(frontend)?;

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
        self.link_preprocessed(Migration::from_mdc(
            card,
            migration_limit,
            migration_max_seq_len,
            metrics,
        ))
    }

    /// The engine-native Generate pipeline: requests migrate only through
    /// their `/generate` format's continuation.
    pub fn build_generate_pipeline(
        &self,
        card: &ModelDeploymentCard,
        migration_limit: u32,
        migration_max_seq_len: Option<u32>,
        metrics: Arc<Metrics>,
    ) -> anyhow::Result<
        ServiceEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>>,
    > {
        self.link_preprocessed(Migration::for_generate(
            card,
            migration_limit,
            migration_max_seq_len,
            metrics,
        ))
    }

    fn link_preprocessed(
        &self,
        migration: Arc<Migration>,
    ) -> anyhow::Result<
        ServiceEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>>,
    > {
        let frontend = SegmentSource::<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
        >::new();
        let migration = migration.into_operator_for::<LLMEngineOutput>();
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
            .link_terminal(frontend)?;

        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn invalid_runtime_lora_configuration_fails_before_waiting_for_workers() {
        // Invalid process-wide flags must not affect concurrent frontend startup tests.
        const TEST: &str = concat!(
            module_path!(),
            "::invalid_runtime_lora_configuration_fails_before_waiting_for_workers"
        );
        let test_name = TEST.split_once("::").unwrap().1;
        if std::env::var("DYNAMO_INVALID_LORA_CONFIG_TEST").as_deref() != Ok(test_name) {
            let mut child = tokio::process::Command::new(std::env::current_exe().unwrap());
            child
                .args(["--exact", test_name, "--nocapture"])
                .env("DYNAMO_INVALID_LORA_CONFIG_TEST", test_name)
                .env("DYN_LORA_ENABLED", "false")
                .env("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
                .env("DYN_ROUTER_MIN_INITIAL_WORKERS", "1")
                .kill_on_drop(true);
            let output = tokio::time::timeout(Duration::from_secs(30), child.output())
                .await
                .expect("invalid configuration subprocess must finish within its deadline")
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("test result: ok. 1 passed;"),
                "{stdout}\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let runtime = dynamo_runtime::Runtime::from_current().unwrap();
        let distributed = DistributedRuntime::new(
            runtime.clone(),
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let client = distributed
            .namespace("invalid-runtime-lora".to_string())
            .unwrap()
            .component("workers")
            .unwrap()
            .endpoint("generate")
            .client()
            .await
            .unwrap();
        let load_context = RoutingLoadContext::start(
            client.clone(),
            crate::kv_router::RouterLoadSource::Aggregated,
            crate::discovery::LoadThresholdHandle::new(Default::default()),
            &distributed.child_token(),
            None,
        )
        .await
        .unwrap();
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            build_preprocessed_routing(
                &client,
                Arc::new(ModelManager::new()),
                RouterMode::RoundRobin,
                load_context,
                None,
                None,
                None,
                false,
                None,
            ),
        )
        .await
        .expect("invalid configuration must fail before the worker wait");
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("requires DYN_LORA_ENABLED")
        );
        runtime.shutdown();
    }

    #[test]
    fn test_validate_router_mode_for_lora() {
        use RouterMode::*;
        let all = [
            Direct,
            KV,
            Random,
            RoundRobin,
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
        for m in [Direct, LeastLoaded, DeviceAwareWeighted] {
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
