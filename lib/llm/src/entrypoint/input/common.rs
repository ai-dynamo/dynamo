// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;

use crate::{
    backend::{Backend, ExecutionContext},
    discovery::{ModelManager, ModelWatcher},
    engines::StreamingEngineAdapter,
    entrypoint::{self, EngineConfig},
    kv_router::{KvPushRouter, KvRouter, PrefillRouter},
    migration::Migration,
    model_card::{self, ModelDeploymentCard},
    preprocessor::{OpenAIPreprocessor, prompt::PromptFormatter},
    protocols::common::llm_backend::{BackendOutput, LLMEngineOutput, PreprocessedRequest},
    request_template::RequestTemplate,
    types::{
        Annotated,
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            OpenAIChatCompletionsStreamingEngine,
        },
    },
};

use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::Client,
    distributed::DistributedConfig,
    engine::{AsyncEngineStream, Data},
    pipeline::{
        Context, ManyOut, Operator, PushRouter, RouterMode, SegmentSource, ServiceBackend,
        ServiceEngine, ServiceFrontend, SingleIn, Source,
    },
};
use std::sync::Arc;

pub struct PreparedEngine {
    pub service_name: String,
    pub engine: OpenAIChatCompletionsStreamingEngine,
    pub inspect_template: bool,
    pub card: Option<ModelDeploymentCard>,
    pub request_template: Option<RequestTemplate>,
}

impl PreparedEngine {
    pub fn has_tokenizer(&self) -> bool {
        if let Some(card) = self.card.as_ref() {
            card.has_tokenizer()
        } else {
            false
        }
    }
}

/// Turns an EngineConfig into an OpenAI chat-completions and completions supported StreamingEngine.
pub async fn prepare_engine(
    runtime: Runtime,
    engine_config: EngineConfig,
) -> anyhow::Result<PreparedEngine> {
    match engine_config {
        EngineConfig::Dynamic(local_model) => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;

            let Some(etcd_client) = distributed_runtime.etcd_client() else {
                anyhow::bail!("Cannot be both static mode and run with dynamic discovery.");
            };
            let model_manager = Arc::new(ModelManager::new());
            let watch_obj = Arc::new(ModelWatcher::new(
                distributed_runtime,
                model_manager.clone(),
                dynamo_runtime::pipeline::RouterMode::RoundRobin,
                None,
                None,
            ));
            let models_watcher = etcd_client
                .kv_get_and_watch_prefix(model_card::ROOT_PATH)
                .await?;
            let (_prefix, _watcher, receiver) = models_watcher.dissolve();

            let inner_watch_obj = watch_obj.clone();
            let _watcher_task = tokio::spawn(async move {
                inner_watch_obj.watch(receiver, None).await;
            });
            tracing::info!("Waiting for remote model..");

            // TODO: We use the first model to appear, usually we have only one
            // We should add slash commands to text input `/model <name>` to choose,
            // '/models` to list, and notifications when models are added / removed.

            let model_service_name = watch_obj.wait_for_chat_model().await;
            tracing::info!("Connected to {model_service_name}");
            let engine = model_manager.get_chat_completions_engine(&model_service_name)?;
            Ok(PreparedEngine {
                service_name: model_service_name,
                engine,
                inspect_template: false,
                card: None,
                request_template: local_model.request_template(),
            })
        }
        EngineConfig::StaticRemote(local_model) => {
            // For now we only do ModelType.Backend
            // For batch/text we only do Chat Completions

            // The card should have been loaded at 'build' phase earlier
            let card = local_model.card();
            let router_mode = local_model.router_config().router_mode;

            let dst_config = DistributedConfig::from_settings(true);
            let distributed_runtime = DistributedRuntime::new(runtime, dst_config).await?;

            let endpoint_id = local_model.endpoint_id();
            let component = distributed_runtime
                .namespace(&endpoint_id.namespace)?
                .component(&endpoint_id.component)?;

            let client = component.endpoint(&endpoint_id.name).client().await?;

            let kv_chooser = if router_mode == RouterMode::KV {
                let model_manager = Arc::new(ModelManager::new());
                Some(
                    model_manager
                        .kv_chooser_for(
                            &component,
                            card.kv_cache_block_size,
                            Some(local_model.router_config().kv_router_config),
                        )
                        .await?,
                )
            } else {
                None
            };

            let hf_tokenizer = card.tokenizer_hf()?;
            let chat_engine = entrypoint::build_routed_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(
                card,
                &client,
                router_mode,
                None,
                kv_chooser.clone(),
                hf_tokenizer,
                None, // No prefill chooser in static mode
            )
            .await?;

            let service_name = local_model.service_name().to_string();
            tracing::info!("Static connecting to {service_name}");
            Ok(PreparedEngine {
                service_name,
                engine: chat_engine,
                inspect_template: false,
                request_template: local_model.request_template(),
                card: Some(local_model.into_card()),
            })
        }
        EngineConfig::StaticFull { engine, model, .. } => {
            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with engine pre-processing");
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            Ok(PreparedEngine {
                service_name,
                engine,
                inspect_template: false,
                request_template: model.request_template(),
                card: Some(model.into_card()),
            })
        }
        EngineConfig::StaticCore {
            engine: inner_engine,
            model,
            ..
        } => {
            let pipeline = build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(model.card(), inner_engine, model.card().tokenizer_hf()?)
            .await?;

            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with Dynamo pre-processing");
            Ok(PreparedEngine {
                service_name,
                engine: pipeline,
                inspect_template: true,
                request_template: model.request_template(),
                card: Some(model.into_card()),
            })
        }
    }
}

pub async fn build_pipeline<Req, Resp>(
    card: &ModelDeploymentCard,
    engine: ExecutionContext,
    hf_tokenizer: tokenizers::Tokenizer,
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
    let PromptFormatter::OAI(formatter) = PromptFormatter::from_mdc(card)?;
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, hf_tokenizer.clone())?
            .into_operator();
    let backend = Backend::from_tokenizer(hf_tokenizer).into_operator();
    let engine = ServiceBackend::from_engine(engine);

    Ok(frontend
        .link(preprocessor.forward_edge())?
        .link(backend.forward_edge())?
        .link(engine)?
        .link(backend.backward_edge())?
        .link(preprocessor.backward_edge())?
        .link(frontend)?)
}

pub async fn build_routed_pipeline<Req, Resp>(
    card: &ModelDeploymentCard,
    client: &Client,
    router_mode: RouterMode,
    busy_threshold: Option<f64>,
    chooser: Option<Arc<KvRouter>>,
    hf_tokenizer: tokenizers::Tokenizer,
    prefill_chooser: Option<Arc<PrefillRouter>>,
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
    let PromptFormatter::OAI(formatter) = PromptFormatter::from_mdc(card)?;
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, hf_tokenizer.clone())?;
    build_routed_pipeline_with_preprocessor(
        card,
        client,
        router_mode,
        busy_threshold,
        chooser,
        preprocessor,
        hf_tokenizer,
        prefill_chooser,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn build_routed_pipeline_with_preprocessor<Req, Resp>(
    card: &ModelDeploymentCard,
    client: &Client,
    router_mode: RouterMode,
    busy_threshold: Option<f64>,
    chooser: Option<Arc<KvRouter>>,
    preprocessor: Arc<OpenAIPreprocessor>,
    hf_tokenizer: tokenizers::Tokenizer,
    prefill_chooser: Option<Arc<PrefillRouter>>,
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
    let backend = Backend::from_tokenizer(hf_tokenizer).into_operator();
    let migration = Migration::from_mdc(card).into_operator();

    // Create worker monitor only if busy_threshold is set
    let worker_monitor = busy_threshold.map(|threshold| {
        Arc::new(crate::discovery::KvWorkerMonitor::new(
            Arc::new(client.clone()),
            threshold,
        )) as Arc<dyn dynamo_runtime::pipeline::WorkerLoadMonitor>
    });

    let router =
        PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
            client.clone(),
            router_mode,
            busy_threshold,
            worker_monitor,
        )
        .await?;

    let service_backend = match router_mode {
        RouterMode::Random | RouterMode::RoundRobin | RouterMode::Direct(_) => {
            ServiceBackend::from_engine(Arc::new(router))
        }
        RouterMode::KV => {
            let Some(chooser) = chooser else {
                anyhow::bail!("RouterMode::KV requires KVRouter to not be null");
            };
            let kv_push_router = KvPushRouter::new(router, chooser);
            ServiceBackend::from_engine(Arc::new(kv_push_router))
        }
    };

    // Use the provided prefill chooser, or create a disabled one if not provided
    let prefill_chooser = prefill_chooser.unwrap_or_else(|| PrefillRouter::disabled(router_mode));
    let prefill_op = prefill_chooser.into_operator();

    // Link with prefill chooser including backward edge for response flow
    let engine = frontend
        .link(preprocessor_op.forward_edge())?
        .link(backend.forward_edge())?
        .link(migration.forward_edge())?
        .link(prefill_op.forward_edge())?
        .link(service_backend)?
        .link(prefill_op.backward_edge())?
        .link(migration.backward_edge())?
        .link(backend.backward_edge())?
        .link(preprocessor_op.backward_edge())?
        .link(frontend)?;

    Ok(engine)
}
