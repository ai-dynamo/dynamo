// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;

use crate::{
    backend::{Backend, ExecutionContext},
    discovery::{KvWorkerMonitor, ModelManager, ModelWatcher},
    engines::StreamingEngineAdapter,
    entrypoint::{EngineConfig, RouterConfig},
    kv_router::{KvPushRouter, KvRouter, PrefillRouter},
    migration::Migration,
    model_card::ModelDeploymentCard,
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
    DistributedRuntime,
    component::Client,
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
    distributed_runtime: DistributedRuntime,
    engine_config: EngineConfig,
) -> anyhow::Result<PreparedEngine> {
    match engine_config {
        EngineConfig::Dynamic(local_model) => {
            let model_manager = Arc::new(ModelManager::new());
            let watch_obj = Arc::new(ModelWatcher::new(
                distributed_runtime.clone(),
                model_manager.clone(),
                RouterConfig::default(),
            ));
            let discovery = distributed_runtime.discovery();
            let discovery_stream = discovery
                .list_and_watch(
                    dynamo_runtime::discovery::DiscoveryQuery::AllModels,
                    Some(distributed_runtime.primary_token().clone()),
                )
                .await?;
            let inner_watch_obj = watch_obj.clone();
            let _watcher_task = tokio::spawn(async move {
                inner_watch_obj.watch(discovery_stream, None).await;
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
        EngineConfig::InProcessText { engine, model, .. } => {
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
        EngineConfig::InProcessTokens {
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

#[allow(clippy::too_many_arguments)]
pub async fn build_routed_pipeline<Req, Resp>(
    card: &ModelDeploymentCard,
    client: &Client,
    router_mode: RouterMode,
    worker_monitor: Option<KvWorkerMonitor>,
    chooser: Option<Arc<KvRouter>>,
    hf_tokenizer: tokenizers::Tokenizer,
    prefill_chooser: Option<Arc<PrefillRouter>>,
    enforce_disagg: bool,
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
        worker_monitor,
        chooser,
        preprocessor,
        hf_tokenizer,
        prefill_chooser,
        enforce_disagg,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn build_routed_pipeline_with_preprocessor<Req, Resp>(
    card: &ModelDeploymentCard,
    client: &Client,
    router_mode: RouterMode,
    worker_monitor: Option<KvWorkerMonitor>,
    chooser: Option<Arc<KvRouter>>,
    preprocessor: Arc<OpenAIPreprocessor>,
    hf_tokenizer: tokenizers::Tokenizer,
    prefill_chooser: Option<Arc<PrefillRouter>>,
    enforce_disagg: bool,
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

    // For KV routing, use the client from the chooser to ensure shared state
    let router_client = if router_mode == RouterMode::KV {
        let Some(ref chooser) = chooser else {
            anyhow::bail!("RouterMode::KV requires KVRouter to not be null");
        };
        chooser.client().clone()
    } else {
        client.clone()
    };

    // Get threshold value and wrap monitor for PushRouter
    let threshold_value = worker_monitor.as_ref().map(|m| m.threshold());
    let monitor_arc =
        worker_monitor.map(|m| Arc::new(m) as Arc<dyn dynamo_runtime::pipeline::WorkerLoadMonitor>);

    let router =
        PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
            router_client,
            router_mode,
            threshold_value,
            monitor_arc,
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
    let prefill_chooser =
        prefill_chooser.unwrap_or_else(|| PrefillRouter::disabled(router_mode, enforce_disagg));
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
