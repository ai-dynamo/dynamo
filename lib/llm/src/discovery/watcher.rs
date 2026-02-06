// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use tokio::sync::Notify;
use tokio::sync::mpsc::Sender;

use anyhow::Context as _;
use dashmap::DashSet;
use futures::StreamExt;

use dynamo_runtime::{
    DistributedRuntime,
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, DiscoveryStream,
        ModelCardInstanceId,
    },
    pipeline::{
        ManyOut, Operator, RouterMode, SegmentSource, ServiceBackend, SingleIn, Source,
        network::egress::push_router::PushRouter,
    },
    protocols::{EndpointId, annotated::Annotated},
};

use crate::{
    backend::Backend,
    discovery::{WORKER_TYPE_DECODE, WorkerSet},
    entrypoint::{self, ChatEngineFactoryCallback, RouterConfig},
    http::service::metrics::Metrics,
    kv_router::PrefillRouter,
    model_card::ModelDeploymentCard,
    model_type::{ModelInput, ModelType},
    preprocessor::{OpenAIPreprocessor, PreprocessedEmbeddingRequest, prompt::PromptFormatter},
    protocols::{
        common::llm_backend::EmbeddingsEngineOutput,
        openai::{
            chat_completions::{
                NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            },
            completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
            embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
            images::{NvCreateImageRequest, NvImagesResponse},
            videos::{NvCreateVideoRequest, NvVideosResponse},
        },
        tensor::{NvCreateTensorRequest, NvCreateTensorResponse},
    },
};

use super::ModelManager;
use crate::namespace::NamespaceFilter;

#[derive(Debug, Clone)]
pub enum ModelUpdate {
    Added(ModelDeploymentCard),
    Removed(ModelDeploymentCard),
}

pub struct ModelWatcher {
    manager: Arc<ModelManager>,
    drt: DistributedRuntime,
    router_config: RouterConfig,
    migration_limit: u32,
    notify_on_model: Notify,
    model_update_tx: Option<Sender<ModelUpdate>>,
    chat_engine_factory: Option<ChatEngineFactoryCallback>,
    metrics: Arc<Metrics>,
    /// Guards against concurrent pipeline construction for the same (model, namespace).
    registering_worker_sets: DashSet<String>,
}

const ALL_MODEL_TYPES: &[ModelType] = &[
    ModelType::Chat,
    ModelType::Completions,
    ModelType::Embedding,
    ModelType::Images,
    ModelType::Audios,
    ModelType::Videos,
    ModelType::TensorBased,
    ModelType::Prefill,
];

impl ModelWatcher {
    pub fn new(
        runtime: DistributedRuntime,
        model_manager: Arc<ModelManager>,
        router_config: RouterConfig,
        migration_limit: u32,
        chat_engine_factory: Option<ChatEngineFactoryCallback>,
        metrics: Arc<Metrics>,
    ) -> ModelWatcher {
        Self {
            manager: model_manager,
            drt: runtime,
            router_config,
            migration_limit,
            notify_on_model: Notify::new(),
            model_update_tx: None,
            chat_engine_factory,
            metrics,
            registering_worker_sets: DashSet::new(),
        }
    }

    pub fn set_notify_on_model_update(&mut self, tx: Sender<ModelUpdate>) {
        self.model_update_tx = Some(tx);
    }

    /// Wait until we have at least one chat completions model and return it's name.
    pub async fn wait_for_chat_model(&self) -> String {
        // Loop in case it gets added and immediately deleted
        loop {
            if let Some(model_name) = self.manager.list_chat_completions_models().first() {
                return model_name.to_owned();
            }
            self.notify_on_model.notified().await
        }
    }

    /// Common watch logic with optional namespace filtering
    pub async fn watch(
        &self,
        mut discovery_stream: DiscoveryStream,
        namespace_filter: NamespaceFilter,
    ) {
        while let Some(result) = discovery_stream.next().await {
            let event = match result {
                Ok(event) => event,
                Err(err) => {
                    tracing::error!(%err, "Error in discovery stream");
                    continue;
                }
            };

            match event {
                DiscoveryEvent::Added(instance) => {
                    // Extract ModelCardInstanceId and card from the discovery instance
                    let (mcid, mut card) = match &instance {
                        DiscoveryInstance::Model {
                            namespace,
                            component,
                            endpoint,
                            instance_id,
                            model_suffix,
                            ..
                        } => {
                            let mcid = ModelCardInstanceId {
                                namespace: namespace.clone(),
                                component: component.clone(),
                                endpoint: endpoint.clone(),
                                instance_id: *instance_id,
                                model_suffix: model_suffix.clone(),
                            };

                            match instance.deserialize_model::<ModelDeploymentCard>() {
                                Ok(card) => (mcid, card),
                                Err(err) => {
                                    tracing::error!(%err, instance_id, "Failed to deserialize model card");
                                    continue;
                                }
                            }
                        }
                        _ => {
                            tracing::error!(
                                "Unexpected discovery instance type (expected ModelCard)"
                            );
                            continue;
                        }
                    };

                    // Filter by namespace using the configured filter
                    if !namespace_filter.matches(&mcid.namespace) {
                        tracing::debug!(
                            model_namespace = mcid.namespace,
                            namespace_filter = ?namespace_filter,
                            "Skipping model due to namespace filter"
                        );
                        continue;
                    }

                    // If we already have a WorkerSet for this (model, namespace) and the
                    // checksums don't match, reject the new worker. Different namespaces
                    // may legitimately have different checksums (rolling updates).
                    let can_add = self.manager.is_valid_checksum(
                        card.model_type,
                        card.name(),
                        &mcid.namespace,
                        card.mdcsum(),
                    );
                    if can_add.is_some_and(|is_valid| !is_valid) {
                        tracing::error!(
                            model_name = card.name(),
                            namespace = mcid.namespace,
                            "Checksum for new worker does not match existing WorkerSet in same namespace."
                        );

                        // TODO: mark that instance down in clients
                        continue;
                    }

                    match self.handle_put(&mcid, &mut card).await {
                        Ok(()) => {
                            tracing::info!(
                                model_name = card.name(),
                                namespace = mcid.namespace,
                                "added model"
                            );
                            self.notify_on_model.notify_waiters();
                        }
                        Err(err) => {
                            tracing::error!(
                                model_name = card.name(),
                                namespace = mcid.namespace,
                                error = format!("{err:#}"),
                                "Error adding model from discovery",
                            );
                        }
                    }
                }
                DiscoveryEvent::Removed(id) => {
                    // Extract ModelCardInstanceId from the removal event
                    let model_card_instance_id = match &id {
                        DiscoveryInstanceId::Model(mcid) => mcid,
                        DiscoveryInstanceId::Endpoint(_) | DiscoveryInstanceId::EventChannel(_) => {
                            tracing::error!(
                                "Unexpected discovery instance type in removal (expected Model)"
                            );
                            continue;
                        }
                    };

                    match self
                        .handle_delete(model_card_instance_id, &namespace_filter)
                        .await
                    {
                        Ok(Some(model_name)) => {
                            tracing::info!(model_name, "removed model");
                        }
                        Ok(None) => {
                            // There are other instances running this model, nothing to do
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "error removing model");
                        }
                    }
                }
            }
        }
    }

    /// Handle a worker removal. Cleans up per-namespace WorkerSets and the Model itself
    /// when no instances remain. Returns the model name if the entire Model was removed.
    async fn handle_delete(
        &self,
        mcid: &ModelCardInstanceId,
        namespace_filter: &NamespaceFilter,
    ) -> anyhow::Result<Option<String>> {
        let key = mcid.to_path();
        let card = match self.manager.remove_model_card(&key) {
            Some(card) => card,
            None => {
                anyhow::bail!("Missing ModelDeploymentCard for {}", key);
            }
        };
        let model_name = card.name().to_string();
        let worker_namespace = &mcid.namespace;

        // Decrement worker count for this namespace's WorkerSet
        if let Some(model) = self.manager.get_model(&model_name) {
            if let Some(ws) = model.get_worker_set(worker_namespace) {
                let remaining = ws.decrement_workers();
                tracing::debug!(
                    model_name,
                    namespace = %worker_namespace,
                    remaining_workers = remaining,
                    "Decremented worker count for WorkerSet"
                );
            }
        }

        // Query discovery for all remaining instances of this model
        let active_instances = self
            .cards_for_model_with_endpoints(&model_name, namespace_filter)
            .await
            .with_context(|| model_name.clone())?;

        // Check if any instances remain in this worker's specific namespace
        let ns_has_instances = active_instances
            .iter()
            .any(|(eid, _)| eid.namespace == *worker_namespace);

        if !ns_has_instances {
            // No more workers in this namespace — remove its WorkerSet and associated state
            if let Some(_removed_ws) = self.manager.remove_worker_set(&model_name, worker_namespace) {
                self.manager.remove_prefill_activator(&model_name, worker_namespace);
                tracing::info!(
                    model_name,
                    namespace = %worker_namespace,
                    "Removed WorkerSet (no remaining instances in namespace)"
                );
            }
        }

        // Check if the Model still has instances in any namespace
        if !active_instances.is_empty() {
            tracing::debug!(
                model_name,
                active_instance_count = active_instances.len(),
                "Model has other active instances in other namespaces"
            );
            return Ok(None);
        }

        // No instances remain anywhere — remove the entire Model
        let _ = self.manager.remove_model(&model_name);

        if let Some(tx) = &self.model_update_tx {
            for model_type in ALL_MODEL_TYPES {
                if card.model_type.intersects(*model_type) {
                    tx.send(ModelUpdate::Removed(card.clone())).await.ok();
                }
            }
        }

        Ok(Some(model_name))
    }

    // Handles a PUT event from store, this usually means adding a new model to the list of served
    // models.
    async fn handle_put(
        &self,
        mcid: &ModelCardInstanceId,
        card: &mut ModelDeploymentCard,
    ) -> anyhow::Result<()> {
        // Check if this specific (model, namespace) WorkerSet already exists.
        // If so, this is just another worker joining an existing set — no pipeline build needed.
        let model_name = card.name().to_string();
        let namespace = mcid.namespace.clone();

        if let Some(model) = self.manager.get_model(&model_name) {
            if let Some(ws) = model.get_worker_set(&namespace) {
                self.manager
                    .save_model_card(&mcid.to_path(), card.clone())?;
                let count = ws.increment_workers();
                tracing::debug!(
                    model_name = card.name(),
                    namespace = namespace,
                    worker_count = count,
                    "Worker joined existing WorkerSet, skipping pipeline build"
                );
                return Ok(());
            }
        }

        // Guard against concurrent pipeline construction for the same (model, namespace)
        let registration_key = ModelManager::model_namespace_key(&model_name, &namespace);
        if !self.registering_worker_sets.insert(registration_key.clone()) {
            self.manager
                .save_model_card(&mcid.to_path(), card.clone())?;
            tracing::debug!(
                model_name = card.name(),
                namespace = namespace,
                "WorkerSet registration in progress, skipping"
            );
            return Ok(());
        }

        let result = self.do_worker_set_registration(mcid, card).await;

        // Always remove from registering set
        self.registering_worker_sets.remove(&registration_key);

        result
    }

    /// Build a complete WorkerSet with all engines for this (model, namespace)
    /// and add it to the Model.
    async fn do_worker_set_registration(
        &self,
        mcid: &ModelCardInstanceId,
        card: &mut ModelDeploymentCard,
    ) -> anyhow::Result<()> {
        card.download_config().await?;

        let component = self
            .drt
            .namespace(&mcid.namespace)?
            .component(&mcid.component)?;
        let endpoint = component.endpoint(&mcid.endpoint);
        let client = endpoint.client().await?;
        tracing::debug!(model_name = card.name(), namespace = mcid.namespace, "building worker set pipeline");
        self.manager
            .save_model_card(&mcid.to_path(), card.clone())?;

        if let Some(tx) = &self.model_update_tx {
            tx.send(ModelUpdate::Added(card.clone())).await.ok();
        }

        let checksum = card.mdcsum();
        let namespace = mcid.namespace.clone();

        // Build the WorkerSet with all applicable engines
        let mut worker_set = WorkerSet::new(
            namespace.clone(),
            checksum.to_string(),
            card.clone(),
        );

        if card.model_input == ModelInput::Tokens
            && (card.model_type.supports_chat() || card.model_type.supports_completions())
        {
            // Case 1: Tokens + (Chat OR Completions OR Both)
            let endpoint = component.endpoint(&mcid.endpoint);
            // Create the KV router whenever any local routed pipeline will be built.
            // The chat factory builds its own router, but completions currently always
            // uses the local routed pipeline and therefore still needs a chooser.
            let needs_local_chat_pipeline =
                card.model_type.supports_chat() && self.chat_engine_factory.is_none();
            let needs_local_completions_pipeline = card.model_type.supports_completions();
            let kv_chooser = if self.router_config.router_mode == RouterMode::KV
                && (needs_local_chat_pipeline || needs_local_completions_pipeline)
            {
                Some(
                    self.manager
                        .kv_chooser_for(
                            &endpoint,
                            card.kv_cache_block_size,
                            Some(self.router_config.kv_router_config),
                            WORKER_TYPE_DECODE,
                        )
                        .await?,
                )
            } else {
                None
            };

            let tokenizer_hf = card.tokenizer_hf().context("tokenizer_hf")?;

            let model_name = card.name().to_string();
            let prefill_chooser = self
                .manager
                .register_prefill_router(model_name.clone(), &namespace)
                .map(|rx| {
                    let mut prefill_config = self.router_config.kv_router_config;
                    prefill_config.router_track_active_blocks = false;

                    PrefillRouter::new(
                        rx,
                        self.manager.clone(),
                        self.router_config.router_mode,
                        card.kv_cache_block_size,
                        Some(prefill_config),
                        self.router_config.enforce_disagg,
                        model_name.clone(),
                    )
                });

            let worker_monitor = Some(self.manager.get_or_create_worker_monitor(
                card.name(),
                client.clone(),
                self.router_config.load_threshold_config.clone(),
            ));

            // Store KV router and worker monitor on the WorkerSet
            worker_set.kv_router = kv_chooser.clone();
            worker_set.worker_monitor = worker_monitor.clone();

            if card.model_type.supports_chat() {
                let factory_engine = if let Some(ref factory) = self.chat_engine_factory {
                    match factory(mcid.clone(), card.clone()).await {
                        Ok(engine) => Some(engine),
                        Err(err) => return Err(err).context("python chat_engine_factory"),
                    }
                } else {
                    None
                };

                let chat_engine = if let Some(engine) = factory_engine {
                    engine
                } else {
                    entrypoint::build_routed_pipeline::<
                        NvCreateChatCompletionRequest,
                        NvCreateChatCompletionStreamResponse,
                    >(
                        card,
                        &client,
                        self.manager.clone(),
                        self.router_config.router_mode,
                        worker_monitor.clone(),
                        kv_chooser.clone(),
                        tokenizer_hf.clone(),
                        prefill_chooser.clone(),
                        self.router_config.enforce_disagg,
                        self.migration_limit,
                        self.metrics.clone(),
                    )
                    .await
                    .context("build_routed_pipeline")?
                };
                worker_set.chat_engine = Some(chat_engine);
                tracing::info!("Chat completions is ready");
            }

            // Add completions engine only if the model supports completions.
            if card.model_type.supports_completions() {
                let formatter = PromptFormatter::no_op();
                let PromptFormatter::OAI(formatter) = formatter;
                let preprocessor = OpenAIPreprocessor::new_with_parts(
                    card.clone(),
                    formatter,
                    tokenizer_hf.clone(),
                )
                .context("OpenAIPreprocessor::new_with_parts")?;
                let completions_engine = entrypoint::build_routed_pipeline_with_preprocessor::<
                    NvCreateCompletionRequest,
                    NvCreateCompletionResponse,
                >(
                    card,
                    &client,
                    self.manager.clone(),
                    self.router_config.router_mode,
                    worker_monitor,
                    kv_chooser,
                    preprocessor,
                    tokenizer_hf,
                    prefill_chooser,
                    self.router_config.enforce_disagg,
                    self.migration_limit,
                    self.metrics.clone(),
                )
                .await
                .context("build_routed_pipeline_with_preprocessor")?;
                worker_set.completions_engine = Some(completions_engine);
                tracing::info!("Completions is ready");
            }
        } else if card.model_input == ModelInput::Text && card.model_type.supports_embedding() {
            let push_router = PushRouter::<
                NvCreateEmbeddingRequest,
                Annotated<NvCreateEmbeddingResponse>,
            >::from_client_with_threshold(
                client, self.router_config.router_mode, None, None
            )
            .await?;
            worker_set.embeddings_engine = Some(Arc::new(push_router));
        }
        // Case: Text + (Images, Audio, Videos)
        // Must come before the plain Text+Chat / Text+Completions branches because
        // diffusion models often set both Images and Chat flags. The branch below
        // handles the chat registration internally when supports_chat() is true.
        else if card.model_input == ModelInput::Text
            && (card.model_type.supports_images()
                || card.model_type.supports_audios()
                || card.model_type.supports_videos())
        {
            // Image/Audio/Video models can also support chat completions (vLLM omni way)
            if card.model_type.supports_chat() {
                let chat_router = PushRouter::<
                    NvCreateChatCompletionRequest,
                    Annotated<NvCreateChatCompletionStreamResponse>,
                >::from_client_with_threshold(
                    client.clone(),
                    self.router_config.router_mode,
                    None,
                    None,
                )
                .await?;
                worker_set.chat_engine = Some(Arc::new(chat_router));
            }

            if card.model_type.supports_images() {
                let images_router = PushRouter::<
                    NvCreateImageRequest,
                    Annotated<NvImagesResponse>,
                >::from_client_with_threshold(
                    client.clone(), self.router_config.router_mode, None, None
                )
                .await?;
                worker_set.images_engine = Some(Arc::new(images_router));
            }

            if card.model_type.supports_videos() {
                let videos_router = PushRouter::<
                    NvCreateVideoRequest,
                    Annotated<NvVideosResponse>,
                >::from_client_with_threshold(
                    client.clone(), self.router_config.router_mode, None, None
                )
                .await?;
                worker_set.videos_engine = Some(Arc::new(videos_router));
            }

            // TODO: add audio models support
        } else if card.model_input == ModelInput::Text && card.model_type.supports_chat() {
            // Case: Text + Chat (pure text-to-text, no diffusion)
            let push_router = PushRouter::<
                NvCreateChatCompletionRequest,
                Annotated<NvCreateChatCompletionStreamResponse>,
            >::from_client_with_threshold(
                client, self.router_config.router_mode, None, None
            )
            .await?;
            worker_set.chat_engine = Some(Arc::new(push_router));
        } else if card.model_input == ModelInput::Text && card.model_type.supports_completions() {
            // Case: Text + Completions
            let push_router = PushRouter::<
                NvCreateCompletionRequest,
                Annotated<NvCreateCompletionResponse>,
            >::from_client_with_threshold(
                client, self.router_config.router_mode, None, None
            )
            .await?;
            worker_set.completions_engine = Some(Arc::new(push_router));
        } else if card.model_input == ModelInput::Tokens && card.model_type.supports_embedding() {
            let frontend = SegmentSource::<
                SingleIn<NvCreateEmbeddingRequest>,
                ManyOut<Annotated<NvCreateEmbeddingResponse>>,
            >::new();

            let preprocessor = OpenAIPreprocessor::new(card.clone())?.into_operator();
            let backend = Backend::from_mdc(card).into_operator();

            let router = PushRouter::<
                PreprocessedEmbeddingRequest,
                Annotated<EmbeddingsEngineOutput>,
            >::from_client_with_threshold(
                client, self.router_config.router_mode, None, None
            )
            .await?;

            let service_backend = ServiceBackend::from_engine(Arc::new(router));

            let embedding_engine = frontend
                .link(preprocessor.forward_edge())?
                .link(backend.forward_edge())?
                .link(service_backend)?
                .link(backend.backward_edge())?
                .link(preprocessor.backward_edge())?
                .link(frontend)?;

            worker_set.embeddings_engine = Some(embedding_engine);
        } else if card.model_input == ModelInput::Tensor && card.model_type.supports_tensor() {
            // Case 6: Tensor + TensorBased (non-LLM)
            // No KV cache concepts - not an LLM model
            let push_router = PushRouter::<
                NvCreateTensorRequest,
                Annotated<NvCreateTensorResponse>,
            >::from_client_with_threshold(
                client, self.router_config.router_mode, None, None
            )
            .await?;
            worker_set.tensor_engine = Some(Arc::new(push_router));
        } else if card.model_type.supports_prefill() {
            if card.model_input != ModelInput::Tokens {
                anyhow::bail!(
                    "Prefill models must use ModelInput::Tokens, got {}",
                    card.model_input.as_str()
                );
            }

            tracing::info!(
                model_name = card.name(),
                "Prefill model detected, registering and activating prefill router"
            );

            // Prefill sets have no engines — we add the WorkerSet first for tracking,
            // then activate the prefill router.
            worker_set.increment_workers();
            self.manager.add_worker_set(card.name(), &namespace, worker_set);

            let Ok(()) = self.manager.activate_prefill_router(card.name(), &namespace, endpoint) else {
                tracing::warn!(
                    model_name = card.name(),
                    "Failed to activate prefill router - prefill model may already be activated"
                );
                return Ok(());
            };

            tracing::info!(
                model_name = card.name(),
                "Prefill model registered and router activated successfully"
            );

            return Ok(());
        } else {
            anyhow::bail!(
                "Unsupported model configuration: {} with {} input. Supported combinations: \
                Tokens+(Chat|Completions|Prefill), Text+(Chat|Completions|Images), Tokens+Embeddings, Tensor+TensorBased",
                card.model_type,
                card.model_input.as_str()
            );
        }

        // Count the first worker that triggered this registration
        worker_set.increment_workers();

        // Add the completed WorkerSet to the Model
        self.manager.add_worker_set(card.name(), &namespace, worker_set);

        Ok(())
    }

    /// All the registered ModelDeploymentCard with the EndpointId they are attached to, one per instance
    async fn all_cards(&self) -> anyhow::Result<Vec<(EndpointId, ModelDeploymentCard)>> {
        let discovery = self.drt.discovery();
        let instances = discovery.list(DiscoveryQuery::AllModels).await?;

        let mut results = Vec::with_capacity(instances.len());
        for instance in instances {
            match instance.deserialize_model::<ModelDeploymentCard>() {
                Ok(card) => {
                    let endpoint_id = match &instance {
                        dynamo_runtime::discovery::DiscoveryInstance::Model {
                            namespace,
                            component,
                            endpoint,
                            ..
                        } => EndpointId {
                            namespace: namespace.clone(),
                            component: component.clone(),
                            name: endpoint.clone(),
                        },
                        _ => {
                            tracing::error!(
                                "Unexpected discovery instance type (expected ModelCard)"
                            );
                            continue;
                        }
                    };
                    results.push((endpoint_id, card));
                }
                Err(err) => {
                    tracing::error!(%err, "Failed to deserialize model card");
                    continue;
                }
            }
        }
        Ok(results)
    }

    pub async fn cards_for_model(
        &self,
        model_name: &str,
        namespace_filter: &NamespaceFilter,
    ) -> anyhow::Result<Vec<ModelDeploymentCard>> {
        Ok(self
            .cards_for_model_with_endpoints(model_name, namespace_filter)
            .await?
            .into_iter()
            .map(|(_, card)| card)
            .collect())
    }

    /// Like `cards_for_model` but also returns the EndpointId for each card,
    /// allowing callers to filter by namespace.
    async fn cards_for_model_with_endpoints(
        &self,
        model_name: &str,
        namespace_filter: &NamespaceFilter,
    ) -> anyhow::Result<Vec<(EndpointId, ModelDeploymentCard)>> {
        let mut all = self.all_cards().await?;
        all.retain(|(endpoint_id, card)| {
            let matches_name = card.name() == model_name;
            let matches_namespace = namespace_filter.matches(&endpoint_id.namespace);
            matches_name && matches_namespace
        });
        Ok(all)
    }
}
