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
    entrypoint::{self, EngineFactoryCallback, RouterConfig},
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
        },
        tensor::{NvCreateTensorRequest, NvCreateTensorResponse},
    },
};

use super::ModelManager;
use crate::namespace::is_global_namespace;

#[derive(Debug, Clone)]
pub enum ModelUpdate {
    Added(ModelDeploymentCard),
    Removed(ModelDeploymentCard),
}

pub struct ModelWatcher {
    manager: Arc<ModelManager>,
    drt: DistributedRuntime,
    router_config: RouterConfig,
    notify_on_model: Notify,
    model_update_tx: Option<Sender<ModelUpdate>>,
    engine_factory: Option<EngineFactoryCallback>,
    metrics: Arc<Metrics>,
    registering_models: DashSet<String>,
}

const ALL_MODEL_TYPES: &[ModelType] = &[
    ModelType::Chat,
    ModelType::Completions,
    ModelType::Embedding,
    ModelType::TensorBased,
    ModelType::Prefill,
];

impl ModelWatcher {
    pub fn new(
        runtime: DistributedRuntime,
        model_manager: Arc<ModelManager>,
        router_config: RouterConfig,
        engine_factory: Option<EngineFactoryCallback>,
        metrics: Arc<Metrics>,
    ) -> ModelWatcher {
        Self {
            manager: model_manager,
            drt: runtime,
            router_config,
            notify_on_model: Notify::new(),
            model_update_tx: None,
            engine_factory,
            metrics,
            registering_models: DashSet::new(),
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
        target_namespace: Option<&str>,
    ) {
        let global_namespace = target_namespace.is_none_or(is_global_namespace);

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

                    // Filter by namespace if target_namespace is specified
                    if !global_namespace
                        && let Some(target_ns) = target_namespace
                        && mcid.namespace != target_ns
                    {
                        tracing::debug!(
                            model_namespace = mcid.namespace,
                            target_namespace = target_ns,
                            "Skipping model from different namespace"
                        );
                        continue;
                    }

                    // If we already have a worker for this model, and the ModelDeploymentCard
                    // cards don't match, alert, and don't add the new instance
                    let can_add =
                        self.manager
                            .is_valid_checksum(card.model_type, card.name(), card.mdcsum());
                    if can_add.is_some_and(|is_valid| !is_valid) {
                        tracing::error!(
                            model_name = card.name(),
                            "Checksum for new model does not match existing model."
                        );

                        // TODO: mark that instance down in clients
                        // Not obvious how to do that given the current design
                        // Instances come from an `InstanceSource` in a `Client` in a `PushRouter`.
                        // Calling `report_instance_down` on the Client should do it (although
                        // needs more testing).
                        // The `PushRouter` is in `ModelMananger` (`self.manager` here), but inside
                        // interface `AsyncEngine` which only has a `generate` method.

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
                        .handle_delete(model_card_instance_id, target_namespace, global_namespace)
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

    /// If the last instance running this model has gone delete it.
    /// Returns the name of the model we just deleted, if any.
    async fn handle_delete(
        &self,
        mcid: &ModelCardInstanceId,
        target_namespace: Option<&str>,
        is_global_namespace: bool,
    ) -> anyhow::Result<Option<String>> {
        let key = mcid.to_path();
        let card = match self.manager.remove_model_card(&key) {
            Some(card) => card,
            None => {
                anyhow::bail!("Missing ModelDeploymentCard for {}", key);
            }
        };
        let model_name = card.name().to_string();
        let active_instances = self
            .cards_for_model(&model_name, target_namespace, is_global_namespace)
            .await
            .with_context(|| model_name.clone())?;
        if !active_instances.is_empty() {
            tracing::debug!(
                model_name,
                target_namespace = ?target_namespace,
                active_instance_count = active_instances.len(),
                "Model has other active instances, not removing"
            );
            return Ok(None);
        }

        // Ignore the errors because model could be either type
        let chat_model_remove_err = self.manager.remove_chat_completions_model(&model_name);
        let completions_model_remove_err = self.manager.remove_completions_model(&model_name);
        let embeddings_model_remove_err = self.manager.remove_embeddings_model(&model_name);
        let tensor_model_remove_err = self.manager.remove_tensor_model(&model_name);
        let prefill_model_remove_err = self.manager.remove_prefill_model(&model_name);

        let mut chat_model_removed = false;
        let mut completions_model_removed = false;
        let mut embeddings_model_removed = false;
        let mut tensor_model_removed = false;
        let mut prefill_model_removed = false;

        if chat_model_remove_err.is_ok() && self.manager.list_chat_completions_models().is_empty() {
            chat_model_removed = true;
        }
        if completions_model_remove_err.is_ok() && self.manager.list_completions_models().is_empty()
        {
            completions_model_removed = true;
        }
        if embeddings_model_remove_err.is_ok() && self.manager.list_embeddings_models().is_empty() {
            embeddings_model_removed = true;
        }
        if tensor_model_remove_err.is_ok() && self.manager.list_tensor_models().is_empty() {
            tensor_model_removed = true;
        }
        if prefill_model_remove_err.is_ok() && self.manager.list_prefill_models().is_empty() {
            prefill_model_removed = true;
        }

        if !chat_model_removed
            && !completions_model_removed
            && !embeddings_model_removed
            && !tensor_model_removed
            && !prefill_model_removed
        {
            tracing::debug!(
                "No updates to send for model {}: chat_model_removed: {}, completions_model_removed: {}, embeddings_model_removed: {}, tensor_model_removed: {}, prefill_model_removed: {}",
                model_name,
                chat_model_removed,
                completions_model_removed,
                embeddings_model_removed,
                tensor_model_removed,
                prefill_model_removed
            );
        } else {
            for model_type in ALL_MODEL_TYPES {
                if ((chat_model_removed && *model_type == ModelType::Chat)
                    || (completions_model_removed && *model_type == ModelType::Completions)
                    || (embeddings_model_removed && *model_type == ModelType::Embedding)
                    || (tensor_model_removed && *model_type == ModelType::TensorBased)
                    || (prefill_model_removed && *model_type == ModelType::Prefill))
                    && let Some(tx) = &self.model_update_tx
                {
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
        // Check if model is already registered before downloading config.
        // This prevents duplicate HuggingFace API calls when multiple workers register
        // the same model.
        // Prefill and decode models are tracked separately, so registering one
        // doesn't block the other (they can arrive in any order).
        let already_registered = if card.model_type.supports_prefill() {
            self.manager.has_prefill_model(card.name())
        } else {
            self.manager.has_decode_model(card.name())
        };

        if already_registered {
            self.manager
                .save_model_card(&mcid.to_path(), card.clone())?;
            tracing::debug!(
                model_name = card.name(),
                namespace = mcid.namespace,
                model_type = %card.model_type,
                "Model already registered, skipping config download"
            );
            return Ok(());
        }

        // Use registering_models set to prevent concurrent registrations.
        let model_key = card.name().to_string();
        if !self.registering_models.insert(model_key.clone()) {
            self.manager
                .save_model_card(&mcid.to_path(), card.clone())?;
            tracing::debug!(
                model_name = card.name(),
                namespace = mcid.namespace,
                "Model registration in progress by another worker, skipping"
            );
            return Ok(());
        }

        // We acquired the registration lock. Use a helper to ensure cleanup on all exit paths.
        let result = self.do_model_registration(mcid, card).await;

        // Always remove from registering set, whether success or failure
        self.registering_models.remove(&model_key);

        result
    }

    /// Inner function that performs the actual model registration.
    /// Called by handle_put after acquiring the registration lock.
    async fn do_model_registration(
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
        tracing::debug!(model_name = card.name(), "adding model");
        self.manager
            .save_model_card(&mcid.to_path(), card.clone())?;

        if let Some(tx) = &self.model_update_tx {
            tx.send(ModelUpdate::Added(card.clone())).await.ok();
        }

        let checksum = card.mdcsum();

        if card.model_input == ModelInput::Tokens
            && (card.model_type.supports_chat() || card.model_type.supports_completions())
        {
            // Case 1: Tokens + (Chat OR Completions OR Both)
            // A model that expects pre-processed requests meaning it's up to us whether we
            // handle Chat or Completions requests, so handle whatever the model supports.

            let endpoint = component.endpoint(&mcid.endpoint);
            let kv_chooser = if self.router_config.router_mode == RouterMode::KV {
                Some(
                    self.manager
                        .kv_chooser_for(
                            &endpoint,
                            card.kv_cache_block_size,
                            Some(self.router_config.kv_router_config),
                        )
                        .await?,
                )
            } else {
                None
            };

            // This is expensive, we are loading ~10MiB JSON, so only do it once
            let tokenizer_hf = card.tokenizer_hf().context("tokenizer_hf")?;

            // Create prefill chooser once if we're building pipelines
            // Both chat and completions will share the same prefill chooser instance
            let prefill_chooser = self
                .manager
                .register_prefill_router(card.name().to_string())
                .map(|rx| {
                    // Create prefill-specific config with track_active_blocks disabled
                    let mut prefill_config = self.router_config.kv_router_config;
                    prefill_config.router_track_active_blocks = false;

                    PrefillRouter::new(
                        rx,
                        self.manager.clone(),
                        self.router_config.router_mode,
                        card.kv_cache_block_size,
                        Some(prefill_config),
                        self.router_config.enforce_disagg,
                    )
                });

            // Get or create the worker monitor for this model
            // This allows dynamic threshold updates via the ModelManager
            // Create monitor if any threshold is configured
            let worker_monitor = if self.router_config.load_threshold_config.is_configured() {
                Some(self.manager.get_or_create_worker_monitor(
                    card.name(),
                    client.clone(),
                    self.router_config.load_threshold_config.clone(),
                ))
            } else {
                None
            };

            // Add chat engine only if the model supports chat
            if card.model_type.supports_chat() {
                // Work in progress. This will allow creating  a chat_engine from Python.
                let chat_engine = if let Some(ref factory) = self.engine_factory {
                    factory(card.clone())
                        .await
                        .context("python engine_factory")?
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
                        self.metrics.clone(),
                    )
                    .await
                    .context("build_routed_pipeline")?
                };
                self.manager
                    .add_chat_completions_model(card.name(), checksum, chat_engine)
                    .context("add_chat_completions_model")?;
                tracing::info!("Chat completions is ready");
            }

            // Add completions engine only if the model supports completions
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
                    self.metrics.clone(),
                )
                .await
                .context("build_routed_pipeline_with_preprocessor")?;
                self.manager
                    .add_completions_model(card.name(), checksum, completions_engine)
                    .context("add_completions_model")?;
                tracing::info!("Completions is ready");
            }
        } else if card.model_input == ModelInput::Text && card.model_type.supports_embedding() {
            // Case: Text + Embeddings
            let push_router = PushRouter::<
                NvCreateEmbeddingRequest,
                Annotated<NvCreateEmbeddingResponse>,
            >::from_client_with_threshold(
                client, self.router_config.router_mode, None, None
            )
            .await?;
            let engine = Arc::new(push_router);
            self.manager
                .add_embeddings_model(card.name(), checksum, engine)?;
        } else if card.model_input == ModelInput::Text && card.model_type.supports_chat() {
            // Case 3: Text + Chat
            let push_router = PushRouter::<
                NvCreateChatCompletionRequest,
                Annotated<NvCreateChatCompletionStreamResponse>,
            >::from_client_with_threshold(
                client, self.router_config.router_mode, None, None
            )
            .await?;
            let engine = Arc::new(push_router);
            self.manager
                .add_chat_completions_model(card.name(), checksum, engine)?;
        } else if card.model_input == ModelInput::Text && card.model_type.supports_completions() {
            // Case 2: Text + Completions
            let push_router = PushRouter::<
                NvCreateCompletionRequest,
                Annotated<NvCreateCompletionResponse>,
            >::from_client_with_threshold(
                client, self.router_config.router_mode, None, None
            )
            .await?;
            let engine = Arc::new(push_router);
            self.manager
                .add_completions_model(card.name(), checksum, engine)?;
        } else if card.model_input == ModelInput::Tokens && card.model_type.supports_embedding() {
            // Case 4: Tokens + Embeddings

            // Create preprocessing pipeline similar to Backend
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

            // Note: Embeddings don't need KV routing complexity or load monitoring
            let service_backend = ServiceBackend::from_engine(Arc::new(router));

            // Link the pipeline: frontend -> preprocessor -> backend -> service_backend -> backend -> preprocessor -> frontend
            let embedding_engine = frontend
                .link(preprocessor.forward_edge())?
                .link(backend.forward_edge())?
                .link(service_backend)?
                .link(backend.backward_edge())?
                .link(preprocessor.backward_edge())?
                .link(frontend)?;

            self.manager
                .add_embeddings_model(card.name(), checksum, embedding_engine)?;
        } else if card.model_input == ModelInput::Tensor && card.model_type.supports_tensor() {
            // Case 5: Tensor + Tensor (non-LLM)
            // No KV cache concepts - not an LLM model
            let push_router = PushRouter::<
                NvCreateTensorRequest,
                Annotated<NvCreateTensorResponse>,
            >::from_client_with_threshold(
                client, self.router_config.router_mode, None, None
            )
            .await?;
            let engine = Arc::new(push_router);
            self.manager
                .add_tensor_model(card.name(), checksum, engine)?;
        } else if card.model_input == ModelInput::Text && card.model_type.supports_images() {
            // Case: Text + Images (diffusion models)
            // Takes text prompts as input, generates images
            let push_router = PushRouter::<
                NvCreateImageRequest,
                Annotated<NvImagesResponse>,
            >::from_client_with_threshold(
                client, self.router_config.router_mode, None, None
            )
            .await?;
            let engine = Arc::new(push_router);
            self.manager
                .add_images_model(card.name(), checksum, engine)?;
        } else if card.model_type.supports_prefill() {
            // Case 6: Prefill
            // Guardrail: Verify model_input is Tokens
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

            // Register prefill model for tracking (no engine needed, just lifecycle)
            self.manager
                .add_prefill_model(card.name(), checksum)
                .context("add_prefill_model")?;

            // Activate the prefill router with the endpoint for this prefill model
            let Ok(()) = self.manager.activate_prefill_router(card.name(), endpoint) else {
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
        } else {
            // Reject unsupported combinations
            anyhow::bail!(
                "Unsupported model configuration: {} with {} input. Supported combinations: \
                Tokens+(Chat|Completions|Prefill), Text+(Chat|Completions|Images), Tokens+Embeddings, Tensor+TensorBased",
                card.model_type,
                card.model_input.as_str()
            );
        }

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
                    // Extract EndpointId from the instance
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
        target_namespace: Option<&str>,
        is_global_namespace: bool,
    ) -> anyhow::Result<Vec<ModelDeploymentCard>> {
        let mut all = self.all_cards().await?;
        all.retain(|(endpoint_id, card)| {
            let matches_name = card.name() == model_name;
            let matches_namespace = match (is_global_namespace, target_namespace) {
                (true, _) => true,
                (false, None) => true,
                (false, Some(target_ns)) => endpoint_id.namespace == target_ns,
            };
            matches_name && matches_namespace
        });
        Ok(all.into_iter().map(|(_eid, card)| card).collect())
    }
}
