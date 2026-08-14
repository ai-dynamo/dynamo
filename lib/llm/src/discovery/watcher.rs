// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Notify, mpsc::Sender};

use anyhow::Context as _;
use async_trait::async_trait;
use dynamo_kv_router::{
    DEFAULT_ROUTING_GROUP, PrefillLoadEstimator, RoutingPartitionRef,
    selector::{DefaultWorkerSelector, WorkerSelector},
};
use dynamo_runtime::{
    DistributedRuntime,
    discovery::{DiscoveryInstance, DiscoveryQuery, DiscoveryStream, ModelCardInstanceId},
    pipeline::{
        ManyOut, Operator, RouterMode, SegmentSource, ServiceBackend, SingleIn, Source,
        network::egress::push_router::PushRouter,
    },
    protocols::{EndpointId, annotated::Annotated},
};

use dynamo_renderer::PromptFormatter;

use crate::{
    backend::Backend,
    discovery::{KvWorkerMonitor, WORKER_TYPE_DECODE, WorkerSet},
    entrypoint::{self, ChatEngineFactoryCallback, RouterConfig},
    http::service::metrics::Metrics,
    kv_router::{EncoderRouter, PrefillRouter, WorkerSelectorFactory},
    local_model::runtime_config::{
        ModelRuntimeConfig, TokenizerBackend, VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
    },
    model_card::ModelDeploymentCard,
    model_type::{ModelInput, ModelType},
    preprocessor::{
        OpenAIPreprocessor, PreprocessedEmbeddingRequest, prompt::prompt_formatter_from_mdc,
    },
    protocols::{
        common::llm_backend::EmbeddingsEngineOutput,
        openai::{
            audios::{NvAudioSpeechResponse, NvCreateAudioSpeechRequest},
            chat_completions::{
                NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            },
            classify::{NvCreateClassifyRequest, NvCreateClassifyResponse},
            completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
            embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
            images::{NvCreateImageRequest, NvImagesResponse},
            pooling::{NvCreatePoolingRequest, NvCreatePoolingResponse},
            videos::{NvCreateVideoRequest, NvVideosResponse},
        },
        tensor::{NvCreateTensorRequest, NvCreateTensorResponse},
    },
    types::generic::realtime::{RealtimeClientEvent, RealtimeServerEvent},
    worker_type::WorkerType,
};

use super::{
    ModelManager,
    controller::{ControllerHost, DesiredInstance, GroupKey, GroupSpec, ModelDiscoveryController},
};
use crate::namespace::NamespaceFilter;
use tokio_util::sync::CancellationToken;

/// Constructs a collision-free WorkerSet storage key from its exact endpoint,
/// model type, and worker role.
///
/// Each `(EndpointId, model_type, worker_type)` combination gets its own
/// WorkerSet bucket. This generalizes the old `{ns}` / `{ns}:prefill` split:
/// prefill, decode, encode, and aggregated workers within the same namespace
/// (and even the same model_type) cleanly separate by `worker_type`. Encode
/// workers, which register with [`ModelType::empty`], end up under
/// `{ns}::encode` — distinct from a decode `{ns}:chat|completions:decode`.
///
/// `worker_type` arrives as `Option<WorkerType>` because the
/// serving-readiness fields on the MDC are still optional at the type
/// level; the compat shim renders missing values via
/// [`effective_worker_type`] so legacy cards bucket and route correctly.
fn worker_set_key(
    endpoint_id: &EndpointId,
    model_type: ModelType,
    worker_type: Option<WorkerType>,
) -> String {
    let mt = model_type.as_vec().join("|");
    let wt = effective_worker_type(worker_type, model_type);
    serde_json::to_string(&(
        &endpoint_id.namespace,
        &endpoint_id.component,
        &endpoint_id.name,
        mt,
        wt.as_str(),
    ))
    .expect("serializing WorkerSet key strings cannot fail")
}

fn model_card_endpoint_id(mcid: &ModelCardInstanceId) -> EndpointId {
    EndpointId {
        namespace: mcid.namespace.clone(),
        component: mcid.component.clone(),
        name: mcid.endpoint.clone(),
    }
}

fn model_card_instance_id(instance: &DiscoveryInstance) -> anyhow::Result<ModelCardInstanceId> {
    match instance {
        DiscoveryInstance::Model {
            namespace,
            component,
            endpoint,
            instance_id,
            model_suffix,
            ..
        } => Ok(ModelCardInstanceId {
            namespace: namespace.clone(),
            component: component.clone(),
            endpoint: endpoint.clone(),
            instance_id: *instance_id,
            model_suffix: model_suffix.clone(),
        }),
        _ => anyhow::bail!("Unexpected discovery instance type (expected ModelCard)"),
    }
}

fn uses_multimodal_cache_routing(card: &ModelDeploymentCard) -> bool {
    card.worker_type == Some(WorkerType::Encode)
        || card.media_decoder.is_some()
        || card.model_type.supports_images()
        || card.model_type.supports_videos()
        || card
            .needs
            .iter()
            .flatten()
            .any(|worker_type| *worker_type == WorkerType::Encode)
}

fn supports_generate_capability(card: &ModelDeploymentCard, capability: &str) -> bool {
    matches!(
        card.runtime_config.runtime_data.get(capability),
        Some(serde_json::Value::Bool(true))
    )
}

fn supports_enabled_engine_generate(card: &ModelDeploymentCard, capabilities: &[&str]) -> bool {
    capabilities
        .iter()
        .any(|capability| supports_generate_capability(card, capability))
}

// Generate's opaque request state is not yet verified for migration replay.
const GENERATE_MIGRATION_LIMIT: u32 = 0;

/// Project the topology implicit in a pre-`worker_type` prefill card into the
/// explicit contract used by current workers.
///
/// TODO(v1.5): Remove this projection together with the missing-role fallback
/// in `effective_worker_type` and the legacy readiness bypass after the v1.2
/// MDC compatibility window expires.
fn normalize_legacy_prefill_topology(card: &mut ModelDeploymentCard) {
    if card.worker_type.is_some() || !card.model_type.supports_prefill() {
        return;
    }

    card.worker_type = Some(WorkerType::Prefill);
    if card.needs.is_empty() {
        card.needs = vec![vec![WorkerType::Decode]];
    }
}

/// Resolve the effective [`WorkerType`] for a card during the
/// cross-version rollout.
///
/// A card from a **new** worker carries an explicit `worker_type`, used
/// verbatim. A card from an **old** (legacy) worker has no `worker_type`;
/// we reconstruct its role from the signal an old frontend itself used — the
/// legacy `ModelType::Prefill` marker bit:
///
/// - legacy prefill card (`ModelType::Prefill` set, no `worker_type`) → `Prefill`
/// - any other legacy card → `Aggregated`
///
/// This lets a new frontend activate the prefill router for, and correctly
/// bucket, an old prefill worker. (Old *decode* workers are indistinguishable
/// from old *aggregated* workers on the wire, so they resolve to `Aggregated`;
/// the readiness path handles that by not topology-gating namespaces that
/// still contain legacy cards — see `Model::is_workers_ready`.)
fn effective_worker_type(worker_type: Option<WorkerType>, model_type: ModelType) -> WorkerType {
    ModelDeploymentCard::resolve_worker_type(worker_type, model_type)
}

#[derive(Debug, Clone)]
pub enum ModelUpdate {
    Added(ModelDeploymentCard),
    Removed(ModelDeploymentCard),
}

pub struct ModelWatcher<Sel = DefaultWorkerSelector>
where
    Sel: WorkerSelector<ModelRuntimeConfig>,
{
    manager: Arc<ModelManager>,
    drt: DistributedRuntime,
    router_config: RouterConfig,
    migration_limit: u32,
    migration_max_seq_len: Option<u32>,
    notify_on_model: Notify,
    model_update_tx: Option<Sender<ModelUpdate>>,
    model_update_dispatch:
        parking_lot::Mutex<Option<tokio::sync::mpsc::UnboundedSender<ModelUpdate>>>,
    chat_engine_factory: Option<ChatEngineFactoryCallback>,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    metrics: Arc<Metrics>,
    /// Frontend's `--model-path`. Threaded into `download_config` so
    /// `file://` slots can fall back here when the worker's path is
    /// unreachable on this host.
    local_model_path: Option<PathBuf>,
    /// Frontend-level tokenizer backend override for discovered model cards.
    tokenizer_backend: Option<TokenizerBackend>,
    /// Frontend-level tokenizer fallback override for discovered model cards.
    tokenizer_fallback_enabled: Option<bool>,
    /// Worker capabilities accepted by the frontend's engine-native Generate routes.
    /// Keep raw pipelines out of default-off and backend-mismatched paths.
    generate_engine_capabilities: Vec<&'static str>,
    worker_selector_factory: WorkerSelectorFactory<Sel>,
    /// Custom selector dispatch cannot infer whether an untyped legacy card is decode or aggregated.
    require_typed_worker_role: bool,
}

pub(crate) struct PreparedWorkerSet {
    worker_set: Option<WorkerSet>,
    card: ModelDeploymentCard,
}

const ALL_MODEL_TYPES: &[ModelType] = &[
    ModelType::Chat,
    ModelType::Completions,
    ModelType::Embedding,
    ModelType::Images,
    ModelType::Audios,
    ModelType::Videos,
    ModelType::TensorBased,
    ModelType::Realtime,
    ModelType::Classify,
    ModelType::Pooling,
];

/// Returns true if no models in the manager support the given model type.
fn is_model_type_list_empty(manager: &ModelManager, model_type: ModelType) -> bool {
    if model_type == ModelType::Chat {
        manager.list_chat_completions_models().is_empty()
    } else if model_type == ModelType::Completions {
        manager.list_completions_models().is_empty()
    } else if model_type == ModelType::Embedding {
        manager.list_embeddings_models().is_empty()
    } else if model_type == ModelType::Images {
        manager.list_images_models().is_empty()
    } else if model_type == ModelType::Audios {
        manager.list_audios_models().is_empty()
    } else if model_type == ModelType::Videos {
        manager.list_videos_models().is_empty()
    } else if model_type == ModelType::TensorBased {
        manager.list_tensor_models().is_empty()
    } else if model_type == ModelType::Realtime {
        manager.list_realtime_models().is_empty()
    } else if model_type == ModelType::Classify {
        manager.list_classify_models().is_empty()
    } else if model_type == ModelType::Pooling {
        manager.list_pooling_models().is_empty()
    } else {
        true
    }
}

fn removed_model_cards(
    manager: &ModelManager,
    card: &ModelDeploymentCard,
) -> Vec<ModelDeploymentCard> {
    ALL_MODEL_TYPES
        .iter()
        .filter_map(|model_type| {
            if card.model_type.intersects(*model_type)
                && is_model_type_list_empty(manager, *model_type)
            {
                let mut removed_card = card.clone();
                removed_card.model_type = *model_type;
                Some(removed_card)
            } else {
                None
            }
        })
        .collect()
}

impl ModelWatcher<DefaultWorkerSelector> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: DistributedRuntime,
        model_manager: Arc<ModelManager>,
        router_config: RouterConfig,
        migration_limit: u32,
        migration_max_seq_len: Option<u32>,
        chat_engine_factory: Option<ChatEngineFactoryCallback>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        metrics: Arc<Metrics>,
    ) -> ModelWatcher {
        Self::new_with_worker_selector_factory(
            runtime,
            model_manager,
            router_config,
            migration_limit,
            migration_max_seq_len,
            chat_engine_factory,
            prefill_load_estimator,
            metrics,
            false,
            Arc::new(|config, worker_type, _partition| {
                DefaultWorkerSelector::new(
                    Some(config.clone()),
                    worker_type.default_selector_label(),
                )
            }),
        )
    }
}

impl<Sel> ModelWatcher<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_worker_selector_factory(
        runtime: DistributedRuntime,
        model_manager: Arc<ModelManager>,
        router_config: RouterConfig,
        migration_limit: u32,
        migration_max_seq_len: Option<u32>,
        chat_engine_factory: Option<ChatEngineFactoryCallback>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        metrics: Arc<Metrics>,
        require_typed_worker_role: bool,
        worker_selector_factory: WorkerSelectorFactory<Sel>,
    ) -> Self {
        Self {
            manager: model_manager,
            drt: runtime,
            router_config,
            migration_limit,
            migration_max_seq_len,
            notify_on_model: Notify::new(),
            model_update_tx: None,
            model_update_dispatch: parking_lot::Mutex::new(None),
            chat_engine_factory,
            prefill_load_estimator,
            metrics,
            local_model_path: None,
            tokenizer_backend: None,
            tokenizer_fallback_enabled: None,
            generate_engine_capabilities: Vec::new(),
            worker_selector_factory,
            require_typed_worker_role,
        }
    }

    pub fn set_notify_on_model_update(&mut self, tx: Sender<ModelUpdate>) {
        self.model_update_tx = Some(tx);
    }

    pub fn set_local_model_path(&mut self, path: Option<PathBuf>) {
        self.local_model_path = path;
    }

    pub fn set_tokenizer_backend(&mut self, tokenizer_backend: Option<TokenizerBackend>) {
        self.tokenizer_backend = tokenizer_backend;
    }

    pub fn set_tokenizer_fallback_enabled(&mut self, enabled: Option<bool>) {
        self.tokenizer_fallback_enabled = enabled;
    }

    pub(crate) fn set_generate_engine_capabilities(&mut self, capabilities: Vec<&'static str>) {
        self.generate_engine_capabilities = capabilities;
    }
    /// Compatibility wrapper for callers that enable the vLLM Generate route.
    pub fn set_generate_engine_enabled(&mut self, enabled: bool) {
        self.generate_engine_capabilities = enabled
            .then_some(VLLM_INFERENCE_V1_GENERATE_CAPABILITY)
            .into_iter()
            .collect();
    }

    fn apply_tokenizer_overrides(&self, card: &mut ModelDeploymentCard) {
        if let Some(tokenizer_backend) = self.tokenizer_backend {
            card.runtime_config.tokenizer_backend = Some(tokenizer_backend);
        }
        if let Some(enabled) = self.tokenizer_fallback_enabled {
            card.runtime_config.tokenizer_fallback_enabled = Some(enabled);
        }
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

    /// Run the ordered desired-state controller for model discovery.
    pub async fn watch(
        self: Arc<Self>,
        discovery_stream: DiscoveryStream,
        namespace_filter: NamespaceFilter,
    ) {
        let dispatch_handle = self.model_update_tx.clone().map(|external_tx| {
            let (dispatch_tx, mut dispatch_rx) = tokio::sync::mpsc::unbounded_channel();
            *self.model_update_dispatch.lock() = Some(dispatch_tx);
            tokio::spawn(async move {
                while let Some(update) = dispatch_rx.recv().await {
                    if external_tx.send(update).await.is_err() {
                        break;
                    }
                }
            })
        });

        ModelDiscoveryController::new(Arc::clone(&self))
            .run(discovery_stream, namespace_filter)
            .await;

        self.model_update_dispatch.lock().take();
        if let Some(mut dispatch_handle) = dispatch_handle
            && tokio::time::timeout(Duration::from_secs(1), &mut dispatch_handle)
                .await
                .is_err()
        {
            dispatch_handle.abort();
        }
    }

    /// Build a complete WorkerSet off-side. The controller is the only caller that may publish it.
    async fn prepare_worker_set(
        &self,
        spec: &GroupSpec,
        admitted_ids: tokio::sync::watch::Receiver<Vec<u64>>,
        cancellation: CancellationToken,
    ) -> anyhow::Result<PreparedWorkerSet> {
        let mcid = &spec.representative.mcid;
        let mut prepared_card = spec.representative.card.clone();
        let card = &mut prepared_card;

        card.download_config(self.local_model_path.as_deref())
            .await?;

        validate_selector_worker_role(card, self.require_typed_worker_role)?;

        // Use per-worker-set router config if the worker provided one in its MDC,
        // otherwise fall back to the frontend-level global config. Policy selections
        // are process-local, so preserve them when the MDC supplies the base config.
        let router_config =
            effective_router_config(card.router_config.as_ref(), &self.router_config);

        let component = self
            .drt
            .namespace(&mcid.namespace)?
            .component(&mcid.component)?;
        let endpoint = component.endpoint(&mcid.endpoint);
        let client = endpoint
            .client()
            .await?
            .with_admitted_instances_and_cancellation(admitted_ids, cancellation.clone());
        let instance_watcher = client.instance_avail_watcher();
        tracing::debug!(
            model_name = card.name(),
            namespace = mcid.namespace,
            "building worker set pipeline"
        );
        let checksum = card.mdcsum();
        let namespace = mcid.namespace.clone();
        // Build the WorkerSet with all applicable engines
        let mut worker_set = WorkerSet::new(namespace.clone(), checksum.to_string(), card.clone());
        worker_set.set_lifecycle_cancellation(cancellation);
        worker_set.set_topology_endpoint(endpoint.clone());
        worker_set.set_instance_watcher(instance_watcher);

        // A surface-less Encode worker is reached only through EncoderRouter.
        // Register it for serving readiness, publish its endpoint to any
        // waiting token pipeline, and do not build a public OpenAI surface.
        if effective_worker_type(card.worker_type, card.model_type) == WorkerType::Encode
            && card.model_type.is_empty()
        {
            if card.model_input != ModelInput::Tokens {
                anyhow::bail!(
                    "Encode workers must use ModelInput::Tokens, got {}",
                    card.model_input.as_str()
                );
            }
            return Ok(PreparedWorkerSet {
                worker_set: Some(worker_set),
                card: card.clone(),
            });
        }

        // worker_type-driven short circuit for Prefill.
        //
        // A prefill worker carries no OpenAI-style engine — it is reached only
        // through the dedicated prefill router, never by the frontend — so we
        // dispatch it off `worker_type` here, *before* the model_type-based
        // branches below. Everything else is routed by its OpenAI surface: a
        // card that declares a surface builds the matching pipeline (so an
        // sglang multimodal encode worker, which fronts the model, serves like
        // any other worker), while a surface-less (`ModelType::empty()`) card
        // is registered for serving-readiness only (see the `is_empty()` arm at
        // the end of the chain). The role is carried by `worker_type`; serving
        // is driven by `model_type`.
        //
        // `effective_worker_type` also resolves a legacy prefill card (the
        // `ModelType::Prefill` marker bit with no `worker_type`, from an old
        // worker registering against a new frontend) to `Prefill` here, so it
        // activates the prefill router just like a new prefill worker.
        if effective_worker_type(card.worker_type, card.model_type) == WorkerType::Prefill {
            // Guardrail: prefill workers still expect Tokens input downstream.
            if card.model_input != ModelInput::Tokens {
                anyhow::bail!(
                    "Prefill workers must use ModelInput::Tokens, got {}",
                    card.model_input.as_str()
                );
            }

            tracing::info!(
                model_name = card.name(),
                "Prefill worker detected, registering and activating prefill router"
            );

            return Ok(PreparedWorkerSet {
                worker_set: Some(worker_set),
                card: card.clone(),
            });
        }

        if card.model_input == ModelInput::Tokens
            && (card.model_type.supports_chat() || card.model_type.supports_completions())
        {
            // Case 1: Tokens + (Chat OR Completions OR Both)
            // A model that expects pre-processed requests meaning it's up to us whether we
            // handle Chat or Completions requests, so handle whatever the model supports.

            let endpoint = component.endpoint(&mcid.endpoint);
            // Loading the tokenizer is expensive (~10 MiB JSON), so only do it
            // once and only when a local pipeline actually needs it.  Models
            // without tokenizer.json (e.g. Qwen3-Omni) set tokenizer = None;
            // they rely on a Python chat_engine_factory for tokenization.
            // When a chat_engine_factory handles chat and no completions are
            // needed, skip tokenizer loading entirely — even if the file exists.
            let needs_local_chat_pipeline =
                card.model_type.supports_chat() && self.chat_engine_factory.is_none();
            let needs_local_completions_pipeline = card.model_type.supports_completions();
            let tokenizer = if (needs_local_chat_pipeline || needs_local_completions_pipeline)
                && card.has_tokenizer()
            {
                Some(card.tokenizer().context("tokenizer")?)
            } else {
                None
            };

            // Routing is required whenever any pipeline (factory chat or local) will exist.
            // tokenizer.is_some() implies a local chat or completions pipeline will be built.
            let needs_factory_chat_pipeline =
                card.model_type.supports_chat() && self.chat_engine_factory.is_some();
            let needs_generate_pipeline =
                supports_enabled_engine_generate(card, &self.generate_engine_capabilities);
            let needs_preprocessed_routing =
                needs_factory_chat_pipeline || tokenizer.is_some() || needs_generate_pipeline;

            // Create the KV router whenever any routed pipeline will be built.
            // Python chat factories receive a Rust-routed engine, so they also
            // need the shared chooser in KV mode.
            let kv_chooser =
                if router_config.router_mode == RouterMode::KV && needs_preprocessed_routing {
                    let selector = (self.worker_selector_factory)(
                        &router_config.kv_router_config,
                        effective_worker_type(card.worker_type, card.model_type),
                        RoutingPartitionRef::new(&card.display_name, DEFAULT_ROUTING_GROUP),
                    );
                    Some(
                        self.manager
                            .kv_chooser_for_with_selector_and_client(
                                &endpoint,
                                client.clone(),
                                card.kv_cache_block_size,
                                selector,
                                Some(router_config.kv_router_config.clone()),
                                self.prefill_load_estimator.clone(),
                                card.worker_type,
                                WORKER_TYPE_DECODE, // This is the decode router
                                Some(card.display_name.clone()),
                                card.runtime_config.enable_eagle,
                            )
                            .await?,
                    )
                } else {
                    None
                };

            // Create the worker monitor for this WorkerSet BEFORE the prefill router so the
            // monitor can be handed directly to PrefillRouter::new_with_selector_factory. Each
            // WorkerSet gets its own monitor (1-to-1), scoped to this WorkerSet's Client/namespace.
            // The monitor tracks Prometheus metrics (active_decode_blocks, active_prefill_tokens,
            // worker TTFT/ITL
            // cleanup); thresholds control overload detection. The monitor and prefill router are
            // created together here, so the monitor is passed into the prefill router directly.
            //
            // IMPORTANT: When KV routing is active, the monitor must use the KvRouter's Client
            // so that overload-state updates (via set_overloaded_instances) are visible to the
            // PushRouter, which also uses the KvRouter's Client (see common.rs:258-263).
            // Using a different Client instance would cause the PushRouter to never see
            // overloaded workers, since each Client::new() creates independent ArcSwap state.
            let worker_monitor = if needs_preprocessed_routing {
                let monitor_client = kv_chooser
                    .as_ref()
                    .map(|chooser| chooser.client().clone())
                    .unwrap_or_else(|| client.clone());
                Some(KvWorkerMonitor::new(
                    monitor_client,
                    router_config.load_threshold_config.clone(),
                ))
            } else {
                None
            };

            // Only a typed Decode endpoint participates in the namespace-level
            // P/D rendezvous. Aggregated and Encode endpoints are independent
            // serving leaves and must not claim or perturb that pairing.
            let model_name = card.name().to_string();
            let prefill_chooser = if needs_preprocessed_routing
                && effective_worker_type(card.worker_type, card.model_type) == WorkerType::Decode
            {
                let mut prefill_config = router_config.kv_router_config.clone();
                prefill_config.router_track_active_blocks = false;
                let prefill_enable_eagle = false;

                Some(PrefillRouter::new_with_selector_factory(
                    None,
                    self.manager.clone(),
                    router_config.router_mode,
                    card.kv_cache_block_size,
                    Some(prefill_config),
                    kv_chooser.clone(),
                    self.worker_selector_factory.clone(),
                    self.prefill_load_estimator.clone(),
                    router_config.session_affinity_ttl_secs,
                    model_name.clone(),
                    namespace.clone(),
                    prefill_enable_eagle,
                    worker_monitor.clone(),
                ))
            } else {
                None
            };

            let encoder_chooser = if needs_preprocessed_routing {
                Some(EncoderRouter::new(model_name.clone(), namespace.clone()))
            } else {
                None
            };

            // Store the worker monitor and prefill router on the WorkerSet.
            // The prefill router is stored so the watcher can deactivate/reactivate it
            // when prefill workers die or rejoin.
            worker_set.worker_monitor = worker_monitor.clone();
            worker_set.prefill_router = prefill_chooser.clone().map(|router| {
                router as Arc<dyn crate::kv_router::prefill_router::PrefillRouterLifecycle>
            });
            worker_set.encoder_router = encoder_chooser.clone();

            let preprocessed_routing = if needs_preprocessed_routing {
                Some(
                    entrypoint::input::build_preprocessed_routing_with_selector(
                        &client,
                        self.manager.clone(),
                        router_config.router_mode,
                        worker_monitor.clone(),
                        kv_chooser.clone(),
                        prefill_chooser.clone(),
                        encoder_chooser.clone(),
                        uses_multimodal_cache_routing(card),
                        router_config.session_affinity_ttl_secs,
                    )
                    .await
                    .context("build_preprocessed_routing")?,
                )
            } else {
                None
            };

            // Add chat engine only if the model supports chat
            if card.model_type.supports_chat() {
                let routing = preprocessed_routing.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("chat pipeline requires preprocessed routing")
                })?;
                let chat_engine = if let Some(ref factory) = self.chat_engine_factory {
                    let routed_engine = routing
                        .build_preprocessed_pipeline(
                            card,
                            self.migration_limit,
                            self.migration_max_seq_len,
                            self.metrics.clone(),
                        )
                        .context("PreprocessedRouting::build_preprocessed_pipeline")?;
                    Some(
                        factory(mcid.clone(), card.clone(), routed_engine)
                            .await
                            .context("python chat_engine_factory")?,
                    )
                } else if let Some(tk) = tokenizer.clone() {
                    let PromptFormatter::OAI(formatter) =
                        prompt_formatter_from_mdc(card).context("prompt_formatter_from_mdc")?;
                    let preprocessor =
                        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tk.clone())
                            .context("OpenAIPreprocessor.new_with_parts")?;
                    Some(
                        routing
                            .build_pipeline::<
                                NvCreateChatCompletionRequest,
                                NvCreateChatCompletionStreamResponse,
                            >(
                                card,
                                preprocessor,
                                tk,
                                self.migration_limit,
                                self.migration_max_seq_len,
                                self.metrics.clone(),
                            )
                            .context("PreprocessedRouting::build_pipeline")?,
                        )
                } else if needs_generate_pipeline {
                    tracing::warn!(
                        "Skipping chat engine: no supported Rust tokenizer or chat_engine_factory; Generate remains available"
                    );
                    None
                } else {
                    anyhow::bail!(
                        "Model has no supported Rust tokenizer and no chat_engine_factory. \
                         Use --dyn-chat-processor vllm/sglang or provide a supported \
                         tokenizer file (tokenizer.json, tiktoken.model, or *.tiktoken)."
                    );
                };
                if let Some(chat_engine) = chat_engine {
                    worker_set.chat_engine = Some(chat_engine);
                    tracing::info!("Chat completions is ready");
                }
            }

            // Add completions engine only if the model supports completions
            // and we have a tokenizer (completions always uses the Rust preprocessor).
            if card.model_type.supports_completions() {
                if let Some(tk) = tokenizer {
                    let formatter = PromptFormatter::no_op();
                    let PromptFormatter::OAI(formatter) = formatter;
                    let preprocessor =
                        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tk.clone())
                            .context("OpenAIPreprocessor::new_with_parts")?;
                    let routing = preprocessed_routing.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("completions pipeline requires preprocessed routing")
                    })?;
                    let completions_engine = routing
                        .build_pipeline::<NvCreateCompletionRequest, NvCreateCompletionResponse>(
                            card,
                            preprocessor,
                            tk,
                            self.migration_limit,
                            self.migration_max_seq_len,
                            self.metrics.clone(),
                        )
                        .context("PreprocessedRouting::build_pipeline")?;
                    worker_set.completions_engine = Some(completions_engine);
                    tracing::info!("Completions is ready");
                } else {
                    tracing::warn!(
                        "Skipping completions engine: no Rust tokenizer available for this model"
                    );
                }
            }

            // Generate is a frontend-native token-in/token-out surface. It
            // reuses the raw routed pipeline so the complete request envelope
            // reaches the worker without passing through the OpenAI decoder.
            if needs_generate_pipeline {
                let routing = preprocessed_routing.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("generate pipeline requires preprocessed routing")
                })?;
                let generate_engine = routing
                    .build_preprocessed_pipeline(
                        card,
                        GENERATE_MIGRATION_LIMIT,
                        None,
                        self.metrics.clone(),
                    )
                    .context("build generate (preprocessed) pipeline")?;
                worker_set.generate_engine = Some(generate_engine);
                tracing::info!("Generate (token-in/token-out) is ready");
            }

            // Verify we built at least one serving engine. Generate can be the
            // sole engine because token-native requests need no frontend tokenizer.
            if !worker_set.has_any_serving_engine() {
                anyhow::bail!(
                    "Model '{}' requires frontend tokenization/preprocessing (ModelInput::Tokens) \
                     but no serving engine could be built. Provide a working tokenizer config or \
                     perform tokenization in the backend (ModelInput::Text).",
                    card.name()
                );
            }
        } else if card.model_input == ModelInput::Text {
            // Text workers tokenize in the backend and can advertise multiple
            // OpenAI surfaces. Build each declared surface independently:
            // ModelType is a bitflag, so choosing one mutually-exclusive branch
            // would silently omit engines for mixed-capability cards.
            if card.model_type.supports_embedding() {
                let push_router = PushRouter::<
                    NvCreateEmbeddingRequest,
                    Annotated<NvCreateEmbeddingResponse>,
                >::from_client_with_monitor(
                    client.clone(), router_config.router_mode, None
                )
                .await?;
                worker_set.embeddings_engine = Some(Arc::new(push_router));
            }

            if card.model_type.supports_classify() {
                let push_router = PushRouter::<
                    NvCreateClassifyRequest,
                    Annotated<NvCreateClassifyResponse>,
                >::from_client_with_monitor(
                    client.clone(), router_config.router_mode, None
                )
                .await?;
                worker_set.classify_engine = Some(Arc::new(push_router));
            }

            if card.model_type.supports_pooling() {
                let push_router = PushRouter::<
                    NvCreatePoolingRequest,
                    Annotated<NvCreatePoolingResponse>,
                >::from_client_with_monitor(
                    client.clone(), router_config.router_mode, None
                )
                .await?;
                worker_set.pooling_engine = Some(Arc::new(push_router));
            }

            if card.model_type.supports_chat() {
                let chat_router = PushRouter::<
                    NvCreateChatCompletionRequest,
                    Annotated<NvCreateChatCompletionStreamResponse>,
                >::from_client_with_monitor(
                    client.clone(), router_config.router_mode, None
                )
                .await?;
                worker_set.chat_engine = Some(Arc::new(chat_router));
            }

            if card.model_type.supports_completions() {
                let completions_router = PushRouter::<
                    NvCreateCompletionRequest,
                    Annotated<NvCreateCompletionResponse>,
                >::from_client_with_monitor(
                    client.clone(), router_config.router_mode, None
                )
                .await?;
                worker_set.completions_engine = Some(Arc::new(completions_router));
            }

            if card.model_type.supports_images() {
                let images_router = PushRouter::<
                    NvCreateImageRequest,
                    Annotated<NvImagesResponse>,
                >::from_client_with_monitor(client.clone(), router_config.router_mode, None)
                .await?;
                worker_set.images_engine = Some(Arc::new(images_router));
            }

            if card.model_type.supports_videos() {
                let videos_router = PushRouter::<
                    NvCreateVideoRequest,
                    Annotated<NvVideosResponse>,
                >::from_client_with_monitor(client.clone(), router_config.router_mode, None)
                .await?;
                worker_set.videos_engine = Some(Arc::new(videos_router));
            }

            if card.model_type.supports_audios() {
                let audios_router = PushRouter::<
                    NvCreateAudioSpeechRequest,
                    Annotated<NvAudioSpeechResponse>,
                >::from_client_with_monitor(
                    client.clone(), router_config.router_mode, None
                )
                .await?;
                worker_set.audios_engine = Some(Arc::new(audios_router));
            }

            if card.model_type.supports_realtime() {
                // `Text` is overloaded for Realtime; its I/O passes through.
                let realtime_router = PushRouter::<
                    RealtimeClientEvent,
                    Annotated<RealtimeServerEvent>,
                >::from_client_with_monitor(
                    client.clone(), router_config.router_mode, None
                )
                .await?;
                worker_set.realtime_engine = Some(Arc::new(realtime_router));
            }

            if card.model_type.is_empty() {
                tracing::info!(
                    model_name = card.name(),
                    "Topology-only worker (empty model_type), registering for serving readiness only"
                );
            } else if !worker_set.has_any_serving_engine() {
                anyhow::bail!(
                    "Unsupported model configuration: {} with Text input",
                    card.model_type
                );
            }
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
            >::from_client_with_monitor(
                client, router_config.router_mode, None
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

            worker_set.embeddings_engine = Some(embedding_engine);
        } else if card.model_input == ModelInput::Tensor && card.model_type.supports_tensor() {
            // Case 6: Tensor + TensorBased (non-LLM)
            // No KV cache concepts - not an LLM model
            let push_router = PushRouter::<
                NvCreateTensorRequest,
                Annotated<NvCreateTensorResponse>,
            >::from_client_with_monitor(
                client, router_config.router_mode, None
            )
            .await?;
            worker_set.tensor_engine = Some(Arc::new(push_router));
        } else if card.model_type.is_empty() {
            // No OpenAI surface declared: a topology-only worker that exists
            // purely for serving-readiness accounting — e.g. a surface-less
            // encode helper, or an internal disaggregated worker fronted by
            // another worker (reached over RPC, never by the frontend). Build
            // no pipeline; the shared tail below registers the engine-less
            // WorkerSet so the readiness gate counts it. (Prefill is handled by
            // its own branch above.)
            tracing::info!(
                model_name = card.name(),
                "Topology-only worker (empty model_type), registering for serving readiness only"
            );
        } else {
            // A worker that declares an OpenAI surface but with an incompatible
            // model_input. (Surface-less workers hit the `is_empty()` arm above;
            // prefill is routed off `worker_type`.)
            anyhow::bail!(
                "Unsupported model configuration: {} with {} input. Supported combinations: \
                Tokens+(Chat|Completions), Text+(Chat|Completions|Images|Audios|Videos|Embeddings|Classify|Pooling|Realtime), \
                Tokens+Embeddings, Tensor+TensorBased",
                card.model_type,
                card.model_input.as_str()
            );
        }

        Ok(PreparedWorkerSet {
            worker_set: Some(worker_set),
            card: card.clone(),
        })
    }

    fn emit_update(&self, update: ModelUpdate) {
        if let Some(dispatch) = self.model_update_dispatch.lock().as_ref() {
            let _ = dispatch.send(update);
        }
    }
}

#[async_trait]
impl<Sel> ControllerHost for ModelWatcher<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    type Prepared = PreparedWorkerSet;

    fn normalize(
        &self,
        instance: DiscoveryInstance,
        namespace_filter: &NamespaceFilter,
    ) -> anyhow::Result<Option<DesiredInstance>> {
        let mcid = model_card_instance_id(&instance)?;
        if !namespace_filter.matches(&mcid.namespace) {
            return Ok(None);
        }

        let mut card = instance.deserialize_model::<ModelDeploymentCard>()?;
        normalize_legacy_prefill_topology(&mut card);
        self.apply_tokenizer_overrides(&mut card);
        validate_card_shape(&card)?;
        anyhow::ensure!(
            mcid.model_suffix.is_some() == card.lora.is_some(),
            "LoRA discovery identity and card metadata disagree"
        );
        let endpoint_id = model_card_endpoint_id(&mcid);
        let group_key = GroupKey {
            model_name: card.name().to_string(),
            worker_set_key: worker_set_key(&endpoint_id, card.model_type, card.worker_type),
        };
        let fingerprint = materialization_fingerprint(&card, &self.router_config)?;
        let projection_fingerprint = lora_projection_fingerprint(&card)?;
        Ok(Some(DesiredInstance {
            key: mcid.to_path(),
            mcid,
            endpoint_id,
            card,
            group_key,
            fingerprint,
            projection_fingerprint,
        }))
    }

    async fn prepare(
        &self,
        spec: GroupSpec,
        admitted_ids: tokio::sync::watch::Receiver<Vec<u64>>,
        cancellation: CancellationToken,
    ) -> anyhow::Result<Self::Prepared> {
        self.prepare_worker_set(&spec, admitted_ids, cancellation)
            .await
    }

    fn commit_group(
        &self,
        spec: &GroupSpec,
        mut prepared: Self::Prepared,
        members: &[DesiredInstance],
        adapters: &[DesiredInstance],
    ) -> anyhow::Result<()> {
        let adapter_was_available = adapters
            .iter()
            .map(|adapter| {
                (
                    adapter.card.name().to_string(),
                    self.manager
                        .get_committed_model(adapter.card.name())
                        .is_some(),
                )
            })
            .collect::<HashMap<_, _>>();
        let worker_set = prepared
            .worker_set
            .take()
            .ok_or_else(|| anyhow::anyhow!("prepared WorkerSet was already consumed"))?;
        let mut committed_members = members
            .iter()
            .map(|member| (member.key.clone(), member.card.clone()))
            .collect::<Vec<_>>();
        if let Some((_, card)) = committed_members
            .iter_mut()
            .find(|(key, _)| key == &spec.representative.key)
        {
            *card = prepared.card.clone();
        }
        self.manager.commit_discovery_group(
            &spec.key.id(),
            &spec.key.worker_set_key,
            worker_set,
            committed_members,
            adapters
                .iter()
                .map(|adapter| (adapter.key.clone(), adapter.card.clone()))
                .collect(),
        )?;
        self.emit_update(ModelUpdate::Added(prepared.card.clone()));
        let mut adapter_names = HashSet::new();
        for adapter in adapters {
            if adapter_names.insert(adapter.card.name().to_string())
                && !adapter_was_available
                    .get(adapter.card.name())
                    .copied()
                    .unwrap_or(false)
            {
                self.emit_update(ModelUpdate::Added(adapter.card.clone()));
            }
        }
        if prepared.card.model_type.supports_chat() {
            self.notify_on_model.notify_waiters();
        }
        tracing::info!(
            model_name = prepared.card.name(),
            group = %spec.key.id(),
            members = members.len(),
            "Committed discovered model group"
        );
        Ok(())
    }

    fn replace_group(
        &self,
        key: &GroupKey,
        members: &[DesiredInstance],
        adapters: &[DesiredInstance],
    ) -> anyhow::Result<()> {
        let group_id = key.id();
        let previous = self
            .manager
            .discovery_group_adapter_cards(&group_id)
            .into_iter()
            .map(|card| (card.name().to_string(), card))
            .collect::<HashMap<_, _>>();
        let desired = adapters
            .iter()
            .map(|adapter| (adapter.card.name().to_string(), adapter.card.clone()))
            .collect::<HashMap<_, _>>();
        let was_available = previous
            .keys()
            .chain(desired.keys())
            .map(|name| {
                (
                    name.clone(),
                    self.manager.get_committed_model(name).is_some(),
                )
            })
            .collect::<HashMap<_, _>>();
        self.manager.replace_discovery_group(
            &group_id,
            members
                .iter()
                .map(|member| (member.key.clone(), member.card.clone()))
                .collect(),
            adapters
                .iter()
                .map(|adapter| (adapter.key.clone(), adapter.card.clone()))
                .collect(),
        )?;
        for (name, card) in &desired {
            if !was_available.get(name).copied().unwrap_or(false)
                && self.manager.get_committed_model(name).is_some()
            {
                self.emit_update(ModelUpdate::Added(card.clone()));
            }
        }
        for (name, card) in previous {
            if was_available.get(&name).copied().unwrap_or(false)
                && self.manager.get_committed_model(&name).is_none()
            {
                self.emit_update(ModelUpdate::Removed(card));
            }
        }
        Ok(())
    }

    fn remove_group(&self, key: &GroupKey) {
        let Some(removed) = self.manager.remove_discovery_group(&key.id()) else {
            return;
        };
        let removed_members = removed.cards.len();
        let mut removed_adapter_names = HashSet::new();
        for removed_card in &removed.cards {
            if removed_card.lora.is_some()
                && removed_adapter_names.insert(removed_card.name().to_string())
                && self
                    .manager
                    .get_committed_model(removed_card.name())
                    .is_none()
            {
                self.emit_update(ModelUpdate::Removed(removed_card.clone()));
            }
        }
        let card = removed.representative;
        for removed_card in removed_model_cards(&self.manager, &card) {
            self.emit_update(ModelUpdate::Removed(removed_card));
        }
        tracing::info!(
            model_name = card.name(),
            group = %key.id(),
            members = removed_members,
            "Removed discovered model group"
        );
    }

    fn discard_prepared(&self, prepared: Self::Prepared) {
        drop(prepared);
    }

    async fn list_instances(&self) -> anyhow::Result<Vec<DiscoveryInstance>> {
        self.drt.discovery().list(DiscoveryQuery::AllModels).await
    }
}

fn validate_card_shape(card: &ModelDeploymentCard) -> anyhow::Result<()> {
    anyhow::ensure!(!card.name().is_empty(), "model name cannot be empty");
    let worker_type = effective_worker_type(card.worker_type, card.model_type);
    if worker_type == WorkerType::Prefill {
        anyhow::ensure!(
            card.model_input == ModelInput::Tokens,
            "prefill workers must use token input"
        );
        return Ok(());
    }
    if worker_type == WorkerType::Encode && card.model_type.is_empty() {
        anyhow::ensure!(
            card.model_input == ModelInput::Tokens,
            "surface-less encode workers must use token input"
        );
        return Ok(());
    }

    let supported = card.model_type.is_empty()
        || card.model_input == ModelInput::Text
        || (card.model_input == ModelInput::Tokens
            && (card.model_type.supports_chat()
                || card.model_type.supports_completions()
                || card.model_type.supports_embedding()))
        || (card.model_input == ModelInput::Tensor && card.model_type.supports_tensor());
    anyhow::ensure!(
        supported,
        "unsupported model configuration: {} with {} input",
        card.model_type,
        card.model_input.as_str()
    );
    Ok(())
}

fn materialization_fingerprint(
    card: &ModelDeploymentCard,
    default_router_config: &RouterConfig,
) -> anyhow::Result<String> {
    let effective_router = card.router_config.as_ref().unwrap_or(default_router_config);
    let mut value = serde_json::to_value(card)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("model card must serialize as an object"))?;
    object.insert(
        "worker_type".to_string(),
        serde_json::to_value(effective_worker_type(card.worker_type, card.model_type))?,
    );
    object.remove("router_config");
    let normalized: ModelDeploymentCard = serde_json::from_value(value)?;

    let mut bytes = normalized.mdcsum().as_bytes().to_vec();
    let mut router_value = serde_json::to_value(effective_router)?;
    canonicalize_json(&mut router_value);
    bytes.extend(serde_json::to_vec(&router_value)?);
    Ok(blake3::hash(&bytes).to_string())
}

fn effective_router_config<'a>(
    worker_config: Option<&'a RouterConfig>,
    frontend_config: &'a RouterConfig,
) -> Cow<'a, RouterConfig> {
    let Some(worker_config) = worker_config else {
        return Cow::Borrowed(frontend_config);
    };

    let mut effective = worker_config.clone();
    effective.kv_router_config.router_prefill_policy = frontend_config
        .kv_router_config
        .router_prefill_policy
        .clone();
    effective.kv_router_config.router_decode_policy = frontend_config
        .kv_router_config
        .router_decode_policy
        .clone();
    Cow::Owned(effective)
}

fn validate_selector_worker_role(
    card: &ModelDeploymentCard,
    require_typed_worker_role: bool,
) -> anyhow::Result<()> {
    if require_typed_worker_role && card.worker_type.is_none() {
        anyhow::bail!(
            "custom worker-selection policies require model cards with an explicit worker_type"
        );
    }
    Ok(())
}

fn lora_projection_fingerprint(card: &ModelDeploymentCard) -> anyhow::Result<String> {
    let mut value = serde_json::json!({
        "display_name": &card.display_name,
        "aliases": &card.aliases,
        "lora": &card.lora,
        "base_capacity": card.runtime_config.max_gpu_lora_count,
    });
    canonicalize_json(&mut value);
    Ok(blake3::hash(&serde_json::to_vec(&value)?).to_string())
}

fn canonicalize_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            let mut entries = std::mem::take(object).into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            for (key, mut value) in entries {
                canonicalize_json(&mut value);
                object.insert(key, value);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                canonicalize_json(value);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_model::runtime_config::VLLM_INFERENCE_V1_GENERATE_CAPABILITY;
    use crate::model_card::ModelDeploymentCard;
    use dynamo_runtime::engine::AsyncEngine;
    use dynamo_runtime::pipeline::Error;

    fn test_endpoint_id(name: &str) -> EndpointId {
        EndpointId {
            namespace: "ns1".to_string(),
            component: "workers".to_string(),
            name: name.to_string(),
        }
    }

    #[test]
    fn generate_requires_enabled_matching_worker_capability() {
        const OTHER_GENERATE_CAPABILITY: &str = "other_generate";
        let mut card = ModelDeploymentCard::with_name_only("model");
        card.model_type = ModelType::Chat | ModelType::Completions;
        card.runtime_config
            .set_engine_specific(VLLM_INFERENCE_V1_GENERATE_CAPABILITY, true)
            .unwrap();

        assert!(supports_enabled_engine_generate(
            &card,
            &[VLLM_INFERENCE_V1_GENERATE_CAPABILITY]
        ));
        assert!(!supports_enabled_engine_generate(&card, &[]));
        assert!(!supports_enabled_engine_generate(
            &card,
            &[OTHER_GENERATE_CAPABILITY]
        ));
    }

    #[tokio::test]
    async fn tokenizer_fallback_override_applies_to_discovered_card() {
        use dynamo_runtime::{Runtime, distributed::DistributedConfig};

        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let mut watcher = ModelWatcher::new(
            drt,
            Arc::new(ModelManager::new()),
            RouterConfig::default(),
            0,
            None,
            None,
            None,
            Arc::new(Metrics::new()),
        );
        watcher.set_tokenizer_fallback_enabled(Some(false));

        let mut card = ModelDeploymentCard::with_name_only("strict-tokenizer");
        card.runtime_config.tokenizer_fallback_enabled = Some(true);
        let instance = DiscoveryInstance::Model {
            namespace: "ns1".to_string(),
            component: "workers".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 1,
            card_json: serde_json::to_value(card).unwrap(),
            model_suffix: None,
        };

        let desired = watcher
            .normalize(instance, &NamespaceFilter::Global)
            .unwrap()
            .unwrap();
        assert_eq!(
            desired.card.runtime_config.tokenizer_fallback_enabled,
            Some(false)
        );
    }

    #[tokio::test]
    async fn text_pooling_family_preserves_chat_engine() {
        use dynamo_runtime::{Runtime, distributed::DistributedConfig};

        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let manager = Arc::new(ModelManager::new());
        let watcher = Arc::new(ModelWatcher::new(
            drt,
            manager.clone(),
            RouterConfig::default(),
            0,
            None,
            None,
            None,
            Arc::new(Metrics::new_with_prefix(Some(
                "watcher_mixed_text_test".to_string(),
            ))),
        ));
        let mcid = ModelCardInstanceId {
            namespace: "mixed-text-ns".to_string(),
            component: "workers".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 1,
            model_suffix: None,
        };
        let mut card = ModelDeploymentCard::with_name_only("mixed-text-model");
        card.model_input = ModelInput::Text;
        card.model_type = ModelType::Chat | ModelType::Classify | ModelType::Pooling;
        card.worker_type = Some(WorkerType::Aggregated);

        let endpoint_id = model_card_endpoint_id(&mcid);
        let key = GroupKey {
            model_name: card.name().to_string(),
            worker_set_key: worker_set_key(&endpoint_id, card.model_type, card.worker_type),
        };
        let desired = DesiredInstance {
            key: mcid.to_path(),
            mcid,
            endpoint_id,
            fingerprint: materialization_fingerprint(&card, &RouterConfig::default()).unwrap(),
            projection_fingerprint: lora_projection_fingerprint(&card).unwrap(),
            card,
            group_key: key.clone(),
        };
        let spec = GroupSpec {
            key,
            fingerprint: desired.fingerprint.clone(),
            generation: 1,
            representative: desired.clone(),
        };
        let (_admission_tx, admission_rx) = tokio::sync::watch::channel(vec![1]);
        let prepared = watcher
            .prepare_worker_set(&spec, admission_rx, CancellationToken::new())
            .await
            .unwrap();
        watcher
            .commit_group(&spec, prepared, &[desired], &[])
            .unwrap();

        let model = manager.get_model("mixed-text-model").unwrap();
        assert!(model.has_chat_engine());
        assert!(model.has_classify_engine());
        assert!(model.has_pooling_engine());
    }

    #[test]
    fn test_is_model_type_list_empty_on_empty_manager() {
        let mm = ModelManager::new();
        assert!(is_model_type_list_empty(&mm, ModelType::Chat));
        assert!(is_model_type_list_empty(&mm, ModelType::Completions));
        assert!(is_model_type_list_empty(&mm, ModelType::Embedding));
        assert!(is_model_type_list_empty(&mm, ModelType::Images));
        assert!(is_model_type_list_empty(&mm, ModelType::Audios));
        assert!(is_model_type_list_empty(&mm, ModelType::Videos));
        assert!(is_model_type_list_empty(&mm, ModelType::TensorBased));
        assert!(is_model_type_list_empty(&mm, ModelType::Realtime));
        assert!(is_model_type_list_empty(&mm, ModelType::Classify));
        assert!(is_model_type_list_empty(&mm, ModelType::Pooling));
    }

    #[test]
    fn removal_cards_contain_only_the_empty_model_type() {
        let mm = ModelManager::new();
        let mut card = ModelDeploymentCard::with_name_only("model");
        card.model_type = ModelType::Classify | ModelType::Pooling;

        let removed_cards = removed_model_cards(&mm, &card);
        assert_eq!(removed_cards.len(), 2);
        assert!(
            removed_cards
                .iter()
                .any(|card| card.model_type == ModelType::Classify)
        );
        assert!(
            removed_cards
                .iter()
                .any(|card| card.model_type == ModelType::Pooling)
        );
        assert!(
            removed_cards
                .iter()
                .all(|card| card.model_type.bits().count_ones() == 1)
        );
    }

    #[test]
    fn test_is_model_type_list_empty_realtime_after_register() {
        let mm = ModelManager::new();
        let engine = std::sync::Arc::new(crate::engines::EchoBidirectionalEngine);
        mm.add_realtime_model("rt-echo", "0", engine).unwrap();
        assert!(!is_model_type_list_empty(&mm, ModelType::Realtime));
    }

    /// Stand-in engine for registration-only tests; never invoked.
    struct UncalledEngine;

    #[async_trait::async_trait]
    impl
        AsyncEngine<
            SingleIn<NvCreatePoolingRequest>,
            ManyOut<Annotated<NvCreatePoolingResponse>>,
            Error,
        > for UncalledEngine
    {
        async fn generate(
            &self,
            _request: SingleIn<NvCreatePoolingRequest>,
        ) -> Result<ManyOut<Annotated<NvCreatePoolingResponse>>, Error> {
            anyhow::bail!("engine is never invoked by this test")
        }
    }

    #[async_trait::async_trait]
    impl
        AsyncEngine<
            SingleIn<NvCreateClassifyRequest>,
            ManyOut<Annotated<NvCreateClassifyResponse>>,
            Error,
        > for UncalledEngine
    {
        async fn generate(
            &self,
            _request: SingleIn<NvCreateClassifyRequest>,
        ) -> Result<ManyOut<Annotated<NvCreateClassifyResponse>>, Error> {
            anyhow::bail!("engine is never invoked by this test")
        }
    }

    /// Removing one model of a type must not retract the endpoint for the
    /// models of that type that are still registered: the frontend maps a
    /// `ModelUpdate::Removed` card onto process-wide endpoint flags, so an
    /// over-eager removal card would 404 the surviving models' endpoint.
    #[test]
    fn removing_one_model_keeps_the_endpoint_for_surviving_models() {
        let mm = ModelManager::new();
        mm.add_pooling_model("model-a", "ck-a", std::sync::Arc::new(UncalledEngine))
            .unwrap();
        mm.add_pooling_model("model-b", "ck-b", std::sync::Arc::new(UncalledEngine))
            .unwrap();
        mm.add_classify_model("model-a", "ck-a", std::sync::Arc::new(UncalledEngine))
            .unwrap();
        mm.add_classify_model("model-b", "ck-b", std::sync::Arc::new(UncalledEngine))
            .unwrap();

        let mut card = ModelDeploymentCard::with_name_only("model-a");
        card.model_type = ModelType::Classify | ModelType::Pooling;

        // Mirrors `handle_delete`, which drops the model from the manager
        // before computing the removal cards.
        mm.remove_model("model-a");
        assert!(
            removed_model_cards(&mm, &card).is_empty(),
            "removing model-a must emit no removal card while model-b is still registered"
        );

        // The last model of each type going away must still retract both.
        mm.remove_model("model-b");
        let removed = removed_model_cards(&mm, &card);
        assert_eq!(removed.len(), 2);
        assert!(
            removed
                .iter()
                .any(|card| card.model_type == ModelType::Classify)
        );
        assert!(
            removed
                .iter()
                .any(|card| card.model_type == ModelType::Pooling)
        );
    }

    #[test]
    fn test_realtime_in_all_model_types() {
        assert!(ALL_MODEL_TYPES.contains(&ModelType::Realtime));
    }

    #[test]
    fn ws_key_format_per_role() {
        let endpoint_id = test_endpoint_id("generate");
        // Decode worker with Chat | Completions
        let dk = worker_set_key(
            &endpoint_id,
            ModelType::Chat | ModelType::Completions,
            Some(WorkerType::Decode),
        );
        assert_eq!(
            dk,
            r#"["ns1","workers","generate","chat|completions","decode"]"#
        );

        // Prefill worker registers with empty ModelType (no OpenAI surface)
        let pk = worker_set_key(&endpoint_id, ModelType::empty(), Some(WorkerType::Prefill));
        assert_eq!(pk, r#"["ns1","workers","generate","","prefill"]"#);

        // Encode worker, same pattern as prefill
        let ek = worker_set_key(&endpoint_id, ModelType::empty(), Some(WorkerType::Encode));
        assert_eq!(ek, r#"["ns1","workers","generate","","encode"]"#);

        // Aggregated worker
        let ak = worker_set_key(
            &endpoint_id,
            ModelType::Chat | ModelType::Completions,
            Some(WorkerType::Aggregated),
        );
        assert_eq!(
            ak,
            r#"["ns1","workers","generate","chat|completions","aggregated"]"#
        );

        // Legacy card with no worker_type set falls under the compat shim,
        // which renders it as `aggregated` in the key.
        let legacy = worker_set_key(&endpoint_id, ModelType::Chat | ModelType::Completions, None);
        assert_eq!(
            legacy,
            r#"["ns1","workers","generate","chat|completions","aggregated"]"#
        );
    }

    #[test]
    fn ws_key_separates_endpoints_in_same_component() {
        let a = worker_set_key(
            &test_endpoint_id("generate-a"),
            ModelType::Chat,
            Some(WorkerType::Decode),
        );
        let b = worker_set_key(
            &test_endpoint_id("generate-b"),
            ModelType::Chat,
            Some(WorkerType::Decode),
        );
        assert_ne!(a, b);
    }

    #[test]
    fn ws_key_new_and_legacy_prefill_share_a_bucket() {
        let endpoint_id = test_endpoint_id("generate");
        // A NEW prefill worker dual-emits ModelType::Prefill + worker_type=Prefill.
        let new_prefill =
            worker_set_key(&endpoint_id, ModelType::Prefill, Some(WorkerType::Prefill));
        assert_eq!(
            new_prefill,
            r#"["ns1","workers","generate","prefill","prefill"]"#
        );

        // A LEGACY prefill card (ModelType::Prefill marker bit, no worker_type)
        // must resolve to the SAME bucket via effective_worker_type, so old and
        // new prefill workers in one namespace don't split into two buckets.
        let legacy_prefill = worker_set_key(&endpoint_id, ModelType::Prefill, None);
        assert_eq!(
            legacy_prefill,
            r#"["ns1","workers","generate","prefill","prefill"]"#
        );
        assert_eq!(new_prefill, legacy_prefill);
    }

    #[test]
    fn effective_worker_type_resolution() {
        // Explicit worker_type is used verbatim.
        assert_eq!(
            effective_worker_type(Some(WorkerType::Decode), ModelType::Chat),
            WorkerType::Decode
        );
        assert_eq!(
            effective_worker_type(Some(WorkerType::Prefill), ModelType::Prefill),
            WorkerType::Prefill
        );
        // Legacy prefill card (Prefill marker bit, no worker_type) → Prefill.
        assert_eq!(
            effective_worker_type(None, ModelType::Prefill),
            WorkerType::Prefill
        );
        // Any other legacy card → Aggregated.
        assert_eq!(
            effective_worker_type(None, ModelType::Chat | ModelType::Completions),
            WorkerType::Aggregated
        );
        assert_eq!(
            effective_worker_type(None, ModelType::empty()),
            WorkerType::Aggregated
        );
    }

    #[test]
    fn custom_selector_requires_explicit_worker_type() {
        let mut card = ModelDeploymentCard::with_name_only("model");
        assert!(validate_selector_worker_role(&card, false).is_ok());
        assert!(validate_selector_worker_role(&card, true).is_err());

        card.worker_type = Some(WorkerType::Decode);
        assert!(validate_selector_worker_role(&card, true).is_ok());
    }

    #[test]
    fn materialization_fingerprint_normalizes_legacy_prefill_topology() {
        let mut legacy = ModelDeploymentCard::with_name_only("model");
        legacy.model_type = ModelType::Prefill;
        let mut current = legacy.clone();
        current.worker_type = Some(WorkerType::Prefill);
        current.needs = vec![vec![WorkerType::Decode]];

        normalize_legacy_prefill_topology(&mut legacy);
        assert_eq!(legacy.worker_type, Some(WorkerType::Prefill));
        assert_eq!(legacy.needs, vec![vec![WorkerType::Decode]]);

        assert_eq!(
            materialization_fingerprint(&legacy, &RouterConfig::default()).unwrap(),
            materialization_fingerprint(&current, &RouterConfig::default()).unwrap()
        );

        current.runtime_config.max_gpu_lora_count = Some(4);
        current.runtime_config.kv_event_publishing_enabled = Some(true);
        current.runtime_config.data_parallel_start_rank = 4;
        assert_eq!(
            materialization_fingerprint(&legacy, &RouterConfig::default()).unwrap(),
            materialization_fingerprint(&current, &RouterConfig::default()).unwrap()
        );

        current.aliases.push("new-serving-name".to_string());
        assert_ne!(
            materialization_fingerprint(&legacy, &RouterConfig::default()).unwrap(),
            materialization_fingerprint(&current, &RouterConfig::default()).unwrap()
        );

        let mut legacy_wire = ModelDeploymentCard::with_name_only("model");
        legacy_wire.model_type = ModelType::Prefill;
        let mut legacy_wire = serde_json::to_value(&legacy_wire).unwrap();
        let object = legacy_wire.as_object_mut().unwrap();
        object.remove("worker_type");
        object.remove("needs");
        legacy_wire["context_length"] = serde_json::json!(8_192);
        let mut legacy_wire: ModelDeploymentCard = serde_json::from_value(legacy_wire).unwrap();
        normalize_legacy_prefill_topology(&mut legacy_wire);
        let mut current_wire = ModelDeploymentCard::with_name_only("model");
        current_wire.model_type = ModelType::Prefill;
        current_wire.worker_type = Some(WorkerType::Prefill);
        current_wire.needs = vec![vec![WorkerType::Decode]];
        current_wire.runtime_config.context_length = Some(8_192);
        assert_eq!(
            materialization_fingerprint(&legacy_wire, &RouterConfig::default()).unwrap(),
            materialization_fingerprint(&current_wire, &RouterConfig::default()).unwrap()
        );
    }

    #[test]
    fn worker_router_config_preserves_frontend_policy_selections() {
        let mut frontend = RouterConfig::default();
        frontend.kv_router_config.router_prefill_policy = Some("frontend-prefill".to_string());
        frontend.kv_router_config.router_decode_policy = Some("frontend-decode".to_string());

        let mut worker = RouterConfig::default();
        worker.kv_router_config.router_temperature = 0.75;
        worker.kv_router_config.router_policy_config = Some("worker-policy.yaml".to_string());

        let effective = effective_router_config(Some(&worker), &frontend);
        assert_eq!(effective.kv_router_config.router_temperature, 0.75);
        assert_eq!(
            effective.kv_router_config.router_policy_config.as_deref(),
            Some("worker-policy.yaml")
        );
        assert_eq!(
            effective.kv_router_config.router_prefill_policy.as_deref(),
            Some("frontend-prefill")
        );
        assert_eq!(
            effective.kv_router_config.router_decode_policy.as_deref(),
            Some("frontend-decode")
        );
        assert!(worker.kv_router_config.router_prefill_policy.is_none());
        assert!(worker.kv_router_config.router_decode_policy.is_none());
    }

    #[tokio::test]
    async fn discovery_normalization_joins_v12_and_current_prefill() {
        use dynamo_runtime::{Runtime, distributed::DistributedConfig};

        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let watcher = ModelWatcher::new(
            drt,
            Arc::new(ModelManager::new()),
            RouterConfig::default(),
            0,
            None,
            None,
            None,
            Arc::new(Metrics::new_with_prefix(Some(
                "watcher_v12_prefill_compat_test".to_string(),
            ))),
        );

        let mut legacy_card = ModelDeploymentCard::with_name_only("model");
        legacy_card.model_type = ModelType::Prefill;
        legacy_card.model_input = ModelInput::Tokens;
        let mut legacy_json = serde_json::to_value(legacy_card).unwrap();
        let legacy_object = legacy_json.as_object_mut().unwrap();
        legacy_object.remove("worker_type");
        legacy_object.remove("needs");

        let mut current_card = ModelDeploymentCard::with_name_only("model");
        current_card.model_type = ModelType::Prefill;
        current_card.model_input = ModelInput::Tokens;
        current_card.worker_type = Some(WorkerType::Prefill);
        current_card.needs = vec![vec![WorkerType::Decode]];

        let instance = |instance_id, card_json| DiscoveryInstance::Model {
            namespace: "ns1".to_string(),
            component: "workers".to_string(),
            endpoint: "generate".to_string(),
            instance_id,
            card_json,
            model_suffix: None,
        };
        let legacy = watcher
            .normalize(instance(1, legacy_json), &NamespaceFilter::Global)
            .unwrap()
            .unwrap();
        let current = watcher
            .normalize(
                instance(2, serde_json::to_value(current_card).unwrap()),
                &NamespaceFilter::Global,
            )
            .unwrap()
            .unwrap();

        assert_eq!(legacy.card.worker_type, Some(WorkerType::Prefill));
        assert_eq!(legacy.card.needs, vec![vec![WorkerType::Decode]]);
        assert_eq!(legacy.group_key, current.group_key);
        assert_eq!(legacy.fingerprint, current.fingerprint);
    }

    #[test]
    fn ws_key_separates_prefill_from_decode_in_same_namespace() {
        let endpoint_id = test_endpoint_id("generate");
        // Prefill and decode in the same deployment namespace must hash to
        // distinct keys so they live in separate WorkerSet buckets.
        let decode = worker_set_key(
            &endpoint_id,
            ModelType::Chat | ModelType::Completions,
            Some(WorkerType::Decode),
        );
        let prefill = worker_set_key(&endpoint_id, ModelType::empty(), Some(WorkerType::Prefill));
        assert_ne!(decode, prefill);
    }

    #[test]
    fn worker_set_key_encode_and_aggregated_coexist_in_same_namespace() {
        let endpoint_id = EndpointId {
            namespace: "dynamo".to_string(),
            ..test_endpoint_id("generate")
        };
        // Regression for the Encode/Aggregated key collision: Encode and
        // Aggregated workers in the same namespace MUST map to different keys,
        // so both can register without an MDC checksum mismatch. Under the
        // role-in-key scheme, an Encode worker registers surface-less
        // (ModelType::empty()) and lands in `{ns}::encode`, while Aggregated
        // keeps its `{ns}:chat|completions:aggregated` bucket.
        let agg_key = worker_set_key(
            &endpoint_id,
            ModelType::Chat | ModelType::Completions,
            Some(WorkerType::Aggregated),
        );
        let enc_key = worker_set_key(&endpoint_id, ModelType::empty(), Some(WorkerType::Encode));
        assert_ne!(agg_key, enc_key);
        assert_eq!(
            agg_key,
            r#"["dynamo","workers","generate","chat|completions","aggregated"]"#
        );
        assert_eq!(enc_key, r#"["dynamo","workers","generate","","encode"]"#);
    }
}
