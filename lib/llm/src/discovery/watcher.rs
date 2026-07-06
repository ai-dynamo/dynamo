// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;
use tokio::sync::{Mutex, Notify, mpsc::Sender};
use tokio::task::{JoinHandle, JoinSet};

use anyhow::Context as _;
use dashmap::{DashMap, DashSet};
use dynamo_http_server::metrics::Metrics;
use dynamo_kv_router::PrefillLoadEstimator;
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

use dynamo_renderer::PromptFormatter;

use crate::{
    backend::Backend,
    discovery::{KvWorkerMonitor, WORKER_TYPE_DECODE, WorkerSet},
    entrypoint::{self, ChatEngineFactoryCallback, RouterConfig},
    kv_router::{EncoderRouter, PrefillRouter},
    local_model::runtime_config::{TokenizerBackend, VLLM_INFERENCE_V1_GENERATE_CAPABILITY},
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
            completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
            embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
            images::{NvCreateImageRequest, NvImagesResponse},
            videos::{NvCreateVideoRequest, NvVideosResponse},
        },
        tensor::{NvCreateTensorRequest, NvCreateTensorResponse},
    },
    types::generic::realtime::{RealtimeClientEvent, RealtimeServerEvent},
    worker_type::WorkerType,
};

use super::ModelManager;
use crate::namespace::NamespaceFilter;

const RECONCILIATION_INTERVAL: Duration = Duration::from_secs(30);

/// Constructs the WorkerSet storage key as `{namespace}:{model_type}:{worker_type}`.
///
/// Each `(namespace, model_type, worker_type)` combination gets its own
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
    namespace: &str,
    model_type: ModelType,
    worker_type: Option<WorkerType>,
) -> String {
    let mt = model_type.as_vec().join("|");
    let wt = effective_worker_type(worker_type, model_type).as_str();
    format!("{}:{}:{}", namespace, mt, wt)
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

/// A source card is fully represented locally only after both the per-instance
/// card and the shared WorkerSet have been recorded. `handle_put` can save the
/// card before later pipeline construction fails, so either check alone would
/// allow a failed registration to escape reconciliation.
fn is_registration_complete(
    manager: &ModelManager,
    mcid: &ModelCardInstanceId,
    card: &ModelDeploymentCard,
) -> bool {
    let ws_key = worker_set_key(&mcid.namespace, card.model_type, card.worker_type);
    manager
        .get_model_card(&mcid.to_path())
        .is_some_and(|saved| saved.name() == card.name() && saved.mdcsum() == card.mdcsum())
        && manager.get_model(card.name()).is_some_and(|model| {
            model.has_worker_set(&ws_key) && model.is_checksum_compatible(&ws_key, card.mdcsum())
        })
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

fn supports_vllm_generate(card: &ModelDeploymentCard) -> bool {
    matches!(
        card.runtime_config
            .runtime_data
            .get(VLLM_INFERENCE_V1_GENERATE_CAPABILITY),
        Some(serde_json::Value::Bool(true))
    )
}

const ENCODER_RESULT_HANDOFF_CAPABILITY: &str = "encoder_result_handoff";

fn supports_encoder_result_handoff(card: &ModelDeploymentCard) -> bool {
    matches!(
        card.runtime_config
            .runtime_data
            .get(ENCODER_RESULT_HANDOFF_CAPABILITY),
        Some(serde_json::Value::Bool(true))
    )
}

// Generate's opaque request state is not yet verified for migration replay.
const GENERATE_MIGRATION_LIMIT: u32 = 0;

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
    worker_type.unwrap_or_else(|| {
        if model_type.supports_prefill() {
            WorkerType::Prefill
        } else {
            WorkerType::Aggregated
        }
    })
}

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
    migration_max_seq_len: Option<u32>,
    notify_on_model: Notify,
    model_update_tx: Option<Sender<ModelUpdate>>,
    chat_engine_factory: Option<ChatEngineFactoryCallback>,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    metrics: Arc<Metrics>,
    /// Guards against concurrent pipeline construction for the same (model, namespace).
    registering_worker_sets: DashSet<String>,
    /// Wakes tasks blocked in `recover_concurrent_registration` when a
    /// `RegistrationGuard` drops (i.e. a registration completes or panics).
    registration_notify: Notify,
    /// Tracks in-flight `handle_put` tasks by instance path so a later operation
    /// can cancel a superseded registration.
    pending_puts: DashMap<String, PendingPut>,
    /// Serializes state changes for one model-card instance. The generation
    /// fence preserves discovery-event order even when spawned tasks acquire
    /// the per-instance lock in a different order.
    instance_operations: DashMap<String, Arc<InstanceOperation>>,
    /// Maps an MDC instance path to the LoRA adapter name recorded in the pre-spawn state-tracker
    /// addition, so a removal whose model card was never durably saved (handle_put failed before
    /// save) can still remove exactly that adapter from the tracker instead of leaving phantom
    /// state or wiping a live worker (R4-2 / RR3-3).
    pending_lora_adds: DashMap<String, String>,
    /// Frontend's `--model-path`. Threaded into `download_config` so
    /// `file://` slots can fall back here when the worker's path is
    /// unreachable on this host.
    local_model_path: Option<PathBuf>,
    /// Frontend-level tokenizer backend override for discovered model cards.
    tokenizer_backend: Option<TokenizerBackend>,
    /// Whether the frontend configured the vLLM-compatible Generate API.
    /// Keep the raw Generate pipeline out of non-HTTP and default-off paths.
    generate_engine_enabled: bool,
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
    } else {
        true
    }
}

/// RAII guard that removes a key from a `DashSet` on drop and wakes any tasks
/// waiting for the registration to finish via the shared [`Notify`].
/// Ensures `registering_worker_sets` is cleaned up even if the registration
/// task panics, preventing permanent poisoning of the registration key.
struct RegistrationGuard<'a> {
    set: &'a DashSet<String>,
    key: String,
    notify: &'a Notify,
}

struct PendingPut {
    generation: u64,
    handle: JoinHandle<()>,
}

#[derive(Default)]
struct InstanceOperation {
    generation: AtomicU64,
    lock: Mutex<()>,
}

impl InstanceOperation {
    fn current_generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn begin(&self) -> u64 {
        self.generation
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1)
    }

    fn reserve_after(&self, expected: u64) -> Option<u64> {
        let next = expected.wrapping_add(1);
        self.generation
            .compare_exchange(expected, next, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| next)
    }

    fn is_current(&self, generation: u64) -> bool {
        self.current_generation() == generation
    }
}

enum DeleteOutcome {
    Superseded,
    Processed(Option<String>),
}

impl Drop for RegistrationGuard<'_> {
    fn drop(&mut self) {
        self.set.remove(&self.key);
        self.notify.notify_waiters();
    }
}

impl ModelWatcher {
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
        Self {
            manager: model_manager,
            drt: runtime,
            router_config,
            migration_limit,
            migration_max_seq_len,
            notify_on_model: Notify::new(),
            model_update_tx: None,
            chat_engine_factory,
            prefill_load_estimator,
            metrics,
            registering_worker_sets: DashSet::new(),
            registration_notify: Notify::new(),
            pending_puts: DashMap::new(),
            instance_operations: DashMap::new(),
            pending_lora_adds: DashMap::new(),
            local_model_path: None,
            tokenizer_backend: None,
            generate_engine_enabled: false,
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

    pub fn set_generate_engine_enabled(&mut self, enabled: bool) {
        self.generate_engine_enabled = enabled;
    }

    fn apply_tokenizer_backend_override(&self, card: &mut ModelDeploymentCard) {
        if let Some(tokenizer_backend) = self.tokenizer_backend {
            card.runtime_config.tokenizer_backend = Some(tokenizer_backend);
        }
    }

    fn instance_operation(&self, key: &str) -> Arc<InstanceOperation> {
        self.instance_operations
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(InstanceOperation::default()))
            .clone()
    }

    fn begin_instance_operation(&self, key: &str) -> (Arc<InstanceOperation>, u64) {
        let operation = self.instance_operation(key);
        let generation = operation.begin();
        (operation, generation)
    }

    fn observe_instance_operation(&self, key: &str) -> (Arc<InstanceOperation>, u64) {
        let operation = self.instance_operation(key);
        let generation = operation.current_generation();
        (operation, generation)
    }

    fn prune_instance_operation(&self, key: &str, operation: &Arc<InstanceOperation>) {
        self.instance_operations.remove_if(key, |_, current| {
            Arc::ptr_eq(current, operation) && Arc::strong_count(current) == 2
        });
    }

    fn seed_lora_state_for_put(&self, mcid: &ModelCardInstanceId, card: &ModelDeploymentCard) {
        use crate::kv_router::protocols::WorkerWithDpRank;

        let worker = WorkerWithDpRank::new(mcid.instance_id, 0);
        if let Some(adapter_name) =
            seed_lora_state_from_card(self.manager.lora_state_tracker(), worker, card)
        {
            self.pending_lora_adds.insert(mcid.to_path(), adapter_name);
        }
    }

    async fn handle_put_if_current(
        &self,
        mcid: &ModelCardInstanceId,
        card: &mut ModelDeploymentCard,
        operation: &InstanceOperation,
        generation: u64,
    ) -> Option<anyhow::Result<()>> {
        let _guard = operation.lock.lock().await;
        if !operation.is_current(generation) {
            return None;
        }

        self.seed_lora_state_for_put(mcid, card);
        Some(self.handle_put(mcid, card).await)
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

    /// Common watch logic with optional namespace filtering.
    ///
    /// Takes `Arc<Self>` so that each `handle_put` call can be spawned into its own
    /// tokio task, preventing a slow HuggingFace config download for one model from
    /// blocking discovery events for all subsequent models.
    pub async fn watch(
        self: Arc<Self>,
        mut discovery_stream: DiscoveryStream,
        namespace_filter: NamespaceFilter,
    ) {
        let mut reconciliation = tokio::time::interval(RECONCILIATION_INTERVAL);
        reconciliation.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // list_and_watch supplies the initial snapshot. Consume the immediate
        // first tick so reconciliation begins after one complete interval.
        reconciliation.tick().await;
        let mut reconciliation_tasks = JoinSet::new();

        loop {
            let result = tokio::select! {
                result = discovery_stream.next() => {
                    let Some(result) = result else {
                        break;
                    };
                    result
                }
                _ = reconciliation.tick(), if reconciliation_tasks.is_empty() => {
                    let watcher = Arc::clone(&self);
                    let namespace_filter = namespace_filter.clone();
                    reconciliation_tasks.spawn(async move {
                        if let Err(err) = watcher.reconcile(&namespace_filter).await {
                            tracing::warn!(
                                error = format!("{err:#}"),
                                "Frontend model registration reconciliation failed"
                            );
                        }
                    });
                    continue;
                }
                result = reconciliation_tasks.join_next(), if !reconciliation_tasks.is_empty() => {
                    if let Some(Err(err)) = result {
                        tracing::warn!(
                            error = %err,
                            "Frontend model registration reconciliation task failed"
                        );
                    }
                    continue;
                }
            };
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

                    self.apply_tokenizer_backend_override(&mut card);

                    // If a WorkerSet already exists for this (model, namespace, type),
                    // validate that the new worker's checksum matches. Different
                    // WorkerSets (different namespaces) are allowed to have different checksums to support rolling updates.
                    let ws_key = worker_set_key(&mcid.namespace, card.model_type, card.worker_type);
                    if let Some(model) = self.manager.get_model(card.name())
                        && !model.is_checksum_compatible(&ws_key, card.mdcsum())
                    {
                        tracing::error!(
                            model_name = card.name(),
                            namespace = mcid.namespace,
                            new_checksum = card.mdcsum(),
                            "Checksum for new worker does not match existing WorkerSet's checksum. \
                             Drain all old workers in this namespace before deploying a new version."
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

                    // Spawn each handle_put into its own task so that a slow
                    // HuggingFace config download for one model cannot block
                    // discovery events for all subsequent models.
                    //
                    // The per-instance operation generation preserves event
                    // order if this task is scheduled after a later operation.
                    let instance_key = mcid.to_path();
                    let task_key = instance_key.clone();
                    let (operation, generation) = self.begin_instance_operation(&instance_key);
                    let watcher = Arc::clone(&self);
                    let handle = tokio::spawn(async move {
                        match watcher
                            .handle_put_if_current(&mcid, &mut card, &operation, generation)
                            .await
                        {
                            Some(Ok(())) => {
                                tracing::info!(
                                    model_name = card.name(),
                                    namespace = mcid.namespace,
                                    "added model"
                                );
                                watcher.notify_on_model.notify_waiters();
                            }
                            Some(Err(err)) => {
                                tracing::error!(
                                    model_name = card.name(),
                                    namespace = mcid.namespace,
                                    error = format!("{err:#}"),
                                    "Error adding model from discovery",
                                );
                            }
                            None => {
                                tracing::debug!(
                                    key = %task_key,
                                    generation,
                                    "Skipping superseded model registration"
                                );
                            }
                        }
                        watcher.prune_instance_operation(&task_key, &operation);
                        // Note: we intentionally do NOT remove from pending_puts here.
                        // Only the watch loop (on duplicate events) and handle_delete
                        // manage pending_puts, avoiding a race where a completed task's
                        // cleanup could remove a newer task's entry.
                    });
                    // If a duplicate Added event arrives while the first task is still
                    // in-flight, abort the old task to cancel redundant work.
                    //
                    // `instance_key` is `mcid.to_path()` = "{ns}/{component}/{endpoint}/{instance_id:x}",
                    // so this is keyed per-worker-instance, NOT per-model. Two different workers
                    // registering the same model produce two different keys and run independently.
                    // The only case that hits this branch is the etcd watch replaying the same
                    // worker's Added event (reconnect or re-sync) — where cancelling the earlier
                    // redundant task is exactly what we want.
                    if let Some((_, old_put)) = self.pending_puts.remove(&instance_key) {
                        old_put.handle.abort();
                    }
                    self.pending_puts
                        .insert(instance_key, PendingPut { generation, handle });
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

                    let instance_key = model_card_instance_id.to_path();
                    let (operation, generation) = self.begin_instance_operation(&instance_key);
                    match self
                        .handle_delete(
                            model_card_instance_id,
                            &namespace_filter,
                            operation,
                            generation,
                        )
                        .await
                    {
                        Ok(DeleteOutcome::Processed(Some(model_name))) => {
                            tracing::info!(model_name, "removed model");
                        }
                        Ok(DeleteOutcome::Processed(None)) => {
                            // There are other instances running this model, nothing to do
                        }
                        Ok(DeleteOutcome::Superseded) => {}
                        Err(e) => {
                            tracing::error!(error = %e, "error removing model");
                        }
                    }
                }
            }
        }

        reconciliation_tasks.abort_all();
        while reconciliation_tasks.join_next().await.is_some() {}
    }

    /// Compare the local frontend registry with a fresh discovery snapshot.
    /// Missing or incomplete source cards are retried through `handle_put`; a
    /// locally recorded card absent from the snapshot is removed through
    /// `handle_delete`. The snapshot is sorted so repeated failures are logged
    /// and retried in a deterministic order.
    async fn reconcile(self: &Arc<Self>, namespace_filter: &NamespaceFilter) -> anyhow::Result<()> {
        // Snapshot local keys and their operation generations before awaiting
        // discovery. A registration that starts after this point advances its
        // generation and prevents this snapshot from deleting the fresh card.
        let local_entries = self
            .manager
            .get_model_card_keys()
            .into_iter()
            .map(|key| {
                let (operation, generation) = self.observe_instance_operation(&key);
                (key, operation, generation)
            })
            .collect::<Vec<_>>();

        let mut instances = self.drt.discovery().list(DiscoveryQuery::AllModels).await?;
        instances.sort_by_key(|instance| {
            model_card_instance_id(instance)
                .map(|mcid| mcid.to_path())
                .unwrap_or_default()
        });

        let mut desired_keys = HashSet::with_capacity(instances.len());
        let mut retry_count = 0usize;

        for instance in instances {
            let mcid = match model_card_instance_id(&instance) {
                Ok(mcid) => mcid,
                Err(err) => {
                    tracing::error!(
                        error = format!("{err:#}"),
                        "Unexpected discovery entry during model reconciliation"
                    );
                    continue;
                }
            };
            if !namespace_filter.matches(&mcid.namespace) {
                continue;
            }

            // Record the source key before deserializing the card. A malformed
            // source entry must not cause a valid local entry with the same key
            // to be treated as stale and deleted.
            desired_keys.insert(mcid.to_path());

            let mut card = match instance.deserialize_model::<ModelDeploymentCard>() {
                Ok(card) => card,
                Err(err) => {
                    tracing::error!(
                        %err,
                        instance_id = mcid.instance_id,
                        "Failed to deserialize model card during reconciliation"
                    );
                    continue;
                }
            };
            self.apply_tokenizer_backend_override(&mut card);

            if !is_registration_complete(&self.manager, &mcid, &card)
                && self.retry_model_registration(mcid, card).await
            {
                retry_count += 1;
            }
        }

        let mut stale_entries = Vec::new();
        for (key, operation, generation) in local_entries {
            let mcid = match ModelCardInstanceId::from_path(&key) {
                Ok(mcid) => mcid,
                Err(err) => {
                    tracing::warn!(%err, key, "Ignoring malformed local model-card key");
                    continue;
                }
            };
            if namespace_filter.matches(&mcid.namespace) && !desired_keys.contains(&key) {
                stale_entries.push((key, mcid, operation, generation));
            }
        }
        stale_entries.sort_by(|left, right| left.0.cmp(&right.0));
        let stale_entry_count = stale_entries.len();

        let mut removed_stale_count = 0usize;
        let mut superseded_stale_count = 0usize;
        for (key, mcid, operation, snapshot_generation) in stale_entries {
            let Some(delete_generation) = operation.reserve_after(snapshot_generation) else {
                superseded_stale_count += 1;
                tracing::debug!(
                    key = %key,
                    snapshot_generation,
                    current_generation = operation.current_generation(),
                    "Skipping stale cleanup superseded after reconciliation snapshot"
                );
                continue;
            };
            match self
                .handle_delete(&mcid, namespace_filter, operation, delete_generation)
                .await
            {
                Ok(DeleteOutcome::Processed(_)) => removed_stale_count += 1,
                Ok(DeleteOutcome::Superseded) => superseded_stale_count += 1,
                Err(err) => {
                    tracing::error!(
                        key = %key,
                        error = format!("{err:#}"),
                        "Error removing stale model during reconciliation"
                    );
                }
            }
        }

        tracing::debug!(
            expected_model_cards = desired_keys.len(),
            retried_model_cards = retry_count,
            removed_stale_model_cards = removed_stale_count,
            superseded_stale_model_cards = superseded_stale_count,
            failed_stale_model_cards =
                stale_entry_count.saturating_sub(removed_stale_count + superseded_stale_count),
            "Completed frontend model registration reconciliation"
        );
        Ok(())
    }

    /// Start one reconciliation retry unless the original event-driven
    /// registration for this instance is still running. Completed failed tasks
    /// remain in `pending_puts`, so they are replaced here by the retry task.
    async fn retry_model_registration(
        self: &Arc<Self>,
        mcid: ModelCardInstanceId,
        mut card: ModelDeploymentCard,
    ) -> bool {
        let instance_key = mcid.to_path();
        if self
            .pending_puts
            .get(&instance_key)
            .is_some_and(|pending| !pending.handle.is_finished())
        {
            return false;
        }

        let ws_key = worker_set_key(&mcid.namespace, card.model_type, card.worker_type);
        if let Some(model) = self.manager.get_model(card.name())
            && !model.is_checksum_compatible(&ws_key, card.mdcsum())
        {
            tracing::error!(
                model_name = card.name(),
                namespace = mcid.namespace,
                new_checksum = card.mdcsum(),
                "Reconciliation found a model-card checksum that does not match the existing WorkerSet. \
                 Drain all old workers in this namespace before deploying a new version."
            );
            return false;
        }

        tracing::warn!(
            model_name = card.name(),
            namespace = mcid.namespace,
            instance_key,
            "Retrying incomplete frontend model registration from discovery snapshot"
        );

        let task_key = instance_key.clone();
        let (operation, generation) = self.begin_instance_operation(&instance_key);
        let watcher = Arc::clone(self);
        let handle = tokio::spawn(async move {
            match watcher
                .handle_put_if_current(&mcid, &mut card, &operation, generation)
                .await
            {
                Some(Ok(())) => {
                    tracing::info!(
                        model_name = card.name(),
                        namespace = mcid.namespace,
                        "Reconciled model registration"
                    );
                    watcher.notify_on_model.notify_waiters();
                }
                Some(Err(err)) => {
                    tracing::error!(
                        model_name = card.name(),
                        namespace = mcid.namespace,
                        error = format!("{err:#}"),
                        "Model registration reconciliation retry failed"
                    );
                }
                None => {
                    tracing::debug!(
                        key = %task_key,
                        generation,
                        "Skipping superseded model registration retry"
                    );
                }
            }
            watcher.prune_instance_operation(&task_key, &operation);
        });

        if let Some((_, old_put)) = self.pending_puts.remove(&instance_key) {
            old_put.handle.abort();
        }
        self.pending_puts
            .insert(instance_key, PendingPut { generation, handle });
        true
    }

    /// Handle a worker removal. Cleans up per-namespace WorkerSets and the Model itself
    /// when no instances remain. Returns the model name if the entire Model was removed.
    async fn handle_delete(
        &self,
        mcid: &ModelCardInstanceId,
        namespace_filter: &NamespaceFilter,
        operation: Arc<InstanceOperation>,
        generation: u64,
    ) -> anyhow::Result<DeleteOutcome> {
        let key = mcid.to_path();

        let operation_guard = match tokio::time::timeout(
            Duration::from_secs(60),
            operation.lock.lock(),
        )
        .await
        {
            Ok(guard) => guard,
            Err(_) => {
                if !operation.is_current(generation) {
                    self.prune_instance_operation(&key, &operation);
                    return Ok(DeleteOutcome::Superseded);
                }

                if let Some((_, pending)) = self
                    .pending_puts
                    .remove_if(&key, |_, pending| pending.generation < generation)
                {
                    pending.handle.abort();
                    let _ = pending.handle.await;
                }
                tracing::warn!(
                    key = %key,
                    "Timed out waiting for an earlier model registration; aborted it before delete"
                );
                operation.lock.lock().await
            }
        };

        if !operation.is_current(generation) {
            drop(operation_guard);
            self.prune_instance_operation(&key, &operation);
            tracing::debug!(
                key = %key,
                generation,
                current_generation = operation.current_generation(),
                "Skipping superseded model removal"
            );
            return Ok(DeleteOutcome::Superseded);
        }

        // An earlier put either released the operation lock or is waiting on
        // it with an obsolete generation. Cancel its task before mutating the
        // local card and WorkerSet state.
        if let Some((_, pending)) = self
            .pending_puts
            .remove_if(&key, |_, pending| pending.generation < generation)
        {
            pending.handle.abort();
            let _ = pending.handle.await;
        }

        let result = self.handle_delete_serialized(mcid, namespace_filter).await;
        drop(operation_guard);
        self.prune_instance_operation(&key, &operation);
        result.map(DeleteOutcome::Processed)
    }

    async fn handle_delete_serialized(
        &self,
        mcid: &ModelCardInstanceId,
        namespace_filter: &NamespaceFilter,
    ) -> anyhow::Result<Option<String>> {
        let key = mcid.to_path();

        let card = match self.manager.get_model_card(&key) {
            Some(card) => card,
            None => {
                // The card was never durably saved (e.g. an Added event whose handle_put failed
                // before save, after the pre-spawn tracker addition). Reconcile tracker state at
                // the right granularity:
                //   - base worker card (`model_suffix == None`): the worker instance is gone, so
                //     clear it entirely;
                //   - LoRA-adapter card (`model_suffix == Some`): remove ONLY that adapter, using
                //     the name recorded in `pending_lora_adds`, so a still-live worker's other
                //     adapters/capacity are untouched (RR3-3 / R4-2).
                use crate::kv_router::protocols::WorkerWithDpRank;
                let worker = WorkerWithDpRank::new(mcid.instance_id, 0);
                if mcid.model_suffix.is_none() {
                    self.manager
                        .lora_state_tracker()
                        .handle_worker_removal(worker);
                    // Sweep any pending fallback entries for this worker's LoRA cards so they
                    // can't linger if their own remove events never arrive (RF-2).
                    let prefix = format!("{}/", mcid.to_path());
                    self.pending_lora_adds
                        .retain(|k, _| !k.starts_with(&prefix));
                } else if let Some((_, lora_name)) = self.pending_lora_adds.remove(&key) {
                    self.manager
                        .lora_state_tracker()
                        .handle_mdc_removal(worker, &lora_name);
                }
                tracing::warn!(
                    key = %key,
                    "ModelDeploymentCard absent during removal; reconciled LoRA tracker state from pending add"
                );
                return Ok(None);
            }
        };
        let model_name = card.name().to_string();

        // Complete the only fallible discovery query before removing local
        // state. If discovery is temporarily unavailable, retaining the card
        // lets the next reconciliation pass retry this stale entry instead of
        // losing the key while leaving its WorkerSet behind.
        let active_instances = self
            .cards_for_model_with_endpoints(&model_name, namespace_filter)
            .await
            .with_context(|| model_name.clone())?;

        let card = match self.manager.remove_model_card(&key) {
            Some(card) => card,
            None => {
                tracing::debug!(
                    key = %key,
                    "ModelDeploymentCard was removed by concurrent cleanup"
                );
                return Ok(None);
            }
        };

        // Feed the LoRA state tracker now that any in-flight handle_put has completed and the
        // card is available (N2 — avoids the race where a Removed event outran the add). A LoRA
        // adapter card unregisters just that adapter; the base worker card means the worker
        // instance is gone, so drop its capacity + all loaded-LoRA bookkeeping (F4/F5).
        {
            use crate::kv_router::protocols::WorkerWithDpRank;
            let worker = WorkerWithDpRank::new(mcid.instance_id, 0);
            match card.lora {
                Some(ref lora_info) => self
                    .manager
                    .lora_state_tracker()
                    .handle_mdc_removal(worker, &lora_info.name),
                None => self
                    .manager
                    .lora_state_tracker()
                    .handle_worker_removal(worker),
            }
        }
        // The card-based cleanup above is authoritative; drop the pending fallback entry for a
        // LoRA card, or sweep all of this worker's suffixed entries for a base card (RF-2).
        if card.lora.is_some() {
            self.pending_lora_adds.remove(&key);
        } else {
            let prefix = format!("{}/", mcid.to_path());
            self.pending_lora_adds
                .retain(|k, _| !k.starts_with(&prefix));
        }

        let worker_namespace = &mcid.namespace;
        let worker_component = &mcid.component;
        let ws_key = worker_set_key(&mcid.namespace, card.model_type, card.worker_type);

        // Check if instances of the SAME role and component remain in
        // this namespace. In disaggregated deployments, prefill and
        // decode are different components in the same namespace -- so
        // checking only (ns, component) is necessary but not sufficient.
        // Encode workers can share a (ns, component) with Aggregated, so
        // we ALSO require the remaining instance to map to the same
        // computed `ws_key` (which folds in worker_type for Encode). If
        // we used only (ns, component), removing the last Encode
        // instance from a namespace that still has an Aggregated worker
        // in the same component would see "instances exist" and skip
        // remove_worker_set, leaking the Encode WorkerSet forever.
        let component_has_instances = active_instances.iter().any(|(eid, other_card)| {
            eid.namespace == *worker_namespace
                && eid.component == *worker_component
                && worker_set_key(
                    &eid.namespace,
                    other_card.model_type,
                    other_card.worker_type,
                ) == ws_key
        });

        if !component_has_instances {
            // No more workers of this component in this namespace — remove its WorkerSet
            let removed = self.manager.remove_worker_set(&model_name, &ws_key);
            if removed.is_some() {
                tracing::info!(
                    model_name,
                    namespace = %worker_namespace,
                    "Removed WorkerSet (no remaining instances in namespace)"
                );
            }

            // Activator-state cleanup depends on which component just went away.
            //
            // PREFILL teardown (cached endpoint is stale): drop everything for
            // this key and deactivate the decode-side router so requests fall
            // back to aggregated mode.
            //
            // DECODE teardown: keep `PrefillReady` (the cached endpoint is still
            // valid for future decode rebuilds — that's PR 8965's primary
            // contribution) but DO drop any stale `DecodeWaiting(sender)`. The
            // sender pointed at a `oneshot::Receiver` held by the now-dropped
            // PrefillRouter; leaving it in the map causes the next decode
            // rebuild's `register_prefill_router` to find a stale `DecodeWaiting`,
            // return `None`, and produce a WorkerSet with no PrefillRouter at
            // all. The stale-DecodeWaiting cleanup tests cover this rebuild
            // path.
            match card.worker_type {
                Some(WorkerType::Prefill) => {
                    if removed.is_some() {
                        self.manager
                            .remove_prefill_activator(&model_name, worker_namespace);
                    }
                    self.manager
                        .deactivate_prefill_router_for_decode(&model_name, worker_namespace);
                }
                Some(WorkerType::Encode) if card.model_type.is_empty() => {
                    if removed.is_some() {
                        self.manager
                            .remove_encoder_activator(&model_name, worker_namespace);
                    }
                    self.manager
                        .deactivate_encoder_router_for_consumers(&model_name, worker_namespace);
                }
                Some(WorkerType::Decode)
                | Some(WorkerType::Aggregated)
                | Some(WorkerType::Encode)
                | None => {
                    // Decode-component teardown — and any surface-bearing worker
                    // that built a pipeline via the model_type chain (including
                    // an sglang multimodal encode front door, which registers a
                    // prefill router just like a decode worker): always run the
                    // waiter cleanup, regardless of whether `remove_worker_set`
                    // found an entry. If a decode worker registered (creating a
                    // `DecodeWaiting` activator entry) but `handle_add_helper`
                    // later failed before `add_worker_set`, the WorkerSet is
                    // absent here yet the stale `DecodeWaiting` still needs to be
                    // cleared. The helper is state-safe
                    // (`remove_if(|_, v| matches!(v, DecodeWaiting(_)))`) so
                    // calling it on a key that's vacant or holds `PrefillReady`
                    // is a no-op.
                    self.manager
                        .remove_decode_prefill_waiter(&model_name, worker_namespace);
                    self.manager
                        .remove_consumer_encoder_waiter(&model_name, worker_namespace);
                }
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
                if card.model_type.intersects(*model_type)
                    && is_model_type_list_empty(&self.manager, *model_type)
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
        // Check if this specific (model, namespace, type) WorkerSet already exists.
        // If so, this is just another worker joining an existing set — no pipeline build needed.
        let model_name = card.name().to_string();
        let namespace = mcid.namespace.clone();
        let ws_key = worker_set_key(&namespace, card.model_type, card.worker_type);

        if let Some(model) = self.manager.get_model(&model_name)
            && model.has_worker_set(&ws_key)
        {
            if !model.is_checksum_compatible(&ws_key, card.mdcsum()) {
                tracing::error!(
                    model_name = card.name(),
                    namespace = namespace,
                    new_checksum = card.mdcsum(),
                    "Checksum for new worker does not match existing WorkerSet's checksum. \
                     Drain all old workers in this namespace before deploying a new version."
                );
                return Err(anyhow::anyhow!(
                    "Checksum mismatch for worker in namespace {namespace}"
                ));
            }
            self.manager
                .save_model_card(&mcid.to_path(), card.clone())?;
            tracing::debug!(
                model_name = card.name(),
                namespace = namespace,
                "Worker joined existing WorkerSet, skipping pipeline build"
            );
            return Ok(());
        }

        // Guard against concurrent pipeline construction for the same (model, namespace, type)
        let registration_key = ModelManager::model_namespace_key(&model_name, &ws_key);
        if !self
            .registering_worker_sets
            .insert(registration_key.clone())
            && !self
                .recover_concurrent_registration(
                    mcid,
                    card,
                    &model_name,
                    &namespace,
                    &ws_key,
                    &registration_key,
                )
                .await?
        {
            return Ok(());
        }

        // RAII guard ensures the registration key is removed even if
        // do_worker_set_registration panics, preventing permanent poisoning.
        // It also wakes any waiters in recover_concurrent_registration.
        let _guard = RegistrationGuard {
            set: &self.registering_worker_sets,
            key: registration_key,
            notify: &self.registration_notify,
        };

        self.do_worker_set_registration(mcid, card).await
    }

    /// Handle the case where another task is already building the pipeline for this
    /// (model, namespace, type). This is a recovery path — it waits for the in-flight
    /// registration to finish, then either joins the resulting WorkerSet or retries.
    ///
    /// Returns `true` if the caller should proceed with its own registration
    /// (i.e. the other task failed), `false` if the worker was handled (joined or rejected).
    async fn recover_concurrent_registration(
        &self,
        mcid: &ModelCardInstanceId,
        card: &mut ModelDeploymentCard,
        model_name: &str,
        namespace: &str,
        ws_key: &str,
        registration_key: &str,
    ) -> anyhow::Result<bool> {
        // Wait for the in-flight registration to complete so we can validate
        // the new worker's checksum. Without this, a concurrent worker with a
        // mismatched checksum could sneak past the early check in `watch`.
        //
        // Uses a Notify + enable() loop instead of polling to wake up
        // immediately when the RegistrationGuard drops, avoiding up to 100ms
        // of unnecessary latency and wasted CPU cycles.
        // An absolute deadline ensures spurious wakeups (from unrelated
        // registrations sharing the same Notify) cannot extend the wait
        // beyond 30 seconds.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            let notified = self.registration_notify.notified();
            tokio::pin!(notified);
            // Register interest in the notification BEFORE checking the
            // condition to avoid a race where the guard drops between
            // our check and the .await.
            notified.as_mut().enable();
            if !self.registering_worker_sets.contains(registration_key) {
                break;
            }
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            if tokio::time::timeout(remaining, notified).await.is_err() {
                break;
            }
        }

        // If we timed out and the other task is still running, bail out rather
        // than proceeding with concurrent pipeline construction.
        if self.registering_worker_sets.contains(registration_key) {
            // Save the model card so handle_delete can find it for cleanup.
            self.manager
                .save_model_card(&mcid.to_path(), card.clone())?;
            tracing::warn!(
                model_name = card.name(),
                namespace = namespace,
                "Timed out waiting for concurrent registration to complete, skipping"
            );
            return Ok(false);
        }

        // Validate checksum against the registered model
        if let Some(model) = self.manager.get_model(model_name)
            && !model.is_checksum_compatible(ws_key, card.mdcsum())
        {
            tracing::error!(
                model_name = card.name(),
                namespace = namespace,
                new_checksum = card.mdcsum(),
                "Checksum for new worker does not match existing WorkerSet's checksum. \
                 Drain all old workers in this namespace before deploying a new version."
            );
            return Ok(false);
        }

        // If the first registration failed or timed out, no WorkerSet exists.
        // Fall through to do_worker_set_registration instead of becoming a ghost
        // worker (registered in cards but with no serving pipeline).
        if self
            .manager
            .get_model(model_name)
            .is_none_or(|m| !m.has_worker_set(ws_key))
        {
            // Only the first waiter to re-insert the key should proceed with
            // registration. Other waiters return false to avoid concurrent builds.
            if !self
                .registering_worker_sets
                .insert(registration_key.to_string())
            {
                // Save the model card so handle_delete can find it for cleanup.
                self.manager
                    .save_model_card(&mcid.to_path(), card.clone())?;
                tracing::debug!(
                    model_name = card.name(),
                    namespace = namespace,
                    "Another waiter won the re-registration race, skipping"
                );
                return Ok(false);
            }
            tracing::warn!(
                model_name = card.name(),
                namespace = namespace,
                "Concurrent registration produced no WorkerSet, retrying"
            );
            return Ok(true);
        }

        self.manager
            .save_model_card(&mcid.to_path(), card.clone())?;
        tracing::debug!(
            model_name = card.name(),
            namespace = namespace,
            "Worker joined existing WorkerSet, skipping pipeline build"
        );
        Ok(false)
    }

    /// Build a complete WorkerSet with all engines for this (model, namespace)
    /// and add it to the Model.
    async fn do_worker_set_registration(
        &self,
        mcid: &ModelCardInstanceId,
        card: &mut ModelDeploymentCard,
    ) -> anyhow::Result<()> {
        card.download_config(self.local_model_path.as_deref())
            .await?;

        // Use per-worker-set router config if the worker provided one in its MDC,
        // otherwise fall back to the frontend-level global config.
        let router_config = card.router_config.as_ref().unwrap_or(&self.router_config);

        let component = self
            .drt
            .namespace(&mcid.namespace)?
            .component(&mcid.component)?;
        let endpoint = component.endpoint(&mcid.endpoint);
        let client = endpoint.client().await?;
        let instance_watcher = client.instance_avail_watcher();
        tracing::debug!(
            model_name = card.name(),
            namespace = mcid.namespace,
            "building worker set pipeline"
        );
        self.manager
            .save_model_card(&mcid.to_path(), card.clone())?;

        let checksum = card.mdcsum();
        let namespace = mcid.namespace.clone();
        let ws_key = worker_set_key(&namespace, card.model_type, card.worker_type);

        // Build the WorkerSet with all applicable engines
        let mut worker_set = WorkerSet::new(namespace.clone(), checksum.to_string(), card.clone());
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
            self.manager
                .add_worker_set(card.name(), &ws_key, worker_set);

            if let Some(tx) = &self.model_update_tx {
                tx.send(ModelUpdate::Added(card.clone())).await.ok();
            }
            self.manager.activate_encoder_router(
                card.name(),
                &namespace,
                component.endpoint(&mcid.endpoint),
            );
            tracing::info!(
                model_name = card.name(),
                namespace = %namespace,
                "Encode worker registered and router activated"
            );
            return Ok(());
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

            // No engine on the worker set — just lifecycle tracking so the
            // prefill router can be activated/deactivated as workers come
            // and go.
            self.manager
                .add_worker_set(card.name(), &ws_key, worker_set);

            if supports_encoder_result_handoff(card) {
                self.manager.enable_encoder_routing(card.name(), &namespace);
            }

            if let Some(tx) = &self.model_update_tx {
                tx.send(ModelUpdate::Added(card.clone())).await.ok();
            }

            // activate_prefill_router is keyed by deployment namespace (not
            // ws_key) because it coordinates between decode and prefill
            // worker sets that share the same deployment namespace but have
            // different ws_keys (decode `{ns}:chat|completions:decode` vs
            // prefill `{ns}::prefill`).
            let endpoint = component.endpoint(&mcid.endpoint);
            let Ok(()) = self
                .manager
                .activate_prefill_router(card.name(), &namespace, endpoint)
            else {
                tracing::warn!(
                    model_name = card.name(),
                    "Failed to activate prefill router - prefill worker may already be activated"
                );
                return Ok(());
            };

            tracing::info!(
                model_name = card.name(),
                "Prefill worker registered and router activated successfully"
            );

            return Ok(());
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
                self.generate_engine_enabled && supports_vllm_generate(card);
            let needs_preprocessed_routing =
                needs_factory_chat_pipeline || tokenizer.is_some() || needs_generate_pipeline;

            // Create the KV router whenever any routed pipeline will be built.
            // Python chat factories receive a Rust-routed engine, so they also
            // need the shared chooser in KV mode.
            let kv_chooser =
                if router_config.router_mode == RouterMode::KV && needs_preprocessed_routing {
                    Some(
                        self.manager
                            .kv_chooser_for(
                                &endpoint,
                                card.kv_cache_block_size,
                                Some(router_config.kv_router_config.clone()),
                                self.prefill_load_estimator.clone(),
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
            // monitor can be handed directly to PrefillRouter::new. Each WorkerSet gets its own
            // monitor (1-to-1), scoped to this WorkerSet's Client/namespace. The monitor tracks
            // Prometheus metrics (active_decode_blocks, active_prefill_tokens, worker TTFT/ITL
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

            // Create prefill chooser once if we're building pipelines
            // Both chat and completions will share the same prefill chooser instance
            let model_name = card.name().to_string();
            let prefill_chooser = if needs_preprocessed_routing {
                self.manager
                    .register_prefill_router(&model_name, &namespace)
                    .map(|rx| {
                        // Create prefill-specific config with track_active_blocks disabled
                        let mut prefill_config = router_config.kv_router_config.clone();
                        prefill_config.router_track_active_blocks = false;
                        // Prefill KV events are emitted by prefill workers; do not inherit
                        // decode-only speculative hash mode.
                        let prefill_enable_eagle = false;

                        PrefillRouter::new(
                            rx,
                            self.manager.clone(),
                            router_config.router_mode,
                            card.kv_cache_block_size,
                            Some(prefill_config),
                            self.prefill_load_estimator.clone(),
                            router_config.session_affinity_ttl_secs,
                            model_name.clone(),
                            namespace.clone(),
                            prefill_enable_eagle,
                            // Hand the monitor directly so the prefill Client can be attached
                            // to it on activation (no namespace lookup).
                            worker_monitor.clone(),
                        )
                    })
            } else {
                None
            };

            let encoder_chooser = if needs_preprocessed_routing {
                if supports_encoder_result_handoff(card) {
                    self.manager.enable_encoder_routing(&model_name, &namespace);
                }
                self.manager
                    .register_encoder_router(&model_name, &namespace)
                    .map(|rx| EncoderRouter::new(rx, model_name.clone(), namespace.clone()))
            } else {
                None
            };

            // Store KV router, worker monitor, and prefill router on the WorkerSet.
            // The prefill router is stored so the watcher can deactivate/reactivate it
            // when prefill workers die or rejoin.
            worker_set.kv_router = kv_chooser.clone();
            worker_set.worker_monitor = worker_monitor.clone();
            worker_set.prefill_router = prefill_chooser.clone();
            worker_set.encoder_router = encoder_chooser.clone();

            let preprocessed_routing = if needs_preprocessed_routing {
                Some(
                    entrypoint::build_preprocessed_routing(
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
        } else if card.model_input == ModelInput::Text && card.model_type.supports_embedding() {
            // Case: Text + Embeddings
            let push_router = PushRouter::<
                NvCreateEmbeddingRequest,
                Annotated<NvCreateEmbeddingResponse>,
            >::from_client_with_monitor(
                client, router_config.router_mode, None
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
                >::from_client_with_monitor(
                    client.clone(), router_config.router_mode, None
                )
                .await?;
                worker_set.chat_engine = Some(Arc::new(chat_router));
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
        } else if card.model_input == ModelInput::Text && card.model_type.supports_chat() {
            // Case: Text + Chat (pure text-to-text, no diffusion)
            let push_router =
                PushRouter::<
                    NvCreateChatCompletionRequest,
                    Annotated<NvCreateChatCompletionStreamResponse>,
                >::from_client_with_monitor(client, router_config.router_mode, None)
                .await?;
            worker_set.chat_engine = Some(Arc::new(push_router));
        } else if card.model_input == ModelInput::Text && card.model_type.supports_completions() {
            // Case: Text + Completions
            let push_router = PushRouter::<
                NvCreateCompletionRequest,
                Annotated<NvCreateCompletionResponse>,
            >::from_client_with_monitor(
                client, router_config.router_mode, None
            )
            .await?;
            worker_set.completions_engine = Some(Arc::new(push_router));
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
        } else if card.model_input == ModelInput::Text && card.model_type.supports_realtime() {
            // Case 7: Text + Realtime
            // 'Text' is being overloaded here, it simply means the I/O will be passed through
            let realtime_router = PushRouter::<
                RealtimeClientEvent,
                Annotated<RealtimeServerEvent>,
            >::from_client_with_monitor(
                client, router_config.router_mode, None
            )
            .await?;
            worker_set.realtime_engine = Some(Arc::new(realtime_router));
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
                Tokens+(Chat|Completions), Text+(Chat|Completions|Images|Audios|Videos|Embeddings|Realtime), \
                Tokens+Embeddings, Tensor+TensorBased",
                card.model_type,
                card.model_input.as_str()
            );
        }

        // Add the completed WorkerSet to the Model
        self.manager
            .add_worker_set(card.name(), &ws_key, worker_set);

        if let Some(tx) = &self.model_update_tx {
            tx.send(ModelUpdate::Added(card.clone())).await.ok();
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
                Ok(mut card) => {
                    self.apply_tokenizer_backend_override(&mut card);
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

/// Seed the LoRA state tracker from a worker's MDC.
///
/// - Adapter card (`card.lora` is `Some`): register the loaded adapter (which also records the
///   worker's capacity) and return the adapter name so the caller can track it for failed-save
///   removal reconciliation.
/// - Base worker card advertising `runtime_config.max_gpu_lora_count`: seed capacity-only (no
///   phantom adapter) so the controller sees idle-but-LoRA-capable workers before the first
///   adapter load. Returns `None`.
/// - Otherwise (non-LoRA worker): no-op, returns `None`.
///
/// Split out of the discovery loop so the base-card capacity data flow is unit-testable without
/// constructing a full `ModelWatcher`.
fn seed_lora_state_from_card(
    state_tracker: &crate::lora::LoraStateTracker,
    worker: crate::kv_router::protocols::WorkerWithDpRank,
    card: &ModelDeploymentCard,
) -> Option<String> {
    if let Some(lora_info) = card.lora.as_ref() {
        state_tracker.handle_mdc_addition(worker, lora_info);
        Some(lora_info.name.clone())
    } else if let Some(capacity) = card.runtime_config.max_gpu_lora_count {
        state_tracker.set_worker_capacity(worker, capacity);
        None
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_card::ModelDeploymentCard;

    #[test]
    fn vllm_generate_requires_explicit_worker_capability() {
        let mut card = ModelDeploymentCard::with_name_only("model");
        card.model_type = ModelType::Chat | ModelType::Completions;
        assert!(!supports_vllm_generate(&card));

        card.runtime_config
            .set_engine_specific(VLLM_INFERENCE_V1_GENERATE_CAPABILITY, true)
            .unwrap();
        assert!(supports_vllm_generate(&card));

        card.runtime_config
            .set_engine_specific(VLLM_INFERENCE_V1_GENERATE_CAPABILITY, false)
            .unwrap();
        assert!(!supports_vllm_generate(&card));
    }

    #[test]
    fn encoder_result_handoff_requires_explicit_worker_capability() {
        let mut card = ModelDeploymentCard::with_name_only("model");
        card.needs = vec![vec![WorkerType::Encode]];
        assert!(!supports_encoder_result_handoff(&card));

        card.runtime_config
            .set_engine_specific(ENCODER_RESULT_HANDOFF_CAPABILITY, true)
            .unwrap();
        assert!(supports_encoder_result_handoff(&card));

        card.runtime_config
            .set_engine_specific(ENCODER_RESULT_HANDOFF_CAPABILITY, false)
            .unwrap();
        assert!(!supports_encoder_result_handoff(&card));
    }

    #[test]
    fn base_card_with_capacity_seeds_idle_lora_capable_worker() {
        // jh-nv (watcher base-card seeding): a base worker card (lora=None) carrying
        // runtime_config.max_gpu_lora_count must seed capacity-only so the controller sees the
        // idle LoRA-capable worker before any adapter loads. Pins the base-card -> capacity flow
        // that the discovery loop relies on.
        let st = crate::lora::LoraStateTracker::new();
        let worker = crate::kv_router::protocols::WorkerWithDpRank::new(7, 0);
        let mut card = ModelDeploymentCard::with_name_only("base-model");
        card.runtime_config.max_gpu_lora_count = Some(4);
        assert!(card.lora.is_none());

        let adapter = seed_lora_state_from_card(&st, worker, &card);
        assert_eq!(adapter, None, "a base card registers no adapter name");
        assert_eq!(
            st.list_workers(),
            vec![worker],
            "idle LoRA-capable worker must be visible to the controller"
        );
        assert_eq!(st.total_lora_slots(), 4);
    }

    #[test]
    fn base_card_without_capacity_seeds_nothing() {
        // A non-LoRA base card must not seed any worker capacity.
        let st = crate::lora::LoraStateTracker::new();
        let card = ModelDeploymentCard::with_name_only("base-model");
        assert!(card.runtime_config.max_gpu_lora_count.is_none());

        let adapter = seed_lora_state_from_card(
            &st,
            crate::kv_router::protocols::WorkerWithDpRank::new(1, 0),
            &card,
        );
        assert_eq!(adapter, None);
        assert!(
            st.list_workers().is_empty(),
            "a non-LoRA base card must not seed capacity"
        );
    }

    #[test]
    fn adapter_card_registers_adapter_and_returns_name() {
        // An adapter card registers the loaded adapter (+capacity) and returns its name so the
        // caller can track it in pending_lora_adds.
        let st = crate::lora::LoraStateTracker::new();
        let worker = crate::kv_router::protocols::WorkerWithDpRank::new(3, 0);
        let mut card = ModelDeploymentCard::with_name_only("base-model");
        card.lora = Some(crate::model_card::LoraInfo {
            name: "adapter-x".to_string(),
            max_gpu_lora_count: Some(2),
        });

        let adapter = seed_lora_state_from_card(&st, worker, &card);
        assert_eq!(adapter.as_deref(), Some("adapter-x"));
        assert!(st.is_loaded("adapter-x", &worker));
        assert_eq!(st.total_lora_slots(), 2);
    }

    #[test]
    fn registration_is_complete_only_after_card_and_worker_set_exist() {
        let manager = ModelManager::new();
        let mcid = ModelCardInstanceId {
            namespace: "deployment-a".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 7,
            model_suffix: None,
        };
        let mut card = ModelDeploymentCard::with_name_only("llama");
        card.model_type = ModelType::Chat;
        card.worker_type = Some(WorkerType::Aggregated);

        assert!(!is_registration_complete(&manager, &mcid, &card));

        // `do_worker_set_registration` saves the card before all pipeline
        // construction has completed. A failure after this point must remain a
        // reconciliation candidate.
        manager
            .save_model_card(&mcid.to_path(), card.clone())
            .unwrap();
        assert!(!is_registration_complete(&manager, &mcid, &card));

        let ws_key = worker_set_key(&mcid.namespace, card.model_type, card.worker_type);
        manager.add_worker_set(
            card.name(),
            &ws_key,
            WorkerSet::new(
                mcid.namespace.clone(),
                "stale-checksum".to_string(),
                card.clone(),
            ),
        );
        assert!(
            !is_registration_complete(&manager, &mcid, &card),
            "a WorkerSet from a different model-card checksum must be retried"
        );

        manager.add_worker_set(
            card.name(),
            &ws_key,
            WorkerSet::new(
                mcid.namespace.clone(),
                card.mdcsum().to_string(),
                card.clone(),
            ),
        );
        assert!(is_registration_complete(&manager, &mcid, &card));

        manager.remove_model_card(&mcid.to_path());
        assert!(!is_registration_complete(&manager, &mcid, &card));
    }

    #[test]
    fn stale_cleanup_cannot_reserve_after_a_new_registration() {
        let operation = InstanceOperation::default();
        let snapshot_generation = operation.current_generation();

        let registration_generation = operation.begin();

        assert!(operation.is_current(registration_generation));
        assert_eq!(operation.reserve_after(snapshot_generation), None);
    }

    #[tokio::test]
    async fn newer_registration_waits_for_cleanup_and_remains_current() {
        let operation = Arc::new(InstanceOperation::default());
        let cleanup_generation = operation
            .reserve_after(operation.current_generation())
            .expect("cleanup should reserve the unchanged snapshot generation");
        let cleanup_guard = operation.lock.lock().await;

        let registration_generation = operation.begin();
        assert!(!operation.is_current(cleanup_generation));

        let registration_operation = Arc::clone(&operation);
        let registration = tokio::spawn(async move {
            let _guard = registration_operation.lock.lock().await;
            registration_operation.is_current(registration_generation)
        });
        tokio::task::yield_now().await;
        assert!(
            !registration.is_finished(),
            "the newer registration must wait until cleanup releases the instance lock"
        );

        drop(cleanup_guard);
        assert!(registration.await.unwrap());
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
    }

    #[test]
    fn test_is_model_type_list_empty_realtime_after_register() {
        let mm = ModelManager::new();
        let engine = std::sync::Arc::new(crate::engines::EchoBidirectionalEngine);
        mm.add_realtime_model("rt-echo", "0", engine).unwrap();
        assert!(!is_model_type_list_empty(&mm, ModelType::Realtime));
    }

    #[test]
    fn test_realtime_in_all_model_types() {
        assert!(ALL_MODEL_TYPES.contains(&ModelType::Realtime));
    }

    #[test]
    fn ws_key_format_per_role() {
        // Decode worker with Chat | Completions
        let dk = worker_set_key(
            "ns1",
            ModelType::Chat | ModelType::Completions,
            Some(WorkerType::Decode),
        );
        assert_eq!(dk, "ns1:chat|completions:decode");

        // Prefill worker registers with empty ModelType (no OpenAI surface)
        let pk = worker_set_key("ns1", ModelType::empty(), Some(WorkerType::Prefill));
        assert_eq!(pk, "ns1::prefill");

        // Encode worker, same pattern as prefill
        let ek = worker_set_key("ns1", ModelType::empty(), Some(WorkerType::Encode));
        assert_eq!(ek, "ns1::encode");

        // Aggregated worker
        let ak = worker_set_key(
            "ns1",
            ModelType::Chat | ModelType::Completions,
            Some(WorkerType::Aggregated),
        );
        assert_eq!(ak, "ns1:chat|completions:aggregated");

        // Legacy card with no worker_type set falls under the compat shim,
        // which renders it as `aggregated` in the key.
        let legacy = worker_set_key("ns1", ModelType::Chat | ModelType::Completions, None);
        assert_eq!(legacy, "ns1:chat|completions:aggregated");
    }

    #[test]
    fn ws_key_new_and_legacy_prefill_share_a_bucket() {
        // A NEW prefill worker dual-emits ModelType::Prefill + worker_type=Prefill.
        let new_prefill = worker_set_key("ns1", ModelType::Prefill, Some(WorkerType::Prefill));
        assert_eq!(new_prefill, "ns1:prefill:prefill");

        // A LEGACY prefill card (ModelType::Prefill marker bit, no worker_type)
        // must resolve to the SAME bucket via effective_worker_type, so old and
        // new prefill workers in one namespace don't split into two buckets.
        let legacy_prefill = worker_set_key("ns1", ModelType::Prefill, None);
        assert_eq!(legacy_prefill, "ns1:prefill:prefill");
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
    fn ws_key_separates_prefill_from_decode_in_same_namespace() {
        // Prefill and decode in the same deployment namespace must hash to
        // distinct keys so they live in separate WorkerSet buckets.
        let decode = worker_set_key(
            "ns1",
            ModelType::Chat | ModelType::Completions,
            Some(WorkerType::Decode),
        );
        let prefill = worker_set_key("ns1", ModelType::empty(), Some(WorkerType::Prefill));
        assert_ne!(decode, prefill);
    }

    #[test]
    fn worker_set_key_encode_and_aggregated_coexist_in_same_namespace() {
        // Regression for the Encode/Aggregated key collision: Encode and
        // Aggregated workers in the same namespace MUST map to different keys,
        // so both can register without an MDC checksum mismatch. Under the
        // role-in-key scheme, an Encode worker registers surface-less
        // (ModelType::empty()) and lands in `{ns}::encode`, while Aggregated
        // keeps its `{ns}:chat|completions:aggregated` bucket.
        let agg_key = worker_set_key(
            "dynamo",
            ModelType::Chat | ModelType::Completions,
            Some(WorkerType::Aggregated),
        );
        let enc_key = worker_set_key("dynamo", ModelType::empty(), Some(WorkerType::Encode));
        assert_ne!(agg_key, enc_key);
        assert_eq!(agg_key, "dynamo:chat|completions:aggregated");
        assert_eq!(enc_key, "dynamo::encode");
    }
}
