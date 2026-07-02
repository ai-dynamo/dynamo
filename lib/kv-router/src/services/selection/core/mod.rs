// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dynamo_tokens::SequenceHash;
use parking_lot::RwLock;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use crate::indexer::TieredMatchDetails;
use crate::protocols::{
    ActiveSequenceEvent, LocalBlockHash, RoutingConstraints, WorkerId, WorkerWithDpRank,
};
use crate::scheduling::config::RouterConfigOverride;
use crate::scheduling::selector::DefaultWorkerSelector;
use crate::scheduling::{
    KvSchedulerError, LocalScheduler, OverlapAnalysis, OverlapSignals, PotentialLoad, ScheduleMode,
    ScheduleRequest, TieredOverlapRefresher, effective_prefill_tokens,
    prefill_load_hint_from_effective_tokens,
};
use crate::sequences::{
    ActiveSequencesMultiWorker, ReplicaWorkerPolicy, SequenceError, SequenceRequest,
};
use crate::services::common::replica_sync::{
    ReplicaSyncConfig, ScopedReplicaEvent, ScopedSequencePublisher, setup_scoped_replica_sync,
};
use crate::services::indexer::backend::Indexer;
use crate::services::indexer::recovery;
use crate::services::indexer::registry::WorkerRegistry;
use crate::services::overlap::MooncakeOverlapSummary;

use super::catalog::WorkerCatalog;
use super::error::SelectionError;
use super::input::PromptRequest;
use super::types::{
    ModelLoadResponse, OverlapScoresRequest, OverlapScoresResponse, PotentialLoadsRequest,
    ReadyResponse, ReservationRequest, ReservationResponse, SelectAndReserveRequest, SelectRequest,
    SelectResponse, SelectionKey, SelectionWorkerConfig, WORKER_TYPE, WorkerCatalogRecord,
    WorkerLifecycle, WorkerPatchRequest, WorkerRequest,
};

type SelectionScheduler = LocalScheduler<
    ScopedSequencePublisher,
    SelectionWorkerConfig,
    DefaultWorkerSelector,
    TieredOverlapRefresher<Indexer>,
>;

struct SelectionEntry {
    key: SelectionKey,
    block_size: u32,
    is_eagle: bool,
    indexer: Indexer,
    workers_tx: watch::Sender<HashMap<WorkerId, SelectionWorkerConfig>>,
    scheduler: SelectionScheduler,
    replica_tx: Option<mpsc::Sender<ActiveSequenceEvent>>,
}

struct PreparedSelectionInputs {
    block_hashes: Vec<LocalBlockHash>,
    sequence_hashes: Vec<SequenceHash>,
    isl_tokens: usize,
    overlap: OverlapSignals,
}

struct SelectionOperation {
    key: SelectionKey,
    selection_id: Option<String>,
    reservation_id: Option<String>,
    prompt: PromptRequest,
    router_config_override: Option<RouterConfigOverride>,
    expected_output_tokens: Option<u32>,
    priority_jump: f64,
    strict_priority: u32,
    policy_class: Option<String>,
    session_id: Option<String>,
    pinned_worker: Option<WorkerWithDpRank>,
    allowed_worker_ids: Option<HashSet<WorkerId>>,
    routing_constraints: RoutingConstraints,
}

#[derive(Debug, Clone)]
pub struct SelectionServiceConfig {
    pub port: u16,
    pub threads: usize,
    pub indexer_peers: Vec<String>,
    pub replica_sync_port: Option<u16>,
    pub replica_sync_peers: Vec<String>,
    pub kv_router_config: crate::config::KvRouterConfig,
}

pub struct SelectionCore {
    catalog: WorkerCatalog,
    entries: RwLock<HashMap<SelectionKey, Arc<SelectionEntry>>>,
    indexer_registry: Arc<WorkerRegistry>,
    kv_router_config: crate::config::KvRouterConfig,
    cancel_token: CancellationToken,
    replica_config: Option<ReplicaSyncConfig>,
}

impl SelectionCore {
    pub fn new(
        kv_router_config: crate::config::KvRouterConfig,
        indexer_threads: usize,
        cancel_token: CancellationToken,
    ) -> Self {
        Self::new_inner(kv_router_config, indexer_threads, cancel_token, None, true)
    }

    pub(crate) fn new_for_server(
        kv_router_config: crate::config::KvRouterConfig,
        indexer_threads: usize,
        cancel_token: CancellationToken,
        replica_config: Option<ReplicaSyncConfig>,
    ) -> Self {
        Self::new_inner(
            kv_router_config,
            indexer_threads,
            cancel_token,
            replica_config,
            false,
        )
    }

    fn new_inner(
        kv_router_config: crate::config::KvRouterConfig,
        indexer_threads: usize,
        cancel_token: CancellationToken,
        replica_config: Option<ReplicaSyncConfig>,
        signal_indexer_ready: bool,
    ) -> Self {
        let cancel_token = cancel_token.child_token();
        let indexer_registry = Arc::new(WorkerRegistry::new_with_cancel_token(
            indexer_threads,
            cancel_token.clone(),
        ));
        if signal_indexer_ready {
            indexer_registry.signal_ready();
        }
        Self {
            catalog: WorkerCatalog::default(),
            entries: RwLock::new(HashMap::new()),
            indexer_registry,
            kv_router_config,
            cancel_token,
            replica_config,
        }
    }

    /// Cancel core-scoped tasks (KV-event listeners, scheduling, replica sync,
    /// periodic expiry) without cancelling the parent token. In-flight and
    /// queued selections then fail fast.
    ///
    /// The KV indexer thread pool is owned by the registry and released when
    /// this `SelectionCore` is dropped. Idempotent.
    pub fn shutdown(&self) {
        self.cancel_token.cancel();
    }

    fn ensure_running(&self) -> Result<(), SelectionError> {
        if self.cancel_token.is_cancelled() {
            return Err(SelectionError::NotReady(
                "selection service is shutting down".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) async fn recover_indexer_from_peers(
        &self,
        peers: &[String],
    ) -> anyhow::Result<bool> {
        recovery::recover_from_peers(peers, &self.indexer_registry).await
    }

    pub(crate) fn signal_indexer_ready(&self) {
        self.indexer_registry.signal_ready();
    }

    pub(crate) async fn dump_indexer_events(&self) -> serde_json::Value {
        crate::services::indexer::server::dump_registry(&self.indexer_registry).await
    }

    pub(crate) fn dispatch_replica_event(&self, envelope: ScopedReplicaEvent) {
        if self
            .replica_config
            .as_ref()
            .is_some_and(|config| config.is_self_event(&envelope.event))
        {
            return;
        }

        let key = SelectionKey::new(envelope.model_name, envelope.tenant_id);
        let Some(entry) = self.entries.read().get(&key).cloned() else {
            tracing::trace!(%key, "Dropping replica event for unknown selector entry");
            return;
        };
        if entry.block_size != envelope.block_size {
            tracing::debug!(
                %key,
                expected_block_size = entry.block_size,
                received_block_size = envelope.block_size,
                "Dropping selector replica event with mismatched block size"
            );
            return;
        }
        let Some(replica_tx) = &entry.replica_tx else {
            return;
        };
        match replica_tx.try_send(envelope.event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(event)) => {
                tracing::trace!(
                    %key,
                    request_id = %event.request_id,
                    "Selector replica subscriber channel full; dropping event"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!(%key, "Selector replica subscriber channel closed");
            }
        }
    }

    pub async fn upsert_worker(
        &self,
        req: WorkerRequest,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        self.ensure_running()?;
        let (previous, record) = self.catalog.upsert(req);
        self.reconcile_worker(record.worker_id, previous).await
    }

    pub async fn patch_worker(
        &self,
        worker_id: WorkerId,
        patch: WorkerPatchRequest,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        self.ensure_running()?;
        let (previous, record) = self.catalog.patch(worker_id, patch)?;
        self.reconcile_worker(record.worker_id, Some(previous))
            .await
    }

    pub async fn delete_worker(
        &self,
        worker_id: WorkerId,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        let Some(previous) = self.catalog.get(worker_id) else {
            return Err(SelectionError::NotFound(format!(
                "worker {worker_id} not found"
            )));
        };
        let key = previous.key();
        self.catalog
            .set_lifecycle(worker_id, WorkerLifecycle::Draining, Vec::new());
        self.publish_scheduler_config(&key)?;
        self.cleanup_indexer_registration(&previous).await;
        let record = self
            .catalog
            .set_lifecycle(worker_id, WorkerLifecycle::Unschedulable, Vec::new())
            .ok_or_else(|| SelectionError::NotFound(format!("worker {worker_id} not found")))?;
        self.publish_scheduler_config(&key)?;
        Ok(record)
    }

    pub fn list_workers(
        &self,
        model_name: Option<&str>,
        tenant_id: Option<&str>,
    ) -> Vec<WorkerCatalogRecord> {
        self.catalog.list(model_name, tenant_id)
    }

    pub fn ready(&self) -> ReadyResponse {
        let schedulable_workers = self.catalog.schedulable_count();
        let workers = self.catalog.list(None, None);
        ReadyResponse {
            ready: !self.cancel_token.is_cancelled() && schedulable_workers > 0,
            schedulable_workers,
            workers,
        }
    }

    async fn reconcile_worker(
        &self,
        worker_id: WorkerId,
        previous: Option<WorkerCatalogRecord>,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        let Some(record) = self.catalog.get(worker_id) else {
            return Err(SelectionError::NotFound(format!(
                "worker {worker_id} not found"
            )));
        };

        if previous
            .as_ref()
            .is_some_and(|record| record.lifecycle == WorkerLifecycle::Schedulable)
        {
            self.catalog
                .set_lifecycle(worker_id, WorkerLifecycle::Draining, Vec::new());
            self.publish_scheduler_config(&previous.as_ref().expect("checked").key())?;
            self.cleanup_indexer_registration(previous.as_ref().expect("checked"))
                .await;
        }

        let reasons = record.missing_schedulable_metadata(
            self.kv_router_config.router_queue_threshold.is_some()
                || self.kv_router_config.router_policy_config.is_some(),
            self.kv_router_config.use_kv_events,
        );
        if !reasons.is_empty() {
            let updated = self
                .catalog
                .set_lifecycle(worker_id, WorkerLifecycle::Incomplete, reasons)
                .ok_or_else(|| SelectionError::NotFound(format!("worker {worker_id} not found")))?;
            self.publish_scheduler_config(&updated.key())?;
            return Ok(updated);
        }

        if let Err(error) = self.ensure_entry(&record) {
            return self.mark_incomplete_after_reconcile_error(worker_id, record.key(), error);
        }
        if self.kv_router_config.use_kv_events
            && let Err(error) = self.register_indexer_listeners(&record).await
        {
            self.cleanup_indexer_registration(&record).await;
            return self.mark_incomplete_after_reconcile_error(worker_id, record.key(), error);
        }

        let updated = self
            .catalog
            .set_lifecycle(worker_id, WorkerLifecycle::Schedulable, Vec::new())
            .ok_or_else(|| SelectionError::NotFound(format!("worker {worker_id} not found")))?;
        self.publish_scheduler_config(&updated.key())?;
        Ok(updated)
    }

    fn mark_incomplete_after_reconcile_error(
        &self,
        worker_id: WorkerId,
        key: SelectionKey,
        error: SelectionError,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        let updated = self
            .catalog
            .set_lifecycle(
                worker_id,
                WorkerLifecycle::Incomplete,
                vec![format!("reconciliation failed: {error}")],
            )
            .ok_or_else(|| SelectionError::NotFound(format!("worker {worker_id} not found")))?;
        self.publish_scheduler_config(&key)?;
        Ok(updated)
    }

    fn ensure_entry(
        &self,
        record: &WorkerCatalogRecord,
    ) -> Result<Arc<SelectionEntry>, SelectionError> {
        let block_size = record
            .block_size
            .ok_or_else(|| SelectionError::BadRequest("block_size is required".to_string()))?;
        let is_eagle = record.is_eagle.unwrap_or(false);
        let key = record.key();

        if let Some(entry) = self.entries.read().get(&key).cloned() {
            if entry.block_size != block_size {
                return Err(SelectionError::Conflict(format!(
                    "block_size mismatch for {key}: existing={} requested={block_size}",
                    entry.block_size
                )));
            }
            if entry.is_eagle != is_eagle {
                return Err(SelectionError::Conflict(format!(
                    "is_eagle mismatch for {key}: existing={} requested={is_eagle}",
                    entry.is_eagle
                )));
            }
            return Ok(entry);
        }

        let mut entries = self.entries.write();
        if let Some(entry) = entries.get(&key).cloned() {
            if entry.block_size != block_size {
                return Err(SelectionError::Conflict(format!(
                    "block_size mismatch for {key}: existing={} requested={block_size}",
                    entry.block_size
                )));
            }
            if entry.is_eagle != is_eagle {
                return Err(SelectionError::Conflict(format!(
                    "is_eagle mismatch for {key}: existing={} requested={is_eagle}",
                    entry.is_eagle
                )));
            }
            return Ok(entry);
        }

        let (workers_tx, workers_rx) = watch::channel(HashMap::new());
        let scoped_replica_sync = setup_scoped_replica_sync(
            self.replica_config.as_ref(),
            &key.model_name,
            &key.tenant_id,
            block_size,
        );
        let slots = Arc::new(ActiveSequencesMultiWorker::new_with_replica_worker_policy(
            scoped_replica_sync.publisher,
            block_size as usize,
            HashMap::new(),
            scoped_replica_sync.enabled,
            scoped_replica_sync.process_id,
            WORKER_TYPE,
            ReplicaWorkerPolicy::RequireRegistered,
        ));
        let replica_tx = scoped_replica_sync.channel.map(|(replica_tx, subscriber)| {
            slots.start_replica_sync(subscriber, self.cancel_token.child_token());
            replica_tx
        });
        slots.start_periodic_force_expiry_across_all_workers(self.cancel_token.child_token());

        let indexer = self
            .indexer_registry
            .get_or_create_indexer(key.indexer_key(), block_size);
        let overlap_refresh = Arc::new(TieredOverlapRefresher::new(
            indexer.clone(),
            self.kv_router_config.clone(),
            block_size,
        ));
        let selector = DefaultWorkerSelector::new(Some(self.kv_router_config.clone()), WORKER_TYPE);
        let profile = self
            .kv_router_config
            .policy_profile(Some(&key.model_name))
            .map_err(|error| SelectionError::BadRequest(error.to_string()))?;
        let scheduler = LocalScheduler::new_with_policy_profile(
            slots,
            workers_rx,
            profile,
            block_size,
            selector,
            None,
            Some(overlap_refresh),
            None,
            self.kv_router_config.router_queue_recheck_interval(),
            self.kv_router_config.router_track_prefill_tokens,
            self.cancel_token.child_token(),
            WORKER_TYPE,
            true,
        );
        let entry = Arc::new(SelectionEntry {
            key: key.clone(),
            block_size,
            is_eagle,
            indexer,
            workers_tx,
            scheduler,
            replica_tx,
        });
        entries.insert(key, entry.clone());
        Ok(entry)
    }

    async fn register_indexer_listeners(
        &self,
        record: &WorkerCatalogRecord,
    ) -> Result<(), SelectionError> {
        let block_size = record
            .block_size
            .ok_or_else(|| SelectionError::BadRequest("block_size is required".to_string()))?;
        let mut endpoints: Vec<_> = record.listener_endpoints().into_iter().collect();
        endpoints.sort_by_key(|(dp_rank, _)| *dp_rank);
        for (dp_rank, endpoint) in endpoints {
            crate::services::common::zmq::validate_endpoint(&endpoint).map_err(|error| {
                SelectionError::BadRequest(format!(
                    "invalid kv_events endpoint for worker {} dp_rank {dp_rank}: {error}",
                    record.worker_id
                ))
            })?;
            if let Some(replay_endpoint) = record.replay_endpoint.as_deref() {
                crate::services::common::zmq::validate_endpoint(replay_endpoint).map_err(
                    |error| {
                        SelectionError::BadRequest(format!(
                            "invalid replay endpoint for worker {} dp_rank {dp_rank}: {error}",
                            record.worker_id
                        ))
                    },
                )?;
            }
            self.indexer_registry
                .register(
                    record.worker_id,
                    endpoint,
                    dp_rank,
                    record.model_name.clone(),
                    record.tenant_id.clone(),
                    block_size,
                    record.replay_endpoint.clone(),
                )
                .await
                .map_err(|error| SelectionError::BadRequest(error.to_string()))?;
        }
        Ok(())
    }

    async fn cleanup_indexer_registration(&self, record: &WorkerCatalogRecord) {
        if self.kv_router_config.use_kv_events {
            if let Err(error) = self
                .indexer_registry
                .deregister(record.worker_id, &record.model_name, &record.tenant_id)
                .await
            {
                tracing::debug!(
                    worker_id = record.worker_id,
                    error = %error,
                    "indexer deregistration skipped or failed"
                );
            }
            return;
        }

        let key = record.key().indexer_key();
        let indexer = self
            .indexer_registry
            .get_indexer(&key)
            .map(|entry| entry.indexer.clone());
        if let Some(indexer) = indexer {
            indexer.remove_worker(record.worker_id).await;
        }
    }

    fn publish_scheduler_config(&self, key: &SelectionKey) -> Result<(), SelectionError> {
        let Some(entry) = self.entries.read().get(key).cloned() else {
            return Ok(());
        };
        let workers = self.catalog.scheduler_configs_for_key(key);
        entry.workers_tx.send(workers).map_err(|_| {
            SelectionError::Internal(format!("scheduler worker watch closed for {key}"))
        })
    }

    fn ready_entry(&self, key: &SelectionKey) -> Result<Arc<SelectionEntry>, SelectionError> {
        if self.catalog.schedulable_count() == 0 {
            return Err(SelectionError::NotReady(
                "no schedulable workers are available".to_string(),
            ));
        }

        let Some(entry) = self.entries.read().get(key).cloned() else {
            return Err(SelectionError::NotReady(format!(
                "no schedulable workers for {key}"
            )));
        };
        if !self.catalog.has_schedulable_for_key(key) {
            return Err(SelectionError::NotReady(format!(
                "no schedulable workers for {key}"
            )));
        }
        Ok(entry)
    }

    pub async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
        self.select_with_policy_class(req, None).await
    }

    pub async fn select_with_policy_class(
        &self,
        req: SelectRequest,
        policy_class: Option<String>,
    ) -> Result<SelectResponse, SelectionError> {
        self.schedule_selection(
            SelectionOperation {
                key: SelectionKey::new(req.model_name, req.tenant_id),
                selection_id: req.selection_id,
                reservation_id: None,
                prompt: req.prompt,
                router_config_override: req.router_config_override,
                expected_output_tokens: req.expected_output_tokens,
                priority_jump: req.priority_jump.unwrap_or_default(),
                strict_priority: req.strict_priority.unwrap_or(0),
                policy_class,
                session_id: req.session_id,
                pinned_worker: req.pinned_worker,
                allowed_worker_ids: req.allowed_worker_ids,
                routing_constraints: req.routing_constraints,
            },
            false,
        )
        .await
    }

    pub async fn select_and_reserve(
        &self,
        req: SelectAndReserveRequest,
    ) -> Result<SelectResponse, SelectionError> {
        self.select_and_reserve_with_policy_class(req, None).await
    }

    pub async fn select_and_reserve_with_policy_class(
        &self,
        req: SelectAndReserveRequest,
        policy_class: Option<String>,
    ) -> Result<SelectResponse, SelectionError> {
        let reservation_id = req
            .reservation_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        self.schedule_selection(
            SelectionOperation {
                key: SelectionKey::new(req.model_name, req.tenant_id),
                selection_id: req.selection_id,
                reservation_id: Some(reservation_id),
                prompt: req.prompt,
                router_config_override: req.router_config_override,
                expected_output_tokens: req.expected_output_tokens,
                priority_jump: req.priority_jump.unwrap_or_default(),
                strict_priority: req.strict_priority.unwrap_or(0),
                policy_class,
                session_id: req.session_id,
                pinned_worker: req.pinned_worker,
                allowed_worker_ids: req.allowed_worker_ids,
                routing_constraints: req.routing_constraints,
            },
            true,
        )
        .await
    }

    async fn schedule_selection(
        &self,
        operation: SelectionOperation,
        book: bool,
    ) -> Result<SelectResponse, SelectionError> {
        let SelectionOperation {
            key,
            selection_id,
            reservation_id,
            prompt,
            router_config_override,
            expected_output_tokens,
            priority_jump,
            strict_priority,
            policy_class,
            session_id,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
        } = operation;
        self.ensure_running()?;

        let entry = self.ready_entry(&key)?;
        let PreparedSelectionInputs {
            block_hashes,
            sequence_hashes,
            isl_tokens,
            overlap,
        } = self.prepare_selection_inputs(&entry, &prompt).await?;
        let mode = if book {
            ScheduleMode::Tracked {
                request_id: reservation_id.clone().ok_or_else(|| {
                    SelectionError::Internal(
                        "booked selection did not include a reservation ID".to_string(),
                    )
                })?,
            }
        } else {
            ScheduleMode::QueryOnly {
                request_id: selection_id.clone(),
            }
        };
        let schedule_request = ScheduleRequest {
            mode,
            token_seq: Some(sequence_hashes),
            block_hashes: Some(block_hashes),
            isl_tokens,
            overlap,
            router_config_override,
            lora_name: prompt.lora_name,
            priority_jump,
            strict_priority,
            policy_class,
            session_id,
            expected_output_tokens,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
            shared_cache_hits: None,
        };
        let response = tokio::select! {
            biased;
            _ = self.cancel_token.cancelled() => {
                return Err(SelectionError::Scheduler(KvSchedulerError::SubscriberShutdown));
            }
            result = entry.scheduler.schedule_request(schedule_request) => result?,
        };
        let endpoint = self
            .catalog
            .schedulable_endpoint(response.best_worker.worker_id, &key)
            .ok_or_else(|| {
                SelectionError::Internal(format!(
                    "selected worker {} is no longer schedulable",
                    response.best_worker.worker_id
                ))
            })?;
        let overlap = MooncakeOverlapSummary::from_selected_worker_tiers(
            &response.selected_worker_tiers,
            entry.block_size,
        );

        Ok(SelectResponse {
            selection_id,
            reservation_id,
            model_name: key.model_name,
            tenant_id: key.tenant_id,
            worker_id: response.best_worker.worker_id,
            dp_rank: response.best_worker.dp_rank,
            endpoint,
            block_size: entry.block_size,
            overlap,
            effective_prefill_tokens: effective_prefill_tokens(isl_tokens, response.cached_tokens),
        })
    }

    pub async fn create_reservation(
        &self,
        req: ReservationRequest,
    ) -> Result<ReservationResponse, SelectionError> {
        self.ensure_running()?;
        let key = SelectionKey::new(req.model_name.clone(), req.tenant_id.clone());
        let entry = self.ready_entry(&key)?;
        let normalized = req
            .prompt
            .normalize_for_reservation(entry.block_size, entry.is_eagle)?;
        let prefill_load_hint = req
            .effective_prefill_tokens
            .map(|tokens| {
                prefill_load_hint_from_effective_tokens(normalized.isl_tokens, tokens)
                    .map_err(|error| SelectionError::BadRequest(error.to_string()))
            })
            .transpose()?;
        let worker = WorkerWithDpRank::new(req.worker_id, req.dp_rank.unwrap_or(0));
        let endpoint = self
            .catalog
            .schedulable_endpoint(req.worker_id, &key)
            .ok_or_else(|| {
                SelectionError::NotFound(format!(
                    "schedulable worker {} not found for {key}",
                    req.worker_id
                ))
            })?;
        let track_prefill_tokens = req.effective_prefill_tokens.is_some()
            || req
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.track_prefill_tokens)
                .unwrap_or(self.kv_router_config.router_track_prefill_tokens);

        entry
            .scheduler
            .add_request(SequenceRequest {
                request_id: req.reservation_id.clone(),
                token_sequence: Some(normalized.sequence_hashes),
                track_prefill_tokens,
                expected_output_tokens: req.expected_output_tokens,
                prefill_load_hint,
                worker,
                lora_name: req.prompt.lora_name.clone(),
            })
            .await?;

        Ok(ReservationResponse {
            reservation_id: req.reservation_id,
            model_name: key.model_name,
            tenant_id: key.tenant_id,
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
            endpoint,
        })
    }

    pub async fn prefill_complete(&self, reservation_id: &str) -> Result<(), SelectionError> {
        let entries = { self.entries.read().values().cloned().collect::<Vec<_>>() };
        for entry in entries {
            match entry.scheduler.mark_prefill_completed(reservation_id).await {
                Ok(()) => return Ok(()),
                Err(SequenceError::RequestNotFound { .. }) => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(SelectionError::NotFound(format!(
            "reservation {reservation_id} not found"
        )))
    }

    pub async fn free_reservation(&self, reservation_id: &str) -> Result<(), SelectionError> {
        let entries = { self.entries.read().values().cloned().collect::<Vec<_>>() };
        for entry in entries {
            match entry.scheduler.free(reservation_id).await {
                Ok(()) => return Ok(()),
                Err(SequenceError::RequestNotFound { .. }) => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(SelectionError::NotFound(format!(
            "reservation {reservation_id} not found"
        )))
    }

    pub fn add_output_block(
        &self,
        reservation_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SelectionError> {
        if let Some(frac) = decay_fraction
            && !(0.0..=1.0).contains(&frac)
        {
            return Err(SelectionError::BadRequest(
                "decay_fraction must be between 0.0 and 1.0".to_string(),
            ));
        }

        let entries = { self.entries.read().values().cloned().collect::<Vec<_>>() };
        for entry in entries {
            match entry
                .scheduler
                .add_output_block(reservation_id, decay_fraction)
            {
                Ok(()) => return Ok(()),
                Err(SequenceError::RequestNotFound { .. }) => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(SelectionError::NotFound(format!(
            "reservation {reservation_id} not found"
        )))
    }

    pub fn loads(
        &self,
        model_name: Option<&str>,
        tenant_id: Option<&str>,
    ) -> Vec<ModelLoadResponse> {
        let entries: Vec<_> = self.entries.read().values().cloned().collect();
        let mut loads = Vec::new();
        for entry in entries {
            if model_name.is_some_and(|model_name| entry.key.model_name != model_name)
                || tenant_id.is_some_and(|tenant_id| entry.key.tenant_id != tenant_id)
            {
                continue;
            }
            loads.push(ModelLoadResponse {
                model_name: entry.key.model_name.clone(),
                tenant_id: entry.key.tenant_id.clone(),
                loads: entry
                    .scheduler
                    .get_potential_loads(None, 0, HashMap::new(), false),
                pending_count: entry.scheduler.pending_count(),
                pending_isl_tokens: entry.scheduler.pending_isl_tokens(),
            });
        }
        loads.sort_by(|a, b| (&a.model_name, &a.tenant_id).cmp(&(&b.model_name, &b.tenant_id)));
        loads
    }

    pub async fn potential_loads(
        &self,
        req: PotentialLoadsRequest,
    ) -> Result<Vec<PotentialLoad>, SelectionError> {
        let key = SelectionKey::new(req.model_name.clone(), req.tenant_id.clone());
        let entry = self.ready_entry(&key)?;
        let prepared = self.prepare_selection_inputs(&entry, &req.prompt).await?;
        let track_prefill_tokens = req
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.track_prefill_tokens)
            .unwrap_or(self.kv_router_config.router_track_prefill_tokens);
        Ok(entry.scheduler.get_potential_loads(
            Some(prepared.sequence_hashes),
            prepared.isl_tokens,
            prepared.overlap.effective_cached_tokens,
            track_prefill_tokens,
        ))
    }

    pub async fn overlap_scores(
        &self,
        req: OverlapScoresRequest,
    ) -> Result<OverlapScoresResponse, SelectionError> {
        let key = SelectionKey::new(req.model_name.clone(), req.tenant_id.clone());
        let entry = self.ready_entry(&key)?;
        let normalized = req
            .prompt
            .normalize_for_selection(entry.block_size, entry.is_eagle)?;
        let num_blocks = normalized.block_hashes.len();
        let tiered = entry
            .indexer
            .find_tiered_matches(normalized.block_hashes)
            .await
            .map_err(|error| SelectionError::Internal(error.to_string()))?;
        let schedulable_workers = self.schedulable_worker_ranks(&key);
        Ok(
            OverlapAnalysis::new(&self.kv_router_config, entry.block_size, &tiered)
                .scores_response(
                    req.router_config_override.as_ref(),
                    num_blocks,
                    schedulable_workers,
                    false,
                    None,
                    None,
                ),
        )
    }

    async fn prepare_selection_inputs(
        &self,
        entry: &SelectionEntry,
        prompt: &PromptRequest,
    ) -> Result<PreparedSelectionInputs, SelectionError> {
        let normalized = prompt.normalize_for_selection(entry.block_size, entry.is_eagle)?;
        let tiered = if normalized.block_hashes.is_empty() {
            TieredMatchDetails::default()
        } else {
            entry
                .indexer
                .find_tiered_matches(normalized.block_hashes.clone())
                .await
                .map_err(|error| SelectionError::Internal(error.to_string()))?
        };
        let overlap =
            OverlapAnalysis::new(&self.kv_router_config, entry.block_size, &tiered).signals();
        drop(tiered);
        Ok(PreparedSelectionInputs {
            block_hashes: normalized.block_hashes,
            sequence_hashes: normalized.sequence_hashes,
            isl_tokens: normalized.isl_tokens,
            overlap,
        })
    }

    fn schedulable_worker_ranks(&self, key: &SelectionKey) -> Vec<WorkerWithDpRank> {
        let configs = self.catalog.scheduler_configs_for_key(key);
        let mut workers = Vec::new();
        for (worker_id, config) in configs {
            let start = config.data_parallel_start_rank;
            let end = start.saturating_add(config.data_parallel_size);
            for dp_rank in start..end {
                workers.push(WorkerWithDpRank::new(worker_id, dp_rank));
            }
        }
        workers
    }
}

impl Drop for SelectionCore {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::StorageTier;
    use crate::services::indexer::backend::test_util::store_event;
    use std::time::Duration;

    fn test_config(use_kv_events: bool) -> crate::config::KvRouterConfig {
        crate::config::KvRouterConfig {
            use_kv_events,
            router_queue_threshold: None,
            ..Default::default()
        }
    }

    fn worker(worker_id: WorkerId) -> WorkerRequest {
        WorkerRequest {
            worker_id,
            model_name: "model".to_string(),
            tenant_id: "default".to_string(),
            endpoint: Some(format!("http://worker-{worker_id}:8000")),
            kv_events_endpoint: None,
            kv_events_endpoints: HashMap::new(),
            replay_endpoint: None,
            block_size: Some(4),
            data_parallel_start_rank: None,
            data_parallel_size: None,
            max_num_batched_tokens: Some(1024),
            total_kv_blocks: None,
            stable_routing_id: None,
            is_eagle: None,
            taints: HashSet::new(),
            topology_domains: HashMap::new(),
            kv_transfer_domain: None,
            kv_transfer_enforcement: None,
            kv_transfer_preferred_weight: None,
        }
    }

    fn worker_with_kv_events(worker_id: WorkerId) -> WorkerRequest {
        WorkerRequest {
            kv_events_endpoint: Some("tcp://127.0.0.1:5557".to_string()),
            ..worker(worker_id)
        }
    }

    fn prompt() -> PromptRequest {
        PromptRequest {
            token_ids: Some(vec![1, 2, 3, 4]),
            mm_routing_info: None,
            block_mm_infos: None,
            block_hashes: None,
            sequence_hashes: None,
            isl_tokens: None,
            lora_name: None,
            is_eagle: None,
        }
    }

    fn select_request() -> SelectRequest {
        SelectRequest {
            model_name: "model".to_string(),
            tenant_id: "default".to_string(),
            selection_id: None,
            prompt: prompt(),
            router_config_override: None,
            expected_output_tokens: None,
            priority_jump: None,
            strict_priority: None,
            session_id: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
        }
    }

    fn reserve_request(reservation_id: &str) -> SelectAndReserveRequest {
        SelectAndReserveRequest {
            model_name: "model".to_string(),
            tenant_id: "default".to_string(),
            selection_id: None,
            reservation_id: Some(reservation_id.to_string()),
            prompt: prompt(),
            router_config_override: None,
            expected_output_tokens: None,
            priority_jump: None,
            strict_priority: None,
            session_id: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
        }
    }

    async fn wait_for_pending_selection(core: &SelectionCore) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if core.loads(Some("model"), Some("default"))[0].pending_count == 1 {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("selection did not queue");
    }

    fn assert_shutdown_error(error: SelectionError) {
        assert!(matches!(
            error,
            SelectionError::NotReady(message)
                if message == "selection service is shutting down"
        ));
    }

    #[test]
    fn parent_cancel_cancels_core() {
        let parent = CancellationToken::new();
        let core = SelectionCore::new(test_config(false), 1, parent.clone());

        assert!(!core.cancel_token.is_cancelled());
        parent.cancel();
        assert!(core.cancel_token.is_cancelled());
    }

    #[test]
    fn shutdown_keeps_parent_alive() {
        let parent = CancellationToken::new();
        let core = SelectionCore::new(test_config(false), 1, parent.clone());

        core.shutdown();

        assert!(core.cancel_token.is_cancelled());
        assert!(!parent.is_cancelled());
    }

    #[tokio::test]
    async fn shutdown_cancels_listeners() {
        let parent = CancellationToken::new();
        let core = SelectionCore::new(test_config(true), 1, parent);

        let record = core
            .upsert_worker(worker_with_kv_events(1))
            .await
            .expect("worker upsert");
        assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable);
        assert_eq!(core.indexer_registry.listener_cancelled(1, 0), Some(false));

        core.shutdown();
        assert_eq!(core.indexer_registry.listener_cancelled(1, 0), Some(true));
    }

    #[tokio::test]
    async fn shutdown_reports_not_ready_and_rejects_new_work() {
        let core = SelectionCore::new(test_config(false), 1, CancellationToken::new());
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        assert!(core.ready().ready);

        core.shutdown();

        let ready = core.ready();
        assert!(!ready.ready);
        assert_eq!(ready.schedulable_workers, 1);

        let upsert_error = core
            .upsert_worker(worker(2))
            .await
            .expect_err("upsert should fail after shutdown");
        assert_shutdown_error(upsert_error);

        let patch = serde_json::from_value(serde_json::json!({
            "endpoint": "http://worker-1:9000"
        }))
        .expect("worker patch");
        let patch_error = core
            .patch_worker(1, patch)
            .await
            .expect_err("patch should fail after shutdown");
        assert_shutdown_error(patch_error);

        let select_error = core
            .select(select_request())
            .await
            .expect_err("selection should fail after shutdown");
        assert_shutdown_error(select_error);

        let reservation_error = core
            .create_reservation(ReservationRequest {
                model_name: "model".to_string(),
                tenant_id: "default".to_string(),
                reservation_id: "res-after-shutdown".to_string(),
                worker_id: 1,
                dp_rank: None,
                prompt: prompt(),
                router_config_override: None,
                expected_output_tokens: None,
                effective_prefill_tokens: None,
            })
            .await
            .expect_err("reservation should fail after shutdown");
        assert_shutdown_error(reservation_error);

        assert_eq!(core.list_workers(None, None).len(), 1);
        assert_eq!(core.loads(None, None).len(), 1);
        let deleted = core
            .delete_worker(1)
            .await
            .expect("delete should remain available after shutdown");
        assert_eq!(deleted.lifecycle, WorkerLifecycle::Unschedulable);
    }

    #[tokio::test]
    async fn queued_selection_errors_on_shutdown() {
        let mut config = test_config(false);
        config.router_queue_threshold = Some(0.0);
        let core = Arc::new(SelectionCore::new(config, 1, CancellationToken::new()));

        let record = core.upsert_worker(worker(1)).await.expect("worker upsert");
        assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable);
        core.select_and_reserve(reserve_request("res-a"))
            .await
            .expect("initial reservation");

        let queued_core = core.clone();
        let queued = tokio::spawn(async move { queued_core.select(select_request()).await });
        wait_for_pending_selection(&core).await;

        core.shutdown();
        let err = tokio::time::timeout(Duration::from_secs(1), queued)
            .await
            .expect("queued selection timed out")
            .expect("queued selection task panicked")
            .expect_err("queued selection should fail");

        assert!(matches!(
            err,
            SelectionError::Scheduler(KvSchedulerError::SubscriberShutdown)
        ));
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn queued_selection_returns_refreshed_overlap_snapshot() {
        let mut config = test_config(false);
        config.router_queue_threshold = Some(0.0);
        let core = Arc::new(SelectionCore::new(config, 1, CancellationToken::new()));

        for worker_id in [1, 2] {
            let mut request = worker(worker_id);
            request.max_num_batched_tokens = Some(8);
            core.upsert_worker(request).await.expect("worker upsert");
        }
        let key = SelectionKey::new("model".to_string(), "default".to_string());
        let entry = core.entries.read().get(&key).cloned().expect("entry");
        entry
            .indexer
            .apply_event_routed(store_event(1, 0, 1, &[], &[11], StorageTier::Device))
            .await;
        entry.indexer.dump_events().await.expect("flush indexer");

        for worker_id in [1, 2] {
            core.create_reservation(ReservationRequest {
                model_name: "model".to_string(),
                tenant_id: "default".to_string(),
                reservation_id: format!("occupy-{worker_id}"),
                worker_id,
                dp_rank: Some(0),
                prompt: PromptRequest {
                    token_ids: None,
                    mm_routing_info: None,
                    block_mm_infos: None,
                    block_hashes: None,
                    sequence_hashes: Some(vec![1, 2]),
                    isl_tokens: Some(8),
                    lora_name: None,
                    is_eagle: None,
                },
                router_config_override: None,
                expected_output_tokens: None,
                effective_prefill_tokens: Some(8),
            })
            .await
            .expect("occupy worker");
        }

        let queued_core = Arc::clone(&core);
        let queued = tokio::spawn(async move {
            queued_core
                .select_and_reserve(SelectAndReserveRequest {
                    model_name: "model".to_string(),
                    tenant_id: "default".to_string(),
                    selection_id: Some("refresh-selection".to_string()),
                    reservation_id: Some("refreshed-request".to_string()),
                    prompt: PromptRequest {
                        token_ids: None,
                        mm_routing_info: None,
                        block_mm_infos: None,
                        block_hashes: Some(vec![11, 12]),
                        sequence_hashes: Some(vec![101, 102]),
                        isl_tokens: Some(8),
                        lora_name: None,
                        is_eagle: None,
                    },
                    router_config_override: None,
                    expected_output_tokens: None,
                    priority_jump: None,
                    strict_priority: None,
                    session_id: None,
                    pinned_worker: None,
                    allowed_worker_ids: None,
                    routing_constraints: RoutingConstraints::default(),
                })
                .await
        });
        wait_for_pending_selection(&core).await;

        entry
            .indexer
            .apply_event_routed(store_event(2, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await;
        entry.indexer.dump_events().await.expect("flush indexer");
        tokio::time::advance(Duration::from_secs(11)).await;
        core.free_reservation("occupy-2")
            .await
            .expect("release worker 2");

        let response = queued.await.expect("selection task").expect("selection");
        assert_eq!(response.worker_id, 2);
        assert_eq!(response.effective_prefill_tokens, 0);
        assert_eq!(response.overlap.gpu, 8);
        assert_eq!(response.overlap.cpu, 8);
        assert_eq!(response.overlap.disk, 8);
        assert_eq!(response.overlap.dp, HashMap::from([("0".to_string(), 8)]));
    }
}
