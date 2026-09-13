// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker membership: catalog upserts, patches and deletes, partition
//! creation, indexer registration and scheduler config publication.

use super::reservations::{ReservationIndexObserver, spawn_reservation_index_sweep};
use super::*;
use crate::services::selection::ingress::remove_worker_from_index;

impl SelectionCore {
    pub async fn upsert_worker(
        &self,
        req: WorkerRequest,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        self.ensure_running()?;
        let mut record = WorkerCatalogRecord::new(req);
        // Partition policy factories may construct independently. Only committing
        // membership and reconciling ingress needs the catalog mutation lock.
        self.prepare_worker(&mut record)?;
        let _update = self.catalog_updates.lock().await;
        self.ensure_running()?;
        let previous = self.catalog.get(record.worker_id);
        self.reconcile_worker(record, previous).await
    }

    pub async fn patch_worker(
        &self,
        worker_id: WorkerId,
        patch: WorkerPatchRequest,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        let _update = self.catalog_updates.lock().await;
        self.ensure_running()?;
        let previous = self
            .catalog
            .get(worker_id)
            .ok_or_else(|| SelectionError::NotFound(format!("worker {worker_id} not found")))?;
        let mut record = previous.clone();
        record.apply_patch(patch);
        self.prepare_worker(&mut record)?;
        self.reconcile_worker(record, Some(previous)).await
    }

    pub async fn delete_worker(
        &self,
        worker_id: WorkerId,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        let _update = self.catalog_updates.lock().await;
        let Some(previous) = self.catalog.get(worker_id) else {
            return Err(SelectionError::NotFound(format!(
                "worker {worker_id} not found"
            )));
        };
        let key = previous.key();
        self.catalog
            .set_lifecycle(worker_id, WorkerLifecycle::Draining, Vec::new());
        self.publish_scheduler_config(&key);
        self.cleanup_indexer_registration(&previous).await;
        let record = self
            .catalog
            .set_lifecycle(worker_id, WorkerLifecycle::Unschedulable, Vec::new())
            .ok_or_else(|| SelectionError::NotFound(format!("worker {worker_id} not found")))?;
        self.publish_scheduler_config(&key);
        // Departed workers do not linger: every reader treats a missing record
        // like an unschedulable one, and the catalog stays bounded by live ids.
        self.catalog.remove(worker_id);
        Ok(record)
    }

    pub fn list_workers(
        &self,
        model_name: Option<&str>,
        routing_group: Option<&str>,
    ) -> Vec<WorkerCatalogRecord> {
        self.catalog.list(model_name, routing_group)
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

    fn prepare_worker(&self, record: &mut WorkerCatalogRecord) -> Result<(), SelectionError> {
        let queueing_enabled = self
            .kv_router_config
            .queueing_enabled(Some(&record.model_name))
            .map_err(|error| SelectionError::BadRequest(error.to_string()))?;
        record.not_schedulable_reasons = record.missing_schedulable_metadata(queueing_enabled);
        if let Some(ingress) = self.ingress() {
            record
                .not_schedulable_reasons
                .extend(ingress.missing_metadata(record));
        }
        if record.not_schedulable_reasons.is_empty()
            && let Err(error) = self.ensure_entry(record)
        {
            record
                .not_schedulable_reasons
                .push(format!("reconciliation failed: {error}"));
        }

        Ok(())
    }

    async fn reconcile_worker(
        &self,
        mut record: WorkerCatalogRecord,
        previous: Option<WorkerCatalogRecord>,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        let previous = previous.filter(|old| old.lifecycle == WorkerLifecycle::Schedulable);
        let previous = if let Some(old) = previous.as_ref()
            && (old.key() != record.key() || !record.not_schedulable_reasons.is_empty())
        {
            self.catalog
                .set_lifecycle(old.worker_id, WorkerLifecycle::Draining, Vec::new());
            self.publish_scheduler_config(&old.key());
            self.cleanup_indexer_registration(old).await;
            None
        } else {
            previous
        };

        if record.not_schedulable_reasons.is_empty()
            && let Some(ingress) = self.ingress()
            && let Err(error) = ingress
                .reconcile(&self.indexer_registry, previous.as_ref(), &record)
                .await
        {
            self.cleanup_indexer_registration(&record).await;
            record
                .not_schedulable_reasons
                .push(format!("reconciliation failed: {error}"));
        }
        record.lifecycle = if record.not_schedulable_reasons.is_empty() {
            WorkerLifecycle::Schedulable
        } else {
            WorkerLifecycle::Incomplete
        };
        // Readers see only committed metadata. A valid capacity/topology update preserves
        // live bookings on ranks present in both the old and new snapshots.
        self.catalog.replace(record.clone());
        self.publish_scheduler_config(&record.key());
        Ok(record)
    }

    fn ensure_entry(
        &self,
        record: &WorkerCatalogRecord,
    ) -> Result<Arc<SelectionEntry>, SelectionError> {
        let block_size = record
            .block_size
            .ok_or_else(|| SelectionError::BadRequest("block_size is required".to_string()))?;
        self.ensure_entry_for(record.key(), block_size, record.is_eagle.unwrap_or(false))
    }

    /// Create the partition scheduler and indexer for `key` before any worker
    /// registers, so an embedding host can hold the scheduler handle from
    /// construction. Idempotent; a later worker with a different block size or
    /// eagle setting is rejected at reconciliation.
    pub fn ensure_partition(
        &self,
        key: RoutingPartitionId,
        block_size: u32,
        is_eagle: bool,
    ) -> Result<SelectionPartition, SelectionError> {
        self.ensure_running()?;
        if block_size == 0 {
            return Err(SelectionError::BadRequest(
                "block_size must be greater than 0".to_string(),
            ));
        }
        self.ensure_entry_for(key, block_size, is_eagle)
            .map(SelectionPartition)
    }

    fn ensure_entry_for(
        &self,
        key: RoutingPartitionId,
        block_size: u32,
        is_eagle: bool,
    ) -> Result<Arc<SelectionEntry>, SelectionError> {
        self.reservation_sweep_started.get_or_init(|| {
            spawn_reservation_index_sweep(
                Arc::clone(&self.entries),
                Arc::clone(&self.reservation_index),
                self.cancel_token.child_token(),
            );
        });

        let entry_cell = { self.entries.read().get(&key).cloned() };
        let entry_cell = entry_cell.unwrap_or_else(|| {
            self.entries
                .write()
                .entry(key.clone())
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        });
        let entry = entry_cell
            .get_or_try_init(|| -> Result<Arc<SelectionEntry>, SelectionError> {
                let (workers_tx, workers_rx) = watch::channel(HashMap::new());
                let host_replica = self
                    .host
                    .replication
                    .channels
                    .as_ref()
                    .and_then(|factory| factory(&key));
                let scoped_replica_sync = setup_scoped_replica_sync(
                    self.replica_config.as_ref(),
                    &key,
                    block_size,
                    host_replica,
                );
                let worker_label = self.worker_type.as_str();
                let slots = Arc::new(ActiveSequencesMultiWorker::new_with_options(
                    scoped_replica_sync
                        .publisher
                        .with_load_sink(self.host.telemetry.scheduler_load.clone()),
                    block_size as usize,
                    HashMap::new(),
                    scoped_replica_sync.enabled,
                    scoped_replica_sync.process_id,
                    worker_label,
                    SequenceTrackerOptions {
                        replica_worker_policy: ReplicaWorkerPolicy::RequireRegistered,
                        expiry_duration: self
                            .host
                            .replication
                            .request_leases
                            .is_none()
                            .then(active_request_expiry_duration),
                    },
                ));
                slots.set_replica_request_lease_observer(Arc::new(ReservationIndexObserver {
                    index: Arc::clone(&self.reservation_index),
                    partition: key.clone(),
                    host: self.host.replication.request_leases.clone(),
                }));
                let replica_tx = scoped_replica_sync.channel.map(|(replica_tx, subscriber)| {
                    slots.start_replica_sync(subscriber, self.cancel_token.child_token());
                    replica_tx
                });
                if self.host.replication.request_leases.is_none() {
                    slots.start_periodic_force_expiry_across_all_workers(
                        self.cancel_token.child_token(),
                    );
                }

                let indexer = match &self.host.cache.index {
                    KvIndexSource::Owned(ingress) => {
                        ingress.open(&self.indexer_registry, &key, block_size)
                    }
                    KvIndexSource::Remote(_) => self
                        .indexer_registry
                        .get_or_create_indexer(key.clone(), block_size),
                };
                let overlap_refresh = indexer.supports_overlap_refresh().then(|| {
                    Arc::new(TieredOverlapRefresher::new(
                        indexer.clone(),
                        self.kv_router_config.clone(),
                        block_size,
                    ))
                });
                let selector = self.worker_selection_policy_factory.as_ref().map_or_else(
                    || WorkerSelectionPolicy::default(self.kv_router_config.clone(), worker_label),
                    |factory| factory(&self.kv_router_config, self.worker_type, key.as_ref()),
                );
                let profile = self
                    .kv_router_config
                    .policy_profile(Some(&key.model_name))
                    .map_err(|error| SelectionError::BadRequest(error.to_string()))?;
                let scheduler = LocalScheduler::new(
                    slots,
                    workers_rx,
                    profile,
                    block_size,
                    selector,
                    self.host.load.prefill_estimator.clone(),
                    overlap_refresh,
                    // Standalone selection has no router Client snapshot, so
                    // these stay `None` unless an embedding host injects them.
                    self.host.load.overloaded_workers.clone(),
                    self.host.load.available_workers.clone(),
                    self.kv_router_config.router_queue_recheck_interval(),
                    self.kv_router_config.router_track_prefill_tokens,
                    self.cancel_token.child_token(),
                    worker_label,
                    true,
                );
                Ok(Arc::new(SelectionEntry {
                    key: key.clone(),
                    block_size,
                    is_eagle,
                    indexer,
                    workers_tx,
                    hint_capable: AtomicBool::new(false),
                    scheduler,
                    replica_tx,
                    affinity: OnceCell::new(),
                    replica_config: self.replica_config.clone(),
                }))
            })?
            .clone();
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
        if let Some(config) = self.session_affinity {
            entry.session_affinity(config)?;
        }
        Ok(entry)
    }

    /// The ingress feeding core-owned indexes, when this core listens for KV events.
    fn ingress(&self) -> Option<&dyn KvEventIngress> {
        match &self.host.cache.index {
            KvIndexSource::Owned(ingress) if self.listens_for_kv_events => Some(ingress.as_ref()),
            _ => None,
        }
    }

    async fn cleanup_indexer_registration(&self, record: &WorkerCatalogRecord) {
        if let KvIndexSource::Owned(ingress) = &self.host.cache.index {
            ingress.detach(&self.indexer_registry, record).await;
            return;
        }
        remove_worker_from_index(&self.indexer_registry, record).await;
    }

    pub(super) fn publish_scheduler_config(&self, key: &RoutingPartitionId) {
        let Some(entry) = self.entry(key) else {
            return;
        };
        let (workers, hint_capable) = self.catalog.partition_view(key);
        entry.hint_capable.store(hint_capable, Ordering::Release);
        // Lifecycle transitions between non-schedulable states publish the same
        // map; skipping them saves the scheduler a wake and a full map clone.
        entry.workers_tx.send_if_modified(|current| {
            if *current == workers {
                false
            } else {
                *current = workers;
                true
            }
        });
    }
}
