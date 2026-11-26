// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::{Result, bail};
use dashmap::DashMap;
use dashmap::mapref::one::Ref;
use parking_lot::Mutex;
use serde::Serialize;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use dynamo_kv_indexer::KvIndexerMetrics;
use dynamo_router_protocol::identity::RoutingPartitionId;
use dynamo_router_protocol::worker::WorkerId;

use super::backend::Indexer;
#[cfg(test)]
use super::listener::BatchGate;
use super::listener::spawn_zmq_listener;

pub use dynamo_kv_indexer::{IndexClaimError, IndexRegistry, IndexerEntry, LifecycleGuard};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ListenerStatus {
    Pending,
    Active,
    Paused,
    Failed,
}

impl ListenerStatus {
    pub const ALL: [Self; 4] = [Self::Pending, Self::Active, Self::Paused, Self::Failed];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Failed => "failed",
        }
    }

    pub fn metric_index(self) -> usize {
        match self {
            Self::Pending => 0,
            Self::Active => 1,
            Self::Paused => 2,
            Self::Failed => 3,
        }
    }

    pub fn aggregate(statuses: impl IntoIterator<Item = Self>) -> Self {
        let mut saw_pending = false;
        let mut saw_active = false;

        for status in statuses {
            match status {
                Self::Failed => return Self::Failed,
                Self::Pending => saw_pending = true,
                Self::Active => saw_active = true,
                Self::Paused => {}
            }
        }

        if saw_pending {
            Self::Pending
        } else if saw_active {
            Self::Active
        } else {
            Self::Paused
        }
    }
}

impl fmt::Display for ListenerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerSource {
    Zmq,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListenerInfo {
    endpoint: String,
    engine_owned_mm_identity: bool,
    idempotent_stores: bool,
    status: ListenerStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerInfo {
    instance_id: WorkerId,
    source: WorkerSource,
    status: ListenerStatus,
    model_name: String,
    routing_group: String,
    block_size: u32,
    endpoints: HashMap<u32, String>,
    listeners: HashMap<u32, ListenerInfo>,
}

#[derive(Debug, thiserror::Error)]
pub enum ListenerControlError {
    #[error("instance {instance_id} not found")]
    WorkerNotFound { instance_id: WorkerId },

    #[error("instance {instance_id} dp_rank {dp_rank} not found")]
    ListenerNotFound { instance_id: WorkerId, dp_rank: u32 },

    #[error("instance {instance_id} dp_rank {dp_rank} cannot be paused from status {status}")]
    InvalidPauseState {
        instance_id: WorkerId,
        dp_rank: u32,
        status: ListenerStatus,
    },

    #[error("instance {instance_id} dp_rank {dp_rank} cannot be resumed from status {status}")]
    InvalidResumeState {
        instance_id: WorkerId,
        dp_rank: u32,
        status: ListenerStatus,
    },
}

struct ListenerRuntime {
    status: ListenerStatus,
    last_error: Option<String>,
    cancel_token: Option<CancellationToken>,
    generation: u64,
    /// The newest attempt's task. Each attempt waits for the one before it, so joining
    /// this joins them all. Kept while paused or failed: a cancelled task may not have
    /// exited yet.
    task: Option<JoinHandle<()>>,
}

pub struct ListenerRecord {
    endpoint: String,
    replay_endpoint: Option<String>,
    engine_owned_mm_identity: bool,
    idempotent_stores: bool,
    block_size: u32,
    indexer: Indexer,
    watermark: Arc<AtomicU64>,
    runtime: Mutex<ListenerRuntime>,
    /// Cancelled by deregistration, which drops every attempt at its next await point.
    /// Pausing cancels only the attempt's own token, which lets a batch finish. Read and
    /// cancelled under `runtime`, so a resume never starts after the retirement.
    retired: CancellationToken,
    /// Test hook: handed to the first listener attempt that reaches its receive loop.
    #[cfg(test)]
    batch_gate: Mutex<Option<BatchGate>>,
}

impl ListenerRecord {
    fn new(
        endpoint: String,
        replay_endpoint: Option<String>,
        engine_owned_mm_identity: bool,
        idempotent_stores: bool,
        block_size: u32,
        indexer: Indexer,
        watermark: Arc<AtomicU64>,
    ) -> Self {
        Self {
            endpoint,
            replay_endpoint,
            engine_owned_mm_identity,
            idempotent_stores,
            block_size,
            indexer,
            watermark,
            runtime: Mutex::new(ListenerRuntime {
                status: ListenerStatus::Pending,
                last_error: None,
                cancel_token: None,
                generation: 0,
                task: None,
            }),
            retired: CancellationToken::new(),
            #[cfg(test)]
            batch_gate: Mutex::new(None),
        }
    }

    #[cfg(test)]
    pub(super) fn take_batch_gate(&self) -> Option<BatchGate> {
        self.batch_gate.lock().take()
    }

    pub(super) fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub(super) fn replay_endpoint(&self) -> Option<&str> {
        self.replay_endpoint.as_deref()
    }

    pub(super) fn engine_owned_mm_identity(&self) -> bool {
        self.engine_owned_mm_identity
    }

    pub(super) fn idempotent_stores(&self) -> bool {
        self.idempotent_stores
    }

    pub(super) fn block_size(&self) -> u32 {
        self.block_size
    }

    pub(super) fn indexer(&self) -> Indexer {
        self.indexer.clone()
    }

    pub(super) fn watermark(&self) -> Arc<AtomicU64> {
        self.watermark.clone()
    }

    pub(super) fn retired(&self) -> CancellationToken {
        self.retired.clone()
    }

    /// Start every listener through here, with the runtime lock held: a listener whose
    /// handle is not yet in the record cannot be joined by a deregistration.
    fn spawn_attempt_locked(
        self: &Arc<Self>,
        runtime: &mut ListenerRuntime,
        worker_id: WorkerId,
        dp_rank: u32,
        root_cancel_token: &CancellationToken,
        ready: watch::Receiver<bool>,
    ) {
        runtime.generation += 1;
        let cancel_token = root_cancel_token.child_token();
        runtime.status = ListenerStatus::Pending;
        runtime.last_error = None;
        runtime.cancel_token = Some(cancel_token.clone());
        let previous = runtime.task.take();
        runtime.task = Some(spawn_zmq_listener(
            worker_id,
            dp_rank,
            self.clone(),
            ready,
            runtime.generation,
            cancel_token.child_token(),
            previous,
        ));
    }

    pub(super) fn start(
        self: &Arc<Self>,
        worker_id: WorkerId,
        dp_rank: u32,
        root_cancel_token: &CancellationToken,
        ready: watch::Receiver<bool>,
    ) {
        let mut runtime = self.runtime.lock();
        self.spawn_attempt_locked(&mut runtime, worker_id, dp_rank, root_cancel_token, ready);
    }

    /// Stops the attempt once its current batch is applied, so the watermark covers what
    /// it indexed.
    pub(super) fn pause(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
    ) -> std::result::Result<(), ListenerControlError> {
        let mut runtime = self.runtime.lock();
        if self.retired.is_cancelled() {
            return Err(ListenerControlError::ListenerNotFound {
                instance_id,
                dp_rank,
            });
        }
        match runtime.status {
            ListenerStatus::Pending | ListenerStatus::Active => {
                let cancel_token =
                    runtime
                        .cancel_token
                        .take()
                        .ok_or(ListenerControlError::InvalidPauseState {
                            instance_id,
                            dp_rank,
                            status: runtime.status,
                        })?;
                cancel_token.cancel();
                runtime.status = ListenerStatus::Paused;
                runtime.last_error = None;
                Ok(())
            }
            status => Err(ListenerControlError::InvalidPauseState {
                instance_id,
                dp_rank,
                status,
            }),
        }
    }

    pub(super) fn resume(
        self: &Arc<Self>,
        instance_id: WorkerId,
        dp_rank: u32,
        root_cancel_token: &CancellationToken,
        ready: watch::Receiver<bool>,
    ) -> std::result::Result<(), ListenerControlError> {
        let mut runtime = self.runtime.lock();
        if self.retired.is_cancelled() {
            // The caller cloned this record out of the map before it was deregistered.
            return Err(ListenerControlError::ListenerNotFound {
                instance_id,
                dp_rank,
            });
        }
        match runtime.status {
            ListenerStatus::Paused | ListenerStatus::Failed => {
                self.spawn_attempt_locked(
                    &mut runtime,
                    instance_id,
                    dp_rank,
                    root_cancel_token,
                    ready,
                );
                Ok(())
            }
            status => Err(ListenerControlError::InvalidResumeState {
                instance_id,
                dp_rank,
                status,
            }),
        }
    }

    pub(super) fn is_current_attempt(&self, generation: u64) -> bool {
        let runtime = self.runtime.lock();
        runtime.generation == generation && runtime.cancel_token.is_some()
    }

    pub(super) fn try_mark_active(&self, generation: u64) -> bool {
        let mut runtime = self.runtime.lock();
        if runtime.generation != generation || runtime.cancel_token.is_none() {
            return false;
        }
        runtime.status = ListenerStatus::Active;
        runtime.last_error = None;
        true
    }

    pub(super) fn try_mark_failed(&self, generation: u64, error: impl Into<String>) {
        let mut runtime = self.runtime.lock();
        if runtime.generation != generation || runtime.cancel_token.is_none() {
            return;
        }
        runtime.status = ListenerStatus::Failed;
        runtime.last_error = Some(error.into());
        runtime.cancel_token = None;
    }

    /// Stops this record for good and returns its task for the caller to join. A retired
    /// record cannot be paused or resumed, and `try_mark_*` refuse once the token is gone.
    fn retire(&self) -> Option<JoinHandle<()>> {
        let mut runtime = self.runtime.lock();
        self.retired.cancel();
        if let Some(cancel_token) = runtime.cancel_token.take() {
            cancel_token.cancel();
        }
        runtime.task.take()
    }

    fn snapshot(&self) -> ListenerInfo {
        let runtime = self.runtime.lock();
        ListenerInfo {
            endpoint: self.endpoint.clone(),
            engine_owned_mm_identity: self.engine_owned_mm_identity,
            idempotent_stores: self.idempotent_stores,
            status: runtime.status,
            last_error: runtime.last_error.clone(),
        }
    }

    #[cfg(all(feature = "metrics", feature = "http"))]
    fn status(&self) -> ListenerStatus {
        self.runtime.lock().status
    }
}

pub struct WorkerEntry {
    key: RoutingPartitionId,
    listeners: HashMap<u32, Arc<ListenerRecord>>,
}

/// A retired listener's task, by dp_rank.
type ListenerTask = (u32, JoinHandle<()>);

enum Removal {
    Worker,
    DpRank(u32),
}

#[derive(Default)]
struct WorkerLock {
    mutex: Arc<tokio::sync::Mutex<()>>,
    /// Operations holding or waiting for `mutex`; the entry goes when none is left.
    users: usize,
}

type WorkerLocks = Arc<DashMap<WorkerId, WorkerLock>>;

/// Held by every registration and removal of one worker, until a removal is queued.
/// Counted as a user from before it waits, so a waiter dropped mid-wait is counted out.
struct WorkerLifecycleGuard {
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
    locks: WorkerLocks,
    worker_id: WorkerId,
}

impl Drop for WorkerLifecycleGuard {
    fn drop(&mut self) {
        drop(self.guard.take());
        self.locks.remove_if_mut(&self.worker_id, |_, lock| {
            lock.users -= 1;
            lock.users == 0
        });
    }
}

/// Joins the listeners, then enqueues the removal behind everything they sent.
///
/// Both run in a task of their own: the worker has already left the registry, so a
/// caller dropped in between (a client gone from `/unregister`) must not leave it indexed.
/// The task also keeps `lifecycle` until the removal is queued, so a new registration of
/// this worker cannot index blocks the removal would then delete.
async fn join_then_remove(
    lifecycle: WorkerLifecycleGuard,
    worker_id: WorkerId,
    tasks: Vec<ListenerTask>,
    indexer: Option<Indexer>,
    removal: Removal,
) {
    let task = tokio::spawn(async move {
        let _lifecycle = lifecycle;
        for (dp_rank, task) in tasks {
            if let Err(error) = task.await {
                tracing::error!(worker_id, dp_rank, %error, "ZMQ listener task ended abnormally");
            }
        }
        let Some(indexer) = indexer else {
            return;
        };
        match removal {
            Removal::Worker => indexer.remove_worker(worker_id).await,
            // Waits for the lane, so a lookup after deregistration no longer scores the rank.
            Removal::DpRank(dp_rank) => {
                if let Err(error) = indexer
                    .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                    .await
                {
                    tracing::error!(worker_id, dp_rank, %error, "Failed to remove worker rank");
                }
            }
        }
    });
    if let Err(error) = task.await {
        tracing::error!(worker_id, %error, "Worker removal task ended abnormally");
    }
}

pub struct WorkerRegistry {
    workers: DashMap<WorkerId, WorkerEntry>,
    watermarks: DashMap<(WorkerId, u32), Arc<AtomicU64>>,
    peers: DashMap<String, ()>,
    root_cancel_token: CancellationToken,
    /// The per-partition indexers these listeners feed; shared with whoever reads them.
    index: Arc<IndexRegistry>,
    worker_locks: WorkerLocks,
}

impl WorkerRegistry {
    pub fn new(num_threads: usize) -> Self {
        Self::new_with_cancel_token(num_threads, CancellationToken::new())
    }

    pub fn new_with_cancel_token(num_threads: usize, root_cancel_token: CancellationToken) -> Self {
        Self::new_inner(
            num_threads,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            root_cancel_token,
        )
    }

    #[cfg(all(feature = "metrics", feature = "http"))]
    pub(super) fn new_with_indexer_metrics_and_cancel_token(
        num_threads: usize,
        indexer_metrics: Arc<KvIndexerMetrics>,
        root_cancel_token: CancellationToken,
    ) -> Self {
        Self::new_inner(num_threads, indexer_metrics, root_cancel_token)
    }

    fn new_inner(
        num_threads: usize,
        indexer_metrics: Arc<KvIndexerMetrics>,
        root_cancel_token: CancellationToken,
    ) -> Self {
        Self::with_index(
            Arc::new(IndexRegistry::new(num_threads, indexer_metrics)),
            root_cancel_token,
        )
    }

    /// Listeners over an index registry someone else owns too (the selection core's).
    pub fn with_index(index: Arc<IndexRegistry>, root_cancel_token: CancellationToken) -> Self {
        Self {
            workers: DashMap::new(),
            watermarks: DashMap::new(),
            peers: DashMap::new(),
            root_cancel_token,
            index,
            worker_locks: Arc::default(),
        }
    }

    /// Serializes registrations and removals of one worker; other workers are not blocked.
    /// Take it before the index lifecycle lock.
    async fn lock_worker(&self, worker_id: WorkerId) -> WorkerLifecycleGuard {
        let mutex = {
            let mut lock = self.worker_locks.entry(worker_id).or_default();
            lock.users += 1;
            lock.mutex.clone()
        };
        let mut lifecycle = WorkerLifecycleGuard {
            guard: None,
            locks: self.worker_locks.clone(),
            worker_id,
        };
        lifecycle.guard = Some(mutex.lock_owned().await);
        lifecycle
    }

    pub fn index(&self) -> &Arc<IndexRegistry> {
        &self.index
    }

    pub fn signal_ready(&self) {
        self.index.signal_ready();
    }

    pub fn ready_rx(&self) -> watch::Receiver<bool> {
        self.index.ready_rx()
    }

    pub fn register_peer(&self, url: String) {
        self.peers.entry(url).or_insert(());
    }

    pub fn deregister_peer(&self, url: &str) -> bool {
        self.peers.remove(url).is_some()
    }

    pub fn list_peers(&self) -> Vec<String> {
        self.peers.iter().map(|entry| entry.key().clone()).collect()
    }

    #[cfg(all(feature = "metrics", feature = "http"))]
    pub fn refresh_metrics(&self) {
        let models = self.index.len();
        let workers = self.workers.len();

        let mut listener_counts = [0_i64; 4];
        for entry in self.workers.iter() {
            for record in entry.value().listeners.values() {
                listener_counts[record.status().metric_index()] += 1;
            }
        }

        super::metrics::set_worker_state(models, workers, listener_counts);
    }

    #[expect(clippy::too_many_arguments)]
    pub async fn register(
        &self,
        instance_id: WorkerId,
        endpoint: String,
        dp_rank: u32,
        model_name: String,
        routing_group: String,
        block_size: u32,
        replay_endpoint: Option<String>,
        engine_owned_mm_identity: bool,
        idempotent_stores: bool,
    ) -> Result<()> {
        let key = RoutingPartitionId::new(model_name, routing_group);
        let _lifecycle = self.lock_worker(instance_id).await;
        let registration = self.index.lock_lifecycle().await;

        if let Some(entry) = self.workers.get(&instance_id) {
            if entry.key != key {
                bail!(
                    "instance {instance_id} is already registered for model={} routing_group={}",
                    entry.key.model_name,
                    entry.key.routing_group
                );
            }

            if entry.listeners.contains_key(&dp_rank) {
                bail!("instance {instance_id} dp_rank {dp_rank} already registered");
            }
        }

        let indexer = self.index.claim(&registration, &key, block_size)?;
        let bs = block_size;

        let watermark = self
            .watermarks
            .entry((instance_id, dp_rank))
            .or_insert_with(|| Arc::new(AtomicU64::new(u64::MAX)))
            .clone();

        let record = Arc::new(ListenerRecord::new(
            endpoint,
            replay_endpoint,
            engine_owned_mm_identity,
            idempotent_stores,
            bs,
            indexer,
            watermark,
        ));
        // Start before the record is visible in the map, so a deregistration that finds the
        // record also finds its task.
        record.start(
            instance_id,
            dp_rank,
            &self.root_cancel_token,
            self.ready_rx(),
        );

        {
            let mut entry = self
                .workers
                .entry(instance_id)
                .or_insert_with(|| WorkerEntry {
                    key: key.clone(),
                    listeners: HashMap::new(),
                });
            entry.listeners.insert(dp_rank, record);
        }

        drop(registration);
        Ok(())
    }

    pub async fn deregister(
        &self,
        instance_id: WorkerId,
        model_name: &str,
        routing_group: &str,
    ) -> Result<()> {
        let key = RoutingPartitionId::new(model_name, routing_group);
        let lifecycle = self.lock_worker(instance_id).await;

        if let Some(entry) = self.workers.get(&instance_id) {
            if entry.key != key {
                bail!(
                    "instance {instance_id} is registered for model={} routing_group={}",
                    entry.key.model_name,
                    entry.key.routing_group
                );
            }
        } else {
            bail!("instance {instance_id} not found");
        }

        self.remove_worker_entry(lifecycle, instance_id, &key).await;
        Ok(())
    }

    /// Takes the worker out of the registry, joins its listeners, then removes it from the
    /// indexer. The maps change before the first await, so no map guard is held across one.
    async fn remove_worker_entry(
        &self,
        lifecycle: WorkerLifecycleGuard,
        instance_id: WorkerId,
        key: &RoutingPartitionId,
    ) {
        let (tasks, indexer) = match self.workers.remove(&instance_id) {
            Some((_, entry)) => {
                for &dp_rank in entry.listeners.keys() {
                    self.watermarks.remove(&(instance_id, dp_rank));
                }
                // The listeners' own indexer: while they are joined, the partition may be
                // released or created again.
                let indexer = entry
                    .listeners
                    .values()
                    .next()
                    .map(|record| record.indexer());
                let tasks = entry
                    .listeners
                    .iter()
                    .filter_map(|(&dp_rank, record)| record.retire().map(|task| (dp_rank, task)))
                    .collect();
                (tasks, indexer)
            }
            None => (Vec::new(), None),
        };
        // Registration needs this map's write lock while removal can wait on indexer queues.
        let indexer = indexer.or_else(|| self.index.indexer(key));
        join_then_remove(lifecycle, instance_id, tasks, indexer, Removal::Worker).await;
        self.release_indexer_if_unused(key).await;
    }

    pub async fn deregister_dp_rank(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
        model_name: &str,
        routing_group: &str,
    ) -> Result<()> {
        let key = RoutingPartitionId::new(model_name, routing_group);
        let lifecycle = self.lock_worker(instance_id).await;

        let (record, last_rank) = {
            let mut entry = self
                .workers
                .get_mut(&instance_id)
                .ok_or_else(|| anyhow::anyhow!("instance {instance_id} not found"))?;

            if entry.key != key {
                bail!(
                    "instance {instance_id} is registered for model={} routing_group={}",
                    entry.key.model_name,
                    entry.key.routing_group
                );
            }

            let record = entry.listeners.remove(&dp_rank).ok_or_else(|| {
                anyhow::anyhow!("instance {instance_id} dp_rank {dp_rank} not found")
            })?;
            let last_rank = entry.listeners.is_empty();
            (record, last_rank)
        };

        // Everything that changes the maps stays before the first await.
        let tasks = record
            .retire()
            .map(|task| (dp_rank, task))
            .into_iter()
            .collect();
        self.watermarks.remove(&(instance_id, dp_rank));
        let remove_whole_worker = last_rank
            && self
                .workers
                .remove_if(&instance_id, |_, entry| entry.listeners.is_empty())
                .is_some();

        let removal = if remove_whole_worker {
            Removal::Worker
        } else {
            Removal::DpRank(dp_rank)
        };
        join_then_remove(
            lifecycle,
            instance_id,
            tasks,
            Some(record.indexer()),
            removal,
        )
        .await;
        if remove_whole_worker {
            self.release_indexer_if_unused(&key).await;
        }

        Ok(())
    }

    pub async fn deregister_all_routing_groups(
        &self,
        instance_id: WorkerId,
        model_name: &str,
    ) -> Result<()> {
        let lifecycle = self.lock_worker(instance_id).await;
        let key = if let Some(entry) = self.workers.get(&instance_id) {
            if entry.key.model_name != model_name {
                bail!(
                    "instance {instance_id} is registered for model={} routing_group={}",
                    entry.key.model_name,
                    entry.key.routing_group
                );
            }
            entry.key.clone()
        } else {
            bail!("instance {instance_id} not found");
        };

        self.remove_worker_entry(lifecycle, instance_id, &key).await;
        Ok(())
    }

    pub fn pause_listener(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
    ) -> std::result::Result<(), ListenerControlError> {
        let record = if let Some(entry) = self.workers.get(&instance_id) {
            entry.listeners.get(&dp_rank).cloned().ok_or(
                ListenerControlError::ListenerNotFound {
                    instance_id,
                    dp_rank,
                },
            )?
        } else {
            return Err(ListenerControlError::WorkerNotFound { instance_id });
        };

        record.pause(instance_id, dp_rank)?;
        tracing::info!(instance_id, dp_rank, "Paused ZMQ listener");
        Ok(())
    }

    pub async fn resume_listener(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
    ) -> std::result::Result<(), ListenerControlError> {
        let record = if let Some(entry) = self.workers.get(&instance_id) {
            entry.listeners.get(&dp_rank).cloned().ok_or(
                ListenerControlError::ListenerNotFound {
                    instance_id,
                    dp_rank,
                },
            )?
        } else {
            return Err(ListenerControlError::WorkerNotFound { instance_id });
        };

        record.resume(
            instance_id,
            dp_rank,
            &self.root_cancel_token,
            self.ready_rx(),
        )?;
        tracing::info!(instance_id, dp_rank, "Resumed ZMQ listener");
        Ok(())
    }

    /// Whether `worker_id` has listeners registered here.
    pub fn has_worker(&self, worker_id: WorkerId) -> bool {
        self.workers.contains_key(&worker_id)
    }

    pub fn has_listener(&self, worker_id: WorkerId, dp_rank: u32) -> bool {
        self.workers
            .get(&worker_id)
            .is_some_and(|entry| entry.listeners.contains_key(&dp_rank))
    }

    /// Drop what a rank's listener fed into the index after the listener itself
    /// is gone (its deregistration was cancelled before reaching this step).
    pub async fn remove_dp_rank_blocks(
        &self,
        worker_id: WorkerId,
        dp_rank: u32,
        key: &RoutingPartitionId,
    ) {
        // Clone the handle so the shard guard drops before the await.
        let indexer = self.index.indexer(key);
        if let Some(indexer) = indexer {
            indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
        }
    }

    pub fn list(&self) -> Vec<WorkerInfo> {
        self.list_filtered(None, None)
    }

    /// Return registered workers, optionally filtered by `model_name` and/or
    /// `routing_group`.  Pass `None` for a field to skip that filter.
    ///
    /// Workers that are mid-deregistration (listener map temporarily empty
    /// before the worker entry is removed) are silently omitted to avoid
    /// exposing `block_size = 0` in the response.
    pub fn list_filtered(
        &self,
        model_name: Option<&str>,
        routing_group: Option<&str>,
    ) -> Vec<WorkerInfo> {
        self.workers
            .iter()
            .filter_map(|entry| {
                let worker = entry.value();
                let key = &worker.key;

                // Apply caller-supplied filters.
                if model_name.is_some_and(|m| key.model_name != m)
                    || routing_group.is_some_and(|t| key.routing_group != t)
                {
                    return None;
                }

                // Skip workers that are mid-deregistration.  deregister_dp_rank
                // removes the last listener from the HashMap *before* calling
                // workers.remove(), so a concurrent list() can observe an empty
                // listener set.  Omit these entries rather than serializing
                // block_size = 0.
                if worker.listeners.is_empty() {
                    return None;
                }

                // block_size is authoritative on IndexerEntry (validated on
                // every register() call), not on individual listener records.
                // Read it from there to avoid the same TOCTOU.
                let block_size = self.index.block_size(key).unwrap_or(0);

                let listeners: HashMap<u32, ListenerInfo> = worker
                    .listeners
                    .iter()
                    .map(|(dp_rank, record)| (*dp_rank, record.snapshot()))
                    .collect();
                let endpoints: HashMap<u32, String> = listeners
                    .iter()
                    .map(|(dp_rank, info)| (*dp_rank, info.endpoint.clone()))
                    .collect();
                let status = ListenerStatus::aggregate(listeners.values().map(|info| info.status));
                Some(WorkerInfo {
                    instance_id: *entry.key(),
                    source: WorkerSource::Zmq,
                    status,
                    model_name: key.model_name.clone(),
                    routing_group: key.routing_group.clone(),
                    block_size,
                    endpoints,
                    listeners,
                })
            })
            .collect()
    }

    pub fn get_indexer(
        &self,
        key: &RoutingPartitionId,
    ) -> Option<Ref<'_, RoutingPartitionId, IndexerEntry>> {
        self.index.get_indexer(key)
    }

    pub fn get_or_create_indexer(&self, key: RoutingPartitionId, block_size: u32) -> Indexer {
        self.index.get_or_create_indexer(key, block_size)
    }

    pub fn all_indexers_with_block_size(&self) -> Vec<(RoutingPartitionId, Indexer, u32)> {
        self.index.all_indexers_with_block_size()
    }

    #[cfg(test)]
    pub(crate) fn forget_listener(&self, instance_id: WorkerId, dp_rank: u32) {
        if let Some(mut entry) = self.workers.get_mut(&instance_id) {
            entry.listeners.remove(&dp_rank);
        }
    }

    #[cfg(test)]
    pub(crate) fn record_for_test(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
    ) -> Option<Arc<ListenerRecord>> {
        self.workers
            .get(&instance_id)
            .and_then(|entry| entry.listeners.get(&dp_rank).cloned())
    }

    /// `resume_listener` on a record the caller cloned out of the map earlier.
    #[cfg(test)]
    pub(crate) fn resume_record_for_test(
        &self,
        record: &Arc<ListenerRecord>,
        instance_id: WorkerId,
        dp_rank: u32,
    ) -> std::result::Result<(), ListenerControlError> {
        record.resume(
            instance_id,
            dp_rank,
            &self.root_cancel_token,
            self.ready_rx(),
        )
    }

    #[cfg(test)]
    pub(crate) fn listener_cancelled(&self, instance_id: WorkerId, dp_rank: u32) -> Option<bool> {
        self.workers.get(&instance_id).and_then(|entry| {
            entry.listeners.get(&dp_rank).and_then(|record| {
                record
                    .runtime
                    .lock()
                    .cancel_token
                    .as_ref()
                    .map(|t| t.is_cancelled())
            })
        })
    }

    /// Drop the partition's indexer once no registered worker names it (unless the
    /// index retains empty indexers); the check runs under the index lifecycle lock.
    async fn release_indexer_if_unused(&self, key: &RoutingPartitionId) {
        self.index
            .release_if_unused(key, || {
                self.workers.iter().any(|entry| entry.value().key == *key)
            })
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_events::residency::StorageTier;
    use dynamo_kv_hashing::LocalBlockHash;
    use dynamo_kv_indexer::testing::store_event;
    use dynamo_router_protocol::worker::WorkerWithDpRank;
    use std::future::Future;
    use std::sync::atomic::Ordering;
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;
    use tokio::sync::oneshot;

    fn test_registry() -> WorkerRegistry {
        WorkerRegistry::new(1)
    }

    #[rstest::rstest]
    #[case("worker")]
    #[case("all_groups")]
    #[case("last_rank")]
    #[tokio::test]
    async fn registration_progresses_while_removal_is_backpressured(#[case] removal: &str) {
        let registry = test_registry();
        let key = RoutingPartitionId::new("test-model", "default");
        registry
            .register(
                1,
                "tcp://127.0.0.1:15557".into(),
                0,
                "test-model".into(),
                "default".into(),
                1,
                None,
                false,
                false,
            )
            .await
            .unwrap();
        let indexer = registry.index().indexer(&key).unwrap();
        let Indexer::Single { primary, .. } = indexer else {
            unreachable!();
        };
        // Reserve the entire queue so removal must yield without blocking the executor.
        let permits = primary.saturate_mutation_queue_for_test().await;
        let removal = async {
            match removal {
                "worker" => registry.deregister(1, "test-model", "default").await,
                "all_groups" => {
                    registry
                        .deregister_all_routing_groups(1, "test-model")
                        .await
                }
                "last_rank" => {
                    registry
                        .deregister_dp_rank(1, 0, "test-model", "default")
                        .await
                }
                _ => unreachable!(),
            }
        };
        tokio::pin!(removal);
        assert!(matches!(
            removal
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        // A nonblocking probe makes the regression fail instead of deadlocking the test.
        assert!(registry.index().contains(&key));
        assert!(registry.index().shard_is_writable(&key));
        registry
            .register(
                2,
                "tcp://127.0.0.1:15558".into(),
                0,
                "test-model".into(),
                "default".into(),
                1,
                None,
                false,
                false,
            )
            .await
            .unwrap();
        drop(permits);
        tokio::time::timeout(Duration::from_secs(5), removal)
            .await
            .expect("removal should finish after queue capacity is released")
            .unwrap();
        assert!(registry.workers.contains_key(&2));
        assert!(registry.index().contains(&key));
        registry.root_cancel_token.cancel();
    }

    #[rstest::rstest]
    #[case("worker")]
    #[case("all_groups")]
    #[case("last_rank")]
    #[tokio::test]
    async fn empty_indexer_removal_waits_for_registration(#[case] removal: &str) {
        let registry = test_registry();
        let key = RoutingPartitionId::new("test-model", "default");
        registry
            .register(
                1,
                "tcp://127.0.0.1:15557".into(),
                0,
                "test-model".into(),
                "default".into(),
                1,
                None,
                false,
                false,
            )
            .await
            .unwrap();

        // Queue registration ahead of pruning, without timing or thread scheduling assumptions.
        let lifecycle = registry.index().lock_lifecycle().await;
        let registration = registry.register(
            2,
            "tcp://127.0.0.1:15558".into(),
            0,
            "test-model".into(),
            "default".into(),
            1,
            None,
            false,
            false,
        );
        tokio::pin!(registration);
        assert!(matches!(
            registration
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));

        let removal = async {
            match removal {
                "worker" => registry.deregister(1, "test-model", "default").await,
                "all_groups" => {
                    registry
                        .deregister_all_routing_groups(1, "test-model")
                        .await
                }
                "last_rank" => {
                    registry
                        .deregister_dp_rank(1, 0, "test-model", "default")
                        .await
                }
                _ => unreachable!(),
            }
        };
        tokio::pin!(removal);
        assert!(matches!(
            removal
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        assert!(registry.workers.is_empty());
        assert!(!registry.watermarks.contains_key(&(1, 0)));
        assert!(registry.get_indexer(&key).is_some());
        drop(lifecycle);
        registration.await.unwrap();
        removal.await.unwrap();

        // Events written by the new listener must remain visible through the query registry.
        let listener_indexer = registry.workers.get(&2).unwrap().listeners[&0]
            .indexer
            .clone();
        listener_indexer
            .apply_event_routed(store_event(2, 0, 0, &[], &[11], StorageTier::Device))
            .await
            .unwrap();
        listener_indexer.dump_events().await.unwrap();
        let query_indexer = registry
            .get_indexer(&key)
            .expect("registered indexer")
            .indexer
            .clone();
        let scores = query_indexer
            .find_matches(vec![LocalBlockHash(11)])
            .await
            .unwrap();
        assert_eq!(scores.scores.get(&WorkerWithDpRank::new(2, 0)), Some(&1));
        registry.root_cancel_token.cancel();
    }

    #[tokio::test]
    async fn register_records_engine_owned_mm_identity_per_listener() {
        let registry = test_registry();
        for (dp_rank, engine_owned) in [(0, false), (1, true)] {
            registry
                .register(
                    1,
                    format!("tcp://127.0.0.1:{}", 15570 + dp_rank),
                    dp_rank,
                    "test-model".to_string(),
                    "default".to_string(),
                    1,
                    None,
                    engine_owned,
                    false,
                )
                .await
                .unwrap();
        }

        let entry = registry.workers.get(&1).unwrap();
        assert!(!entry.listeners[&0].engine_owned_mm_identity());
        assert!(entry.listeners[&1].engine_owned_mm_identity());
        drop(entry);
        let listed = serde_json::to_value(registry.list()).unwrap();
        assert_eq!(
            listed[0]["listeners"]["1"]["engine_owned_mm_identity"],
            true
        );
        registry.root_cancel_token.cancel();
    }

    #[tokio::test]
    async fn register_records_idempotent_stores_per_listener() {
        let registry = test_registry();
        for (dp_rank, idempotent_stores) in [(0, false), (1, true)] {
            registry
                .register(
                    1,
                    format!("tcp://127.0.0.1:{}", 15580 + dp_rank),
                    dp_rank,
                    "test-model".to_string(),
                    "default".to_string(),
                    1,
                    None,
                    false,
                    idempotent_stores,
                )
                .await
                .unwrap();
        }

        let entry = registry.workers.get(&1).unwrap();
        assert!(!entry.listeners[&0].idempotent_stores());
        assert!(entry.listeners[&1].idempotent_stores());
        assert!(!entry.listeners[&1].engine_owned_mm_identity());
        drop(entry);
        let listed = serde_json::to_value(registry.list()).unwrap();
        assert_eq!(listed[0]["listeners"]["0"]["idempotent_stores"], false);
        assert_eq!(listed[0]["listeners"]["1"]["idempotent_stores"], true);
        registry.root_cancel_token.cancel();
    }

    #[tokio::test]
    async fn deregister_dp_rank_removes_watermark() {
        let registry = test_registry();
        registry.signal_ready();

        registry
            .register(
                1,
                "tcp://127.0.0.1:15558".to_string(),
                0,
                "test-model".to_string(),
                "default".to_string(),
                1,
                None,
                false,
                false,
            )
            .await
            .unwrap();

        registry
            .register(
                1,
                "tcp://127.0.0.1:15559".to_string(),
                1,
                "test-model".to_string(),
                "default".to_string(),
                1,
                None,
                false,
                false,
            )
            .await
            .unwrap();

        assert!(registry.watermarks.contains_key(&(1, 0)));
        assert!(registry.watermarks.contains_key(&(1, 1)));

        registry
            .deregister_dp_rank(1, 1, "test-model", "default")
            .await
            .unwrap();

        assert!(
            registry.watermarks.contains_key(&(1, 0)),
            "watermark for dp_rank 0 should remain"
        );
        assert!(
            !registry.watermarks.contains_key(&(1, 1)),
            "watermark for dp_rank 1 should be removed"
        );
    }

    #[tokio::test]
    async fn listener_cancelled_by_root() {
        let root = CancellationToken::new();
        let registry = WorkerRegistry::new_with_cancel_token(1, root.clone());

        registry
            .register(
                1,
                "tcp://127.0.0.1:15560".to_string(),
                0,
                "test-model".to_string(),
                "default".to_string(),
                1,
                None,
                false,
                false,
            )
            .await
            .unwrap();
        assert_eq!(registry.listener_cancelled(1, 0), Some(false));

        root.cancel();
        assert_eq!(registry.listener_cancelled(1, 0), Some(true));
    }

    #[tokio::test]
    async fn listener_inherits_cancelled_root() {
        let root = CancellationToken::new();
        root.cancel();
        let registry = WorkerRegistry::new_with_cancel_token(1, root);

        registry
            .register(
                1,
                "tcp://127.0.0.1:15561".to_string(),
                0,
                "test-model".to_string(),
                "default".to_string(),
                1,
                None,
                false,
                false,
            )
            .await
            .unwrap();

        assert_eq!(registry.listener_cancelled(1, 0), Some(true));
    }

    #[tokio::test]
    async fn re_register_gets_fresh_watermark() {
        let registry = test_registry();
        registry.signal_ready();

        registry
            .register(
                1,
                "tcp://127.0.0.1:15560".to_string(),
                0,
                "test-model".to_string(),
                "default".to_string(),
                1,
                None,
                false,
                false,
            )
            .await
            .unwrap();

        // Simulate that the listener advanced the watermark.
        registry
            .watermarks
            .get(&(1, 0))
            .unwrap()
            .store(42, Ordering::Release);

        registry
            .deregister(1, "test-model", "default")
            .await
            .unwrap();

        assert!(
            registry
                .get_indexer(&RoutingPartitionId::new("test-model", "default"))
                .is_none()
        );

        registry
            .register(
                1,
                "tcp://127.0.0.1:15561".to_string(),
                0,
                "test-model".to_string(),
                "default".to_string(),
                1,
                None,
                false,
                false,
            )
            .await
            .unwrap();

        let wm = registry
            .watermarks
            .get(&(1, 0))
            .expect("watermark should exist after re-register");
        assert_eq!(
            wm.load(Ordering::Acquire),
            u64::MAX,
            "re-registered watermark should be fresh (u64::MAX)"
        );
    }

    // ── list_filtered tests ───────────────────────────────────────────────────

    #[tokio::test]
    async fn list_filtered_returns_metadata_and_applies_filters() {
        let registry = test_registry();
        registry.signal_ready();

        registry
            .register(
                10,
                "tcp://127.0.0.1:15570".to_string(),
                0,
                "llama3".to_string(),
                "acme".to_string(),
                4,
                None,
                false,
                false,
            )
            .await
            .unwrap();

        registry
            .register(
                11,
                "tcp://127.0.0.1:15571".to_string(),
                0,
                "mistral".to_string(),
                "other-group".to_string(),
                8,
                None,
                false,
                false,
            )
            .await
            .unwrap();

        let workers = registry.list();
        assert_eq!(workers.len(), 2);

        let llama = workers.iter().find(|w| w.model_name == "llama3").unwrap();
        assert_eq!(llama.block_size, 4);
        assert_eq!(llama.routing_group, "acme");

        let mistral = workers.iter().find(|w| w.model_name == "mistral").unwrap();
        assert_eq!(mistral.block_size, 8);
        assert_eq!(mistral.routing_group, "other-group");

        let filtered = registry.list_filtered(Some("llama3"), None);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].model_name, "llama3");

        let acme = registry.list_filtered(None, Some("acme"));
        assert_eq!(acme.len(), 1);
        assert_eq!(acme[0].routing_group, "acme");

        let other = registry.list_filtered(None, Some("other-group"));
        assert_eq!(other.len(), 1);
        assert_eq!(other[0].routing_group, "other-group");

        assert!(
            registry
                .list_filtered(Some("llama3"), Some("other-group"))
                .is_empty()
        );
        assert!(registry.list_filtered(Some("nonexistent"), None).is_empty());
    }

    #[test]
    fn list_filtered_skips_empty_listener_workers() {
        // Directly inject a WorkerEntry with an empty listener map into the
        // registry's workers DashMap.  This is the exact state that exists in
        // the TOCTOU window inside deregister_dp_rank: the last listener has
        // been removed from the HashMap (inside the shard lock at line ~525)
        // but workers.remove() has not yet been called (line ~538, outside the
        // lock).  Calling list() on this state must not return the entry.
        //
        // The previous approach (register + deregister both dp_ranks) was
        // vacuous: after the second deregister_dp_rank returns, the worker is
        // already gone, so workers.iter() is empty and the assertion held
        // trivially without exercising the filter.
        let registry = test_registry();

        let key = RoutingPartitionId::new("llama3", "acme");

        // Inject the empty-listener WorkerEntry directly.
        registry.workers.insert(
            99u64,
            WorkerEntry {
                key,
                listeners: HashMap::new(),
            },
        );

        let workers = registry.list();
        assert!(
            workers.is_empty(),
            "worker with empty listener map must be omitted from list(); \
             got {} entries",
            workers.len()
        );
    }

    // ── deregistration vs an in-flight listener batch ────────────────
    //
    // Timeouts below are watchdogs against a deadlock; ordering comes from the batch gate.

    const WATCHDOG: Duration = Duration::from_secs(10);

    /// A PUB socket standing in for an engine's KV event publisher.
    struct TestPublisher {
        socket: zmq::Socket,
        endpoint: String,
    }

    fn test_zmq_context() -> &'static zmq::Context {
        static CONTEXT: std::sync::LazyLock<zmq::Context> =
            std::sync::LazyLock::new(zmq::Context::new);
        &CONTEXT
    }

    impl TestPublisher {
        fn bind() -> Self {
            let socket = test_zmq_context().socket(zmq::PUB).unwrap();
            socket.set_linger(0).unwrap();
            socket.bind("tcp://127.0.0.1:*").unwrap();
            let endpoint = socket.get_last_endpoint().unwrap().unwrap();
            Self { socket, endpoint }
        }

        fn publish(&self, seq: u64, payload: Vec<u8>) {
            self.socket
                .send_multipart(
                    [b"kv-events".to_vec(), seq.to_be_bytes().to_vec(), payload],
                    0,
                )
                .unwrap();
        }
    }

    /// One `BlockStored` per hash, each a root block with its own tokens, so a block is
    /// accepted whether or not the ones before it in the batch were indexed.
    fn root_batch(dp_rank: i32, hashes: &[u64]) -> Vec<u8> {
        let events: Vec<_> = hashes
            .iter()
            .map(|&hash| {
                (
                    "BlockStored",
                    vec![hash],
                    Option::<u64>::None,
                    vec![hash as u32; 4],
                    4_usize,
                    Option::<u64>::None,
                    "GPU",
                )
            })
            .collect();
        rmp_serde::to_vec(&(0.0_f64, events, Some(dp_rank))).unwrap()
    }

    /// The test's side of a [`BatchGate`].
    struct GateHandle {
        reached: oneshot::Receiver<()>,
        resume: oneshot::Sender<()>,
        /// Resolves with `Err` once the listener loop holding the gate is dropped.
        exited: oneshot::Receiver<()>,
        exit_seen: bool,
    }

    impl GateHandle {
        /// Whether the listener loop is already gone, without waiting for it.
        fn listener_exited(&mut self) -> bool {
            self.exit_seen = self.exit_seen
                || matches!(
                    self.exited.try_recv(),
                    Err(oneshot::error::TryRecvError::Closed)
                );
            self.exit_seen
        }

        async fn release_and_wait_for_exit(&mut self) {
            // The listener may already be gone, which closes the receiver.
            let (spare, _) = oneshot::channel();
            let _ = std::mem::replace(&mut self.resume, spare).send(());
            if !self.exit_seen {
                tokio::time::timeout(WATCHDOG, &mut self.exited)
                    .await
                    .expect("listener did not exit")
                    .unwrap_err();
                self.exit_seen = true;
            }
        }
    }

    /// Parks the listener of `(instance_id, dp_rank)` after the first event of batch `seq`.
    /// Install it before `signal_ready`: the listener takes it when its receive loop starts.
    fn install_gate(
        registry: &WorkerRegistry,
        instance_id: WorkerId,
        dp_rank: u32,
        seq: u64,
    ) -> GateHandle {
        let (reached_tx, reached) = oneshot::channel();
        let (resume, resume_rx) = oneshot::channel();
        let (alive, exited) = oneshot::channel();
        let record = registry.record_for_test(instance_id, dp_rank).unwrap();
        *record.batch_gate.lock() = Some(BatchGate {
            seq,
            reached: Some(reached_tx),
            resume: Some(resume_rx),
            _alive: alive,
        });
        GateHandle {
            reached,
            resume,
            exited,
            exit_seen: false,
        }
    }

    async fn register_for_test(
        registry: &WorkerRegistry,
        instance_id: WorkerId,
        dp_rank: u32,
        endpoint: &str,
        replay_endpoint: Option<String>,
    ) {
        registry
            .register(
                instance_id,
                endpoint.to_string(),
                dp_rank,
                "test-model".into(),
                "default".into(),
                4,
                replay_endpoint,
                false,
                false,
            )
            .await
            .unwrap();
    }

    /// Events of `worker_id`'s `dp_rank` the indexer holds, after everything queued so far.
    async fn indexed_events(indexer: &Indexer, worker_id: WorkerId, dp_rank: u32) -> usize {
        indexer
            .dump_events()
            .await
            .unwrap()
            .iter()
            .filter(|e| e.worker_id == worker_id && e.event.dp_rank == dp_rank)
            .count()
    }

    /// Publishes batch `seq = 0` until the listener, whose SUB socket may still be
    /// connecting, has indexed it.
    async fn publish_probe(publisher: &TestPublisher, indexer: &Indexer, worker_id: WorkerId) {
        tokio::time::timeout(WATCHDOG, async {
            loop {
                publisher.publish(0, root_batch(0, &[1]));
                tokio::time::sleep(Duration::from_millis(10)).await;
                if indexed_events(indexer, worker_id, 0).await > 0 {
                    return;
                }
            }
        })
        .await
        .expect("listener never indexed the probe batch");
    }

    /// A registry with worker 1 (rank 0) listening to `rank0` and worker 2 (so the partition
    /// survives). With `sibling_rank`, worker 1 also has rank 1, which holds a seeded block.
    /// The rank 0 listener is parked after the first event of batch 1, with the probe indexed.
    async fn registry_with_parked_listener(
        threads: usize,
        sibling_rank: bool,
        rank0: &TestPublisher,
        replay_endpoint: Option<String>,
    ) -> (WorkerRegistry, Indexer, GateHandle, Vec<TestPublisher>) {
        let registry = WorkerRegistry::new(threads);
        let key = RoutingPartitionId::new("test-model", "default");
        let mut others = vec![TestPublisher::bind()];
        register_for_test(&registry, 1, 0, &rank0.endpoint, replay_endpoint).await;
        register_for_test(&registry, 2, 0, &others[0].endpoint, None).await;
        if sibling_rank {
            others.push(TestPublisher::bind());
            register_for_test(&registry, 1, 1, &others[1].endpoint, None).await;
        }
        let indexer = registry.index().indexer(&key).unwrap();
        indexer
            .apply_event_routed(store_event(2, 0, 0, &[], &[21], StorageTier::Device))
            .await
            .unwrap();
        if sibling_rank {
            indexer
                .apply_event_routed(store_event(1, 1, 0, &[], &[11], StorageTier::Device))
                .await
                .unwrap();
        }

        let mut gate = install_gate(&registry, 1, 0, 1);
        registry.signal_ready();
        publish_probe(rank0, &indexer, 1).await;
        rank0.publish(1, root_batch(0, &[100, 101, 102]));
        tokio::time::timeout(WATCHDOG, &mut gate.reached)
            .await
            .expect("listener never reached the batch gate")
            .unwrap();
        (registry, indexer, gate, others)
    }

    /// After the listener is gone, the rank has no block and its neighbours are intact.
    async fn assert_rank_gone(indexer: &Indexer, sibling_rank: bool) {
        assert_eq!(
            indexed_events(indexer, 1, 0).await,
            0,
            "deregistered rank came back from its in-flight batch"
        );
        assert!(
            indexed_events(indexer, 2, 0).await > 0,
            "unrelated worker was removed"
        );
        if sibling_rank {
            assert!(
                indexed_events(indexer, 1, 1).await > 0,
                "sibling rank was removed"
            );
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Deregistration {
        Worker,
        LastRank,
        OtherRank,
    }

    #[rstest::rstest]
    #[case::single_worker(1, Deregistration::Worker)]
    #[case::single_last_rank(1, Deregistration::LastRank)]
    #[case::single_other_rank(1, Deregistration::OtherRank)]
    #[case::concurrent_worker(2, Deregistration::Worker)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deregister_stops_listener_mid_batch(
        #[case] threads: usize,
        #[case] removal: Deregistration,
    ) {
        let sibling_rank = matches!(removal, Deregistration::OtherRank);
        let rank0 = TestPublisher::bind();
        let (registry, indexer, mut gate, _others) =
            registry_with_parked_listener(threads, sibling_rank, &rank0, None).await;

        // The listener is parked mid-batch. A fix that cannot stop it makes this time out.
        let deregistration = tokio::time::timeout(WATCHDOG, async {
            match removal {
                Deregistration::Worker => registry.deregister(1, "test-model", "default").await,
                Deregistration::LastRank | Deregistration::OtherRank => {
                    registry
                        .deregister_dp_rank(1, 0, "test-model", "default")
                        .await
                }
            }
        })
        .await
        .expect("deregistration did not finish");
        deregistration.unwrap();
        let stopped_before_return = gate.listener_exited();

        gate.release_and_wait_for_exit().await;

        assert_rank_gone(&indexer, sibling_rank).await;
        assert!(
            stopped_before_return,
            "deregistration returned while the listener was still running"
        );
        registry.root_cancel_token.cancel();
    }

    /// A pause lets the batch in flight finish, so the watermark covers what was indexed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pause_finishes_in_flight_batch() {
        let rank0 = TestPublisher::bind();
        let (registry, indexer, mut gate, _others) =
            registry_with_parked_listener(1, false, &rank0, None).await;

        registry.pause_listener(1, 0).unwrap();
        gate.release_and_wait_for_exit().await;

        assert_eq!(
            indexed_events(&indexer, 1, 0).await,
            4,
            "pause dropped the rest of the batch (expected the probe and blocks 100-102)"
        );
        let watermark = registry.watermarks.get(&(1, 0)).unwrap().clone();
        assert_eq!(watermark.load(Ordering::Acquire), 1);
        registry.root_cancel_token.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deregister_joins_paused_listener() {
        let rank0 = TestPublisher::bind();
        let (registry, indexer, mut gate, _others) =
            registry_with_parked_listener(1, false, &rank0, None).await;

        registry.pause_listener(1, 0).unwrap();
        tokio::time::timeout(WATCHDOG, registry.deregister(1, "test-model", "default"))
            .await
            .expect("deregistration did not finish")
            .unwrap();
        let stopped_before_return = gate.listener_exited();

        gate.release_and_wait_for_exit().await;

        assert_rank_gone(&indexer, false).await;
        assert!(stopped_before_return);
        registry.root_cancel_token.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pause_resume_deregister_while_batch_in_flight() {
        let rank0 = TestPublisher::bind();
        let (registry, indexer, mut gate, _others) =
            registry_with_parked_listener(1, false, &rank0, None).await;

        registry.pause_listener(1, 0).unwrap();
        registry.resume_listener(1, 0).await.unwrap();
        tokio::time::timeout(WATCHDOG, registry.deregister(1, "test-model", "default"))
            .await
            .expect("deregistration did not finish")
            .unwrap();
        let stopped_before_return = gate.listener_exited();

        gate.release_and_wait_for_exit().await;

        assert_rank_gone(&indexer, false).await;
        assert!(stopped_before_return);
        registry.root_cancel_token.cancel();
    }

    /// A gap makes the listener request a replay from an engine that never answers;
    /// deregistering must still stop it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deregister_returns_when_replay_hangs() {
        let replay = test_zmq_context().socket(zmq::ROUTER).unwrap();
        replay.set_linger(0).unwrap();
        replay.bind("tcp://127.0.0.1:*").unwrap();
        let replay_endpoint = replay.get_last_endpoint().unwrap().unwrap();
        let rank0 = TestPublisher::bind();
        let other = TestPublisher::bind();
        let registry = WorkerRegistry::new(1);
        register_for_test(&registry, 1, 0, &rank0.endpoint, Some(replay_endpoint)).await;
        register_for_test(&registry, 2, 0, &other.endpoint, None).await;
        // Never reached: the gate is only here to tell when the listener loop is dropped.
        let mut gate = install_gate(&registry, 1, 0, u64::MAX);
        registry.signal_ready();

        // The first batch seen is seq 5, so the listener requests a replay of 0..5. Publish
        // until the request arrives: the SUB socket may still be connecting.
        tokio::time::timeout(WATCHDOG, async {
            loop {
                rank0.publish(5, root_batch(0, &[100]));
                tokio::time::sleep(Duration::from_millis(10)).await;
                if replay.recv_multipart(zmq::DONTWAIT).is_ok() {
                    return;
                }
            }
        })
        .await
        .expect("listener never requested a replay");

        tokio::time::timeout(WATCHDOG, registry.deregister(1, "test-model", "default"))
            .await
            .expect("deregistration hung behind the replay request")
            .unwrap();
        assert!(
            gate.listener_exited(),
            "deregistration returned while the listener was still waiting on the replay"
        );
        registry.root_cancel_token.cancel();
    }

    /// The removal still lands when the caller of `deregister` goes away part-way.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_deregister_still_removes_worker() {
        let rank0 = TestPublisher::bind();
        let (registry, indexer, mut gate, _others) =
            registry_with_parked_listener(1, false, &rank0, None).await;

        {
            let deregistration = registry.deregister(1, "test-model", "default");
            tokio::pin!(deregistration);
            let _ = deregistration
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()));
        }
        gate.release_and_wait_for_exit().await;

        tokio::time::timeout(WATCHDOG, async {
            while indexed_events(&indexer, 1, 0).await > 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("worker stayed indexed after its deregistration was dropped");
        assert!(!registry.has_worker(1));
        registry.root_cancel_token.cancel();
    }

    fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        future.poll(&mut Context::from_waker(Waker::noop()))
    }

    /// A re-registration of the worker waits until its removal is queued, so the removal
    /// cannot delete what the new listener indexes. Other workers register meanwhile.
    #[tokio::test]
    async fn reregistration_waits_for_removal() {
        let registry = test_registry();
        let key = RoutingPartitionId::new("test-model", "default");
        register_for_test(&registry, 1, 0, "tcp://127.0.0.1:15557", None).await;
        let Indexer::Single { primary, .. } = registry.index().indexer(&key).unwrap() else {
            unreachable!();
        };
        let permits = primary.saturate_mutation_queue_for_test().await;

        let removal = registry.deregister(1, "test-model", "default");
        tokio::pin!(removal);
        assert!(poll_once(removal.as_mut()).is_pending());
        let reregistration = register_for_test(&registry, 1, 0, "tcp://127.0.0.1:15558", None);
        tokio::pin!(reregistration);
        assert!(
            poll_once(reregistration.as_mut()).is_pending(),
            "the worker registered again before its removal was queued"
        );
        tokio::time::timeout(
            WATCHDOG,
            register_for_test(&registry, 2, 0, "tcp://127.0.0.1:15559", None),
        )
        .await
        .expect("another worker's registration waited for the removal");

        drop(permits);
        tokio::time::timeout(WATCHDOG, removal)
            .await
            .expect("removal did not finish")
            .unwrap();
        tokio::time::timeout(WATCHDOG, reregistration)
            .await
            .expect("re-registration did not finish");
        assert!(registry.has_listener(1, 0));
        registry.root_cancel_token.cancel();
    }

    /// Removing the whole worker waits for a rank removal in flight, which no longer has
    /// that rank in the worker's entry to join.
    #[tokio::test]
    async fn worker_removal_waits_for_rank_removal() {
        let registry = test_registry();
        let key = RoutingPartitionId::new("test-model", "default");
        register_for_test(&registry, 1, 0, "tcp://127.0.0.1:15557", None).await;
        register_for_test(&registry, 1, 1, "tcp://127.0.0.1:15558", None).await;
        let Indexer::Single { primary, .. } = registry.index().indexer(&key).unwrap() else {
            unreachable!();
        };
        let permits = primary.saturate_mutation_queue_for_test().await;

        let rank_removal = registry.deregister_dp_rank(1, 0, "test-model", "default");
        tokio::pin!(rank_removal);
        assert!(poll_once(rank_removal.as_mut()).is_pending());
        let worker_removal = registry.deregister(1, "test-model", "default");
        tokio::pin!(worker_removal);
        assert!(poll_once(worker_removal.as_mut()).is_pending());
        assert!(
            registry.has_listener(1, 1),
            "the worker was removed while a rank removal was in flight"
        );

        drop(permits);
        tokio::time::timeout(WATCHDOG, rank_removal)
            .await
            .expect("rank removal did not finish")
            .unwrap();
        tokio::time::timeout(WATCHDOG, worker_removal)
            .await
            .expect("worker removal did not finish")
            .unwrap();
        assert!(!registry.has_worker(1));
        registry.root_cancel_token.cancel();
    }

    /// A waiter dropped after the holder released the lock, before it ran, leaves no entry.
    #[tokio::test]
    async fn dropped_lock_waiter_leaves_no_entry() {
        let registry = test_registry();
        let holder = registry.lock_worker(1).await;
        {
            let waiter = registry.lock_worker(1);
            tokio::pin!(waiter);
            assert!(poll_once(waiter.as_mut()).is_pending());
            drop(holder);
        }
        assert!(
            registry.worker_locks.is_empty(),
            "a dropped lock waiter left its worker's entry behind"
        );
        drop(registry.lock_worker(1).await);
        assert!(registry.worker_locks.is_empty());
    }

    #[tokio::test]
    async fn retired_record_cannot_resume() {
        let registry = test_registry();
        register_for_test(&registry, 1, 0, "tcp://127.0.0.1:15590", None).await;
        let stale = registry.record_for_test(1, 0).unwrap();
        registry.pause_listener(1, 0).unwrap();

        registry
            .deregister(1, "test-model", "default")
            .await
            .unwrap();

        // A resume that cloned the record before the deregistration must not restart it.
        let resumed = registry.resume_record_for_test(&stale, 1, 0);
        assert!(
            matches!(resumed, Err(ListenerControlError::ListenerNotFound { .. })),
            "a deregistered record started a listener: {resumed:?}"
        );
        registry.root_cancel_token.cancel();
    }
}
