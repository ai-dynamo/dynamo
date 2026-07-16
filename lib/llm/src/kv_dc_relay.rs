// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-scoped, single-DC KV DC Relay.
//!
//! Dynamo discovery and worker-local recovery feed one serialized actor. The
//! actor's exact member ownership is authoritative; its CKF publications expose
//! one physical layout for a future Relay-to-global-router adapter.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use dynamo_kv_router::indexer::cuckoo::{
    CkfConfig, DcCkfDelta, DcCkfSnapshot, DcCkfState, DcCkfStats,
};
use dynamo_kv_router::protocols::{
    DpRank, ExternalSequenceBlockHash, KvCacheEventData, RouterEvent, StorageTier, WorkerId,
    WorkerWithDpRank,
};
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use parking_lot::Mutex;
use rustc_hash::FxHashSet;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::discovery::runtime_config_watch;
use crate::kv_router::indexer::{RecoveryTarget, WorkerQueryClient, start_kv_dc_relay_subscriber};

pub const DEFAULT_EXPECTED_UNIQUE_BLOCKS: usize = 1_048_576;
const DEFAULT_MAILBOX_CAPACITY: usize = 1_024;
const DEFAULT_SUBSCRIBER_CAPACITY: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum KvDcRelayError {
    #[error("KV DC Relay is shutting down")]
    ShuttingDown,
    #[error("KV DC Relay actor stopped before completing an accepted command")]
    ActorStopped,
    #[error("invalid tree dump for worker {worker_id} rank {dp_rank}: {message}")]
    InvalidTreeDump {
        worker_id: WorkerId,
        dp_rank: DpRank,
        message: String,
    },
    #[error(transparent)]
    Build(#[from] dynamo_kv_router::indexer::cuckoo::CkfBuildError),
    #[error(transparent)]
    Event(#[from] dynamo_kv_router::protocols::KvCacheEventError),
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayMemberStats {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub blocks: usize,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayStats {
    pub identity: KvDcRelayIdentityStats,
    pub aggregation: KvDcRelayAggregationStats,
    pub publication: KvDcRelayPublicationStats,
    pub recovery: KvDcRelayRecoveryStats,
    pub memory: KvDcRelayMemoryStats,
    pub actor: KvDcRelayActorStats,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayIdentityStats {
    pub dc_id: String,
    pub model_name: String,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayAggregationStats {
    pub members: Vec<KvDcRelayMemberStats>,
    pub member_count: usize,
    pub contribution_count: usize,
    pub unique_block_count: usize,
    pub unknown_removals: u64,
    pub capacity_failures: u64,
    pub occupied_bucket_count: usize,
    pub occupied_slot_count: usize,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayPublicationStats {
    pub sequence: u64,
    pub pending_events: usize,
    pub publication_count: u64,
    pub unchanged_publication_count: u64,
    pub physical_touches: u64,
    pub distinct_touched_buckets: u64,
    pub emitted_images: u64,
    pub net_reverted_buckets: u64,
    pub reset_count: u64,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayRecoveryStats {
    pub degraded_resets: u64,
    pub rebuild_count: u64,
    pub rebuild_ns: u64,
    pub rebuild_max_ns: u64,
    pub worker_count: usize,
    pub rank_count: usize,
    pub recovering_rank_count: usize,
    pub pending_live_event_count: usize,
    pub discovered_endpoint_count: usize,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayMemoryStats {
    pub filter_bytes: usize,
    pub dirty_tracking_bytes: usize,
    pub member_set_capacity: usize,
    pub refcount_capacity: usize,
    pub insertion_scratch_capacity: usize,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayActorStats {
    pub mailbox_depth: usize,
    pub mailbox_capacity: usize,
    pub mailbox_wait_ns: u64,
    pub mailbox_max_wait_ns: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct KvDcRelayHealth {
    pub healthy: bool,
    pub shutting_down: bool,
    pub active_command: Option<String>,
    pub active_command_age_ms: Option<u64>,
    pub mailbox_depth: usize,
    pub worker_count: usize,
    pub rank_count: usize,
    pub recovering_rank_count: usize,
    pub pending_live_event_count: usize,
    pub discovered_endpoint_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct KvDcRelayDiagnosticSnapshot {
    pub dc_id: String,
    pub model_name: String,
    pub sequence: u64,
    pub member_count: usize,
    pub contribution_count: usize,
    pub unique_block_count: usize,
    pub format_version: u16,
    pub seed: u64,
    pub bucket_count: usize,
    pub fingerprint_bits: u8,
    pub slots_per_bucket: u8,
    pub buckets: Vec<u64>,
}

#[derive(Debug)]
#[allow(dead_code)] // Reserved for the future Relay-to-global-router adapter.
pub(crate) struct DcCkfSubscription {
    pub(crate) snapshot: DcCkfSnapshot,
    pub(crate) deltas: mpsc::Receiver<DcCkfDelta>,
}

#[derive(Debug, Default)]
struct ActorCounters {
    mailbox_wait_ns: AtomicU64,
    mailbox_max_wait_ns: AtomicU64,
    degraded_resets: AtomicU64,
    publications: AtomicU64,
    unchanged_publications: AtomicU64,
    rebuild_count: AtomicU64,
    rebuild_ns: AtomicU64,
    rebuild_max_ns: AtomicU64,
}

#[derive(Debug, Default)]
struct ActorActivity {
    active_command: Option<&'static str>,
    active_since: Option<Instant>,
    shutting_down: bool,
}

#[derive(Debug, Clone)]
pub struct KvDcRelayHandle {
    sender: mpsc::Sender<ActorCommand>,
    counters: Arc<ActorCounters>,
    activity: Arc<Mutex<ActorActivity>>,
}

impl KvDcRelayHandle {
    pub fn spawn(config: CkfConfig) -> Result<Self, KvDcRelayError> {
        Self::spawn_with_capacity(config, DEFAULT_MAILBOX_CAPACITY)
    }

    fn spawn_with_capacity(config: CkfConfig, capacity: usize) -> Result<Self, KvDcRelayError> {
        let state = DcCkfState::new(config)?;
        let (sender, receiver) = mpsc::channel(capacity);
        let counters = Arc::new(ActorCounters::default());
        let activity = Arc::new(Mutex::new(ActorActivity::default()));
        tokio::spawn(run_actor(
            state,
            receiver,
            counters.clone(),
            activity.clone(),
        ));
        Ok(Self {
            sender,
            counters,
            activity,
        })
    }

    async fn submit<T>(
        &self,
        make_command: impl FnOnce(oneshot::Sender<Result<T, KvDcRelayError>>) -> ActorCommand,
    ) -> Result<T, KvDcRelayError> {
        let (response_tx, response_rx) = oneshot::channel();
        let wait_started = Instant::now();
        self.sender
            .send(make_command(response_tx))
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        let waited = wait_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.counters
            .mailbox_wait_ns
            .fetch_add(waited, Ordering::Relaxed);
        self.counters
            .mailbox_max_wait_ns
            .fetch_max(waited, Ordering::Relaxed);
        response_rx
            .await
            .map_err(|_| KvDcRelayError::ActorStopped)?
    }

    pub async fn apply_event(&self, event: RouterEvent) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::Apply { event, response })
            .await
    }

    pub async fn replace_rank(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::ReplaceRank {
            worker_id,
            dp_rank,
            events,
            response,
        })
        .await
    }

    pub async fn remove_rank(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        degraded: bool,
    ) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::RemoveRank {
            worker_id,
            dp_rank,
            degraded,
            response,
        })
        .await
    }

    pub async fn remove_worker(&self, worker_id: WorkerId) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::RemoveWorker {
            worker_id,
            response,
        })
        .await
    }

    pub async fn flush(&self) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::Flush { response })
            .await
    }

    pub async fn snapshot(&self) -> Result<DcCkfSnapshot, KvDcRelayError> {
        self.snapshot_with_stats()
            .await
            .map(|snapshot| snapshot.snapshot)
    }

    async fn snapshot_with_stats(&self) -> Result<ActorSnapshot, KvDcRelayError> {
        self.submit(|response| ActorCommand::Snapshot { response })
            .await
    }

    async fn state_stats(
        &self,
    ) -> Result<(DcCkfStats, Vec<(WorkerWithDpRank, usize)>), KvDcRelayError> {
        self.submit(|response| ActorCommand::Stats { response })
            .await
    }

    #[allow(dead_code)] // Reserved for the future Relay-to-global-router adapter.
    pub(crate) async fn subscribe(&self) -> Result<DcCkfSubscription, KvDcRelayError> {
        self.submit(|response| ActorCommand::Subscribe { response })
            .await
    }

    pub async fn shutdown(&self) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::Shutdown { response })
            .await
    }

    fn mailbox_depth(&self) -> usize {
        self.sender
            .max_capacity()
            .saturating_sub(self.sender.capacity())
    }
}

#[derive(Clone)]
struct KvDcRelayRecoveryTarget {
    handle: KvDcRelayHandle,
}

#[async_trait]
impl RecoveryTarget for KvDcRelayRecoveryTarget {
    async fn apply_event(&self, event: RouterEvent) -> anyhow::Result<()> {
        self.handle.apply_event(event).await.map_err(Into::into)
    }

    async fn replace_rank(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> anyhow::Result<()> {
        self.handle
            .replace_rank(worker_id, dp_rank, events)
            .await
            .map_err(Into::into)
    }

    async fn remove_rank(&self, worker_id: WorkerId, dp_rank: DpRank) -> anyhow::Result<()> {
        self.handle
            .remove_rank(worker_id, dp_rank, false)
            .await
            .map_err(Into::into)
    }

    async fn degraded_reset_rank(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) -> anyhow::Result<()> {
        self.handle
            .remove_rank(worker_id, dp_rank, true)
            .await
            .map_err(Into::into)
    }

    async fn remove_worker(&self, worker_id: WorkerId) -> anyhow::Result<()> {
        self.handle
            .remove_worker(worker_id)
            .await
            .map_err(Into::into)
    }
}

pub struct ModelKvDcRelay {
    dc_id: String,
    model_name: String,
    handle: KvDcRelayHandle,
    recovery: Arc<WorkerQueryClient>,
    intake_cancel: CancellationToken,
}

impl ModelKvDcRelay {
    pub async fn start(
        endpoint: Endpoint,
        model_name: String,
        dc_id: String,
    ) -> anyhow::Result<Self> {
        let mut config = CkfConfig::new(DEFAULT_EXPECTED_UNIQUE_BLOCKS);
        config.publish_every_n_events = 1;
        let handle = KvDcRelayHandle::spawn(config)?;
        let target: Arc<dyn RecoveryTarget> = Arc::new(KvDcRelayRecoveryTarget {
            handle: handle.clone(),
        });
        let workers_with_configs = runtime_config_watch(&endpoint).await?;
        let component = endpoint.component().clone();
        let intake_cancel = component.drt().child_token();
        let recovery = match start_kv_dc_relay_subscriber(
            component,
            target,
            workers_with_configs,
            model_name.clone(),
            intake_cancel.child_token(),
        )
        .await
        {
            Ok(recovery) => recovery,
            Err(error) => {
                intake_cancel.cancel();
                let _ = handle.shutdown().await;
                return Err(error);
            }
        };
        Ok(Self {
            dc_id,
            model_name,
            handle,
            recovery,
            intake_cancel,
        })
    }

    pub async fn stats(&self) -> Result<KvDcRelayStats, KvDcRelayError> {
        let (stats, members) = self.handle.state_stats().await?;
        let aggregation = stats.aggregation();
        let publication = stats.publication();
        let memory = stats.memory();
        let recovery = self.recovery.health_snapshot().await;
        Ok(KvDcRelayStats {
            identity: KvDcRelayIdentityStats {
                dc_id: self.dc_id.clone(),
                model_name: self.model_name.clone(),
            },
            aggregation: KvDcRelayAggregationStats {
                members: members
                    .into_iter()
                    .map(|(worker, blocks)| KvDcRelayMemberStats {
                        worker_id: worker.worker_id,
                        dp_rank: worker.dp_rank,
                        blocks,
                    })
                    .collect(),
                member_count: aggregation.member_count(),
                contribution_count: aggregation.contribution_count(),
                unique_block_count: aggregation.unique_block_count(),
                unknown_removals: aggregation.unknown_removals(),
                capacity_failures: aggregation.capacity_failures(),
                occupied_bucket_count: aggregation.occupied_bucket_count(),
                occupied_slot_count: aggregation.occupied_slot_count(),
            },
            publication: KvDcRelayPublicationStats {
                sequence: publication.sequence(),
                pending_events: publication.pending_events(),
                publication_count: self.handle.counters.publications.load(Ordering::Relaxed),
                unchanged_publication_count: self
                    .handle
                    .counters
                    .unchanged_publications
                    .load(Ordering::Relaxed),
                physical_touches: publication.physical_touches(),
                distinct_touched_buckets: publication.distinct_touched_buckets(),
                emitted_images: publication.emitted_images(),
                net_reverted_buckets: publication.net_reverted_buckets(),
                reset_count: publication.reset_count(),
            },
            recovery: KvDcRelayRecoveryStats {
                degraded_resets: self.handle.counters.degraded_resets.load(Ordering::Relaxed),
                rebuild_count: self.handle.counters.rebuild_count.load(Ordering::Relaxed),
                rebuild_ns: self.handle.counters.rebuild_ns.load(Ordering::Relaxed),
                rebuild_max_ns: self.handle.counters.rebuild_max_ns.load(Ordering::Relaxed),
                worker_count: recovery.worker_count,
                rank_count: recovery.rank_count,
                recovering_rank_count: recovery.recovering_rank_count,
                pending_live_event_count: recovery.pending_live_event_count,
                discovered_endpoint_count: recovery.discovered_endpoint_count,
            },
            memory: KvDcRelayMemoryStats {
                filter_bytes: memory.filter_bytes(),
                dirty_tracking_bytes: memory.dirty_tracking_bytes(),
                member_set_capacity: memory.member_set_capacity(),
                refcount_capacity: memory.refcount_capacity(),
                insertion_scratch_capacity: memory.insertion_scratch_capacity(),
            },
            actor: KvDcRelayActorStats {
                mailbox_depth: self.handle.mailbox_depth(),
                mailbox_capacity: self.handle.sender.max_capacity(),
                mailbox_wait_ns: self.handle.counters.mailbox_wait_ns.load(Ordering::Relaxed),
                mailbox_max_wait_ns: self
                    .handle
                    .counters
                    .mailbox_max_wait_ns
                    .load(Ordering::Relaxed),
            },
        })
    }

    pub async fn snapshot(&self) -> Result<DcCkfSnapshot, KvDcRelayError> {
        self.handle.snapshot().await
    }

    pub async fn diagnostic_snapshot(&self) -> Result<KvDcRelayDiagnosticSnapshot, KvDcRelayError> {
        let actor_snapshot = self.handle.snapshot_with_stats().await?;
        let format = actor_snapshot.snapshot.format();
        let aggregation = actor_snapshot.stats.aggregation();
        Ok(KvDcRelayDiagnosticSnapshot {
            dc_id: self.dc_id.clone(),
            model_name: self.model_name.clone(),
            sequence: actor_snapshot.snapshot.sequence(),
            member_count: aggregation.member_count(),
            contribution_count: aggregation.contribution_count(),
            unique_block_count: aggregation.unique_block_count(),
            format_version: format.format_version(),
            seed: format.seed(),
            bucket_count: format.bucket_count(),
            fingerprint_bits: format.fingerprint_bits(),
            slots_per_bucket: format.slots_per_bucket(),
            buckets: actor_snapshot.snapshot.buckets().to_vec(),
        })
    }

    pub async fn health(&self) -> KvDcRelayHealth {
        let recovery = self.recovery.health_snapshot().await;
        let activity = self.handle.activity.lock();
        KvDcRelayHealth {
            healthy: !activity.shutting_down,
            shutting_down: activity.shutting_down,
            active_command: activity.active_command.map(str::to_string),
            active_command_age_ms: activity
                .active_since
                .map(|started| started.elapsed().as_millis().min(u64::MAX as u128) as u64),
            mailbox_depth: self.handle.mailbox_depth(),
            worker_count: recovery.worker_count,
            rank_count: recovery.rank_count,
            recovering_rank_count: recovery.recovering_rank_count,
            pending_live_event_count: recovery.pending_live_event_count,
            discovered_endpoint_count: recovery.discovered_endpoint_count,
        }
    }

    pub async fn shutdown(&self) -> Result<(), KvDcRelayError> {
        self.intake_cancel.cancel();
        self.handle.shutdown().await
    }
}

type ActorStatsResult = Result<(DcCkfStats, Vec<(WorkerWithDpRank, usize)>), KvDcRelayError>;

enum ActorCommand {
    Apply {
        event: RouterEvent,
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    ReplaceRank {
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    RemoveRank {
        worker_id: WorkerId,
        dp_rank: DpRank,
        degraded: bool,
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    RemoveWorker {
        worker_id: WorkerId,
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    Flush {
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    Snapshot {
        response: oneshot::Sender<Result<ActorSnapshot, KvDcRelayError>>,
    },
    Stats {
        response: oneshot::Sender<ActorStatsResult>,
    },
    #[allow(dead_code)] // Reserved for the future Relay-to-global-router adapter.
    Subscribe {
        response: oneshot::Sender<Result<DcCkfSubscription, KvDcRelayError>>,
    },
    Shutdown {
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    #[cfg(test)]
    Pause {
        entered: oneshot::Sender<()>,
        release: oneshot::Receiver<()>,
    },
}

struct ActorSnapshot {
    snapshot: DcCkfSnapshot,
    stats: DcCkfStats,
}

impl ActorCommand {
    fn kind(&self) -> &'static str {
        match self {
            Self::Apply { .. } => "apply_event",
            Self::ReplaceRank { .. } => "replace_rank",
            Self::RemoveRank { .. } => "remove_rank",
            Self::RemoveWorker { .. } => "remove_worker",
            Self::Flush { .. } => "flush",
            Self::Snapshot { .. } => "snapshot",
            Self::Stats { .. } => "stats",
            Self::Subscribe { .. } => "subscribe",
            Self::Shutdown { .. } => "shutdown",
            #[cfg(test)]
            Self::Pause { .. } => "test_pause",
        }
    }
}

async fn run_actor(
    mut state: DcCkfState,
    mut receiver: mpsc::Receiver<ActorCommand>,
    counters: Arc<ActorCounters>,
    activity: Arc<Mutex<ActorActivity>>,
) {
    let mut subscribers: Vec<mpsc::Sender<DcCkfDelta>> = Vec::new();
    let mut shutdown_response = None;
    while let Some(command) = receiver.recv().await {
        let kind = command.kind();
        {
            let mut active = activity.lock();
            active.active_command = Some(kind);
            active.active_since = Some(Instant::now());
        }
        match command {
            ActorCommand::Apply { event, response } => {
                let worker_id = event.worker_id;
                let dp_rank = event.event.dp_rank;
                let event_id = event.event.event_id;
                let outcome = state.apply_event(event);
                let first_error = outcome.first_error().copied();
                let publication_boundary = outcome.publication_boundary();
                if outcome.unknown_removals() != 0 {
                    tracing::warn!(
                        worker_id,
                        dp_rank,
                        event_id,
                        unknown_removals = outcome.unknown_removals(),
                        "Ignoring KV DC Relay removals not owned by this worker/rank"
                    );
                }
                if let Some(delta) = outcome.into_delta() {
                    publish_delta(delta, &mut subscribers, &counters);
                } else if publication_boundary {
                    counters
                        .unchanged_publications
                        .fetch_add(1, Ordering::Relaxed);
                }
                let result = first_error.map_or(Ok(()), |error| Err(error.into()));
                let _ = response.send(result);
            }
            ActorCommand::ReplaceRank {
                worker_id,
                dp_rank,
                events,
                response,
            } => {
                let rebuild_started = Instant::now();
                let result = replacement_hashes(worker_id, dp_rank, events)
                    .and_then(|hashes| {
                        state
                            .replace_rank(WorkerWithDpRank::new(worker_id, dp_rank), hashes)
                            .map_err(Into::into)
                    })
                    .map(|delta| {
                        if let Some(delta) = delta {
                            publish_delta(delta, &mut subscribers, &counters);
                        } else {
                            counters
                                .unchanged_publications
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    });
                let elapsed = rebuild_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                counters.rebuild_count.fetch_add(1, Ordering::Relaxed);
                counters.rebuild_ns.fetch_add(elapsed, Ordering::Relaxed);
                counters
                    .rebuild_max_ns
                    .fetch_max(elapsed, Ordering::Relaxed);
                let _ = response.send(result);
            }
            ActorCommand::RemoveRank {
                worker_id,
                dp_rank,
                degraded,
                response,
            } => {
                let result = state
                    .remove_rank(WorkerWithDpRank::new(worker_id, dp_rank))
                    .map(|delta| {
                        if degraded {
                            counters.degraded_resets.fetch_add(1, Ordering::Relaxed);
                        }
                        if let Some(delta) = delta {
                            publish_delta(delta, &mut subscribers, &counters);
                        } else {
                            counters
                                .unchanged_publications
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    })
                    .map_err(Into::into);
                let _ = response.send(result);
            }
            ActorCommand::RemoveWorker {
                worker_id,
                response,
            } => {
                let result = state
                    .remove_worker(worker_id)
                    .map(|delta| {
                        if let Some(delta) = delta {
                            publish_delta(delta, &mut subscribers, &counters);
                        } else {
                            counters
                                .unchanged_publications
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    })
                    .map_err(Into::into);
                let _ = response.send(result);
            }
            ActorCommand::Flush { response } => {
                if let Some(delta) = state.flush() {
                    publish_delta(delta, &mut subscribers, &counters);
                } else {
                    counters
                        .unchanged_publications
                        .fetch_add(1, Ordering::Relaxed);
                }
                let _ = response.send(Ok(()));
            }
            ActorCommand::Snapshot { response } => {
                let result = state
                    .snapshot()
                    .map_err(Into::into)
                    .map(|(delta, snapshot)| {
                        if let Some(delta) = delta {
                            publish_delta(delta, &mut subscribers, &counters);
                        }
                        ActorSnapshot {
                            snapshot,
                            stats: state.stats(),
                        }
                    });
                let _ = response.send(result);
            }
            ActorCommand::Stats { response } => {
                let _ = response.send(Ok((state.stats(), state.member_counts())));
            }
            ActorCommand::Subscribe { response } => {
                let result = state
                    .snapshot()
                    .map_err(Into::into)
                    .map(|(delta, snapshot)| {
                        if let Some(delta) = delta {
                            publish_delta(delta, &mut subscribers, &counters);
                        }
                        let (sender, deltas) = mpsc::channel(DEFAULT_SUBSCRIBER_CAPACITY);
                        subscribers.push(sender);
                        DcCkfSubscription { snapshot, deltas }
                    });
                let _ = response.send(result);
            }
            ActorCommand::Shutdown { response } => {
                if shutdown_response.is_some() {
                    let _ = response.send(Err(KvDcRelayError::ShuttingDown));
                } else {
                    receiver.close();
                    activity.lock().shutting_down = true;
                    shutdown_response = Some(response);
                }
            }
            #[cfg(test)]
            ActorCommand::Pause { entered, release } => {
                let _ = entered.send(());
                let _ = release.await;
            }
        }
        let mut active = activity.lock();
        active.active_command = None;
        active.active_since = None;
    }

    if let Some(delta) = state.flush() {
        publish_delta(delta, &mut subscribers, &counters);
    }
    subscribers.clear();
    if let Some(response) = shutdown_response {
        let _ = response.send(Ok(()));
    }
}

fn replacement_hashes(
    worker_id: WorkerId,
    dp_rank: DpRank,
    events: Vec<RouterEvent>,
) -> Result<FxHashSet<ExternalSequenceBlockHash>, KvDcRelayError> {
    let mut hashes = FxHashSet::default();
    for event in events {
        if event.worker_id != worker_id || event.event.dp_rank != dp_rank {
            return Err(KvDcRelayError::InvalidTreeDump {
                worker_id,
                dp_rank,
                message: "event identity does not match replacement rank".to_string(),
            });
        }
        if event.storage_tier != StorageTier::Device {
            continue;
        }
        let KvCacheEventData::Stored(store) = event.event.data else {
            return Err(KvDcRelayError::InvalidTreeDump {
                worker_id,
                dp_rank,
                message: "tree dump contains a non-Stored event".to_string(),
            });
        };
        hashes.try_reserve(store.blocks.len()).map_err(|_| {
            KvDcRelayError::Build(
                dynamo_kv_router::indexer::cuckoo::CkfBuildError::AllocationFailed,
            )
        })?;
        hashes.extend(store.blocks.into_iter().map(|block| block.block_hash));
    }
    Ok(hashes)
}

fn publish_delta(
    delta: DcCkfDelta,
    subscribers: &mut Vec<mpsc::Sender<DcCkfDelta>>,
    counters: &ActorCounters,
) {
    counters.publications.fetch_add(1, Ordering::Relaxed);
    subscribers.retain(|subscriber| match subscriber.try_send(delta.clone()) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Closed(_)) | Err(mpsc::error::TrySendError::Full(_)) => {
            false
        }
    });
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_kv_router::protocols::{
        KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
    };

    use super::*;

    fn stored(worker: WorkerWithDpRank, event_id: u64, hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: hashes
                        .iter()
                        .copied()
                        .map(|hash| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(hash),
                            tokens_hash: LocalBlockHash(hash),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: worker.dp_rank,
            },
        )
    }

    async fn pause_actor(handle: &KvDcRelayHandle) -> oneshot::Sender<()> {
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        handle
            .sender
            .send(ActorCommand::Pause {
                entered: entered_tx,
                release: release_rx,
            })
            .await
            .unwrap();
        entered_rx.await.unwrap();
        release_tx
    }

    #[tokio::test]
    async fn subscriber_gets_snapshot_then_one_atomic_replacement_delta() {
        let worker = WorkerWithDpRank::new(1, 0);
        let handle = KvDcRelayHandle::spawn(CkfConfig::new(32)).unwrap();
        handle
            .apply_event(stored(worker, 1, &[1, 2]))
            .await
            .unwrap();
        let mut subscription = handle.subscribe().await.unwrap();
        let base_sequence = subscription.snapshot.sequence();

        handle
            .replace_rank(
                worker.worker_id,
                worker.dp_rank,
                vec![stored(worker, 0, &[3, 4])],
            )
            .await
            .unwrap();
        let delta = subscription.deltas.recv().await.unwrap();
        assert_eq!(delta.base_sequence(), base_sequence);
        assert_eq!(delta.sequence(), base_sequence + 1);
        assert!(!delta.reset());
        assert!(
            tokio::time::timeout(Duration::from_millis(20), subscription.deltas.recv())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn shared_owner_survives_transactional_rank_replacement() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let handle = KvDcRelayHandle::spawn(CkfConfig::new(32)).unwrap();
        handle.apply_event(stored(first, 1, &[9])).await.unwrap();
        handle.apply_event(stored(second, 1, &[9])).await.unwrap();

        handle
            .replace_rank(first.worker_id, first.dp_rank, Vec::new())
            .await
            .unwrap();
        let (stats, members) = handle.state_stats().await.unwrap();
        assert_eq!(stats.aggregation().unique_block_count(), 1);
        assert_eq!(stats.aggregation().contribution_count(), 1);
        assert_eq!(members, vec![(second, 1)]);
    }

    #[tokio::test]
    async fn shutdown_drains_accepted_commands_and_rejects_new_ones() {
        let worker = WorkerWithDpRank::new(1, 0);
        let handle = KvDcRelayHandle::spawn_with_capacity(CkfConfig::new(32), 4).unwrap();
        let release = pause_actor(&handle).await;

        let apply_handle = handle.clone();
        let apply =
            tokio::spawn(async move { apply_handle.apply_event(stored(worker, 1, &[1])).await });
        tokio::task::yield_now().await;
        let shutdown_handle = handle.clone();
        let shutdown = tokio::spawn(async move { shutdown_handle.shutdown().await });
        tokio::task::yield_now().await;
        release.send(()).unwrap();

        apply.await.unwrap().unwrap();
        shutdown.await.unwrap().unwrap();
        assert!(matches!(
            handle.apply_event(stored(worker, 2, &[2])).await,
            Err(KvDcRelayError::ShuttingDown)
        ));
    }

    #[tokio::test]
    async fn bounded_mailbox_applies_awaited_backpressure() {
        let worker = WorkerWithDpRank::new(1, 0);
        let handle = KvDcRelayHandle::spawn_with_capacity(CkfConfig::new(32), 1).unwrap();
        let release = pause_actor(&handle).await;

        let first_handle = handle.clone();
        let first =
            tokio::spawn(async move { first_handle.apply_event(stored(worker, 1, &[1])).await });
        tokio::task::yield_now().await;
        let second_handle = handle.clone();
        let mut second =
            tokio::spawn(async move { second_handle.apply_event(stored(worker, 2, &[2])).await });
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut second)
                .await
                .is_err()
        );

        release.send(()).unwrap();
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
        let (stats, _) = handle.state_stats().await.unwrap();
        assert_eq!(stats.aggregation().unique_block_count(), 2);
    }

    #[tokio::test]
    async fn lagging_publication_subscriber_is_closed_before_updates_are_lost() {
        let worker = WorkerWithDpRank::new(1, 0);
        let handle = KvDcRelayHandle::spawn(CkfConfig::new(128)).unwrap();
        let mut subscription = handle.subscribe().await.unwrap();

        for event_id in 1..=DEFAULT_SUBSCRIBER_CAPACITY as u64 + 1 {
            handle
                .apply_event(stored(worker, event_id, &[event_id]))
                .await
                .unwrap();
        }

        let mut received = 0;
        while subscription.deltas.recv().await.is_some() {
            received += 1;
        }
        assert_eq!(received, DEFAULT_SUBSCRIBER_CAPACITY);
    }
}
