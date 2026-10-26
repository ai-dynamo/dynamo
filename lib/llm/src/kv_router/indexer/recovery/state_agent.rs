// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Version-aware KV state-source selection and router recovery.
//!
//! This controller lives at one router/indexer boundary. It owns source-mode
//! selection and publishes one aggregate CacheOwner projection; router-core
//! remains unaware of discovery and attachment lifecycles.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result};
use dashmap::DashMap;
use dynamo_kv_router::{
    identity::CacheOwnerId,
    indexer::{
        KvStateAgentIdentity, KvStateAgentStatus, KvStateProtocolVersion, WorkerKvQueryResponse,
    },
    protocols::{
        KvCacheEvent, KvCacheEventData, ResetScope, ResidencyDomain, ResidencyProjection,
        ResidencyRoutingSnapshot, RouterEvent, RouterHintSourceMetadata, StorageTier,
        WorkerWithDpRank,
    },
};
use dynamo_runtime::{
    component::Component,
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, DiscoveryStream,
        EventScope, EventSourceInstanceId, EventSourceQuery,
    },
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{Codec, EventSubscriber, ValidatedEnvelope, uses_direct_zmq},
};
use futures::{FutureExt, StreamExt};
use tokio::{
    sync::{Mutex, Semaphore, mpsc, oneshot, watch},
    time::Instant,
};
use tokio_util::sync::CancellationToken;

use crate::discovery::kv_state_agent::{
    KV_STATE_ATTACHMENT_TOPIC_V2, KV_STATE_EVENT_TOPIC_V2, KV_STATE_SOURCE_TOPIC_V2,
    KvStateAttachmentAdvertisement, KvStateSourceAdvertisement, attachment_record_id,
};
use crate::{
    direct_zmq_fan_in::{
        ContinuityMode, FanInEvent, FanInObservation, start_direct_zmq_fan_in_async,
    },
    direct_zmq_sub_pool::KV_ZMQ_RCVHWM,
    discovery::{KvSourceMembershipView, KvSourceMembershipWatch, KvSourceStatus},
    kv_router::Indexer,
};

use super::{
    RuntimeWorkerQueryTransport, recovery_lane::RecoveryLane, worker_query_state::RankState,
};

const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);
const RECOVERY_TIMEOUT: Duration = Duration::from_secs(30);
const RECONCILE_RETRY_INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const RECONCILE_RETRY_MAX_BACKOFF: Duration = Duration::from_secs(5);
/// Concurrent status handshakes per router; snapshots share the recovery lane's
/// separate budget.
const STATUS_CONCURRENCY_LIMIT: usize = 64;

#[derive(Clone)]
struct OwnerRuntime {
    publisher_id: Option<u64>,
    /// Live ingest of `publisher_id`, shared with its event source.
    ingest: Option<SharedIngest>,
    attachment_generation: Option<u64>,
    last_worker: Option<WorkerWithDpRank>,
    ready: bool,
    attached_worker: Option<WorkerWithDpRank>,
    router_hint_source: Option<RouterHintSourceMetadata>,
    hint_ready: bool,
    /// Fence revision this state was planned at; it is routable only while
    /// that revision is current.
    revision: OwnerRevision,
}

/// Live-ingest state of one source publisher bound to its owner.
///
/// The publisher's event source applies batches under this lock, and
/// reconciliation binds, recovers, and retires it under the same lock, so each
/// batch lands entirely before or after a transition.
struct OwnerIngest {
    owner: CacheOwnerId,
    publisher_id: u64,
    recovered_cursor: u64,
    attachment_generation: Option<u64>,
    last_worker: Option<WorkerWithDpRank>,
    recovery_required: bool,
    /// Present while recovery is in flight: live events are buffered, then
    /// replayed after the recovered cursor.
    tail: Option<OwnerRecoveryTail>,
    /// Unbound from its owner; late batches are dropped.
    retired: bool,
}

type SharedIngest = Arc<Mutex<OwnerIngest>>;

/// Source publishers bound to owners, shared with every event source.
type IngestBindings = Arc<DashMap<u64, SharedIngest>>;

struct OwnerRecoveryTail {
    attachment_generation: Option<u64>,
    schedule_generation: u64,
    rank: RankState,
}

/// Applies live state-agent batches on the event source that received them.
struct LiveIngest {
    indexer: Indexer,
    bindings: IngestBindings,
    /// Owners whose live stream newly requires recovery; reconciliation
    /// withdraws them from routing.
    withdrawn: mpsc::UnboundedSender<(CacheOwnerId, u64)>,
}

impl LiveIngest {
    async fn apply(&self, publisher_id: u64, events: Vec<RouterEvent>) {
        // An unbound publisher's events precede its owner's recovery snapshot,
        // which covers them.
        let Some(ingest) = self
            .bindings
            .get(&publisher_id)
            .map(|entry| entry.value().clone())
        else {
            return;
        };
        let mut ingest = ingest.lock().await;
        if ingest.retired {
            return;
        }
        // Recovery replays the tail even when it was scheduled because the live
        // stream itself required recovery.
        if let Some(tail) = ingest.tail.as_mut() {
            for event in events {
                tail.rank.buffer_recovery_tail(event);
            }
            return;
        }
        if ingest.recovery_required {
            return;
        }
        apply_events_for_owner(&self.indexer, events, &mut ingest).await;
        if ingest.recovery_required {
            let _ = self.withdrawn.send((ingest.owner, publisher_id));
        }
    }
}

#[derive(Clone)]
struct OwnerRecoveryPlan {
    owner: CacheOwnerId,
    source: KvStateSourceAdvertisement,
    attachment: Option<KvStateAttachmentAdvertisement>,
    expected: KvStateAgentIdentity,
    recognized_worker: Option<WorkerWithDpRank>,
    previous_publisher_id: Option<u64>,
    previous_cursor: u64,
    recovery_required: bool,
    revision: OwnerRevision,
    schedule_generation: u64,
}

#[derive(Clone, PartialEq)]
struct RecoveryRetryFence {
    source: KvStateSourceAdvertisement,
    attachment: Option<KvStateAttachmentAdvertisement>,
    revision: OwnerRevision,
    activation_expected: bool,
}

struct RetryTimer {
    deadline: Option<Instant>,
    next_delay: Duration,
}

impl Default for RetryTimer {
    fn default() -> Self {
        Self {
            deadline: None,
            next_delay: RECONCILE_RETRY_INITIAL_BACKOFF,
        }
    }
}

impl RetryTimer {
    fn schedule(&mut self) {
        if self.deadline.is_some() {
            return;
        }
        self.deadline = Some(Instant::now() + self.next_delay);
        self.next_delay = (self.next_delay * 2).min(RECONCILE_RETRY_MAX_BACKOFF);
    }

    fn take_due(&mut self, now: Instant) -> bool {
        let due = self.deadline.is_some_and(|deadline| deadline <= now);
        if due {
            self.deadline = None;
        }
        due
    }
}

/// Backoff for reconciling again without a discovery change: per owner after its
/// recovery fails or misses readiness, and for every owner after activation or
/// a reconciliation fails.
#[derive(Default)]
struct RecoveryRetries {
    owners: HashMap<CacheOwnerId, (RetryTimer, RecoveryRetryFence)>,
    all: RetryTimer,
}

impl RecoveryRetries {
    fn schedule(&mut self, owner: CacheOwnerId, fence: RecoveryRetryFence) {
        let (timer, current) = self
            .owners
            .entry(owner)
            .or_insert_with(|| (RetryTimer::default(), fence.clone()));
        if timer.deadline.is_none() {
            *current = fence;
            timer.schedule();
        }
    }

    fn reset(&mut self, owner: &CacheOwnerId) {
        self.owners.remove(owner);
    }

    fn next_deadline(&self) -> Option<Instant> {
        self.owners
            .values()
            .filter_map(|(timer, _)| timer.deadline)
            .chain(self.all.deadline)
            .min()
    }

    fn take_due(&mut self) -> (bool, Vec<(CacheOwnerId, RecoveryRetryFence)>) {
        let now = Instant::now();
        let all = self.all.take_due(now);
        let owners = self
            .owners
            .iter_mut()
            .filter_map(|(owner, (timer, fence))| {
                timer.take_due(now).then(|| (*owner, fence.clone()))
            })
            .collect();
        (all, owners)
    }
}

/// Owners waiting for reconciliation.
#[derive(Default)]
struct DirtyOwners {
    owners: HashSet<CacheOwnerId>,
    all: bool,
}

impl DirtyOwners {
    fn mark(&mut self, owner: CacheOwnerId) {
        self.owners.insert(owner);
    }

    fn is_empty(&self) -> bool {
        !self.all && self.owners.is_empty()
    }

    /// The owners to reconcile now: every known owner after `all`.
    fn take(
        &mut self,
        runtime: &HashMap<CacheOwnerId, OwnerRuntime>,
        sources: &HashMap<u64, KvStateSourceAdvertisement>,
        attachments: &HashMap<u64, KvStateAttachmentAdvertisement>,
    ) -> HashSet<CacheOwnerId> {
        let mut owners = std::mem::take(&mut self.owners);
        if std::mem::take(&mut self.all) {
            owners.extend(runtime.keys().copied());
            owners.extend(sources.values().map(|source| source.cache_owner_id));
            owners.extend(
                attachments
                    .values()
                    .map(|attachment| attachment.cache_owner_id),
            );
        }
        owners
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct OwnerRevision {
    /// Bumped when a discovery event cannot be attributed to owners.
    epoch: u64,
    owner: u64,
}

/// Owners a discovery event may affect.
#[derive(Debug)]
enum Affected {
    Owners(HashSet<CacheOwnerId>),
    All,
}

impl Affected {
    fn merge(&mut self, other: Affected) {
        match (self, other) {
            (Affected::All, _) => {}
            (this, Affected::All) => *this = Affected::All,
            (Affected::Owners(owners), Affected::Owners(more)) => owners.extend(more),
        }
    }
}

struct PublishedOwner {
    worker: Option<WorkerWithDpRank>,
    hint: Option<(RouterHintSourceMetadata, Option<WorkerWithDpRank>)>,
}

/// The CacheOwner routing projection and the per-owner discovery revisions that
/// fence it.
///
/// The observation fence bumps the revision of every owner a discovery event may
/// affect and withdraws those owners, even while reconciliation is blocked on a
/// clear or a snapshot. Reconciliation publishes an owner only from a plan made
/// at its current revision, so a stale plan can never republish it.
#[derive(Default)]
struct ProjectionFence {
    epoch: u64,
    owners: HashMap<CacheOwnerId, u64>,
    published: HashMap<CacheOwnerId, PublishedOwner>,
}

type SharedFence = Arc<Mutex<ProjectionFence>>;

impl ProjectionFence {
    fn revision(&self, owner: &CacheOwnerId) -> OwnerRevision {
        OwnerRevision {
            epoch: self.epoch,
            owner: self.owners.get(owner).copied().unwrap_or(0),
        }
    }

    /// Advances the affected owners' revisions and withdraws them from routing.
    fn invalidate(&mut self, indexer: &Indexer, affected: &Affected) {
        let withdrawn = match affected {
            Affected::All => {
                self.epoch = self.epoch.wrapping_add(1);
                let withdrawn = !self.published.is_empty();
                self.published.clear();
                withdrawn
            }
            Affected::Owners(owners) => {
                let mut withdrawn = false;
                for owner in owners {
                    let revision = self.owners.entry(*owner).or_default();
                    *revision = revision.wrapping_add(1);
                    withdrawn |= self.published.remove(owner).is_some();
                }
                withdrawn
            }
        };
        if withdrawn {
            self.publish(indexer);
        }
    }

    /// Fails closed: nothing is routable until owners are planned again.
    fn withdraw_all(&mut self, indexer: &Indexer) {
        self.invalidate(indexer, &Affected::All);
        indexer.set_residency_routing_snapshot(ResidencyRoutingSnapshot::default());
    }

    /// Publishes every ready owner whose state was planned at its current revision.
    fn commit(&mut self, indexer: &Indexer, runtime: &HashMap<CacheOwnerId, OwnerRuntime>) {
        let published = runtime
            .iter()
            .filter(|(owner, state)| state.revision == self.revision(owner))
            .filter_map(|(owner, state)| {
                let worker = state.ready.then_some(state.last_worker).flatten();
                let hint = state
                    .hint_ready
                    .then(|| {
                        state
                            .router_hint_source
                            .clone()
                            .map(|metadata| (metadata, state.attached_worker))
                    })
                    .flatten();
                (worker.is_some() || hint.is_some())
                    .then_some((*owner, PublishedOwner { worker, hint }))
            })
            .collect();
        self.published = published;
        self.publish(indexer);
    }

    fn publish(&self, indexer: &Indexer) {
        let projection = ResidencyProjection::new(
            self.published
                .iter()
                .filter_map(|(owner, published)| published.worker.map(|worker| (*owner, worker))),
        )
        .unwrap_or_default();
        let hint_sources = self.published.iter().filter_map(|(owner, published)| {
            published
                .hint
                .clone()
                .map(|(metadata, worker)| (*owner, metadata, worker))
        });
        indexer.set_residency_routing_snapshot(ResidencyRoutingSnapshot::new(
            projection,
            hint_sources,
        ));
    }
}

/// What the observation fence knows from its own watches: enough to name the
/// owners each discovery event can affect.
#[derive(Default)]
struct FenceIndex {
    source_owners: HashMap<u64, CacheOwnerId>,
    attachments: HashMap<u64, (CacheOwnerId, WorkerWithDpRank)>,
    attached: HashMap<WorkerWithDpRank, HashSet<CacheOwnerId>>,
    live_workers: HashSet<WorkerWithDpRank>,
}

impl FenceIndex {
    /// Starts from the membership already in the watch, which `changed()` never
    /// delivers, so a worker present at startup names its owners when it leaves.
    fn new(membership: &KvSourceMembershipView) -> Self {
        Self {
            live_workers: membership.sources.keys().copied().collect(),
            ..Self::default()
        }
    }

    fn source(&mut self, event: Result<DiscoveryEvent>) -> Affected {
        match event {
            Ok(DiscoveryEvent::Added(DiscoveryInstance::EventSource {
                publisher_id,
                metadata,
                ..
            })) => match serde_json::from_value::<KvStateSourceAdvertisement>(metadata) {
                Ok(source) => {
                    self.source_owners
                        .insert(publisher_id, source.cache_owner_id);
                    Affected::Owners(HashSet::from([source.cache_owner_id]))
                }
                Err(_) => Affected::All,
            },
            Ok(DiscoveryEvent::Removed(DiscoveryInstanceId::EventSource(
                EventSourceInstanceId { publisher_id, .. },
            ))) => self
                .source_owners
                .remove(&publisher_id)
                .map_or(Affected::All, |owner| {
                    Affected::Owners(HashSet::from([owner]))
                }),
            // A resync follows the changes it reconciled, already seen above.
            Ok(_) => Affected::Owners(HashSet::new()),
            Err(_) => Affected::All,
        }
    }

    /// An attachment also affects every owner attached to the same worker, which
    /// closes the window where a worker moves between owners.
    fn attachment(&mut self, event: Result<DiscoveryEvent>) -> Affected {
        let (owner, worker) = match event {
            Ok(DiscoveryEvent::Added(DiscoveryInstance::EventSource {
                publisher_id,
                metadata,
                ..
            })) => {
                let Ok(attachment) =
                    serde_json::from_value::<KvStateAttachmentAdvertisement>(metadata)
                else {
                    return Affected::All;
                };
                let entry = (attachment.cache_owner_id, attachment.worker);
                self.attachments.insert(publisher_id, entry);
                self.attached
                    .entry(attachment.worker)
                    .or_default()
                    .insert(attachment.cache_owner_id);
                entry
            }
            Ok(DiscoveryEvent::Removed(DiscoveryInstanceId::EventSource(
                EventSourceInstanceId { publisher_id, .. },
            ))) => {
                let Some((owner, worker)) = self.attachments.remove(&publisher_id) else {
                    return Affected::All;
                };
                if !self
                    .attachments
                    .values()
                    .any(|attached| *attached == (owner, worker))
                    && let Some(owners) = self.attached.get_mut(&worker)
                {
                    owners.remove(&owner);
                }
                (owner, worker)
            }
            Ok(_) => return Affected::Owners(HashSet::new()),
            Err(_) => return Affected::All,
        };
        let mut owners = self.attached.get(&worker).cloned().unwrap_or_default();
        owners.insert(owner);
        Affected::Owners(owners)
    }

    /// Folds the discovery events already queued on either watch into
    /// `affected`, so a burst withdraws its owners with one projection rebuild.
    /// Returns `None` once a watch ends.
    fn drain_ready(
        &mut self,
        sources: &mut DiscoveryStream,
        attachments: &mut DiscoveryStream,
        mut affected: Affected,
    ) -> Option<Affected> {
        loop {
            let next = if let Some(event) = sources.next().now_or_never() {
                event.map(|event| self.source(event))
            } else if let Some(event) = attachments.next().now_or_never() {
                event.map(|event| self.attachment(event))
            } else {
                return Some(affected);
            };
            affected.merge(next?);
        }
    }

    /// Workers that joined or left affect the owners attached to them.
    fn membership(&mut self, view: &KvSourceMembershipView) -> Affected {
        let live_workers: HashSet<_> = view.sources.keys().copied().collect();
        let owners = live_workers
            .symmetric_difference(&self.live_workers)
            .filter_map(|worker| self.attached.get(worker))
            .flatten()
            .copied()
            .collect();
        self.live_workers = live_workers;
        Affected::Owners(owners)
    }
}

enum OwnerRecoveryResult {
    Initial {
        plan: OwnerRecoveryPlan,
        status: Result<KvStateAgentStatus>,
        recovery: Option<Result<WorkerKvQueryResponse>>,
    },
    PostStatus {
        plan: OwnerRecoveryPlan,
        recovered_cursor: u64,
        status: Result<KvStateAgentStatus>,
    },
}

/// Router-owned state-agent work plus the filtered legacy source view.
pub(crate) struct KvStateRouterHandle {
    pub(crate) legacy_membership: KvSourceMembershipWatch,
    completion: oneshot::Receiver<()>,
}

impl KvStateRouterHandle {
    pub(crate) fn into_parts(self) -> (KvSourceMembershipWatch, oneshot::Receiver<()>) {
        (self.legacy_membership, self.completion)
    }
}

pub(crate) async fn start_state_agent_router(
    component: Component,
    indexer: Indexer,
    membership: KvSourceMembershipWatch,
    expected_block_size: u32,
    cancellation_token: CancellationToken,
) -> Result<KvStateRouterHandle> {
    let endpoint = membership
        .borrow()
        .resolved_kv_state_endpoint()
        .cloned()
        .context("KV state endpoint is ambiguous while starting state-agent reconciliation")?;
    let discovery = component.drt().discovery();
    let source_cancel = cancellation_token.child_token();
    let sources = discovery
        .list_and_watch(
            DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
                endpoint.clone(),
                KV_STATE_SOURCE_TOPIC_V2,
            )),
            Some(source_cancel),
        )
        .await
        .context("failed to watch V2 KV state sources")?;

    let (recognized_tx, recognized_rx) =
        watch::channel(source_mode_suppressed_workers(&membership.borrow()));
    let legacy_membership = filtered_legacy_membership(
        membership.clone(),
        recognized_rx,
        cancellation_token.child_token(),
    );
    let (completion_tx, completion_rx) = oneshot::channel();
    let fence = SharedFence::default();
    let (fenced_tx, fenced_rx) = mpsc::unbounded_channel();
    start_observation_fence(
        component.clone(),
        indexer.clone(),
        membership.clone(),
        endpoint.clone(),
        recognized_tx.clone(),
        fence.clone(),
        fenced_tx,
        cancellation_token.child_token(),
    )
    .await?;
    component.drt().runtime().secondary().spawn(async move {
        run_state_agent_router(
            indexer,
            membership,
            component,
            endpoint,
            expected_block_size,
            sources,
            recognized_tx,
            fence,
            fenced_rx,
            cancellation_token,
        )
        .await;
        let _ = completion_tx.send(());
    });
    Ok(KvStateRouterHandle {
        legacy_membership,
        completion: completion_rx,
    })
}

#[allow(clippy::too_many_arguments)]
async fn start_observation_fence(
    component: Component,
    indexer: Indexer,
    mut membership: KvSourceMembershipWatch,
    endpoint: EndpointId,
    recognized_tx: watch::Sender<HashSet<WorkerWithDpRank>>,
    fence: SharedFence,
    fenced_tx: mpsc::UnboundedSender<Affected>,
    cancel: CancellationToken,
) -> Result<()> {
    let discovery = component.drt().discovery();
    let mut sources = discovery
        .list_and_watch(
            DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
                endpoint.clone(),
                KV_STATE_SOURCE_TOPIC_V2,
            )),
            Some(cancel.child_token()),
        )
        .await?;
    let mut attachments = discovery
        .list_and_watch(
            DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
                endpoint,
                KV_STATE_ATTACHMENT_TOPIC_V2,
            )),
            Some(cancel.child_token()),
        )
        .await?;
    component.drt().runtime().secondary().spawn(async move {
        // NOTE: This independent watch must withdraw affected owners while the
        // reconciliation loop is blocked on acknowledged clears or applying a
        // snapshot. Temporary loss of KV-aware routing for those owners is
        // intentional; ordinary serving continues without a stale projection.
        // TODO(#13044): share these model-wide watches.
        let mut index = FenceIndex::new(&membership.borrow());
        loop {
            let affected = tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                changed = membership.changed() => changed.ok().map(|()| {
                    let view = membership.borrow_and_update().clone();
                    recognized_tx.send_modify(|recognized| {
                        recognized.extend(source_mode_suppressed_workers(&view));
                    });
                    index.membership(&view)
                }),
                event = sources.next() => event.map(|event| index.source(event)),
                event = attachments.next() => event.map(|event| index.attachment(event)),
            };
            let Some(affected) = affected
                .and_then(|affected| index.drain_ready(&mut sources, &mut attachments, affected))
            else {
                // Dropping `fenced_tx` ends reconciliation, which keeps V2
                // routing withdrawn.
                tracing::error!("KV state observation fence lost a discovery watch");
                fence.lock().await.withdraw_all(&indexer);
                break;
            };
            if matches!(&affected, Affected::Owners(owners) if owners.is_empty()) {
                continue;
            }
            fence.lock().await.invalidate(&indexer, &affected);
            let _ = fenced_tx.send(affected);
        }
    });
    Ok(())
}

fn filtered_legacy_membership(
    membership: KvSourceMembershipWatch,
    mut recognized: watch::Receiver<HashSet<WorkerWithDpRank>>,
    cancel: CancellationToken,
) -> KvSourceMembershipWatch {
    let mut source = membership.clone();
    let initial = suppress_recognized(source.borrow().clone(), &recognized.borrow());
    let (tx, rx) = watch::channel(initial);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                changed = source.changed() => if changed.is_err() { break; },
                changed = recognized.changed() => if changed.is_err() { break; },
            }
            let view = suppress_recognized(
                source.borrow_and_update().clone(),
                &recognized.borrow_and_update(),
            );
            tx.send_replace(view);
        }
    });
    membership.with_receiver(rx)
}

fn suppress_recognized(
    mut view: KvSourceMembershipView,
    recognized: &HashSet<WorkerWithDpRank>,
) -> KvSourceMembershipView {
    for worker in recognized {
        if let Some(status) = view.sources.get_mut(worker) {
            *status = KvSourceStatus::Suppressed;
        }
    }
    view
}

fn source_mode_suppressed_workers(view: &KvSourceMembershipView) -> HashSet<WorkerWithDpRank> {
    view.sources
        .keys()
        .copied()
        .filter(|worker| {
            !matches!(
                view.kv_event_source_mode(worker.worker_id),
                None | Some("framework_v1")
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
async fn run_state_agent_router(
    indexer: Indexer,
    mut membership: KvSourceMembershipWatch,
    component: Component,
    endpoint: EndpointId,
    expected_block_size: u32,
    mut source_stream: DiscoveryStream,
    recognized_tx: watch::Sender<HashSet<WorkerWithDpRank>>,
    fence: SharedFence,
    mut fenced_rx: mpsc::UnboundedReceiver<Affected>,
    cancel: CancellationToken,
) {
    let mut attachment_stream: Option<DiscoveryStream> = None;
    let mut ingest_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut transport: Option<Arc<RuntimeWorkerQueryTransport>> = None;
    let mut sources = HashMap::new();
    let mut attachments = HashMap::new();
    let mut runtime: HashMap<CacheOwnerId, OwnerRuntime> = HashMap::new();
    let mut live_workers: HashSet<WorkerWithDpRank> =
        membership.borrow().sources.keys().copied().collect();
    let bindings = IngestBindings::default();
    let (withdrawn_tx, mut withdrawn_rx) = mpsc::unbounded_channel();
    let live_ingest = Arc::new(LiveIngest {
        indexer: indexer.clone(),
        bindings: bindings.clone(),
        withdrawn: withdrawn_tx,
    });
    let recovery_lane = Arc::new(RecoveryLane::new());
    let status_permits = Arc::new(Semaphore::new(STATUS_CONCURRENCY_LIMIT));
    let (recovery_tx, mut recovery_rx) = mpsc::unbounded_channel();
    let mut next_recovery_schedule = 0u64;
    let mut retries = RecoveryRetries::default();
    let mut dirty = DirtyOwners::default();

    loop {
        enum Input {
            Membership,
            Source(Result<DiscoveryEvent>),
            Attachment(Result<DiscoveryEvent>),
            Fenced(Affected),
            Withdrawn((CacheOwnerId, u64)),
            Recovery(Option<Box<OwnerRecoveryResult>>),
            Retry,
            /// An input V2 routing depends on ended; it stays withdrawn.
            Ended(&'static str),
            Cancelled,
        }
        let input = tokio::select! {
            biased;
            _ = cancel.cancelled() => Input::Cancelled,
            changed = membership.changed() => {
                if changed.is_err() { Input::Cancelled } else { Input::Membership }
            },
            event = source_stream.next() => event.map_or(Input::Ended(WATCH_ENDED), Input::Source),
            event = async {
                match attachment_stream.as_mut() {
                    Some(stream) => stream.next().await,
                    None => std::future::pending().await,
                }
            } => event.map_or(Input::Ended(WATCH_ENDED), Input::Attachment),
            affected = fenced_rx.recv() => {
                affected.map_or(Input::Ended("KV state observation fence ended"), Input::Fenced)
            },
            Some(withdrawn) = withdrawn_rx.recv() => Input::Withdrawn(withdrawn),
            _ = async {
                match ingest_task.as_mut() {
                    Some(task) => { let _ = task.await; }
                    None => std::future::pending().await,
                }
            } => Input::Ended("V2 KV state event ingest ended"),
            result = recovery_rx.recv() => Input::Recovery(result.map(Box::new)),
            _ = wait_for_retry_deadline(retries.next_deadline()) => Input::Retry,
        };
        match input {
            Input::Cancelled => break,
            Input::Membership => {
                membership.borrow_and_update();
                let snapshot = membership.borrow().clone();
                let mut recognized = source_mode_suppressed_workers(&snapshot);
                recognized.extend(
                    recognized_tx
                        .borrow()
                        .iter()
                        .copied()
                        .filter(|worker| snapshot.sources.contains_key(worker)),
                );
                recognized_tx.send_replace(recognized);
                let workers: HashSet<_> = snapshot.sources.keys().copied().collect();
                let changed: HashSet<_> = workers
                    .symmetric_difference(&live_workers)
                    .copied()
                    .collect();
                live_workers = workers;
                for owner in owners_on_workers(&changed, &attachments, &runtime) {
                    retries.reset(&owner);
                    dirty.mark(owner);
                }
            }
            Input::Source(event) => apply_discovery_event(
                event,
                &endpoint,
                KV_STATE_SOURCE_TOPIC_V2,
                &mut sources,
                &mut retries,
                &mut dirty,
            ),
            Input::Attachment(event) => apply_discovery_event(
                event,
                &endpoint,
                KV_STATE_ATTACHMENT_TOPIC_V2,
                &mut attachments,
                &mut retries,
                &mut dirty,
            ),
            Input::Ended(reason) => {
                fail_closed_until_cancelled(&endpoint, reason, &fence, &indexer, &cancel).await;
                break;
            }
            Input::Fenced(Affected::All) => dirty.all = true,
            Input::Fenced(Affected::Owners(owners)) => {
                // An owner already planned at its new revision needs nothing more.
                let fence = fence.lock().await;
                for owner in owners {
                    if runtime
                        .get(&owner)
                        .is_none_or(|state| state.revision != fence.revision(&owner))
                    {
                        dirty.mark(owner);
                    }
                }
            }
            Input::Withdrawn((owner, publisher_id)) => {
                if withdraw_owner(
                    owner,
                    publisher_id,
                    &mut runtime,
                    &sources,
                    &attachments,
                    &mut retries,
                ) {
                    fence.lock().await.commit(&indexer, &runtime);
                }
            }
            Input::Recovery(Some(result)) => {
                let owner = result.owner();
                let publisher_id = result.publisher_id();
                let retry_fence = result.retry_fence();
                recovery_lane.finish((owner, result.schedule_generation()));
                if let Err(error) = finish_owner_recovery(
                    *result,
                    &indexer,
                    transport.as_ref(),
                    &status_permits,
                    &membership,
                    &sources,
                    &attachments,
                    &mut runtime,
                    &recovery_lane,
                    &recovery_tx,
                    &fence,
                )
                .await
                {
                    fail_owner_recovery(owner, publisher_id, &mut runtime).await;
                    tracing::warn!(%owner, %error, "KV state-source recovery completion failed closed");
                }
                let retry = match runtime
                    .get(&owner)
                    .filter(|state| state.publisher_id == Some(publisher_id))
                {
                    Some(state) => {
                        (retry_fence.activation_expected && !state.ready)
                            || match &state.ingest {
                                Some(ingest) => ingest.lock().await.recovery_required,
                                None => true,
                            }
                    }
                    None => false,
                };
                if retry {
                    retries.schedule(owner, retry_fence);
                }
            }
            Input::Recovery(None) => break,
            Input::Retry => {
                let (all, due) = retries.take_due();
                dirty.all |= all;
                let fence = fence.lock().await;
                for (owner, retry_fence) in due {
                    if recovery_retry_is_current(
                        &retry_fence,
                        &sources,
                        &attachments,
                        fence.revision(&owner),
                    ) {
                        dirty.mark(owner);
                    } else {
                        retries.reset(&owner);
                    }
                }
            }
        }
        if dirty.is_empty() {
            continue;
        }
        // One reconciliation covers every discovery update already queued: the
        // initial watch snapshot delivers each source and attachment as its own
        // event.
        let watches_open = apply_queued_discovery(
            &mut source_stream,
            &endpoint,
            KV_STATE_SOURCE_TOPIC_V2,
            &mut sources,
            &mut retries,
            &mut dirty,
        ) && attachment_stream.as_mut().is_none_or(|stream| {
            apply_queued_discovery(
                stream,
                &endpoint,
                KV_STATE_ATTACHMENT_TOPIC_V2,
                &mut attachments,
                &mut retries,
                &mut dirty,
            )
        });
        if !watches_open {
            fail_closed_until_cancelled(&endpoint, WATCH_ENDED, &fence, &indexer, &cancel).await;
            break;
        }
        if !sources.is_empty() && transport.is_none() {
            match start_state_agent_consumers(
                &component,
                &endpoint,
                live_ingest.clone(),
                cancel.child_token(),
            )
            .await
            {
                Ok((mut attachment_watch, state_transport, task)) => {
                    ingest_task = Some(task);
                    transport = Some(Arc::new(state_transport));
                    // Otherwise every owner is planned once without its
                    // attachment and again when the snapshot arrives.
                    let snapshot_applied = apply_initial_discovery(
                        &mut attachment_watch,
                        &endpoint,
                        KV_STATE_ATTACHMENT_TOPIC_V2,
                        &mut attachments,
                        &mut retries,
                        &mut dirty,
                        &cancel,
                    )
                    .await;
                    attachment_stream = Some(attachment_watch);
                    if !snapshot_applied {
                        fail_closed_until_cancelled(
                            &endpoint,
                            WATCH_ENDED,
                            &fence,
                            &indexer,
                            &cancel,
                        )
                        .await;
                        break;
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, %endpoint, "Failed to activate V2 KV state consumers");
                    fence.lock().await.withdraw_all(&indexer);
                    recognized_tx
                        .send_replace(membership.borrow().sources.keys().copied().collect());
                    retries.all.schedule();
                    continue;
                }
            }
        }
        let Some(transport) = transport.as_ref() else {
            continue;
        };
        let owners = dirty.take(&runtime, &sources, &attachments);
        let membership_snapshot = membership.borrow().clone();
        let Some(schedule_generation) = next_recovery_schedule.checked_add(1) else {
            tracing::error!("KV state recovery schedule generation exhausted");
            break;
        };
        next_recovery_schedule = schedule_generation;
        recovery_lane
            .cancel_where(|(owner, _)| owners.contains(owner))
            .await;
        if let Err(error) = schedule_state_sources(
            &indexer,
            transport.clone(),
            &status_permits,
            membership_snapshot,
            &endpoint,
            expected_block_size,
            &sources,
            &attachments,
            &mut runtime,
            &bindings,
            &recovery_lane,
            &recovery_tx,
            &recognized_tx,
            &fence,
            &owners,
            schedule_generation,
        )
        .await
        {
            tracing::warn!(%error, %endpoint, "KV state-source reconciliation failed closed");
            retries.all.schedule();
        } else {
            retries.all = RetryTimer::default();
        }
    }
    recovery_lane.cancel_all().await;
    fence.lock().await.withdraw_all(&indexer);
}

const WATCH_ENDED: &str = "KV state discovery watch ended";

/// Keeps V2 routing withdrawn until shutdown. Waiting keeps the recognized
/// sender alive, so the derived legacy view continues forwarding membership for
/// unrecognized ranks.
async fn fail_closed_until_cancelled(
    endpoint: &EndpointId,
    reason: &str,
    fence: &SharedFence,
    indexer: &Indexer,
    cancel: &CancellationToken,
) {
    tracing::error!(%endpoint, "{reason}; keeping recognized ranks fail-closed");
    fence.lock().await.withdraw_all(indexer);
    cancel.cancelled().await;
}

/// Applies one discovery event and queues its owner for reconciliation.
fn apply_discovery_event<T: Advertisement>(
    event: Result<DiscoveryEvent>,
    endpoint: &EndpointId,
    topic: &str,
    advertisements: &mut HashMap<u64, T>,
    retries: &mut RecoveryRetries,
    dirty: &mut DirtyOwners,
) {
    match update_advertisements(event, endpoint, topic, advertisements) {
        Ok(Some(owner)) => {
            retries.reset(&owner);
            dirty.mark(owner);
        }
        Ok(None) => {}
        Err(error) => tracing::warn!(%error, topic, "Ignoring invalid V2 KV state advertisement"),
    }
}

/// Applies the discovery events already queued on `stream`. Returns false once
/// the watch has ended.
fn apply_queued_discovery<T: Advertisement>(
    stream: &mut DiscoveryStream,
    endpoint: &EndpointId,
    topic: &str,
    advertisements: &mut HashMap<u64, T>,
    retries: &mut RecoveryRetries,
    dirty: &mut DirtyOwners,
) -> bool {
    loop {
        match stream.next().now_or_never() {
            Some(Some(event)) => {
                apply_discovery_event(event, endpoint, topic, advertisements, retries, dirty)
            }
            Some(None) => return false,
            None => return true,
        }
    }
}

/// Applies a new watch's initial snapshot: its `Added` events up to the first
/// `Resync`. A backend may still be delivering it when the watch is returned.
/// Returns false if the watch ends or the router is cancelled first.
#[allow(clippy::too_many_arguments)]
async fn apply_initial_discovery<T: Advertisement>(
    stream: &mut DiscoveryStream,
    endpoint: &EndpointId,
    topic: &str,
    advertisements: &mut HashMap<u64, T>,
    retries: &mut RecoveryRetries,
    dirty: &mut DirtyOwners,
    cancel: &CancellationToken,
) -> bool {
    loop {
        let event = tokio::select! {
            biased;
            _ = cancel.cancelled() => return false,
            event = stream.next() => event,
        };
        match event {
            Some(Ok(DiscoveryEvent::Resync(_))) => return true,
            Some(event) => {
                apply_discovery_event(event, endpoint, topic, advertisements, retries, dirty)
            }
            None => return false,
        }
    }
}

/// Owners whose plan depends on any of `workers`: attached to one, or last
/// projected onto one.
fn owners_on_workers(
    workers: &HashSet<WorkerWithDpRank>,
    attachments: &HashMap<u64, KvStateAttachmentAdvertisement>,
    runtime: &HashMap<CacheOwnerId, OwnerRuntime>,
) -> HashSet<CacheOwnerId> {
    if workers.is_empty() {
        return HashSet::new();
    }
    attachments
        .values()
        .filter(|attachment| workers.contains(&attachment.worker))
        .map(|attachment| attachment.cache_owner_id)
        .chain(runtime.iter().filter_map(|(owner, state)| {
            [state.last_worker, state.attached_worker]
                .into_iter()
                .flatten()
                .any(|worker| workers.contains(&worker))
                .then_some(*owner)
        }))
        .collect()
}

/// Withdraws an owner whose live stream now requires recovery and schedules
/// that recovery. Returns whether its routing changed.
fn withdraw_owner(
    owner: CacheOwnerId,
    publisher_id: u64,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
    sources: &HashMap<u64, KvStateSourceAdvertisement>,
    attachments: &HashMap<u64, KvStateAttachmentAdvertisement>,
    retries: &mut RecoveryRetries,
) -> bool {
    let Some(state) = runtime
        .get_mut(&owner)
        .filter(|state| state.publisher_id == Some(publisher_id))
    else {
        return false;
    };
    // Without its source record, the discovery change re-plans the owner.
    if let Some(source) = sources.get(&publisher_id) {
        let attachment = state
            .attachment_generation
            .and_then(|generation| attachments.get(&attachment_record_id(publisher_id, generation)))
            .cloned();
        retries.schedule(
            owner,
            RecoveryRetryFence {
                source: source.clone(),
                activation_expected: attachment.is_some() && state.last_worker.is_some(),
                attachment,
                revision: state.revision,
            },
        );
    }
    let was_published = state.ready || state.hint_ready;
    state.ready = false;
    state.hint_ready = false;
    was_published
}

async fn wait_for_retry_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

fn recovery_retry_is_current(
    fence: &RecoveryRetryFence,
    sources: &HashMap<u64, KvStateSourceAdvertisement>,
    attachments: &HashMap<u64, KvStateAttachmentAdvertisement>,
    revision: OwnerRevision,
) -> bool {
    if revision != fence.revision || sources.get(&fence.source.publisher_id) != Some(&fence.source)
    {
        return false;
    }
    fence.attachment.as_ref().is_none_or(|attachment| {
        attachments.get(&attachment_record_id(
            attachment.publisher_id,
            attachment.attachment_generation,
        )) == Some(attachment)
    })
}

async fn start_state_agent_consumers(
    component: &Component,
    endpoint: &EndpointId,
    live_ingest: Arc<LiveIngest>,
    cancel: CancellationToken,
) -> Result<(
    dynamo_runtime::discovery::DiscoveryStream,
    RuntimeWorkerQueryTransport,
    tokio::task::JoinHandle<()>,
)> {
    let attachments = component
        .drt()
        .discovery()
        .list_and_watch(
            DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
                endpoint.clone(),
                KV_STATE_ATTACHMENT_TOPIC_V2,
            )),
            Some(cancel.clone()),
        )
        .await
        .context("failed to watch V2 KV state attachments")?;
    let ingest = start_live_ingest(component, endpoint, live_ingest, cancel)
        .await
        .context("failed to subscribe to V2 KV state events")?;
    let transport = RuntimeWorkerQueryTransport::new(component).await?;
    Ok((attachments, transport, ingest))
}

/// Consumes every slot publisher's events. Direct ZMQ runs one source task per
/// publisher on the pooled SUB sockets the legacy KV path uses, so publishers
/// decode and apply concurrently and a full source backpressures its socket
/// instead of dropping. Brokered transports deliver one merged stream.
async fn start_live_ingest(
    component: &Component,
    endpoint: &EndpointId,
    live_ingest: Arc<LiveIngest>,
    cancel: CancellationToken,
) -> Result<tokio::task::JoinHandle<()>> {
    let transport_kind = component.drt().default_event_transport_kind();
    if uses_direct_zmq(transport_kind) {
        let codec = Codec::default();
        let handler = move |envelope: ValidatedEnvelope| {
            let live_ingest = live_ingest.clone();
            let events = codec.decode_payload::<Vec<RouterEvent>>(&envelope.payload);
            async move {
                live_ingest.apply(envelope.publisher_id, events?).await;
                Ok(())
            }
        };
        return start_direct_zmq_fan_in_async(
            component.clone(),
            endpoint.clone(),
            KV_STATE_EVENT_TOPIC_V2,
            KV_ZMQ_RCVHWM,
            None,
            ContinuityMode::Disabled,
            true,
            cancel,
            handler,
            |observation: FanInObservation| {
                if observation.event == FanInEvent::SourceStarted {
                    tracing::debug!(
                        publisher_id = observation.publisher_id,
                        "Consuming KV state-agent publisher"
                    );
                }
            },
        )
        .await;
    }

    let mut events = EventSubscriber::for_endpoint_id_with_transport(
        component.drt(),
        endpoint,
        KV_STATE_EVENT_TOPIC_V2,
        transport_kind,
    )
    .await?
    .typed::<Vec<RouterEvent>>();
    let endpoint = endpoint.clone();
    Ok(tokio::spawn(async move {
        loop {
            let event = tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                event = events.next() => event,
            };
            match event {
                Some(Ok((envelope, batch))) => {
                    live_ingest.apply(envelope.publisher_id, batch).await
                }
                Some(Err(error)) => {
                    tracing::warn!(%error, %endpoint, "Failed to decode V2 KV state event batch")
                }
                None => return,
            }
        }
    }))
}

trait Advertisement: serde::de::DeserializeOwned + Clone + PartialEq {
    fn owner(&self) -> CacheOwnerId;
}

impl Advertisement for KvStateSourceAdvertisement {
    fn owner(&self) -> CacheOwnerId {
        self.cache_owner_id
    }
}

impl Advertisement for KvStateAttachmentAdvertisement {
    fn owner(&self) -> CacheOwnerId {
        self.cache_owner_id
    }
}

/// Applies one discovery event and returns the owner it changed.
fn update_advertisements<T: Advertisement>(
    event: Result<DiscoveryEvent>,
    endpoint: &EndpointId,
    expected_topic: &str,
    values: &mut HashMap<u64, T>,
) -> Result<Option<CacheOwnerId>> {
    let event = event?;
    let expected_scope = EventScope::Endpoint {
        endpoint: endpoint.clone(),
    };
    match event {
        DiscoveryEvent::Added(DiscoveryInstance::EventSource {
            scope,
            topic,
            publisher_id,
            metadata,
        }) => {
            if scope != expected_scope || topic != expected_topic {
                return Ok(None);
            }
            let value: T = serde_json::from_value(metadata)?;
            if values
                .get(&publisher_id)
                .is_some_and(|previous| previous != &value)
            {
                anyhow::bail!("discovery identity changed its immutable advertisement");
            }
            let owner = value.owner();
            values.insert(publisher_id, value);
            Ok(Some(owner))
        }
        DiscoveryEvent::Removed(DiscoveryInstanceId::EventSource(EventSourceInstanceId {
            scope,
            topic,
            publisher_id,
        })) => {
            if scope == expected_scope && topic == expected_topic {
                Ok(values.remove(&publisher_id).map(|value| value.owner()))
            } else {
                Ok(None)
            }
        }
        DiscoveryEvent::Added(_)
        | DiscoveryEvent::Removed(_)
        | DiscoveryEvent::ModelTaintsUpdated(_)
        | DiscoveryEvent::Resync(_) => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
async fn schedule_state_sources(
    indexer: &Indexer,
    transport: Arc<RuntimeWorkerQueryTransport>,
    status_permits: &Arc<Semaphore>,
    membership: KvSourceMembershipView,
    endpoint: &EndpointId,
    expected_block_size: u32,
    sources: &HashMap<u64, KvStateSourceAdvertisement>,
    attachments: &HashMap<u64, KvStateAttachmentAdvertisement>,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
    bindings: &IngestBindings,
    recovery_lane: &Arc<RecoveryLane<(CacheOwnerId, u64)>>,
    recovery_tx: &mpsc::UnboundedSender<OwnerRecoveryResult>,
    recognized_tx: &watch::Sender<HashSet<WorkerWithDpRank>>,
    fence: &SharedFence,
    owners: &HashSet<CacheOwnerId>,
    schedule_generation: u64,
) -> Result<()> {
    let revisions: HashMap<_, _> = {
        let fence = fence.lock().await;
        owners
            .iter()
            .map(|owner| (*owner, fence.revision(owner)))
            .collect()
    };
    let live_workers: HashSet<_> = membership.sources.keys().copied().collect();
    let mut source_by_owner: HashMap<CacheOwnerId, Vec<&KvStateSourceAdvertisement>> =
        HashMap::new();
    for (discovery_id, source) in sources {
        let expected_ingress_protocol =
            crate::discovery::kv_state_agent::KvStateIngressProtocol::VllmResidencyV1;
        let expected_domain = source.cache_owner_id.pool().indexer_domain();
        if source.publisher_id != *discovery_id
            || source.protocol_version != KvStateProtocolVersion::V2
            || source.event_topic != KV_STATE_EVENT_TOPIC_V2
            || source.kv_state_endpoint != *endpoint
            || source.kv_block_size != expected_block_size
            || source.ingress_protocol != expected_ingress_protocol
            || expected_domain != source.indexer_domain_id
        {
            tracing::warn!(
                discovery_id = *discovery_id,
                advertised_publisher_id = source.publisher_id,
                advertised_protocol = ?source.protocol_version,
                expected_protocol = ?KvStateProtocolVersion::V2,
                advertised_event_topic = %source.event_topic,
                expected_event_topic = KV_STATE_EVENT_TOPIC_V2,
                advertised_endpoint = ?source.kv_state_endpoint,
                expected_endpoint = ?endpoint,
                advertised_block_size = source.kv_block_size,
                expected_block_size,
                advertised_ingress_protocol = ?source.ingress_protocol,
                expected_ingress_protocol = ?expected_ingress_protocol,
                advertised_domain = %source.indexer_domain_id,
                expected_domain = %expected_domain,
                owner = %source.cache_owner_id,
                "Ignoring incompatible V2 KV state source advertisement"
            );
            continue;
        }
        source_by_owner
            .entry(source.cache_owner_id)
            .or_default()
            .push(source);
    }
    let mut attachment_by_owner: HashMap<CacheOwnerId, Vec<&KvStateAttachmentAdvertisement>> =
        HashMap::new();
    for (discovery_id, attachment) in attachments {
        if attachment_record_id(attachment.publisher_id, attachment.attachment_generation)
            != *discovery_id
            || attachment.protocol_version != KvStateProtocolVersion::V2
        {
            continue;
        }
        attachment_by_owner
            .entry(attachment.cache_owner_id)
            .or_default()
            .push(attachment);
    }

    // Publish source-mode selection before applying any replacement reset or
    // snapshot. The legacy event client consults this watch at admission, so a
    // late legacy event cannot repopulate Worker ownership during activation.
    let mut provisional_recognized = source_mode_suppressed_workers(&membership);
    for (owner, matching_sources) in &source_by_owner {
        let [source] = matching_sources.as_slice() else {
            if matching_sources.len() > 1
                && let Some(worker) = runtime
                    .get(owner)
                    .and_then(|state| state.last_worker)
                    .filter(|worker| live_workers.contains(worker))
            {
                provisional_recognized.insert(worker);
            }
            continue;
        };
        let matching_attachments = attachment_by_owner
            .get(owner)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if let [attachment] = matching_attachments
            && attachment.publisher_id == source.publisher_id
            && attachment.protocol_version == source.protocol_version
            && attachment.worker.dp_rank == source.global_dp_rank
            && live_workers.contains(&attachment.worker)
        {
            provisional_recognized.insert(attachment.worker);
        } else if let Some(worker) = runtime
            .get(owner)
            .and_then(|state| state.last_worker)
            .filter(|worker| live_workers.contains(worker))
        {
            provisional_recognized.insert(worker);
        }
    }
    // Attachment presence is itself an explicit state-agent mode claim. This
    // closes the source/attachment discovery ordering window without treating
    // the attachment as sufficient for routing readiness.
    for matching_attachments in attachment_by_owner.values() {
        for attachment in matching_attachments {
            if live_workers.contains(&attachment.worker) {
                provisional_recognized.insert(attachment.worker);
            }
        }
    }
    recognized_tx.send_replace(provisional_recognized.clone());

    // Only the owners whose sources, attachments, or workers changed are planned
    // again; every other owner keeps its readiness and live ingest.
    let known_owners: Vec<_> = owners
        .iter()
        .copied()
        .filter(|owner| {
            runtime.contains_key(owner)
                || source_by_owner.contains_key(owner)
                || attachment_by_owner.contains_key(owner)
        })
        .collect();
    for owner in &known_owners {
        if let Some(state) = runtime.get_mut(owner) {
            state.ready = false;
            state.attached_worker = None;
            state.hint_ready = false;
        }
    }
    // Reconciliation performs asynchronous clears and recovery. Withdraw the
    // affected owners before any fallible work so an early error cannot leave a
    // stale CacheOwner mapping routable.
    fence.lock().await.commit(indexer, runtime);
    for owner in known_owners {
        let revision = revisions[&owner];
        let matching_sources = source_by_owner
            .get(&owner)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if matching_sources.len() > 1 {
            if let Some(state) = runtime.get_mut(&owner) {
                // No overlapping source incarnation is authoritative. Fence
                // live traffic until one exact source wins and recovers.
                if let Some(ingest) = state.ingest.take() {
                    retire_ingest(bindings, &ingest).await;
                }
                state.publisher_id = None;
                state.attachment_generation = None;
                state.ready = false;
                state.attached_worker = None;
                state.hint_ready = false;
                state.revision = revision;
            }
            tracing::warn!(%owner, "Ambiguous KV state-source incarnations; retaining exact ownership unprojected");
            continue;
        }
        let Some(source) = matching_sources.first().copied() else {
            if let Some(previous) = runtime.remove(&owner) {
                if let Some(ingest) = &previous.ingest {
                    retire_ingest(bindings, ingest).await;
                }
                if let Some(worker) = previous.last_worker {
                    clear_worker(indexer, worker, owner).await?;
                }
                clear_cache_owner(indexer, owner, 0).await?;
            }
            if let Some(candidates) = attachment_by_owner.get(&owner) {
                for attachment in candidates {
                    if live_workers.contains(&attachment.worker) {
                        clear_worker(indexer, attachment.worker, owner).await?;
                    }
                }
            }
            continue;
        };
        let matching = attachment_by_owner
            .get(&owner)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let attachment = match matching {
            [attachment]
                if attachment.publisher_id == source.publisher_id
                    && attachment.protocol_version == source.protocol_version
                    && attachment.worker.dp_rank == source.global_dp_rank
                    && attachment.recovery_control_target == source.recovery_control_target
                    && attachment.ingress_protocol == source.ingress_protocol =>
            {
                Some(*attachment)
            }
            _ => None,
        };

        let previous = runtime.get(&owner).cloned();
        let expected_generation = attachment.map(|value| value.attachment_generation);
        let last_worker = attachment
            .map(|value| value.worker)
            .or_else(|| previous.as_ref().and_then(|value| value.last_worker));
        let recognized_worker = last_worker.filter(|worker| live_workers.contains(worker));
        let attachment_changed = previous.as_ref().is_none_or(|value| {
            value.publisher_id != Some(source.publisher_id)
                || value.attachment_generation != expected_generation
        });
        // Fence the previous live ingest before clearing anything: a batch it
        // applied after a clear would survive the recovery snapshot.
        let mut reused_ingest = None;
        if let Some(previous) = previous.as_ref()
            && let Some(ingest) = previous.ingest.as_ref()
        {
            if previous.publisher_id == Some(source.publisher_id) {
                let (cursor, recovery_required) = buffer_live_ingest(
                    ingest,
                    expected_generation,
                    recognized_worker,
                    attachment_changed,
                    schedule_generation,
                )
                .await;
                reused_ingest = Some((ingest.clone(), cursor, recovery_required));
            } else {
                retire_ingest(bindings, ingest).await;
            }
        }

        if attachment.is_none() {
            for candidate in matching {
                if live_workers.contains(&candidate.worker) {
                    clear_worker(indexer, candidate.worker, owner).await?;
                }
            }
        }
        if attachment_changed
            && let Some(worker) = previous.as_ref().and_then(|value| value.last_worker)
        {
            clear_worker(indexer, worker, owner).await?;
        }
        if let Some(worker) = last_worker
            && !live_workers.contains(&worker)
            && previous
                .as_ref()
                .is_some_and(|value| value.last_worker == Some(worker))
        {
            clear_worker(indexer, worker, owner).await?;
        }
        if previous.is_none()
            && let Some(worker) = recognized_worker
        {
            clear_worker(indexer, worker, owner).await?;
        }

        let expected = KvStateAgentIdentity {
            cache_owner_id: owner,
            publisher_id: source.publisher_id,
            protocol_version: source.protocol_version,
        };
        let (ingest, previous_cursor, recovery_required) = match reused_ingest {
            Some(reused) => reused,
            None => {
                // Buffer live events from the first batch until recovery replays them.
                let ingest = Arc::new(Mutex::new(OwnerIngest {
                    owner,
                    publisher_id: source.publisher_id,
                    recovered_cursor: 0,
                    attachment_generation: expected_generation,
                    last_worker: recognized_worker,
                    recovery_required: attachment_changed,
                    tail: Some(OwnerRecoveryTail {
                        attachment_generation: expected_generation,
                        schedule_generation,
                        rank: RankState::default(),
                    }),
                    retired: false,
                }));
                bindings.insert(source.publisher_id, ingest.clone());
                (ingest, 0, attachment_changed)
            }
        };
        runtime.insert(
            owner,
            OwnerRuntime {
                publisher_id: Some(source.publisher_id),
                ingest: Some(ingest),
                attachment_generation: expected_generation,
                last_worker: recognized_worker,
                ready: false,
                attached_worker: attachment
                    .map(|value| value.worker)
                    .filter(|worker| live_workers.contains(worker)),
                router_hint_source: source.router_hint_source.clone(),
                hint_ready: false,
                revision,
            },
        );
        launch_initial_recovery(
            recovery_lane,
            recovery_tx,
            transport.clone(),
            status_permits.clone(),
            OwnerRecoveryPlan {
                owner,
                source: source.clone(),
                attachment: attachment.cloned(),
                expected,
                recognized_worker,
                previous_publisher_id: previous.and_then(|value| value.publisher_id),
                previous_cursor,
                recovery_required,
                revision,
                schedule_generation,
            },
        );
    }

    let mut recognized = provisional_recognized;
    recognized.extend(runtime.values().filter_map(|value| value.last_worker));
    recognized_tx.send_replace(recognized);
    fence.lock().await.commit(indexer, runtime);
    Ok(())
}

impl OwnerRecoveryResult {
    fn owner(&self) -> CacheOwnerId {
        match self {
            Self::Initial { plan, .. } | Self::PostStatus { plan, .. } => plan.owner,
        }
    }

    fn publisher_id(&self) -> u64 {
        match self {
            Self::Initial { plan, .. } | Self::PostStatus { plan, .. } => plan.source.publisher_id,
        }
    }

    fn schedule_generation(&self) -> u64 {
        match self {
            Self::Initial { plan, .. } | Self::PostStatus { plan, .. } => plan.schedule_generation,
        }
    }

    fn retry_fence(&self) -> RecoveryRetryFence {
        let plan = match self {
            Self::Initial { plan, .. } | Self::PostStatus { plan, .. } => plan,
        };
        RecoveryRetryFence {
            source: plan.source.clone(),
            attachment: plan.attachment.clone(),
            revision: plan.revision,
            activation_expected: plan.attachment.is_some() && plan.recognized_worker.is_some(),
        }
    }
}

fn launch_initial_recovery(
    lane: &Arc<RecoveryLane<(CacheOwnerId, u64)>>,
    tx: &mpsc::UnboundedSender<OwnerRecoveryResult>,
    transport: Arc<RuntimeWorkerQueryTransport>,
    status_permits: Arc<Semaphore>,
    plan: OwnerRecoveryPlan,
) {
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let task_tx = tx.clone();
    let semaphore = lane.semaphore();
    let owner = plan.owner;
    let schedule_generation = plan.schedule_generation;
    let handle = tokio::spawn(async move {
        let Ok(status_permit) = status_permits.acquire_owned().await else {
            return;
        };
        let status_query = transport.query_status(
            plan.attachment
                .as_ref()
                .map_or(0, |value| value.worker.worker_id),
            plan.source.global_dp_rank,
            plan.source.recovery_control_target.clone(),
            plan.expected.clone(),
            plan.attachment
                .as_ref()
                .map(|value| value.attachment_generation),
            CONTROL_TIMEOUT,
        );
        let status = tokio::select! {
            biased;
            _ = task_cancel.cancelled() => return,
            result = status_query => result,
        };
        drop(status_permit);
        let needs_recovery = status.as_ref().is_ok_and(|status| {
            plan.recovery_required
                || plan.previous_cursor
                    < plan
                        .attachment
                        .as_ref()
                        .map_or(status.outbound_cursor, |value| {
                            value.ready_at_outbound_cursor
                        })
        });
        let recovery = if needs_recovery {
            let permit = tokio::select! {
                biased;
                _ = task_cancel.cancelled() => return,
                permit = semaphore.acquire_owned() => match permit {
                    Ok(permit) => permit,
                    Err(_) => return,
                },
            };
            let query = transport.query_state_agent_recovery(
                plan.attachment
                    .as_ref()
                    .map_or(0, |value| value.worker.worker_id),
                plan.source.global_dp_rank,
                plan.source.recovery_control_target.clone(),
                plan.expected.clone(),
                plan.attachment
                    .as_ref()
                    .map(|value| value.attachment_generation),
                RECOVERY_TIMEOUT,
            );
            let result = tokio::select! {
                biased;
                _ = task_cancel.cancelled() => return,
                result = query => result,
            };
            drop(permit);
            Some(result)
        } else {
            None
        };
        let _ = task_tx.send(OwnerRecoveryResult::Initial {
            plan,
            status,
            recovery,
        });
    });
    lane.insert((owner, schedule_generation), cancel, handle);
}

fn launch_post_status(
    lane: &Arc<RecoveryLane<(CacheOwnerId, u64)>>,
    tx: &mpsc::UnboundedSender<OwnerRecoveryResult>,
    transport: Arc<RuntimeWorkerQueryTransport>,
    status_permits: Arc<Semaphore>,
    plan: OwnerRecoveryPlan,
    recovered_cursor: u64,
) {
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let task_tx = tx.clone();
    let owner = plan.owner;
    let schedule_generation = plan.schedule_generation;
    let handle = tokio::spawn(async move {
        let Ok(_status_permit) = status_permits.acquire_owned().await else {
            return;
        };
        let query = transport.query_status(
            plan.attachment
                .as_ref()
                .map_or(0, |value| value.worker.worker_id),
            plan.source.global_dp_rank,
            plan.source.recovery_control_target.clone(),
            plan.expected.clone(),
            plan.attachment
                .as_ref()
                .map(|value| value.attachment_generation),
            CONTROL_TIMEOUT,
        );
        let status = tokio::select! {
            biased;
            _ = task_cancel.cancelled() => return,
            result = query => result,
        };
        let _ = task_tx.send(OwnerRecoveryResult::PostStatus {
            plan,
            recovered_cursor,
            status,
        });
    });
    lane.insert((owner, schedule_generation), cancel, handle);
}

#[allow(clippy::too_many_arguments)]
async fn finish_owner_recovery(
    result: OwnerRecoveryResult,
    indexer: &Indexer,
    transport: Option<&Arc<RuntimeWorkerQueryTransport>>,
    status_permits: &Arc<Semaphore>,
    membership: &KvSourceMembershipWatch,
    sources: &HashMap<u64, KvStateSourceAdvertisement>,
    attachments: &HashMap<u64, KvStateAttachmentAdvertisement>,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
    lane: &Arc<RecoveryLane<(CacheOwnerId, u64)>>,
    tx: &mpsc::UnboundedSender<OwnerRecoveryResult>,
    fence: &SharedFence,
) -> Result<()> {
    let plan = match &result {
        OwnerRecoveryResult::Initial { plan, .. }
        | OwnerRecoveryResult::PostStatus { plan, .. } => plan,
    };
    let revision = fence.lock().await.revision(&plan.owner);
    if !owner_recovery_is_current(plan, membership, sources, attachments, revision) {
        return Ok(());
    }
    let Some(ingest) = runtime
        .get(&plan.owner)
        .filter(|state| state.publisher_id == Some(plan.source.publisher_id))
        .and_then(|state| state.ingest.clone())
    else {
        return Ok(());
    };
    if ingest
        .lock()
        .await
        .tail
        .as_ref()
        .is_none_or(|tail| tail.schedule_generation != plan.schedule_generation)
    {
        return Ok(());
    }
    match result {
        OwnerRecoveryResult::Initial {
            plan,
            status,
            recovery,
        } => {
            let status = match status {
                Ok(status) => status,
                Err(error) => {
                    fail_owner_recovery(plan.owner, plan.source.publisher_id, runtime).await;
                    tracing::warn!(owner = %plan.owner, %error, "KV state-agent status handshake failed");
                    return Ok(());
                }
            };
            let recovered_cursor = if let Some(recovery) = recovery {
                let response = match recovery {
                    Ok(response) => response,
                    Err(error) => {
                        fail_owner_recovery(plan.owner, plan.source.publisher_id, runtime).await;
                        tracing::warn!(owner = %plan.owner, %error, "KV state-agent recovery query failed");
                        return Ok(());
                    }
                };
                if plan.recovery_required
                    && plan.previous_publisher_id != Some(plan.source.publisher_id)
                {
                    clear_cache_owner(indexer, plan.owner, 0).await?;
                }
                if let Some(worker) = plan.recognized_worker {
                    clear_worker(indexer, worker, plan.owner).await?;
                }
                let recovered_cursor = apply_recovery_response(
                    indexer,
                    plan.owner,
                    plan.recognized_worker,
                    &plan.expected,
                    plan.attachment
                        .as_ref()
                        .map(|value| value.attachment_generation),
                    response,
                )
                .await?;
                // Keep buffering until the post-recovery status confirms readiness.
                apply_recovery_tail(indexer, &plan, recovered_cursor, &ingest, false).await?;
                let Some(transport) = transport else {
                    anyhow::bail!("state-agent recovery transport disappeared");
                };
                launch_post_status(
                    lane,
                    tx,
                    (*transport).clone(),
                    status_permits.clone(),
                    plan,
                    recovered_cursor,
                );
                return Ok(());
            } else {
                apply_recovery_tail(indexer, &plan, plan.previous_cursor, &ingest, true).await?
            };
            finish_owner_readiness(&plan, status, recovered_cursor, membership, runtime);
        }
        OwnerRecoveryResult::PostStatus {
            plan,
            recovered_cursor,
            status,
        } => {
            let status = match status {
                Ok(status) => status,
                Err(error) => {
                    fail_owner_recovery(plan.owner, plan.source.publisher_id, runtime).await;
                    tracing::warn!(owner = %plan.owner, %error, "KV state-agent post-recovery status check failed");
                    return Ok(());
                }
            };
            let recovered_cursor =
                apply_recovery_tail(indexer, &plan, recovered_cursor, &ingest, true).await?;
            finish_owner_readiness(&plan, status, recovered_cursor, membership, runtime);
        }
    }
    fence.lock().await.commit(indexer, runtime);
    Ok(())
}

fn owner_recovery_is_current(
    plan: &OwnerRecoveryPlan,
    membership: &KvSourceMembershipWatch,
    sources: &HashMap<u64, KvStateSourceAdvertisement>,
    attachments: &HashMap<u64, KvStateAttachmentAdvertisement>,
    revision: OwnerRevision,
) -> bool {
    if revision != plan.revision
        || sources.get(&plan.source.publisher_id) != Some(&plan.source)
        || plan
            .recognized_worker
            .is_some_and(|worker| !membership.borrow().sources.contains_key(&worker))
    {
        return false;
    }
    match &plan.attachment {
        Some(attachment) => {
            attachments.get(&attachment_record_id(
                attachment.publisher_id,
                attachment.attachment_generation,
            )) == Some(attachment)
        }
        None => true,
    }
}

/// Replays the buffered live tail after `recovered_cursor`. With `finish`, live
/// events apply directly afterwards; the drain and the switch happen under one
/// lock, so no event lands between them.
async fn apply_recovery_tail(
    indexer: &Indexer,
    plan: &OwnerRecoveryPlan,
    recovered_cursor: u64,
    ingest: &SharedIngest,
    finish: bool,
) -> Result<u64> {
    let attachment_generation = plan
        .attachment
        .as_ref()
        .map(|value| value.attachment_generation);
    let mut ingest = ingest.lock().await;
    let events = ingest
        .tail
        .as_mut()
        .filter(|tail| tail.attachment_generation == attachment_generation)
        .map_or_else(Vec::new, |tail| {
            tail.rank.drain_advisory_tail_after(recovered_cursor)
        });
    ingest.recovered_cursor = recovered_cursor;
    ingest.recovery_required = false;
    apply_events_for_owner(indexer, events, &mut ingest).await;
    if ingest.recovery_required {
        anyhow::bail!("buffered state-agent recovery tail was rejected");
    }
    if finish {
        ingest.tail = None;
    }
    Ok(ingest.recovered_cursor)
}

fn finish_owner_readiness(
    plan: &OwnerRecoveryPlan,
    status: KvStateAgentStatus,
    recovered_cursor: u64,
    membership: &KvSourceMembershipWatch,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
) {
    let membership = membership.borrow();
    let live_workers = &membership.sources;
    let identity_matches = status.identity == plan.expected;
    let hint_ready = identity_matches
        && status.cache_owner_ready
        && recovered_cursor >= status.outbound_cursor
        && plan.source.router_hint_source.is_some();
    let ready = plan.attachment.as_ref().is_some_and(|attachment| {
        live_workers.contains_key(&attachment.worker)
            && status.cache_owner_ready
            && identity_matches
            && status.attachment.as_ref().is_some_and(|status_attachment| {
                status_attachment.ready
                    && status_attachment.generation == attachment.attachment_generation
                    && status_attachment.worker == attachment.worker
                    && status_attachment.ready_at_outbound_cursor
                        == attachment.ready_at_outbound_cursor
            })
            && recovered_cursor >= attachment.ready_at_outbound_cursor
    });
    let Some(state) = runtime
        .get_mut(&plan.owner)
        .filter(|state| state.publisher_id == Some(plan.source.publisher_id))
    else {
        return;
    };
    state.attachment_generation = plan
        .attachment
        .as_ref()
        .map(|value| value.attachment_generation);
    state.last_worker = plan.recognized_worker;
    state.ready = ready;
    state.attached_worker = plan
        .attachment
        .as_ref()
        .map(|attachment| attachment.worker)
        .filter(|worker| live_workers.contains_key(worker));
    state.router_hint_source = plan.source.router_hint_source.clone();
    state.hint_ready = hint_ready;
}

async fn fail_owner_recovery(
    owner: CacheOwnerId,
    publisher_id: u64,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
) {
    let Some(state) = runtime
        .get_mut(&owner)
        .filter(|state| state.publisher_id == Some(publisher_id))
    else {
        return;
    };
    state.ready = false;
    state.attached_worker = None;
    state.hint_ready = false;
    if let Some(ingest) = &state.ingest {
        let mut ingest = ingest.lock().await;
        ingest.tail = None;
        ingest.recovery_required = true;
    }
}

/// Buffers the live events of a reused ingest until recovery replays them, and
/// returns its recovered cursor and whether recovery must rebuild its state.
async fn buffer_live_ingest(
    ingest: &SharedIngest,
    attachment_generation: Option<u64>,
    last_worker: Option<WorkerWithDpRank>,
    attachment_changed: bool,
    schedule_generation: u64,
) -> (u64, bool) {
    let mut ingest = ingest.lock().await;
    let recovery_required = attachment_changed || ingest.recovery_required;
    ingest.attachment_generation = attachment_generation;
    ingest.last_worker = last_worker;
    ingest.recovery_required = recovery_required;
    match ingest.tail.as_mut() {
        Some(tail) if tail.attachment_generation == attachment_generation => {
            tail.schedule_generation = schedule_generation;
        }
        _ => {
            ingest.tail = Some(OwnerRecoveryTail {
                attachment_generation,
                schedule_generation,
                rank: RankState::default(),
            });
        }
    }
    (ingest.recovered_cursor, recovery_required)
}

async fn retire_ingest(bindings: &IngestBindings, ingest: &SharedIngest) {
    let mut guard = ingest.lock().await;
    guard.retired = true;
    guard.tail = None;
    bindings.remove_if(&guard.publisher_id, |_, bound| Arc::ptr_eq(bound, ingest));
}

async fn apply_recovery_response(
    indexer: &Indexer,
    owner: CacheOwnerId,
    worker: Option<WorkerWithDpRank>,
    expected: &KvStateAgentIdentity,
    expected_generation: Option<u64>,
    response: WorkerKvQueryResponse,
) -> Result<u64> {
    let WorkerKvQueryResponse::StateAgentRecovery { response, receipt } = response else {
        anyhow::bail!("state-agent recovery returned an unexpected response");
    };
    if receipt.identity != *expected || receipt.attachment_generation != expected_generation {
        anyhow::bail!("state-agent recovery receipt does not match the selected incarnation");
    }
    match *response {
        WorkerKvQueryResponse::TreeDump {
            events,
            last_event_id,
            reset_scope,
        } => {
            validate_recovery_events(&events, owner, worker)?;
            match reset_scope {
                ResetScope::All => {
                    if let Some(worker) = worker {
                        clear_worker(indexer, worker, owner).await?;
                    }
                    clear_cache_owner(indexer, owner, last_event_id).await?;
                }
                ResetScope::Domain(ResidencyDomain::CacheOwner) => {
                    clear_cache_owner(indexer, owner, last_event_id).await?;
                }
                ResetScope::Domain(ResidencyDomain::Worker) => {
                    if let Some(worker) = worker {
                        clear_worker(indexer, worker, owner).await?;
                    }
                }
            }
            for event in events {
                indexer.try_apply_event(event).await?;
            }
            Ok(receipt.recovered_through_cursor.max(last_event_id))
        }
        WorkerKvQueryResponse::Events {
            events,
            last_event_id,
        } => {
            validate_recovery_events(&events, owner, worker)?;
            for event in events {
                indexer.try_apply_event(event).await?;
            }
            Ok(receipt.recovered_through_cursor.max(last_event_id))
        }
        WorkerKvQueryResponse::TreeDumpFailed { message, .. } => {
            anyhow::bail!("state-agent snapshot failed: {message}")
        }
        WorkerKvQueryResponse::Error(message) => {
            anyhow::bail!("state-agent recovery failed: {message}")
        }
        _ => anyhow::bail!("state-agent recovery returned no applicable state"),
    }
}

fn validate_recovery_events(
    events: &[RouterEvent],
    owner: CacheOwnerId,
    worker: Option<WorkerWithDpRank>,
) -> Result<()> {
    for event in events {
        let valid = event
            .resolved_residency_domain()
            .is_ok_and(|domain| match domain {
                ResidencyDomain::CacheOwner => event.state_source == Some(owner),
                ResidencyDomain::Worker => worker.is_some_and(|worker| {
                    worker.worker_id == event.worker_id && worker.dp_rank == event.event.dp_rank
                }),
            });
        if !valid {
            anyhow::bail!(
                "state-agent recovery contains ownership outside the selected source or attachment"
            );
        }
    }
    Ok(())
}

async fn apply_events_for_owner(
    indexer: &Indexer,
    events: Vec<RouterEvent>,
    state: &mut OwnerIngest,
) {
    let owner = state.owner;
    let publisher_id = state.publisher_id;
    for event in events {
        if state.recovery_required {
            break;
        }
        let event_id = event.event.event_id;
        if event_id <= state.recovered_cursor {
            continue;
        }
        if let Some(expected) = state.recovered_cursor.checked_add(1)
            && event_id > expected
        {
            tracing::warn!(
                publisher_id,
                expected,
                actual = event_id,
                "KV state-agent event gap; continuing advisory ingestion"
            );
        }
        let domain = event.resolved_residency_domain();
        if matches!(domain.as_ref(), Ok(ResidencyDomain::Worker))
            && state.attachment_generation.is_none()
        {
            tracing::warn!(
                publisher_id,
                event_id,
                "Ignoring stale Worker event while state source is detached"
            );
            state.recovered_cursor = event_id;
            continue;
        }
        let valid = domain.is_ok_and(|domain| match domain {
            ResidencyDomain::CacheOwner => event.state_source == Some(owner),
            ResidencyDomain::Worker => state.last_worker.is_some_and(|worker| {
                worker.worker_id == event.worker_id && worker.dp_rank == event.event.dp_rank
            }),
        });
        if !valid {
            tracing::warn!(
                publisher_id,
                event_id,
                "Ignoring incorrectly attributed KV state event"
            );
            state.recovery_required = true;
            break;
        }
        let is_clear = matches!(event.event.data, KvCacheEventData::Cleared);
        if let Err(error) = indexer.try_apply_event(event).await {
            tracing::warn!(publisher_id, event_id, %error, "Failed to apply advisory KV state event");
            if is_clear {
                state.recovery_required = true;
                break;
            }
        }
        state.recovered_cursor = event_id;
    }
}

async fn clear_worker(
    indexer: &Indexer,
    worker: WorkerWithDpRank,
    owner: CacheOwnerId,
) -> Result<()> {
    indexer
        .try_apply_event(
            RouterEvent::with_residency_domain(
                worker.worker_id,
                KvCacheEvent {
                    event_id: 0,
                    data: KvCacheEventData::Cleared,
                    dp_rank: worker.dp_rank,
                },
                StorageTier::Device,
                ResidencyDomain::Worker,
            )
            .with_state_source(owner),
        )
        .await?;
    Ok(())
}

async fn clear_cache_owner(indexer: &Indexer, owner: CacheOwnerId, event_id: u64) -> Result<()> {
    indexer
        .try_apply_event(RouterEvent::with_cache_owner(
            0,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
            StorageTier::Device,
            owner,
        ))
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
        StableDpSlotId,
    };
    use dynamo_kv_router::indexer::{
        KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvStateAttachmentStatus,
        KvStateRecoveryReceipt, LowerTierIndexers, LowerTierQueryOptions, MatchDetails,
        query_lower_tiers_with_options,
    };
    use dynamo_kv_router::kv_hints::KvTransferCandidateSource;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData,
        LocalBlockHash, ResidencyOwner, RouterHintSourceMetadata, WorkerWithDpRank,
    };
    use dynamo_runtime::{component::TransportType, protocols::EndpointId};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::discovery::{
        KvEventSource, KvSourceMembershipView, KvStateEndpointResolution,
        kv_state_agent::{
            KvStateIngressProtocol, KvStateProjectionResolution, KvStateUnknownReason,
            resolve_kv_state_projection,
        },
    };

    fn owner(slot: u8) -> CacheOwnerId {
        CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            StableDpSlotId::new([slot; 16], IdentitySource::Explicit),
        )
    }

    fn test_indexer() -> (KvIndexer, Indexer) {
        let primary = KvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
        );
        (
            primary.clone(),
            Indexer::Single {
                primary,
                lower_tier: LowerTierIndexers::new(1, 4),
                approx: None,
                primary_records_routing_decisions: false,
                session_updates: None,
            },
        )
    }

    fn instance(instance_id: u64) -> dynamo_runtime::component::Instance {
        dynamo_runtime::component::Instance {
            component: "worker".to_string(),
            endpoint: "state-agent".to_string(),
            namespace: "ns".to_string(),
            instance_id,
            transport: TransportType::Tcp("tcp://127.0.0.1:1234".to_string()),
            device_type: None,
            request_plane_codec: None,
        }
    }

    fn source_advertisement(owner: CacheOwnerId) -> KvStateSourceAdvertisement {
        KvStateSourceAdvertisement {
            cache_owner_id: owner,
            global_dp_rank: 3,
            kv_state_endpoint: "ns.worker.generate".into(),
            indexer_domain_id: owner.pool().indexer_domain(),
            kv_block_size: 4,
            ingress_protocol: KvStateIngressProtocol::VllmResidencyV1,
            publisher_id: 41,
            protocol_version: KvStateProtocolVersion::V2,
            event_topic: KV_STATE_EVENT_TOPIC_V2.to_string(),
            recovery_control_target: instance(17),
            router_hint_source: None,
        }
    }

    fn attachment_advertisement(
        owner: CacheOwnerId,
        generation: u64,
    ) -> KvStateAttachmentAdvertisement {
        KvStateAttachmentAdvertisement {
            cache_owner_id: owner,
            publisher_id: 41,
            protocol_version: KvStateProtocolVersion::V2,
            recovery_control_target: instance(17),
            attachment_generation: generation,
            producer_instance: instance(18),
            intent_incarnation: 71,
            worker: WorkerWithDpRank::new(17, 3),
            ingress_protocol: KvStateIngressProtocol::VllmResidencyV1,
            raw_zmq_endpoint: "tcp://worker:5557".to_string(),
            raw_topic: String::new(),
            ready_at_outbound_cursor: 9,
        }
    }

    fn bound_ingest(
        owner: CacheOwnerId,
        publisher_id: u64,
        attachment_generation: Option<u64>,
        last_worker: Option<WorkerWithDpRank>,
    ) -> SharedIngest {
        Arc::new(Mutex::new(OwnerIngest {
            owner,
            publisher_id,
            recovered_cursor: 0,
            attachment_generation,
            last_worker,
            recovery_required: false,
            tail: None,
            retired: false,
        }))
    }

    fn live_ingest(
        indexer: Indexer,
        ingests: &[&SharedIngest],
    ) -> (LiveIngest, mpsc::UnboundedReceiver<(CacheOwnerId, u64)>) {
        let bindings = IngestBindings::default();
        for ingest in ingests {
            let publisher_id = ingest.try_lock().unwrap().publisher_id;
            bindings.insert(publisher_id, (*ingest).clone());
        }
        let (withdrawn, withdrawn_rx) = mpsc::unbounded_channel();
        (
            LiveIngest {
                indexer,
                bindings,
                withdrawn,
            },
            withdrawn_rx,
        )
    }

    fn worker_store(owner: CacheOwnerId, worker: WorkerWithDpRank, event_id: u64) -> RouterEvent {
        RouterEvent::with_residency_domain(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(event_id),
                        tokens_hash: LocalBlockHash(event_id),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: worker.dp_rank,
            },
            StorageTier::Device,
            ResidencyDomain::Worker,
        )
        .with_state_source(owner)
    }

    #[tokio::test(start_paused = true)]
    async fn recovery_retry_runs_without_discovery_and_fences_replacement() {
        let owner = owner(4);
        let source = source_advertisement(owner);
        let attachment = attachment_advertisement(owner, 7);
        let sources = HashMap::from([(source.publisher_id, source.clone())]);
        let mut attachments = HashMap::from([(
            attachment_record_id(attachment.publisher_id, attachment.attachment_generation),
            attachment.clone(),
        )]);
        let revision = OwnerRevision { epoch: 0, owner: 3 };
        let fence = RecoveryRetryFence {
            source,
            attachment: Some(attachment),
            revision,
            activation_expected: true,
        };
        let mut retries = RecoveryRetries::default();

        retries.schedule(owner, fence.clone());
        tokio::time::advance(RECONCILE_RETRY_INITIAL_BACKOFF).await;
        wait_for_retry_deadline(retries.next_deadline()).await;
        let (all, due) = retries.take_due();
        assert!(!all);
        let [(due_owner, due)] = due.as_slice() else {
            panic!("the owner's retry should fire without discovery");
        };
        assert_eq!(*due_owner, owner);
        assert!(recovery_retry_is_current(
            due,
            &sources,
            &attachments,
            revision,
        ));
        // A newer observation of this owner supersedes the retry.
        assert!(!recovery_retry_is_current(
            due,
            &sources,
            &attachments,
            OwnerRevision { epoch: 0, owner: 4 },
        ));

        attachments.clear();
        let replacement = attachment_advertisement(owner, 8);
        attachments.insert(
            attachment_record_id(replacement.publisher_id, replacement.attachment_generation),
            replacement,
        );
        assert!(!recovery_retry_is_current(
            due,
            &sources,
            &attachments,
            revision,
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn withdrawn_owner_retries_recovery_without_discovery() {
        let owner = owner(4);
        let source = source_advertisement(owner);
        let attachment = attachment_advertisement(owner, 7);
        let sources = HashMap::from([(source.publisher_id, source)]);
        let attachments = HashMap::from([(
            attachment_record_id(attachment.publisher_id, attachment.attachment_generation),
            attachment.clone(),
        )]);
        let mut runtime = HashMap::from([(owner, ready_runtime(attachment.worker))]);
        let mut retries = RecoveryRetries::default();

        assert!(withdraw_owner(
            owner,
            41,
            &mut runtime,
            &sources,
            &attachments,
            &mut retries,
        ));
        assert!(!runtime[&owner].ready);

        tokio::time::advance(RECONCILE_RETRY_INITIAL_BACKOFF).await;
        let (_, due) = retries.take_due();
        let [(due_owner, due)] = due.as_slice() else {
            panic!("the withdrawn owner's recovery should be retried");
        };
        assert_eq!(*due_owner, owner);
        assert!(due.activation_expected);
        assert!(recovery_retry_is_current(
            due,
            &sources,
            &attachments,
            runtime[&owner].revision,
        ));
    }

    #[tokio::test]
    async fn attachment_recovers_pre_advertisement_tail_without_readability_toggle() {
        let owner = owner(4);
        let worker = WorkerWithDpRank::new(17, 3);
        let source = source_advertisement(owner);
        let attachment = attachment_advertisement(owner, 7);
        let identity = KvStateAgentIdentity {
            cache_owner_id: owner,
            publisher_id: source.publisher_id,
            protocol_version: source.protocol_version,
        };
        let plan = OwnerRecoveryPlan {
            owner,
            source,
            attachment: Some(attachment),
            expected: identity.clone(),
            recognized_worker: Some(worker),
            previous_publisher_id: None,
            previous_cursor: 0,
            recovery_required: true,
            revision: OwnerRevision::default(),
            schedule_generation: 1,
        };
        let mut rank = RankState::default();
        rank.buffer_recovery_tail(RouterEvent::with_cache_owner(
            worker.worker_id,
            KvCacheEvent {
                event_id: 10,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(10),
                        tokens_hash: LocalBlockHash(10),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: worker.dp_rank,
            },
            StorageTier::HostPinned,
            owner,
        ));
        let ingest = bound_ingest(owner, 41, Some(7), Some(worker));
        {
            let mut ingest = ingest.try_lock().unwrap();
            ingest.recovery_required = true;
            ingest.tail = Some(OwnerRecoveryTail {
                attachment_generation: Some(7),
                schedule_generation: 1,
                rank,
            });
        }

        let status = KvStateAgentStatus {
            identity: identity.clone(),
            attachment: Some(KvStateAttachmentStatus {
                generation: 7,
                worker,
                ready: true,
                ready_at_outbound_cursor: 9,
            }),
            cache_owner_ready: true,
            outbound_cursor: 10,
        };
        let live_workers = HashSet::from([worker]);
        let receipt_before_tail = KvStateRecoveryReceipt {
            identity: identity.clone(),
            attachment_generation: Some(7),
            recovered_through_cursor: 8,
        };
        assert_eq!(
            resolve_kv_state_projection(
                true,
                owner,
                std::slice::from_ref(&plan.source),
                std::slice::from_ref(plan.attachment.as_ref().unwrap()),
                &live_workers,
                Some(&status),
                Some(&receipt_before_tail),
            ),
            KvStateProjectionResolution::Unknown(KvStateUnknownReason::RecoveryBehindBarrier)
        );

        let recovered_cursor = apply_recovery_tail(&Indexer::None, &plan, 9, &ingest, false)
            .await
            .unwrap();
        assert_eq!(recovered_cursor, 10);
        assert_eq!(ingest.lock().await.recovered_cursor, 10);

        let receipt_after_tail = KvStateRecoveryReceipt {
            identity,
            attachment_generation: Some(7),
            recovered_through_cursor: recovered_cursor,
        };
        assert!(matches!(
            resolve_kv_state_projection(
                true,
                owner,
                std::slice::from_ref(&plan.source),
                std::slice::from_ref(plan.attachment.as_ref().unwrap()),
                &live_workers,
                Some(&status),
                Some(&receipt_after_tail),
            ),
            KvStateProjectionResolution::Ready { worker: ready, .. } if ready == worker
        ));
    }

    #[tokio::test]
    async fn detached_owner_remains_hint_source_without_scheduling_projection() {
        let owner = owner(4);
        let owner_key = ResidencyOwner::cache_owner(owner).compact_key();
        let worker = WorkerWithDpRank::new(17, 3);
        let primary = KvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
        );
        let lower_tiers = LowerTierIndexers::new(1, 4);
        let indexer = Indexer::Single {
            primary,
            lower_tier: lower_tiers.clone(),
            approx: None,
            primary_records_routing_decisions: false,
            session_updates: None,
        };
        lower_tiers
            .get_or_create(StorageTier::HostPinned)
            .apply_event_and_wait(RouterEvent::with_cache_owner(
                worker.worker_id,
                KvCacheEvent {
                    event_id: 1,
                    dp_rank: worker.dp_rank,
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: None,
                        start_position: None,
                        blocks: vec![
                            KvCacheStoredBlockData {
                                block_hash: ExternalSequenceBlockHash(101),
                                tokens_hash: LocalBlockHash(11),
                                mm_extra_info: None,
                            },
                            KvCacheStoredBlockData {
                                block_hash: ExternalSequenceBlockHash(102),
                                tokens_hash: LocalBlockHash(12),
                                mm_extra_info: None,
                            },
                        ],
                    }),
                },
                StorageTier::HostPinned,
                owner,
            ))
            .await
            .unwrap();
        let mut runtime = HashMap::from([(
            owner,
            OwnerRuntime {
                publisher_id: Some(41),
                ingest: None,
                attachment_generation: Some(7),
                last_worker: Some(worker),
                ready: true,
                attached_worker: Some(worker),
                router_hint_source: Some(RouterHintSourceMetadata {
                    source_control_endpoint: "tcp://persistent-owner:23280".to_string(),
                    worker_type: "prefill".to_string(),
                    topology_domains: Default::default(),
                }),
                hint_ready: true,
                revision: OwnerRevision::default(),
            },
        )]);
        let mut fence = ProjectionFence::default();
        let lookup = || {
            query_lower_tiers_with_options(
                &lower_tiers,
                &[LocalBlockHash(11), LocalBlockHash(12)],
                &MatchDetails::default(),
                LowerTierQueryOptions {
                    retain_kv_transfer_chain: true,
                },
            )
        };

        fence.commit(&indexer, &runtime);
        let attached = lookup();
        assert_eq!(
            attached[&StorageTier::HostPinned].hits.get(&worker),
            Some(&2)
        );

        let state = runtime.get_mut(&owner).unwrap();
        state.ready = false;
        state.attached_worker = None;
        state.attachment_generation = None;
        fence.commit(&indexer, &runtime);
        let detached = lookup();
        let details = &detached[&StorageTier::HostPinned];
        assert!(details.hits.is_empty());
        let candidates = details.kv_transfer_candidates.as_ref().unwrap();
        assert_eq!(
            candidates.owner_prefix_blocks,
            vec![(KvTransferCandidateSource::CacheOwner(owner_key), 2)]
        );
        assert_eq!(
            candidates
                .routing_snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.router_hint_source(owner_key))
                .and_then(|source| source.attached_worker),
            None
        );

        runtime.get_mut(&owner).unwrap().hint_ready = false;
        fence.commit(&indexer, &runtime);
        let cleared = lookup();
        assert!(
            cleared[&StorageTier::HostPinned]
                .kv_transfer_candidates
                .is_none()
        );
    }

    #[test]
    fn recognized_state_source_suppresses_legacy_without_reporting_missing() {
        let worker = WorkerWithDpRank::new(17, 3);
        let endpoint = EndpointId::from("ns.backend.generate");
        let source = KvEventSource {
            kv_state_endpoint: endpoint.clone(),
            worker,
            publisher_id: 41,
            recovery_target: None,
        };
        let view = KvSourceMembershipView {
            serving_endpoint: endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(endpoint),
            sources: HashMap::from([(worker, KvSourceStatus::ActiveLiveOnly(source))]),
            kv_event_publishing_enabled: HashMap::from([(worker.worker_id, Some(true))]),
            kv_event_source_mode: HashMap::new(),
            recovery_expected: HashMap::from([(worker, false)]),
        };
        let filtered = suppress_recognized(view, &HashSet::from([worker]));
        assert_eq!(filtered.status(&worker), Some(&KvSourceStatus::Suppressed));
    }

    #[test]
    fn explicit_source_mode_suppresses_legacy_before_state_source_is_ready() {
        let worker = WorkerWithDpRank::new(7, 3);
        let endpoint = EndpointId::from("ns.backend.generate");
        let mut view = KvSourceMembershipView {
            serving_endpoint: endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(endpoint),
            sources: HashMap::from([(worker, KvSourceStatus::Missing)]),
            kv_event_publishing_enabled: HashMap::new(),
            kv_event_source_mode: HashMap::from([(
                worker.worker_id,
                Some("state_agent_v2".to_string()),
            )]),
            recovery_expected: HashMap::new(),
        };
        let recognized = source_mode_suppressed_workers(&view);
        assert_eq!(recognized, HashSet::from([worker]));
        view.kv_event_source_mode
            .insert(worker.worker_id, Some("future_mode".to_string()));
        assert_eq!(source_mode_suppressed_workers(&view), recognized);
    }

    #[tokio::test]
    async fn recovery_rejects_ownership_from_another_state_source() {
        let selected = owner(4);
        let other = owner(5);
        let identity = KvStateAgentIdentity {
            cache_owner_id: selected,
            publisher_id: 41,
            protocol_version: KvStateProtocolVersion::V2,
        };
        let response = WorkerKvQueryResponse::StateAgentRecovery {
            response: Box::new(WorkerKvQueryResponse::Events {
                events: vec![RouterEvent::with_cache_owner(
                    0,
                    KvCacheEvent {
                        event_id: 1,
                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                            block_hashes: vec![ExternalSequenceBlockHash(7)],
                        }),
                        dp_rank: 0,
                    },
                    StorageTier::HostPinned,
                    other,
                )],
                last_event_id: 1,
            }),
            receipt: KvStateRecoveryReceipt {
                identity: identity.clone(),
                attachment_generation: None,
                recovered_through_cursor: 1,
            },
        };

        let error =
            apply_recovery_response(&Indexer::None, selected, None, &identity, None, response)
                .await
                .unwrap_err();
        assert!(error.to_string().contains("ownership outside"));
    }

    fn ready_runtime(worker: WorkerWithDpRank) -> OwnerRuntime {
        OwnerRuntime {
            publisher_id: Some(41),
            ingest: None,
            attachment_generation: Some(7),
            last_worker: Some(worker),
            ready: true,
            attached_worker: Some(worker),
            router_hint_source: None,
            hint_ready: false,
            revision: OwnerRevision::default(),
        }
    }

    fn published(fence: &ProjectionFence) -> HashSet<CacheOwnerId> {
        fence.published.keys().copied().collect()
    }

    fn event_source_added<T: serde::Serialize>(
        topic: &str,
        publisher_id: u64,
        advertisement: &T,
    ) -> Result<DiscoveryEvent> {
        Ok(DiscoveryEvent::Added(DiscoveryInstance::EventSource {
            scope: EventScope::Endpoint {
                endpoint: EndpointId::from("ns.worker.generate"),
            },
            topic: topic.to_string(),
            publisher_id,
            metadata: serde_json::to_value(advertisement).unwrap(),
        }))
    }

    fn event_source_removed(topic: &str, publisher_id: u64) -> Result<DiscoveryEvent> {
        Ok(DiscoveryEvent::Removed(DiscoveryInstanceId::EventSource(
            EventSourceInstanceId {
                scope: EventScope::Endpoint {
                    endpoint: EndpointId::from("ns.worker.generate"),
                },
                topic: topic.to_string(),
                publisher_id,
            },
        )))
    }

    fn membership_view(workers: &[WorkerWithDpRank]) -> KvSourceMembershipView {
        let endpoint = EndpointId::from("ns.worker.generate");
        KvSourceMembershipView {
            serving_endpoint: endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(endpoint),
            sources: workers
                .iter()
                .map(|worker| (*worker, KvSourceStatus::Missing))
                .collect(),
            kv_event_publishing_enabled: HashMap::new(),
            kv_event_source_mode: HashMap::new(),
            recovery_expected: HashMap::new(),
        }
    }

    fn owners(affected: Affected) -> HashSet<CacheOwnerId> {
        match affected {
            Affected::Owners(owners) => owners,
            Affected::All => panic!("expected specific owners"),
        }
    }

    #[test]
    fn fence_withdraws_only_affected_owners_and_drops_stale_plans() {
        let (first, second) = (owner(4), owner(5));
        let mut runtime = HashMap::from([
            (first, ready_runtime(WorkerWithDpRank::new(17, 3))),
            (second, ready_runtime(WorkerWithDpRank::new(18, 0))),
        ]);
        let mut fence = ProjectionFence::default();
        fence.commit(&Indexer::None, &runtime);
        assert_eq!(published(&fence), HashSet::from([first, second]));

        fence.invalidate(&Indexer::None, &Affected::Owners(HashSet::from([first])));
        assert_eq!(published(&fence), HashSet::from([second]));
        // A plan made before the observation can never republish its owner.
        fence.commit(&Indexer::None, &runtime);
        assert_eq!(published(&fence), HashSet::from([second]));

        runtime.get_mut(&first).unwrap().revision = fence.revision(&first);
        fence.commit(&Indexer::None, &runtime);
        assert_eq!(published(&fence), HashSet::from([first, second]));

        fence.invalidate(&Indexer::None, &Affected::All);
        assert!(published(&fence).is_empty());
        fence.commit(&Indexer::None, &runtime);
        assert!(published(&fence).is_empty());
    }

    #[test]
    fn fence_index_names_owners_sharing_a_worker() {
        let (first, second) = (owner(4), owner(5));
        let worker = WorkerWithDpRank::new(17, 3);
        let mut index = FenceIndex::default();

        let added = index.attachment(event_source_added(
            KV_STATE_ATTACHMENT_TOPIC_V2,
            1,
            &attachment_advertisement(first, 7),
        ));
        assert_eq!(owners(added), HashSet::from([first]));
        // The worker moves to another owner before the old attachment is removed:
        // both owners are affected.
        let moved = index.attachment(event_source_added(
            KV_STATE_ATTACHMENT_TOPIC_V2,
            2,
            &attachment_advertisement(second, 1),
        ));
        assert_eq!(owners(moved), HashSet::from([first, second]));
        let removed = index.attachment(event_source_removed(KV_STATE_ATTACHMENT_TOPIC_V2, 1));
        assert_eq!(owners(removed), HashSet::from([first, second]));

        assert_eq!(
            owners(index.membership(&membership_view(&[worker]))),
            HashSet::from([second])
        );
        assert!(owners(index.membership(&membership_view(&[worker]))).is_empty());
        assert_eq!(
            owners(index.membership(&membership_view(&[]))),
            HashSet::from([second])
        );

        // An unattributable removal invalidates every owner.
        assert!(matches!(
            index.source(event_source_removed(KV_STATE_SOURCE_TOPIC_V2, 99)),
            Affected::All
        ));
    }

    #[test]
    fn fence_index_names_owners_of_a_worker_present_at_startup() {
        let owner = owner(4);
        let worker = WorkerWithDpRank::new(17, 3);
        let mut index = FenceIndex::new(&membership_view(&[worker]));
        index.attachment(event_source_added(
            KV_STATE_ATTACHMENT_TOPIC_V2,
            1,
            &attachment_advertisement(owner, 7),
        ));

        assert_eq!(
            owners(index.membership(&membership_view(&[]))),
            HashSet::from([owner])
        );
    }

    #[test]
    fn fence_folds_queued_discovery_events_into_one_withdrawal() {
        let (first, second) = (owner(4), owner(5));
        let mut index = FenceIndex::default();
        let queued = |event| -> DiscoveryStream {
            Box::pin(futures::stream::iter([event]).chain(futures::stream::pending()))
        };
        let mut sources = queued(event_source_added(
            KV_STATE_SOURCE_TOPIC_V2,
            41,
            &source_advertisement(first),
        ));
        let mut attachments = queued(event_source_added(
            KV_STATE_ATTACHMENT_TOPIC_V2,
            1,
            &attachment_advertisement(second, 7),
        ));
        let affected = index.drain_ready(
            &mut sources,
            &mut attachments,
            Affected::Owners(HashSet::new()),
        );
        assert_eq!(owners(affected.unwrap()), HashSet::from([first, second]));

        // A watch that ends fails the fence closed.
        let mut ended: DiscoveryStream = Box::pin(futures::stream::empty());
        assert!(
            index
                .drain_ready(&mut ended, &mut attachments, Affected::All)
                .is_none()
        );
    }

    #[tokio::test]
    async fn initial_discovery_waits_for_the_snapshot() {
        let owner = owner(4);
        let record_id = attachment_record_id(41, 7);
        let added = event_source_added(
            KV_STATE_ATTACHMENT_TOPIC_V2,
            record_id,
            &attachment_advertisement(owner, 7),
        );
        // Delivered after the watch is returned, as a backend's spawned task does.
        let mut stream: DiscoveryStream = Box::pin(
            futures::stream::once(async move {
                tokio::task::yield_now().await;
                added
            })
            .chain(futures::stream::iter([Ok(DiscoveryEvent::Resync(
                Vec::new(),
            ))]))
            .chain(futures::stream::pending()),
        );
        let endpoint = EndpointId::from("ns.worker.generate");
        let mut attachments = HashMap::<u64, KvStateAttachmentAdvertisement>::new();
        let mut retries = RecoveryRetries::default();
        let mut dirty = DirtyOwners::default();
        let cancel = CancellationToken::new();

        assert!(
            apply_initial_discovery(
                &mut stream,
                &endpoint,
                KV_STATE_ATTACHMENT_TOPIC_V2,
                &mut attachments,
                &mut retries,
                &mut dirty,
                &cancel,
            )
            .await
        );
        assert!(attachments.contains_key(&record_id));
        assert!(dirty.owners.contains(&owner));

        let mut ended: DiscoveryStream = Box::pin(futures::stream::empty());
        assert!(
            !apply_initial_discovery(
                &mut ended,
                &endpoint,
                KV_STATE_ATTACHMENT_TOPIC_V2,
                &mut attachments,
                &mut retries,
                &mut dirty,
                &cancel,
            )
            .await
        );
    }

    #[test]
    fn owners_on_workers_names_attached_and_projected_owners() {
        let (first, second, third) = (owner(4), owner(5), owner(6));
        let (attached_worker, projected_worker) =
            (WorkerWithDpRank::new(17, 3), WorkerWithDpRank::new(18, 0));
        let attachments = HashMap::from([(1, attachment_advertisement(first, 7))]);
        let runtime = HashMap::from([
            (second, ready_runtime(projected_worker)),
            (third, ready_runtime(WorkerWithDpRank::new(19, 0))),
        ]);
        assert_eq!(
            owners_on_workers(
                &HashSet::from([attached_worker, projected_worker]),
                &attachments,
                &runtime,
            ),
            HashSet::from([first, second])
        );
    }

    #[tokio::test]
    async fn detached_source_ignores_late_worker_events_until_reattached() {
        let owner = owner(4);
        let worker = WorkerWithDpRank::new(17, 3);
        let (primary, indexer) = test_indexer();
        let ingest = bound_ingest(owner, 41, None, Some(worker));
        let (live, _withdrawn) = live_ingest(indexer, &[&ingest]);

        live.apply(41, vec![worker_store(owner, worker, 1)]).await;
        assert!(primary.dump_events().await.unwrap().is_empty());

        ingest.lock().await.attachment_generation = Some(2);
        live.apply(41, vec![worker_store(owner, worker, 2)]).await;
        assert_eq!(primary.dump_events().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn attachment_change_buffers_old_worker_events_before_clearing() {
        let (primary, indexer) = test_indexer();
        let owner = owner(4);
        let old_worker = WorkerWithDpRank::new(17, 3);
        let ingest = bound_ingest(owner, 41, Some(7), Some(old_worker));
        let (live, _withdrawn) = live_ingest(indexer, &[&ingest]);

        buffer_live_ingest(
            &ingest,
            Some(8),
            Some(WorkerWithDpRank::new(18, 3)),
            true,
            1,
        )
        .await;
        // A late batch from the old attachment, after the old worker is cleared.
        live.apply(41, vec![worker_store(owner, old_worker, 1)])
            .await;

        assert!(primary.dump_events().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn recovery_tail_hands_off_to_live_ingest_and_retirement_drops_late_batches() {
        let owner = owner(4);
        let worker = WorkerWithDpRank::new(17, 3);
        let (primary, indexer) = test_indexer();
        let ingest = bound_ingest(owner, 41, Some(7), Some(worker));
        buffer_live_ingest(&ingest, Some(7), Some(worker), true, 1).await;
        let (live, _withdrawn) = live_ingest(indexer.clone(), &[&ingest]);

        // Recovery is in flight: live batches are buffered, not applied.
        live.apply(41, vec![worker_store(owner, worker, 1)]).await;
        live.apply(41, vec![worker_store(owner, worker, 2)]).await;
        assert!(primary.dump_events().await.unwrap().is_empty());

        let plan = OwnerRecoveryPlan {
            owner,
            source: source_advertisement(owner),
            attachment: Some(attachment_advertisement(owner, 7)),
            expected: KvStateAgentIdentity {
                cache_owner_id: owner,
                publisher_id: 41,
                protocol_version: KvStateProtocolVersion::V2,
            },
            recognized_worker: Some(worker),
            previous_publisher_id: None,
            previous_cursor: 0,
            recovery_required: true,
            revision: OwnerRevision::default(),
            schedule_generation: 1,
        };
        // The snapshot covered event 1; only event 2 replays.
        assert_eq!(
            apply_recovery_tail(&indexer, &plan, 1, &ingest, true)
                .await
                .unwrap(),
            2
        );
        assert!(ingest.lock().await.tail.is_none());
        assert_eq!(primary.dump_events().await.unwrap().len(), 1);

        live.apply(41, vec![worker_store(owner, worker, 3)]).await;
        assert_eq!(ingest.lock().await.recovered_cursor, 3);

        retire_ingest(&live.bindings, &ingest).await;
        assert!(live.bindings.is_empty());
        assert!(ingest.lock().await.retired);
        live.apply(41, vec![worker_store(owner, worker, 4)]).await;
        assert_eq!(primary.dump_events().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn incorrectly_attributed_live_event_withdraws_detached_hint_eligibility() {
        let selected = owner(4);
        let other = owner(5);
        let worker = WorkerWithDpRank::new(17, 3);
        let ingest = bound_ingest(selected, 41, None, Some(worker));
        let (live, mut withdrawn) = live_ingest(Indexer::None, &[&ingest]);
        let event = RouterEvent::with_cache_owner(
            worker.worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(7)],
                }),
                dp_rank: worker.dp_rank,
            },
            StorageTier::HostPinned,
            other,
        );

        live.apply(41, vec![event]).await;
        assert_eq!(withdrawn.try_recv().unwrap(), (selected, 41));
        let state = ingest.lock().await;
        assert!(state.recovery_required);
        assert_eq!(state.recovered_cursor, 0);
    }
}
