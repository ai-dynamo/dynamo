// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::{
    ConsumerInstanceId, DcCkfDelta, LaneLease, ProducerIdentity,
};
use futures_util::FutureExt as _;
use parking_lot::Mutex;
use tokio::sync::{Semaphore, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::actor::KvDcRelayHandle;
use super::publication_codec::{self, PublicationFrame};

pub(super) const DEFAULT_PUBLICATION_QUEUE_CAPACITY: usize = 16;
pub(super) const DEFAULT_PUBLICATION_QUEUE_BYTES: usize = 16 * 1024 * 1024;
pub(super) const DEFAULT_POOL_SUBSCRIBERS: usize = 64;
const DEFAULT_PUBLICATION_ENCODING_CONCURRENCY: usize = 2;

#[derive(Clone)]
pub(super) struct PublicationHubConfig {
    pub(super) queue_capacity: usize,
    pub(super) queue_bytes: usize,
    pub(super) max_subscribers: usize,
    pub(super) max_delta_images: usize,
    pub(super) encoding_permits: Arc<Semaphore>,
    #[cfg(test)]
    pub(super) initialization_gate: Option<Arc<Semaphore>>,
}

impl Default for PublicationHubConfig {
    fn default() -> Self {
        Self {
            queue_capacity: DEFAULT_PUBLICATION_QUEUE_CAPACITY,
            queue_bytes: DEFAULT_PUBLICATION_QUEUE_BYTES,
            max_subscribers: DEFAULT_POOL_SUBSCRIBERS,
            max_delta_images: super::protocol::wire::images::max_delta_images(),
            encoding_permits: Arc::new(Semaphore::new(DEFAULT_PUBLICATION_ENCODING_CONCURRENCY)),
            #[cfg(test)]
            initialization_gate: None,
        }
    }
}

impl PublicationHubConfig {
    fn validate(&self) -> Result<(), String> {
        if self.queue_capacity == 0 {
            return Err("publication queue capacity must be nonzero".to_string());
        }
        if self.max_subscribers == 0 {
            return Err("publication subscriber limit must be nonzero".to_string());
        }
        let maximum = super::protocol::wire::images::max_delta_images();
        if self.max_delta_images > maximum {
            return Err(format!(
                "publication delta image limit {} exceeds CBI1 maximum {maximum}",
                self.max_delta_images
            ));
        }
        let minimum_bytes = super::protocol::wire::images::IMAGES_HEADER_LEN
            .saturating_add(12)
            .saturating_add(self.max_delta_images.saturating_mul(12))
            .saturating_add(256);
        if self.queue_bytes < minimum_bytes {
            return Err(format!(
                "publication queue byte limit {} cannot hold one maximum frame of {minimum_bytes} bytes",
                self.queue_bytes
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HubSnapshot {
    identity: ProducerIdentity,
    lease: LaneLease,
    sequence: u64,
    buckets: Arc<[u64]>,
}

impl HubSnapshot {
    pub(super) fn from_actor(
        identity: ProducerIdentity,
        lease: LaneLease,
        sequence: u64,
        buckets: &[u64],
    ) -> Self {
        Self {
            identity,
            lease,
            sequence,
            buckets: Arc::from(buckets),
        }
    }

    pub(crate) const fn identity(&self) -> ProducerIdentity {
        self.identity
    }

    pub(crate) const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(crate) fn buckets(&self) -> &[u64] {
        &self.buckets
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum PublicationHubError {
    #[error("unknown or inactive pool {0}")]
    UnknownPool(PoolId),
    #[error("active producer identity for pool {0} does not match the subscription request")]
    ProducerMismatch(PoolId),
    #[error("publication hub is unavailable: {0}")]
    Unavailable(String),
    #[error("pool {pool_id} reached its subscriber limit {limit}")]
    SubscriberLimit { pool_id: PoolId, limit: usize },
    #[error("Relay reached its initialized publication hub limit {limit}")]
    InitializedHubLimit { limit: usize },
    #[error("pool {0} subscriber exceeded its bounded publication queue")]
    SubscriberLagged(PoolId),
    #[error("publication identity changed from {expected:?} to {actual:?}")]
    IdentityChanged {
        expected: Box<ProducerIdentity>,
        actual: Box<ProducerIdentity>,
    },
    #[error("publication lease changed from {expected:?} to {actual:?}")]
    LeaseChanged {
        expected: LaneLease,
        actual: LaneLease,
    },
    #[error("publication sequence gap: mirror is at {current}, delta extends {base} -> {next}")]
    SequenceGap { current: u64, base: u64, next: u64 },
    #[error("delta bucket {bucket} is outside mirror length {bucket_count}")]
    BucketOutOfRange { bucket: usize, bucket_count: usize },
}

struct QueuedFrame {
    frame: Arc<PublicationFrame>,
    bytes: usize,
    pending_bytes: Arc<AtomicUsize>,
}

impl Drop for QueuedFrame {
    fn drop(&mut self) {
        let previous = self.pending_bytes.fetch_sub(self.bytes, Ordering::AcqRel);
        debug_assert!(previous >= self.bytes);
    }
}

enum SubscriberSendError {
    Closed,
    Full,
}

struct HubSubscriber {
    sender: mpsc::Sender<QueuedFrame>,
    pending_bytes: Arc<AtomicUsize>,
    byte_limit: usize,
    close_reason: Arc<AtomicU8>,
}

const SUBSCRIBER_ACTIVE: u8 = 0;
const SUBSCRIBER_LAGGED: u8 = 1;
const SUBSCRIBER_RETIRED: u8 = 2;

impl HubSubscriber {
    fn try_send(&self, frame: Arc<PublicationFrame>) -> Result<(), SubscriberSendError> {
        let bytes = frame.queued_bytes();
        reserve_bytes(&self.pending_bytes, bytes, self.byte_limit)?;
        let queued = QueuedFrame {
            frame,
            bytes,
            pending_bytes: self.pending_bytes.clone(),
        };
        self.sender.try_send(queued).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => SubscriberSendError::Full,
            mpsc::error::TrySendError::Closed(_) => SubscriberSendError::Closed,
        })
    }
}

fn reserve_bytes(
    pending: &AtomicUsize,
    bytes: usize,
    limit: usize,
) -> Result<(), SubscriberSendError> {
    let mut current = pending.load(Ordering::Acquire);
    loop {
        let next = current
            .checked_add(bytes)
            .ok_or(SubscriberSendError::Full)?;
        if next > limit {
            return Err(SubscriberSendError::Full);
        }
        match pending.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(()),
            Err(observed) => current = observed,
        }
    }
}

struct HubState {
    snapshot: HubSnapshot,
    is_ready: bool,
    last_error: Option<String>,
    next_subscriber_id: u64,
    subscribers: HashMap<u64, HubSubscriber>,
}

pub(super) type TerminalFailure = Arc<dyn Fn(String) + Send + Sync>;

struct HubTask {
    actor: KvDcRelayHandle,
    lease: LaneLease,
    pool_id: PoolId,
    state: Arc<Mutex<HubState>>,
    actor_deltas: tokio::sync::broadcast::Receiver<DcCkfDelta>,
    max_delta_images: usize,
    encoding_permits: Arc<Semaphore>,
    terminal_failure: TerminalFailure,
    cancel: CancellationToken,
    stopped: CancellationToken,
}

struct PublicationHubInner {
    pool_id: PoolId,
    config: PublicationHubConfig,
    state: Arc<Mutex<HubState>>,
    cancel: CancellationToken,
    stopped: CancellationToken,
    task: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Clone)]
pub(crate) struct PublicationHub {
    inner: Arc<PublicationHubInner>,
}

impl PublicationHub {
    pub(super) async fn start(
        actor: KvDcRelayHandle,
        lease: LaneLease,
        config: PublicationHubConfig,
        terminal_failure: TerminalFailure,
    ) -> Result<Self, super::host::KvDcRelayError> {
        config
            .validate()
            .map_err(super::host::KvDcRelayError::Publisher)?;
        #[cfg(test)]
        if let Some(gate) = &config.initialization_gate {
            let permit = gate.acquire().await;
            if permit.is_err() {
                return Err(super::host::KvDcRelayError::ShuttingDown);
            }
        }

        let subscription = actor.subscribe(lease).await?;
        let snapshot = copy_actor_snapshot(subscription.snapshot, config.encoding_permits.clone())
            .await
            .map_err(super::host::KvDcRelayError::Publisher)?;
        validate_snapshot(actor.identity(), lease, &snapshot)
            .map_err(|error| super::host::KvDcRelayError::Publisher(error.to_string()))?;
        let pool_id = snapshot.identity.pool_id();
        let state = Arc::new(Mutex::new(HubState {
            snapshot,
            is_ready: true,
            last_error: None,
            next_subscriber_id: 1,
            subscribers: HashMap::new(),
        }));
        let cancel = CancellationToken::new();
        let stopped = CancellationToken::new();
        let task = tokio::spawn(run_hub(HubTask {
            actor,
            lease,
            pool_id,
            state: state.clone(),
            actor_deltas: subscription.deltas,
            max_delta_images: config.max_delta_images,
            encoding_permits: config.encoding_permits.clone(),
            terminal_failure,
            cancel: cancel.clone(),
            stopped: stopped.clone(),
        }));
        Ok(Self {
            inner: Arc::new(PublicationHubInner {
                pool_id,
                config,
                state,
                cancel,
                stopped,
                task: Mutex::new(Some(task)),
            }),
        })
    }

    pub(crate) fn subscribe(&self) -> Result<PublicationHubSubscription, PublicationHubError> {
        let mut state = self.inner.state.lock();
        if !state.is_ready {
            return Err(PublicationHubError::Unavailable(
                state
                    .last_error
                    .clone()
                    .unwrap_or_else(|| "stopped".to_string()),
            ));
        }
        if state.subscribers.len() >= self.inner.config.max_subscribers {
            return Err(PublicationHubError::SubscriberLimit {
                pool_id: self.inner.pool_id,
                limit: self.inner.config.max_subscribers,
            });
        }
        let subscriber_id = state.next_subscriber_id;
        state.next_subscriber_id = state.next_subscriber_id.checked_add(1).ok_or_else(|| {
            PublicationHubError::Unavailable("subscriber ID space exhausted".to_string())
        })?;
        let (sender, receiver) = mpsc::channel(self.inner.config.queue_capacity);
        let close_reason = Arc::new(AtomicU8::new(SUBSCRIBER_ACTIVE));
        state.subscribers.insert(
            subscriber_id,
            HubSubscriber {
                sender,
                pending_bytes: Arc::new(AtomicUsize::new(0)),
                byte_limit: self.inner.config.queue_bytes,
                close_reason: close_reason.clone(),
            },
        );
        Ok(PublicationHubSubscription {
            snapshot: Some(state.snapshot.clone()),
            receiver,
            subscriber_id,
            pool_id: self.inner.pool_id,
            close_reason,
            hub: Arc::downgrade(&self.inner),
        })
    }

    pub(crate) fn retire(&self) {
        self.inner.cancel.cancel();
        let mut state = self.inner.state.lock();
        state.is_ready = false;
        close_subscribers(&mut state, SUBSCRIBER_RETIRED);
    }

    pub(crate) async fn shutdown(&self) {
        self.inner.cancel.cancel();
        let task = self.inner.task.lock().take();
        if let Some(task) = task {
            if let Err(error) = task.await
                && !error.is_cancelled()
            {
                tracing::warn!(pool_id = %self.inner.pool_id, %error, "KV Relay publication hub monitor failed during shutdown");
            }
        } else {
            self.inner.stopped.cancelled().await;
        }
    }
}

pub(crate) struct PublicationHubSubscription {
    // Bootstrap owns the generation snapshot. Keeping this Arc for the lifetime of a slow
    // network stream would force the hub's next in-place delta to clone the full CKF image.
    snapshot: Option<HubSnapshot>,
    receiver: mpsc::Receiver<QueuedFrame>,
    subscriber_id: u64,
    pool_id: PoolId,
    close_reason: Arc<AtomicU8>,
    hub: Weak<PublicationHubInner>,
}

impl PublicationHubSubscription {
    #[cfg(test)]
    pub(crate) const fn snapshot(&self) -> Option<&HubSnapshot> {
        self.snapshot.as_ref()
    }

    pub(crate) fn take_snapshot(&mut self) -> Result<HubSnapshot, PublicationHubError> {
        self.snapshot.take().ok_or_else(|| {
            PublicationHubError::Unavailable(
                "publication subscription snapshot was already consumed".to_string(),
            )
        })
    }

    pub(crate) async fn recv(&mut self) -> Result<Arc<PublicationFrame>, PublicationHubError> {
        self.ensure_active()?;
        let queued = self.receiver.recv().await;
        self.ensure_active()?;
        match queued {
            Some(queued) => Ok(queued.frame.clone()),
            None => Err(PublicationHubError::Unavailable(
                "publication subscription requires a fresh snapshot".to_string(),
            )),
        }
    }

    pub(crate) fn ensure_active(&self) -> Result<(), PublicationHubError> {
        match self.close_reason.load(Ordering::Acquire) {
            SUBSCRIBER_ACTIVE if !self.receiver.is_closed() => Ok(()),
            SUBSCRIBER_LAGGED => Err(PublicationHubError::SubscriberLagged(self.pool_id)),
            _ => Err(PublicationHubError::Unavailable(
                "publication subscription requires a fresh snapshot".to_string(),
            )),
        }
    }
}

impl Drop for PublicationHubSubscription {
    fn drop(&mut self) {
        if let Some(hub) = self.hub.upgrade() {
            hub.state.lock().subscribers.remove(&self.subscriber_id);
        }
    }
}

async fn run_hub(mut task: HubTask) {
    let result = AssertUnwindSafe(run_hub_loop(&mut task))
        .catch_unwind()
        .await;
    let failure = match result {
        Ok(failure) => failure,
        Err(_) => Some("publication hub task panicked".to_string()),
    };
    if let Some(reason) = failure {
        {
            let mut state = task.state.lock();
            state.is_ready = false;
            state.last_error = Some(reason.clone());
            close_subscribers(&mut state, SUBSCRIBER_RETIRED);
        }
        tracing::error!(pool_id = %task.pool_id, error = %reason, "KV DC Relay publication hub stopped");
        (task.terminal_failure)(reason);
    } else {
        let mut state = task.state.lock();
        state.is_ready = false;
        close_subscribers(&mut state, SUBSCRIBER_RETIRED);
    }
    task.stopped.cancel();
}

async fn run_hub_loop(task: &mut HubTask) -> Option<String> {
    loop {
        let delta = tokio::select! {
            biased;
            _ = task.cancel.cancelled() => return None,
            delta = task.actor_deltas.recv() => delta,
        };
        match delta {
            Ok(delta) if delta.images().len() > task.max_delta_images => {
                if let Err(error) = recover_snapshot(task).await {
                    return Some(format!("failed to recover oversized publication: {error}"));
                }
            }
            Ok(delta) => {
                let (delta, frame) = match encode_delta(delta, task.encoding_permits.clone()).await
                {
                    Ok((delta, frame)) => (delta, Arc::new(frame)),
                    Err(error) => return Some(format!("failed to encode publication: {error}")),
                };
                if task.cancel.is_cancelled() {
                    return None;
                }
                let shared_snapshot = {
                    let mut state = task.state.lock();
                    if Arc::strong_count(&state.snapshot.buckets) == 1 {
                        if let Err(error) = apply_delta(&mut state.snapshot, &delta) {
                            return Some(error.to_string());
                        }
                        fan_out(&mut state, frame.clone());
                        None
                    } else {
                        Some(state.snapshot.clone())
                    }
                };
                let Some(snapshot) = shared_snapshot else {
                    continue;
                };
                let base_sequence = delta.base_sequence();
                let next_sequence = delta.sequence();
                let snapshot =
                    match apply_delta_off_thread(snapshot, delta, task.encoding_permits.clone())
                        .await
                    {
                        Ok(snapshot) => snapshot,
                        Err(error) => return Some(format!("failed to apply publication: {error}")),
                    };
                if task.cancel.is_cancelled() {
                    return None;
                }
                let mut state = task.state.lock();
                if state.snapshot.sequence != base_sequence {
                    return Some(
                        PublicationHubError::SequenceGap {
                            current: state.snapshot.sequence,
                            base: base_sequence,
                            next: next_sequence,
                        }
                        .to_string(),
                    );
                }
                state.snapshot = snapshot;
                fan_out(&mut state, frame);
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                if let Err(error) = recover_snapshot(task).await {
                    return Some(format!(
                        "failed to recover after internal lag of {skipped} deltas: {error}"
                    ));
                }
                tracing::warn!(pool_id = %task.pool_id, skipped, "KV DC Relay publication hub recovered from internal lag");
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                return (!task.cancel.is_cancelled())
                    .then(|| "actor publication stream closed".to_string());
            }
        }
    }
}

async fn recover_snapshot(task: &mut HubTask) -> Result<(), PublicationHubError> {
    let subscription = task
        .actor
        .subscribe(task.lease)
        .await
        .map_err(|error| PublicationHubError::Unavailable(error.to_string()))?;
    let replacement = copy_actor_snapshot(subscription.snapshot, task.encoding_permits.clone())
        .await
        .map_err(PublicationHubError::Unavailable)?;
    validate_snapshot(task.actor.identity(), task.lease, &replacement)?;
    let mut state = task.state.lock();
    state.snapshot = replacement;
    close_subscribers(&mut state, SUBSCRIBER_RETIRED);
    drop(state);
    task.actor_deltas = subscription.deltas;
    Ok(())
}

async fn copy_actor_snapshot(
    snapshot: dynamo_kv_router::indexer::cuckoo::DcCkfSnapshot,
    permits: Arc<Semaphore>,
) -> Result<HubSnapshot, String> {
    run_blocking_publication_work(permits, move || {
        HubSnapshot::from_actor(
            snapshot.identity(),
            snapshot.lease(),
            snapshot.sequence(),
            snapshot.buckets(),
        )
    })
    .await
}

async fn encode_delta(
    delta: DcCkfDelta,
    permits: Arc<Semaphore>,
) -> Result<(DcCkfDelta, PublicationFrame), String> {
    run_blocking_publication_work(permits, move || {
        publication_codec::encode_delta(&delta)
            .map(|frame| (delta, frame))
            .map_err(|error| error.to_string())
    })
    .await?
}

async fn apply_delta_off_thread(
    mut snapshot: HubSnapshot,
    delta: DcCkfDelta,
    permits: Arc<Semaphore>,
) -> Result<HubSnapshot, String> {
    run_blocking_publication_work(permits, move || {
        apply_delta(&mut snapshot, &delta)
            .map(|()| snapshot)
            .map_err(|error| error.to_string())
    })
    .await?
}

async fn run_blocking_publication_work<T, Work>(
    permits: Arc<Semaphore>,
    work: Work,
) -> Result<T, String>
where
    T: Send + 'static,
    Work: FnOnce() -> T + Send + 'static,
{
    let permit = permits
        .acquire_owned()
        .await
        .map_err(|_| "publication encoder is shutting down".to_string())?;
    tokio::task::spawn_blocking(move || {
        let result = work();
        drop(permit);
        result
    })
    .await
    .map_err(|error| format!("publication blocking task failed: {error}"))
}

fn validate_snapshot(
    identity: ProducerIdentity,
    lease: LaneLease,
    snapshot: &HubSnapshot,
) -> Result<(), PublicationHubError> {
    if snapshot.identity != identity {
        return Err(PublicationHubError::IdentityChanged {
            expected: Box::new(identity),
            actual: Box::new(snapshot.identity),
        });
    }
    if snapshot.lease != lease {
        return Err(PublicationHubError::LeaseChanged {
            expected: lease,
            actual: snapshot.lease,
        });
    }
    Ok(())
}

fn apply_delta(snapshot: &mut HubSnapshot, delta: &DcCkfDelta) -> Result<(), PublicationHubError> {
    validate_snapshot(delta.identity(), delta.lease(), snapshot)?;
    let Some(next_sequence) = delta.base_sequence().checked_add(1) else {
        return Err(PublicationHubError::SequenceGap {
            current: snapshot.sequence,
            base: delta.base_sequence(),
            next: delta.sequence(),
        });
    };
    if delta.base_sequence() != snapshot.sequence || delta.sequence() != next_sequence {
        return Err(PublicationHubError::SequenceGap {
            current: snapshot.sequence,
            base: delta.base_sequence(),
            next: delta.sequence(),
        });
    }
    let buckets = Arc::make_mut(&mut snapshot.buckets);
    for image in delta.images() {
        let bucket = image.bucket();
        let Some(value) = buckets.get_mut(bucket) else {
            return Err(PublicationHubError::BucketOutOfRange {
                bucket,
                bucket_count: buckets.len(),
            });
        };
        *value = image.value();
    }
    snapshot.sequence = delta.sequence();
    Ok(())
}

fn fan_out(state: &mut HubState, frame: Arc<PublicationFrame>) {
    state
        .subscribers
        .retain(|_, subscriber| match subscriber.try_send(frame.clone()) {
            Ok(()) => true,
            Err(SubscriberSendError::Closed) => false,
            Err(SubscriberSendError::Full) => {
                subscriber
                    .close_reason
                    .store(SUBSCRIBER_LAGGED, Ordering::Release);
                false
            }
        });
}

fn close_subscribers(state: &mut HubState, reason: u8) {
    for subscriber in state.subscribers.values() {
        subscriber.close_reason.store(reason, Ordering::Release);
    }
    state.subscribers.clear();
}

pub(super) fn publication_lease(identity: ProducerIdentity) -> LaneLease {
    LaneLease::new(
        ConsumerInstanceId::new(identity.producer_incarnation()),
        0,
        identity.layout_generation(),
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::{
        CkfConfig, DcCkfBucketImage, DcCkfState, ProducerIdentity,
    };
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent,
    };

    use super::super::actor::StreamScope;
    use super::*;

    fn identity() -> ProducerIdentity {
        let format = DcCkfState::new(CkfConfig::new(32))
            .expect("fixture state")
            .format();
        ProducerIdentity::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            7,
            11,
            format,
        )
    }

    fn snapshot() -> HubSnapshot {
        let identity = identity();
        HubSnapshot {
            identity,
            lease: publication_lease(identity),
            sequence: 4,
            buckets: vec![0; identity.format().bucket_count()].into(),
        }
    }

    fn stored(event_id: u64, hash: u64) -> RouterEvent {
        const EXTERNAL_MASK: u64 = 0x7A11_0C4D_EE55_9911;
        RouterEvent::new(
            7,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(hash ^ EXTERNAL_MASK),
                        tokens_hash: LocalBlockHash(hash),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        )
    }

    async fn actor_and_hub(
        hub_config: PublicationHubConfig,
        terminal_failures: Arc<AtomicUsize>,
    ) -> (KvDcRelayHandle, PublicationHub) {
        let mut config = CkfConfig::new(256);
        config.publish_every_n_events = 1;
        let pool_id = identity().pool_id();
        let (actor, _faults) = KvDcRelayHandle::spawn_with_publication_delay(
            config,
            StreamScope {
                relay_incarnation: 7,
                layout_generation: 11,
                pool_id,
            },
            Duration::from_millis(1),
        )
        .expect("actor");
        let hub = PublicationHub::start(
            actor.clone(),
            publication_lease(actor.identity()),
            hub_config,
            Arc::new(move |_| {
                terminal_failures.fetch_add(1, Ordering::Relaxed);
            }),
        )
        .await
        .expect("hub");
        (actor, hub)
    }

    #[test]
    fn delta_requires_exact_identity_lease_and_sequence() {
        let mut snapshot = snapshot();
        let identity = snapshot.identity;
        let lease = snapshot.lease;
        let delta = DcCkfDelta::new(
            identity,
            lease,
            4,
            5,
            vec![DcCkfBucketImage::new(1, 0xABCD)],
        );
        apply_delta(&mut snapshot, &delta).expect("contiguous delta");
        assert_eq!(snapshot.sequence, 5);
        assert_eq!(snapshot.buckets[1], 0xABCD);

        let gap = DcCkfDelta::new(identity, lease, 4, 5, Vec::new());
        assert!(matches!(
            apply_delta(&mut snapshot, &gap),
            Err(PublicationHubError::SequenceGap { .. })
        ));
    }

    #[tokio::test]
    async fn subscriber_queue_is_bounded_by_messages_and_bytes() {
        let (sender, mut receiver) = mpsc::channel(1);
        let pending_bytes = Arc::new(AtomicUsize::new(0));
        let subscriber = HubSubscriber {
            sender,
            pending_bytes: pending_bytes.clone(),
            byte_limit: 300,
            close_reason: Arc::new(AtomicU8::new(SUBSCRIBER_ACTIVE)),
        };
        let frame = Arc::new(publication_codec::encode_heartbeat(identity(), 1));
        assert!(subscriber.try_send(frame.clone()).is_ok());
        assert!(matches!(
            subscriber.try_send(frame),
            Err(SubscriberSendError::Full)
        ));
        assert!(pending_bytes.load(Ordering::Acquire) <= 300);
        drop(receiver.recv().await);
        assert_eq!(pending_bytes.load(Ordering::Acquire), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blocking_publication_work_keeps_tokio_worker_responsive() {
        let entered = Arc::new(Semaphore::new(0));
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let task = tokio::spawn({
            let entered = entered.clone();
            let release = release.clone();
            async move {
                run_blocking_publication_work(Arc::new(Semaphore::new(1)), move || {
                    entered.add_permits(1);
                    let (released, changed) = &*release;
                    let guard = released.lock().unwrap_or_else(|error| error.into_inner());
                    drop(
                        changed
                            .wait_while(guard, |released| !*released)
                            .unwrap_or_else(|error| error.into_inner()),
                    );
                })
                .await
            }
        });

        tokio::time::timeout(Duration::from_secs(1), entered.acquire())
            .await
            .expect("blocking work started")
            .expect("entry semaphore open")
            .forget();
        tokio::time::timeout(
            Duration::from_millis(100),
            tokio::time::sleep(Duration::from_millis(1)),
        )
        .await
        .expect("Tokio worker remained responsive");

        let (released, changed) = &*release;
        *released.lock().unwrap_or_else(|error| error.into_inner()) = true;
        changed.notify_all();
        task.await.expect("publication task joined").unwrap();
    }

    #[tokio::test]
    async fn subscriber_limit_rejects_excess_streams() {
        let failures = Arc::new(AtomicUsize::new(0));
        let config = PublicationHubConfig {
            max_subscribers: 1,
            ..PublicationHubConfig::default()
        };
        let (actor, hub) = actor_and_hub(config, failures.clone()).await;
        let subscription = hub.subscribe().unwrap();

        assert!(matches!(
            hub.subscribe(),
            Err(PublicationHubError::SubscriberLimit { limit: 1, .. })
        ));

        drop(subscription);
        hub.shutdown().await;
        actor.shutdown().await.unwrap();
        assert_eq!(failures.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn bootstrap_snapshot_is_released_after_ownership_transfer() {
        let failures = Arc::new(AtomicUsize::new(0));
        let (actor, hub) = actor_and_hub(PublicationHubConfig::default(), failures.clone()).await;
        let mut subscription = hub.subscribe().unwrap();
        let hub_buckets = hub.inner.state.lock().snapshot.buckets.clone();

        let bootstrap = subscription.take_snapshot().unwrap();
        assert!(subscription.snapshot().is_none());
        assert!(subscription.take_snapshot().is_err());
        assert_eq!(Arc::strong_count(&hub_buckets), 3);

        drop(bootstrap);
        assert_eq!(Arc::strong_count(&hub_buckets), 2);
        drop((subscription, hub_buckets));
        hub.shutdown().await;
        actor.shutdown().await.unwrap();
        assert_eq!(failures.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn delta_keeps_the_unique_hub_bucket_allocation() {
        let failures = Arc::new(AtomicUsize::new(0));
        let (actor, hub) = actor_and_hub(PublicationHubConfig::default(), failures.clone()).await;
        let (before_ptr, before_sequence) = {
            let state = hub.inner.state.lock();
            (
                state.snapshot.buckets.as_ptr() as usize,
                state.snapshot.sequence(),
            )
        };

        actor.admit_event(1, stored(1, 99)).await.unwrap();
        actor.flush().await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if hub.inner.state.lock().snapshot.sequence() > before_sequence {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("hub must apply the actor delta");

        {
            let state = hub.inner.state.lock();
            assert_eq!(state.snapshot.buckets.as_ptr() as usize, before_ptr);
        }
        hub.shutdown().await;
        actor.shutdown().await.unwrap();
        assert_eq!(failures.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn one_actor_lease_fans_out_one_contiguous_frame_instance() {
        let failures = Arc::new(AtomicUsize::new(0));
        let (actor, hub) = actor_and_hub(PublicationHubConfig::default(), failures.clone()).await;
        let mut first = hub.subscribe().unwrap();
        let mut second = hub.subscribe().unwrap();
        assert_eq!(
            first.snapshot().unwrap().sequence(),
            second.snapshot().unwrap().sequence()
        );

        actor.admit_event(1, stored(1, 99)).await.unwrap();
        actor.flush().await.unwrap();
        let first_frame = first.recv().await.unwrap();
        let second_frame = second.recv().await.unwrap();
        assert!(Arc::ptr_eq(&first_frame, &second_frame));
        assert_eq!(
            first_frame.base_sequence,
            first.snapshot().unwrap().sequence()
        );
        assert_eq!(first_frame.sequence, first_frame.base_sequence + 1);

        let third = hub.subscribe().unwrap();
        assert_eq!(third.snapshot().unwrap().sequence(), first_frame.sequence);
        drop((first, second, third));
        hub.shutdown().await;
        actor.shutdown().await.unwrap();
        assert_eq!(failures.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn slow_subscriber_is_removed_without_affecting_fast_subscriber() {
        let failures = Arc::new(AtomicUsize::new(0));
        let config = PublicationHubConfig {
            queue_capacity: 1,
            ..PublicationHubConfig::default()
        };
        let (actor, hub) = actor_and_hub(config, failures.clone()).await;
        let mut slow = hub.subscribe().unwrap();
        let mut fast = hub.subscribe().unwrap();

        for event_id in 1..=3 {
            actor
                .admit_event(1, stored(event_id, event_id + 100))
                .await
                .unwrap();
            actor.flush().await.unwrap();
            let frame = fast.recv().await.unwrap();
            assert_eq!(frame.sequence, event_id);
        }
        assert!(matches!(
            slow.recv().await,
            Err(PublicationHubError::SubscriberLagged(_))
        ));
        assert_eq!(hub.inner.state.lock().subscribers.len(), 1);

        drop((slow, fast));
        hub.shutdown().await;
        actor.shutdown().await.unwrap();
        assert_eq!(failures.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn oversized_delta_resnapshots_and_closes_existing_subscribers() {
        let failures = Arc::new(AtomicUsize::new(0));
        let config = PublicationHubConfig {
            max_delta_images: 0,
            ..PublicationHubConfig::default()
        };
        let (actor, hub) = actor_and_hub(config, failures.clone()).await;
        let mut old = hub.subscribe().unwrap();

        actor.admit_event(1, stored(1, 777)).await.unwrap();
        actor.flush().await.unwrap();
        assert!(old.recv().await.is_err());
        let replacement = hub.subscribe().unwrap();
        assert_eq!(replacement.snapshot().unwrap().sequence(), 1);
        assert!(hub.inner.state.lock().is_ready);

        drop((old, replacement));
        hub.shutdown().await;
        actor.shutdown().await.unwrap();
        assert_eq!(failures.load(Ordering::Relaxed), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn internal_actor_lag_resnapshots_and_closes_existing_subscribers() {
        let failures = Arc::new(AtomicUsize::new(0));
        let (actor, hub) = actor_and_hub(PublicationHubConfig::default(), failures.clone()).await;
        let mut old = hub.subscribe().unwrap();
        let state = hub.inner.state.clone();
        let (locked_tx, locked_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let blocker = std::thread::spawn(move || {
            let _guard = state.lock();
            let _ = locked_tx.send(());
            release_rx.recv().expect("release hub state");
        });
        locked_rx.await.expect("hub state lock");

        for event_id in 1..=70 {
            actor
                .admit_event(1, stored(event_id, event_id + 1_000))
                .await
                .unwrap();
            actor.flush().await.unwrap();
        }
        release_tx.send(()).unwrap();
        blocker.join().unwrap();

        tokio::time::timeout(Duration::from_secs(2), async {
            while old.recv().await.is_ok() {}
        })
        .await
        .expect("old subscription must close after resnapshot");
        let replacement = hub.subscribe().unwrap();
        assert_eq!(replacement.snapshot().unwrap().sequence(), 70);
        assert!(hub.inner.state.lock().is_ready);

        drop((old, replacement));
        hub.shutdown().await;
        actor.shutdown().await.unwrap();
        assert_eq!(failures.load(Ordering::Relaxed), 0);
    }
}
