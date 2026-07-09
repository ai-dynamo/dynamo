// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Weak,
        atomic::{AtomicU8, AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::{Result, bail};
use tokio::sync::{Notify, RwLock, mpsc, oneshot};

use dynamo_kv_router::RouterEventSink;
use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::{
    KvCacheEvent, KvCacheEventData, RouterEvent, StorageTier, WorkerId,
};
use dynamo_runtime::transports::event_plane::EventPublisher;
use dynamo_runtime::transports::nats::NatsQueue;

use crate::kv_router::KV_EVENT_SUBJECT;
use crate::kv_router::indexer::valkey::ValkeyIndexer;
use crate::kv_router::metrics::kv_publisher_metrics;

/// Bound one direct metadata mutation. The indexer deliberately retries
/// retryable transport/WAIT errors forever, so the publisher adds the
/// worker-level integrity boundary which keeps its input memory bounded.
#[cfg(not(test))]
const VALKEY_EVENT_MUTATION_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(test)]
const VALKEY_EVENT_MUTATION_TIMEOUT: Duration = Duration::from_millis(50);
const INTEGRITY_HEALTHY: u8 = 0;
const INTEGRITY_FAULTED: u8 = 1;
const INTEGRITY_FENCING: u8 = 2;
const INTEGRITY_FENCED: u8 = 3;
/// Cross-rank publishers feed one worker-owned collector. Its queue remains
/// bounded so a stalled primary eventually backpressures the already-bounded
/// raw ingress instead of growing forever, while a larger pipeline amortizes
/// one replica acknowledgement over bursty interleaved request chains.
const DIRECT_EVENT_BATCH_MAX_EVENTS: usize = 128;
const DIRECT_EVENT_BATCH_QUEUE_CAPACITY: usize = 4_096;
// Per-rank normalization already has a 1ms coalescing window. A second full
// millisecond would cap a lockstep 8-rank worker near 8k normalized events/s;
// 100us is enough to catch sibling wakeups without becoming the drain limit.
const DIRECT_EVENT_BATCH_COLLECT_WINDOW: Duration = Duration::from_micros(100);

pub(super) type ValkeyOperation<'a> = Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
pub(super) type ValkeyGcOperation<'a> = Pin<Box<dyn Future<Output = Result<[u64; 8]>> + Send + 'a>>;

/// Operations needed by the worker integrity barrier. Keeping the interface
/// narrow makes the fault protocol independently testable without a Valkey
/// process and keeps all production commands on the existing indexer.
pub(super) trait ValkeyWorkerEventOps: Send + Sync {
    fn apply_owned_batch<'a>(&'a self, events: &'a [RouterEvent]) -> ValkeyOperation<'a>;
    fn unregister(&self) -> ValkeyOperation<'_>;
    fn renew(&self) -> ValkeyOperation<'_>;
    fn gc_step(&self, inspection_budget: u32) -> ValkeyGcOperation<'_>;
}

struct IndexerWorkerEventOps {
    indexer: ValkeyIndexer,
    worker_id: WorkerId,
    owner_nonce: u64,
    lease_ms: u64,
}

impl ValkeyWorkerEventOps for IndexerWorkerEventOps {
    fn apply_owned_batch<'a>(&'a self, events: &'a [RouterEvent]) -> ValkeyOperation<'a> {
        Box::pin(async move {
            self.indexer
                .apply_events_owned(events, self.owner_nonce)
                .await
        })
    }

    fn unregister(&self) -> ValkeyOperation<'_> {
        Box::pin(
            self.indexer
                .unregister_worker_lease(self.worker_id, self.owner_nonce),
        )
    }

    fn renew(&self) -> ValkeyOperation<'_> {
        Box::pin(
            self.indexer
                .renew_worker_lease(self.worker_id, self.owner_nonce, self.lease_ms),
        )
    }

    fn gc_step(&self, inspection_budget: u32) -> ValkeyGcOperation<'_> {
        Box::pin(self.indexer.gc_step(inspection_budget))
    }
}

struct ValkeyPublisherIntegrityInner {
    backend: Arc<dyn ValkeyWorkerEventOps>,
    state: AtomicU8,
    mutation_gate: RwLock<()>,
    batch_tx: mpsc::Sender<DirectEventBatchRequest>,
    pending_events: AtomicUsize,
    pending_events_changed: Notify,
    state_changed: Notify,
    worker_id: WorkerId,
    owner_nonce: u64,
}

struct DirectEventBatchRequest {
    event: RouterEvent,
    completion: Option<oneshot::Sender<std::result::Result<(), Arc<str>>>>,
}

/// One integrity domain is shared by every DP-rank publisher and the worker
/// lease heartbeat. A fault stops new input immediately and permanently. An
/// owner-fenced UNREGISTER is attempted once; there is no automatic resume
/// until an atomic clear-and-reregister module primitive exists.
#[derive(Clone)]
pub(super) struct ValkeyPublisherIntegrity {
    inner: Arc<ValkeyPublisherIntegrityInner>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum IntegrityState {
    Healthy,
    Faulted,
    Fencing,
    Fenced,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FenceAttempt {
    NotNeeded,
    InProgress,
    Confirmed,
    Unconfirmed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GcStepOutcome {
    Completed([u64; 8]),
    SkippedBusy,
    SkippedFenced,
}

impl ValkeyPublisherIntegrity {
    pub(super) fn new(
        indexer: ValkeyIndexer,
        worker_id: WorkerId,
        owner_nonce: u64,
        lease_ms: u64,
    ) -> Self {
        Self::with_backend(
            worker_id,
            owner_nonce,
            Arc::new(IndexerWorkerEventOps {
                indexer,
                worker_id,
                owner_nonce,
                lease_ms,
            }),
        )
    }

    pub(super) fn with_backend(
        worker_id: WorkerId,
        owner_nonce: u64,
        backend: Arc<dyn ValkeyWorkerEventOps>,
    ) -> Self {
        let (batch_tx, batch_rx) = mpsc::channel(DIRECT_EVENT_BATCH_QUEUE_CAPACITY);
        let integrity = Self {
            inner: Arc::new(ValkeyPublisherIntegrityInner {
                backend,
                state: AtomicU8::new(INTEGRITY_HEALTHY),
                mutation_gate: RwLock::new(()),
                batch_tx,
                pending_events: AtomicUsize::new(0),
                pending_events_changed: Notify::new(),
                state_changed: Notify::new(),
                worker_id,
                owner_nonce,
            }),
        };
        spawn_direct_event_batcher(Arc::downgrade(&integrity.inner), batch_rx);
        integrity
    }

    pub(super) fn state(&self) -> IntegrityState {
        match self.inner.state.load(Ordering::Acquire) {
            INTEGRITY_HEALTHY => IntegrityState::Healthy,
            INTEGRITY_FAULTED => IntegrityState::Faulted,
            INTEGRITY_FENCING => IntegrityState::Fencing,
            INTEGRITY_FENCED => IntegrityState::Fenced,
            _ => unreachable!("publisher integrity state is internal"),
        }
    }

    pub(super) fn is_healthy(&self) -> bool {
        self.state() == IntegrityState::Healthy
    }

    pub(super) fn mark_fault(&self, reason: &'static str) {
        if self
            .inner
            .state
            .compare_exchange(
                INTEGRITY_HEALTHY,
                INTEGRITY_FAULTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            if let Some(metrics) = kv_publisher_metrics() {
                metrics.increment_integrity_fault(reason);
            }
            tracing::error!(
                worker_id = self.inner.worker_id,
                owner_nonce = self.inner.owner_nonce,
                reason,
                "Direct Valkey KV metadata integrity fault; stopping input and heartbeat while the worker is fenced"
            );
            self.inner.state_changed.notify_waiters();
            // Preserve one permit for a processor which races between its
            // health check and registering the notification future.
            self.inner.state_changed.notify_one();
        }
    }

    pub(super) async fn wait_for_state_change(&self) {
        self.inner.state_changed.notified().await;
    }

    async fn bounded_operation(
        &self,
        operation: &'static str,
        future: ValkeyOperation<'_>,
    ) -> Result<()> {
        match tokio::time::timeout(VALKEY_EVENT_MUTATION_TIMEOUT, future).await {
            Ok(result) => result,
            Err(_) => bail!(
                "Valkey {operation} exceeded the {}ms worker integrity deadline",
                VALKEY_EVENT_MUTATION_TIMEOUT.as_millis()
            ),
        }
    }

    async fn enqueue_event(
        &self,
        event: &RouterEvent,
        completion: Option<oneshot::Sender<std::result::Result<(), Arc<str>>>>,
    ) -> Result<()> {
        // Avoid joining the lock queue once another rank has already faulted.
        // A second check after acquisition closes the race with fencing.
        if !self.is_healthy() {
            bail!("direct Valkey publisher is fenced by an integrity fault");
        }
        if event.storage_tier.is_gpu() && matches!(event.event.data, KvCacheEventData::Cleared) {
            self.mark_fault("worker_clear");
            bail!("worker-wide CLEAR requires direct Valkey owner fencing and restart");
        }

        // Reserve before incrementing the drain counter. Cancellation while
        // waiting for capacity then cannot strand a phantom pending event;
        // after reserve succeeds, send is synchronous and infallible.
        let permit = self
            .inner
            .batch_tx
            .reserve()
            .await
            .map_err(|_| anyhow::anyhow!("direct Valkey event batcher stopped unexpectedly"));
        let permit = match permit {
            Ok(permit) => permit,
            Err(error) => {
                self.mark_fault("batcher_unavailable");
                return Err(error);
            }
        };
        if !self.is_healthy() {
            bail!("direct Valkey publisher is fenced by an integrity fault");
        }
        self.inner.pending_events.fetch_add(1, Ordering::AcqRel);
        permit.send(DirectEventBatchRequest {
            event: event.clone(),
            completion,
        });
        Ok(())
    }

    pub(super) async fn publish_event(&self, event: &RouterEvent) -> Result<()> {
        if !event.storage_tier.is_gpu() {
            return Ok(());
        }
        let (completion, completed) = oneshot::channel();
        self.enqueue_event(event, Some(completion)).await?;
        match completed.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => bail!("{error}"),
            Err(_) => {
                self.mark_fault("batcher_unavailable");
                bail!("direct Valkey event batcher stopped before completing an accepted event")
            }
        }
    }

    /// Queue an event for the direct Valkey batcher without waiting for its
    /// individual commit and without mirroring it to a generic event plane.
    ///
    /// The event processor already preserves raw worker ordering. Keeping this
    /// enqueue non-blocking lets a burst from all ranks coalesce into one
    /// replica-acknowledged APPLY_OWNED batch in direct Valkey mode.
    async fn enqueue_event_without_relay(&self, event: &RouterEvent) -> Result<()> {
        self.enqueue_event(event, None).await?;
        if !self.is_healthy() {
            bail!("direct Valkey publisher faulted after accepting a queued event");
        }
        Ok(())
    }

    /// Wait until every normalized event accepted by this worker has either
    /// committed (including replica acknowledgement) or entered the
    /// worker-wide integrity fence. Publisher shutdown calls this only after
    /// its own raw ingress has been drained.
    pub(super) async fn wait_for_idle(&self) {
        loop {
            // Register the waiter first so a transition to zero between the
            // load and await cannot be lost.
            let changed = self.inner.pending_events_changed.notified();
            if self.inner.pending_events.load(Ordering::Acquire) == 0 {
                return;
            }
            changed.await;
        }
    }

    fn complete_pending_events(&self, completed: usize) {
        let previous = self
            .inner
            .pending_events
            .fetch_sub(completed, Ordering::AcqRel);
        debug_assert!(previous >= completed);
        if previous == completed {
            self.inner.pending_events_changed.notify_waiters();
            self.inner.pending_events_changed.notify_one();
        }
    }

    async fn publish_events_after_gate(&self, events: &[RouterEvent]) -> Result<()> {
        if !self.is_healthy() {
            bail!("direct Valkey publisher is fenced by an integrity fault");
        }
        let result = self
            .bounded_operation(
                "APPLY_OWNED batch",
                self.inner.backend.apply_owned_batch(events),
            )
            .await;
        if let Err(error) = &result {
            let reason = if error.to_string().contains("integrity deadline") {
                "apply_timeout"
            } else {
                "apply_error"
            };
            if let Some(metrics) = kv_publisher_metrics() {
                metrics.increment_publish_error(reason);
            }
            tracing::error!(
                worker_id = self.inner.worker_id,
                owner_nonce = self.inner.owner_nonce,
                batch_events = events.len(),
                first_event_id = events.first().map(|event| event.event.event_id),
                last_event_id = events.last().map(|event| event.event.event_id),
                error = %error,
                "Direct Valkey APPLY_OWNED batch failed; events were not silently discarded"
            );
            self.mark_fault(reason);
        }
        result?;
        // An unrelated rank/input can fault while this batch is in flight.
        // Do not report success (or trigger the legacy relay) across that
        // boundary; the owner fence will retire the just-applied metadata.
        if !self.is_healthy() {
            bail!("direct Valkey publisher faulted while APPLY_OWNED batch was in flight");
        }
        Ok(())
    }

    /// Best-effort immediate admission fence for the current fault. There is
    /// deliberately no automatic re-registration here: UNREGISTER followed by
    /// REGISTER cannot prove a clear if either response/WAIT is ambiguous. The
    /// worker remains permanently fenced until process restart; server-side
    /// lease expiry is the definitive backstop when this attempt is unconfirmed.
    pub(super) async fn fence_once(&self) -> FenceAttempt {
        match self.inner.state.compare_exchange(
            INTEGRITY_FAULTED,
            INTEGRITY_FENCING,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(INTEGRITY_HEALTHY) => return FenceAttempt::NotNeeded,
            Err(INTEGRITY_FENCING | INTEGRITY_FENCED) => return FenceAttempt::InProgress,
            Err(_) => unreachable!("publisher integrity state is internal"),
        }

        // State changes to FENCING before this exclusive permit is requested.
        // Existing shared APPLY operations drain first; queued operations
        // acquire later, fail their health recheck, and never reach Valkey.
        let _gate = self.inner.mutation_gate.write().await;
        let unregister = self
            .bounded_operation("UNREGISTER_WORKER", self.inner.backend.unregister())
            .await;
        self.inner.state.store(INTEGRITY_FENCED, Ordering::Release);

        match unregister {
            Ok(()) => {
                if let Some(metrics) = kv_publisher_metrics() {
                    metrics.increment_integrity_fence("confirmed");
                }
                tracing::error!(
                    worker_id = self.inner.worker_id,
                    owner_nonce = self.inner.owner_nonce,
                    "Direct Valkey worker unregister was acknowledged; publisher remains permanently fenced pending process restart"
                );
                self.inner.state_changed.notify_waiters();
                self.inner.state_changed.notify_one();
                FenceAttempt::Confirmed
            }
            Err(error) => {
                if let Some(metrics) = kv_publisher_metrics() {
                    metrics.increment_integrity_fence("unconfirmed");
                }
                tracing::error!(
                    worker_id = self.inner.worker_id,
                    owner_nonce = self.inner.owner_nonce,
                    error = %error,
                    "Direct Valkey unregister could not be proved; input and heartbeat remain permanently fenced and server-side lease expiry is the admission backstop"
                );
                self.inner.state_changed.notify_waiters();
                self.inner.state_changed.notify_one();
                FenceAttempt::Unconfirmed
            }
        }
    }

    /// Heartbeats take the exclusive integrity permit. Once any integrity
    /// fault is visible they cannot renew or reacquire the old lease.
    pub(super) async fn renew_lease(&self) -> Result<()> {
        let _gate = self.inner.mutation_gate.write().await;
        if !self.is_healthy() {
            return Ok(());
        }
        let result = self
            .bounded_operation("RENEW_WORKER_LEASE", self.inner.backend.renew())
            .await;
        if let Err(error) = &result {
            tracing::error!(
                worker_id = self.inner.worker_id,
                owner_nonce = self.inner.owner_nonce,
                error = %error,
                "Valkey worker lease renewal failed; entering the worker integrity fence"
            );
            self.mark_fault("lease_renewal_error");
        }
        result
    }

    /// Run one opportunistic lifecycle-GC tick without delaying an active
    /// APPLY, lease renewal, or integrity fence. GC failure is maintenance
    /// telemetry only and never changes worker health.
    pub(super) async fn gc_step_if_idle(&self, inspection_budget: u32) -> Result<GcStepOutcome> {
        if !self.is_healthy() {
            return Ok(GcStepOutcome::SkippedFenced);
        }
        let Ok(_gate) = self.inner.mutation_gate.try_write() else {
            return Ok(GcStepOutcome::SkippedBusy);
        };
        if !self.is_healthy() {
            return Ok(GcStepOutcome::SkippedFenced);
        }
        match tokio::time::timeout(
            VALKEY_EVENT_MUTATION_TIMEOUT,
            self.inner.backend.gc_step(inspection_budget),
        )
        .await
        {
            Ok(result) => result.map(GcStepOutcome::Completed),
            Err(_) => bail!(
                "Valkey DYNKV.GC exceeded the {}ms lifecycle deadline",
                VALKEY_EVENT_MUTATION_TIMEOUT.as_millis()
            ),
        }
    }

    /// Registration teardown uses the same integrity state so a confirmed or
    /// ambiguous fault fence is never followed by a duplicate UNREGISTER. A
    /// second owner-fenced unregister can report STALE after the first one
    /// committed but lost its reply, creating a false shutdown failure.
    pub(super) async fn unregister_for_shutdown(&self) -> Result<()> {
        loop {
            match self.state() {
                IntegrityState::Faulted => {
                    // The processor may have been cancelled before it could
                    // claim the single fence. Claim it here; fence_once owns
                    // the same CAS and mutation gate as the normal path.
                    let _ = self.fence_once().await;
                    return Ok(());
                }
                IntegrityState::Fencing | IntegrityState::Fenced => {
                    tracing::warn!(
                        worker_id = self.inner.worker_id,
                        owner_nonce = self.inner.owner_nonce,
                        integrity_state = ?self.state(),
                        "Skipping duplicate Valkey worker unregister after an integrity fence; the prior attempt or lease expiry is authoritative"
                    );
                    return Ok(());
                }
                IntegrityState::Healthy => {
                    let gate = self.inner.mutation_gate.write().await;
                    if !self.is_healthy() {
                        // A fault raced the gate acquisition. Drop the gate and
                        // re-enter through the one-shot fence state machine.
                        drop(gate);
                        continue;
                    }
                    let result = self
                        .bounded_operation("UNREGISTER_WORKER", self.inner.backend.unregister())
                        .await;
                    // Shutdown never resumes this owner, whether the response
                    // was acknowledged or ambiguous.
                    self.inner.state.store(INTEGRITY_FENCED, Ordering::Release);
                    return result;
                }
            }
        }
    }
}

fn spawn_direct_event_batcher(
    integrity: Weak<ValkeyPublisherIntegrityInner>,
    batch_rx: mpsc::Receiver<DirectEventBatchRequest>,
) {
    tokio::spawn(run_direct_event_batcher(integrity, batch_rx));
}

async fn run_direct_event_batcher(
    integrity: Weak<ValkeyPublisherIntegrityInner>,
    mut batch_rx: mpsc::Receiver<DirectEventBatchRequest>,
) {
    while let Some(first) = batch_rx.recv().await {
        let mut requests = Vec::with_capacity(DIRECT_EVENT_BATCH_MAX_EVENTS);
        let mut saw_clear = first.event.storage_tier.is_gpu()
            && matches!(first.event.event.data, KvCacheEventData::Cleared);
        let mut saw_relay_only_barrier = !first.event.storage_tier.is_gpu();
        requests.push(first);

        let Some(inner) = integrity.upgrade() else {
            complete_direct_event_batch(
                requests,
                Err(anyhow::anyhow!("direct Valkey publisher was dropped")),
                None,
            );
            return;
        };
        let publisher = ValkeyPublisherIntegrity {
            inner: Arc::clone(&inner),
        };
        if !publisher.is_healthy() {
            while let Ok(request) = batch_rx.try_recv() {
                requests.push(request);
            }
            complete_direct_event_batch(
                requests,
                Err(anyhow::anyhow!(
                    "direct Valkey publisher is fenced by an integrity fault"
                )),
                Some(&publisher),
            );
            continue;
        }

        if !saw_clear && !saw_relay_only_barrier {
            let deadline = tokio::time::Instant::now() + DIRECT_EVENT_BATCH_COLLECT_WINDOW;
            while requests.len() < DIRECT_EVENT_BATCH_MAX_EVENTS {
                match tokio::time::timeout_at(deadline, batch_rx.recv()).await {
                    Ok(Some(request)) => {
                        saw_clear = request.event.storage_tier.is_gpu()
                            && matches!(request.event.event.data, KvCacheEventData::Cleared);
                        saw_relay_only_barrier = !request.event.storage_tier.is_gpu();
                        requests.push(request);
                        if saw_clear || saw_relay_only_barrier {
                            break;
                        }
                    }
                    Ok(None) | Err(_) => break,
                }
            }
        }

        if saw_clear {
            // CLEAR is worker-wide, but sibling rank events generated before
            // it may still be buffered outside this shared FIFO. Applying it
            // could therefore be followed by stale STORE events which
            // resurrect state. Fail closed: owner-fenced UNREGISTER is the
            // only operation that atomically retires every rank and causes
            // every processor to discard its bounded ingress.
            publisher.mark_fault("worker_clear");
            while let Ok(request) = batch_rx.try_recv() {
                requests.push(request);
            }
            complete_direct_event_batch(
                requests,
                Err(anyhow::anyhow!(
                    "worker-wide CLEAR requires direct Valkey owner fencing and restart"
                )),
                Some(&publisher),
            );
            continue;
        }

        // The executor owns this permit, not any caller. Dropping a caller's
        // oneshot cannot let fencing pass while the pipeline is still doing
        // wire I/O. The internal deadline drops the backend future (and its
        // taken socket) before this guard is released.
        let _gate = inner.mutation_gate.read().await;
        let result = if publisher.is_healthy() {
            let events = requests
                .iter()
                .filter(|request| request.event.storage_tier.is_gpu())
                .map(|request| request.event.clone())
                .collect::<Vec<_>>();
            if events.is_empty() {
                Ok(())
            } else {
                publisher.publish_events_after_gate(&events).await
            }
        } else {
            Err(anyhow::anyhow!(
                "direct Valkey publisher faulted before queued batch execution"
            ))
        };
        drop(_gate);
        complete_direct_event_batch(requests, result, Some(&publisher));
    }
}

fn complete_direct_event_batch(
    requests: Vec<DirectEventBatchRequest>,
    result: Result<()>,
    publisher: Option<&ValkeyPublisherIntegrity>,
) {
    let completed = requests.len();
    let error = result
        .err()
        .map(|error| Arc::<str>::from(format!("{error:#}")));
    for request in requests {
        let completion = match &error {
            Some(error) => Err(Arc::clone(error)),
            None => Ok(()),
        };
        if let Some(sender) = request.completion {
            let _ = sender.send(completion);
        }
    }
    if let Some(publisher) = publisher {
        publisher.complete_pending_events(completed);
    }
}

pub(super) struct EventPlanePublisher(pub(super) EventPublisher);

impl RouterEventSink for EventPlanePublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        self.0.publish(event)
    }
}

pub(super) struct JetStreamPublisher(pub(super) NatsQueue);

impl RouterEventSink for JetStreamPublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        NatsQueue::publish_event(&self.0, KV_EVENT_SUBJECT, event)
    }
}

/// Writes normalized, batched worker KV events directly to the shared
/// primary-backed Valkey index. The module replicates each mutation to its
/// standby; frontends only query the shared index in this mode.
#[derive(Clone)]
pub(super) struct ValkeyEventPublisher {
    integrity: ValkeyPublisherIntegrity,
}

impl ValkeyEventPublisher {
    pub(super) fn new(integrity: ValkeyPublisherIntegrity) -> Self {
        Self { integrity }
    }

    pub(super) fn integrity(&self) -> &ValkeyPublisherIntegrity {
        &self.integrity
    }
}

impl RouterEventSink for ValkeyEventPublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        self.integrity.publish_event(event)
    }
}

/// Enqueues GPU events for primary-backed, replica-acknowledged Valkey batches.
pub(super) struct QueuedValkeyPublisher {
    valkey: ValkeyEventPublisher,
}

impl QueuedValkeyPublisher {
    pub(super) fn new(valkey: ValkeyEventPublisher) -> Self {
        Self { valkey }
    }
}

impl RouterEventSink for QueuedValkeyPublisher {
    async fn publish_event(&self, event: &RouterEvent) -> Result<()> {
        self.valkey
            .integrity
            .enqueue_event_without_relay(event)
            .await
    }
}

pub(super) async fn emit<P: RouterEventSink>(
    publisher: &P,
    local_indexer: &Option<Arc<LocalKvIndexer>>,
    worker_id: u64,
    storage_tier: StorageTier,
    event: KvCacheEvent,
) {
    let router_event = RouterEvent::with_storage_tier(worker_id, event, storage_tier);
    if let Some(indexer) = local_indexer
        && let Err(error) = indexer.apply_event_with_buffer(router_event.clone()).await
    {
        tracing::warn!(worker_id, error = %error, "Failed to apply event to local indexer");
    }
    if let Err(error) = publisher.publish_event(&router_event).await {
        tracing::error!(worker_id, error = %error, "Failed to publish event");
    }
}

#[cfg(test)]
mod tests;
