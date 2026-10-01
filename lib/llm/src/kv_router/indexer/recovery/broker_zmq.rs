// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet, hash_map::Entry},
    future::Future,
    sync::Arc,
    time::Duration,
};

use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, RouterEvent};
use dynamo_runtime::{
    component::Component,
    discovery::EventTransportKind,
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{Codec, EventEnvelope, EventSubscriber},
};
use futures::future::join_all;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

use super::{
    IndexerRecoveryTarget, RecoveryTarget,
    subscriber::{
        MismatchMetricScope, clear_mismatch_metric_on_cancellation, update_mismatch_metric,
        update_subscription_failure_metric,
    },
    worker_query::WorkerQueryClient,
};
use crate::{
    discovery::{KvSourceMembershipView, KvSourceMembershipWatch},
    kv_router::metrics::{KvZmqIngressMetrics, RouterWorkerStatusMetrics},
};

const INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const MAX_BACKOFF: Duration = Duration::from_secs(5);
const PUBLISHER_LANE_CAPACITY: usize = 64;
const PUBLISHER_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

enum ScopeExit {
    Rebind,
    Retry,
    Stop,
}

enum MembershipUpdate {
    View(Box<KvSourceMembershipView>),
    Rebind,
    Stop,
}

struct PublisherLane {
    sender: mpsc::Sender<EventEnvelope>,
    cancel: CancellationToken,
    handle: JoinHandle<()>,
    queue_full_warned: bool,
}

trait PublisherBatchConsumer: Clone + Send + Sync + 'static {
    fn consume(
        &self,
        publisher_id: u64,
        envelope: EventEnvelope,
    ) -> impl Future<Output = ()> + Send;
}

struct KvBatchConsumer<T: RecoveryTarget> {
    client: Arc<WorkerQueryClient<T>>,
    metrics: Arc<KvZmqIngressMetrics>,
}

impl<T: RecoveryTarget> Clone for KvBatchConsumer<T> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

impl<T: RecoveryTarget> PublisherBatchConsumer for KvBatchConsumer<T> {
    async fn consume(&self, publisher_id: u64, envelope: EventEnvelope) {
        let events = match Codec::default().decode_payload::<Vec<RouterEvent>>(&envelope.payload) {
            Ok(events) => events,
            Err(error) => {
                tracing::warn!(%error, publisher_id, "Failed to decode brokered-ZMQ KV payload");
                self.metrics.increment_lifecycle("payload_decode_error");
                return;
            }
        };
        self.client.handle_live_batch(publisher_id, events).await;
        self.metrics.increment_batch();
    }
}

struct PublisherLanes<C: PublisherBatchConsumer> {
    lanes: HashMap<u64, PublisherLane>,
    retired: Vec<JoinHandle<()>>,
    consumer: C,
    metrics: Arc<KvZmqIngressMetrics>,
    cancellation_token: CancellationToken,
}

impl<C: PublisherBatchConsumer> PublisherLanes<C> {
    fn new(
        consumer: C,
        metrics: Arc<KvZmqIngressMetrics>,
        supervisor_token: &CancellationToken,
    ) -> Self {
        Self {
            lanes: HashMap::new(),
            retired: Vec::new(),
            consumer,
            metrics,
            cancellation_token: supervisor_token.child_token(),
        }
    }

    fn dispatch(&mut self, envelope: EventEnvelope, active_publishers: &HashSet<u64>) {
        let publisher_id = envelope.publisher_id;
        if !active_publishers.contains(&publisher_id) {
            self.metrics.increment_lifecycle("inactive_publisher");
            return;
        }

        let consumer = self.consumer.clone();
        let metrics = self.metrics.clone();
        let cancellation_token = self.cancellation_token.child_token();
        let lane = match self.lanes.entry(publisher_id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(spawn_publisher_lane(
                publisher_id,
                consumer,
                metrics,
                cancellation_token,
            )),
        };

        let closed = match lane.sender.try_send(envelope) {
            Ok(()) => false,
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.metrics.increment_lifecycle("queue_full");
                if !lane.queue_full_warned {
                    tracing::warn!(
                        publisher_id,
                        capacity = PUBLISHER_LANE_CAPACITY,
                        "Brokered-ZMQ publisher lane is full; dropping the newest batch"
                    );
                    lane.queue_full_warned = true;
                }
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.metrics.increment_lifecycle("lane_closed");
                true
            }
        };
        if closed {
            self.retire(publisher_id);
        }
    }

    fn reconcile(&mut self, active_publishers: &HashSet<u64>) {
        let obsolete = self
            .lanes
            .keys()
            .filter(|publisher_id| !active_publishers.contains(publisher_id))
            .copied()
            .collect::<Vec<_>>();
        for publisher_id in obsolete {
            self.retire(publisher_id);
        }
    }

    async fn shutdown(mut self) {
        self.cancellation_token.cancel();
        let publisher_ids = self.lanes.keys().copied().collect::<Vec<_>>();
        for publisher_id in publisher_ids {
            self.retire(publisher_id);
        }

        let handles = std::mem::take(&mut self.retired);
        if handles.is_empty() {
            return;
        }
        match tokio::time::timeout(PUBLISHER_JOIN_TIMEOUT, join_all(handles)).await {
            Ok(results) => {
                for result in results {
                    if let Err(error) = result
                        && !error.is_cancelled()
                    {
                        tracing::warn!(%error, "Brokered-ZMQ publisher lane failed");
                    }
                }
            }
            Err(_) => {
                // Dropping JoinHandle detaches the task. It may finish its current batch,
                // but it is never aborted between index admission and state commit.
                self.metrics.increment_lifecycle("join_timeout");
            }
        }
    }

    fn retire(&mut self, publisher_id: u64) {
        if let Some(lane) = self.lanes.remove(&publisher_id) {
            lane.cancel.cancel();
            self.retired.push(lane.handle);
        }
    }
}

impl<C: PublisherBatchConsumer> Drop for PublisherLanes<C> {
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

fn spawn_publisher_lane<C: PublisherBatchConsumer>(
    publisher_id: u64,
    consumer: C,
    metrics: Arc<KvZmqIngressMetrics>,
    cancellation_token: CancellationToken,
) -> PublisherLane {
    let (sender, receiver) = mpsc::channel(PUBLISHER_LANE_CAPACITY);
    metrics.increment_sources("active");
    metrics.increment_lifecycle("started");
    let metrics_guard = LaneMetricsGuard(metrics.clone());
    let handle = tokio::spawn(run_publisher_lane(
        publisher_id,
        receiver,
        consumer,
        metrics,
        cancellation_token.clone(),
        metrics_guard,
    ));
    PublisherLane {
        sender,
        cancel: cancellation_token,
        handle,
        queue_full_warned: false,
    }
}

struct LaneMetricsGuard(Arc<KvZmqIngressMetrics>);

impl Drop for LaneMetricsGuard {
    fn drop(&mut self) {
        self.0.decrement_sources("active");
        self.0.increment_lifecycle("stopped");
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn run_broker_zmq_supervisor(
    component: Component,
    serving_endpoint: EndpointId,
    client: Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    mut membership_watch: KvSourceMembershipWatch,
    model: String,
    worker_type: &'static str,
    metric_scope: MismatchMetricScope,
    cancellation_token: CancellationToken,
    mut startup_ready: Option<oneshot::Sender<()>>,
) {
    let status_metrics = RouterWorkerStatusMetrics::from_component(&component);
    let ingress_metrics = KvZmqIngressMetrics::from_component(&component);
    let mut retry_delay = INITIAL_BACKOFF;

    loop {
        let view = membership_watch.borrow_and_update().clone();
        update_mismatch_metric(
            &status_metrics,
            &view,
            &model,
            worker_type,
            &serving_endpoint,
            metric_scope,
        );

        let subscriber = if let Some(kv_state_endpoint) = view.resolved_kv_state_endpoint() {
            match EventSubscriber::for_endpoint_id_with_transport(
                component.drt(),
                kv_state_endpoint,
                KV_EVENT_SUBJECT,
                EventTransportKind::Zmq,
            )
            .await
            {
                Ok(subscriber) => Some((kv_state_endpoint.clone(), subscriber)),
                Err(error) => {
                    tracing::error!(%error, %kv_state_endpoint, "Failed to subscribe to brokered KV events");
                    ingress_metrics.increment_lifecycle("connect_error");
                    update_subscription_failure_metric(
                        &status_metrics,
                        &view,
                        &model,
                        worker_type,
                        &serving_endpoint,
                        metric_scope,
                    );
                    if let Some(ready) = startup_ready.take() {
                        let _ = ready.send(());
                    }
                    if !wait_for_retry(retry_delay, &mut membership_watch, &cancellation_token)
                        .await
                    {
                        break;
                    }
                    retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
                    continue;
                }
            }
        } else {
            tracing::error!(
                serving_endpoint = %serving_endpoint,
                resolution = ?view.endpoint_resolution,
                "KV event handling disabled because active base cards disagree on their KV-state endpoint"
            );
            None
        };

        if membership_watch.borrow().resolved_kv_state_endpoint()
            != subscriber.as_ref().map(|(endpoint, _)| endpoint)
        {
            continue;
        }
        let view = client.sync_membership().await;
        if let Some(ready) = startup_ready.take() {
            let _ = ready.send(());
        }

        let Some((kv_state_endpoint, subscriber)) = subscriber else {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                result = membership_watch.changed() => {
                    if result.is_err() {
                        break;
                    }
                }
            }
            continue;
        };

        match consume_scope(
            subscriber,
            &client,
            &kv_state_endpoint,
            &mut membership_watch,
            &status_metrics,
            &ingress_metrics,
            &model,
            worker_type,
            &serving_endpoint,
            metric_scope,
            &cancellation_token,
            &mut retry_delay,
            active_publishers(&view),
        )
        .await
        {
            ScopeExit::Rebind => retry_delay = INITIAL_BACKOFF,
            ScopeExit::Retry => {
                ingress_metrics.increment_lifecycle("reconnect");
                let view = client.sync_membership().await;
                update_subscription_failure_metric(
                    &status_metrics,
                    &view,
                    &model,
                    worker_type,
                    &serving_endpoint,
                    metric_scope,
                );
                if !wait_for_retry(retry_delay, &mut membership_watch, &cancellation_token).await {
                    break;
                }
                retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
            }
            ScopeExit::Stop => break,
        }
    }

    client.shutdown().await;
    clear_mismatch_metric_on_cancellation(
        &status_metrics,
        &cancellation_token,
        &model,
        worker_type,
        &serving_endpoint,
    );
}

#[allow(clippy::too_many_arguments)]
async fn consume_scope<T: RecoveryTarget>(
    mut subscriber: EventSubscriber,
    client: &Arc<WorkerQueryClient<T>>,
    kv_state_endpoint: &EndpointId,
    membership_watch: &mut KvSourceMembershipWatch,
    status_metrics: &RouterWorkerStatusMetrics,
    ingress_metrics: &Arc<KvZmqIngressMetrics>,
    model: &str,
    worker_type: &str,
    serving_endpoint: &EndpointId,
    metric_scope: MismatchMetricScope,
    cancellation_token: &CancellationToken,
    retry_delay: &mut Duration,
    mut active_publishers: HashSet<u64>,
) -> ScopeExit {
    let consumer = KvBatchConsumer {
        client: client.clone(),
        metrics: ingress_metrics.clone(),
    };
    let mut lanes = PublisherLanes::new(consumer, ingress_metrics.clone(), cancellation_token);
    let membership_cancel = cancellation_token.child_token();
    let (membership_tx, mut membership_rx) = mpsc::channel(1);
    let membership_handle = tokio::spawn(run_membership_updates(
        client.clone(),
        membership_watch.fork_receiver(),
        kv_state_endpoint.clone(),
        membership_tx,
        membership_cancel.clone(),
    ));
    let exit = loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => break ScopeExit::Stop,
            update = membership_rx.recv() => {
                if update.is_some() {
                    membership_watch.borrow_and_update();
                }
                match update {
                    Some(MembershipUpdate::View(view)) => {
                        update_mismatch_metric(
                            status_metrics,
                            &view,
                            model,
                            worker_type,
                            serving_endpoint,
                            metric_scope,
                        );
                        active_publishers = self::active_publishers(&view);
                        lanes.reconcile(&active_publishers);
                    }
                    Some(MembershipUpdate::Rebind) => break ScopeExit::Rebind,
                    Some(MembershipUpdate::Stop) => break ScopeExit::Stop,
                    None => {
                        tracing::error!(%kv_state_endpoint, "Brokered KV membership task ended unexpectedly");
                        break ScopeExit::Retry;
                    }
                }
            }
            result = subscriber.next() => {
                let Some(result) = result else {
                    tracing::error!(%kv_state_endpoint, "Brokered KV event stream ended unexpectedly");
                    break ScopeExit::Retry;
                };
                *retry_delay = INITIAL_BACKOFF;
                match result {
                    Ok(envelope) => lanes.dispatch(envelope, &active_publishers),
                    Err(error) => {
                        tracing::warn!(%error, %kv_state_endpoint, "Failed to receive or decode brokered KV event envelope");
                        ingress_metrics.increment_lifecycle("stream_error");
                    }
                }
            }
        }
    };
    membership_cancel.cancel();
    drop(membership_rx);
    if let Err(error) = membership_handle.await
        && !error.is_cancelled()
    {
        tracing::warn!(%error, "Brokered KV membership task failed");
    }
    lanes.shutdown().await;
    exit
}

async fn run_membership_updates<T: RecoveryTarget>(
    client: Arc<WorkerQueryClient<T>>,
    mut membership_watch: KvSourceMembershipWatch,
    kv_state_endpoint: EndpointId,
    updates: mpsc::Sender<MembershipUpdate>,
    cancellation_token: CancellationToken,
) {
    loop {
        let changed = tokio::select! {
            _ = cancellation_token.cancelled() => return,
            changed = membership_watch.changed() => changed,
        };
        if changed.is_err() {
            let _ = updates.send(MembershipUpdate::Stop).await;
            return;
        }

        membership_watch.borrow_and_update();
        if membership_watch.borrow().resolved_kv_state_endpoint() != Some(&kv_state_endpoint) {
            let _ = updates.send(MembershipUpdate::Rebind).await;
            return;
        }

        // This may reset or recover ranks. Keep it outside the socket reader and do not
        // cancel it between index admission and the matching state commit.
        let view = client.sync_membership().await;
        let update = if view.resolved_kv_state_endpoint() == Some(&kv_state_endpoint) {
            MembershipUpdate::View(Box::new(view))
        } else {
            MembershipUpdate::Rebind
        };
        let sent = tokio::select! {
            _ = cancellation_token.cancelled() => return,
            sent = updates.send(update) => sent,
        };
        if sent.is_err() {
            return;
        }
    }
}

async fn run_publisher_lane<C: PublisherBatchConsumer>(
    publisher_id: u64,
    mut receiver: mpsc::Receiver<EventEnvelope>,
    consumer: C,
    metrics: Arc<KvZmqIngressMetrics>,
    cancellation_token: CancellationToken,
    _metrics_guard: LaneMetricsGuard,
) {
    let mut high_watermark = None;
    loop {
        let envelope = tokio::select! {
            biased;
            _ = cancellation_token.cancelled() => break,
            envelope = receiver.recv() => {
                let Some(envelope) = envelope else {
                    break;
                };
                envelope
            }
        };

        observe_sequence(envelope.sequence, &mut high_watermark, &metrics);
        consumer.consume(publisher_id, envelope).await;
    }
}

fn observe_sequence(
    sequence: u64,
    high_watermark: &mut Option<u64>,
    metrics: &KvZmqIngressMetrics,
) -> (u64, bool) {
    let observation = match *high_watermark {
        None => {
            *high_watermark = Some(sequence);
            (0, false)
        }
        Some(previous) if sequence <= previous => (0, true),
        Some(previous) => {
            let missing = sequence - previous - 1;
            *high_watermark = Some(sequence);
            (missing, false)
        }
    };
    if observation.0 > 0 {
        metrics.increment_lifecycle_by("sequence_gap", observation.0);
    }
    if observation.1 {
        metrics.increment_lifecycle("out_of_order");
    }
    observation
}

fn active_publishers(view: &KvSourceMembershipView) -> HashSet<u64> {
    view.sources
        .values()
        .filter_map(|status| status.active_source())
        .map(|source| source.publisher_id)
        .collect()
}

async fn wait_for_retry(
    delay: Duration,
    membership_watch: &mut KvSourceMembershipWatch,
    cancellation_token: &CancellationToken,
) -> bool {
    tokio::select! {
        _ = cancellation_token.cancelled() => false,
        changed = membership_watch.changed() => changed.is_ok(),
        _ = tokio::time::sleep(delay) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use tokio::sync::{Mutex, Notify};

    #[derive(Clone)]
    struct TestConsumer {
        blocked_publisher: Option<u64>,
        blocked_started: Arc<Notify>,
        blocked_release: Arc<Notify>,
        other_admitted: Arc<Notify>,
        seen: Arc<Mutex<Vec<(u64, u64)>>>,
    }

    impl TestConsumer {
        fn new(blocked_publisher: Option<u64>) -> Self {
            Self {
                blocked_publisher,
                blocked_started: Arc::new(Notify::new()),
                blocked_release: Arc::new(Notify::new()),
                other_admitted: Arc::new(Notify::new()),
                seen: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl PublisherBatchConsumer for TestConsumer {
        async fn consume(&self, publisher_id: u64, envelope: EventEnvelope) {
            if self.blocked_publisher == Some(publisher_id) && envelope.sequence == 0 {
                self.blocked_started.notify_one();
                self.blocked_release.notified().await;
            }
            self.seen
                .lock()
                .await
                .push((publisher_id, envelope.sequence));
            if self.blocked_publisher != Some(publisher_id) {
                self.other_admitted.notify_one();
            }
        }
    }

    async fn test_metrics() -> Arc<KvZmqIngressMetrics> {
        let drt = DistributedRuntime::new(
            Runtime::from_current().unwrap(),
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let component = drt
            .namespace("broker-zmq-lane-test")
            .unwrap()
            .component("router")
            .unwrap();
        KvZmqIngressMetrics::from_component(&component)
    }

    fn envelope(publisher_id: u64, sequence: u64) -> EventEnvelope {
        EventEnvelope {
            publisher_id,
            sequence,
            published_at: 0,
            topic: KV_EVENT_SUBJECT.to_string(),
            payload: Bytes::new(),
        }
    }

    #[tokio::test]
    async fn slow_publisher_does_not_block_sibling_and_order_is_preserved() {
        let consumer = TestConsumer::new(Some(1));
        let cancel = CancellationToken::new();
        let mut lanes = PublisherLanes::new(consumer.clone(), test_metrics().await, &cancel);
        let active = HashSet::from([1, 2]);

        lanes.dispatch(envelope(1, 0), &active);
        tokio::time::timeout(Duration::from_secs(1), consumer.blocked_started.notified())
            .await
            .unwrap();
        lanes.dispatch(envelope(1, 1), &active);
        lanes.dispatch(envelope(2, 0), &active);

        tokio::time::timeout(Duration::from_secs(1), consumer.other_admitted.notified())
            .await
            .expect("a sibling publisher should progress independently");
        consumer.blocked_release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if consumer.seen.lock().await.len() == 3 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let seen = consumer.seen.lock().await.clone();
        let first = seen.iter().position(|item| *item == (1, 0)).unwrap();
        let second = seen.iter().position(|item| *item == (1, 1)).unwrap();
        assert!(first < second);
        lanes.shutdown().await;
    }

    #[tokio::test]
    async fn full_lane_drops_only_its_newest_batch() {
        let consumer = TestConsumer::new(Some(1));
        let cancel = CancellationToken::new();
        let mut lanes = PublisherLanes::new(consumer.clone(), test_metrics().await, &cancel);
        let active = HashSet::from([1]);

        lanes.dispatch(envelope(1, 0), &active);
        tokio::time::timeout(Duration::from_secs(1), consumer.blocked_started.notified())
            .await
            .unwrap();
        for sequence in 1..=PUBLISHER_LANE_CAPACITY as u64 {
            lanes.dispatch(envelope(1, sequence), &active);
        }
        assert_eq!(lanes.lanes.get(&1).unwrap().sender.capacity(), 0);
        lanes.dispatch(envelope(1, PUBLISHER_LANE_CAPACITY as u64 + 1), &active);

        consumer.blocked_release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if consumer.seen.lock().await.len() == PUBLISHER_LANE_CAPACITY + 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(
            !consumer
                .seen
                .lock()
                .await
                .contains(&(1, PUBLISHER_LANE_CAPACITY as u64 + 1))
        );
        lanes.shutdown().await;
    }

    #[tokio::test]
    async fn membership_reconcile_finishes_current_batch_without_blocking() {
        let consumer = TestConsumer::new(Some(7));
        let cancel = CancellationToken::new();
        let mut lanes = PublisherLanes::new(consumer.clone(), test_metrics().await, &cancel);
        lanes.dispatch(envelope(7, 0), &HashSet::from([7]));
        tokio::time::timeout(Duration::from_secs(1), consumer.blocked_started.notified())
            .await
            .unwrap();

        lanes.reconcile(&HashSet::new());
        assert!(lanes.lanes.is_empty());
        lanes.dispatch(envelope(7, 1), &HashSet::new());
        assert!(lanes.lanes.is_empty());

        consumer.blocked_release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if consumer.seen.lock().await.contains(&(7, 0)) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the admitted batch should finish after its lane is retired");
        lanes.shutdown().await;
    }

    #[tokio::test]
    async fn supervisor_cancellation_reaches_publisher_lanes() {
        let consumer = TestConsumer::new(None);
        let cancel = CancellationToken::new();
        let mut lanes = PublisherLanes::new(consumer.clone(), test_metrics().await, &cancel);
        lanes.dispatch(envelope(7, 0), &HashSet::from([7]));
        tokio::time::timeout(Duration::from_secs(1), consumer.other_admitted.notified())
            .await
            .unwrap();

        let mut lane = lanes.lanes.remove(&7).unwrap();
        cancel.cancel();
        assert!(lane.cancel.is_cancelled());
        tokio::time::timeout(Duration::from_secs(1), &mut lane.handle)
            .await
            .expect("publisher lane should stop after supervisor cancellation")
            .expect("publisher lane should exit cleanly");
        lanes.shutdown().await;
    }

    #[tokio::test]
    async fn first_sequence_seeds_watermark_without_counting_a_gap() {
        let metrics = test_metrics().await;
        let mut high_watermark = None;
        let gaps_before = metrics.lifecycle_count("sequence_gap");
        let out_of_order_before = metrics.lifecycle_count("out_of_order");

        assert_eq!(
            observe_sequence(1_000_000, &mut high_watermark, &metrics),
            (0, false)
        );
        assert_eq!(metrics.lifecycle_count("sequence_gap"), gaps_before);
        assert_eq!(
            observe_sequence(1_000_003, &mut high_watermark, &metrics),
            (2, false)
        );
        assert_eq!(
            observe_sequence(1_000_002, &mut high_watermark, &metrics),
            (0, true)
        );
        assert_eq!(high_watermark, Some(1_000_003));
        assert_eq!(metrics.lifecycle_count("sequence_gap"), gaps_before + 2);
        assert_eq!(
            metrics.lifecycle_count("out_of_order"),
            out_of_order_before + 1
        );
    }
}
