// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
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

struct PublisherLane {
    sender: mpsc::Sender<EventEnvelope>,
    cancel: CancellationToken,
    handle: JoinHandle<()>,
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
    consumer: C,
    metrics: Arc<KvZmqIngressMetrics>,
}

impl<C: PublisherBatchConsumer> PublisherLanes<C> {
    fn new(consumer: C, metrics: Arc<KvZmqIngressMetrics>) -> Self {
        Self {
            lanes: HashMap::new(),
            consumer,
            metrics,
        }
    }

    fn dispatch(&mut self, envelope: EventEnvelope, active_publishers: &HashSet<u64>) {
        let publisher_id = envelope.publisher_id;
        if !active_publishers.contains(&publisher_id) {
            self.metrics.increment_lifecycle("inactive_publisher");
            return;
        }

        if !self.lanes.contains_key(&publisher_id) {
            self.lanes.insert(publisher_id, self.spawn(publisher_id));
        }

        let result = self
            .lanes
            .get(&publisher_id)
            .expect("publisher lane was just inserted")
            .sender
            .try_send(envelope);
        match result {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.metrics.increment_lifecycle("queue_full");
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.metrics.increment_lifecycle("lane_closed");
                if let Some(lane) = self.lanes.remove(&publisher_id) {
                    lane.cancel.cancel();
                    lane.handle.abort();
                    self.metrics.decrement_sources("active");
                }
            }
        }
    }

    fn spawn(&self, publisher_id: u64) -> PublisherLane {
        let (sender, receiver) = mpsc::channel(PUBLISHER_LANE_CAPACITY);
        let cancel = CancellationToken::new();
        let handle = tokio::spawn(run_publisher_lane(
            publisher_id,
            receiver,
            self.consumer.clone(),
            self.metrics.clone(),
            cancel.clone(),
        ));
        self.metrics.increment_sources("active");
        self.metrics.increment_lifecycle("started");
        PublisherLane {
            sender,
            cancel,
            handle,
        }
    }

    async fn reconcile(&mut self, active_publishers: &HashSet<u64>) {
        let obsolete = self
            .lanes
            .keys()
            .filter(|publisher_id| !active_publishers.contains(publisher_id))
            .copied()
            .collect::<Vec<_>>();
        self.stop(obsolete).await;
    }

    async fn shutdown(mut self) {
        let publisher_ids = self.lanes.keys().copied().collect::<Vec<_>>();
        self.stop(publisher_ids).await;
    }

    async fn stop(&mut self, publisher_ids: Vec<u64>) {
        let mut removed = Vec::with_capacity(publisher_ids.len());
        for publisher_id in publisher_ids {
            if let Some(lane) = self.lanes.remove(&publisher_id) {
                lane.cancel.cancel();
                removed.push(lane);
            }
        }
        for lane in removed {
            stop_publisher_lane(lane, &self.metrics).await;
        }
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

        let current_view = membership_watch.borrow().clone();
        if current_view.resolved_kv_state_endpoint()
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
    let mut lanes = PublisherLanes::new(consumer, ingress_metrics.clone());
    let exit = loop {
        tokio::select! {
            biased;
            _ = cancellation_token.cancelled() => break ScopeExit::Stop,
            changed = membership_watch.changed() => {
                if changed.is_err() {
                    break ScopeExit::Stop;
                }
                membership_watch.borrow_and_update();
                let view = client.sync_membership().await;
                update_mismatch_metric(
                    status_metrics,
                    &view,
                    model,
                    worker_type,
                    serving_endpoint,
                    metric_scope,
                );
                if view.resolved_kv_state_endpoint() != Some(kv_state_endpoint) {
                    break ScopeExit::Rebind;
                }
                active_publishers = self::active_publishers(&view);
                lanes.reconcile(&active_publishers).await;
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
                        tracing::warn!(%error, %kv_state_endpoint, "Failed to decode brokered KV event envelope");
                        ingress_metrics.increment_lifecycle("envelope_decode_error");
                    }
                }
            }
        }
    };
    lanes.shutdown().await;
    exit
}

async fn run_publisher_lane<C: PublisherBatchConsumer>(
    publisher_id: u64,
    mut receiver: mpsc::Receiver<EventEnvelope>,
    consumer: C,
    metrics: Arc<KvZmqIngressMetrics>,
    cancellation_token: CancellationToken,
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
) {
    match *high_watermark {
        None => {
            if sequence > 0 {
                metrics.increment_lifecycle_by("sequence_gap", sequence);
            }
            *high_watermark = Some(sequence);
        }
        Some(previous) if sequence <= previous => {
            metrics.increment_lifecycle("out_of_order");
        }
        Some(previous) => {
            let missing = sequence - previous - 1;
            if missing > 0 {
                metrics.increment_lifecycle_by("sequence_gap", missing);
            }
            *high_watermark = Some(sequence);
        }
    }
}

fn active_publishers(view: &KvSourceMembershipView) -> HashSet<u64> {
    view.sources
        .values()
        .filter_map(|status| status.active_source())
        .map(|source| source.publisher_id)
        .collect()
}

async fn stop_publisher_lane(mut lane: PublisherLane, metrics: &KvZmqIngressMetrics) {
    match tokio::time::timeout(PUBLISHER_JOIN_TIMEOUT, &mut lane.handle).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) if error.is_cancelled() => {}
        Ok(Err(error)) => tracing::warn!(%error, "Brokered-ZMQ publisher lane failed"),
        Err(_) => {
            lane.handle.abort();
            let _ = lane.handle.await;
            metrics.increment_lifecycle("forced_abort");
        }
    }
    metrics.decrement_sources("active");
    metrics.increment_lifecycle("stopped");
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
        let mut lanes = PublisherLanes::new(consumer.clone(), test_metrics().await);
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
        let mut lanes = PublisherLanes::new(consumer.clone(), test_metrics().await);
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
    async fn membership_reconcile_removes_inactive_lane() {
        let consumer = TestConsumer::new(None);
        let mut lanes = PublisherLanes::new(consumer.clone(), test_metrics().await);
        lanes.dispatch(envelope(7, 0), &HashSet::from([7]));
        tokio::time::timeout(Duration::from_secs(1), consumer.other_admitted.notified())
            .await
            .unwrap();

        lanes.reconcile(&HashSet::new()).await;
        assert!(lanes.lanes.is_empty());
        lanes.dispatch(envelope(7, 1), &HashSet::new());
        assert!(lanes.lanes.is_empty());
        lanes.shutdown().await;
    }
}
