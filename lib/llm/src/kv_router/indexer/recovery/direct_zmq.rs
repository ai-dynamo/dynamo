// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, RouterEvent};
use dynamo_runtime::{
    component::Component,
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
        EventChannelInstanceId, EventChannelQuery, EventTransport,
    },
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::event_plane::zmq_transport::ZmqWireStream,
    transports::event_plane::{Codec, EventScope, ZmqSubTransport},
};
use futures::StreamExt;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

use super::{
    IndexerRecoveryTarget,
    subscriber::{update_mismatch_metric, update_subscription_failure_metric},
    worker_query::{ReadyKvSource, WorkerQueryClient},
};
use crate::{
    discovery::{KvSourceMembershipView, KvSourceMembershipWatch},
    kv_router::metrics::{KvZmqIngressMetrics, RouterWorkerStatusMetrics},
};

const INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const MAX_BACKOFF: Duration = Duration::from_secs(5);
const SOURCE_JOIN_TIMEOUT: Duration = Duration::from_secs(5);
const SIGNAL_CAPACITY: usize = 1024;

enum ScopeExit {
    Rebind,
    Retry,
    Stop,
}

enum SourceSignal {
    Ready {
        publisher_id: u64,
        task_generation: u64,
        activate: oneshot::Sender<()>,
    },
    Disconnected {
        publisher_id: u64,
        task_generation: u64,
    },
}

struct SourceTask {
    endpoint: String,
    task_generation: u64,
    cancel: CancellationToken,
    handle: JoinHandle<()>,
    state: SourceState,
}

enum SourceState {
    Connecting,
    Preconnected { activate: oneshot::Sender<()> },
    Active { bindings: HashSet<ReadyKvSource> },
    Fenced,
}

impl SourceState {
    fn is_ready(&self) -> bool {
        matches!(self, Self::Preconnected { .. } | Self::Active { .. })
    }

    fn active_bindings(&self) -> Option<&HashSet<ReadyKvSource>> {
        match self {
            Self::Active { bindings } => Some(bindings),
            _ => None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn run_direct_zmq_supervisor(
    component: Component,
    serving_endpoint: EndpointId,
    client: Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    mut membership_watch: KvSourceMembershipWatch,
    model: String,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
    mut startup_ready: Option<oneshot::Sender<Result<(), String>>>,
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
        );

        let Some(kv_state_endpoint) = view.resolved_kv_state_endpoint().cloned() else {
            client
                .sync_membership_with_ready_sources(&HashSet::new())
                .await;
            if let Some(ready) = startup_ready.take() {
                let _ = ready.send(Ok(()));
            }
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                changed = membership_watch.changed() => {
                    if changed.is_err() {
                        break;
                    }
                }
            }
            continue;
        };

        let scope_cancel = cancellation_token.child_token();
        let query = DiscoveryQuery::EventChannels(EventChannelQuery::endpoint_topic(
            kv_state_endpoint.clone(),
            KV_EVENT_SUBJECT,
        ));
        let stream = match component
            .drt()
            .discovery()
            .list_and_watch(query, Some(scope_cancel.clone()))
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                tracing::error!(%error, %kv_state_endpoint, "Failed to watch direct-ZMQ KV event channels");
                update_subscription_failure_metric(
                    &status_metrics,
                    &view,
                    &model,
                    worker_type,
                    &serving_endpoint,
                );
                client
                    .sync_membership_with_ready_sources(&HashSet::new())
                    .await;
                if let Some(ready) = startup_ready.take() {
                    let _ = ready.send(Ok(()));
                }
                if !wait_for_retry(retry_delay, &mut membership_watch, &cancellation_token).await {
                    break;
                }
                retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
                continue;
            }
        };

        client
            .sync_membership_with_ready_sources(&HashSet::new())
            .await;
        if let Some(ready) = startup_ready.take() {
            let _ = ready.send(Ok(()));
        }

        let exit = consume_scope(
            stream,
            &client,
            &kv_state_endpoint,
            &mut membership_watch,
            &status_metrics,
            &ingress_metrics,
            &model,
            worker_type,
            &serving_endpoint,
            &cancellation_token,
        )
        .await;
        scope_cancel.cancel();

        match exit {
            ScopeExit::Rebind => retry_delay = INITIAL_BACKOFF,
            ScopeExit::Retry => {
                let view = membership_watch.borrow().clone();
                update_subscription_failure_metric(
                    &status_metrics,
                    &view,
                    &model,
                    worker_type,
                    &serving_endpoint,
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
}

#[allow(clippy::too_many_arguments)]
async fn consume_scope(
    mut discovery_stream: dynamo_runtime::discovery::DiscoveryStream,
    client: &Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    kv_state_endpoint: &EndpointId,
    membership_watch: &mut KvSourceMembershipWatch,
    status_metrics: &RouterWorkerStatusMetrics,
    ingress_metrics: &Arc<KvZmqIngressMetrics>,
    model: &str,
    worker_type: &str,
    serving_endpoint: &EndpointId,
    cancellation_token: &CancellationToken,
) -> ScopeExit {
    let expected_scope = EventScope::Endpoint {
        endpoint: kv_state_endpoint.clone(),
    };
    let (signal_tx, mut signal_rx) = mpsc::channel(SIGNAL_CAPACITY);
    let mut sources = HashMap::<u64, SourceTask>::new();
    let mut invalid_publishers = HashSet::new();
    let mut next_task_generation = 1_u64;
    let mut exit = ScopeExit::Retry;

    loop {
        tokio::select! {
            biased;
            _ = cancellation_token.cancelled() => {
                exit = ScopeExit::Stop;
                break;
            }
            changed = membership_watch.changed() => {
                if changed.is_err() {
                    exit = ScopeExit::Stop;
                    break;
                }
                membership_watch.borrow_and_update();
                if membership_watch.borrow().resolved_kv_state_endpoint() != Some(kv_state_endpoint) {
                    exit = ScopeExit::Rebind;
                    break;
                }
                let view = membership_watch.borrow().clone();
                reconcile_sources(
                    client,
                    view.clone(),
                    &mut sources,
                    ingress_metrics,
                    &signal_tx,
                    cancellation_token,
                    &mut next_task_generation,
                ).await;
                update_mismatch_metric(
                    status_metrics,
                    &view,
                    model,
                    worker_type,
                    serving_endpoint,
                );
            }
            signal = signal_rx.recv() => {
                let Some(signal) = signal else {
                    break;
                };
                match signal {
                    SourceSignal::Ready { publisher_id, task_generation, activate } => {
                        let Some(source) = sources.get_mut(&publisher_id) else {
                            continue;
                        };
                        if source.task_generation != task_generation {
                            continue;
                        }
                        transition_source_state(
                            source,
                            SourceState::Preconnected { activate },
                            ingress_metrics,
                        );
                        ingress_metrics.increment_lifecycle("preconnected");
                        let view = membership_watch.borrow().clone();
                        reconcile_sources(
                            client,
                            view,
                            &mut sources,
                            ingress_metrics,
                            &signal_tx,
                            cancellation_token,
                            &mut next_task_generation,
                        ).await;
                    }
                    SourceSignal::Disconnected { publisher_id, task_generation } => {
                        let Some(source) = sources.get_mut(&publisher_id) else {
                            continue;
                        };
                        if source.task_generation != task_generation {
                            continue;
                        }
                        transition_source_state(source, SourceState::Fenced, ingress_metrics);
                        ingress_metrics.increment_lifecycle("reconnect");
                        client.fence_transport(publisher_id).await;
                        let view = membership_watch.borrow().clone();
                        reconcile_sources(
                            client,
                            view,
                            &mut sources,
                            ingress_metrics,
                            &signal_tx,
                            cancellation_token,
                            &mut next_task_generation,
                        ).await;
                    }
                }
            }
            event = discovery_stream.next() => {
                let Some(event) = event else {
                    tracing::error!(%kv_state_endpoint, "Direct-ZMQ event-channel discovery stream ended");
                    break;
                };
                match event {
                    Ok(DiscoveryEvent::Added(DiscoveryInstance::EventChannel {
                        scope,
                        topic,
                        instance_id,
                        transport,
                    })) if scope == expected_scope && topic == KV_EVENT_SUBJECT => {
                        let EventTransport::Zmq { endpoint } = transport else {
                            tracing::warn!(publisher_id = instance_id, "Ignoring non-direct-ZMQ event channel in direct ingress");
                            continue;
                        };
                        if invalid_publishers.contains(&instance_id) {
                            continue;
                        }
                        if let Some(existing) = sources.get(&instance_id) {
                            if existing.endpoint == endpoint {
                                continue;
                            }
                            tracing::error!(
                                publisher_id = instance_id,
                                old_endpoint = %existing.endpoint,
                                new_endpoint = %endpoint,
                                "Direct-ZMQ publisher changed its immutable channel endpoint"
                            );
                            let existing = sources.remove(&instance_id).expect("entry was present");
                            stop_source(existing, ingress_metrics).await;
                            client.fence_transport(instance_id).await;
                            invalid_publishers.insert(instance_id);
                            let view = membership_watch.borrow().clone();
                            reconcile_sources(
                                client,
                                view,
                                &mut sources,
                                ingress_metrics,
                                &signal_tx,
                                cancellation_token,
                                &mut next_task_generation,
                            ).await;
                            continue;
                        }

                        let task_generation = next_task_generation;
                        next_task_generation = next_task_generation.wrapping_add(1);
                        let source = spawn_source(
                            instance_id,
                            endpoint,
                            task_generation,
                            signal_tx.clone(),
                            client.clone(),
                            ingress_metrics.clone(),
                            cancellation_token.child_token(),
                        );
                        sources.insert(instance_id, source);
                        ingress_metrics.increment_lifecycle("started");
                    }
                    Ok(DiscoveryEvent::Removed(DiscoveryInstanceId::EventChannel(
                        EventChannelInstanceId { scope, topic, instance_id },
                    ))) if scope == expected_scope && topic == KV_EVENT_SUBJECT => {
                        invalid_publishers.remove(&instance_id);
                        if let Some(source) = sources.remove(&instance_id) {
                            stop_source(source, ingress_metrics).await;
                            client.fence_transport(instance_id).await;
                            ingress_metrics.increment_lifecycle("removed");
                            let view = membership_watch.borrow().clone();
                            reconcile_sources(
                                client,
                                view,
                                &mut sources,
                                ingress_metrics,
                                &signal_tx,
                                cancellation_token,
                                &mut next_task_generation,
                            ).await;
                        }
                    }
                    Ok(DiscoveryEvent::Added(_)) | Ok(DiscoveryEvent::Removed(_)) => {}
                    Err(error) => {
                        tracing::error!(%error, %kv_state_endpoint, "Direct-ZMQ event-channel discovery failed");
                        break;
                    }
                }
            }
        }
    }

    let stopped_sources = sources.into_iter().collect::<Vec<_>>();
    for (_, source) in &stopped_sources {
        source.cancel.cancel();
    }
    let publisher_ids = stopped_sources
        .iter()
        .map(|(publisher_id, _)| *publisher_id)
        .collect::<Vec<_>>();
    futures::future::join_all(
        stopped_sources
            .into_iter()
            .map(|(_, source)| stop_source(source, ingress_metrics)),
    )
    .await;
    for publisher_id in publisher_ids {
        client.fence_transport(publisher_id).await;
    }
    client
        .sync_membership_with_ready_sources(&HashSet::new())
        .await;
    exit
}

async fn reconcile_sources(
    client: &Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    preliminary_view: KvSourceMembershipView,
    sources: &mut HashMap<u64, SourceTask>,
    metrics: &Arc<KvZmqIngressMetrics>,
    signal_tx: &mpsc::Sender<SourceSignal>,
    cancellation_token: &CancellationToken,
    next_task_generation: &mut u64,
) {
    let preliminary_ready = ready_sources(&preliminary_view, sources);
    let obsolete: Vec<_> = sources
        .iter()
        .filter_map(|(publisher_id, source)| {
            source.state.active_bindings().and_then(|active| {
                (active != &bindings_for_publisher(&preliminary_ready, *publisher_id))
                    .then_some(*publisher_id)
            })
        })
        .collect();
    for publisher_id in obsolete {
        restart_source(
            publisher_id,
            client,
            sources,
            metrics,
            signal_tx,
            cancellation_token,
            next_task_generation,
        )
        .await;
    }

    let preconnected_ready = ready_sources(&preliminary_view, sources);
    let current_view = client
        .sync_membership_with_ready_sources(&preconnected_ready)
        .await;
    let current_ready = ready_sources(&current_view, sources);
    let stale_after_sync: Vec<_> = sources
        .iter()
        .filter_map(|(publisher_id, source)| {
            source.state.active_bindings().and_then(|active| {
                (active != &bindings_for_publisher(&current_ready, *publisher_id))
                    .then_some(*publisher_id)
            })
        })
        .collect();
    for publisher_id in stale_after_sync {
        restart_source(
            publisher_id,
            client,
            sources,
            metrics,
            signal_tx,
            cancellation_token,
            next_task_generation,
        )
        .await;
    }

    let ready_publishers = current_ready
        .iter()
        .map(|ready| ready.source_id.publisher_id)
        .collect::<HashSet<_>>();
    for publisher_id in ready_publishers {
        let bindings = bindings_for_publisher(&current_ready, publisher_id);
        if !bindings.is_subset(&preconnected_ready) {
            continue;
        }
        let Some(source) = sources.get_mut(&publisher_id) else {
            continue;
        };
        if source.state.active_bindings() == Some(&bindings) {
            continue;
        }
        let SourceState::Preconnected { activate } =
            std::mem::replace(&mut source.state, SourceState::Fenced)
        else {
            continue;
        };
        metrics.decrement_sources("preconnected");
        if activate.send(()).is_ok() {
            source.state = SourceState::Active { bindings };
            metrics.increment_sources("active");
            metrics.increment_lifecycle("activated");
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn restart_source(
    publisher_id: u64,
    client: &Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    sources: &mut HashMap<u64, SourceTask>,
    metrics: &Arc<KvZmqIngressMetrics>,
    signal_tx: &mpsc::Sender<SourceSignal>,
    cancellation_token: &CancellationToken,
    next_task_generation: &mut u64,
) {
    let Some(source) = sources.remove(&publisher_id) else {
        return;
    };
    let endpoint = source.endpoint.clone();
    stop_source(source, metrics).await;
    client.fence_transport(publisher_id).await;

    let task_generation = *next_task_generation;
    *next_task_generation = (*next_task_generation).wrapping_add(1);
    sources.insert(
        publisher_id,
        spawn_source(
            publisher_id,
            endpoint,
            task_generation,
            signal_tx.clone(),
            client.clone(),
            metrics.clone(),
            cancellation_token.child_token(),
        ),
    );
    metrics.increment_lifecycle("replaced");
}

fn ready_sources(
    view: &KvSourceMembershipView,
    sources: &HashMap<u64, SourceTask>,
) -> HashSet<ReadyKvSource> {
    view.sources
        .iter()
        .filter_map(|(worker, status)| {
            let source = status.active_source()?;
            let transport = sources.get(&source.publisher_id)?;
            if !transport.state.is_ready() {
                return None;
            }
            Some(ReadyKvSource {
                source_id: source.source_id(),
                lifecycle_generation: view.lifecycle_generation(worker).unwrap_or(0),
            })
        })
        .collect()
}

fn bindings_for_publisher(
    ready: &HashSet<ReadyKvSource>,
    publisher_id: u64,
) -> HashSet<ReadyKvSource> {
    ready
        .iter()
        .filter(|binding| binding.source_id.publisher_id == publisher_id)
        .cloned()
        .collect()
}

fn spawn_source(
    publisher_id: u64,
    endpoint: String,
    task_generation: u64,
    signal_tx: mpsc::Sender<SourceSignal>,
    client: Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    metrics: Arc<KvZmqIngressMetrics>,
    cancel: CancellationToken,
) -> SourceTask {
    let task_cancel = cancel.clone();
    let task_endpoint = endpoint.clone();
    let handle = tokio::spawn(async move {
        run_source(
            publisher_id,
            task_endpoint,
            task_generation,
            signal_tx,
            client,
            metrics,
            task_cancel,
        )
        .await;
    });
    SourceTask {
        endpoint,
        task_generation,
        cancel,
        handle,
        state: SourceState::Connecting,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_source(
    publisher_id: u64,
    endpoint: String,
    task_generation: u64,
    signal_tx: mpsc::Sender<SourceSignal>,
    client: Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    metrics: Arc<KvZmqIngressMetrics>,
    cancel: CancellationToken,
) {
    let mut retry_delay = INITIAL_BACKOFF;
    loop {
        let stream = tokio::select! {
            _ = cancel.cancelled() => return,
            stream = ZmqSubTransport::connect_single_consumer(&endpoint, KV_EVENT_SUBJECT) => stream,
        };
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                tracing::warn!(%error, publisher_id, %endpoint, "Failed to connect direct-ZMQ KV source");
                if signal_tx
                    .send(SourceSignal::Disconnected {
                        publisher_id,
                        task_generation,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
                if !sleep_or_cancel(retry_delay, &cancel).await {
                    return;
                }
                retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
                continue;
            }
        };
        retry_delay = INITIAL_BACKOFF;

        let (activate, activation) = oneshot::channel();
        if signal_tx
            .send(SourceSignal::Ready {
                publisher_id,
                task_generation,
                activate,
            })
            .await
            .is_err()
        {
            return;
        }

        let activated = tokio::select! {
            _ = cancel.cancelled() => return,
            activated = activation => activated.is_ok(),
        };
        if !activated {
            return;
        }

        tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            _ = consume_connection(publisher_id, &mut stream, &client, &metrics) => {}
        }
        if signal_tx
            .send(SourceSignal::Disconnected {
                publisher_id,
                task_generation,
            })
            .await
            .is_err()
        {
            return;
        }
        if !sleep_or_cancel(retry_delay, &cancel).await {
            return;
        }
        retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
    }
}

async fn consume_connection(
    publisher_id: u64,
    stream: &mut ZmqWireStream,
    client: &Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    metrics: &KvZmqIngressMetrics,
) {
    let codec = Codec::default();
    loop {
        let result = stream.next().await;
        let Some(result) = result else {
            return;
        };
        let message = match result {
            Ok(message) => message,
            Err(error) => {
                tracing::warn!(%error, publisher_id, "Direct-ZMQ KV source stream failed");
                return;
            }
        };

        let envelope = match codec.decode_envelope(&message.payload) {
            Ok(envelope) => envelope,
            Err(error) => {
                tracing::warn!(%error, publisher_id, "Failed to decode direct-ZMQ KV envelope");
                metrics.increment_lifecycle("envelope_decode_error");
                continue;
            }
        };
        if envelope.publisher_id != publisher_id
            || envelope.publisher_id != message.publisher_id
            || envelope.sequence != message.sequence
            || envelope.topic != KV_EVENT_SUBJECT
        {
            tracing::warn!(
                publisher_id,
                frame_publisher_id = message.publisher_id,
                frame_sequence = message.sequence,
                envelope_publisher_id = envelope.publisher_id,
                envelope_sequence = envelope.sequence,
                envelope_topic = %envelope.topic,
                "Dropping direct-ZMQ KV envelope with inconsistent attribution"
            );
            metrics.increment_lifecycle("identity_mismatch");
            continue;
        }
        let events = match codec.decode_payload::<Vec<RouterEvent>>(&envelope.payload) {
            Ok(events) => events,
            Err(error) => {
                tracing::warn!(%error, publisher_id, "Failed to decode direct-ZMQ KV payload");
                metrics.increment_lifecycle("payload_decode_error");
                continue;
            }
        };
        client.handle_live_batch(publisher_id, events).await;
        metrics.increment_batch();
    }
}

async fn stop_source(source: SourceTask, metrics: &KvZmqIngressMetrics) {
    leave_source_state(&source.state, metrics);
    source.cancel.cancel();
    stop_source_handle(source.handle, metrics).await;
}

fn transition_source_state(
    source: &mut SourceTask,
    next: SourceState,
    metrics: &KvZmqIngressMetrics,
) {
    leave_source_state(&source.state, metrics);
    enter_source_state(&next, metrics);
    source.state = next;
}

fn enter_source_state(state: &SourceState, metrics: &KvZmqIngressMetrics) {
    match state {
        SourceState::Preconnected { .. } => metrics.increment_sources("preconnected"),
        SourceState::Active { .. } => metrics.increment_sources("active"),
        SourceState::Connecting | SourceState::Fenced => {}
    }
}

fn leave_source_state(state: &SourceState, metrics: &KvZmqIngressMetrics) {
    match state {
        SourceState::Preconnected { .. } => metrics.decrement_sources("preconnected"),
        SourceState::Active { .. } => metrics.decrement_sources("active"),
        SourceState::Connecting | SourceState::Fenced => {}
    }
}

async fn stop_source_handle(mut handle: JoinHandle<()>, metrics: &KvZmqIngressMetrics) {
    match tokio::time::timeout(SOURCE_JOIN_TIMEOUT, &mut handle).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) if error.is_cancelled() => {}
        Ok(Err(error)) => {
            tracing::warn!(%error, "Direct-ZMQ source task failed during shutdown");
        }
        Err(_) => {
            handle.abort();
            let _ = handle.await;
            metrics.increment_lifecycle("forced_abort");
        }
    }
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

async fn sleep_or_cancel(delay: Duration, cancellation_token: &CancellationToken) -> bool {
    tokio::select! {
        _ = cancellation_token.cancelled() => false,
        _ = tokio::time::sleep(delay) => true,
    }
}
