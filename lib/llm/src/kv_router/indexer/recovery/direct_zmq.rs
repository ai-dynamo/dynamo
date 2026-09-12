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
    transports::event_plane::{Codec, EventScope, ValidatedZmqSource, ValidatedZmqSourceError},
};
use futures::StreamExt;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

use super::{
    IndexerRecoveryTarget,
    subscriber::{
        MismatchMetricScope, clear_mismatch_metric_on_cancellation, update_mismatch_metric,
        update_subscription_failure_metric,
    },
    worker_query::WorkerQueryClient,
};
use crate::{
    direct_zmq_sub_pool::{
        DirectZmqSubConnection, DirectZmqSubItem, DirectZmqSubPool, ENDPOINTS_PER_SUB_ENV,
        KV_ZMQ_RCVHWM, endpoints_per_sub_from_env,
    },
    discovery::{KvSourceId, KvSourceMembershipView, KvSourceMembershipWatch},
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
        group_id: Option<u64>,
        activate: oneshot::Sender<()>,
    },
    Disconnected {
        publisher_id: u64,
        task_generation: u64,
        group_id: Option<u64>,
    },
}

struct SourceTask {
    endpoint: String,
    task_generation: u64,
    cancel: CancellationToken,
    handle: JoinHandle<()>,
    state: SourceState,
    group_id: Option<u64>,
}

enum SourceState {
    Connecting,
    Preconnected { activate: oneshot::Sender<()> },
    Active { bindings: HashSet<KvSourceId> },
    Fenced,
}

impl SourceState {
    fn is_ready(&self) -> bool {
        matches!(self, Self::Preconnected { .. } | Self::Active { .. })
    }

    fn active_bindings(&self) -> Option<&HashSet<KvSourceId>> {
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
    metric_scope: MismatchMetricScope,
    cancellation_token: CancellationToken,
    mut startup_ready: Option<oneshot::Sender<Result<(), String>>>,
) {
    let status_metrics = RouterWorkerStatusMetrics::from_component(&component);
    let ingress_metrics = KvZmqIngressMetrics::from_component(&component);
    let endpoints_per_sub = match endpoints_per_sub_from_env() {
        Ok(value) => value,
        Err(error) => {
            tracing::error!(%error, "Invalid direct-ZMQ KV ingress configuration");
            if let Some(ready) = startup_ready.take() {
                let _ = ready.send(Err(error.to_string()));
            }
            return;
        }
    };
    tracing::info!(
        endpoints_per_sub,
        env = ENDPOINTS_PER_SUB_ENV,
        "Configured direct-ZMQ KV ingress SUB fan-in"
    );
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
                    metric_scope,
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
            metric_scope,
            &cancellation_token,
            endpoints_per_sub,
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
    metric_scope: MismatchMetricScope,
    cancellation_token: &CancellationToken,
    endpoints_per_sub: usize,
) -> ScopeExit {
    let expected_scope = EventScope::Endpoint {
        endpoint: kv_state_endpoint.clone(),
    };
    let (signal_tx, mut signal_rx) = mpsc::channel(SIGNAL_CAPACITY);
    let group_pool = DirectZmqSubPool::new(
        KV_EVENT_SUBJECT,
        endpoints_per_sub,
        KV_ZMQ_RCVHWM,
        cancellation_token.child_token(),
    )
    .expect("validated direct-ZMQ KV ingress configuration");
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
                    &group_pool,
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
                    metric_scope,
                );
            }
            signal = signal_rx.recv() => {
                let Some(signal) = signal else {
                    break;
                };
                match signal {
                    SourceSignal::Ready { publisher_id, task_generation, group_id, activate } => {
                        let Some(source) = sources.get_mut(&publisher_id) else {
                            continue;
                        };
                        if source.task_generation != task_generation {
                            continue;
                        }
                        source.group_id = group_id;
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
                            &group_pool,
                            ingress_metrics,
                            &signal_tx,
                            cancellation_token,
                            &mut next_task_generation,
                        ).await;
                    }
                    SourceSignal::Disconnected { publisher_id, task_generation, group_id } => {
                        let affected = affected_source_ids(
                            &sources,
                            publisher_id,
                            task_generation,
                            group_id,
                        );
                        if affected.is_empty() {
                            continue;
                        }
                        for publisher_id in &affected {
                            let source = sources
                                .get_mut(publisher_id)
                                .expect("affected source must exist");
                            source.group_id = None;
                            transition_source_state(source, SourceState::Fenced, ingress_metrics);
                            ingress_metrics.increment_lifecycle("reconnect");
                        }
                        for publisher_id in affected {
                            client.fence_transport(publisher_id).await;
                        }
                        let view = membership_watch.borrow().clone();
                        reconcile_sources(
                            client,
                            view,
                            &mut sources,
                            &group_pool,
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
                                &group_pool,
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
                            group_pool.clone(),
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
                                &group_pool,
                                ingress_metrics,
                                &signal_tx,
                                cancellation_token,
                                &mut next_task_generation,
                            ).await;
                        }
                    }
                    Ok(DiscoveryEvent::Added(_))
                    | Ok(DiscoveryEvent::ModelTaintsUpdated(_))
                    | Ok(DiscoveryEvent::Removed(_)) => {}
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
    group_pool.shutdown().await;
    for publisher_id in publisher_ids {
        client.fence_transport(publisher_id).await;
    }
    client
        .sync_membership_with_ready_sources(&HashSet::new())
        .await;
    exit
}

fn affected_source_ids(
    sources: &HashMap<u64, SourceTask>,
    publisher_id: u64,
    task_generation: u64,
    group_id: Option<u64>,
) -> Vec<u64> {
    let Some(source) = sources.get(&publisher_id) else {
        return Vec::new();
    };
    if source.task_generation != task_generation || source.group_id != group_id {
        return Vec::new();
    }
    match group_id {
        Some(group_id) => sources
            .iter()
            .filter_map(|(publisher_id, source)| {
                (source.group_id == Some(group_id)).then_some(*publisher_id)
            })
            .collect(),
        None => vec![publisher_id],
    }
}

#[allow(clippy::too_many_arguments)]
async fn reconcile_sources(
    client: &Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    preliminary_view: KvSourceMembershipView,
    sources: &mut HashMap<u64, SourceTask>,
    group_pool: &DirectZmqSubPool,
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
            group_pool,
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
            group_pool,
            metrics,
            signal_tx,
            cancellation_token,
            next_task_generation,
        )
        .await;
    }

    let ready_publishers = current_ready
        .iter()
        .map(|ready| ready.publisher_id)
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
        if !matches!(&source.state, SourceState::Preconnected { .. }) {
            continue;
        }
        let SourceState::Preconnected { activate } =
            std::mem::replace(&mut source.state, SourceState::Fenced)
        else {
            unreachable!("source state was checked above");
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
    group_pool: &DirectZmqSubPool,
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
            group_pool.clone(),
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
) -> HashSet<KvSourceId> {
    view.sources
        .values()
        .filter_map(|status| {
            let source = status.active_source()?;
            let transport = sources.get(&source.publisher_id)?;
            if !transport.state.is_ready() {
                return None;
            }
            Some(source.source_id())
        })
        .collect()
}

fn bindings_for_publisher(ready: &HashSet<KvSourceId>, publisher_id: u64) -> HashSet<KvSourceId> {
    ready
        .iter()
        .filter(|binding| binding.publisher_id == publisher_id)
        .cloned()
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn spawn_source(
    publisher_id: u64,
    endpoint: String,
    task_generation: u64,
    signal_tx: mpsc::Sender<SourceSignal>,
    group_pool: DirectZmqSubPool,
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
            group_pool,
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
        group_id: None,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_source(
    publisher_id: u64,
    endpoint: String,
    task_generation: u64,
    signal_tx: mpsc::Sender<SourceSignal>,
    group_pool: DirectZmqSubPool,
    client: Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    metrics: Arc<KvZmqIngressMetrics>,
    cancel: CancellationToken,
) {
    let mut retry_delay = INITIAL_BACKOFF;
    loop {
        let connection = tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            connection = group_pool.connect(publisher_id, &endpoint, task_generation) => connection,
        };
        let mut connection = match connection {
            Ok(connection) => connection,
            Err(error) => {
                tracing::warn!(%error, publisher_id, %endpoint, "Failed to connect direct-ZMQ KV source");
                if !send_source_signal(
                    &signal_tx,
                    SourceSignal::Disconnected {
                        publisher_id,
                        task_generation,
                        group_id: None,
                    },
                    &cancel,
                )
                .await
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
        if cancel.is_cancelled() {
            connection.close().await;
            return;
        }
        retry_delay = INITIAL_BACKOFF;
        let group_id = connection.group_id();
        let disconnected = connection.disconnected();

        let (activate, activation) = oneshot::channel();
        if !send_source_signal(
            &signal_tx,
            SourceSignal::Ready {
                publisher_id,
                task_generation,
                group_id,
                activate,
            },
            &cancel,
        )
        .await
        {
            return;
        }

        enum ActivationOutcome {
            Activated,
            Cancelled,
            Disconnected,
        }

        let activation_outcome = tokio::select! {
            _ = cancel.cancelled() => ActivationOutcome::Cancelled,
            activated = activation => {
                if activated.is_ok() {
                    ActivationOutcome::Activated
                } else {
                    ActivationOutcome::Cancelled
                }
            },
            _ = wait_for_disconnect(disconnected.clone()) => ActivationOutcome::Disconnected,
        };
        if !matches!(activation_outcome, ActivationOutcome::Activated) {
            connection.close().await;
            if matches!(activation_outcome, ActivationOutcome::Cancelled) {
                return;
            }
            if !send_source_signal(
                &signal_tx,
                SourceSignal::Disconnected {
                    publisher_id,
                    task_generation,
                    group_id,
                },
                &cancel,
            )
            .await
            {
                return;
            }
            if !sleep_or_cancel(retry_delay, &cancel).await {
                return;
            }
            retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
            continue;
        }

        let cancelled = tokio::select! {
            biased;
            _ = cancel.cancelled() => true,
            _ = wait_for_disconnect(disconnected) => false,
            _ = consume_connection(publisher_id, &mut connection, &client, &metrics) => false,
        };
        connection.close().await;
        if cancelled {
            return;
        }
        if !send_source_signal(
            &signal_tx,
            SourceSignal::Disconnected {
                publisher_id,
                task_generation,
                group_id,
            },
            &cancel,
        )
        .await
        {
            return;
        }
        if !sleep_or_cancel(retry_delay, &cancel).await {
            return;
        }
        retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
    }
}

async fn wait_for_disconnect(disconnected: Option<CancellationToken>) {
    match disconnected {
        Some(disconnected) => disconnected.cancelled().await,
        None => std::future::pending().await,
    }
}

async fn send_source_signal(
    signal_tx: &mpsc::Sender<SourceSignal>,
    signal: SourceSignal,
    cancel: &CancellationToken,
) -> bool {
    tokio::select! {
        biased;
        _ = cancel.cancelled() => false,
        result = signal_tx.send(signal) => result.is_ok(),
    }
}

async fn consume_connection(
    publisher_id: u64,
    connection: &mut DirectZmqSubConnection,
    client: &Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    metrics: &KvZmqIngressMetrics,
) {
    match connection {
        DirectZmqSubConnection::Dedicated(source) => {
            consume_dedicated_connection(publisher_id, source, client, metrics).await
        }
        DirectZmqSubConnection::Grouped(registration) => {
            consume_grouped_connection(publisher_id, &mut registration.receiver, client, metrics)
                .await
        }
    }
}

async fn consume_grouped_connection(
    publisher_id: u64,
    receiver: &mut mpsc::Receiver<DirectZmqSubItem>,
    client: &Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    metrics: &KvZmqIngressMetrics,
) {
    let codec = Codec::default();
    while let Some(item) = receiver.recv().await {
        let envelope = match item {
            DirectZmqSubItem::Envelope(envelope) => envelope,
            DirectZmqSubItem::EnvelopeDecodeError => {
                metrics.increment_lifecycle("envelope_decode_error");
                continue;
            }
            DirectZmqSubItem::IdentityMismatch => {
                metrics.increment_lifecycle("identity_mismatch");
                continue;
            }
        };
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

async fn consume_dedicated_connection(
    publisher_id: u64,
    source: &mut ValidatedZmqSource,
    client: &Arc<WorkerQueryClient<IndexerRecoveryTarget>>,
    metrics: &KvZmqIngressMetrics,
) {
    let codec = Codec::default();
    loop {
        let Some(result) = source.next().await else {
            return;
        };
        let envelope = match result {
            Ok(envelope) => envelope,
            Err(ValidatedZmqSourceError::Receive(error)) => {
                tracing::warn!(%error, publisher_id, "Direct-ZMQ KV source stream failed");
                return;
            }
            Err(ValidatedZmqSourceError::EnvelopeDecode(error)) => {
                tracing::warn!(%error, publisher_id, "Failed to decode direct-ZMQ KV envelope");
                metrics.increment_lifecycle("envelope_decode_error");
                continue;
            }
            Err(error @ ValidatedZmqSourceError::IdentityMismatch { .. }) => {
                tracing::warn!(%error, publisher_id, "Dropping direct-ZMQ KV envelope with inconsistent attribution");
                metrics.increment_lifecycle("identity_mismatch");
                continue;
            }
        };
        let events = match codec.decode_payload::<Vec<RouterEvent>>(&envelope.payload) {
            Ok(events) => events,
            Err(error) => {
                tracing::warn!(%error, publisher_id, "Failed to decode direct-ZMQ KV payload");
                metrics.increment_lifecycle("payload_decode_error");
                continue;
            }
        };
        let event_count = events.len();
        let first_event_id = events.first().map(|event| event.event.event_id);
        let last_event_id = events.last().map(|event| event.event.event_id);
        client.handle_live_batch(publisher_id, events).await;
        tracing::trace!(
            publisher_id,
            event_count,
            ?first_event_id,
            ?last_event_id,
            "Received KV event batch from worker event plane"
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_source(task_generation: u64, group_id: Option<u64>) -> SourceTask {
        SourceTask {
            endpoint: "tcp://127.0.0.1:1".to_string(),
            task_generation,
            cancel: CancellationToken::new(),
            handle: tokio::spawn(std::future::pending()),
            state: SourceState::Connecting,
            group_id,
        }
    }

    #[tokio::test]
    async fn grouped_disconnect_selects_its_failure_domain_once() {
        let mut sources = HashMap::from([
            (1, test_source(10, Some(7))),
            (2, test_source(20, Some(7))),
            (3, test_source(30, Some(8))),
        ]);

        let affected = affected_source_ids(&sources, 1, 10, Some(7))
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(affected, HashSet::from([1, 2]));
        for publisher_id in affected {
            sources.get_mut(&publisher_id).unwrap().group_id = None;
        }
        assert!(affected_source_ids(&sources, 2, 20, Some(7)).is_empty());

        for source in sources.into_values() {
            source.handle.abort();
        }
    }

    #[tokio::test]
    async fn full_source_status_channel_does_not_delay_shutdown() {
        let (signal_tx, _signal_rx) = mpsc::channel(1);
        signal_tx
            .send(SourceSignal::Disconnected {
                publisher_id: 1,
                task_generation: 1,
                group_id: None,
            })
            .await
            .unwrap();

        let cancel = CancellationToken::new();
        cancel.cancel();
        let sent = tokio::time::timeout(
            Duration::from_secs(1),
            send_source_signal(
                &signal_tx,
                SourceSignal::Disconnected {
                    publisher_id: 2,
                    task_generation: 1,
                    group_id: None,
                },
                &cancel,
            ),
        )
        .await
        .expect("cancelled status send must finish promptly");

        assert!(!sent);
    }
}
