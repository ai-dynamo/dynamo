// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, RouterEvent};
use dynamo_runtime::{
    component::{Component, Endpoint},
    discovery::EventTransportKind,
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{EventSubscriber, TypedEventSubscriber, uses_direct_zmq},
};
use tokio::sync::{Semaphore, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::{
    IndexerRecoveryTarget, RecoveryTarget, direct_zmq::run_direct_zmq_supervisor, source_health,
    worker_query::WorkerQueryClient,
};
use crate::{
    discovery::{KvSourceMembershipView, KvSourceMembershipWatch, KvSourceStatus},
    kv_router::{Indexer, KvEventSourceRequirement, metrics::RouterWorkerStatusMetrics},
    worker_type::WorkerType,
};

const SUBSCRIPTION_INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const SUBSCRIPTION_MAX_BACKOFF: Duration = Duration::from_secs(5);

enum ScopeExit {
    Rebind,
    Retry,
    Stop,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum MismatchMetricScope {
    Generic,
    Router(KvEventSourceRequirement),
}

#[allow(clippy::too_many_arguments)]
async fn run_subscription_supervisor<T: RecoveryTarget>(
    component: Component,
    serving_endpoint: EndpointId,
    client: Arc<WorkerQueryClient<T>>,
    transport_kind: EventTransportKind,
    mut membership_watch: KvSourceMembershipWatch,
    model: String,
    worker_type: &'static str,
    metric_scope: MismatchMetricScope,
    cancellation_token: CancellationToken,
    mut startup_ready: Option<oneshot::Sender<()>>,
) {
    let metrics = RouterWorkerStatusMetrics::from_component(&component);
    let mut retry_delay = SUBSCRIPTION_INITIAL_BACKOFF;

    loop {
        let view = membership_watch.borrow_and_update().clone();
        update_mismatch_metric(
            &metrics,
            &view,
            &model,
            worker_type,
            &serving_endpoint,
            metric_scope,
        );

        let subscriber = if let Some(kv_state_endpoint) = view.resolved_kv_state_endpoint() {
            tracing::debug!(
                serving_endpoint = %serving_endpoint,
                %kv_state_endpoint,
                "Resolved KV-state event source"
            );
            match EventSubscriber::for_endpoint_id_with_transport(
                component.drt(),
                kv_state_endpoint,
                KV_EVENT_SUBJECT,
                transport_kind,
            )
            .await
            {
                Ok(subscriber) => Some((
                    kv_state_endpoint.clone(),
                    subscriber.typed::<Vec<RouterEvent>>(),
                )),
                Err(error) => {
                    tracing::error!(%error, %kv_state_endpoint, "Failed to subscribe to KV-state endpoint");
                    update_subscription_failure_metric(
                        &metrics,
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
                    retry_delay = (retry_delay * 2).min(SUBSCRIPTION_MAX_BACKOFF);
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
        client.sync_membership().await;

        // Subscriber construction establishes buffering before membership activation starts
        // initial recovery. Re-reading the watch above rejects a stale endpoint binding.
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
            &metrics,
            &model,
            worker_type,
            &serving_endpoint,
            metric_scope,
            &cancellation_token,
            &mut retry_delay,
        )
        .await
        {
            ScopeExit::Rebind => {
                retry_delay = SUBSCRIPTION_INITIAL_BACKOFF;
                continue;
            }
            ScopeExit::Retry => {
                let view = client.sync_membership().await;
                update_subscription_failure_metric(
                    &metrics,
                    &view,
                    &model,
                    worker_type,
                    &serving_endpoint,
                    metric_scope,
                );
                if !wait_for_retry(retry_delay, &mut membership_watch, &cancellation_token).await {
                    break;
                }
                retry_delay = (retry_delay * 2).min(SUBSCRIPTION_MAX_BACKOFF);
            }
            ScopeExit::Stop => break,
        }
    }

    client.shutdown().await;
    clear_mismatch_metric_on_cancellation(
        &metrics,
        &cancellation_token,
        &model,
        worker_type,
        &serving_endpoint,
    );
}

#[allow(clippy::too_many_arguments)]
async fn consume_scope<T: RecoveryTarget>(
    mut subscriber: TypedEventSubscriber<Vec<RouterEvent>>,
    client: &Arc<WorkerQueryClient<T>>,
    kv_state_endpoint: &EndpointId,
    membership_watch: &mut KvSourceMembershipWatch,
    metrics: &RouterWorkerStatusMetrics,
    model: &str,
    worker_type: &str,
    serving_endpoint: &EndpointId,
    metric_scope: MismatchMetricScope,
    cancellation_token: &CancellationToken,
    retry_delay: &mut Duration,
) -> ScopeExit {
    loop {
        tokio::select! {
            biased;
            _ = cancellation_token.cancelled() => return ScopeExit::Stop,
            changed = membership_watch.changed() => {
                if changed.is_err() {
                    return ScopeExit::Stop;
                }
                membership_watch.borrow_and_update();
                let view = client.sync_membership().await;
                update_mismatch_metric(
                    metrics,
                    &view,
                    model,
                    worker_type,
                    serving_endpoint,
                    metric_scope,
                );
                if view.resolved_kv_state_endpoint() != Some(kv_state_endpoint) {
                    return ScopeExit::Rebind;
                }
            }
            result = subscriber.next() => {
                let Some(result) = result else {
                    tracing::error!(%kv_state_endpoint, "KV event-plane stream ended unexpectedly");
                    return ScopeExit::Retry;
                };
                *retry_delay = SUBSCRIPTION_INITIAL_BACKOFF;
                match result {
                    Ok((envelope, events)) => {
                        client.handle_live_batch(envelope.publisher_id, events).await;
                    }
                    Err(error) => {
                        tracing::warn!(%error, %kv_state_endpoint, "Failed to decode KV event batch");
                    }
                }
            }
        }
    }
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

pub(super) fn update_mismatch_metric(
    metrics: &RouterWorkerStatusMetrics,
    view: &KvSourceMembershipView,
    model: &str,
    worker_type: &str,
    serving_endpoint: &EndpointId,
    metric_scope: MismatchMetricScope,
) {
    let mismatch_count = mismatch_count(view, metric_scope);
    metrics.set_kv_event_source_mismatch_workers(
        model,
        worker_type,
        &serving_endpoint.namespace,
        &serving_endpoint.component,
        &serving_endpoint.name,
        mismatch_count,
    );
}

fn mismatch_count(view: &KvSourceMembershipView, metric_scope: MismatchMetricScope) -> usize {
    match metric_scope {
        MismatchMetricScope::Router(KvEventSourceRequirement::NotRequired) => 0,
        MismatchMetricScope::Router(KvEventSourceRequirement::Unknown)
        | MismatchMetricScope::Generic => legacy_mismatch_count(view),
        MismatchMetricScope::Router(requirement) => {
            if !requirement.requires_source() {
                0
            } else {
                view.sources
                    .iter()
                    .filter(|(worker, status)| {
                        view.kv_event_publishing_enabled(worker.worker_id) == Some(false)
                            || if view.observation_state.is_bound() {
                                source_status_is_mismatch(view, worker, status)
                            } else {
                                matches!(
                                    status,
                                    KvSourceStatus::Ambiguous(
                                        crate::discovery::KvSourceAmbiguity::EndpointMapping { .. }
                                    )
                                )
                            }
                    })
                    .count()
            }
        }
    }
}

fn legacy_mismatch_count(view: &KvSourceMembershipView) -> usize {
    view.sources
        .iter()
        .filter(|(worker, status)| source_status_is_mismatch(view, worker, status))
        .count()
}

fn source_status_is_mismatch(
    view: &KvSourceMembershipView,
    worker: &dynamo_kv_router::protocols::WorkerWithDpRank,
    status: &KvSourceStatus,
) -> bool {
    match status {
        KvSourceStatus::Missing | KvSourceStatus::Ambiguous(_) => true,
        KvSourceStatus::ActiveLiveOnly(_) => view.recovery_expected(worker).unwrap_or(false),
        KvSourceStatus::ActiveRecoverable(_) => false,
    }
}

pub(super) fn update_subscription_failure_metric(
    metrics: &RouterWorkerStatusMetrics,
    view: &KvSourceMembershipView,
    model: &str,
    worker_type: &str,
    serving_endpoint: &EndpointId,
    metric_scope: MismatchMetricScope,
) {
    let count = match metric_scope {
        MismatchMetricScope::Router(KvEventSourceRequirement::NotRequired) => 0,
        MismatchMetricScope::Generic
        | MismatchMetricScope::Router(KvEventSourceRequirement::Unknown)
        | MismatchMetricScope::Router(_) => view.sources.len(),
    };
    metrics.set_kv_event_source_mismatch_workers(
        model,
        worker_type,
        &serving_endpoint.namespace,
        &serving_endpoint.component,
        &serving_endpoint.name,
        count,
    );
}

pub(super) fn clear_mismatch_metric_on_cancellation(
    metrics: &RouterWorkerStatusMetrics,
    cancellation_token: &CancellationToken,
    model: &str,
    worker_type: &str,
    serving_endpoint: &EndpointId,
) {
    if !cancellation_token.is_cancelled() {
        return;
    }
    metrics.set_kv_event_source_mismatch_workers(
        model,
        worker_type,
        &serving_endpoint.namespace,
        &serving_endpoint.component,
        &serving_endpoint.name,
        0,
    );
}

/// Dropping this handle cancels the KV event subscription and its health monitor.
#[must_use = "dropping the handle cancels the KV event subscription and health monitor"]
pub(crate) struct KvEventSubscriptionHandle {
    cancel: CancellationToken,
    source_health_cancel: CancellationToken,
    completions: Vec<oneshot::Receiver<()>>,
}

impl KvEventSubscriptionHandle {
    pub(crate) fn stop_source_health_monitor(&self) {
        self.source_health_cancel.cancel();
    }

    pub(crate) async fn shutdown(mut self) {
        self.cancel.cancel();
        for completion in self.completions.drain(..) {
            let _ = completion.await;
        }
    }
}

impl Drop for KvEventSubscriptionHandle {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn start_subscriber(
    endpoint: Endpoint,
    indexer: Indexer,
    membership_watch: KvSourceMembershipWatch,
    model: String,
    worker_role: Option<WorkerType>,
    source_requirement: KvEventSourceRequirement,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
) -> Result<KvEventSubscriptionHandle> {
    let transport_kind = endpoint.component().drt().default_event_transport_kind();
    let direct_zmq = uses_direct_zmq(transport_kind);
    let cancel = cancellation_token.child_token();
    let cancellation_guard = cancel.clone().drop_guard();
    let client = WorkerQueryClient::spawn(
        endpoint.component().clone(),
        IndexerRecoveryTarget::new(indexer),
        membership_watch.clone(),
        cancel.child_token(),
    )
    .await?;
    let source_health_cancel = cancel.child_token();
    let health_completion = source_health::spawn(
        membership_watch.clone(),
        model.clone(),
        worker_role,
        source_requirement,
        endpoint.id(),
        source_health_cancel.clone(),
    );
    let metric_scope = MismatchMetricScope::Router(source_requirement);

    if !direct_zmq {
        tracing::info!(
            transport = ?transport_kind,
            "Using aggregated KV event subscriber"
        );
        let (startup_tx, startup_rx) = oneshot::channel();
        let (completion_tx, completion_rx) = oneshot::channel();
        let task_cancel = cancel.clone();
        tokio::spawn(async move {
            run_subscription_supervisor(
                endpoint.component().clone(),
                endpoint.id(),
                client,
                transport_kind,
                membership_watch,
                model,
                worker_type,
                metric_scope,
                task_cancel,
                Some(startup_tx),
            )
            .await;
            let _ = completion_tx.send(());
        });
        startup_rx.await.map_err(|_| {
            anyhow::anyhow!("KV event subscription supervisor exited before reporting readiness")
        })?;
        let cancel = cancellation_guard.disarm();
        return Ok(KvEventSubscriptionHandle {
            cancel,
            source_health_cancel,
            completions: vec![completion_rx, health_completion],
        });
    }

    tracing::info!("Using direct-ZMQ KV event ingress on the application runtime");
    let (startup_tx, startup_rx) = oneshot::channel();
    let (completion_tx, completion_rx) = oneshot::channel();
    let task_cancel = cancel.clone();
    tokio::spawn(async move {
        run_direct_zmq_supervisor(
            endpoint.component().clone(),
            endpoint.id(),
            client,
            membership_watch,
            model,
            worker_type,
            metric_scope,
            task_cancel,
            Some(startup_tx),
        )
        .await;
        let _ = completion_tx.send(());
    });

    startup_rx
        .await
        .map_err(|_| {
            anyhow::anyhow!("Direct-ZMQ ingress supervisor exited before reporting readiness")
        })?
        .map_err(anyhow::Error::msg)?;
    let cancel = cancellation_guard.disarm();
    Ok(KvEventSubscriptionHandle {
        cancel,
        source_health_cancel,
        completions: vec![completion_rx, health_completion],
    })
}

pub(crate) struct RecoverySupervisor<T: RecoveryTarget> {
    client: Arc<WorkerQueryClient<T>>,
    cancel: CancellationToken,
    task: JoinHandle<()>,
}

impl<T: RecoveryTarget> RecoverySupervisor<T> {
    pub(crate) fn client(&self) -> &Arc<WorkerQueryClient<T>> {
        &self.client
    }

    pub(crate) async fn shutdown(self) {
        self.cancel.cancel();
        self.client.shutdown().await;
        if let Err(error) = self.task.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "KV source subscription supervisor failed during shutdown");
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn start_target_subscriber<T: RecoveryTarget>(
    component: Component,
    serving_endpoint: EndpointId,
    target: T,
    membership_watch: KvSourceMembershipWatch,
    model: String,
    worker_type: &'static str,
    recovery_semaphore: Arc<Semaphore>,
    recovery_attempt_timeout: Duration,
    cancellation_token: CancellationToken,
) -> Result<RecoverySupervisor<T>> {
    let transport_kind = component.drt().default_event_transport_kind();
    let cancel = cancellation_token.child_token();
    let client = WorkerQueryClient::spawn_with_recovery_limit(
        component.clone(),
        target,
        membership_watch.clone(),
        recovery_semaphore,
        recovery_attempt_timeout,
        cancel.child_token(),
    )
    .await?;
    let (startup_tx, startup_rx) = oneshot::channel();
    let task = tokio::spawn(run_subscription_supervisor(
        component,
        serving_endpoint,
        client.clone(),
        transport_kind,
        membership_watch,
        model,
        worker_type,
        MismatchMetricScope::Generic,
        cancel.clone(),
        Some(startup_tx),
    ));
    startup_rx.await.map_err(|_| {
        anyhow::anyhow!("KV event subscription supervisor exited before reporting readiness")
    })?;
    Ok(RecoverySupervisor {
        client,
        cancel,
        task,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::WorkerWithDpRank;

    use crate::discovery::{
        KvEventSource, KvSourceAmbiguity, KvSourceObservationState, KvStateEndpointResolution,
    };

    fn metric_view(
        observation_state: KvSourceObservationState,
        status: KvSourceStatus,
        capability: Option<bool>,
    ) -> KvSourceMembershipView {
        let serving_endpoint = EndpointId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            name: "generate".to_string(),
        };
        let worker = WorkerWithDpRank::new(7, 0);
        KvSourceMembershipView {
            serving_endpoint: serving_endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(serving_endpoint),
            observation_state,
            sources: HashMap::from([(worker, status)]),
            kv_event_publishing_enabled: HashMap::from([(7, capability)]),
            lifecycle_generations: HashMap::from([(worker, 0)]),
            recovery_expected: HashMap::from([(worker, false)]),
        }
    }

    #[test]
    fn router_mismatch_metric_is_requirement_and_observation_aware() {
        let rebinding = metric_view(
            KvSourceObservationState::Rebinding,
            KvSourceStatus::Missing,
            Some(true),
        );
        assert_eq!(mismatch_count(&rebinding, MismatchMetricScope::Generic), 1);
        assert_eq!(
            mismatch_count(
                &rebinding,
                MismatchMetricScope::Router(KvEventSourceRequirement::Unknown)
            ),
            1
        );
        assert_eq!(
            mismatch_count(
                &rebinding,
                MismatchMetricScope::Router(KvEventSourceRequirement::CacheAwareRouting)
            ),
            0
        );

        let worker = WorkerWithDpRank::new(7, 0);
        let active_disabled = metric_view(
            KvSourceObservationState::Bound,
            KvSourceStatus::ActiveLiveOnly(KvEventSource {
                kv_state_endpoint: rebinding.serving_endpoint.clone(),
                worker,
                publisher_id: 11,
                recovery_target: None,
            }),
            Some(false),
        );
        assert_eq!(
            mismatch_count(&active_disabled, MismatchMetricScope::Generic),
            0
        );
        assert_eq!(
            mismatch_count(
                &active_disabled,
                MismatchMetricScope::Router(KvEventSourceRequirement::CacheAwareRouting)
            ),
            1
        );
        assert_eq!(
            mismatch_count(
                &rebinding,
                MismatchMetricScope::Router(KvEventSourceRequirement::NotRequired)
            ),
            0
        );

        let endpoint_mapping_ambiguity = metric_view(
            KvSourceObservationState::Rebinding,
            KvSourceStatus::Ambiguous(KvSourceAmbiguity::EndpointMapping {
                endpoints: vec![rebinding.serving_endpoint.clone()],
            }),
            Some(true),
        );
        assert_eq!(
            mismatch_count(
                &endpoint_mapping_ambiguity,
                MismatchMetricScope::Router(KvEventSourceRequirement::CacheAwareRouting)
            ),
            1
        );
    }

    #[tokio::test]
    async fn subscription_handle_shutdown_waits_for_all_owned_tasks() {
        let cancel = CancellationToken::new();
        let source_health_cancel = cancel.child_token();
        let mut completions = Vec::new();
        for _ in 0..2 {
            let task_cancel = cancel.clone();
            let (completion_tx, completion_rx) = oneshot::channel();
            completions.push(completion_rx);
            tokio::spawn(async move {
                task_cancel.cancelled().await;
                let _ = completion_tx.send(());
            });
        }
        let handle = KvEventSubscriptionHandle {
            cancel,
            source_health_cancel,
            completions,
        };

        tokio::time::timeout(Duration::from_secs(1), handle.shutdown())
            .await
            .expect("subscription shutdown should complete");
    }

    #[tokio::test]
    async fn stopping_source_health_preserves_subscription_for_inflight_requests() {
        let cancel = CancellationToken::new();
        let source_health_cancel = cancel.child_token();
        let health_observer = source_health_cancel.clone();
        let handle = KvEventSubscriptionHandle {
            cancel: cancel.clone(),
            source_health_cancel,
            completions: Vec::new(),
        };

        handle.stop_source_health_monitor();

        health_observer.cancelled().await;
        assert!(!cancel.is_cancelled());
        handle.shutdown().await;
    }
}
