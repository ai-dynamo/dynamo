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
    IndexerRecoveryTarget, RecoveryTarget, direct_zmq::run_direct_zmq_supervisor,
    worker_query::WorkerQueryClient,
};
use crate::{
    discovery::{KvSourceMembershipView, KvSourceMembershipWatch, KvSourceStatus},
    kv_router::{Indexer, metrics::RouterWorkerStatusMetrics},
};

const SUBSCRIPTION_INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const SUBSCRIPTION_MAX_BACKOFF: Duration = Duration::from_secs(5);

enum ScopeExit {
    Rebind,
    Retry,
    Stop,
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
    cancellation_token: CancellationToken,
    mut startup_ready: Option<oneshot::Sender<()>>,
) {
    let metrics = RouterWorkerStatusMetrics::from_component(&component);
    let mut retry_delay = SUBSCRIPTION_INITIAL_BACKOFF;

    loop {
        let view = membership_watch.borrow_and_update().clone();
        update_mismatch_metric(&metrics, &view, &model, worker_type, &serving_endpoint);

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
                update_mismatch_metric(metrics, &view, model, worker_type, serving_endpoint);
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
) {
    let mismatch_count = view
        .sources
        .iter()
        .filter(|(worker, status)| match status {
            KvSourceStatus::Missing | KvSourceStatus::Ambiguous(_) => true,
            KvSourceStatus::ActiveLiveOnly(_) => view.recovery_expected(worker).unwrap_or(false),
            KvSourceStatus::ActiveRecoverable(_) => false,
        })
        .count();
    metrics.set_kv_event_source_mismatch_workers(
        model,
        worker_type,
        &serving_endpoint.namespace,
        &serving_endpoint.component,
        &serving_endpoint.name,
        mismatch_count,
    );
}

pub(super) fn update_subscription_failure_metric(
    metrics: &RouterWorkerStatusMetrics,
    view: &KvSourceMembershipView,
    model: &str,
    worker_type: &str,
    serving_endpoint: &EndpointId,
) {
    metrics.set_kv_event_source_mismatch_workers(
        model,
        worker_type,
        &serving_endpoint.namespace,
        &serving_endpoint.component,
        &serving_endpoint.name,
        view.sources.len(),
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

pub(crate) struct KvEventSubscriptionHandle {
    cancel: CancellationToken,
    completion: Option<oneshot::Receiver<()>>,
}

impl KvEventSubscriptionHandle {
    pub(crate) async fn shutdown(mut self) {
        self.cancel.cancel();
        if let Some(completion) = self.completion.take() {
            let _ = completion.await;
        }
    }
}

impl Drop for KvEventSubscriptionHandle {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

pub async fn start_subscriber(
    endpoint: Endpoint,
    indexer: Indexer,
    membership_watch: KvSourceMembershipWatch,
    model: String,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
) -> Result<KvEventSubscriptionHandle> {
    let transport_kind = endpoint.component().drt().default_event_transport_kind();
    let direct_zmq = uses_direct_zmq(transport_kind);
    let cancel = cancellation_token.child_token();
    let client = WorkerQueryClient::spawn(
        endpoint.component().clone(),
        IndexerRecoveryTarget::new(indexer),
        membership_watch.clone(),
        cancel.child_token(),
    )
    .await?;

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
                task_cancel,
                Some(startup_tx),
            )
            .await;
            let _ = completion_tx.send(());
        });
        startup_rx.await.map_err(|_| {
            anyhow::anyhow!("KV event subscription supervisor exited before reporting readiness")
        })?;
        return Ok(KvEventSubscriptionHandle {
            cancel,
            completion: Some(completion_rx),
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
    Ok(KvEventSubscriptionHandle {
        cancel,
        completion: Some(completion_rx),
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

    #[tokio::test]
    async fn subscription_handle_shutdown_waits_for_completion() {
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let (completion_tx, completion_rx) = oneshot::channel();
        tokio::spawn(async move {
            task_cancel.cancelled().await;
            let _ = completion_tx.send(());
        });
        let handle = KvEventSubscriptionHandle {
            cancel,
            completion: Some(completion_rx),
        };

        tokio::time::timeout(Duration::from_secs(1), handle.shutdown())
            .await
            .expect("subscription shutdown should complete");
    }
}
