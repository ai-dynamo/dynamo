// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, RouterEvent};
use dynamo_runtime::{
    component::Endpoint,
    discovery::EventTransportKind,
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{EventSubscriber, TypedEventSubscriber},
};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use super::worker_query::WorkerQueryClient;
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
async fn run_subscription_supervisor(
    endpoint: Endpoint,
    indexer: Indexer,
    transport_kind: EventTransportKind,
    mut membership_watch: KvSourceMembershipWatch,
    model: String,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
    mut startup_ready: Option<oneshot::Sender<()>>,
) {
    let serving_endpoint = endpoint.id();
    let component = endpoint.component().clone();
    let metrics = RouterWorkerStatusMetrics::from_component(&component);
    let client = match WorkerQueryClient::spawn(
        component.clone(),
        indexer,
        membership_watch.clone(),
        cancellation_token.child_token(),
    )
    .await
    {
        Ok(started) => started,
        Err(error) => {
            tracing::error!(%error, "Failed to start KV source recovery coordinator");
            if let Some(ready) = startup_ready.take() {
                let _ = ready.send(());
            }
            return;
        }
    };
    let mut retry_delay = SUBSCRIPTION_INITIAL_BACKOFF;

    loop {
        membership_watch.borrow_and_update();
        let view = client.sync_membership().await;
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

        // Ordinary serving readiness is independent of recovery. In the healthy path this is
        // sent only after the first exact transport subscription has been constructed.
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
    if cancellation_token.is_cancelled() {
        metrics.set_kv_event_source_mismatch_workers(
            &model,
            worker_type,
            &serving_endpoint.namespace,
            &serving_endpoint.component,
            &serving_endpoint.name,
            0,
        );
    }
}

#[allow(clippy::too_many_arguments)]
async fn consume_scope(
    mut subscriber: TypedEventSubscriber<Vec<RouterEvent>>,
    client: &Arc<WorkerQueryClient>,
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

fn update_mismatch_metric(
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

fn update_subscription_failure_metric(
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

pub async fn start_subscriber(
    endpoint: Endpoint,
    indexer: Indexer,
    membership_watch: KvSourceMembershipWatch,
    model: String,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
) -> Result<()> {
    let transport_kind = endpoint.component().drt().default_event_transport_kind();
    let (startup_tx, startup_rx) = oneshot::channel();
    tokio::spawn(run_subscription_supervisor(
        endpoint,
        indexer,
        transport_kind,
        membership_watch,
        model,
        worker_type,
        cancellation_token,
        Some(startup_tx),
    ));
    startup_rx.await.map_err(|_| {
        anyhow::anyhow!("KV event subscription supervisor exited before reporting readiness")
    })?;
    Ok(())
}
