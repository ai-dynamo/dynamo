// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc, time::Instant};

use anyhow::Context;
use dynamo_runtime::{
    DistributedRuntime,
    discovery::{DiscoveryEvent, DiscoveryInstance, DiscoveryQuery},
};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    app::{AppEvent, NatsConnectionState, NatsSnapshot},
    metrics::{PrometheusSample, process_metrics},
};

pub async fn spawn_discovery_pipeline(
    drt: Arc<DistributedRuntime>,
    tx: mpsc::Sender<AppEvent>,
    cancel: CancellationToken,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let discovery = drt.discovery();
    let mut instances_by_id: HashMap<u64, dynamo_runtime::component::Instance> = HashMap::new();

    let discovery_instances = discovery
        .list(DiscoveryQuery::AllEndpoints)
        .await
        .context("failed to list discovery instances")?;

    let mut endpoints = Vec::new();
    for di in discovery_instances {
        if let DiscoveryInstance::Endpoint(instance) = di {
            instances_by_id.insert(instance.instance_id, instance.clone());
            endpoints.push(instance);
        }
    }

    tx.send(AppEvent::DiscoverySnapshot(endpoints)).await.ok();

    let stream = discovery
        .list_and_watch(DiscoveryQuery::AllEndpoints, Some(cancel.clone()))
        .await
        .context("failed to watch discovery events")?;

    let cancel_listener = cancel.clone();

    let handle = tokio::spawn(async move {
        let tx = tx;
        let mut instances_by_id = instances_by_id;
        let mut stream = stream;
        loop {
            tokio::select! {
                _ = cancel_listener.cancelled() => break,
                event = stream.next() => {
                    let Some(event) = event else { break };
                    match event {
                        Ok(DiscoveryEvent::Added(DiscoveryInstance::Endpoint(instance))) => {
                            instances_by_id.insert(instance.instance_id, instance.clone());
                            if tx.send(AppEvent::InstanceUp(instance)).await.is_err() {
                                break;
                            }
                        }
                        Ok(DiscoveryEvent::Removed(instance_id)) => {
                            if let Some(instance) = instances_by_id.remove(&instance_id) {
                                let event = AppEvent::InstanceDown {
                                    namespace: instance.namespace.clone(),
                                    component: instance.component.clone(),
                                    endpoint: instance.endpoint.clone(),
                                    instance_id,
                                };
                                if tx.send(event).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Ok(_) => {
                            // Ignore other discovery events (e.g. model cards)
                        }
                        Err(err) => {
                            let _ = tx
                                .send(AppEvent::Error(format!("Discovery stream error: {err}")))
                                .await;
                        }
                    }
                }
            }
        }
    });

    Ok(handle)
}

pub async fn spawn_nats_pipeline(
    drt: Arc<DistributedRuntime>,
    tx: mpsc::Sender<AppEvent>,
    cancel: CancellationToken,
    interval: std::time::Duration,
) -> anyhow::Result<Option<tokio::task::JoinHandle<()>>> {
    let Some(nats_client) = drt.nats_client().cloned() else {
        tx.send(AppEvent::Info(
            "NATS client not configured; skipping NATS monitoring".to_string(),
        ))
        .await
        .ok();
        return Ok(None);
    };

    let snapshot = build_nats_snapshot(&nats_client);
    tx.send(AppEvent::Nats(snapshot)).await.ok();

    let mut ticker = tokio::time::interval(interval);
    // Fire first tick immediately
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let handle = tokio::spawn(async move {
        let tx = tx;
        let nats_client = nats_client;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = ticker.tick() => {
                    let snapshot = build_nats_snapshot(&nats_client);

                    if tx.send(AppEvent::Nats(snapshot)).await.is_err() {
                        break;
                    }
                }
            }
        }
    });

    Ok(Some(handle))
}

fn build_nats_snapshot(client: &dynamo_runtime::transports::nats::Client) -> NatsSnapshot {
    let stats = client.client().statistics();
    let connection_state = match client.client().connection_state() {
        async_nats::connection::State::Connected => NatsConnectionState::Connected,
        _ => NatsConnectionState::Disconnected,
    };
    use std::sync::atomic::Ordering;
    NatsSnapshot {
        in_bytes: stats.in_bytes.load(Ordering::Relaxed),
        out_bytes: stats.out_bytes.load(Ordering::Relaxed),
        in_messages: stats.in_messages.load(Ordering::Relaxed),
        out_messages: stats.out_messages.load(Ordering::Relaxed),
        connects: stats.connects.load(Ordering::Relaxed),
        connection_state,
        last_updated: Instant::now(),
    }
}

pub async fn spawn_metrics_pipeline(
    metrics_url: String,
    interval: std::time::Duration,
    tx: mpsc::Sender<AppEvent>,
    cancel: CancellationToken,
) -> anyhow::Result<Option<tokio::task::JoinHandle<()>>> {
    let client = reqwest::Client::builder()
        .user_agent("dynamo-tui/0.1")
        .gzip(true)
        .build()
        .context("failed to build reqwest client")?;

    tx.send(AppEvent::Info(format!(
        "Scraping metrics from {metrics_url}"
    )))
    .await
    .ok();

    let prev_sample: Option<PrometheusSample> = None;
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let handle = tokio::spawn(async move {
        let tx = tx;
        let mut prev_sample = prev_sample;

        if collect_metrics_once(&client, &metrics_url, &mut prev_sample, &tx)
            .await
            .is_err()
        {
            // error already reported
        }

        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = ticker.tick() => {
                    if collect_metrics_once(&client, &metrics_url, &mut prev_sample, &tx).await.is_err() {
                        continue;
                    }
                }
            }
        }
    });

    Ok(Some(handle))
}

async fn collect_metrics_once(
    client: &reqwest::Client,
    metrics_url: &str,
    prev_sample: &mut Option<PrometheusSample>,
    tx: &mpsc::Sender<AppEvent>,
) -> anyhow::Result<()> {
    match client.get(metrics_url).send().await {
        Ok(response) => match response.text().await {
            Ok(body) => {
                let now = Instant::now();
                let (snapshot, sample) = process_metrics(&body, now, prev_sample.as_ref());
                *prev_sample = Some(sample);
                if tx.send(AppEvent::Metrics(snapshot)).await.is_err() {
                    anyhow::bail!("metrics channel closed");
                }
            }
            Err(err) => {
                tracing::warn!(error = %err, "failed to read metrics response body");
                let _ = tx
                    .send(AppEvent::Error(format!("Metrics body error: {err}")))
                    .await;
            }
        },
        Err(err) => {
            tracing::warn!(error = %err, "failed to fetch metrics endpoint");
            let _ = tx
                .send(AppEvent::Error(format!("Metrics fetch error: {err}")))
                .await;
        }
    }
    Ok(())
}
