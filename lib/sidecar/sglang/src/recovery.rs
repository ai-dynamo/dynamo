// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One recovery controller for all ranks attached to one SGLang engine.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, bail};
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, PlacementEvent};
use dynamo_llm::kv_router::publisher::KvEventPublisher;
use dynamo_sidecar_common::kv_replay::{RecoveryStatus, ReplaySource, run_source};
use serde_json::Value;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::{client::Client, proto as pb};

pub(crate) struct Recovery {
    ready: watch::Sender<bool>,
    publishers: Mutex<BTreeMap<u32, Arc<KvEventPublisher>>>,
    sources: Vec<ReplaySource>,
    client: Client,
    worker_id: u64,
    descriptor: Value,
    cancel: CancellationToken,
}

impl Recovery {
    pub(crate) fn new(
        client: Client,
        sources: Vec<ReplaySource>,
        worker_id: u64,
        descriptor: Value,
        cancel: CancellationToken,
    ) -> Arc<Self> {
        Arc::new(Self {
            ready: watch::channel(false).0,
            publishers: Mutex::new(BTreeMap::new()),
            sources,
            client,
            worker_id,
            descriptor,
            cancel,
        })
    }

    pub(crate) fn is_active(&self) -> bool {
        !self.publishers.lock().unwrap().is_empty()
    }

    pub(crate) fn readiness(&self) -> watch::Receiver<bool> {
        self.ready.subscribe()
    }

    pub(crate) fn attach(self: &Arc<Self>, rank: u32, publisher: Arc<KvEventPublisher>) {
        let mut publishers = self.publishers.lock().unwrap();
        assert!(
            publishers.insert(rank, publisher).is_none(),
            "duplicate recovery publisher attachment"
        );
        if publishers.len() == self.sources.len() {
            let publishers = publishers.clone();
            let this = self.clone();
            tokio::spawn(async move {
                let result = tokio::select! {
                    _ = this.cancel.cancelled() => Ok(()),
                    result = this.run(publishers) => result,
                };
                this.ready.send_replace(false);
                if let Err(error) = result {
                    tracing::error!(%error, "SGLang KV recovery stopped; sidecar remains unready");
                }
            });
        }
    }

    async fn run(&self, publishers: BTreeMap<u32, Arc<KvEventPublisher>>) -> Result<()> {
        let mut shutdown_requested = None;
        loop {
            self.ready.send_replace(false);
            let mut client = self.client.clone();
            let stream = tokio::time::timeout(
                Duration::from_secs(5),
                client.watch_engine_state(pb::WatchEngineStateRequest {}),
            )
            .await;
            let mut stream = match stream {
                Ok(Ok(response)) => response.into_inner(),
                Ok(Err(error)) if error.code() == tonic::Code::Unimplemented => {
                    bail!("KV recovery requires SGLang WatchEngineState support")
                }
                other => {
                    tracing::warn!(?other, "Waiting for SGLang engine state before KV recovery");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            };
            let snapshot = match tokio::time::timeout(Duration::from_secs(5), stream.message())
                .await
            {
                Ok(Ok(Some(snapshot))) if snapshot.instance_id != 0 && snapshot.healthy => snapshot,
                _ => {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            };
            let instance = snapshot.instance_id;
            if shutdown_requested == Some(instance) {
                // Delivery/acknowledgement can be lost. Retry only after observing
                // the same instance, with bounded rate, while staying unready.
                // The empty upstream request cannot fence a replacement racing
                // this dispatch; that may cause one additional engine restart.
                let result = tokio::time::timeout(
                    Duration::from_secs(5),
                    client.shutdown(pb::ShutdownRequest {}),
                )
                .await;
                tracing::debug!(
                    ?result,
                    instance,
                    "Retrying shutdown for unrecoverable engine"
                );
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
            let info: Value = serde_json::from_str(
                &snapshot
                    .server_info
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("engine state omitted server info"))?
                    .json_info,
            )?;
            if [
                "kv_events",
                "kv_events_config",
                "model_path",
                "tp_size",
                "dp_size",
                "disaggregation_mode",
            ]
            .iter()
            .any(|key| info.get(*key) != self.descriptor.get(*key))
            {
                bail!(
                    "SGLang KV source topology changed; sidecar must be relaunched with new metadata"
                );
            }

            // Sessions are cancelled and joined before clearing. No old task can
            // enqueue an old-incarnation event after this reset boundary.
            for (&rank, publisher) in &publishers {
                publisher
                    .publish_recovery_batch(vec![PlacementEvent::local_gpu(
                        self.worker_id,
                        KvCacheEvent {
                            event_id: publisher.next_event_id(),
                            data: KvCacheEventData::Cleared,
                            dp_rank: rank,
                        },
                    )])
                    .await?;
            }
            let session = self.cancel.child_token();
            let mut tasks = tokio::task::JoinSet::new();
            let mut statuses = Vec::new();
            for source in &self.sources {
                let (tx, rx) = watch::channel(RecoveryStatus::Recovering);
                statuses.push(rx);
                let publisher = publishers[&source.dp_rank].clone();
                let source = source.clone();
                let cancel = session.clone();
                let worker_id = self.worker_id;
                tasks.spawn(
                    async move { tokio::select! {
                        _ = cancel.cancelled() => Ok(()),
                        result = run_source(source, publisher, worker_id, cancel.clone(), tx) => result,
                    } },
                );
            }
            let mut missing = None;
            let mut engine_serving = !snapshot.is_pause;
            loop {
                tokio::select! {
                    biased;
                    _ = self.cancel.cancelled() => { session.cancel(); tasks.abort_all(); return Ok(()); }
                    snapshot = stream.message() => {
                        match snapshot {
                            Ok(Some(snapshot)) if snapshot.instance_id == instance && snapshot.healthy => { engine_serving = !snapshot.is_pause; }
                            _ => break,
                        }
                    }
                    task = tasks.join_next() => {
                        tracing::warn!(?task, "KV source stopped; retrying bootstrap without assuming missing history");
                        break;
                    }
                    _ = futures::future::select_all(statuses.iter_mut().map(|rx| Box::pin(rx.changed()))) => {}
                }
                let states: Vec<_> = statuses.iter().map(|rx| rx.borrow().clone()).collect();
                missing = states.iter().find_map(|s| match s {
                    RecoveryStatus::MissingHistory { expected, got } => Some((*expected, *got)),
                    _ => None,
                });
                if missing.is_some() {
                    break;
                }
                let now = states.iter().all(|s| matches!(s, RecoveryStatus::Ready));
                if now && engine_serving && !*self.ready.borrow() {
                    // Both ZMQ handshakes and replay have completed. Recheck
                    // the engine identity before advertising their result.
                    // Source sockets cannot reconnect within this session.
                    let mut verifier = self.client.clone();
                    let check = tokio::time::timeout(Duration::from_secs(5), async {
                        let mut state = verifier
                            .watch_engine_state(pb::WatchEngineStateRequest {})
                            .await?
                            .into_inner();
                        state.message().await
                    })
                    .await;
                    match check {
                        Ok(Ok(Some(snapshot)))
                            if snapshot.instance_id == instance && snapshot.healthy =>
                        {
                            engine_serving = !snapshot.is_pause;
                        }
                        _ => break,
                    }
                }
                // A rank may have failed while instance verification awaited.
                let now = engine_serving
                    && now
                    && statuses
                        .iter()
                        .all(|rx| matches!(*rx.borrow(), RecoveryStatus::Ready));
                self.ready.send_if_modified(|ready| {
                    let changed = *ready != now;
                    *ready = now;
                    changed
                });
            }
            self.ready.send_replace(false);
            session.cancel();
            while tasks.join_next().await.is_some() {}
            if let Some((expected, got)) = missing {
                tracing::warn!(
                    instance,
                    expected,
                    got,
                    "KV history unavailable; requesting configured SGLang shutdown"
                );
                shutdown_requested = Some(instance);
                // A healthy state update is not a reason to cancel dispatch.
                // The deadline bounds delivery; completion is observed via the
                // engine-state stream, never inferred from an acknowledgement.
                let result = tokio::time::timeout(
                    Duration::from_secs(5),
                    client.shutdown(pb::ShutdownRequest {}),
                )
                .await;
                tracing::info!(
                    ?result,
                    instance,
                    "Shutdown attempted; waiting for a new engine instance"
                );
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
}
