// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-free cross-replica in-flight-load sync for the EPP router.
//!
//! When the EPP runs in vanilla (non-Dynamo-runtime) mode there is no NATS
//! event plane and no `DynamoWorkerMetadata` CRD, so the standalone router's
//! replica-sync mechanism (which discovers peer routers and exchanges
//! `ActiveSequenceEvent`s over those) is unavailable. This module provides an
//! equivalent that depends only on facilities already present in a GAIE
//! deployment:
//!
//! - **Peer discovery** via the `EndpointSlice`s of the EPP's own Kubernetes
//!   `Service` (one ready entry per EPP pod). The Downward-API `POD_IP`
//!   excludes this replica from its own peer set.
//! - **Transport** via a fixed-port ZMQ PUB/SUB mesh: each EPP binds a PUB on
//!   `DYN_EPP_REPLICA_SYNC_PORT` and a single SUB socket connects (fan-in) to
//!   every peer's PUB. ZMQ handles reconnection as peers come and go.
//!
//! The received events are fed into the decode router's active-sequence
//! tracker via [`dynamo_llm::kv_router::KvRouter::start_replica_sync`], so a
//! replica's projected load reflects requests booked on its peers.
//!
//! See ai-dynamo/dynamo#10384.
//!
//! ## v1 limitation
//!
//! Published `AddRequest` events carry `token_sequence: None`: the EPP books
//! requests through [`crate::epp::Router::add_request`] but does not retain the
//! router-internal block hashes. Peers therefore mirror in-flight request
//! *count* per worker exactly, but the token-level prefill/decode load
//! contribution of peer requests is approximated as zero until the events carry
//! a token count or block hashes.

use std::collections::HashSet;

use anyhow::Result;
use dynamo_kv_router::protocols::{ActiveSequenceEvent, ActiveSequenceEventData, WorkerWithDpRank};
use dynamo_kv_router::{SequenceRequest, SequenceSubscriber};
use dynamo_runtime::CancellationToken;
use futures::{SinkExt, StreamExt};
use std::collections::HashMap;
use tmq::{AsZmqSocket, Context, Multipart, publish::Publish, subscribe::Subscribe};
use tokio::sync::{Mutex, mpsc, watch};

/// ZMQ topic shared by all EPP replicas for active-sequence events.
const TOPIC: &[u8] = b"epp-replica-sync";

/// Publishes this replica's active-sequence events to peer EPPs over a
/// fixed-port ZMQ PUB socket.
///
/// `Free` and `MarkPrefillCompleted` are addressed to a worker, but the EPP
/// call sites for those only carry a request id, so this tracks
/// `request_id -> worker` (populated on `on_add`, consumed on `on_free`).
pub struct ReplicaPublisher {
    socket: Mutex<Publish>,
    router_id: u64,
    req_workers: Mutex<HashMap<String, WorkerWithDpRank>>,
}

impl ReplicaPublisher {
    /// Bind the PUB socket on `0.0.0.0:port`. `router_id` must match the decode
    /// router's own id (`drt.discovery().instance_id()`) so that peers apply
    /// these events and this replica never re-applies its own.
    pub async fn bind(port: u16, router_id: u64) -> Result<Self> {
        let endpoint = format!("tcp://0.0.0.0:{port}");
        let ctx = Context::new();
        let socket = tmq::publish::publish(&ctx).bind(&endpoint)?;
        tracing::info!(%endpoint, router_id, "EPP replica-sync PUB bound");
        Ok(Self {
            socket: Mutex::new(socket),
            router_id,
            req_workers: Mutex::new(HashMap::new()),
        })
    }

    async fn send(&self, event: &ActiveSequenceEvent) {
        let payload = match serde_json::to_vec(event) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, "failed to serialize replica-sync event");
                return;
            }
        };
        let frames = Multipart::from(vec![TOPIC.to_vec(), payload]);
        if let Err(e) = self.socket.lock().await.send(frames).await {
            tracing::warn!(error = %e, "failed to publish replica-sync event");
        }
    }

    /// Mirror a freshly-booked request to peers, at full fidelity.
    ///
    /// Takes the exact [`SequenceRequest`] the router booked (returned by
    /// `KvRouter::add_request`) so the published event carries the real token
    /// hashes, prefill-token tracking flag, expected output tokens and
    /// prefill-load hint — matching what the standalone router's own
    /// replica-sync publisher would emit.
    pub async fn on_add(&self, req: &SequenceRequest) {
        self.req_workers
            .lock()
            .await
            .insert(req.request_id.clone(), req.worker);
        let event = ActiveSequenceEvent {
            request_id: req.request_id.clone(),
            worker: req.worker,
            data: ActiveSequenceEventData::AddRequest {
                token_sequence: req.token_sequence.clone(),
                track_prefill_tokens: req.track_prefill_tokens,
                expected_output_tokens: req.expected_output_tokens,
                prefill_load_hint: req.prefill_load_hint,
            },
            router_id: self.router_id,
            lora_name: req.lora_name.clone(),
        };
        self.send(&event).await;
    }

    /// Mirror a prefill-completion transition to peers.
    pub async fn on_mark_prefill(&self, request_id: &str) {
        let Some(worker) = self.req_workers.lock().await.get(request_id).copied() else {
            return;
        };
        let event = ActiveSequenceEvent {
            request_id: request_id.to_string(),
            worker,
            data: ActiveSequenceEventData::MarkPrefillCompleted,
            router_id: self.router_id,
            lora_name: None,
        };
        self.send(&event).await;
    }

    /// Mirror a request completion (free) to peers.
    pub async fn on_free(&self, request_id: &str) {
        let Some(worker) = self.req_workers.lock().await.remove(request_id) else {
            return;
        };
        let event = ActiveSequenceEvent {
            request_id: request_id.to_string(),
            worker,
            data: ActiveSequenceEventData::Free,
            router_id: self.router_id,
            lora_name: None,
        };
        self.send(&event).await;
    }
}

/// [`SequenceSubscriber`] backed by an mpsc channel that the peer SUB pump
/// feeds with decoded peer events.
pub struct ReplicaSubscriber {
    rx: mpsc::Receiver<ActiveSequenceEvent>,
}

impl SequenceSubscriber for ReplicaSubscriber {
    async fn next_event(&mut self) -> Option<anyhow::Result<ActiveSequenceEvent>> {
        self.rx.recv().await.map(Ok)
    }
}

/// Bound on buffered peer events; under a burst or a stalled sequence-sync
/// consumer the newest events are dropped (with a warning) rather than growing
/// memory without limit.
const REPLICA_SYNC_EVENT_BUFFER: usize = 8192;

/// Start peer discovery (EndpointSlice reflector) plus the ZMQ SUB pump and
/// return a [`ReplicaSubscriber`] yielding peer events. Wire the returned
/// subscriber into `KvRouter::start_replica_sync`.
pub async fn spawn_peer_sync(
    service_name: String,
    namespace: String,
    self_pod_ip: String,
    port: u16,
    cancel: CancellationToken,
) -> Result<ReplicaSubscriber> {
    let peers_rx =
        spawn_endpointslice_reflector(service_name, namespace, self_pod_ip, port).await?;
    let (event_tx, event_rx) = mpsc::channel(REPLICA_SYNC_EVENT_BUFFER);
    tokio::spawn(sub_manager(peers_rx, event_tx, cancel));
    Ok(ReplicaSubscriber { rx: event_rx })
}

/// Watch the EndpointSlices of the EPP's Service and publish the current set of
/// peer PUB endpoints (`tcp://ip:port`, self excluded) to a watch channel.
async fn spawn_endpointslice_reflector(
    service_name: String,
    namespace: String,
    self_pod_ip: String,
    port: u16,
) -> Result<watch::Receiver<Vec<String>>> {
    use k8s_openapi::api::discovery::v1::EndpointSlice;
    use kube::{Api, Client, runtime::reflector, runtime::watcher};

    let client = Client::try_default().await?;
    let api: Api<EndpointSlice> = Api::namespaced(client, &namespace);
    let selector = format!("kubernetes.io/service-name={service_name}");
    let cfg = watcher::Config::default().labels(&selector);

    let (reader, writer) = reflector::store();
    let stream = reflector::reflector(writer, watcher(api, cfg));
    let (tx, rx) = watch::channel(Vec::new());

    tracing::info!(
        namespace = %namespace,
        selector = %selector,
        port,
        "Starting EPP EndpointSlice reflector for peer discovery"
    );

    tokio::spawn(async move {
        tokio::pin!(stream);
        while stream.next().await.is_some() {
            let peers = compute_peers(&reader, &self_pod_ip, port);
            tracing::debug!(peer_count = peers.len(), peers = ?peers, "EPP peer set updated");
            // Ignore send errors: a closed receiver means sync has shut down.
            let _ = tx.send(peers);
        }
        tracing::warn!("EPP EndpointSlice reflector stream ended");
    });

    Ok(rx)
}

/// Collect ready peer endpoints from the EndpointSlice store, excluding self.
fn compute_peers(
    store: &kube::runtime::reflector::Store<k8s_openapi::api::discovery::v1::EndpointSlice>,
    self_pod_ip: &str,
    port: u16,
) -> Vec<String> {
    let mut peers = HashSet::new();
    for slice in store.state() {
        for endpoint in slice.endpoints.iter() {
            // Kubernetes treats a missing `ready` condition as ready.
            let ready = endpoint
                .conditions
                .as_ref()
                .and_then(|c| c.ready)
                .unwrap_or(true);
            if !ready {
                continue;
            }
            for addr in endpoint.addresses.iter() {
                if addr == self_pod_ip {
                    continue;
                }
                // Bracket IPv6 literals so the ZMQ `tcp://` endpoint is valid.
                let endpoint = match addr.parse::<std::net::IpAddr>() {
                    Ok(std::net::IpAddr::V6(v6)) => format!("tcp://[{v6}]:{port}"),
                    Ok(std::net::IpAddr::V4(v4)) => format!("tcp://{v4}:{port}"),
                    Err(_) => {
                        tracing::warn!(addr, "skipping unparsable peer address");
                        continue;
                    }
                };
                peers.insert(endpoint);
            }
        }
    }
    let mut peers: Vec<String> = peers.into_iter().collect();
    peers.sort();
    peers
}

/// Rebuild the SUB socket whenever the peer set changes and pump decoded events
/// into `event_tx`.
async fn sub_manager(
    mut peers_rx: watch::Receiver<Vec<String>>,
    event_tx: mpsc::Sender<ActiveSequenceEvent>,
    cancel: CancellationToken,
) {
    let ctx = Context::new();
    loop {
        let peers = peers_rx.borrow_and_update().clone();
        let pump = if peers.is_empty() {
            None
        } else {
            match build_sub(&ctx, &peers) {
                Ok(sub) => {
                    tracing::info!(peer_count = peers.len(), "EPP replica-sync SUB connected");
                    Some(tokio::spawn(pump_sub(sub, event_tx.clone())))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to build replica-sync SUB socket");
                    None
                }
            }
        };

        tokio::select! {
            _ = cancel.cancelled() => {
                if let Some(h) = pump { h.abort(); }
                break;
            }
            changed = peers_rx.changed() => {
                if let Some(h) = pump { h.abort(); }
                if changed.is_err() {
                    break;
                }
            }
        }
    }
}

/// Build a single SUB socket connected (fan-in) to every peer PUB endpoint.
fn build_sub(ctx: &Context, peers: &[String]) -> Result<Subscribe> {
    let mut iter = peers.iter();
    let first = iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("build_sub called with no peers"))?;
    let sub = tmq::subscribe::subscribe(ctx)
        .connect(first)?
        .subscribe(TOPIC)?;
    for endpoint in iter {
        sub.get_socket().connect(endpoint)?;
    }
    Ok(sub)
}

/// Read multipart frames, decode the payload, and forward events.
async fn pump_sub(mut sub: Subscribe, event_tx: mpsc::Sender<ActiveSequenceEvent>) {
    while let Some(result) = sub.next().await {
        let msg = match result {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!(error = %e, "replica-sync SUB recv error");
                continue;
            }
        };
        let Some(payload) = msg.0.back() else {
            continue;
        };
        match serde_json::from_slice::<ActiveSequenceEvent>(&payload[..]) {
            Ok(event) => match event_tx.try_send(event) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    tracing::warn!("replica-sync event buffer full; dropping peer event");
                }
                Err(mpsc::error::TrySendError::Closed(_)) => break, // subscriber dropped
            },
            Err(e) => tracing::debug!(error = %e, "failed to decode replica-sync event"),
        }
    }
}
