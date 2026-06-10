// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::future::{self, Future};
use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::{Context as _, Result, anyhow};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::protocols::{ActiveLoad, ActiveSequenceEvent, WorkerWithDpRank};
use crate::sequences::{SequencePublisher, SequenceSubscriber};
use crate::services::zmq::{create_bound_pub_socket, create_sub_socket, validate_endpoint};

use super::registry::SlotTrackerRegistry;

pub(crate) const REPLICA_EVENT_CHANNEL_CAPACITY: usize = 100_000;
const PEER_COMMAND_CHANNEL_CAPACITY: usize = 64;
const REPLICA_TOPIC: &[u8] = b"dynamo.slot-tracker.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SlotReplicaEvent {
    pub model_name: String,
    pub tenant_id: String,
    pub block_size: u32,
    pub event: ActiveSequenceEvent,
}

pub(crate) type ReplicaEventSender = mpsc::Sender<SlotReplicaEvent>;

#[derive(Clone)]
pub(crate) struct ScopedSequencePublisher {
    replica: Option<ScopedReplicaPublisher>,
}

#[derive(Clone)]
struct ScopedReplicaPublisher {
    model_name: Arc<str>,
    tenant_id: Arc<str>,
    block_size: u32,
    tx: ReplicaEventSender,
}

impl ScopedSequencePublisher {
    pub(crate) fn disabled() -> Self {
        Self { replica: None }
    }

    pub(crate) fn enabled(
        model_name: Arc<str>,
        tenant_id: Arc<str>,
        block_size: u32,
        tx: ReplicaEventSender,
    ) -> Self {
        Self {
            replica: Some(ScopedReplicaPublisher {
                model_name,
                tenant_id,
                block_size,
                tx,
            }),
        }
    }
}

impl SequencePublisher for ScopedSequencePublisher {
    fn publish_event(
        &self,
        event: &ActiveSequenceEvent,
    ) -> impl Future<Output = Result<()>> + Send {
        let Some(replica) = &self.replica else {
            return future::ready(Ok(()));
        };
        let envelope = SlotReplicaEvent {
            model_name: replica.model_name.to_string(),
            tenant_id: replica.tenant_id.to_string(),
            block_size: replica.block_size,
            event: event.clone(),
        };
        let result = match replica.tx.try_send(envelope) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(event)) => {
                tracing::trace!(
                    model_name = %event.model_name,
                    tenant_id = %event.tenant_id,
                    request_id = %event.event.request_id,
                    "Replica publisher channel full; dropping event"
                );
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(anyhow!("replica publisher channel is closed"))
            }
        };
        future::ready(result)
    }

    fn publish_load(&self, _load: ActiveLoad) {}

    fn observe_load(
        &self,
        _worker: &WorkerWithDpRank,
        _worker_type: &str,
        _blocks: usize,
        _tokens: usize,
    ) {
    }
}

pub(crate) struct ChannelSequenceSubscriber {
    rx: mpsc::Receiver<ActiveSequenceEvent>,
}

impl ChannelSequenceSubscriber {
    pub(crate) fn new(rx: mpsc::Receiver<ActiveSequenceEvent>) -> Self {
        Self { rx }
    }
}

impl SequenceSubscriber for ChannelSequenceSubscriber {
    async fn next_event(&mut self) -> Option<Result<ActiveSequenceEvent>> {
        self.rx.recv().await.map(Ok)
    }

    fn poll_next_event(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<ActiveSequenceEvent>>> {
        self.rx.poll_recv(cx).map(|event| event.map(Ok))
    }
}

pub(crate) fn generate_process_id() -> u64 {
    loop {
        let id = rand::random();
        if id != 0 {
            return id;
        }
    }
}

pub(crate) fn start_replica_publisher(
    bind_endpoint: &str,
    cancel_token: CancellationToken,
) -> Result<ReplicaEventSender> {
    validate_endpoint(bind_endpoint)?;
    let mut socket = create_bound_pub_socket(bind_endpoint)
        .with_context(|| format!("failed to bind replica publisher to `{bind_endpoint}`"))?;
    let (tx, mut rx) = mpsc::channel::<SlotReplicaEvent>(REPLICA_EVENT_CHANNEL_CAPACITY);

    tokio::spawn(async move {
        loop {
            let event = tokio::select! {
                _ = cancel_token.cancelled() => break,
                event = rx.recv() => {
                    let Some(event) = event else {
                        break;
                    };
                    event
                }
            };

            let request_id = event.event.request_id.clone();
            let payload = match rmp_serde::to_vec_named(&event) {
                Ok(payload) => payload,
                Err(error) => {
                    tracing::error!(
                        request_id = %request_id,
                        "Failed to encode slot-tracker replica event: {error}"
                    );
                    continue;
                }
            };
            if let Err(error) = socket
                .send_multipart(vec![REPLICA_TOPIC.to_vec(), payload])
                .await
            {
                tracing::error!(
                    request_id = %request_id,
                    "Failed to publish slot-tracker replica event: {error}"
                );
            }
        }
    });

    Ok(tx)
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum PeerError {
    #[error(transparent)]
    InvalidEndpoint(#[from] anyhow::Error),

    #[error("peer endpoint matches this replica's advertised endpoint")]
    SelfEndpoint,

    #[error("replica peer manager is unavailable")]
    Unavailable,
}

#[derive(Clone)]
pub(crate) struct PeerManager {
    command_tx: mpsc::Sender<PeerCommand>,
    peers: Arc<RwLock<HashSet<String>>>,
    advertised_endpoint: Option<Arc<str>>,
}

enum PeerCommand {
    Register {
        endpoint: String,
        response: oneshot::Sender<Result<bool>>,
    },
    Deregister {
        endpoint: String,
        response: oneshot::Sender<Result<bool>>,
    },
}

impl PeerManager {
    pub(crate) fn start(
        registry: Arc<SlotTrackerRegistry>,
        initial_peers: Vec<String>,
        advertised_endpoint: Option<String>,
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        let advertised_endpoint = advertised_endpoint.map(Arc::<str>::from);
        if let Some(endpoint) = advertised_endpoint.as_deref() {
            validate_endpoint(endpoint)
                .with_context(|| format!("invalid advertised replica endpoint `{endpoint}`"))?;
        }
        let mut socket = create_sub_socket(REPLICA_TOPIC)?;
        let mut configured_peers = HashSet::new();
        for endpoint in initial_peers {
            validate_peer_endpoint(&endpoint, advertised_endpoint.as_deref())?;
            if configured_peers.insert(endpoint.clone()) {
                socket
                    .connect(&endpoint)
                    .with_context(|| format!("failed to register replica peer `{endpoint}`"))?;
            }
        }

        let peers = Arc::new(RwLock::new(configured_peers));
        let (command_tx, mut command_rx) = mpsc::channel(PEER_COMMAND_CHANNEL_CAPACITY);
        let task_peers = Arc::clone(&peers);
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => break,
                    command = command_rx.recv() => {
                        let Some(command) = command else {
                            break;
                        };
                        handle_peer_command(&socket, &task_peers, command);
                    }
                    message = socket.recv_multipart() => {
                        match message {
                            Ok(frames) => handle_replica_message(&registry, frames),
                            Err(error) => {
                                tracing::error!("Failed to receive slot-tracker replica event: {error}");
                            }
                        }
                    }
                }
            }
        });

        Ok(Self {
            command_tx,
            peers,
            advertised_endpoint,
        })
    }

    pub(crate) async fn register_peer(&self, endpoint: String) -> Result<bool, PeerError> {
        validate_peer_endpoint(&endpoint, self.advertised_endpoint.as_deref())?;
        let (response, result) = oneshot::channel();
        self.command_tx
            .send(PeerCommand::Register { endpoint, response })
            .await
            .map_err(|_| PeerError::Unavailable)?;
        result
            .await
            .map_err(|_| PeerError::Unavailable)?
            .map_err(PeerError::InvalidEndpoint)
    }

    pub(crate) async fn deregister_peer(&self, endpoint: String) -> Result<bool, PeerError> {
        validate_endpoint(&endpoint).map_err(PeerError::InvalidEndpoint)?;
        let (response, result) = oneshot::channel();
        self.command_tx
            .send(PeerCommand::Deregister { endpoint, response })
            .await
            .map_err(|_| PeerError::Unavailable)?;
        result
            .await
            .map_err(|_| PeerError::Unavailable)?
            .map_err(PeerError::InvalidEndpoint)
    }

    pub(crate) fn list_peers(&self) -> Vec<String> {
        let mut peers: Vec<_> = self.peers.read().iter().cloned().collect();
        peers.sort();
        peers
    }
}

fn validate_peer_endpoint(
    endpoint: &str,
    advertised_endpoint: Option<&str>,
) -> Result<(), PeerError> {
    validate_endpoint(endpoint).map_err(PeerError::InvalidEndpoint)?;
    if advertised_endpoint.is_some_and(|advertised| endpoint == advertised) {
        return Err(PeerError::SelfEndpoint);
    }
    Ok(())
}

fn handle_peer_command(
    socket: &crate::services::zmq::ZmqSocket,
    peers: &RwLock<HashSet<String>>,
    command: PeerCommand,
) {
    match command {
        PeerCommand::Register { endpoint, response } => {
            if peers.read().contains(&endpoint) {
                let _ = response.send(Ok(false));
                return;
            }
            let result = socket
                .connect(&endpoint)
                .with_context(|| format!("failed to register replica peer `{endpoint}`"))
                .map(|()| {
                    peers.write().insert(endpoint);
                    true
                });
            let _ = response.send(result);
        }
        PeerCommand::Deregister { endpoint, response } => {
            if !peers.read().contains(&endpoint) {
                let _ = response.send(Ok(false));
                return;
            }
            let result = socket
                .disconnect(&endpoint)
                .with_context(|| format!("failed to deregister replica peer `{endpoint}`"))
                .map(|()| {
                    peers.write().remove(&endpoint);
                    true
                });
            let _ = response.send(result);
        }
    }
}

fn handle_replica_message(
    registry: &SlotTrackerRegistry,
    frames: crate::services::zmq::MultipartMessage,
) {
    let [topic, payload] = frames.as_slice() else {
        tracing::debug!(
            frame_count = frames.len(),
            "Dropping malformed slot-tracker replica message"
        );
        return;
    };
    if topic.as_slice() != REPLICA_TOPIC {
        tracing::debug!("Dropping slot-tracker replica message with unexpected topic");
        return;
    }
    let event: SlotReplicaEvent = match rmp_serde::from_slice(payload) {
        Ok(event) => event,
        Err(error) => {
            tracing::debug!("Dropping malformed slot-tracker replica payload: {error}");
            return;
        }
    };
    registry.dispatch_replica_event(event);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{ActiveSequenceEventData, WorkerWithDpRank};
    use crate::services::slot_tracker::registry::TrackerKey;

    fn event() -> SlotReplicaEvent {
        SlotReplicaEvent {
            model_name: "model".to_string(),
            tenant_id: "tenant".to_string(),
            block_size: 16,
            event: ActiveSequenceEvent {
                request_id: "request".to_string(),
                worker: WorkerWithDpRank::new(1, 0),
                data: ActiveSequenceEventData::Free,
                router_id: 42,
                lora_name: None,
            },
        }
    }

    #[test]
    fn replica_event_round_trips_named_messagepack() {
        let encoded = rmp_serde::to_vec_named(&event()).unwrap();
        let decoded: SlotReplicaEvent = rmp_serde::from_slice(&encoded).unwrap();

        assert_eq!(decoded.model_name, "model");
        assert_eq!(decoded.tenant_id, "tenant");
        assert_eq!(decoded.block_size, 16);
        assert_eq!(decoded.event.request_id, "request");
    }

    #[test]
    fn process_identity_is_nonzero() {
        assert_ne!(generate_process_id(), 0);
    }

    #[test]
    fn advertised_endpoint_cannot_be_registered_as_peer() {
        let error = validate_peer_endpoint("tcp://127.0.0.1:8092", Some("tcp://127.0.0.1:8092"))
            .unwrap_err();

        assert!(error.to_string().contains("advertised endpoint"));
    }

    #[tokio::test]
    async fn scoped_publisher_drops_when_outbound_channel_is_full() {
        let (tx, mut rx) = mpsc::channel(1);
        let publisher =
            ScopedSequencePublisher::enabled(Arc::from("model"), Arc::from("tenant"), 16, tx);
        let event = event().event;

        publisher.publish_event(&event).await.unwrap();
        publisher.publish_event(&event).await.unwrap();

        assert_eq!(rx.len(), 1);
        assert_eq!(rx.recv().await.unwrap().event.request_id, "request");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn peer_manager_registers_and_deregisters_endpoints() {
        let cancel_token = CancellationToken::new();
        let registry = Arc::new(SlotTrackerRegistry::new(cancel_token.clone()));
        let manager = PeerManager::start(
            registry,
            Vec::new(),
            Some("tcp://127.0.0.1:8092".to_string()),
            cancel_token.clone(),
        )
        .unwrap();
        let endpoint = "tcp://127.0.0.1:8093".to_string();

        assert!(manager.register_peer(endpoint.clone()).await.unwrap());
        assert!(!manager.register_peer(endpoint.clone()).await.unwrap());
        assert_eq!(manager.list_peers(), vec![endpoint.clone()]);
        assert!(manager.deregister_peer(endpoint.clone()).await.unwrap());
        assert!(!manager.deregister_peer(endpoint).await.unwrap());
        assert!(manager.list_peers().is_empty());

        cancel_token.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn zmq_replica_sync_propagates_request_lifecycle() {
        let endpoint_a = reserve_tcp_endpoint();
        let endpoint_b = reserve_tcp_endpoint();
        let cancel_token = CancellationToken::new();
        let outbound_a = start_replica_publisher(&endpoint_a, cancel_token.child_token()).unwrap();
        let outbound_b = start_replica_publisher(&endpoint_b, cancel_token.child_token()).unwrap();
        let registry_a = Arc::new(SlotTrackerRegistry::new_with_replica_sync(
            cancel_token.clone(),
            11,
            outbound_a,
        ));
        let registry_b = Arc::new(SlotTrackerRegistry::new_with_replica_sync(
            cancel_token.clone(),
            22,
            outbound_b,
        ));
        let _peer_b = PeerManager::start(
            Arc::clone(&registry_b),
            vec![endpoint_a],
            Some(endpoint_b),
            cancel_token.child_token(),
        )
        .unwrap();
        let key = TrackerKey::new("model".to_string(), Some("tenant".to_string()));
        registry_a.register(key.clone(), 1, 16, 0, 1).unwrap();
        registry_b.register(key.clone(), 1, 16, 0, 1).unwrap();
        let worker = WorkerWithDpRank::new(1, 0);

        let mut warmup_requests = Vec::new();
        for attempt in 0..40 {
            let request_id = format!("warmup-{attempt}");
            registry_a
                .add_request(&key, request_id.clone(), worker, vec![attempt], 0)
                .unwrap();
            warmup_requests.push(request_id);
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            if registry_b.list_loads(None, None)[0].active_decode_blocks > 0 {
                break;
            }
        }
        assert!(registry_b.list_loads(None, None)[0].active_decode_blocks > 0);

        for request_id in &warmup_requests {
            registry_a.free(&key, request_id).unwrap();
        }
        wait_for_load(&registry_b, 0, 0).await;

        registry_a
            .add_request(&key, "target".to_string(), worker, vec![1, 2, 3], 8)
            .unwrap();
        wait_for_load(&registry_b, 3, 8).await;

        registry_a.mark_prefill_completed(&key, "target").unwrap();
        wait_for_load(&registry_b, 3, 0).await;

        registry_a.free(&key, "target").unwrap();
        wait_for_load(&registry_b, 0, 0).await;
        cancel_token.cancel();
    }

    async fn wait_for_load(
        registry: &SlotTrackerRegistry,
        expected_blocks: usize,
        expected_tokens: usize,
    ) {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                let load = &registry.list_loads(None, None)[0];
                if load.active_decode_blocks == expected_blocks
                    && load.active_prefill_tokens == expected_tokens
                {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }

    fn reserve_tcp_endpoint() -> String {
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("failed to reserve TCP port");
        let endpoint = format!("tcp://127.0.0.1:{}", listener.local_addr().unwrap().port());
        drop(listener);
        endpoint
    }
}
