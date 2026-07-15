// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::future::{self, Future};
use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::{Context as _, Result, anyhow};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::protocols::{ActiveLoad, ActiveSequenceEvent, WorkerWithDpRank};
use crate::sequences::{SequencePublisher, SequenceSubscriber};
use crate::services::common::zmq::{create_bound_pub_socket, create_sub_socket, validate_endpoint};

pub(crate) const REPLICA_EVENT_CHANNEL_CAPACITY: usize = 100_000;
const PEER_COMMAND_CHANNEL_CAPACITY: usize = 64;
const REPLICA_TOPIC: &[u8] = b"dynamo.slot-tracker.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ScopedReplicaEvent {
    pub model_name: String,
    pub routing_group: String,
    pub block_size: u32,
    pub event: ActiveSequenceEvent,
}

pub(crate) type ReplicaEventSender = mpsc::Sender<ScopedReplicaEvent>;

#[derive(Debug, Clone)]
pub(crate) struct ReplicaSyncConfig {
    process_id: u64,
    outbound_tx: ReplicaEventSender,
}

#[derive(Debug)]
pub(crate) struct ReplicaSyncRuntime {
    config: ReplicaSyncConfig,
    cancel_token: CancellationToken,
    publisher_task: Mutex<Option<JoinHandle<()>>>,
}

impl ReplicaSyncRuntime {
    pub(crate) fn config(&self) -> ReplicaSyncConfig {
        self.config.clone()
    }

    pub(crate) async fn shutdown(&self) {
        self.cancel_token.cancel();
        if let Some(task) = self.publisher_task.lock().await.take() {
            let _ = task.await;
        }
    }

    fn abort(&self) {
        self.cancel_token.cancel();
        if let Ok(mut task) = self.publisher_task.try_lock()
            && let Some(task) = task.take()
        {
            task.abort();
        }
    }
}

impl Drop for ReplicaSyncRuntime {
    fn drop(&mut self) {
        self.abort();
    }
}

impl ReplicaSyncConfig {
    pub(crate) fn new(process_id: u64, outbound_tx: ReplicaEventSender) -> Self {
        Self {
            process_id,
            outbound_tx,
        }
    }

    #[cfg(feature = "standalone-slot-tracker")]
    pub(crate) fn process_id(&self) -> u64 {
        self.process_id
    }

    pub(crate) fn is_self_event(&self, event: &ActiveSequenceEvent) -> bool {
        event.router_id == self.process_id
    }
}

pub(crate) struct ScopedReplicaSync {
    pub publisher: ScopedSequencePublisher,
    pub enabled: bool,
    pub process_id: u64,
    pub channel: Option<(mpsc::Sender<ActiveSequenceEvent>, ChannelSequenceSubscriber)>,
}

#[derive(Clone)]
pub(crate) struct ScopedSequencePublisher {
    replica: Option<ScopedReplicaPublisher>,
}

#[derive(Clone)]
struct ScopedReplicaPublisher {
    model_name: Arc<str>,
    routing_group: Arc<str>,
    block_size: u32,
    tx: ReplicaEventSender,
}

impl ScopedSequencePublisher {
    pub(crate) fn disabled() -> Self {
        Self { replica: None }
    }

    pub(crate) fn enabled(
        model_name: Arc<str>,
        routing_group: Arc<str>,
        block_size: u32,
        tx: ReplicaEventSender,
    ) -> Self {
        Self {
            replica: Some(ScopedReplicaPublisher {
                model_name,
                routing_group,
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
        let envelope = ScopedReplicaEvent {
            model_name: replica.model_name.to_string(),
            routing_group: replica.routing_group.to_string(),
            block_size: replica.block_size,
            event: event.clone(),
        };
        let result = match replica.tx.try_send(envelope) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(event)) => {
                tracing::trace!(
                    model_name = %event.model_name,
                    routing_group = %event.routing_group,
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

fn generate_process_id() -> u64 {
    loop {
        let id = fastrand::u64(..);
        if id != 0 {
            return id;
        }
    }
}

fn replica_sync_bind_endpoint(port: u16) -> Result<String> {
    if port == 0 {
        anyhow::bail!("replica sync port must be greater than zero");
    }
    Ok(format!("tcp://*:{port}"))
}

pub(crate) fn setup_replica_sync(
    port: Option<u16>,
    initial_peers: &[String],
    cancel_token: CancellationToken,
) -> Result<Option<ReplicaSyncRuntime>> {
    let Some(port) = port else {
        if !initial_peers.is_empty() {
            anyhow::bail!("--replica-sync-peers requires --replica-sync-port");
        }
        return Ok(None);
    };

    let bind_endpoint = replica_sync_bind_endpoint(port)?;
    let process_id = generate_process_id();
    let (outbound_tx, publisher_task) =
        start_replica_publisher(&bind_endpoint, cancel_token.clone())?;
    Ok(Some(ReplicaSyncRuntime {
        config: ReplicaSyncConfig::new(process_id, outbound_tx),
        cancel_token,
        publisher_task: Mutex::new(Some(publisher_task)),
    }))
}

pub(crate) fn setup_scoped_replica_sync(
    config: Option<&ReplicaSyncConfig>,
    model_name: &str,
    routing_group: &str,
    block_size: u32,
) -> ScopedReplicaSync {
    let Some(config) = config else {
        return ScopedReplicaSync {
            publisher: ScopedSequencePublisher::disabled(),
            enabled: false,
            process_id: 0,
            channel: None,
        };
    };

    let (replica_tx, replica_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
    ScopedReplicaSync {
        publisher: ScopedSequencePublisher::enabled(
            Arc::from(model_name),
            Arc::from(routing_group),
            block_size,
            config.outbound_tx.clone(),
        ),
        enabled: true,
        process_id: config.process_id,
        channel: Some((replica_tx, ChannelSequenceSubscriber::new(replica_rx))),
    }
}

pub(crate) fn start_replica_publisher(
    bind_endpoint: &str,
    cancel_token: CancellationToken,
) -> Result<(ReplicaEventSender, JoinHandle<()>)> {
    validate_endpoint(bind_endpoint)?;
    let mut socket = create_bound_pub_socket(bind_endpoint)
        .with_context(|| format!("failed to bind replica publisher to `{bind_endpoint}`"))?;
    let (tx, mut rx) = mpsc::channel::<ScopedReplicaEvent>(REPLICA_EVENT_CHANNEL_CAPACITY);

    let task = tokio::spawn(async move {
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
                        "Failed to encode active-sequence replica event: {error}"
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
                    "Failed to publish active-sequence replica event: {error}"
                );
            }
        }
    });

    Ok((tx, task))
}

#[derive(Debug, thiserror::Error)]
#[cfg_attr(not(feature = "standalone-slot-tracker"), allow(dead_code))]
pub enum ReplicaPeerError {
    #[error(transparent)]
    InvalidEndpoint(#[from] anyhow::Error),

    #[allow(dead_code)]
    #[error("replica sync is disabled")]
    Disabled,

    #[error("replica peer manager is unavailable")]
    Unavailable,
}

#[cfg_attr(not(feature = "standalone-slot-tracker"), allow(dead_code))]
pub(crate) struct PeerManager {
    command_tx: mpsc::Sender<PeerCommand>,
    peers: Arc<RwLock<HashSet<String>>>,
    cancel_token: CancellationToken,
    subscriber_task: Mutex<Option<JoinHandle<()>>>,
}

#[cfg_attr(not(feature = "standalone-slot-tracker"), allow(dead_code))]
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
    pub(crate) fn start<F>(
        initial_peers: Vec<String>,
        cancel_token: CancellationToken,
        handle_event: F,
    ) -> Result<Self>
    where
        F: Fn(ScopedReplicaEvent) + Send + Sync + 'static,
    {
        let mut socket = create_sub_socket(REPLICA_TOPIC)?;
        let mut configured_peers = HashSet::new();
        for endpoint in initial_peers {
            validate_endpoint(&endpoint)
                .with_context(|| format!("invalid replica peer endpoint `{endpoint}`"))?;
            if configured_peers.insert(endpoint.clone()) {
                socket
                    .connect(&endpoint)
                    .with_context(|| format!("failed to register replica peer `{endpoint}`"))?;
            }
        }

        let peers = Arc::new(RwLock::new(configured_peers));
        let (command_tx, mut command_rx) = mpsc::channel(PEER_COMMAND_CHANNEL_CAPACITY);
        let task_peers = Arc::clone(&peers);
        let task_cancel = cancel_token.clone();
        let subscriber_task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = task_cancel.cancelled() => break,
                    command = command_rx.recv() => {
                        let Some(command) = command else {
                            break;
                        };
                        handle_peer_command(&socket, &task_peers, command);
                    }
                    message = socket.recv_multipart() => {
                        match message {
                            Ok(frames) => handle_replica_message(&handle_event, frames),
                            Err(error) => {
                                tracing::error!("Failed to receive active-sequence replica event: {error}");
                            }
                        }
                    }
                }
            }
        });

        Ok(Self {
            command_tx,
            peers,
            cancel_token,
            subscriber_task: Mutex::new(Some(subscriber_task)),
        })
    }

    #[cfg_attr(not(feature = "standalone-slot-tracker"), allow(dead_code))]
    pub(crate) async fn register_peer(&self, endpoint: String) -> Result<bool, ReplicaPeerError> {
        validate_endpoint(&endpoint).map_err(ReplicaPeerError::InvalidEndpoint)?;
        let (response, result) = oneshot::channel();
        self.command_tx
            .send(PeerCommand::Register { endpoint, response })
            .await
            .map_err(|_| ReplicaPeerError::Unavailable)?;
        result
            .await
            .map_err(|_| ReplicaPeerError::Unavailable)?
            .map_err(ReplicaPeerError::InvalidEndpoint)
    }

    #[cfg_attr(not(feature = "standalone-slot-tracker"), allow(dead_code))]
    pub(crate) async fn deregister_peer(&self, endpoint: String) -> Result<bool, ReplicaPeerError> {
        validate_endpoint(&endpoint).map_err(ReplicaPeerError::InvalidEndpoint)?;
        let (response, result) = oneshot::channel();
        self.command_tx
            .send(PeerCommand::Deregister { endpoint, response })
            .await
            .map_err(|_| ReplicaPeerError::Unavailable)?;
        result
            .await
            .map_err(|_| ReplicaPeerError::Unavailable)?
            .map_err(ReplicaPeerError::InvalidEndpoint)
    }

    #[cfg_attr(not(feature = "standalone-slot-tracker"), allow(dead_code))]
    pub(crate) fn list_peers(&self) -> Vec<String> {
        let mut peers: Vec<_> = self.peers.read().iter().cloned().collect();
        peers.sort();
        peers
    }

    pub(crate) async fn shutdown(&self) {
        self.cancel_token.cancel();
        if let Some(task) = self.subscriber_task.lock().await.take() {
            let _ = task.await;
        }
    }

    fn abort(&self) {
        self.cancel_token.cancel();
        if let Ok(mut task) = self.subscriber_task.try_lock()
            && let Some(task) = task.take()
        {
            task.abort();
        }
    }
}

impl Drop for PeerManager {
    fn drop(&mut self) {
        self.abort();
    }
}

fn handle_peer_command(
    socket: &crate::services::common::zmq::ZmqSocket,
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

fn handle_replica_message<F>(
    handle_event: &F,
    frames: crate::services::common::zmq::MultipartMessage,
) where
    F: Fn(ScopedReplicaEvent),
{
    let [topic, payload] = frames.as_slice() else {
        tracing::debug!(
            frame_count = frames.len(),
            "Dropping malformed active-sequence replica message"
        );
        return;
    };
    if topic.as_slice() != REPLICA_TOPIC {
        tracing::debug!("Dropping active-sequence replica message with unexpected topic");
        return;
    }
    let event: ScopedReplicaEvent = match rmp_serde::from_slice(payload) {
        Ok(event) => event,
        Err(error) => {
            tracing::debug!("Dropping malformed active-sequence replica payload: {error}");
            return;
        }
    };
    handle_event(event);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{ActiveSequenceEventData, WorkerWithDpRank};
    #[cfg(feature = "standalone-slot-tracker")]
    use crate::services::slot_tracker::registry::{SlotTrackerRegistry, TrackerKey};

    fn event() -> ScopedReplicaEvent {
        ScopedReplicaEvent {
            model_name: "model".to_string(),
            routing_group: "group".to_string(),
            block_size: 16,
            event: ActiveSequenceEvent {
                request_id: "request".to_string(),
                worker: WorkerWithDpRank::new(1, 0),
                data: ActiveSequenceEventData::Free,
                router_id: 42,
                stride: None,
                lora_name: None,
            },
        }
    }

    #[test]
    fn replica_sync_port_builds_wildcard_bind_endpoint() {
        assert_eq!(replica_sync_bind_endpoint(8092).unwrap(), "tcp://*:8092");
        assert!(replica_sync_bind_endpoint(0).is_err());
    }

    #[test]
    fn replica_sync_requires_port_for_initial_peers() {
        let error = setup_replica_sync(
            None,
            &["tcp://127.0.0.1:8092".to_string()],
            CancellationToken::new(),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("--replica-sync-peers requires --replica-sync-port")
        );
    }

    #[test]
    fn replica_event_wire_schema_uses_routing_group() {
        let payload = rmp_serde::to_vec_named(&event()).unwrap();
        let value: serde_json::Value = rmp_serde::from_slice(&payload).unwrap();

        assert_eq!(value["routing_group"], "group");
        assert!(value.get("tenant_id").is_none());

        let decoded: ScopedReplicaEvent = rmp_serde::from_slice(&payload).unwrap();
        assert_eq!(decoded.routing_group, "group");
    }

    #[test]
    fn legacy_replica_event_wire_schema_is_rejected() {
        let legacy = serde_json::json!({
            "model_name": "model",
            "tenant_id": "tenant",
            "block_size": 16,
            "event": event().event,
        });
        let payload = rmp_serde::to_vec_named(&legacy).unwrap();

        assert!(rmp_serde::from_slice::<ScopedReplicaEvent>(&payload).is_err());
    }

    #[tokio::test]
    async fn scoped_publisher_drops_when_outbound_channel_is_full() {
        let (tx, mut rx) = mpsc::channel(1);
        let publisher =
            ScopedSequencePublisher::enabled(Arc::from("model"), Arc::from("group"), 16, tx);
        let event = event().event;

        publisher.publish_event(&event).await.unwrap();
        publisher.publish_event(&event).await.unwrap();

        assert_eq!(rx.len(), 1);
        assert_eq!(rx.recv().await.unwrap().event.request_id, "request");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dynamic_peer_registration_controls_delivery() {
        let endpoint = reserve_tcp_endpoint();
        let cancel_token = CancellationToken::new();
        let (outbound, publisher_task) =
            start_replica_publisher(&endpoint, cancel_token.child_token()).expect("publisher");
        let (received_tx, mut received_rx) = mpsc::channel(16);
        let manager = PeerManager::start(Vec::new(), cancel_token.child_token(), move |event| {
            let _ = received_tx.try_send(event);
        })
        .expect("peer manager");

        assert!(manager.register_peer(endpoint.clone()).await.unwrap());

        let mut delivered = false;
        for attempt in 0..40 {
            let mut event = event();
            event.event.request_id = format!("warmup-{attempt}");
            outbound.send(event).await.unwrap();
            if tokio::time::timeout(std::time::Duration::from_millis(50), received_rx.recv())
                .await
                .is_ok()
            {
                delivered = true;
                break;
            }
        }
        assert!(delivered, "dynamically registered peer received no events");

        assert!(manager.deregister_peer(endpoint).await.unwrap());
        while received_rx.try_recv().is_ok() {}

        let mut after_disconnect = event();
        after_disconnect.event.request_id = "after-disconnect".to_string();
        outbound.send(after_disconnect).await.unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(200), received_rx.recv(),)
                .await
                .is_err()
        );

        cancel_token.cancel();
        manager.shutdown().await;
        publisher_task.await.unwrap();
    }

    #[cfg(feature = "standalone-slot-tracker")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn zmq_replica_sync_propagates_request_lifecycle() {
        let endpoint_a = reserve_tcp_endpoint();
        let endpoint_b = reserve_tcp_endpoint();
        let cancel_token = CancellationToken::new();
        let (outbound_a, publisher_a) =
            start_replica_publisher(&endpoint_a, cancel_token.child_token()).unwrap();
        let (outbound_b, publisher_b) =
            start_replica_publisher(&endpoint_b, cancel_token.child_token()).unwrap();
        let registry_a = Arc::new(SlotTrackerRegistry::new_with_replica_sync(
            cancel_token.clone(),
            ReplicaSyncConfig::new(11, outbound_a),
        ));
        let registry_b = Arc::new(SlotTrackerRegistry::new_with_replica_sync(
            cancel_token.clone(),
            ReplicaSyncConfig::new(22, outbound_b),
        ));
        let dispatch_registry_b = Arc::clone(&registry_b);
        let peer_b =
            PeerManager::start(vec![endpoint_a], cancel_token.child_token(), move |event| {
                dispatch_registry_b.dispatch_replica_event(event)
            })
            .unwrap();
        let key = TrackerKey::new("model".to_string(), Some("group".to_string()));
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
        peer_b.shutdown().await;
        publisher_a.await.unwrap();
        publisher_b.await.unwrap();
    }

    #[cfg(feature = "standalone-slot-tracker")]
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
