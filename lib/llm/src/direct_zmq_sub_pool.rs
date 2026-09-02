// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared direct-ZMQ SUB socket grouping for high-fanout KV ingress.

use std::{
    collections::HashMap,
    ffi::{OsStr, OsString},
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use dynamo_runtime::transports::event_plane::{
    Codec, DynamicZmqSubSocket, ValidatedEnvelope, ZmqWireMessage,
};
use tokio::{
    sync::{Mutex, mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

const GROUP_JOIN_TIMEOUT: Duration = Duration::from_secs(5);
const GROUP_COMMAND_CAPACITY: usize = 128;
const PUBLISHER_LANE_CAPACITY: usize = 64;

pub(crate) const ENDPOINTS_PER_SUB_ENV: &str = "DYN_ROUTER_KV_ZMQ_ENDPOINTS_PER_SUB";
pub(crate) const DEFAULT_ENDPOINTS_PER_SUB: usize = 64;

pub(crate) fn endpoints_per_sub_from_env() -> Result<usize> {
    endpoints_per_sub_from_lookup(|key| std::env::var_os(key))
}

fn endpoints_per_sub_from_lookup(
    mut lookup: impl FnMut(&str) -> Option<OsString>,
) -> Result<usize> {
    let Some(raw) = lookup(ENDPOINTS_PER_SUB_ENV) else {
        return Ok(DEFAULT_ENDPOINTS_PER_SUB);
    };
    parse_endpoints_per_sub(&raw)
}

fn parse_endpoints_per_sub(raw: &OsStr) -> Result<usize> {
    let value = raw
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("{ENDPOINTS_PER_SUB_ENV} must be valid UTF-8"))?
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("{ENDPOINTS_PER_SUB_ENV} must be a positive integer"))?;
    anyhow::ensure!(value > 0, "{ENDPOINTS_PER_SUB_ENV} must be positive");
    Ok(value)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectZmqSubPoolEvent {
    SocketGroups(i64),
    ConnectedEndpoints(i64),
    Lifecycle(&'static str),
}

pub(crate) type DirectZmqSubPoolObserver =
    Arc<dyn Fn(DirectZmqSubPoolEvent) + Send + Sync + 'static>;

struct GroupRoute {
    endpoint: String,
    generation: u64,
    sender: mpsc::Sender<ValidatedEnvelope>,
    disconnected: CancellationToken,
}

impl Drop for GroupRoute {
    fn drop(&mut self) {
        self.disconnected.cancel();
    }
}

enum GroupCommand {
    Add {
        publisher_id: u64,
        route: GroupRoute,
        completed: oneshot::Sender<Result<()>>,
    },
    Remove {
        publisher_id: u64,
        generation: u64,
        completed: oneshot::Sender<Result<()>>,
    },
}

struct SocketGroup {
    members: HashMap<u64, GroupMember>,
    command_tx: mpsc::Sender<GroupCommand>,
    cancel: CancellationToken,
    handle: JoinHandle<()>,
}

struct GroupMember {
    generation: u64,
    connected: bool,
}

struct PoolInner {
    groups: HashMap<u64, SocketGroup>,
    next_group_id: u64,
}

#[derive(Clone)]
pub(crate) struct DirectZmqSubPool {
    inner: Arc<Mutex<PoolInner>>,
    topic: Arc<str>,
    endpoints_per_sub: usize,
    rcvhwm: Option<i32>,
    observer: DirectZmqSubPoolObserver,
    cancellation_token: CancellationToken,
}

pub(crate) struct DirectZmqSubRegistration {
    pub(crate) group_id: u64,
    pub(crate) receiver: mpsc::Receiver<ValidatedEnvelope>,
    pub(crate) disconnected: CancellationToken,
}

impl DirectZmqSubPool {
    pub(crate) fn new(
        topic: impl Into<Arc<str>>,
        endpoints_per_sub: usize,
        observer: DirectZmqSubPoolObserver,
        cancellation_token: CancellationToken,
    ) -> Result<Self> {
        Self::new_inner(topic, endpoints_per_sub, None, observer, cancellation_token)
    }

    pub(crate) fn new_with_rcvhwm(
        topic: impl Into<Arc<str>>,
        endpoints_per_sub: usize,
        rcvhwm: i32,
        observer: DirectZmqSubPoolObserver,
        cancellation_token: CancellationToken,
    ) -> Result<Self> {
        anyhow::ensure!(rcvhwm > 0, "ZMQ receive HWM must be greater than zero");
        Self::new_inner(
            topic,
            endpoints_per_sub,
            Some(rcvhwm),
            observer,
            cancellation_token,
        )
    }

    fn new_inner(
        topic: impl Into<Arc<str>>,
        endpoints_per_sub: usize,
        rcvhwm: Option<i32>,
        observer: DirectZmqSubPoolObserver,
        cancellation_token: CancellationToken,
    ) -> Result<Self> {
        anyhow::ensure!(endpoints_per_sub > 0, "endpoints per SUB must be positive");
        Ok(Self {
            inner: Arc::new(Mutex::new(PoolInner {
                groups: HashMap::new(),
                next_group_id: 1,
            })),
            topic: topic.into(),
            endpoints_per_sub,
            rcvhwm,
            observer,
            cancellation_token,
        })
    }

    pub(crate) async fn register(
        &self,
        publisher_id: u64,
        endpoint: &str,
        generation: u64,
    ) -> Result<DirectZmqSubRegistration> {
        let (sender, receiver) = mpsc::channel(PUBLISHER_LANE_CAPACITY);
        let disconnected = CancellationToken::new();
        let mut inner = self.inner.lock().await;
        self.reap_failed_groups(&mut inner);
        anyhow::ensure!(
            inner
                .groups
                .values()
                .all(|group| !group.members.contains_key(&publisher_id)),
            "publisher {publisher_id} is already registered"
        );

        let group_id = inner
            .groups
            .iter()
            .filter(|(_, group)| {
                group.members.len() < self.endpoints_per_sub
                    && !group.command_tx.is_closed()
                    && !group.handle.is_finished()
            })
            .min_by_key(|(group_id, group)| (group.members.len(), **group_id))
            .map(|(group_id, _)| *group_id);

        if let Some(group_id) = group_id {
            let group = inner
                .groups
                .get_mut(&group_id)
                .expect("selected socket group must exist");
            group.members.insert(
                publisher_id,
                GroupMember {
                    generation,
                    connected: false,
                },
            );
            let command_tx = group.command_tx.clone();
            drop(inner);
            let (completed, completion) = oneshot::channel();
            let result = async {
                command_tx
                    .send(GroupCommand::Add {
                        publisher_id,
                        route: GroupRoute {
                            endpoint: endpoint.to_string(),
                            generation,
                            sender,
                            disconnected: disconnected.clone(),
                        },
                        completed,
                    })
                    .await
                    .map_err(|_| anyhow::anyhow!("direct-ZMQ socket group stopped"))?;
                completion
                    .await
                    .map_err(|_| anyhow::anyhow!("direct-ZMQ socket group stopped"))?
            }
            .await;
            if let Err(error) = result {
                self.rollback_registration(group_id, publisher_id, generation)
                    .await;
                return Err(error);
            }
            let marked_connected = {
                let mut inner = self.inner.lock().await;
                self.reap_failed_groups(&mut inner);
                inner
                    .groups
                    .get_mut(&group_id)
                    .and_then(|group| group.members.get_mut(&publisher_id))
                    .filter(|member| member.generation == generation)
                    .is_some_and(|member| {
                        member.connected = true;
                        true
                    })
            };
            anyhow::ensure!(marked_connected, "direct-ZMQ socket group stopped");
            self.observe(DirectZmqSubPoolEvent::ConnectedEndpoints(1));
            return Ok(DirectZmqSubRegistration {
                group_id,
                receiver,
                disconnected,
            });
        }

        let socket = match self.rcvhwm {
            Some(rcvhwm) => {
                DynamicZmqSubSocket::connect_with_rcvhwm(endpoint, &self.topic, rcvhwm)?
            }
            None => DynamicZmqSubSocket::connect(endpoint, &self.topic)?,
        };
        let group_id = inner.next_group_id;
        inner.next_group_id = inner.next_group_id.wrapping_add(1);
        let (command_tx, command_rx) = mpsc::channel(GROUP_COMMAND_CAPACITY);
        let cancel = self.cancellation_token.child_token();
        let mut routes = HashMap::new();
        routes.insert(
            publisher_id,
            GroupRoute {
                endpoint: endpoint.to_string(),
                generation,
                sender,
                disconnected: disconnected.clone(),
            },
        );
        let handle = tokio::spawn(run_socket_group(
            group_id,
            self.topic.clone(),
            socket,
            routes,
            command_rx,
            self.observer.clone(),
            cancel.clone(),
        ));
        inner.groups.insert(
            group_id,
            SocketGroup {
                members: HashMap::from([(
                    publisher_id,
                    GroupMember {
                        generation,
                        connected: true,
                    },
                )]),
                command_tx,
                cancel,
                handle,
            },
        );
        self.observe(DirectZmqSubPoolEvent::SocketGroups(1));
        self.observe(DirectZmqSubPoolEvent::ConnectedEndpoints(1));
        self.observe(DirectZmqSubPoolEvent::Lifecycle("group_started"));
        Ok(DirectZmqSubRegistration {
            group_id,
            receiver,
            disconnected,
        })
    }

    pub(crate) async fn unregister(&self, group_id: u64, publisher_id: u64, generation: u64) {
        let mut group_to_stop = None;
        let command_tx = {
            let mut inner = self.inner.lock().await;
            self.reap_failed_groups(&mut inner);
            let Some(group) = inner.groups.get_mut(&group_id) else {
                return;
            };
            let Some(member) = group.members.get(&publisher_id) else {
                return;
            };
            if member.generation != generation {
                return;
            }
            let connected = member.connected;
            group.members.remove(&publisher_id);
            if connected {
                self.observe(DirectZmqSubPoolEvent::ConnectedEndpoints(-1));
            }
            if group.members.is_empty() {
                group_to_stop = inner.groups.remove(&group_id);
                self.observe(DirectZmqSubPoolEvent::SocketGroups(-1));
                None
            } else {
                Some(group.command_tx.clone())
            }
        };
        if let Some(command_tx) = command_tx {
            let (completed, completion) = oneshot::channel();
            let removed = command_tx
                .send(GroupCommand::Remove {
                    publisher_id,
                    generation,
                    completed,
                })
                .await
                .is_ok()
                && completion.await.is_ok_and(|result| result.is_ok());
            if !removed {
                self.observe(DirectZmqSubPoolEvent::Lifecycle("group_command_error"));
            }
        }
        if let Some(group) = group_to_stop {
            stop_group(group, &self.observer).await;
        }
    }

    async fn rollback_registration(&self, group_id: u64, publisher_id: u64, generation: u64) {
        let mut group_to_stop = None;
        let command_tx = {
            let mut inner = self.inner.lock().await;
            self.reap_failed_groups(&mut inner);
            let Some(group) = inner.groups.get_mut(&group_id) else {
                return;
            };
            let Some(member) = group.members.get(&publisher_id) else {
                return;
            };
            if member.generation != generation {
                return;
            }
            group.members.remove(&publisher_id);
            if group.members.is_empty() {
                group_to_stop = inner.groups.remove(&group_id);
                self.observe(DirectZmqSubPoolEvent::SocketGroups(-1));
                None
            } else {
                Some(group.command_tx.clone())
            }
        };
        if let Some(command_tx) = command_tx {
            let (completed, completion) = oneshot::channel();
            let _ = command_tx
                .send(GroupCommand::Remove {
                    publisher_id,
                    generation,
                    completed,
                })
                .await;
            let _ = completion.await;
        }
        if let Some(group) = group_to_stop {
            stop_group(group, &self.observer).await;
        }
    }

    pub(crate) async fn shutdown(&self) {
        let groups = {
            let mut inner = self.inner.lock().await;
            let groups = inner
                .groups
                .drain()
                .map(|(_, group)| group)
                .collect::<Vec<_>>();
            for group in &groups {
                self.observe(DirectZmqSubPoolEvent::SocketGroups(-1));
                self.observe(DirectZmqSubPoolEvent::ConnectedEndpoints(
                    -(group
                        .members
                        .values()
                        .filter(|member| member.connected)
                        .count() as i64),
                ));
                group.cancel.cancel();
            }
            groups
        };
        futures::future::join_all(
            groups
                .into_iter()
                .map(|group| stop_group(group, &self.observer)),
        )
        .await;
    }

    #[cfg(test)]
    pub(crate) async fn group_count(&self) -> usize {
        self.inner.lock().await.groups.len()
    }

    fn reap_failed_groups(&self, inner: &mut PoolInner) {
        let failed = inner
            .groups
            .iter()
            .filter_map(|(group_id, group)| {
                (group.command_tx.is_closed() || group.handle.is_finished()).then_some(*group_id)
            })
            .collect::<Vec<_>>();
        for group_id in failed {
            if let Some(group) = inner.groups.remove(&group_id) {
                self.observe(DirectZmqSubPoolEvent::SocketGroups(-1));
                self.observe(DirectZmqSubPoolEvent::ConnectedEndpoints(
                    -(group
                        .members
                        .values()
                        .filter(|member| member.connected)
                        .count() as i64),
                ));
                self.observe(DirectZmqSubPoolEvent::Lifecycle("group_reaped"));
                group.cancel.cancel();
            }
        }
    }

    fn observe(&self, event: DirectZmqSubPoolEvent) {
        (self.observer)(event);
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_socket_group(
    group_id: u64,
    topic: Arc<str>,
    mut socket: DynamicZmqSubSocket,
    mut routes: HashMap<u64, GroupRoute>,
    mut command_rx: mpsc::Receiver<GroupCommand>,
    observer: DirectZmqSubPoolObserver,
    cancellation_token: CancellationToken,
) {
    let codec = Codec::default();
    loop {
        tokio::select! {
            biased;
            _ = cancellation_token.cancelled() => return,
            command = command_rx.recv() => {
                let Some(command) = command else {
                    return;
                };
                match command {
                    GroupCommand::Add { publisher_id, route, completed } => {
                        let result = if routes.contains_key(&publisher_id) {
                            Err(anyhow::anyhow!("publisher {publisher_id} is already registered"))
                        } else if routes.values().any(|existing| existing.endpoint == route.endpoint) {
                            Err(anyhow::anyhow!("endpoint {} is already registered", route.endpoint))
                        } else {
                            socket.add_endpoint(&route.endpoint).map(|()| {
                                routes.insert(publisher_id, route);
                            })
                        };
                        let _ = completed.send(result);
                    }
                    GroupCommand::Remove { publisher_id, generation, completed } => {
                        let result = match routes.get(&publisher_id) {
                            Some(route) if route.generation == generation => {
                                let route = routes.remove(&publisher_id).expect("route was present");
                                socket.remove_endpoint(&route.endpoint)
                            }
                            _ => Ok(()),
                        };
                        let _ = completed.send(result);
                    }
                }
            }
            message = socket.next() => {
                let Some(message) = message else {
                    observer(DirectZmqSubPoolEvent::Lifecycle("group_failure"));
                    return;
                };
                let message = match message {
                    Ok(message) => message,
                    Err(error) => {
                        tracing::warn!(%error, group_id, topic = %topic, "Direct-ZMQ socket group receive failed");
                        observer(DirectZmqSubPoolEvent::Lifecycle("group_failure"));
                        return;
                    }
                };
                dispatch_group_message(group_id, &topic, message, &codec, &mut routes, &observer);
            }
        }
    }
}

fn dispatch_group_message(
    group_id: u64,
    topic: &str,
    message: ZmqWireMessage,
    codec: &Codec,
    routes: &mut HashMap<u64, GroupRoute>,
    observer: &DirectZmqSubPoolObserver,
) {
    let envelope = match codec.decode_envelope(&message.payload) {
        Ok(envelope) => envelope,
        Err(error) => {
            tracing::warn!(%error, group_id, topic, publisher_id = message.publisher_id, "Failed to decode direct-ZMQ envelope");
            observer(DirectZmqSubPoolEvent::Lifecycle("envelope_decode_error"));
            return;
        }
    };
    if envelope.publisher_id != message.publisher_id
        || envelope.sequence != message.sequence
        || envelope.topic != topic
    {
        tracing::warn!(
            group_id,
            topic,
            frame_publisher_id = message.publisher_id,
            frame_sequence = message.sequence,
            envelope_publisher_id = envelope.publisher_id,
            envelope_sequence = envelope.sequence,
            envelope_topic = %envelope.topic,
            "Dropping direct-ZMQ envelope with inconsistent attribution"
        );
        observer(DirectZmqSubPoolEvent::Lifecycle("identity_mismatch"));
        return;
    }
    let Some(route) = routes.get(&message.publisher_id) else {
        observer(DirectZmqSubPoolEvent::Lifecycle("unknown_publisher"));
        return;
    };
    let envelope = ValidatedEnvelope {
        publisher_id: envelope.publisher_id,
        sequence: envelope.sequence,
        published_at: envelope.published_at,
        payload: envelope.payload,
    };
    match route.sender.try_send(envelope) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(_)) => {
            observer(DirectZmqSubPoolEvent::Lifecycle("lane_full"));
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            observer(DirectZmqSubPoolEvent::Lifecycle("lane_closed"));
        }
    }
}

async fn stop_group(mut group: SocketGroup, observer: &DirectZmqSubPoolObserver) {
    group.cancel.cancel();
    match tokio::time::timeout(GROUP_JOIN_TIMEOUT, &mut group.handle).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) if error.is_cancelled() => {}
        Ok(Err(error)) => tracing::warn!(%error, "Direct-ZMQ socket group failed during shutdown"),
        Err(_) => {
            group.handle.abort();
            let _ = group.handle.await;
            observer(DirectZmqSubPoolEvent::Lifecycle("group_forced_abort"));
        }
    }
    observer(DirectZmqSubPoolEvent::Lifecycle("group_stopped"));
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    fn config_lookup(value: Option<&str>) -> impl FnMut(&str) -> Option<OsString> {
        let value = value.map(OsString::from);
        move |key| {
            (key == ENDPOINTS_PER_SUB_ENV)
                .then(|| value.clone())
                .flatten()
        }
    }

    fn observer() -> DirectZmqSubPoolObserver {
        Arc::new(|_| {})
    }

    fn wire_message(topic: &str, publisher_id: u64, sequence: u64) -> ZmqWireMessage {
        let payload = Codec::default()
            .encode_envelope_parts(publisher_id, sequence, 1, topic, b"payload")
            .unwrap();
        ZmqWireMessage {
            publisher_id,
            sequence,
            payload,
        }
    }

    #[test]
    fn parses_endpoints_per_sub_configuration() {
        assert_eq!(
            endpoints_per_sub_from_lookup(config_lookup(None)).unwrap(),
            DEFAULT_ENDPOINTS_PER_SUB
        );
        assert_eq!(
            endpoints_per_sub_from_lookup(config_lookup(Some("1"))).unwrap(),
            1
        );
        assert_eq!(
            endpoints_per_sub_from_lookup(config_lookup(Some("128"))).unwrap(),
            128
        );
        for invalid in ["", "0", "-1", "not-a-number"] {
            assert!(endpoints_per_sub_from_lookup(config_lookup(Some(invalid))).is_err());
        }
    }

    #[tokio::test]
    async fn creates_three_groups_for_129_publishers() {
        let pool =
            DirectZmqSubPool::new("kv-events", 64, observer(), CancellationToken::new()).unwrap();
        let registrations = futures::future::join_all((1..=129).map(|publisher_id| {
            let pool = pool.clone();
            async move {
                let endpoint = format!("tcp://127.0.0.1:{}", 30_000 + publisher_id);
                pool.register(publisher_id, &endpoint, publisher_id)
                    .await
                    .unwrap()
            }
        }))
        .await;

        let mut sizes = pool
            .inner
            .lock()
            .await
            .groups
            .values()
            .map(|group| group.members.len())
            .collect::<Vec<_>>();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 64, 64]);
        assert_eq!(pool.group_count().await, 3);

        drop(registrations);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn full_publisher_lane_does_not_block_a_sibling() {
        let (full_tx, mut full_rx) = mpsc::channel(1);
        full_tx
            .try_send(ValidatedEnvelope {
                publisher_id: 1,
                sequence: 0,
                published_at: 0,
                payload: Bytes::new(),
            })
            .unwrap();
        let (sibling_tx, mut sibling_rx) = mpsc::channel(2);
        let mut routes = HashMap::from([
            (
                1,
                GroupRoute {
                    endpoint: "tcp://127.0.0.1:1".to_string(),
                    generation: 1,
                    sender: full_tx,
                    disconnected: CancellationToken::new(),
                },
            ),
            (
                2,
                GroupRoute {
                    endpoint: "tcp://127.0.0.1:2".to_string(),
                    generation: 1,
                    sender: sibling_tx,
                    disconnected: CancellationToken::new(),
                },
            ),
        ]);
        let codec = Codec::default();
        let observer = observer();

        dispatch_group_message(
            1,
            "kv_metrics",
            wire_message("kv_metrics", 1, 1),
            &codec,
            &mut routes,
            &observer,
        );
        dispatch_group_message(
            1,
            "kv_metrics",
            wire_message("kv_metrics", 2, 1),
            &codec,
            &mut routes,
            &observer,
        );
        dispatch_group_message(
            1,
            "kv_metrics",
            wire_message("kv_metrics", 2, 2),
            &codec,
            &mut routes,
            &observer,
        );

        assert_eq!(full_rx.recv().await.unwrap().sequence, 0);
        assert_eq!(sibling_rx.recv().await.unwrap().sequence, 1);
        assert_eq!(sibling_rx.recv().await.unwrap().sequence, 2);
    }

    #[tokio::test]
    async fn validates_topic_and_identity_before_dispatch() {
        let (sender, mut receiver) = mpsc::channel(2);
        let mut routes = HashMap::from([(
            1,
            GroupRoute {
                endpoint: "tcp://127.0.0.1:1".to_string(),
                generation: 1,
                sender,
                disconnected: CancellationToken::new(),
            },
        )]);
        let codec = Codec::default();
        let observer = observer();

        dispatch_group_message(
            1,
            "kv_metrics",
            wire_message("kv-events", 1, 1),
            &codec,
            &mut routes,
            &observer,
        );
        let mut wrong_publisher = wire_message("kv_metrics", 1, 2);
        wrong_publisher.publisher_id = 2;
        dispatch_group_message(
            1,
            "kv_metrics",
            wrong_publisher,
            &codec,
            &mut routes,
            &observer,
        );
        dispatch_group_message(
            1,
            "kv_metrics",
            wire_message("kv_metrics", 1, 3),
            &codec,
            &mut routes,
            &observer,
        );

        assert_eq!(receiver.recv().await.unwrap().sequence, 3);
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn failed_group_is_replaced_without_affecting_other_groups() {
        let pool =
            DirectZmqSubPool::new("kv_metrics", 1, observer(), CancellationToken::new()).unwrap();
        let first = pool.register(1, "tcp://127.0.0.1:31001", 1).await.unwrap();
        let unaffected = pool.register(2, "tcp://127.0.0.1:31002", 2).await.unwrap();
        {
            let inner = pool.inner.lock().await;
            inner.groups.get(&first.group_id).unwrap().handle.abort();
        }
        tokio::time::timeout(Duration::from_secs(1), first.disconnected.cancelled())
            .await
            .expect("failed group must disconnect its publishers");
        assert!(!unaffected.disconnected.is_cancelled());

        let replacement = pool.register(3, "tcp://127.0.0.1:31003", 3).await.unwrap();
        assert_ne!(first.group_id, replacement.group_id);
        let inner = pool.inner.lock().await;
        assert!(!inner.groups.contains_key(&first.group_id));
        assert!(inner.groups.contains_key(&unaffected.group_id));
        assert!(inner.groups.contains_key(&replacement.group_id));
        drop(inner);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn removing_last_publisher_stops_group() {
        let pool =
            DirectZmqSubPool::new("kv-events", 64, observer(), CancellationToken::new()).unwrap();
        let registration = pool.register(1, "tcp://127.0.0.1:31001", 1).await.unwrap();

        pool.unregister(registration.group_id, 1, 1).await;

        assert!(registration.disconnected.is_cancelled());
        assert_eq!(pool.group_count().await, 0);
        pool.shutdown().await;
    }
}
