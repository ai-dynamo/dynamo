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
use parking_lot::Mutex;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

const GROUP_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) const ENDPOINTS_PER_SUB_ENV: &str = "DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB";
pub(crate) const DEFAULT_ENDPOINTS_PER_SUB: usize = 64;
pub(crate) const KV_ZMQ_RCVHWM: i32 = 100_000;

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

#[derive(Debug)]
pub(crate) enum DirectZmqSubItem {
    Envelope(ValidatedEnvelope),
    EnvelopeDecodeError,
    IdentityMismatch,
}

struct GroupRoute {
    endpoint: String,
    generation: u64,
    sender: mpsc::Sender<DirectZmqSubItem>,
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
    },
    #[cfg(test)]
    Pause {
        started: oneshot::Sender<()>,
        release: oneshot::Receiver<()>,
    },
    #[cfg(test)]
    Dispatch {
        message: ZmqWireMessage,
        completed: oneshot::Sender<DispatchOutcome>,
    },
}

struct SocketGroup {
    assignments: HashMap<u64, u64>,
    command_tx: mpsc::UnboundedSender<GroupCommand>,
    cancel: CancellationToken,
    handle: JoinHandle<()>,
}

struct PoolInner {
    groups: HashMap<u64, SocketGroup>,
    next_group_id: u64,
    closed: bool,
}

#[derive(Clone)]
pub(crate) struct DirectZmqSubPool {
    inner: Arc<Mutex<PoolInner>>,
    topic: Arc<str>,
    endpoints_per_sub: usize,
    rcvhwm: i32,
    cancellation_token: CancellationToken,
}

pub(crate) struct DirectZmqSubRegistration {
    pub(crate) group_id: u64,
    // The receiver must close before the lease queues endpoint removal. This
    // also wakes a socket group that is waiting for publisher-lane capacity.
    pub(crate) receiver: mpsc::Receiver<DirectZmqSubItem>,
    pub(crate) disconnected: CancellationToken,
    lease: RegistrationLease,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DispatchOutcome {
    Delivered,
    RouteClosed { publisher_id: u64, generation: u64 },
    GroupCancelled,
}

struct RegistrationLease {
    pool: DirectZmqSubPool,
    group_id: u64,
    publisher_id: u64,
    generation: u64,
    armed: bool,
}

impl RegistrationLease {
    fn new(pool: DirectZmqSubPool, group_id: u64, publisher_id: u64, generation: u64) -> Self {
        Self {
            pool,
            group_id,
            publisher_id,
            generation,
            armed: true,
        }
    }

    fn remove(mut self) -> Option<SocketGroup> {
        self.armed = false;
        self.pool
            .remove_registration(self.group_id, self.publisher_id, self.generation)
    }
}

impl Drop for RegistrationLease {
    fn drop(&mut self) {
        if self.armed
            && let Some(group) =
                self.pool
                    .remove_registration(self.group_id, self.publisher_id, self.generation)
        {
            tokio::spawn(stop_group(group));
        }
    }
}

impl DirectZmqSubPool {
    pub(crate) fn new(
        topic: impl Into<Arc<str>>,
        endpoints_per_sub: usize,
        rcvhwm: i32,
        cancellation_token: CancellationToken,
    ) -> Result<Self> {
        anyhow::ensure!(endpoints_per_sub > 0, "endpoints per SUB must be positive");
        anyhow::ensure!(rcvhwm > 0, "ZMQ receive HWM must be greater than zero");
        Ok(Self {
            inner: Arc::new(Mutex::new(PoolInner {
                groups: HashMap::new(),
                next_group_id: 1,
                closed: false,
            })),
            topic: topic.into(),
            endpoints_per_sub,
            rcvhwm,
            cancellation_token,
        })
    }

    pub(crate) async fn register(
        &self,
        publisher_id: u64,
        endpoint: &str,
        generation: u64,
    ) -> Result<DirectZmqSubRegistration> {
        enum RegistrationStart {
            Pending {
                registration: DirectZmqSubRegistration,
                completion: oneshot::Receiver<Result<()>>,
            },
            Ready(DirectZmqSubRegistration),
        }

        let (sender, receiver) = mpsc::channel(self.rcvhwm as usize);
        let disconnected = CancellationToken::new();
        let start = {
            let mut inner = self.inner.lock();
            Self::reap_failed_groups(&mut inner);
            anyhow::ensure!(!inner.closed, "direct-ZMQ socket pool is closed");
            anyhow::ensure!(
                inner
                    .groups
                    .values()
                    .all(|group| !group.assignments.contains_key(&publisher_id)),
                "publisher {publisher_id} is already registered"
            );

            let group_id = inner
                .groups
                .iter()
                .filter(|(_, group)| {
                    group.assignments.len() < self.endpoints_per_sub
                        && !group.command_tx.is_closed()
                        && !group.handle.is_finished()
                })
                .min_by_key(|(group_id, group)| (group.assignments.len(), **group_id))
                .map(|(group_id, _)| *group_id);

            if let Some(group_id) = group_id {
                let group = inner
                    .groups
                    .get_mut(&group_id)
                    .expect("selected socket group must exist");
                group.assignments.insert(publisher_id, generation);
                let (completed, completion) = oneshot::channel();
                let command = GroupCommand::Add {
                    publisher_id,
                    route: GroupRoute {
                        endpoint: endpoint.to_string(),
                        generation,
                        sender,
                        disconnected: disconnected.clone(),
                    },
                    completed,
                };
                if group.command_tx.send(command).is_err() {
                    group.assignments.remove(&publisher_id);
                    anyhow::bail!("direct-ZMQ socket group stopped");
                }
                RegistrationStart::Pending {
                    registration: DirectZmqSubRegistration {
                        group_id,
                        receiver,
                        disconnected,
                        lease: RegistrationLease::new(
                            self.clone(),
                            group_id,
                            publisher_id,
                            generation,
                        ),
                    },
                    completion,
                }
            } else {
                // libzmq connects asynchronously. Keep group creation serialized so
                // concurrent registrations cannot create excess transient sockets.
                let socket =
                    DynamicZmqSubSocket::connect_with_rcvhwm(endpoint, &self.topic, self.rcvhwm)?;
                let group_id = inner.next_group_id;
                inner.next_group_id = inner.next_group_id.wrapping_add(1);
                let (command_tx, command_rx) = mpsc::unbounded_channel();
                let cancel = self.cancellation_token.child_token();
                let routes = HashMap::from([(
                    publisher_id,
                    GroupRoute {
                        endpoint: endpoint.to_string(),
                        generation,
                        sender,
                        disconnected: disconnected.clone(),
                    },
                )]);
                let handle = tokio::spawn(run_socket_group(
                    group_id,
                    self.topic.clone(),
                    socket,
                    routes,
                    command_rx,
                    cancel.clone(),
                ));
                inner.groups.insert(
                    group_id,
                    SocketGroup {
                        assignments: HashMap::from([(publisher_id, generation)]),
                        command_tx,
                        cancel,
                        handle,
                    },
                );
                RegistrationStart::Ready(DirectZmqSubRegistration {
                    group_id,
                    receiver,
                    disconnected,
                    lease: RegistrationLease::new(self.clone(), group_id, publisher_id, generation),
                })
            }
        };

        let (registration, completion) = match start {
            RegistrationStart::Pending {
                registration,
                completion,
            } => (registration, completion),
            RegistrationStart::Ready(registration) => return Ok(registration),
        };

        completion
            .await
            .map_err(|_| anyhow::anyhow!("direct-ZMQ socket group stopped"))??;
        let valid = {
            let mut inner = self.inner.lock();
            Self::reap_failed_groups(&mut inner);
            !inner.closed
                && inner
                    .groups
                    .get(&registration.group_id)
                    .is_some_and(|group| group.assignments.get(&publisher_id) == Some(&generation))
        };
        anyhow::ensure!(valid, "direct-ZMQ socket group stopped");
        Ok(registration)
    }

    pub(crate) async fn unregister(
        &self,
        registration: DirectZmqSubRegistration,
        publisher_id: u64,
        generation: u64,
    ) {
        let DirectZmqSubRegistration {
            receiver,
            disconnected: _,
            lease,
            group_id: _,
        } = registration;
        // Wake a group task that may be waiting to send to this publisher before
        // queuing the ordered socket removal below.
        drop(receiver);

        debug_assert_eq!(lease.publisher_id, publisher_id);
        debug_assert_eq!(lease.generation, generation);
        let group_to_stop = lease.remove();

        if let Some(group) = group_to_stop {
            stop_group(group).await;
        }
    }

    fn remove_registration(
        &self,
        group_id: u64,
        publisher_id: u64,
        generation: u64,
    ) -> Option<SocketGroup> {
        let mut inner = self.inner.lock();
        Self::reap_failed_groups(&mut inner);
        let group = inner.groups.get_mut(&group_id)?;
        if group.assignments.get(&publisher_id) != Some(&generation) {
            return None;
        }
        group.assignments.remove(&publisher_id);
        if group.assignments.is_empty() {
            let group = inner
                .groups
                .remove(&group_id)
                .expect("empty socket group must exist");
            group.cancel.cancel();
            return Some(group);
        }

        let _ = group.command_tx.send(GroupCommand::Remove {
            publisher_id,
            generation,
        });
        None
    }

    pub(crate) async fn shutdown(&self) {
        let groups = {
            let mut inner = self.inner.lock();
            inner.closed = true;
            let groups = inner
                .groups
                .drain()
                .map(|(_, group)| group)
                .collect::<Vec<_>>();
            for group in &groups {
                group.cancel.cancel();
            }
            groups
        };
        futures::future::join_all(groups.into_iter().map(stop_group)).await;
    }

    #[cfg(test)]
    pub(crate) fn group_count(&self) -> usize {
        self.inner.lock().groups.len()
    }

    fn reap_failed_groups(inner: &mut PoolInner) {
        let failed = inner
            .groups
            .iter()
            .filter_map(|(group_id, group)| {
                (group.command_tx.is_closed() || group.handle.is_finished()).then_some(*group_id)
            })
            .collect::<Vec<_>>();
        for group_id in failed {
            if let Some(group) = inner.groups.remove(&group_id) {
                group.cancel.cancel();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_socket_group(
    group_id: u64,
    topic: Arc<str>,
    mut socket: DynamicZmqSubSocket,
    mut routes: HashMap<u64, GroupRoute>,
    mut command_rx: mpsc::UnboundedReceiver<GroupCommand>,
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
                    GroupCommand::Remove { publisher_id, generation } => {
                        if let Err(error) = remove_route(&mut socket, &mut routes, publisher_id, generation) {
                            tracing::warn!(
                                %error,
                                group_id,
                                topic = %topic,
                                publisher_id,
                                generation,
                                "Failed to remove direct-ZMQ publisher endpoint"
                            );
                        }
                    }
                    #[cfg(test)]
                    GroupCommand::Pause { started, release } => {
                        let _ = started.send(());
                        tokio::select! {
                            _ = cancellation_token.cancelled() => return,
                            _ = release => {}
                        }
                    }
                    #[cfg(test)]
                    GroupCommand::Dispatch { message, completed } => {
                        let outcome = dispatch_group_message(
                            group_id,
                            &topic,
                            message,
                            &codec,
                            &routes,
                            &cancellation_token,
                        )
                        .await;
                        let keep_running = handle_dispatch_outcome(
                            outcome,
                            group_id,
                            &topic,
                            &mut socket,
                            &mut routes,
                        );
                        let _ = completed.send(outcome);
                        if !keep_running {
                            return;
                        }
                    }
                }
            }
            message = socket.next() => {
                let Some(message) = message else {
                    tracing::warn!(group_id, topic = %topic, "Direct-ZMQ socket group stopped");
                    return;
                };
                let message = match message {
                    Ok(message) => message,
                    Err(error) => {
                        tracing::warn!(%error, group_id, topic = %topic, "Direct-ZMQ socket group receive failed");
                        return;
                    }
                };
                let outcome = dispatch_group_message(
                    group_id,
                    &topic,
                    message,
                    &codec,
                    &routes,
                    &cancellation_token,
                )
                .await;
                if !handle_dispatch_outcome(
                    outcome,
                    group_id,
                    &topic,
                    &mut socket,
                    &mut routes,
                ) {
                    return;
                }
                tokio::task::consume_budget().await;
            }
        }
    }
}

fn handle_dispatch_outcome(
    outcome: DispatchOutcome,
    group_id: u64,
    topic: &str,
    socket: &mut DynamicZmqSubSocket,
    routes: &mut HashMap<u64, GroupRoute>,
) -> bool {
    let DispatchOutcome::RouteClosed {
        publisher_id,
        generation,
    } = outcome
    else {
        return outcome == DispatchOutcome::Delivered;
    };

    if let Err(error) = remove_route(socket, routes, publisher_id, generation) {
        tracing::warn!(
            %error,
            group_id,
            topic,
            publisher_id,
            generation,
            "Failed to disconnect closed direct-ZMQ publisher lane"
        );
    }
    true
}

fn remove_route(
    socket: &mut DynamicZmqSubSocket,
    routes: &mut HashMap<u64, GroupRoute>,
    publisher_id: u64,
    generation: u64,
) -> Result<()> {
    match routes.get(&publisher_id) {
        Some(route) if route.generation == generation => {
            let route = routes.remove(&publisher_id).expect("route was present");
            socket.remove_endpoint(&route.endpoint)
        }
        _ => Ok(()),
    }
}

async fn dispatch_group_message(
    group_id: u64,
    topic: &str,
    message: ZmqWireMessage,
    codec: &Codec,
    routes: &HashMap<u64, GroupRoute>,
    cancellation_token: &CancellationToken,
) -> DispatchOutcome {
    let Some(route) = routes.get(&message.publisher_id) else {
        tracing::warn!(
            group_id,
            topic,
            publisher_id = message.publisher_id,
            "Dropping direct-ZMQ envelope from an unknown publisher"
        );
        return DispatchOutcome::Delivered;
    };
    let item = match codec.decode_envelope(&message.payload) {
        Ok(envelope)
            if envelope.publisher_id == message.publisher_id
                && envelope.sequence == message.sequence
                && envelope.topic == topic =>
        {
            DirectZmqSubItem::Envelope(ValidatedEnvelope {
                publisher_id: envelope.publisher_id,
                sequence: envelope.sequence,
                published_at: envelope.published_at,
                payload: envelope.payload,
            })
        }
        Ok(envelope) => {
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
            DirectZmqSubItem::IdentityMismatch
        }
        Err(error) => {
            tracing::warn!(
                %error,
                group_id,
                topic,
                publisher_id = message.publisher_id,
                "Failed to decode direct-ZMQ envelope"
            );
            DirectZmqSubItem::EnvelopeDecodeError
        }
    };

    match route.sender.try_send(item) {
        Ok(()) => DispatchOutcome::Delivered,
        Err(mpsc::error::TrySendError::Full(item)) => {
            let result = tokio::select! {
                biased;
                _ = cancellation_token.cancelled() => return DispatchOutcome::GroupCancelled,
                result = route.sender.send(item) => result,
            };
            match result {
                Ok(()) => DispatchOutcome::Delivered,
                Err(_) => DispatchOutcome::RouteClosed {
                    publisher_id: message.publisher_id,
                    generation: route.generation,
                },
            }
        }
        Err(mpsc::error::TrySendError::Closed(_)) => DispatchOutcome::RouteClosed {
            publisher_id: message.publisher_id,
            generation: route.generation,
        },
    }
}

async fn stop_group(mut group: SocketGroup) {
    group.cancel.cancel();
    match tokio::time::timeout(GROUP_JOIN_TIMEOUT, &mut group.handle).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) if error.is_cancelled() => {}
        Ok(Err(error)) => tracing::warn!(%error, "Direct-ZMQ socket group failed during shutdown"),
        Err(_) => {
            group.handle.abort();
            let _ = group.handle.await;
            tracing::warn!("Direct-ZMQ socket group was aborted during shutdown");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_lookup(value: Option<&str>) -> impl FnMut(&str) -> Option<OsString> {
        let value = value.map(OsString::from);
        move |key| {
            (key == ENDPOINTS_PER_SUB_ENV)
                .then(|| value.clone())
                .flatten()
        }
    }

    fn pool(topic: &str, endpoints_per_sub: usize, rcvhwm: i32) -> DirectZmqSubPool {
        DirectZmqSubPool::new(topic, endpoints_per_sub, rcvhwm, CancellationToken::new()).unwrap()
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

    fn sequence(item: DirectZmqSubItem) -> u64 {
        match item {
            DirectZmqSubItem::Envelope(envelope) => envelope.sequence,
            other => panic!("expected envelope, got {other:?}"),
        }
    }

    fn dispatch_on_group(
        pool: &DirectZmqSubPool,
        group_id: u64,
        message: ZmqWireMessage,
    ) -> oneshot::Receiver<DispatchOutcome> {
        let (completed, completion) = oneshot::channel();
        pool.inner
            .lock()
            .groups
            .get(&group_id)
            .expect("socket group must exist")
            .command_tx
            .send(GroupCommand::Dispatch { message, completed })
            .expect("socket group must be running");
        completion
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
        let pool = pool("kv-events", 64, 128);
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
            .groups
            .values()
            .map(|group| group.assignments.len())
            .collect::<Vec<_>>();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 64, 64]);
        assert_eq!(pool.group_count(), 3);
        assert!(
            registrations
                .iter()
                .all(|registration| registration.receiver.max_capacity() == 128)
        );

        drop(registrations);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn full_lane_backpressures_and_preserves_order() {
        let pool = pool("kv_metrics", 64, 1);
        let mut source = pool.register(1, "tcp://127.0.0.1:31001", 1).await.unwrap();
        let mut sibling = pool.register(2, "tcp://127.0.0.1:31002", 1).await.unwrap();
        assert_eq!(source.group_id, sibling.group_id);

        assert_eq!(
            dispatch_on_group(&pool, source.group_id, wire_message("kv_metrics", 1, 0),)
                .await
                .unwrap(),
            DispatchOutcome::Delivered
        );
        let mut blocked =
            dispatch_on_group(&pool, source.group_id, wire_message("kv_metrics", 1, 1));
        tokio::task::yield_now().await;
        assert!(matches!(
            blocked.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        let mut sibling_blocked =
            dispatch_on_group(&pool, source.group_id, wire_message("kv_metrics", 2, 0));
        tokio::task::yield_now().await;
        assert!(matches!(
            sibling_blocked.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));

        assert_eq!(sequence(source.receiver.recv().await.unwrap()), 0);
        assert_eq!(blocked.await.unwrap(), DispatchOutcome::Delivered);
        assert_eq!(sequence(source.receiver.recv().await.unwrap()), 1);
        assert_eq!(sibling_blocked.await.unwrap(), DispatchOutcome::Delivered);
        assert_eq!(sequence(sibling.receiver.recv().await.unwrap()), 0);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn remove_then_add_stays_ordered_while_a_sibling_lane_is_blocked() {
        let pool = pool("kv_metrics", 64, 1);
        let mut blocked_source = pool.register(1, "tcp://127.0.0.1:31001", 1).await.unwrap();
        let old = pool.register(2, "tcp://127.0.0.1:31002", 1).await.unwrap();
        let group_id = blocked_source.group_id;

        assert_eq!(
            dispatch_on_group(&pool, group_id, wire_message("kv_metrics", 1, 0))
                .await
                .unwrap(),
            DispatchOutcome::Delivered
        );
        let blocked = dispatch_on_group(&pool, group_id, wire_message("kv_metrics", 1, 1));
        tokio::task::yield_now().await;

        drop(old);
        let replacement_pool = pool.clone();
        let replacement = tokio::spawn(async move {
            replacement_pool
                .register(2, "tcp://127.0.0.1:31002", 2)
                .await
        });
        tokio::task::yield_now().await;
        assert!(!replacement.is_finished());

        assert_eq!(sequence(blocked_source.receiver.recv().await.unwrap()), 0);
        assert_eq!(blocked.await.unwrap(), DispatchOutcome::Delivered);
        let mut replacement = replacement.await.unwrap().unwrap();
        assert_eq!(replacement.group_id, group_id);
        assert_eq!(
            dispatch_on_group(&pool, group_id, wire_message("kv_metrics", 2, 0))
                .await
                .unwrap(),
            DispatchOutcome::Delivered
        );
        assert_eq!(sequence(replacement.receiver.recv().await.unwrap()), 0);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn unregister_unblocks_a_full_lane_and_keeps_the_group_alive() {
        let pool = pool("kv_metrics", 64, 1);
        let source = pool.register(1, "tcp://127.0.0.1:31001", 1).await.unwrap();
        let mut sibling = pool.register(2, "tcp://127.0.0.1:31002", 1).await.unwrap();
        let disconnected = source.disconnected.clone();
        assert_eq!(source.group_id, sibling.group_id);

        assert_eq!(
            dispatch_on_group(&pool, source.group_id, wire_message("kv_metrics", 1, 0),)
                .await
                .unwrap(),
            DispatchOutcome::Delivered
        );
        let blocked = dispatch_on_group(&pool, source.group_id, wire_message("kv_metrics", 1, 1));
        tokio::task::yield_now().await;

        tokio::time::timeout(Duration::from_secs(1), pool.unregister(source, 1, 1))
            .await
            .expect("unregister must interrupt a blocked lane send");
        assert_eq!(
            blocked.await.unwrap(),
            DispatchOutcome::RouteClosed {
                publisher_id: 1,
                generation: 1,
            }
        );
        assert!(disconnected.is_cancelled());

        assert_eq!(
            dispatch_on_group(&pool, sibling.group_id, wire_message("kv_metrics", 2, 0),)
                .await
                .unwrap(),
            DispatchOutcome::Delivered
        );
        assert_eq!(sequence(sibling.receiver.recv().await.unwrap()), 0);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_interrupts_a_blocked_lane_send() {
        let pool = pool("kv_metrics", 64, 1);
        let source = pool.register(1, "tcp://127.0.0.1:31001", 1).await.unwrap();
        let disconnected = source.disconnected.clone();

        assert_eq!(
            dispatch_on_group(&pool, source.group_id, wire_message("kv_metrics", 1, 0),)
                .await
                .unwrap(),
            DispatchOutcome::Delivered
        );
        let blocked = dispatch_on_group(&pool, source.group_id, wire_message("kv_metrics", 1, 1));
        tokio::task::yield_now().await;

        tokio::time::timeout(Duration::from_secs(1), pool.shutdown())
            .await
            .expect("shutdown must interrupt a blocked lane send");
        assert_eq!(blocked.await.unwrap(), DispatchOutcome::GroupCancelled);
        assert!(disconnected.is_cancelled());
        assert_eq!(pool.group_count(), 0);
        assert!(pool.register(2, "tcp://127.0.0.1:31002", 1).await.is_err());
    }

    #[tokio::test]
    async fn reports_validation_errors_to_the_affected_publisher() {
        let (sender, mut receiver) = mpsc::channel(3);
        let routes = HashMap::from([(
            1,
            GroupRoute {
                endpoint: "tcp://127.0.0.1:1".to_string(),
                generation: 1,
                sender,
                disconnected: CancellationToken::new(),
            },
        )]);
        let codec = Codec::default();

        assert_eq!(
            dispatch_group_message(
                1,
                "kv_metrics",
                wire_message("kv-events", 1, 1),
                &codec,
                &routes,
                &CancellationToken::new(),
            )
            .await,
            DispatchOutcome::Delivered
        );
        assert_eq!(
            dispatch_group_message(
                1,
                "kv_metrics",
                wire_message("kv_metrics", 1, 2),
                &codec,
                &routes,
                &CancellationToken::new(),
            )
            .await,
            DispatchOutcome::Delivered
        );

        assert!(matches!(
            receiver.recv().await.unwrap(),
            DirectZmqSubItem::IdentityMismatch
        ));
        assert_eq!(sequence(receiver.recv().await.unwrap()), 2);
    }

    #[tokio::test]
    async fn failed_group_is_replaced_without_affecting_other_groups() {
        let pool = pool("kv_metrics", 2, 128);
        let first = pool.register(1, "tcp://127.0.0.1:31001", 1).await.unwrap();
        let second = pool.register(2, "tcp://127.0.0.1:31002", 1).await.unwrap();
        let mut unaffected = pool.register(3, "tcp://127.0.0.1:31003", 1).await.unwrap();
        assert_eq!(first.group_id, second.group_id);
        assert_ne!(first.group_id, unaffected.group_id);
        {
            let inner = pool.inner.lock();
            inner.groups.get(&first.group_id).unwrap().handle.abort();
        }
        tokio::time::timeout(Duration::from_secs(1), first.disconnected.cancelled())
            .await
            .expect("failed group must disconnect its publishers");
        tokio::time::timeout(Duration::from_secs(1), second.disconnected.cancelled())
            .await
            .expect("failed group must disconnect all its publishers");
        assert!(!unaffected.disconnected.is_cancelled());
        assert_eq!(
            dispatch_on_group(&pool, unaffected.group_id, wire_message("kv_metrics", 3, 0),)
                .await
                .unwrap(),
            DispatchOutcome::Delivered
        );
        assert_eq!(sequence(unaffected.receiver.recv().await.unwrap()), 0);

        let replacement = pool.register(1, "tcp://127.0.0.1:31001", 2).await.unwrap();
        assert_ne!(first.group_id, replacement.group_id);
        {
            let inner = pool.inner.lock();
            assert!(!inner.groups.contains_key(&first.group_id));
            assert!(inner.groups.contains_key(&unaffected.group_id));
            assert!(inner.groups.contains_key(&replacement.group_id));
        }
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn removing_last_publisher_stops_group_and_shutdown_closes_pool() {
        let pool = pool("kv-events", 64, 128);
        let registration = pool.register(1, "tcp://127.0.0.1:31001", 1).await.unwrap();
        let disconnected = registration.disconnected.clone();

        pool.unregister(registration, 1, 1).await;
        assert!(disconnected.is_cancelled());
        assert_eq!(pool.group_count(), 0);

        pool.shutdown().await;
        assert!(pool.register(2, "tcp://127.0.0.1:31002", 2).await.is_err());
    }

    #[tokio::test]
    async fn dropping_an_active_registration_removes_its_assignment() {
        let pool = pool("kv-events", 64, 128);
        let registration = pool.register(1, "tcp://127.0.0.1:31001", 1).await.unwrap();
        let disconnected = registration.disconnected.clone();

        drop(registration);

        assert_eq!(pool.group_count(), 0);
        tokio::time::timeout(Duration::from_secs(1), disconnected.cancelled())
            .await
            .expect("registration drop must stop its final socket group");
        let replacement = pool.register(1, "tcp://127.0.0.1:31001", 2).await.unwrap();
        assert_eq!(pool.group_count(), 1);
        drop(replacement);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn cancelled_registration_rolls_back_and_can_register_again() {
        let pool = pool("kv-events", 64, 128);
        let first = pool.register(1, "tcp://127.0.0.1:31001", 1).await.unwrap();
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        pool.inner
            .lock()
            .groups
            .get(&first.group_id)
            .unwrap()
            .command_tx
            .send(GroupCommand::Pause {
                started: started_tx,
                release: release_rx,
            })
            .unwrap();
        started_rx.await.unwrap();

        let pending_pool = pool.clone();
        let pending =
            tokio::spawn(async move { pending_pool.register(2, "tcp://127.0.0.1:31002", 1).await });
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if pool
                    .inner
                    .lock()
                    .groups
                    .get(&first.group_id)
                    .unwrap()
                    .assignments
                    .contains_key(&2)
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("registration should reserve a group slot");
        pending.abort();
        match pending.await {
            Err(error) => assert!(error.is_cancelled()),
            Ok(_) => panic!("registration task should be cancelled"),
        }
        assert!(
            !pool
                .inner
                .lock()
                .groups
                .get(&first.group_id)
                .unwrap()
                .assignments
                .contains_key(&2)
        );

        release_tx.send(()).unwrap();
        let replacement = pool.register(2, "tcp://127.0.0.1:31002", 2).await.unwrap();
        assert_eq!(replacement.group_id, first.group_id);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_rejects_an_inflight_registration() {
        let pool = pool("kv-events", 64, 128);
        let first = pool.register(1, "tcp://127.0.0.1:31001", 1).await.unwrap();
        let (started_tx, started_rx) = oneshot::channel();
        let (_release_tx, release_rx) = oneshot::channel();
        pool.inner
            .lock()
            .groups
            .get(&first.group_id)
            .unwrap()
            .command_tx
            .send(GroupCommand::Pause {
                started: started_tx,
                release: release_rx,
            })
            .unwrap();
        started_rx.await.unwrap();

        let pending_pool = pool.clone();
        let pending =
            tokio::spawn(async move { pending_pool.register(2, "tcp://127.0.0.1:31002", 1).await });
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if pool
                    .inner
                    .lock()
                    .groups
                    .get(&first.group_id)
                    .unwrap()
                    .assignments
                    .contains_key(&2)
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("registration should reserve a group slot");

        pool.shutdown().await;
        match pending.await {
            Ok(Err(_)) => {}
            Ok(Ok(_)) => panic!("in-flight registration must fail during shutdown"),
            Err(error) => panic!("registration task failed: {error}"),
        }
        assert_eq!(pool.group_count(), 0);
        assert!(first.disconnected.is_cancelled());
    }
}
