// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared direct-ZMQ SUB socket grouping for high-fanout KV ingress.

use std::{
    collections::{HashMap, HashSet},
    ffi::{OsStr, OsString},
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use dynamo_runtime::transports::event_plane::{
    Codec, DynamicZmqSubSocket, ValidatedEnvelope, ValidatedZmqSource, ZmqWireMessage,
};
use parking_lot::Mutex;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

use crate::utils::zmq::multipart_message;

const GROUP_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) const ENDPOINTS_PER_SUB_ENV: &str = "DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB";
pub(crate) const DEFAULT_ENDPOINTS_PER_SUB: usize = 1;
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

/// Undecoded multipart frames from a pool that routes by raw topic frame.
pub(crate) type RawZmqFrames = Vec<Vec<u8>>;

enum RouteSender {
    Envelope(mpsc::Sender<DirectZmqSubItem>),
    Raw(mpsc::Sender<RawZmqFrames>),
}

struct GroupRoute {
    endpoint: String,
    generation: u64,
    sender: RouteSender,
    /// Raw-topic pools subscribe per route and route by this exact topic frame.
    topic: Option<Vec<u8>>,
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
    /// Routes per connected endpoint. Publishers that share one PUB endpoint share
    /// one connection, so they must land in the same group.
    endpoints: HashMap<String, usize>,
    /// Raw-topic routes. Frames route by exact topic, so topics are unique per group.
    topics: HashSet<Vec<u8>>,
    command_tx: mpsc::UnboundedSender<GroupCommand>,
    cancel: CancellationToken,
    handle: JoinHandle<()>,
}

impl SocketGroup {
    fn accepts_topic(&self, topic: Option<&[u8]>) -> bool {
        topic.is_none_or(|topic| !self.topics.contains(topic))
    }
}

struct PoolInner {
    groups: HashMap<u64, SocketGroup>,
    next_group_id: u64,
    closed: bool,
    /// Publishers discovered behind each shared endpoint, registered or not.
    discovered: HashMap<String, usize>,
}

impl PoolInner {
    /// Publishers a group carries: per endpoint, the larger of its registered
    /// routes and the publishers discovered behind it.
    fn load(&self, group: &SocketGroup) -> usize {
        group
            .endpoints
            .iter()
            .map(|(endpoint, routes)| (*routes).max(self.discovered_on(endpoint)))
            .sum()
    }

    fn discovered_on(&self, endpoint: &str) -> usize {
        self.discovered.get(endpoint).copied().unwrap_or(0)
    }
}

#[derive(Clone)]
pub(crate) struct DirectZmqSubPool {
    inner: Arc<Mutex<PoolInner>>,
    topic: Arc<str>,
    endpoints_per_sub: usize,
    rcvhwm: i32,
    shared_endpoints: bool,
    raw_topics: bool,
    cancellation_token: CancellationToken,
}

pub(crate) struct DirectZmqSubRegistration<T = DirectZmqSubItem> {
    pub(crate) group_id: u64,
    pub(crate) receiver: mpsc::Receiver<T>,
    pub(crate) disconnected: CancellationToken,
    pool: DirectZmqSubPool,
    publisher_id: u64,
    endpoint: String,
    topic: Option<Vec<u8>>,
    generation: u64,
    armed: bool,
}

pub(crate) enum DirectZmqSubConnection {
    Dedicated(ValidatedZmqSource),
    Grouped(DirectZmqSubRegistration),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DispatchOutcome {
    Delivered,
    RouteClosed { publisher_id: u64, generation: u64 },
    GroupCancelled,
}

impl<T> DirectZmqSubRegistration<T> {
    fn release(&mut self) -> Option<SocketGroup> {
        if !self.armed {
            return None;
        }
        // Wake a group task that may be waiting for this publisher before
        // queuing the ordered endpoint removal.
        self.receiver.close();
        self.armed = false;
        self.pool.remove_registration(
            self.group_id,
            self.publisher_id,
            &self.endpoint,
            self.topic.as_deref(),
            self.generation,
        )
    }

    pub(crate) async fn close(mut self) {
        if let Some(group) = self.release() {
            stop_group(group).await;
        }
    }
}

impl DirectZmqSubConnection {
    pub(crate) fn group_id(&self) -> Option<u64> {
        match self {
            Self::Dedicated(_) => None,
            Self::Grouped(registration) => Some(registration.group_id),
        }
    }

    pub(crate) fn disconnected(&self) -> Option<CancellationToken> {
        match self {
            Self::Dedicated(_) => None,
            Self::Grouped(registration) => Some(registration.disconnected.clone()),
        }
    }

    pub(crate) async fn close(self) {
        if let Self::Grouped(registration) = self {
            registration.close().await;
        }
    }
}

impl<T> Drop for DirectZmqSubRegistration<T> {
    fn drop(&mut self) {
        if let Some(group) = self.release() {
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
                discovered: HashMap::new(),
            })),
            topic: topic.into(),
            endpoints_per_sub,
            rcvhwm,
            shared_endpoints: false,
            raw_topics: false,
            cancellation_token,
        })
    }

    /// A pool for raw publishers, such as vLLM KV event streams, whose frames
    /// carry no publisher ID. Each route subscribes to its own topic and receives
    /// the frames whose topic frame matches it exactly, undecoded.
    pub(crate) fn new_raw(
        endpoints_per_sub: usize,
        rcvhwm: i32,
        cancellation_token: CancellationToken,
    ) -> Result<Self> {
        let mut pool = Self::new("raw", endpoints_per_sub, rcvhwm, cancellation_token)?;
        pool.raw_topics = true;
        Ok(pool)
    }

    /// Expect several publishers behind one PUB endpoint, as a KV state-agent host
    /// multiplexes its slots. Every source then joins a group, so each endpoint is
    /// connected once and routed by publisher ID; a dedicated socket would receive
    /// every co-located publisher's traffic.
    pub(crate) fn with_shared_endpoints(mut self) -> Self {
        self.shared_endpoints = true;
        self
    }

    /// Record a publisher discovered behind `endpoint`, before it registers. A
    /// shared endpoint is placed by every publisher it carries, since all of them
    /// will join the group that connects it.
    pub(crate) fn expect_publisher(&self, endpoint: &str) {
        if self.shared_endpoints {
            *self
                .inner
                .lock()
                .discovered
                .entry(endpoint.to_string())
                .or_default() += 1;
        }
    }

    /// Undo [`Self::expect_publisher`] once the publisher is no longer discovered.
    pub(crate) fn forget_publisher(&self, endpoint: &str) {
        if self.shared_endpoints {
            release_endpoint(&mut self.inner.lock().discovered, endpoint);
        }
    }

    pub(crate) async fn connect(
        &self,
        publisher_id: u64,
        endpoint: &str,
        generation: u64,
    ) -> Result<DirectZmqSubConnection> {
        if self.endpoints_per_sub == 1 && !self.shared_endpoints {
            anyhow::ensure!(
                !self.inner.lock().closed,
                "direct-ZMQ socket pool is closed"
            );
            let source =
                ValidatedZmqSource::connect(endpoint, &self.topic, publisher_id, self.rcvhwm)
                    .await?;
            anyhow::ensure!(
                !self.inner.lock().closed,
                "direct-ZMQ socket pool is closed"
            );
            return Ok(DirectZmqSubConnection::Dedicated(source));
        }

        self.register_grouped(publisher_id, endpoint, generation)
            .await
            .map(DirectZmqSubConnection::Grouped)
    }

    async fn register_grouped(
        &self,
        publisher_id: u64,
        endpoint: &str,
        generation: u64,
    ) -> Result<DirectZmqSubRegistration> {
        anyhow::ensure!(!self.raw_topics, "raw-topic pools route raw sources only");
        self.register_route(
            publisher_id,
            endpoint,
            None,
            generation,
            RouteSender::Envelope,
        )
        .await
    }

    /// Route the frames `endpoint` publishes on exactly `topic` to `route_id`.
    /// Topics must be non-empty, and a group holds each topic once: subscriptions
    /// are prefix filters, so frames are routed only by an exact topic match.
    pub(crate) async fn register_raw(
        &self,
        route_id: u64,
        endpoint: &str,
        topic: &str,
        generation: u64,
    ) -> Result<DirectZmqSubRegistration<RawZmqFrames>> {
        anyhow::ensure!(self.raw_topics, "pool routes event-plane envelopes");
        anyhow::ensure!(!topic.is_empty(), "pooled raw sources need a topic");
        self.register_route(
            route_id,
            endpoint,
            Some(topic),
            generation,
            RouteSender::Raw,
        )
        .await
    }

    async fn register_route<T>(
        &self,
        publisher_id: u64,
        endpoint: &str,
        topic: Option<&str>,
        generation: u64,
        route_sender: fn(mpsc::Sender<T>) -> RouteSender,
    ) -> Result<DirectZmqSubRegistration<T>> {
        let (sender, receiver) = mpsc::channel(self.rcvhwm as usize);
        let topic_bytes = topic.map(|topic| topic.as_bytes().to_vec());
        let disconnected = CancellationToken::new();
        let route = GroupRoute {
            endpoint: endpoint.to_string(),
            generation,
            sender: route_sender(sender),
            topic: topic_bytes.clone(),
            disconnected: disconnected.clone(),
        };
        let (group_id, completion) = {
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

            let running =
                |group: &SocketGroup| !group.command_tx.is_closed() && !group.handle.is_finished();
            // A group already connected to this endpoint takes the route even when
            // full: a second connection would duplicate every co-located publisher.
            let affine = inner
                .groups
                .iter()
                .find(|(_, group)| group.endpoints.contains_key(endpoint) && running(group));
            if let Some((_, group)) = affine {
                anyhow::ensure!(
                    group.accepts_topic(topic_bytes.as_deref()),
                    "duplicate raw topic on endpoint {endpoint}"
                );
            }
            // A new endpoint brings every publisher discovered behind it, so place it
            // by that count; otherwise discovery order decides which group ends up
            // reading most shared endpoints.
            let weight = inner.discovered_on(endpoint).max(1);
            let group_id = affine.map(|(group_id, _)| *group_id).or_else(|| {
                inner
                    .groups
                    .iter()
                    .map(|(group_id, group)| (*group_id, group, inner.load(group)))
                    .filter(|(_, group, load)| {
                        load + weight <= self.endpoints_per_sub
                            && running(group)
                            && group.accepts_topic(topic_bytes.as_deref())
                    })
                    .min_by_key(|(group_id, _, load)| (*load, *group_id))
                    .map(|(group_id, _, _)| group_id)
            });

            if let Some(group_id) = group_id {
                let group = inner
                    .groups
                    .get_mut(&group_id)
                    .expect("selected socket group must exist");
                group.assignments.insert(publisher_id, generation);
                *group.endpoints.entry(endpoint.to_string()).or_default() += 1;
                group.topics.extend(topic_bytes.clone());
                let (completed, completion) = oneshot::channel();
                let command = GroupCommand::Add {
                    publisher_id,
                    route,
                    completed,
                };
                if group.command_tx.send(command).is_err() {
                    group.assignments.remove(&publisher_id);
                    release_endpoint(&mut group.endpoints, endpoint);
                    release_topic(&mut group.topics, topic_bytes.as_deref());
                    anyhow::bail!("direct-ZMQ socket group stopped");
                }
                (group_id, Some(completion))
            } else {
                // libzmq connects asynchronously. Keep group creation serialized so
                // concurrent registrations cannot create excess transient sockets.
                let socket = DynamicZmqSubSocket::connect_with_rcvhwm(
                    endpoint,
                    topic.unwrap_or(&self.topic),
                    self.rcvhwm,
                )?;
                let group_id = inner.next_group_id;
                inner.next_group_id = inner.next_group_id.wrapping_add(1);
                let (command_tx, command_rx) = mpsc::unbounded_channel();
                let cancel = self.cancellation_token.child_token();
                let routes = HashMap::from([(publisher_id, route)]);
                let handle = tokio::spawn(run_socket_group(
                    group_id,
                    self.topic.clone(),
                    socket,
                    routes,
                    command_rx,
                    self.shared_endpoints,
                    self.raw_topics,
                    cancel.clone(),
                ));
                inner.groups.insert(
                    group_id,
                    SocketGroup {
                        assignments: HashMap::from([(publisher_id, generation)]),
                        endpoints: HashMap::from([(endpoint.to_string(), 1)]),
                        topics: topic_bytes.clone().into_iter().collect(),
                        command_tx,
                        cancel,
                        handle,
                    },
                );
                (group_id, None)
            }
        };
        let registration = DirectZmqSubRegistration {
            pool: self.clone(),
            group_id,
            publisher_id,
            endpoint: endpoint.to_string(),
            topic: topic_bytes,
            generation,
            receiver,
            disconnected,
            armed: true,
        };

        let Some(completion) = completion else {
            return Ok(registration);
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

    fn remove_registration(
        &self,
        group_id: u64,
        publisher_id: u64,
        endpoint: &str,
        topic: Option<&[u8]>,
        generation: u64,
    ) -> Option<SocketGroup> {
        let mut inner = self.inner.lock();
        Self::reap_failed_groups(&mut inner);
        let group = inner.groups.get_mut(&group_id)?;
        if group.assignments.get(&publisher_id) != Some(&generation) {
            return None;
        }
        group.assignments.remove(&publisher_id);
        release_endpoint(&mut group.endpoints, endpoint);
        release_topic(&mut group.topics, topic);
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

/// One group socket's routes, and for raw-topic pools the topic index into them.
struct GroupRoutes {
    routes: HashMap<u64, GroupRoute>,
    by_topic: HashMap<Vec<u8>, u64>,
}

impl GroupRoutes {
    fn new(routes: HashMap<u64, GroupRoute>) -> Self {
        let by_topic = routes
            .iter()
            .filter_map(|(publisher_id, route)| Some((route.topic.clone()?, *publisher_id)))
            .collect();
        Self { routes, by_topic }
    }

    fn add(
        &mut self,
        socket: &mut DynamicZmqSubSocket,
        publisher_id: u64,
        route: GroupRoute,
    ) -> Result<()> {
        anyhow::ensure!(
            !self.routes.contains_key(&publisher_id),
            "publisher {publisher_id} is already registered"
        );
        // Already connected for a co-located publisher.
        if !self
            .routes
            .values()
            .any(|existing| existing.endpoint == route.endpoint)
        {
            socket.add_endpoint(&route.endpoint)?;
        }
        if let Some(topic) = &route.topic {
            socket.subscribe(topic)?;
            self.by_topic.insert(topic.clone(), publisher_id);
        }
        self.routes.insert(publisher_id, route);
        Ok(())
    }

    fn remove(
        &mut self,
        socket: &mut DynamicZmqSubSocket,
        publisher_id: u64,
        generation: u64,
    ) -> Result<()> {
        match self.routes.get(&publisher_id) {
            Some(route) if route.generation == generation => {
                let route = self
                    .routes
                    .remove(&publisher_id)
                    .expect("route was present");
                if let Some(topic) = &route.topic {
                    self.by_topic.remove(topic);
                    socket.unsubscribe(topic)?;
                }
                if self
                    .routes
                    .values()
                    .any(|other| other.endpoint == route.endpoint)
                {
                    return Ok(());
                }
                socket.remove_endpoint(&route.endpoint)
            }
            _ => Ok(()),
        }
    }
}

enum GroupMessage {
    Envelope(ZmqWireMessage),
    Raw(tmq::Multipart),
}

async fn next_group_message(
    socket: &mut DynamicZmqSubSocket,
    raw_topics: bool,
) -> Option<Result<GroupMessage>> {
    if raw_topics {
        let frames = socket.next_frames().await?;
        Some(frames.map(GroupMessage::Raw))
    } else {
        let message = socket.next().await?;
        Some(message.map(GroupMessage::Envelope))
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_socket_group(
    group_id: u64,
    topic: Arc<str>,
    mut socket: DynamicZmqSubSocket,
    routes: HashMap<u64, GroupRoute>,
    mut command_rx: mpsc::UnboundedReceiver<GroupCommand>,
    shared_endpoints: bool,
    raw_topics: bool,
    cancellation_token: CancellationToken,
) {
    let codec = Codec::default();
    let mut routes = GroupRoutes::new(routes);
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
                        let _ = completed.send(routes.add(&mut socket, publisher_id, route));
                    }
                    GroupCommand::Remove { publisher_id, generation } => {
                        if let Err(error) = routes.remove(&mut socket, publisher_id, generation) {
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
                            &routes.routes,
                            shared_endpoints,
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
            message = next_group_message(&mut socket, raw_topics) => {
                let Some(message) = message else {
                    tracing::warn!(group_id, topic = %topic, "Direct-ZMQ socket group stopped");
                    return;
                };
                let outcome = match message {
                    Ok(GroupMessage::Envelope(message)) => {
                        dispatch_group_message(
                            group_id,
                            &topic,
                            message,
                            &codec,
                            &routes.routes,
                            shared_endpoints,
                            &cancellation_token,
                        )
                        .await
                    }
                    Ok(GroupMessage::Raw(frames)) => {
                        dispatch_raw_message(group_id, frames, &routes, &cancellation_token).await
                    }
                    Err(error) => {
                        tracing::warn!(%error, group_id, topic = %topic, "Direct-ZMQ socket group receive failed");
                        return;
                    }
                };
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
    routes: &mut GroupRoutes,
) -> bool {
    let DispatchOutcome::RouteClosed {
        publisher_id,
        generation,
    } = outcome
    else {
        return outcome == DispatchOutcome::Delivered;
    };

    if let Err(error) = routes.remove(socket, publisher_id, generation) {
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

async fn dispatch_group_message(
    group_id: u64,
    topic: &str,
    message: ZmqWireMessage,
    codec: &Codec,
    routes: &HashMap<u64, GroupRoute>,
    shared_endpoints: bool,
    cancellation_token: &CancellationToken,
) -> DispatchOutcome {
    let Some(route) = routes.get(&message.publisher_id) else {
        // A shared endpoint also carries co-located publishers this router has not
        // discovered yet; that is expected rather than a misrouted connection.
        if shared_endpoints {
            tracing::trace!(
                group_id,
                topic,
                publisher_id = message.publisher_id,
                "Dropping direct-ZMQ envelope from an undiscovered co-located publisher"
            );
        } else {
            tracing::warn!(
                group_id,
                topic,
                publisher_id = message.publisher_id,
                "Dropping direct-ZMQ envelope from an unknown publisher"
            );
        }
        return DispatchOutcome::Delivered;
    };
    let RouteSender::Envelope(sender) = &route.sender else {
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
    deliver(
        sender,
        item,
        message.publisher_id,
        route.generation,
        cancellation_token,
    )
    .await
}

async fn dispatch_raw_message(
    group_id: u64,
    frames: tmq::Multipart,
    routes: &GroupRoutes,
    cancellation_token: &CancellationToken,
) -> DispatchOutcome {
    // Subscriptions are prefix filters; only an exact topic frame is this route's.
    let route = frames
        .0
        .front()
        .and_then(|topic| routes.by_topic.get(&topic[..]))
        .and_then(|publisher_id| Some((*publisher_id, routes.routes.get(publisher_id)?)));
    let Some((publisher_id, route)) = route else {
        tracing::trace!(
            group_id,
            "Dropping raw ZMQ frames without an exact topic route"
        );
        return DispatchOutcome::Delivered;
    };
    let RouteSender::Raw(sender) = &route.sender else {
        return DispatchOutcome::Delivered;
    };
    deliver(
        sender,
        multipart_message(frames),
        publisher_id,
        route.generation,
        cancellation_token,
    )
    .await
}

/// Hand one item to its route, waiting while the route is full so a slow
/// consumer backpressures the socket instead of losing frames.
async fn deliver<T>(
    sender: &mpsc::Sender<T>,
    item: T,
    publisher_id: u64,
    generation: u64,
    cancellation_token: &CancellationToken,
) -> DispatchOutcome {
    match sender.try_send(item) {
        Ok(()) => DispatchOutcome::Delivered,
        Err(mpsc::error::TrySendError::Full(item)) => {
            let result = tokio::select! {
                biased;
                _ = cancellation_token.cancelled() => return DispatchOutcome::GroupCancelled,
                result = sender.send(item) => result,
            };
            match result {
                Ok(()) => DispatchOutcome::Delivered,
                Err(_) => DispatchOutcome::RouteClosed {
                    publisher_id,
                    generation,
                },
            }
        }
        Err(mpsc::error::TrySendError::Closed(_)) => DispatchOutcome::RouteClosed {
            publisher_id,
            generation,
        },
    }
}

fn release_endpoint(endpoints: &mut HashMap<String, usize>, endpoint: &str) {
    if let Some(count) = endpoints.get_mut(endpoint) {
        *count -= 1;
        if *count == 0 {
            endpoints.remove(endpoint);
        }
    }
}

fn release_topic(topics: &mut HashSet<Vec<u8>>, topic: Option<&[u8]>) {
    if let Some(topic) = topic {
        topics.remove(topic);
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

    fn endpoint(publisher_id: u64) -> String {
        format!("tcp://127.0.0.1:{}", 31_000 + publisher_id)
    }

    async fn register(
        pool: &DirectZmqSubPool,
        publisher_id: u64,
        generation: u64,
    ) -> DirectZmqSubRegistration {
        pool.register_grouped(publisher_id, &endpoint(publisher_id), generation)
            .await
            .unwrap()
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

    fn dispatch(
        pool: &DirectZmqSubPool,
        group_id: u64,
        publisher_id: u64,
        sequence: u64,
    ) -> oneshot::Receiver<DispatchOutcome> {
        let (completed, completion) = oneshot::channel();
        let message = wire_message(&pool.topic, publisher_id, sequence);
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

    async fn delivered(pool: &DirectZmqSubPool, group_id: u64, publisher_id: u64, sequence: u64) {
        assert_eq!(
            dispatch(pool, group_id, publisher_id, sequence)
                .await
                .unwrap(),
            DispatchOutcome::Delivered
        );
    }

    async fn pause_group(pool: &DirectZmqSubPool, group_id: u64) -> oneshot::Sender<()> {
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        pool.inner
            .lock()
            .groups
            .get(&group_id)
            .unwrap()
            .command_tx
            .send(GroupCommand::Pause {
                started: started_tx,
                release: release_rx,
            })
            .unwrap();
        started_rx.await.unwrap();
        release_tx
    }

    async fn wait_for_assignment(pool: &DirectZmqSubPool, group_id: u64, publisher_id: u64) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while !pool
                .inner
                .lock()
                .groups
                .get(&group_id)
                .unwrap()
                .assignments
                .contains_key(&publisher_id)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("registration should reserve a group slot");
    }

    #[test]
    fn parses_endpoints_per_sub_configuration() {
        assert_eq!(DEFAULT_ENDPOINTS_PER_SUB, 1);
        for (value, expected) in [(None, 1), (Some("1"), 1), (Some("128"), 128)] {
            assert_eq!(
                endpoints_per_sub_from_lookup(config_lookup(value)).unwrap(),
                expected
            );
        }
        for invalid in ["", "0", "-1", "not-a-number"] {
            assert!(endpoints_per_sub_from_lookup(config_lookup(Some(invalid))).is_err());
        }
    }

    #[tokio::test]
    async fn one_endpoint_uses_dedicated_source_without_a_group() {
        let pool = pool("kv-events", 1, 128);
        let connection = pool.connect(1, "tcp://127.0.0.1:31001", 1).await.unwrap();

        assert!(matches!(connection, DirectZmqSubConnection::Dedicated(_)));
        assert_eq!(pool.group_count(), 0);
        connection.close().await;
    }

    #[tokio::test]
    async fn creates_three_groups_for_129_publishers() {
        let pool = pool("kv-events", 64, 128);
        let registrations = futures::future::join_all((1..=129).map(|publisher_id| {
            let pool = pool.clone();
            async move { register(&pool, publisher_id, publisher_id).await }
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
        let mut source = register(&pool, 1, 1).await;
        let mut sibling = register(&pool, 2, 1).await;
        assert_eq!(source.group_id, sibling.group_id);

        delivered(&pool, source.group_id, 1, 0).await;
        let mut blocked = dispatch(&pool, source.group_id, 1, 1);
        tokio::task::yield_now().await;
        assert!(matches!(
            blocked.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        let mut sibling_blocked = dispatch(&pool, source.group_id, 2, 0);
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
        let mut blocked_source = register(&pool, 1, 1).await;
        let old = register(&pool, 2, 1).await;
        let group_id = blocked_source.group_id;

        delivered(&pool, group_id, 1, 0).await;
        let blocked = dispatch(&pool, group_id, 1, 1);
        tokio::task::yield_now().await;

        drop(old);
        let replacement_pool = pool.clone();
        let replacement = tokio::spawn(async move { register(&replacement_pool, 2, 2).await });
        tokio::task::yield_now().await;
        assert!(!replacement.is_finished());

        assert_eq!(sequence(blocked_source.receiver.recv().await.unwrap()), 0);
        assert_eq!(blocked.await.unwrap(), DispatchOutcome::Delivered);
        let mut replacement = replacement.await.unwrap();
        assert_eq!(replacement.group_id, group_id);
        delivered(&pool, group_id, 2, 0).await;
        assert_eq!(sequence(replacement.receiver.recv().await.unwrap()), 0);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn close_unblocks_a_full_lane_and_keeps_the_group_alive() {
        let pool = pool("kv_metrics", 64, 1);
        let source = register(&pool, 1, 1).await;
        let mut sibling = register(&pool, 2, 1).await;
        let disconnected = source.disconnected.clone();
        assert_eq!(source.group_id, sibling.group_id);

        delivered(&pool, source.group_id, 1, 0).await;
        let blocked = dispatch(&pool, source.group_id, 1, 1);
        tokio::task::yield_now().await;

        tokio::time::timeout(Duration::from_secs(1), source.close())
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

        delivered(&pool, sibling.group_id, 2, 0).await;
        assert_eq!(sequence(sibling.receiver.recv().await.unwrap()), 0);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_interrupts_a_blocked_lane_send() {
        let pool = pool("kv_metrics", 64, 1);
        let source = register(&pool, 1, 1).await;
        let disconnected = source.disconnected.clone();

        delivered(&pool, source.group_id, 1, 0).await;
        let blocked = dispatch(&pool, source.group_id, 1, 1);
        tokio::task::yield_now().await;

        tokio::time::timeout(Duration::from_secs(1), pool.shutdown())
            .await
            .expect("shutdown must interrupt a blocked lane send");
        assert_eq!(blocked.await.unwrap(), DispatchOutcome::GroupCancelled);
        assert!(disconnected.is_cancelled());
        assert_eq!(pool.group_count(), 0);
        assert!(
            pool.register_grouped(2, "tcp://127.0.0.1:31002", 1)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn reports_validation_errors_to_the_affected_publisher() {
        let (sender, mut receiver) = mpsc::channel(3);
        let routes = HashMap::from([(
            1,
            GroupRoute {
                endpoint: "tcp://127.0.0.1:1".to_string(),
                generation: 1,
                sender: RouteSender::Envelope(sender),
                topic: None,
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
                false,
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
                false,
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
    async fn co_located_publishers_share_one_connection() {
        let pool = pool("kv-state-events-v2", 2, 128).with_shared_endpoints();
        let shared = "tcp://127.0.0.1:31100";
        let DirectZmqSubConnection::Grouped(mut first) = pool.connect(1, shared, 1).await.unwrap()
        else {
            panic!("shared endpoints must not use a dedicated socket");
        };
        let mut second = pool.register_grouped(2, shared, 1).await.unwrap();
        let other = register(&pool, 3, 1).await;
        // The shared endpoint's group is full, but a co-located publisher must still
        // join it rather than open a second connection.
        let mut third = pool.register_grouped(4, shared, 1).await.unwrap();
        assert_eq!(first.group_id, second.group_id);
        assert_eq!(first.group_id, third.group_id);
        assert_ne!(first.group_id, other.group_id);

        delivered(&pool, first.group_id, 1, 0).await;
        delivered(&pool, first.group_id, 2, 5).await;
        delivered(&pool, first.group_id, 4, 7).await;
        assert_eq!(sequence(first.receiver.recv().await.unwrap()), 0);
        assert_eq!(sequence(second.receiver.recv().await.unwrap()), 5);
        assert_eq!(sequence(third.receiver.recv().await.unwrap()), 7);

        let group_id = first.group_id;
        first.close().await;
        delivered(&pool, group_id, 2, 6).await;
        assert_eq!(sequence(second.receiver.recv().await.unwrap()), 6);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn shared_endpoints_are_placed_by_discovered_publishers() {
        // Discovery order is unrelated to which PUB a publisher sits on. Placed as
        // one publisher each, both endpoints would share the first reader.
        let pool = pool("kv-state-events-v2", 4, 128).with_shared_endpoints();
        let endpoints = ["tcp://127.0.0.1:31300", "tcp://127.0.0.1:31301"];
        for endpoint in endpoints {
            for _ in 0..4 {
                pool.expect_publisher(endpoint);
            }
        }
        let mut registrations = Vec::new();
        for publisher_id in 0..8 {
            let endpoint = endpoints[publisher_id as usize % 2];
            registrations.push(
                pool.register_grouped(publisher_id, endpoint, 1)
                    .await
                    .unwrap(),
            );
        }
        let groups: Vec<_> = registrations.iter().map(|r| r.group_id).collect();
        assert_ne!(groups[0], groups[1]);
        for (index, group) in groups.iter().enumerate() {
            assert_eq!(*group, groups[index % 2]);
        }

        // Endpoints with one publisher each still pack to the cap.
        let mut singles = Vec::new();
        for publisher_id in 10..15 {
            pool.expect_publisher(&endpoint(publisher_id));
            singles.push(register(&pool, publisher_id, 1).await);
        }
        let groups: Vec<_> = singles.iter().map(|r| r.group_id).collect();
        assert!(groups[..4].iter().all(|group| *group == groups[0]));
        assert_ne!(groups[4], groups[0]);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn raw_topics_route_exact_frames_and_separate_duplicates() {
        use tmq::AsZmqSocket;

        let pool = DirectZmqSubPool::new_raw(4, 128, CancellationToken::new()).unwrap();
        let publisher = crate::utils::zmq::bind_pub_socket("tcp://127.0.0.1:*")
            .await
            .unwrap();
        let endpoint = publisher
            .lock()
            .await
            .get_socket()
            .get_last_endpoint()
            .unwrap()
            .unwrap();
        let mut first = pool
            .register_raw(1, &endpoint, "slot-0001", 1)
            .await
            .unwrap();
        // A prefix of another route's topic shares the socket; routing is exact.
        let mut prefix = pool
            .register_raw(2, "tcp://127.0.0.1:31201", "slot-000", 1)
            .await
            .unwrap();
        // The same topic on another endpoint could not be told apart.
        let duplicate = pool
            .register_raw(3, "tcp://127.0.0.1:31202", "slot-0001", 1)
            .await
            .unwrap();
        assert_eq!(first.group_id, prefix.group_id);
        assert_ne!(first.group_id, duplicate.group_id);
        assert!(
            pool.register_raw(4, "tcp://127.0.0.1:31203", "", 1)
                .await
                .is_err()
        );

        let frames = |topic: &str, sequence: u64| {
            vec![
                topic.as_bytes().to_vec(),
                sequence.to_be_bytes().to_vec(),
                b"payload".to_vec(),
            ]
        };
        let received = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                // Passes the "slot-0001" filter but is not that route's topic.
                crate::utils::zmq::send_multipart(&publisher, frames("slot-00010", 0))
                    .await
                    .unwrap();
                crate::utils::zmq::send_multipart(&publisher, frames("slot-0001", 1))
                    .await
                    .unwrap();
                if let Ok(Some(frames)) =
                    tokio::time::timeout(Duration::from_millis(50), first.receiver.recv()).await
                {
                    return frames;
                }
            }
        })
        .await
        .expect("the exact topic route should receive frames");
        assert_eq!(received, frames("slot-0001", 1));
        assert!(prefix.receiver.try_recv().is_err());
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn failed_group_is_replaced_without_affecting_other_groups() {
        let pool = pool("kv_metrics", 2, 128);
        let first = register(&pool, 1, 1).await;
        let second = register(&pool, 2, 1).await;
        let mut unaffected = register(&pool, 3, 1).await;
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
        delivered(&pool, unaffected.group_id, 3, 0).await;
        assert_eq!(sequence(unaffected.receiver.recv().await.unwrap()), 0);

        let replacement = register(&pool, 1, 2).await;
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
    async fn registrations_clean_up_on_close_and_drop() {
        let pool = pool("kv-events", 64, 128);
        let registration = register(&pool, 1, 1).await;
        let disconnected = registration.disconnected.clone();

        registration.close().await;
        assert!(disconnected.is_cancelled());
        assert_eq!(pool.group_count(), 0);

        let registration = register(&pool, 1, 2).await;
        let disconnected = registration.disconnected.clone();
        drop(registration);
        assert_eq!(pool.group_count(), 0);
        tokio::time::timeout(Duration::from_secs(1), disconnected.cancelled())
            .await
            .expect("registration drop must stop its final socket group");

        let replacement = register(&pool, 1, 3).await;
        assert_eq!(pool.group_count(), 1);
        drop(replacement);
        pool.shutdown().await;
        assert!(
            pool.register_grouped(2, "tcp://127.0.0.1:31002", 1)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn cancelled_registration_rolls_back_and_can_register_again() {
        let pool = pool("kv-events", 64, 128);
        let first = register(&pool, 1, 1).await;
        let release = pause_group(&pool, first.group_id).await;

        let pending_pool = pool.clone();
        let pending = tokio::spawn(async move { register(&pending_pool, 2, 1).await });
        wait_for_assignment(&pool, first.group_id, 2).await;
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

        release.send(()).unwrap();
        let replacement = register(&pool, 2, 2).await;
        assert_eq!(replacement.group_id, first.group_id);
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_rejects_an_inflight_registration() {
        let pool = pool("kv-events", 64, 128);
        let first = register(&pool, 1, 1).await;
        let _release = pause_group(&pool, first.group_id).await;

        let pending_pool = pool.clone();
        let pending =
            tokio::spawn(async move { pending_pool.register_grouped(2, &endpoint(2), 1).await });
        wait_for_assignment(&pool, first.group_id, 2).await;

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
