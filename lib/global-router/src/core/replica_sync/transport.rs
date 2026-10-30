// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The transport's state in one replica.
//!
//! The core gives each partition a pair of host replica channels
//! ([`HostReplicaChannels`]): it writes its booking events to the outbound
//! one and reads its peers' events from the inbound one. [`ReplicaHub`]
//! supplies those channels through the factory the core calls once per
//! partition, batches each partition's outbound events into
//! [`proto::ReplicaFrame`]s on one broadcast every subscriber reads, and
//! routes inbound frames to the partition they name. Session bindings go
//! the same way through the core's [`SessionBindingUpdates`] and
//! [`SelectionService::apply_session_binding`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use anyhow::{Result, bail};
use dashmap::DashMap;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::protocols::{
    ActiveSequenceEvent, ActiveSequenceEventData, MAX_REPLICA_BATCH_DURATION,
    MAX_REPLICA_BATCH_EVENTS, WorkerWithDpRank,
};
use dynamo_kv_router::services::selection::{
    HostReplicaChannels, HostReplicaSyncFactory, SelectionService, SessionBindingUpdates,
};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{Notify, broadcast, mpsc};
use tokio_util::sync::CancellationToken;
use url::Url;

use super::proto;
use super::proto::replica_frame::Frame;
use super::{from_wire_binding, from_wire_event, to_wire_binding, to_wire_event};

/// Events a partition's channels hold before they drop, as the core's own
/// mesh sizes them (`REPLICA_EVENT_CHANNEL_CAPACITY`).
const EVENT_CHANNEL_CAPACITY: usize = 100_000;
/// Frames a subscriber may fall behind before it loses the oldest.
const FRAME_BUFFER: usize = 1024;

/// Replicate bookings and session bindings with other Global Frontends over
/// the gRPC `ReplicaSync` stream, served on this replica's API port.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GrpcReplicaSync {
    /// Peers to subscribe to at start, as `http://host:port`: each peer's
    /// API address. More can join through
    /// [`crate::CoreSelector::register_peer`].
    pub peers: Vec<Url>,
}

/// A peer's API address as the dialer accepts it: `http://host:port`.
/// `https://` is refused until the transport carries TLS.
pub fn peer_url(endpoint: &str) -> Result<Url> {
    let url: Url = endpoint
        .parse()
        .map_err(|error| anyhow::anyhow!("peer {endpoint}: {error}"))?;
    match url.scheme() {
        "http" => {}
        "https" => bail!("peer {endpoint}: TLS peers are not supported yet; use http://"),
        other => bail!("peer {endpoint}: expected http://host:port, got scheme {other}://"),
    }
    if url.host_str().is_none() {
        bail!("peer {endpoint}: expected http://host:port");
    }
    Ok(url)
}

/// Counts of what the transport dropped or saw, for metrics and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplicaSyncStats {
    /// Peers whose stream is open right now.
    pub connected_peers: u64,
    /// Frames this replica published to its subscribers.
    pub frames_published: u64,
    /// Frames this replica received from its peers.
    pub frames_received: u64,
    /// Inbound events dropped because the partition's channel was full.
    pub events_dropped: u64,
    /// Inbound events dropped because they named no partition this core has,
    /// or did not convert.
    pub events_unroutable: u64,
    /// Inbound bindings the core refused or that did not convert.
    pub bindings_rejected: u64,
    /// Frames subscribers missed for reading too slowly.
    pub subscribers_lagged: u64,
    /// Bookings freed on this replica's behalf of peers that left
    /// ([`ReplicaHub::remove_peer`]).
    pub bookings_released: u64,
}

/// What a peer's `AddRequest` opened, as its `Free` would name it.
struct OpenBooking {
    worker: WorkerWithDpRank,
    lora_name: Option<String>,
}

#[derive(Default)]
struct Counters {
    connected_peers: AtomicU64,
    frames_published: AtomicU64,
    frames_received: AtomicU64,
    events_dropped: AtomicU64,
    events_unroutable: AtomicU64,
    bindings_rejected: AtomicU64,
    subscribers_lagged: AtomicU64,
    bookings_released: AtomicU64,
}

impl Counters {
    fn released(&self, peer: &Url, released: u64) {
        if released > 0 {
            self.bookings_released.fetch_add(released, Relaxed);
            tracing::info!(%peer, released, "freed the bookings of a replica that left");
        }
    }
}

/// One replica's end of the stream: the core's channels in, frames out, and
/// the peers it follows.
pub struct ReplicaHub {
    process_id: u64,
    /// Each partition's inbound channel, filled as the core creates them.
    partitions: DashMap<RoutingPartitionId, mpsc::Sender<ActiveSequenceEvent>>,
    /// Every frame this replica publishes; each subscriber reads a copy.
    frames: broadcast::Sender<proto::ReplicaFrame>,
    /// The peers to follow, each with the token that stops following it.
    peers: DashMap<Url, CancellationToken>,
    peers_changed: Notify,
    /// The bookings each peer's events have opened and not yet freed, by
    /// the router id the events carry (the peer's process), so a peer that
    /// leaves can have them freed rather than left to the core's expiry.
    open: DashMap<u64, HashMap<(RoutingPartitionId, String), OpenBooking>>,
    /// The router ids each peer's stream has carried events for.
    routers_of: DashMap<Url, HashSet<u64>>,
    cancel: CancellationToken,
    counters: Arc<Counters>,
}

impl ReplicaHub {
    /// A hub whose tasks and streams end when `cancel` fires. Its process
    /// id, which stamps every event this replica publishes, is random and
    /// never zero.
    pub fn new(cancel: CancellationToken) -> Arc<Self> {
        let process_id = loop {
            let id: u64 = rand::random();
            if id != 0 {
                break id;
            }
        };
        let (frames, _) = broadcast::channel(FRAME_BUFFER);
        Arc::new(Self {
            process_id,
            partitions: DashMap::new(),
            frames,
            peers: DashMap::new(),
            peers_changed: Notify::new(),
            open: DashMap::new(),
            routers_of: DashMap::new(),
            cancel,
            counters: Arc::default(),
        })
    }

    /// The id this replica stamps on its events as `router_id` and
    /// `writer_id`.
    pub fn process_id(&self) -> u64 {
        self.process_id
    }

    /// Fires when the hub shuts down.
    pub fn cancelled(&self) -> tokio_util::sync::WaitForCancellationFuture<'_> {
        self.cancel.cancelled()
    }

    pub fn stats(&self) -> ReplicaSyncStats {
        let c = &self.counters;
        ReplicaSyncStats {
            connected_peers: c.connected_peers.load(Relaxed),
            frames_published: c.frames_published.load(Relaxed),
            frames_received: c.frames_received.load(Relaxed),
            events_dropped: c.events_dropped.load(Relaxed),
            events_unroutable: c.events_unroutable.load(Relaxed),
            bindings_rejected: c.bindings_rejected.load(Relaxed),
            subscribers_lagged: c.subscribers_lagged.load(Relaxed),
            bookings_released: c.bookings_released.load(Relaxed),
        }
    }

    /// The factory the core calls once per partition. It may run under the
    /// core's locks, so it only opens channels and starts the partition's
    /// publisher task; it needs a Tokio runtime.
    pub fn factory(self: &Arc<Self>) -> HostReplicaSyncFactory {
        let hub = self.clone();
        Arc::new(move |partition: &RoutingPartitionId| {
            let (outbound_tx, outbound_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
            let (inbound_tx, inbound_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
            hub.partitions.insert(partition.clone(), inbound_tx.clone());
            tokio::spawn(publish_events(hub.clone(), partition.clone(), outbound_rx));
            Some(HostReplicaChannels {
                outbound: Some(outbound_tx),
                inbound_tx,
                inbound_rx,
                process_id: hub.process_id,
                ingress_observer: None,
            })
        })
    }

    /// Starts following `peer`. `false` if it was already a peer.
    pub fn add_peer(&self, peer: Url) -> bool {
        let added = match self.peers.entry(peer) {
            dashmap::Entry::Occupied(_) => false,
            dashmap::Entry::Vacant(entry) => {
                entry.insert(self.cancel.child_token());
                true
            }
        };
        if added {
            self.peers_changed.notify_one();
        }
        added
    }

    /// Stops following `peer` and frees the bookings its events opened,
    /// since the frees for them will never arrive: an authoritative removal
    /// (Global Discovery no longer lists the replica) means its requests
    /// are gone with it. `false` if it was not a peer.
    pub fn remove_peer(&self, peer: &Url) -> bool {
        let Some((_, follow)) = self.peers.remove(peer) else {
            return false;
        };
        follow.cancel();
        self.peers_changed.notify_one();
        self.release_bookings_of(peer);
        true
    }

    /// Frees, in the core, every booking still open from the router ids
    /// `peer`'s stream carried, unless another peer still carries them. A
    /// free the partition cannot take at once is delivered as soon as it
    /// can, never lost.
    fn release_bookings_of(&self, peer: &Url) {
        let Some((_, routers)) = self.routers_of.remove(peer) else {
            return;
        };
        let mut released = 0u64;
        let mut pending = Vec::new();
        for router_id in routers {
            let still_carried = self
                .routers_of
                .iter()
                .any(|entry| entry.value().contains(&router_id));
            if still_carried {
                continue;
            }
            let Some((_, bookings)) = self.open.remove(&router_id) else {
                continue;
            };
            for ((partition, request_id), booking) in bookings {
                let free = ActiveSequenceEvent {
                    request_id,
                    worker: booking.worker,
                    data: ActiveSequenceEventData::Free,
                    router_id,
                    lora_name: booking.lora_name,
                };
                let Some(inbound) = self.partitions.get(&partition) else {
                    self.counters.events_unroutable.fetch_add(1, Relaxed);
                    continue;
                };
                match inbound.try_send(free) {
                    Ok(()) => released += 1,
                    Err(TrySendError::Full(free)) => pending.push((inbound.clone(), free)),
                    Err(TrySendError::Closed(_)) => {
                        self.counters.events_dropped.fetch_add(1, Relaxed);
                    }
                }
            }
        }
        self.counters.released(peer, released);
        if !pending.is_empty() {
            let (counters, peer) = (self.counters.clone(), peer.clone());
            tokio::spawn(async move {
                let mut released = 0u64;
                for (inbound, free) in pending {
                    match inbound.send(free).await {
                        Ok(()) => released += 1,
                        Err(_) => {
                            counters.events_dropped.fetch_add(1, Relaxed);
                        }
                    }
                }
                counters.released(&peer, released);
            });
        }
    }

    pub fn peers(&self) -> Vec<Url> {
        self.peers.iter().map(|peer| peer.key().clone()).collect()
    }

    /// Each peer with the token that stops following it.
    pub(super) fn peer_tokens(&self) -> Vec<(Url, CancellationToken)> {
        self.peers
            .iter()
            .map(|peer| (peer.key().clone(), peer.value().clone()))
            .collect()
    }

    /// Resolves after the next [`Self::add_peer`] or [`Self::remove_peer`].
    pub(super) async fn peers_changed(&self) {
        self.peers_changed.notified().await;
    }

    /// A reader of every frame published from now on.
    pub(super) fn subscribe(&self) -> broadcast::Receiver<proto::ReplicaFrame> {
        self.frames.subscribe()
    }

    pub(super) fn publish(&self, frame: proto::ReplicaFrame) {
        self.counters.frames_published.fetch_add(1, Relaxed);
        // An error means no subscriber is reading, which is not a loss.
        let _ = self.frames.send(frame);
    }

    pub(super) fn subscriber_lagged(&self, frames: u64) {
        self.counters.subscribers_lagged.fetch_add(frames, Relaxed);
    }

    pub(super) fn peer_connected(&self) {
        self.counters.connected_peers.fetch_add(1, Relaxed);
    }

    pub(super) fn peer_disconnected(&self) {
        self.counters.connected_peers.fetch_sub(1, Relaxed);
    }

    /// Hands a frame to the core: each event to its partition's inbound
    /// channel, each binding to the session table. The core drops events
    /// stamped with this replica's id; they are skipped here first.
    pub fn apply(&self, frame: proto::ReplicaFrame, service: &SelectionService) {
        self.apply_inner(None, frame, service);
    }

    /// [`Self::apply`] for a frame read from `peer`'s stream, so the
    /// bookings it opens can be freed if the peer leaves. A frame from a
    /// peer no longer followed is ignored: it cannot re-book what
    /// [`Self::remove_peer`] freed.
    pub fn apply_from(&self, peer: &Url, frame: proto::ReplicaFrame, service: &SelectionService) {
        // Held while the frame is applied, so `remove_peer` waits for it:
        // nothing lands after the peer's bookings are freed.
        let Some(_followed) = self.peers.get(peer) else {
            return;
        };
        self.apply_inner(Some(peer), frame, service);
    }

    fn apply_inner(
        &self,
        peer: Option<&Url>,
        frame: proto::ReplicaFrame,
        service: &SelectionService,
    ) {
        self.counters.frames_received.fetch_add(1, Relaxed);
        let (events, bindings) = match frame.frame {
            Some(Frame::Snapshot(snapshot)) => (snapshot.open_bookings, snapshot.bindings),
            Some(Frame::Events(batch)) => (batch.events, batch.bindings),
            None => return,
        };
        for event in events {
            if event.router_id == self.process_id {
                continue;
            }
            let (partition, event) = match from_wire_event(event) {
                Ok(converted) => converted,
                Err(error) => {
                    self.counters.events_unroutable.fetch_add(1, Relaxed);
                    tracing::debug!(%error, "a replica event did not convert");
                    continue;
                }
            };
            let Some(inbound) = self.partitions.get(&partition) else {
                self.counters.events_unroutable.fetch_add(1, Relaxed);
                tracing::debug!(%partition, "a replica event named a partition this core has not created");
                continue;
            };
            let opened = match &event.data {
                ActiveSequenceEventData::AddRequest { .. } => Some(Some(OpenBooking {
                    worker: event.worker,
                    lora_name: event.lora_name.clone(),
                })),
                ActiveSequenceEventData::Free => Some(None),
                ActiveSequenceEventData::MarkPrefillCompleted => None,
            };
            let (router_id, request_id) = (event.router_id, event.request_id.clone());
            if let Err(error) = inbound.try_send(event) {
                self.counters.events_dropped.fetch_add(1, Relaxed);
                tracing::debug!(%partition, %error, "a replica event was dropped");
                continue;
            }
            // Only what the core received is tracked, so a peer's departure
            // frees exactly what it booked here.
            if let Some(peer) = peer {
                self.routers_of
                    .entry(peer.clone())
                    .or_default()
                    .insert(router_id);
            }
            match opened {
                Some(Some(booking)) => {
                    self.open
                        .entry(router_id)
                        .or_default()
                        .insert((partition, request_id), booking);
                }
                Some(None) => {
                    if let Some(mut bookings) = self.open.get_mut(&router_id) {
                        bookings.remove(&(partition, request_id));
                    }
                }
                None => {}
            }
        }
        for binding in bindings {
            if binding.writer_id == self.process_id {
                continue;
            }
            let binding = match from_wire_binding(binding) {
                Ok(binding) => binding,
                Err(error) => {
                    self.counters.bindings_rejected.fetch_add(1, Relaxed);
                    tracing::debug!(%error, "a replica binding did not convert");
                    continue;
                }
            };
            if let Err(error) = service.apply_session_binding(binding) {
                self.counters.bindings_rejected.fetch_add(1, Relaxed);
                tracing::debug!(%error, "a replica binding was not applied");
            }
        }
    }

    /// Publishes the core's session bindings as they change, until the
    /// updates end or the hub shuts down. Run it as a task.
    pub async fn publish_bindings(self: Arc<Self>, updates: SessionBindingUpdates) {
        loop {
            let bindings = tokio::select! {
                _ = self.cancel.cancelled() => return,
                bindings = updates.recv_many() => match bindings {
                    Some(bindings) => bindings,
                    None => return,
                },
            };
            self.publish(proto::ReplicaFrame {
                frame: Some(Frame::Events(proto::ReplicaEventBatch {
                    events: Vec::new(),
                    bindings: bindings.iter().map(to_wire_binding).collect(),
                })),
            });
        }
    }
}

/// Batches `partition`'s outbound events into frames: a frame closes at
/// [`MAX_REPLICA_BATCH_EVENTS`] events or [`MAX_REPLICA_BATCH_DURATION`]
/// after its first, as the core's own mesh batches. Ends when the channel
/// closes or the hub shuts down.
async fn publish_events(
    hub: Arc<ReplicaHub>,
    partition: RoutingPartitionId,
    mut outbound: mpsc::Receiver<ActiveSequenceEvent>,
) {
    let mut events = Vec::with_capacity(MAX_REPLICA_BATCH_EVENTS);
    loop {
        let received = tokio::select! {
            _ = hub.cancel.cancelled() => return,
            received = outbound.recv_many(&mut events, MAX_REPLICA_BATCH_EVENTS) => received,
        };
        if received == 0 {
            // The core dropped the partition: forget its inbound channel too,
            // unless a new partition of the same key has replaced it.
            hub.partitions
                .remove_if(&partition, |_, inbound| inbound.is_closed());
            return;
        }
        if events.len() < MAX_REPLICA_BATCH_EVENTS {
            let deadline = tokio::time::sleep(MAX_REPLICA_BATCH_DURATION);
            tokio::pin!(deadline);
            loop {
                let room = MAX_REPLICA_BATCH_EVENTS - events.len();
                let more = tokio::select! {
                    _ = &mut deadline => break,
                    more = outbound.recv_many(&mut events, room) => more,
                };
                if more == 0 || events.len() >= MAX_REPLICA_BATCH_EVENTS {
                    break;
                }
            }
        }
        hub.publish(proto::ReplicaFrame {
            frame: Some(Frame::Events(proto::ReplicaEventBatch {
                events: events
                    .drain(..)
                    .map(|event| to_wire_event(&partition, &event))
                    .collect(),
                bindings: Vec::new(),
            })),
        });
    }
}
