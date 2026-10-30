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

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use dashmap::{DashMap, Entry};
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::multi_worker_sequence::{
    ReplicaRequestLeaseObserver, active_request_expiry_duration,
};
use dynamo_kv_router::protocols::{
    ActiveSequenceEvent, ActiveSequenceEventData, MAX_REPLICA_BATCH_DURATION,
    MAX_REPLICA_BATCH_EVENTS, WorkerWithDpRank,
};
use dynamo_kv_router::queue::SchedulerBookingDescriptor;
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
    /// Bookings peers have opened here and not yet freed, as tracked for
    /// [`ReplicaHub::remove_peer`].
    pub bookings_tracked: u64,
    /// Bookings freed on this replica's behalf of peers that left
    /// ([`ReplicaHub::remove_peer`]).
    pub bookings_released: u64,
}

/// A booking a peer's `AddRequest` opened here, as its `Free` would name
/// it, and whether the core has taken it (the lease observer says so).
#[derive(Clone)]
struct OpenBooking {
    partition: RoutingPartitionId,
    /// The stream it arrived on; `None` for a frame applied directly.
    peer: Option<Url>,
    router_id: u64,
    worker: WorkerWithDpRank,
    lora_name: Option<String>,
    /// When it was opened, or last given a second chance by the sweep.
    opened_at: Instant,
    admitted: bool,
    /// The core saw progress on it since `opened_at`.
    progressed: bool,
}

impl OpenBooking {
    fn free(&self, request_id: String) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id,
            worker: self.worker,
            data: ActiveSequenceEventData::Free,
            router_id: self.router_id,
            lora_name: self.lora_name.clone(),
        }
    }
}

/// One following of a peer. A peer removed and added back is a new
/// generation, so a frame the old stream still holds is not applied.
#[derive(Clone)]
pub(crate) struct Follow {
    pub(crate) generation: u64,
    pub(crate) cancel: CancellationToken,
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
    fn released(&self, why: &str, released: u64) {
        if released > 0 {
            self.bookings_released.fetch_add(released, Relaxed);
            tracing::info!(why, released, "freed mirrored bookings in the core");
        }
    }
}

/// One replica's end of the stream: the core's channels in, frames out, and
/// the peers it follows.
pub struct ReplicaHub {
    process_id: u64,
    /// Each partition's inbound channel, filled as the core creates them.
    partitions: DashMap<RoutingPartitionId, mpsc::Sender<ActiveSequenceEvent>>,
    /// Events a full partition has not taken yet, in order; while a
    /// partition has a backlog every new event joins it, so a free never
    /// overtakes its add.
    backlog: DashMap<RoutingPartitionId, VecDeque<ActiveSequenceEvent>>,
    me: Weak<ReplicaHub>,
    /// Every frame this replica publishes; each subscriber reads a copy.
    frames: broadcast::Sender<proto::ReplicaFrame>,
    /// The peers to follow, each with the token that stops following it
    /// and the generation of that following.
    peers: DashMap<Url, Follow>,
    generations: AtomicU64,
    peers_changed: Notify,
    /// The bookings peers' events have opened here and not yet freed, by
    /// request id. The hub holds the core's request leases for them: the
    /// core keeps a booking until it is freed, and the hub frees what a
    /// departed peer left ([`Self::remove_peer`]) or what no free ever
    /// reached ([`Self::sweep_at`]).
    open: DashMap<String, OpenBooking>,
    /// How long a mirrored booking may go without progress or its free
    /// before the hub frees it; the core's own expiry length, applied here
    /// as the embedded router applies it: a booking that progressed in the
    /// last interval gets one more.
    booking_expiry: Duration,
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
        let hub = Arc::new_cyclic(|me| Self {
            me: me.clone(),
            process_id,
            partitions: DashMap::new(),
            backlog: DashMap::new(),
            frames,
            peers: DashMap::new(),
            generations: AtomicU64::new(0),
            peers_changed: Notify::new(),
            open: DashMap::new(),
            booking_expiry: active_request_expiry_duration(),
            cancel,
            counters: Arc::default(),
        });
        tokio::spawn(sweep_bookings(Arc::downgrade(&hub)));
        hub
    }

    /// Frees, in the core, every mirrored booking that went `booking_expiry`
    /// without progress or its free, and forgets one the core never
    /// admitted (it dropped the event) after the same time. A booking that
    /// progressed in the interval starts another one instead. With the host
    /// holding the leases this is the only expiry a mirrored booking has; a
    /// local booking ends with its reservation.
    fn sweep_at(&self, now: Instant) {
        let mut stale = Vec::new();
        self.open.retain(|request_id, booking| {
            if now.saturating_duration_since(booking.opened_at) < self.booking_expiry {
                return true;
            }
            if booking.progressed {
                booking.progressed = false;
                booking.opened_at = now;
                return true;
            }
            if booking.admitted {
                stale.push((booking.partition.clone(), booking.free(request_id.clone())));
            }
            false
        });
        if !stale.is_empty() {
            self.free_in_core(stale, "swept");
        }
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
            bookings_tracked: self.open.len() as u64,
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
                entry.insert(Follow {
                    generation: self.generations.fetch_add(1, Relaxed),
                    cancel: self.cancel.child_token(),
                });
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
        follow.cancel.cancel();
        self.peers_changed.notify_one();
        self.release_bookings_of(peer);
        true
    }

    /// Frees, in the core, every booking still open that `peer`'s stream
    /// carried.
    fn release_bookings_of(&self, peer: &Url) {
        let mut frees = Vec::new();
        self.open.retain(|request_id, booking| {
            if booking.peer.as_ref() == Some(peer) {
                frees.push((booking.partition.clone(), booking.free(request_id.clone())));
                return false;
            }
            true
        });
        if !frees.is_empty() {
            self.free_in_core(frees, peer.as_str());
        }
    }

    /// Hands `frees` to their partitions, in order and never lost.
    fn free_in_core(&self, frees: Vec<(RoutingPartitionId, ActiveSequenceEvent)>, why: &str) {
        let mut released = 0u64;
        for (partition, free) in frees {
            if self.deliver(&partition, free) {
                released += 1;
            }
        }
        self.counters.released(why, released);
    }

    /// Hands `event` to its partition: straight to the channel while the
    /// partition has no backlog, else behind the backlog, which one task
    /// per partition drains as the channel takes it. `false` if the
    /// partition is unknown or gone.
    fn deliver(&self, partition: &RoutingPartitionId, event: ActiveSequenceEvent) -> bool {
        let Some(inbound) = self
            .partitions
            .get(partition)
            .map(|inbound| inbound.clone())
        else {
            self.counters.events_unroutable.fetch_add(1, Relaxed);
            return false;
        };
        match self.backlog.entry(partition.clone()) {
            Entry::Occupied(mut queued) => {
                queued.get_mut().push_back(event);
                true
            }
            Entry::Vacant(slot) => match inbound.try_send(event) {
                Ok(()) => true,
                Err(TrySendError::Full(event)) => {
                    slot.insert(VecDeque::from([event]));
                    tokio::spawn(drain_backlog(self.me.clone(), partition.clone(), inbound));
                    true
                }
                Err(TrySendError::Closed(_)) => {
                    self.counters.events_dropped.fetch_add(1, Relaxed);
                    tracing::debug!(%partition, "a replica event was dropped: the partition is gone");
                    false
                }
            },
        }
    }

    pub fn peers(&self) -> Vec<Url> {
        self.peers.iter().map(|peer| peer.key().clone()).collect()
    }

    /// Each peer with its current following.
    pub(crate) fn follows(&self) -> Vec<(Url, Follow)> {
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

    /// [`Self::apply`] for a frame read from `peer`'s stream in following
    /// `generation`, so the bookings it opens can be freed if the peer
    /// leaves. A frame from a peer no longer followed, or from an earlier
    /// following of it, is ignored: it cannot re-book what
    /// [`Self::remove_peer`] freed.
    pub(crate) fn apply_from(
        &self,
        peer: &Url,
        generation: u64,
        frame: proto::ReplicaFrame,
        service: &SelectionService,
    ) {
        // Held while the frame is applied, so `remove_peer` waits for it:
        // nothing lands after the peer's bookings are freed.
        let Some(followed) = self.peers.get(peer) else {
            return;
        };
        if followed.generation != generation {
            return;
        }
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
            if !self.partitions.contains_key(&partition) {
                self.counters.events_unroutable.fetch_add(1, Relaxed);
                tracing::debug!(%partition, "a replica event named a partition this core has not created");
                continue;
            }
            let request_id = event.request_id.clone();
            // An add is tracked before it is sent, so the core's `admitted`
            // finds it; a free is left to the core's `completed`, so one the
            // partition is slow to take is still remembered.
            let added = matches!(event.data, ActiveSequenceEventData::AddRequest { .. });
            if added {
                self.open.insert(
                    request_id.clone(),
                    OpenBooking {
                        partition: partition.clone(),
                        peer: peer.cloned(),
                        router_id: event.router_id,
                        worker: event.worker,
                        lora_name: event.lora_name.clone(),
                        opened_at: Instant::now(),
                        admitted: false,
                        progressed: false,
                    },
                );
            }
            if !self.deliver(&partition, event) && added {
                self.open.remove(&request_id);
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
/// Delivers a partition's backlog in order as its channel takes it, then
/// retires the backlog.
async fn drain_backlog(
    weak: Weak<ReplicaHub>,
    partition: RoutingPartitionId,
    inbound: mpsc::Sender<ActiveSequenceEvent>,
) {
    loop {
        let Some(hub) = weak.upgrade() else {
            return;
        };
        let next = match hub.backlog.entry(partition.clone()) {
            Entry::Occupied(mut queued) => match queued.get_mut().pop_front() {
                Some(event) => event,
                None => {
                    queued.remove();
                    return;
                }
            },
            Entry::Vacant(_) => return,
        };
        drop(hub);
        if inbound.send(next).await.is_err() {
            if let Some(hub) = weak.upgrade() {
                let dropped = hub
                    .backlog
                    .remove(&partition)
                    .map_or(0, |(_, queued)| queued.len() as u64 + 1);
                hub.counters.events_dropped.fetch_add(dropped, Relaxed);
            }
            return;
        }
    }
}

/// Sweeps mirrored bookings twice per expiry window while the hub lives.
async fn sweep_bookings(hub: Weak<ReplicaHub>) {
    let Some(expiry) = hub.upgrade().map(|hub| hub.booking_expiry) else {
        return;
    };
    let mut ticks = tokio::time::interval(expiry / 2);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ticks.tick().await;
    loop {
        ticks.tick().await;
        let Some(hub) = hub.upgrade() else {
            return;
        };
        if hub.cancel.is_cancelled() {
            return;
        }
        hub.sweep_at(Instant::now());
    }
}

/// The core's request leases for what peers mirror here: the core keeps
/// such a booking until it is freed, and tells the hub when it takes one
/// and when it lets one go.
impl ReplicaRequestLeaseObserver for ReplicaHub {
    fn admitted(&self, booking: SchedulerBookingDescriptor) {
        self.booking_admitted(&booking.request_id);
    }

    fn progressed(&self, booking: &SchedulerBookingDescriptor) {
        if let Some(mut open) = self.open.get_mut(&booking.request_id) {
            open.progressed = true;
        }
    }

    fn completed(&self, booking: &SchedulerBookingDescriptor) {
        self.booking_completed(&booking.request_id);
    }
}

impl ReplicaHub {
    /// The core took the mirrored booking `request_id`.
    fn booking_admitted(&self, request_id: &str) {
        if let Some(mut open) = self.open.get_mut(request_id) {
            open.admitted = true;
        }
    }

    /// The core let the mirrored booking `request_id` go.
    fn booking_completed(&self, request_id: &str) {
        self.open.remove(request_id);
    }

    /// The core saw progress on the mirrored booking `request_id`.
    #[cfg(test)]
    fn booking_progressed(&self, request_id: &str) {
        if let Some(mut open) = self.open.get_mut(request_id) {
            open.progressed = true;
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_sweep_frees_an_admitted_booking_whose_free_never_came() {
        let hub = ReplicaHub::new(CancellationToken::new());
        let partition = RoutingPartitionId::new("qwen", "global");
        let mut channels = (hub.factory())(&partition).unwrap();
        let start = Instant::now();
        let booking = |admitted: bool| OpenBooking {
            partition: partition.clone(),
            peer: None,
            router_id: 7,
            worker: WorkerWithDpRank::new(1, 0),
            lora_name: None,
            opened_at: start,
            admitted,
            progressed: false,
        };
        hub.open.insert("taken".to_owned(), booking(false));
        hub.open.insert("dropped".to_owned(), booking(false));
        hub.open.insert("busy".to_owned(), booking(false));
        hub.booking_admitted("taken");
        hub.booking_admitted("busy");
        hub.booking_progressed("busy");
        assert_eq!(hub.stats().bookings_tracked, 3);

        hub.sweep_at(start + hub.booking_expiry / 2);
        assert_eq!(hub.stats().bookings_tracked, 3);

        // Old enough: the admitted one is freed in the core, the one the
        // core never took is only forgotten, and the one that progressed
        // gets another interval.
        hub.sweep_at(start + hub.booking_expiry);
        assert_eq!(hub.stats().bookings_tracked, 1);
        let free = channels.inbound_rx.recv().await.unwrap();
        assert_eq!(free.request_id, "taken");
        assert!(matches!(free.data, ActiveSequenceEventData::Free));
        assert!(channels.inbound_rx.try_recv().is_err());
        assert_eq!(hub.stats().bookings_released, 1);

        // No further progress: the second interval ends it.
        hub.sweep_at(start + hub.booking_expiry * 2);
        assert_eq!(hub.stats().bookings_tracked, 0);
        assert_eq!(channels.inbound_rx.recv().await.unwrap().request_id, "busy");

        hub.open.insert("done".to_owned(), booking(true));
        hub.booking_completed("done");
        assert_eq!(hub.stats().bookings_tracked, 0);
    }

    #[tokio::test]
    async fn a_full_partition_takes_its_events_later_in_order() {
        let hub = ReplicaHub::new(CancellationToken::new());
        let partition = RoutingPartitionId::new("qwen", "global");
        let mut channels = (hub.factory())(&partition).unwrap();
        let event = |request: &str| ActiveSequenceEvent {
            request_id: request.to_owned(),
            worker: WorkerWithDpRank::new(1, 0),
            data: ActiveSequenceEventData::Free,
            router_id: 7,
            lora_name: None,
        };
        for i in 0..EVENT_CHANNEL_CAPACITY {
            assert!(hub.deliver(&partition, event(&format!("fill-{i}"))));
        }
        assert!(hub.deliver(&partition, event("late-1")));
        assert!(hub.deliver(&partition, event("late-2")));
        assert_eq!(
            hub.backlog.get(&partition).map(|queued| queued.len()),
            Some(2)
        );

        let mut seen = Vec::new();
        while seen.len() < EVENT_CHANNEL_CAPACITY + 2 {
            seen.push(channels.inbound_rx.recv().await.unwrap().request_id);
        }
        assert_eq!(&seen[EVENT_CHANNEL_CAPACITY..], ["late-1", "late-2"]);
        tokio::time::timeout(Duration::from_secs(2), async {
            while hub.backlog.contains_key(&partition) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the backlog retires once drained");
    }
}
