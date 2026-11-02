// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The transport's state in one replica.
//!
//! [`ReplicaHub`] publishes the core's session bindings, as the core's
//! [`SessionBindingUpdates`] reports them, on one broadcast every
//! subscriber reads, and applies its peers' bindings to the core's session
//! table through [`SelectionService::apply_session_binding`]. With
//! [`ReplicaHub::carry_in_flight`], it also publishes this replica's requests in
//! flight per pool on a timer and hands peers' to [`FrontendSignals`],
//! outside the core. Bookings are not replicated: no policy reads them.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Result, bail};
use dashmap::DashMap;
use dynamo_kv_router::services::selection::{SelectionService, SessionBindingUpdates};
use tokio::sync::{Notify, broadcast};
use tokio_util::sync::CancellationToken;
use url::Url;

use super::proto;
use super::proto::replica_frame::Frame;
use super::{from_wire_binding, to_wire_binding};
use crate::core::worker_id;
use crate::frontend_signals::FrontendSignals;

/// Frames a subscriber may fall behind before it loses the oldest.
const FRAME_BUFFER: usize = 1024;
/// How often a replica publishes its requests in flight per pool.
pub const IN_FLIGHT_INTERVAL: Duration = Duration::from_millis(100);
/// How long a peer's report counts without a newer one: five reports, so a
/// late frame or two does not drop the peer, and a peer that is gone stops
/// counting within half a second.
const IN_FLIGHT_FRESH: Duration = Duration::from_millis(500);

/// Replicate session bindings and requests in flight with other Global
/// Frontends over the gRPC `ReplicaSync` stream, served on this replica's
/// API port.
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
    /// Inbound bindings the core refused or that did not convert.
    pub bindings_rejected: u64,
    /// Frames subscribers missed for reading too slowly.
    pub subscribers_lagged: u64,
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
    bindings_rejected: AtomicU64,
    subscribers_lagged: AtomicU64,
}

/// One replica's end of the stream: frames out, peers' frames in, and the
/// peers it follows.
pub struct ReplicaHub {
    process_id: u64,
    /// Every frame this replica publishes; each subscriber reads a copy.
    frames: broadcast::Sender<proto::ReplicaFrame>,
    /// The peers to follow, each with the token that stops following it
    /// and the generation of that following.
    peers: DashMap<Url, Follow>,
    generations: AtomicU64,
    peers_changed: Notify,
    cancel: CancellationToken,
    counters: Counters,
    /// Where peers' requests in flight go, and this replica's come from,
    /// once [`Self::carry_in_flight`] is called.
    in_flight: OnceLock<Arc<FrontendSignals>>,
}

impl ReplicaHub {
    /// A hub whose tasks and streams end when `cancel` fires. Its process
    /// id, which stamps every frame this replica publishes, is random and
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
            frames,
            peers: DashMap::new(),
            generations: AtomicU64::new(0),
            peers_changed: Notify::new(),
            cancel,
            counters: Counters::default(),
            in_flight: OnceLock::new(),
        })
    }

    /// The id this replica stamps on its frames as `writer_id` and
    /// `replica_id`.
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
            bindings_rejected: c.bindings_rejected.load(Relaxed),
            subscribers_lagged: c.subscribers_lagged.load(Relaxed),
        }
    }

    /// Carries the frontends' counts: publishes this replica's requests in
    /// flight per pool, from `signals`, every [`IN_FLIGHT_INTERVAL`], and
    /// records each peer's in `signals`. Once per hub; a second call is ignored.
    pub fn carry_in_flight(self: &Arc<Self>, signals: Arc<FrontendSignals>) {
        if self.in_flight.set(signals).is_ok() {
            tokio::spawn(publish_in_flight(self.clone()));
        }
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

    /// Stops following `peer`. Its requests in flight stop counting once
    /// its last report goes stale. `false` if it was not a peer.
    pub fn remove_peer(&self, peer: &Url) -> bool {
        let Some((_, follow)) = self.peers.remove(peer) else {
            return false;
        };
        follow.cancel.cancel();
        self.peers_changed.notify_one();
        true
    }

    pub fn peers(&self) -> Vec<Url> {
        self.peers.iter().map(|peer| peer.key().clone()).collect()
    }

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

    /// Hands a frame to the core: each binding to the session table, each
    /// peer's requests in flight to [`FrontendSignals`]. This replica's own
    /// frames, echoed back, are skipped.
    pub fn apply(&self, frame: proto::ReplicaFrame, service: &SelectionService) {
        self.counters.frames_received.fetch_add(1, Relaxed);
        let bindings = match frame.frame {
            Some(Frame::Snapshot(snapshot)) => snapshot.bindings,
            Some(Frame::Bindings(batch)) => batch.bindings,
            Some(Frame::InflightCount(inflight)) => return self.apply_inflight(inflight),
            None => return,
        };
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

    /// [`Self::apply`] for a frame read from `peer`'s stream in following
    /// `generation`. A frame from a peer no longer followed, or from an
    /// earlier following of it, is ignored.
    pub(crate) fn apply_from(
        &self,
        peer: &Url,
        generation: u64,
        frame: proto::ReplicaFrame,
        service: &SelectionService,
    ) {
        let followed = self
            .peers
            .get(peer)
            .is_some_and(|followed| followed.generation == generation);
        if followed {
            self.apply(frame, service);
        }
    }

    /// This replica's requests in flight per pool now, as a frame, if this
    /// hub carries in-flight counts: what [`publish_in_flight`] sends every interval, and
    /// what a new subscriber gets right after the snapshot.
    pub(super) fn inflight_frame(&self) -> Option<proto::ReplicaFrame> {
        let signals = self.in_flight.get()?;
        let pools = signals
            .local()
            .into_iter()
            .map(|(pool, count)| proto::PoolInflightCount {
                worker_id: worker_id(&pool),
                count,
            })
            .collect();
        Some(proto::ReplicaFrame {
            frame: Some(Frame::InflightCount(proto::InflightCount {
                replica_id: self.process_id,
                pools,
            })),
        })
    }

    /// Records a peer's requests in flight per pool, if this hub carries
    /// in-flight counts. A replica's own counts, echoed back, are skipped.
    fn apply_inflight(&self, inflight: proto::InflightCount) {
        if inflight.replica_id == self.process_id {
            return;
        }
        let Some(signals) = self.in_flight.get() else {
            return;
        };
        let counts = inflight
            .pools
            .into_iter()
            .map(|pool| (pool.worker_id, pool.count))
            .collect::<HashMap<_, _>>();
        signals.record_peer(inflight.replica_id, counts, IN_FLIGHT_FRESH);
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
                frame: Some(Frame::Bindings(proto::BindingBatch {
                    bindings: bindings.iter().map(to_wire_binding).collect(),
                })),
            });
        }
    }
}

/// Publishes this replica's requests in flight per pool every
/// [`IN_FLIGHT_INTERVAL`] until the hub shuts down: one small frame per tick,
/// whatever the request rate.
async fn publish_in_flight(hub: Arc<ReplicaHub>) {
    let mut tick = tokio::time::interval(IN_FLIGHT_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = hub.cancel.cancelled() => return,
            _ = tick.tick() => {}
        }
        if let Some(frame) = hub.inflight_frame() {
            hub.publish(frame);
        }
    }
}
