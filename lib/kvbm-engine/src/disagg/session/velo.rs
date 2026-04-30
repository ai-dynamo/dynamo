// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production [`Session`] / [`SessionFactory`] impl backed by velo.
//!
//! One bidirectional `Frame` stream per session (vs. the old
//! `DisaggSession`'s two enum types). The session's monitor task
//! reads inbound frames and demuxes them into:
//!
//! - `mpsc::UnboundedSender<CommitDelta>` for [`Session::commits`]
//! - `mpsc::UnboundedSender<AvailabilityDelta>` for [`Session::availability`]
//! - `mpsc::UnboundedSender<LifecycleEvent>` for [`Session::lifecycle`]
//! - a `DashMap<u64, oneshot>` for in-flight `pull` correlation
//!
//! Each of the three streams is single-consumer (panic on second
//! subscribe) with replay-on-first-subscribe semantics: prior
//! `Commit` frames coalesce into one `Added` delta, prior
//! `Available` frames coalesce into one `Available` delta.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use anyhow::{Context as _, Result, anyhow};
use dashmap::DashMap;
use futures::Stream;
use futures::future::BoxFuture;
use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::blocks::{ImmutableBlock, MutableBlock};
use parking_lot::Mutex;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};

use serde::{Deserialize, Serialize};

use super::{
    AvailabilityDelta, AvailabilityStream, CommitDelta, CommitStream, CommittedBlock, Frame,
    LifecycleEvent, LifecycleStream, Session, SessionFactory, SessionId,
};
use crate::disagg::{RemoteBlockRef, RemoteBlockSet, SessionEndpoint};
use crate::leader::InstanceLeader;
use crate::{BlockId, G2, InstanceId, SequenceHash};

/// Endpoint kind tag for the new symmetric-session wire format.
/// Distinct from the legacy `kvbm_conditional_disagg_v1` so the
/// two impls cannot accidentally interop.
pub const SESSION_STREAM_SCHEMA: &str = "kvbm_cd_session_v2";

/// Wire-level wrapper assigning a monotonically increasing
/// per-direction sequence number to each `Frame`. velo's
/// active-message dispatcher spawns a fresh tokio task per
/// inbound message (`InlineDispatcher::dispatch`); those tasks
/// race for the inbound flume channel, so AM bytes can be
/// reordered relative to the sender's call order. The receive
/// side reorders by `seq` before invoking the per-variant
/// dispatch, restoring the FIFO guarantee the protocol depends
/// on (e.g. `Available` must precede `Drained`, `CommitsClosed`
/// must follow all `Commit` frames).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SeqFrame {
    seq: u64,
    frame: Frame,
}

// ============================================================================
// Replay-buffered single-consumer stream
// ============================================================================

/// Holds pre-subscribe items in a buffer until first subscribe,
/// then switches to live mpsc forwarding. Subscribing twice
/// panics.
struct ReplayStream<T> {
    state: Mutex<ReplayState<T>>,
}

enum ReplayState<T> {
    NotSubscribed(Vec<T>),
    Subscribed(mpsc::UnboundedSender<T>),
}

impl<T> ReplayStream<T> {
    fn new() -> Self {
        Self {
            state: Mutex::new(ReplayState::NotSubscribed(Vec::new())),
        }
    }

    /// Push an item. Buffers if not yet subscribed; sends via
    /// mpsc otherwise.
    fn push(&self, item: T) {
        let mut state = self.state.lock();
        match &mut *state {
            ReplayState::NotSubscribed(buf) => buf.push(item),
            ReplayState::Subscribed(tx) => {
                let _ = tx.send(item);
            }
        }
    }

    /// Subscribe (transition to live mode), returning the
    /// receiver and the pre-subscribe buffer.
    ///
    /// Panics if called twice.
    fn subscribe(&self) -> (mpsc::UnboundedReceiver<T>, Vec<T>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut state = self.state.lock();
        let buffered = match std::mem::replace(&mut *state, ReplayState::Subscribed(tx)) {
            ReplayState::NotSubscribed(buf) => buf,
            ReplayState::Subscribed(_) => panic!("ReplayStream::subscribe called twice"),
        };
        (rx, buffered)
    }
}

// ============================================================================
// Inner state shared between the session, its monitor, and futures
// ============================================================================

/// Command queue for the per-session outbound sender task.
///
/// Public session methods enqueue these; a single sender task
/// drains the mpsc in receive order and forwards to the velo
/// outbound `StreamSender`, guaranteeing FIFO frame ordering on
/// the wire (no spawn race between `commit` /
/// `make_available` / `finish_*`).
enum OutboundCommand {
    Send(Frame),
    Finalize,
}

/// Receive-side reorder state. Frames arrive over velo's
/// AM-dispatched inbound flume in non-deterministic order
/// despite single-task FIFO sends; this small buffer holds
/// future seqs until their predecessors arrive, then drains
/// the contiguous prefix.
struct RecvState {
    buffer: BTreeMap<u64, Frame>,
    next_seq: u64,
}

struct VeloSessionInner {
    session_id: SessionId,
    velo: Arc<velo::Velo>,
    leader: Arc<InstanceLeader>,
    /// Endpoint we advertised — peers attach to this. None on
    /// puller-side after we ourselves attached to a peer.
    local_endpoint: Mutex<Option<SessionEndpoint>>,
    /// Synchronous enqueue side of the outbound queue. The single
    /// sender task drains the matching receiver in order.
    outbound_tx: mpsc::UnboundedSender<OutboundCommand>,
    /// One-shot for installing the velo outbound `StreamSender`
    /// once it exists. Holder side: installed on inbound
    /// `Frame::Attach`. Puller side: installed inline by the
    /// factory before the sender task starts. Taken exactly once.
    outbound_install_tx: Mutex<Option<oneshot::Sender<velo::StreamSender<SeqFrame>>>>,
    /// Monotonic outbound sequence counter assigned by the
    /// sender task as it drains the mpsc, so wire seqs match
    /// drain order. Receivers buffer + reorder by seq.
    outbound_seq: AtomicU64,
    /// Receive-side reorder buffer + expected seq. Velo's AM
    /// dispatcher races concurrent inbound tasks; the monitor
    /// inserts arriving `SeqFrame`s here and drains in seq order.
    recv_state: Mutex<RecvState>,

    /// Peer's identity, set when we receive `Frame::Attach`
    /// (holder side) or pre-known via the `attach` path
    /// (puller side, learned from the `Attach` frame the peer
    /// will send back? — actually the puller side learns from
    /// the inbound `Attach` only if the peer is also running
    /// this impl. For the symmetric attach, only the puller
    /// sends Attach; the holder doesn't reciprocate. Keep it
    /// `None` on the puller side until we extend the protocol.)
    peer_instance_id: Mutex<Option<InstanceId>>,

    // Local state vectors
    committed: Mutex<BTreeSet<SequenceHash>>,
    available_pins: Mutex<BTreeMap<SequenceHash, ImmutableBlock<G2>>>,

    // Peer state vectors (replicated from inbound frames)
    peer_committed: Mutex<BTreeSet<SequenceHash>>,
    peer_available: Mutex<BTreeMap<SequenceHash, BlockId>>,

    // Pending pulls keyed by pull_id. The oneshot resolves on
    // inbound `PullComplete`.
    pending_pulls: DashMap<u64, oneshot::Sender<()>>,
    /// Inbound `Pull` frames recorded so we can drop the
    /// matching pins on `PullAck`.
    inbound_pulls: DashMap<u64, Vec<SequenceHash>>,
    /// Counter for new pull_ids on this side.
    next_pull_id: AtomicU64,

    // Stream replay buffers
    commit_stream: ReplayStream<CommitDelta>,
    avail_stream: ReplayStream<AvailabilityDelta>,
    lifecycle_stream: ReplayStream<LifecycleEvent>,

    /// `finish_commits` was called locally — we send
    /// `CommitsClosed` exactly once.
    commits_closed: Mutex<bool>,
    avail_drained: Mutex<bool>,

    /// Set when monitor detects shutdown, used to short-circuit.
    closed: Mutex<bool>,

    /// Owned tokio `Handle` for spawning the outbound sender task
    /// and dispatch-side helpers.
    ///
    /// Sync trait methods (`commit`, `make_available`, `finish_*`,
    /// `close`) may be invoked from a thread that has no current
    /// tokio runtime — e.g. vLLM's scheduler calling into the
    /// connector via PyO3. Sync methods enqueue synchronously and
    /// do not need this handle, but the dispatch path (Attach
    /// install, etc.) does.
    runtime: Handle,
}

// ============================================================================
// VeloSession — public type
// ============================================================================

/// Production session. Both holder and puller sides are the
/// same type — symmetry is the point.
#[derive(Clone)]
pub struct VeloSession {
    inner: Arc<VeloSessionInner>,
}

impl std::fmt::Debug for VeloSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VeloSession")
            .field("session_id", &self.inner.session_id)
            .field("commits_closed", &*self.inner.commits_closed.lock())
            .field("avail_drained", &*self.inner.avail_drained.lock())
            .finish()
    }
}

impl VeloSession {
    /// Build inner state plus the outbound queue. The caller must
    /// also call [`spawn_outbound_sender`] with `outbound_rx` and
    /// `install_rx` to start draining; the install half of the
    /// oneshot stays inside the inner so the holder-side dispatch
    /// path can install it after `Frame::Attach` arrives.
    fn new_inner(
        session_id: SessionId,
        velo: Arc<velo::Velo>,
        leader: Arc<InstanceLeader>,
        local_endpoint: Option<SessionEndpoint>,
        runtime: Handle,
        outbound_tx: mpsc::UnboundedSender<OutboundCommand>,
        outbound_install_tx: oneshot::Sender<velo::StreamSender<SeqFrame>>,
    ) -> Arc<VeloSessionInner> {
        Arc::new(VeloSessionInner {
            session_id,
            velo,
            leader,
            local_endpoint: Mutex::new(local_endpoint),
            outbound_tx,
            outbound_install_tx: Mutex::new(Some(outbound_install_tx)),
            outbound_seq: AtomicU64::new(0),
            recv_state: Mutex::new(RecvState {
                buffer: BTreeMap::new(),
                next_seq: 0,
            }),
            peer_instance_id: Mutex::new(None),
            committed: Mutex::new(BTreeSet::new()),
            available_pins: Mutex::new(BTreeMap::new()),
            peer_committed: Mutex::new(BTreeSet::new()),
            peer_available: Mutex::new(BTreeMap::new()),
            pending_pulls: DashMap::new(),
            inbound_pulls: DashMap::new(),
            next_pull_id: AtomicU64::new(1),
            commit_stream: ReplayStream::new(),
            avail_stream: ReplayStream::new(),
            lifecycle_stream: ReplayStream::new(),
            commits_closed: Mutex::new(false),
            avail_drained: Mutex::new(false),
            closed: Mutex::new(false),
            runtime,
        })
    }

    /// Synchronously enqueue a frame for outbound dispatch.
    /// Returns Err only when the sender task has already
    /// terminated (session closed / failed).
    fn enqueue_frame(&self, frame: Frame) -> Result<()> {
        self.inner
            .outbound_tx
            .send(OutboundCommand::Send(frame))
            .map_err(|_| anyhow!("session outbound channel closed"))
    }

    /// Synchronously enqueue stream finalization. After draining
    /// any frames already in the queue, the sender task calls
    /// `finalize()` on the velo `StreamSender` and exits.
    fn enqueue_finalize(&self) -> Result<()> {
        self.inner
            .outbound_tx
            .send(OutboundCommand::Finalize)
            .map_err(|_| anyhow!("session outbound channel closed"))
    }
}

/// Single per-session sender task. Awaits installation of the
/// velo outbound `StreamSender`, then drains the outbound mpsc
/// in receive order and forwards each frame. This is the only
/// task that ever calls `sender.send(frame).await`, eliminating
/// the spawn race that previously reordered Commit/Available
/// vs. Drained.
fn spawn_outbound_sender(
    mut rx: mpsc::UnboundedReceiver<OutboundCommand>,
    install_rx: oneshot::Receiver<velo::StreamSender<SeqFrame>>,
    inner: Arc<VeloSessionInner>,
    runtime: &Handle,
) {
    let session_id = inner.session_id;
    runtime.spawn(async move {
        let sender = match install_rx.await {
            Ok(s) => s,
            Err(_) => {
                // Inner dropped before outbound was installed;
                // any queued frames go nowhere.
                return;
            }
        };
        while let Some(cmd) = rx.recv().await {
            match cmd {
                OutboundCommand::Send(frame) => {
                    let seq = inner.outbound_seq.fetch_add(1, Ordering::Relaxed);
                    let kind = frame_kind(&frame);
                    crate::engine_audit!(
                        "session_outbound_send",
                        session_id = %session_id,
                        seq,
                        kind = kind
                    );
                    if let Err(err) = sender.send(SeqFrame { seq, frame }).await {
                        tracing::error!(error = %err, "velo outbound send failed");
                        break;
                    }
                }
                OutboundCommand::Finalize => {
                    crate::engine_audit!(
                        "session_outbound_finalize",
                        session_id = %session_id
                    );
                    let _ = sender.finalize();
                    break;
                }
            }
        }
    });
}

fn frame_kind(frame: &Frame) -> &'static str {
    match frame {
        Frame::Attach { .. } => "Attach",
        Frame::Commit { .. } => "Commit",
        Frame::CommitsClosed => "CommitsClosed",
        Frame::Available { .. } => "Available",
        Frame::Drained => "Drained",
        Frame::Pull { .. } => "Pull",
        Frame::PullComplete { .. } => "PullComplete",
        Frame::PullAck { .. } => "PullAck",
        Frame::Detach => "Detach",
        Frame::Error { .. } => "Error",
    }
}

// ============================================================================
// Frame handling — runs in the per-session monitor task
// ============================================================================

/// Wire-side dispatch: insert into the seq buffer and drain
/// the contiguous prefix in seq order. Velo's AM dispatcher
/// races concurrent inbound tasks, so seqs may arrive out of
/// order; this function restores the sender's call order
/// before invoking [`dispatch_frame_inner`] for each variant.
fn dispatch_seq_frame(inner: &Arc<VeloSessionInner>, seq_frame: SeqFrame, runtime: &Handle) {
    let session_id = inner.session_id;
    let to_dispatch: Vec<Frame> = {
        let mut state = inner.recv_state.lock();
        if seq_frame.seq < state.next_seq {
            tracing::warn!(
                %session_id,
                seq = seq_frame.seq,
                expected = state.next_seq,
                "duplicate or stale seq dropped"
            );
            return;
        }
        state.buffer.insert(seq_frame.seq, seq_frame.frame);
        let mut ready = Vec::new();
        loop {
            let next = state.next_seq;
            match state.buffer.remove(&next) {
                Some(frame) => {
                    ready.push(frame);
                    state.next_seq += 1;
                }
                None => break,
            }
        }
        ready
    };
    for frame in to_dispatch {
        dispatch_frame_inner(inner, frame, runtime);
    }
}

fn dispatch_frame_inner(inner: &Arc<VeloSessionInner>, frame: Frame, runtime: &Handle) {
    let session_id = inner.session_id;
    match frame {
        Frame::Attach {
            instance_id,
            endpoint,
        } => {
            crate::engine_audit!(
                "session_recv_attach",
                session_id = %session_id,
                peer_instance_id = %instance_id
            );
            *inner.peer_instance_id.lock() = Some(instance_id);
            // Holder side: open the outbound velo sender, deliver
            // it to the per-session sender task via the install
            // oneshot, ensure peer's worker metadata is imported,
            // then push Attached. Caller can rely on "Attached
            // means ready to handle inbound Pull": the underlying
            // RDMA call (pull_remote_block_sets) requires
            // metadata, and we pay the roundtrip eagerly here so
            // it's hot when the first Pull frame arrives.
            let inner_for_attach = Arc::clone(inner);
            let endpoint_clone = endpoint.clone();
            runtime.spawn(async move {
                let handle = match handle_from_endpoint(&endpoint_clone) {
                    Ok(h) => h,
                    Err(err) => {
                        tracing::error!(error = %err, "decode Attach endpoint failed");
                        inner_for_attach
                            .lifecycle_stream
                            .push(LifecycleEvent::Failed {
                                reason: format!("decode Attach endpoint: {err}"),
                            });
                        return;
                    }
                };
                let sender = match inner_for_attach.velo.attach_anchor::<SeqFrame>(handle).await {
                    Ok(s) => s,
                    Err(err) => {
                        tracing::error!(error = %err, "attach outbound on Attach failed");
                        inner_for_attach
                            .lifecycle_stream
                            .push(LifecycleEvent::Failed {
                                reason: format!("install outbound on Attach: {err}"),
                            });
                        return;
                    }
                };
                let install_tx = inner_for_attach.outbound_install_tx.lock().take();
                match install_tx {
                    Some(tx) => {
                        if tx.send(sender).is_err() {
                            tracing::warn!(
                                "outbound sender task gone before install on Frame::Attach"
                            );
                            return;
                        }
                    }
                    None => {
                        tracing::warn!(
                            "outbound already installed before Frame::Attach (duplicate Attach?)"
                        );
                        return;
                    }
                }
                // Skipped when this side has no workers — see
                // matching comment in `attach()`. Stream-only
                // callers (no pull) remain usable.
                if inner_for_attach.leader.worker_count() > 0 {
                    if let Err(err) = inner_for_attach
                        .leader
                        .ensure_remote_metadata(instance_id)
                        .await
                    {
                        tracing::error!(error = %err, peer = %instance_id, "metadata exchange failed on Attach");
                        inner_for_attach
                            .lifecycle_stream
                            .push(LifecycleEvent::Failed {
                                reason: format!(
                                    "metadata exchange failed for {instance_id}: {err}"
                                ),
                            });
                        return;
                    }
                }
                inner_for_attach
                    .lifecycle_stream
                    .push(LifecycleEvent::Attached {
                        peer_instance_id: instance_id,
                    });
            });
        }
        Frame::Commit { hashes } => {
            crate::engine_audit!(
                "session_recv_commit",
                session_id = %session_id,
                num_hashes = hashes.len()
            );
            {
                let mut peer_committed = inner.peer_committed.lock();
                peer_committed.extend(hashes.iter().copied());
            }
            inner.commit_stream.push(CommitDelta::Added(hashes));
        }
        Frame::CommitsClosed => {
            crate::engine_audit!(
                "session_recv_commits_closed",
                session_id = %session_id
            );
            inner.commit_stream.push(CommitDelta::Closed);
        }
        Frame::Available { blocks } => {
            crate::engine_audit!(
                "session_recv_available",
                session_id = %session_id,
                num_blocks = blocks.len()
            );
            {
                let mut peer_available = inner.peer_available.lock();
                for b in &blocks {
                    peer_available.insert(b.hash, b.peer_block_id);
                }
            }
            inner
                .avail_stream
                .push(AvailabilityDelta::Available(blocks));
        }
        Frame::Drained => {
            crate::engine_audit!(
                "session_recv_drained",
                session_id = %session_id
            );
            inner.avail_stream.push(AvailabilityDelta::Drained);
        }
        Frame::Pull { pull_id, hashes } => {
            crate::engine_audit!(
                "session_recv_pull",
                session_id = %session_id,
                pull_id,
                num_hashes = hashes.len()
            );
            // We are holder. Authorize the puller's RDMA read,
            // remember the hashes so we can drop pins on PullAck.
            inner.inbound_pulls.insert(pull_id, hashes);
            // Synchronous enqueue — preserves causal order with
            // any concurrent Commit/Available emitted by the
            // holder side.
            let session = VeloSession {
                inner: Arc::clone(inner),
            };
            if let Err(err) = session.enqueue_frame(Frame::PullComplete { pull_id }) {
                tracing::error!(error = %err, pull_id, "enqueue PullComplete failed");
            }
        }
        Frame::PullComplete { pull_id } => {
            crate::engine_audit!(
                "session_recv_pull_complete",
                session_id = %session_id,
                pull_id
            );
            // We are puller. Resolve the matching oneshot so the
            // pull future proceeds to do the RDMA read.
            if let Some((_, tx)) = inner.pending_pulls.remove(&pull_id) {
                let _ = tx.send(());
            } else {
                tracing::warn!(pull_id, "PullComplete with no pending pull");
            }
        }
        Frame::PullAck { pull_id } => {
            crate::engine_audit!(
                "session_recv_pull_ack",
                session_id = %session_id,
                pull_id
            );
            // We are holder. Puller confirmed RDMA read settled;
            // drop pins for the hashes correlated with this pull.
            if let Some((_, hashes)) = inner.inbound_pulls.remove(&pull_id) {
                let mut pins = inner.available_pins.lock();
                for h in &hashes {
                    pins.remove(h);
                }
            }
        }
        Frame::Detach => {
            crate::engine_audit!(
                "session_recv_detach",
                session_id = %session_id
            );
            inner.lifecycle_stream.push(LifecycleEvent::Detached {
                reason: Some("peer detached".to_string()),
            });
        }
        Frame::Error { message } => {
            inner
                .lifecycle_stream
                .push(LifecycleEvent::Failed { reason: message });
        }
    }
}

impl VeloSession {
    /// Test-only: route an inbound `Frame` directly through the
    /// session's monitor dispatch, bypassing the wire. Lets unit
    /// tests assert on dispatch_frame's per-variant side effects
    /// (state-vector mutation, pin release on PullAck, lifecycle
    /// emission, etc.) without standing up a paired velo
    /// connection.
    #[cfg(any(test, feature = "testing"))]
    pub fn test_inject_inbound_frame(&self, frame: Frame) {
        dispatch_frame_inner(&self.inner, frame, &self.inner.runtime);
    }

    /// Test-only: count of pins currently held in
    /// `available_pins` (the set holder drops on PullAck).
    #[cfg(any(test, feature = "testing"))]
    pub fn test_available_pin_count(&self) -> usize {
        self.inner.available_pins.lock().len()
    }
}

fn spawn_monitor(
    inner: Arc<VeloSessionInner>,
    mut anchor: velo::StreamAnchor<SeqFrame>,
    runtime: Handle,
) {
    runtime.clone().spawn(async move {
        use futures::StreamExt;
        while let Some(frame) = anchor.next().await {
            match frame {
                Ok(velo::StreamFrame::Item(seq_frame)) => {
                    dispatch_seq_frame(&inner, seq_frame, &runtime)
                }
                Ok(velo::StreamFrame::Finalized) => {
                    inner
                        .lifecycle_stream
                        .push(LifecycleEvent::Detached { reason: None });
                    break;
                }
                Ok(velo::StreamFrame::Detached) => {
                    inner.lifecycle_stream.push(LifecycleEvent::Detached {
                        reason: Some("stream detached".to_string()),
                    });
                    break;
                }
                Ok(_) => {}
                Err(err) => {
                    inner.lifecycle_stream.push(LifecycleEvent::Failed {
                        reason: format!("stream error: {err}"),
                    });
                    break;
                }
            }
        }
        *inner.closed.lock() = true;
    });
}

// ============================================================================
// Stream wrappers — combine pre-subscribe replay with live mpsc
// ============================================================================

/// Adapter that yields a (combined) replay item first if any,
/// then forwards the live mpsc receiver.
struct CombiningStream<T> {
    /// Items to yield before draining the receiver. For
    /// commits/availability this is at most one element (the
    /// coalesced replay) plus optionally a `Closed`/`Drained`
    /// terminator.
    pending: VecDeque<T>,
    rx: mpsc::UnboundedReceiver<T>,
}

impl<T: Unpin> Stream for CombiningStream<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(item) = self.pending.pop_front() {
            return Poll::Ready(Some(item));
        }
        self.rx.poll_recv(cx)
    }
}

fn build_commit_stream(
    rx: mpsc::UnboundedReceiver<CommitDelta>,
    replay: Vec<CommitDelta>,
) -> CommitStream {
    let mut pending: VecDeque<CommitDelta> = VecDeque::new();
    // Coalesce all `Added` deltas in the replay into one, then
    // optionally append a `Closed`.
    let mut coalesced: Vec<SequenceHash> = Vec::new();
    let mut saw_closed = false;
    for d in replay {
        match d {
            CommitDelta::Added(hs) => coalesced.extend(hs),
            CommitDelta::Closed => {
                saw_closed = true;
            }
        }
    }
    if !coalesced.is_empty() {
        pending.push_back(CommitDelta::Added(coalesced));
    }
    if saw_closed {
        pending.push_back(CommitDelta::Closed);
    }
    Box::pin(CombiningStream { pending, rx })
}

fn build_avail_stream(
    rx: mpsc::UnboundedReceiver<AvailabilityDelta>,
    replay: Vec<AvailabilityDelta>,
) -> AvailabilityStream {
    let mut pending: VecDeque<AvailabilityDelta> = VecDeque::new();
    let mut coalesced: Vec<CommittedBlock> = Vec::new();
    let mut saw_drained = false;
    for d in replay {
        match d {
            AvailabilityDelta::Available(bs) => coalesced.extend(bs),
            AvailabilityDelta::Drained => {
                saw_drained = true;
            }
        }
    }
    if !coalesced.is_empty() {
        pending.push_back(AvailabilityDelta::Available(coalesced));
    }
    if saw_drained {
        pending.push_back(AvailabilityDelta::Drained);
    }
    Box::pin(CombiningStream { pending, rx })
}

fn build_lifecycle_stream(
    rx: mpsc::UnboundedReceiver<LifecycleEvent>,
    replay: Vec<LifecycleEvent>,
) -> LifecycleStream {
    let mut pending: VecDeque<LifecycleEvent> = VecDeque::new();
    pending.extend(replay);
    Box::pin(CombiningStream { pending, rx })
}

// ============================================================================
// Session trait impl
// ============================================================================

impl Session for VeloSession {
    fn session_id(&self) -> SessionId {
        self.inner.session_id
    }

    fn endpoint(&self) -> Option<SessionEndpoint> {
        self.inner.local_endpoint.lock().clone()
    }

    fn commit(&self, hashes: Vec<SequenceHash>) -> Result<()> {
        crate::engine_audit!(
            "session_commit",
            session_id = %self.inner.session_id,
            num_hashes = hashes.len()
        );
        if hashes.is_empty() {
            return Ok(());
        }
        {
            let mut committed = self.inner.committed.lock();
            committed.extend(hashes.iter().copied());
        }
        // Synchronous FIFO enqueue — frames go out in the order
        // their public methods are called, with no spawn race.
        if let Err(err) = self.enqueue_frame(Frame::Commit { hashes }) {
            tracing::error!(error = %err, "enqueue Commit failed");
        }
        Ok(())
    }

    fn finish_commits(&self) -> Result<()> {
        crate::engine_audit!(
            "session_finish_commits",
            session_id = %self.inner.session_id
        );
        {
            let mut closed = self.inner.commits_closed.lock();
            if *closed {
                return Ok(());
            }
            *closed = true;
        }
        if let Err(err) = self.enqueue_frame(Frame::CommitsClosed) {
            tracing::error!(error = %err, "enqueue CommitsClosed failed");
        }
        Ok(())
    }

    fn make_available(&self, blocks: Vec<ImmutableBlock<G2>>) -> Result<()> {
        crate::engine_audit!(
            "session_make_available",
            session_id = %self.inner.session_id,
            num_blocks = blocks.len()
        );
        if blocks.is_empty() {
            return Ok(());
        }
        // Validate every block.hash ∈ committed.
        {
            let committed = self.inner.committed.lock();
            for b in &blocks {
                let h = b.sequence_hash();
                if !committed.contains(&h) {
                    anyhow::bail!("make_available: block hash {:?} is not in committed set", h);
                }
            }
        }

        // Pin the blocks and build the wire payload.
        let payload: Vec<CommittedBlock> = blocks
            .iter()
            .map(|b| CommittedBlock {
                hash: b.sequence_hash(),
                peer_block_id: b.block_id(),
            })
            .collect();
        {
            let mut pins = self.inner.available_pins.lock();
            for b in blocks {
                let h = b.sequence_hash();
                pins.insert(h, b);
            }
        }

        if let Err(err) = self.enqueue_frame(Frame::Available { blocks: payload }) {
            tracing::error!(error = %err, "enqueue Available failed");
        }
        Ok(())
    }

    fn finish_availability(&self) -> Result<()> {
        crate::engine_audit!(
            "session_finish_availability",
            session_id = %self.inner.session_id
        );
        {
            let mut drained = self.inner.avail_drained.lock();
            if *drained {
                return Ok(());
            }
            *drained = true;
        }
        if let Err(err) = self.enqueue_frame(Frame::Drained) {
            tracing::error!(error = %err, "enqueue Drained failed");
        }
        Ok(())
    }

    fn commits(&self) -> CommitStream {
        let (rx, replay) = self.inner.commit_stream.subscribe();
        build_commit_stream(rx, replay)
    }

    fn availability(&self) -> AvailabilityStream {
        let (rx, replay) = self.inner.avail_stream.subscribe();
        build_avail_stream(rx, replay)
    }

    fn peer_committed(&self) -> Vec<SequenceHash> {
        self.inner.peer_committed.lock().iter().copied().collect()
    }

    fn peer_available(&self) -> Vec<CommittedBlock> {
        self.inner
            .peer_available
            .lock()
            .iter()
            .map(|(h, id)| CommittedBlock {
                hash: *h,
                peer_block_id: *id,
            })
            .collect()
    }

    fn pull(
        &self,
        hashes: Vec<SequenceHash>,
        dst: Vec<MutableBlock<G2>>,
    ) -> BoxFuture<'static, Result<Vec<MutableBlock<G2>>>> {
        let session = self.clone();
        crate::engine_audit!(
            "session_pull_request",
            session_id = %self.inner.session_id,
            num_hashes = hashes.len(),
            num_dst = dst.len()
        );
        Box::pin(async move {
            // Validate inputs.
            if hashes.len() != dst.len() {
                anyhow::bail!(
                    "pull: hashes.len() {} != dst.len() {}",
                    hashes.len(),
                    dst.len()
                );
            }
            if hashes.is_empty() {
                return Ok(dst);
            }

            // Resolve peer_block_id per hash from peer_available.
            let peer_block_ids: Vec<BlockId> = {
                let peer_avail = session.inner.peer_available.lock();
                let mut out = Vec::with_capacity(hashes.len());
                for h in &hashes {
                    let id = peer_avail
                        .get(h)
                        .copied()
                        .ok_or_else(|| anyhow!("pull: hash {:?} not in peer_available", h))?;
                    out.push(id);
                }
                out
            };

            // Peer instance id (set when we received Attach, or
            // — if we are puller — set by the Attach we sent).
            // For this design, the puller sends Attach. So the
            // holder side has peer_instance_id; the puller side
            // does NOT have one (the holder hasn't told us its
            // identity over the wire). The puller side falls
            // back to the velo instance_id baked into the peer
            // endpoint we attached to.
            //
            // Fix: extract the peer's velo instance_id from the
            // outbound StreamSender's metadata. For now, we
            // require the caller to have set peer_instance_id
            // out-of-band before pull() — the symmetric trait
            // doesn't expose this yet, so we read from the
            // peer_endpoint we held.
            let peer_instance_id = {
                let stored = session.inner.peer_instance_id.lock();
                stored.ok_or_else(|| {
                    anyhow!(
                        "pull: peer_instance_id unknown — \
                             holder side requires `Attach` frame to have arrived"
                    )
                })?
            };

            // Allocate pull_id and install oneshot.
            let pull_id = session.inner.next_pull_id.fetch_add(1, Ordering::Relaxed);
            let (tx, rx) = oneshot::channel();
            session.inner.pending_pulls.insert(pull_id, tx);

            // Send the Pull frame and await PullComplete.
            crate::engine_audit!(
                "session_pull_send",
                session_id = %session.inner.session_id,
                pull_id,
                num_hashes = hashes.len(),
                peer_instance_id = %peer_instance_id
            );
            session.enqueue_frame(Frame::Pull {
                pull_id,
                hashes: hashes.clone(),
            })?;
            rx.await.map_err(|_| {
                anyhow!(
                    "pull {}: PullComplete oneshot dropped (session closed)",
                    pull_id
                )
            })?;
            crate::engine_audit!(
                "session_pull_complete_received",
                session_id = %session.inner.session_id,
                pull_id
            );

            // Now drive the RDMA read. Build RemoteBlockSet.
            let block_set = RemoteBlockSet {
                source_layout: LogicalLayoutHandle::G2,
                blocks: hashes
                    .iter()
                    .zip(peer_block_ids.iter())
                    .map(|(h, id)| RemoteBlockRef {
                        block_id: *id,
                        sequence_hash: *h,
                    })
                    .collect(),
            };
            let dst_block_ids: Vec<BlockId> = dst.iter().map(|b| b.block_id()).collect();
            crate::engine_audit!(
                "session_pull_rdma_start",
                session_id = %session.inner.session_id,
                pull_id,
                num_blocks = dst_block_ids.len()
            );
            let notification = session
                .inner
                .leader
                .pull_remote_block_sets(peer_instance_id, &[block_set], &dst_block_ids)
                .await
                .context("pull_remote_block_sets enqueue")?;
            notification
                .await
                .context("pull_remote_block_sets notification")?;
            crate::engine_audit!(
                "session_pull_rdma_done",
                session_id = %session.inner.session_id,
                pull_id
            );

            // Enqueue PullAck — sender task forwards in order
            // after any earlier outbound frames.
            session
                .enqueue_frame(Frame::PullAck { pull_id })
                .context("enqueue PullAck")?;
            crate::engine_audit!(
                "session_pull_ack_sent",
                session_id = %session.inner.session_id,
                pull_id
            );

            Ok(dst)
        })
    }

    fn lifecycle(&self) -> LifecycleStream {
        let (rx, replay) = self.inner.lifecycle_stream.subscribe();
        build_lifecycle_stream(rx, replay)
    }

    fn close(&self, reason: Option<String>) {
        crate::engine_audit!(
            "session_close",
            session_id = %self.inner.session_id,
            reason = ?reason
        );
        // close() implies finish_commits + finish_availability —
        // enqueue terminators on the wire if not already sent so
        // the peer's commit/availability streams close cleanly.
        // FIFO mpsc enqueue keeps everything in causal order;
        // the per-session sender task drains, sends each frame,
        // then runs `Finalize` to close the velo stream.
        let need_commits_closed = {
            let mut flag = self.inner.commits_closed.lock();
            let was_open = !*flag;
            *flag = true;
            was_open
        };
        if need_commits_closed {
            let _ = self.enqueue_frame(Frame::CommitsClosed);
        }

        let need_drained = {
            let mut flag = self.inner.avail_drained.lock();
            let was_open = !*flag;
            *flag = true;
            was_open
        };
        if need_drained {
            let _ = self.enqueue_frame(Frame::Drained);
        }

        // Best-effort: enqueue Detach. Ignore errors because
        // close is called even when the peer is gone.
        let _ = self.enqueue_frame(Frame::Detach);
        if let Some(reason) = reason {
            self.inner
                .lifecycle_stream
                .push(LifecycleEvent::Detached {
                    reason: Some(reason),
                });
        }
        // Finalize the outbound stream after any in-flight
        // frames are flushed.
        let _ = self.enqueue_finalize();
        *self.inner.closed.lock() = true;
    }
}

// ============================================================================
// Endpoint helpers
// ============================================================================

fn endpoint_from_handle(handle: velo::StreamAnchorHandle) -> SessionEndpoint {
    SessionEndpoint {
        kind: SESSION_STREAM_SCHEMA.to_string(),
        payload: serde_json::to_value(handle).expect("serialize stream anchor handle"),
    }
}

fn handle_from_endpoint(endpoint: &SessionEndpoint) -> Result<velo::StreamAnchorHandle> {
    if endpoint.kind != SESSION_STREAM_SCHEMA {
        anyhow::bail!(
            "unsupported session endpoint kind: {} (expected {})",
            endpoint.kind,
            SESSION_STREAM_SCHEMA
        );
    }
    serde_json::from_value(endpoint.payload.clone()).context("decode stream anchor handle")
}

// ============================================================================
// VeloSessionFactory
// ============================================================================

pub struct VeloSessionFactory {
    velo: Arc<velo::Velo>,
    leader: Arc<InstanceLeader>,
    runtime: Handle,
}

impl VeloSessionFactory {
    pub fn new(velo: Arc<velo::Velo>, leader: Arc<InstanceLeader>, runtime: Handle) -> Arc<Self> {
        Arc::new(Self {
            velo,
            leader,
            runtime,
        })
    }

    /// Test-only: same as the trait `open` but returns the
    /// concrete `Arc<VeloSession>` so tests can call
    /// `test_inject_inbound_frame` / `test_available_pin_count`
    /// without downcasting from `Arc<dyn Session>`.
    #[cfg(any(test, feature = "testing"))]
    pub fn open_concrete(&self, session_id: SessionId) -> Result<Arc<VeloSession>> {
        let anchor = self.velo.create_anchor::<SeqFrame>();
        let endpoint = endpoint_from_handle(anchor.handle());
        let (outbound_tx, outbound_rx) = mpsc::unbounded_channel();
        let (install_tx, install_rx) = oneshot::channel();
        let inner = VeloSession::new_inner(
            session_id,
            Arc::clone(&self.velo),
            Arc::clone(&self.leader),
            Some(endpoint),
            self.runtime.clone(),
            outbound_tx,
            install_tx,
        );
        spawn_outbound_sender(outbound_rx, install_rx, Arc::clone(&inner), &self.runtime);
        spawn_monitor(Arc::clone(&inner), anchor, self.runtime.clone());
        Ok(Arc::new(VeloSession { inner }))
    }
}

impl SessionFactory for VeloSessionFactory {
    fn open(&self, session_id: SessionId) -> Result<Arc<dyn Session>> {
        crate::engine_audit!(
            "session_factory_open",
            session_id = %session_id
        );
        let anchor = self.velo.create_anchor::<SeqFrame>();
        let endpoint = endpoint_from_handle(anchor.handle());
        let (outbound_tx, outbound_rx) = mpsc::unbounded_channel();
        let (install_tx, install_rx) = oneshot::channel();
        let inner = VeloSession::new_inner(
            session_id,
            Arc::clone(&self.velo),
            Arc::clone(&self.leader),
            Some(endpoint),
            self.runtime.clone(),
            outbound_tx,
            install_tx,
        );
        spawn_outbound_sender(outbound_rx, install_rx, Arc::clone(&inner), &self.runtime);
        spawn_monitor(Arc::clone(&inner), anchor, self.runtime.clone());
        Ok(Arc::new(VeloSession { inner }))
    }

    fn attach(
        &self,
        session_id: SessionId,
        peer_instance_id: InstanceId,
        peer_endpoint: SessionEndpoint,
    ) -> BoxFuture<'static, Result<Arc<dyn Session>>> {
        crate::engine_audit!(
            "session_factory_attach",
            session_id = %session_id,
            peer_instance_id = %peer_instance_id
        );
        let velo = Arc::clone(&self.velo);
        let leader = Arc::clone(&self.leader);
        let runtime = self.runtime.clone();
        Box::pin(async move {
            // 1. Eager metadata exchange — ensures the peer is
            //    velo-registered (the unary AM call surfaces a
            //    clear error otherwise) AND that the holder's
            //    worker metadata is imported into our
            //    parallel_worker before any wire I/O. Cached
            //    per-peer-instance on InstanceLeader so repeat
            //    attaches between the same pair pay nothing.
            //    Hot pull path: first session.pull(...) no
            //    longer pays the metadata roundtrip.
            //
            //    Skipped when the local leader has no workers —
            //    there's nothing to import into and pull(...)
            //    would fail at the worker boundary regardless.
            //    This keeps the session usable for stream-only
            //    callers (e.g. tests that don't pull).
            if leader.worker_count() > 0 {
                leader
                    .ensure_remote_metadata(peer_instance_id)
                    .await
                    .with_context(|| {
                        format!(
                            "attach: metadata exchange failed for peer {peer_instance_id} \
                             (peer not velo-registered, or remote leader unreachable)"
                        )
                    })?;
            }

            // 2. Open outbound to peer.
            let peer_handle = handle_from_endpoint(&peer_endpoint)?;
            let outbound = velo
                .attach_anchor::<SeqFrame>(peer_handle)
                .await
                .context("attach outbound to peer endpoint")?;

            // 3. Open our inbound anchor for the holder to attach back.
            let anchor = velo.create_anchor::<SeqFrame>();
            let local_endpoint = endpoint_from_handle(anchor.handle());

            let our_instance = velo.instance_id();
            let (outbound_tx, outbound_rx) = mpsc::unbounded_channel();
            let (install_tx, install_rx) = oneshot::channel();
            let inner = VeloSession::new_inner(
                session_id,
                Arc::clone(&velo),
                leader,
                Some(local_endpoint.clone()),
                runtime.clone(),
                outbound_tx,
                install_tx,
            );
            // Puller knows peer's identity out-of-band.
            *inner.peer_instance_id.lock() = Some(peer_instance_id);
            // Install outbound immediately on the puller side via
            // the install oneshot. The sender task will start
            // forwarding as soon as Attach is enqueued below.
            {
                let install_tx = inner.outbound_install_tx.lock().take();
                match install_tx {
                    Some(tx) => {
                        if tx.send(outbound).is_err() {
                            anyhow::bail!("attach: outbound install receiver dropped");
                        }
                    }
                    None => anyhow::bail!("attach: install slot already taken"),
                }
            }
            spawn_outbound_sender(outbound_rx, install_rx, Arc::clone(&inner), &runtime);
            spawn_monitor(Arc::clone(&inner), anchor, runtime.clone());

            // 4. Enqueue Attach so the holder learns our identity +
            //    can attach its outbound to our anchor + run its
            //    own ensure_remote_metadata for the reverse
            //    direction.
            let session = VeloSession { inner };
            session
                .enqueue_frame(Frame::Attach {
                    instance_id: our_instance,
                    endpoint: local_endpoint,
                })
                .context("enqueue initial Attach")?;

            Ok(Arc::new(session) as Arc<dyn Session>)
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- replay-stream unit tests (pure logic, no velo) -----

    #[test]
    fn commit_stream_replay_coalesces_adds() {
        let rs = ReplayStream::<CommitDelta>::new();
        rs.push(CommitDelta::Added(vec![mk_hash(1)]));
        rs.push(CommitDelta::Added(vec![mk_hash(2), mk_hash(3)]));
        let (rx, replay) = rs.subscribe();
        let mut stream = build_commit_stream(rx, replay);

        // Replay should yield one combined Added(3 hashes).
        use futures::StreamExt;
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            let first = stream.next().await.expect("first");
            match first {
                CommitDelta::Added(hs) => {
                    assert_eq!(hs.len(), 3);
                    assert_eq!(hs[0], mk_hash(1));
                    assert_eq!(hs[1], mk_hash(2));
                    assert_eq!(hs[2], mk_hash(3));
                }
                other => panic!("expected Added, got {other:?}"),
            }
        });
    }

    #[test]
    fn commit_stream_replay_preserves_closed_terminator() {
        let rs = ReplayStream::<CommitDelta>::new();
        rs.push(CommitDelta::Added(vec![mk_hash(1)]));
        rs.push(CommitDelta::Closed);
        let (rx, replay) = rs.subscribe();
        let mut stream = build_commit_stream(rx, replay);

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        use futures::StreamExt;
        rt.block_on(async {
            assert!(matches!(stream.next().await, Some(CommitDelta::Added(_))));
            assert!(matches!(stream.next().await, Some(CommitDelta::Closed)));
        });
    }

    #[test]
    fn avail_stream_replay_coalesces_blocks() {
        let rs = ReplayStream::<AvailabilityDelta>::new();
        rs.push(AvailabilityDelta::Available(vec![mk_committed(1, 10)]));
        rs.push(AvailabilityDelta::Available(vec![
            mk_committed(2, 11),
            mk_committed(3, 12),
        ]));
        rs.push(AvailabilityDelta::Drained);
        let (rx, replay) = rs.subscribe();
        let mut stream = build_avail_stream(rx, replay);

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        use futures::StreamExt;
        rt.block_on(async {
            match stream.next().await.unwrap() {
                AvailabilityDelta::Available(bs) => assert_eq!(bs.len(), 3),
                other => panic!("expected Available, got {other:?}"),
            }
            assert!(matches!(
                stream.next().await,
                Some(AvailabilityDelta::Drained)
            ));
        });
    }

    #[test]
    #[should_panic(expected = "subscribe called twice")]
    fn replay_stream_subscribe_twice_panics() {
        let rs = ReplayStream::<CommitDelta>::new();
        let _ = rs.subscribe();
        let _ = rs.subscribe();
    }

    #[test]
    fn lifecycle_stream_replay_preserves_order() {
        let rs = ReplayStream::<LifecycleEvent>::new();
        rs.push(LifecycleEvent::Attached {
            peer_instance_id: uuid::Uuid::new_v4().into(),
        });
        rs.push(LifecycleEvent::Detached { reason: None });
        let (rx, replay) = rs.subscribe();
        let mut stream = build_lifecycle_stream(rx, replay);

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        use futures::StreamExt;
        rt.block_on(async {
            assert!(matches!(
                stream.next().await,
                Some(LifecycleEvent::Attached { .. })
            ));
            assert!(matches!(
                stream.next().await,
                Some(LifecycleEvent::Detached { .. })
            ));
        });
    }

    fn mk_hash(seed: u64) -> SequenceHash {
        SequenceHash::new(seed, None, seed)
    }

    fn mk_committed(seed: u64, block_id: BlockId) -> CommittedBlock {
        CommittedBlock {
            hash: mk_hash(seed),
            peer_block_id: block_id,
        }
    }
}
