// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory [`MockSession`] + [`MockSessionFactory`] for unit tests.
//!
//! Feature-gated: compile only with `--features testing`.
//!
//! # Observable parity
//!
//! `MockSession` matches `VeloSession`'s observable behaviour for callers
//! using only the public [`super::Session`] trait surface:
//!
//! * **Subscribe-once**: calling `commits()` / `availability()` /
//!   `lifecycle()` a second time panics.
//! * **Replay-on-subscribe**: if injects arrived before the caller subscribed,
//!   they are drained as a single `Added` / `Available` item on the first
//!   stream read, followed by live items.  Contiguous `Added`/`Available` runs
//!   in the pre-subscribe buffer are *concatenated* (not emitted as separate
//!   items) so the caller sees one batch; terminators (`Closed` / `Drained`)
//!   are preserved in their original position.
//! * **`close()` implies `finish_commits` + `finish_availability`**: calling
//!   `close()` drives both stream terminators if they haven't been sent yet.
//! * **Invariant validation is synchronous and pre-mutation**:
//!   `make_available` / `pull` error before any state mutation when their
//!   preconditions are violated.
//!
//! # Buffer-after-closed policy
//!
//! `inject_peer_commit` / `inject_peer_available` after a `Closed` / `Drained`
//! terminator has already been sent is a no-op (silently ignored).  The real
//! wire won't generate this; tests that do it have a bug.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use dashmap::DashMap;

use anyhow::{Result, anyhow};
use futures::FutureExt;
use futures::future::BoxFuture;
use kvbm_disagg_protocol::SessionEndpoint;
use kvbm_logical::blocks::{ImmutableBlock, MutableBlock};
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};

use crate::{BlockId, G2, InstanceId, SequenceHash};

use super::{
    AvailabilityDelta, AvailabilityStream, CommitDelta, CommitStream, CommittedBlock,
    LifecycleEvent, LifecycleStream, Session, SessionFactory, SessionId,
};

// ============================================================================
// Stream adapter
// ============================================================================

/// Wraps an mpsc receiver as a `Stream`.
struct MpscStream<T> {
    rx: mpsc::UnboundedReceiver<T>,
}

impl<T: Send + 'static> futures::Stream for MpscStream<T> {
    type Item = T;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

// ============================================================================
// Per-stream state
// ============================================================================

enum StreamState<T: Clone> {
    /// No subscriber yet.  Buffer items here.
    Buffering(Vec<T>),
    /// No subscriber yet, terminator already buffered. No more pushes accepted;
    /// subscribe drains the buffer (containing the terminator) into the live
    /// channel and transitions to Terminated.
    BufferingClosed(Vec<T>),
    /// Subscriber attached.  Forward directly.
    Live(mpsc::UnboundedSender<T>),
    /// Terminal item already emitted.  Subsequent injects are no-ops.
    Terminated,
}

impl<T: Clone> StreamState<T> {
    fn new() -> Self {
        Self::Buffering(Vec::new())
    }

    fn push(&mut self, item: T) {
        match self {
            Self::Buffering(buf) => buf.push(item),
            Self::BufferingClosed(_) => {}
            Self::Live(tx) => {
                let _ = tx.send(item);
            }
            Self::Terminated => {}
        }
    }

    fn terminate(&mut self) {
        match self {
            Self::Buffering(buf) => {
                let buf = std::mem::take(buf);
                *self = Self::BufferingClosed(buf);
            }
            Self::BufferingClosed(_) => {}
            Self::Live(_) | Self::Terminated => {
                *self = Self::Terminated;
            }
        }
    }

    fn is_terminated(&self) -> bool {
        matches!(self, Self::BufferingClosed(_) | Self::Terminated)
    }
}

// ============================================================================
// Pull tracking
// ============================================================================

/// One pending `pull` call.  `resolve_pull` sends through this channel.
/// The sender carries `Result<Vec<MutableBlock<G2>>>` so dst is passed back
/// on success.
struct PendingPull {
    /// Destination blocks, held here until `resolve_pull` is called.
    dst: Vec<MutableBlock<G2>>,
    /// Test drives this to completion.
    resolver: oneshot::Sender<Result<()>>,
    /// The future's end of the channel.  `pull()` returns a future that
    /// awaits this and, on `Ok`, returns `dst`.
    ///
    /// We use a second oneshot to ship `dst` back to the caller's future.
    dst_tx: oneshot::Sender<Vec<MutableBlock<G2>>>,
}

// ============================================================================
// Inner locked state
// ============================================================================

struct MockSessionInner {
    // ---- peer-side read model (driven by inject knobs) ----
    peer_committed: BTreeSet<SequenceHash>,
    peer_available: BTreeMap<SequenceHash, BlockId>,

    // ---- holder-side local state ----
    committed: BTreeSet<SequenceHash>,
    available_pins: BTreeMap<SequenceHash, ImmutableBlock<G2>>,

    // ---- recorders ----
    commit_calls: Vec<Vec<SequenceHash>>,
    make_available_calls: Vec<Vec<SequenceHash>>,
    finish_commits_called: bool,
    finish_availability_called: bool,
    pull_calls: Vec<(Vec<SequenceHash>, Vec<BlockId>)>,
    closed_reason: Option<Option<String>>,
    finished_reason: Option<Option<String>>,
    local_finished: bool,
    peer_finished: bool,
    rendezvous_finalized: bool,

    // ---- per-stream states ----
    commits_state: StreamState<CommitDelta>,
    availability_state: StreamState<AvailabilityDelta>,
    lifecycle_state: StreamState<LifecycleEvent>,

    // ---- pending pull futures ----
    pending_pulls: Vec<Option<PendingPull>>,
}

impl MockSessionInner {
    fn new() -> Self {
        Self {
            peer_committed: BTreeSet::new(),
            peer_available: BTreeMap::new(),
            committed: BTreeSet::new(),
            available_pins: BTreeMap::new(),
            commit_calls: Vec::new(),
            make_available_calls: Vec::new(),
            finish_commits_called: false,
            finish_availability_called: false,
            pull_calls: Vec::new(),
            closed_reason: None,
            finished_reason: None,
            local_finished: false,
            peer_finished: false,
            rendezvous_finalized: false,
            commits_state: StreamState::new(),
            availability_state: StreamState::new(),
            lifecycle_state: StreamState::new(),
            pending_pulls: Vec::new(),
        }
    }
}

// ============================================================================
// MockSession
// ============================================================================

pub struct MockSession {
    id: SessionId,
    endpoint: Option<SessionEndpoint>,
    /// Peer's velo `InstanceId`, set on attach (puller side)
    /// or when `Frame::Attach` arrives (holder side). Stored
    /// for parity with `VeloSession` even though the mock's
    /// `pull` doesn't drive RDMA.
    peer_instance_id: Mutex<Option<InstanceId>>,
    inner: Mutex<MockSessionInner>,
    pull_index: AtomicU64,
    /// Cross-wired partner session (paired-mode). When set,
    /// every holder-side action (`commit`, `make_available`,
    /// `finish_*`, `close`) pushes the equivalent peer-side
    /// delta onto the partner's puller streams, and `pull`
    /// auto-resolves Ok by dropping the partner's pins for
    /// the pulled hashes.
    partner: Mutex<Option<Weak<MockSession>>>,
    /// When the session was created via a factory that tracks
    /// active sessions, holds an Arc to the factory's gauge so
    /// the `Drop` impl can decrement.
    active_count: Option<Arc<std::sync::atomic::AtomicUsize>>,
}

impl Drop for MockSession {
    fn drop(&mut self) {
        if let Some(c) = &self.active_count {
            c.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl MockSession {
    #[cfg(test)]
    fn new(id: SessionId, endpoint: Option<SessionEndpoint>) -> Arc<Self> {
        Self::new_with_peer(id, endpoint, None, None)
    }

    fn new_with_peer(
        id: SessionId,
        endpoint: Option<SessionEndpoint>,
        peer_instance_id: Option<InstanceId>,
        active_count: Option<Arc<std::sync::atomic::AtomicUsize>>,
    ) -> Arc<Self> {
        if let Some(c) = active_count.as_ref() {
            c.fetch_add(1, Ordering::AcqRel);
        }
        Arc::new(Self {
            id,
            endpoint,
            peer_instance_id: Mutex::new(peer_instance_id),
            inner: Mutex::new(MockSessionInner::new()),
            pull_index: AtomicU64::new(0),
            partner: Mutex::new(None),
            active_count,
        })
    }

    /// Cross-wire this session with a partner. Mutual: the
    /// partner is also linked back to this session.
    /// Subsequent holder-side actions on either side
    /// auto-deliver to the other's puller streams. Any
    /// committed / made-available state already published on
    /// either side at pair time is replayed onto the partner's
    /// peer streams so late-attach matches the velo
    /// replay-on-subscribe semantics.
    pub fn pair_with(self: &Arc<Self>, partner: &Arc<MockSession>) {
        *self.partner.lock() = Some(Arc::downgrade(partner));
        *partner.partner.lock() = Some(Arc::downgrade(self));
        replay_to_partner(self, partner);
        replay_to_partner(partner, self);
    }

    fn partner(&self) -> Option<Arc<MockSession>> {
        self.partner.lock().as_ref().and_then(Weak::upgrade)
    }

    /// Test accessor for `peer_instance_id` (set on attach or
    /// via `inject_peer_attached`).
    pub fn peer_instance_id(&self) -> Option<InstanceId> {
        *self.peer_instance_id.lock()
    }

    // ---- subscribe helpers ----

    fn take_commits_stream(&self) -> CommitStream {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut inner = self.inner.lock();
        match &mut inner.commits_state {
            StreamState::Buffering(buf) => {
                for item in drain_commit_buffer(std::mem::take(buf)) {
                    let _ = tx.send(item);
                }
                inner.commits_state = StreamState::Live(tx);
            }
            StreamState::BufferingClosed(buf) => {
                for item in drain_commit_buffer(std::mem::take(buf)) {
                    let _ = tx.send(item);
                }
                inner.commits_state = StreamState::Terminated;
            }
            StreamState::Live(_) => {
                panic!("MockSession::commits called twice");
            }
            StreamState::Terminated => {
                // Already closed; return empty stream.
            }
        }
        Box::pin(MpscStream { rx })
    }

    fn take_availability_stream(&self) -> AvailabilityStream {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut inner = self.inner.lock();
        match &mut inner.availability_state {
            StreamState::Buffering(buf) => {
                for item in drain_availability_buffer(std::mem::take(buf)) {
                    let _ = tx.send(item);
                }
                inner.availability_state = StreamState::Live(tx);
            }
            StreamState::BufferingClosed(buf) => {
                for item in drain_availability_buffer(std::mem::take(buf)) {
                    let _ = tx.send(item);
                }
                inner.availability_state = StreamState::Terminated;
            }
            StreamState::Live(_) => {
                panic!("MockSession::availability called twice");
            }
            StreamState::Terminated => {}
        }
        Box::pin(MpscStream { rx })
    }

    fn take_lifecycle_stream(&self) -> LifecycleStream {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut inner = self.inner.lock();
        match &mut inner.lifecycle_state {
            StreamState::Buffering(buf) => {
                for item in std::mem::take(buf) {
                    let _ = tx.send(item);
                }
                inner.lifecycle_state = StreamState::Live(tx);
            }
            StreamState::BufferingClosed(buf) => {
                for item in std::mem::take(buf) {
                    let _ = tx.send(item);
                }
                inner.lifecycle_state = StreamState::Terminated;
            }
            StreamState::Live(_) => {
                panic!("MockSession::lifecycle called twice");
            }
            StreamState::Terminated => {}
        }
        Box::pin(MpscStream { rx })
    }

    // ========================================================================
    // Test-side knobs
    // ========================================================================

    pub fn inject_peer_commit(&self, hashes: Vec<SequenceHash>) {
        let mut inner = self.inner.lock();
        if inner.commits_state.is_terminated() {
            return;
        }
        for &h in &hashes {
            inner.peer_committed.insert(h);
        }
        inner.commits_state.push(CommitDelta::Added(hashes));
    }

    pub fn inject_peer_finish_commits(&self) {
        let mut inner = self.inner.lock();
        if inner.commits_state.is_terminated() {
            return;
        }
        inner.commits_state.push(CommitDelta::Closed);
        inner.commits_state.terminate();
    }

    pub fn inject_peer_available(&self, blocks: Vec<CommittedBlock>) {
        let mut inner = self.inner.lock();
        if inner.availability_state.is_terminated() {
            return;
        }
        for b in &blocks {
            inner.peer_available.insert(b.hash, b.peer_block_id);
        }
        inner
            .availability_state
            .push(AvailabilityDelta::Available(blocks));
    }

    pub fn inject_peer_drained(&self) {
        let mut inner = self.inner.lock();
        if inner.availability_state.is_terminated() {
            return;
        }
        inner.availability_state.push(AvailabilityDelta::Drained);
        inner.availability_state.terminate();
    }

    pub fn inject_lifecycle(&self, event: LifecycleEvent) {
        self.inner.lock().lifecycle_state.push(event);
    }

    /// Resolve the Nth `pull` call (0-indexed).
    ///
    /// On `Ok(())`, `dst` is shipped back to the future's caller.
    /// On `Err(e)`, the error is forwarded to the future.
    pub fn resolve_pull(&self, index: usize, result: Result<()>) {
        let pending = {
            let mut inner = self.inner.lock();
            inner
                .pending_pulls
                .get_mut(index)
                .expect("resolve_pull: index out of bounds")
                .take()
                .expect("resolve_pull: already resolved")
        };
        match result {
            Ok(()) => {
                let _ = pending.dst_tx.send(pending.dst);
                let _ = pending.resolver.send(Ok(()));
            }
            Err(e) => {
                drop(pending.dst); // drop dst on error
                let _ = pending.resolver.send(Err(e));
            }
        }
    }

    /// Resolve the Nth `pull` call with a SHORT (`returned_len <
    /// dst.len()`) or LONG (`returned_len > dst.len()`) result.
    ///
    /// Used to exercise the caller's length-mismatch guards. The
    /// pull future receives a `MutableBlock` vec of `returned_len`
    /// elements (truncated from the original `dst` for short, or
    /// padded by allocating extras for long — out of scope here;
    /// short is the realistic case).
    ///
    /// Panics if `returned_len > dst.len()` (long-result mocking
    /// would require allocating fresh mutables outside the original
    /// slot, which the harness does not support).
    pub fn resolve_pull_short(&self, index: usize, returned_len: usize) {
        let pending = {
            let mut inner = self.inner.lock();
            inner
                .pending_pulls
                .get_mut(index)
                .expect("resolve_pull_short: index out of bounds")
                .take()
                .expect("resolve_pull_short: already resolved")
        };
        let mut dst = pending.dst;
        assert!(
            returned_len <= dst.len(),
            "resolve_pull_short: returned_len ({}) > dst.len() ({})",
            returned_len,
            dst.len()
        );
        dst.truncate(returned_len);
        let _ = pending.dst_tx.send(dst);
        let _ = pending.resolver.send(Ok(()));
    }

    // ---- recorders ----

    pub fn commit_calls(&self) -> Vec<Vec<SequenceHash>> {
        self.inner.lock().commit_calls.clone()
    }

    pub fn make_available_calls(&self) -> Vec<Vec<SequenceHash>> {
        self.inner.lock().make_available_calls.clone()
    }

    pub fn pull_calls(&self) -> Vec<(Vec<SequenceHash>, Vec<BlockId>)> {
        self.inner.lock().pull_calls.clone()
    }

    pub fn closed_reason(&self) -> Option<Option<String>> {
        self.inner.lock().closed_reason.clone()
    }

    pub fn finished_reason(&self) -> Option<Option<String>> {
        self.inner.lock().finished_reason.clone()
    }

    /// Test-only: simulate inbound `Frame::Finished` from the peer.
    /// Marks `peer_finished` true and may trigger the rendezvous
    /// finalize if local has already called `finished()`.
    pub fn inject_peer_finished(&self) {
        {
            let mut inner = self.inner.lock();
            inner.peer_finished = true;
        }
        self.maybe_finalize_paired();
    }

    /// Test-only: when both sides of a paired MockSession have
    /// signalled `finished`, drive the velo-equivalent of
    /// `sender.finalize()` on this side. Mirrors VeloSession's
    /// `maybe_finalize`. Pushes a `Detached` lifecycle event so
    /// callers' watchers fire.
    fn maybe_finalize_paired(&self) {
        let should_finalize = {
            let mut inner = self.inner.lock();
            if !(inner.local_finished && inner.peer_finished) {
                return;
            }
            if inner.rendezvous_finalized {
                return;
            }
            inner.rendezvous_finalized = true;
            true
        };
        if should_finalize {
            let mut inner = self.inner.lock();
            inner
                .lifecycle_state
                .push(LifecycleEvent::Detached {
                    reason: Some("rendezvous".to_string()),
                });
            inner.lifecycle_state.terminate();
        }
    }

    pub async fn wait_pull_count(&self, n: usize) {
        wait_until(|| self.inner.lock().pull_calls.len() >= n).await;
    }
}

/// Replay `holder`'s already-published state onto `puller`'s peer
/// streams. Called from `pair_with` so a late-attach puller
/// observes the holder's pre-attach commits/availability — same
/// semantics velo's replay-on-subscribe gives us on the wire.
fn replay_to_partner(holder: &Arc<MockSession>, puller: &Arc<MockSession>) {
    let (committed, available, finish_commits, finish_avail, closed_reason) = {
        let h = holder.inner.lock();
        let committed: Vec<SequenceHash> = h.committed.iter().copied().collect();
        let available: Vec<CommittedBlock> = h
            .available_pins
            .iter()
            .map(|(hash, block)| CommittedBlock {
                hash: *hash,
                peer_block_id: block.block_id(),
            })
            .collect();
        (
            committed,
            available,
            h.finish_commits_called,
            h.finish_availability_called,
            h.closed_reason.clone(),
        )
    };
    if !committed.is_empty() {
        puller.inject_peer_commit(committed);
    }
    if finish_commits {
        puller.inject_peer_finish_commits();
    }
    if !available.is_empty() {
        puller.inject_peer_available(available);
    }
    if finish_avail {
        puller.inject_peer_drained();
    }
    if let Some(reason) = closed_reason {
        puller.inject_lifecycle(LifecycleEvent::Detached { reason });
    }
}

// ============================================================================
// Replay-buffer helpers
// ============================================================================

fn drain_commit_buffer(buf: Vec<CommitDelta>) -> Vec<CommitDelta> {
    let mut out: Vec<CommitDelta> = Vec::new();
    for item in buf {
        match item {
            CommitDelta::Added(hashes) => match out.last_mut() {
                Some(CommitDelta::Added(prev)) => prev.extend(hashes),
                _ => out.push(CommitDelta::Added(hashes)),
            },
            CommitDelta::Closed => out.push(CommitDelta::Closed),
        }
    }
    out
}

fn drain_availability_buffer(buf: Vec<AvailabilityDelta>) -> Vec<AvailabilityDelta> {
    let mut out: Vec<AvailabilityDelta> = Vec::new();
    for item in buf {
        match item {
            AvailabilityDelta::Available(blocks) => match out.last_mut() {
                Some(AvailabilityDelta::Available(prev)) => prev.extend(blocks),
                _ => out.push(AvailabilityDelta::Available(blocks)),
            },
            AvailabilityDelta::Drained => out.push(AvailabilityDelta::Drained),
        }
    }
    out
}

// ============================================================================
// Session trait impl
// ============================================================================

impl Session for MockSession {
    fn session_id(&self) -> SessionId {
        self.id
    }

    fn endpoint(&self) -> Option<SessionEndpoint> {
        self.endpoint.clone()
    }

    fn commit(&self, hashes: Vec<SequenceHash>) -> Result<()> {
        {
            let mut inner = self.inner.lock();
            inner.commit_calls.push(hashes.clone());
            for h in &hashes {
                inner.committed.insert(*h);
            }
        }
        if let Some(partner) = self.partner() {
            partner.inject_peer_commit(hashes);
        }
        Ok(())
    }

    fn finish_commits(&self) -> Result<()> {
        self.inner.lock().finish_commits_called = true;
        if let Some(partner) = self.partner() {
            partner.inject_peer_finish_commits();
        }
        Ok(())
    }

    fn make_available(&self, blocks: Vec<ImmutableBlock<G2>>) -> Result<()> {
        // Validate and mutate under a single lock guard (no TOCTOU gap).
        let mut peer_committed_blocks: Vec<CommittedBlock> = Vec::with_capacity(blocks.len());
        {
            let mut inner = self.inner.lock();
            for block in &blocks {
                let hash = block.sequence_hash();
                if !inner.committed.contains(&hash) {
                    return Err(anyhow!(
                        "make_available: hash {hash:?} not in committed set"
                    ));
                }
            }
            let hashes: Vec<SequenceHash> = blocks.iter().map(|b| b.sequence_hash()).collect();
            inner.make_available_calls.push(hashes);
            for block in blocks {
                let hash = block.sequence_hash();
                let block_id = block.block_id();
                inner.available_pins.insert(hash, block);
                peer_committed_blocks.push(CommittedBlock {
                    hash,
                    peer_block_id: block_id,
                });
            }
        }
        if let Some(partner) = self.partner() {
            partner.inject_peer_available(peer_committed_blocks);
        }
        Ok(())
    }

    fn finish_availability(&self) -> Result<()> {
        self.inner.lock().finish_availability_called = true;
        if let Some(partner) = self.partner() {
            partner.inject_peer_drained();
        }
        Ok(())
    }

    fn commits(&self) -> CommitStream {
        self.take_commits_stream()
    }

    fn availability(&self) -> AvailabilityStream {
        self.take_availability_stream()
    }

    fn peer_committed(&self) -> Vec<SequenceHash> {
        self.inner.lock().peer_committed.iter().copied().collect()
    }

    fn peer_available(&self) -> Vec<CommittedBlock> {
        self.inner
            .lock()
            .peer_available
            .iter()
            .map(|(&hash, &peer_block_id)| CommittedBlock {
                hash,
                peer_block_id,
            })
            .collect()
    }

    fn pull(
        &self,
        hashes: Vec<SequenceHash>,
        dst: Vec<MutableBlock<G2>>,
    ) -> BoxFuture<'static, Result<Vec<MutableBlock<G2>>>> {
        // Synchronous length validation.
        if hashes.len() != dst.len() {
            let hl = hashes.len();
            let dl = dst.len();
            return async move { Err(anyhow!("pull: hashes.len() ({hl}) != dst.len() ({dl})")) }
                .boxed();
        }

        // Assign a monotonic index before taking the lock (atomic, no contention).
        let index = self.pull_index.fetch_add(1, Ordering::SeqCst) as usize;
        let dst_block_ids: Vec<BlockId> = dst.iter().map(|b| b.block_id()).collect();

        // Validate peer_available and record the call.
        {
            let mut inner = self.inner.lock();
            for hash in &hashes {
                if !inner.peer_available.contains_key(hash) {
                    let hash = *hash;
                    return async move {
                        Err(anyhow!("pull: hash {hash:?} not in peer_available"))
                    }
                    .boxed();
                }
            }
            inner.pull_calls.push((hashes.clone(), dst_block_ids));
        }

        // Paired-mode auto-resolution: drop partner pins for the
        // pulled hashes and immediately return dst.
        if let Some(partner) = self.partner() {
            {
                let mut p_inner = partner.inner.lock();
                for h in &hashes {
                    p_inner.available_pins.remove(h);
                }
            }
            return async move { Ok(dst) }.boxed();
        }

        // Manual mode: install a PendingPull for the test to
        // resolve via `resolve_pull(index, ...)`.
        let (resolver_tx, resolver_rx) = oneshot::channel::<Result<()>>();
        let (dst_tx, dst_rx) = oneshot::channel::<Vec<MutableBlock<G2>>>();
        {
            let mut inner = self.inner.lock();
            while inner.pending_pulls.len() <= index {
                inner.pending_pulls.push(None);
            }
            inner.pending_pulls[index] = Some(PendingPull {
                dst,
                resolver: resolver_tx,
                dst_tx,
            });
        }

        async move {
            let result = resolver_rx
                .await
                .map_err(|_| anyhow!("pull resolver dropped"))?;
            result?;
            dst_rx
                .await
                .map_err(|_| anyhow!("pull dst channel dropped"))
        }
        .boxed()
    }

    fn lifecycle(&self) -> LifecycleStream {
        self.take_lifecycle_stream()
    }

    fn finalize(&self, reason: Option<String>) {
        let need_signal = {
            let mut inner = self.inner.lock();
            inner.finished_reason = Some(reason);
            inner.finish_commits_called = true;
            inner.finish_availability_called = true;
            if !inner.commits_state.is_terminated() {
                inner.commits_state.push(CommitDelta::Closed);
                inner.commits_state.terminate();
            }
            if !inner.availability_state.is_terminated() {
                inner.availability_state.push(AvailabilityDelta::Drained);
                inner.availability_state.terminate();
            }
            let was_open = !inner.local_finished;
            inner.local_finished = true;
            was_open
        };
        // Paired mode: deliver Finished to partner so its
        // peer_finished + maybe_finalize fire.
        if need_signal {
            if let Some(partner) = self.partner() {
                partner.inject_peer_finished();
            }
        }
        self.maybe_finalize_paired();
    }

    fn close(&self, reason: Option<String>) {
        {
            let mut inner = self.inner.lock();
            inner.closed_reason = Some(reason.clone());
            // Imply finish_commits and finish_availability.
            inner.finish_commits_called = true;
            inner.finish_availability_called = true;
            // Drive commit/availability stream terminators.
            if !inner.commits_state.is_terminated() {
                inner.commits_state.push(CommitDelta::Closed);
                inner.commits_state.terminate();
            }
            if !inner.availability_state.is_terminated() {
                inner.availability_state.push(AvailabilityDelta::Drained);
                inner.availability_state.terminate();
            }
            // Push a Detached lifecycle event if a reason was given.
            if let Some(r) = reason.clone() {
                inner
                    .lifecycle_state
                    .push(LifecycleEvent::Detached { reason: Some(r) });
            }
            inner.lifecycle_state.terminate();
        }
        // Paired mode: deliver terminators + Detached to partner.
        if let Some(partner) = self.partner() {
            partner.inject_peer_finish_commits();
            partner.inject_peer_drained();
            partner.inject_lifecycle(LifecycleEvent::Detached { reason });
        }
    }
}

// ============================================================================
// MockSessionFactory
// ============================================================================

struct AttachRecord {
    session_id: SessionId,
    peer_instance_id: InstanceId,
    peer_endpoint: SessionEndpoint,
}

pub struct MockSessionFactory {
    last_opened: Mutex<Option<Arc<MockSession>>>,
    last_attached: Mutex<Option<Arc<MockSession>>>,
    attach_records: Mutex<Vec<AttachRecord>>,
    /// Shared registry for paired-mode. When two factories are
    /// constructed via `make_paired()`, they share this Arc; on
    /// `attach`, the factory looks up the partner factory's
    /// previously-`open`ed session and cross-wires the new
    /// attach session with it.
    paired_registry: Option<Arc<DashMap<SessionId, Arc<MockSession>>>>,
    /// Live session count. Each `MockSession` created via this
    /// factory holds a clone and decrements on Drop. The
    /// factory's own `last_opened` / `last_attached` Arcs
    /// hold strong refs that must be dropped (or replaced) for
    /// the count to reach zero — production callers don't
    /// retain factory-side refs, but tests must clear them
    /// (or drop the factory) to assert the gauge.
    active_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Cumulative count of `open` calls (lifetime, never decremented).
    open_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl MockSessionFactory {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Build a pair of factories that cross-wire opened/attached
    /// sessions in-process. Use this for higher-level wrapper
    /// composition tests where a real velo wire is overkill.
    ///
    /// Returns `(holder_factory, puller_factory)`. The holder
    /// calls `open(session_id)` and the puller calls
    /// `attach(session_id, ...)` with the same id; the two
    /// `MockSession`s are then linked so that any holder-side
    /// action (`commit`, `make_available`, `finish_*`, `close`)
    /// pushes the equivalent peer-side delta onto the puller's
    /// streams, and `pull(...)` auto-resolves Ok by dropping the
    /// partner's pins for the pulled hashes.
    pub fn make_paired() -> (Arc<Self>, Arc<Self>) {
        let registry = Arc::new(DashMap::new());
        (
            Arc::new(Self {
                last_opened: Mutex::new(None),
                last_attached: Mutex::new(None),
                attach_records: Mutex::new(Vec::new()),
                paired_registry: Some(Arc::clone(&registry)),
                active_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                open_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }),
            Arc::new(Self {
                last_opened: Mutex::new(None),
                last_attached: Mutex::new(None),
                attach_records: Mutex::new(Vec::new()),
                paired_registry: Some(registry),
                active_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                open_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }),
        )
    }

    /// The session created by the most recent `open` call.
    pub fn last_opened(&self) -> Option<Arc<MockSession>> {
        self.last_opened.lock().clone()
    }

    /// The session created by the most recent `attach` call.
    pub fn last_attached(&self) -> Option<Arc<MockSession>> {
        self.last_attached.lock().clone()
    }

    /// All `(session_id, peer_instance_id, peer_endpoint)` triples passed to
    /// `attach`, in order.
    pub fn attach_calls(&self) -> Vec<(SessionId, InstanceId, SessionEndpoint)> {
        self.attach_records
            .lock()
            .iter()
            .map(|r| (r.session_id, r.peer_instance_id, r.peer_endpoint.clone()))
            .collect()
    }

    /// Cumulative count of `open` calls. Idempotency tests assert this
    /// stays at 1 across repeat wrapper calls.
    pub fn open_count(&self) -> usize {
        self.open_count.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl Default for MockSessionFactory {
    fn default() -> Self {
        Self {
            last_opened: Mutex::new(None),
            last_attached: Mutex::new(None),
            attach_records: Mutex::new(Vec::new()),
            paired_registry: None,
            active_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            open_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

fn mock_endpoint() -> SessionEndpoint {
    SessionEndpoint {
        kind: "mock".to_string(),
        payload: serde_json::json!({ "session": "mock" }),
    }
}

impl SessionFactory for MockSessionFactory {
    fn active_session_count(&self) -> usize {
        self.active_count.load(Ordering::Acquire)
    }

    fn open(&self, session_id: SessionId) -> Result<Arc<dyn Session>> {
        self.open_count.fetch_add(1, Ordering::AcqRel);
        let session = MockSession::new_with_peer(
            session_id,
            Some(mock_endpoint()),
            None,
            Some(Arc::clone(&self.active_count)),
        );
        if let Some(registry) = &self.paired_registry {
            registry.insert(session_id, Arc::clone(&session));
        }
        *self.last_opened.lock() = Some(Arc::clone(&session));
        Ok(session)
    }

    fn attach(
        &self,
        session_id: SessionId,
        peer_instance_id: InstanceId,
        peer_endpoint: SessionEndpoint,
    ) -> BoxFuture<'static, Result<Arc<dyn Session>>> {
        let session = MockSession::new_with_peer(
            session_id,
            None,
            Some(peer_instance_id),
            Some(Arc::clone(&self.active_count)),
        );
        if let Some(registry) = &self.paired_registry {
            if let Some(holder) = registry.get(&session_id).map(|e| Arc::clone(e.value())) {
                session.pair_with(&holder);
                // Mirror velo: holder receives `Frame::Attach` →
                // pushes `LifecycleEvent::Attached` on its own
                // lifecycle stream.
                holder.inject_lifecycle(LifecycleEvent::Attached { peer_instance_id });
            }
        }
        *self.last_attached.lock() = Some(Arc::clone(&session));
        self.attach_records.lock().push(AttachRecord {
            session_id,
            peer_instance_id,
            peer_endpoint,
        });
        async move { Ok(session as Arc<dyn Session>) }.boxed()
    }
}

// ============================================================================
// Shared async utility
// ============================================================================

/// Poll a sync predicate until true, sleeping 5 ms between polls.
/// Panics after ~1 s (200 attempts).
pub async fn wait_until(predicate: impl Fn() -> bool) {
    for _ in 0..200 {
        if predicate() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    panic!("wait_until: condition not met within ~1 s");
}

// ============================================================================
// Inline tests
// ============================================================================

#[cfg(all(test, feature = "testing"))]
mod tests {
    use futures::StreamExt;

    use crate::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
    use crate::testing::token_blocks::create_token_sequence;
    use crate::{BlockId, G2, SequenceHash};
    use kvbm_logical::blocks::{ImmutableBlock, MutableBlock};
    use kvbm_logical::manager::BlockManager;
    use std::sync::Arc;

    use super::*;

    const TEST_BLOCK_SIZE: usize = 16;

    fn hash(n: u64) -> SequenceHash {
        SequenceHash::new(n, None, n)
    }

    // ---- block helpers ----

    fn make_g2_manager(count: usize) -> Arc<BlockManager<G2>> {
        let registry = TestRegistryBuilder::new().build();
        Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(count)
                .block_size(TEST_BLOCK_SIZE)
                .registry(registry)
                .build(),
        )
    }

    fn alloc_immutable(
        manager: &Arc<BlockManager<G2>>,
        count: usize,
        start_token: u32,
    ) -> Vec<ImmutableBlock<G2>> {
        let token_sequence = create_token_sequence(count, TEST_BLOCK_SIZE, start_token);
        let mutable = manager.allocate_blocks(count).expect("alloc failed");
        let complete: Vec<_> = mutable
            .into_iter()
            .zip(token_sequence.blocks().iter())
            .map(|(b, tb)| b.complete(tb).expect("complete failed"))
            .collect();
        manager.register_blocks(complete)
    }

    fn alloc_mutable(manager: &Arc<BlockManager<G2>>, count: usize) -> Vec<MutableBlock<G2>> {
        manager
            .allocate_blocks(count)
            .expect("alloc mutable failed")
    }

    /// Build a `Vec<CommittedBlock>` from immutable blocks with fake peer IDs.
    fn to_committed(blocks: &[ImmutableBlock<G2>], base_id: BlockId) -> Vec<CommittedBlock> {
        blocks
            .iter()
            .enumerate()
            .map(|(i, b)| CommittedBlock {
                hash: b.sequence_hash(),
                peer_block_id: base_id + i as BlockId,
            })
            .collect()
    }

    // ========================================================================
    // make_available errors if hash not in committed
    // ========================================================================

    #[test]
    fn make_available_errors_if_not_committed() {
        let manager = make_g2_manager(1);
        let blocks = alloc_immutable(&manager, 1, 0);
        let session = MockSession::new(uuid::Uuid::new_v4(), None);
        let err = session.make_available(blocks).unwrap_err();
        assert!(
            err.to_string().contains("not in committed set"),
            "unexpected error: {err}"
        );
    }

    // ========================================================================
    // pull errors if hash not in peer_available
    // ========================================================================

    #[tokio::test]
    async fn pull_errors_if_not_peer_available() {
        let manager = make_g2_manager(1);
        let dst = alloc_mutable(&manager, 1);
        let session = MockSession::new(uuid::Uuid::new_v4(), None);
        let err = session.pull(vec![hash(42)], dst).await.unwrap_err();
        assert!(
            err.to_string().contains("not in peer_available"),
            "unexpected error: {err}"
        );
    }

    // ========================================================================
    // pull errors if hashes.len() != dst.len()
    // ========================================================================

    #[tokio::test]
    async fn pull_errors_on_length_mismatch() {
        let manager = make_g2_manager(1);
        let dst = alloc_mutable(&manager, 1);
        let session = MockSession::new(uuid::Uuid::new_v4(), None);
        let err = session.pull(vec![hash(1), hash(2)], dst).await.unwrap_err();
        assert!(
            err.to_string().contains("hashes.len()"),
            "unexpected error: {err}"
        );
    }

    // ========================================================================
    // Replay-on-subscribe: 3 injects -> single Added
    // ========================================================================

    #[tokio::test]
    async fn replay_on_subscribe_single_added() {
        let session = MockSession::new(uuid::Uuid::new_v4(), None);
        session.inject_peer_commit(vec![hash(1)]);
        session.inject_peer_commit(vec![hash(2)]);
        session.inject_peer_commit(vec![hash(3)]);

        let mut stream = session.commits();
        let delta = stream.next().await.expect("expected a delta");
        match delta {
            CommitDelta::Added(hashes) => {
                assert_eq!(hashes.len(), 3, "expected all 3 hashes in one Added");
                assert!(hashes.contains(&hash(1)));
                assert!(hashes.contains(&hash(2)));
                assert!(hashes.contains(&hash(3)));
            }
            CommitDelta::Closed => panic!("unexpected Closed"),
        }
    }

    // ========================================================================
    // Subscribe-once: second call panics
    // ========================================================================

    #[test]
    #[should_panic(expected = "commits called twice")]
    fn subscribe_once_panics_on_second_call() {
        let session = MockSession::new(uuid::Uuid::new_v4(), None);
        let _s1 = session.commits();
        let _s2 = session.commits(); // must panic
    }

    // ========================================================================
    // pull resolves on resolve_pull(0, Ok)
    // ========================================================================

    #[tokio::test]
    async fn pull_resolves_ok() {
        let mgr_src = make_g2_manager(1);
        let mgr_dst = make_g2_manager(1);
        let src_blocks = alloc_immutable(&mgr_src, 1, 0);
        let committed = to_committed(&src_blocks, 10);
        let dst = alloc_mutable(&mgr_dst, 1);

        let session = MockSession::new(uuid::Uuid::new_v4(), None);
        session.inject_peer_available(committed.clone());

        let hashes = vec![committed[0].hash];
        let pull_fut = session.pull(hashes, dst);
        let handle = tokio::spawn(async move { pull_fut.await });

        session.wait_pull_count(1).await;
        session.resolve_pull(0, Ok(()));

        let returned = handle.await.unwrap().expect("pull should succeed");
        assert_eq!(returned.len(), 1, "expected 1 dst block returned");
    }

    // ========================================================================
    // pull resolves on resolve_pull(0, Err)
    // ========================================================================

    #[tokio::test]
    async fn pull_resolves_err() {
        let mgr_src = make_g2_manager(1);
        let mgr_dst = make_g2_manager(1);
        let src_blocks = alloc_immutable(&mgr_src, 1, 0);
        let committed = to_committed(&src_blocks, 10);
        let dst = alloc_mutable(&mgr_dst, 1);

        let session = MockSession::new(uuid::Uuid::new_v4(), None);
        session.inject_peer_available(committed.clone());

        let hashes = vec![committed[0].hash];
        let session_clone = Arc::clone(&session);
        let pull_fut = session.pull(hashes, dst);
        let handle = tokio::spawn(async move { pull_fut.await });

        session_clone.wait_pull_count(1).await;
        session_clone.resolve_pull(0, Err(anyhow!("simulated transport failure")));

        let err = handle.await.unwrap().unwrap_err();
        assert!(
            err.to_string().contains("simulated transport failure"),
            "unexpected error: {err}"
        );
    }

    // ========================================================================
    // MockSessionFactory::last_opened returns the opened session
    // ========================================================================

    #[test]
    fn factory_last_opened() {
        let factory = MockSessionFactory::new();
        let id = uuid::Uuid::new_v4();
        let session = factory.open(id).unwrap();
        let last = factory.last_opened().expect("last_opened should be Some");
        assert_eq!(session.session_id(), last.session_id());
        assert_eq!(last.endpoint().map(|e| e.kind), Some("mock".to_string()));
    }

    // ========================================================================
    // MockSessionFactory::attach returns session with None endpoint
    // ========================================================================

    #[tokio::test]
    async fn factory_attach_endpoint_is_none() {
        let factory = MockSessionFactory::new();
        let id = uuid::Uuid::new_v4();
        let peer_id: InstanceId = uuid::Uuid::new_v4().into();
        let session = factory.attach(id, peer_id, mock_endpoint()).await.unwrap();
        assert!(
            session.endpoint().is_none(),
            "attach session should have None endpoint"
        );
    }

    #[tokio::test]
    async fn factory_attach_records_peer_instance_id() {
        let factory = MockSessionFactory::new();
        let id = uuid::Uuid::new_v4();
        let peer_id: InstanceId = uuid::Uuid::new_v4().into();
        let _ = factory.attach(id, peer_id, mock_endpoint()).await.unwrap();
        let calls = factory.attach_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, peer_id);
        let last = factory.last_attached().unwrap();
        assert_eq!(last.peer_instance_id(), Some(peer_id));
    }
}
