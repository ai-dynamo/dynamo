// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bidirectional CD session — symmetric trait both decode and
//! prefill program against.
//!
//! Each side advertises *intent* (committed hashes) and
//! *capability* (available blocks, in G2, pinned, ready to
//! pull). Strict invariant: `available ⊆ committed`. Both
//! sets are monotonic-add only within a session lifetime.
//!
//! The wire is one bidirectional `Frame` stream. The session
//! implementation demuxes incoming frames into separate
//! single-consumer mpsc channels — one per
//! trait-surface stream. Callers see independent
//! [`Session::commits`] / [`Session::availability`] /
//! [`Session::lifecycle`] streams; the implementation handles
//! the demux.
//!
//! See `/home/ryan/.claude/plans/cd-session-refactor.md` §1
//! for the full design.

use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use futures::Stream;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use kvbm_logical::blocks::{ImmutableBlock, MutableBlock};

use super::SessionEndpoint;
use crate::{BlockId, G2, InstanceId, SequenceHash};

/// Session correlation id. Re-export of the existing alias so
/// callers can use either path.
pub type SessionId = uuid::Uuid;

// ============================================================================
// Stream payload types
// ============================================================================

/// Block currently advertised as available on the peer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedBlock {
    pub hash: SequenceHash,
    pub peer_block_id: BlockId,
}

/// Delta on the peer's committed-set stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitDelta {
    /// Hashes added to peer's committed set.
    Added(Vec<SequenceHash>),
    /// Peer signaled committed set is final. Stream ends
    /// after this item.
    Closed,
}

/// Delta on the peer's availability stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AvailabilityDelta {
    /// Blocks newly available on peer (pulled-ready).
    Available(Vec<CommittedBlock>),
    /// Peer's available set will receive no more additions.
    /// Stream ends after this item.
    Drained,
}

/// Lifecycle event for an active session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleEvent {
    /// Peer attached. Carries peer's identity for any future
    /// non-session-scoped operations (e.g. metrics).
    Attached { peer_instance_id: InstanceId },
    /// Peer detached cleanly.
    Detached { reason: Option<String> },
    /// Session entered a terminal failed state.
    Failed { reason: String },
}

pub type CommitStream = Pin<Box<dyn Stream<Item = CommitDelta> + Send + 'static>>;
pub type AvailabilityStream =
    Pin<Box<dyn Stream<Item = AvailabilityDelta> + Send + 'static>>;
pub type LifecycleStream = Pin<Box<dyn Stream<Item = LifecycleEvent> + Send + 'static>>;

// ============================================================================
// On-wire frames (single bidirectional `Frame` stream)
// ============================================================================

/// One frame on the bidirectional session wire. The session
/// implementation demuxes by variant into the per-stream
/// mpsc channels and the pull-correlation table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Frame {
    /// Puller→holder. Sent immediately after attach so the
    /// holder learns the peer's identity and (optionally) an
    /// endpoint to attach back on.
    Attach {
        instance_id: InstanceId,
        endpoint: SessionEndpoint,
    },
    /// Holder→puller. Adds hashes to peer's committed set.
    Commit { hashes: Vec<SequenceHash> },
    /// Holder→puller. Terminator for the commit stream.
    CommitsClosed,
    /// Holder→puller. Adds blocks to peer's available set
    /// (each block's hash must already be in committed).
    Available { blocks: Vec<CommittedBlock> },
    /// Holder→puller. Terminator for the availability stream.
    Drained,
    /// Puller→holder. Request the holder authorize a pull
    /// of these hashes.
    Pull {
        pull_id: u64,
        hashes: Vec<SequenceHash>,
    },
    /// Holder→puller. Authorize the puller's RDMA read for
    /// the matching `pull_id`. After PullAck arrives the
    /// holder drops its pins on those hashes.
    PullComplete { pull_id: u64 },
    /// Puller→holder. The puller's RDMA read settled.
    PullAck { pull_id: u64 },
    /// Either side. Initiate teardown.
    Detach,
    /// Either side. Terminal error.
    Error { message: String },
}

// ============================================================================
// Session trait
// ============================================================================

/// Bidirectional CD session.
///
/// Each side holds an instance and programs against the same
/// trait. The implementation tracks the peer's two monotonic
/// state vectors (committed, available) as local read-models
/// replicated via `Frame`s on the wire.
pub trait Session: Send + Sync {
    /// Stable correlation id.
    fn session_id(&self) -> SessionId;

    /// Endpoint the peer uses to attach to us. `None` on the
    /// puller side once we've already attached to a peer.
    fn endpoint(&self) -> Option<SessionEndpoint>;

    // --------------------------------------------------------
    // Holder-side: things we publish to the peer.
    // --------------------------------------------------------

    /// Declare hashes we will provide. Monotonic-add. Sent to
    /// the peer as a `Commit` frame; peer sees a
    /// `CommitDelta::Added`.
    fn commit(&self, hashes: Vec<SequenceHash>) -> Result<()>;

    /// Mark the commit set complete. No more commits will
    /// follow; peer sees `CommitDelta::Closed`.
    fn finish_commits(&self) -> Result<()>;

    /// Mark previously-committed hashes as actually available
    /// for pull. Each block's hash must already be in the
    /// local committed set (validated). Pin held until the
    /// puller's `pull` for that hash completes (PullAck), then
    /// dropped automatically.
    fn make_available(&self, blocks: Vec<ImmutableBlock<G2>>) -> Result<()>;

    /// Mark the availability set complete. No more
    /// `make_available` calls will follow; peer sees
    /// `AvailabilityDelta::Drained`.
    fn finish_availability(&self) -> Result<()>;

    // --------------------------------------------------------
    // Puller-side: things we read from the peer.
    // --------------------------------------------------------

    /// Stream of commit deltas from the peer. Subscribe-once.
    /// Replays prior commits as a single `Added` then yields
    /// live deltas, ending in `Closed`.
    fn commits(&self) -> CommitStream;

    /// Stream of availability deltas from the peer.
    /// Subscribe-once. Same replay-then-live pattern, ending
    /// in `Drained`.
    fn availability(&self) -> AvailabilityStream;

    /// Snapshot of the peer's currently-committed set. Pure
    /// local read.
    fn peer_committed(&self) -> Vec<SequenceHash>;

    /// Snapshot of the peer's currently-available blocks.
    /// Pure local read.
    fn peer_available(&self) -> Vec<CommittedBlock>;

    /// Pull `hashes` from the peer into `dst`. Each hash must
    /// be in `peer_available` at call time (validated).
    /// `hashes.len() == dst.len()`. Data lands in zipped
    /// order; future resolves on transfer completion.
    fn pull(
        &self,
        hashes: Vec<SequenceHash>,
        dst: Vec<MutableBlock<G2>>,
    ) -> BoxFuture<'static, Result<Vec<MutableBlock<G2>>>>;

    // --------------------------------------------------------
    // Either side.
    // --------------------------------------------------------

    /// Lifecycle events stream. Subscribe-once.
    /// Commit/availability changes go through their dedicated
    /// streams; this one carries Attached/Detached/Failed.
    fn lifecycle(&self) -> LifecycleStream;

    /// Finalize and tear down. Implies `finish_commits` and
    /// `finish_availability` if not already called.
    fn close(&self, reason: Option<String>);
}

// ============================================================================
// SessionFactory trait
// ============================================================================

/// Factory for sessions. Owns the runtime + RDMA / mock
/// injection points. Production wraps `velo::Velo` +
/// `InstanceLeader`; tests inject in-memory mocks.
pub trait SessionFactory: Send + Sync {
    /// Open a holder-side session. The returned endpoint is
    /// shared with the peer (e.g. via the hub queue) so they
    /// can attach.
    fn open(&self, session_id: SessionId) -> Result<Arc<dyn Session>>;

    /// Attach to a peer that opened a session. Returned
    /// session is already bound and ready for `commits()` /
    /// `availability()` / `pull(...)`.
    fn attach(
        &self,
        session_id: SessionId,
        peer_endpoint: SessionEndpoint,
    ) -> BoxFuture<'static, Result<Arc<dyn Session>>>;
}

// ============================================================================
// Stage-0 skeleton — todo!() bodies.
//
// Stage 1 splits this into:
//   - `velo.rs` — production `VeloSession` + `VeloSessionFactory`
//   - `testing.rs` (feature `testing`) — `MockSession` + factory
// ============================================================================

pub mod velo;
pub use velo::{SESSION_STREAM_SCHEMA, VeloSession, VeloSessionFactory};

#[cfg(any(test, feature = "testing"))]
pub mod testing;
#[cfg(any(test, feature = "testing"))]
pub use testing::{MockSession, MockSessionFactory};

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_object_safe(_: &dyn Session) {}
    fn assert_factory_object_safe(_: &dyn SessionFactory) {}

    #[test]
    fn frame_round_trip_msgpack() {
        let frame = Frame::Pull {
            pull_id: 42,
            hashes: vec![],
        };
        let encoded = rmp_serde::to_vec(&frame).unwrap();
        let decoded: Frame = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn commit_delta_variants_are_distinct() {
        assert_ne!(
            CommitDelta::Added(vec![]),
            CommitDelta::Closed,
        );
    }

    // Stage 0: skeleton compiles. Real impls land in stage 1.
    #[test]
    fn skeleton_compiles() {
        // Ensures the trait object surface exists and the
        // trait is object-safe. Object-safety is not
        // statically guaranteed by the trait definition alone
        // because of the generic-free signatures; this asserts
        // it.
        let _: Option<&dyn Session> = None;
        let _: Option<&dyn SessionFactory> = None;
        if false {
            assert_object_safe(unreachable!());
            assert_factory_object_safe(unreachable!());
        }
    }
}
