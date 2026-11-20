// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-side KV state: one rank's ordered KV event stream folded into an exact residency
//! ledger, with the history and snapshots that streaming, replay and bootstrap are served from.
//!
//! # Audience
//!
//! An engine's Rust frontend (vLLM's `vllm-server`, SGLang's `sglang-kv-events` relay) or a
//! Dynamo sidecar hosts one [`RankStream`] per data-parallel rank, next to the code that reads
//! the engine's KV events. The host owns the transport (gRPC, the Dynamo request plane) and the
//! wire types; this crate owns the state machine behind them:
//!
//! | API the host serves | What it calls |
//! |---------------------|---------------|
//! | live stream | [`RankStream::apply_batch`], then fan out the batch it just passed in |
//! | replay from a cursor | [`RankStream::resume`] + [`RankStream::backlog`] |
//! | snapshot | [`RankStream::snapshot`], then [`SnapshotImage::runs`] on any thread |
//! | subscribe (server picks) | [`RankStream::admit`], then as its [`Admission`] says |
//!
//! # Contract
//!
//! - **Single owner, synchronous.** Every mutating call takes `&mut self`; there are no threads,
//!   locks, channels or async here. The host decides which thread or task owns the stream.
//! - **The engine's sequence, never renumbered.** A cursor is `(epoch, next batch sequence)`. A
//!   stale or out-of-order batch is refused and nothing is applied; the host decides what a gap
//!   means and calls [`RankStream::start_epoch`] or [`RankStream::reseed`].
//! - **The ledger is the authority.** It is keyed by the engine's own block hash, so the engine
//!   does not hash anything for this crate. A prefix index ([`StreamIndex`]) is optional and is
//!   fed from the same fold.
//! - **Exact cut.** A snapshot is the state after batch `S - 1`; live batches continue from `S`.
//!
//! # Dependencies
//!
//! The standard library, `rustc-hash` and `thiserror`, and nothing from the Dynamo workspace:
//! an engine embeds this crate and builds it with its own toolchain. Dynamo's prefix indexes
//! attach from the other side, in `dynamo_kv_indexer::stream`.

mod history;
mod index;
mod ledger;
mod rank;
mod snapshot;
mod tiers;

#[cfg(test)]
mod tests;

pub use history::History;
pub use index::{NoIndex, StreamIndex};
pub use ledger::{BlockView, Ledger, LedgerStats, Removed, Stored};
pub use rank::{
    Admission, Applied, ApplyError, BatchWriter, Carry, Health, OnMiss, RankStream, Refusal,
    ReseedError, SameEpoch, StreamConfig, StreamInfo, StreamStats,
};
pub use snapshot::{ChunkBy, SnapshotImage, SnapshotRun, chunk_by};
pub use tiers::{MAX_TIERS, Tier, TierCounts, TierMap, TierSet};

/// The engine's identity for one cached block: its sequence hash, narrowed to 64 bits.
///
/// It names the block for parent links and removals. It is opaque here: nothing recomputes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockHash(pub u64);

impl BlockHash {
    /// A signed 64-bit engine hash (SGLang publishes these), bit for bit.
    pub const fn from_i64(hash: i64) -> Self {
        Self(hash as u64)
    }

    /// The low 64 bits of a big-endian digest, which is how vLLM narrows its SHA-256 block
    /// hash to an integer. `None` for an empty digest.
    pub fn from_be_bytes(digest: &[u8]) -> Option<Self> {
        if digest.is_empty() {
            return None;
        }
        let tail = &digest[digest.len().saturating_sub(8)..];
        let mut bytes = [0u8; 8];
        bytes[8 - tail.len()..].copy_from_slice(tail);
        Some(Self(u64::from_be_bytes(bytes)))
    }
}

/// A hash of the block's content that a prefix index descends by (Dynamo's `LocalBlockHash`).
///
/// Only hosts that attach a [`StreamIndex`] supply it; an engine frontend leaves it
/// [`ContentHash::NONE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub u64);

impl ContentHash {
    /// No content hash. This crate reserves 0 for it, so 0 cannot name real content.
    pub const NONE: Self = Self(0);

    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
}

/// One block as a store names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub hash: BlockHash,
    pub content: ContentHash,
}

impl BlockKey {
    pub const fn new(hash: BlockHash, content: ContentHash) -> Self {
        Self { hash, content }
    }

    /// A block with no content hash: held in the ledger, never projected into an index.
    pub const fn opaque(hash: BlockHash) -> Self {
        Self {
            hash,
            content: ContentHash::NONE,
        }
    }
}

/// What a store says about the block before its first block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Parent {
    /// The first block of a sequence.
    Root,
    /// The block that precedes it.
    Block(BlockHash),
    /// The store does not say: a token-less placeholder, or a store whose hashes do not line up
    /// with its tokens. Exported as an orphan; a later store that knows the parent fills it in.
    Unknown,
}

/// One uninterrupted run of an engine's batch sequence, as this host saw it.
///
/// The host mints it (a random value per engine connection is enough) and changes it whenever
/// the sequence can no longer be trusted to continue: engine restart, an unrecoverable gap, a
/// reseed. A host never mints [`Epoch::NONE`], so it can map an unparsable cursor to it and
/// have that cursor refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Epoch(pub u128);

impl Epoch {
    /// No epoch: the value of a cursor that names none.
    pub const NONE: Self = Self(0);
}

/// A consumer's position: every batch below `next_seq` of `epoch` has been applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cursor {
    pub epoch: Epoch,
    pub next_seq: u64,
}
