// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block pool bookkeeping for thread-safe block management.
//!
//! - [`BlockStore<T>`]: unified single-mutex owner of reset and inactive
//!   pool state.
//! - [`InactiveIndex`]: pluggable T-free eviction-order trait for the
//!   inactive pool (crate-private).
//! - Type-safe RAII guards (`MutableBlock`, `CompleteBlock`,
//!   `ImmutableBlock`, `WeakBlock`) live in `crate::blocks` and return to
//!   the store on drop.

mod inactive;
pub(crate) mod store;

#[cfg(test)]
pub mod tests;

#[cfg(test)]
mod block_proptest;

pub(crate) use inactive::backends;
pub(crate) use store::{BlockStore, InactiveIndex};

pub(crate) use crate::SequenceHash;
use crate::blocks::BlockId;

// Re-export block duplication policy
pub use crate::blocks::BlockDuplicationPolicy;

/// A block that is free and available for allocation in the inactive pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InactiveBlock {
    pub block_id: BlockId,
    pub seq_hash: SequenceHash,
}

// ---------------------------------------------------------------------------
// Identity hashing for `SequenceHash`-keyed maps
// ---------------------------------------------------------------------------

use std::collections::HashMap;

/// Identity hasher for `SequenceHash` keys.
///
/// `SequenceHash` (`dynamo_tokens::PositionalLineageHash`) is a
/// `#[derive(Hash)]` transparent newtype over a `u128` content hash —
/// already uniformly mixed. Running SipHash over it is wasted work on the
/// KV-cache hot path: `BlockStore::active_by_hash` and every
/// `SequenceHash`-keyed inactive-pool backend (`LruBackend`,
/// `MultiLruBackend`, `HashMapBackend`) do these lookups under the store
/// mutex. This hasher folds the 128-bit key into a `u64` with one XOR.
///
/// The derived `Hash` for the newtype forwards to `u128::hash`, which calls
/// `Hasher::write_u128` — so `write` (the byte-slice path) is never
/// exercised for these keys; it is `unreachable!` to catch a key-type
/// change that would silently make this hasher lossy.
#[derive(Default)]
pub(crate) struct IdHasher(u64);

impl std::hash::Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write_u128(&mut self, v: u128) {
        self.0 = (v as u64) ^ ((v >> 64) as u64);
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("SequenceHash hashes via write_u128, not the byte-slice path");
    }
}

#[derive(Default, Clone)]
pub(crate) struct IdBuildHasher;

impl std::hash::BuildHasher for IdBuildHasher {
    type Hasher = IdHasher;
    fn build_hasher(&self) -> IdHasher {
        IdHasher::default()
    }
}

/// `HashMap` keyed by `SequenceHash` using the identity [`IdHasher`].
pub(crate) type SeqHashMap<V> = HashMap<SequenceHash, V, IdBuildHasher>;
