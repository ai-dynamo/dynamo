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
