// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard types that enforce the block lifecycle state machine.
//!
//! Every logical block in KVBM progresses through a fixed sequence of states.
//! The guard types in this module make those transitions explicit in the type
//! system: each state is represented by a distinct Rust type, and moving to the
//! next state consumes the current guard and produces the next one. Dropping a
//! guard at any point automatically returns the underlying block to the
//! appropriate pool, so blocks are never leaked. This ensures proper behavior
//! on early exits or exception handling.
//!
//! # State machine
//!
//! ```text
//!                  stage / complete          register_block
//!   Reset ──────────────────────► Staged ─────────────────► Registered
//!     ▲                              │                          │
//!     │          reset               │                          │
//!     ├──────────────────────────────┘                          │
//!     │                          drop (reset + return)          │
//!     └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Public guard types
//!
//! - [`MutableBlock`] -- guards a block in the **Reset** state.
//!   - Created by [`crate::BlockManager::allocate_blocks`] or [`CompleteBlock::reset`].
//!   - Transitions to [`CompleteBlock`] via [`stage`](MutableBlock::stage)
//!     or [`complete`](MutableBlock::complete).
//!
//! - [`CompleteBlock`] -- guards a block in the **Staged** state.
//!   - Created by [`MutableBlock::stage`] or [`MutableBlock::complete`].
//!   - Transitions to [`MutableBlock`] via [`reset`](CompleteBlock::reset),
//!     or to [`ImmutableBlock`] via [`crate::BlockManager::register_block`].
//!
//! - [`ImmutableBlock`] -- guards a block in the **Registered** state.
//!   - Created by [`crate::BlockManager::register_block`],
//!     [`crate::BlockManager::match_blocks`], [`crate::BlockManager::scan_matches`],
//!     or [`WeakBlock::upgrade`].
//!   - Transitions to [`WeakBlock`] via [`downgrade`](ImmutableBlock::downgrade).
//!
//! - [`WeakBlock`] -- non-owning reference to a registered block.
//!   - Created by [`ImmutableBlock::downgrade`].
//!   - Transitions to [`ImmutableBlock`] via [`upgrade`](WeakBlock::upgrade).
//!
//! # Supporting types
//!
//! - [`BlockMetadata`] -- trait bound satisfied by any `Clone + Send + Sync + 'static` type.
//! - [`BlockError`] -- error type that returns the originating block to prevent leaks.
//! - [`BlockDuplicationPolicy`] -- controls whether duplicate sequence hashes are allowed
//!   at registration time.
//!
//! # Internal types (crate-visible only)
//!
//! `PrimaryBlock` and `DuplicateBlock` are RAII guards for the Registered state,
//! used internally by the [`BlockRegistry`] to
//! distinguish between the canonical holder of a sequence hash and any additional
//! logical copies.

mod complete;
mod immutable;
mod mutable;
mod registered;

pub use complete::CompleteBlock;
pub use immutable::{ImmutableBlock, WeakBlock};
pub use mutable::MutableBlock;

pub(crate) mod state;
pub(crate) use registered::{DuplicateBlock, PrimaryBlock, WeakBlockEntry};

// Re-export from the new registry module location for backward compatibility
pub use crate::registry::BlockRegistrationHandle;
pub use crate::registry::BlockRegistry;

/// Marker trait for types that can serve as block-level metadata.
///
/// A blanket implementation covers every type that is
/// `Clone + Send + Sync + 'static`, so callers rarely need to think about
/// this trait directly -- any ordinary data type already satisfies it.
pub trait BlockMetadata: Clone + Send + Sync + 'static {}
impl<T: Clone + Send + Sync + 'static> BlockMetadata for T {}

/// Logical Block Identifier
pub use crate::{BlockId, SequenceHash};
use dynamo_tokens::TokenBlock;
use std::sync::Arc;

/// Return function for blocks transitioning back to Reset state.
/// Used by MutableBlock, DuplicateBlock, and CompleteBlock drop paths.
pub(crate) type ResetReturnFn<T> = Arc<dyn Fn(Block<T, state::Reset>) + Send + Sync>;

/// Return function for registered blocks returning to the inactive pool.
/// Used by PrimaryBlock drop and pool management.
pub(crate) type RegisteredReturnFn<T> = Arc<dyn Fn(Arc<Block<T, state::Registered>>) + Send + Sync>;

/// Upgrade function for finding/promoting blocks by sequence hash.
/// Used by ImmutableBlock and WeakBlock upgrade paths.
pub(crate) type UpgradeFn<T> =
    Arc<dyn Fn(SequenceHash) -> Option<Arc<dyn RegisteredBlock<T>>> + Send + Sync>;

/// Error returned by block state transitions.
///
/// Every variant carries the originating block back to the caller so that
/// the block is never silently leaked on failure. The caller can inspect the
/// error, recover the block from the variant, and retry or drop it as needed.
#[derive(Debug, thiserror::Error)]
pub enum BlockError<B> {
    /// The number of tokens in the provided data did not match the block's
    /// fixed size. The block is returned in the `block` field.
    #[error("Block size mismatch: expected {expected} tokens, got {actual}")]
    BlockSizeMismatch {
        expected: usize,
        actual: usize,
        block: B,
    },
}

/// Controls whether the [`BlockRegistry`] accepts
/// multiple physical blocks that share the same [`SequenceHash`].
///
/// The policy is set once when a
/// [`BlockManager`](crate::manager::BlockManager) is built and applies to
/// every subsequent registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDuplicationPolicy {
    /// Multiple physical blocks may hold the same data for a single logical
    /// block / sequence hash.
    ///
    /// This is the typical choice for G1 / device memory in LLM inference
    /// frameworks.
    Allow,
    /// Only one physical block is retained per sequence hash. Attempting to
    /// register a duplicate returns the existing primary block instead.
    ///
    /// Used internally by KVBM for G2+ storage layers (host, disk,
    /// distributed, object store) where deduplication saves capacity.
    Reject,
}

/// The raw block value parameterised by metadata `T` and a type-state marker.
///
/// External code never sees this type directly; it is always wrapped in one of
/// the public RAII guards ([`MutableBlock`], [`CompleteBlock`], [`ImmutableBlock`]).
#[derive(Debug)]
pub(crate) struct Block<T, State> {
    block_id: BlockId,
    block_size: usize,
    state: State,
    marker: std::marker::PhantomData<T>,
}

/// Common interface for blocks in the Registered state.
///
/// Both [`PrimaryBlock`] and [`DuplicateBlock`] implement this trait so that
/// [`ImmutableBlock`] can hold either variant behind a `dyn RegisteredBlock<T>`.
pub(crate) trait RegisteredBlock<T>: Send + Sync {
    fn block_id(&self) -> BlockId;
    fn sequence_hash(&self) -> SequenceHash;
    fn registration_handle(&self) -> &BlockRegistrationHandle;
}
