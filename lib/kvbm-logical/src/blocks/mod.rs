// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard types that enforce the block lifecycle state machine.
//!
//! Every logical block in KVBM progresses through a fixed sequence of
//! states. The guard types in this module make those transitions explicit
//! at the type level: each transition consumes one guard and produces the
//! next. Dropping a guard at any point automatically returns the
//! underlying slot to the appropriate pool, so blocks are never leaked.
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
//! - [`MutableBlock`] — `Reset` state.
//! - [`CompleteBlock`] — `Staged` state.
//! - [`ImmutableBlock`] — `Registered` state (cheap-to-clone strong handle).
//! - [`WeakBlock`] — non-owning reference to a registered block.
//!
//! Slot bookkeeping lives in [`BlockStore<T>`](crate::pools::BlockStore);
//! `Primary` vs `Duplicate` is captured by an internal flag on
//! [`ImmutableBlockInner`] and the corresponding slot state.

mod complete;
mod immutable;
mod mutable;
mod pin;

pub use complete::CompleteBlock;
pub use immutable::{ImmutableBlock, WeakBlock};
pub use mutable::MutableBlock;
pub use pin::{LifecyclePin, LifecyclePinRef};

pub(crate) use immutable::ImmutableBlockInner;

pub use crate::registry::BlockRegistrationHandle;
pub use crate::registry::BlockRegistry;

pub use crate::{BlockId, SequenceHash};

/// Marker trait for types that can serve as block-level metadata.
///
/// A blanket implementation covers every type that is `Clone + Send + Sync
/// + 'static`, so callers rarely need to think about this trait directly.
pub trait BlockMetadata: Clone + Send + Sync + 'static {}
impl<T: Clone + Send + Sync + 'static> BlockMetadata for T {}

/// Error returned by block state transitions.
///
/// Every variant carries the originating block back to the caller so that
/// the block is never silently leaked on failure. The caller can inspect
/// the error, recover the block from the variant, and retry or drop it as
/// needed.
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

/// Controls whether the [`BlockRegistry`] accepts multiple physical blocks
/// that share the same [`SequenceHash`].
///
/// The policy is set once when a [`BlockManager`](crate::manager::BlockManager)
/// is built and applies to every subsequent registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDuplicationPolicy {
    /// Multiple physical blocks may hold the same data for a single
    /// logical block / sequence hash.
    Allow,
    /// Only one physical block is retained per sequence hash. Attempting
    /// to register a duplicate returns the existing primary block instead.
    Reject,
}
