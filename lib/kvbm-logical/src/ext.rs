// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extension points for custom block lifecycle strategies.
//!
//! - Use [`BlockToken`] (obtained via [`BlockRegistry::token()`](crate::registry::BlockRegistry::token))
//!   together with [`make_mutable_block`] and [`make_immutable_block`] to construct
//!   RAII guard types from external block managers without accessing `pub(crate)` internals.
//! - Implement [`RegisteredBlock`] to provide custom registered block types.

use std::sync::Arc;

pub use crate::BlockId;
pub use crate::SequenceHash;
pub use crate::blocks::Block;
pub use crate::blocks::state::Reset;
pub use crate::blocks::{BlockDuplicationPolicy, BlockError, BlockMetadata};
pub use crate::blocks::{CompleteBlock, ImmutableBlock, RegisteredBlock, ResetReturnFn, UpgradeFn};

/// Sealed capability token proving the holder went through a properly-constructed
/// [`BlockRegistry`](crate::registry::BlockRegistry).
///
/// The constructor is `pub(crate)` — only kvbm-logical internals can mint one.
/// External crates obtain a `BlockToken` via [`BlockRegistry::token()`](crate::registry::BlockRegistry::token).
pub struct BlockToken(pub(crate) ());

/// Construct a [`MutableBlock<T>`] from a raw `Block<T, Reset>` and a drop closure.
///
/// The `on_drop` closure is called when the MutableBlock is dropped (either explicitly
/// or via RAII), receiving the inner `Block<T, Reset>` back. Typically used to free
/// the block in an external allocator (e.g., `fc::free_blocks()`).
///
/// Metrics are set to `None` — external block managers handle their own metrics.
pub fn make_mutable_block<T: BlockMetadata>(
    _token: &BlockToken,
    block: Block<T, Reset>,
    on_drop: impl Fn(Block<T, Reset>) + Send + Sync + 'static,
) -> crate::blocks::MutableBlock<T> {
    crate::blocks::MutableBlock::new(block, Arc::new(on_drop), None)
}

/// Consume a [`CompleteBlock`], extracting its identity without triggering the drop handler.
///
/// This is used by external block managers that handle registration themselves
/// (e.g., via an external cache system) and need to prevent the CompleteBlock's
/// RAII return-to-pool behavior from firing.
pub fn consume_complete_block<T: BlockMetadata>(
    _token: &BlockToken,
    mut block: CompleteBlock<T>,
) -> (BlockId, SequenceHash) {
    let id = block.block_id();
    let hash = block.sequence_hash();
    block.block.take(); // prevent Drop from firing return_fn
    (id, hash)
}

/// Construct an [`ImmutableBlock<T>`] from a registered block and an upgrade function.
///
/// The `registered_block` is obtained from [`crate::registry::BlockRegistrationHandle::register_block`].
/// The `upgrade_fn` closure is called by [`crate::blocks::WeakBlock::upgrade`] slow path — pass
/// `Arc::new(|_| None)` for Phase 2 (FC-based upgrade is Phase 3).
///
/// Metrics are set to `None` — external block managers handle their own metrics.
pub fn make_immutable_block<T: BlockMetadata>(
    _token: &BlockToken,
    registered_block: Arc<dyn RegisteredBlock<T>>,
    upgrade_fn: UpgradeFn<T>,
) -> crate::blocks::ImmutableBlock<T> {
    crate::blocks::ImmutableBlock::new(registered_block, upgrade_fn, None)
}
