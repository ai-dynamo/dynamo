// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block registration logic.
//!
//! Implemented as inherent methods on [`BlockRegistrationHandle`] that
//! interact with the unified [`BlockStore`]. The previous dual-weak-ref
//! resurrection scheme is gone — the single store mutex serializes drop
//! and resurrection, so a single `Weak<ImmutableBlockInner<T>>` per handle
//! is sufficient.

use std::sync::Arc;

use super::handle::BlockRegistrationHandle;

use crate::blocks::{BlockDuplicationPolicy, BlockMetadata, CompleteBlock, ImmutableBlockInner};
use crate::pools::BlockStore;
use crate::pools::store::upgrade_or_resurrect;

impl BlockRegistrationHandle {
    /// Register a [`CompleteBlock`] with this handle, returning the
    /// `Arc<ImmutableBlockInner<T>>` that backs the resulting public
    /// [`ImmutableBlock`](crate::blocks::ImmutableBlock).
    pub(crate) fn register_block<T: BlockMetadata + Sync>(
        &self,
        block: CompleteBlock<T>,
        duplication_policy: BlockDuplicationPolicy,
        store: &Arc<BlockStore<T>>,
    ) -> Arc<ImmutableBlockInner<T>> {
        assert_eq!(
            block.sequence_hash(),
            self.seq_hash(),
            "Attempted to register block with different sequence hash"
        );

        let block_id = block.block_id();
        let seq_hash = block.sequence_hash();

        // Look for an existing primary (live in active or evictable in inactive).
        let existing = upgrade_or_resurrect::<T>(self, store);

        if let Some(existing_primary) = existing {
            if existing_primary.block_id() == block_id {
                panic!("Attempted to register block with same block_id as existing");
            }
            match duplication_policy {
                BlockDuplicationPolicy::Allow => {
                    // Disarm the CompleteBlock guard — store will own the
                    // slot via the new Duplicate state. `transition_to_duplicate`
                    // emits `inc_duplicate_blocks` inside its critical section.
                    let mut block = block;
                    block.armed = false;
                    drop(block);
                    store.transition_to_duplicate(
                        block_id,
                        seq_hash,
                        self.clone(),
                        existing_primary,
                    )
                }
                BlockDuplicationPolicy::Reject => {
                    store.metrics().inc_registration_dedup();
                    // Drop the CompleteBlock — it returns the slot to Reset.
                    drop(block);
                    existing_primary
                }
            }
        } else {
            // Fresh primary.
            let mut block = block;
            block.armed = false;
            drop(block);
            store.transition_to_primary(block_id, seq_hash, self.clone())
        }
    }

    /// Try to fetch a strong [`ImmutableBlockInner`] for this handle —
    /// either an alive existing inner (fast path) or a resurrected one
    /// pulled out of the inactive pool (slow path).
    #[inline]
    pub(crate) fn try_get_inner<T: BlockMetadata + Sync>(
        &self,
        store: &Arc<BlockStore<T>>,
    ) -> Option<Arc<ImmutableBlockInner<T>>> {
        upgrade_or_resurrect::<T>(self, store)
    }
}
