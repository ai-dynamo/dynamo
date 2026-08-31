// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block registration logic.
//!
//! Implemented as inherent methods on [`BlockRegistrationHandle`] that
//! delegate to the unified [`BlockStore`]. Lookup and slot transition
//! happen under a single store-mutex acquisition, closing the
//! register-vs-register race for the same sequence hash.

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
        store.register_completed_block(block, self.clone(), duplication_policy)
    }

    /// Try to fetch a strong [`ImmutableBlockInner`] for this handle —
    /// either an alive existing inner (fast path) or a resurrected one
    /// pulled out of the inactive pool (slow path). `touch` is forwarded
    /// to the inactive resurrection path so frequency tracking sees the
    /// hit even when an active-pool lookup absorbs an inactive entry.
    #[inline]
    pub(crate) fn try_get_inner<T: BlockMetadata + Sync>(
        &self,
        store: &Arc<BlockStore<T>>,
        touch: bool,
    ) -> Option<Arc<ImmutableBlockInner<T>>> {
        upgrade_or_resurrect::<T>(self, store, touch)
    }
}
