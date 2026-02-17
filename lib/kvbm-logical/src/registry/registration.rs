// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block registration logic: register_block, try_find_existing_block, try_get_block.

use super::RegisteredReturnFn;
use super::attachments::AttachmentStore;
use super::handle::BlockRegistrationHandle;

use crate::blocks::{
    Block, BlockDuplicationPolicy, BlockMetadata, CompleteBlock, DuplicateBlock, PrimaryBlock,
    RegisteredBlock, WeakBlockEntry,
    state::{Registered, Reset, Staged},
};
use crate::metrics::BlockPoolMetrics;
use crate::pools::InactivePool;

use std::any::TypeId;
use std::sync::{Arc, Weak};

impl BlockRegistrationHandle {
    pub(crate) fn register_block<T: BlockMetadata + Sync>(
        &self,
        mut block: CompleteBlock<T>,
        duplication_policy: BlockDuplicationPolicy,
        inactive_pool: &InactivePool<T>,
        metrics: Option<&BlockPoolMetrics>,
    ) -> Arc<dyn RegisteredBlock<T>> {
        assert_eq!(
            block.sequence_hash(),
            self.seq_hash(),
            "Attempted to register block with different sequence hash"
        );

        let block_id = block.block_id();
        let inner_block = block.block.take().unwrap();
        let reset_return_fn = block.return_fn.clone();

        register_block_inner(
            self,
            inner_block,
            block_id,
            reset_return_fn,
            duplication_policy,
            inactive_pool,
            metrics,
        )
    }

    #[inline]
    pub(crate) fn try_get_block<T: BlockMetadata + Sync>(
        &self,
        pool_return_fn: RegisteredReturnFn<T>,
    ) -> Option<Arc<dyn RegisteredBlock<T>>> {
        let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
        let attachments = self.inner.attachments.lock();

        let weak_block = attachments
            .weak_blocks
            .get(&type_id)
            .and_then(|weak_any| weak_any.downcast_ref::<WeakBlockEntry<T>>())?;

        if let Some(primary_arc) = weak_block.primary_block.upgrade() {
            drop(attachments);
            return Some(primary_arc as Arc<dyn RegisteredBlock<T>>);
        }

        if let Some(raw_arc) = weak_block.raw_block.upgrade() {
            drop(attachments);
            let primary_arc = PrimaryBlock::new_attached(raw_arc, pool_return_fn);
            return Some(primary_arc as Arc<dyn RegisteredBlock<T>>);
        }

        None
    }
}

/// Core registration logic for register_block.
fn register_block_inner<T: BlockMetadata + Sync>(
    handle: &BlockRegistrationHandle,
    block: Block<T, Staged>,
    block_id: crate::BlockId,
    reset_return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
    duplication_policy: BlockDuplicationPolicy,
    inactive_pool: &InactivePool<T>,
    metrics: Option<&BlockPoolMetrics>,
) -> Arc<dyn RegisteredBlock<T>> {
    let pool_return_fn = inactive_pool.return_fn();

    // CRITICAL: Check for existing blocks BEFORE registering.
    // register()/register_with_handle() calls mark_present::<T>() which would make
    // has_block::<T>() always return true.
    let attachments = handle.inner.attachments.lock();

    // Check for existing block (handles race condition with retry loop)
    if let Some(existing_primary) = try_find_existing_block(handle, inactive_pool, &attachments) {
        // Check if same block_id (shouldn't happen)
        if existing_primary.block_id() == block_id {
            panic!("Attempted to register block with same block_id as existing");
        }

        // Handle duplicate based on policy
        match duplication_policy {
            BlockDuplicationPolicy::Allow => {
                if let Some(m) = metrics {
                    m.inc_duplicate_blocks();
                }
                drop(attachments);
                PrimaryBlock::store_weak_refs(&existing_primary);
                let registered_block = block.register_with_handle(handle.clone());
                let duplicate =
                    DuplicateBlock::new(registered_block, existing_primary, reset_return_fn);
                return Arc::new(duplicate);
            }
            BlockDuplicationPolicy::Reject => {
                if let Some(m) = metrics {
                    m.inc_registration_dedup();
                }
                drop(attachments);
                PrimaryBlock::store_weak_refs(&existing_primary);
                reset_return_fn(block.reset());
                return existing_primary as Arc<dyn RegisteredBlock<T>>;
            }
        }
    }

    // No existing block - register and create new primary
    drop(attachments);
    let registered_block = block.register_with_handle(handle.clone());
    let primary_arc = PrimaryBlock::new_attached(Arc::new(registered_block), pool_return_fn);

    primary_arc as Arc<dyn RegisteredBlock<T>>
}

/// Try to find an existing block with the same sequence hash.
/// Handles race conditions where block may be transitioning between pools.
fn try_find_existing_block<T: BlockMetadata + Sync>(
    handle: &BlockRegistrationHandle,
    inactive_pool: &InactivePool<T>,
    attachments: &AttachmentStore,
) -> Option<Arc<PrimaryBlock<T>>> {
    let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
    const MAX_RETRIES: usize = 100;
    let mut retry_count = 0;

    loop {
        // Check presence first
        if !attachments
            .presence_markers
            .contains_key(&TypeId::of::<T>())
        {
            tracing::debug!(
                seq_hash = %handle.seq_hash(),
                "try_find_existing_block: no presence marker, returning None"
            );
            return None;
        }

        // Try active pool (weak reference)
        if let Some(weak_any) = attachments.weak_blocks.get(&type_id)
            && let Some(weak_block) = weak_any.downcast_ref::<WeakBlockEntry<T>>()
            && let Some(existing_primary) = weak_block.primary_block.upgrade()
        {
            tracing::debug!(
                seq_hash = %handle.seq_hash(),
                block_id = existing_primary.block_id(),
                "try_find_existing_block: found in active pool"
            );
            return Some(existing_primary);
        }

        // Try inactive pool - this acquires the inactive pool lock.
        // find_block_as_primary uses new_unattached because we hold the attachments lock.
        // The caller (register_block_inner) calls store_weak_refs after dropping the lock.
        if let Some(promoted) = inactive_pool.find_block_as_primary(handle.seq_hash(), false) {
            tracing::debug!(
                seq_hash = %handle.seq_hash(),
                block_id = promoted.block_id(),
                "try_find_existing_block: found in inactive pool, promoted"
            );
            return Some(promoted);
        }

        // Block is present but not found in either pool - it's transitioning.
        retry_count += 1;
        if retry_count >= MAX_RETRIES {
            tracing::warn!(
                seq_hash = %handle.seq_hash(),
                retries = retry_count,
                "try_find_existing_block: max retries exceeded, presence marker set but block not found in either pool"
            );
            return None;
        }

        // Brief yield to allow other thread to complete transition
        std::hint::spin_loop();
    }
}
