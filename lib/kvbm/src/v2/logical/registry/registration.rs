// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block registration logic: register_block, register_mutable_block, try_find_existing_block,
//! try_get_block, attach_block, attach_block_ref.

use super::attachments::AttachmentStore;
use super::handle::BlockRegistrationHandle;
use super::RegisteredReturnFn;

use crate::v2::logical::blocks::{
    Block, BlockDuplicationPolicy, BlockMetadata, CompleteBlock, DuplicateBlock, MutableBlock,
    PrimaryBlock, RegisteredBlock,
    state::{Registered, Reset},
};
use crate::v2::logical::pools::InactivePool;

use std::any::TypeId;
use std::sync::{Arc, Weak};

/// Weak references to a block for resurrection during pool transitions.
pub(crate) struct WeakBlockEntry<T: BlockMetadata + Sync> {
    /// Weak reference to the raw block
    pub(crate) raw_block: Weak<Block<T, Registered>>,
    /// Weak reference to the registered block
    pub(crate) primary_block: Weak<PrimaryBlock<T>>,
}

/// Extracted block ready for registration (from either CompleteBlock or MutableBlock).
/// Used to DRY the shared logic between register_block and register_mutable_block.
enum ExtractedBlock<T: BlockMetadata> {
    Complete(Block<T, crate::v2::logical::blocks::state::Complete>),
    Reset(Block<T, Reset>),
}

impl<T: BlockMetadata> ExtractedBlock<T> {
    fn register(self, handle: BlockRegistrationHandle) -> Block<T, Registered> {
        match self {
            Self::Complete(inner) => inner.register(handle),
            Self::Reset(inner) => inner.register_with_handle(handle),
        }
    }

    fn discard(self, return_fn: &Arc<dyn Fn(Block<T, Reset>) + Send + Sync>) {
        match self {
            Self::Complete(inner) => return_fn(inner.reset()),
            Self::Reset(inner) => return_fn(inner),
        }
    }
}

/// Shared registration logic used by both register_block and register_mutable_block.
fn register_block_inner<T: BlockMetadata + Sync>(
    handle: &BlockRegistrationHandle,
    extracted: ExtractedBlock<T>,
    block_id: crate::v2::BlockId,
    reset_return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
    duplication_policy: BlockDuplicationPolicy,
    inactive_pool: &InactivePool<T>,
) -> Arc<dyn RegisteredBlock<T>> {
    let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
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
                drop(attachments);
                attach_block_ref(handle, &existing_primary);
                let registered_block = extracted.register(handle.clone());
                let duplicate =
                    DuplicateBlock::new(registered_block, existing_primary, reset_return_fn);
                return Arc::new(duplicate);
            }
            BlockDuplicationPolicy::Reject => {
                drop(attachments);
                attach_block_ref(handle, &existing_primary);
                extracted.discard(&reset_return_fn);
                return existing_primary as Arc<dyn RegisteredBlock<T>>;
            }
        }
    }

    // No existing block - register and create new primary
    drop(attachments);
    let registered_block = extracted.register(handle.clone());

    let primary = PrimaryBlock::new(Arc::new(registered_block), pool_return_fn);

    // Store weak references for future lookups
    let primary_arc = Arc::new(primary);
    let raw_block = Arc::downgrade(primary_arc.block.as_ref().unwrap());
    let primary_block = Arc::downgrade(&primary_arc);

    let mut attachments = handle.inner.attachments.lock();
    attachments.weak_blocks.insert(
        type_id,
        Box::new(WeakBlockEntry {
            raw_block,
            primary_block,
        }),
    );

    drop(attachments);

    primary_arc as Arc<dyn RegisteredBlock<T>>
}

impl BlockRegistrationHandle {
    pub(crate) fn register_block<T: BlockMetadata + Sync>(
        &self,
        mut block: CompleteBlock<T>,
        duplication_policy: BlockDuplicationPolicy,
        inactive_pool: &InactivePool<T>,
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
            ExtractedBlock::Complete(inner_block),
            block_id,
            reset_return_fn,
            duplication_policy,
            inactive_pool,
        )
    }

    pub(crate) fn register_mutable_block<T: BlockMetadata + Sync>(
        &self,
        mutable_block: MutableBlock<T>,
        duplication_policy: BlockDuplicationPolicy,
        inactive_pool: &InactivePool<T>,
    ) -> Arc<dyn RegisteredBlock<T>> {
        let block_id = mutable_block.block_id();
        let (inner_block, reset_return_fn) = mutable_block.into_parts();

        register_block_inner(
            self,
            ExtractedBlock::Reset(inner_block),
            block_id,
            reset_return_fn,
            duplication_policy,
            inactive_pool,
        )
    }

    pub(crate) fn attach_block<T: BlockMetadata + Sync>(
        &self,
        block: PrimaryBlock<T>,
    ) -> Arc<dyn RegisteredBlock<T>> {
        let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
        let mut attachments = self.inner.attachments.lock();

        #[cfg(debug_assertions)]
        {
            if let Some(weak_any) = attachments.weak_blocks.get(&type_id)
                && let Some(weak) = weak_any.downcast_ref::<WeakBlockEntry<T>>()
            {
                debug_assert!(
                    weak.raw_block.upgrade().is_none(),
                    "Attempted to reattach block when raw block is still alive"
                );
                debug_assert!(
                    weak.primary_block.upgrade().is_none(),
                    "Attempted to reattach block when registered block is still alive"
                );
            }
        }

        let raw_block = Arc::downgrade(block.block.as_ref().unwrap());
        let reg_arc = Arc::new(block);
        let primary_block = Arc::downgrade(&reg_arc);

        attachments.weak_blocks.insert(
            type_id,
            Box::new(WeakBlockEntry {
                raw_block,
                primary_block,
            }),
        );

        reg_arc as Arc<dyn RegisteredBlock<T>>
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
            let primary = PrimaryBlock::new(raw_arc, pool_return_fn);
            let primary_arc = Arc::new(primary);

            let new_weak = Arc::downgrade(&primary_arc);
            let weak_block_mut = WeakBlockEntry {
                raw_block: weak_block.raw_block.clone(),
                primary_block: new_weak,
            };

            drop(attachments);

            let mut attachments = self.inner.attachments.lock();
            attachments
                .weak_blocks
                .insert(type_id, Box::new(weak_block_mut));
            drop(attachments);

            return Some(primary_arc as Arc<dyn RegisteredBlock<T>>);
        }

        None
    }
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
        {
            if let Some(existing_primary) = weak_block.primary_block.upgrade() {
                tracing::debug!(
                    seq_hash = %handle.seq_hash(),
                    block_id = existing_primary.block_id(),
                    "try_find_existing_block: found in active pool"
                );
                return Some(existing_primary);
            }
        }

        // Try inactive pool - this acquires the inactive pool lock
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

/// Attach a weak reference to an existing PrimaryBlock for future lookups.
fn attach_block_ref<T: BlockMetadata + Sync>(
    handle: &BlockRegistrationHandle,
    primary_arc: &Arc<PrimaryBlock<T>>,
) {
    let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
    let mut attachments = handle.inner.attachments.lock();

    let raw_block = Arc::downgrade(primary_arc.block.as_ref().unwrap());
    let primary_block = Arc::downgrade(primary_arc);

    attachments.weak_blocks.insert(
        type_id,
        Box::new(WeakBlockEntry {
            raw_block,
            primary_block,
        }),
    );
}
