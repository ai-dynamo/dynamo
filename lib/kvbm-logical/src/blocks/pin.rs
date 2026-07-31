// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Type-erased lifecycle pin for cross-policy block references.
//!
//! [`LifecyclePinRef`] is a thin wrapper around `Arc<dyn LifecyclePin>`
//! obtained from [`ImmutableBlock::pin`](super::ImmutableBlock::pin). It
//! serves two purposes for downstream callers (e.g., a KV offload
//! connector that must stash a heterogeneous list of in-flight transfers):
//!
//! 1. **Keepalive.** As long as the pin is alive, the underlying slot
//!    will not transition from `Active` to `Inactive` and will not be
//!    recycled to a different sequence hash. Cloning the pin is one
//!    `Arc::clone` — no extra heap allocation beyond the original
//!    `ImmutableBlockInner`.
//! 2. **Logical address.** The pin exposes
//!    `(manager_id, block_id, sequence_hash)`, which uniquely identifies
//!    the slot at runtime even after the originating
//!    `ImmutableBlock<T>`'s metadata parameter `T` has been type-erased
//!    away. Two pins with the same `manager_id` come from the same
//!    physical pool.
//!
//! kvbm-logical does **not** own bytes — the pin keeps the *logical*
//! lifecycle alive. The executor / runtime owning the physical buffer
//! is responsible for honoring the `(manager_id, block_id)` address as
//! the lookup key into its own per-pool storage.

use std::sync::Arc;

use crate::ManagerId;
use crate::blocks::{BlockId, BlockRegistrationHandle, SequenceHash};

/// Type-erased view of a registered slot's lifecycle. Implemented on the
/// crate-private `ImmutableBlockInner<T>` so the policy parameter `T`
/// drops out at the trait-object boundary.
pub trait LifecyclePin: Send + Sync {
    fn block_id(&self) -> BlockId;
    fn sequence_hash(&self) -> SequenceHash;
    fn manager_id(&self) -> ManagerId;
    fn registration_handle(&self) -> BlockRegistrationHandle;
}

/// Cheap-to-clone strong reference to a registered slot's lifecycle.
///
/// Each `Clone` is a single `Arc::clone`. Drop releases the keepalive;
/// when the last pin (and the last `ImmutableBlock` clone) are dropped,
/// the underlying slot transitions to `Inactive` and may be evicted.
#[derive(Clone)]
pub struct LifecyclePinRef(Arc<dyn LifecyclePin>);

impl LifecyclePinRef {
    pub(crate) fn new(inner: Arc<dyn LifecyclePin>) -> Self {
        Self(inner)
    }

    pub fn block_id(&self) -> BlockId {
        self.0.block_id()
    }

    pub fn sequence_hash(&self) -> SequenceHash {
        self.0.sequence_hash()
    }

    pub fn manager_id(&self) -> ManagerId {
        self.0.manager_id()
    }

    pub fn registration_handle(&self) -> BlockRegistrationHandle {
        self.0.registration_handle()
    }

    /// Strong-count of the underlying `Arc<dyn LifecyclePin>`. Useful for
    /// metrics and tests; do not branch on this in production code.
    pub fn use_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

impl std::fmt::Debug for LifecyclePinRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LifecyclePinRef")
            .field("manager_id", &self.0.manager_id())
            .field("block_id", &self.0.block_id())
            .field("sequence_hash", &self.0.sequence_hash())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::ManagerId;
    use crate::testing::{TestMeta, create_iota_token_block, create_test_manager};

    /// Helper: register a single block in `manager` and return both the
    /// resulting `ImmutableBlock` and the `ManagerId` derived from its pin.
    fn one_block_and_manager_id(
        manager: &crate::BlockManager<TestMeta>,
        salt: u32,
    ) -> (crate::ImmutableBlock<TestMeta>, ManagerId) {
        let token = create_iota_token_block(salt, 4);
        let mb = manager
            .allocate_blocks(1)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let cb = mb.complete(&token).unwrap();
        let block = manager
            .register_blocks(vec![cb])
            .into_iter()
            .next()
            .unwrap();
        let id = block.pin().manager_id();
        (block, id)
    }

    #[test]
    fn manager_id_unique_across_managers() {
        let mgr_a = create_test_manager::<TestMeta>(4);
        let mgr_b = create_test_manager::<TestMeta>(4);
        let (_a, id_a) = one_block_and_manager_id(&mgr_a, 1);
        let (_b, id_b) = one_block_and_manager_id(&mgr_b, 2);
        assert_ne!(id_a, id_b);
        assert_ne!(id_a, ManagerId::NULL);
        assert_ne!(id_b, ManagerId::NULL);
    }

    #[test]
    fn manager_id_stable_across_clones() {
        let manager = create_test_manager::<TestMeta>(4);
        let (block, manager_id) = one_block_and_manager_id(&manager, 0);

        let pin1 = block.pin();
        let pin2 = block.clone().pin();
        let pin3 = pin1.clone();

        assert_eq!(pin1.manager_id(), manager_id);
        assert_eq!(pin2.manager_id(), manager_id);
        assert_eq!(pin3.manager_id(), manager_id);
        assert_eq!(pin1.block_id(), block.block_id());
        assert_eq!(pin1.sequence_hash(), block.sequence_hash());
    }

    #[test]
    fn pin_keeps_slot_active_after_block_drops() {
        let manager = create_test_manager::<TestMeta>(4);
        let initial = manager.available_blocks();
        let token = create_iota_token_block(100, 4);
        let mb = manager
            .allocate_blocks(1)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let cb = mb.complete(&token).unwrap();
        let block = manager
            .register_blocks(vec![cb])
            .into_iter()
            .next()
            .unwrap();

        // Slot now Active: not in reset+inactive count.
        assert_eq!(manager.available_blocks(), initial - 1);

        let pin = block.pin();
        // Dropping the block while pin is held: slot must stay Active.
        drop(block);
        assert_eq!(
            manager.available_blocks(),
            initial - 1,
            "pin should prevent the slot from transitioning to Inactive"
        );

        // Dropping the pin releases the keepalive: slot transitions to
        // Inactive (which counts as available).
        drop(pin);
        assert_eq!(manager.available_blocks(), initial);
    }

    #[test]
    fn pin_clone_is_arc_bump() {
        let manager = create_test_manager::<TestMeta>(4);
        let token = create_iota_token_block(200, 4);
        let mb = manager
            .allocate_blocks(1)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let cb = mb.complete(&token).unwrap();
        let block = manager
            .register_blocks(vec![cb])
            .into_iter()
            .next()
            .unwrap();

        let pin = block.pin();
        let count_before = pin.use_count();
        let _clone = pin.clone();
        assert_eq!(pin.use_count(), count_before + 1);
    }

    #[test]
    fn pins_from_different_managers_have_different_addresses() {
        let mgr_a = create_test_manager::<TestMeta>(4);
        let mgr_b = create_test_manager::<TestMeta>(4);

        let token_a = create_iota_token_block(300, 4);
        let mb_a = mgr_a
            .allocate_blocks(1)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let block_a = mgr_a
            .register_blocks(vec![mb_a.complete(&token_a).unwrap()])
            .into_iter()
            .next()
            .unwrap();

        let token_b = create_iota_token_block(300, 4);
        let mb_b = mgr_b
            .allocate_blocks(1)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let block_b = mgr_b
            .register_blocks(vec![mb_b.complete(&token_b).unwrap()])
            .into_iter()
            .next()
            .unwrap();

        let pin_a = block_a.pin();
        let pin_b = block_b.pin();

        // Same content (same iota tokens) ⇒ same SequenceHash.
        assert_eq!(pin_a.sequence_hash(), pin_b.sequence_hash());
        // ...but different physical pools.
        assert_ne!(pin_a.manager_id(), pin_b.manager_id());
        // The (manager_id, block_id) pair is the disambiguating address.
    }
}
