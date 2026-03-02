// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Delegate trait for block presence transitions.
//!
//! [`PresenceDelegate`] receives notifications when blocks transition to/from
//! the `Registered` state. Implementations must be lightweight and non-blocking
//! — callbacks fire synchronously on the transitioning thread, outside any lock.

use super::handle::BlockRegistrationHandle;
use crate::BlockId;
use crate::blocks::{BlockMetadata, SequenceHash};

use std::any::TypeId;
use std::sync::Arc;

/// Delegate notified when blocks transition to/from Registered state.
///
/// Implementations must be lightweight and non-blocking — callbacks fire
/// synchronously on the transitioning thread, outside any lock.
pub trait PresenceDelegate: Send + Sync {
    /// Called after a block is marked present (Staged → Registered).
    fn on_present(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
        type_id: TypeId,
        handle: &BlockRegistrationHandle,
    );

    /// Called after a block is marked absent (Registered → Reset).
    fn on_absent(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
        type_id: TypeId,
        handle: &BlockRegistrationHandle,
    );
}

/// A delegate that only fires for a specific `T: BlockMetadata` type,
/// filtering by `TypeId` internally.
struct TypedPresenceDelegate<F1, F2> {
    type_id: TypeId,
    on_present: F1,
    on_absent: F2,
}

impl<F1, F2> PresenceDelegate for TypedPresenceDelegate<F1, F2>
where
    F1: Fn(SequenceHash, BlockId, &BlockRegistrationHandle) + Send + Sync,
    F2: Fn(SequenceHash, BlockId, &BlockRegistrationHandle) + Send + Sync,
{
    fn on_present(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
        type_id: TypeId,
        handle: &BlockRegistrationHandle,
    ) {
        if type_id == self.type_id {
            (self.on_present)(seq_hash, block_id, handle);
        }
    }

    fn on_absent(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
        type_id: TypeId,
        handle: &BlockRegistrationHandle,
    ) {
        if type_id == self.type_id {
            (self.on_absent)(seq_hash, block_id, handle);
        }
    }
}

/// Create a delegate that only fires for blocks with metadata type `T`.
///
/// # Example
///
/// ```ignore
/// let delegate = typed_presence_delegate::<GpuMeta>(
///     |seq_hash, block_id, handle| { /* on present */ },
///     |seq_hash, block_id, handle| { /* on absent */ },
/// );
/// ```
pub fn typed_presence_delegate<T: BlockMetadata>(
    on_present: impl Fn(SequenceHash, BlockId, &BlockRegistrationHandle) + Send + Sync + 'static,
    on_absent: impl Fn(SequenceHash, BlockId, &BlockRegistrationHandle) + Send + Sync + 'static,
) -> Arc<dyn PresenceDelegate> {
    Arc::new(TypedPresenceDelegate {
        type_id: TypeId::of::<T>(),
        on_present,
        on_absent,
    })
}
