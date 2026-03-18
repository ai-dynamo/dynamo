// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! High-level interface for the block manager connector.
//!
//! This module can be used to framework connector apis or provide the touch points to build
//! a full blown scheduler + kvbm + framework connector.

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod protocol;
pub mod scheduler;

use super::*;

use crate::{
    block_manager::{block::BlockId, pool::BlockPoolError},
    tokens::{SaltHash, TokenBlockSequence},
};

use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum SlotError {
    #[error("slot not found")]
    NotFound,

    #[error("slot is in an invalid state: {0}")]
    InvalidState(String),

    #[error("slot operation failed: {0}")]
    InvalidOperation(String),

    #[error(transparent)]
    BlockPoolError(#[from] BlockPoolError),
}

pub trait RequestKey:
    std::hash::Hash
    + std::cmp::Eq
    + std::fmt::Debug
    + std::fmt::Display
    + tracing::Value
    + Clone
    + Send
    + Sync
    + 'static
{
}

impl RequestKey for String {}
impl RequestKey for u64 {}
impl RequestKey for usize {}

pub trait SlotManager<R: RequestKey>: Send + Sync {
    type SlotType: Slot + ?Sized;

    fn has_slot(&self, request_id: &R) -> bool;

    /// Create a new slot for the given request ID, initial tokens and salt hash.
    fn create_slot(
        &self,
        request_id: &R,
        tokens: Vec<u32>,
        salt_hash: SaltHash,
    ) -> Result<(), SlotError>;

    fn get_slot(&self, request_id: &R) -> Result<Arc<Mutex<Self::SlotType>>, SlotError>;
    fn remove_slot(&self, request_id: &R) -> Result<(), SlotError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// The slot was not scheduled in the previous iteration.
    Initialized,

    /// The slot was previously scheduled, but not in the last iteration.
    NotScheduled,

    /// The slot is prepared to load kv blocks from external storage; however, the onboarding operation
    /// has not been triggered yet. The usize is the number of tokens that are ready for onboarding.
    OnboardStaged(usize),

    /// The slot is actively copying blocks to device storage from some external storage(s).
    /// The u64 is the iteration at which the onboarding operation was triggered.
    Onboarding(u64),

    /// The slot is actively prefilling the sequence.
    Prefilling,

    /// The slot is actively participating in a forward pass which will result in one more more tokens
    /// to be applied to the sequence.
    Decoding,

    /// The slot is marked as finished, but not all resources have been released.
    Finishing,

    /// The slot is finished and all resources have been released.
    Finished,
}

pub trait Slot: std::fmt::Debug {
    fn state(&self) -> SlotState;

    fn sequence(&self) -> &TokenBlockSequence;

    /// The number of tokens that have been computed on the device, i.e. the number of tokens for which we have ownership
    /// of computed kv blocks in the device storage.
    fn computed_tokens(&self) -> usize;

    fn mark_as_scheduled(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_prefilling(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_decoding(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_onboarding(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_not_scheduled(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_finished(&mut self, iteration: u64) -> Result<(), SlotError>;

    /// The number of device blocks that have been allocated to the slot.
    fn num_device_blocks_allocated(&self) -> usize;

    /// Find all possible block matches for remaining known tokens in some local storage, i.e. look up and take ownership
    /// of any kv blocks for tokens in the isl that are not already in memory on the device, but on some local storage.
    ///
    /// If external tokens are matched, then the slot will transition to the [`SlotState::Onboarding`] state.
    fn acquire_all_local_matches(&mut self) -> Result<(), SlotError>;

    /// Take all pending operations for the slot.
    fn take_pending_operations(&mut self) -> Vec<String>;
}

pub trait ExternallyManagedDeviceSlot: Slot {
    /// Since we do not control the device pool, nor do we have insight in how the device pool is managed,
    /// we must accept external updates to the computed position.
    fn advance_computed_position(&mut self, num_tokens: usize) -> Result<(), SlotError>;

    /// Append the given block ids to the slot.
    ///
    /// The external device block manager has provided a set of mutable blocks to the slot.
    fn append_mutable_device_blocks(&mut self, block_ids: Vec<BlockId>) -> Result<(), SlotError>;
}
