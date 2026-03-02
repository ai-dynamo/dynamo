// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::sync::{Arc, Mutex};

use crate::block_manager::{
    block::BlockId,
    connector::RequestKey,
    pool::BlockPoolError,
};
use crate::tokens::{SaltHash, TokenBlockSequence};

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

pub trait SlotManager<R: RequestKey>: Send + Sync {
    type SlotType: Slot + ?Sized;

    fn has_slot(&self, request_id: &R) -> bool;

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
#[allow(dead_code)]
pub enum SlotState {
    Initialized,
    OnboardStaged(usize),
    Onboarding(usize),
    Prefilling,
    SkippedPrefill,
    Decoding,
    SkippedDecode,
    Finishing,
    Finished,
    Preempted,
}

#[allow(dead_code)]
pub trait Slot: std::fmt::Debug {
    fn request_id(&self) -> &str;
    fn state(&self) -> SlotState;
    fn sequence(&self) -> &TokenBlockSequence;
    fn computed_tokens(&self) -> usize;

    fn apply_scheduler_output(
        &mut self,
        tokens: &[u32],
        block_ids: &[usize],
        num_computed_tokens: usize,
        num_scheduled_tokens: usize,
        priorities: Option<&[u32]>,
    ) -> Result<(), SlotError>;

    fn record_start_iteration(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_prefilling(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_decoding(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_finished(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn num_device_blocks_allocated(&self) -> usize;
    fn acquire_local_matches(&mut self, num_computed_tokens: usize) -> Result<(), SlotError>;
    fn trigger_onboarding(&mut self, num_external_tokens: usize) -> Result<(), SlotError>;
    fn take_pending_operations(
        &mut self,
    ) -> Option<Vec<crate::block_manager::connector::protocol::WorkerTransferRequest>>;
    fn record_cached_device_tokens(&mut self, num_tokens: usize);
    fn record_cached_host_tokens(&mut self, num_tokens: usize);
    fn record_cached_disk_tokens(&mut self, num_tokens: usize);
    fn reset_after_preemption(&mut self);
    fn reset(&mut self);
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait ExternallyManagedDeviceSlot: Slot {
    fn advance_computed_position(&mut self, num_tokens: usize) -> Result<(), SlotError>;
    fn append_mutable_device_blocks(&mut self, block_ids: &[BlockId]) -> Result<(), SlotError>;
}
