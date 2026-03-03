// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified event handle encoded in a single `u128` value.

use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

use crate::status::Generation;

const SYSTEM_BITS: u32 = 64;
const LOCAL_BITS: u32 = 32;
const GENERATION_BITS: u32 = 32;

const LOCAL_SHIFT: u32 = GENERATION_BITS;
const SYSTEM_SHIFT: u32 = LOCAL_SHIFT + LOCAL_BITS;

const SYSTEM_MASK: u128 = ((1u128 << SYSTEM_BITS) - 1) << SYSTEM_SHIFT;
const LOCAL_MASK: u128 = ((1u128 << LOCAL_BITS) - 1) << LOCAL_SHIFT;
const GENERATION_MASK: u128 = (1u128 << GENERATION_BITS) - 1;

/// Bit 31 of `local_index` marks handles as local vs distributed.
pub(crate) const LOCAL_FLAG: u32 = 1 << 31;

/// Mask for the counter portion of `local_index` (strips the local flag bit).
pub(crate) const INDEX_COUNTER_MASK: u32 = LOCAL_FLAG - 1;

/// Public event handle encoded in a single u128 value.
///
/// Layout (MSB to LSB): `[system_id: 64 bits][local_index: 32 bits][generation: 32 bits]`
///
/// The `local_index` field uses bit 31 as a local/distributed flag:
/// - Bit 31 = 1: local event (created by `LocalEventSystem::new()`)
/// - Bit 31 = 0: distributed event (created via `DistributedEventFactory`)
///
/// Both local and distributed systems have unique non-zero `system_id` values.
/// Use `is_local()` / `is_distributed()` to check origin type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventHandle(u128);

impl EventHandle {
    /// Create a handle with an explicit system id.
    pub(crate) fn new(system_id: u64, local_index: u32, generation: Generation) -> Self {
        let raw = ((system_id as u128) << SYSTEM_SHIFT)
            | ((local_index as u128) << LOCAL_SHIFT)
            | (generation as u128);
        Self(raw)
    }

    /// Reconstruct a handle from its raw u128 representation.
    pub fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    /// Return the raw u128 representation.
    pub fn raw(&self) -> u128 {
        self.0
    }

    /// Extract the system id (upper 64 bits).
    pub fn system_id(&self) -> u64 {
        ((self.0 & SYSTEM_MASK) >> SYSTEM_SHIFT) as u64
    }

    /// Extract the local index (middle 32 bits), including the local flag bit.
    pub fn local_index(&self) -> u32 {
        ((self.0 & LOCAL_MASK) >> LOCAL_SHIFT) as u32
    }

    /// Extract the generation counter (lower 32 bits).
    pub fn generation(&self) -> Generation {
        (self.0 & GENERATION_MASK) as Generation
    }

    /// Returns `true` when the handle was created by a local event system.
    pub fn is_local(&self) -> bool {
        (self.local_index() & LOCAL_FLAG) != 0
    }

    /// Returns `true` when the handle was created by a distributed event system.
    pub fn is_distributed(&self) -> bool {
        !self.is_local()
    }

    /// Extract the counter portion of the local index (strips the flag bit).
    pub(crate) fn index_counter(&self) -> u32 {
        self.local_index() & INDEX_COUNTER_MASK
    }

    /// Return a copy of this handle with a different generation.
    pub fn with_generation(&self, generation: Generation) -> Self {
        Self::new(self.system_id(), self.local_index(), generation)
    }
}

impl Display for EventHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "EventHandle {{ system={}, index={}, generation={}, {} }}",
            self.system_id(),
            self.index_counter(),
            self.generation(),
            if self.is_local() {
                "local"
            } else {
                "distributed"
            }
        )
    }
}
