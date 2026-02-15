// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified event handle encoded in a single `u128` value.

use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};

use crate::status::Generation;

const WORKER_BITS: u32 = 64;
const LOCAL_BITS: u32 = 32;
const GENERATION_BITS: u32 = 32;

const LOCAL_SHIFT: u32 = GENERATION_BITS;
const WORKER_SHIFT: u32 = LOCAL_SHIFT + LOCAL_BITS;

const WORKER_MASK: u128 = ((1u128 << WORKER_BITS) - 1) << WORKER_SHIFT;
const LOCAL_MASK: u128 = ((1u128 << LOCAL_BITS) - 1) << LOCAL_SHIFT;
const GENERATION_MASK: u128 = (1u128 << GENERATION_BITS) - 1;

/// Public event handle encoded in a single u128 value.
///
/// Layout (MSB to LSB): `[worker_id: 64 bits][local_index: 32 bits][generation: 32 bits]`
///
/// Local-only handles have `worker_id == 0`. Distributed handles embed a
/// non-zero worker identifier so they are globally unique across workers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventHandle(u128);

impl EventHandle {
    /// Create a handle with an explicit worker id.
    pub(crate) fn new(worker_id: u64, local_index: u32, generation: Generation) -> Self {
        let raw = ((worker_id as u128) << WORKER_SHIFT)
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

    /// Extract the worker id (upper 64 bits).
    pub fn worker_id(&self) -> u64 {
        ((self.0 & WORKER_MASK) >> WORKER_SHIFT) as u64
    }

    /// Extract the local index (middle 32 bits).
    pub fn local_index(&self) -> u32 {
        ((self.0 & LOCAL_MASK) >> LOCAL_SHIFT) as u32
    }

    /// Extract the generation counter (lower 32 bits).
    pub fn generation(&self) -> Generation {
        (self.0 & GENERATION_MASK) as Generation
    }

    /// Returns `true` when the handle was created without a worker id.
    pub fn is_local_only(&self) -> bool {
        self.worker_id() == 0
    }

    /// Return a copy of this handle with a different generation.
    pub fn with_generation(&self, generation: Generation) -> Self {
        Self::new(self.worker_id(), self.local_index(), generation)
    }
}

impl Display for EventHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EventHandle(index={}, generation={})",
            self.local_index(),
            self.generation()
        )
    }
}
