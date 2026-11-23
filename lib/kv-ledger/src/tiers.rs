// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tier slots: where a block is resident, as small integers the host assigns.

/// How many tier slots a ledger has.
pub const MAX_TIERS: usize = 8;

/// One place a block can be resident. The host picks which slot means what.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tier(u8);

impl Tier {
    /// `None` when `slot` is not below [`MAX_TIERS`].
    pub const fn new(slot: u8) -> Option<Self> {
        if (slot as usize) < MAX_TIERS {
            Some(Self(slot))
        } else {
            None
        }
    }

    pub const fn slot(self) -> usize {
        self.0 as usize
    }
}

/// A set of tier slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TierSet(u8);

impl TierSet {
    pub(crate) const fn of(tier: Tier) -> Self {
        Self(1 << tier.0)
    }

    pub(crate) const fn with(self, tier: Tier) -> Self {
        Self(self.0 | (1 << tier.0))
    }

    pub(crate) const fn without(self, tier: Tier) -> Self {
        Self(self.0 & !(1 << tier.0))
    }

    pub const fn contains(self, tier: Tier) -> bool {
        self.0 & (1 << tier.0) != 0
    }

    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }
}
