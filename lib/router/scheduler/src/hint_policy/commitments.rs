// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decisions a worker-selection policy commits the booked worker to, which a hint policy carries
//! out.

use std::ops::BitOr;

/// Kinds of commitment a hint policy turns into hints, declared through
/// [`super::HintPolicy::commitments`].
///
/// A selection commits the booked worker to a decision its cost assumed, today only a KV copy.
/// The router pairs each kind with the one hint policy of the role that declares it, runs that
/// policy first, and fails the booking when it sends nothing for a committed decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Commitments(u8);

impl Commitments {
    /// No commitments.
    pub const NONE: Self = Self(0);
    /// A KV copy: the booked worker copies the blocks of a
    /// [`KvCopySource`](crate::kv_hints::KvCopySource) from its source.
    pub const KV_COPY: Self = Self(1 << 0);

    /// Whether every kind in `other` is also in `self`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl BitOr for Commitments {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}
