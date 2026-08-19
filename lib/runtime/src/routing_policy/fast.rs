// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU64, Ordering};

/// Host-neutral constant-work primitives shared by builtin routing policies.
#[derive(Debug, Default)]
pub(crate) struct FastPicker {
    round_robin_cursor: AtomicU64,
}

impl FastPicker {
    pub(crate) const fn new() -> Self {
        Self {
            round_robin_cursor: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub(crate) fn round_robin_index(&self, candidate_count: usize, commit: bool) -> Option<usize> {
        if candidate_count == 0 {
            return None;
        }
        let cursor = if commit {
            self.round_robin_cursor.fetch_add(1, Ordering::Relaxed)
        } else {
            self.round_robin_cursor.load(Ordering::Relaxed)
        };
        Some(cursor as usize % candidate_count)
    }

    #[inline(always)]
    pub(crate) fn random_index(candidate_count: usize) -> Option<usize> {
        Self::random_index_by(candidate_count, |upper| fastrand::usize(..upper))
    }

    #[inline(always)]
    pub(crate) fn random_index_by(
        candidate_count: usize,
        mut sample: impl FnMut(usize) -> usize,
    ) -> Option<usize> {
        (candidate_count != 0).then(|| sample(candidate_count))
    }

    #[inline(always)]
    pub(crate) fn power_of_two_choices_index(
        candidate_count: usize,
        load: impl Fn(usize) -> u64,
    ) -> Option<usize> {
        Self::power_of_two_choices_index_by(candidate_count, |upper| fastrand::usize(..upper), load)
    }

    #[inline(always)]
    pub(crate) fn power_of_two_choices_index_by(
        candidate_count: usize,
        mut sample: impl FnMut(usize) -> usize,
        load: impl Fn(usize) -> u64,
    ) -> Option<usize> {
        let first = Self::random_index_by(candidate_count, &mut sample)?;
        if candidate_count == 1 {
            return Some(first);
        }
        let second = (first + 1 + sample(candidate_count - 1)) % candidate_count;
        Some(if load(first) <= load(second) {
            first
        } else {
            second
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn round_robin_peek_does_not_advance() {
        let picker = FastPicker::new();
        assert_eq!(picker.round_robin_index(4, false), Some(0));
        assert_eq!(picker.round_robin_index(4, false), Some(0));
        assert_eq!(picker.round_robin_index(4, true), Some(0));
        assert_eq!(picker.round_robin_index(4, true), Some(1));
    }

    #[test]
    fn p2c_reads_two_loads_at_large_worker_counts() {
        let reads = Cell::new(0);
        let selected = FastPicker::power_of_two_choices_index(65_536, |_| {
            reads.set(reads.get() + 1);
            0
        });
        assert!(selected.is_some());
        assert_eq!(reads.get(), 2);
    }
}
