// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small, host-neutral primitives for constant-work routing policies.
//!
//! Routing hosts retain candidate ownership, eligibility, and admission. This module owns only
//! the state and random sampling shared by round-robin, random, P2C, and least-load tie-breaking.

use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Stateful round-robin cursor plus stateless random and P2C selection helpers.
#[derive(Debug, Default)]
pub struct FastPicker {
    round_robin_cursor: AtomicU64,
}

impl FastPicker {
    /// Construct a picker with its round-robin cursor at the first candidate.
    pub const fn new() -> Self {
        Self {
            round_robin_cursor: AtomicU64::new(0),
        }
    }

    /// Select a round-robin candidate, advancing only for a committed selection.
    #[inline(always)]
    pub fn round_robin_index(&self, candidate_count: usize, commit: bool) -> Option<usize> {
        if candidate_count == 0 {
            return None;
        }
        let cursor = if commit {
            self.round_robin_cursor
                .fetch_add(1, AtomicOrdering::Relaxed)
        } else {
            self.round_robin_cursor.load(AtomicOrdering::Relaxed)
        };
        Some(cursor as usize % candidate_count)
    }

    /// Sample one candidate uniformly at random.
    #[inline(always)]
    pub fn random_index(candidate_count: usize) -> Option<usize> {
        if candidate_count == 0 {
            None
        } else {
            Some(fastrand::usize(..candidate_count))
        }
    }

    /// Sample two distinct candidates and choose the lower load.
    ///
    /// The load callback is invoked zero times for zero candidates, zero times for one candidate,
    /// and exactly twice for every larger candidate set.
    #[inline(always)]
    pub fn power_of_two_choices_index(
        candidate_count: usize,
        load: impl Fn(usize) -> u64,
    ) -> Option<usize> {
        Self::power_of_two_choices_index_by(candidate_count, |first, second| {
            load(first) <= load(second)
        })
    }

    /// Sample two distinct candidates and choose the one preferred by `compare`.
    ///
    /// The comparison callback is invoked zero times for zero or one candidate, and exactly once
    /// for every larger candidate set.
    #[inline(always)]
    pub fn power_of_two_choices_index_by(
        candidate_count: usize,
        compare: impl FnOnce(usize, usize) -> bool,
    ) -> Option<usize> {
        let first = Self::random_index(candidate_count)?;
        if candidate_count == 1 {
            return Some(first);
        }
        let second = (first + 1 + fastrand::usize(..candidate_count - 1)) % candidate_count;
        Some(if compare(first, second) {
            first
        } else {
            second
        })
    }
}

/// Find the lowest candidate using reservoir sampling to break equal-cost ties.
///
/// This helper is for hosts that already need to scan their candidate set. A cache-free host
/// with an O(1) maintained minimum must use that minimum directly instead.
#[inline(always)]
pub fn reservoir_least_index_by(
    candidate_count: usize,
    mut compare: impl FnMut(usize, usize) -> Ordering,
    mut sample: impl FnMut(usize) -> usize,
) -> Option<usize> {
    if candidate_count == 0 {
        return None;
    }
    let mut best = 0;
    let mut ties = 1usize;
    for index in 1..candidate_count {
        match compare(index, best) {
            Ordering::Less => {
                best = index;
                ties = 1;
            }
            Ordering::Equal => {
                ties += 1;
                if sample(ties) == 0 {
                    best = index;
                }
            }
            Ordering::Greater => {}
        }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn round_robin_peek_is_read_only() {
        let picker = FastPicker::new();
        assert_eq!(picker.round_robin_index(4, false), Some(0));
        assert_eq!(picker.round_robin_index(4, false), Some(0));
        assert_eq!(picker.round_robin_index(4, true), Some(0));
        assert_eq!(picker.round_robin_index(4, true), Some(1));
    }

    #[test]
    fn p2c_reads_two_loads_at_any_nontrivial_size() {
        let reads = Cell::new(0);
        let selected = FastPicker::power_of_two_choices_index(1024, |_| {
            reads.set(reads.get() + 1);
            0
        });
        assert!(selected.is_some());
        assert_eq!(reads.get(), 2);
    }

    #[test]
    fn reservoir_selection_keeps_any_tied_candidate_eligible() {
        for _ in 0..32 {
            let selected = reservoir_least_index_by(
                3,
                |_, _| Ordering::Equal,
                |upper| fastrand::usize(..upper),
            )
            .unwrap();
            assert!(selected < 3);
        }
    }
}
