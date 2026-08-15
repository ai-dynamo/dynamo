// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Array-backed min-max heap.
//!
//! A policy class stores runnable requests ordered by due time, so scheduling
//! needs both ends of the same structure: the minimum is the earliest deadline
//! that normal dispatch serves, and the maximum is the latest deadline that a
//! later deadline-aware dispatch policy will serve. `BinaryHeap` exposes only
//! one end, and keeping a second structure would be a second scheduling queue
//! per class.
//!
//! This is the classic Atkinson-Sack-Santoro-Strothotte min-max heap: one `Vec`
//! whose levels alternate between min levels (even depth, starting at the root)
//! and max levels (odd depth). Every node on a min level is no greater than all
//! of its descendants, and every node on a max level is no less than all of its
//! descendants. `peek_min`/`peek_max` are O(1), `push`/`pop_min`/`pop_max` are
//! O(log n), and building from a `Vec` is O(n).
//!
//! Ordering follows `T: Ord` directly: `pop_min` returns the least element.
//! Callers that want "most urgent first" order their key ascending by urgency.

use std::slice;
use std::vec;

pub(crate) struct MinMaxHeap<T> {
    data: Vec<T>,
}

impl<T> Default for MinMaxHeap<T> {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}

impl<T> MinMaxHeap<T> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate the backing array in unspecified order.
    pub(crate) fn iter(&self) -> slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Remove every element, yielding them in unspecified order.
    pub(crate) fn drain(&mut self) -> vec::Drain<'_, T> {
        self.data.drain(..)
    }

    /// Return the least element in O(1).
    pub(crate) fn peek_min(&self) -> Option<&T> {
        self.data.first()
    }
}

/// True when `index` sits on a min level of the implicit binary tree.
///
/// Depth is `floor(log2(index + 1))`; even depths are min levels, so the root
/// is a min level.
#[inline]
fn is_min_level(index: usize) -> bool {
    (index + 1).ilog2().is_multiple_of(2)
}

#[inline]
fn parent(index: usize) -> usize {
    (index - 1) / 2
}

impl<T: Ord> MinMaxHeap<T> {
    /// Return the greatest element in O(1).
    ///
    /// Normal Stage 0 dispatch only reads the minimum end; the maximum end is
    /// part of the data structure's contract and is exercised by its tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn peek_max(&self) -> Option<&T> {
        self.max_index().map(|index| &self.data[index])
    }

    pub(crate) fn push(&mut self, value: T) {
        self.data.push(value);
        self.bubble_up(self.data.len() - 1);
    }

    /// Remove and return the least element.
    pub(crate) fn pop_min(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let value = self.data.swap_remove(0);
        if !self.data.is_empty() {
            self.trickle_down(0);
        }
        Some(value)
    }

    /// Remove and return the greatest element.
    ///
    /// See [`Self::peek_max`] for why this has no Stage 0 production caller.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn pop_max(&mut self) -> Option<T> {
        let index = self.max_index()?;
        let value = self.data.swap_remove(index);
        if index < self.data.len() {
            self.trickle_down(index);
        }
        Some(value)
    }

    /// The greatest element is the root when the heap holds one element and
    /// otherwise the larger of the root's children, which form the first max
    /// level.
    fn max_index(&self) -> Option<usize> {
        match self.data.len() {
            0 => None,
            1 => Some(0),
            2 => Some(1),
            _ => Some(if self.data[1] >= self.data[2] { 1 } else { 2 }),
        }
    }

    fn bubble_up(&mut self, index: usize) {
        if index == 0 {
            return;
        }
        let parent = parent(index);
        if is_min_level(index) {
            if self.data[index] > self.data[parent] {
                self.data.swap(index, parent);
                self.bubble_up_on_level(parent, false);
            } else {
                self.bubble_up_on_level(index, true);
            }
        } else if self.data[index] < self.data[parent] {
            self.data.swap(index, parent);
            self.bubble_up_on_level(parent, true);
        } else {
            self.bubble_up_on_level(index, false);
        }
    }

    /// Walk grandparents on the same level kind until the element is ordered.
    fn bubble_up_on_level(&mut self, mut index: usize, min_level: bool) {
        while index > 2 {
            let grandparent = parent(parent(index));
            let ordered = if min_level {
                self.data[index] >= self.data[grandparent]
            } else {
                self.data[index] <= self.data[grandparent]
            };
            if ordered {
                return;
            }
            self.data.swap(index, grandparent);
            index = grandparent;
        }
    }

    fn trickle_down(&mut self, index: usize) {
        if is_min_level(index) {
            self.trickle_down_on_level(index, true);
        } else {
            self.trickle_down_on_level(index, false);
        }
    }

    fn trickle_down_on_level(&mut self, mut index: usize, min_level: bool) {
        while let Some((extreme, is_grandchild)) = self.extreme_descendant(index, min_level) {
            let displaces = if min_level {
                self.data[extreme] < self.data[index]
            } else {
                self.data[extreme] > self.data[index]
            };
            if !displaces {
                return;
            }
            self.data.swap(extreme, index);
            if !is_grandchild {
                return;
            }
            // The moved element landed on the same level kind two levels down,
            // so only its new parent - on the opposite level kind - can now be
            // out of order with it.
            let parent = parent(extreme);
            let inverted = if min_level {
                self.data[extreme] > self.data[parent]
            } else {
                self.data[extreme] < self.data[parent]
            };
            if inverted {
                self.data.swap(extreme, parent);
            }
            index = extreme;
        }
    }

    /// Return the least (or greatest) of `index`'s children and grandchildren,
    /// and whether it is a grandchild. `None` when `index` is a leaf.
    fn extreme_descendant(&self, index: usize, min_level: bool) -> Option<(usize, bool)> {
        let len = self.data.len();
        let first_child = 2 * index + 1;
        if first_child >= len {
            return None;
        }

        let better = |candidate: usize, current: usize| {
            if min_level {
                self.data[candidate] < self.data[current]
            } else {
                self.data[candidate] > self.data[current]
            }
        };

        let mut extreme = first_child;
        let mut is_grandchild = false;
        let second_child = first_child + 1;
        if second_child < len && better(second_child, extreme) {
            extreme = second_child;
        }
        let first_grandchild = 2 * first_child + 1;
        for grandchild in first_grandchild..(first_grandchild + 4).min(len) {
            if better(grandchild, extreme) {
                extreme = grandchild;
                is_grandchild = true;
            }
        }
        Some((extreme, is_grandchild))
    }
}

impl<T: Ord> From<Vec<T>> for MinMaxHeap<T> {
    /// Heapify in O(n) with the usual bottom-up pass over internal nodes.
    fn from(data: Vec<T>) -> Self {
        let mut heap = Self { data };
        for index in (0..heap.data.len() / 2).rev() {
            heap.trickle_down(index);
        }
        heap
    }
}

impl<T: Ord> FromIterator<T> for MinMaxHeap<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(iter.into_iter().collect::<Vec<_>>())
    }
}

impl<T> IntoIterator for MinMaxHeap<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    /// Yield every element in unspecified order.
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a MinMaxHeap<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift so the randomized checks stay reproducible and
    /// KV-router keeps its dependency footprint unchanged.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound
        }
    }

    /// Assert the min-max ordering invariant over the whole backing array.
    fn assert_invariant<T: Ord + std::fmt::Debug>(heap: &MinMaxHeap<T>) {
        for index in 0..heap.data.len() {
            let min_level = is_min_level(index);
            let mut descendants = vec![2 * index + 1, 2 * index + 2];
            let mut cursor = 0;
            while cursor < descendants.len() {
                let descendant = descendants[cursor];
                cursor += 1;
                if descendant >= heap.data.len() {
                    continue;
                }
                if min_level {
                    assert!(
                        heap.data[index] <= heap.data[descendant],
                        "min level {index} > descendant {descendant}"
                    );
                } else {
                    assert!(
                        heap.data[index] >= heap.data[descendant],
                        "max level {index} < descendant {descendant}"
                    );
                }
                descendants.push(2 * descendant + 1);
                descendants.push(2 * descendant + 2);
            }
        }
    }

    #[test]
    fn level_parity_alternates_by_depth() {
        assert!(is_min_level(0));
        assert!(!is_min_level(1));
        assert!(!is_min_level(2));
        for index in 3..7 {
            assert!(is_min_level(index), "depth 2 is a min level");
        }
        for index in 7..15 {
            assert!(!is_min_level(index), "depth 3 is a max level");
        }
    }

    #[test]
    fn empty_heap_has_no_ends() {
        let mut heap = MinMaxHeap::<u32>::new();
        assert!(heap.is_empty());
        assert_eq!(heap.peek_min(), None);
        assert_eq!(heap.peek_max(), None);
        assert_eq!(heap.pop_min(), None);
        assert_eq!(heap.pop_max(), None);
    }

    #[test]
    fn single_element_is_both_ends() {
        let mut heap = MinMaxHeap::new();
        heap.push(7);
        assert_eq!(heap.peek_min(), Some(&7));
        assert_eq!(heap.peek_max(), Some(&7));
        assert_eq!(heap.pop_max(), Some(7));
        assert!(heap.is_empty());
    }

    #[test]
    fn push_keeps_both_ends_correct() {
        let mut heap = MinMaxHeap::new();
        for value in [5, 1, 9, 3, 7, 2, 8, 6, 4, 0] {
            heap.push(value);
            assert_invariant(&heap);
        }
        assert_eq!(heap.peek_min(), Some(&0));
        assert_eq!(heap.peek_max(), Some(&9));
        assert_eq!(heap.len(), 10);
    }

    #[test]
    fn pop_min_yields_ascending_order() {
        let mut heap: MinMaxHeap<u32> = (0..64u32).map(|value| value * 7 % 64).collect();
        let mut popped = Vec::new();
        while let Some(value) = heap.pop_min() {
            assert_invariant(&heap);
            popped.push(value);
        }
        assert_eq!(popped, (0..64).collect::<Vec<_>>());
    }

    #[test]
    fn pop_max_yields_descending_order() {
        let mut heap: MinMaxHeap<u32> = (0..64u32).map(|value| value * 11 % 64).collect();
        let mut popped = Vec::new();
        while let Some(value) = heap.pop_max() {
            assert_invariant(&heap);
            popped.push(value);
        }
        assert_eq!(popped, (0..64).rev().collect::<Vec<_>>());
    }

    #[test]
    fn from_vec_heapifies_every_length() {
        for len in 0..40u32 {
            let heap =
                MinMaxHeap::from((0..len).map(|value| (value * 13) % 40).collect::<Vec<_>>());
            assert_invariant(&heap);
            assert_eq!(heap.len(), len as usize);
            if len > 0 {
                let expected = (0..len).map(|value| (value * 13) % 40);
                assert_eq!(heap.peek_min().copied(), expected.clone().min());
                assert_eq!(heap.peek_max().copied(), expected.max());
            }
        }
    }

    #[test]
    fn duplicate_keys_keep_both_ends_valid() {
        let mut heap: MinMaxHeap<u32> = std::iter::repeat_n(4u32, 17).collect();
        assert_invariant(&heap);
        assert_eq!(heap.peek_min(), Some(&4));
        assert_eq!(heap.peek_max(), Some(&4));
        heap.push(1);
        heap.push(9);
        assert_eq!(heap.peek_min(), Some(&1));
        assert_eq!(heap.peek_max(), Some(&9));
    }

    #[test]
    fn interleaved_operations_match_a_sorted_model() {
        let mut rng = Rng(0x5eed_1234_9abc_def1);
        let mut heap = MinMaxHeap::new();
        let mut model: Vec<u64> = Vec::new();

        for step in 0..20_000u64 {
            match rng.below(4) {
                0 | 1 => {
                    let value = rng.below(512);
                    heap.push(value);
                    model.push(value);
                }
                2 => {
                    model.sort_unstable();
                    let expected = if model.is_empty() {
                        None
                    } else {
                        Some(model.remove(0))
                    };
                    assert_eq!(heap.pop_min(), expected, "pop_min at step {step}");
                }
                _ => {
                    model.sort_unstable();
                    assert_eq!(heap.pop_max(), model.pop(), "pop_max at step {step}");
                }
            }
            assert_eq!(heap.len(), model.len());
            if step % 97 == 0 {
                assert_invariant(&heap);
                model.sort_unstable();
                assert_eq!(heap.peek_min().copied(), model.first().copied());
                assert_eq!(heap.peek_max().copied(), model.last().copied());
            }
        }
    }

    #[test]
    fn drain_and_into_iter_yield_every_element() {
        let mut heap: MinMaxHeap<u32> = (0..25u32).collect();
        let mut drained: Vec<u32> = heap.drain().collect();
        drained.sort_unstable();
        assert_eq!(drained, (0..25).collect::<Vec<_>>());
        assert!(heap.is_empty());

        let heap: MinMaxHeap<u32> = (0..25u32).collect();
        let mut collected: Vec<u32> = heap.into_iter().collect();
        collected.sort_unstable();
        assert_eq!(collected, (0..25).collect::<Vec<_>>());
    }
}
