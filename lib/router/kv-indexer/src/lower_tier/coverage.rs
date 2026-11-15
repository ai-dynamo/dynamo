// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which owners hold which positions of one run.
//!
//! Owner sets are run-length encoded: consecutive positions held by the same
//! owners share one segment, so a store or remove of many blocks is one
//! operation and a walk does owner-set work only where the owners change.

use std::ops::Range;

use super::{IndexedResidencyOwner, OwnerId, OwnerSet, OwnerTable, Removal};

#[derive(Debug, Clone)]
struct Segment {
    start: u32,
    /// `None` is a hole: positions nobody holds.
    owners: Option<OwnerSet>,
}

/// Owner sets over positions `0..len`.
///
/// Segments are sorted by `start`, the first starts at 0, and neighbours never
/// hold equal owner sets.
#[derive(Debug, Default)]
pub(super) struct Coverage {
    segments: Vec<Segment>,
    len: u32,
    /// Positions held by at least one owner.
    live: u32,
}

fn same(a: &Option<OwnerSet>, b: &Option<OwnerSet>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.same_owners(b),
        _ => false,
    }
}

impl Coverage {
    pub(super) fn len(&self) -> u32 {
        self.len
    }

    pub(super) fn live(&self) -> u32 {
        self.live
    }

    /// One past the last held position; 0 when nothing is held.
    pub(super) fn live_end(&self) -> u32 {
        (0..self.segments.len())
            .rev()
            .find(|&idx| self.segments[idx].owners.is_some())
            .map_or(0, |idx| self.segment_end(idx))
    }

    /// Index of the segment holding `pos`, which must be below `len`.
    pub(super) fn segment_index(&self, pos: u32) -> usize {
        debug_assert!(pos < self.len);
        self.segments
            .partition_point(|segment| segment.start <= pos)
            - 1
    }

    pub(super) fn segment_end(&self, idx: usize) -> u32 {
        self.segments
            .get(idx + 1)
            .map_or(self.len, |next| next.start)
    }

    pub(super) fn segment_owners(&self, idx: usize) -> Option<&OwnerSet> {
        self.segments[idx].owners.as_ref()
    }

    pub(super) fn owners_at(&self, pos: u32) -> Option<&OwnerSet> {
        self.segment_owners(self.segment_index(pos))
    }

    /// Every segment's positions and owners, in order.
    pub(super) fn segments(&self) -> impl Iterator<Item = (Range<u32>, Option<&OwnerSet>)> + '_ {
        (0..self.segments.len()).map(|idx| {
            (
                self.segments[idx].start..self.segment_end(idx),
                self.segments[idx].owners.as_ref(),
            )
        })
    }

    /// Whether the owner holds every position of `range`.
    pub(super) fn covers(
        &self,
        range: Range<u32>,
        id: OwnerId,
        owner: IndexedResidencyOwner,
    ) -> bool {
        self.overlapping(range).all(|idx| {
            self.segments[idx]
                .owners
                .as_ref()
                .is_some_and(|set| set.contains(id, owner))
        })
    }

    /// Whether the owner holds any position of `range`.
    pub(super) fn intersects(
        &self,
        range: Range<u32>,
        id: OwnerId,
        owner: IndexedResidencyOwner,
    ) -> bool {
        self.overlapping(range).any(|idx| {
            self.segments[idx]
                .owners
                .as_ref()
                .is_some_and(|set| set.contains(id, owner))
        })
    }

    fn overlapping(&self, range: Range<u32>) -> Range<usize> {
        if range.start >= range.end {
            return 0..0;
        }
        self.segment_index(range.start)..self.segment_index(range.end - 1) + 1
    }

    /// Appends `count` positions held by `owners`.
    pub(super) fn push(&mut self, count: u32, owners: Option<OwnerSet>) {
        if count == 0 {
            return;
        }
        if owners.is_some() {
            self.live += count;
        }
        if !self
            .segments
            .last()
            .is_some_and(|last| same(&last.owners, &owners))
        {
            self.segments.push(Segment {
                start: self.len,
                owners,
            });
        }
        self.len += count;
    }

    /// Drops every position at or past `len`.
    pub(super) fn truncate(&mut self, len: u32) {
        if len >= self.len {
            return;
        }
        let first = self.split_at(len);
        for idx in first..self.segments.len() {
            if self.segments[idx].owners.is_some() {
                self.live -= self.segment_end(idx) - self.segments[idx].start;
            }
        }
        self.segments.truncate(first);
        self.len = len;
    }

    /// Adds the owner to every position of `range`. Calls `changed` with each
    /// span that did not have it and returns how many positions those are.
    pub(super) fn add(
        &mut self,
        range: Range<u32>,
        id: OwnerId,
        owner: IndexedResidencyOwner,
        table: &OwnerTable,
        mut changed: impl FnMut(Range<u32>),
    ) -> u32 {
        debug_assert!(range.start < range.end && range.end <= self.len);
        let first = self.split_at(range.start);
        let last = self.split_at(range.end);
        let mut added = 0;
        for idx in first..last {
            let span = self.segments[idx].start..self.segment_end(idx);
            let inserted = match &mut self.segments[idx].owners {
                Some(owners) => owners.insert(id, owner, table),
                hole => {
                    *hole = Some(OwnerSet::Single(id));
                    self.live += span.end - span.start;
                    true
                }
            };
            if inserted {
                added += span.end - span.start;
                changed(span);
            }
        }
        self.merge_around(first, last);
        added
    }

    /// Removes the owner from every position of `range`. Calls `changed` with
    /// each span that had it and returns how many positions those are.
    pub(super) fn remove(
        &mut self,
        range: Range<u32>,
        id: OwnerId,
        owner: IndexedResidencyOwner,
        table: &OwnerTable,
        mut changed: impl FnMut(Range<u32>),
    ) -> u32 {
        debug_assert!(range.start < range.end && range.end <= self.len);
        let first = self.split_at(range.start);
        let last = self.split_at(range.end);
        let mut removed = 0;
        for idx in first..last {
            let span = self.segments[idx].start..self.segment_end(idx);
            let segment = &mut self.segments[idx];
            let Some(owners) = &mut segment.owners else {
                continue;
            };
            match owners.remove(id, owner, table) {
                Removal::Absent => continue,
                Removal::Removed => {}
                Removal::Emptied => {
                    segment.owners = None;
                    self.live -= span.end - span.start;
                }
            }
            removed += span.end - span.start;
            changed(span);
        }
        self.merge_around(first, last);
        removed
    }

    /// Makes `pos` a segment boundary and returns the index of the segment
    /// starting there, or the segment count when `pos` is `len`.
    fn split_at(&mut self, pos: u32) -> usize {
        if pos >= self.len {
            return self.segments.len();
        }
        let idx = self.segment_index(pos);
        if self.segments[idx].start == pos {
            return idx;
        }
        let owners = self.segments[idx].owners.clone();
        self.segments
            .insert(idx + 1, Segment { start: pos, owners });
        idx + 1
    }

    /// Merges equal neighbours across the boundaries `first..=last`, where
    /// boundary `i` sits between segments `i - 1` and `i`.
    fn merge_around(&mut self, first: usize, last: usize) {
        let Some(max) = self.segments.len().checked_sub(1) else {
            return;
        };
        let (lo, hi) = (first.max(1), last.min(max));
        if lo > hi {
            return;
        }
        let mut keep = lo;
        for idx in lo..=hi {
            if same(&self.segments[keep - 1].owners, &self.segments[idx].owners) {
                continue;
            }
            self.segments.swap(keep, idx);
            keep += 1;
        }
        self.segments.drain(keep..=hi);
    }

    #[cfg(test)]
    pub(super) fn assert_invariants(&self) {
        assert_eq!(self.segments.is_empty(), self.len == 0);
        let mut live = 0;
        for (idx, segment) in self.segments.iter().enumerate() {
            let end = self.segment_end(idx);
            assert!(segment.start < end, "empty segment at {idx}");
            if idx == 0 {
                assert_eq!(segment.start, 0);
            } else {
                assert!(
                    !same(&self.segments[idx - 1].owners, &segment.owners),
                    "equal neighbours at {idx}"
                );
            }
            if segment.owners.is_some() {
                live += end - segment.start;
            }
        }
        assert_eq!(live, self.live);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::super::test_owners::indexed_owners;
    use super::*;

    fn ids_of(owners: Option<&OwnerSet>, table: &OwnerTable) -> BTreeSet<u32> {
        match owners {
            None => BTreeSet::new(),
            Some(OwnerSet::Single(id)) => BTreeSet::from([id.0]),
            Some(OwnerSet::Pair(ids)) => ids.iter().map(|id| id.0).collect(),
            Some(OwnerSet::Multi(multi)) => multi.owners().map(|owner| table.id(owner).0).collect(),
        }
    }

    /// Random adds, removes, appends and truncations against one owner set per
    /// position.
    #[test]
    fn matches_a_per_position_model() {
        for seed in 0..64 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let table = OwnerTable::default();
            let owners = indexed_owners(6);
            let ids: Vec<OwnerId> = owners.iter().map(|owner| table.intern(*owner)).collect();
            let mut coverage = Coverage::default();
            let mut model: Vec<BTreeSet<u32>> = Vec::new();

            for _ in 0..400 {
                let len = model.len() as u32;
                let which = rng.usize(..owners.len());
                let (id, owner) = (ids[which], owners[which]);
                match rng.u32(..10) {
                    0 | 1 => {
                        let count = rng.u32(1..6);
                        let held = rng.bool();
                        coverage.push(count, held.then_some(OwnerSet::Single(id)));
                        let set = if held {
                            BTreeSet::from([id.0])
                        } else {
                            BTreeSet::new()
                        };
                        model.extend(std::iter::repeat_n(set, count as usize));
                    }
                    2 if len > 0 => {
                        let keep = rng.u32(..=len);
                        coverage.truncate(keep);
                        model.truncate(keep as usize);
                    }
                    op if len > 0 => {
                        let start = rng.u32(..len);
                        let end = rng.u32(start + 1..=len.min(start + 8));
                        let add = op % 2 == 0;
                        let expected = (start..end)
                            .filter(|&pos| model[pos as usize].contains(&id.0) != add)
                            .count() as u32;
                        assert_eq!(
                            coverage.covers(start..end, id, owner),
                            (start..end).all(|pos| model[pos as usize].contains(&id.0))
                        );
                        assert_eq!(
                            coverage.intersects(start..end, id, owner),
                            (start..end).any(|pos| model[pos as usize].contains(&id.0))
                        );
                        let mut reported = 0;
                        let changed = if add {
                            coverage.add(start..end, id, owner, &table, |span| {
                                reported += span.end - span.start
                            })
                        } else {
                            coverage.remove(start..end, id, owner, &table, |span| {
                                reported += span.end - span.start
                            })
                        };
                        assert_eq!(changed, expected, "seed {seed}");
                        assert_eq!(reported, expected, "seed {seed}");
                        for pos in start..end {
                            if add {
                                model[pos as usize].insert(id.0);
                            } else {
                                model[pos as usize].remove(&id.0);
                            }
                        }
                    }
                    _ => {}
                }

                coverage.assert_invariants();
                assert_eq!(coverage.len() as usize, model.len());
                assert_eq!(
                    coverage.live() as usize,
                    model.iter().filter(|set| !set.is_empty()).count()
                );
                assert_eq!(
                    coverage.live_end() as usize,
                    model
                        .iter()
                        .rposition(|set| !set.is_empty())
                        .map_or(0, |pos| pos + 1)
                );
                for (pos, expected) in model.iter().enumerate() {
                    assert_eq!(
                        &ids_of(coverage.owners_at(pos as u32), &table),
                        expected,
                        "seed {seed} pos {pos}"
                    );
                }
                let spans: Vec<_> = coverage.segments().map(|(span, _)| span).collect();
                assert!(spans.windows(2).all(|pair| pair[0].end == pair[1].start));
            }
        }
    }
}
