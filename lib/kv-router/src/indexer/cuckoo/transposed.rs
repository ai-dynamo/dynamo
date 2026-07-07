// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::ensure;

use super::{CuckooFilter, DeltaEntry, OVERLAP_VERIFY_WINDOW, Probe, SLOTS};

pub struct TransposedTable {
    num_dcs: usize,
    num_buckets: usize,
    lanes: Vec<AtomicU64>,
    generations: Vec<AtomicU64>,
}

#[derive(Debug)]
pub struct MaskLookup {
    pub depths: Vec<u32>,
    pub conflict_mask: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchPhase {
    GenerationSnapshot,
    FirstProbe,
    ExponentialProbe,
    BinaryProbe,
    VerificationProbe,
    GenerationValidation,
}

impl TransposedTable {
    pub fn from_filters(filters: &[CuckooFilter]) -> anyhow::Result<Self> {
        let Some(first) = filters.first() else {
            anyhow::bail!("transposed table requires at least one filter");
        };
        ensure!(
            filters.len() <= 16,
            "transposed table supports at most 16 DCs"
        );
        ensure!(
            filters.iter().all(|filter| {
                filter.num_buckets() == first.num_buckets() && filter.seed() == first.seed()
            }),
            "all transposed filters must have identical shape and seed"
        );
        let num_dcs = filters.len();
        let num_buckets = first.num_buckets();
        let lanes = (0..num_buckets)
            .flat_map(|bucket| {
                filters
                    .iter()
                    .map(move |filter| AtomicU64::new(pack_slots(filter.bucket_slots(bucket))))
            })
            .collect();
        let table = Self {
            num_dcs,
            num_buckets,
            lanes,
            generations: (0..num_dcs).map(|_| AtomicU64::new(0)).collect(),
        };
        table.verify_filters(filters)?;
        Ok(table)
    }

    pub fn num_dcs(&self) -> usize {
        self.num_dcs
    }

    pub fn begin_update(&self, dc: usize) -> u64 {
        let previous = self.generations[dc].fetch_add(1, Ordering::Release);
        assert_eq!(previous & 1, 0, "concurrent writers for DC {dc}");
        previous
    }

    pub fn apply_entries(&self, dc: usize, entries: &[DeltaEntry]) {
        for entry in entries {
            self.lanes[self.lane_index(entry.bucket, dc)]
                .store(pack_slots(entry.slots), Ordering::Release);
        }
    }

    pub fn rebuild_dc(&self, dc: usize, filter: &CuckooFilter) {
        assert_eq!(filter.num_buckets(), self.num_buckets);
        for bucket in 0..self.num_buckets {
            self.lanes[self.lane_index(bucket, dc)]
                .store(pack_slots(filter.bucket_slots(bucket)), Ordering::Release);
        }
    }

    pub fn end_update(&self, dc: usize, previous_even: u64) {
        self.generations[dc].store(previous_even + 2, Ordering::Release);
    }

    pub fn search(&self, probes: &[Probe], available_mask: u16) -> MaskLookup {
        self.search_with_phase_hook(probes, available_mask, |_| {})
    }

    fn search_with_phase_hook(
        &self,
        probes: &[Probe],
        available_mask: u16,
        mut hook: impl FnMut(SearchPhase),
    ) -> MaskLookup {
        let all_mask = if self.num_dcs == 16 {
            u16::MAX
        } else {
            (1u16 << self.num_dcs) - 1
        };
        let eligible = available_mask & all_mask;
        let before: Vec<u64> = self
            .generations
            .iter()
            .map(|generation| generation.load(Ordering::Acquire))
            .collect();
        hook(SearchPhase::GenerationSnapshot);
        let mut stable = eligible;
        for (dc, generation) in before.iter().enumerate() {
            if generation & 1 != 0 {
                stable &= !(1u16 << dc);
            }
        }
        let mut depths = self.search_stable(probes, stable, &mut hook);
        let mut conflict_mask = eligible & !stable;
        hook(SearchPhase::GenerationValidation);
        for (dc, generation) in self.generations.iter().enumerate() {
            let after = generation.load(Ordering::Acquire);
            if after != before[dc] || after & 1 != 0 {
                conflict_mask |= 1u16 << dc;
                depths[dc] = 0;
            }
        }
        MaskLookup {
            depths,
            conflict_mask,
        }
    }

    pub fn verify_filters(&self, filters: &[CuckooFilter]) -> anyhow::Result<()> {
        ensure!(filters.len() == self.num_dcs, "filter count mismatch");
        for (dc, filter) in filters.iter().enumerate() {
            ensure!(
                filter.num_buckets() == self.num_buckets,
                "bucket count mismatch"
            );
            for bucket in 0..self.num_buckets {
                let actual = self.lanes[self.lane_index(bucket, dc)].load(Ordering::Acquire);
                let expected = pack_slots(filter.bucket_slots(bucket));
                ensure!(
                    actual == expected,
                    "transposed lane mismatch: dc={dc}, bucket={bucket}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "bench")]
    pub fn touch_for_benchmark(&self) {
        for lane in &self.lanes {
            std::hint::black_box(lane.load(Ordering::Acquire));
        }
        for generation in &self.generations {
            std::hint::black_box(generation.load(Ordering::Acquire));
        }
    }

    fn search_stable(
        &self,
        probes: &[Probe],
        stable_mask: u16,
        hook: &mut impl FnMut(SearchPhase),
    ) -> Vec<u32> {
        let mut depths = vec![0u32; self.num_dcs];
        if probes.is_empty() || stable_mask == 0 {
            return depths;
        }
        let first = self.presence_mask(&probes[0], stable_mask);
        hook(SearchPhase::FirstProbe);
        let active = stable_mask & first;
        let mut lo = vec![0usize; self.num_dcs];
        let mut hi = vec![1usize; self.num_dcs];
        let mut unresolved = active;

        let mut probe_index = 1usize;
        while unresolved != 0 && probe_index < probes.len() {
            let present = self.presence_mask(&probes[probe_index], unresolved);
            hook(SearchPhase::ExponentialProbe);
            let missed = unresolved & !present;
            for_each_dc(missed, |dc| hi[dc] = probe_index);
            for_each_dc(present, |dc| lo[dc] = probe_index);
            unresolved = present;
            probe_index <<= 1;
        }
        if unresolved != 0 {
            for_each_dc(unresolved, |dc| hi[dc] = probes.len());
        }

        loop {
            let mut groups = BTreeMap::<usize, u16>::new();
            for_each_dc(active, |dc| {
                if hi[dc] - lo[dc] > 1 {
                    let midpoint = lo[dc] + (hi[dc] - lo[dc]) / 2;
                    *groups.entry(midpoint).or_default() |= 1u16 << dc;
                }
            });
            if groups.is_empty() {
                break;
            }
            for (midpoint, group) in groups {
                let present = self.presence_mask(&probes[midpoint], group);
                hook(SearchPhase::BinaryProbe);
                for_each_dc(group & present, |dc| lo[dc] = midpoint);
                for_each_dc(group & !present, |dc| hi[dc] = midpoint);
            }
        }

        for_each_dc(active, |dc| depths[dc] = hi[dc] as u32);
        let mut verification = BTreeMap::<usize, u16>::new();
        for_each_dc(active, |dc| {
            let end = depths[dc] as usize;
            for index in end.saturating_sub(OVERLAP_VERIFY_WINDOW)..end {
                *verification.entry(index).or_default() |= 1u16 << dc;
            }
        });
        let mut verified = active;
        for (index, candidates) in verification {
            let candidates = candidates & verified;
            if candidates == 0 {
                continue;
            }
            let present = self.presence_mask(&probes[index], candidates);
            hook(SearchPhase::VerificationProbe);
            let missed = candidates & !present;
            for_each_dc(missed, |dc| depths[dc] = index as u32);
            verified &= !missed;
        }
        depths
    }

    #[inline]
    fn presence_mask(&self, probe: &Probe, candidates: u16) -> u16 {
        let mask = (self.num_buckets - 1) as u64;
        let (first, second) = probe.bucket_indices(mask);
        let fingerprint = probe.fingerprint();
        let mut present = 0u16;
        for_each_dc(candidates, |dc| {
            let first_lane = self.lanes[self.lane_index(first, dc)].load(Ordering::Acquire);
            if packed_contains(first_lane, fingerprint) {
                present |= 1u16 << dc;
                return;
            }
            let second_lane = self.lanes[self.lane_index(second, dc)].load(Ordering::Acquire);
            if packed_contains(second_lane, fingerprint) {
                present |= 1u16 << dc;
            }
        });
        present
    }

    #[inline]
    fn lane_index(&self, bucket: usize, dc: usize) -> usize {
        bucket * self.num_dcs + dc
    }
}

#[inline]
fn pack_slots(slots: [u16; SLOTS]) -> u64 {
    slots
        .iter()
        .enumerate()
        .fold(0u64, |packed, (index, slot)| {
            packed | (u64::from(*slot) << (index * 16))
        })
}

#[inline]
fn packed_contains(packed: u64, fingerprint: u16) -> bool {
    (0..SLOTS).any(|index| ((packed >> (index * 16)) as u16) == fingerprint)
}

fn for_each_dc(mut mask: u16, mut callback: impl FnMut(usize)) {
    while mask != 0 {
        let dc = mask.trailing_zeros() as usize;
        callback(dc);
        mask &= mask - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::{DEFAULT_FILTER_SEED, overlap_depth_searched, probes_for};
    use super::*;

    #[test]
    fn transposed_and_native_search_agree() {
        let mut filters: Vec<CuckooFilter> = (0..4)
            .map(|_| CuckooFilter::provisioned(512, DEFAULT_FILTER_SEED))
            .collect();
        let sequence: Vec<u64> = (1..=128).collect();
        for (dc, filter) in filters.iter_mut().enumerate() {
            for &hash in sequence.iter().take(17 + dc * 13) {
                assert!(filter.insert(hash));
            }
        }
        let probes = probes_for(&sequence, DEFAULT_FILTER_SEED);
        let table = TransposedTable::from_filters(&filters).unwrap();
        let result = table.search(&probes, 0b1111);
        assert_eq!(result.conflict_mask, 0);
        for (dc, filter) in filters.iter().enumerate() {
            assert_eq!(
                result.depths[dc],
                overlap_depth_searched(filter, &probes),
                "DC {dc}"
            );
        }
    }

    #[test]
    fn updating_one_generation_only_conflicts_that_dc() {
        let filters: Vec<CuckooFilter> = (0..2)
            .map(|_| CuckooFilter::provisioned(64, DEFAULT_FILTER_SEED))
            .collect();
        let table = TransposedTable::from_filters(&filters).unwrap();
        let even = table.begin_update(1);
        let probes = probes_for(&[1, 2, 3], DEFAULT_FILTER_SEED);
        let result = table.search(&probes, 0b11);
        assert_eq!(result.conflict_mask, 0b10);
        table.end_update(1, even);
    }

    #[test]
    fn generation_changes_are_detected_during_every_search_phase() {
        let sequence: Vec<u64> = (1..=128).collect();
        let mut filters: Vec<CuckooFilter> = (0..2)
            .map(|_| CuckooFilter::provisioned(512, DEFAULT_FILTER_SEED))
            .collect();
        for filter in &mut filters {
            for &hash in sequence.iter().take(40) {
                assert!(filter.insert(hash));
            }
        }
        let probes = probes_for(&sequence, DEFAULT_FILTER_SEED);
        let table = TransposedTable::from_filters(&filters).unwrap();
        for target in [
            SearchPhase::GenerationSnapshot,
            SearchPhase::FirstProbe,
            SearchPhase::ExponentialProbe,
            SearchPhase::BinaryProbe,
            SearchPhase::VerificationProbe,
            SearchPhase::GenerationValidation,
        ] {
            let mut changed = false;
            let result = table.search_with_phase_hook(&probes, 0b11, |phase| {
                if phase == target && !changed {
                    let generation = table.begin_update(1);
                    table.end_update(1, generation);
                    changed = true;
                }
            });
            assert!(changed, "search did not reach {target:?}");
            assert_eq!(result.conflict_mask, 0b10, "phase {target:?}");
            assert_eq!(
                result.depths[0],
                overlap_depth_searched(&filters[0], &probes),
                "stable DC was lost during {target:?}"
            );
        }
    }
}
