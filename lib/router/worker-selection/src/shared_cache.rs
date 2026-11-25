// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block positions an external shared cache holds for a request, as the host looked them
//! up and the cache column reports them.

use std::ops::Range;

/// Shared cache hit information, represented as sorted non-overlapping half-open ranges.
///
/// Ranges encode which block positions exist in the external shared KV cache pool.
/// Using ranges instead of `Vec<bool>` avoids iterating over potentially thousands
/// of blocks per worker. Typical shared cache patterns produce few contiguous regions,
/// making `hits_beyond` and `loadable_beyond` O(num_ranges) ~ O(1-5).
#[derive(Debug, Clone, Default)]
pub struct SharedCacheHits {
    /// Ranges of block positions that exist in the shared cache.
    /// Half-open ranges [start, end), sorted and non-overlapping.
    ranges: Vec<Range<u32>>,
    /// Total number of hits (sum of range lengths).
    total_hits: u32,
    /// Shortest run, in blocks, the engine loads from the shared cache; it skips
    /// shorter runs. Zero sets no minimum.
    min_load_blocks: u32,
}

impl SharedCacheHits {
    /// Create from sorted, non-overlapping ranges.
    pub fn from_ranges(ranges: Vec<Range<u32>>) -> Self {
        let total_hits = ranges.iter().map(|r| r.end - r.start).sum();
        Self {
            ranges,
            total_hits,
            min_load_blocks: 0,
        }
    }

    /// The block positions the shared cache holds, as sorted, non-overlapping half-open
    /// ranges.
    #[inline]
    pub fn ranges(&self) -> &[Range<u32>] {
        &self.ranges
    }

    /// The number of block positions the shared cache holds: the sum of the range lengths.
    #[inline]
    pub fn total_hits(&self) -> u32 {
        self.total_hits
    }

    /// Set the shortest run, in blocks, the engine loads from the shared cache.
    pub fn with_min_load_blocks(mut self, min_load_blocks: u32) -> Self {
        self.min_load_blocks = min_load_blocks;
        self
    }

    /// Create from a boolean hit vector (convenience for tests and simple backends).
    /// Coalesces consecutive `true` entries into ranges.
    pub fn from_hits(hits: &[bool]) -> Self {
        let mut ranges = Vec::new();
        let mut i = 0;
        while i < hits.len() {
            if hits[i] {
                let start = i as u32;
                while i < hits.len() && hits[i] {
                    i += 1;
                }
                ranges.push(start..i as u32);
            } else {
                i += 1;
            }
        }
        Self::from_ranges(ranges)
    }

    /// Blocks a worker holding the first `local_blocks` blocks can load from the
    /// shared cache: the unbroken run that starts at `local_blocks`, or zero when
    /// that run is shorter than `min_load_blocks`. An engine extends its prefix in
    /// order, so hits after a missing block cannot be used.
    pub fn loadable_beyond(&self, local_blocks: u32) -> u32 {
        let mut end = local_blocks;
        for range in &self.ranges {
            if range.end <= end {
                continue;
            }
            if range.start > end {
                break;
            }
            end = range.end;
        }
        let run = end - local_blocks;
        if run < self.min_load_blocks { 0 } else { run }
    }

    /// Count hits at positions >= `from_position`, including hits after a gap
    /// that an engine cannot load; see [`Self::loadable_beyond`].
    /// O(num_ranges), not O(num_blocks).
    pub fn hits_beyond(&self, from_position: u32) -> u32 {
        self.ranges
            .iter()
            .map(|r| {
                if r.end <= from_position {
                    0
                } else if r.start >= from_position {
                    r.end - r.start
                } else {
                    r.end - from_position
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_cache_hits_from_hits() {
        // All hits contiguous
        let hits = SharedCacheHits::from_hits(&[true, true, true, true]);
        assert_eq!(hits.ranges, vec![0..4]);
        assert_eq!(hits.total_hits, 4);

        // Sparse hits
        let hits = SharedCacheHits::from_hits(&[true, false, true, true, false, true]);
        assert_eq!(hits.ranges, vec![0..1, 2..4, 5..6]);
        assert_eq!(hits.total_hits, 4);

        // No hits
        let hits = SharedCacheHits::from_hits(&[false, false, false]);
        assert!(hits.ranges.is_empty());
        assert_eq!(hits.total_hits, 0);

        // Empty
        let hits = SharedCacheHits::from_hits(&[]);
        assert!(hits.ranges.is_empty());
        assert_eq!(hits.total_hits, 0);
    }

    #[test]
    fn test_shared_cache_hits_beyond() {
        // Shared has [A, B, C, D] => range 0..4
        #[allow(clippy::single_range_in_vec_init)]
        let hits = SharedCacheHits::from_ranges(vec![0..4]);

        // Device has overlap=2 (positions 0,1 on device) => shared_beyond should count positions 2,3
        assert_eq!(hits.hits_beyond(2), 2);

        // Device has overlap=0 => all 4 shared hits count
        assert_eq!(hits.hits_beyond(0), 4);

        // Device has overlap=4 => nothing beyond
        assert_eq!(hits.hits_beyond(4), 0);

        // Device overlap exceeds range
        assert_eq!(hits.hits_beyond(10), 0);
    }

    #[test]
    fn test_shared_cache_hits_beyond_sparse() {
        // Ranges: [1..3, 5..8] => positions 1,2,5,6,7
        let hits = SharedCacheHits::from_ranges(vec![1..3, 5..8]);
        assert_eq!(hits.total_hits, 5);

        // from_position=0 => all 5 hits
        assert_eq!(hits.hits_beyond(0), 5);
        // from_position=2 => pos 2 (from first range) + 5,6,7 (from second) = 4
        assert_eq!(hits.hits_beyond(2), 4);
        // from_position=3 => only second range: 3 hits
        assert_eq!(hits.hits_beyond(3), 3);
        // from_position=6 => positions 6,7 from second range = 2
        assert_eq!(hits.hits_beyond(6), 2);
        // from_position=8 => nothing
        assert_eq!(hits.hits_beyond(8), 0);
    }

    #[test]
    fn test_shared_cache_loadable_beyond_stops_at_first_gap() {
        // Positions 0,1,2 and 5,6,7: the engine cannot load past the gap at 3.
        let hits = SharedCacheHits::from_ranges(vec![0..3, 5..8]);
        assert_eq!(hits.loadable_beyond(0), 3);
        assert_eq!(hits.loadable_beyond(1), 2);
        assert_eq!(hits.loadable_beyond(3), 0);
        assert_eq!(hits.loadable_beyond(4), 0);
        assert_eq!(hits.loadable_beyond(5), 3);
        assert_eq!(hits.loadable_beyond(8), 0);

        // Adjacent ranges form one run.
        let hits = SharedCacheHits::from_ranges(vec![0..2, 2..4]);
        assert_eq!(hits.loadable_beyond(1), 3);
    }

    #[test]
    fn test_shared_cache_loadable_beyond_respects_min_load_blocks() {
        #[allow(clippy::single_range_in_vec_init)]
        let hits = SharedCacheHits::from_ranges(vec![0..8]).with_min_load_blocks(4);
        assert_eq!(hits.loadable_beyond(0), 8);
        assert_eq!(hits.loadable_beyond(4), 4);
        assert_eq!(hits.loadable_beyond(5), 0);
    }
}
