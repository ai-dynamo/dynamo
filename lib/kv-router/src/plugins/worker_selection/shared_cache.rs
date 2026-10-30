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
/// making `hits_beyond` O(num_ranges) ~ O(1-5).
#[derive(Debug, Clone, Default)]
pub struct SharedCacheHits {
    /// Ranges of block positions that exist in the shared cache.
    /// Half-open ranges [start, end), sorted and non-overlapping.
    pub ranges: Vec<Range<u32>>,
    /// Total number of hits (sum of range lengths).
    pub total_hits: u32,
}

impl SharedCacheHits {
    /// Create from sorted, non-overlapping ranges.
    pub fn from_ranges(ranges: Vec<Range<u32>>) -> Self {
        let total_hits = ranges.iter().map(|r| r.end - r.start).sum();
        Self { ranges, total_hits }
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

    /// Count hits at positions >= `from_position`.
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
