// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::config::M3Config;
use crate::disk_tier::DiskTier;
use crate::fast_tier::FastTier;
use dynamo_memory::{ArenaBuffer, MemoryDescription, SystemStorage, nixl::NixlRegistered};
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// Main M3 storage instance
pub struct M3Store {
    pub(crate) disk_tier: DiskTier,
    pub(crate) fast_tier: Option<FastTier>,
    pub(crate) stats: Arc<SizeStatistics>,
    pub(crate) config: M3Config,
}

/// Buffer returned from M3Store operations
#[derive(Debug)]
pub enum M3Buffer {
    /// Owned buffer from disk tier
    Owned(Vec<u8>),

    /// Reference to fast tier buffer (NIXL-registered, RDMA-capable)
    FastTier(Arc<ArenaBuffer<NixlRegistered<SystemStorage>>>),
}

impl M3Buffer {
    /// Get the size of the buffer
    pub fn size(&self) -> usize {
        match self {
            M3Buffer::Owned(v) => v.len(),
            M3Buffer::FastTier(buf) => buf.size(),
        }
    }

    /// Get a slice of the buffer data
    pub fn as_slice(&self) -> &[u8] {
        match self {
            M3Buffer::Owned(v) => v.as_slice(),
            M3Buffer::FastTier(buf) => unsafe {
                std::slice::from_raw_parts(buf.addr() as *const u8, buf.size())
            },
        }
    }

    /// Copy buffer contents into destination
    pub fn copy_to(&self, dst: &mut [u8]) -> Result<(), crate::error::M3Error> {
        if dst.len() < self.size() {
            return Err(crate::error::M3Error::BufferTooSmall);
        }

        match self {
            M3Buffer::Owned(v) => {
                dst[..v.len()].copy_from_slice(v);
            }
            M3Buffer::FastTier(buf) => unsafe {
                std::ptr::copy_nonoverlapping(
                    buf.addr() as *const u8,
                    dst.as_mut_ptr(),
                    buf.size(),
                );
            },
        }

        Ok(())
    }

    /// Convert to owned Vec<u8>
    pub fn to_vec(self) -> Vec<u8> {
        match self {
            M3Buffer::Owned(v) => v,
            M3Buffer::FastTier(buf) => {
                let mut v = vec![0u8; buf.size()];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf.addr() as *const u8,
                        v.as_mut_ptr(),
                        buf.size(),
                    );
                }
                v
            }
        }
    }
}

/// Statistics tracker for object sizes
pub struct SizeStatistics {
    count: AtomicU64,
    total_bytes: AtomicU64,
    histogram: Mutex<Histogram>,
}

impl SizeStatistics {
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            histogram: Mutex::new(Histogram::new()),
        }
    }

    /// Record a new object size
    pub fn record(&self, size: usize) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size as u64, Ordering::Relaxed);
        self.histogram.lock().record(size);
    }

    /// Get the total number of objects recorded
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the total bytes recorded
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Get the average object size
    pub fn average_size(&self) -> u64 {
        let count = self.count();
        if count == 0 {
            0
        } else {
            self.total_bytes() / count
        }
    }

    /// Predict space needed for an incoming object
    /// Returns a conservative estimate based on historical data
    pub fn predict_space_needed(&self, incoming_size: usize) -> usize {
        // For now, simple heuristic: return the incoming size
        // Could be enhanced with percentile-based predictions
        incoming_size
    }

    /// Determine if eager eviction should be triggered
    pub fn should_eager_evict(&self, current_used: usize, max: usize, threshold: f64) -> bool {
        let usage_ratio = current_used as f64 / max as f64;
        usage_ratio >= threshold
    }

    /// Get snapshot of histogram
    pub fn get_histogram_snapshot(&self) -> HistogramSnapshot {
        self.histogram.lock().snapshot()
    }
}

impl Default for SizeStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple histogram for tracking size distribution
struct Histogram {
    // Buckets: < 1KB, 1KB-10KB, 10KB-100KB, 100KB-1MB, 1MB-10MB, 10MB-100MB, > 100MB
    buckets: [u64; 7],
}

impl Histogram {
    fn new() -> Self {
        Self { buckets: [0; 7] }
    }

    fn record(&mut self, size: usize) {
        let bucket = Self::size_to_bucket(size);
        self.buckets[bucket] += 1;
    }

    fn size_to_bucket(size: usize) -> usize {
        match size {
            0..=1023 => 0,             // < 1KB
            1024..=10239 => 1,         // 1KB-10KB
            10240..=102399 => 2,       // 10KB-100KB
            102400..=1048575 => 3,     // 100KB-1MB
            1048576..=10485759 => 4,   // 1MB-10MB
            10485760..=104857599 => 5, // 10MB-100MB
            _ => 6,                    // > 100MB
        }
    }

    fn snapshot(&self) -> HistogramSnapshot {
        HistogramSnapshot {
            buckets: self.buckets,
        }
    }
}

/// Snapshot of histogram for external access
#[derive(Debug, Clone, Copy)]
pub struct HistogramSnapshot {
    pub buckets: [u64; 7],
}

impl HistogramSnapshot {
    pub fn bucket_labels() -> &'static [&'static str] {
        &[
            "< 1KB",
            "1KB-10KB",
            "10KB-100KB",
            "100KB-1MB",
            "1MB-10MB",
            "10MB-100MB",
            "> 100MB",
        ]
    }

    pub fn get_bucket(&self, index: usize) -> u64 {
        self.buckets.get(index).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_statistics() {
        let stats = SizeStatistics::new();

        stats.record(1000);
        stats.record(2000);
        stats.record(3000);

        assert_eq!(stats.count(), 3);
        assert_eq!(stats.total_bytes(), 6000);
        assert_eq!(stats.average_size(), 2000);
    }

    #[test]
    fn test_should_eager_evict() {
        let stats = SizeStatistics::new();

        // 85% threshold
        assert!(!stats.should_eager_evict(84, 100, 0.85));
        assert!(stats.should_eager_evict(85, 100, 0.85));
        assert!(stats.should_eager_evict(90, 100, 0.85));
    }

    #[test]
    fn test_histogram_buckets() {
        let stats = SizeStatistics::new();

        stats.record(500); // < 1KB
        stats.record(5000); // 1KB-10KB
        stats.record(50000); // 10KB-100KB
        stats.record(500000); // 100KB-1MB
        stats.record(5000000); // 1MB-10MB

        let snapshot = stats.get_histogram_snapshot();
        assert_eq!(snapshot.buckets[0], 1); // < 1KB
        assert_eq!(snapshot.buckets[1], 1); // 1KB-10KB
        assert_eq!(snapshot.buckets[2], 1); // 10KB-100KB
        assert_eq!(snapshot.buckets[3], 1); // 100KB-1MB
        assert_eq!(snapshot.buckets[4], 1); // 1MB-10MB
    }
}
