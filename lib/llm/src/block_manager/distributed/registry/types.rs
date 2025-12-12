// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core types for the distributed object registry.

use super::protocol::SequenceHash;


pub type ObjectKey = u64;

/// Result of `can_offload()` - which hashes need to be stored vs already exist.
///
/// When `can_offload` is called, the hub atomically **leases** the hashes that
/// are available. This prevents race conditions where multiple workers try to
/// offload the same hash. The lease expires after a timeout if `register` is
/// not called.
///
/// # Example
/// ```ignore
/// let result = registry.can_offload(&[A, B, C, D]).await?;
/// // If A, C already in object and D is leased by another worker:
/// // result.can_offload = [B]         // You have the lease, store this
/// // result.already_stored = [A, C]   // Already in object, skip
/// // result.leased = [D]              // Another worker is handling this
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OffloadResult {
    /// Hashes that CAN be offloaded (lease granted to you, store these).
    /// You MUST call `register()` after storing, or the lease will expire.
    pub can_offload: Vec<SequenceHash>,

    /// Hashes that are ALREADY stored (in registry, skip storing).
    pub already_stored: Vec<SequenceHash>,

    /// Hashes that are currently LEASED by another worker.
    /// Another worker is in the process of storing these. Skip them.
    pub leased: Vec<SequenceHash>,
}

impl OffloadResult {
    /// Create empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create result with pre-allocated capacity.
    pub fn with_capacity(can_offload: usize, already_stored: usize) -> Self {
        Self {
            can_offload: Vec::with_capacity(can_offload),
            already_stored: Vec::with_capacity(already_stored),
            leased: Vec::new(),
        }
    }

    /// Number of hashes that need to be stored (leases granted to you).
    pub fn offload_count(&self) -> usize {
        self.can_offload.len()
    }

    /// Number of hashes already in storage (dedup hits).
    pub fn dedup_count(&self) -> usize {
        self.already_stored.len()
    }

    /// Number of hashes leased by other workers.
    pub fn leased_count(&self) -> usize {
        self.leased.len()
    }

    /// Total hashes checked.
    pub fn total(&self) -> usize {
        self.can_offload.len() + self.already_stored.len() + self.leased.len()
    }

    /// Deduplication ratio (0.0 = no dedup, 1.0 = all already stored or leased).
    /// Includes both stored and leased as "handled" since you don't need to store them.
    pub fn dedup_ratio(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            0.0
        } else {
            (self.already_stored.len() + self.leased.len()) as f64 / total as f64
        }
    }

    /// Check if any hashes can be offloaded (you have leases).
    pub fn has_offloadable(&self) -> bool {
        !self.can_offload.is_empty()
    }

    /// Check if all hashes were already stored or leased (nothing for you to do).
    pub fn all_handled(&self) -> bool {
        self.can_offload.is_empty() && (!self.already_stored.is_empty() || !self.leased.is_empty())
    }

    /// Check if all hashes were already stored.
    pub fn all_already_stored(&self) -> bool {
        self.can_offload.is_empty() && self.leased.is_empty() && !self.already_stored.is_empty()
    }
}

/// Result of `match_sequence_hashes()` - what can be loaded from object.
///
/// Contains contiguous prefix of hashes that exist in object storage.
///
/// # Example
/// ```ignore
/// // Query: [A, B, C, D, E] where A, B, C exist in object
/// let result = registry.match_sequence_hashes(&[A, B, C, D, E]).await?;
/// // result.matched = [(A, key_a), (B, key_b), (C, key_c)]
/// // Stops at D because it doesn't exist
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MatchResult {
    /// Contiguous prefix of hashes that exist, with their object keys.
    /// Stops at first hash that doesn't exist.
    pub matched: Vec<(SequenceHash, ObjectKey)>,
}

impl MatchResult {
    /// Create empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create result with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            matched: Vec::with_capacity(capacity),
        }
    }

    /// Create from matched entries.
    pub fn from_matched(matched: Vec<(SequenceHash, ObjectKey)>) -> Self {
        Self { matched }
    }

    /// Number of hashes that matched.
    pub fn match_count(&self) -> usize {
        self.matched.len()
    }

    /// Check if any hashes matched.
    pub fn has_matches(&self) -> bool {
        !self.matched.is_empty()
    }

    /// Check if empty (no matches).
    pub fn is_empty(&self) -> bool {
        self.matched.is_empty()
    }

    /// Get just the hashes (without keys).
    pub fn hashes(&self) -> Vec<SequenceHash> {
        self.matched.iter().map(|(h, _)| *h).collect()
    }

    /// Get just the object keys.
    pub fn keys(&self) -> Vec<ObjectKey> {
        self.matched.iter().map(|(_, k)| *k).collect()
    }

    /// Iterate over matched entries.
    pub fn iter(&self) -> impl Iterator<Item = &(SequenceHash, ObjectKey)> {
        self.matched.iter()
    }
}

impl IntoIterator for MatchResult {
    type Item = (SequenceHash, ObjectKey);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.matched.into_iter()
    }
}

/// Hub statistics for monitoring and debugging.
#[derive(Debug, Clone, Default)]
pub struct HubStats {
    /// Total entries registered.
    pub total_registered: u64,

    /// Total `can_offload()` queries.
    pub total_offload_queries: u64,

    /// Total `match_sequence_hashes()` queries.
    pub total_match_queries: u64,

    /// Deduplication hits (hashes that already existed when can_offload was called).
    pub dedup_hits: u64,

    /// Total hashes checked for offload.
    pub dedup_total_checked: u64,
}

impl HubStats {
    /// Create new stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Deduplication ratio (0.0 = no dedup, 1.0 = all were duplicates).
    pub fn dedup_ratio(&self) -> f64 {
        if self.dedup_total_checked == 0 {
            0.0
        } else {
            self.dedup_hits as f64 / self.dedup_total_checked as f64
        }
    }

    /// Total queries (all types).
    pub fn total_queries(&self) -> u64 {
        self.total_offload_queries + self.total_match_queries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offload_result_empty() {
        let result = OffloadResult::new();
        assert_eq!(result.offload_count(), 0);
        assert_eq!(result.dedup_count(), 0);
        assert_eq!(result.dedup_ratio(), 0.0);
        assert!(!result.has_offloadable());
        assert!(!result.all_already_stored());
    }

    #[test]
    fn test_offload_result_mixed() {
        let result = OffloadResult {
            can_offload: vec![1, 2],
            already_stored: vec![3, 4, 5],
            leased: vec![],
        };
        assert_eq!(result.offload_count(), 2);
        assert_eq!(result.dedup_count(), 3);
        assert_eq!(result.total(), 5);
        assert!((result.dedup_ratio() - 0.6).abs() < 0.001);
        assert!(result.has_offloadable());
        assert!(!result.all_already_stored());
    }

    #[test]
    fn test_offload_result_all_stored() {
        let result = OffloadResult {
            can_offload: vec![],
            already_stored: vec![1, 2, 3],
            leased: vec![],
        };
        assert!(!result.has_offloadable());
        assert!(result.all_already_stored());
        assert_eq!(result.dedup_ratio(), 1.0);
    }

    #[test]
    fn test_offload_result_with_leased() {
        let result = OffloadResult {
            can_offload: vec![1],
            already_stored: vec![2, 3],
            leased: vec![4, 5],
        };
        assert_eq!(result.offload_count(), 1);
        assert_eq!(result.dedup_count(), 2);
        assert_eq!(result.leased_count(), 2);
        assert_eq!(result.total(), 5);
        // Dedup ratio includes leased (4 out of 5 = 80% handled by others)
        assert!((result.dedup_ratio() - 0.8).abs() < 0.001);
        assert!(result.has_offloadable());
        assert!(!result.all_handled()); // We have work to do
    }

    #[test]
    fn test_offload_result_all_handled() {
        let result = OffloadResult {
            can_offload: vec![],
            already_stored: vec![1, 2],
            leased: vec![3],
        };
        assert!(!result.has_offloadable());
        assert!(result.all_handled());
        assert!(!result.all_already_stored()); // Some are leased, not stored
    }

    #[test]
    fn test_match_result_empty() {
        let result = MatchResult::new();
        assert_eq!(result.match_count(), 0);
        assert!(!result.has_matches());
        assert!(result.is_empty());
    }

    #[test]
    fn test_match_result_with_matches() {
        let result = MatchResult::from_matched(vec![(100, 100), (200, 200), (300, 300)]);
        assert_eq!(result.match_count(), 3);
        assert!(result.has_matches());
        assert!(!result.is_empty());
        assert_eq!(result.hashes(), vec![100, 200, 300]);
        assert_eq!(result.keys(), vec![100, 200, 300]);
    }

    #[test]
    fn test_hub_stats() {
        let mut stats = HubStats::new();
        stats.dedup_hits = 30;
        stats.dedup_total_checked = 100;
        assert!((stats.dedup_ratio() - 0.3).abs() < 0.001);

        stats.total_offload_queries = 10;
        stats.total_match_queries = 20;
        assert_eq!(stats.total_queries(), 30);
    }
}

