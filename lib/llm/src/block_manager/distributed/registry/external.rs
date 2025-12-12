// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! External registry trait for remote storage tier catalogs.
//!
//! Provides an abstraction over external storage systems
//! for tracking which keys have been offloaded.
//!
//! # Overview
//!
//! The `ExternalRegistry` trait defines the interface for storage tier catalogs
//! that track which keys exist in external/remote storage. This enables:
//!
//! - Cache lookups before expensive network transfers
//! - Multiple backend implementations
//! - Easy mocking for tests
//!
//! # Example
//!
//! ```ignore
//! use dynamo_llm::block_manager::v2::logical::external_registry::ExternalRegistry;
//!
//! fn check_cache<R: ExternalRegistry<u64>>(registry: &R, hashes: &[u64]) {
//!     let matched = registry.match_keys(hashes);
//!     println!("Found {} keys in external storage", matched.len());
//! }
//! ```

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

/// Trait for keys that can be stored in an external registry.
///
/// Keys must be:
/// - Copyable (cheap to pass around)
/// - Hashable (for fast lookups)
/// - Comparable (for deduplication)
/// - Debuggable (for logging)
/// - Thread-safe (Send + Sync)
pub trait RegistryKey: Copy + Hash + Eq + Debug + Send + Sync + 'static {}

// Blanket implementation for all types that satisfy the bounds
impl<T> RegistryKey for T where T: Copy + Hash + Eq + Debug + Send + Sync + 'static {}

/// Trait for external storage registries.
///
/// Implementors track which keys have been stored in external storage systems
/// (S3, GCS, Azure Blob, etc.). This enables cache lookups before initiating
/// expensive network transfers.
///
/// # Type Parameter
///
/// * `K` - The key type (must implement [`RegistryKey`])
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` for concurrent access.
///
/// # Example Implementation
///
/// ```ignore
/// impl ExternalRegistry<u64> for ObjectRegistry {
///     fn match_keys(&self, keys: &[u64]) -> Vec<u64> {
///         keys.iter()
///             .copied()
///             .take_while(|k| self.cache.contains_key(k))
///             .collect()
///     }
///     // ...
/// }
/// ```
pub trait ExternalRegistry<K: RegistryKey>: Send + Sync + Debug {
    /// Find matching keys (contiguous prefix).
    ///
    /// Returns the longest contiguous prefix of keys that exist in the registry.
    /// Stops at the first miss. This matches KV cache block semantics where
    /// we need contiguous prefix matches.
    ///
    /// # Arguments
    ///
    /// * `keys` - Keys to look up (in order)
    ///
    /// # Returns
    ///
    /// Vector of keys that exist (contiguous prefix only)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // keys = [A, B, C, D, E]
    /// // registry contains: A, B, D, E
    /// // returns: [A, B]  (stops at C which is missing)
    /// ```
    fn match_keys(&self, keys: &[K]) -> Vec<K>;

    /// Register keys as stored in external storage.
    ///
    /// # Arguments
    ///
    /// * `keys` - Keys to register
    ///
    /// # Returns
    ///
    /// Number of keys successfully registered (may be less than input if at capacity
    /// and eviction is disabled)
    fn register(&self, keys: &[K]) -> usize;

    /// Remove keys from the registry.
    ///
    /// Note: This does NOT delete objects from external storage,
    /// only removes them from the local catalog.
    ///
    /// # Arguments
    ///
    /// * `keys` - Keys to unregister
    ///
    /// # Returns
    ///
    /// Number of keys that were actually removed (excludes keys not in registry)
    fn unregister(&self, keys: &[K]) -> usize;

    /// Get the storage backend type name (for logging/metrics).
    ///
    /// # Returns
    ///
    /// Static string identifying the backend (e.g., "object_storage", "redis", "mock")
    fn backend_name(&self) -> &'static str;

    /// Check if the registry can accept more registrations.
    ///
    /// # Returns
    ///
    /// * `true` if registry has capacity for new entries (or will evict to make room)
    /// * `false` if registry is at capacity and cannot accept new entries
    fn can_register(&self) -> bool;
}

/// Type alias for a shared external registry reference.
pub type SharedExternalRegistry<K> = Arc<dyn ExternalRegistry<K>>;

/// Common type alias for sequence hash registries (u64 keys).
pub type SequenceHashRegistry = SharedExternalRegistry<u64>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::RwLock;

    /// Mock registry for testing
    #[derive(Debug)]
    struct MockRegistry {
        entries: RwLock<HashSet<u64>>,
        capacity: usize,
    }

    impl MockRegistry {
        fn new(capacity: usize) -> Self {
            Self {
                entries: RwLock::new(HashSet::new()),
                capacity,
            }
        }
    }

    impl ExternalRegistry<u64> for MockRegistry {
        fn match_keys(&self, keys: &[u64]) -> Vec<u64> {
            let entries = self.entries.read().unwrap();
            keys.iter()
                .copied()
                .take_while(|k| entries.contains(k))
                .collect()
        }

        fn register(&self, keys: &[u64]) -> usize {
            let mut entries = self.entries.write().unwrap();
            let mut count = 0;
            for key in keys {
                if entries.len() < self.capacity {
                    entries.insert(*key);
                    count += 1;
                }
            }
            count
        }

        fn unregister(&self, keys: &[u64]) -> usize {
            let mut entries = self.entries.write().unwrap();
            keys.iter().filter(|k| entries.remove(*k)).count()
        }

        fn backend_name(&self) -> &'static str {
            "mock"
        }

        fn can_register(&self) -> bool {
            self.entries.read().unwrap().len() < self.capacity
        }
    }

    #[test]
    fn test_mock_registry_match_keys() {
        let registry = MockRegistry::new(100);
        registry.register(&[1, 2, 3]);

        // Contiguous match
        let matched = registry.match_keys(&[1, 2, 3, 4, 5]);
        assert_eq!(matched, vec![1, 2, 3]);

        // Stops at first miss
        let matched = registry.match_keys(&[1, 4, 2, 3]);
        assert_eq!(matched, vec![1]);
    }

    #[test]
    fn test_mock_registry_register_unregister() {
        let registry = MockRegistry::new(100);

        let count = registry.register(&[10, 20, 30]);
        assert_eq!(count, 3);

        let count = registry.unregister(&[20, 40]);
        assert_eq!(count, 1); // Only 20 was in registry

        let matched = registry.match_keys(&[10, 20, 30]);
        assert_eq!(matched, vec![10]); // 20 is gone, stops there
    }

    #[test]
    fn test_mock_registry_capacity() {
        let registry = MockRegistry::new(2);

        assert!(registry.can_register());
        registry.register(&[1, 2]);
        assert!(!registry.can_register());

        // Can't register more
        let count = registry.register(&[3]);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_backend_name() {
        let registry = MockRegistry::new(10);
        assert_eq!(registry.backend_name(), "mock");
    }

    #[test]
    fn test_shared_registry() {
        let registry: SharedExternalRegistry<u64> = Arc::new(MockRegistry::new(100));
        registry.register(&[1, 2, 3]);

        let matched = registry.match_keys(&[1, 2, 3]);
        assert_eq!(matched.len(), 3);
    }
}

