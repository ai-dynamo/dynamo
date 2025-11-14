// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::error::Result;
use dynamo_memory::{
    ArenaAllocator, ArenaBuffer, MemoryDescription, SystemStorage,
    nixl::{NixlAgent, NixlRegistered, register_with_nixl},
};
use moka::future::Cache;
use std::sync::Arc;

/// Fast tier using ArenaAllocator with NIXL-registered memory
pub struct FastTier {
    allocator: ArenaAllocator<NixlRegistered<SystemStorage>>,
    cache: Cache<String, Arc<ArenaBuffer<NixlRegistered<SystemStorage>>>>,
    #[allow(dead_code)]
    max_bytes: usize,
}

impl FastTier {
    /// Create a new fast tier
    pub fn new(max_bytes: usize, nixl_agent: &Arc<NixlAgent>) -> Result<Self> {
        // Create NIXL-registered storage
        let storage = SystemStorage::new(max_bytes)?;
        let registered = register_with_nixl(storage, nixl_agent, None)
            .map_err(|_| crate::error::M3Error::Internal("NIXL registration failed".into()))?;

        // Create arena allocator with 2MB pages
        let page_size = 2 << 20; // 2MiB pages
        let allocator = ArenaAllocator::new(registered, page_size)?;

        // Setup Moka cache with LRU eviction
        // ArenaBuffer is automatically freed when Arc drops (via Drop trait)
        let cache = Cache::builder()
            .max_capacity(max_bytes as u64)
            .weigher(
                |_k: &String, v: &Arc<ArenaBuffer<NixlRegistered<SystemStorage>>>| {
                    (v.size() as u64).try_into().unwrap_or(u32::MAX)
                },
            )
            .build();

        Ok(Self {
            allocator,
            cache,
            max_bytes,
        })
    }

    /// Get value from cache
    pub async fn get(&self, key: &str) -> Option<Arc<ArenaBuffer<NixlRegistered<SystemStorage>>>> {
        self.cache.get(key).await
    }

    /// Put value into cache
    pub async fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        // Allocate from arena
        let buffer = self.allocator.allocate(data.len())?;

        // Copy data into buffer
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.addr() as *mut u8, data.len());
        }

        // Insert into cache (Arc wrapper for shared ownership)
        self.cache.insert(key.to_string(), Arc::new(buffer)).await;

        Ok(())
    }

    /// Invalidate a key from cache
    pub async fn invalidate(&self, key: &str) {
        self.cache.invalidate(key).await;
    }

    /// Invalidate multiple keys
    #[allow(dead_code)]
    pub async fn invalidate_batch(&self, keys: &[&str]) {
        for key in keys {
            self.cache.invalidate(*key).await;
        }
    }

    /// Get current cache size (approximate, based on weighted size)
    pub fn current_bytes(&self) -> u64 {
        self.cache.weighted_size()
    }

    /// Get max capacity
    #[allow(dead_code)]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Get current entry count
    #[allow(dead_code)]
    pub fn entry_count(&self) -> u64 {
        self.cache.entry_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_agent() -> Arc<NixlAgent> {
        Arc::new(NixlAgent::new("test_fast_tier").expect("failed to create agent"))
    }

    #[tokio::test]
    async fn test_fast_tier_put_get() {
        let agent = create_test_agent();
        let tier = FastTier::new(10 << 20, &agent).unwrap(); // 10MB

        let data = vec![42u8; 1000];
        tier.put("test_key", &data).await.unwrap();

        let buffer = tier.get("test_key").await.unwrap();
        let retrieved =
            unsafe { std::slice::from_raw_parts(buffer.addr() as *const u8, buffer.size()) };
        assert_eq!(retrieved, data.as_slice());
    }

    #[tokio::test]
    async fn test_fast_tier_invalidate() {
        let agent = create_test_agent();
        let tier = FastTier::new(10 << 20, &agent).unwrap();

        let data = vec![42u8; 1000];
        tier.put("test_key", &data).await.unwrap();

        assert!(tier.get("test_key").await.is_some());

        tier.invalidate("test_key").await;

        assert!(tier.get("test_key").await.is_none());
    }

    #[tokio::test]
    async fn test_fast_tier_multiple_entries() {
        let agent = create_test_agent();
        let tier = FastTier::new(10 << 20, &agent).unwrap();

        for i in 0..10 {
            let key = format!("key_{}", i);
            let data = vec![i as u8; 1000];
            tier.put(&key, &data).await.unwrap();
        }

        assert_eq!(tier.entry_count(), 10);

        for i in 0..10 {
            let key = format!("key_{}", i);
            let buffer = tier.get(&key).await.unwrap();
            let retrieved =
                unsafe { std::slice::from_raw_parts(buffer.addr() as *const u8, buffer.size()) };
            assert_eq!(retrieved[0], i as u8);
        }
    }

    #[tokio::test]
    async fn test_fast_tier_eviction() {
        let agent = create_test_agent();
        let small_size = 5 << 20; // 5MB
        let tier = FastTier::new(small_size, &agent).unwrap();

        // Fill with 10 x 1MB entries, should trigger evictions
        for i in 0..10 {
            let key = format!("key_{}", i);
            let data = vec![i as u8; 1 << 20]; // 1MB each
            tier.put(&key, &data).await.unwrap();
        }

        // Not all entries should be present due to eviction
        assert!(tier.entry_count() < 10);

        // More recent entries should still be available
        assert!(tier.get("key_9").await.is_some());
    }
}
