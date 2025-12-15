// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for G2→G4 (object storage) offload flow with distributed locking.
//!
//! This test demonstrates:
//! - Using the locking mechanism (.lock/.meta files) for distributed coordination
//! - Verifying lock acquisition and release
//! - Verifying meta file creation marks blocks as offloaded
//! - Verifying re-offload is skipped for blocks that already have meta files
//!
//! Note: Uses a mock in-memory object storage implementation for testing without
//! requiring a real S3/MinIO backend.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    use anyhow::Result;
    use bytes::Bytes;
    use futures::future::BoxFuture;

    use crate::v2::distributed::object::{LockFileContent, ObjectBlockOps, ObjectLockManager};
    use crate::v2::distributed::offload::{
        BoxFuture as PolicyBoxFuture, EvalContext, ObjectLockPresenceFilter, OffloadPolicy,
        PendingTracker,
    };
    use crate::v2::physical::transfer::PhysicalLayout;
    use crate::v2::{BlockId, G2, SequenceHash};

    /// Create a test sequence hash from a simple integer.
    fn test_hash(n: u64) -> SequenceHash {
        SequenceHash::new(n, 0, 0)
    }

    /// Get current time as seconds since epoch
    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Format timestamp as RFC3339-like string
    fn timestamp_to_string(secs: u64) -> String {
        format!("{}", secs)
    }

    /// Parse timestamp from string
    fn parse_timestamp(s: &str) -> Option<u64> {
        s.parse().ok()
    }

    /// Check if deadline timestamp is expired
    fn is_expired_timestamp(deadline_str: &str) -> bool {
        if let Some(deadline) = parse_timestamp(deadline_str) {
            now_secs() > deadline
        } else {
            true // Can't parse = treat as expired
        }
    }

    // =========================================================================
    // Mock Object Storage Implementation
    // =========================================================================

    /// In-memory mock object storage for testing.
    #[derive(Debug, Default)]
    struct MockObjectStorage {
        objects: RwLock<HashMap<String, Bytes>>,
    }

    impl MockObjectStorage {
        fn new() -> Self {
            Self {
                objects: RwLock::new(HashMap::new()),
            }
        }

        fn has_object(&self, key: &str) -> bool {
            self.objects.read().unwrap().contains_key(key)
        }

        fn get_object(&self, key: &str) -> Option<Bytes> {
            self.objects.read().unwrap().get(key).cloned()
        }

        fn put_object(&self, key: &str, data: Bytes) {
            self.objects.write().unwrap().insert(key.to_string(), data);
        }

        fn delete_object(&self, key: &str) -> bool {
            self.objects.write().unwrap().remove(key).is_some()
        }

        fn put_if_not_exists(&self, key: &str, data: Bytes) -> bool {
            let mut objects = self.objects.write().unwrap();
            if objects.contains_key(key) {
                false
            } else {
                objects.insert(key.to_string(), data);
                true
            }
        }

        #[allow(dead_code)]
        fn list_keys(&self) -> Vec<String> {
            self.objects.read().unwrap().keys().cloned().collect()
        }
    }

    /// Mock ObjectBlockOps implementation using in-memory storage.
    #[allow(dead_code)]
    struct MockObjectBlockClient {
        storage: Arc<MockObjectStorage>,
    }

    #[allow(dead_code)]
    impl MockObjectBlockClient {
        fn new(storage: Arc<MockObjectStorage>) -> Self {
            Self { storage }
        }
    }

    impl ObjectBlockOps for MockObjectBlockClient {
        fn has_blocks(
            &self,
            keys: Vec<SequenceHash>,
        ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
            let storage = self.storage.clone();
            Box::pin(async move {
                keys.into_iter()
                    .map(|hash| {
                        let key = format!("{:?}", hash);
                        let size = storage.get_object(&key).map(|b| b.len());
                        (hash, size)
                    })
                    .collect()
            })
        }

        fn put_blocks(
            &self,
            keys: Vec<SequenceHash>,
            _layout: PhysicalLayout,
            _block_ids: Vec<BlockId>,
        ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
            let storage = self.storage.clone();
            Box::pin(async move {
                keys.into_iter()
                    .map(|hash| {
                        let key = format!("{:?}", hash);
                        storage.put_object(&key, Bytes::from("mock_block_data"));
                        Ok(hash)
                    })
                    .collect()
            })
        }

        fn get_blocks(
            &self,
            keys: Vec<SequenceHash>,
            _layout: PhysicalLayout,
            _block_ids: Vec<BlockId>,
        ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
            let storage = self.storage.clone();
            Box::pin(async move {
                keys.into_iter()
                    .map(|hash| {
                        let key = format!("{:?}", hash);
                        if storage.has_object(&key) {
                            Ok(hash)
                        } else {
                            Err(hash)
                        }
                    })
                    .collect()
            })
        }
    }

    /// Mock ObjectLockManager implementation using in-memory storage.
    struct MockLockManager {
        storage: Arc<MockObjectStorage>,
        instance_id: String,
        lock_timeout_secs: u64,
    }

    impl MockLockManager {
        fn new(storage: Arc<MockObjectStorage>, instance_id: String) -> Self {
            Self {
                storage,
                instance_id,
                lock_timeout_secs: 300,
            }
        }

        fn lock_key(hash: &SequenceHash) -> String {
            format!("{:?}.lock", hash)
        }

        fn meta_key(hash: &SequenceHash) -> String {
            format!("{:?}.meta", hash)
        }

        fn create_lock_content(&self) -> LockFileContent {
            LockFileContent {
                instance_id: self.instance_id.clone(),
                acquired_at: timestamp_to_string(now_secs()),
                deadline: timestamp_to_string(now_secs() + self.lock_timeout_secs),
            }
        }
    }

    impl ObjectLockManager for MockLockManager {
        fn has_meta(&self, hash: SequenceHash) -> PolicyBoxFuture<'static, Result<bool>> {
            let storage = self.storage.clone();
            Box::pin(async move {
                let meta_key = Self::meta_key(&hash);
                Ok(storage.has_object(&meta_key))
            })
        }

        fn try_acquire_lock(&self, hash: SequenceHash) -> PolicyBoxFuture<'static, Result<bool>> {
            let storage = self.storage.clone();
            let lock_content = self.create_lock_content();
            let our_instance_id = self.instance_id.clone();

            Box::pin(async move {
                let lock_key = Self::lock_key(&hash);
                let lock_data =
                    serde_json::to_vec(&lock_content).expect("Failed to serialize lock content");

                // Try conditional put
                if storage.put_if_not_exists(&lock_key, Bytes::from(lock_data.clone())) {
                    return Ok(true); // Acquired lock
                }

                // Lock exists, check if we own it or if it's expired
                if let Some(existing_data) = storage.get_object(&lock_key) {
                    if let Ok(existing_lock) =
                        serde_json::from_slice::<LockFileContent>(&existing_data)
                    {
                        // Check if we own the lock
                        if existing_lock.instance_id == our_instance_id {
                            return Ok(true);
                        }

                        // Check if expired
                        if is_expired_timestamp(&existing_lock.deadline) {
                            // Expired, overwrite
                            storage.put_object(&lock_key, Bytes::from(lock_data));
                            return Ok(true);
                        }
                    }
                }

                Ok(false) // Lock held by another instance
            })
        }

        fn create_meta(&self, hash: SequenceHash) -> PolicyBoxFuture<'static, Result<()>> {
            let storage = self.storage.clone();
            Box::pin(async move {
                let meta_key = Self::meta_key(&hash);
                storage.put_object(&meta_key, Bytes::new());
                Ok(())
            })
        }

        fn release_lock(&self, hash: SequenceHash) -> PolicyBoxFuture<'static, Result<()>> {
            let storage = self.storage.clone();
            Box::pin(async move {
                let lock_key = Self::lock_key(&hash);
                storage.delete_object(&lock_key);
                Ok(())
            })
        }
    }

    // =========================================================================
    // Lock Manager Tests
    // =========================================================================

    /// Test basic lock acquisition and release.
    #[tokio::test]
    async fn test_lock_manager_acquire_and_release() -> Result<()> {
        let storage = Arc::new(MockObjectStorage::new());
        let lock_manager = MockLockManager::new(storage.clone(), "test-instance".to_string());

        let hash = test_hash(12345);

        // Initially no lock or meta
        assert!(!storage.has_object(&MockLockManager::lock_key(&hash)));
        assert!(!storage.has_object(&MockLockManager::meta_key(&hash)));

        // Acquire lock
        let acquired = lock_manager.try_acquire_lock(hash).await?;
        assert!(acquired, "Should acquire lock");
        assert!(storage.has_object(&MockLockManager::lock_key(&hash)));

        // Verify lock content
        let lock_data = storage
            .get_object(&MockLockManager::lock_key(&hash))
            .unwrap();
        let lock_content: LockFileContent = serde_json::from_slice(&lock_data)?;
        assert_eq!(lock_content.instance_id, "test-instance");

        // Create meta
        lock_manager.create_meta(hash).await?;
        assert!(storage.has_object(&MockLockManager::meta_key(&hash)));

        // Release lock
        lock_manager.release_lock(hash).await?;
        assert!(!storage.has_object(&MockLockManager::lock_key(&hash)));

        // Meta should still exist
        assert!(storage.has_object(&MockLockManager::meta_key(&hash)));

        eprintln!("✓ Lock acquisition and release test passed");
        Ok(())
    }

    /// Test that same instance can re-acquire its own lock.
    #[tokio::test]
    async fn test_lock_manager_reacquire_own_lock() -> Result<()> {
        let storage = Arc::new(MockObjectStorage::new());
        let lock_manager = MockLockManager::new(storage.clone(), "test-instance".to_string());

        let hash = test_hash(12345);

        // Acquire lock
        let acquired1 = lock_manager.try_acquire_lock(hash).await?;
        assert!(acquired1, "Should acquire lock first time");

        // Re-acquire same lock (same instance)
        let acquired2 = lock_manager.try_acquire_lock(hash).await?;
        assert!(
            acquired2,
            "Same instance should be able to re-acquire its lock"
        );

        eprintln!("✓ Lock re-acquisition test passed");
        Ok(())
    }

    /// Test that different instance cannot acquire a valid lock.
    #[tokio::test]
    async fn test_lock_manager_contention() -> Result<()> {
        let storage = Arc::new(MockObjectStorage::new());
        let lock_manager1 = MockLockManager::new(storage.clone(), "instance-1".to_string());
        let lock_manager2 = MockLockManager::new(storage.clone(), "instance-2".to_string());

        let hash = test_hash(12345);

        // Instance 1 acquires lock
        let acquired1 = lock_manager1.try_acquire_lock(hash).await?;
        assert!(acquired1, "Instance 1 should acquire lock");

        // Instance 2 tries to acquire same lock
        let acquired2 = lock_manager2.try_acquire_lock(hash).await?;
        assert!(
            !acquired2,
            "Instance 2 should NOT acquire lock held by instance 1"
        );

        // Instance 1 can still re-acquire its own lock
        let acquired1_again = lock_manager1.try_acquire_lock(hash).await?;
        assert!(acquired1_again, "Instance 1 should re-acquire its own lock");

        eprintln!("✓ Lock contention test passed");
        Ok(())
    }

    /// Test that expired locks can be overwritten.
    #[tokio::test]
    async fn test_lock_manager_expired_lock_overwrite() -> Result<()> {
        let storage = Arc::new(MockObjectStorage::new());
        let lock_manager = MockLockManager::new(storage.clone(), "new-instance".to_string());

        let hash = test_hash(12345);

        // Pre-populate an expired lock from another instance
        let expired_lock = LockFileContent {
            instance_id: "old-instance".to_string(),
            acquired_at: timestamp_to_string(0), // Ancient time
            deadline: timestamp_to_string(1),    // Long expired
        };
        let lock_key = MockLockManager::lock_key(&hash);
        storage.put_object(&lock_key, Bytes::from(serde_json::to_vec(&expired_lock)?));

        // New instance should be able to overwrite expired lock
        let acquired = lock_manager.try_acquire_lock(hash).await?;
        assert!(acquired, "Should acquire expired lock");

        // Verify new instance owns the lock
        let lock_data = storage.get_object(&lock_key).unwrap();
        let lock_content: LockFileContent = serde_json::from_slice(&lock_data)?;
        assert_eq!(lock_content.instance_id, "new-instance");

        eprintln!("✓ Expired lock overwrite test passed");
        Ok(())
    }

    /// Test has_meta checks correctly.
    #[tokio::test]
    async fn test_lock_manager_has_meta() -> Result<()> {
        let storage = Arc::new(MockObjectStorage::new());
        let lock_manager = MockLockManager::new(storage.clone(), "test-instance".to_string());

        let hash = test_hash(12345);

        // Initially no meta
        let has_meta_before = lock_manager.has_meta(hash).await?;
        assert!(!has_meta_before, "Should not have meta initially");

        // Create meta
        lock_manager.create_meta(hash).await?;

        // Now has meta
        let has_meta_after = lock_manager.has_meta(hash).await?;
        assert!(has_meta_after, "Should have meta after creation");

        eprintln!("✓ Has meta test passed");
        Ok(())
    }

    // =========================================================================
    // Policy Tests
    // =========================================================================

    /// Test ObjectLockPresenceFilter passes blocks without meta/lock.
    #[tokio::test]
    async fn test_policy_passes_new_blocks() -> Result<()> {
        let storage = Arc::new(MockObjectStorage::new());
        let lock_manager: Arc<dyn ObjectLockManager> = Arc::new(MockLockManager::new(
            storage.clone(),
            "test-instance".to_string(),
        ));
        let filter = ObjectLockPresenceFilter::<G2>::new(lock_manager);

        let hash = test_hash(12345);
        let ctx = EvalContext::<G2>::from_weak(BlockId::default(), hash);

        // Evaluate policy
        let result = match filter.evaluate(&ctx) {
            futures::future::Either::Left(ready) => ready.await,
            futures::future::Either::Right(boxed) => boxed.await,
        };

        assert!(result?, "Policy should pass for new block");

        // Lock should be acquired during evaluation
        assert!(storage.has_object(&MockLockManager::lock_key(&hash)));

        eprintln!("✓ Policy passes new blocks test passed");
        Ok(())
    }

    /// Test ObjectLockPresenceFilter filters blocks with existing meta.
    #[tokio::test]
    async fn test_policy_filters_blocks_with_meta() -> Result<()> {
        let storage = Arc::new(MockObjectStorage::new());

        // Pre-populate meta file
        let hash = test_hash(12345);
        let meta_key = MockLockManager::meta_key(&hash);
        storage.put_object(&meta_key, Bytes::new());

        let lock_manager: Arc<dyn ObjectLockManager> = Arc::new(MockLockManager::new(
            storage.clone(),
            "test-instance".to_string(),
        ));
        let filter = ObjectLockPresenceFilter::<G2>::new(lock_manager);

        let ctx = EvalContext::<G2>::from_weak(BlockId::default(), hash);

        // Evaluate policy
        let result = match filter.evaluate(&ctx) {
            futures::future::Either::Left(ready) => ready.await,
            futures::future::Either::Right(boxed) => boxed.await,
        };

        assert!(!result?, "Policy should filter block with existing meta");

        eprintln!("✓ Policy filters blocks with meta test passed");
        Ok(())
    }

    /// Test ObjectLockPresenceFilter filters blocks with valid lock from another instance.
    #[tokio::test]
    async fn test_policy_filters_locked_blocks() -> Result<()> {
        let storage = Arc::new(MockObjectStorage::new());

        // Pre-populate lock from another instance
        let hash = test_hash(12345);
        let lock_content = LockFileContent {
            instance_id: "other-instance".to_string(),
            acquired_at: timestamp_to_string(now_secs()),
            deadline: timestamp_to_string(now_secs() + 300), // 5 min in future
        };
        let lock_key = MockLockManager::lock_key(&hash);
        storage.put_object(&lock_key, Bytes::from(serde_json::to_vec(&lock_content)?));

        let lock_manager: Arc<dyn ObjectLockManager> = Arc::new(MockLockManager::new(
            storage.clone(),
            "test-instance".to_string(),
        ));
        let filter = ObjectLockPresenceFilter::<G2>::new(lock_manager);

        let ctx = EvalContext::<G2>::from_weak(BlockId::default(), hash);

        // Evaluate policy
        let result = match filter.evaluate(&ctx) {
            futures::future::Either::Left(ready) => ready.await,
            futures::future::Either::Right(boxed) => boxed.await,
        };

        assert!(
            !result?,
            "Policy should filter block locked by another instance"
        );

        eprintln!("✓ Policy filters locked blocks test passed");
        Ok(())
    }

    /// Test ObjectLockPresenceFilter with pending tracker.
    #[tokio::test]
    async fn test_policy_filters_pending_blocks() -> Result<()> {
        let storage = Arc::new(MockObjectStorage::new());
        let lock_manager: Arc<dyn ObjectLockManager> = Arc::new(MockLockManager::new(
            storage.clone(),
            "test-instance".to_string(),
        ));

        let pending_tracker = Arc::new(PendingTracker::new());
        let filter = ObjectLockPresenceFilter::<G2>::new(lock_manager)
            .with_pending_tracker(pending_tracker.clone());

        let hash = test_hash(12345);

        // Mark as pending
        let _guard = pending_tracker.guard(hash);

        let ctx = EvalContext::<G2>::from_weak(BlockId::default(), hash);

        // Evaluate policy
        let result = match filter.evaluate(&ctx) {
            futures::future::Either::Left(ready) => ready.await,
            futures::future::Either::Right(boxed) => boxed.await,
        };

        assert!(!result?, "Policy should filter pending block");

        eprintln!("✓ Policy filters pending blocks test passed");
        Ok(())
    }

    /// Test batch evaluation filters correctly.
    #[tokio::test]
    async fn test_policy_batch_evaluation() -> Result<()> {
        let storage = Arc::new(MockObjectStorage::new());

        // Pre-populate meta for some blocks
        let hash1 = test_hash(1);
        let hash2 = test_hash(2);
        let hash3 = test_hash(3);

        storage.put_object(&MockLockManager::meta_key(&hash1), Bytes::new()); // Has meta

        let lock_manager: Arc<dyn ObjectLockManager> = Arc::new(MockLockManager::new(
            storage.clone(),
            "test-instance".to_string(),
        ));
        let filter = ObjectLockPresenceFilter::<G2>::new(lock_manager);

        let contexts = vec![
            EvalContext::<G2>::from_weak(0, hash1), // Has meta - should filter
            EvalContext::<G2>::from_weak(1, hash2), // New - should pass
            EvalContext::<G2>::from_weak(2, hash3), // New - should pass
        ];

        // Evaluate batch
        let result = match filter.evaluate_batch(&contexts) {
            futures::future::Either::Left(ready) => ready.await,
            futures::future::Either::Right(boxed) => boxed.await,
        };

        let results = result?;
        assert_eq!(results.len(), 3);
        assert!(!results[0], "Block with meta should be filtered");
        assert!(results[1], "New block should pass");
        assert!(results[2], "New block should pass");

        eprintln!("✓ Policy batch evaluation test passed");
        Ok(())
    }
}
