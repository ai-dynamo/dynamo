// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed read-write lock implementation using etcd atomic transactions

use std::time::Duration;

use etcd_client::{Compare, CompareOp, PutOptions, Txn, TxnOp};

use anyhow::Result;

use super::Client;

/// Timeout for acquiring read lock when downloading snapshots
const DEFAULT_READ_LOCK_TIMEOUT_SECS: u64 = 30;

/// Distributed read-write lock for coordinating operations across multiple processes
///
/// This implementation uses etcd atomic transactions to prevent race conditions:
/// - Write locks are exclusive (no readers or writers can coexist)
/// - Read locks are shared (multiple readers allowed, but no writers)
/// - All lock operations use atomic compare-and-set to ensure correctness
/// - Locks are bound to leases for automatic cleanup on client failure
#[derive(Clone)]
pub struct DistributedRWLock {
    lock_prefix: String,
}

pub struct WriteLockGuard<'a> {
    rwlock: &'a DistributedRWLock,
    etcd_client: &'a Client,
}

impl Drop for WriteLockGuard<'_> {
    fn drop(&mut self) {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let rwlock = self.rwlock.clone();
                let etcd_client = self.etcd_client.clone();
                handle.spawn(async move {
                    let write_key = format!("v1/{}/writer", rwlock.lock_prefix);
                    if let Err(e) = etcd_client.kv_delete(write_key.as_str(), None).await {
                        tracing::warn!("Failed to release write lock in drop: {e:?}");
                    }
                });
            }
            Err(_) => {
                tracing::error!(
                    "WriteLockGuard dropped outside tokio runtime - lock not released! \
                     Lock will be cleaned up when etcd lease expires."
                );
            }
        }
    }
}

pub struct ReadLockGuard<'a> {
    rwlock: &'a DistributedRWLock,
    etcd_client: &'a Client,
    reader_id: String,
}

impl Drop for ReadLockGuard<'_> {
    fn drop(&mut self) {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let rwlock = self.rwlock.clone();
                let etcd_client = self.etcd_client.clone();
                let reader_id = self.reader_id.clone();
                handle.spawn(async move {
                    let reader_key = format!("v1/{}/readers/{reader_id}", rwlock.lock_prefix);
                    if let Err(e) = etcd_client.kv_delete(reader_key.as_str(), None).await {
                        tracing::warn!("Failed to release read lock in drop: {e:?}");
                    }
                });
            }
            Err(_) => {
                tracing::error!(
                    "ReadLockGuard dropped outside tokio runtime - lock not released! \
                     Lock will be cleaned up when etcd lease expires."
                );
            }
        }
    }
}

impl DistributedRWLock {
    /// Create a new distributed RWLock with the given prefix
    ///
    /// The lock will create keys under:
    /// - `v1/{prefix}/writer` for the write lock
    /// - `v1/{prefix}/readers/{reader_id}` for read locks
    pub fn new(lock_prefix: String) -> Self {
        Self { lock_prefix }
    }

    /// Try to acquire exclusive write lock (non-blocking)
    ///
    /// Returns `Some(WriteLockGuard)` if acquired, `None` if readers exist or lock unavailable.
    /// The guard automatically releases the lock when dropped.
    ///
    /// Implementation strategy:
    /// 1. Atomically create writer key if it doesn't exist
    /// 2. Immediately check if any readers exist
    /// 3. If readers found, rollback (delete writer key) and return None
    ///
    /// Note: There is still a small race window (sub-millisecond) where a reader could acquire
    /// a lock between steps 2-3.
    pub async fn try_write_lock<'a>(
        &'a self,
        etcd_client: &'a Client,
    ) -> Option<WriteLockGuard<'a>> {
        let write_key = format!("v1/{}/writer", self.lock_prefix);
        let lease_id = etcd_client.lease_id();
        let put_options = PutOptions::new().with_lease(lease_id as i64);

        // Step 1: Atomically create write lock only if it doesn't exist
        let txn = Txn::new()
            .when(vec![Compare::version(
                write_key.as_str(),
                CompareOp::Equal,
                0,
            )])
            .and_then(vec![TxnOp::put(
                write_key.as_str(),
                b"writing",
                Some(put_options),
            )]);

        // Execute the atomic transaction
        match etcd_client.etcd_client().kv_client().txn(txn).await {
            Ok(response) if response.succeeded() => {
                // Step 2: Immediately check if any readers exist
                let reader_prefix = format!("v1/{}/readers/", self.lock_prefix);
                match etcd_client.kv_get_prefix(&reader_prefix).await {
                    Ok(readers) if !readers.is_empty() => {
                        // Readers exist! Rollback - delete our writer key
                        tracing::debug!(
                            "Found {} reader(s) after acquiring write lock, rolling back",
                            readers.len()
                        );
                        if let Err(e) = etcd_client.kv_delete(write_key.as_str(), None).await {
                            tracing::warn!("Failed to rollback write lock: {e:?}");
                        }
                        None
                    }
                    Ok(_) => {
                        // No readers, we successfully hold the write lock
                        tracing::debug!("Successfully acquired write lock with no readers");
                        Some(WriteLockGuard {
                            rwlock: self,
                            etcd_client,
                        })
                    }
                    Err(e) => {
                        // Error checking for readers - rollback to be safe
                        tracing::warn!(
                            "Failed to check for readers, rolling back write lock: {e:?}"
                        );
                        let _ = etcd_client.kv_delete(write_key.as_str(), None).await;
                        None
                    }
                }
            }
            Ok(_) => {
                tracing::debug!("Write lock already exists, transaction failed");
                None
            }
            Err(e) => {
                tracing::warn!("Failed to execute write lock transaction: {e:?}");
                None
            }
        }
    }

    /// Acquire shared read lock with polling retry
    ///
    /// Polls every 100ms until write lock is released, then atomically acquires read lock.
    /// The guard automatically releases the lock when dropped.
    /// Uses atomic transaction to prevent race with writer - the check for no write lock
    /// and creation of read lock happen in a single atomic operation.
    ///
    /// # Arguments
    /// * `etcd_client` - The etcd client
    /// * `reader_id` - Unique identifier for this reader
    /// * `timeout` - Optional timeout, defaults to 5 seconds
    pub async fn read_lock_with_wait<'a>(
        &'a self,
        etcd_client: &'a Client,
        reader_id: &str,
        timeout: Option<Duration>,
    ) -> Result<ReadLockGuard<'a>> {
        let timeout = timeout.unwrap_or(Duration::from_secs(DEFAULT_READ_LOCK_TIMEOUT_SECS));
        let write_key = format!("v1/{}/writer", self.lock_prefix);
        let reader_key = format!("v1/{}/readers/{reader_id}", self.lock_prefix);
        let deadline = tokio::time::Instant::now() + timeout;
        let lease_id = etcd_client.lease_id();

        loop {
            // Check if timeout exceeded
            if tokio::time::Instant::now() > deadline {
                anyhow::bail!("Timeout waiting for read lock after {:?}", timeout);
            }

            // Try to atomically acquire read lock
            // The transaction checks that no writer exists and creates reader key atomically
            let put_options = PutOptions::new().with_lease(lease_id as i64);

            // Build atomic transaction: create reader key only if write_key doesn't exist
            let txn = Txn::new()
                .when(vec![Compare::version(
                    write_key.as_str(),
                    CompareOp::Equal,
                    0,
                )])
                .and_then(vec![TxnOp::put(
                    reader_key.as_str(),
                    b"reading",
                    Some(put_options),
                )]);

            // Execute the atomic transaction
            match etcd_client.etcd_client().kv_client().txn(txn).await {
                Ok(response) if response.succeeded() => {
                    tracing::debug!("Acquired read lock for reader {}", reader_id);
                    return Ok(ReadLockGuard {
                        rwlock: self,
                        etcd_client,
                        reader_id: reader_id.to_string(),
                    });
                }
                Ok(_) => {
                    tracing::trace!("Write lock exists or was created, retrying after delay");
                }
                Err(e) => {
                    tracing::warn!("Failed to execute read lock transaction: {e:?}");
                }
            }

            // Wait before next retry
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

#[cfg(feature = "testing-etcd")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Runtime;
    use std::sync::Arc;
    use tokio::sync::Barrier;

    /// Test the DistributedRWLock behavior
    ///
    /// This test verifies:
    /// 1. Multiple readers can acquire read locks simultaneously
    /// 2. Write lock fails when readers are active
    /// 3. Write lock succeeds when no locks are held
    /// 4. Read lock waits for write lock to be released
    #[tokio::test]
    async fn test_distributed_rwlock() {
        // Setup: Create etcd client
        let runtime = Runtime::from_settings().unwrap();
        let etcd_client = Client::builder()
            .etcd_url(vec!["http://localhost:2379".to_string()])
            .build()
            .unwrap();
        let etcd_client = Client::new(etcd_client, runtime).await.unwrap();

        // Prevent runtime from being dropped in async context at end of test
        let etcd_client = std::mem::ManuallyDrop::new(etcd_client);

        // Create RWLock with unique prefix for this test
        let test_id = uuid::Uuid::new_v4();
        let lock_prefix = format!("/test/rwlock/{}", test_id);
        let rwlock = DistributedRWLock::new(lock_prefix.clone());

        // Step 1: Acquire first read lock
        let _reader1_guard = rwlock
            .read_lock_with_wait(&etcd_client, "reader1", Some(Duration::from_secs(5)))
            .await
            .expect("First read lock should succeed");
        println!("✓ Acquired first read lock");

        // Step 2: Acquire second read lock (should succeed - multiple readers allowed)
        let _reader2_guard = rwlock
            .read_lock_with_wait(&etcd_client, "reader2", Some(Duration::from_secs(5)))
            .await
            .expect("Second read lock should succeed");
        println!("✓ Acquired second read lock");

        // Step 3: Try to acquire write lock (should fail - readers are active)
        let write_result = rwlock.try_write_lock(&etcd_client).await;
        assert!(
            write_result.is_none(),
            "Write lock should fail when readers are active"
        );
        println!("✓ Write lock correctly failed with active readers");

        // Step 4: Drop first read lock
        drop(_reader1_guard);
        tokio::time::sleep(Duration::from_millis(50)).await; // Give time for async drop
        println!("✓ Released first read lock");

        // Verify write lock still fails with one reader active
        let write_result_with_one_reader = rwlock.try_write_lock(&etcd_client).await;
        assert!(
            write_result_with_one_reader.is_none(),
            "Write lock should still fail when one reader is active"
        );
        println!("✓ Write lock correctly failed with one reader still active");

        drop(_reader2_guard);
        tokio::time::sleep(Duration::from_millis(50)).await; // Give time for async drop
        println!("✓ Released second read lock");

        // Give etcd a moment to process the deletions
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Step 5: Acquire write lock (should succeed now - no locks held)
        let _write_guard = rwlock
            .try_write_lock(&etcd_client)
            .await
            .expect("Write lock should succeed with no readers");
        println!("✓ Acquired write lock");

        // Step 5a: Try to acquire write lock again (should fail immediately - already held)
        let write_result_already_held = rwlock.try_write_lock(&etcd_client).await;
        assert!(
            write_result_already_held.is_none(),
            "Write lock should fail when another write lock is already held"
        );
        println!("✓ Write lock correctly failed when already held");

        // Step 6: Spawn background task to acquire read lock
        // It should wait because write lock is held
        let barrier = Arc::new(Barrier::new(2));
        let barrier_clone = barrier.clone();
        let rwlock_clone = rwlock.clone();
        let etcd_client_clone = etcd_client.clone();

        let read_task = tokio::spawn(async move {
            println!("→ Background: Attempting to acquire read lock (should wait)...");
            barrier_clone.wait().await; // Signal that we've started

            let start = std::time::Instant::now();
            let _guard = rwlock_clone
                .read_lock_with_wait(&etcd_client_clone, "reader3", Some(Duration::from_secs(10)))
                .await
                .expect("Read lock should eventually succeed");

            let elapsed = start.elapsed();
            println!("✓ Background: Acquired read lock after {:?}", elapsed);

            // Verify it actually waited (should be > 100ms since we sleep before releasing write lock)
            assert!(
                elapsed > Duration::from_millis(50),
                "Read lock should have waited for write lock to be released"
            );

            // Guard will be dropped here, releasing the lock
        });

        // Wait for background task to start
        barrier.wait().await;

        // Give the background task a moment to start polling
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Step 7: Release write lock by dropping guard
        println!("→ Releasing write lock...");
        drop(_write_guard);
        tokio::time::sleep(Duration::from_millis(50)).await; // Give time for async drop
        println!("✓ Released write lock");

        // Step 8: Background task should now succeed
        read_task
            .await
            .expect("Background task should complete successfully");

        // Final cleanup: verify all locks are released
        tokio::time::sleep(Duration::from_millis(100)).await;
        let remaining_locks = etcd_client
            .kv_get_prefix(&format!("v1/{lock_prefix}"))
            .await
            .expect("Should be able to check remaining locks");
        assert!(
            remaining_locks.is_empty(),
            "All locks should be released at end of test"
        );
        println!("✓ All locks cleaned up successfully");

        println!("\n🎉 All DistributedRWLock tests passed!");
    }
}
