// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis-based distributed session locking using Redlock algorithm
//!
//! Provides distributed locking for session-level concurrency control
//! when multiple instances need to coordinate access to the same session.

use async_trait::async_trait;
use std::time::{Duration, Instant};

use super::{LockError, LockGuard, SessionLock};

#[cfg(feature = "redis-storage")]
use deadpool_redis::Pool;
#[cfg(feature = "redis-storage")]
use redis::AsyncCommands;

/// Default lock TTL for auto-release if process crashes
const DEFAULT_LOCK_TTL_MS: u64 = 30_000; // 30 seconds

/// Retry delay for lock acquisition
const LOCK_RETRY_DELAY_MS: u64 = 50;

/// Lua script for safe lock release (only release if we own it)
const RELEASE_LOCK_SCRIPT: &str = r#"
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"#;

/// Redis-based distributed session lock implementation
///
/// Uses the Redlock algorithm for distributed locking:
/// - SET with NX (only if not exists) and PX (with expiration)
/// - Unique lock ID (UUID) for safe release
/// - Lua script for atomic check-and-delete on release
///
/// # Example
///
/// ```ignore
/// use dynamo_llm::storage::RedisSessionLock;
/// use std::time::Duration;
///
/// let lock = RedisSessionLock::new("redis://localhost:6379").await?;
/// let guard = lock.acquire("tenant:session", Duration::from_secs(30)).await?;
/// // ... do work ...
/// // Lock automatically released when guard is dropped
/// ```
#[cfg(feature = "redis-storage")]
pub struct RedisSessionLock {
    pool: Pool,
    lock_ttl_ms: u64,
}

#[cfg(feature = "redis-storage")]
impl RedisSessionLock {
    /// Create a new Redis session lock instance
    ///
    /// # Arguments
    /// * `redis_url` - Redis connection URL (e.g., "redis://localhost:6379")
    ///
    /// # Errors
    /// Returns error if connection pool cannot be created
    pub async fn new(redis_url: &str) -> Result<Self, LockError> {
        let cfg = deadpool_redis::Config::from_url(redis_url);
        let pool = cfg
            .create_pool(Some(deadpool_redis::Runtime::Tokio1))
            .map_err(|e| LockError::BackendError(format!("Failed to create Redis pool: {}", e)))?;

        // Test connection
        let mut conn = pool
            .get()
            .await
            .map_err(|e| LockError::BackendError(format!("Failed to connect to Redis: {}", e)))?;

        redis::cmd("PING")
            .query_async::<String>(&mut conn)
            .await
            .map_err(|e| LockError::BackendError(format!("Redis ping failed: {}", e)))?;

        Ok(Self {
            pool,
            lock_ttl_ms: DEFAULT_LOCK_TTL_MS,
        })
    }

    /// Create lock from an existing pool
    pub fn from_pool(pool: Pool) -> Self {
        Self {
            pool,
            lock_ttl_ms: DEFAULT_LOCK_TTL_MS,
        }
    }

    /// Set custom lock TTL (safety timeout)
    ///
    /// This is the time after which the lock automatically releases
    /// if the holding process crashes without explicit release.
    pub fn with_lock_ttl(mut self, ttl: Duration) -> Self {
        self.lock_ttl_ms = ttl.as_millis() as u64;
        self
    }

    /// Get the connection pool
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// Generate the lock key
    fn lock_key(key: &str) -> String {
        format!("lock:{}", key)
    }

    /// Try to acquire the lock once (non-blocking)
    async fn try_acquire_once(&self, key: &str) -> Result<Option<String>, LockError> {
        let lock_key = Self::lock_key(key);
        let lock_id = uuid::Uuid::new_v4().to_string();

        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| LockError::BackendError(format!("Failed to get connection: {}", e)))?;

        // SET key value NX PX milliseconds
        // NX = only set if not exists
        // PX = expiration in milliseconds
        let result: Option<String> = redis::cmd("SET")
            .arg(&lock_key)
            .arg(&lock_id)
            .arg("NX")
            .arg("PX")
            .arg(self.lock_ttl_ms)
            .query_async(&mut conn)
            .await
            .map_err(|e| LockError::BackendError(format!("Failed to acquire lock: {}", e)))?;

        if result.is_some() {
            Ok(Some(lock_id))
        } else {
            Ok(None)
        }
    }

    /// Release a lock by ID (atomic check-and-delete)
    async fn release_lock(&self, key: &str, lock_id: &str) -> Result<bool, LockError> {
        let lock_key = Self::lock_key(key);

        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| LockError::BackendError(format!("Failed to get connection: {}", e)))?;

        // Use Lua script for atomic check-and-delete
        let result: i32 = redis::Script::new(RELEASE_LOCK_SCRIPT)
            .key(&lock_key)
            .arg(lock_id)
            .invoke_async(&mut conn)
            .await
            .map_err(|e| LockError::BackendError(format!("Failed to release lock: {}", e)))?;

        Ok(result == 1)
    }
}

#[cfg(feature = "redis-storage")]
#[async_trait]
impl SessionLock for RedisSessionLock {
    async fn acquire(&self, key: &str, timeout: Duration) -> Result<LockGuard, LockError> {
        let start = Instant::now();
        let retry_delay = Duration::from_millis(LOCK_RETRY_DELAY_MS);

        loop {
            // Try to acquire the lock
            match self.try_acquire_once(key).await? {
                Some(lock_id) => {
                    // Successfully acquired - create guard with release function
                    let pool = self.pool.clone();
                    let key_owned = key.to_string();
                    let lock_id_owned = lock_id.clone();

                    let release_fn = move || {
                        // Spawn a task to release the lock
                        // We need to do this because Drop cannot be async
                        let pool = pool.clone();
                        let lock_key = Self::lock_key(&key_owned);
                        let lock_id = lock_id_owned.clone();

                        // Use tokio::spawn to handle the async release
                        if let Ok(handle) = tokio::runtime::Handle::try_current() {
                            handle.spawn(async move {
                                if let Ok(mut conn) = pool.get().await {
                                    let _: Result<i32, _> = redis::Script::new(RELEASE_LOCK_SCRIPT)
                                        .key(&lock_key)
                                        .arg(&lock_id)
                                        .invoke_async(&mut conn)
                                        .await;
                                }
                            });
                        }
                    };

                    return Ok(LockGuard::with_release_fn(key.to_string(), release_fn));
                }
                None => {
                    // Lock not acquired, check timeout
                    if start.elapsed() >= timeout {
                        return Err(LockError::Timeout(start.elapsed()));
                    }

                    // Wait before retrying
                    tokio::time::sleep(retry_delay).await;
                }
            }
        }
    }

    async fn try_acquire(&self, key: &str) -> Result<LockGuard, LockError> {
        match self.try_acquire_once(key).await? {
            Some(lock_id) => {
                let pool = self.pool.clone();
                let key_owned = key.to_string();
                let lock_id_owned = lock_id.clone();

                let release_fn = move || {
                    let pool = pool.clone();
                    let lock_key = Self::lock_key(&key_owned);
                    let lock_id = lock_id_owned.clone();

                    if let Ok(handle) = tokio::runtime::Handle::try_current() {
                        handle.spawn(async move {
                            if let Ok(mut conn) = pool.get().await {
                                let _: Result<i32, _> = redis::Script::new(RELEASE_LOCK_SCRIPT)
                                    .key(&lock_key)
                                    .arg(&lock_id)
                                    .invoke_async(&mut conn)
                                    .await;
                            }
                        });
                    }
                };

                Ok(LockGuard::with_release_fn(key.to_string(), release_fn))
            }
            None => Err(LockError::AlreadyHeld),
        }
    }

    async fn is_locked(&self, key: &str) -> bool {
        let lock_key = Self::lock_key(key);

        let result: Result<bool, _> = async {
            let mut conn =
                self.pool.get().await.map_err(|e| {
                    LockError::BackendError(format!("Failed to get connection: {}", e))
                })?;

            let exists: bool = conn
                .exists(&lock_key)
                .await
                .map_err(|e| LockError::BackendError(format!("Failed to check lock: {}", e)))?;

            Ok(exists)
        }
        .await;

        result.unwrap_or(false)
    }
}

#[cfg(test)]
#[cfg(feature = "redis-storage")]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Helper to check if Redis is available
    async fn redis_available() -> bool {
        match RedisSessionLock::new("redis://localhost:6379").await {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    #[tokio::test]
    #[ignore = "Requires Redis server running locally"]
    async fn test_redis_lock_acquire_release() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let lock = RedisSessionLock::new("redis://localhost:6379")
            .await
            .unwrap();

        // Acquire lock
        let guard = lock
            .acquire("test:lock:1", Duration::from_secs(5))
            .await
            .unwrap();

        assert_eq!(guard.key(), "test:lock:1");
        assert!(lock.is_locked("test:lock:1").await);

        // Release by dropping
        drop(guard);

        // Give time for async release
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(!lock.is_locked("test:lock:1").await);
    }

    #[tokio::test]
    #[ignore = "Requires Redis server running locally"]
    async fn test_redis_lock_contention() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let lock = RedisSessionLock::new("redis://localhost:6379")
            .await
            .unwrap();

        // Acquire lock
        let _guard = lock
            .acquire("test:lock:contention", Duration::from_secs(5))
            .await
            .unwrap();

        // Try to acquire again - should fail
        let result = lock.try_acquire("test:lock:contention").await;
        assert!(matches!(result, Err(LockError::AlreadyHeld)));
    }

    #[tokio::test]
    #[ignore = "Requires Redis server running locally"]
    async fn test_redis_lock_timeout() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let lock = RedisSessionLock::new("redis://localhost:6379")
            .await
            .unwrap();

        // Acquire lock
        let _guard = lock
            .acquire("test:lock:timeout", Duration::from_secs(10))
            .await
            .unwrap();

        // Try to acquire with short timeout - should timeout
        let result = lock
            .acquire("test:lock:timeout", Duration::from_millis(100))
            .await;

        assert!(matches!(result, Err(LockError::Timeout(_))));
    }

    #[tokio::test]
    #[ignore = "Requires Redis server running locally"]
    async fn test_redis_lock_distributed() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        // Simulate distributed locking with multiple "instances"
        let lock1 = Arc::new(
            RedisSessionLock::new("redis://localhost:6379")
                .await
                .unwrap(),
        );
        let lock2 = Arc::new(
            RedisSessionLock::new("redis://localhost:6379")
                .await
                .unwrap(),
        );

        let key = "test:lock:distributed";
        let counter = Arc::new(tokio::sync::Mutex::new(0u64));

        let mut handles = vec![];

        // Spawn tasks using different lock instances
        for i in 0..10 {
            let lock = if i % 2 == 0 {
                lock1.clone()
            } else {
                lock2.clone()
            };
            let counter = counter.clone();
            let key = key.to_string();

            handles.push(tokio::spawn(async move {
                let _guard = lock.acquire(&key, Duration::from_secs(10)).await.unwrap();

                // Critical section
                let current = *counter.lock().await;
                tokio::time::sleep(Duration::from_millis(10)).await;
                *counter.lock().await = current + 1;
            }));
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Counter should be exactly 10 (no lost updates)
        assert_eq!(*counter.lock().await, 10);
    }

    #[tokio::test]
    #[ignore = "Requires Redis server running locally"]
    async fn test_redis_lock_auto_expire() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let lock = RedisSessionLock::new("redis://localhost:6379")
            .await
            .unwrap()
            .with_lock_ttl(Duration::from_millis(500)); // Very short TTL

        // Acquire lock
        let guard = lock
            .acquire("test:lock:expire", Duration::from_secs(1))
            .await
            .unwrap();

        // Forget the guard (simulate crash - don't drop properly)
        std::mem::forget(guard);

        // Wait for auto-expire
        tokio::time::sleep(Duration::from_millis(600)).await;

        // Lock should have expired, allowing new acquisition
        let result = lock.try_acquire("test:lock:expire").await;
        assert!(result.is_ok());

        // Cleanup
        drop(result);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
