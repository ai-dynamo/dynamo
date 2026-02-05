//! Session locking for concurrent access control
//!
//! Provides distributed locking when multiple requests target the same session
//! with `previous_response_id`. This prevents race conditions where two requests
//! might read the same previous response and create conflicting state.
//!
//! # Usage Pattern
//!
//! ```ignore
//! // When handling a request with previous_response_id:
//! let guard = lock.acquire("tenant:session", Duration::from_secs(30)).await?;
//! // ... process request ...
//! // Lock automatically released when guard is dropped
//! ```

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};

/// Errors from session locking operations
#[derive(Debug, thiserror::Error)]
pub enum LockError {
    /// Lock acquisition timed out
    #[error("Lock acquisition timed out after {0:?}")]
    Timeout(Duration),

    /// Lock is held by another request
    #[error("Lock is currently held")]
    AlreadyHeld,

    /// Backend error
    #[error("Lock backend error: {0}")]
    BackendError(String),
}

/// Guard that releases the lock when dropped
pub struct LockGuard {
    _permit: Option<OwnedSemaphorePermit>,
    key: String,
    release_fn: Option<Box<dyn FnOnce() + Send>>,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if let Some(release_fn) = self.release_fn.take() {
            release_fn();
        }
        // Permit is automatically released when dropped
    }
}

impl LockGuard {
    /// Create a guard with a semaphore permit
    pub fn with_permit(key: String, permit: OwnedSemaphorePermit) -> Self {
        Self {
            _permit: Some(permit),
            key,
            release_fn: None,
        }
    }

    /// Create a guard with a custom release function
    pub fn with_release_fn<F: FnOnce() + Send + 'static>(key: String, release_fn: F) -> Self {
        Self {
            _permit: None,
            key,
            release_fn: Some(Box::new(release_fn)),
        }
    }

    /// Get the lock key
    pub fn key(&self) -> &str {
        &self.key
    }
}

/// Trait for session locking implementations
///
/// Implementations should provide distributed locking for production use.
/// The in-memory implementation is suitable for single-instance deployments
/// and testing.
#[async_trait]
pub trait SessionLock: Send + Sync {
    /// Acquire a lock for the given key
    ///
    /// # Arguments
    /// * `key` - Lock key (typically `{tenant_id}:{session_id}`)
    /// * `timeout` - Maximum time to wait for lock acquisition
    ///
    /// # Returns
    /// A guard that releases the lock when dropped
    async fn acquire(&self, key: &str, timeout: Duration) -> Result<LockGuard, LockError>;

    /// Try to acquire a lock without waiting
    ///
    /// # Returns
    /// `Ok(guard)` if lock acquired, `Err(AlreadyHeld)` if not available
    async fn try_acquire(&self, key: &str) -> Result<LockGuard, LockError>;

    /// Check if a lock is currently held
    async fn is_locked(&self, key: &str) -> bool;
}

/// In-memory session lock for single-instance deployments and testing
///
/// Uses per-key semaphores to ensure mutual exclusion within a single process.
/// For multi-instance deployments, use a distributed lock implementation
/// (Redis, etcd, etc.).
pub struct InMemorySessionLock {
    locks: Arc<Mutex<HashMap<String, Arc<Semaphore>>>>,
}

impl InMemorySessionLock {
    pub fn new() -> Self {
        Self {
            locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    async fn get_or_create_semaphore(&self, key: &str) -> Arc<Semaphore> {
        let mut locks = self.locks.lock().await;
        locks
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(1)))
            .clone()
    }
}

impl Default for InMemorySessionLock {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SessionLock for InMemorySessionLock {
    async fn acquire(&self, key: &str, timeout: Duration) -> Result<LockGuard, LockError> {
        let semaphore = self.get_or_create_semaphore(key).await;
        let start = Instant::now();

        // Try to acquire with timeout
        match tokio::time::timeout(timeout, semaphore.clone().acquire_owned()).await {
            Ok(Ok(permit)) => Ok(LockGuard::with_permit(key.to_string(), permit)),
            Ok(Err(_)) => Err(LockError::BackendError("Semaphore closed".to_string())),
            Err(_) => Err(LockError::Timeout(start.elapsed())),
        }
    }

    async fn try_acquire(&self, key: &str) -> Result<LockGuard, LockError> {
        let semaphore = self.get_or_create_semaphore(key).await;

        match semaphore.clone().try_acquire_owned() {
            Ok(permit) => Ok(LockGuard::with_permit(key.to_string(), permit)),
            Err(_) => Err(LockError::AlreadyHeld),
        }
    }

    async fn is_locked(&self, key: &str) -> bool {
        let semaphore = self.get_or_create_semaphore(key).await;
        semaphore.available_permits() == 0
    }
}

/// Configuration for session locking behavior
#[derive(Debug, Clone)]
pub struct LockConfig {
    /// Default timeout for lock acquisition
    pub default_timeout: Duration,
    /// Whether to require locks for previous_response_id operations
    pub require_lock: bool,
}

impl Default for LockConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            require_lock: true,
        }
    }
}

impl LockConfig {
    /// Create config from environment variables
    ///
    /// - `DYNAMO_RESPONSES_LOCK_TIMEOUT_SECS`: Lock timeout in seconds (default: 30)
    /// - `DYNAMO_RESPONSES_LOCK_REQUIRED`: Whether locking is required (default: true)
    pub fn from_env() -> Self {
        let default_timeout = std::env::var("DYNAMO_RESPONSES_LOCK_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(30));

        let require_lock = std::env::var("DYNAMO_RESPONSES_LOCK_REQUIRED")
            .ok()
            .map(|s| s.to_lowercase() != "false" && s != "0")
            .unwrap_or(true);

        Self {
            default_timeout,
            require_lock,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_acquire_and_release() {
        let lock = InMemorySessionLock::new();

        // Acquire lock
        let guard = lock
            .acquire("test:session", Duration::from_secs(1))
            .await
            .unwrap();

        assert!(lock.is_locked("test:session").await);

        // Release by dropping
        drop(guard);

        // Should be unlocked now
        assert!(!lock.is_locked("test:session").await);
    }

    #[tokio::test]
    async fn test_try_acquire_when_held() {
        let lock = InMemorySessionLock::new();

        // Acquire lock
        let _guard = lock
            .acquire("test:session", Duration::from_secs(1))
            .await
            .unwrap();

        // Try to acquire again - should fail
        let result = lock.try_acquire("test:session").await;
        assert!(matches!(result, Err(LockError::AlreadyHeld)));
    }

    #[tokio::test]
    async fn test_timeout() {
        let lock = InMemorySessionLock::new();

        // Acquire lock
        let _guard = lock
            .acquire("test:session", Duration::from_secs(1))
            .await
            .unwrap();

        // Try to acquire with short timeout - should timeout
        let result = lock
            .acquire("test:session", Duration::from_millis(100))
            .await;

        assert!(matches!(result, Err(LockError::Timeout(_))));
    }

    #[tokio::test]
    async fn test_different_keys_independent() {
        let lock = InMemorySessionLock::new();

        // Acquire lock on key1
        let _guard1 = lock
            .acquire("tenant:session1", Duration::from_secs(1))
            .await
            .unwrap();

        // Should be able to acquire lock on key2
        let guard2 = lock
            .acquire("tenant:session2", Duration::from_secs(1))
            .await;

        assert!(guard2.is_ok());
    }

    #[tokio::test]
    async fn test_concurrent_acquisition() {
        let lock = Arc::new(InMemorySessionLock::new());
        let key = "test:concurrent";

        // Track which tasks acquired the lock
        let acquired = Arc::new(Mutex::new(Vec::new()));

        let mut handles = vec![];

        for i in 0..5 {
            let lock = lock.clone();
            let acquired = acquired.clone();
            let key = key.to_string();

            handles.push(tokio::spawn(async move {
                // Each task tries to acquire the lock
                if let Ok(guard) = lock.acquire(&key, Duration::from_secs(5)).await {
                    // Record acquisition order
                    acquired.lock().await.push(i);
                    // Hold lock briefly
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    drop(guard);
                }
            }));
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // All 5 tasks should have acquired the lock
        let acquired = acquired.lock().await;
        assert_eq!(acquired.len(), 5);
    }

    #[tokio::test]
    async fn test_config_from_env() {
        // Test default config
        let config = LockConfig::default();
        assert_eq!(config.default_timeout, Duration::from_secs(30));
        assert!(config.require_lock);

        // Test env config (can't really test env vars in unit tests without side effects)
        let config = LockConfig::from_env();
        assert!(config.default_timeout.as_secs() > 0);
    }
}
