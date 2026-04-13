// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Failover lock using POSIX `flock(LOCK_EX)`.
//!
//! Port of `lib/gpu_memory_service/failover_lock/flock/lock.py`.
//!
//! Uses `flock(LOCK_EX)` on a shared file as the lock primitive. The Linux
//! kernel is the lock manager — no server process, no sidecar, no protocol.
//! The lock is automatically released when the holding process dies (even
//! via SIGKILL), because the kernel closes all file descriptors.

use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::error::{GmsError, GmsResult};

/// Failover lock backed by `flock(2)`.
pub struct FlockFailoverLock {
    lock_path: PathBuf,
    file: Option<std::fs::File>,
    engine_id: Option<String>,
}

impl FlockFailoverLock {
    /// Create a new failover lock instance (does NOT acquire).
    pub fn new(lock_path: PathBuf) -> Self {
        Self {
            lock_path,
            file: None,
            engine_id: None,
        }
    }

    /// Acquire the exclusive lock via non-blocking poll loop.
    pub async fn acquire(
        &mut self,
        engine_id: &str,
        timeout: Option<Duration>,
    ) -> GmsResult<()> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.lock_path)
            .map_err(GmsError::Io)?;

        let fd = file.as_raw_fd();
        let start = Instant::now();
        let poll_interval = Duration::from_millis(100);

        loop {
            let result = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
            if result == 0 {
                // Got the lock — write engine_id for observability
                use std::io::Write;
                file.set_len(0).map_err(GmsError::Io)?;
                (&file).write_all(engine_id.as_bytes()).map_err(GmsError::Io)?;

                self.file = Some(file);
                self.engine_id = Some(engine_id.to_string());
                tracing::info!("Failover lock acquired: {engine_id}");
                return Ok(());
            }

            // EWOULDBLOCK — lock is held by someone else
            if let Some(timeout) = timeout {
                if start.elapsed() >= timeout {
                    return Err(GmsError::LockTimeout);
                }
            }
            tokio::time::sleep(poll_interval).await;
        }
    }

    /// Release the lock.
    pub fn release(&mut self) {
        if let Some(file) = self.file.take() {
            tracing::info!("Failover lock released: {:?}", self.engine_id);
            // Dropping the File closes the fd, which releases the flock
            drop(file);
            self.engine_id = None;
        }
    }

    /// Read the current lock owner from the lock file.
    pub fn owner(&self) -> Option<String> {
        match std::fs::read_to_string(&self.lock_path) {
            Ok(content) => {
                let trimmed = content.trim().to_string();
                if trimmed.is_empty() { None } else { Some(trimmed) }
            }
            Err(_) => None,
        }
    }

    /// Path to the lock file.
    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }

    /// Whether this instance currently holds the lock.
    pub fn is_held(&self) -> bool {
        self.file.is_some()
    }
}

impl Drop for FlockFailoverLock {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_acquire_release() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("test.lock");

        let mut lock = FlockFailoverLock::new(lock_path.clone());
        assert!(!lock.is_held());

        lock.acquire("engine-1", Some(Duration::from_secs(1)))
            .await
            .unwrap();
        assert!(lock.is_held());
        assert_eq!(lock.owner(), Some("engine-1".to_string()));

        lock.release();
        assert!(!lock.is_held());
    }

    #[tokio::test]
    async fn test_drop_releases_lock() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("test.lock");

        {
            let mut lock = FlockFailoverLock::new(lock_path.clone());
            lock.acquire("engine-1", None).await.unwrap();
        }

        // Should be able to acquire again
        let mut lock2 = FlockFailoverLock::new(lock_path);
        lock2
            .acquire("engine-2", Some(Duration::from_secs(1)))
            .await
            .unwrap();
        assert!(lock2.is_held());
    }
}
