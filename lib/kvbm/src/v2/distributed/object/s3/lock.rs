// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3-based distributed lock manager implementation.
//!
//! This module provides [`S3LockManager`], an implementation of [`ObjectLockManager`]
//! that uses S3 conditional PUT operations for atomic lock acquisition.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::future::BoxFuture;

use super::S3ObjectBlockClient;
use crate::v2::distributed::object::{LockFileContent, ObjectLockManager};
use crate::SequenceHash;

/// S3-based implementation of [`ObjectLockManager`].
///
/// Uses conditional PUT (If-None-Match: *) for atomic lock acquisition.
/// Lock files contain instance_id and deadline; stale locks (past deadline)
/// can be overwritten.
///
/// # Lock File Format
///
/// Lock files are stored at `{hash}.lock` as JSON:
/// ```json
/// {
///   "instance_id": "uuid-of-leader-instance",
///   "acquired_at": "2025-12-14T10:30:00Z",
///   "deadline": "2025-12-14T10:35:00Z"
/// }
/// ```
///
/// # Meta File Format
///
/// Meta files are stored at `{hash}.meta` as empty objects (presence-only).
pub struct S3LockManager {
    client: Arc<S3ObjectBlockClient>,
    instance_id: String,
    lock_timeout: Duration,
}

impl S3LockManager {
    /// Default lock timeout: 300 seconds (5 minutes).
    pub const DEFAULT_LOCK_TIMEOUT: Duration = Duration::from_secs(300);

    /// Create a new S3 lock manager.
    ///
    /// # Arguments
    /// * `client` - S3 client for object operations
    /// * `instance_id` - Unique identifier for this instance (e.g., UUID)
    pub fn new(client: Arc<S3ObjectBlockClient>, instance_id: String) -> Self {
        Self {
            client,
            instance_id,
            lock_timeout: Self::DEFAULT_LOCK_TIMEOUT,
        }
    }

    /// Create with a custom lock timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.lock_timeout = timeout;
        self
    }

    /// Format the lock key for a given hash.
    fn lock_key(&self, hash: &SequenceHash) -> String {
        format!("{:?}.lock", hash)
    }

    /// Format the meta key for a given hash.
    fn meta_key(&self, hash: &SequenceHash) -> String {
        format!("{:?}.meta", hash)
    }

    /// Create lock file content with current timestamp.
    fn create_lock_content(&self) -> LockFileContent {
        let now = chrono::Utc::now();
        let deadline = now + chrono::Duration::from_std(self.lock_timeout).unwrap_or_default();
        LockFileContent {
            instance_id: self.instance_id.clone(),
            acquired_at: now.to_rfc3339(),
            deadline: deadline.to_rfc3339(),
        }
    }

    /// Check if a lock's deadline has been breached.
    fn is_lock_expired(lock: &LockFileContent) -> bool {
        if let Ok(deadline) = chrono::DateTime::parse_from_rfc3339(&lock.deadline) {
            let now = chrono::Utc::now();
            now > deadline.with_timezone(&chrono::Utc)
        } else {
            // If we can't parse the deadline, consider it expired
            true
        }
    }
}

impl ObjectLockManager for S3LockManager {
    fn has_meta(&self, hash: SequenceHash) -> BoxFuture<'static, Result<bool>> {
        let client = self.client.clone();
        let meta_key = self.meta_key(&hash);

        Box::pin(async move { client.has_object(&meta_key).await })
    }

    fn try_acquire_lock(&self, hash: SequenceHash) -> BoxFuture<'static, Result<bool>> {
        let client = self.client.clone();
        let lock_key = self.lock_key(&hash);
        let lock_content = self.create_lock_content();
        let our_instance_id = self.instance_id.clone();

        Box::pin(async move {
            // Serialize lock content
            let lock_data = serde_json::to_vec(&lock_content)
                .map_err(|e| anyhow::anyhow!("failed to serialize lock content: {}", e))?;

            // Try conditional PUT (If-None-Match: *)
            match client
                .put_if_not_exists(&lock_key, bytes::Bytes::from(lock_data.clone()))
                .await
            {
                Ok(true) => {
                    // Successfully acquired lock
                    tracing::debug!(lock_key = %lock_key, "Acquired lock");
                    Ok(true)
                }
                Ok(false) => {
                    // Lock exists, read it to check deadline
                    tracing::debug!(lock_key = %lock_key, "Lock exists, checking deadline");
                    match client.get_object(&lock_key).await? {
                        Some(existing_data) => {
                            match serde_json::from_slice::<LockFileContent>(&existing_data) {
                                Ok(existing_lock) => {
                                    // Check if we own the lock
                                    if existing_lock.instance_id == our_instance_id {
                                        tracing::debug!(lock_key = %lock_key, "We own this lock");
                                        return Ok(true);
                                    }

                                    // Check if the lock is expired
                                    if Self::is_lock_expired(&existing_lock) {
                                        tracing::debug!(
                                            lock_key = %lock_key,
                                            old_instance = %existing_lock.instance_id,
                                            deadline = %existing_lock.deadline,
                                            "Lock expired, overwriting"
                                        );
                                        // Overwrite the expired lock
                                        client
                                            .put_object(&lock_key, bytes::Bytes::from(lock_data))
                                            .await?;
                                        Ok(true)
                                    } else {
                                        tracing::debug!(
                                            lock_key = %lock_key,
                                            owner = %existing_lock.instance_id,
                                            deadline = %existing_lock.deadline,
                                            "Lock held by another instance"
                                        );
                                        Ok(false)
                                    }
                                }
                                Err(e) => {
                                    // Malformed lock file, overwrite it
                                    tracing::warn!(
                                        lock_key = %lock_key,
                                        error = %e,
                                        "Malformed lock file, overwriting"
                                    );
                                    client
                                        .put_object(&lock_key, bytes::Bytes::from(lock_data))
                                        .await?;
                                    Ok(true)
                                }
                            }
                        }
                        None => {
                            // Lock was deleted between checks, try to acquire again
                            tracing::debug!(lock_key = %lock_key, "Lock disappeared, retrying");
                            match client
                                .put_if_not_exists(&lock_key, bytes::Bytes::from(lock_data))
                                .await
                            {
                                Ok(created) => Ok(created),
                                Err(e) => Err(e),
                            }
                        }
                    }
                }
                Err(e) => Err(e),
            }
        })
    }

    fn create_meta(&self, hash: SequenceHash) -> BoxFuture<'static, Result<()>> {
        let client = self.client.clone();
        let meta_key = self.meta_key(&hash);

        Box::pin(async move {
            // Create empty meta file to mark block as offloaded
            client.put_object(&meta_key, bytes::Bytes::new()).await?;
            tracing::debug!(meta_key = %meta_key, "Created meta file");
            Ok(())
        })
    }

    fn release_lock(&self, hash: SequenceHash) -> BoxFuture<'static, Result<()>> {
        let client = self.client.clone();
        let lock_key = self.lock_key(&hash);

        Box::pin(async move {
            client.delete_object(&lock_key).await?;
            tracing::debug!(lock_key = %lock_key, "Released lock");
            Ok(())
        })
    }
}


