// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response storage manager
//!
//! Provides a simple in-memory implementation of ResponseStorage.

use super::{ResponseStorage, StorageError, StoredResponse, validate_key_component};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Simple in-memory response storage implementation
///
/// Users can later replace this with Redis, Postgres, etc.
///
/// Expired entries are cleaned up lazily: on `get_response` (evicted immediately)
/// and filtered out on `list_responses`. A background sweep could be added for
/// production use, but lazy eviction is sufficient for now.
pub struct InMemoryResponseStorage {
    storage: Arc<RwLock<HashMap<String, StoredResponse>>>,
    max_responses_per_session: usize,
}

impl Default for InMemoryResponseStorage {
    fn default() -> Self {
        Self::new(0)
    }
}

impl InMemoryResponseStorage {
    pub fn new(max_responses_per_session: usize) -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
            max_responses_per_session,
        }
    }

    fn make_key(tenant_id: &str, response_id: &str) -> String {
        format!("{tenant_id}:responses:{response_id}")
    }

    fn now_epoch_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn is_expired(stored: &StoredResponse, now: u64) -> bool {
        stored.expires_at.is_some_and(|exp| now > exp)
    }
}

#[async_trait::async_trait]
impl ResponseStorage for InMemoryResponseStorage {
    async fn store_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: Option<&str>,
        response: serde_json::Value,
        ttl: Option<Duration>,
    ) -> Result<String, StorageError> {
        validate_key_component(tenant_id)?;
        validate_key_component(session_id)?;
        if let Some(id) = response_id {
            validate_key_component(id)?;
        }

        let response_id = response_id
            .map(|s| s.to_string())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let key = Self::make_key(tenant_id, &response_id);
        let now = Self::now_epoch_secs();
        let expires_at = ttl.map(|d| now + d.as_secs());

        let stored = StoredResponse {
            response_id: response_id.clone(),
            tenant_id: tenant_id.to_string(),
            session_id: session_id.to_string(),
            response,
            created_at: now,
            expires_at,
        };

        // Hold a single write lock for both the count check and insert to
        // prevent TOCTOU races where concurrent requests could exceed the limit.
        let mut storage = self.storage.write().await;

        // Enforce max_responses_per_session (0 = unlimited)
        if self.max_responses_per_session > 0 {
            let prefix = format!("{tenant_id}:responses:");
            let count = storage
                .iter()
                .filter(|(k, _)| k.starts_with(&prefix))
                .filter(|(_, v)| v.session_id == session_id)
                .filter(|(_, v)| !Self::is_expired(v, now))
                .count();
            if count >= self.max_responses_per_session {
                return Err(StorageError::SessionFull);
            }
        }

        storage.insert(key, stored);

        Ok(response_id)
    }

    async fn get_response(
        &self,
        tenant_id: &str,
        _session_id: &str,
        response_id: &str,
    ) -> Result<StoredResponse, StorageError> {
        validate_key_component(tenant_id)?;
        validate_key_component(response_id)?;

        let key = Self::make_key(tenant_id, response_id);

        // First try with a read lock
        {
            let storage = self.storage.read().await;
            let stored = storage.get(&key).ok_or(StorageError::NotFound)?;

            // Check expiration — if expired, drop read lock and evict below
            if let Some(expires_at) = stored.expires_at {
                let now = Self::now_epoch_secs();
                if now > expires_at {
                    // Drop read lock before acquiring write lock
                    drop(storage);
                    // Lazy eviction: upgrade to write lock and remove the expired entry
                    self.storage.write().await.remove(&key);
                    return Err(StorageError::NotFound);
                }
            }

            // Defense-in-depth: key scheme guarantees tenant match,
            // but return NotFound (not TenantMismatch) per trait contract
            // to avoid leaking existence to other tenants.
            if stored.tenant_id != tenant_id {
                return Err(StorageError::NotFound);
            }

            Ok(stored.clone())
        }
    }

    async fn delete_response(
        &self,
        tenant_id: &str,
        _session_id: &str,
        response_id: &str,
    ) -> Result<(), StorageError> {
        validate_key_component(tenant_id)?;
        validate_key_component(response_id)?;

        let key = Self::make_key(tenant_id, response_id);

        let mut storage = self.storage.write().await;
        let stored = storage.get(&key).ok_or(StorageError::NotFound)?;

        // Defense-in-depth: key scheme guarantees tenant match,
        // but return NotFound (not TenantMismatch) per trait contract
        // to avoid leaking existence to other tenants.
        if stored.tenant_id != tenant_id {
            return Err(StorageError::NotFound);
        }

        // Check expiration — expired entries should be treated as NotFound
        if Self::is_expired(stored, Self::now_epoch_secs()) {
            storage.remove(&key);
            return Err(StorageError::NotFound);
        }

        storage.remove(&key);
        Ok(())
    }

    async fn list_responses(
        &self,
        tenant_id: &str,
        session_id: &str,
        limit: Option<usize>,
        after: Option<&str>,
    ) -> Result<Vec<StoredResponse>, StorageError> {
        validate_key_component(tenant_id)?;
        validate_key_component(session_id)?;
        if let Some(cursor_id) = after {
            validate_key_component(cursor_id)?;
        }

        let storage = self.storage.read().await;
        let prefix = format!("{tenant_id}:responses:");
        let now = Self::now_epoch_secs();

        let mut responses: Vec<StoredResponse> = storage
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .filter(|(_, v)| v.session_id == session_id)
            // Filter out expired entries
            .filter(|(_, v)| !Self::is_expired(v, now))
            .map(|(_, v)| v.clone())
            .collect();

        // Sort by creation time, then by response_id for stable ordering
        responses.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.response_id.cmp(&b.response_id))
        });

        // Apply cursor: skip all responses up to and including the cursor
        if let Some(cursor_id) = after {
            // Find the cursor response to get its position
            if let Some(cursor_pos) = responses.iter().position(|r| r.response_id == cursor_id) {
                // Skip all responses up to and including the cursor
                responses.drain(..=cursor_pos);
            }
            // If cursor not found, return all responses (cursor may have been deleted)
        }

        // Apply limit
        if let Some(limit) = limit {
            responses.truncate(limit);
        }

        Ok(responses)
    }
}
