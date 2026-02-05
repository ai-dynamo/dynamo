//! Response storage manager
//!
//! Provides a simple manager wrapper around ResponseStorage implementations.

use super::{ResponseStorage, StorageError, StoredResponse};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Simple in-memory storage manager (for initial implementation)
///
/// Users can later replace this with Redis, Postgres, etc.
pub struct ResponseStorageManager {
    storage: Arc<RwLock<HashMap<String, StoredResponse>>>,
}

impl ResponseStorageManager {
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn make_key(tenant_id: &str, session_id: &str, response_id: &str) -> String {
        format!("{tenant_id}:{session_id}:responses:{response_id}")
    }
}

#[async_trait::async_trait]
impl ResponseStorage for ResponseStorageManager {
    async fn store_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: Option<&str>,
        response: serde_json::Value,
        ttl: Option<Duration>,
    ) -> Result<String, StorageError> {
        let response_id = response_id
            .map(|s| s.to_string())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let key = Self::make_key(tenant_id, session_id, &response_id);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let expires_at = ttl.map(|d| now + d.as_secs());

        let stored = StoredResponse {
            response_id: response_id.clone(),
            tenant_id: tenant_id.to_string(),
            session_id: session_id.to_string(),
            response,
            created_at: now,
            expires_at,
        };

        self.storage.write().await.insert(key, stored);

        Ok(response_id)
    }

    async fn get_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: &str,
    ) -> Result<StoredResponse, StorageError> {
        let key = Self::make_key(tenant_id, session_id, response_id);

        let storage = self.storage.read().await;
        let stored = storage.get(&key).ok_or(StorageError::NotFound)?;

        // Check expiration
        if let Some(expires_at) = stored.expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            if now > expires_at {
                return Err(StorageError::NotFound);
            }
        }

        // Validate tenant and session
        if stored.tenant_id != tenant_id {
            return Err(StorageError::TenantMismatch);
        }
        if stored.session_id != session_id {
            return Err(StorageError::SessionMismatch);
        }

        Ok(stored.clone())
    }

    async fn delete_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: &str,
    ) -> Result<(), StorageError> {
        let key = Self::make_key(tenant_id, session_id, response_id);

        let mut storage = self.storage.write().await;
        let stored = storage.get(&key).ok_or(StorageError::NotFound)?;

        // Validate before deletion
        if stored.tenant_id != tenant_id {
            return Err(StorageError::TenantMismatch);
        }
        if stored.session_id != session_id {
            return Err(StorageError::SessionMismatch);
        }

        storage.remove(&key);
        Ok(())
    }
}
