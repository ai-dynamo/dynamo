//! Response storage trait and implementations
//!
//! Provides pluggable storage for stateful responses with session scoping.
//! Users can bring their own storage backend (Redis, Postgres, S3, etc.)
//! by implementing the ResponseStorage trait.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Stored response with session metadata
///
/// This struct represents a response that has been stored with full
/// tenant and session context for later retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResponse {
    /// Unique response identifier
    pub response_id: String,

    /// Tenant identifier (for isolation)
    pub tenant_id: String,

    /// Session identifier (conversation context)
    pub session_id: String,

    /// The actual response data
    pub response: serde_json::Value,

    /// Creation timestamp (Unix epoch seconds)
    pub created_at: u64,

    /// Expiration timestamp (Unix epoch seconds), if TTL was set
    pub expires_at: Option<u64>,
}

/// Storage errors
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// Response not found (may have expired or never existed)
    #[error("Response not found")]
    NotFound,

    /// Session mismatch - attempted to access response from different session
    #[error("Session mismatch: response belongs to different session")]
    SessionMismatch,

    /// Tenant mismatch - attempted to access response from different tenant
    #[error("Tenant mismatch: response belongs to different tenant")]
    TenantMismatch,

    /// Backend-specific error
    #[error("Storage backend error: {0}")]
    BackendError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Pluggable storage trait for stateful responses
///
/// # Storage Key Pattern
/// Implementations should use the key pattern:
/// `{tenant_id}:{session_id}:responses:{response_id}`
///
/// This ensures:
/// - Tenant isolation (different prefixes)
/// - Session isolation (different prefixes within tenant)
/// - Easy querying by tenant or session
///
/// # Implementations
/// - `InMemoryResponseStorage`: For testing and single-instance deployments
/// - `RedisResponseStorage`: Reference implementation (users can provide)
/// - `PostgresResponseStorage`: Alternative backend (users can provide)
/// - `S3ResponseStorage`: Archival storage (users can provide)
#[async_trait]
pub trait ResponseStorage: Send + Sync {
    /// Store a response in a specific session
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier from request context
    /// * `session_id` - Session identifier from request context
    /// * `response_id` - Optional response ID (uses existing if provided, generates UUID if None)
    /// * `response` - The response data to store
    /// * `ttl` - Optional time-to-live for automatic expiration
    ///
    /// # Returns
    /// The response_id (either provided or generated)
    ///
    /// # Errors
    /// Returns `StorageError::BackendError` if the storage operation fails
    async fn store_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: Option<&str>,
        response: serde_json::Value,
        ttl: Option<Duration>,
    ) -> Result<String, StorageError>;

    /// Get a response, validating tenant and session
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier from request context
    /// * `session_id` - Session identifier from request context
    /// * `response_id` - The response ID to retrieve
    ///
    /// # Returns
    /// The stored response with metadata
    ///
    /// # Errors
    /// * `StorageError::NotFound` - Response doesn't exist or has expired
    /// * `StorageError::TenantMismatch` - Response belongs to different tenant
    /// * `StorageError::SessionMismatch` - Response belongs to different session
    /// * `StorageError::BackendError` - Storage operation failed
    async fn get_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: &str,
    ) -> Result<StoredResponse, StorageError>;

    /// Delete a response
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier from request context
    /// * `session_id` - Session identifier from request context
    /// * `response_id` - The response ID to delete
    ///
    /// # Errors
    /// * `StorageError::NotFound` - Response doesn't exist
    /// * `StorageError::TenantMismatch` - Response belongs to different tenant
    /// * `StorageError::SessionMismatch` - Response belongs to different session
    /// * `StorageError::BackendError` - Storage operation failed
    async fn delete_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: &str,
    ) -> Result<(), StorageError>;

    /// List all responses in a session (optional, for debugging)
    ///
    /// This is useful for testing and debugging, but not required for
    /// core functionality.
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier
    /// * `session_id` - Session identifier
    /// * `limit` - Maximum number of responses to return
    ///
    /// # Returns
    /// List of responses in the session, ordered by creation time
    async fn list_responses(
        &self,
        tenant_id: &str,
        session_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>, StorageError> {
        // Default implementation returns empty list
        // Implementations can override for better functionality
        let _ = (tenant_id, session_id, limit);
        Ok(Vec::new())
    }

    /// Clone a session by copying all responses up to a specific point
    /// 
    /// This enables "branching" conversations - starting a new session
    /// from a checkpoint in an existing one (rewinding).
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant identifier (must be same for source and target)
    /// * `source_session_id` - Source session to clone from
    /// * `target_session_id` - New session to clone into
    /// * `up_to_response_id` - Optional: only clone responses up to this ID (rewind point)
    ///
    /// # Returns
    /// Number of responses cloned
    async fn clone_session(
        &self,
        tenant_id: &str,
        source_session_id: &str,
        target_session_id: &str,
        up_to_response_id: Option<&str>,
    ) -> Result<usize, StorageError> {
        // Default implementation - override for efficiency in production backends
        let responses = self.list_responses(tenant_id, source_session_id, None).await?;
        
        let mut cloned = 0;
        for response in responses {
            // Stop at rewind point if specified
            if let Some(stop_id) = up_to_response_id {
                if response.response_id == stop_id {
                    // Clone this one and stop
                    self.store_response(
                        tenant_id,
                        target_session_id,
                        Some(&response.response_id),
                        response.response.clone(),
                        None,
                    ).await?;
                    cloned += 1;
                    break;
                }
            }
            
            self.store_response(
                tenant_id,
                target_session_id,
                Some(&response.response_id),
                response.response.clone(),
                None,
            ).await?;
            cloned += 1;
        }
        
        Ok(cloned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    /// In-memory storage implementation for testing
    struct InMemoryResponseStorage {
        storage: Arc<RwLock<HashMap<String, StoredResponse>>>,
    }

    impl InMemoryResponseStorage {
        fn new() -> Self {
            Self {
                storage: Arc::new(RwLock::new(HashMap::new())),
            }
        }

        fn make_key(tenant_id: &str, session_id: &str, response_id: &str) -> String {
            format!("{tenant_id}:{session_id}:responses:{response_id}")
        }
    }

    #[async_trait]
    impl ResponseStorage for InMemoryResponseStorage {
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

            // Validate tenant and session before deletion
            if stored.tenant_id != tenant_id {
                return Err(StorageError::TenantMismatch);
            }
            if stored.session_id != session_id {
                return Err(StorageError::SessionMismatch);
            }

            storage.remove(&key);
            Ok(())
        }

        async fn list_responses(
            &self,
            tenant_id: &str,
            session_id: &str,
            limit: Option<usize>,
        ) -> Result<Vec<StoredResponse>, StorageError> {
            let storage = self.storage.read().await;
            let prefix = format!("{tenant_id}:{session_id}:responses:");

            let mut responses: Vec<StoredResponse> = storage
                .iter()
                .filter(|(k, _)| k.starts_with(&prefix))
                .map(|(_, v)| v.clone())
                .collect();

            // Sort by creation time
            responses.sort_by_key(|r| r.created_at);

            // Apply limit
            if let Some(limit) = limit {
                responses.truncate(limit);
            }

            Ok(responses)
        }
    }

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let storage = InMemoryResponseStorage::new();
        let response_data = serde_json::json!({"message": "Hello, world!"});

        let response_id = storage
            .store_response("tenant_a", "session_1", None, response_data.clone(), None)
            .await
            .unwrap();

        let retrieved = storage
            .get_response("tenant_a", "session_1", &response_id)
            .await
            .unwrap();

        assert_eq!(retrieved.tenant_id, "tenant_a");
        assert_eq!(retrieved.session_id, "session_1");
        assert_eq!(retrieved.response, response_data);
    }

    #[tokio::test]
    async fn test_tenant_isolation() {
        let storage = InMemoryResponseStorage::new();
        let response_data = serde_json::json!({"secret": "tenant_a_data"});

        let response_id = storage
            .store_response("tenant_a", "session_1", None, response_data, None)
            .await
            .unwrap();

        // Tenant B should not be able to access tenant A's response
        let result = storage
            .get_response("tenant_b", "session_1", &response_id)
            .await;

        assert!(matches!(result, Err(StorageError::NotFound)));
    }

    #[tokio::test]
    async fn test_session_isolation() {
        let storage = InMemoryResponseStorage::new();
        let response_data = serde_json::json!({"data": "session_1"});

        let response_id = storage
            .store_response("tenant_a", "session_1", None, response_data, None)
            .await
            .unwrap();

        // Same tenant, different session should not be able to access
        let result = storage
            .get_response("tenant_a", "session_2", &response_id)
            .await;

        assert!(matches!(result, Err(StorageError::NotFound)));
    }

    #[tokio::test]
    async fn test_delete_response() {
        let storage = InMemoryResponseStorage::new();
        let response_data = serde_json::json!({"data": "test"});

        let response_id = storage
            .store_response("tenant_a", "session_1", None, response_data, None)
            .await
            .unwrap();

        // Delete the response
        storage
            .delete_response("tenant_a", "session_1", &response_id)
            .await
            .unwrap();

        // Should no longer exist
        let result = storage
            .get_response("tenant_a", "session_1", &response_id)
            .await;

        assert!(matches!(result, Err(StorageError::NotFound)));
    }

    #[tokio::test]
    async fn test_ttl_expiration() {
        let storage = InMemoryResponseStorage::new();
        let response_data = serde_json::json!({"data": "expires soon"});

        let now_before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Store with 10-second TTL
        let response_id = storage
            .store_response("tenant_a", "session_1", None, response_data, Some(Duration::from_secs(10)),
            )
            .await
            .unwrap();

        // Should NOT be expired yet
        let result = storage
            .get_response("tenant_a", "session_1", &response_id)
            .await;

        assert!(result.is_ok());
        let stored = result.unwrap();
        assert!(stored.expires_at.is_some());
        assert!(stored.expires_at.unwrap() >= now_before + 10);
    }

    #[tokio::test]
    async fn test_list_responses() {
        let storage = InMemoryResponseStorage::new();

        // Store multiple responses in same session
        for i in 1..=5 {
            storage
                .store_response(
                    "tenant_a",
                    "session_1",
                    None,
                    serde_json::json!({"turn": i}),
                    None,
                )
                .await
                .unwrap();
        }

        // Store response in different session
        storage
            .store_response("tenant_a", "session_2", None, serde_json::json!({"other": 1}), None)
            .await
            .unwrap();

        let responses = storage
            .list_responses("tenant_a", "session_1", None)
            .await
            .unwrap();

        assert_eq!(responses.len(), 5);
    }

    #[tokio::test]
    async fn test_list_responses_with_limit() {
        let storage = InMemoryResponseStorage::new();

        for i in 1..=10 {
            storage
                .store_response(
                    "tenant_a",
                    "session_1",
                    None,
                    serde_json::json!({"turn": i}),
                    None,
                )
                .await
                .unwrap();
        }

        let responses = storage
            .list_responses("tenant_a", "session_1", Some(3))
            .await
            .unwrap();

        assert_eq!(responses.len(), 3);
    }}
