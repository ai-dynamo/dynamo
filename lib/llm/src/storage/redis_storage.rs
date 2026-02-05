//! Redis storage backend for stateful responses
//!
//! Provides a production-ready storage implementation using Redis for
//! horizontal scaling across multiple instances.

use async_trait::async_trait;
use std::time::Duration;

use super::{ResponseStorage, StorageError, StoredResponse};

#[cfg(feature = "redis-storage")]
use deadpool_redis::{Config, Pool, Runtime};
#[cfg(feature = "redis-storage")]
use redis::AsyncCommands;

/// Redis-based storage for stateful responses
///
/// Uses Redis for storage with the following key patterns:
/// - Response data: `{tenant_id}:{session_id}:responses:{response_id}` -> JSON
/// - Response index: `{tenant_id}:{session_id}:response_ids` -> SET of response IDs
///
/// # Example
///
/// ```ignore
/// use dynamo_llm::storage::RedisResponseStorage;
///
/// let storage = RedisResponseStorage::new("redis://localhost:6379").await?;
/// storage.store_response("tenant_1", "session_1", None, json!({"data": "value"}), None).await?;
/// ```
#[cfg(feature = "redis-storage")]
pub struct RedisResponseStorage {
    pool: Pool,
}

#[cfg(feature = "redis-storage")]
impl RedisResponseStorage {
    /// Create a new Redis storage instance
    ///
    /// # Arguments
    /// * `redis_url` - Redis connection URL (e.g., "redis://localhost:6379")
    ///
    /// # Errors
    /// Returns error if connection pool cannot be created
    pub async fn new(redis_url: &str) -> Result<Self, StorageError> {
        let cfg = Config::from_url(redis_url);
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1))
            .map_err(|e| StorageError::BackendError(format!("Failed to create Redis pool: {}", e)))?;

        // Test connection
        let mut conn = pool
            .get()
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to connect to Redis: {}", e)))?;

        // Ping to verify connection
        redis::cmd("PING")
            .query_async::<String>(&mut conn)
            .await
            .map_err(|e| StorageError::BackendError(format!("Redis ping failed: {}", e)))?;

        Ok(Self { pool })
    }

    /// Create storage from an existing pool
    pub fn from_pool(pool: Pool) -> Self {
        Self { pool }
    }

    /// Get the connection pool
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// Generate the response data key
    fn response_key(tenant_id: &str, session_id: &str, response_id: &str) -> String {
        format!("{}:{}:responses:{}", tenant_id, session_id, response_id)
    }

    /// Generate the response index key (SET of response IDs)
    fn index_key(tenant_id: &str, session_id: &str) -> String {
        format!("{}:{}:response_ids", tenant_id, session_id)
    }
}

#[cfg(feature = "redis-storage")]
#[async_trait]
impl ResponseStorage for RedisResponseStorage {
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

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| StorageError::BackendError(format!("System time error: {}", e)))?
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

        let json_data = serde_json::to_string(&stored)
            .map_err(|e| StorageError::SerializationError(format!("Failed to serialize: {}", e)))?;

        let response_key = Self::response_key(tenant_id, session_id, &response_id);
        let index_key = Self::index_key(tenant_id, session_id);

        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to get connection: {}", e)))?;

        // Store the response data with optional TTL
        if let Some(ttl) = ttl {
            conn.set_ex::<_, _, ()>(&response_key, &json_data, ttl.as_secs())
                .await
                .map_err(|e| StorageError::BackendError(format!("Failed to store response: {}", e)))?;
        } else {
            conn.set::<_, _, ()>(&response_key, &json_data)
                .await
                .map_err(|e| StorageError::BackendError(format!("Failed to store response: {}", e)))?;
        }

        // Add response ID to the session index
        conn.sadd::<_, _, ()>(&index_key, &response_id)
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to add to index: {}", e)))?;

        // Keep index TTL aligned with response TTLs
        if let Some(ttl) = ttl {
            // Only set expire if it would extend the current TTL
            let current_ttl: i64 = conn
                .ttl(&index_key)
                .await
                .unwrap_or(-1);

            let new_ttl = ttl.as_secs() as i64;
            if current_ttl < 0 || new_ttl > current_ttl {
                conn.expire::<_, ()>(&index_key, new_ttl)
                    .await
                    .map_err(|e| StorageError::BackendError(format!("Failed to set index TTL: {}", e)))?;
            }
        } else {
            // Remove any existing TTL so non-expiring responses remain listable
            redis::cmd("PERSIST")
                .arg(&index_key)
                .query_async::<i32>(&mut conn)
                .await
                .map_err(|e| StorageError::BackendError(format!("Failed to persist index: {}", e)))?;
        }

        Ok(response_id)
    }

    async fn get_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: &str,
    ) -> Result<StoredResponse, StorageError> {
        let response_key = Self::response_key(tenant_id, session_id, response_id);

        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to get connection: {}", e)))?;

        let json_data: Option<String> = conn
            .get(&response_key)
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to get response: {}", e)))?;

        let json_data = json_data.ok_or(StorageError::NotFound)?;

        let stored: StoredResponse = serde_json::from_str(&json_data)
            .map_err(|e| StorageError::SerializationError(format!("Failed to deserialize: {}", e)))?;

        // Check expiration (Redis handles this, but double-check for safety)
        if let Some(expires_at) = stored.expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| StorageError::BackendError(format!("System time error: {}", e)))?
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

        Ok(stored)
    }

    async fn delete_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: &str,
    ) -> Result<(), StorageError> {
        // First verify the response exists and belongs to this tenant/session
        let _ = self.get_response(tenant_id, session_id, response_id).await?;

        let response_key = Self::response_key(tenant_id, session_id, response_id);
        let index_key = Self::index_key(tenant_id, session_id);

        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to get connection: {}", e)))?;

        // Delete the response data
        conn.del::<_, ()>(&response_key)
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to delete response: {}", e)))?;

        // Remove from index
        conn.srem::<_, _, ()>(&index_key, response_id)
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to remove from index: {}", e)))?;

        Ok(())
    }

    async fn list_responses(
        &self,
        tenant_id: &str,
        session_id: &str,
        limit: Option<usize>,
        after: Option<&str>,
    ) -> Result<Vec<StoredResponse>, StorageError> {
        let index_key = Self::index_key(tenant_id, session_id);

        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to get connection: {}", e)))?;

        // Get all response IDs from the index
        let response_ids: Vec<String> = conn
            .smembers(&index_key)
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to get response IDs: {}", e)))?;

        if response_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Build keys for MGET
        let keys: Vec<String> = response_ids
            .iter()
            .map(|id| Self::response_key(tenant_id, session_id, id))
            .collect();

        // Get all responses in one call
        let values: Vec<Option<String>> = conn
            .mget(&keys)
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to get responses: {}", e)))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| StorageError::BackendError(format!("System time error: {}", e)))?
            .as_secs();

        let mut responses: Vec<StoredResponse> = values
            .into_iter()
            .filter_map(|v| v)
            .filter_map(|json_data| serde_json::from_str::<StoredResponse>(&json_data).ok())
            .filter(|stored| {
                // Filter out expired responses
                if let Some(expires_at) = stored.expires_at {
                    now <= expires_at
                } else {
                    true
                }
            })
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
                responses = responses.into_iter().skip(cursor_pos + 1).collect();
            }
            // If cursor not found, return all responses (cursor may have been deleted)
        }

        // Apply limit
        if let Some(limit) = limit {
            responses.truncate(limit);
        }

        Ok(responses)
    }

    async fn fork_session(
        &self,
        tenant_id: &str,
        source_session_id: &str,
        target_session_id: &str,
        up_to_response_id: Option<&str>,
    ) -> Result<usize, StorageError> {
        // Get all responses from source session
        let responses = self.list_responses(tenant_id, source_session_id, None, None).await?;

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
                    )
                    .await?;
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
            )
            .await?;
            cloned += 1;
        }

        Ok(cloned)
    }
}

#[cfg(test)]
#[cfg(feature = "redis-storage")]
mod tests {
    use super::*;

    /// Helper to check if Redis is available
    async fn redis_available() -> bool {
        match RedisResponseStorage::new("redis://localhost:6379").await {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    #[tokio::test]
    #[ignore = "Requires Redis server running locally"]
    async fn test_redis_storage_basic() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let storage = RedisResponseStorage::new("redis://localhost:6379")
            .await
            .unwrap();

        let response_data = serde_json::json!({"message": "Hello from Redis!"});

        // Store
        let response_id = storage
            .store_response(
                "test_tenant",
                "test_session",
                None,
                response_data.clone(),
                Some(Duration::from_secs(60)),
            )
            .await
            .unwrap();

        // Retrieve
        let retrieved = storage
            .get_response("test_tenant", "test_session", &response_id)
            .await
            .unwrap();

        assert_eq!(retrieved.tenant_id, "test_tenant");
        assert_eq!(retrieved.session_id, "test_session");
        assert_eq!(retrieved.response, response_data);

        // Cleanup
        storage
            .delete_response("test_tenant", "test_session", &response_id)
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "Requires Redis server running locally"]
    async fn test_redis_storage_tenant_isolation() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let storage = RedisResponseStorage::new("redis://localhost:6379")
            .await
            .unwrap();

        let response_data = serde_json::json!({"secret": "tenant_a_data"});

        let response_id = storage
            .store_response("tenant_a", "session_1", None, response_data, None)
            .await
            .unwrap();

        // Tenant B should not be able to access
        let result = storage
            .get_response("tenant_b", "session_1", &response_id)
            .await;

        assert!(matches!(result, Err(StorageError::NotFound)));

        // Cleanup
        storage
            .delete_response("tenant_a", "session_1", &response_id)
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "Requires Redis server running locally"]
    async fn test_redis_storage_list_responses() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let storage = RedisResponseStorage::new("redis://localhost:6379")
            .await
            .unwrap();

        let tenant_id = "test_list_tenant";
        let session_id = "test_list_session";

        // Store multiple responses
        let mut ids = Vec::new();
        for i in 1..=5 {
            let id = storage
                .store_response(
                    tenant_id,
                    session_id,
                    None,
                    serde_json::json!({"turn": i}),
                    None,
                )
                .await
                .unwrap();
            ids.push(id);
        }

        // List all
        let responses = storage
            .list_responses(tenant_id, session_id, None, None)
            .await
            .unwrap();

        assert_eq!(responses.len(), 5);

        // List with limit
        let limited = storage
            .list_responses(tenant_id, session_id, Some(3), None)
            .await
            .unwrap();

        assert_eq!(limited.len(), 3);

        // Cleanup
        for id in ids {
            storage
                .delete_response(tenant_id, session_id, &id)
                .await
                .unwrap();
        }
    }
}
