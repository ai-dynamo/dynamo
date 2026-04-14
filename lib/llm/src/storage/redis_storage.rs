// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis storage backend for stateful responses
//!
//! Provides a production-ready storage implementation using Redis for
//! horizontal scaling across multiple instances.

use async_trait::async_trait;
use std::time::Duration;

use super::{ResponseStorage, StorageError, StoredResponse, validate_key_component};

use deadpool_redis::{Config, Pool, Runtime};
use redis::AsyncCommands;

/// Redis-based storage for stateful responses
///
/// Uses Redis for storage with the following key patterns:
/// - Response data: `{tenant_id}:responses:{response_id}` -> JSON
/// - Session index: `{tenant_id}:session:{session_id}:response_ids` -> SET of response IDs
///
/// Session is stored as metadata inside the response value, NOT as part of
/// the response data key. This allows cross-session access within a tenant.
pub struct RedisResponseStorage {
    pool: Pool,
    max_responses_per_session: usize,
}

impl RedisResponseStorage {
    /// Create a new Redis storage instance
    ///
    /// # Arguments
    /// * `redis_url` - Redis connection URL (e.g., "redis://localhost:6379")
    /// * `max_responses_per_session` - Maximum responses per session (0 = unlimited)
    pub async fn new(
        redis_url: &str,
        max_responses_per_session: usize,
    ) -> Result<Self, StorageError> {
        let cfg = Config::from_url(redis_url);
        let pool = cfg.create_pool(Some(Runtime::Tokio1)).map_err(|e| {
            StorageError::BackendError(format!("Failed to create Redis pool: {}", e))
        })?;

        // Test connection
        let mut conn = pool.get().await.map_err(|e| {
            StorageError::BackendError(format!("Failed to connect to Redis: {}", e))
        })?;

        redis::cmd("PING")
            .query_async::<String>(&mut conn)
            .await
            .map_err(|e| StorageError::BackendError(format!("Redis ping failed: {}", e)))?;

        Ok(Self {
            pool,
            max_responses_per_session,
        })
    }

    fn response_key(tenant_id: &str, response_id: &str) -> String {
        format!("{}:responses:{}", tenant_id, response_id)
    }

    fn session_index_key(tenant_id: &str, session_id: &str) -> String {
        format!("{}:session:{}:response_ids", tenant_id, session_id)
    }
}

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
        validate_key_component(tenant_id)?;
        validate_key_component(session_id)?;
        if let Some(id) = response_id {
            validate_key_component(id)?;
        }

        let response_id = response_id
            .map(|s| s.to_string())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let mut conn =
            self.pool.get().await.map_err(|e| {
                StorageError::BackendError(format!("Failed to get connection: {}", e))
            })?;

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

        let response_key = Self::response_key(tenant_id, &response_id);
        let index_key = Self::session_index_key(tenant_id, session_id);

        let ttl_secs = ttl.map(|d| d.as_secs()).unwrap_or(0);

        if self.max_responses_per_session > 0 {
            // Lua script: atomically check count + SADD + SET
            let script = redis::Script::new(
                r#"
                local count = redis.call('SCARD', KEYS[1])
                if count >= tonumber(ARGV[1]) then
                    return 0
                end
                redis.call('SADD', KEYS[1], ARGV[2])
                redis.call('SET', KEYS[2], ARGV[3])
                if tonumber(ARGV[4]) > 0 then
                    redis.call('EXPIRE', KEYS[2], ARGV[4])
                end
                return 1
                "#,
            );

            let result: i32 = script
                .key(&index_key)
                .key(&response_key)
                .arg(self.max_responses_per_session)
                .arg(&response_id)
                .arg(&json_data)
                .arg(ttl_secs)
                .invoke_async(&mut conn)
                .await
                .map_err(|e| {
                    StorageError::BackendError(format!(
                        "Failed to check/add session response: {}",
                        e
                    ))
                })?;

            if result == 0 {
                return Err(StorageError::SessionFull);
            }
        } else {
            // Unlimited: atomically SADD + SET in a pipeline
            let mut pipe = redis::pipe();
            pipe.atomic();
            pipe.cmd("SADD").arg(&index_key).arg(&response_id);
            if ttl_secs > 0 {
                pipe.cmd("SET")
                    .arg(&response_key)
                    .arg(&json_data)
                    .arg("EX")
                    .arg(ttl_secs);
            } else {
                pipe.cmd("SET").arg(&response_key).arg(&json_data);
            }
            pipe.query_async::<Vec<redis::Value>>(&mut conn)
                .await
                .map_err(|e| {
                    StorageError::BackendError(format!("Failed to store response: {}", e))
                })?;
        }

        // Keep index TTL aligned with response TTLs (fire-and-forget)
        if let Some(ttl) = ttl {
            let new_ttl = ttl.as_secs() as i64;
            if let Err(e) = redis::Script::new(
                r#"
                local cur = redis.call('TTL', KEYS[1])
                if cur == -2 or (cur >= 0 and tonumber(ARGV[1]) > cur) then
                    redis.call('EXPIRE', KEYS[1], ARGV[1])
                end
                return 1
                "#,
            )
            .key(&index_key)
            .arg(new_ttl)
            .invoke_async::<i32>(&mut conn)
            .await
            {
                tracing::warn!("Failed to set session index TTL (non-fatal): {e}");
            }
        } else {
            if let Err(e) = redis::cmd("PERSIST")
                .arg(&index_key)
                .query_async::<i32>(&mut conn)
                .await
            {
                tracing::warn!("Failed to persist session index TTL (non-fatal): {e}");
            }
        }

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

        let response_key = Self::response_key(tenant_id, response_id);

        let mut conn =
            self.pool.get().await.map_err(|e| {
                StorageError::BackendError(format!("Failed to get connection: {}", e))
            })?;

        let json_data: Option<String> = conn
            .get(&response_key)
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to get response: {}", e)))?;

        let json_data = json_data.ok_or(StorageError::NotFound)?;

        let stored: StoredResponse = serde_json::from_str(&json_data).map_err(|e| {
            StorageError::SerializationError(format!("Failed to deserialize: {}", e))
        })?;

        if let Some(expires_at) = stored.expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| StorageError::BackendError(format!("System time error: {}", e)))?
                .as_secs();
            if now > expires_at {
                return Err(StorageError::NotFound);
            }
        }

        if stored.tenant_id != tenant_id {
            return Err(StorageError::NotFound);
        }

        Ok(stored)
    }

    async fn delete_response(
        &self,
        tenant_id: &str,
        _session_id: &str,
        response_id: &str,
    ) -> Result<(), StorageError> {
        validate_key_component(tenant_id)?;
        validate_key_component(response_id)?;

        let response_key = Self::response_key(tenant_id, response_id);

        let mut conn =
            self.pool.get().await.map_err(|e| {
                StorageError::BackendError(format!("Failed to get connection: {}", e))
            })?;

        let json_data: Option<String> = conn
            .get(&response_key)
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to get response: {}", e)))?;

        let json_data = json_data.ok_or(StorageError::NotFound)?;

        let stored: StoredResponse = serde_json::from_str(&json_data).map_err(|e| {
            StorageError::SerializationError(format!("Failed to deserialize: {}", e))
        })?;

        if stored.tenant_id != tenant_id {
            return Err(StorageError::NotFound);
        }

        if let Some(expires_at) = stored.expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| StorageError::BackendError(format!("System time error: {}", e)))?
                .as_secs();
            if now > expires_at {
                conn.del::<_, ()>(&response_key).await.ok();
                return Err(StorageError::NotFound);
            }
        }

        conn.del::<_, ()>(&response_key)
            .await
            .map_err(|e| StorageError::BackendError(format!("Failed to delete response: {}", e)))?;

        let index_key = Self::session_index_key(tenant_id, &stored.session_id);
        conn.srem::<_, _, ()>(&index_key, response_id)
            .await
            .map_err(|e| {
                StorageError::BackendError(format!("Failed to remove from index: {}", e))
            })?;

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

        let index_key = Self::session_index_key(tenant_id, session_id);

        let mut conn =
            self.pool.get().await.map_err(|e| {
                StorageError::BackendError(format!("Failed to get connection: {}", e))
            })?;

        let response_ids: Vec<String> = conn.smembers(&index_key).await.map_err(|e| {
            StorageError::BackendError(format!("Failed to get response IDs: {}", e))
        })?;

        if response_ids.is_empty() {
            return Ok(Vec::new());
        }

        let keys: Vec<String> = response_ids
            .iter()
            .map(|id| Self::response_key(tenant_id, id))
            .collect();

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
            .flatten()
            .filter_map(|json_data| {
                serde_json::from_str::<StoredResponse>(&json_data)
                    .inspect_err(|e| tracing::warn!("Skipping corrupt stored response: {e}"))
                    .ok()
            })
            .filter(|stored| stored.expires_at.map_or(true, |exp| now <= exp))
            .filter(|stored| stored.session_id == session_id)
            .collect();

        responses.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.response_id.cmp(&b.response_id))
        });

        if let Some(cursor_id) = after {
            if let Some(cursor_pos) = responses.iter().position(|r| r.response_id == cursor_id) {
                responses.drain(..=cursor_pos);
            }
        }

        if let Some(limit) = limit {
            responses.truncate(limit);
        }

        Ok(responses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn redis_available() -> bool {
        RedisResponseStorage::new("redis://localhost:6379", 0)
            .await
            .is_ok()
    }

    #[tokio::test]
    #[ignore = "Requires Redis server running locally"]
    async fn test_redis_storage_basic() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let storage = RedisResponseStorage::new("redis://localhost:6379", 0)
            .await
            .unwrap();

        let response_data = serde_json::json!({"message": "Hello from Redis!"});

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

        let retrieved = storage
            .get_response("test_tenant", "test_session", &response_id)
            .await
            .unwrap();

        assert_eq!(retrieved.tenant_id, "test_tenant");
        assert_eq!(retrieved.session_id, "test_session");
        assert_eq!(retrieved.response, response_data);

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

        let storage = RedisResponseStorage::new("redis://localhost:6379", 0)
            .await
            .unwrap();

        let response_data = serde_json::json!({"secret": "tenant_a_data"});

        let response_id = storage
            .store_response("tenant_a", "session_1", None, response_data, None)
            .await
            .unwrap();

        let result = storage
            .get_response("tenant_b", "session_1", &response_id)
            .await;

        assert!(matches!(result, Err(StorageError::NotFound)));

        storage
            .delete_response("tenant_a", "session_1", &response_id)
            .await
            .unwrap();
    }
}
