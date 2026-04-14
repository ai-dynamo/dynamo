// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

    /// Reserved for custom storage backends that enforce session boundaries.
    /// Built-in backends return [`NotFound`](StorageError::NotFound) instead
    /// (see trait-level isolation contract).
    #[error("Session mismatch: response belongs to different session")]
    SessionMismatch,

    /// Reserved for custom storage backends that enforce session boundaries.
    /// Built-in backends return [`NotFound`](StorageError::NotFound) instead
    /// (see trait-level isolation contract).
    #[error("Tenant mismatch: response belongs to different tenant")]
    TenantMismatch,

    /// Invalid key component (contains forbidden characters or exceeds length)
    #[error("Invalid key: {0}")]
    InvalidKey(String),

    /// Session has reached its maximum response count
    #[error("Session full: maximum responses per session reached")]
    SessionFull,

    /// Backend-specific error
    #[error("Storage backend error: {0}")]
    BackendError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Maximum allowed length for key components (tenant_id, session_id, response_id).
const MAX_KEY_COMPONENT_LEN: usize = 256;

/// Validate a key component (tenant_id, session_id, or response_id).
///
/// Rejects values that contain `:` (used as key separator), exceed 256 chars,
/// or contain characters outside `[a-zA-Z0-9._-]`.
pub fn validate_key_component(value: &str) -> Result<(), StorageError> {
    if value.len() > MAX_KEY_COMPONENT_LEN {
        return Err(StorageError::InvalidKey(format!(
            "key component exceeds {} characters",
            MAX_KEY_COMPONENT_LEN,
        )));
    }
    if value.is_empty() {
        return Err(StorageError::InvalidKey(
            "key component must not be empty".to_string(),
        ));
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
    {
        return Err(StorageError::InvalidKey(format!(
            "key component contains invalid characters (allowed: a-zA-Z0-9._-): '{}'",
            value,
        )));
    }
    Ok(())
}

/// Pluggable storage backend for stateful responses.
///
/// # Isolation Contract
///
/// - **Tenant isolation is mandatory.** `get_response` and `delete_response`
///   must return `NotFound` (not `TenantMismatch`) when the tenant does not
///   match, to avoid leaking existence information.
/// - **Session is metadata, not a boundary.** Cross-session access within the
///   same tenant is intentionally allowed for multi-agent workflows.
///
/// # Key Schema
///
/// Implementations should key on `(tenant_id, response_id)`. The recommended
/// storage key pattern is `{tenant_id}:responses:{response_id}`, with
/// `session_id` stored as metadata on the [`StoredResponse`].
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
    async fn store_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: Option<&str>,
        response: serde_json::Value,
        ttl: Option<Duration>,
    ) -> Result<String, StorageError>;

    /// Get a response, validating tenant ownership.
    ///
    /// `session_id` is accepted for interface uniformity but cross-session
    /// reads within a tenant are permitted by design.
    async fn get_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: &str,
    ) -> Result<StoredResponse, StorageError>;

    /// Delete a response, validating tenant ownership.
    async fn delete_response(
        &self,
        tenant_id: &str,
        session_id: &str,
        response_id: &str,
    ) -> Result<(), StorageError>;

    /// List all responses in a session (optional, for debugging)
    async fn list_responses(
        &self,
        tenant_id: &str,
        session_id: &str,
        limit: Option<usize>,
        after: Option<&str>,
    ) -> Result<Vec<StoredResponse>, StorageError> {
        let _ = (tenant_id, session_id, limit, after);
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryResponseStorage;

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let storage = InMemoryResponseStorage::new(0);
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
        let storage = InMemoryResponseStorage::new(0);
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
    async fn test_cross_session_access_within_tenant() {
        let storage = InMemoryResponseStorage::new(0);
        let response_data = serde_json::json!({"data": "session_1"});

        let response_id = storage
            .store_response("tenant_a", "session_1", None, response_data.clone(), None)
            .await
            .unwrap();

        // Same tenant, different session CAN access (session is metadata, not boundary)
        let result = storage
            .get_response("tenant_a", "session_2", &response_id)
            .await;

        assert!(result.is_ok());
        let retrieved = result.unwrap();
        assert_eq!(retrieved.session_id, "session_1");
        assert_eq!(retrieved.response, response_data);
    }

    #[tokio::test]
    async fn test_delete_response() {
        let storage = InMemoryResponseStorage::new(0);
        let response_data = serde_json::json!({"data": "test"});

        let response_id = storage
            .store_response("tenant_a", "session_1", None, response_data, None)
            .await
            .unwrap();

        storage
            .delete_response("tenant_a", "session_1", &response_id)
            .await
            .unwrap();

        let result = storage
            .get_response("tenant_a", "session_1", &response_id)
            .await;

        assert!(matches!(result, Err(StorageError::NotFound)));
    }

    #[tokio::test]
    async fn test_ttl_expiration() {
        let storage = InMemoryResponseStorage::new(0);
        let response_data = serde_json::json!({"data": "expires soon"});

        // Store with 10-second TTL
        let response_id = storage
            .store_response(
                "tenant_a",
                "session_1",
                None,
                response_data,
                Some(Duration::from_secs(10)),
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
    }

    #[tokio::test]
    async fn test_list_responses() {
        let storage = InMemoryResponseStorage::new(0);

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
            .store_response(
                "tenant_a",
                "session_2",
                None,
                serde_json::json!({"other": 1}),
                None,
            )
            .await
            .unwrap();

        let responses = storage
            .list_responses("tenant_a", "session_1", None, None)
            .await
            .unwrap();

        assert_eq!(responses.len(), 5);
    }

    #[tokio::test]
    async fn test_list_responses_with_limit() {
        let storage = InMemoryResponseStorage::new(0);

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
            .list_responses("tenant_a", "session_1", Some(3), None)
            .await
            .unwrap();

        assert_eq!(responses.len(), 3);
    }

    #[tokio::test]
    async fn test_list_responses_with_cursor() {
        let storage = InMemoryResponseStorage::new(0);

        for i in 1..=10 {
            storage
                .store_response(
                    "tenant_a",
                    "session_1",
                    Some(&format!("resp_{:02}", i)),
                    serde_json::json!({"turn": i}),
                    None,
                )
                .await
                .unwrap();
        }

        let page1 = storage
            .list_responses("tenant_a", "session_1", Some(3), None)
            .await
            .unwrap();

        assert_eq!(page1.len(), 3);
        assert_eq!(page1[0].response_id, "resp_01");
        assert_eq!(page1[2].response_id, "resp_03");

        let page2 = storage
            .list_responses("tenant_a", "session_1", Some(3), Some("resp_03"))
            .await
            .unwrap();

        assert_eq!(page2.len(), 3);
        assert_eq!(page2[0].response_id, "resp_04");
        assert_eq!(page2[2].response_id, "resp_06");
    }
}
