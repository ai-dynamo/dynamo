// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for stateful responses API
//!
//! Tests the end-to-end flow of:
//! 1. Session middleware extracting tenant_id and session_id from headers
//! 2. Response storage with store: true flag
//! 3. Tenant and session isolation
//! 4. Multi-turn conversation handling

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware,
    routing::post,
    Extension, Json, Router,
};
use dynamo_llm::{
    http::middleware::session::{extract_session_middleware, RequestSession},
    storage::{
        InMemoryResponseStorage, InMemorySessionLock, LockError, ResponseStorage,
        SessionLock, StorageError, parse_trace_content, parse_trace_file, replay_trace,
    },
};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tower::ServiceExt;

/// Shared storage for tests
type SharedStorage = Arc<InMemoryResponseStorage>;

/// Mock handler that echoes back session info and stores data
async fn mock_responses_handler(
    State(storage): State<SharedStorage>,
    Extension(session): Extension<RequestSession>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    // Extract store flag from request
    let should_store = payload
        .get("store")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Create mock response
    let response = json!({
        "id": format!("resp_{}", uuid::Uuid::new_v4()),
        "object": "response",
        "created": chrono::Utc::now().timestamp(),
        "model": payload.get("model").and_then(|v| v.as_str()).unwrap_or("test-model"),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Mock response from test handler"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    });

    // Store if requested
    if should_store {
        let store_result = storage
            .store_response(
                &session.tenant_id,
                &session.session_id,
                None,
                response.clone(),
                Some(std::time::Duration::from_secs(3600)),
            )
            .await;

        if let Ok(stored_id) = store_result {
            // Add stored_id to response for verification
            let mut response_with_id = response.as_object().unwrap().clone();
            response_with_id.insert("stored_id".to_string(), json!(stored_id));
            return Json(json!(response_with_id));
        }
    }

    Json(response)
}

fn create_test_router() -> (Router, SharedStorage) {
    let storage = Arc::new(InMemoryResponseStorage::new());
    let router = Router::new()
        .route("/v1/responses", post(mock_responses_handler))
        .layer(middleware::from_fn(extract_session_middleware))
        .with_state(storage.clone());
    (router, storage)
}

fn create_test_router_with_storage(storage: SharedStorage) -> Router {
    Router::new()
        .route("/v1/responses", post(mock_responses_handler))
        .layer(middleware::from_fn(extract_session_middleware))
        .with_state(storage)
}

#[tokio::test]
async fn test_session_middleware_extraction() {
    let (app, _storage) = create_test_router();

    let request = Request::builder()
        .uri("/v1/responses")
        .method("POST")
        .header("x-tenant-id", "tenant_test_123")
        .header("x-session-id", "session_test_456")
        .header("x-user-id", "user_test_789")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "store": false
            })
            .to_string(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Verify response structure
    assert!(body_json.get("id").is_some());
    assert_eq!(body_json.get("object").unwrap(), "response");
}

#[tokio::test]
async fn test_response_storage_with_store_true() {
    let (app, _storage) = create_test_router();

    let request = Request::builder()
        .uri("/v1/responses")
        .method("POST")
        .header("x-tenant-id", "tenant_store_test")
        .header("x-session-id", "session_store_test")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Store this"}],
                "store": true
            })
            .to_string(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Verify that stored_id was added (indicating storage occurred)
    assert!(body_json.get("stored_id").is_some());
    let stored_id = body_json.get("stored_id").unwrap().as_str().unwrap();
    assert!(!stored_id.is_empty());
}

#[tokio::test]
async fn test_tenant_isolation() {
    let storage = InMemoryResponseStorage::new();

    // Store response for tenant A
    let response_a = json!({"data": "tenant_a_secret", "value": 42});
    let response_id = storage
        .store_response("tenant_a", "session_1", None, response_a.clone(), None)
        .await
        .unwrap();

    // Attempt to retrieve from tenant B - should fail (NotFound due to key mismatch)
    let result = storage
        .get_response("tenant_b", "session_1", &response_id)
        .await;

    assert!(matches!(result, Err(StorageError::NotFound)));

    // Tenant A should be able to retrieve
    let retrieved = storage
        .get_response("tenant_a", "session_1", &response_id)
        .await
        .unwrap();

    assert_eq!(retrieved.tenant_id, "tenant_a");
    assert_eq!(retrieved.response, response_a);
}

#[tokio::test]
async fn test_session_isolation() {
    let storage = InMemoryResponseStorage::new();

    // Store response in session 1
    let response_data = json!({"data": "session_1_data", "turn": 1});
    let response_id = storage
        .store_response("tenant_a", "session_1", None, response_data.clone(), None)
        .await
        .unwrap();

    // Attempt to retrieve from session 2 (same tenant) - should fail
    let result = storage
        .get_response("tenant_a", "session_2", &response_id)
        .await;

    assert!(matches!(result, Err(StorageError::NotFound)));

    // Session 1 should be able to retrieve
    let retrieved = storage
        .get_response("tenant_a", "session_1", &response_id)
        .await
        .unwrap();

    assert_eq!(retrieved.session_id, "session_1");
    assert_eq!(retrieved.response, response_data);
}

#[tokio::test]
async fn test_multi_turn_conversation() {
    let storage = Arc::new(InMemoryResponseStorage::new());
    let tenant_id = "tenant_conversation";
    let session_id = "session_multi_turn";

    // Simulate a multi-turn conversation
    let turns = vec![
        json!({"role": "user", "content": "What is 2+2?"}),
        json!({"role": "assistant", "content": "2+2 equals 4."}),
        json!({"role": "user", "content": "What about 3+3?"}),
        json!({"role": "assistant", "content": "3+3 equals 6."}),
    ];

    let mut stored_ids = Vec::new();

    // Store each turn
    for (idx, turn) in turns.iter().enumerate() {
        let response_id = storage
            .store_response(
                tenant_id,
                session_id,
                None,
                turn.clone(),
                Some(std::time::Duration::from_secs(3600)),
            )
            .await
            .unwrap();

        stored_ids.push(response_id.clone());

        // Verify immediate retrieval
        let retrieved = storage
            .get_response(tenant_id, session_id, &response_id)
            .await
            .unwrap();

        assert_eq!(retrieved.tenant_id, tenant_id);
        assert_eq!(retrieved.session_id, session_id);
        assert_eq!(retrieved.response, *turn);

        println!("Turn {}: stored with ID {}", idx, response_id);
    }

    // Verify all turns are still accessible
    assert_eq!(stored_ids.len(), turns.len());

    for (idx, response_id) in stored_ids.iter().enumerate() {
        let retrieved = storage
            .get_response(tenant_id, session_id, response_id)
            .await
            .unwrap();

        assert_eq!(retrieved.response, turns[idx]);
    }
}

#[tokio::test]
async fn test_missing_tenant_header() {
    let (app, _storage) = create_test_router();

    let request = Request::builder()
        .uri("/v1/responses")
        .method("POST")
        .header("x-session-id", "session_test")
        // Missing x-tenant-id
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}]
            })
            .to_string(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_missing_session_header() {
    let (app, _storage) = create_test_router();

    let request = Request::builder()
        .uri("/v1/responses")
        .method("POST")
        .header("x-tenant-id", "tenant_test")
        // Missing x-session-id
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}]
            })
            .to_string(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_empty_tenant_id() {
    let (app, _storage) = create_test_router();

    let request = Request::builder()
        .uri("/v1/responses")
        .method("POST")
        .header("x-tenant-id", "") // Empty
        .header("x-session-id", "session_test")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}]
            })
            .to_string(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_concurrent_sessions_same_tenant() {
    let storage = Arc::new(InMemoryResponseStorage::new());
    let tenant_id = "tenant_concurrent";

    // Simulate concurrent requests in different sessions
    let mut handles = vec![];

    for session_num in 0..5 {
        let storage = storage.clone();
        let session_id = format!("session_{}", session_num);

        let handle = tokio::spawn(async move {
            let response_data = json!({
                "session": session_id,
                "data": format!("Data for session {}", session_num)
            });

            let response_id = storage
                .store_response(&tenant_id, &session_id, None, response_data.clone(), None)
                .await
                .unwrap();

            // Verify retrieval
            let retrieved = storage
                .get_response(&tenant_id, &session_id, &response_id)
                .await
                .unwrap();

            assert_eq!(retrieved.session_id, session_id);
            assert_eq!(retrieved.response, response_data);

            (session_id, response_id)
        });

        handles.push(handle);
    }

    // Wait for all to complete
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(results.len(), 5);

    // Verify cross-session isolation - session 0 can't access session 1's data
    let (session_0_id, _) = &results[0];
    let (_, session_1_response_id) = &results[1];

    let cross_access = storage
        .get_response(tenant_id, session_0_id, session_1_response_id)
        .await;

    assert!(matches!(cross_access, Err(StorageError::NotFound)));
}

#[tokio::test]
async fn test_ttl_metadata() {
    let storage = InMemoryResponseStorage::new();

    let response_data = json!({"test": "ttl_data"});
    let ttl = std::time::Duration::from_secs(300); // 5 minutes

    let now_before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let response_id = storage
        .store_response("tenant_ttl", "session_ttl", None, response_data.clone(), Some(ttl))
        .await
        .unwrap();

    let retrieved = storage
        .get_response("tenant_ttl", "session_ttl", &response_id)
        .await
        .unwrap();

    // Verify TTL was set
    assert!(retrieved.expires_at.is_some());
    let expires_at = retrieved.expires_at.unwrap();

    // Should expire approximately 300 seconds from now
    assert!(expires_at >= now_before + 300);
    assert!(expires_at <= now_before + 301); // Allow 1 second tolerance

    // Verify other metadata
    assert_eq!(retrieved.tenant_id, "tenant_ttl");
    assert_eq!(retrieved.session_id, "session_ttl");
    assert!(retrieved.created_at >= now_before);
}

#[tokio::test]
async fn test_response_metadata_structure() {
    let storage = InMemoryResponseStorage::new();

    let response_data = json!({
        "id": "resp_123",
        "model": "test-model",
        "choices": [{"message": {"content": "test"}}]
    });

    let response_id = storage
        .store_response(
            "tenant_meta",
            "session_meta",
            None,
            response_data.clone(),
            Some(std::time::Duration::from_secs(3600)),
        )
        .await
        .unwrap();

    let stored = storage
        .get_response("tenant_meta", "session_meta", &response_id)
        .await
        .unwrap();

    // Verify StoredResponse structure
    assert_eq!(stored.response_id, response_id);
    assert_eq!(stored.tenant_id, "tenant_meta");
    assert_eq!(stored.session_id, "session_meta");
    assert_eq!(stored.response, response_data);
    assert!(stored.created_at > 0);
    assert!(stored.expires_at.is_some());

    // Verify response_id is a valid UUID
    assert!(uuid::Uuid::parse_str(&stored.response_id).is_ok());
}

#[tokio::test]
async fn test_storage_key_pattern() {
    // This test verifies the storage key pattern indirectly by testing isolation
    let storage = InMemoryResponseStorage::new();

    // Store responses with similar IDs but different tenant/session
    let response_id_1 = storage
        .store_response("tenant1", "session1", None, json!({"data": 1}), None)
        .await
        .unwrap();

    let response_id_2 = storage
        .store_response("tenant2", "session2", None, json!({"data": 2}), None)
        .await
        .unwrap();

    // Each should only be accessible with correct tenant/session combo
    let retrieved_1 = storage
        .get_response("tenant1", "session1", &response_id_1)
        .await
        .unwrap();
    assert_eq!(retrieved_1.response["data"], 1);

    let retrieved_2 = storage
        .get_response("tenant2", "session2", &response_id_2)
        .await
        .unwrap();
    assert_eq!(retrieved_2.response["data"], 2);

    // Cross-access should fail
    assert!(storage
        .get_response("tenant1", "session1", &response_id_2)
        .await
        .is_err());

    assert!(storage
        .get_response("tenant2", "session2", &response_id_1)
        .await
        .is_err());
}

#[tokio::test]
async fn test_realistic_multi_turn_with_headers() {
    // Simulate a realistic multi-turn conversation using the full HTTP stack
    let storage = Arc::new(InMemoryResponseStorage::new());
    let app = create_test_router_with_storage(storage.clone());

    let tenant_id = "tenant_realistic";
    let session_id = "session_realistic_001";

    // Turn 1: User asks a question
    let request_1 = Request::builder()
        .uri("/v1/responses")
        .method("POST")
        .header("x-tenant-id", tenant_id)
        .header("x-session-id", session_id)
        .header("x-user-id", "user_alice")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "store": true
            })
            .to_string(),
        ))
        .unwrap();

    let response_1 = app.clone().oneshot(request_1).await.unwrap();
    assert_eq!(response_1.status(), StatusCode::OK);

    let body_1 = axum::body::to_bytes(response_1.into_body(), usize::MAX)
        .await
        .unwrap();
    let json_1: serde_json::Value = serde_json::from_slice(&body_1).unwrap();
    let stored_id_1 = json_1.get("stored_id").unwrap().as_str().unwrap();

    // Turn 2: Follow-up question
    let request_2 = Request::builder()
        .uri("/v1/responses")
        .method("POST")
        .header("x-tenant-id", tenant_id)
        .header("x-session-id", session_id)
        .header("x-user-id", "user_alice")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."},
                    {"role": "user", "content": "What is its population?"}
                ],
                "store": true
            })
            .to_string(),
        ))
        .unwrap();

    let response_2 = app.clone().oneshot(request_2).await.unwrap();
    assert_eq!(response_2.status(), StatusCode::OK);

    let body_2 = axum::body::to_bytes(response_2.into_body(), usize::MAX)
        .await
        .unwrap();
    let json_2: serde_json::Value = serde_json::from_slice(&body_2).unwrap();
    let stored_id_2 = json_2.get("stored_id").unwrap().as_str().unwrap();

    // Verify both responses were stored with different IDs
    assert_ne!(stored_id_1, stored_id_2);

    // Verify both are retrievable from storage (using the shared storage instance)
    let retrieved_1 = storage
        .get_response(tenant_id, session_id, stored_id_1)
        .await
        .unwrap();

    let retrieved_2 = storage
        .get_response(tenant_id, session_id, stored_id_2)
        .await
        .unwrap();

    assert_eq!(retrieved_1.tenant_id, tenant_id);
    assert_eq!(retrieved_1.session_id, session_id);
    assert_eq!(retrieved_2.tenant_id, tenant_id);
    assert_eq!(retrieved_2.session_id, session_id);
}

// ============================================================================
// GET /v1/responses/{id} Tests
// ============================================================================

#[tokio::test]
async fn test_get_response_success() {
    let storage = InMemoryResponseStorage::new();

    // Store a response
    let response_data = json!({
        "id": "resp_get_test",
        "model": "test-model",
        "output": [{"type": "message", "content": "Hello!"}]
    });

    let response_id = storage
        .store_response("tenant_get", "session_get", Some("resp_get_test"), response_data.clone(), None)
        .await
        .unwrap();

    // Retrieve it
    let retrieved = storage
        .get_response("tenant_get", "session_get", &response_id)
        .await
        .unwrap();

    assert_eq!(retrieved.response_id, "resp_get_test");
    assert_eq!(retrieved.response, response_data);
}

#[tokio::test]
async fn test_get_response_not_found() {
    let storage = InMemoryResponseStorage::new();

    let result = storage
        .get_response("tenant_notfound", "session_notfound", "nonexistent_id")
        .await;

    assert!(matches!(result, Err(StorageError::NotFound)));
}

#[tokio::test]
async fn test_get_response_wrong_tenant() {
    let storage = InMemoryResponseStorage::new();

    // Store for tenant_a
    let response_id = storage
        .store_response("tenant_a", "session_1", None, json!({"secret": "data"}), None)
        .await
        .unwrap();

    // Try to get with tenant_b - should fail
    let result = storage
        .get_response("tenant_b", "session_1", &response_id)
        .await;

    assert!(matches!(result, Err(StorageError::NotFound)));
}

// ============================================================================
// DELETE /v1/responses/{id} Tests
// ============================================================================

#[tokio::test]
async fn test_delete_response_success() {
    let storage = InMemoryResponseStorage::new();

    // Store a response
    let response_id = storage
        .store_response("tenant_del", "session_del", None, json!({"data": "to_delete"}), None)
        .await
        .unwrap();

    // Verify it exists
    let exists = storage.get_response("tenant_del", "session_del", &response_id).await;
    assert!(exists.is_ok());

    // Delete it
    let delete_result = storage
        .delete_response("tenant_del", "session_del", &response_id)
        .await;
    assert!(delete_result.is_ok());

    // Verify it's gone
    let after_delete = storage
        .get_response("tenant_del", "session_del", &response_id)
        .await;
    assert!(matches!(after_delete, Err(StorageError::NotFound)));
}

#[tokio::test]
async fn test_delete_response_not_found() {
    let storage = InMemoryResponseStorage::new();

    let result = storage
        .delete_response("tenant_del", "session_del", "nonexistent_id")
        .await;

    assert!(matches!(result, Err(StorageError::NotFound)));
}

#[tokio::test]
async fn test_delete_response_wrong_session() {
    let storage = InMemoryResponseStorage::new();

    // Store for session_1
    let response_id = storage
        .store_response("tenant_a", "session_1", None, json!({"data": "protected"}), None)
        .await
        .unwrap();

    // Try to delete from session_2 - should fail
    let result = storage
        .delete_response("tenant_a", "session_2", &response_id)
        .await;

    assert!(matches!(result, Err(StorageError::NotFound)));

    // Original should still exist
    let still_exists = storage.get_response("tenant_a", "session_1", &response_id).await;
    assert!(still_exists.is_ok());
}

// ============================================================================
// Session Cloning Tests
// ============================================================================

#[tokio::test]
async fn test_fork_session_full() {
    let storage = InMemoryResponseStorage::new();

    // Create original session with 3 responses
    for i in 1..=3 {
        storage
            .store_response(
                "tenant_clone",
                "original_session",
                Some(&format!("resp_{}", i)),
                json!({"turn": i, "content": format!("Message {}", i)}),
                None,
            )
            .await
            .unwrap();
    }

    // Clone to new session
    let cloned_count = storage
        .fork_session("tenant_clone", "original_session", "cloned_session", None)
        .await
        .unwrap();

    assert_eq!(cloned_count, 3);

    // Verify cloned session has all responses
    let cloned_responses = storage
        .list_responses("tenant_clone", "cloned_session", None, None)
        .await
        .unwrap();

    assert_eq!(cloned_responses.len(), 3);
}

#[tokio::test]
async fn test_fork_session_rewind() {
    let storage = InMemoryResponseStorage::new();

    // Create original session with 5 responses
    for i in 1..=5 {
        storage
            .store_response(
                "tenant_rewind",
                "original_session",
                Some(&format!("resp_{}", i)),
                json!({"turn": i}),
                None,
            )
            .await
            .unwrap();
    }

    // Clone only up to resp_3 (rewind point)
    let cloned_count = storage
        .fork_session("tenant_rewind", "original_session", "rewound_session", Some("resp_3"))
        .await
        .unwrap();

    assert_eq!(cloned_count, 3);

    // Verify rewound session only has 3 responses
    let rewound_responses = storage
        .list_responses("tenant_rewind", "rewound_session", None, None)
        .await
        .unwrap();

    assert_eq!(rewound_responses.len(), 3);
}

// ============================================================================
// previous_response_id Tests
// ============================================================================

#[tokio::test]
async fn test_previous_response_id_retrieval() {
    let storage = InMemoryResponseStorage::new();

    // Simulate a first turn - store a response with output
    let first_response = json!({
        "id": "resp_turn1",
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "The answer is 4."}]
            }
        ]
    });

    storage
        .store_response("tenant_session", "session_1", Some("resp_turn1"), first_response.clone(), None)
        .await
        .unwrap();

    // Verify we can retrieve it (simulating what handler does)
    let retrieved = storage
        .get_response("tenant_session", "session_1", "resp_turn1")
        .await
        .unwrap();

    // Check we can extract output items
    let output_items = retrieved.response.get("output")
        .and_then(|o| o.as_array())
        .expect("Should have output items");

    assert_eq!(output_items.len(), 1);
    assert_eq!(output_items[0]["role"], "assistant");
}

#[tokio::test]
async fn test_multi_turn_with_previous_response_id() {
    let storage = InMemoryResponseStorage::new();

    // Turn 1: User asks a question
    let turn1_response = json!({
        "id": "resp_001",
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "2+2 equals 4."}]
            }
        ]
    });

    storage
        .store_response("tenant_math", "calc_session", Some("resp_001"), turn1_response, None)
        .await
        .unwrap();

    // Turn 2: Follow-up question (would use previous_response_id: "resp_001")
    let turn2_response = json!({
        "id": "resp_002",
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "4 times 3 equals 12."}]
            }
        ]
    });

    storage
        .store_response("tenant_math", "calc_session", Some("resp_002"), turn2_response, None)
        .await
        .unwrap();

    // Verify conversation history is accessible
    let responses = storage
        .list_responses("tenant_math", "calc_session", None, None)
        .await
        .unwrap();

    assert_eq!(responses.len(), 2);
    assert_eq!(responses[0].response_id, "resp_001");
    assert_eq!(responses[1].response_id, "resp_002");

    // Verify we can chain by fetching previous response
    let prev = storage
        .get_response("tenant_math", "calc_session", "resp_001")
        .await
        .unwrap();

    let prev_output = prev.response.get("output")
        .and_then(|o| o.as_array())
        .unwrap();

    assert!(prev_output[0]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("4"));
}

// ============================================================================
// Trace-Based Tests
// ============================================================================

/// Test that simulates loading a real trace and verifying session flow
#[tokio::test]
async fn test_trace_based_session_flow() {
    let storage = InMemoryResponseStorage::new();

    // Simulate the trace from ../agentic-traces/4de2bcf7-793e-494f-a8a6-ef8556bd352f.jsonl
    let session_id = "4de2bcf7-793e-494f-a8a6-ef8556bd352f";
    let tenant_id = "trace_test_tenant";

    // Turn 1: User asks "how many lines in the README.md?"
    // Model responds with tool call, then final answer
    let turn1_response = json!({
        "id": "resp_turn1_trace",
        "model": "claude-opus-4-5-20251101",
        "created_at": 1770087123,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "The README.md has **225 lines**."}]
            }
        ]
    });

    storage
        .store_response(tenant_id, session_id, Some("resp_turn1_trace"), turn1_response.clone(), None)
        .await
        .unwrap();

    // Verify we can retrieve turn 1
    let retrieved_turn1 = storage
        .get_response(tenant_id, session_id, "resp_turn1_trace")
        .await
        .unwrap();

    assert_eq!(retrieved_turn1.session_id, session_id);

    // Turn 2: User follows up (using previous_response_id)
    // This would reference resp_turn1_trace
    let turn2_response = json!({
        "id": "resp_turn2_trace",
        "model": "claude-opus-4-5-20251101",
        "created_at": 1770087200,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "That's 225 lines in the README file."}]
            }
        ]
    });

    storage
        .store_response(tenant_id, session_id, Some("resp_turn2_trace"), turn2_response, None)
        .await
        .unwrap();

    // Verify full session history
    let all_responses = storage
        .list_responses(tenant_id, session_id, None, None)
        .await
        .unwrap();

    assert_eq!(all_responses.len(), 2);

    // Verify chronological order
    assert_eq!(all_responses[0].response_id, "resp_turn1_trace");
    assert_eq!(all_responses[1].response_id, "resp_turn2_trace");

    // Verify we can chain sessions (what previous_response_id enables)
    let prev_context = storage
        .get_response(tenant_id, session_id, "resp_turn1_trace")
        .await
        .unwrap();

    let prev_output = prev_context.response.get("output")
        .and_then(|o| o.as_array())
        .unwrap();

    // The previous output contains the context for the next turn
    assert!(!prev_output.is_empty());
}

/// Test loading actual trace file format
#[tokio::test]
async fn test_braintrust_trace_format_parsing() {
    // Verify we can parse the Braintrust trace format
    let trace_line = r#"{"id": "4de2bcf7-793e-494f-a8a6-ef8556bd352f", "span_id": "4de2bcf7-793e-494f-a8a6-ef8556bd352f", "metadata": {"session_id": "4de2bcf7-793e-494f-a8a6-ef8556bd352f"}, "input": "Session: claude-trace"}"#;

    let parsed: serde_json::Value = serde_json::from_str(trace_line).unwrap();

    assert_eq!(parsed["id"], "4de2bcf7-793e-494f-a8a6-ef8556bd352f");
    assert_eq!(parsed["metadata"]["session_id"], "4de2bcf7-793e-494f-a8a6-ef8556bd352f");

    // This session_id is what we'd use for our storage
    let session_id = parsed["metadata"]["session_id"].as_str().unwrap();
    assert!(!session_id.is_empty());
}

// ============================================================================
// Trace Replay Tests (using trace_replay module)
// ============================================================================

/// Test parsing and replaying the actual trace file format
#[tokio::test]
async fn test_trace_replay_parse_real_trace() {
    // This is the actual trace data from /Users/mkosec/work/agentic-traces/
    let trace_content = r#"{"id": "4de2bcf7-793e-494f-a8a6-ef8556bd352f", "span_id": "4de2bcf7-793e-494f-a8a6-ef8556bd352f", "root_span_id": "4de2bcf7-793e-494f-a8a6-ef8556bd352f", "created": "2026-02-03T02:52:03.000Z", "input": "Session: claude-trace", "metadata": {"session_id": "4de2bcf7-793e-494f-a8a6-ef8556bd352f", "workspace": "/Users/idhanani/Desktop/agents/claude-trace", "hostname": "idhanani-mlt", "username": "idhanani", "os": "Darwin", "source": "claude-code"}, "span_attributes": {"name": "Claude Code: claude-trace", "type": "task"}, "_written_at": "2026-02-03T02:52:03.000Z", "_project": "claude-code"}
{"id": "8554433b-2545-450c-808b-79cc3c91164e", "span_id": "8554433b-2545-450c-808b-79cc3c91164e", "root_span_id": "4de2bcf7-793e-494f-a8a6-ef8556bd352f", "span_parents": ["4de2bcf7-793e-494f-a8a6-ef8556bd352f"], "created": "2026-02-03T02:52:03.000Z", "input": "how many lines in the README.md?", "metrics": {"start": 1770087123}, "span_attributes": {"name": "Turn 1", "type": "task"}, "_written_at": "2026-02-03T02:52:03.000Z", "_project": "claude-code"}
{"id": "82c33a10-2d22-4816-9edb-0b30f0853046", "span_id": "82c33a10-2d22-4816-9edb-0b30f0853046", "root_span_id": "4de2bcf7-793e-494f-a8a6-ef8556bd352f", "span_parents": ["8554433b-2545-450c-808b-79cc3c91164e"], "created": "2026-02-03T02:52:07.000Z", "input": {"command": "wc -l /Users/idhanani/Desktop/agents/claude-trace/README.md", "description": "Count lines in README.md"}, "output": {"stdout": "     225 /Users/idhanani/Desktop/agents/claude-trace/README.md", "stderr": "", "interrupted": false, "isImage": false}, "metrics": {"start": 1770087127, "end": 1770087127, "duration_ms": 497, "success": "true"}, "metadata": {"tool_name": "Bash"}, "span_attributes": {"name": "Terminal: wc -l /Users/idhanani/Desktop/agents/claude-trace/", "type": "tool"}, "_written_at": "2026-02-03T02:52:07.000Z", "_project": "claude-code", "otel": {"duration_ms": 497, "success": "true"}}
{"id": "3aaca646-c3b7-44c8-abce-396c78d8b99d", "span_id": "3aaca646-c3b7-44c8-abce-396c78d8b99d", "root_span_id": "4de2bcf7-793e-494f-a8a6-ef8556bd352f", "span_parents": ["8554433b-2545-450c-808b-79cc3c91164e"], "created": "2026-02-03T02:52:09.352Z", "input": [{"role": "user", "content": "how many lines in the README.md?"}], "output": {"role": "assistant", "content": "The README.md has **225 lines**."}, "metrics": {"start": 1770087129, "end": 1770087129, "prompt_tokens": 1, "completion_tokens": 6, "tokens": 7, "duration_ms": 2293, "cost_usd": 0.010486}, "metadata": {"model": "claude-opus-4-5-20251101"}, "span_attributes": {"name": "claude-opus-4-5-20251101", "type": "llm"}, "_written_at": "2026-02-03T02:52:09.000Z", "_project": "claude-code"}"#;

    // Parse the trace
    let parsed_trace = parse_trace_content(trace_content).expect("Should parse trace");

    // Verify session extraction
    assert_eq!(parsed_trace.session_id, "4de2bcf7-793e-494f-a8a6-ef8556bd352f");
    assert_eq!(parsed_trace.raw_spans.len(), 4); // root + turn + tool + llm

    // Verify we found the turn
    assert_eq!(parsed_trace.turns.len(), 1);
    assert_eq!(parsed_trace.turns[0].user_input, "how many lines in the README.md?");
    assert_eq!(parsed_trace.turns[0].assistant_output, Some("The README.md has **225 lines**.".to_string()));

    // Verify tool calls were captured
    assert_eq!(parsed_trace.turns[0].tool_calls.len(), 1);
    assert_eq!(parsed_trace.turns[0].tool_calls[0].tool_name, "Bash");
}

/// Test replaying a trace through the storage system
#[tokio::test]
async fn test_trace_replay_through_storage() {
    let storage = InMemoryResponseStorage::new();

    // Parse trace content
    let trace_content = r#"{"id": "replay-test-root", "span_id": "replay-test-root", "root_span_id": "replay-test-root", "created": "2026-02-03T02:52:03.000Z", "input": "Session: replay-test", "metadata": {"session_id": "replay-test-session"}, "span_attributes": {"name": "Test Session", "type": "task"}}
{"id": "turn-1", "span_id": "turn-1", "root_span_id": "replay-test-root", "span_parents": ["replay-test-root"], "created": "2026-02-03T02:52:04.000Z", "input": "What is 2+2?", "span_attributes": {"name": "Turn 1", "type": "task"}}
{"id": "llm-1", "span_id": "llm-1", "root_span_id": "replay-test-root", "span_parents": ["turn-1"], "created": "2026-02-03T02:52:05.000Z", "input": [{"role": "user", "content": "What is 2+2?"}], "output": {"role": "assistant", "content": "2+2 equals 4."}, "metadata": {"model": "test"}, "span_attributes": {"name": "llm", "type": "llm"}}
{"id": "turn-2", "span_id": "turn-2", "root_span_id": "replay-test-root", "span_parents": ["replay-test-root"], "created": "2026-02-03T02:52:06.000Z", "input": "And 3+3?", "span_attributes": {"name": "Turn 2", "type": "task"}}
{"id": "llm-2", "span_id": "llm-2", "root_span_id": "replay-test-root", "span_parents": ["turn-2"], "created": "2026-02-03T02:52:07.000Z", "input": [{"role": "user", "content": "And 3+3?"}], "output": {"role": "assistant", "content": "3+3 equals 6."}, "metadata": {"model": "test"}, "span_attributes": {"name": "llm", "type": "llm"}}"#;

    let parsed_trace = parse_trace_content(trace_content).expect("Should parse");

    // Verify parsing
    assert_eq!(parsed_trace.turns.len(), 2);
    assert_eq!(parsed_trace.turns[0].user_input, "What is 2+2?");
    assert_eq!(parsed_trace.turns[1].user_input, "And 3+3?");

    // Replay through storage
    let replay_result = replay_trace(&storage, &parsed_trace, "tenant_replay")
        .await
        .expect("Replay should succeed");

    assert_eq!(replay_result.turns_replayed, 2);
    assert_eq!(replay_result.session_id, "replay-test-session");
    assert_eq!(replay_result.response_ids.len(), 2);

    // Verify responses are stored and retrievable
    let responses = storage
        .list_responses("tenant_replay", "replay-test-session", None, None)
        .await
        .unwrap();

    assert_eq!(responses.len(), 2);

    // Verify the content was stored correctly
    let first_response = &responses[0];
    assert!(first_response.response["metadata"]["user_input"]
        .as_str()
        .unwrap()
        .contains("2+2"));

    // Verify we can chain via previous_response_id
    let second_response = &responses[1];
    let prev_id = second_response.response["metadata"]["previous_response_id"]
        .as_str()
        .unwrap();
    assert!(!prev_id.is_empty());
}

/// Test that trace replay maintains tenant isolation with independent data
#[tokio::test]
async fn test_trace_replay_tenant_isolation() {
    let storage = InMemoryResponseStorage::new();

    let trace_content = r#"{"id": "iso-root", "span_id": "iso-root", "root_span_id": "iso-root", "created": "2026-02-03T02:52:03.000Z", "input": "Session", "metadata": {"session_id": "isolated-session"}, "span_attributes": {"name": "Test", "type": "task"}}
{"id": "iso-turn", "span_id": "iso-turn", "root_span_id": "iso-root", "span_parents": ["iso-root"], "created": "2026-02-03T02:52:04.000Z", "input": "Hello", "span_attributes": {"name": "Turn 1", "type": "task"}}
{"id": "iso-llm", "span_id": "iso-llm", "root_span_id": "iso-root", "span_parents": ["iso-turn"], "created": "2026-02-03T02:52:05.000Z", "output": {"role": "assistant", "content": "Hi!"}, "span_attributes": {"name": "llm", "type": "llm"}}"#;

    let parsed_trace = parse_trace_content(trace_content).expect("Should parse");

    // Replay for tenant A
    let result_a = replay_trace(&storage, &parsed_trace, "tenant_A")
        .await
        .expect("Replay A should succeed");

    // Replay for tenant B
    let result_b = replay_trace(&storage, &parsed_trace, "tenant_B")
        .await
        .expect("Replay B should succeed");

    // Both should have replayed 1 turn
    assert_eq!(result_a.turns_replayed, 1);
    assert_eq!(result_b.turns_replayed, 1);

    // Each tenant has their own isolated copy
    let tenant_a_responses = storage
        .list_responses("tenant_A", "isolated-session", None, None)
        .await
        .unwrap();
    let tenant_b_responses = storage
        .list_responses("tenant_B", "isolated-session", None, None)
        .await
        .unwrap();

    assert_eq!(tenant_a_responses.len(), 1);
    assert_eq!(tenant_b_responses.len(), 1);

    // Verify the data is stored with correct tenant IDs (proving isolation)
    assert_eq!(tenant_a_responses[0].tenant_id, "tenant_A");
    assert_eq!(tenant_b_responses[0].tenant_id, "tenant_B");

    // Delete tenant A's response - tenant B's should still exist
    storage
        .delete_response("tenant_A", "isolated-session", &result_a.response_ids[0])
        .await
        .unwrap();

    // Tenant A's is gone
    let tenant_a_after = storage
        .list_responses("tenant_A", "isolated-session", None, None)
        .await
        .unwrap();
    assert_eq!(tenant_a_after.len(), 0);

    // Tenant B's still exists (proving isolation)
    let tenant_b_after = storage
        .list_responses("tenant_B", "isolated-session", None, None)
        .await
        .unwrap();
    assert_eq!(tenant_b_after.len(), 1);
}

/// Test loading and replaying a real trace file from disk
/// This test validates compatibility with actual Claude Code trace output
#[tokio::test]
async fn test_trace_replay_from_real_file() {
    use std::path::PathBuf;

    // Look for trace files in the user's traces directory
    let traces_dir = std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".claude/traces"))
        .filter(|p| p.exists());

    let Some(traces_dir) = traces_dir else {
        println!("Skipping: No ~/.claude/traces directory found");
        return;
    };

    // Find the most recent trace file
    let trace_file = std::fs::read_dir(&traces_dir)
        .ok()
        .and_then(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map(|ext| ext == "jsonl").unwrap_or(false))
                .filter(|e| e.file_name().to_string_lossy() != "claude-code.jsonl") // Skip aggregated
                .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
                .map(|e| e.path())
        });

    let Some(trace_path) = trace_file else {
        println!("Skipping: No .jsonl trace files found in {:?}", traces_dir);
        return;
    };

    println!("Testing with trace file: {:?}", trace_path);

    // Parse the trace file
    let trace = parse_trace_file(&trace_path).expect("Should parse real trace file");

    println!("  Session ID: {}", trace.session_id);
    println!("  Total spans: {}", trace.raw_spans.len());
    println!("  Turns found: {}", trace.turns.len());

    // Verify we found at least one turn
    assert!(!trace.turns.is_empty(), "Real trace should have at least one turn");

    // Verify session_id was extracted
    assert!(!trace.session_id.is_empty(), "Session ID should be extracted");

    // Replay through storage
    let storage = InMemoryResponseStorage::new();
    let result = replay_trace(&storage, &trace, "real_trace_tenant")
        .await
        .expect("Replay should succeed");

    // Verify replay results
    assert_eq!(result.turns_replayed, trace.turns.len());
    assert_eq!(result.response_ids.len(), trace.turns.len());

    // Verify responses are stored and retrievable
    let stored_responses = storage
        .list_responses("real_trace_tenant", &trace.session_id, None, None)
        .await
        .unwrap();

    assert_eq!(stored_responses.len(), trace.turns.len());

    // Verify chaining - each response after the first should have previous_response_id
    // that points to another response in the set
    let response_ids: std::collections::HashSet<_> = stored_responses
        .iter()
        .map(|r| r.response_id.as_str())
        .collect();

    let mut has_first = false; // One response should have no previous (the first turn)
    let mut has_chain = false; // At least one response should chain to another

    for response in &stored_responses {
        match response.response["metadata"]["previous_response_id"].as_str() {
            None | Some("") => {
                // This is turn 1 - no previous
                has_first = true;
            }
            Some(prev_id) if prev_id == "null" => {
                has_first = true;
            }
            Some(prev_id) => {
                // Verify the previous_response_id points to a valid response
                assert!(
                    response_ids.contains(prev_id),
                    "previous_response_id '{}' should point to a stored response",
                    prev_id
                );
                has_chain = true;
            }
        }
    }

    assert!(has_first, "Should have at least one response with no previous (turn 1)");
    if stored_responses.len() > 1 {
        assert!(has_chain, "Multi-turn trace should have chained responses");
    }

    println!("✓ Real trace replay successful: {} turns", result.turns_replayed);
}

// ============================================================================
// Session Lock Tests
// ============================================================================

/// Test basic session lock acquire and release
#[tokio::test]
async fn test_session_lock_basic() {
    let lock = InMemorySessionLock::new();

    // Acquire lock
    let guard = lock.acquire("test:session", Duration::from_secs(1)).await.unwrap();
    assert_eq!(guard.key(), "test:session");

    // Should be locked
    assert!(lock.is_locked("test:session").await);

    // Drop releases the lock
    drop(guard);

    // Should be unlocked
    assert!(!lock.is_locked("test:session").await);
}

/// Test that try_acquire fails when lock is held
#[tokio::test]
async fn test_session_lock_contention() {
    let lock = InMemorySessionLock::new();

    // Acquire lock
    let _guard = lock.acquire("contention:test", Duration::from_secs(1)).await.unwrap();

    // Try acquire should fail
    let result = lock.try_acquire("contention:test").await;
    assert!(matches!(result, Err(LockError::AlreadyHeld)));
}

/// Test lock timeout behavior
#[tokio::test]
async fn test_session_lock_timeout() {
    let lock = InMemorySessionLock::new();

    // Acquire lock
    let _guard = lock.acquire("timeout:test", Duration::from_secs(10)).await.unwrap();

    // Try to acquire with short timeout - should timeout
    let result = lock.acquire("timeout:test", Duration::from_millis(50)).await;
    assert!(matches!(result, Err(LockError::Timeout(_))));
}

/// Test that different sessions can be locked independently
#[tokio::test]
async fn test_session_lock_independence() {
    let lock = InMemorySessionLock::new();

    // Lock session 1
    let _guard1 = lock.acquire("tenant:session1", Duration::from_secs(1)).await.unwrap();

    // Session 2 should still be lockable
    let guard2 = lock.acquire("tenant:session2", Duration::from_secs(1)).await;
    assert!(guard2.is_ok());

    // Verify both are locked
    assert!(lock.is_locked("tenant:session1").await);
    assert!(lock.is_locked("tenant:session2").await);
}

/// Test concurrent lock acquisition serializes access
#[tokio::test]
async fn test_session_lock_serialization() {
    let lock = Arc::new(InMemorySessionLock::new());
    let counter = Arc::new(Mutex::new(0u64));
    let key = "serialize:test";

    let mut handles = vec![];

    // Spawn 10 concurrent tasks all trying to increment the counter
    for _ in 0..10 {
        let lock = lock.clone();
        let counter = counter.clone();
        let key = key.to_string();

        handles.push(tokio::spawn(async move {
            // Acquire lock before modifying counter
            let _guard = lock.acquire(&key, Duration::from_secs(5)).await.unwrap();

            // Critical section - read, sleep, increment
            let current = *counter.lock().await;
            tokio::time::sleep(Duration::from_millis(5)).await;
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

// ============================================================================
// Concurrent Storage Access with Locking Tests
// ============================================================================

/// Test concurrent operations on same session with locking
#[tokio::test]
async fn test_concurrent_storage_with_lock() {
    let storage = Arc::new(InMemoryResponseStorage::new());
    let lock = Arc::new(InMemorySessionLock::new());
    let tenant_id = "tenant_concurrent_lock";
    let session_id = "session_concurrent_lock";

    let mut handles = vec![];
    let stored_ids = Arc::new(Mutex::new(Vec::new()));

    // Spawn concurrent tasks that store responses
    for i in 0..10 {
        let storage = storage.clone();
        let lock = lock.clone();
        let stored_ids = stored_ids.clone();
        let lock_key = format!("{}:{}", tenant_id, session_id);

        handles.push(tokio::spawn(async move {
            // Acquire session lock
            let _guard = lock.acquire(&lock_key, Duration::from_secs(5)).await.unwrap();

            // Store response while holding lock
            let response_data = json!({
                "turn": i,
                "message": format!("Response from task {}", i)
            });

            let response_id = storage
                .store_response(tenant_id, session_id, None, response_data, None)
                .await
                .unwrap();

            stored_ids.lock().await.push(response_id);
        }));
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all 10 responses were stored
    let responses = storage
        .list_responses(tenant_id, session_id, None, None)
        .await
        .unwrap();

    assert_eq!(responses.len(), 10);
    assert_eq!(stored_ids.lock().await.len(), 10);
}

/// Test that previous_response_id chaining works under concurrent access
#[tokio::test]
async fn test_previous_response_id_concurrent_chaining() {
    let storage = Arc::new(InMemoryResponseStorage::new());
    let lock = Arc::new(InMemorySessionLock::new());
    let tenant_id = "tenant_chain";
    let session_id = "session_chain";
    let lock_key = format!("{}:{}", tenant_id, session_id);

    // Store initial response
    let _initial_id = storage
        .store_response(
            tenant_id,
            session_id,
            Some("resp_initial"),
            json!({"turn": 0, "content": "Initial"}),
            None,
        )
        .await
        .unwrap();

    let last_response_id = Arc::new(Mutex::new("resp_initial".to_string()));

    let mut handles = vec![];

    // Spawn tasks that chain responses
    for i in 1..=5 {
        let storage = storage.clone();
        let lock = lock.clone();
        let last_response_id = last_response_id.clone();
        let lock_key = lock_key.clone();

        handles.push(tokio::spawn(async move {
            // Acquire lock to prevent race on previous_response_id
            let _guard = lock.acquire(&lock_key, Duration::from_secs(5)).await.unwrap();

            // Read current last response ID
            let prev_id = last_response_id.lock().await.clone();

            // Store new response with reference to previous
            let response_data = json!({
                "turn": i,
                "previous_response_id": prev_id,
                "content": format!("Turn {}", i)
            });

            let new_id = storage
                .store_response(tenant_id, session_id, None, response_data, None)
                .await
                .unwrap();

            // Update last response ID
            *last_response_id.lock().await = new_id.clone();

            new_id
        }));
    }

    // Wait for all tasks
    let mut response_ids = vec![];
    for handle in handles {
        response_ids.push(handle.await.unwrap());
    }

    // Verify chain integrity
    let responses = storage
        .list_responses(tenant_id, session_id, None, None)
        .await
        .unwrap();

    assert_eq!(responses.len(), 6); // initial + 5 chained

    // Verify chained responses (not the initial one) have previous_response_id
    let chained_responses: Vec<_> = responses
        .iter()
        .filter(|r| r.response_id != "resp_initial")
        .collect();

    assert_eq!(chained_responses.len(), 5);
    for resp in chained_responses {
        assert!(
            resp.response.get("previous_response_id").is_some(),
            "Response {} should have previous_response_id",
            resp.response_id
        );
    }
}

// ============================================================================
// Load Tests
// ============================================================================

/// Load test: 1000 parallel sessions storing responses
#[tokio::test]
async fn test_load_1000_parallel_sessions() {
    let storage = Arc::new(InMemoryResponseStorage::new());
    let tenant_id = "tenant_load_test";

    let mut handles = vec![];

    // Spawn 1000 concurrent tasks, each with their own session
    for i in 0..1000 {
        let storage = storage.clone();
        let session_id = format!("load_session_{}", i);

        handles.push(tokio::spawn(async move {
            // Each session stores 5 responses
            let mut response_ids = vec![];

            for turn in 0..5 {
                let response_data = json!({
                    "session": &session_id,
                    "turn": turn,
                    "data": format!("Session {} Turn {}", i, turn)
                });

                let response_id = storage
                    .store_response(tenant_id, &session_id, None, response_data, None)
                    .await
                    .unwrap();

                response_ids.push(response_id);
            }

            // Verify all responses are retrievable
            let responses = storage
                .list_responses(tenant_id, &session_id, None, None)
                .await
                .unwrap();

            assert_eq!(responses.len(), 5);

            (session_id, response_ids)
        }));
    }

    // Wait for all tasks to complete
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // Verify all sessions completed
    assert_eq!(results.len(), 1000);

    // Verify total storage (5000 responses across 1000 sessions)
    let mut total_responses = 0;
    for (session_id, _) in results.iter().take(10) {
        // Spot check 10 sessions
        let responses = storage
            .list_responses(tenant_id, session_id, None, None)
            .await
            .unwrap();
        total_responses += responses.len();
    }

    assert_eq!(total_responses, 50); // 10 sessions * 5 responses each
}

/// Load test: Concurrent reads and writes to same session
#[tokio::test]
async fn test_load_concurrent_read_write_same_session() {
    let storage = Arc::new(InMemoryResponseStorage::new());
    let lock = Arc::new(InMemorySessionLock::new());
    let tenant_id = "tenant_rw_load";
    let session_id = "session_rw_load";
    let lock_key = format!("{}:{}", tenant_id, session_id);

    // Pre-populate with some responses
    for i in 0..10 {
        storage
            .store_response(
                tenant_id,
                session_id,
                Some(&format!("initial_{}", i)),
                json!({"initial": i}),
                None,
            )
            .await
            .unwrap();
    }

    let mut handles = vec![];

    // Spawn 50 readers
    for _ in 0..50 {
        let storage = storage.clone();
        handles.push(tokio::spawn(async move {
            let responses = storage
                .list_responses(tenant_id, session_id, None, None)
                .await
                .unwrap();
            responses.len()
        }));
    }

    // Spawn 50 writers (with locking)
    for i in 0..50 {
        let storage = storage.clone();
        let lock = lock.clone();
        let lock_key = lock_key.clone();

        handles.push(tokio::spawn(async move {
            let _guard = lock.acquire(&lock_key, Duration::from_secs(10)).await.unwrap();
            storage
                .store_response(
                    tenant_id,
                    session_id,
                    None,
                    json!({"concurrent_write": i}),
                    None,
                )
                .await
                .unwrap();
            1usize
        }));
    }

    // Wait for all tasks
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // All tasks should complete without error
    assert_eq!(results.len(), 100);

    // Final count should be 10 initial + 50 concurrent writes = 60
    let final_responses = storage
        .list_responses(tenant_id, session_id, None, None)
        .await
        .unwrap();

    assert_eq!(final_responses.len(), 60);
}

/// Load test: Many tenants, many sessions
#[tokio::test]
async fn test_load_many_tenants_many_sessions() {
    let storage = Arc::new(InMemoryResponseStorage::new());

    let mut handles = vec![];

    // 10 tenants, 100 sessions each, 5 responses per session
    for tenant_num in 0..10 {
        let storage = storage.clone();
        let tenant_id = format!("tenant_{}", tenant_num);

        handles.push(tokio::spawn(async move {
            for session_num in 0..100 {
                let session_id = format!("session_{}", session_num);

                for turn in 0..5 {
                    storage
                        .store_response(
                            &tenant_id,
                            &session_id,
                            None,
                            json!({"tenant": tenant_num, "session": session_num, "turn": turn}),
                            None,
                        )
                        .await
                        .unwrap();
                }
            }
            tenant_num
        }));
    }

    // Wait for all tenants to complete
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(results.len(), 10);

    // Verify isolation: tenant_0's data
    let tenant_0_session_0 = storage
        .list_responses("tenant_0", "session_0", None, None)
        .await
        .unwrap();

    assert_eq!(tenant_0_session_0.len(), 5);

    // Cross-tenant access should fail
    let cross_access = storage
        .get_response("tenant_1", "session_0", &tenant_0_session_0[0].response_id)
        .await;

    assert!(matches!(cross_access, Err(StorageError::NotFound)));
}

// ============================================================================
// Streaming Response Storage Tests
// ============================================================================

/// Test that ResponseStreamConverter storage callback works correctly
#[tokio::test]
async fn test_streaming_response_storage_callback() {
    use dynamo_llm::protocols::openai::responses::stream_converter::ResponseStreamConverter;
    use std::sync::atomic::{AtomicBool, Ordering};

    let storage = Arc::new(InMemoryResponseStorage::new());
    let callback_invoked = Arc::new(AtomicBool::new(false));
    let stored_response = Arc::new(Mutex::new(None));

    // Create converter with storage callback
    let callback_invoked_clone = callback_invoked.clone();
    let stored_response_clone = stored_response.clone();
    let storage_clone = storage.clone();
    let tenant_id = "streaming_test_tenant";
    let session_id = "streaming_test_session";

    let mut converter = ResponseStreamConverter::new("test-model".to_string());
    let response_id = converter.response_id().to_string();

    converter = converter.with_storage_callback(move |response_json| {
        callback_invoked_clone.store(true, Ordering::SeqCst);
        let storage = storage_clone.clone();
        let response_json_clone = response_json.clone();
        let response_id = response_id.clone();

        // Store the response
        tokio::spawn(async move {
            let _ = storage
                .store_response(
                    tenant_id,
                    session_id,
                    Some(&response_id),
                    response_json_clone,
                    Some(Duration::from_secs(3600)),
                )
                .await;
        });

        // Also capture for test verification
        let stored_response = stored_response_clone.clone();
        tokio::spawn(async move {
            *stored_response.lock().await = Some(response_json);
        });
    });

    // Simulate streaming: emit start events
    let _start_events = converter.emit_start_events();

    // Simulate some text content chunks
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
    use dynamo_async_openai::types::{
        ChatCompletionStreamResponseDelta, ChatChoiceStream,
    };

    #[allow(deprecated)]
    let chunk1 = NvCreateChatCompletionStreamResponse {
        id: "chatcmpl-test".to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: Some("Hello, ".to_string()),
                function_call: None,
                tool_calls: None,
                role: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            stop_reason: None,
            logprobs: None,
        }],
        created: 1726000000,
        model: "test-model".to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
        nvext: None,
    };

    #[allow(deprecated)]
    let chunk2 = NvCreateChatCompletionStreamResponse {
        id: "chatcmpl-test".to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: Some("world!".to_string()),
                function_call: None,
                tool_calls: None,
                role: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            stop_reason: None,
            logprobs: None,
        }],
        created: 1726000000,
        model: "test-model".to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
        nvext: None,
    };

    // Process chunks
    let _events1 = converter.process_chunk(&chunk1);
    let _events2 = converter.process_chunk(&chunk2);

    // Emit end events (this should invoke the storage callback)
    let _end_events = converter.emit_end_events();

    // Give the spawned tasks time to complete
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify callback was invoked
    assert!(callback_invoked.load(Ordering::SeqCst), "Storage callback should have been invoked");

    // Verify the stored response contains the accumulated text
    let stored = stored_response.lock().await;
    assert!(stored.is_some(), "Response should have been captured");

    let response_json = stored.as_ref().unwrap();
    let output = response_json.get("output")
        .and_then(|o| o.as_array())
        .expect("Response should have output array");

    assert!(!output.is_empty(), "Output should not be empty");

    // Find the message output item
    let message_item = output.iter()
        .find(|item| item.get("type").and_then(|t| t.as_str()) == Some("message"))
        .expect("Should have a message output item");

    let content = message_item.get("content")
        .and_then(|c| c.as_array())
        .expect("Message should have content array");

    let text = content.iter()
        .find(|c| c.get("type").and_then(|t| t.as_str()) == Some("output_text"))
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .expect("Should have text content");

    assert_eq!(text, "Hello, world!", "Accumulated text should match");
}

/// Test streaming storage with function calls
#[tokio::test]
async fn test_streaming_response_storage_with_function_calls() {
    use dynamo_llm::protocols::openai::responses::stream_converter::ResponseStreamConverter;
    use std::sync::atomic::{AtomicBool, Ordering};

    let callback_invoked = Arc::new(AtomicBool::new(false));
    let stored_response = Arc::new(Mutex::new(None));

    let callback_invoked_clone = callback_invoked.clone();
    let stored_response_clone = stored_response.clone();

    let mut converter = ResponseStreamConverter::new("test-model".to_string());
    converter = converter.with_storage_callback(move |response_json| {
        callback_invoked_clone.store(true, Ordering::SeqCst);
        let stored_response = stored_response_clone.clone();
        tokio::spawn(async move {
            *stored_response.lock().await = Some(response_json);
        });
    });

    // Emit start events
    let _start_events = converter.emit_start_events();

    // Simulate function call chunks
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
    use dynamo_async_openai::types::{
        ChatCompletionStreamResponseDelta, ChatChoiceStream,
        ChatCompletionMessageToolCallChunk, FunctionCallStream,
    };

    #[allow(deprecated)]
    let fc_chunk = NvCreateChatCompletionStreamResponse {
        id: "chatcmpl-test".to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: None,
                function_call: None,
                tool_calls: Some(vec![ChatCompletionMessageToolCallChunk {
                    index: 0,
                    id: Some("call_abc123".to_string()),
                    r#type: Some(dynamo_async_openai::types::ChatCompletionToolType::Function),
                    function: Some(FunctionCallStream {
                        name: Some("get_weather".to_string()),
                        arguments: Some(r#"{"location":"SF"}"#.to_string()),
                    }),
                }]),
                role: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            stop_reason: None,
            logprobs: None,
        }],
        created: 1726000000,
        model: "test-model".to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
        nvext: None,
    };

    // Process chunk
    let _events = converter.process_chunk(&fc_chunk);

    // Emit end events
    let _end_events = converter.emit_end_events();

    // Give the spawned task time to complete
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify callback was invoked
    assert!(callback_invoked.load(Ordering::SeqCst), "Storage callback should have been invoked");

    // Verify the stored response contains the function call
    let stored = stored_response.lock().await;
    assert!(stored.is_some(), "Response should have been captured");

    let response_json = stored.as_ref().unwrap();
    let output = response_json.get("output")
        .and_then(|o| o.as_array())
        .expect("Response should have output array");

    // Find the function call output item
    let fc_item = output.iter()
        .find(|item| item.get("type").and_then(|t| t.as_str()) == Some("function_call"))
        .expect("Should have a function_call output item");

    assert_eq!(fc_item.get("name").and_then(|n| n.as_str()), Some("get_weather"));
    assert_eq!(fc_item.get("call_id").and_then(|c| c.as_str()), Some("call_abc123"));
}

/// Test that streaming without store flag does not invoke callback
#[tokio::test]
async fn test_streaming_without_store_flag_no_callback() {
    use dynamo_llm::protocols::openai::responses::stream_converter::ResponseStreamConverter;

    // Create converter WITHOUT storage callback
    let mut converter = ResponseStreamConverter::new("test-model".to_string());

    // Emit start events
    let _start_events = converter.emit_start_events();

    // Simulate a chunk
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
    use dynamo_async_openai::types::{
        ChatCompletionStreamResponseDelta, ChatChoiceStream,
    };

    #[allow(deprecated)]
    let chunk = NvCreateChatCompletionStreamResponse {
        id: "chatcmpl-test".to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: Some("Test content".to_string()),
                function_call: None,
                tool_calls: None,
                role: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            stop_reason: None,
            logprobs: None,
        }],
        created: 1726000000,
        model: "test-model".to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
        nvext: None,
    };

    let _events = converter.process_chunk(&chunk);

    // Emit end events - should complete without error even without callback
    let end_events = converter.emit_end_events();

    // Should still emit proper end events
    assert!(!end_events.is_empty(), "Should emit end events even without storage callback");
}

// ============================================================================
// Redis Storage Tests (requires redis-storage feature and running Redis)
// ============================================================================

#[cfg(feature = "redis-storage")]
mod redis_tests {
    use super::*;
    use dynamo_llm::storage::{RedisResponseStorage, RedisSessionLock};

    /// Helper to check if Redis is available
    async fn redis_available() -> bool {
        match RedisResponseStorage::new("redis://localhost:6379").await {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    /// Test basic Redis storage operations
    #[tokio::test]
    #[ignore = "Requires Redis server running locally. Run with: cargo test --features redis-storage -- --ignored"]
    async fn test_redis_storage_basic_operations() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let storage = RedisResponseStorage::new("redis://localhost:6379")
            .await
            .unwrap();

        let tenant_id = "redis_test_tenant";
        let session_id = "redis_test_session";

        // Store
        let response_data = json!({"message": "Hello from Redis!", "value": 42});
        let response_id = storage
            .store_response(
                tenant_id,
                session_id,
                None,
                response_data.clone(),
                Some(Duration::from_secs(60)),
            )
            .await
            .unwrap();

        // Get
        let retrieved = storage
            .get_response(tenant_id, session_id, &response_id)
            .await
            .unwrap();

        assert_eq!(retrieved.tenant_id, tenant_id);
        assert_eq!(retrieved.session_id, session_id);
        assert_eq!(retrieved.response, response_data);
        assert!(retrieved.expires_at.is_some());

        // Delete
        storage
            .delete_response(tenant_id, session_id, &response_id)
            .await
            .unwrap();

        // Verify deleted
        let result = storage
            .get_response(tenant_id, session_id, &response_id)
            .await;
        assert!(matches!(result, Err(StorageError::NotFound)));
    }

    /// Test Redis tenant isolation
    #[tokio::test]
    #[ignore = "Requires Redis server running locally. Run with: cargo test --features redis-storage -- --ignored"]
    async fn test_redis_storage_tenant_isolation() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let storage = RedisResponseStorage::new("redis://localhost:6379")
            .await
            .unwrap();

        let response_data = json!({"secret": "tenant_a_only"});

        // Store for tenant A
        let response_id = storage
            .store_response("tenant_a_redis", "session_1", None, response_data.clone(), None)
            .await
            .unwrap();

        // Tenant A can access
        let result_a = storage
            .get_response("tenant_a_redis", "session_1", &response_id)
            .await;
        assert!(result_a.is_ok());

        // Tenant B cannot access
        let result_b = storage
            .get_response("tenant_b_redis", "session_1", &response_id)
            .await;
        assert!(matches!(result_b, Err(StorageError::NotFound)));

        // Cleanup
        storage
            .delete_response("tenant_a_redis", "session_1", &response_id)
            .await
            .unwrap();
    }

    /// Test Redis list_responses
    #[tokio::test]
    #[ignore = "Requires Redis server running locally. Run with: cargo test --features redis-storage -- --ignored"]
    async fn test_redis_storage_list_responses() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let storage = RedisResponseStorage::new("redis://localhost:6379")
            .await
            .unwrap();

        let tenant_id = "redis_list_tenant";
        let session_id = "redis_list_session";

        // Store multiple responses
        let mut ids = Vec::new();
        for i in 1..=5 {
            let id = storage
                .store_response(
                    tenant_id,
                    session_id,
                    None,
                    json!({"turn": i, "data": format!("Response {}", i)}),
                    None,
                )
                .await
                .unwrap();
            ids.push(id);
            // Small delay to ensure ordering by created_at
            tokio::time::sleep(Duration::from_millis(10)).await;
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

    /// Test Redis session cloning
    #[tokio::test]
    #[ignore = "Requires Redis server running locally. Run with: cargo test --features redis-storage -- --ignored"]
    async fn test_redis_storage_fork_session() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let storage = RedisResponseStorage::new("redis://localhost:6379")
            .await
            .unwrap();

        let tenant_id = "redis_clone_tenant";

        // Create source session
        let mut source_ids = Vec::new();
        for i in 1..=3 {
            let id = storage
                .store_response(
                    tenant_id,
                    "source_session",
                    Some(&format!("resp_{}", i)),
                    json!({"turn": i}),
                    None,
                )
                .await
                .unwrap();
            source_ids.push(id);
        }

        // Clone session
        let cloned = storage
            .fork_session(tenant_id, "source_session", "cloned_session", None)
            .await
            .unwrap();
        assert_eq!(cloned, 3);

        // Verify cloned session
        let cloned_responses = storage
            .list_responses(tenant_id, "cloned_session", None, None)
            .await
            .unwrap();
        assert_eq!(cloned_responses.len(), 3);

        // Cleanup
        for id in &source_ids {
            let _ = storage.delete_response(tenant_id, "source_session", id).await;
        }
        for resp in &cloned_responses {
            let _ = storage.delete_response(tenant_id, "cloned_session", &resp.response_id).await;
        }
    }

    /// Test Redis distributed lock basic operations
    #[tokio::test]
    #[ignore = "Requires Redis server running locally. Run with: cargo test --features redis-storage -- --ignored"]
    async fn test_redis_lock_basic() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let lock = RedisSessionLock::new("redis://localhost:6379")
            .await
            .unwrap();

        // Acquire
        let guard = lock
            .acquire("redis_test:lock", Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(guard.key(), "redis_test:lock");

        // Should be locked
        assert!(lock.is_locked("redis_test:lock").await);

        // Release
        drop(guard);

        // Wait for async release
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should be unlocked
        assert!(!lock.is_locked("redis_test:lock").await);
    }

    /// Test Redis lock contention
    #[tokio::test]
    #[ignore = "Requires Redis server running locally. Run with: cargo test --features redis-storage -- --ignored"]
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
            .acquire("redis_contention:lock", Duration::from_secs(5))
            .await
            .unwrap();

        // Try to acquire again - should fail
        let result = lock.try_acquire("redis_contention:lock").await;
        assert!(matches!(result, Err(LockError::AlreadyHeld)));
    }

    /// Test Redis lock timeout
    #[tokio::test]
    #[ignore = "Requires Redis server running locally. Run with: cargo test --features redis-storage -- --ignored"]
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
            .acquire("redis_timeout:lock", Duration::from_secs(10))
            .await
            .unwrap();

        // Try to acquire with short timeout - should timeout
        let result = lock
            .acquire("redis_timeout:lock", Duration::from_millis(100))
            .await;
        assert!(matches!(result, Err(LockError::Timeout(_))));
    }

    /// Test Redis distributed locking with multiple connections (simulating multiple instances)
    #[tokio::test]
    #[ignore = "Requires Redis server running locally. Run with: cargo test --features redis-storage -- --ignored"]
    async fn test_redis_lock_distributed_multi_instance() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        // Create multiple lock instances (simulating multiple app instances)
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

        let key = "redis_distributed:test";
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
                let _guard = lock.acquire(&key, Duration::from_secs(30)).await.unwrap();

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

    /// Test Redis storage with locking for concurrent session access
    #[tokio::test]
    #[ignore = "Requires Redis server running locally. Run with: cargo test --features redis-storage -- --ignored"]
    async fn test_redis_storage_with_locking() {
        if !redis_available().await {
            println!("Redis not available, skipping test");
            return;
        }

        let storage = Arc::new(
            RedisResponseStorage::new("redis://localhost:6379")
                .await
                .unwrap(),
        );
        let lock = Arc::new(
            RedisSessionLock::new("redis://localhost:6379")
                .await
                .unwrap(),
        );

        let tenant_id = "redis_locked_tenant";
        let session_id = "redis_locked_session";
        let lock_key = format!("{}:{}", tenant_id, session_id);

        let stored_ids = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut handles = vec![];

        // Spawn concurrent tasks
        for i in 0..5 {
            let storage = storage.clone();
            let lock = lock.clone();
            let stored_ids = stored_ids.clone();
            let lock_key = lock_key.clone();

            handles.push(tokio::spawn(async move {
                let _guard = lock.acquire(&lock_key, Duration::from_secs(10)).await.unwrap();

                let response_data = json!({
                    "turn": i,
                    "message": format!("Concurrent response {}", i)
                });

                let response_id = storage
                    .store_response(tenant_id, session_id, None, response_data, None)
                    .await
                    .unwrap();

                stored_ids.lock().await.push(response_id);
            }));
        }

        // Wait for all
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all stored
        let responses = storage
            .list_responses(tenant_id, session_id, None, None)
            .await
            .unwrap();
        assert_eq!(responses.len(), 5);

        // Cleanup
        for id in stored_ids.lock().await.iter() {
            storage
                .delete_response(tenant_id, session_id, id)
                .await
                .unwrap();
        }
    }
}
