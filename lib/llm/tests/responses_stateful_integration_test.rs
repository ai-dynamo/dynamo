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
    storage::{ResponseStorage, ResponseStorageManager, StorageError},
};
use serde_json::json;
use std::sync::Arc;
use tower::ServiceExt;

/// Shared storage for tests
type SharedStorage = Arc<ResponseStorageManager>;

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
    let storage = Arc::new(ResponseStorageManager::new());
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
    let storage = ResponseStorageManager::new();

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
    let storage = ResponseStorageManager::new();

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
    let storage = Arc::new(ResponseStorageManager::new());
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
    let storage = Arc::new(ResponseStorageManager::new());
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
    let storage = ResponseStorageManager::new();

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
    let storage = ResponseStorageManager::new();

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
    let storage = ResponseStorageManager::new();

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
    let storage = Arc::new(ResponseStorageManager::new());
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
