// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for stateful responses API
//!
//! Tests the end-to-end flow of:
//! 1. Response storage with store: true flag
//! 2. Tenant and session isolation
//! 3. Multi-turn conversation handling via previous_response_id

use dynamo_llm::storage::{InMemoryResponseStorage, ResponseStorage, StorageError};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn test_storage_basic_crud() {
    let storage = InMemoryResponseStorage::new(0);
    let response_data =
        json!({"id": "resp_123", "output": [{"type": "message", "content": [{"text": "Hello"}]}]});

    // Store
    let id = storage
        .store_response(
            "tenant_a",
            "session_1",
            Some("resp_123"),
            response_data.clone(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(id, "resp_123");

    // Retrieve
    let stored = storage
        .get_response("tenant_a", "session_1", "resp_123")
        .await
        .unwrap();
    assert_eq!(stored.response, response_data);

    // Delete
    storage
        .delete_response("tenant_a", "session_1", "resp_123")
        .await
        .unwrap();

    let result = storage
        .get_response("tenant_a", "session_1", "resp_123")
        .await;
    assert!(matches!(result, Err(StorageError::NotFound)));
}

#[tokio::test]
async fn test_tenant_isolation() {
    let storage = InMemoryResponseStorage::new(0);
    let data = json!({"secret": "tenant_a_only"});

    storage
        .store_response("tenant_a", "session_1", Some("resp_secret"), data, None)
        .await
        .unwrap();

    // Tenant B cannot access tenant A's response
    let result = storage
        .get_response("tenant_b", "session_1", "resp_secret")
        .await;
    assert!(matches!(result, Err(StorageError::NotFound)));

    // Tenant B cannot delete tenant A's response
    let result = storage
        .delete_response("tenant_b", "session_1", "resp_secret")
        .await;
    assert!(matches!(result, Err(StorageError::NotFound)));
}

#[tokio::test]
async fn test_cross_session_access_within_tenant() {
    let storage = InMemoryResponseStorage::new(0);
    let data = json!({"data": "from_session_1"});

    storage
        .store_response(
            "tenant_a",
            "session_1",
            Some("resp_cross"),
            data.clone(),
            None,
        )
        .await
        .unwrap();

    // Same tenant, different session CAN access (session is metadata, not boundary)
    let result = storage
        .get_response("tenant_a", "session_2", "resp_cross")
        .await;
    assert!(result.is_ok());
    let stored = result.unwrap();
    assert_eq!(stored.session_id, "session_1"); // Metadata preserved
}

#[tokio::test]
async fn test_multi_turn_conversation_storage() {
    let storage = InMemoryResponseStorage::new(0);

    // Turn 1: user asks, gets response
    let turn1 = json!({
        "id": "resp_turn1",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I am an AI assistant."}]
        }]
    });
    storage
        .store_response("tenant_a", "session_1", Some("resp_turn1"), turn1, None)
        .await
        .unwrap();

    // Turn 2: user references turn 1
    let turn2 = json!({
        "id": "resp_turn2",
        "previous_response_id": "resp_turn1",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I remember our conversation."}]
        }]
    });
    storage
        .store_response("tenant_a", "session_1", Some("resp_turn2"), turn2, None)
        .await
        .unwrap();

    // Can retrieve both turns
    let t1 = storage
        .get_response("tenant_a", "session_1", "resp_turn1")
        .await
        .unwrap();
    let t2 = storage
        .get_response("tenant_a", "session_1", "resp_turn2")
        .await
        .unwrap();
    assert_eq!(t1.response["id"], "resp_turn1");
    assert_eq!(t2.response["id"], "resp_turn2");
    assert_eq!(t2.response["previous_response_id"], "resp_turn1");
}

#[tokio::test]
async fn test_ttl_sets_expiration() {
    let storage = InMemoryResponseStorage::new(0);

    let id = storage
        .store_response(
            "tenant_a",
            "session_1",
            None,
            json!({"data": "ttl_test"}),
            Some(Duration::from_secs(3600)),
        )
        .await
        .unwrap();

    let stored = storage
        .get_response("tenant_a", "session_1", &id)
        .await
        .unwrap();
    assert!(stored.expires_at.is_some());
}

#[tokio::test]
async fn test_session_limit_enforcement() {
    let storage = InMemoryResponseStorage::new(3);

    for i in 0..3 {
        storage
            .store_response(
                "tenant_a",
                "session_1",
                Some(&format!("resp_{i}")),
                json!({"turn": i}),
                None,
            )
            .await
            .unwrap();
    }

    // Fourth should fail
    let result = storage
        .store_response("tenant_a", "session_1", None, json!({"turn": 3}), None)
        .await;
    assert!(matches!(result, Err(StorageError::SessionFull)));

    // Different session can still store
    let result = storage
        .store_response("tenant_a", "session_2", None, json!({"turn": 0}), None)
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_list_responses_pagination() {
    let storage = InMemoryResponseStorage::new(0);

    for i in 1..=10 {
        storage
            .store_response(
                "tenant_a",
                "session_1",
                Some(&format!("resp_{:02}", i)),
                json!({"turn": i}),
                None,
            )
            .await
            .unwrap();
    }

    // First page
    let page1 = storage
        .list_responses("tenant_a", "session_1", Some(3), None)
        .await
        .unwrap();
    assert_eq!(page1.len(), 3);
    assert_eq!(page1[0].response_id, "resp_01");

    // Second page
    let page2 = storage
        .list_responses("tenant_a", "session_1", Some(3), Some("resp_03"))
        .await
        .unwrap();
    assert_eq!(page2.len(), 3);
    assert_eq!(page2[0].response_id, "resp_04");
}

#[tokio::test]
async fn test_session_middleware_header_validation() {
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode},
        middleware,
        response::IntoResponse,
        routing::get,
    };
    use dynamo_llm::http::middleware::session::{RequestSession, extract_session_middleware};
    use tower::ServiceExt;

    async fn handler(axum::Extension(ctx): axum::Extension<RequestSession>) -> impl IntoResponse {
        format!("tenant={}, session={}", ctx.tenant_id, ctx.session_id)
    }

    let app = Router::new()
        .route("/test", get(handler))
        .layer(middleware::from_fn(extract_session_middleware));

    // Valid headers
    let req = Request::builder()
        .uri("/test")
        .header("x-tenant-id", "my-org")
        .header("x-session-id", "conv-123")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Missing tenant
    let req = Request::builder().uri("/test").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    // Invalid characters in tenant
    let req = Request::builder()
        .uri("/test")
        .header("x-tenant-id", "bad/chars")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}
