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
    Extension, Json, Router,
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware,
    routing::post,
};
use dynamo_async_openai::types::ChatCompletionMessageContent;
use dynamo_llm::{
    http::middleware::session::{RequestSession, extract_session_middleware},
    storage::{InMemoryResponseStorage, ResponseStorage, StorageError},
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
    let storage = Arc::new(InMemoryResponseStorage::new(0));
    let router = Router::new()
        .route("/v1/responses", post(mock_responses_handler))
        .layer(middleware::from_fn(extract_session_middleware))
        .with_state(storage.clone());
    (router, storage)
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
    let storage = InMemoryResponseStorage::new(0);

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
async fn test_cross_session_access_within_tenant() {
    let storage = InMemoryResponseStorage::new(0);

    // Store response in session 1
    let response_data = json!({"data": "session_1_data", "turn": 1});
    let response_id = storage
        .store_response("tenant_a", "session_1", None, response_data.clone(), None)
        .await
        .unwrap();

    // Cross-session access within same tenant should SUCCEED
    // (session is metadata, not a security boundary — enables multi-agent workflows)
    let result = storage
        .get_response("tenant_a", "session_2", &response_id)
        .await;

    assert!(result.is_ok());
    let retrieved = result.unwrap();
    assert_eq!(retrieved.session_id, "session_1"); // Original session metadata preserved
    assert_eq!(retrieved.response, response_data);

    // Session 1 should also still retrieve
    let retrieved = storage
        .get_response("tenant_a", "session_1", &response_id)
        .await
        .unwrap();

    assert_eq!(retrieved.session_id, "session_1");
    assert_eq!(retrieved.response, response_data);
}

#[tokio::test]
async fn test_multi_turn_conversation() {
    let storage = Arc::new(InMemoryResponseStorage::new(0));
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

/// Tests missing tenant header AND empty tenant_id (merged from test_empty_tenant_id
/// and test_missing_session_header).
#[tokio::test]
async fn test_missing_tenant_header() {
    // Case 1: Missing x-tenant-id header entirely
    {
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

    // Case 2: Empty x-tenant-id header
    {
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

    // Case 3: Missing x-session-id header
    {
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
}

// ============================================================================
// GET /v1/responses/{id} Tests
// ============================================================================

#[tokio::test]
async fn test_get_response_success() {
    let storage = InMemoryResponseStorage::new(0);

    // Store a response
    let response_data = json!({
        "id": "resp_get_test",
        "model": "test-model",
        "output": [{"type": "message", "content": "Hello!"}]
    });

    let response_id = storage
        .store_response(
            "tenant_get",
            "session_get",
            Some("resp_get_test"),
            response_data.clone(),
            None,
        )
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
    let storage = InMemoryResponseStorage::new(0);

    let result = storage
        .get_response("tenant_notfound", "session_notfound", "nonexistent_id")
        .await;

    assert!(matches!(result, Err(StorageError::NotFound)));
}

// ============================================================================
// DELETE /v1/responses/{id} Tests
// ============================================================================

#[tokio::test]
async fn test_delete_response_success() {
    let storage = InMemoryResponseStorage::new(0);

    // Store a response
    let response_id = storage
        .store_response(
            "tenant_del",
            "session_del",
            None,
            json!({"data": "to_delete"}),
            None,
        )
        .await
        .unwrap();

    // Verify it exists
    let exists = storage
        .get_response("tenant_del", "session_del", &response_id)
        .await;
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

// ============================================================================
// Session Cloning Tests
// ============================================================================

#[tokio::test]
async fn test_fork_session_full() {
    let storage = InMemoryResponseStorage::new(0);

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

// ============================================================================
// previous_response_id Tests
// ============================================================================

#[tokio::test]
async fn test_previous_response_id_retrieval() {
    let storage = InMemoryResponseStorage::new(0);

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
        .store_response(
            "tenant_session",
            "session_1",
            Some("resp_turn1"),
            first_response.clone(),
            None,
        )
        .await
        .unwrap();

    // Verify we can retrieve it (simulating what handler does)
    let retrieved = storage
        .get_response("tenant_session", "session_1", "resp_turn1")
        .await
        .unwrap();

    // Check we can extract output items
    let output_items = retrieved
        .response
        .get("output")
        .and_then(|o| o.as_array())
        .expect("Should have output items");

    assert_eq!(output_items.len(), 1);
    assert_eq!(output_items[0]["role"], "assistant");
}

// ============================================================================
// Streaming Response Storage Tests
// ============================================================================

/// Test that ResponseStreamConverter storage callback works correctly
#[tokio::test]
async fn test_streaming_response_storage_callback() {
    use dynamo_llm::protocols::openai::responses::ResponseParams;
    use dynamo_llm::protocols::openai::responses::stream_converter::ResponseStreamConverter;
    use std::sync::atomic::{AtomicBool, Ordering};

    let storage = Arc::new(InMemoryResponseStorage::new(0));
    let callback_invoked = Arc::new(AtomicBool::new(false));
    let stored_response = Arc::new(Mutex::new(None));

    // Create converter with storage callback
    let callback_invoked_clone = callback_invoked.clone();
    let stored_response_clone = stored_response.clone();
    let storage_clone = storage.clone();
    let tenant_id = "streaming_test_tenant";
    let session_id = "streaming_test_session";

    let mut converter =
        ResponseStreamConverter::new("test-model".to_string(), ResponseParams::default());
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
    use dynamo_async_openai::types::{ChatChoiceStream, ChatCompletionStreamResponseDelta};
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

    #[allow(deprecated)]
    let chunk1 = NvCreateChatCompletionStreamResponse {
        id: "chatcmpl-test".to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: Some(ChatCompletionMessageContent::Text("Hello, ".to_string())),
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
                content: Some(ChatCompletionMessageContent::Text("world!".to_string())),
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
    assert!(
        callback_invoked.load(Ordering::SeqCst),
        "Storage callback should have been invoked"
    );

    // Verify the stored response contains the accumulated text
    let stored = stored_response.lock().await;
    assert!(stored.is_some(), "Response should have been captured");

    let response_json = stored.as_ref().unwrap();
    let output = response_json
        .get("output")
        .and_then(|o| o.as_array())
        .expect("Response should have output array");

    assert!(!output.is_empty(), "Output should not be empty");

    // Find the message output item
    let message_item = output
        .iter()
        .find(|item| item.get("type").and_then(|t| t.as_str()) == Some("message"))
        .expect("Should have a message output item");

    let content = message_item
        .get("content")
        .and_then(|c| c.as_array())
        .expect("Message should have content array");

    let text = content
        .iter()
        .find(|c| c.get("type").and_then(|t| t.as_str()) == Some("output_text"))
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .expect("Should have text content");

    assert_eq!(text, "Hello, world!", "Accumulated text should match");
}

/// Test that streaming without store flag does not invoke callback
#[tokio::test]
async fn test_streaming_without_store_flag_no_callback() {
    use dynamo_llm::protocols::openai::responses::ResponseParams;
    use dynamo_llm::protocols::openai::responses::stream_converter::ResponseStreamConverter;

    // Create converter WITHOUT storage callback
    let mut converter =
        ResponseStreamConverter::new("test-model".to_string(), ResponseParams::default());

    // Emit start events
    let _start_events = converter.emit_start_events();

    // Simulate a chunk
    use dynamo_async_openai::types::{ChatChoiceStream, ChatCompletionStreamResponseDelta};
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

    #[allow(deprecated)]
    let chunk = NvCreateChatCompletionStreamResponse {
        id: "chatcmpl-test".to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: Some(ChatCompletionMessageContent::Text(
                    "Test content".to_string(),
                )),
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
    assert!(
        !end_events.is_empty(),
        "Should emit end events even without storage callback"
    );
}
