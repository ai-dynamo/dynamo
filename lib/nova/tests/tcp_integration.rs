// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Nova active message system with TCP transport

use dynamo_nova::am::{Nova, NovaHandler};
use dynamo_nova_backend::tcp::TcpTransportBuilder;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

/// Helper to create a test listener bound to a random port
///
/// This returns the TcpListener itself, which can be passed to TcpTransportBuilder::from_listener()
/// to avoid the port race condition between binding and starting the transport.
fn create_test_listener() -> std::net::TcpListener {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FooRequest {
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FooResponse {
    echo: String,
    processed: bool,
}

/// Request type for the error test handler
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ErrorRequest {
    should_fail: bool,
    fail_message: String,
}

/// Response type for the error test handler
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ErrorResponse {
    success: bool,
}

/// Tests that handler errors are properly propagated to clients with the actual error message.
///
/// Before the fix, the client would receive "Failed to deserialize response: expected value at line 1 column 1"
/// because the error payload was treated as a success payload and JSON deserialization failed.
/// After the fix, the client receives the actual error message from the handler.
#[tokio::test]
async fn test_handler_error_propagation() {
    // 1. Create nova_a with TCP transport
    let transport_a = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(create_test_listener())
            .unwrap()
            .build()
            .unwrap(),
    );
    let nova_a = Nova::new(vec![transport_a], vec![]).await.unwrap();

    // 2. Create nova_b with TCP transport
    let transport_b = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(create_test_listener())
            .unwrap()
            .build()
            .unwrap(),
    );
    let nova_b = Nova::new(vec![transport_b], vec![]).await.unwrap();

    // Give transports a moment to bind and start accepting connections
    sleep(Duration::from_millis(100)).await;

    // 3. Register peers with each other
    let peer_info_a = nova_a.peer_info();
    let peer_info_b = nova_b.peer_info();

    nova_a.register_peer(peer_info_b).unwrap();
    nova_b.register_peer(peer_info_a).unwrap();

    // 4. Create a typed_unary_async handler on nova_a that can return errors
    let error_handler = NovaHandler::typed_unary_async::<ErrorRequest, ErrorResponse, _, _>(
        "may_fail",
        |ctx| async move {
            if ctx.input.should_fail {
                // Return an error with a specific message
                Err(anyhow::anyhow!("{}", ctx.input.fail_message))
            } else {
                Ok(ErrorResponse { success: true })
            }
        },
    )
    .build();

    nova_a.register_handler(error_handler).unwrap();

    // 5. Wait for the handler to become available
    nova_b
        .wait_for_handler(nova_a.instance_id(), "may_fail")
        .await
        .unwrap();

    // 6. Test successful case first
    let success_request = ErrorRequest {
        should_fail: false,
        fail_message: String::new(),
    };

    let success_response: ErrorResponse = nova_b
        .typed_unary("may_fail")
        .unwrap()
        .payload(success_request)
        .unwrap()
        .instance(nova_a.instance_id())
        .send()
        .await
        .unwrap();

    assert!(
        success_response.success,
        "Success case should return success=true"
    );

    // 7. Test error case - the handler returns an error with a specific message
    let error_message = "No pending state - call register_kv_caches first";
    let error_request = ErrorRequest {
        should_fail: true,
        fail_message: error_message.to_string(),
    };

    let error_result: Result<ErrorResponse, _> = nova_b
        .typed_unary("may_fail")
        .unwrap()
        .payload(error_request)
        .unwrap()
        .instance(nova_a.instance_id())
        .send()
        .await;

    // 8. Verify the error is properly propagated
    assert!(error_result.is_err(), "Error case should return an error");

    let err = error_result.unwrap_err();
    let err_str = err.to_string();

    // The key assertion: we should receive the ACTUAL error message from the handler,
    // NOT "Failed to deserialize response" which would indicate the bug
    assert!(
        err_str.contains(error_message),
        "Error should contain the handler's error message: '{}'. Got: '{}'",
        error_message,
        err_str
    );

    // Make sure we don't have the old bug's error message
    assert!(
        !err_str.contains("Failed to deserialize response"),
        "Error should NOT be a deserialization error. Got: '{}'",
        err_str
    );
}

#[tokio::test]
async fn test_handler_discovery_via_handshake() {
    // 1. Create nova_a with TCP transport
    let transport_a = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(create_test_listener())
            .unwrap()
            .build()
            .unwrap(),
    );
    let nova_a = Nova::new(vec![transport_a], vec![]).await.unwrap();

    // 2. Create nova_b with TCP transport
    let transport_b = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(create_test_listener())
            .unwrap()
            .build()
            .unwrap(),
    );
    let nova_b = Nova::new(vec![transport_b], vec![]).await.unwrap();

    // Give transports a moment to bind and start accepting connections
    sleep(Duration::from_millis(100)).await;

    // 3. Register peers with each other
    let peer_info_a = nova_a.peer_info();
    let peer_info_b = nova_b.peer_info();

    nova_a.register_peer(peer_info_b).unwrap();
    nova_b.register_peer(peer_info_a).unwrap();

    // 4. Create "foo" unary handler on nova_a that returns a response
    let foo_handler = NovaHandler::typed_unary::<FooRequest, FooResponse, _>("foo", |ctx| {
        Ok(FooResponse {
            echo: format!("Echo: {}", ctx.input.message),
            processed: true,
        })
    })
    .build();

    nova_a.register_handler(foo_handler).unwrap();

    // 5. Wait for the "foo" handler to become available on nova_a (from nova_b's perspective)
    nova_b
        .wait_for_handler(nova_a.instance_id(), "foo")
        .await
        .unwrap();

    // 6. Query handlers from nova_b
    let handlers = nova_b
        .available_handlers(nova_a.instance_id())
        .await
        .unwrap();

    // 7. Verify "foo" handler is visible
    assert!(
        handlers.contains(&"foo".to_string()),
        "Handler 'foo' should be discoverable. Found: {:?}",
        handlers
    );

    // 8. Verify system handlers are also present
    assert!(
        handlers.contains(&"_hello".to_string()),
        "System handler '_hello' should be present. Found: {:?}",
        handlers
    );
    assert!(
        handlers.contains(&"_list_handlers".to_string()),
        "System handler '_list_handlers' should be present. Found: {:?}",
        handlers
    );

    // 9. Call the "foo" handler from nova_b and verify the response
    let request = FooRequest {
        message: "Hello from nova_b".to_string(),
    };

    let response: FooResponse = nova_b
        .typed_unary("foo")
        .unwrap()
        .payload(request)
        .unwrap()
        .instance(nova_a.instance_id())
        .send()
        .await
        .unwrap();

    // 10. Verify the response
    assert_eq!(
        response.echo, "Echo: Hello from nova_b",
        "Response echo should match request message"
    );
    assert!(
        response.processed,
        "Response should indicate processing was successful"
    );
}
