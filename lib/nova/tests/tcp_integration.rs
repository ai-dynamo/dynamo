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
