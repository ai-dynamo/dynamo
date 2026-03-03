// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Basic messaging integration test: two Velo instances exchange unary ping-pong.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use velo::*;
use velo_backend::tcp::TcpTransportBuilder;

fn new_transport() -> Arc<velo_backend::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_instance_ping_pong() {
    // Create server
    let server = Velo::builder()
        .add_transport(new_transport())
        .build()
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Register ping handler on server — echoes payload back
    let ping_handler = Handler::unary_handler("ping", |ctx| Ok(Some(ctx.payload))).build();
    server.register_handler(ping_handler).unwrap();

    // Create client
    let client = Velo::builder()
        .add_transport(new_transport())
        .build()
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Register peers with each other
    client.register_peer(server.peer_info()).unwrap();
    server.register_peer(client.peer_info()).unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    // Send ping from client to server
    let payload = Bytes::from_static(b"hello");
    let response: Bytes = client
        .unary("ping")
        .unwrap()
        .raw_payload(payload.clone())
        .instance(server.instance_id())
        .send()
        .await
        .unwrap();

    assert_eq!(response, payload, "Server should echo payload back");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn typed_unary_json() {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct AddRequest {
        a: i32,
        b: i32,
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct AddResponse {
        result: i32,
    }

    // Create server
    let server = Velo::builder()
        .add_transport(new_transport())
        .build()
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Register typed add handler (Handler::typed_unary, field is `input`)
    let handler = Handler::typed_unary("add", |ctx: TypedContext<AddRequest>| {
        Ok(AddResponse {
            result: ctx.input.a + ctx.input.b,
        })
    })
    .build();
    server.register_handler(handler).unwrap();

    // Create client
    let client = Velo::builder()
        .add_transport(new_transport())
        .build()
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect
    client.register_peer(server.peer_info()).unwrap();
    server.register_peer(client.peer_info()).unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    // Send typed unary request (payload method serializes, send returns TypedUnaryResult)
    let request = AddRequest { a: 17, b: 25 };
    let response: AddResponse = client
        .typed_unary("add")
        .unwrap()
        .payload(&request)
        .unwrap()
        .instance(server.instance_id())
        .send()
        .await
        .unwrap();

    assert_eq!(response.result, 42);
}
