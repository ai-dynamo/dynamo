// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test for configure_layouts RPC pattern.
//!
//! This test verifies that the leader must wait_for_handler before sending RPCs.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_nova::am::{Nova, NovaHandler};
use dynamo_nova_backend::{Transport, tcp::TcpTransportBuilder};
use serde::{Deserialize, Serialize};
use std::net::TcpListener;

/// Simple test request/response types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestConfig {
    value: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestResponse {
    success: bool,
}

/// Create a Nova instance with TCP transport on a random port.
async fn create_nova_tcp() -> Result<Arc<Nova>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let transport: Arc<dyn Transport> = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)?
            .build()?,
    );
    let nova = Nova::new(vec![transport], vec![]).await?;
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok(nova)
}

/// Test that demonstrates the correct pattern: wait_for_handler before RPC.
#[tokio::test]
async fn test_rpc_works_with_wait_for_handler() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    // Create leader Nova
    let leader = create_nova_tcp().await.expect("Create leader Nova");
    println!(
        "Leader Nova created: {:?}",
        leader.instance_id().worker_id()
    );

    // Create worker Nova
    let worker = create_nova_tcp().await.expect("Create worker Nova");
    println!(
        "Worker Nova created: {:?}",
        worker.instance_id().worker_id()
    );

    // Register handler on worker (simulates ConnectorWorkerImpl::new)
    let handler =
        NovaHandler::typed_unary::<TestConfig, TestResponse, _>("test.configure", |ctx| {
            println!("Handler received config: value={}", ctx.input.value);
            Ok(TestResponse { success: true })
        })
        .build();
    worker.register_handler(handler).expect("Register handler");
    println!("Handler registered on worker");

    // Register each as peer of the other (bidirectional)
    leader
        .register_peer(worker.peer_info())
        .expect("Register worker peer");
    worker
        .register_peer(leader.peer_info())
        .expect("Register leader peer");
    println!("Peers registered bidirectionally");

    // CRITICAL: Wait for the handler to become available
    println!("Waiting for handler to become available...");
    leader
        .wait_for_handler(worker.instance_id(), "test.configure")
        .await
        .expect("Handler should become available");
    println!("Handler is now available");

    // Send RPC
    println!("Sending RPC...");
    let response: TestResponse = leader
        .typed_unary("test.configure")
        .unwrap()
        .payload(TestConfig { value: 42 })
        .unwrap()
        .instance(worker.instance_id())
        .send()
        .await
        .expect("RPC should succeed");

    assert!(response.success);
    println!("\n=== Test Passed ===");
}

/// Test that demonstrates the race condition without wait_for_handler.
/// This simulates what the Python code currently does.
#[tokio::test]
async fn test_rpc_may_fail_without_wait_for_handler() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    // Create leader Nova
    let leader = create_nova_tcp().await.expect("Create leader Nova");

    // Create worker Nova
    let worker = create_nova_tcp().await.expect("Create worker Nova");

    // Register handler on worker
    let handler =
        NovaHandler::typed_unary::<TestConfig, TestResponse, _>("test.configure", |_ctx| {
            Ok(TestResponse { success: true })
        })
        .build();
    worker.register_handler(handler).expect("Register handler");

    // Register peers (bidirectional, as Python does)
    leader
        .register_peer(worker.peer_info())
        .expect("Register worker peer");
    worker
        .register_peer(leader.peer_info())
        .expect("Register leader peer");

    // NO wait_for_handler - this is what Python does!
    println!("Sending RPC WITHOUT wait_for_handler (may race)...");

    // Try with a short timeout
    let result = tokio::time::timeout(Duration::from_secs(2), async {
        leader
            .typed_unary::<TestResponse>("test.configure")
            .unwrap()
            .payload(TestConfig { value: 42 })
            .unwrap()
            .instance(worker.instance_id())
            .send()
            .await
    })
    .await;

    match result {
        Ok(Ok(response)) => {
            println!("RPC succeeded (lucky timing): {:?}", response.success);
        }
        Ok(Err(e)) => {
            println!("RPC failed with error: {:?}", e);
        }
        Err(_) => {
            println!("RPC timed out (expected race condition)");
        }
    }

    println!("\n=== Test demonstrates timing sensitivity ===");
}
