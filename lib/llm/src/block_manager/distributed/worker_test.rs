// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests and examples for the active message handling system

use crate::block_manager::distributed::active_message::{
    examples, ActiveMessageFactory, ActiveMessageHandlerFactory, MessageHandler,
};
use anyhow::Result;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

#[tokio::test]
async fn test_active_message_flow() -> Result<()> {
    // Create cancellation token
    let cancellation_token = CancellationToken::new();

    // Create active message system
    let (mut receiver, sender) = ActiveMessageFactory::create(2, cancellation_token.clone())?;

    // Create and register handlers
    let handlers = examples::create_example_handlers();
    receiver.register_handlers(handlers)?;

    // Start the receiver
    receiver.start()?;

    // Get response receiver
    let response_receiver = receiver.get_response_receiver();

    // Collect responses for verification
    let responses = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let responses_clone = responses.clone();

    // Spawn response handler
    let response_task = tokio::spawn(async move {
        while let Some(response) = response_receiver.recv().await {
            info!("Received response: {}", response.notification);
            responses_clone.lock().await.push(response);
        }
    });

    // Send test messages
    let messages = vec![
        ("ping", "test data", Some("ping_test")),
        ("data_transfer", "transfer data", Some("transfer_test")),
        ("ping", "", Some("error_test")), // Empty data will cause error
        ("unknown", "unknown", Some("unknown_test")),
    ];

    for (msg_type, data, response_id) in messages {
        if let Some(response_id) = response_id {
            sender.send_with_response(
                msg_type.to_string(),
                data.as_bytes().to_vec(),
                response_id.to_string(),
            )?;
        } else {
            sender.send_fire_and_forget(msg_type.to_string(), data.as_bytes().to_vec())?;
        }
    }

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Stop the receiver
    receiver.stop().await?;

    // Wait for response task
    let _ = tokio::time::timeout(Duration::from_secs(1), response_task).await;

    // Verify responses
    let collected_responses = responses.lock().await;
    assert_eq!(collected_responses.len(), 4);

    // Check specific responses
    let ping_response = collected_responses
        .iter()
        .find(|r| r.notification.starts_with("ping_test:"))
        .expect("Should have ping response");
    assert!(ping_response.is_success);
    assert_eq!(ping_response.notification, "ping_test:ok");

    let transfer_response = collected_responses
        .iter()
        .find(|r| r.notification.starts_with("transfer_test:"))
        .expect("Should have transfer response");
    assert!(transfer_response.is_success);

    let error_response = collected_responses
        .iter()
        .find(|r| r.notification.starts_with("error_test:"))
        .expect("Should have error response");
    assert!(!error_response.is_success);
    assert!(error_response.notification.contains("err("));

    let unknown_response = collected_responses
        .iter()
        .find(|r| r.notification.starts_with("unknown_test:"))
        .expect("Should have unknown response");
    assert!(!unknown_response.is_success);
    assert!(unknown_response.notification.contains("No handler"));

    info!("All tests passed! 🎉");
    Ok(())
}

/// Example of a custom handler that captures resources
#[derive(Clone)]
struct ResourceCapturingHandler {
    name: String,
    shared_state: std::sync::Arc<tokio::sync::Mutex<Vec<String>>>,
}

impl ResourceCapturingHandler {
    fn new(name: String) -> Self {
        Self {
            name,
            shared_state: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
        }
    }

    async fn get_state(&self) -> Vec<String> {
        self.shared_state.lock().await.clone()
    }
}

impl MessageHandler for ResourceCapturingHandler {
    fn handle_message(
        &self,
        data: Vec<u8>,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        let message = String::from_utf8_lossy(&data).to_string();
        let name = self.name.clone();
        let shared_state = self.shared_state.clone();

        async move {
            info!("Handler {} processing: {}", name, message);

            // Simulate capturing and modifying shared state
            let mut state = shared_state.lock().await;
            state.push(format!("{}: {}", name, message));

            // Simulate some async work
            tokio::time::sleep(Duration::from_millis(50)).await;

            Ok(())
        }
    }
}

#[tokio::test]
async fn test_resource_capturing_handler() -> Result<()> {
    let cancellation_token = CancellationToken::new();
    let (mut receiver, sender) = ActiveMessageFactory::create(1, cancellation_token.clone())?;

    // Create a handler that captures resources
    let handler = ResourceCapturingHandler::new("resource_handler".to_string());
    let handler_clone = handler.clone();

    // Register the handler
    receiver.register_handler(
        "resource_test".to_string(),
        ActiveMessageFactory::create_object_handler(handler_clone),
    )?;

    // Start the receiver
    receiver.start()?;

    // Send some messages
    for i in 0..3 {
        sender.send_fire_and_forget(
            "resource_test".to_string(),
            format!("Message {}", i).into_bytes(),
        )?;
    }

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Check the captured state
    let state = handler.get_state().await;
    assert_eq!(state.len(), 3);
    assert!(state[0].contains("Message 0"));
    assert!(state[1].contains("Message 1"));
    assert!(state[2].contains("Message 2"));

    receiver.stop().await?;
    info!("Resource capturing test passed! 🎯");
    Ok(())
}

/// Integration test showing the complete flow with communication layer
#[tokio::test]
async fn test_communication_integration() -> Result<()> {
    let cancellation_token = CancellationToken::new();
    let (mut receiver, sender) = ActiveMessageFactory::create(2, cancellation_token.clone())?;

    // Register handlers
    let handlers = examples::create_example_handlers();
    receiver.register_handlers(handlers)?;

    // Start the receiver
    receiver.start()?;

    // Create communication layer
    let comm_layer = examples::CommunicationLayer::new(sender, receiver.get_response_receiver());

    // Start response handler
    let response_task = comm_layer.start_response_handler();

    // Simulate incoming network messages
    comm_layer
        .handle_incoming_network_message(
            "ping".to_string(),
            b"network ping".to_vec(),
            Some("net_ping_1".to_string()),
        )
        .await?;

    comm_layer
        .handle_incoming_network_message(
            "data_transfer".to_string(),
            b"network data".to_vec(),
            Some("net_transfer_1".to_string()),
        )
        .await?;

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Cleanup
    receiver.stop().await?;
    response_task.abort();

    info!("Communication integration test passed! 🌐");
    Ok(())
}

/// Performance test to verify concurrency handling
#[tokio::test]
async fn test_concurrency_performance() -> Result<()> {
    let cancellation_token = CancellationToken::new();
    let concurrency = 4;
    let (mut receiver, sender) =
        ActiveMessageFactory::create(concurrency, cancellation_token.clone())?;

    // Create a slow handler for testing concurrency
    let slow_handler = ActiveMessageFactory::create_handler(|data: Vec<u8>| async move {
        // Simulate slow processing
        tokio::time::sleep(Duration::from_millis(100)).await;
        if data.is_empty() {
            anyhow::bail!("Empty data");
        }
        Ok(())
    });

    receiver.register_handler("slow".to_string(), slow_handler)?;

    // Start the receiver
    receiver.start()?;

    let response_receiver = receiver.get_response_receiver();

    // Count responses
    let response_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count_clone = response_count.clone();

    let response_task = tokio::spawn(async move {
        while let Some(_response) = response_receiver.recv().await {
            count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });

    // Send many messages concurrently
    let start_time = std::time::Instant::now();
    let num_messages = 10;

    for i in 0..num_messages {
        sender.send_with_response(
            "slow".to_string(),
            format!("Message {}", i).into_bytes(),
            format!("perf_test_{}", i),
        )?;
    }

    // Wait for all responses
    while response_count.load(std::sync::atomic::Ordering::Relaxed) < num_messages {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let elapsed = start_time.elapsed();
    info!("Processed {} messages in {:?}", num_messages, elapsed);

    // With concurrency=4 and 100ms per message, we should finish in ~300ms instead of 1000ms
    assert!(
        elapsed < Duration::from_millis(400),
        "Should benefit from concurrency"
    );

    receiver.stop().await?;
    response_task.abort();

    info!("Concurrency performance test passed! ⚡");
    Ok(())
}
