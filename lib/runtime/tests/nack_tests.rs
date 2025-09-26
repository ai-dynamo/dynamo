// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_runtime::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    handler::{ActiveMessage, ActiveMessageHandler},
    manager::ActiveMessageManager,
    response::ResponseContext,
    zmq::ZmqActiveMessageManager,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
struct ValidationTestHandler {
    received_messages: Arc<Mutex<Vec<String>>>,
}

impl ValidationTestHandler {
    fn new() -> Self {
        Self {
            received_messages: Arc::new(Mutex::new(Vec::new())),
        }
    }

    async fn get_messages(&self) -> Vec<String> {
        self.received_messages.lock().await.clone()
    }
}

#[async_trait]
impl ActiveMessageHandler for ValidationTestHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        _response: ResponseContext,
    ) -> Result<()> {
        let payload = String::from_utf8(message.payload.to_vec())?;
        self.received_messages.lock().await.push(payload);
        Ok(())
    }

    fn name(&self) -> &str {
        "validation_test_handler"
    }

    fn schema(&self) -> Option<&serde_json::Value> {
        // This handler only accepts objects with a "message" field
        static SCHEMA: once_cell::sync::Lazy<serde_json::Value> =
            once_cell::sync::Lazy::new(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    },
                    "required": ["message"]
                })
            });
        Some(&SCHEMA)
    }
}

#[tokio::test]
async fn test_nack_on_payload_validation_failure() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 = ZmqActiveMessageManager::new(
        "ipc:///tmp/test-nack-validation-1.sock".to_string(),
        cancel_token.clone(),
    )
    .await?;
    let manager2 = ZmqActiveMessageManager::new(
        "ipc:///tmp/test-nack-validation-2.sock".to_string(),
        cancel_token.clone(),
    )
    .await?;

    // Register handler with schema validation
    let handler = Arc::new(ValidationTestHandler::new());
    manager2.register_handler(handler.clone(), None).await?;

    let client1 = manager1.zmq_client();
    let client2 = manager2.zmq_client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send valid payload - should get ACK
    let valid_payload = serde_json::json!({"message": "valid message"});

    let _status = client1
        .message("validation_test_handler")
        .payload(valid_payload)?
        .send(client2.instance_id())
        .await?;

    // Should receive ACK (which happens during send confirmation)
    // If we get here without error, the ACK was successful

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify handler was executed
    let messages = handler.get_messages().await;
    assert_eq!(messages.len(), 1);
    assert!(messages[0].contains("valid message"));

    // Send invalid payload - should get NACK
    let invalid_payload = serde_json::json!({"wrong_field": "invalid"});

    // This should fail due to payload validation
    let result = client1
        .message("validation_test_handler")
        .payload(invalid_payload)?
        .send(client2.instance_id())
        .await;

    // Should get an error due to NACK
    match result {
        Err(e) => {
            // NACK received as expected
            let error_msg = format!("{}", e);
            assert!(
                error_msg.contains("Handler acceptance timeout")
                    || error_msg.contains("validation"),
                "Expected validation error or timeout due to NACK, got: {}",
                error_msg
            );
        }
        Ok(_) => {
            panic!("Expected NACK/error but got successful acceptance");
        }
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify handler was NOT executed for invalid payload
    let messages_after_nack = handler.get_messages().await;
    assert_eq!(messages_after_nack.len(), 1); // Still only 1 message from valid payload

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_cohort_broadcast_with_nack() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    // Create leader and two workers
    let leader = ZmqActiveMessageManager::new(
        "ipc:///tmp/test-cohort-nack-leader.sock".to_string(),
        cancel_token.clone(),
    )
    .await?;

    let worker1 = ZmqActiveMessageManager::new(
        "ipc:///tmp/test-cohort-nack-worker1.sock".to_string(),
        cancel_token.clone(),
    )
    .await?;

    let worker2 = ZmqActiveMessageManager::new(
        "ipc:///tmp/test-cohort-nack-worker2.sock".to_string(),
        cancel_token.clone(),
    )
    .await?;

    // Register handlers with different validation requirements
    let handler1 = Arc::new(ValidationTestHandler::new()); // Accepts valid payloads
    worker1.register_handler(handler1.clone(), None).await?;

    let handler2 = Arc::new(ValidationTestHandler::new()); // Also accepts valid payloads
    worker2.register_handler(handler2.clone(), None).await?;

    let leader_client = leader.zmq_client();
    let worker1_client = worker1.zmq_client();
    let worker2_client = worker2.zmq_client();

    // Connect leader to workers
    let worker1_peer = PeerInfo::new(
        worker1_client.instance_id(),
        worker1_client.endpoint().to_string(),
    );
    let worker2_peer = PeerInfo::new(
        worker2_client.instance_id(),
        worker2_client.endpoint().to_string(),
    );

    leader_client.connect_to_peer(worker1_peer).await?;
    leader_client.connect_to_peer(worker2_peer).await?;

    // Connect workers to leader (for potential responses)
    let leader_peer = PeerInfo::new(
        leader_client.instance_id(),
        leader_client.endpoint().to_string(),
    );
    worker1_client.connect_to_peer(leader_peer.clone()).await?;
    worker2_client.connect_to_peer(leader_peer).await?;

    // Create cohort
    let cohort = dynamo_runtime::active_message::zmq::LeaderWorkerCohort::new(
        leader_client.instance_id(),
        vec![worker1_client.instance_id(), worker2_client.instance_id()],
        leader_client.clone(),
    );

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Broadcast valid payload - should get ACKs from both workers
    let valid_payload = serde_json::json!({"message": "broadcast message"});
    let payload_bytes = Bytes::from(serde_json::to_vec(&valid_payload)?);

    cohort
        .broadcast_to_workers_with_acks(
            "validation_test_handler",
            payload_bytes,
            Duration::from_secs(5),
        )
        .await?;

    tokio::time::sleep(Duration::from_millis(200)).await;

    // Verify both handlers received the message
    let messages1 = handler1.get_messages().await;
    let messages2 = handler2.get_messages().await;
    assert_eq!(messages1.len(), 1);
    assert_eq!(messages2.len(), 1);
    assert!(messages1[0].contains("broadcast message"));
    assert!(messages2[0].contains("broadcast message"));

    leader.shutdown().await?;
    worker1.shutdown().await?;
    worker2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}
