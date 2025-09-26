// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_runtime::active_message::{
    client::ActiveMessageClient,
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
struct TestHandler {
    received_messages: Arc<Mutex<Vec<String>>>,
}

impl TestHandler {
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
impl ActiveMessageHandler for TestHandler {
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
        "test_handler"
    }
}

#[tokio::test]
async fn test_basic_message_send_receive() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 = ZmqActiveMessageManager::new(
        "ipc:///tmp/test-am-1.sock".to_string(),
        cancel_token.clone(),
    )
    .await?;
    let manager2 = ZmqActiveMessageManager::new(
        "ipc:///tmp/test-am-2.sock".to_string(),
        cancel_token.clone(),
    )
    .await?;

    let handler = Arc::new(TestHandler::new());
    manager2.register_handler(handler.clone(), None).await?;

    let client1 = manager1.client();
    let client2 = manager2.client();

    let peer2 = dynamo_runtime::active_message::client::PeerInfo::new(
        client2.instance_id(),
        client2.endpoint().to_string(),
    );
    client1.connect_to_peer(peer2).await?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    let test_message = "Hello from manager1";
    client1
        .send_message(
            client2.instance_id(),
            "test_handler",
            Bytes::from(test_message),
        )
        .await?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    let messages = handler.get_messages().await;
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0], test_message);

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_handler_registration_events() -> Result<()> {
    let cancel_token = CancellationToken::new();

    let manager =
        ZmqActiveMessageManager::new("tcp://127.0.0.1:0".to_string(), cancel_token.clone()).await?;

    let mut events_rx = manager.handler_events();

    let handler = Arc::new(TestHandler::new());
    manager.register_handler(handler, None).await?;

    let event = tokio::time::timeout(Duration::from_secs(1), events_rx.recv()).await??;

    match event {
        dynamo_runtime::active_message::handler::HandlerEvent::Registered { name, .. } => {
            assert_eq!(name, "test_handler");
        }
        _ => panic!("Expected Registered event"),
    }

    manager.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}
