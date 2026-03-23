// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for NATS JetStream audit sink
//!
//! These tests verify audit records are published to NATS JetStream.
//!
//! **Manual Testing Only** (not run in CI - requires network connectivity)
//!
//! Test Requirements:
//! - NATS server with JetStream enabled on localhost:4222
//! - etcd server on localhost:2379
//!
//! Recommended setup:
//! ```bash
//! cd deploy && docker compose up -d nats-server etcd-server
//! ```
//!
//! Run tests:
//! ```bash
//! cargo test --test audit_nats_integration -- --ignored --nocapture
//! ```

#[cfg(test)]
mod tests {
    use dynamo_llm::audit::{bus, handle, sink};
    use dynamo_llm::protocols::openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
    };
    use dynamo_runtime::transports::nats;
    use futures::StreamExt;
    use serde_json::Value;
    use std::sync::Arc;
    use std::time::Duration;
    use temp_env::async_with_vars;
    use tokio::time;
    use uuid::Uuid;

    /// Helper to create a test NATS client
    async fn create_test_nats_client() -> nats::Client {
        nats::ClientOptions::builder()
            .server("nats://localhost:4222")
            .build()
            .expect("Failed to build NATS client options")
            .connect()
            .await
            .expect("Failed to connect to NATS server")
    }

    /// Helper to create a minimal test request
    fn create_test_request(model: &str, store: bool) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "test message"}],
            "store": store
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    /// Helper to create a minimal test response
    fn create_test_response(model: &str, content: &str) -> NvCreateChatCompletionResponse {
        let json = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }]
        });
        serde_json::from_value(json).expect("Failed to create test response")
    }

    /// Helper to setup a NATS stream for testing
    async fn setup_test_stream(client: &nats::Client, stream_name: &str, subject: &str) {
        let js = client.jetstream();
        let _ = js.delete_stream(stream_name).await;

        let config = async_nats::jetstream::stream::Config {
            name: stream_name.to_string(),
            subjects: vec![subject.to_string()],
            max_age: Duration::from_secs(3600),
            ..Default::default()
        };

        js.get_or_create_stream(config)
            .await
            .expect("Failed to create test stream");
    }

    /// Helper to consume messages from a NATS stream
    async fn consume_messages(
        client: &nats::Client,
        stream_name: &str,
        consumer_name: &str,
        max_messages: usize,
        timeout: Duration,
    ) -> Vec<Value> {
        let js = client.jetstream();
        let stream = js
            .get_stream(stream_name)
            .await
            .expect("Failed to get stream");

        let consumer_config = async_nats::jetstream::consumer::pull::Config {
            durable_name: Some(consumer_name.to_string()),
            deliver_policy: async_nats::jetstream::consumer::DeliverPolicy::All,
            ack_policy: async_nats::jetstream::consumer::AckPolicy::Explicit,
            ..Default::default()
        };

        let consumer = stream
            .create_consumer(consumer_config)
            .await
            .expect("Failed to create consumer");

        let mut messages = Vec::new();
        let mut batch = consumer
            .fetch()
            .max_messages(max_messages)
            .expires(timeout)
            .messages()
            .await
            .expect("Failed to fetch messages");

        while let Some(Ok(msg)) = batch.next().await {
            let json: Value =
                serde_json::from_slice(&msg.payload).expect("Failed to parse message as JSON");
            messages.push(json);
            msg.ack().await.expect("Failed to ack message");
        }

        messages
    }

    #[tokio::test]
    #[ignore] // Manual testing only - requires NATS on localhost:4222
    async fn test_audit_nats_basic_flow() {
        const TEST_SUBJECT: &str = "test.audit.basic";
        // Core test: audit records are published to NATS with correct structure
        async_with_vars(
            [
                ("DYN_AUDIT_SINKS", Some("nats")),
                ("DYN_AUDIT_NATS_SUBJECT", Some(TEST_SUBJECT)),
            ],
            async {
                let stream_name = format!("test_basic_{}", Uuid::new_v4());

                let client = create_test_nats_client().await;
                setup_test_stream(&client, &stream_name, TEST_SUBJECT).await;

                bus::init(100);
                sink::spawn_workers_from_env().await.unwrap();
                time::sleep(Duration::from_millis(100)).await;

                // Emit audit record
                let request = create_test_request("nemotron", true);
                let mut handle = handle::create_handle(&request, "test-req-1")
                    .expect("Failed to create audit handle");
                handle.set_request(Arc::new(request.clone()));
                handle.set_response(Arc::new(create_test_response("nemotron", "test response")));
                handle.emit();

                time::sleep(Duration::from_millis(200)).await;

                // Verify message in NATS
                let messages = consume_messages(
                    &client,
                    &stream_name,
                    "test-consumer",
                    1,
                    Duration::from_secs(2),
                )
                .await;

                assert_eq!(messages.len(), 1, "Should receive exactly one audit record");
                let record = &messages[0];
                assert_eq!(record["schema_version"], 1);
                assert_eq!(record["request_id"], "test-req-1");
                assert_eq!(record["model"], "nemotron");
                assert!(record["request"].is_object());
                assert!(record["response"].is_object());

                client.jetstream().delete_stream(&stream_name).await.ok();
            },
        )
        .await;
    }

    #[tokio::test]
    #[ignore] // Manual testing only - requires NATS on localhost:4222
    async fn test_audit_nats_store_flag() {
        // Test that store flag controls whether records are audited
        const TEST_SUBJECT: &str = "test.audit.store";

        async_with_vars(
            [
                ("DYN_AUDIT_SINKS", Some("nats")),
                ("DYN_AUDIT_NATS_SUBJECT", Some(TEST_SUBJECT)),
            ],
            async {
                let stream_name = format!("test_store_{}", Uuid::new_v4());

                let client = create_test_nats_client().await;
                setup_test_stream(&client, &stream_name, TEST_SUBJECT).await;

                bus::init(100);
                sink::spawn_workers_from_env().await.unwrap();
                time::sleep(Duration::from_millis(100)).await;

                // Request with store=true (should be audited)
                let request_true = create_test_request("nemotron", true);
                if let Some(mut handle) = handle::create_handle(&request_true, "store-true") {
                    handle.set_request(Arc::new(request_true.clone()));
                    handle.emit();
                }

                // Request with store=false (should NOT be audited)
                let request_false = create_test_request("nemotron", false);
                assert!(
                    handle::create_handle(&request_false, "store-false").is_none(),
                    "Should not create handle when store=false"
                );

                time::sleep(Duration::from_millis(200)).await;

                let messages = consume_messages(
                    &client,
                    &stream_name,
                    "test-consumer",
                    2,
                    Duration::from_secs(2),
                )
                .await;
                assert_eq!(messages.len(), 1, "Should only audit when store=true");
                assert_eq!(messages[0]["request_id"], "store-true");

                client.jetstream().delete_stream(&stream_name).await.ok();
            },
        )
        .await;
    }
}
