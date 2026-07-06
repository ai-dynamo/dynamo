// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for NATS JetStream request trace payload sink
//!
//! These tests verify request_payload records are published to NATS JetStream.
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
    use dynamo_llm::audit::handle;
    use dynamo_llm::audit::init_from_env_with_shutdown;
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionResponse;
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
        const TEST_SUBJECT: &str = "test.request_trace.basic";
        // Core test: request_payload records are published to NATS with correct structure
        async_with_vars(
            [
                ("DYN_REQUEST_TRACE", Some("1")),
                ("DYN_REQUEST_TRACE_SINKS", Some("nats")),
                ("DYN_REQUEST_TRACE_RECORDS", Some("request_payload")),
                ("DYN_REQUEST_TRACE_NATS_SUBJECT", Some(TEST_SUBJECT)),
            ],
            async {
                let stream_name = format!("test_basic_{}", Uuid::new_v4());

                let client = create_test_nats_client().await;
                setup_test_stream(&client, &stream_name, TEST_SUBJECT).await;

                // Drive the full request trace lifecycle (bus::init, spawn
                // workers, mark_capture_active) so `create_handle` succeeds.
                let shutdown = tokio_util::sync::CancellationToken::new();
                init_from_env_with_shutdown(shutdown.clone()).await.unwrap();
                time::sleep(Duration::from_millis(100)).await;

                // Emit a single combined request+response payload record.
                let request = create_test_request("nemotron", true);
                let handle = handle::create_handle(&request, "test-req-1")
                    .expect("Failed to create payload handle");
                handle.emit(Some(Arc::new(create_test_response(
                    "nemotron",
                    "test response",
                ))));

                time::sleep(Duration::from_millis(200)).await;

                // Verify the single request_payload record in NATS.
                let messages = consume_messages(
                    &client,
                    &stream_name,
                    "test-consumer",
                    1,
                    Duration::from_secs(2),
                )
                .await;

                assert_eq!(
                    messages.len(),
                    1,
                    "Should receive one request_payload record"
                );
                let record = &messages[0];

                assert_eq!(record["schema"], "dynamo.request.trace.v1");
                assert_eq!(record["event_type"], "request_payload");
                assert_eq!(record["payload"]["request_id"], "test-req-1");
                assert_eq!(record["payload"]["model"], "nemotron");
                assert!(record["payload"]["request"].is_object());
                assert!(record["payload"]["response"].is_object());

                client.jetstream().delete_stream(&stream_name).await.ok();
                shutdown.cancel();
            },
        )
        .await;
    }

    #[tokio::test]
    #[ignore] // Manual testing only - requires NATS on localhost:4222
    async fn test_audit_nats_request_payload_records_ignore_store_flag() {
        // Test that request_payload records are emitted regardless of store.
        const TEST_SUBJECT: &str = "test.request_trace.request_payload";

        async_with_vars(
            [
                ("DYN_REQUEST_TRACE", Some("1")),
                ("DYN_REQUEST_TRACE_SINKS", Some("nats")),
                ("DYN_REQUEST_TRACE_RECORDS", Some("request_payload")),
                ("DYN_REQUEST_TRACE_NATS_SUBJECT", Some(TEST_SUBJECT)),
            ],
            async {
                let stream_name = format!("test_include_{}", Uuid::new_v4());

                let client = create_test_nats_client().await;
                setup_test_stream(&client, &stream_name, TEST_SUBJECT).await;

                // Drive the full request trace lifecycle (bus::init, spawn
                // workers, mark_capture_active) so `create_handle` succeeds.
                let shutdown = tokio_util::sync::CancellationToken::new();
                init_from_env_with_shutdown(shutdown.clone()).await.unwrap();
                time::sleep(Duration::from_millis(100)).await;

                let request_true = create_test_request("nemotron", true);
                handle::create_handle(&request_true, "store-true")
                    .expect("store=true handle")
                    .emit(None);

                let request_false = create_test_request("nemotron", false);
                handle::create_handle(&request_false, "store-false")
                    .expect("store=false handle")
                    .emit(None);

                time::sleep(Duration::from_millis(200)).await;

                let messages = consume_messages(
                    &client,
                    &stream_name,
                    "test-consumer",
                    2,
                    Duration::from_secs(2),
                )
                .await;
                assert_eq!(
                    messages.len(),
                    2,
                    "Should emit records for store=true and store=false when request_payload records are enabled"
                );
                let request_ids: std::collections::HashSet<_> = messages
                    .iter()
                    .map(|message| message["payload"]["request_id"].as_str().unwrap())
                    .collect();
                assert!(request_ids.contains("store-true"));
                assert!(request_ids.contains("store-false"));
                for message in &messages {
                    assert_eq!(message["schema"], "dynamo.request.trace.v1");
                    assert_eq!(message["event_type"], "request_payload");
                    assert!(message["payload"]["request"].is_object());
                }

                client.jetstream().delete_stream(&stream_name).await.ok();
                shutdown.cancel();
            },
        )
        .await;
    }
}
