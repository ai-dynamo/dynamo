// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn encode_uses_the_unary_rpc_and_generation_forwards_its_reference() {
    let service = FakeTrtllm::default();
    *service.encoder_role.lock().await = Some("encode".into());
    let server = FakeServer::start(service).await;
    let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
    encoder.start(0).await.unwrap();
    let mut input = request();
    add_images(&mut input, ["https://example.com/image.png"]);
    let encoded = collect(&encoder, input.clone()).await;
    assert_eq!(encoded.len(), 1);
    assert!(encoded[0].token_ids.is_empty());
    assert!(server.service.requests.lock().await.is_empty());
    assert_eq!(server.service.encode_requests.lock().await.len(), 1);
    input.encoder_result = encoded[0].encoder_result.as_deref().cloned();
    let consumer_service = FakeTrtllm::default();
    *consumer_service.encoder_role.lock().await = Some("consume".into());
    let consumer_server = FakeServer::start(consumer_service).await;
    let consumer = encoder_consumer_engine(&consumer_server.endpoint);
    let config = consumer.start(0).await.unwrap();
    assert_eq!(
        config
            .runtime_data
            .get(dynamo_llm::protocols::common::preprocessor::REQUIRES_ENCODER_RESULT),
        Some(&json!(true))
    );
    collect(&consumer, input).await;
    assert_eq!(
        consumer_server.service.requests.lock().await[0].encoded_media,
        Some(pb::EncodedMediaRef {
            opaque_state: vec![1, 2, 3]
        })
    );
    consumer.cleanup().await.unwrap();
    encoder.cleanup().await.unwrap();
}

#[tokio::test]
async fn encoder_routing_rejects_unverified_schema_or_role_before_dispatch() {
    for (native_role, release, expected) in [
        (
            "encode",
            "old-generation-contract",
            "pinned OpenEngine schema release",
        ),
        (
            "consume",
            include_str!("../../proto/REVISION").trim(),
            "sidecar role",
        ),
    ] {
        let service = FakeTrtllm::default();
        *service.encoder_role.lock().await = Some(native_role.into());
        *service.encoder_schema_release.lock().await = Some(release.into());
        let server = FakeServer::start(service).await;
        let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
        let error = encoder
            .start(0)
            .await
            .err()
            .expect("mismatched encoder deployment must fail startup");
        assert!(error.to_string().contains(expected), "{error}");
        assert!(server.service.encode_requests.lock().await.is_empty());
    }
}
