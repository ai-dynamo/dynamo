// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Encoder capability negotiation and image handoffs over fake gRPC.

use super::*;

#[tokio::test]
async fn incompatible_encoder_advertisements_fail_startup_before_dispatch() {
    tokio::time::timeout(Duration::from_secs(10), async {
        for role in ["encode", "consume"] {
            for case in [
                "schema",
                "backend",
                "image",
                "ambiguous",
                "shared",
                "mixed",
                "release",
            ] {
                if role == "consume" && matches!(case, "mixed" | "release") {
                    continue;
                }
                let mut info = encoder_model_info(role);
                let encoder = info.encoder.as_mut().unwrap();
                match case {
                    "schema" => encoder.schema_version = Some(2),
                    "backend" if role == "encode" => {
                        encoder.transfer_backend = Some("unknown".into())
                    }
                    "backend" => encoder.supported_transfer_backends = vec!["unknown".into()],
                    "image" => info.extra = None,
                    "release" => {
                        encoder.supports_abort_cleanup = Some(false);
                    }
                    "ambiguous" => {
                        encoder.encoder_only = Some(true);
                        encoder.accepts_handoff = Some(true);
                    }
                    "shared" if role == "encode" => {
                        encoder.transfer_backend = Some("trtllm-shared-safetensors-v1".into())
                    }
                    "shared" => {
                        encoder.supported_transfer_backends =
                            vec!["trtllm-shared-safetensors-v1".into()]
                    }
                    "mixed" => {
                        encoder
                            .supported_transfer_backends
                            .push("trtllm-shared-safetensors-v1".into());
                    }
                    _ => unreachable!(),
                }
                let service = FakeTrtllm::default();
                *service.model_info_override.lock().await = Some(info);
                let server = FakeServer::start(service).await;
                let sidecar = engine(&server.endpoint, 1);
                let error = sidecar.start(0).await.expect_err("invalid advertised role");
                assert!(
                    error
                        .to_string()
                        .contains("incompatible TensorRT-LLM encoder role"),
                    "{role}/{case}: {error}"
                );
                assert_eq!(server.service.model_info_calls.load(Ordering::SeqCst), 1);
                assert!(server.service.requests.lock().await.is_empty());
            }
        }
    })
    .await
    .expect("invalid capability startup must fail promptly");
}

#[tokio::test]
async fn optional_encoder_consumers_can_start_without_encoder_routing() {
    tokio::time::timeout(Duration::from_secs(10), async {
        for mode in [AGG, PREFILL] {
            for required in [false, true] {
                let mut info = encoder_model_info("consume");
                info.encoder.as_mut().unwrap().requires_handoff = Some(required);
                info.encoder
                    .as_mut()
                    .unwrap()
                    .supported_transfer_backends
                    .push("trtllm-shared-safetensors-v1".into());
                info.extra.as_mut().unwrap().fields.insert(
                    "trtllm_requires_encoder_cache".into(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::BoolValue(!required)),
                    },
                );
                let service = FakeTrtllm::default();
                *service.model_info_override.lock().await = Some(info);
                let server = FakeServer::start(service).await;
                let engine = engine_in_mode(&server.endpoint, 1, mode);
                let result = engine.start(0).await;
                if required {
                    assert!(
                        result
                            .unwrap_err()
                            .to_string()
                            .contains("--route-to-encoder")
                    );
                } else {
                    result.expect("optional consumer can serve media inline");
                    engine.cleanup().await.unwrap();
                }
                assert!(server.service.requests.lock().await.is_empty());
            }
        }
    })
    .await
    .expect("consumer capability discovery must finish promptly");
}

#[tokio::test]
async fn optional_encoder_routing_keeps_inline_consumer_policy() {
    use dynamo_llm::protocols::common::preprocessor::{
        ENCODER_RESULT_REQUIRES_PREFILL, REQUIRES_ENCODER_RESULT,
    };

    tokio::time::timeout(Duration::from_secs(10), async {
        for mode in [AGG, PREFILL, DisaggregationMode::Decode] {
            for required in [false, true] {
                let mut info = encoder_model_info("consume");
                info.encoder.as_mut().unwrap().requires_handoff = Some(required);
                info.extra = Some(
                    dynamo_sidecar_common::json_to_struct(
                        json!({"trtllm_supported_modalities": ["image", "video"]}),
                        "test",
                    )
                    .unwrap(),
                );
                let service = FakeTrtllm::default();
                *service.model_info_override.lock().await = Some(info);
                let server = FakeServer::start(service).await;
                let mut args = vec![
                    "sidecar".into(),
                    "--grpc-endpoint".into(),
                    server.endpoint.clone(),
                    "--model-path".into(),
                    "model-source".into(),
                ];
                if mode != AGG {
                    args.extend([
                        "--disaggregation-mode".into(),
                        if mode == PREFILL {
                            "prefill".into()
                        } else {
                            "decode".into()
                        },
                    ]);
                }
                if !mode.is_decode() {
                    args.push("--route-to-encoder".into());
                }
                let consumer = TrtllmSidecarEngine::from_args(args).unwrap().0;
                let config = consumer.start(0).await.unwrap();
                assert_eq!(
                    config
                        .runtime_data
                        .get(REQUIRES_ENCODER_RESULT)
                        .and_then(serde_json::Value::as_bool),
                    (required && !mode.is_decode()).then_some(true)
                );
                assert_eq!(
                    config
                        .runtime_data
                        .get(ENCODER_RESULT_REQUIRES_PREFILL)
                        .and_then(serde_json::Value::as_bool),
                    mode.is_prefill().then_some(true),
                    "prefill must consume its handoff even when inline encoding is available"
                );
                if !required && mode != DisaggregationMode::Decode {
                    let mut req = request();
                    req.multi_modal_data = Some(std::collections::HashMap::from([(
                        "video_url".into(),
                        vec![MultimodalData::RawUrl(
                            "https://example.com/video.mp4".into(),
                        )],
                    )]));
                    collect(&consumer, req).await;
                    let requests = server.service.requests.lock().await;
                    assert_eq!(requests.len(), 1);
                    assert!(requests[0].encoder.is_none());
                    assert_eq!(requests[0].media[0].modality, pb::Modality::Video as i32);
                }
                let requests_before_handoff = server.service.requests.lock().await.len();
                let mut handoff = request();
                add_images(&mut handoff, ["https://example.com/image.png"]);
                handoff.encoder_result = Some(json!({
                    "transfer_backend": "trtllm-nixl-encoder-v1",
                    "opaque_state": "AQID",
                }));
                if mode.is_decode() {
                    let context = dynamo_backend_common::testing::mock_context();
                    let error = match consumer
                        .generate(handoff, GenerateContext::new(context, None))
                        .await
                    {
                        Err(error) => error,
                        Ok(_) => panic!("decode cannot consume an unclaimed encoder handoff"),
                    };
                    assert_eq!(
                        error.class(),
                        dynamo_runtime::error::ErrorClass::InvalidRequest
                    );
                    assert!(
                        error
                            .to_string()
                            .contains("encoder_result requires an aggregated or prefill worker")
                    );
                    assert_eq!(
                        server.service.requests.lock().await.len(),
                        requests_before_handoff,
                        "invalid decode handoff must be rejected before native dispatch"
                    );
                } else {
                    collect(&consumer, handoff).await;
                    let requests = server.service.requests.lock().await;
                    assert_eq!(requests.len(), requests_before_handoff + 1);
                    assert!(matches!(
                        requests.last().unwrap().encoder.as_ref().unwrap().mode.as_ref(),
                        Some(pb::encoder_options::Mode::Handoff(reference))
                            if reference.opaque_state == [1, 2, 3]
                    ));
                }
                consumer.cleanup().await.unwrap();
            }
        }
    })
    .await
    .expect("consumer metadata and inline fallback stay bounded");
}

#[tokio::test]
async fn image_encoder_and_consumer_exchange_opaque_handoff_over_grpc() {
    tokio::time::timeout(Duration::from_secs(10), async {
        use dynamo_backend_common::metrics::{EngineMetrics, TestHierarchy};
        let source_service = FakeTrtllm::default();
        *source_service.encoder_role.lock().await = Some("encode".into());
        let source_server = FakeServer::start(source_service).await;
        let encoder = engine_in_mode(&source_server.endpoint, 1, DisaggregationMode::Encode);
        encoder.start(0).await.unwrap();
        let metrics = EngineMetrics::from_hierarchy(TestHierarchy::new());
        let bindings = encoder
            .setup_metrics(dynamo_backend_common::MetricsCtx {
                model: "model",
                component: "encode",
                model_load_time_seconds: 0.0,
                metrics: &metrics,
            })
            .await
            .unwrap();
        assert!(bindings.dp_ranks.is_empty());
        assert!(encoder.kv_event_sources().await.unwrap().is_empty());
        let mut input = request();
        add_images(&mut input, ["https://example.com/image.png"]);
        let outputs = collect(&encoder, input.clone()).await;
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].token_ids.is_empty());
        input.encoder_result = outputs[0].encoder_result.as_deref().cloned();

        let consumer_service = FakeTrtllm::default();
        *consumer_service.encoder_role.lock().await = Some("consume".into());
        let consumer_server = FakeServer::start(consumer_service).await;
        let mut argv = vec![
            "sidecar".to_string(),
            "--grpc-endpoint".into(),
            consumer_server.endpoint.clone(),
            "--model-path".into(),
            "model-source".into(),
            "--route-to-encoder".into(),
        ];
        let consumer = TrtllmSidecarEngine::from_args(std::mem::take(&mut argv))
            .unwrap()
            .0;
        consumer.start(1).await.unwrap();
        let outputs = collect(&consumer, input).await;
        assert_eq!(
            outputs.last().unwrap().finish_reason,
            Some(FinishReason::Stop)
        );
        let requests = consumer_server.service.requests.lock().await;
        assert_eq!(
            crate::test_fixtures::encoder_handoff(requests[0].encoder.as_ref().unwrap())
                .opaque_state,
            [1, 2, 3]
        );
        encoder.cleanup().await.unwrap();
        consumer.cleanup().await.unwrap();
    })
    .await
    .expect("encoder handoff timed out");
}

#[tokio::test]
async fn prefill_consumes_encoder_reference_and_decode_uses_only_kv_over_grpc() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let encoder_service = FakeTrtllm::default();
        *encoder_service.encoder_role.lock().await = Some("encode".into());
        let encoder_server = FakeServer::start(encoder_service).await;
        let encoder = engine_in_mode(&encoder_server.endpoint, 1, DisaggregationMode::Encode);
        encoder.start(0).await.unwrap();

        let mut input = request();
        add_images(&mut input, ["https://example.com/image.png"]);
        input.encoder_result = collect(&encoder, input.clone()).await[0]
            .encoder_result
            .as_deref()
            .cloned();

        let prefill_service = FakeTrtllm::default();
        *prefill_service.encoder_role.lock().await = Some("consume".into());
        let prefill_server = FakeServer::start(prefill_service).await;
        let prefill = TrtllmSidecarEngine::from_args(vec![
            "sidecar".into(),
            "--grpc-endpoint".into(),
            prefill_server.endpoint.clone(),
            "--model-path".into(),
            "model-source".into(),
            "--disaggregation-mode".into(),
            "prefill".into(),
            "--route-to-encoder".into(),
        ])
        .expect("prefill encoder routing")
        .0;
        prefill.start(1).await.unwrap();
        let outputs = collect(&prefill, input.clone()).await;
        input.prefill_result = Some(PrefillResult {
            disaggregated_params: outputs
                .last()
                .unwrap()
                .disaggregated_params
                .as_deref()
                .cloned()
                .expect("prefill KV handoff"),
            prompt_tokens_details: None,
        });

        let decode_service = FakeTrtllm::default();
        *decode_service.encoder_role.lock().await = Some("consume".into());
        let decode_server = FakeServer::start(decode_service).await;
        let decode = engine_in_mode(&decode_server.endpoint, 1, DisaggregationMode::Decode);
        decode.start(2).await.unwrap();
        let outputs = collect(&decode, input).await;
        assert!(outputs.iter().any(|output| output.token_ids == [42]));
        assert_eq!(
            outputs.last().unwrap().finish_reason,
            Some(FinishReason::Stop)
        );

        let encoder_requests = encoder_server.service.requests.lock().await;
        assert_eq!(encoder_requests.len(), 1);
        assert!(matches!(
            encoder_requests[0].encoder.as_ref().unwrap().mode,
            Some(pb::encoder_options::Mode::EncodeOnly(_))
        ));
        let prefill_requests = prefill_server.service.requests.lock().await;
        assert_eq!(prefill_requests.len(), 1);
        assert_eq!(
            crate::test_fixtures::encoder_handoff(prefill_requests[0].encoder.as_ref().unwrap())
                .opaque_state,
            [1, 2, 3]
        );
        assert!(is_context_only(&prefill_requests[0]));
        let decode_requests = decode_server.service.requests.lock().await;
        assert_eq!(decode_requests.len(), 1);
        assert!(decode_requests[0].encoder.is_none());
        assert_eq!(
            decode_requests[0]
                .kv
                .as_ref()
                .unwrap()
                .session
                .as_ref()
                .unwrap(),
            &fake_session()
        );
        assert_eq!(decode_requests[0].media, prefill_requests[0].media);
        drop(decode_requests);
        drop(prefill_requests);
        drop(encoder_requests);

        encoder.cleanup().await.unwrap();
        prefill.cleanup().await.unwrap();
        decode.cleanup().await.unwrap();
    })
    .await
    .expect("three-role handoff timed out");
}
