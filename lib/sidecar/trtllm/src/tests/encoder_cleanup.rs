// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Encoder export release, cancellation, and health-probe cleanup.

use super::*;

#[tokio::test]
async fn encoder_release_uses_its_rpc_without_generating_or_aborting_a_request() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let service = FakeTrtllm::default();
        *service.encoder_role.lock().await = Some("encode".into());
        let server = FakeServer::start(service).await;
        let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
        encoder.start(0).await.unwrap();
        let mut req = request_with_encoder_result();
        req.token_ids = Default::default();
        req.multi_modal_data = None;
        req.extra_args = Some(json!({"release_encoder_result": true}));
        let outputs = collect(&encoder, req).await;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].finish_reason, Some(FinishReason::Stop));
        assert!(outputs[0].token_ids.is_empty());
        assert_eq!(
            server
                .service
                .released_encoder_handoffs
                .lock()
                .await
                .as_slice(),
            [pb::EncodedMediaRef {
                opaque_state: vec![1, 2, 3],
            }]
        );
        assert!(server.service.requests.lock().await.is_empty());
        assert!(server.service.aborts.lock().await.is_empty());
        encoder.cleanup().await.unwrap();
    })
    .await
    .expect("encoder handoff release must finish promptly");
}

#[tokio::test]
async fn pending_encode_emits_one_cancelled_terminal_on_stop_kill_and_shutdown() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for cancellation in ["stop", "kill", "shutdown"] {
            let service = FakeTrtllm::default();
            *service.encoder_role.lock().await = Some("encode".into());
            service.hang_before_stream.store(true, Ordering::SeqCst);
            let server = FakeServer::start(service).await;
            let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
            encoder.start(0).await.unwrap();
            let mut req = request();
            add_images(&mut req, ["https://example.com/image.png"]);
            let context = dynamo_backend_common::testing::mock_context();
            let stream = encoder
                .generate(req, GenerateContext::new(context.clone(), None))
                .await
                .unwrap();
            let outputs = tokio::spawn(async move { stream.collect::<Vec<_>>().await });
            server.service.encode_started.notified().await;

            match cancellation {
                "stop" => context.stop_generating(),
                "kill" => context.kill(),
                "shutdown" => encoder.cleanup().await.unwrap(),
                _ => unreachable!(),
            }

            let outputs = outputs.await.unwrap();
            assert_eq!(outputs.len(), 1, "{cancellation}");
            let output = outputs.into_iter().next().unwrap().unwrap();
            assert_eq!(output.finish_reason, Some(FinishReason::Cancelled));
            assert!(output.encoder_result.is_none());
            assert!(output.token_ids.is_empty());
            assert_eq!(server.service.encode_requests.lock().await.len(), 1);
            assert!(server.service.requests.lock().await.is_empty());
            assert!(
                server
                    .service
                    .released_encoder_handoffs
                    .lock()
                    .await
                    .is_empty()
            );
            if cancellation != "shutdown" {
                encoder.cleanup().await.unwrap();
            }
        }
    })
    .await
    .expect("an in-flight Encode must settle on cancellation or engine shutdown");
}

#[tokio::test]
async fn malformed_encoder_release_is_rejected_without_inference_or_release() {
    let service = FakeTrtllm::default();
    *service.encoder_role.lock().await = Some("encode".into());
    let server = FakeServer::start(service).await;
    let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
    encoder.start(0).await.unwrap();
    let mut release = request_with_encoder_result();
    release.token_ids = Default::default();
    release.multi_modal_data = None;
    release.extra_args = Some(json!({"release_encoder_result": true}));
    let mut with_tokens = release.clone();
    with_tokens.token_ids = vec![1].into();
    let mut with_media = release.clone();
    add_images(&mut with_media, ["https://example.com/image.png"]);
    let mut without_reference = release.clone();
    without_reference.encoder_result = None;
    let mut invalid_reference = release;
    invalid_reference.encoder_result = Some(json!({"unexpected": "not a reference"}));
    for req in [
        with_tokens,
        with_media,
        without_reference,
        invalid_reference,
    ] {
        let error = encoder
            .generate(
                req,
                GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
            )
            .await
            .err()
            .expect("malformed releases must not become inference requests");
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(dynamo_backend_common::BackendError::InvalidArgument)
        );
    }
    assert!(server.service.encode_requests.lock().await.is_empty());
    assert!(server.service.requests.lock().await.is_empty());
    assert!(
        server
            .service
            .released_encoder_handoffs
            .lock()
            .await
            .is_empty()
    );
    encoder.cleanup().await.unwrap();
}

#[tokio::test]
async fn encoder_cancelled_terminal_releases_unobserved_export() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for killed in [false, true] {
            let service = FakeTrtllm::default();
            *service.encoder_role.lock().await = Some("encode".into());
            let server = FakeServer::start(service).await;
            let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
            let config = encoder.start(0).await.unwrap();
            assert_eq!(
                config
                    .runtime_data
                    .get(dynamo_llm::local_model::runtime_config::ENCODER_SUPPORTED_MODALITIES),
                Some(&json!(["image_url"]))
            );
            let mut req = request();
            add_images(&mut req, ["https://example.com/image.png"]);
            let context = dynamo_backend_common::testing::mock_context();
            let mut stream = encoder
                .generate(req, GenerateContext::new(context.clone(), None))
                .await
                .unwrap();
            let output = stream.next().await.unwrap().unwrap();
            assert!(output.encoder_result.is_some());
            // The response publisher signals a failed send through this context.
            if killed {
                context.kill();
            } else {
                context.stop_generating();
            }
            drop(stream);
            server.service.encoder_release_started.notified().await;
            assert_eq!(
                server
                    .service
                    .released_encoder_handoffs
                    .lock()
                    .await
                    .as_slice(),
                [pb::EncodedMediaRef {
                    opaque_state: vec![1, 2, 3],
                }]
            );
            encoder.cleanup().await.unwrap();
        }
    })
    .await
    .expect("a cancelled encoder terminal must release its unobserved export");
}

#[tokio::test]
async fn encoder_invalid_terminal_releases_export() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let service = FakeTrtllm::default();
        *service.encoder_role.lock().await = Some("encode".into());
        service
            .invalid_encode_response
            .store(true, Ordering::SeqCst);
        let server = FakeServer::start(service).await;
        let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
        encoder.start(0).await.unwrap();
        let mut req = request();
        add_images(&mut req, ["https://example.com/image.png"]);
        let mut stream = encoder
            .generate(
                req,
                GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
            )
            .await
            .unwrap();
        assert!(
            stream
                .next()
                .await
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("different request_id")
        );
        drop(stream);
        server.service.encoder_release_started.notified().await;
        assert_eq!(
            server.service.released_encoder_handoffs.lock().await.len(),
            1
        );
        encoder.cleanup().await.unwrap();
    })
    .await
    .expect("an invalid encoder terminal must release its export");
}

#[tokio::test]
async fn encoder_probe_releases_export_before_success_without_changing_normal_encodes() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let service = FakeTrtllm::default();
        *service.encoder_role.lock().await = Some("encode".into());
        let server = FakeServer::start(service).await;
        let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
        encoder.start(0).await.unwrap();
        for is_probe in [false, true] {
            let mut req = request();
            add_images(&mut req, ["https://example.com/image.png"]);
            req.is_probe = is_probe;
            let outputs = collect(&encoder, req).await;
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].finish_reason, Some(FinishReason::Stop));
            assert_eq!(outputs[0].encoder_result.is_none(), is_probe);
            assert_eq!(
                server.service.released_encoder_handoffs.lock().await.len(),
                usize::from(is_probe)
            );
        }
        assert_eq!(server.service.encode_requests.lock().await.len(), 2);
        assert!(server.service.aborts.lock().await.is_empty());
        encoder.cleanup().await.unwrap();
    })
    .await
    .expect("encoder canary must release its export before succeeding");
}

#[tokio::test]
async fn encoder_probe_release_survives_cancellation_and_stream_drop_during_cleanup() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let service = FakeTrtllm::default();
        *service.encoder_role.lock().await = Some("encode".into());
        service.hang_encoder_release.store(true, Ordering::SeqCst);
        let server = FakeServer::start(service).await;
        let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
        encoder.start(0).await.unwrap();
        let mut req = request();
        add_images(&mut req, ["https://example.com/image.png"]);
        req.is_probe = true;
        let context = dynamo_backend_common::testing::mock_context();
        let mut stream = encoder
            .generate(req, GenerateContext::new(context.clone(), None))
            .await
            .unwrap();
        let mut next = Box::pin(stream.next());
        tokio::select! {
            _ = server.service.encoder_release_started.notified() => {},
            _ = &mut next => panic!("probe completed before its release RPC"),
        }
        context.stop_generating();
        drop(next);
        drop(stream);
        server
            .service
            .hang_encoder_release
            .store(false, Ordering::SeqCst);
        server.service.encoder_release_resume.notify_one();
        server.service.encoder_release_started.notified().await;
        assert_eq!(
            server.service.released_encoder_handoffs.lock().await.len(),
            2
        );
        let completions =
            std::mem::take(&mut *server.service.encoder_release_completions.lock().await);
        assert_eq!(completions.len(), 2);
        for completion in completions {
            assert!(
                completion.await.is_err(),
                "release handler is still running"
            );
        }
        encoder.cleanup().await.unwrap();
    })
    .await
    .expect("dropped encoder canary must release its export");
}

#[tokio::test]
async fn encoder_probe_reports_release_failure_and_retries_cleanup() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let service = FakeTrtllm::default();
        *service.encoder_role.lock().await = Some("encode".into());
        *service.encoder_release_error.lock().await = Some(tonic::Code::Unavailable);
        let server = FakeServer::start(service).await;
        let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
        encoder.start(0).await.unwrap();
        let mut req = request();
        add_images(&mut req, ["https://example.com/image.png"]);
        req.is_probe = true;
        let mut stream = encoder
            .generate(
                req,
                GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
            )
            .await
            .unwrap();
        assert!(
            stream
                .next()
                .await
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("ReleaseEncodedMedia")
        );
        server.service.encoder_release_started.notified().await;
        drop(stream);
        server.service.encoder_release_started.notified().await;
        assert_eq!(
            server.service.released_encoder_handoffs.lock().await.len(),
            2
        );
        encoder.cleanup().await.unwrap();
    })
    .await
    .expect("failed encoder canary cleanup must not report healthy");
}
