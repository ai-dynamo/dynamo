// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Encoder export release, cancellation, and health-probe cleanup.

use super::*;

#[tokio::test]
async fn encoder_release_uses_abort_without_generating_or_aborting_a_request() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let service = FakeTrtllm::default();
        *service.encoder_role.lock().await = Some("encode".into());
        let server = FakeServer::start(service).await;
        let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
        encoder.start(0).await.unwrap();
        let mut req = request_with_encoder_result();
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
            [pb::EncoderHandoffRef {
                transfer_backend: "trtllm-nixl-encoder-v1".into(),
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
async fn encoder_rejects_invalid_sampling_before_export() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let service = FakeTrtllm::default();
        *service.encoder_role.lock().await = Some("encode".into());
        let server = FakeServer::start(service).await;
        let encoder = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Encode);
        encoder.start(0).await.unwrap();
        for invalid_seed in [false, true] {
            let mut req = request();
            add_images(&mut req, ["https://example.com/image.png"]);
            if invalid_seed {
                req.sampling_options.seed = Some(-1);
            } else {
                req.sampling_options.top_k = Some(-5);
            }
            let error = encoder
                .generate(
                    req,
                    GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
                )
                .await
                .err()
                .expect("invalid sampling must fail before dispatch");
            assert!(error.to_string().contains(if invalid_seed {
                "seed must be non-negative"
            } else {
                "top_k must be"
            }));
        }
        assert!(server.service.requests.lock().await.is_empty());
        encoder.cleanup().await.unwrap();
    })
    .await
    .expect("encoder validation must finish promptly");
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
                [pb::EncoderHandoffRef {
                    transfer_backend: "trtllm-nixl-encoder-v1".into(),
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
        service.encode_ready_usage.store(true, Ordering::SeqCst);
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
                .contains("must omit generation usage")
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
        assert_eq!(server.service.requests.lock().await.len(), 2);
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
                .contains("Abort")
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
