// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

use dynamo_backend_common::testing::mock_context;
use dynamo_backend_common::{
    FinishReason, GenerateContext, LLMEngine, MultimodalData, PreprocessedRequest,
};
use dynamo_sidecar_common::EngineBootstrapResult;
use dynamo_sidecar_testkit::bounded;
use dynamo_sidecar_testkit::control::{Controller, RequestPlan};
use dynamo_sidecar_testkit::fixtures::{collect, request};
use dynamo_trtllm_sidecar::TrtllmSidecarEngine;
use dynamo_trtllm_sidecar::proto as pb;

#[allow(dead_code)]
mod support;
use support::trtllm::{Adapter, Fixture};
use support::{FixtureConfig, ProcessFixture, SidecarFixture};

fn image_request() -> PreprocessedRequest {
    let mut request = request("mocker-model", vec![11, 22, 33], 1);
    request.stop_conditions.max_tokens = None;
    request.multi_modal_data = Some(HashMap::from([(
        "image_url".into(),
        vec![
            MultimodalData::RawUrl("https://example.com/first.png".into()),
            MultimodalData::RawUrl("data:image/png;base64,aW1hZ2U=".into()),
        ],
    )]));
    request.multi_modal_uuids = Some(HashMap::from([(
        "image_url".into(),
        vec![Some("first".into()), None],
    )]));
    request
}

fn finished() -> pb::GenerateResponse {
    pb::GenerateResponse {
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Stop as i32,
                ..Default::default()
            },
        )),
        usage: Some(pb::Usage {
            prompt_tokens: 131,
            total_tokens: 131,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn advertise_images(fixture: &Fixture) {
    fixture
        .discovery
        .portable_image
        .store(true, Ordering::SeqCst);
    fixture
        .discovery
        .advertises_image
        .store(true, Ordering::SeqCst);
}

async fn wait_for_calls(fixture: &Fixture, count: usize) {
    bounded("Control discovery reached", async {
        while fixture.discovery.calls.load(Ordering::SeqCst) < count {
            tokio::task::yield_now().await;
        }
    })
    .await;
}

#[tokio::test]
async fn images_preserve_order_uuids_and_engine_expanded_usage() {
    let control = Controller::<Adapter>::default();
    let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
    advertise_images(&fixture);
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let ctx = mock_context();
    let handle = control.request(ctx.id(), RequestPlan::default());
    fixture.respond(ctx.id(), vec![finished()]);
    let outputs = collect(&engine, image_request(), GenerateContext::new(ctx, None)).await;
    let output = outputs.last().unwrap().as_ref().unwrap();
    assert_eq!(output.finish_reason, Some(FinishReason::Stop));
    assert_eq!(output.completion_usage.as_ref().unwrap().prompt_tokens, 131);
    let native = handle.native_request().unwrap();
    assert_eq!(native.stopping.unwrap().max_tokens, None);
    assert_eq!(native.media.len(), 2);
    assert!(
        native
            .media
            .iter()
            .all(|item| item.modality == pb::Modality::Image as i32)
    );
    assert_eq!(native.media[0].uuid, "first");
    assert_eq!(native.media[1].uuid, "");
    assert_eq!(
        native.media[0].source,
        Some(pb::media_item::Source::Url(
            "https://example.com/first.png".into()
        ))
    );
    assert_eq!(
        native.media[1].source,
        Some(pb::media_item::Source::DataUri(
            "data:image/png;base64,aW1hZ2U=".into()
        ))
    );
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn both_image_capability_signals_are_required() {
    for (portable, image) in [(true, false), (false, true)] {
        let control = Controller::<Adapter>::default();
        let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
        fixture
            .discovery
            .portable_image
            .store(portable, Ordering::SeqCst);
        fixture
            .discovery
            .advertises_image
            .store(image, Ordering::SeqCst);
        let engine = fixture.engine().await;
        engine.start(0).await.unwrap();
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        assert!(
            engine
                .generate(image_request(), GenerateContext::new(ctx, None))
                .await
                .is_err()
        );
        assert!(handle.native_request().is_none());
        engine.cleanup().await.unwrap();
        fixture.shutdown().await;
    }
}

fn bootstrap(
    fixture: &Fixture,
    context_length: Option<u32>,
    deadline_secs: u32,
) -> impl std::future::Future<Output = EngineBootstrapResult<TrtllmSidecarEngine>> + use<> {
    let mut args = vec![
        "dynamo-trtllm-sidecar".into(),
        "--grpc-endpoint".into(),
        fixture.endpoint(),
        "--model-path".into(),
        "mocker-model".into(),
        "--grpc-startup-deadline-secs".into(),
        deadline_secs.to_string(),
    ];
    if let Some(context_length) = context_length {
        args.extend(["--context-length".into(), context_length.to_string()]);
    }
    TrtllmSidecarEngine::try_from_args_async(args).unwrap().1
}

#[tokio::test]
async fn startup_recovers_image_capabilities_before_registration() {
    for omit_window in [false, true] {
        let control = Controller::<Adapter>::default();
        let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
        advertise_images(&fixture);
        fixture
            .discovery
            .omit_context_length
            .store(omit_window, Ordering::SeqCst);
        fixture.discovery.unavailable.store(true, Ordering::SeqCst);
        let mut startup = tokio::spawn(bootstrap(&fixture, Some(32768), 5));
        wait_for_calls(&fixture, 2).await;
        assert!(
            !startup.is_finished(),
            "unresolved capabilities must prevent registration"
        );
        fixture.discovery.unavailable.store(false, Ordering::SeqCst);
        let (engine, _) = bounded("recover startup discovery", &mut startup)
            .await
            .unwrap()
            .unwrap();
        let config = engine.start(0).await.unwrap();
        assert_eq!(config.llm.unwrap().context_length, Some(32768));
        let calls = fixture.discovery.calls.load(Ordering::SeqCst);
        // Serving must use startup metadata, including image support when the
        // engine supplied no window. No Generate may rediscover capabilities.
        fixture.discovery.unavailable.store(true, Ordering::SeqCst);
        for _ in 0..2 {
            let ctx = mock_context();
            let handle = control.request(ctx.id(), RequestPlan::default());
            if omit_window {
                let error = engine
                    .generate(image_request(), GenerateContext::new(ctx, None))
                    .await
                    .err()
                    .expect("media still needs a reported engine window");
                assert!(
                    error.to_string().contains("--context-length to match"),
                    "{error}"
                );
                assert!(handle.native_request().is_none());
            } else {
                fixture.respond(ctx.id(), vec![finished()]);
                let outputs =
                    collect(&engine, image_request(), GenerateContext::new(ctx, None)).await;
                assert_eq!(
                    outputs.last().unwrap().as_ref().unwrap().finish_reason,
                    Some(FinishReason::Stop)
                );
                assert_eq!(handle.native_request().unwrap().media.len(), 2);
            }
        }
        assert_eq!(fixture.discovery.calls.load(Ordering::SeqCst), calls);
        engine.cleanup().await.unwrap();
        fixture.shutdown().await;
    }
}

#[tokio::test]
async fn discovery_model_and_authentication_errors_fail_startup_without_retry() {
    for configured in [None, Some(32768)] {
        for code in [
            tonic::Code::InvalidArgument,
            tonic::Code::NotFound,
            tonic::Code::Unauthenticated,
            tonic::Code::PermissionDenied,
        ] {
            let mut fixture =
                Fixture::start(Controller::<Adapter>::default(), FixtureConfig::default()).await;
            *fixture.discovery.error.lock().unwrap() = Some(code);
            let result = bounded(
                "terminal discovery error",
                bootstrap(&fixture, configured, 60),
            )
            .await;
            let error = result
                .err()
                .expect("model/authentication errors must prevent registration");
            assert!(
                error.to_string().contains("model discovery rejected"),
                "{error}"
            );
            assert!(error.to_string().contains(&format!("{code:?}")), "{error}");
            assert_eq!(fixture.discovery.calls.load(Ordering::SeqCst), 1);
            fixture.shutdown().await;
        }
    }
}

#[tokio::test]
async fn startup_deadline_bounds_retrying_and_stalled_discovery() {
    for configured in [None, Some(32768)] {
        for held in [false, true] {
            let mut fixture =
                Fixture::start(Controller::<Adapter>::default(), FixtureConfig::default()).await;
            fixture.discovery.unavailable.store(true, Ordering::SeqCst);
            fixture.discovery.hold.store(held, Ordering::SeqCst);
            let result =
                tokio::time::timeout(Duration::from_secs(4), bootstrap(&fixture, configured, 2))
                    .await
                    .expect("discovery must respect the startup deadline");
            let error = result
                .err()
                .expect("unresolved capabilities must prevent registration");
            assert!(error.to_string().contains("startup deadline"), "{error}");
            let calls = fixture.discovery.calls.load(Ordering::SeqCst);
            if held {
                assert_eq!(calls, 1);
            } else {
                assert!(calls >= 2);
            }
            fixture.release_discovery();
            fixture.shutdown().await;
        }
    }
}

#[tokio::test]
async fn shutdown_cancels_retrying_and_stalled_bootstrap_discovery() {
    for held in [false, true] {
        let mut fixture =
            Fixture::start(Controller::<Adapter>::default(), FixtureConfig::default()).await;
        fixture.discovery.unavailable.store(true, Ordering::SeqCst);
        fixture.discovery.hold.store(held, Ordering::SeqCst);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let cancel = shutdown.clone();
        let startup = bootstrap(&fixture, Some(32768), 60);
        // Mirrors the shared runner's cancellation of the bootstrap future.
        let task = tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {},
                _ = startup => panic!("unresolved discovery must not register a worker"),
            }
        });
        wait_for_calls(&fixture, 1).await;
        cancel.cancel();
        bounded("shutdown cancels bootstrap", task).await.unwrap();
        let calls = fixture.discovery.calls.load(Ordering::SeqCst);
        fixture.release_discovery();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(fixture.discovery.calls.load(Ordering::SeqCst), calls);
        fixture.shutdown().await;
    }
}

#[tokio::test]
async fn image_window_override_is_rejected_before_dispatch() {
    let control = Controller::<Adapter>::default();
    let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
    advertise_images(&fixture);
    let engine = fixture.engine_with_context_length(4096);
    engine.start(0).await.unwrap();
    for max_tokens in [None, Some(1)] {
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        let mut req = image_request();
        req.stop_conditions.max_tokens = max_tokens;
        let result = engine.generate(req, GenerateContext::new(ctx, None)).await;
        let error = result.err().expect("window mismatch must fail");
        assert!(error.to_string().contains("--context-length to match"));
        assert!(handle.native_request().is_none());
    }
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
}
