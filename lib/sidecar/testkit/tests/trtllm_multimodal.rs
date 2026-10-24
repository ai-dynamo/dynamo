// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

use dynamo_backend_common::testing::mock_context;
use dynamo_backend_common::{
    BackendError, ErrorType, FinishReason, GenerateContext, LLMEngine, MultimodalData,
    PreprocessedRequest,
};
use dynamo_sidecar_testkit::bounded;
use dynamo_sidecar_testkit::control::{Controller, RequestPlan};
use dynamo_sidecar_testkit::fixtures::{collect, request};
use dynamo_trtllm_sidecar::proto as pb;
use futures::future::join_all;

#[allow(dead_code)]
mod support;
use support::trtllm::{Adapter, Fixture};
use support::{FixtureConfig, SidecarFixture};

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

#[tokio::test]
async fn concurrent_discovery_failure_is_shared_and_later_recovers() {
    let control = Controller::<Adapter>::default();
    let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
    fixture.discovery.unavailable.store(true, Ordering::SeqCst);
    let engine = fixture.engine_with_context_length(32768);
    engine.start(0).await.unwrap();
    fixture.discovery.hold.store(true, Ordering::SeqCst);
    let requests = (0..4)
        .map(|_| engine.generate(image_request(), GenerateContext::new(mock_context(), None)));
    let (results, ()) = tokio::join!(join_all(requests), async {
        wait_for_calls(&fixture, 2).await;
        fixture.release_discovery();
    });
    for result in results {
        let error = result.err().expect("discovery must fail for every waiter");
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::CannotConnect)
        );
    }
    assert_eq!(fixture.discovery.calls.load(Ordering::SeqCst), 2);
    // Failures are retained for one second; recovery is not permanently latched.
    fixture.discovery.unavailable.store(false, Ordering::SeqCst);
    advertise_images(&fixture);
    tokio::time::sleep(Duration::from_millis(1100)).await;
    for _ in 0..2 {
        let ctx = mock_context();
        control.request(ctx.id(), RequestPlan::default());
        fixture.respond(ctx.id(), vec![finished()]);
        let outputs = collect(&engine, image_request(), GenerateContext::new(ctx, None)).await;
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].as_ref().unwrap().finish_reason,
            Some(FinishReason::Stop)
        );
    }
    assert_eq!(fixture.discovery.calls.load(Ordering::SeqCst), 3);
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn stalled_discovery_observes_request_and_shutdown_cancellation() {
    for shutdown in [false, true] {
        let control = Controller::<Adapter>::default();
        let mut fixture = Fixture::start(control, FixtureConfig::default()).await;
        fixture.discovery.unavailable.store(true, Ordering::SeqCst);
        let engine = fixture.engine_with_context_length(32768);
        engine.start(0).await.unwrap();
        fixture.discovery.hold.store(true, Ordering::SeqCst);
        let ctx = mock_context();
        let cancelling_ctx = ctx.clone();
        let (outputs, ()) = bounded("cancel stalled discovery", async {
            tokio::join!(
                collect(&engine, image_request(), GenerateContext::new(ctx, None)),
                async {
                    wait_for_calls(&fixture, 2).await;
                    if shutdown {
                        engine.cleanup().await.unwrap();
                    } else {
                        cancelling_ctx.stop_generating();
                    }
                }
            )
        })
        .await;
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].as_ref().unwrap().finish_reason,
            Some(FinishReason::Cancelled)
        );
        fixture.release_discovery();
        engine.cleanup().await.unwrap();
        fixture.shutdown().await;
    }
}

#[tokio::test]
async fn unresponsive_discovery_has_one_bounded_timeout_for_all_waiters() {
    let control = Controller::<Adapter>::default();
    let mut fixture = Fixture::start(control, FixtureConfig::default()).await;
    fixture.discovery.unavailable.store(true, Ordering::SeqCst);
    let engine = fixture.engine_with_context_length(32768);
    engine.start(0).await.unwrap();
    fixture.discovery.hold.store(true, Ordering::SeqCst);
    let requests = (0..4)
        .map(|_| engine.generate(image_request(), GenerateContext::new(mock_context(), None)));
    let results = tokio::time::timeout(Duration::from_secs(35), join_all(requests))
        .await
        .expect("all four requests must share the 30-second RPC timeout");
    for result in results {
        let error = result
            .err()
            .expect("discovery must time out for every waiter");
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::ConnectionTimeout)
        );
    }
    assert_eq!(fixture.discovery.calls.load(Ordering::SeqCst), 2);
    fixture.release_discovery();
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
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
