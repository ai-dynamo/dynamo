// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::protocols::common::preprocessor::{MigrationState, MultimodalData};
use dynamo_kv_router::plugins::request_classifier::ClassifyEvent;

async fn wait_for_booking(fixture: &ConditionalFixture, count: usize) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let loads = fixture
                .host
                .kv_router()
                .get_potential_loads(&[], None, None, None, None)
                .await
                .unwrap();
            if loads.iter().map(|load| load.active_requests).sum::<usize>() == count {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

async fn wait_for_terminal(fixture: &ConditionalFixture) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while fixture.aborts.load(Ordering::Relaxed) + fixture.completions.load(Ordering::Relaxed)
            != 1
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

// Replace the binding while generate holds an admitted conditional Decode plan.
// The new handoff policy vetoes bypass, so the real Prefill path must resume it.
async fn start_replan(
    fixture: &ConditionalFixture,
    mut request: PreprocessedRequest,
) -> tokio::task::JoinHandle<Result<ManyOut<Annotated<LLMEngineOutput>>, Error>> {
    wait_for_booking(fixture, 0).await;
    fixture
        .router
        .set_test_binding(Some(fixture.binding(false)));
    fixture
        .router
        .set_target(Some(WorkerSetTarget::Legacy(fixture.endpoint.clone())));
    let tracker = Arc::new(RequestTracker::new());
    let barrier = tracker.set_phase(RequestPhase::Aggregated).await;
    request.tracker = Some(tracker);
    if request.multi_modal_data.is_none() {
        request.encoder_result = Some(serde_json::json!({"opaque": "replan"}));
    }
    let task = tokio::spawn({
        let router = fixture.router.clone();
        let host = fixture.host.clone();
        async move {
            let mut request = SingleIn::new(request);
            request.insert(
                ENCODER_RESULT_RELEASE_SUPPORTED,
                EncoderResultReleaseSupported(true),
            );
            router.generate(request, host).await
        }
    });
    wait_for_booking(fixture, 1).await;
    fixture.router.set_test_binding(Some(fixture.binding(true)));
    drop(barrier);
    task
}

async fn replan(
    fixture: &ConditionalFixture,
    request: PreprocessedRequest,
) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
    start_replan(fixture, request).await.await.unwrap()
}

#[tokio::test]
async fn raw_media_encode_fallback_keeps_prefill_but_text_can_bypass() {
    let fixture = ConditionalFixture::new("raw-media-prefill-fallback").await;
    fixture.router.set_test_binding(Some(fixture.binding(true)));
    fixture
        .router
        .set_target(Some(WorkerSetTarget::Legacy(fixture.endpoint.clone())));
    let mut raw_media = inference_request();
    raw_media.multi_modal_data = Some(HashMap::from([(
        "image".into(),
        vec![MultimodalData::RawUrl(
            "https://example.com/image.png".into(),
        )],
    )]));
    assert!(raw_media.encoder_result.is_none());
    let mut response = fixture
        .router
        .generate(SingleIn::new(raw_media.clone()), fixture.host.clone())
        .await
        .unwrap();
    while response.next().await.is_some() {}
    assert_eq!(*fixture.dispatch.phases.lock(), vec![RequestPhase::Prefill]);
    assert!(
        !fixture.dispatch.annotations.lock()[0]
            .iter()
            .any(|a| a == BYPASS_REMOTE_PREFILL_ANNOTATION)
    );

    let mut response = fixture
        .router
        .generate(SingleIn::new(inference_request()), fixture.host.clone())
        .await
        .unwrap();
    while response.next().await.is_some() {}
    assert_eq!(
        *fixture.dispatch.phases.lock(),
        vec![RequestPhase::Prefill, RequestPhase::Decode]
    );
    assert!(
        fixture.dispatch.annotations.lock()[1]
            .iter()
            .any(|a| a == BYPASS_REMOTE_PREFILL_ANNOTATION)
    );
    let mut response = replan(&fixture, raw_media).await.unwrap();
    while response.next().await.is_some() {}
    assert_eq!(
        *fixture.dispatch.phases.lock(),
        vec![
            RequestPhase::Prefill,
            RequestPhase::Decode,
            RequestPhase::Prefill
        ],
        "raw-media plans must recheck a replacement binding after admission"
    );
    fixture.runtime.shutdown();
}

#[tokio::test]
async fn replan_prefill_only_success_completes_classification() {
    for retries in [false, true] {
        let fixture = ConditionalFixture::new(&format!("replan-prefill-complete-{retries}")).await;
        let mut request = inference_request();
        let migration = retries.then(MigrationState::default);
        request.migration_state = migration.clone();
        let mut response = replan(&fixture, request).await.unwrap();
        assert_eq!(
            response.next().await.unwrap().data.unwrap().token_ids,
            vec![7]
        );
        assert!(response.next().await.is_none());
        wait_for_terminal(&fixture).await;
        assert_eq!(fixture.classifications.load(Ordering::Relaxed), 1);
        assert_eq!(fixture.completions.load(Ordering::Relaxed), 1);
        assert_eq!(fixture.aborts.load(Ordering::Relaxed), 0);
        assert_eq!(*fixture.dispatch.phases.lock(), vec![RequestPhase::Prefill]);
        assert!(
            migration
                .as_ref()
                .is_none_or(|state| state.take_request_lifecycle().is_none())
        );
        fixture.runtime.shutdown();
    }
}

#[tokio::test]
async fn replan_prefill_terminal_failure_aborts_and_preserves_output() {
    for (index, (finish_reason, retries, cause_type)) in [
        (
            FinishReason::Error("prefill engine failure".into()),
            false,
            ErrorType::Internal,
        ),
        (
            FinishReason::Error("prefill engine failure".into()),
            true,
            ErrorType::Internal,
        ),
        (FinishReason::Cancelled, false, ErrorType::Cancelled),
        (FinishReason::Cancelled, true, ErrorType::Cancelled),
    ]
    .into_iter()
    .enumerate()
    {
        let fixture =
            ConditionalFixture::new(&format!("replan-prefill-terminal-failure-{index}")).await;
        *fixture.dispatch.prefill_output.lock() = Some(LLMEngineOutput {
            token_ids: vec![7],
            finish_reason: Some(finish_reason.clone()),
            ..Default::default()
        });
        let mut request = inference_request();
        let migration = retries.then(MigrationState::default);
        request.migration_state = migration.clone();
        let mut response = replan(&fixture, request).await.unwrap();
        let output = response.next().await.unwrap().data.unwrap();
        assert_eq!(output.finish_reason, Some(finish_reason));
        assert_eq!(output.token_ids, vec![7]);
        assert!(response.next().await.is_none());
        wait_for_terminal(&fixture).await;
        assert_eq!(fixture.classifications.load(Ordering::Relaxed), 1);
        assert_eq!(fixture.aborts.load(Ordering::Relaxed), 1);
        assert_eq!(fixture.completions.load(Ordering::Relaxed), 0);
        assert_eq!(*fixture.dispatch.phases.lock(), vec![RequestPhase::Prefill]);
        assert!(
            migration
                .as_ref()
                .is_none_or(|state| state.take_request_lifecycle().is_none())
        );
        let events = fixture.events.lock();
        let cause = events
            .iter()
            .find_map(|event| match event {
                ClassifyEvent::Aborted { error, .. } => error.as_ref(),
                _ => None,
            })
            .expect("failed terminal must retain a cause");
        assert!(
            match_error_chain(cause.as_ref(), &[cause_type], &[]),
            "{cause:#}"
        );
        if cause_type == ErrorType::Internal {
            assert!(cause.to_string().contains("prefill engine failure"));
        }
        drop(events);
        fixture.runtime.shutdown();
    }
}

#[tokio::test]
async fn replan_failures_preserve_cause_and_outer_retry_ownership() {
    for (index, (phase, retries, stream_failure)) in [
        (RequestPhase::Prefill, false, false),
        (RequestPhase::Prefill, true, false),
        (RequestPhase::Decode, false, false),
        (RequestPhase::Decode, true, false),
        (RequestPhase::Decode, false, true),
        (RequestPhase::Decode, true, true),
    ]
    .into_iter()
    .enumerate()
    {
        let fixture = ConditionalFixture::new(&format!("replan-typed-failure-{index}")).await;
        if stream_failure {
            *fixture.dispatch.fail_stream_phase.lock() = Some(phase);
        } else {
            *fixture.dispatch.fail_phase.lock() = Some(phase);
        }
        *fixture.dispatch.prefill_output.lock() = Some(LLMEngineOutput {
            token_ids: vec![7],
            disaggregated_params: Some(Box::new(
                serde_json::json!({"ctx_request_id": "prefill-request"}),
            )),
            ..Default::default()
        });
        let mut request = inference_request();
        let migration = retries.then(MigrationState::default);
        request.migration_state = migration.clone();
        let error = match replan(&fixture, request).await {
            Err(error) => error,
            Ok(mut response) => {
                use dynamo_runtime::protocols::maybe_error::MaybeError;
                response
                    .next()
                    .await
                    .expect("Decode must return its stream failure")
                    .err()
                    .expect("expected the typed Decode error")
                    .into()
            }
        };
        assert!(
            match_error_chain(error.as_ref(), &[ErrorType::WorkerUnavailable], &[]),
            "{error:#}"
        );
        if let Some(state) = migration {
            assert_eq!(
                fixture.aborts.load(Ordering::Relaxed),
                0,
                "outer Migration owns retry settlement"
            );
            let mut lifecycle = state
                .take_request_lifecycle()
                .expect("retry must retain the original classification");
            lifecycle.abort(Some(
                crate::protocols::common::preprocessor::owned_abort_error(error.as_ref()),
            ));
        }
        wait_for_terminal(&fixture).await;
        assert_eq!(fixture.classifications.load(Ordering::Relaxed), 1);
        assert_eq!(fixture.aborts.load(Ordering::Relaxed), 1);
        assert_eq!(fixture.completions.load(Ordering::Relaxed), 0);
        let events = fixture.events.lock();
        let cause = events
            .iter()
            .find_map(|event| match event {
                ClassifyEvent::Aborted { error, .. } => error.as_ref(),
                _ => None,
            })
            .expect("aborted classification must retain the actual dispatch cause");
        assert!(
            match_error_chain(cause.as_ref(), &[ErrorType::WorkerUnavailable], &[]),
            "{cause:#}"
        );
        assert!(
            cause
                .to_string()
                .contains("injected replan dispatch failure")
        );
        drop(events);
        fixture.runtime.shutdown();
    }
}

#[tokio::test]
async fn dropping_replan_while_prefill_is_pending_aborts_once() {
    let fixture = ConditionalFixture::new("replan-cancelled-prefill").await;
    let gate = Arc::new(tokio::sync::Notify::new());
    *fixture.dispatch.gate.lock() = Some(gate);
    let task = start_replan(&fixture, inference_request()).await;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while fixture.dispatch.phases.lock().as_slice() != [RequestPhase::Prefill] {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    wait_for_terminal(&fixture).await;
    assert_eq!(fixture.classifications.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.aborts.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.completions.load(Ordering::Relaxed), 0);
    fixture.runtime.shutdown();
}
