// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn cancelled_encoder_handoff_never_starts_independent_prefill() {
    let binding = local_prefill_binding("cancelled-encoder").await;
    tokio::time::pause();
    for stop_before_routing in [true, false] {
        for kill in [false, true] {
            let router = conditional_router();
            router.set_test_binding(Some(binding.clone()));
            let tracker = Arc::new(RequestTracker::new());
            let barrier = tracker.set_phase(RequestPhase::Prefill).await;
            let mut request = inference_request();
            request.encoder_result = Some(serde_json::json!({"opaque": "handoff"}));
            request.tracker = Some(tracker);
            let mut request = SingleIn::new(request);
            request.insert(
                ENCODER_RESULT_RELEASE_SUPPORTED,
                EncoderResultReleaseSupported(true),
            );
            let context = request.context();
            let host = Arc::new(CountingDecodeHost::default());
            let cancel = || {
                if kill {
                    context.kill()
                } else {
                    context.stop_generating()
                }
            };
            if stop_before_routing {
                cancel();
            }
            let task = tokio::spawn({
                let router = router.clone();
                let host = host.clone();
                async move { router.generate(request, host).await }
            });
            if !stop_before_routing {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                cancel();
            }
            drop(barrier);
            let error = task.await.unwrap().unwrap_err();
            assert!(
                match_error_chain(error.as_ref(), &[ErrorType::Cancelled], &[]),
                "{error}"
            );
            assert_eq!(host.requests.load(Ordering::Relaxed), 0);
        }
    }
}

#[tokio::test]
async fn conditional_bypass_retains_text_releasable_and_staged_requests() {
    let fixture = ConditionalFixture::new("encoder-bypass-controls").await;
    fixture
        .router
        .set_test_binding(Some(fixture.binding(false)));
    fixture
        .router
        .set_target(Some(WorkerSetTarget::Legacy(fixture.endpoint.clone())));
    let downstream = Arc::new(QueryOnlyDecodeHost::default());
    for (has_encoder, release_supported, staged_kv) in [
        (false, false, false),
        (true, true, false),
        (true, false, true),
    ] {
        let mut request = SingleIn::new(inference_request());
        request.encoder_result = has_encoder.then(|| serde_json::json!({"opaque":"handoff"}));
        request.staged_kv_cleanup = staged_kv;
        request.insert(
            ENCODER_RESULT_RELEASE_SUPPORTED,
            EncoderResultReleaseSupported(release_supported),
        );
        let mut response = fixture
            .router
            .generate(request, downstream.clone())
            .await
            .unwrap();
        assert_eq!(
            response.next().await.unwrap().data.unwrap().token_ids,
            vec![7]
        );
        assert!(response.next().await.is_none());
        assert_eq!(
            fixture.dispatch.phases.lock().last(),
            Some(&RequestPhase::Decode)
        );
        assert!(
            fixture
                .dispatch
                .annotations
                .lock()
                .last()
                .unwrap()
                .iter()
                .any(|value| value == BYPASS_REMOTE_PREFILL_ANNOTATION)
        );
    }
    assert_eq!(downstream.requests.load(Ordering::Relaxed), 0);
    fixture.runtime.shutdown();
}

#[tokio::test]
async fn nonreleasable_handoff_survives_late_client_cancellation() {
    for paired_cancellation in [false, true] {
        let fixture = ConditionalFixture::with_cancellation(
            &format!("nonreleasable-encoder-{paired_cancellation}"),
            paired_cancellation,
        )
        .await;
        let downstream = Arc::new(QueryOnlyDecodeHost::default());
        fixture
            .router
            .set_test_binding(Some(fixture.binding(false)));
        fixture
            .router
            .set_target(Some(WorkerSetTarget::Legacy(fixture.endpoint.clone())));
        let gate = Arc::new(tokio::sync::Notify::new());
        *fixture.dispatch.gate.lock() = Some(gate.clone());
        for kill in [false, true] {
            let session = SessionAffinityId::new(format!("nonreleasable-encoder-{kill}"));
            let leader = fixture.affinity.acquire(&session, None).await.unwrap();
            let mut request = SingleIn::new(inference_request());
            request.encoder_result = Some(serde_json::json!({"opaque": "must-consume"}));
            request.insert(SESSION_AFFINITY_CONTEXT_KEY, session);
            let context = request.context();
            let before = fixture.dispatch.phases.lock().len();
            let mut task = tokio::spawn({
                let router = fixture.router.clone();
                let downstream = downstream.clone();
                async move { router.generate(request, downstream).await }
            });
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                fixture.affinity.wait_for_initializing_waiter(),
            )
            .await
            .unwrap();
            if kill {
                context.kill();
            } else {
                context.stop_generating();
            }
            // Keep admission blocked after stop so cancellation cannot race with its completion.
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(100), &mut task)
                    .await
                    .is_err(),
                "a stopped caller must not abandon an undisposable encoder result"
            );
            assert_eq!(fixture.dispatch.phases.lock().len(), before);
            drop(leader);
            let consumer = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    if let Some(context) = fixture.dispatch.contexts.lock().get(before).cloned() {
                        break context;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            // Paired KV cancellation cannot dispose a separate producer's allocation.
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(100), consumer.stopped())
                    .await
                    .is_err()
            );
            gate.notify_one();
            let mut response = tokio::time::timeout(std::time::Duration::from_secs(5), task)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let output = response.next().await.unwrap().data.unwrap();
            assert_eq!(output.token_ids, vec![7]);
            assert_eq!(
                fixture.dispatch.phases.lock().last(),
                Some(&RequestPhase::Prefill)
            );
            assert!(response.next().await.is_none());
        }
        fixture.runtime.shutdown();
    }
}
