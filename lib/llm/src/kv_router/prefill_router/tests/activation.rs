// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn pending_target_fails_closed_instead_of_forwarding_to_decode() {
    let router = fresh_decode_router();
    mark_target_pending(&router);
    assert!(matches!(
        router.activation_snapshot(),
        PrefillActivation::Pending { .. }
    ));

    let host = Arc::new(CountingDecodeHost::default());
    let error = router
        .generate(SingleIn::new(inference_request()), host.clone())
        .await
        .expect_err("a pending prefill target must not pass through to decode");

    assert!(
        match_error_chain(error.as_ref(), &[ErrorType::WorkerUnavailable], &[]),
        "expected WorkerUnavailable so the error maps to a retryable 503, got: {error}"
    );
    assert!(
        crate::migration::is_migratable(error.as_ref()),
        "the retry layer must be able to re-dispatch this request"
    );
    assert_eq!(
        host.requests.load(Ordering::Relaxed),
        0,
        "decode must not see a request without its prefill hop"
    );
}

#[tokio::test]
async fn untargeted_router_still_passes_through() {
    let router = fresh_decode_router();
    assert!(matches!(
        router.activation_snapshot(),
        PrefillActivation::Untargeted { .. }
    ));

    let host = Arc::new(CountingDecodeHost::default());
    router
        .generate(SingleIn::new(inference_request()), host.clone())
        .await
        .expect("no target means plain passthrough");
    assert_eq!(host.requests.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn cleared_target_still_passes_through() {
    let router = fresh_decode_router();
    mark_target_pending(&router);
    router.set_target(None);
    assert!(matches!(
        router.activation_snapshot(),
        PrefillActivation::Untargeted { .. }
    ));

    let host = Arc::new(CountingDecodeHost::default());
    router
        .generate(SingleIn::new(inference_request()), host.clone())
        .await
        .expect("a deactivated router passes through");
    assert_eq!(host.requests.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn retired_prefill_enforces_handoff_consumption_policy() {
    use crate::kv_router::{RouterLoadSource, RoutingLoadContext};
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("retired-encoder-prefill")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate");
    let replacement = distributed
        .namespace("retired-encoder-prefill")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("replacement");
    let client = endpoint.client().await.unwrap();
    let load_context = RoutingLoadContext::start(
        client.clone(),
        RouterLoadSource::Prefill,
        crate::discovery::LoadThresholdHandle::new(Default::default()),
        &distributed.child_token(),
        None,
    )
    .await
    .unwrap();
    let prefill_host = Arc::new(
        RoutingHost::builder(client, RouterMode::RoundRobin)
            .load_context(load_context)
            .build()
            .await
            .unwrap(),
    );
    tokio::time::pause();
    for (requires_prefill, handoff, staged, query) in [
        (true, true, false, false),
        (false, true, false, false),
        (true, false, false, false),
        (true, true, true, false),
        (true, true, false, true),
    ] {
        for (retire_after_entry, replace_before_retirement) in
            [(false, false), (true, false), (true, true)]
        {
            let router = fresh_decode_router_with_mode(if query {
                RouterMode::KV
            } else {
                RouterMode::RoundRobin
            });
            let mut card =
                crate::model_card::ModelDeploymentCard::with_name_only("optional-prefill");
            card.runtime_config.runtime_data.insert(
                crate::protocols::common::preprocessor::ENCODER_RESULT_REQUIRES_PREFILL.into(),
                serde_json::json!(requires_prefill),
            );
            assert!(!ModelManager::requires_encoder_result(&card));
            let binding = Arc::new(PrefillBinding {
                target_id: WorkerSetTargetId::Legacy(endpoint.id()),
                endpoint_id: endpoint.id(),
                router: prefill_host.clone(),
                prefill_router_mode: RouterMode::RoundRobin,
                encoder_result_requires_prefill: ModelManager::encoder_result_requires_prefill(
                    &card,
                ),
            });
            *router.target.lock() = Some(binding.target_id.clone());
            router.set_test_binding(Some(binding));
            let mut request = if query {
                query_only_request()
            } else {
                inference_request()
            };
            request.encoder_result = handoff.then(|| serde_json::json!({"opaque": "handoff"}));
            request.staged_kv_cleanup = staged;
            let host = Arc::new(CountingDecodeHost::default());
            let result = if retire_after_entry && !query {
                let tracker = Arc::new(RequestTracker::new());
                let barrier = tracker.set_phase(RequestPhase::Decode).await;
                request.tracker = Some(tracker);
                let in_flight = tokio::spawn({
                    let router = router.clone();
                    let host = host.clone();
                    async move { router.generate(SingleIn::new(request), host).await }
                });
                // Paused time advances after generate parks at the phase barrier.
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                if replace_before_retirement {
                    router.set_target(Some(WorkerSetTarget::Legacy(replacement.clone())));
                }
                router.set_target(None);
                drop(barrier);
                tokio::time::timeout(std::time::Duration::from_secs(5), in_flight)
                    .await
                    .expect("retired prefill must resolve the waiting request")
                    .unwrap()
            } else {
                router.set_target(None);
                router.generate(SingleIn::new(request), host.clone()).await
            };
            assert_eq!(
                router.binding().is_none(),
                replace_before_retirement && !query
            );
            if requires_prefill && handoff && !staged && !query {
                let error = result.unwrap_err();
                assert!(crate::migration::is_migratable(error.as_ref()), "{error}");
                assert!(match_error_chain(
                    error.as_ref(),
                    &[ErrorType::WorkerUnavailable],
                    &[]
                ));
                assert_eq!(host.requests.load(Ordering::Relaxed), 0);
            } else {
                drop(result.unwrap());
                assert_eq!(host.requests.load(Ordering::Relaxed), 1);
            }
        }
    }
    tokio::time::resume();
    runtime.shutdown();
}

#[tokio::test]
async fn retired_committed_target_preserves_policy_before_binding_publication() {
    use crate::discovery::CommittedWorkerSetTarget;
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("unpublished-prefill-policy")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate");
    let router = fresh_decode_router();
    let (_admissions, admitted_ids) = watch::channel(Vec::new());
    for (generation, required) in [(1, true), (2, false), (3, true)] {
        let mut card = crate::model_card::ModelDeploymentCard::with_name_only("prefill");
        card.runtime_config.runtime_data.insert(
            crate::protocols::common::preprocessor::ENCODER_RESULT_REQUIRES_PREFILL.into(),
            serde_json::json!(required),
        );
        router.set_target(Some(WorkerSetTarget::Committed(CommittedWorkerSetTarget {
            endpoint: endpoint.clone(),
            group: "prefill".into(),
            generation,
            card: Arc::new(card),
            admitted_ids: admitted_ids.clone(),
        })));
        assert!(matches!(
            router.activation_snapshot(),
            PrefillActivation::Pending { .. }
        ));
        router.set_target(None);
        let mut request = inference_request();
        request.encoder_result = Some(serde_json::json!({"opaque": "handoff"}));
        let host = Arc::new(CountingDecodeHost::default());
        let result = router.generate(SingleIn::new(request), host.clone()).await;
        if required {
            let error = result.unwrap_err();
            assert!(match_error_chain(
                error.as_ref(),
                &[ErrorType::WorkerUnavailable],
                &[]
            ));
            assert_eq!(host.requests.load(Ordering::Relaxed), 0);
        } else {
            drop(result.unwrap());
            assert_eq!(host.requests.load(Ordering::Relaxed), 1);
        }
    }
    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn instance_annotation_does_not_bypass_prefill_on_executing_routes() {
    for mode in [RouterMode::RoundRobin, RouterMode::Random] {
        let router = fresh_decode_router_with_mode(mode);
        mark_target_pending(&router);
        let host = Arc::new(CountingDecodeHost::default());
        let error = router
            .generate(SingleIn::new(query_only_request()), host.clone())
            .await
            .unwrap_err();
        assert!(match_error_chain(
            error.as_ref(),
            &[ErrorType::WorkerUnavailable],
            &[]
        ));
        assert_eq!(host.requests.load(Ordering::Relaxed), 0);
    }
}

#[tokio::test]
async fn activation_snapshot_does_not_wait_for_target_writer() {
    let router = fresh_decode_router();
    mark_target_pending(&router);
    let writer = router.target.lock();
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn({
        let router = router.clone();
        move || {
            tx.send(matches!(
                router.activation_snapshot(),
                PrefillActivation::Pending { .. }
            ))
            .unwrap()
        }
    });
    let result = rx.recv_timeout(std::time::Duration::from_secs(5));
    drop(writer);
    reader.join().unwrap();
    assert!(
        result.unwrap(),
        "routing reads must use the published snapshot"
    );
}

#[tokio::test]
async fn pending_target_still_serves_query_only_requests() {
    // Routing previews never reach the engine, so they are not at risk and
    // must keep working while the prefill binding is being built.
    let router = fresh_decode_router_with_mode(RouterMode::KV);
    mark_target_pending(&router);

    let host = Arc::new(QueryOnlyDecodeHost::default());
    router
        .generate(SingleIn::new(query_only_request()), host.clone())
        .await
        .expect("query-only requests bypass the activation gate");
    assert_eq!(host.requests.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn target_replaced_mid_flight_fails_closed_at_binding_load() {
    // The lifecycle gate runs once at the top of `generate`; the binding is
    // loaded only after the prefill phase barrier. A WorkerSet succession in
    // between clears the binding and moves the router back to `Pending`.
    // Hold the phase barrier so the request is parked in that gap, replace
    // the target, then release it: the request must fail closed rather than
    // reach the missing-binding passthrough.
    let router = fresh_decode_router();
    router.set_test_binding(Some(local_prefill_binding("phase-succession").await));
    tokio::time::pause();
    let tracker = Arc::new(RequestTracker::new());
    let barrier = tracker.set_phase(RequestPhase::Prefill).await;
    let mut request = inference_request();
    request.tracker = Some(tracker.clone());

    let host = Arc::new(CountingDecodeHost::default());
    let in_flight = tokio::spawn({
        let router = router.clone();
        let host = host.clone();
        async move { router.generate(SingleIn::new(request), host).await }
    });
    // With paused time this sleep completes only once every other task is
    // idle, i.e. once the request has passed the gate and parked on the barrier.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    mark_target_pending(&router);
    drop(barrier);

    let error = match in_flight.await.unwrap() {
        Ok(_) => panic!("a replaced target must not pass through to decode"),
        Err(error) => error,
    };
    assert!(
        match_error_chain(error.as_ref(), &[ErrorType::WorkerUnavailable], &[]),
        "expected WorkerUnavailable after a mid-flight target replacement, got: {error}"
    );
    assert_eq!(host.requests.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn dropping_pending_router_releases_activation_tasks() {
    let (_activation_tx, activation_rx) = tokio::sync::oneshot::channel();
    let router = PrefillRouter::new(
        activation_rx,
        Arc::new(crate::discovery::ModelManager::new()),
        RouterMode::RoundRobin,
        16,
        None,
        None,
        None,
        SessionAffinityMode::Hard,
        "test-model".to_string(),
        "test-namespace".to_string(),
        crate::discovery::LoadThresholdHandle::new(Default::default()),
        CancellationToken::new(),
    );
    let task_state = Arc::downgrade(&router.activation_task_state);
    let weak = Arc::downgrade(&router);

    drop(router);

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while weak.strong_count() != 0 || task_state.strong_count() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("pending activation tasks retained their PrefillRouter");
}
