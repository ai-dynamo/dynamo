// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn conditional_disagg_rejects_unknown_decode_target_before_fallback() {
    use crate::{kv_router::KvRouter, local_model::runtime_config::ModelRuntimeConfig};
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("conditional-invalid-worker")
        .unwrap()
        .component("decode")
        .unwrap()
        .endpoint("generate");
    let client = endpoint.client().await.unwrap();
    let (_workers_tx, workers) = watch::channel(HashMap::<u64, ModelRuntimeConfig>::new());
    let config = KvRouterConfig {
        skip_initial_worker_wait: true,
        use_kv_events: false,
        router_track_active_blocks: false,
        ..Default::default()
    };
    let chooser = KvRouter::new(
        endpoint.clone(),
        client.clone(),
        workers,
        None,
        16,
        SelectionPolicySource::Registry,
        Some(config),
        None,
        "decode",
        None,
        false,
        None,
        None,
    )
    .await
    .unwrap();
    let host = Arc::new(
        RoutingHost::builder(client, RouterMode::KV)
            .kv_router(Arc::new(chooser))
            .build()
            .await
            .unwrap(),
    );
    let router = conditional_router();
    router.set_test_binding(Some(Arc::new(PrefillBinding {
        target_id: WorkerSetTargetId::Legacy(endpoint.id()),
        endpoint_id: endpoint.id(),
        router: host.clone(),
        prefill_router_mode: RouterMode::KV,
        encoder_result_requires_prefill: false,
    })));
    assert!(router.decode_routing_host.set(host).is_ok());
    let downstream = Arc::new(QueryOnlyDecodeHost::default());
    let mut request = query_only_request();
    request.annotations.clear();
    request.routing_mut().decode_worker_id = Some(u64::MAX);

    let error = router
        .generate(SingleIn::new(request), downstream.clone())
        .await
        .unwrap_err();
    assert!(match_error_chain(
        error.as_ref(),
        &[ErrorType::InvalidArgument],
        &[]
    ));
    assert_eq!(downstream.requests.load(Ordering::Relaxed), 0);
    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn conditional_disagg_query_only_forwards_to_decode_host() {
    let router = conditional_router();
    let decode_host = Arc::new(QueryOnlyDecodeHost::default());
    let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
        decode_host.clone();

    let mut response = router
        .generate(SingleIn::new(query_only_request()), next)
        .await
        .expect("query-only request should bypass conditional planning");
    let output = response
        .next()
        .await
        .expect("decode host should return routing data")
        .data
        .expect("query-only response should contain data");

    assert_eq!(decode_host.requests.load(Ordering::Relaxed), 1);
    assert_eq!(
        output.routing_data.and_then(|routing| routing.token_ids),
        Some(vec![1, 2, 3])
    );
}

#[tokio::test]
async fn bypass_marker_frame_precedes_engine_output_when_requested() {
    let mut stream = with_bypass_marker(bypassed_stream(), true).unwrap();
    let marker = stream.next().await.expect("marker frame");
    assert_eq!(
        marker.event.as_deref(),
        Some(BYPASS_REMOTE_PREFILL_ANNOTATION)
    );
    assert!(marker.data.is_none());
    assert_eq!(marker.comment, Some(vec!["true".to_string()]));
    let output = stream.next().await.expect("engine output");
    assert_eq!(output.data.map(|d| d.token_ids), Some(vec![7]));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn generate_forwards_bypass_marker_only_to_opted_in_clients() {
    let ConditionalFixture {
        runtime,
        distributed,
        endpoint,
        workers: _workers_tx,
        router,
        host,
        dispatch,
        affinity,
        classifications,
        aborts,
        completions,
        events: _events,
    } = ConditionalFixture::new("conditional-bypass-marker").await;
    let binding = |requires_prefill| {
        let mut card = crate::model_card::ModelDeploymentCard::with_name_only("optional-prefill");
        card.runtime_config.runtime_data.insert(
            crate::protocols::common::preprocessor::ENCODER_RESULT_REQUIRES_PREFILL.into(),
            serde_json::json!(requires_prefill),
        );
        assert!(!ModelManager::requires_encoder_result(&card));
        Arc::new(PrefillBinding {
            target_id: WorkerSetTargetId::Legacy(endpoint.id()),
            endpoint_id: endpoint.id(),
            router: host.clone(),
            prefill_router_mode: RouterMode::KV,
            encoder_result_requires_prefill: ModelManager::encoder_result_requires_prefill(&card),
        })
    };
    router.set_test_binding(Some(binding(false)));
    router.set_target(Some(WorkerSetTarget::Legacy(endpoint.clone())));
    let downstream = Arc::new(QueryOnlyDecodeHost::default());

    for (opted_in, has_encoder) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut request = query_only_request();
        request.encoder_result = has_encoder.then(|| serde_json::json!({"opaque": "optional"}));
        request.annotations = if opted_in {
            vec![BYPASS_REMOTE_PREFILL_ANNOTATION.to_string()]
        } else {
            Vec::new()
        };
        let mut request = SingleIn::new(request);
        request.insert(
            ENCODER_RESULT_RELEASE_SUPPORTED,
            EncoderResultReleaseSupported(true),
        );
        let mut stream = router
            .generate(request, downstream.clone())
            .await
            .expect("conditional disagg should bypass to the decode host");
        let first = stream.next().await.expect("first frame");
        if opted_in {
            assert_eq!(
                first.event.as_deref(),
                Some(BYPASS_REMOTE_PREFILL_ANNOTATION)
            );
            assert!(first.data.is_none());
            let output = stream.next().await.expect("engine output");
            assert_eq!(output.data.map(|d| d.token_ids), Some(vec![7]));
        } else {
            assert!(
                first.event.is_none(),
                "default client must not get the marker"
            );
            assert_eq!(first.data.map(|d| d.token_ids), Some(vec![7]));
        }
        assert!(stream.next().await.is_none());
    }

    // The decode hop saw the router-owned marker exactly once per request:
    // the client-supplied copy was stripped and the router's copy added.
    let seen = dispatch.annotations.lock().clone();
    assert_eq!(seen.len(), 4);
    for annotations in &seen {
        assert_eq!(
            annotations
                .iter()
                .filter(|annotation| *annotation == BYPASS_REMOTE_PREFILL_ANNOTATION)
                .count(),
            1
        );
    }
    assert_eq!(downstream.requests.load(Ordering::Relaxed), 0);
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while completions.load(Ordering::Relaxed) != 4 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let loads = host
                .kv_router()
                .get_potential_loads(&[], None, None, None, None)
                .await
                .unwrap();
            if loads.iter().all(|load| load.active_requests == 0) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    // Hold the real generate path after admission, then remove its prefill
    // binding. Falling back must release the decode booking without ending
    // or registering the logical classifier lifecycle a second time.
    router.set_test_binding(Some(binding(false)));
    let baseline_calls = classifications.load(Ordering::Relaxed);
    let baseline_aborts = aborts.load(Ordering::Relaxed);
    let tracker = Arc::new(RequestTracker::new());
    let barrier = tracker.set_phase(RequestPhase::Aggregated).await;
    let mut request = inference_request();
    request.encoder_result = Some(serde_json::json!({"opaque": "replan"}));
    request.tracker = Some(tracker);
    let fallback = Arc::new(CaptureHandoffHost {
        host: host.clone(),
        request: Mutex::new(None),
    });
    let task = tokio::spawn({
        let router = router.clone();
        let fallback = fallback.clone();
        async move {
            let mut request = SingleIn::new(request);
            request.insert(
                ENCODER_RESULT_RELEASE_SUPPORTED,
                EncoderResultReleaseSupported(true),
            );
            router.generate(request, fallback).await
        }
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let loads = host
                .kv_router()
                .get_potential_loads(&[], None, None, None, None)
                .await
                .unwrap();
            if loads.iter().any(|load| load.active_requests == 1) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    router.set_target(None);
    assert!(router.binding().is_some());
    drop(barrier);
    let mut stream = task.await.unwrap().unwrap();
    while stream.next().await.is_some() {}
    let loads = host
        .kv_router()
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert!(
        loads.iter().all(|load| load.active_requests == 0),
        "abandoned decode booking leaked: {loads:?}"
    );
    assert_eq!(
        fallback.request.lock().take(),
        Some((RequestPhase::Aggregated, false))
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while completions.load(Ordering::Relaxed) != 5 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(classifications.load(Ordering::Relaxed), baseline_calls + 1);
    assert_eq!(aborts.load(Ordering::Relaxed), baseline_aborts);
    router.set_test_binding(Some(binding(false)));
    router.set_target(Some(WorkerSetTarget::Legacy(endpoint.clone())));

    let replacement = distributed
        .namespace("conditional-bypass-marker")
        .unwrap()
        .component("replacement")
        .unwrap()
        .endpoint("generate");
    // Change cancellation or topology while affinity admission is held.
    for action in ["stop", "kill", "remove", "replace"] {
        router.set_test_binding(Some(binding(true)));
        router.set_target(Some(WorkerSetTarget::Legacy(endpoint.clone())));
        let session = SessionAffinityId::new("encoder-held-prefill-admission");
        let leader = affinity.acquire(&session, None).await.unwrap();
        let mut request = SingleIn::new(inference_request());
        request.encoder_result = Some(serde_json::json!({"opaque": "held-admission"}));
        request.insert(
            ENCODER_RESULT_RELEASE_SUPPORTED,
            EncoderResultReleaseSupported(true),
        );
        request.insert(SESSION_AFFINITY_CONTEXT_KEY, session);
        let context = request.context();
        let dispatches_before = dispatch.annotations.lock().len();
        let task = tokio::spawn({
            let router = router.clone();
            let downstream = downstream.clone();
            async move { router.generate(request, downstream).await }
        });
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            affinity.wait_for_initializing_waiter(),
        )
        .await
        .unwrap();
        match action {
            "stop" => context.stop_generating(),
            "kill" => context.kill(),
            "remove" => router.set_target(None),
            "replace" => router.set_target(Some(WorkerSetTarget::Legacy(replacement.clone()))),
            _ => unreachable!(),
        }
        drop(leader);
        let error = tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        let expected = if matches!(action, "stop" | "kill") {
            ErrorType::Cancelled
        } else {
            ErrorType::WorkerUnavailable
        };
        assert!(
            match_error_chain(error.as_ref(), &[expected], &[]),
            "{action}: {error}"
        );
        assert_eq!(dispatch.annotations.lock().len(), dispatches_before);
        let loads = host
            .kv_router()
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert!(
            loads.iter().all(|load| load.active_requests == 0),
            "cancelled prefill booking leaked: {loads:?}"
        );
    }
    router.set_test_binding(Some(binding(false)));
    router.set_target(Some(WorkerSetTarget::Legacy(endpoint.clone())));
    for required in [true, false] {
        router.set_test_binding(Some(binding(required)));
        for has_encoder in [false, true] {
            let mut request = inference_request();
            request.encoder_result = has_encoder.then(|| serde_json::json!({"opaque": "handoff"}));
            let decision = router
                .plan_conditional_disagg_decode(&SingleIn::new(request), "handoff")
                .await
                .unwrap();
            assert_eq!(
                decision.is_some(),
                !required || !has_encoder,
                "handoffs follow the prefill consumption policy; text can bypass"
            );
        }
    }
    router.set_test_binding(Some(binding(false)));
    let mut request = inference_request();
    request.encoder_result = Some(serde_json::json!({"opaque": "handoff"}));
    let decision = router
        .plan_conditional_disagg_decode(&SingleIn::new(request), "succession")
        .await
        .unwrap()
        .unwrap();
    assert!(decision.allows_encoder_bypass(&router));
    router.set_test_binding(Some(binding(true)));
    assert!(
        !decision.allows_encoder_bypass(&router),
        "a held decision must not bypass a successor prefill binding"
    );
    drop(decision);
    router.set_test_binding(Some(binding(false)));
    let mut request = inference_request();
    request.encoder_result = Some(serde_json::json!({"opaque": "handoff"}));
    let decision = router
        .plan_conditional_disagg_decode(&SingleIn::new(request), "removal")
        .await
        .unwrap()
        .unwrap();
    router.set_target(None);
    assert!(
        !decision.allows_encoder_bypass(&router),
        "a removed prefill target must invalidate held handoff decisions"
    );
    drop(decision);

    router.set_test_binding(Some(binding(true)));
    router.set_target(Some(WorkerSetTarget::Legacy(endpoint.clone())));
    for has_encoder in [true, false] {
        let mut request = inference_request();
        request.encoder_result = has_encoder.then(|| serde_json::json!({"opaque": "prefill-only"}));
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let mut stream = router
                .generate(SingleIn::new(request), downstream.clone())
                .await
                .unwrap();
            while stream.next().await.is_some() {}
        })
        .await
        .unwrap();
        assert_eq!(
            dispatch.phases.lock().last(),
            Some(&if has_encoder {
                RequestPhase::Prefill
            } else {
                RequestPhase::Decode
            })
        );
    }
    drop(router);
    drop(host);
    runtime.shutdown();
}
