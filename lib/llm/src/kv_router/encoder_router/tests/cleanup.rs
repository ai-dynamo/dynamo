// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn downstream_completion_or_abandonment_releases_only_unconsumed_handoffs() {
    if isolated(concat!(
        module_path!(),
        "::downstream_completion_or_abandonment_releases_only_unconsumed_handoffs"
    ))
    .await
    {
        return;
    }
    let mut fixture = Fixture::new().await;
    fixture.admit_all().await;
    let router = fixture.router.clone();
    let release_gate = fixture.release_gate.clone();
    let released = &mut fixture.released;
    for behavior in [
        DownstreamBehavior::SetupError,
        DownstreamBehavior::TypedError,
        DownstreamBehavior::Terminal,
        DownstreamBehavior::ErrorTerminal,
        DownstreamBehavior::CancelledTerminal,
        DownstreamBehavior::Token,
        DownstreamBehavior::Empty,
        DownstreamBehavior::Pending,
    ] {
        let downstream = Arc::new(CaptureEngine {
            behavior,
            ..Default::default()
        });
        let mut original = multimodal_request();
        original.sampling_options.n = Some(3);
        original.extra_args = Some(
            json!({"sampling_constraints": "generation-only", "block_release":
                !matches!(behavior, DownstreamBehavior::SetupError | DownstreamBehavior::TypedError | DownstreamBehavior::ErrorTerminal | DownstreamBehavior::CancelledTerminal)
            }),
        );
        let request = SingleIn::new(original);
        let request_id = request.id().to_string();
        let request_context = request.context();
        let result = router.generate(request, downstream.clone()).await;
        if matches!(behavior, DownstreamBehavior::SetupError) {
            assert!(result.is_err());
        } else {
            let mut response = result.unwrap();
            if matches!(
                behavior,
                DownstreamBehavior::TypedError
                    | DownstreamBehavior::ErrorTerminal
                    | DownstreamBehavior::CancelledTerminal
            ) {
                while tokio::time::timeout(Duration::from_secs(5), response.next())
                    .await
                    .unwrap()
                    .is_some()
                {}
            } else if !matches!(behavior, DownstreamBehavior::Pending) {
                while response
                    .next()
                    .now_or_never()
                    .expect("producer cleanup must not block downstream output")
                    .is_some()
                {}
            } else {
                let producer = downstream
                    .request
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .encoder_result
                    .as_ref()
                    .unwrap()["worker_id"]
                    .as_u64()
                    .unwrap();
                let binding = router.binding.load_full().unwrap();
                binding.router.client.set_overloaded_instances(&[producer]);
                let refusal = binding
                    .router
                    .direct(SingleIn::new(multimodal_request()), producer)
                    .await
                    .expect_err("ordinary inference must obey overload admission");
                assert_eq!(
                    refusal.downcast_ref::<DynamoError>().unwrap().error_type(),
                    dynamo_runtime::error::ErrorType::WorkerOverloaded
                );
            }
            if matches!(behavior, DownstreamBehavior::Pending) {
                request_context.stop_generating();
            }
            drop(response);
        }
        if matches!(
            behavior,
            DownstreamBehavior::Token | DownstreamBehavior::Terminal
        ) {
            assert!(
                tokio::time::timeout(Duration::from_millis(500), async {
                    loop {
                        let (_, release) = released.recv().await.unwrap();
                        if release
                            .encoder_result
                            .as_ref()
                            .is_some_and(|payload| payload["request_id"] == request_id)
                        {
                            break;
                        }
                    }
                })
                .await
                .is_err(),
                "a consumed handoff must not send producer release"
            );
            continue;
        }
        let expected = downstream
            .request
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .encoder_result
            .clone()
            .unwrap();
        let (producer, cleanup) = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let cleanup = released.recv().await.unwrap();
                if cleanup
                    .1
                    .encoder_result
                    .as_ref()
                    .is_some_and(|payload| payload["request_id"] == request_id)
                {
                    break cleanup;
                }
            }
        })
        .await
        .expect("every completed or abandoned handoff must release its producer");
        if expected["block_release"] == true {
            release_gate.add_permits(1);
        }
        assert_eq!(producer, expected["worker_id"].as_u64().unwrap());
        assert_eq!(cleanup.encoder_result, Some(expected));
        assert!(cleanup.token_ids.is_empty());
        assert!(cleanup.multi_modal_data.is_none());
        assert!(cleanup.prefill_result.is_none());
        assert!(cleanup.routing.is_none());
        assert!(cleanup.sampling_options.n.is_none());
        assert_eq!(
            cleanup.extra_args,
            Some(json!({RELEASE_ENCODER_RESULT: true}))
        );
    }

    drop(router);
    fixture.shutdown().await;
}

#[tokio::test]
async fn downstream_error_waits_for_producer_disposal() {
    if isolated(concat!(
        module_path!(),
        "::downstream_error_waits_for_producer_disposal"
    ))
    .await
    {
        return;
    }
    let mut fixture = Fixture::new().await;
    fixture.admit_all().await;
    let downstream = Arc::new(CaptureEngine {
        behavior: DownstreamBehavior::TypedError,
        ..Default::default()
    });
    let mut request = multimodal_request();
    request.extra_args = Some(json!({"block_release": true}));
    let request = SingleIn::new(request);
    let request_id = request.id().to_string();
    let mut response = fixture.router.generate(request, downstream).await.unwrap();
    let poll = tokio::spawn(async move { response.next().await });
    let (_, release) = fixture.release_for(&request_id).await;
    assert!(release.encoder_result.is_some());
    assert!(
        !poll.is_finished(),
        "retry observed the error before release completed"
    );
    fixture.release_gate.add_permits(1);
    let error = tokio::time::timeout(Duration::from_secs(5), poll)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(error.err().is_some());
    fixture.shutdown().await;
}

#[tokio::test]
async fn cancelling_pending_disposal_retries_the_original_producer() {
    if isolated(concat!(
        module_path!(),
        "::cancelling_pending_disposal_retries_the_original_producer"
    ))
    .await
    {
        return;
    }
    let mut fixture = Fixture::new().await;
    fixture.admit_all().await;
    let downstream = Arc::new(CaptureEngine {
        behavior: DownstreamBehavior::TypedError,
        ..Default::default()
    });
    let mut request = multimodal_request();
    request.extra_args = Some(json!({"block_release": true}));
    let request = SingleIn::new(request);
    let request_id = request.id().to_string();
    let mut response = fixture.router.generate(request, downstream).await.unwrap();
    let poll = tokio::spawn(async move { response.next().await });
    let first = fixture.release_for(&request_id).await;
    assert_eq!(
        first.0,
        first.1.encoder_result.as_ref().unwrap()["worker_id"]
            .as_u64()
            .unwrap()
    );
    assert!(!poll.is_finished());
    poll.abort();
    assert!(poll.await.unwrap_err().is_cancelled());
    let second = fixture.release_for(&request_id).await;
    assert_eq!(
        second.0, first.0,
        "cleanup changed the producer after cancellation"
    );
    assert_eq!(second.1.encoder_result, first.1.encoder_result);
    fixture.release_gate.add_permits(2);
    fixture.shutdown().await;
}

async fn finalization_failure_waits_for_disposal(required: bool) {
    let mut fixture = Fixture::new().await;
    fixture.router.set_required(required);
    let downstream = Arc::new(CaptureEngine::default());
    let mut request = multimodal_request();
    request.extra_args = Some(json!({"block_release": true, "truncate_after_terminal": true}));
    let request = SingleIn::new(request);
    let request_id = request.id().to_string();
    let mut task = tokio::spawn({
        let router = fixture.router.clone();
        let downstream = downstream.clone();
        async move { router.generate(request, downstream).await }
    });
    let (producer, release) = fixture.release_for(&request_id).await;
    assert_eq!(producer, fixture.ids[0]);
    assert_eq!(
        release.encoder_result.as_ref().unwrap()["worker_id"],
        producer
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(100), &mut task)
            .await
            .is_err(),
        "truncated Encode stream escaped before disposing its export"
    );
    assert!(
        downstream.request.lock().unwrap().is_none(),
        "inline fallback started before export disposal"
    );
    fixture.release_gate.add_permits(1);
    let result = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap();
    if required {
        let error = result.unwrap_err();
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[dynamo_runtime::error::ErrorType::Disconnected],
            &[]
        ));
        assert!(crate::migration::is_migratable(error.as_ref()));
        assert!(downstream.request.lock().unwrap().is_none());
    } else {
        drop(result.unwrap());
        let captured = downstream.request.lock().unwrap();
        let captured = captured.as_ref().unwrap();
        assert!(captured.encoder_result.is_none());
        assert!(captured.multi_modal_data.is_some());
    }
    fixture.shutdown().await;
}

#[tokio::test]
async fn required_encode_finalization_failure_waits_for_disposal() {
    if isolated(concat!(
        module_path!(),
        "::required_encode_finalization_failure_waits_for_disposal"
    ))
    .await
    {
        return;
    }
    finalization_failure_waits_for_disposal(true).await;
}

#[tokio::test]
async fn optional_encode_finalization_failure_waits_for_disposal() {
    if isolated(concat!(
        module_path!(),
        "::optional_encode_finalization_failure_waits_for_disposal"
    ))
    .await
    {
        return;
    }
    finalization_failure_waits_for_disposal(false).await;
}

#[tokio::test]
async fn release_support_is_negotiated_per_handoff() {
    if isolated(concat!(
        module_path!(),
        "::release_support_is_negotiated_per_handoff"
    ))
    .await
    {
        return;
    }
    let fixture = Fixture::new().await;
    let binding = fixture.router.binding.load_full().unwrap();
    let (route, response) = binding
        .router
        .select_and_dispatch_with_cleanup(SingleIn::new(multimodal_request()))
        .await
        .unwrap();
    let _: Vec<_> = response.collect().await;
    let mut cleanup = EncoderResultCleanup {
        route,
        request: None,
    };
    let opaque = json!({"arbitrary": [1, {"engine-owned": "value"}]});
    let mut terminal = LLMEngineOutput::encode_terminal(opaque.as_object().unwrap().clone());
    cleanup.observe("model", &terminal).unwrap();
    assert!(
        cleanup.request.is_none(),
        "ordinary backends must not receive release requests"
    );
    terminal.extra_args = Some(Box::new(json!({ENCODER_RESULT_RELEASE_SUPPORTED: false})));
    cleanup.observe("model", &terminal).unwrap();
    assert!(cleanup.request.is_none());
    terminal.extra_args = Some(Box::new(json!({ENCODER_RESULT_RELEASE_SUPPORTED: true})));
    cleanup.observe("model", &terminal).unwrap();
    assert_eq!(cleanup.request.take().unwrap().encoder_result, Some(opaque));
    fixture.shutdown().await;
}

#[tokio::test]
async fn encode_failure_uses_current_required_policy_after_disposal() {
    if isolated(concat!(
        module_path!(),
        "::encode_failure_uses_current_required_policy_after_disposal"
    ))
    .await
    {
        return;
    }
    for required in [true, false] {
        let mut fixture = Fixture::new().await;
        fixture.router.set_required(!required);
        let downstream = Arc::new(CaptureEngine::default());
        let mut request = multimodal_request();
        request.extra_args = Some(json!({"block_release": true, "truncate_after_terminal": true}));
        let request = SingleIn::new(request);
        let request_id = request.id().to_string();
        let task = tokio::spawn({
            let router = fixture.router.clone();
            let downstream = downstream.clone();
            async move { router.generate(request, downstream).await }
        });
        fixture.release_for(&request_id).await;
        assert!(!task.is_finished());
        fixture.router.set_required(required);
        fixture.release_gate.add_permits(1);
        let result = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap();
        if required {
            let error = result.unwrap_err();
            assert!(dynamo_runtime::error::match_error_chain(
                error.as_ref(),
                &[dynamo_runtime::error::ErrorType::Disconnected],
                &[]
            ));
            assert!(downstream.request.lock().unwrap().is_none());
        } else {
            drop(result.unwrap());
            let captured = downstream.request.lock().unwrap().take().unwrap();
            assert!(captured.encoder_result.is_none());
            assert!(captured.multi_modal_data.is_some());
        }
        fixture.shutdown().await;
    }
}

#[tokio::test]
async fn pending_disposal_retries_retired_producer_without_reselection() {
    if isolated(concat!(
        module_path!(),
        "::pending_disposal_retries_retired_producer_without_reselection"
    ))
    .await
    {
        return;
    }
    let mut fixture = Fixture::new().await;
    let downstream = Arc::new(CaptureEngine {
        behavior: DownstreamBehavior::TypedError,
        ..Default::default()
    });
    let mut request = multimodal_request();
    request.extra_args = Some(json!({"block_release": true}));
    let request = SingleIn::new(request);
    let request_id = request.id().to_string();
    let mut response = fixture.router.generate(request, downstream).await.unwrap();
    let poll = tokio::spawn(async move { response.next().await });
    let first = fixture.release_for(&request_id).await;
    assert_eq!(first.0, fixture.ids[0]);
    fixture.retire_producer().await;
    poll.abort();
    assert!(poll.await.unwrap_err().is_cancelled());
    let second = fixture.release_for(&request_id).await;
    assert_eq!(
        second.0, first.0,
        "cleanup must not target the admitted replacement"
    );
    assert_eq!(second.1.encoder_result, first.1.encoder_result);
    fixture.release_gate.add_permits(2);
    fixture.shutdown().await;
}
