// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `Control` service: model info, health, abort, and the unimplemented RPCs.

use super::*;

/// The context length is the one field the sidecar needs to derive a default
/// `max_tokens`, and the real server answers for any model name -- it loads one
/// model and reports it whatever it is asked about -- so the mocker must not
/// reject a name the engine would have served.
#[tokio::test]
async fn model_info_reports_a_positive_context_length_for_any_model_name() {
    let service = service();
    for model in ["", "mocker-model", "some-other-model"] {
        let info = service
            .get_model_info(Request::new(pb::GetModelInfoRequest {
                model: model.to_string(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            info.max_context_length,
            Some(1_024),
            "asked about {model:?}"
        );
        assert_eq!(info.model_id, "mocker-model");
    }
}

#[tokio::test]
async fn abort_reports_aborted_then_already_finished() {
    let service = TrtllmMockerService::new(
        config(),
        MockEngineArgsBuilder::default()
            .engine_type(EngineType::Trtllm)
            .num_gpu_blocks(4_096usize)
            .block_size(4usize)
            .speedup_ratio(0.01)
            .build()
            .unwrap(),
    )
    .unwrap();
    let mut stream = service
        .generate(Request::new(request("req-abort", 512)))
        .await
        .unwrap()
        .into_inner();
    let _first = stream.next().await.unwrap().unwrap();

    let abort = |target| {
        let service = service.clone();
        async move {
            service
                .abort(Request::new(pb::AbortRequest {
                    target: Some(target),
                }))
                .await
                .unwrap()
                .into_inner()
                .status
        }
    };

    let status = abort(pb::abort_request::Target::RequestId("req-abort".into())).await;
    assert_eq!(status, pb::AbortStatus::Aborted as i32);

    // Drain what is left; the request must still end with one terminal event.
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let Some(pb::generate_response::Event::Finished(finished)) = item.unwrap().event {
            terminal = Some(finished);
        }
    }
    assert_eq!(
        terminal
            .expect("aborted request must still terminate")
            .reason,
        pb::FinishReason::Cancelled as i32
    );

    let status = abort(pb::abort_request::Target::RequestId("req-abort".into())).await;
    assert_eq!(status, pb::AbortStatus::AlreadyFinished as i32);
    let status = abort(pb::abort_request::Target::RequestId("never-existed".into())).await;
    assert_eq!(status, pb::AbortStatus::AlreadyFinished as i32);

    let error = service
        .abort(Request::new(pb::AbortRequest { target: None }))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::InvalidArgument);
}

/// The real TensorRT-LLM server leaves these unimplemented. A mocker that
/// answered them would let a KV-routing test pass here and fail in production.
#[tokio::test]
async fn kv_event_rpcs_are_unimplemented() {
    let service = service();
    let error = service
        .get_kv_event_sources(Request::new(pb::GetKvEventSourcesRequest::default()))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::Unimplemented);
    match service
        .subscribe_kv_events(Request::new(pb::SubscribeKvEventsRequest::default()))
        .await
    {
        Ok(_) => panic!("SubscribeKvEvents must be unimplemented"),
        Err(error) => assert_eq!(error.code(), Code::Unimplemented),
    }
}

#[tokio::test]
async fn lora_rpcs_are_unimplemented() {
    let service = service();
    assert_eq!(
        service
            .list_loras(Request::new(pb::ListLorasRequest::default()))
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
}

#[tokio::test]
async fn health_is_ready_but_the_inference_probe_is_not_simulated() {
    let service = service();
    let health = service
        .health(Request::new(pb::HealthRequest::default()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(health.state, pb::HealthState::Ready as i32);
    assert_eq!(health.checks.len(), 3);

    let error = service
        .health(Request::new(pb::HealthRequest {
            include_inference_probe: true,
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::Unimplemented);
}

/// The engine finishing is not the same as the client being told. The pump
/// makes that window wide -- the engine can run to completion while the
/// consumer has read nothing -- and an abort landing inside it is honoured,
/// because no terminal event has reached the client yet. What must never
/// happen is the two disagreeing.
#[tokio::test]
async fn abort_before_the_terminal_reaches_the_client_cancels_it() {
    let service = service();
    let mut stream = service
        .generate(Request::new(request("req-raced", 4)))
        .await
        .unwrap()
        .into_inner();

    // Let the (instant) engine run ahead of this consumer.
    while service.active_request_count() > 0 {
        tokio::task::yield_now().await;
    }

    let status = service
        .abort(Request::new(pb::AbortRequest {
            target: Some(pb::abort_request::Target::RequestId("req-raced".into())),
        }))
        .await
        .unwrap()
        .into_inner()
        .status;
    assert_eq!(status, pb::AbortStatus::Aborted as i32);

    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let Some(pb::generate_response::Event::Finished(finished)) = item.unwrap().event {
            terminal = Some(finished);
        }
    }
    assert_eq!(
        terminal.expect("the request still terminates").reason,
        pb::FinishReason::Cancelled as i32,
        "the terminal must match the ABORTED the caller was given"
    );
}

/// The abort reply and the terminal event must agree even when the stream is
/// being polled concurrently, which is how tonic always drives it.
/// `LiveEngine::cancel` closes the response channel before it returns, so a
/// transition recorded after the cancel would arrive too late.
#[tokio::test]
async fn concurrent_abort_and_terminal_event_agree() {
    let service = TrtllmMockerService::new(
        config(),
        MockEngineArgsBuilder::default()
            .engine_type(EngineType::Trtllm)
            .num_gpu_blocks(4_096usize)
            .block_size(4usize)
            .speedup_ratio(0.01)
            .build()
            .unwrap(),
    )
    .unwrap();
    let mut stream = service
        .generate(Request::new(request("req-concurrent", 512)))
        .await
        .unwrap()
        .into_inner();

    // Drain concurrently with the abort, rather than holding the stream idle.
    let drain = tokio::spawn(async move {
        let mut terminal = None;
        let mut failed = None;
        while let Some(item) = stream.next().await {
            match item.unwrap().event {
                Some(pb::generate_response::Event::Finished(finished)) => {
                    terminal = Some(finished.reason)
                }
                Some(pb::generate_response::Event::Error(error)) => failed = Some(error),
                _ => {}
            }
        }
        (terminal, failed)
    });

    let status = service
        .abort(Request::new(pb::AbortRequest {
            target: Some(pb::abort_request::Target::RequestId(
                "req-concurrent".into(),
            )),
        }))
        .await
        .unwrap()
        .into_inner()
        .status;

    let (terminal, failed) = drain.await.unwrap();
    assert!(
        failed.is_none(),
        "an aborted request must not report an engine error: {failed:?}"
    );
    if status == pb::AbortStatus::Aborted as i32 {
        assert_eq!(
            terminal,
            Some(pb::FinishReason::Cancelled as i32),
            "ABORTED must be matched by a CANCELLED terminal"
        );
    } else {
        assert_eq!(
            terminal,
            Some(pb::FinishReason::Length as i32),
            "ALREADY_FINISHED must be matched by the real terminal"
        );
    }
}

/// `LiveEngine::cancel` cannot stop a request the scheduler has not seen yet,
/// and the window between registering a request and submitting it is real. The
/// stream honours the claim itself, so an abort that lands there stops
/// generation instead of reporting ABORTED while the request streams its whole
/// budget.
#[tokio::test]
async fn an_abort_that_beats_submission_still_stops_generation() {
    let mut service = service();
    let gate = service.gate_submissions();

    let streaming = service.clone();
    let stream = tokio::spawn(async move {
        streaming
            .generate(Request::new(request("req-early", 512)))
            .await
            .unwrap()
            .into_inner()
    });

    // Wait until the request is registered but still held before submission.
    // The scheduler's own count stays zero throughout: that is the point.
    while service.registered_request_count() == 0 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        service.active_request_count(),
        0,
        "the scheduler must not have seen this request yet"
    );
    let status = service
        .abort(Request::new(pb::AbortRequest {
            target: Some(pb::abort_request::Target::RequestId("req-early".into())),
        }))
        .await
        .unwrap()
        .into_inner()
        .status;
    assert_eq!(
        status,
        pb::AbortStatus::Aborted as i32,
        "a live request reports ABORTED even though the scheduler has not seen it"
    );
    gate.notify_waiters();

    let mut stream = stream.await.unwrap();
    let mut tokens = 0usize;
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item.unwrap().event {
            Some(pb::generate_response::Event::Token(_)) => tokens += 1,
            Some(pb::generate_response::Event::Finished(finished)) => terminal = Some(finished),
            _ => {}
        }
    }
    assert_eq!(
        terminal.expect("the request still terminates").reason,
        pb::FinishReason::Cancelled as i32
    );
    assert_eq!(
        tokens, 0,
        "an abort the scheduler never saw must still stop generation"
    );
}
