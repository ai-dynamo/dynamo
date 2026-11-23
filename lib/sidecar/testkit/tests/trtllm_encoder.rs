// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenEngine's unary encoder contract and encoder-only node lifecycle.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

use dynamo_backend_common::testing::mock_context;
use dynamo_backend_common::{
    BackendError, DisaggregationMode, ErrorType, FinishReason, GenerateContext, LLMEngine,
    MultimodalData, PreprocessedRequest,
};
use dynamo_sidecar_testkit::bounded;
use dynamo_sidecar_testkit::control::{Controller, Event, OpenAction, RequestPlan};
use dynamo_sidecar_testkit::fixtures::{collect, request};
use dynamo_trtllm_sidecar::TrtllmSidecarEngine;
use dynamo_trtllm_sidecar::proto as pb;
use futures::StreamExt;
use serde_json::json;

#[allow(dead_code)]
#[path = "support/process.rs"]
mod process;
#[allow(dead_code)]
mod support;
use process::Environment;
use support::trtllm::{Adapter, Fixture};
use support::{FixtureConfig, ProcessFixture, SidecarFixture};

fn image_request() -> PreprocessedRequest {
    let mut request = request("mocker-model", vec![11, 22, 33], 1);
    request.multi_modal_data = Some(HashMap::from([(
        "image_url".into(),
        vec![MultimodalData::RawUrl(
            "https://example.com/image.png".into(),
        )],
    )]));
    request
}

fn reference() -> pb::EncodedMediaRef {
    pb::EncodedMediaRef {
        opaque_state: vec![1, 2, 3],
    }
}

fn release_request() -> PreprocessedRequest {
    let mut request = request("mocker-model", vec![], 1);
    request.encoder_result = Some(json!({"opaque_state": "AQID"}));
    request.extra_args = Some(json!({"release_encoder_result": true}));
    request
}

async fn fixture(mode: DisaggregationMode) -> (Fixture, Controller<Adapter>) {
    let control = Controller::default();
    let fixture = Fixture::start(
        control.clone(),
        FixtureConfig {
            disaggregation_mode: mode,
            ..Default::default()
        },
    )
    .await;
    *fixture.discovery.model_info.lock().unwrap() = Some(pb::ModelInfo {
        model_id: "mocker-model".into(),
        served_model_name: "mocker-model".into(),
        max_context_length: Some(32768),
        supports_multimodal: Some(true),
        supports_text_input: Some(true),
        supports_token_ids_input: Some(true),
        extra: Some(
            dynamo_sidecar_common::json_to_struct(
                json!({"trtllm_supported_modalities": ["image"]}),
                "test capabilities",
            )
            .unwrap(),
        ),
        ..Default::default()
    });
    (fixture, control)
}

fn server_info(encoder: bool) -> pb::ServerInfo {
    pb::ServerInfo {
        engine_role: if encoder {
            pb::EngineRole::Encoder
        } else {
            pb::EngineRole::Aggregated
        } as i32,
        capacity: Some(pb::DeploymentCapacity {
            kv_block_size: Some(32),
            total_kv_blocks: Some(1024),
            ..Default::default()
        }),
        schema_release: include_str!("../../trtllm/proto/REVISION").trim().into(),
        ..Default::default()
    }
}

fn node_info(leader: bool) -> pb::ServerInfo {
    let mut info = server_info(true);
    info.parallelism = Some(pb::ParallelismInfo {
        data_parallel_size: Some(2),
        ..Default::default()
    });
    info.extra = Some(
        dynamo_sidecar_common::json_to_struct(
            json!({"trtllm_node": {
                "version": 1, "engine_id": "encoder-engine",
                "node_id": if leader { "leader" } else { "follower" },
                "leader": leader, "node_count": 2, "kv_block_size": 0,
                "local_dp_ranks": [], "source_owners": {}
            }}),
            "test node",
        )
        .unwrap(),
    );
    info
}

fn restart(fixture: &Fixture) {
    let mut value = dynamo_sidecar_common::struct_to_json(
        fixture
            .discovery
            .server_info
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .extra
            .clone()
            .unwrap(),
        "test",
        "node",
    )
    .unwrap();
    value["trtllm_node"]["engine_id"] = json!("restarted");
    fixture
        .discovery
        .server_info
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .extra = Some(dynamo_sidecar_common::json_to_struct(value, "test node").unwrap());
}

#[tokio::test]
async fn encode_uses_unary_rpc_and_consumer_forwards_opaque_reference() {
    let (mut encoder_fixture, generate) = fixture(DisaggregationMode::Encode).await;
    let encoder = encoder_fixture.engine().await;
    encoder.start(0).await.unwrap();
    let outputs = collect(
        &encoder,
        image_request(),
        GenerateContext::new(mock_context(), None),
    )
    .await;
    assert_eq!(outputs.len(), 1);
    let output = outputs[0].as_ref().unwrap();
    assert!(output.token_ids.is_empty());
    assert_eq!(encoder_fixture.encoder.encode.received_count(), 1);
    assert_eq!(generate.received_count(), 0);
    let (mut consumer_fixture, generate) = fixture(DisaggregationMode::Aggregated).await;
    *consumer_fixture.discovery.server_info.lock().unwrap() = Some(server_info(false));
    let consumer = consumer_fixture.engine_with_args(&["--route-to-encoder"]);
    let config = consumer.start(0).await.unwrap();
    assert_eq!(
        config.runtime_data.get("requires_encoder_result"),
        Some(&json!(true))
    );
    let context = mock_context();
    let handle = generate.request(context.id(), RequestPlan::default());
    let mut input = image_request();
    input.encoder_result = output.encoder_result.as_deref().cloned();
    consumer_fixture.respond(
        context.id(),
        vec![pb::GenerateResponse {
            event: Some(pb::generate_response::Event::Finished(
                pb::GenerationFinished {
                    output_index: Some(0),
                    reason: pb::FinishReason::Stop as i32,
                    ..Default::default()
                },
            )),
            ..Default::default()
        }],
    );
    let outputs = collect(&consumer, input, GenerateContext::new(context, None)).await;
    assert!(outputs.iter().all(Result::is_ok), "{outputs:?}");
    assert_eq!(
        handle.native_request().unwrap().encoded_media,
        Some(reference())
    );
    consumer.cleanup().await.unwrap();
    encoder.cleanup().await.unwrap();
    consumer_fixture.shutdown().await;
    encoder_fixture.shutdown().await;
}

#[tokio::test]
async fn encoder_rejects_wrong_schema_or_role_before_dispatch() {
    for (role, release, expected) in [
        (
            true,
            "old-generation-contract",
            "pinned OpenEngine schema release",
        ),
        (
            false,
            include_str!("../../trtllm/proto/REVISION").trim(),
            "sidecar role",
        ),
    ] {
        let (mut fixture, _) = fixture(DisaggregationMode::Encode).await;
        let mut info = server_info(role);
        info.schema_release = release.into();
        *fixture.discovery.server_info.lock().unwrap() = Some(info);
        let error = fixture.engine().await.start(0).await.unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(fixture.encoder.encode.received_count(), 0);
        fixture.shutdown().await;
    }
}

#[tokio::test]
async fn encoder_registration_rejects_identity_changed_after_bootstrap() {
    for changed in ["instance", "role", "schema"] {
        let (mut fixture, _) = fixture(DisaggregationMode::Encode).await;
        let mut initial = server_info(true);
        initial.instance_id = "encoder-a".into();
        let mut replacement = initial.clone();
        match changed {
            "instance" => replacement.instance_id = "encoder-b".into(),
            "role" => replacement.engine_role = pb::EngineRole::Aggregated as i32,
            "schema" => replacement.schema_release = "different-release".into(),
            _ => unreachable!(),
        }
        fixture
            .discovery
            .server_info_sequence
            .lock()
            .unwrap()
            .extend([initial, replacement]);
        let encoder = fixture.engine().await;
        let error = encoder
            .start(0)
            .await
            .expect_err("changed engine must not register");
        assert!(error.to_string().contains("identity changed"), "{error}");
        encoder.cleanup().await.unwrap();
        fixture.shutdown().await;
    }
}

#[tokio::test]
async fn encoder_identity_discovery_uses_remaining_startup_budget() {
    let (mut fixture, _) = fixture(DisaggregationMode::Encode).await;
    fixture
        .discovery
        .server_info_hold
        .store(true, Ordering::SeqCst);
    let encoder = TrtllmSidecarEngine::from_args(vec![
        "dynamo-trtllm-sidecar".into(),
        "--grpc-endpoint".into(),
        fixture.endpoint(),
        "--model-path".into(),
        "mocker-model".into(),
        "--disaggregation-mode".into(),
        "encode".into(),
        "--grpc-startup-deadline-secs".into(),
        "1".into(),
    ])
    .unwrap()
    .0;
    let error = tokio::time::timeout(Duration::from_secs(2), encoder.start(0))
        .await
        .expect("identity RPC exceeded startup budget")
        .expect_err("stalled identity must not register");
    assert!(error.to_string().contains("startup deadline"), "{error}");
    encoder.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn reference_release_retries_transient_failure_at_same_encoder() {
    let (mut fixture, generate) = fixture(DisaggregationMode::Encode).await;
    let encoder = fixture.engine().await;
    encoder.start(0).await.unwrap();
    let first = fixture.encoder.release.next_request(RequestPlan {
        open: OpenAction::Fail,
        ..Default::default()
    });
    let retry = fixture.encoder.release.next_request(RequestPlan::default());
    let output = collect(
        &encoder,
        release_request(),
        GenerateContext::new(mock_context(), None),
    )
    .await;
    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0].as_ref().unwrap().finish_reason,
        Some(FinishReason::Stop)
    );
    for handle in [&first, &retry] {
        bounded("release completed", handle.wait(Event::Dropped)).await;
        assert_eq!(
            handle.native_request().unwrap().native.encoded_media,
            Some(reference())
        );
    }
    assert_eq!(fixture.encoder.release.received_count(), 2);
    assert_eq!(generate.received_count(), 0);
    encoder.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn reference_only_release_never_generates_or_aborts() {
    let (mut fixture, generate) = fixture(DisaggregationMode::Encode).await;
    let encoder = fixture.engine().await;
    encoder.start(0).await.unwrap();
    let release = fixture.encoder.release.next_request(RequestPlan::default());
    let outputs = collect(
        &encoder,
        release_request(),
        GenerateContext::new(mock_context(), None),
    )
    .await;
    assert_eq!(outputs.len(), 1);
    let output = outputs[0].as_ref().unwrap();
    assert_eq!(output.finish_reason, Some(FinishReason::Stop));
    assert!(output.token_ids.is_empty());
    bounded("release handler dropped", release.wait(Event::Dropped)).await;
    assert_eq!(
        release.native_request().unwrap().native.encoded_media,
        Some(reference())
    );
    assert_eq!(fixture.encoder.encode.received_count(), 0);
    assert_eq!(generate.received_count(), 0);
    assert_eq!(fixture.encoder.aborts.lock().unwrap().len(), 0);
    encoder.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn pending_encode_emits_one_cancelled_terminal_on_stop_kill_and_shutdown() {
    bounded("pending Encode cancellation", async {
        for cancellation in ["stop", "kill", "shutdown"] {
            let (mut fixture, generate) = fixture(DisaggregationMode::Encode).await;
            let encoder = fixture.engine().await;
            encoder.start(0).await.unwrap();
            let context = mock_context();
            let handle = fixture.encoder.encode.request(
                context.id(),
                RequestPlan {
                    open: OpenAction::Hold,
                    ..Default::default()
                },
            );
            let stream = encoder
                .generate(image_request(), GenerateContext::new(context.clone(), None))
                .await
                .unwrap();
            let outputs = tokio::spawn(async move { stream.collect::<Vec<_>>().await });
            handle.wait(Event::Received).await;
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
            handle.wait(Event::Dropped).await;
            assert_eq!(fixture.encoder.encode.received_count(), 1);
            assert_eq!(generate.received_count(), 0);
            assert_eq!(fixture.encoder.release.received_count(), 0);
            if cancellation != "shutdown" {
                encoder.cleanup().await.unwrap();
            }
            fixture.shutdown().await;
        }
    })
    .await;
}

#[tokio::test]
async fn malformed_release_is_rejected_before_any_native_work() {
    let (mut fixture, generate) = fixture(DisaggregationMode::Encode).await;
    let encoder = fixture.engine().await;
    encoder.start(0).await.unwrap();
    let release = release_request();
    let mut with_tokens = release.clone();
    with_tokens.token_ids = vec![1].into();
    let mut with_media = release.clone();
    with_media.multi_modal_data = image_request().multi_modal_data;
    let mut without_ref = release.clone();
    without_ref.encoder_result = None;
    let mut invalid_ref = release;
    invalid_ref.encoder_result = Some(json!({"unexpected": "not a reference"}));
    for input in [with_tokens, with_media, without_ref, invalid_ref] {
        let error = encoder
            .generate(input, GenerateContext::new(mock_context(), None))
            .await
            .err()
            .expect("invalid release accepted");
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }
    assert_eq!(fixture.encoder.encode.received_count(), 0);
    assert_eq!(fixture.encoder.release.received_count(), 0);
    assert_eq!(generate.received_count(), 0);
    encoder.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn cancelled_terminal_releases_unobserved_reference() {
    bounded("unobserved reference cleanup", async {
        for killed in [false, true] {
            let (mut fixture, _) = fixture(DisaggregationMode::Encode).await;
            let encoder = fixture.engine().await;
            encoder.start(0).await.unwrap();
            let release = fixture.encoder.release.next_request(RequestPlan::default());
            let context = mock_context();
            let mut stream = encoder
                .generate(image_request(), GenerateContext::new(context.clone(), None))
                .await
                .unwrap();
            assert!(
                stream
                    .next()
                    .await
                    .unwrap()
                    .unwrap()
                    .encoder_result
                    .is_some()
            );
            if killed {
                context.kill();
            } else {
                context.stop_generating();
            }
            drop(stream);
            release.wait(Event::Dropped).await;
            assert_eq!(
                release.native_request().unwrap().native.encoded_media,
                Some(reference())
            );
            assert_eq!(fixture.encoder.release.received_count(), 1);
            encoder.cleanup().await.unwrap();
            fixture.shutdown().await;
        }
    })
    .await;
}

#[tokio::test]
async fn invalid_encode_response_releases_its_reference() {
    let (mut fixture, _) = fixture(DisaggregationMode::Encode).await;
    *fixture.encoder.response.lock().unwrap() = Some(pb::EncodeResponse {
        request_id: "wrong-request".into(),
        encoded_media: Some(reference()),
    });
    let encoder = fixture.engine().await;
    encoder.start(0).await.unwrap();
    let release = fixture.encoder.release.next_request(RequestPlan::default());
    let mut stream = encoder
        .generate(image_request(), GenerateContext::new(mock_context(), None))
        .await
        .unwrap();
    assert!(
        bounded("invalid Encode", stream.next())
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("different request_id")
    );
    drop(stream);
    bounded("invalid reference cleanup", release.wait(Event::Dropped)).await;
    assert_eq!(
        release.native_request().unwrap().native.encoded_media,
        Some(reference())
    );
    assert_eq!(fixture.encoder.release.received_count(), 1);
    encoder.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn probe_releases_reference_before_success_without_changing_normal_encode() {
    let (mut fixture, _) = fixture(DisaggregationMode::Encode).await;
    let encoder = fixture.engine().await;
    encoder.start(0).await.unwrap();
    let release = fixture.encoder.release.next_request(RequestPlan::default());
    for is_probe in [false, true] {
        let mut input = image_request();
        input.is_probe = is_probe;
        let outputs = collect(&encoder, input, GenerateContext::new(mock_context(), None)).await;
        assert_eq!(outputs.len(), 1);
        let output = outputs[0].as_ref().unwrap();
        assert_eq!(output.finish_reason, Some(FinishReason::Stop));
        assert_eq!(output.encoder_result.is_none(), is_probe);
        assert!(output.token_ids.is_empty());
        assert_eq!(
            fixture.encoder.release.received_count(),
            usize::from(is_probe)
        );
        if is_probe {
            assert!(release.reached(Event::Dropped));
            assert_eq!(
                release.native_request().unwrap().native.encoded_media,
                Some(reference())
            );
        }
    }
    assert_eq!(fixture.encoder.encode.received_count(), 2);
    assert_eq!(fixture.encoder.aborts.lock().unwrap().len(), 0);
    encoder.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn dropped_probe_retries_held_release_and_drops_both_native_handlers() {
    bounded("probe cancellation during release", async {
        let (mut fixture, _) = fixture(DisaggregationMode::Encode).await;
        let encoder = fixture.engine().await;
        encoder.start(0).await.unwrap();
        let first = fixture.encoder.release.next_request(RequestPlan { open: OpenAction::Hold, ..Default::default() });
        let retry = fixture.encoder.release.next_request(RequestPlan::default());
        let mut input = image_request(); input.is_probe = true;
        let context = mock_context();
        let mut stream = encoder.generate(input, GenerateContext::new(context.clone(), None)).await.unwrap();
        let mut next = Box::pin(stream.next());
        tokio::select! { _ = first.wait(Event::Received) => {}, _ = &mut next => panic!("probe succeeded before release") }
        context.stop_generating(); drop(next); drop(stream);
        first.wait(Event::Dropped).await;
        retry.wait(Event::Dropped).await;
        for handle in [&first, &retry] { assert_eq!(handle.native_request().unwrap().native.encoded_media, Some(reference())); }
        assert_eq!(fixture.encoder.release.received_count(), 2);
        encoder.cleanup().await.unwrap();
    fixture.shutdown().await;
    }).await;
}

#[tokio::test]
async fn probe_release_failure_is_not_success_and_retries_cleanup() {
    bounded("failed probe cleanup", async {
        let (mut fixture, _) = fixture(DisaggregationMode::Encode).await;
        let encoder = fixture.engine().await;
        encoder.start(0).await.unwrap();
        let plan = RequestPlan {
            open: OpenAction::Reject,
            ..Default::default()
        };
        let first = fixture.encoder.release.next_request(plan);
        let retry = fixture.encoder.release.next_request(plan);
        let mut input = image_request();
        input.is_probe = true;
        let mut stream = encoder
            .generate(input, GenerateContext::new(mock_context(), None))
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
        drop(stream);
        first.wait(Event::Dropped).await;
        retry.wait(Event::Dropped).await;
        assert_eq!(fixture.encoder.release.received_count(), 2);
        encoder.cleanup().await.unwrap();
        fixture.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn consumer_cancellation_does_not_submit_or_wait_for_generate() {
    bounded("consumer cancellation", async {
        for before_dispatch in [true, false] {
            let (mut fixture, generate) = fixture(DisaggregationMode::Aggregated).await;
            *fixture.discovery.server_info.lock().unwrap() = Some(server_info(false));
            let engine = fixture.engine_with_args(&["--route-to-encoder"]);
            engine.start(0).await.unwrap();
            let context = mock_context();
            let mut input = image_request(); input.encoder_result = Some(json!({"opaque_state": "AQID"}));
            let handle = generate.request(context.id(), RequestPlan { open: OpenAction::Hold, ..Default::default() });
            if before_dispatch { context.stop_generating(); }
            let dispatch = engine.generate(input, GenerateContext::new(context.clone(), None));
            tokio::pin!(dispatch);
            if !before_dispatch {
                tokio::select! { _ = handle.wait(Event::Received) => {}, _ = &mut dispatch => panic!("dispatch finished before headers") }
                context.stop_generating();
            }
            let mut stream = dispatch.await.unwrap();
            assert_eq!(stream.next().await.unwrap().unwrap().finish_reason, Some(FinishReason::Cancelled));
            if before_dispatch { assert_eq!(generate.received_count(), 0); }
            else { handle.wait(Event::Dropped).await; }
            engine.cleanup().await.unwrap();
            fixture.shutdown().await;
        }
    }).await;
}

#[tokio::test]
async fn encoder_abort_uses_native_abort_even_after_stream_ends() {
    let (mut fixture, _) = fixture(DisaggregationMode::Encode).await;
    let encoder = fixture.engine().await;
    encoder.start(0).await.unwrap();
    let context = mock_context();
    let request_id = context.id().to_owned();
    let output = collect(
        &encoder,
        image_request(),
        GenerateContext::new(context.clone(), None),
    )
    .await;
    assert_eq!(output.len(), 1);
    assert!(output[0].as_ref().unwrap().encoder_result.is_some());
    encoder.abort(context).await;
    assert_eq!(
        fixture.encoder.aborts.lock().unwrap().as_slice(),
        [pb::AbortRequest {
            target: Some(pb::abort_request::Target::RequestId(request_id)),
        }]
    );
    encoder.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn encoder_leader_publishes_and_monitors_identity_without_kv_telemetry() {
    tokio::time::timeout(Duration::from_secs(20), async {
        let (mut fixture, _) = fixture(DisaggregationMode::Encode).await;
        let info = node_info(true);
        let node = dynamo_sidecar_common::struct_to_json(info.extra.clone().unwrap(), "test", "node").unwrap()["trtllm_node"].clone();
        *fixture.discovery.server_info.lock().unwrap() = Some(info);
        let encoder = fixture.engine().await;
        let config = encoder.start(0).await.unwrap();
        assert_eq!(config.runtime_data["trtllm_node"], node);
        let registration = config.llm.unwrap();
        assert_eq!(registration.data_parallel_size, Some(2));
        assert_eq!(registration.data_parallel_start_rank, Some(0));
        assert!(registration.kv_cache_block_size.is_none());
        assert!(registration.total_kv_blocks.is_none());
        assert!(encoder.kv_event_sources().await.unwrap().is_empty());
        let before = fixture.discovery.server_info_calls.load(Ordering::SeqCst);
        let monitor = encoder.engine_lost(); tokio::pin!(monitor);
        let polled = async { while fixture.discovery.server_info_calls.load(Ordering::SeqCst) < before + 2 { tokio::time::sleep(Duration::from_millis(10)).await; } };
        tokio::select! { _ = polled => {}, error = &mut monitor => panic!("premature engine loss: {error}") }
        restart(&fixture);
        let error = monitor.await;
        assert!(error.to_string().contains("Local engine changed"), "{error}");
        assert_eq!(fixture.discovery.load_calls.load(Ordering::SeqCst), 0);
        assert_eq!(fixture.discovery.kv_source_calls.load(Ordering::SeqCst), 0);
        encoder.cleanup().await.unwrap();
    fixture.shutdown().await;
    }).await.expect("encoder identity supervision");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn encoder_follower_monitors_identity_without_serving_or_kv_telemetry() {
    tokio::time::timeout(Duration::from_secs(25), async {
        let env = Environment::new().await;
        let (mut fixture, _) = fixture(DisaggregationMode::Encode).await;
        *fixture.discovery.server_info.lock().unwrap() = Some(node_info(false));
        let mut process = env.spawn::<Fixture>(&fixture.endpoint(), DisaggregationMode::Encode, 60);
        while fixture.discovery.server_info_calls.load(Ordering::SeqCst) < 3 {
            assert!(process.is_running(), "{}", process.logs());
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(env.cards().await.is_empty());
        assert!(env.registrations("encode").await.is_empty());
        assert_eq!(fixture.discovery.load_calls.load(Ordering::SeqCst), 0);
        assert_eq!(fixture.discovery.kv_source_calls.load(Ordering::SeqCst), 0);
        restart(&fixture);
        assert!(!process.exit().await.success(), "{}", process.logs());
        assert!(
            process.logs().contains("Local engine changed"),
            "{}",
            process.logs()
        );
        fixture.shutdown().await;
    })
    .await
    .expect("encoder follower identity supervision");
}

#[tokio::test]
async fn encoder_follower_cannot_start_direct_inference() {
    let (mut fixture, generate) = fixture(DisaggregationMode::Encode).await;
    *fixture.discovery.server_info.lock().unwrap() = Some(node_info(false));
    let error = fixture.engine().await.start(0).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Follower endpoint cannot serve inference"),
        "{error}"
    );
    assert_eq!(generate.received_count(), 0);
    assert_eq!(fixture.encoder.encode.received_count(), 0);
    fixture.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consumer_follower_accepts_route_flag_and_cancels_initial_reconnect() {
    bounded("follower reconnect shutdown", async {
        let env = Environment::new().await;
        let (mut fixture, _) = fixture(DisaggregationMode::Aggregated).await;
        let mut info = node_info(false);
        info.engine_role = pb::EngineRole::Aggregated as i32;
        *fixture.discovery.server_info.lock().unwrap() = Some(info);
        fixture
            .discovery
            .server_info_hold_after
            .store(1, Ordering::SeqCst);
        let mut process = env.spawn_with_args::<Fixture>(
            &fixture.endpoint(),
            DisaggregationMode::Aggregated,
            60,
            &["--route-to-encoder"],
        );
        while fixture.discovery.server_info_blocked.load(Ordering::SeqCst) == 0 {
            assert!(process.is_running(), "{}", process.logs());
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(env.cards().await.is_empty());
        process.shutdown().await;
        fixture.shutdown().await;
    })
    .await;
}
