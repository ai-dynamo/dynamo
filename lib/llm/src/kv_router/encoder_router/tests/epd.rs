// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::protocols::common::preprocessor::{
    ENCODER_RESULT_RELEASE_SUPPORTED, MultimodalData, RELEASE_ENCODER_RESULT,
};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    discovery::EventTransportKind,
    distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
    engine::{AsyncEngine, AsyncEngineContextProvider},
    pipeline::{Error, ResponseStream, network::Ingress},
    storage::kv,
};
use futures::{FutureExt, stream};
use std::sync::atomic::AtomicUsize;
use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
    time::Duration,
};

#[derive(Default)]
struct CaptureEngine {
    request: Mutex<Option<PreprocessedRequest>>,
    behavior: DownstreamBehavior,
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for CaptureEngine
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> std::result::Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        self.request
            .lock()
            .unwrap()
            .replace(request.content().clone());
        let response: futures::stream::BoxStream<'static, Annotated<LLMEngineOutput>> =
            match self.behavior {
                DownstreamBehavior::Empty => Box::pin(stream::empty()),
                DownstreamBehavior::SetupError => {
                    anyhow::bail!("downstream dispatch rejected")
                }
                DownstreamBehavior::TypedError => Box::pin(stream::iter([Annotated::from_error(
                    "downstream stream rejected",
                )])),
                DownstreamBehavior::Terminal => {
                    Box::pin(stream::iter([
                        Annotated::from_data(LLMEngineOutput::stop()),
                    ]))
                }
                DownstreamBehavior::ErrorTerminal | DownstreamBehavior::CancelledTerminal => {
                    Box::pin(stream::iter([Annotated::from_data(LLMEngineOutput {
                        finish_reason: Some(
                            if matches!(self.behavior, DownstreamBehavior::ErrorTerminal) {
                                FinishReason::Error("rejected before consuming handoff".into())
                            } else {
                                FinishReason::Cancelled
                            },
                        ),
                        ..Default::default()
                    })]))
                }
                DownstreamBehavior::Token => Box::pin(stream::iter([
                    Annotated::from_data(LLMEngineOutput {
                        token_ids: vec![42],
                        ..Default::default()
                    }),
                    Annotated::from_data(LLMEngineOutput::stop()),
                ])),
                DownstreamBehavior::Pending => Box::pin(stream::pending()),
            };
        Ok(ResponseStream::new(response, request.context()))
    }
}

fn multimodal_request() -> PreprocessedRequest {
    media_request("image_url")
}

struct EncodeWorker(
    u64,
    tokio::sync::mpsc::UnboundedSender<(u64, PreprocessedRequest)>,
    Arc<tokio::sync::Semaphore>,
    Arc<AtomicUsize>,
);

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for EncodeWorker
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        if request
            .extra_args
            .as_ref()
            .and_then(|args| args.get(RELEASE_ENCODER_RESULT))
            == Some(&json!(true))
        {
            self.1.send((self.0, request.content().clone())).unwrap();
            if request
                .encoder_result
                .as_ref()
                .and_then(|value| value.get("block_release"))
                == Some(&json!(true))
            {
                self.2.acquire().await.unwrap().forget();
            }
            return Ok(ResponseStream::new(
                Box::pin(stream::iter([
                    Annotated::from_data(LLMEngineOutput::stop()),
                ])),
                request.context(),
            ));
        }
        self.3.fetch_add(1, Ordering::SeqCst);
        if matches!(request.sampling_options.n, Some(2 | 4 | 5 | 6)) {
            return Ok(ResponseStream::new(
                Box::pin(stream::iter([Annotated {
                    error: Some(
                        DynamoError::builder()
                            .class(match request.sampling_options.n {
                                Some(4) => ErrorClass::WorkerOverloaded,
                                Some(5) => ErrorClass::WorkerUnavailable,
                                Some(6) => ErrorClass::Cancelled,
                                _ => ErrorClass::InvalidRequest,
                            })
                            .message("encoder rejected request")
                            .build(),
                    ),
                    data: None,
                    id: None,
                    event: Some("error".into()),
                    comment: None,
                }])),
                request.context(),
            ));
        }
        let mut output = LLMEngineOutput::encode_terminal(
            json!({"worker_id": self.0, "request_id": request.id(), "routing": request.routing,
                "block_release": request.extra_args.as_ref().and_then(|extra| extra.get("block_release"))})
                .as_object()
                .unwrap()
                .clone(),
        );
        output.extra_args = Some(Box::new(json!({ENCODER_RESULT_RELEASE_SUPPORTED: true})));
        let mut frames = vec![Annotated::from_data(output)];
        if request
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("truncate_after_terminal"))
            == Some(&json!(true))
        {
            frames.push(Annotated::from_err(
                DynamoError::builder()
                    .error_type(dynamo_runtime::error::ErrorType::Disconnected)
                    .message("stream ended before complete_final")
                    .build(),
            ));
        }
        Ok(ResponseStream::new(
            Box::pin(stream::iter(frames)),
            request.context(),
        ))
    }
}

async fn encoded_worker(router: &EncoderRouter) -> Option<u64> {
    let downstream = Arc::new(CaptureEngine {
        behavior: DownstreamBehavior::Terminal,
        ..Default::default()
    });
    let response = router
        .generate(SingleIn::new(multimodal_request()), downstream.clone())
        .await
        .unwrap();
    let _: Vec<_> = response.collect().await;
    downstream
        .request
        .lock()
        .unwrap()
        .take()
        .unwrap()
        .encoder_result
        .and_then(|result| result["worker_id"].as_u64())
}

#[tokio::test]
async fn query_annotation_only_bypasses_encoder_for_kv_lookup() {
    for (mode, state) in [
        (RouterMode::KV, EncoderLifecycleState::Unavailable),
        (RouterMode::RoundRobin, EncoderLifecycleState::Pending),
        (RouterMode::Random, EncoderLifecycleState::Unavailable),
        (RouterMode::RoundRobin, EncoderLifecycleState::Active),
    ] {
        let mut router = EncoderRouter::disabled();
        Arc::get_mut(&mut router).unwrap().router_mode = mode;
        router.set_required(true);
        router.lifecycle.store(state as u8, Ordering::Release);
        let mut request = media_request("image_url");
        request.annotations.push("query_instance_id:".into());
        let downstream = Arc::new(CaptureEngine::default());
        let result = router
            .generate(SingleIn::new(request), downstream.clone())
            .await;
        if !mode.is_kv_routing() {
            assert!(result.is_err(), "executing routes must require an encoder");
            assert!(downstream.request.lock().unwrap().is_none());
            continue;
        }
        drop(result.unwrap());
        let seen = downstream.request.lock().unwrap();
        assert!(seen.as_ref().unwrap().encoder_result.is_none());
        assert!(
            seen.as_ref()
                .unwrap()
                .multi_modal_data
                .as_ref()
                .unwrap()
                .contains_key("image_url")
        );
    }
}

#[tokio::test]
async fn unavailable_required_encoder_rejects_media_but_passes_text() {
    for state in [
        EncoderLifecycleState::Pending,
        EncoderLifecycleState::Active,
        EncoderLifecycleState::Unavailable,
    ] {
        let router = EncoderRouter::disabled();
        router.set_required(true);
        router.lifecycle.store(state as u8, Ordering::Release);
        let downstream = Arc::new(CaptureEngine::default());
        let error = router
            .generate(SingleIn::new(multimodal_request()), downstream.clone())
            .await
            .unwrap_err();
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[dynamo_runtime::error::ErrorType::WorkerUnavailable],
            &[]
        ));
        assert!(crate::migration::is_migratable(error.as_ref()));
        assert!(downstream.request.lock().unwrap().is_none());
        let mut text = multimodal_request();
        text.multi_modal_data = None;
        drop(
            router
                .generate(SingleIn::new(text), downstream.clone())
                .await
                .unwrap(),
        );
        assert!(downstream.request.lock().unwrap().is_some());
    }
}

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
async fn encode_failure_waits_for_disposal_and_uses_current_policy() {
    if isolated(concat!(
        module_path!(),
        "::encode_failure_waits_for_disposal_and_uses_current_policy"
    ))
    .await
    {
        return;
    }
    for (initially_required, required) in
        [(true, true), (false, false), (false, true), (true, false)]
    {
        let mut fixture = Fixture::new().await;
        fixture.router.set_required(initially_required);
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
            assert!(crate::migration::is_migratable(error.as_ref()));
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
    assert_eq!(
        first.0,
        first.1.encoder_result.as_ref().unwrap()["worker_id"]
            .as_u64()
            .unwrap()
    );
    fixture.retire_producer().await;
    assert!(!poll.is_finished());
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

#[tokio::test]
async fn encoder_errors_remain_typed_and_optional_requests_can_fall_inline() {
    if isolated(concat!(
        module_path!(),
        "::encoder_errors_remain_typed_and_optional_requests_can_fall_inline"
    ))
    .await
    {
        return;
    }
    let fixture = Fixture::new().await;
    let router = fixture.router.clone();
    let ids = fixture.ids.clone();
    router.set_required(true);
    let downstream = Arc::new(CaptureEngine::default());
    let mut invalid = multimodal_request();
    invalid.sampling_options.n = Some(2);
    let error = router
        .generate(SingleIn::new(invalid.clone()), downstream.clone())
        .await
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<DynamoError>().unwrap().class(),
        ErrorClass::InvalidRequest
    );
    assert!(format!("{error:#}").contains("encoder rejected request"));
    assert!(downstream.request.lock().unwrap().is_none());
    for (n, expected, migratable) in [
        (4, dynamo_runtime::error::ErrorType::ResourceExhausted, true),
        (6, dynamo_runtime::error::ErrorType::Cancelled, false),
    ] {
        let mut request = multimodal_request();
        request.sampling_options.n = Some(n);
        let error = router
            .generate(SingleIn::new(request), downstream.clone())
            .await
            .unwrap_err();
        assert!(
            dynamo_runtime::error::match_error_chain(error.as_ref(), &[expected], &[]),
            "n={n}: {error:#}"
        );
        if n == 4 {
            let typed = error.downcast_ref::<DynamoError>().unwrap();
            assert_eq!(typed.class(), ErrorClass::CapacityExhausted);
            assert_eq!(typed.reason().as_str(), "capacity.worker_overloaded");
        }
        assert_eq!(
            crate::migration::is_migratable(error.as_ref()),
            migratable,
            "n={n}: {error:#}"
        );
        assert!(downstream.request.lock().unwrap().is_none());
    }
    let binding = router.binding.load_full().unwrap();
    binding.router.client.set_overloaded_instances(&[ids[0]]);
    let error = router
        .generate(SingleIn::new(multimodal_request()), downstream.clone())
        .await
        .unwrap_err();
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[dynamo_runtime::error::ErrorType::ResourceExhausted],
        &[]
    ));
    assert!(downstream.request.lock().unwrap().is_none());
    binding.router.client.set_overloaded_instances(&[]);
    router.set_required(false);
    let response = router
        .generate(SingleIn::new(invalid), downstream.clone())
        .await
        .unwrap();
    drop(response);
    assert!(
        downstream
            .request
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .encoder_result
            .is_none()
    );

    // A worker-unavailable stream quarantines its producer. Check it last so
    // the real transport's health tracking cannot poison subsequent fixtures.
    router.set_required(true);
    let downstream = Arc::new(CaptureEngine::default());
    let mut request = multimodal_request();
    request.sampling_options.n = Some(5);
    let error = router
        .generate(SingleIn::new(request), downstream.clone())
        .await
        .unwrap_err();
    assert!(
        dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[dynamo_runtime::error::ErrorType::WorkerUnavailable],
            &[]
        ),
        "{error:#}"
    );
    assert!(crate::migration::is_migratable(error.as_ref()), "{error:#}");
    assert!(downstream.request.lock().unwrap().is_none());

    drop(router);
    fixture.shutdown().await;
}

#[tokio::test]
async fn runtime_lora_media_is_rejected_before_encode_but_text_remains_passthrough() {
    if isolated(concat!(
        module_path!(),
        "::runtime_lora_media_is_rejected_before_encode_but_text_remains_passthrough"
    ))
    .await
    {
        return;
    }
    let mut fixture = Fixture::new().await;
    let downstream = Arc::new(CaptureEngine::default());
    let mut request = multimodal_request();
    request.routing_mut().lora_resolution_version =
        Some(crate::lora::runtime::RUNTIME_LORA_PROTOCOL_VERSION);
    let before = fixture.encode_calls.load(Ordering::SeqCst);
    let result = fixture
        .router
        .generate(SingleIn::new(request.clone()), downstream.clone())
        .await;
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("runtime LoRA media must not dispatch to an encoder"),
    };
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[dynamo_runtime::error::ErrorType::InvalidArgument],
        &[],
    ));
    assert!(
        error
            .to_string()
            .contains("runtime_lora_encoder_unsupported")
    );
    assert_eq!(fixture.encode_calls.load(Ordering::SeqCst), before);
    assert!(downstream.request.lock().unwrap().is_none());
    assert!(fixture.released.try_recv().is_err());

    request.multi_modal_data = None;
    let response = fixture
        .router
        .generate(SingleIn::new(request.clone()), downstream.clone())
        .await
        .unwrap();
    let _: Vec<_> = response.collect().await;
    let text = downstream.request.lock().unwrap().take().unwrap();
    assert_eq!(
        serde_json::to_value(text).unwrap(),
        serde_json::to_value(request).unwrap()
    );
    assert_eq!(fixture.encode_calls.load(Ordering::SeqCst), before);
    assert!(fixture.released.try_recv().is_err());

    assert_eq!(encoded_worker(&fixture.router).await, Some(fixture.ids[0]));
    assert_eq!(fixture.encode_calls.load(Ordering::SeqCst), before + 1);
    fixture.shutdown().await;
}

#[derive(Default, Clone, Copy)]
enum DownstreamBehavior {
    #[default]
    Empty,
    SetupError,
    TypedError,
    Terminal,
    ErrorTerminal,
    CancelledTerminal,
    Token,
    Pending,
}

fn media_request(kind: &str) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("model".to_string())
        .token_ids(vec![1, 2, 3])
        .multi_modal_data(Some(HashMap::from([(
            kind.to_string(),
            vec![MultimodalData::RawUrl(
                "data:image/png;base64,cGF5bG9hZA==".to_string(),
            )],
        )])))
        .stop_conditions(StopConditions::default())
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .build()
        .unwrap()
}

async fn isolated(test: &str) -> bool {
    let test_name = test.split_once("::").unwrap().1;
    if std::env::var("DYNAMO_ROUTING_HOP_TEST").as_deref() == Ok(test_name) {
        return false;
    }
    let mut child = tokio::process::Command::new(std::env::current_exe().unwrap());
    child
        .args(["--exact", test_name, "--nocapture"])
        .env("DYNAMO_ROUTING_HOP_TEST", test_name)
        .env("DYN_TCP_RPC_HOST", "127.0.0.1")
        .env("DYN_TCP_RPC_PORT", "0")
        .env("DYN_TCP_RESPONSE_STREAM_HOST", "127.0.0.1")
        .env("DYN_TCP_RESPONSE_STREAM_PORT", "0")
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(30), child.output())
        .await
        .expect("encoder subprocess must finish within its deadline")
        .unwrap();
    assert!(
        output.status.success()
            && String::from_utf8_lossy(&output.stdout).contains("test result: ok. 1 passed;"),
        "encoder subprocess must execute exactly one test\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

struct Fixture {
    router: Arc<EncoderRouter>,
    endpoint: dynamo_runtime::component::Endpoint,
    ids: Vec<u64>,
    admissions: watch::Sender<Vec<u64>>,
    released: tokio::sync::mpsc::UnboundedReceiver<(u64, PreprocessedRequest)>,
    release_gate: Arc<tokio::sync::Semaphore>,
    encode_calls: Arc<AtomicUsize>,
    card: Arc<ModelDeploymentCard>,
    runtime: Runtime,
    workers: Vec<dynamo_runtime::component::StartedEndpoint>,
    _runtimes: Vec<DistributedRuntime>,
    _store: tempfile::TempDir,
}

impl Fixture {
    async fn new() -> Self {
        let card = ModelDeploymentCard::with_name_only("model");
        let runtime = Runtime::from_current().unwrap();
        let store = tempfile::tempdir().unwrap();
        let config = || DistributedConfig {
            discovery_backend: DiscoveryBackend::KvStore(kv::Selector::File(store.path().into())),
            nats_config: None,
            request_plane: RequestPlaneMode::Tcp,
            response_plane: None,
            event_transport_kind: EventTransportKind::Zmq,
        };
        let namespace = format!("encoder-admission-{}", uuid::Uuid::new_v4());
        let mut workers = Vec::new();
        let mut runtimes = Vec::new();
        let (releases, released) = tokio::sync::mpsc::unbounded_channel();
        let release_gate = Arc::new(tokio::sync::Semaphore::new(0));
        let encode_calls = Arc::new(AtomicUsize::new(0));
        for _ in 0..2 {
            let drt = DistributedRuntime::new(runtime.clone(), config())
                .await
                .unwrap();
            let endpoint = drt
                .namespace(namespace.clone())
                .unwrap()
                .component("workers")
                .unwrap()
                .endpoint("encode");
            let worker = endpoint
                .endpoint_builder()
                .handler(
                    Ingress::for_engine(Arc::new(EncodeWorker(
                        drt.connection_id(),
                        releases.clone(),
                        release_gate.clone(),
                        encode_calls.clone(),
                    )))
                    .unwrap(),
                )
                .start_with_registration()
                .await
                .unwrap();
            workers.push(worker);
            runtimes.push(drt);
        }
        let drt = DistributedRuntime::new(runtime.clone(), config())
            .await
            .unwrap();
        let endpoint = drt
            .namespace(namespace.clone())
            .unwrap()
            .component("workers")
            .unwrap()
            .endpoint("encode");
        let raw_client = endpoint.client().await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut instances = raw_client.instance_source.as_ref().clone();
            while instances.borrow_and_update().len() != 2 {
                instances.changed().await.unwrap();
            }
        })
        .await
        .expect("all workers must be visible in raw discovery");
        runtimes.push(drt);
        let ids: Vec<_> = workers
            .iter()
            .map(|worker| worker.instance().id())
            .collect();
        let (admissions, admitted_ids) = watch::channel(vec![ids[0]]);
        let router = EncoderRouter::new("model".into(), namespace);
        let fixture = Self {
            router,
            endpoint,
            ids,
            admissions,
            released,
            release_gate,
            encode_calls,
            card: Arc::new(card),
            runtime,
            workers,
            _runtimes: runtimes,
            _store: store,
        };
        fixture
            .router
            .set_target(Some(fixture.target(1, admitted_ids)));
        fixture.wait_ready(1).await;
        fixture
    }

    fn target(&self, generation: u64, admitted_ids: watch::Receiver<Vec<u64>>) -> WorkerSetTarget {
        WorkerSetTarget::Committed(CommittedWorkerSetTarget {
            endpoint: self.endpoint.clone(),
            group: self.endpoint.id().to_string(),
            generation,
            card: self.card.clone(),
            admitted_ids,
        })
    }

    async fn wait_ready(&self, generation: u64) {
        let expected: HashSet<_> = self.admissions.borrow().iter().copied().collect();
        let target = self.target(generation, self.admissions.subscribe()).id();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let ready = self
                    .router
                    .binding
                    .load()
                    .as_ref()
                    .is_some_and(|binding| binding.target_id == target)
                    && self.router.available_worker_ids_for(
                        &self.endpoint.id().to_string(),
                        &self.endpoint.id(),
                    ) == Some(expected.clone());
                if ready {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("committed encoder admission must activate");
    }

    async fn admit_all(&self) {
        self.admissions.send_replace(self.ids.clone());
        self.wait_ready(1).await;
    }

    async fn release_for(&mut self, request_id: &str) -> (u64, PreprocessedRequest) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let release = self.released.recv().await.unwrap();
                if release.1.encoder_result.as_ref().unwrap()["request_id"] == request_id {
                    break release;
                }
            }
        })
        .await
        .expect("producer must receive the handoff release")
    }

    async fn retire_producer(&self) {
        self.admissions.send_replace(vec![self.ids[1]]);
        self.wait_ready(1).await;
        let binding = self.router.binding.load_full().unwrap();
        self._runtimes[0]
            .discovery()
            .unregister(dynamo_runtime::discovery::DiscoveryInstance::Endpoint(
                self.workers[0].instance().clone(),
            ))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut instances = binding.router.client.instance_source.as_ref().clone();
            while instances
                .borrow_and_update()
                .iter()
                .any(|instance| instance.id() == self.ids[0])
            {
                instances.changed().await.unwrap();
            }
        })
        .await
        .expect("retired producer must leave raw discovery");
    }

    async fn shutdown(self) {
        drop(self.router);
        for worker in self.workers {
            worker.shutdown().await.unwrap();
        }
        self.runtime.shutdown();
    }
}

struct RetiringPrefill {
    admissions: watch::Sender<Vec<u64>>,
    replacement: u64,
    clear: bool,
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for RetiringPrefill
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        self.admissions.send_replace(vec![self.replacement]);
        Err(crate::kv_router::prefill_router::reject_stale_media_plan(request, self.clear).await)
    }
}

#[tokio::test]
async fn stale_media_plan_releases_original_encoder_after_booking_abort() {
    if isolated(concat!(
        module_path!(),
        "::stale_media_plan_releases_original_encoder_after_booking_abort"
    ))
    .await
    {
        return;
    }
    for clear in [false, true] {
        let mut fixture = Fixture::new().await;
        fixture.router.set_required(true);
        let request = SingleIn::new(multimodal_request());
        let request_id = request.id().to_string();
        let downstream = Arc::new(RetiringPrefill {
            admissions: fixture.admissions.clone(),
            replacement: fixture.ids[1],
            clear,
        });
        let error = fixture
            .router
            .generate(request, downstream)
            .await
            .unwrap_err();
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[dynamo_runtime::error::ErrorType::WorkerUnavailable],
            &[]
        ));
        fixture.wait_ready(1).await;
        let (producer, release) = fixture.release_for(&request_id).await;
        assert_eq!(producer, fixture.ids[0]);
        assert_eq!(
            release.encoder_result.as_ref().unwrap()["worker_id"],
            producer
        );
        fixture.shutdown().await;
    }
}
