// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
    time::Duration,
};

use futures::{FutureExt, stream};
use serde_json::json;

use dynamo_runtime::{
    DistributedRuntime, Runtime,
    discovery::EventTransportKind,
    distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
    engine::{AsyncEngine, AsyncEngineContextProvider},
    pipeline::{Error, ResponseStream, context::Controller, network::Ingress},
    storage::kv,
};

use crate::discovery::CommittedWorkerSetTarget;
use crate::model_card::ModelDeploymentCard;
use crate::protocols::common::preprocessor::{
    ENCODER_RESULT_RELEASE_SUPPORTED, MultimodalData, RELEASE_ENCODER_RESULT,
};
use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};

use super::*;

fn stream_of(items: Vec<Annotated<LLMEngineOutput>>) -> ManyOut<Annotated<LLMEngineOutput>> {
    ResponseStream::new(
        Box::pin(stream::iter(items)),
        Arc::new(Controller::default()),
    )
}

#[derive(Default)]
struct CaptureEngine {
    request: Mutex<Option<PreprocessedRequest>>,
    behavior: DownstreamBehavior,
    setup_failed: tokio::sync::Notify,
}

#[derive(Default, Clone, Copy)]
enum DownstreamBehavior {
    #[default]
    Empty,
    SetupError,
    TypedError,
    Terminal,
    Token,
    Pending,
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
                    self.setup_failed.notify_one();
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
    PreprocessedRequest::builder()
        .model("model".to_string())
        .token_ids(vec![1, 2, 3])
        .multi_modal_data(Some(HashMap::from([(
            "image_url".to_string(),
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

struct EncodeWorker(
    u64,
    tokio::sync::mpsc::UnboundedSender<(u64, PreprocessedRequest)>,
    Arc<tokio::sync::Semaphore>,
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
        if request.sampling_options.n == Some(2) {
            return Ok(ResponseStream::new(
                Box::pin(stream::iter([Annotated {
                    error: Some(
                        DynamoError::builder()
                            .class(ErrorClass::InvalidRequest)
                            .message("encoder accepts n=1 only")
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
        output.extra_args = Some(json!({ENCODER_RESULT_RELEASE_SUPPORTED: true}));
        Ok(ResponseStream::new(
            Box::pin(stream::iter([Annotated::from_data(output)])),
            request.context(),
        ))
    }
}

async fn encoded_worker(router: &EncoderRouter) -> Option<u64> {
    let downstream = Arc::new(CaptureEngine::default());
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
async fn committed_encoder_admission_survives_membership_changes_and_endpoint_reuse() {
    // Each request-plane server needs a live runtime for its process-wide
    // accept loop, independent of the surrounding test harness's runtimes.
    const TEST: &str = concat!(
        module_path!(),
        "::committed_encoder_admission_survives_membership_changes_and_endpoint_reuse"
    );
    let test_name = TEST.split_once("::").unwrap().1;
    if std::env::var("DYNAMO_ROUTING_HOP_TEST").as_deref() != Ok(test_name) {
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
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
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
    let (releases, mut released) = tokio::sync::mpsc::unbounded_channel();
    let release_gate = Arc::new(tokio::sync::Semaphore::new(0));
    for _ in 0..3 {
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
        while instances.borrow_and_update().len() != 3 {
            instances.changed().await.unwrap();
        }
    })
    .await
    .expect("all three workers must be visible in raw discovery");
    let ids: Vec<_> = workers
        .iter()
        .map(|worker| worker.instance().id())
        .collect();
    let (admissions, admitted_ids) = watch::channel(vec![ids[0]]);
    let card = Arc::new(ModelDeploymentCard::with_name_only("model"));
    let group_id = endpoint.id().to_string();
    let target = |generation, admitted_ids| {
        WorkerSetTarget::Committed(CommittedWorkerSetTarget {
            endpoint: endpoint.clone(),
            group: endpoint.id().to_string(),
            generation,
            card: card.clone(),
            admitted_ids,
        })
    };
    let router = EncoderRouter::new("model".into(), namespace);
    router.set_target(Some(target(1, admitted_ids)));
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Some(worker) = encoded_worker(&router).await {
                assert_eq!(worker, ids[0]);
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("committed encoder must activate");
    assert_eq!(
        router.available_worker_ids_for(&group_id, &endpoint.id()),
        Some(HashSet::from([ids[0]]))
    );
    assert!(
        router
            .available_worker_ids_for("other-group", &endpoint.id())
            .is_none()
    );
    let mut original = multimodal_request();
    let consumer_routing = json!({
        "backend_instance_id": 11,
        "prefill_worker_id": 12,
        "decode_worker_id": 13,
        "dp_rank": 1,
        "prefill_dp_rank": 2,
        "lora_name": "adapter",
        "cache_salt": "tenant",
        "priority": 3,
    });
    original.routing = Some(serde_json::from_value(consumer_routing.clone()).unwrap());
    let downstream = Arc::new(CaptureEngine::default());
    let response = router
        .generate(SingleIn::new(original), downstream.clone())
        .await
        .unwrap();
    let _: Vec<_> = response.collect().await;
    {
        let captured = downstream.request.lock().unwrap();
        let captured = captured.as_ref().unwrap();
        assert_eq!(
            serde_json::to_value(&captured.routing).unwrap(),
            consumer_routing
        );
        assert_eq!(
            captured.encoder_result.as_ref().unwrap()["routing"],
            json!({"lora_name": "adapter", "cache_salt": "tenant", "priority": 3})
        );
    }
    for _ in 0..6 {
        assert_eq!(encoded_worker(&router).await, Some(ids[0]));
    }

    admissions.send_replace(vec![ids[0], ids[1]]);
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut seen = HashSet::new();
        while seen.len() < 2 {
            let worker = encoded_worker(&router).await.unwrap();
            assert!(
                ids[..2].contains(&worker),
                "rejected encoder received a request"
            );
            seen.insert(worker);
        }
    })
    .await
    .expect("compatible members must become reachable through the admission channel");

    let retired = router.binding.load_full().unwrap();
    admissions.send_replace(Vec::new());
    drop(admissions);
    router.set_target(None);
    assert!(
        router
            .available_worker_ids_for(&group_id, &endpoint.id())
            .is_none()
    );
    let (successor_admissions, successor_ids) = watch::channel(vec![ids[2]]);
    router.set_target(Some(target(2, successor_ids)));
    assert_eq!(
        router.available_worker_ids_for(&group_id, &endpoint.id()),
        Some(HashSet::new())
    );
    assert!(
        router
            .available_worker_ids_for("other-group", &endpoint.id())
            .is_none()
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Some(worker) = encoded_worker(&router).await {
                assert_eq!(worker, ids[2]);
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("same-endpoint successor must activate with its own admission");
    assert_eq!(
        router.available_worker_ids_for(&group_id, &endpoint.id()),
        Some(HashSet::from([ids[2]]))
    );
    for _ in 0..6 {
        assert_eq!(encoded_worker(&router).await, Some(ids[2]));
    }
    assert!(
        tokio::time::timeout(
            Duration::from_secs(5),
            retired
                .router
                .generate(SingleIn::new(multimodal_request()),)
        )
        .await
        .expect("retired admission must fail promptly")
        .is_err()
    );

    let downstream = Arc::new(CaptureEngine::default());
    let mut invalid = multimodal_request();
    invalid.sampling_options.n = Some(2);
    router.set_required(true);
    let error = router
        .generate(SingleIn::new(invalid.clone()), downstream.clone())
        .await
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<DynamoError>().unwrap().class(),
        ErrorClass::InvalidRequest
    );
    assert!(downstream.request.lock().unwrap().is_none());
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

    // Keep several live producers available: cleanup must target the exact
    // producer selected for this request, without another round-robin hop.
    successor_admissions.send_replace(vec![ids[1], ids[2]]);
    for behavior in [
        DownstreamBehavior::SetupError,
        DownstreamBehavior::TypedError,
        DownstreamBehavior::Terminal,
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
        original.extra_args =
            Some(json!({"sampling_constraints": "generation-only", "block_release": true}));
        let request = SingleIn::new(original);
        let request_id = request.id().to_string();
        let request_context = request.context();
        let result = if matches!(behavior, DownstreamBehavior::SetupError) {
            let dispatch = tokio::spawn({
                let router = router.clone();
                let downstream = downstream.clone();
                async move { router.generate(request, downstream).await }
            });
            downstream.setup_failed.notified().await;
            assert!(
                dispatch.is_finished(),
                "setup errors must not await blocked producer release"
            );
            dispatch.await.unwrap()
        } else {
            router.generate(request, downstream.clone()).await
        };
        if matches!(behavior, DownstreamBehavior::SetupError) {
            assert!(result.is_err());
        } else {
            let mut response = result.unwrap();
            if !matches!(behavior, DownstreamBehavior::Pending) {
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
        if matches!(behavior, DownstreamBehavior::Token) {
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
                "a token-confirmed handoff must not send producer release"
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
        release_gate.add_permits(1);
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
    for worker in workers {
        worker.shutdown().await.unwrap();
    }
    runtime.shutdown();
}

#[tokio::test]
async fn binding_availability_tracks_local_inhibition_and_cancellation() {
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use std::collections::HashSet;

    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("encoder-availability")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate");
    let router = EncoderRouter::disabled();
    assert!(
        router
            .available_worker_ids_for("legacy", &endpoint.id())
            .is_none()
    );
    let binding = Arc::new(
        EncoderRouter::build(
            WorkerSetTarget::Legacy(endpoint.clone()),
            CancellationToken::new(),
        )
        .await
        .unwrap(),
    );
    binding
        .router
        .client
        .override_discovered_instances(vec![1, 2]);
    router.binding.store(Some(binding.clone()));
    router
        .lifecycle
        .store(EncoderLifecycleState::Active as u8, Ordering::Release);
    assert_eq!(
        router.available_worker_ids_for("legacy", &endpoint.id()),
        Some(HashSet::from([1, 2]))
    );
    binding.router.client.report_instance_down(1);
    assert_eq!(
        router.available_worker_ids_for("legacy", &endpoint.id()),
        Some(HashSet::from([2]))
    );
    let mut cleanup = EncoderResultCleanup {
        router: binding.router.clone(),
        worker_id: 2,
        request: None,
    };
    let opaque = json!({"arbitrary": [1, {"engine-owned": "value"}]});
    let mut terminal = LLMEngineOutput::encode_terminal(opaque.as_object().unwrap().clone());
    cleanup.observe("model", &terminal).unwrap();
    assert!(
        cleanup.request.is_none(),
        "ordinary backends must not receive release requests"
    );
    terminal.extra_args = Some(json!({ENCODER_RESULT_RELEASE_SUPPORTED: false}));
    cleanup.observe("model", &terminal).unwrap();
    assert!(cleanup.request.is_none());
    terminal.extra_args = Some(json!({ENCODER_RESULT_RELEASE_SUPPORTED: true}));
    cleanup.observe("model", &terminal).unwrap();
    assert_eq!(cleanup.request.take().unwrap().encoder_result, Some(opaque));
    let mut other = endpoint.id();
    other.name = "other".into();
    assert!(router.available_worker_ids_for("legacy", &other).is_none());
    router.cancel_token.cancel();
    assert_eq!(
        router.available_worker_ids_for("legacy", &endpoint.id()),
        Some(HashSet::new())
    );
    runtime.shutdown();
}

#[tokio::test]
async fn binding_availability_waits_for_rebind_publication() {
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use std::{sync::mpsc, time::Duration};

    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let component = distributed
        .namespace("encoder-rebind")
        .unwrap()
        .component("workers")
        .unwrap();
    let original = component.endpoint("original");
    let replacement = Arc::new(
        EncoderRouter::build(
            WorkerSetTarget::Legacy(component.endpoint("replacement")),
            CancellationToken::new(),
        )
        .await
        .unwrap(),
    );
    let router = EncoderRouter::disabled();
    let binding = Arc::new(
        EncoderRouter::build(
            WorkerSetTarget::Legacy(original.clone()),
            CancellationToken::new(),
        )
        .await
        .unwrap(),
    );
    binding.router.client.override_discovered_instances(vec![1]);
    router.binding.store(Some(binding));
    router
        .lifecycle
        .store(EncoderLifecycleState::Active as u8, Ordering::Release);
    let mut target = router.target.lock();
    *target = Some(WorkerSetTargetId::Legacy(original.id()));
    let (started_tx, started_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let reader_router = router.clone();
    let reader = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        result_tx
            .send(reader_router.available_worker_ids_for("legacy", &original.id()))
            .unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(matches!(
        result_rx.recv_timeout(Duration::from_millis(100)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    *target = Some(replacement.target_id.clone());
    router.binding.store(Some(replacement));
    router
        .lifecycle
        .store(EncoderLifecycleState::Active as u8, Ordering::Release);
    drop(target);
    assert_eq!(
        result_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
        None
    );
    reader.join().unwrap();
    runtime.shutdown();
}

#[tokio::test]
async fn pending_activation_does_not_keep_router_alive() {
    let router = EncoderRouter::new("model".into(), "namespace".into());
    let weak = Arc::downgrade(&router);

    drop(router);

    assert!(weak.upgrade().is_none());
}

#[tokio::test]
async fn required_encoder_fails_closed_when_unavailable_or_failing() {
    for active in [false, true] {
        let router = EncoderRouter::disabled();
        router.set_required(true);
        if active {
            router
                .lifecycle
                .store(EncoderLifecycleState::Active as u8, Ordering::Release);
        }
        let downstream = Arc::new(CaptureEngine::default());
        let error = router
            .generate(SingleIn::new(multimodal_request()), downstream.clone())
            .await
            .expect_err("vision-disabled consumers must not receive inline media");
        assert!(error.chain().any(|error| {
            error
                .downcast_ref::<DynamoError>()
                .is_some_and(|error| error.class() == ErrorClass::Unavailable)
        }));
        assert!(downstream.request.lock().unwrap().is_none());
    }
}

#[tokio::test]
async fn required_encoder_rejects_unsupported_media_before_dispatch_but_passes_text() {
    let router = EncoderRouter::disabled();
    router.set_required(true);
    for modality in ["video_url", "audio_url"] {
        let mut request = multimodal_request();
        let images = request
            .multi_modal_data
            .as_mut()
            .unwrap()
            .remove("image_url")
            .unwrap();
        request
            .multi_modal_data
            .as_mut()
            .unwrap()
            .insert(modality.into(), images);
        let downstream = Arc::new(CaptureEngine::default());
        let error = router
            .generate(SingleIn::new(request), downstream.clone())
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<DynamoError>().unwrap().class(),
            ErrorClass::InvalidRequest
        );
        assert!(downstream.request.lock().unwrap().is_none());
    }
    let downstream = Arc::new(CaptureEngine::default());
    let mut text = multimodal_request();
    text.multi_modal_data = None;
    router
        .generate(SingleIn::new(text), downstream.clone())
        .await
        .unwrap();
    assert!(downstream.request.lock().unwrap().is_some());
}

#[tokio::test]
async fn encoder_failure_falls_back_to_downstream() {
    let router = EncoderRouter::disabled();
    router
        .lifecycle
        .store(EncoderLifecycleState::Active as u8, Ordering::Release);
    let downstream = Arc::new(CaptureEngine::default());
    let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
        downstream.clone();

    let _response = router
        .generate(SingleIn::new(multimodal_request()), next)
        .await
        .expect("encoder failure should fall through to downstream");

    let request = downstream
        .request
        .lock()
        .unwrap()
        .take()
        .expect("downstream must receive the original request");
    assert!(request.encoder_result.is_none());
    assert!(request.multi_modal_data.is_some());
}

#[tokio::test]
async fn consumes_object_shaped_encode_terminal() {
    let output =
        LLMEngineOutput::encode_terminal(json!({"schema_version": 1}).as_object().unwrap().clone());
    let (result, _) =
        EncoderRouter::consume_encode_stream(stream_of(vec![Annotated::from_data(output)]), None)
            .await
            .unwrap();
    assert_eq!(result, json!({"schema_version": 1}));
}

#[tokio::test]
async fn rejects_terminal_without_encoder_result() {
    let result = EncoderRouter::consume_encode_stream(
        stream_of(vec![Annotated::from_data(LLMEngineOutput::stop())]),
        None,
    )
    .await;
    assert!(result.is_err());
}
