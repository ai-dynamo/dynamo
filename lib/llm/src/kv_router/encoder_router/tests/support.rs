// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::atomic::AtomicUsize;

pub(super) fn stream_of(
    items: Vec<Annotated<LLMEngineOutput>>,
) -> ManyOut<Annotated<LLMEngineOutput>> {
    ResponseStream::new(
        Box::pin(stream::iter(items)),
        Arc::new(Controller::default()),
    )
}

#[derive(Default)]
pub(super) struct CaptureEngine {
    pub(super) request: Mutex<Option<PreprocessedRequest>>,
    pub(super) behavior: DownstreamBehavior,
}

#[derive(Default, Clone, Copy)]
pub(super) enum DownstreamBehavior {
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

pub(super) fn multimodal_request() -> PreprocessedRequest {
    media_request("image_url")
}

pub(super) fn media_request(kind: &str) -> PreprocessedRequest {
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
        if matches!(request.sampling_options.n, Some(2 | 4 | 5 | 6))
            || request
                .multi_modal_data
                .as_ref()
                .is_some_and(|media| media.contains_key("unsupported_url"))
        {
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

pub(super) async fn encoded_worker(router: &EncoderRouter) -> Option<u64> {
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

pub(super) async fn isolated(test: &str) -> bool {
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
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

pub(super) struct Fixture {
    pub(super) router: Arc<EncoderRouter>,
    pub(super) endpoint: dynamo_runtime::component::Endpoint,
    pub(super) ids: Vec<u64>,
    pub(super) admissions: watch::Sender<Vec<u64>>,
    pub(super) released: tokio::sync::mpsc::UnboundedReceiver<(u64, PreprocessedRequest)>,
    pub(super) release_gate: Arc<tokio::sync::Semaphore>,
    pub(super) encode_calls: Arc<AtomicUsize>,
    card: Arc<ModelDeploymentCard>,
    runtime: Runtime,
    workers: Vec<dynamo_runtime::component::StartedEndpoint>,
    _runtimes: Vec<DistributedRuntime>,
    _store: tempfile::TempDir,
}

impl Fixture {
    pub(super) async fn new() -> Self {
        Self::with_card(ModelDeploymentCard::with_name_only("model")).await
    }

    pub(super) async fn with_card(card: ModelDeploymentCard) -> Self {
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
            while instances.borrow_and_update().len() != 3 {
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

    pub(super) fn target(
        &self,
        generation: u64,
        admitted_ids: watch::Receiver<Vec<u64>>,
    ) -> WorkerSetTarget {
        WorkerSetTarget::Committed(CommittedWorkerSetTarget {
            endpoint: self.endpoint.clone(),
            group: self.endpoint.id().to_string(),
            generation,
            card: self.card.clone(),
            admitted_ids,
        })
    }

    pub(super) async fn wait_ready(&self, generation: u64) {
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

    pub(super) async fn admit_all(&self) {
        self.admissions.send_replace(self.ids.clone());
        self.wait_ready(1).await;
    }

    pub(super) async fn release_for(&mut self, request_id: &str) -> (u64, PreprocessedRequest) {
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

    pub(super) async fn retire_producer(&self) {
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

    pub(super) async fn shutdown(self) {
        drop(self.router);
        for worker in self.workers {
            worker.shutdown().await.unwrap();
        }
        self.runtime.shutdown();
    }
}
