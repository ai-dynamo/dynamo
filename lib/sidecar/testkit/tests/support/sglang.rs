// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SGLang fixture for `sglang.api.v1`, the gRPC contract of SGLang's Rust
//! server.

use std::collections::{HashMap, VecDeque};
use std::process::Command;
use std::sync::{Arc, Mutex};

use dynamo_backend_common::{BackendError, DisaggregationMode, PreprocessedRequest};
use dynamo_llm::local_model::runtime_config::{
    TOKEN_BUDGET_RUNTIME_KEY, TOOL_CALL_STRUCTURAL_TAG_REASONING_GATE_RUNTIME_KEY, TokenBudget,
};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::common::preprocessor::RoutingHints;
use dynamo_mocker::common::protocols::EngineType;
use dynamo_sglang_mocker::{MockerServerConfig, ServerMode, SglangMockerService};
use dynamo_sglang_sidecar::SglangSidecarEngine;
use dynamo_sglang_sidecar::proto::{
    self as pb,
    generate_stream_item::Item,
    sglang_service_server::{SglangService, SglangServiceServer},
};
use dynamo_sidecar_testkit::control::{Controller, Protocol, RequestHandle};
use dynamo_sidecar_testkit::fixtures::Outputs;
use dynamo_sidecar_testkit::server::TestServer;
use futures::stream::BoxStream;
use tokio::sync::watch;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use super::{
    DiscoveryFixture, FixtureConfig, GenerateOpening, HandoffFixture, ProcessFixture,
    SidecarFixture, WireFixture, fast_engine_args, sidecar_command, wait_scheduler_idle,
};

/// Rewrites one discovery response, as a script step.
pub type ModelInfoPatch = Box<dyn Fn(&mut pb::GetModelInfoResponse) + Send + Sync>;
pub type ServerInfoPatch = Box<dyn Fn(&mut pb::GetServerInfoResponse) + Send + Sync>;

pub struct Fixture {
    config: FixtureConfig,
    service: SglangMockerService,
    pub(crate) server: TestServer,
    scripted: Arc<Mutex<HashMap<String, Vec<pb::GenerateStreamItem>>>>,
    model_info: Arc<Mutex<Option<ModelInfoPatch>>>,
    server_info: Arc<Mutex<VecDeque<ServerInfoPatch>>>,
    health: watch::Sender<Option<bool>>,
    health_calls: watch::Receiver<usize>,
    aborts: watch::Receiver<Vec<(String, bool)>>,
    http: Option<http::Peer>,
}

impl Fixture {
    /// The Rust server serves native `/generate` on the HTTP port GetServerInfo
    /// reports; the peer imitates it, and Abort RPCs reach this fixture.
    pub async fn start_http(control: Controller<http::Adapter>, config: FixtureConfig) -> Self {
        let mut fixture = Self::start(Controller::default(), config).await;
        let http = http::Peer::start(control, fixture.aborts.clone()).await;
        let port = u32::from(http.port());
        fixture.override_discovery(None, vec![Box::new(move |info| info.port = port)]);
        fixture.http = Some(http);
        fixture
    }

    pub fn http(&self) -> &http::Peer {
        self.http.as_ref().expect("HTTP peer is enabled")
    }

    /// Replace the Mocker's stream for one request with scripted items.
    pub fn respond(&self, request_id: &str, items: Vec<pb::GenerateStreamItem>) {
        assert!(
            self.scripted
                .lock()
                .unwrap()
                .insert(request_id.to_owned(), items)
                .is_none()
        );
    }

    /// GetServerInfo applies `server_info` patches front-to-back, repeating
    /// the last one.
    pub fn override_discovery(
        &self,
        model_info: Option<ModelInfoPatch>,
        server_info: Vec<ServerInfoPatch>,
    ) {
        *self.model_info.lock().unwrap() = model_info;
        *self.server_info.lock().unwrap() = server_info.into();
    }

    /// Abort RPCs in arrival order, with whether each matched a request.
    pub fn aborts(&self) -> Vec<(String, bool)> {
        self.aborts.borrow().clone()
    }

    pub async fn wait_aborts(&self, count: usize) {
        let mut aborts = self.aborts.clone();
        dynamo_sidecar_testkit::bounded(
            "SGLang Abort RPC",
            aborts.wait_for(|aborts| aborts.len() >= count),
        )
        .await
        .unwrap();
    }
}

impl SidecarFixture for Fixture {
    type Engine = SglangSidecarEngine;
    type Protocol = Adapter;
    const GENERATE_OPENING: GenerateOpening = GenerateOpening::OnStreamPoll;

    async fn start(control: Controller<Adapter>, config: FixtureConfig) -> Self {
        let mut args = fast_engine_args(EngineType::Sglang);
        args.speedup_ratio = config.speedup_ratio;
        let service = SglangMockerService::new(
            MockerServerConfig {
                model: config.model.clone(),
                mode: match config.disaggregation_mode {
                    DisaggregationMode::Aggregated => ServerMode::Aggregated,
                    DisaggregationMode::Prefill => ServerMode::Prefill,
                    DisaggregationMode::Decode => ServerMode::Decode,
                    DisaggregationMode::Encode => panic!("Mocker does not support encode mode"),
                },
                ..Default::default()
            },
            args,
        )
        .unwrap();
        let scripted = Arc::new(Mutex::new(HashMap::new()));
        let model_info = Arc::new(Mutex::new(None));
        let server_info = Arc::new(Mutex::new(VecDeque::new()));
        let (health, health_rx) = watch::channel(Some(true));
        let (health_tx, health_calls) = watch::channel(0);
        let (abort_tx, aborts) = watch::channel(Vec::new());
        let controlled = ControlledService {
            inner: service.clone(),
            control,
            scripted: scripted.clone(),
            model_info: model_info.clone(),
            server_info: server_info.clone(),
            health: health_rx,
            health_calls: health_tx,
            aborts: abort_tx,
        };
        let server = TestServer::start(move |listener, shutdown| async move {
            tonic::transport::Server::builder()
                .add_service(SglangServiceServer::new(controlled))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown.await;
                })
                .await?;
            Ok(())
        })
        .await
        .unwrap();
        Self {
            config,
            service,
            server,
            scripted,
            model_info,
            server_info,
            health,
            health_calls,
            aborts,
            http: None,
        }
    }

    async fn engine(&self) -> Self::Engine {
        let argv: Vec<String> = vec![
            "dynamo-sglang-sidecar".into(),
            "--grpc-endpoint".into(),
            self.server.endpoint(),
            "--disaggregation-mode".into(),
            self.config.disaggregation_mode.to_string(),
            "--grpc-connections".into(),
            self.config.connections.to_string(),
            "--grpc-connect-attempt-timeout-secs".into(),
            "1".into(),
            "--grpc-retry-interval-secs".into(),
            "1".into(),
            "--grpc-startup-deadline-secs".into(),
            "5".into(),
        ];
        tokio::task::spawn_blocking(move || SglangSidecarEngine::from_args(Some(argv)).unwrap().0)
            .await
            .unwrap()
    }

    fn eof_error() -> BackendError {
        BackendError::EngineShutdown
    }

    fn native_model(_request: &pb::GenerateRequest) -> Option<&str> {
        // The Rust server serves one model and its Generate RPC has no model
        // selector.
        None
    }

    fn active_request_count(&self) -> usize {
        self.service.active_request_count()
    }

    async fn scheduler_idle(&self) {
        wait_scheduler_idle(self.service.metrics_receiver(), || {
            self.active_request_count()
        })
        .await;
    }

    async fn shutdown(&mut self) {
        self.server.shutdown().await.unwrap();
        if let Some(http) = self.http.as_mut() {
            http.shutdown().await;
        }
    }
}

/// The single value of each scalar-or-list request field the sidecar sends.
pub fn sampling(request: &pb::GenerateRequest) -> &pb::SamplingParams {
    match request.sampling_params.as_ref().unwrap().value.as_ref() {
        Some(pb::sampling_params_or_list::Value::One(params)) => params,
        other => panic!("expected one sampling-params object: {other:?}"),
    }
}

fn input_ids(request: &pb::GenerateRequest) -> &[i64] {
    match request.input_ids.as_ref().unwrap().value.as_ref() {
        Some(pb::token_ids_or_list::Value::One(ids)) => &ids.ids,
        other => panic!("expected one token-id list: {other:?}"),
    }
}

fn one_bool(value: Option<&pb::BoolOrList>) -> Option<bool> {
    match value?.value.as_ref()? {
        pb::bool_or_list::Value::One(value) => Some(*value),
        pb::bool_or_list::Value::Many(_) => panic!("expected one boolean"),
    }
}

fn one_int64(value: Option<&pb::Int64OrList>) -> Option<i64> {
    match value?.value.as_ref()? {
        pb::int64_or_list::Value::One(value) => Some(*value),
        pb::int64_or_list::Value::Many(_) => panic!("expected one integer"),
    }
}

fn one_string(value: Option<&pb::OptionalStringOrList>) -> Option<String> {
    match value?.value.as_ref()? {
        pb::optional_string_or_list::Value::One(value) => value.value.clone(),
        pb::optional_string_or_list::Value::Many(_) => panic!("expected one string"),
    }
}

fn one_optional_int64(value: Option<&pb::OptionalInt64OrList>) -> Option<i64> {
    match value?.value.as_ref()? {
        pb::optional_int64_or_list::Value::One(value) => value.value,
        pb::optional_int64_or_list::Value::Many(_) => panic!("expected one integer"),
    }
}

pub fn frame(item: &pb::GenerateStreamItem) -> Option<&pb::GenerateResponse> {
    match item.item.as_ref() {
        Some(Item::Frame(frame)) => Some(frame),
        _ => None,
    }
}

fn output_ids(frame: &pb::GenerateResponse) -> &[i64] {
    frame.output_ids.as_ref().map_or(&[], |ids| &ids.ids)
}

/// A frame carrying `tokens`, terminal when `finish` is set.
pub fn token_frame(
    tokens: &[i64],
    finish: Option<pb::finish_reason::Kind>,
) -> pb::GenerateStreamItem {
    pb::GenerateStreamItem {
        item: Some(Item::Frame(pb::GenerateResponse {
            output_ids: (!tokens.is_empty()).then(|| pb::TokenIds {
                ids: tokens.to_vec(),
            }),
            meta_info: Some(pb::GenerateMetaInfo {
                finish_reason: finish.map(|kind| pb::FinishReason { kind: Some(kind) }),
                ..Default::default()
            }),
            ..Default::default()
        })),
    }
}

/// One request's in-stream failure with SGLang's HTTP status code.
pub fn error_item(code: u32, message: &str) -> pb::GenerateStreamItem {
    pb::GenerateStreamItem {
        item: Some(Item::Error(pb::GenerateStreamError {
            error: Some(pb::ErrorBody {
                message: message.to_string(),
                code,
            }),
            index: None,
        })),
    }
}

impl WireFixture for Fixture {
    fn assert_stream(
        handle: &RequestHandle<Adapter>,
        request: &PreprocessedRequest,
        outputs: &Outputs,
    ) {
        let native = handle.native_request().unwrap();
        assert_eq!(
            input_ids(&native),
            request
                .token_ids
                .iter()
                .map(|&token| i64::from(token))
                .collect::<Vec<_>>()
        );
        let sampling = sampling(&native);
        assert_eq!(
            sampling.max_new_tokens,
            request.stop_conditions.max_tokens.map(i64::from)
        );
        assert_eq!(
            sampling.temperature,
            request.sampling_options.temperature.map(f64::from)
        );
        assert_eq!(
            sampling.top_p,
            request.sampling_options.top_p.map(f64::from)
        );
        assert_eq!(
            sampling.top_k,
            request.sampling_options.top_k.map(i64::from)
        );
        assert_eq!(
            sampling.min_p,
            request.sampling_options.min_p.map(f64::from)
        );
        assert_eq!(
            sampling.presence_penalty,
            request.sampling_options.presence_penalty.map(f64::from)
        );
        assert_eq!(
            sampling.frequency_penalty,
            request.sampling_options.frequency_penalty.map(f64::from)
        );
        assert_eq!(
            sampling.repetition_penalty,
            request.sampling_options.repetition_penalty.map(f64::from)
        );
        assert_eq!(
            sampling.min_new_tokens,
            request.stop_conditions.min_tokens.map(i64::from)
        );
        let stop = match sampling.stop.as_ref().and_then(|stop| stop.value.as_ref()) {
            Some(pb::string_or_list::Value::Many(stop)) => stop.items.clone(),
            Some(pb::string_or_list::Value::One(stop)) => vec![stop.clone()],
            None => Vec::new(),
        };
        assert_eq!(
            stop,
            request.stop_conditions.stop.clone().unwrap_or_default()
        );
        let stop_ids = request
            .stop_conditions
            .stop_token_ids
            .iter()
            .chain(&request.stop_conditions.stop_token_ids_hidden)
            .chain(&request.stop_conditions.stop_token_ids_visible)
            .flatten()
            .map(|&token| i64::from(token))
            .collect::<std::collections::BTreeSet<_>>();
        let native_stop_ids = sampling
            .stop_token_ids
            .as_ref()
            .map(|ids| ids.items.clone())
            .unwrap_or_default();
        assert_eq!(native_stop_ids.len(), stop_ids.len());
        assert_eq!(
            native_stop_ids
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            stop_ids
        );
        assert_eq!(sampling.ignore_eos, request.stop_conditions.ignore_eos);
        assert_eq!(native.stream, Some(true));
        assert_eq!(one_bool(native.return_logprob.as_ref()), Some(true));
        assert_eq!(one_int64(native.top_logprobs_num.as_ref()), Some(2));
        assert_eq!(one_int64(native.logprob_start_len.as_ref()), Some(0));
        assert_eq!(native.routing_key, request.mdc_sum);
        assert_eq!(
            one_string(native.lora_path.as_ref()),
            request
                .routing
                .as_ref()
                .and_then(|routing| routing.lora_name.clone())
        );
        // The engine serves one DP rank, so the router's rank 0 is not
        // forwarded.
        assert_eq!(native.routed_dp_rank, None);

        let native_frames = handle
            .native_responses()
            .into_iter()
            .map(|item| {
                frame(&item)
                    .expect("a healthy stream has only frames")
                    .clone()
            })
            .collect::<Vec<_>>();
        let outputs: Vec<_> = outputs
            .iter()
            .map(|output| output.as_ref().unwrap())
            .collect();
        assert_eq!(outputs.len(), native_frames.len());
        assert_eq!(
            outputs.len(),
            request.stop_conditions.max_tokens.unwrap() as usize
        );
        assert!(outputs.iter().all(|output| output.token_ids.len() == 1));
        for (output, native) in outputs.iter().zip(&native_frames) {
            assert_eq!(
                output.token_ids,
                output_ids(native)
                    .iter()
                    .map(|&token| u32::try_from(token).unwrap())
                    .collect::<Vec<_>>()
            );
            assert!(output.text.is_none());
            let meta = native.meta_info.as_ref().unwrap();
            let selected = &meta.output_token_logprobs.as_ref().unwrap().entries;
            assert_eq!(
                output.log_probs.as_ref().unwrap(),
                &selected
                    .iter()
                    .map(|entry| entry.logprob.unwrap())
                    .collect::<Vec<_>>()
            );
            let actual = output.top_logprobs.as_ref().unwrap();
            assert_eq!(actual.len(), output_ids(native).len());
            assert_eq!(actual.len(), meta.output_top_logprobs.len());
            for (actual, expected) in actual.iter().zip(&meta.output_top_logprobs) {
                let expected = &expected.row.as_ref().unwrap().entries;
                assert_eq!(actual.len(), 2);
                assert_eq!(actual.len(), expected.len());
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert_eq!(i64::from(actual.token_id), expected.token_id);
                    assert_eq!(Some(actual.logprob), expected.logprob);
                    assert_eq!(actual.rank, (index + 1) as u32);
                }
            }
        }
        assert!(
            outputs[..outputs.len() - 1]
                .iter()
                .all(|output| output.engine_data.is_none())
        );
        // SGLang's incremental stream reports prompt logprobs on its first
        // frame; the sidecar attaches them to the terminal output only.
        let prompts = &outputs.last().unwrap().engine_data.as_ref().unwrap()["prompt_logprobs"];
        let first = native_frames.first().unwrap().meta_info.as_ref().unwrap();
        let selected = &first.input_token_logprobs.as_ref().unwrap().entries;
        let alternatives = &first.input_top_logprobs;
        assert!(native_frames[1..].iter().all(|frame| {
            let meta = frame.meta_info.as_ref().unwrap();
            meta.input_token_logprobs
                .as_ref()
                .is_none_or(|entries| entries.entries.is_empty())
                && meta.input_top_logprobs.is_empty()
        }));
        assert_eq!(prompts.as_array().unwrap().len(), request.token_ids.len());
        assert_eq!(selected.len(), request.token_ids.len());
        assert_eq!(alternatives.len(), request.token_ids.len());
        assert!(prompts[0].is_null());
        for index in 1..request.token_ids.len() {
            for entry in std::iter::once(&selected[index])
                .chain(&alternatives[index].row.as_ref().unwrap().entries)
            {
                let token = entry.token_id.to_string();
                assert_eq!(prompts[index][token]["logprob"].as_f64(), entry.logprob);
            }
        }
    }

    async fn scheduler_active(&self) {
        let mut metrics = self.service.metrics_receiver();
        dynamo_sidecar_testkit::bounded("Mocker active scheduler work", async {
            loop {
                let snapshot = metrics.borrow_and_update().clone();
                if snapshot.running_requests + snapshot.waiting_requests > 0 {
                    assert!(self.service.active_request_count() > 0);
                    return;
                }
                metrics.changed().await.unwrap();
            }
        })
        .await;
    }
}

#[derive(Clone)]
struct ControlledService {
    inner: SglangMockerService,
    control: Controller<Adapter>,
    scripted: Arc<Mutex<HashMap<String, Vec<pb::GenerateStreamItem>>>>,
    model_info: Arc<Mutex<Option<ModelInfoPatch>>>,
    server_info: Arc<Mutex<VecDeque<ServerInfoPatch>>>,
    health: watch::Receiver<Option<bool>>,
    health_calls: watch::Sender<usize>,
    aborts: watch::Sender<Vec<(String, bool)>>,
}

#[tonic::async_trait]
impl SglangService for ControlledService {
    type GenerateStream = BoxStream<'static, Result<pb::GenerateStreamItem, Status>>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let opened = self.control.open(request.get_ref()).await?;
        let scripted = self
            .scripted
            .lock()
            .unwrap()
            .remove(Adapter::request_id(request.get_ref()));
        if let Some(items) = scripted {
            return Ok(Response::new(
                opened.wrap(Box::pin(futures::stream::iter(items.into_iter().map(Ok)))),
            ));
        }
        let response = self.inner.generate(request).await?;
        Ok(Response::new(opened.wrap(response.into_inner())))
    }

    async fn health_check(
        &self,
        _request: Request<pb::HealthCheckRequest>,
    ) -> Result<Response<pb::HealthCheckResponse>, Status> {
        self.health_calls.send_modify(|count| *count += 1);
        let mut health = self.health.clone();
        let healthy = health.wait_for(Option::is_some).await.unwrap().unwrap();
        if !healthy {
            // SGLang's Rust server reports an unready or stalled scheduler as
            // UNAVAILABLE; it never answers healthy=false.
            return Err(Status::unavailable(
                "no scheduler output within the health-check timeout",
            ));
        }
        Ok(Response::new(pb::HealthCheckResponse { healthy }))
    }

    async fn get_model_info(
        &self,
        request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::GetModelInfoResponse>, Status> {
        let mut response = self.inner.get_model_info(request).await?.into_inner();
        if let Some(patch) = self.model_info.lock().unwrap().as_ref() {
            patch(&mut response);
        }
        Ok(Response::new(response))
    }

    async fn get_server_info(
        &self,
        request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::GetServerInfoResponse>, Status> {
        let mut response = self.inner.get_server_info(request).await?.into_inner();
        let mut scripts = self.server_info.lock().unwrap();
        if let Some(patch) = scripts.front() {
            patch(&mut response);
        }
        if scripts.len() > 1 {
            scripts.pop_front();
        }
        Ok(Response::new(response))
    }

    async fn list_models(
        &self,
        request: Request<pb::ListModelsRequest>,
    ) -> Result<Response<pb::ListModelsResponse>, Status> {
        self.inner.list_models(request).await
    }

    async fn detokenize(
        &self,
        request: Request<pb::DetokenizeRequest>,
    ) -> Result<Response<pb::DetokenizeResponse>, Status> {
        self.inner.detokenize(request).await
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let rid = request.get_ref().rid.clone();
        let response = self.inner.abort(request).await?;
        let success = response.get_ref().success;
        self.aborts
            .send_modify(|aborts| aborts.push((rid, success)));
        Ok(response)
    }

    async fn get_hi_cache_storage(
        &self,
        request: Request<pb::GetHiCacheStorageRequest>,
    ) -> Result<Response<pb::HiCacheStatus>, Status> {
        self.inner.get_hi_cache_storage(request).await
    }

    async fn attach_hi_cache_storage(
        &self,
        request: Request<pb::AttachHiCacheStorageRequest>,
    ) -> Result<Response<pb::HiCacheStorageResponse>, Status> {
        self.inner.attach_hi_cache_storage(request).await
    }

    async fn detach_hi_cache_storage(
        &self,
        request: Request<pb::DetachHiCacheStorageRequest>,
    ) -> Result<Response<pb::HiCacheStorageResponse>, Status> {
        self.inner.detach_hi_cache_storage(request).await
    }

    async fn clear_hi_cache_storage(
        &self,
        request: Request<pb::ClearHiCacheStorageRequest>,
    ) -> Result<Response<pb::HiCacheStorageResponse>, Status> {
        self.inner.clear_hi_cache_storage(request).await
    }
}

#[derive(Clone, Copy)]
pub struct Adapter;

impl Protocol for Adapter {
    type Request = pb::GenerateRequest;
    type Response = pb::GenerateStreamItem;
    type Error = Status;

    fn request_id(request: &Self::Request) -> &str {
        match request.rid.as_ref().and_then(|rid| rid.value.as_ref()) {
            Some(pb::string_or_list::Value::One(rid)) => rid,
            other => panic!("sidecar request ID: {other:?}"),
        }
    }

    fn record_tokens(response: &Self::Response, tokens: &mut Vec<u32>) -> bool {
        let Some(frame) = frame(response) else {
            return false;
        };
        let ids = output_ids(frame);
        tokens.extend(ids.iter().map(|&id| u32::try_from(id).unwrap()));
        !ids.is_empty()
    }

    /// SGLang ends a request's stream after its terminal frame or error item.
    fn is_terminal(response: &Self::Response) -> bool {
        match response.item.as_ref() {
            Some(Item::Frame(frame)) => frame
                .meta_info
                .as_ref()
                .is_some_and(|meta| meta.finish_reason.is_some()),
            Some(Item::Error(_)) => true,
            None => false,
        }
    }

    fn injected_error(message: &'static str) -> Self::Error {
        Status::unavailable(message)
    }

    fn invalid_argument_error(message: &'static str) -> Self::Error {
        Status::invalid_argument(message)
    }
}

impl DiscoveryFixture for Fixture {
    fn set_served_model_name(&self, name: &str) {
        let name = name.to_string();
        self.override_discovery(
            None,
            vec![Box::new(move |info| info.served_model_name = name.clone())],
        );
    }

    fn set_health(&self, is_healthy: Option<bool>) {
        self.health.send_replace(is_healthy);
    }

    async fn health_check_received(&self) {
        let mut calls = self.health_calls.clone();
        dynamo_sidecar_testkit::bounded(
            "SGLang HealthCheck received",
            calls.wait_for(|count| *count > 0),
        )
        .await
        .unwrap();
    }

    fn assert_unhealthy_startup(logs: &str) {
        assert!(logs.contains("did not become healthy"), "{logs}");
        assert!(logs.contains("Unavailable"), "{logs}");
    }
}

impl ProcessFixture for Fixture {
    fn endpoint(&self) -> String {
        self.server.endpoint()
    }

    fn command(_model: &str) -> Command {
        let mut command = sidecar_command("dynamo-sglang-sidecar", "DYNAMO_SGLANG_SIDECAR");
        command.env("SGLANG_DISAGGREGATION_BOOTSTRAP_HOST", "127.0.0.1");
        command
    }

    fn configure_request(request: &mut PreprocessedRequest) {
        request.sampling_options.temperature = Some(0.125);
        request.sampling_options.presence_penalty = Some(0.25);
        request.sampling_options.frequency_penalty = Some(0.75);
        request.output_options.logprobs = Some(2);
        request.output_options.prompt_logprobs = Some(1);
        request.mdc_sum = Some("request-cache-key".into());
        // The router names rank 0, the engine's only DP rank.
        request.routing = Some(RoutingHints {
            lora_name: Some("test-adapter".into()),
            dp_rank: Some(0),
            ..Default::default()
        });
    }

    fn assert_registration(card: &ModelDeploymentCard) {
        assert_eq!(card.kv_cache_block_size, 4);
        assert_eq!(card.runtime_config.total_kv_blocks, Some(4096));
        assert_eq!(card.runtime_config.max_num_seqs, Some(64));
        assert_eq!(card.runtime_config.max_num_batched_tokens, Some(1024));
        assert_eq!(card.runtime_config.data_parallel_start_rank, 0);
        assert_eq!(card.runtime_config.data_parallel_size, 1);
        assert!(card.runtime_config.tool_call_parser.is_none());
        assert!(card.runtime_config.reasoning_parser.is_none());
        assert_eq!(card.effective_context_length(), 32_768);
        assert_eq!(
            card.runtime_config
                .get_engine_specific::<TokenBudget>(TOKEN_BUDGET_RUNTIME_KEY)
                .unwrap(),
            Some(TokenBudget {
                combined_limit: 32_768,
                reject_prompt_overflow: true,
                reject_total_overflow: true,
            })
        );
        assert!(
            !card
                .runtime_config
                .runtime_data
                .contains_key(TOOL_CALL_STRUCTURAL_TAG_REASONING_GATE_RUNTIME_KEY)
        );
        assert_eq!(
            card.runtime_config.runtime_data.get("grpc_service"),
            Some(&serde_json::json!("sglang.api.v1.SglangService"))
        );
    }
}

impl HandoffFixture for Fixture {
    const KV_BLOCK_SIZE: u32 = 4;
    const HAS_BOOTSTRAP: bool = true;

    fn assert_handoff(
        prefill: &RequestHandle<Self::Protocol>,
        decode: &RequestHandle<Self::Protocol>,
        id: &str,
    ) {
        let prefill_wire = prefill.native_request().unwrap();
        let decode_wire = decode.native_request().unwrap();
        assert_eq!(Adapter::request_id(&prefill_wire), id);
        assert_eq!(Adapter::request_id(&decode_wire), id);
        assert_eq!(sampling(&prefill_wire).max_new_tokens, Some(1));
        assert_eq!(sampling(&decode_wire).max_new_tokens, Some(3));
        let rendezvous = |wire: &pb::GenerateRequest| {
            (
                one_string(wire.bootstrap_host.as_ref()),
                one_optional_int64(wire.bootstrap_port.as_ref()),
                one_optional_int64(wire.bootstrap_room.as_ref()),
            )
        };
        let (host, port, room) = rendezvous(&prefill_wire);
        assert_eq!(rendezvous(&decode_wire), (host.clone(), port, room));
        assert!(host.is_some_and(|host| !host.is_empty()));
        assert_eq!(port, Some(8998));
        assert!(room.is_some_and(|room| room >= 0));
    }
}

pub mod http {
    use std::collections::HashMap;
    use std::io;
    use std::sync::{Arc, Mutex};

    use dynamo_sidecar_testkit::control::{Controller, Protocol};
    use dynamo_sidecar_testkit::server::TestServer;
    use futures::StreamExt;
    use serde_json::{Value, json};
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpStream;
    use tokio::sync::{Notify, watch};

    /// The Abort RPCs the gRPC fixture records, with whether each matched.
    /// SGLang's Rust server serves no `/abort_request`; the Abort RPC cancels
    /// its native requests. The peer, not the Mocker, serves them, so none
    /// matches.
    #[derive(Clone)]
    struct Aborts(watch::Receiver<Vec<(String, bool)>>);

    impl Aborts {
        fn ids(&self) -> Vec<String> {
            self.0.borrow().iter().map(|(id, _)| id.clone()).collect()
        }

        async fn wait(&self, request_id: &str) {
            self.0
                .clone()
                .wait_for(|aborts| aborts.iter().any(|(id, _)| id == request_id))
                .await
                .unwrap();
        }
    }

    pub struct Peer {
        server: TestServer,
        aborts: Aborts,
        held: Arc<Mutex<HashMap<String, Arc<ResponseGate>>>>,
    }

    pub struct ResponseGate {
        empty_sent: watch::Sender<bool>,
        release: Notify,
        responses: Vec<Value>,
    }

    impl ResponseGate {
        pub async fn wait_empty_response(&self) {
            dynamo_sidecar_testkit::bounded(
                "HTTP headers and empty response",
                self.empty_sent.subscribe().wait_for(|sent| *sent),
            )
            .await
            .unwrap();
        }

        pub fn release(&self) {
            self.release.notify_one();
        }
    }

    impl Peer {
        pub(crate) async fn start(
            control: Controller<Adapter>,
            aborts: watch::Receiver<Vec<(String, bool)>>,
        ) -> Self {
            let aborts = Aborts(aborts);
            let held = Arc::new(Mutex::new(HashMap::new()));
            let server_held = Arc::clone(&held);
            let server_aborts = aborts.clone();
            let server = TestServer::start(move |listener, _shutdown| async move {
                let mut connections = tokio::task::JoinSet::new();
                loop {
                    tokio::select! {
                        accepted = listener.accept() => {
                            let (socket, _) = accepted?;
                            connections.spawn(serve(socket, control.clone(), server_aborts.clone(), Arc::clone(&server_held)));
                        }
                        result = connections.join_next(), if !connections.is_empty() => {
                            result.unwrap()??;
                        }
                    }
                }
            })
            .await
            .unwrap();
            Self {
                server,
                aborts,
                held,
            }
        }

        pub fn hold_responses(&self, request_id: &str, responses: Vec<Value>) -> Arc<ResponseGate> {
            let gate = Arc::new(ResponseGate {
                empty_sent: watch::channel(false).0,
                release: Notify::new(),
                responses,
            });
            assert!(
                self.held
                    .lock()
                    .unwrap()
                    .insert(request_id.into(), Arc::clone(&gate))
                    .is_none()
            );
            gate
        }

        pub async fn wait_aborted(&self, request_id: &str) {
            dynamo_sidecar_testkit::bounded("native HTTP abort", self.aborts.wait(request_id))
                .await;
        }

        pub fn aborted_requests(&self) -> Vec<String> {
            self.aborts.ids()
        }

        pub fn port(&self) -> u16 {
            self.server
                .endpoint()
                .rsplit_once(':')
                .unwrap()
                .1
                .parse()
                .unwrap()
        }

        pub(crate) async fn shutdown(&mut self) {
            self.server.shutdown().await.unwrap();
        }
    }

    async fn serve(
        socket: TcpStream,
        control: Controller<Adapter>,
        aborts: Aborts,
        held: Arc<Mutex<HashMap<String, Arc<ResponseGate>>>>,
    ) -> anyhow::Result<()> {
        let mut reader = BufReader::new(socket);
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 {
            return Ok(());
        }
        // SGLang's Rust server serves no `/abort_request`.
        let is_health = line.starts_with("GET /health ");
        assert!(is_health || line.starts_with("POST /generate "), "{line}");
        let mut length = 0;
        loop {
            line.clear();
            if reader.read_line(&mut line).await? == 0 {
                return Ok(());
            }
            if line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse()?;
            }
        }
        if is_health {
            reader
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await?;
            return Ok(());
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body).await?;
        let request: Value = serde_json::from_slice(&body)?;
        let (mut reader, mut writer) = reader.into_inner().into_split();
        let gate = held.lock().unwrap().remove(Adapter::request_id(&request));
        let mut eof = [0];
        let mut has_terminal = false;
        tokio::select! {
            received = reader.read(&mut eof) => {
                assert_eq!(received?, 0, "unexpected bytes after /generate body");
                assert!(
                    has_terminal || aborts.ids().iter().any(|id| id == Adapter::request_id(&request)),
                    "unfinished generation disconnected before its abort request"
                );
            }
            result = async {
                let opened = control.open(&request).await?;
                writer.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").await?;
                let responses = if let Some(gate) = gate {
                    // The Rust server omits output_ids while a frame has no tokens.
                    let empty = json!({"text": "", "meta_info": {"finish_reason": null}});
                    writer.write_all(format!("data: {empty}\n\n").as_bytes()).await?;
                    gate.empty_sent.send_replace(true);
                    gate.release.notified().await;
                    gate.responses.clone()
                } else {
                    responses(Adapter::request_id(&request)).into()
                };
                let source = futures::stream::iter(responses.into_iter().map(Ok));
                let mut stream = opened.wrap(Box::pin(source));
                while let Some(response) = stream.next().await {
                    let response = response?;
                    has_terminal = Adapter::is_terminal(&response);
                    writer.write_all(format!("data: {response}\n\n").as_bytes()).await?;
                }
                Ok::<_, io::Error>(())
            } => result?,
        }
        Ok(())
    }

    pub fn responses(request_id: &str) -> [Value; 2] {
        [
            json!({"output_ids": [101], "text": "first", "meta_info": {"id": request_id, "finish_reason": null}}),
            json!({"output_ids": [102], "text": "last", "meta_info": {"id": request_id, "finish_reason": {"type": "length"}}}),
        ]
    }

    pub struct Adapter;

    impl Protocol for Adapter {
        type Request = Value;
        type Response = Value;
        type Error = io::Error;

        fn request_id(request: &Value) -> &str {
            request["rid"].as_str().unwrap()
        }

        fn record_tokens(response: &Value, tokens: &mut Vec<u32>) -> bool {
            // The Rust server omits output_ids while a frame has no tokens.
            let Some(ids) = response.get("output_ids").and_then(Value::as_array) else {
                return false;
            };
            tokens.extend(
                ids.iter()
                    .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap()),
            );
            !ids.is_empty()
        }

        /// SGLang ends a request's stream after its finish or error.
        fn is_terminal(response: &Value) -> bool {
            response.get("error").is_some() || !response["meta_info"]["finish_reason"].is_null()
        }

        fn injected_error(message: &'static str) -> io::Error {
            io::Error::other(message)
        }

        fn invalid_argument_error(message: &'static str) -> io::Error {
            io::Error::new(io::ErrorKind::InvalidInput, message)
        }
    }
}
