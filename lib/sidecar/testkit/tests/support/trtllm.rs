// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use dynamo_backend_common::{BackendError, DisaggregationMode, PreprocessedRequest};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_mocker::common::protocols::EngineType;
use dynamo_sidecar_testkit::control::{Controller, Protocol, RequestHandle};
use dynamo_sidecar_testkit::fixtures::Outputs;
use dynamo_sidecar_testkit::server::TestServer;
use dynamo_trtllm_mocker::{MockerServerConfig, ServerMode, TrtllmMockerService};
use dynamo_trtllm_sidecar::TrtllmSidecarEngine;
use dynamo_trtllm_sidecar::proto::{
    self as pb,
    control_server::{Control, ControlServer},
    encoder_server::{Encoder, EncoderServer},
    inference_server::{Inference, InferenceServer},
};
use futures::stream::BoxStream;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use super::{
    FixtureConfig, GenerateOpening, HandoffFixture, ProcessFixture, SidecarFixture, WireFixture,
    fast_engine_args, sidecar_command, wait_scheduler_active, wait_scheduler_idle,
};

pub struct Fixture {
    config: FixtureConfig,
    service: TrtllmMockerService,
    server: TestServer,
    scripted: Arc<Mutex<HashMap<String, Vec<pb::GenerateResponse>>>>,
    pub discovery: Arc<Discovery>,
    pub encoder: Arc<EncoderControls>,
    health: tonic_health::server::HealthReporter,
}

#[derive(Default)]
pub struct Discovery {
    pub calls: AtomicUsize,
    pub server_info_calls: AtomicUsize,
    pub server_info_hold_after: AtomicUsize,
    pub server_info_hold: AtomicBool,
    pub server_info_blocked: AtomicUsize,
    pub load_calls: AtomicUsize,
    pub kv_source_calls: AtomicUsize,
    pub server_info: Mutex<Option<pb::ServerInfo>>,
    pub server_info_sequence: Mutex<VecDeque<pb::ServerInfo>>,
    pub unavailable: AtomicBool,
    pub hold: AtomicBool,
    pub model_info: Mutex<Option<pb::ModelInfo>>,
    pub omit_context_length: AtomicBool,
    pub error: Mutex<Option<tonic::Code>>,
    release: tokio::sync::Notify,
}

#[derive(Default)]
pub struct EncoderControls {
    pub encode: Controller<EncodeAdapter>,
    pub release: Controller<ReleaseAdapter>,
    pub response: Mutex<Option<pb::EncodeResponse>>,
    pub aborts: Mutex<Vec<pb::AbortRequest>>,
    release_sequence: AtomicUsize,
}

pub struct EncodeAdapter;

impl Protocol for EncodeAdapter {
    type Request = pb::EncodeRequest;
    type Response = pb::EncodeResponse;
    type Error = Status;

    fn request_id(request: &Self::Request) -> &str {
        &request.request_id
    }
    fn record_tokens(_: &Self::Response, _: &mut Vec<u32>) -> bool {
        false
    }
    fn is_terminal(_: &Self::Response) -> bool {
        true
    }
    fn injected_error(message: &'static str) -> Status {
        Adapter::injected_error(message)
    }
    fn invalid_argument_error(message: &'static str) -> Status {
        Adapter::invalid_argument_error(message)
    }
}

#[derive(Clone)]
pub struct ReleaseRequest {
    key: String,
    pub native: pb::ReleaseEncodedMediaRequest,
}

pub struct ReleaseAdapter;

impl Protocol for ReleaseAdapter {
    type Request = ReleaseRequest;
    type Response = pb::ReleaseEncodedMediaResponse;
    type Error = Status;

    fn request_id(request: &Self::Request) -> &str {
        &request.key
    }
    fn record_tokens(_: &Self::Response, _: &mut Vec<u32>) -> bool {
        false
    }
    fn is_terminal(_: &Self::Response) -> bool {
        true
    }
    fn injected_error(message: &'static str) -> Status {
        Adapter::injected_error(message)
    }
    fn invalid_argument_error(message: &'static str) -> Status {
        Adapter::invalid_argument_error(message)
    }
}

impl Fixture {
    /// Sets what `Watch` reports for TensorRT-LLM's `liveness` health.
    pub async fn set_liveness_serving(&self, is_serving: bool) {
        let status = if is_serving {
            tonic_health::ServingStatus::Serving
        } else {
            tonic_health::ServingStatus::NotServing
        };
        self.health.set_service_status("liveness", status).await;
    }

    pub fn release_discovery(&self) {
        self.discovery.hold.store(false, Ordering::SeqCst);
        self.discovery.release.notify_waiters();
    }

    pub fn engine_with_context_length(&self, context_length: u32) -> TrtllmSidecarEngine {
        self.make_engine(Some(context_length), &[])
    }

    pub fn engine_with_args(&self, extra: &[&str]) -> TrtllmSidecarEngine {
        self.make_engine(None, extra)
    }

    fn make_engine(&self, context_length: Option<u32>, extra: &[&str]) -> TrtllmSidecarEngine {
        let mut args = vec![
            "dynamo-trtllm-sidecar".into(),
            "--grpc-endpoint".into(),
            self.server.endpoint(),
            "--model-path".into(),
            self.config.model.clone(),
            "--disaggregation-mode".into(),
            self.config.disaggregation_mode.to_string(),
            "--grpc-connections".into(),
            self.config.connections.to_string(),
            "--grpc-startup-deadline-secs".into(),
            "5".into(),
            "--grpc-connect-attempt-timeout-secs".into(),
            "1".into(),
        ];
        if let Some(context_length) = context_length {
            args.extend(["--context-length".into(), context_length.to_string()]);
        }
        args.extend(extra.iter().map(|arg| (*arg).to_owned()));
        TrtllmSidecarEngine::from_args(args).unwrap().0
    }

    pub fn respond(&self, request_id: &str, responses: Vec<pb::GenerateResponse>) {
        assert!(
            self.scripted
                .lock()
                .unwrap()
                .insert(request_id.to_owned(), responses)
                .is_none()
        );
    }
}

impl SidecarFixture for Fixture {
    type Engine = TrtllmSidecarEngine;
    type Protocol = Adapter;
    const GENERATE_OPENING: GenerateOpening = GenerateOpening::WaitsForHeaders;

    async fn start(control: Controller<Adapter>, config: FixtureConfig) -> Self {
        let mut args = fast_engine_args(EngineType::Trtllm);
        args.speedup_ratio = config.speedup_ratio;
        let service = TrtllmMockerService::new(
            MockerServerConfig {
                model: config.model.clone(),
                mode: match config.disaggregation_mode {
                    DisaggregationMode::Aggregated => ServerMode::Aggregated,
                    DisaggregationMode::Prefill => ServerMode::Prefill,
                    DisaggregationMode::Decode => ServerMode::Decode,
                    DisaggregationMode::Encode => ServerMode::Aggregated,
                },
                ..Default::default()
            },
            args,
        )
        .unwrap();
        let scripted = Arc::new(Mutex::new(HashMap::new()));
        let encoder = Arc::new(EncoderControls::default());
        let controlled = ControlledInference {
            scripted: scripted.clone(),
            inner: service.clone(),
            control,
        };
        let discovery = Arc::new(Discovery::default());
        let control_service = ControlledControl {
            inner: service.clone(),
            discovery: discovery.clone(),
            encoder: encoder.clone(),
            is_encoder: config.disaggregation_mode.is_encode(),
        };
        let encoder_service = ControlledEncoder {
            encoder: encoder.clone(),
        };
        let (health, health_service) = tonic_health::server::health_reporter();
        health
            .set_service_status("liveness", tonic_health::ServingStatus::Serving)
            .await;
        let server = TestServer::start(move |listener, shutdown| async move {
            tonic::transport::Server::builder()
                .add_service(health_service)
                .add_service(InferenceServer::new(controlled))
                .add_service(ControlServer::new(control_service))
                .add_service(EncoderServer::new(encoder_service))
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
            discovery,
            encoder,
            health,
        }
    }

    async fn engine(&self) -> Self::Engine {
        self.make_engine(None, &[])
    }

    fn eof_error() -> BackendError {
        BackendError::StreamIncomplete
    }

    fn native_model(request: &pb::GenerateRequest) -> Option<&str> {
        Some(&request.model)
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
    }
}

struct ControlledControl {
    inner: TrtllmMockerService,
    discovery: Arc<Discovery>,
    encoder: Arc<EncoderControls>,
    is_encoder: bool,
}

#[tonic::async_trait]
impl Control for ControlledControl {
    async fn shutdown(
        &self,
        _request: Request<pb::ShutdownRequest>,
    ) -> Result<Response<pb::ShutdownResponse>, Status> {
        Err(Status::unimplemented("engine shutdown is not implemented"))
    }

    async fn get_model_info(
        &self,
        request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        let released = self.discovery.release.notified();
        self.discovery.calls.fetch_add(1, Ordering::SeqCst);
        if self.discovery.hold.load(Ordering::SeqCst) {
            released.await;
        }
        if self.discovery.unavailable.load(Ordering::SeqCst) {
            return Err(Status::unavailable("discovery temporarily unavailable"));
        }
        if let Some(code) = *self.discovery.error.lock().unwrap() {
            return Err(Status::new(code, "model discovery rejected"));
        }
        let model_info = self.discovery.model_info.lock().unwrap().clone();
        let mut response = if let Some(info) = model_info {
            Response::new(info)
        } else {
            self.inner.get_model_info(request).await?
        };
        if self.discovery.omit_context_length.load(Ordering::SeqCst) {
            response.get_mut().max_context_length = None;
        }
        Ok(response)
    }

    async fn get_server_info(
        &self,
        request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        let call = self
            .discovery
            .server_info_calls
            .fetch_add(1, Ordering::SeqCst)
            + 1;
        let hold_after = self.discovery.server_info_hold_after.load(Ordering::SeqCst);
        if self.discovery.server_info_hold.load(Ordering::SeqCst)
            || (hold_after > 0 && call > hold_after)
        {
            self.discovery
                .server_info_blocked
                .fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
        }
        if let Some(info) = self
            .discovery
            .server_info_sequence
            .lock()
            .unwrap()
            .pop_front()
        {
            return Ok(Response::new(info));
        }
        if let Some(info) = self.discovery.server_info.lock().unwrap().clone() {
            return Ok(Response::new(info));
        }
        let mut response = self.inner.get_server_info(request).await?;
        if self.is_encoder {
            response.get_mut().engine_role = pb::EngineRole::Encoder as i32;
            response.get_mut().schema_release =
                include_str!("../../../trtllm/proto/REVISION").trim().into();
        }
        Ok(response)
    }
    async fn get_load(
        &self,
        request: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
        self.discovery.load_calls.fetch_add(1, Ordering::SeqCst);
        if self.is_encoder {
            return Err(Status::unimplemented("encoder has no KV load"));
        }
        self.inner.get_load(request).await
    }
    async fn health(
        &self,
        request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        self.inner.health(request).await
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        self.encoder
            .aborts
            .lock()
            .unwrap()
            .push(request.get_ref().clone());
        self.inner.abort(request).await
    }
    async fn load_lora(
        &self,
        request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        self.inner.load_lora(request).await
    }
    async fn unload_lora(
        &self,
        request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        self.inner.unload_lora(request).await
    }
    async fn list_loras(
        &self,
        request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        self.inner.list_loras(request).await
    }
    async fn get_kv_event_sources(
        &self,
        request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        self.discovery
            .kv_source_calls
            .fetch_add(1, Ordering::SeqCst);
        if self.is_encoder {
            return Err(Status::invalid_argument("encoder has no KV events"));
        }
        self.inner.get_kv_event_sources(request).await
    }
    type SubscribeKvEventsStream = <TrtllmMockerService as Control>::SubscribeKvEventsStream;
    async fn subscribe_kv_events(
        &self,
        request: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        self.inner.subscribe_kv_events(request).await
    }
}

impl WireFixture for Fixture {
    fn assert_stream(
        handle: &RequestHandle<Adapter>,
        request: &PreprocessedRequest,
        outputs: &Outputs,
    ) {
        let native = handle.native_request().unwrap();
        assert_eq!(native.model, request.model);
        assert_eq!(
            native.input,
            Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                ids: request.token_ids.as_ref().clone(),
            }))
        );
        assert_eq!(
            native.stopping.as_ref().unwrap().max_tokens,
            request.stop_conditions.max_tokens
        );
        assert_eq!(
            native.sampling.as_ref().unwrap().temperature,
            request.sampling_options.temperature.map(f64::from)
        );
        assert_eq!(
            native.stopping.as_ref().unwrap().ignore_eos,
            request.stop_conditions.ignore_eos
        );
        let sampling = native.sampling.as_ref().unwrap();
        assert_eq!(sampling.num_sequences, Some(1));
        assert_eq!(
            sampling.presence_penalty,
            request.sampling_options.presence_penalty.map(f64::from)
        );
        assert_eq!(
            sampling.frequency_penalty,
            request.sampling_options.frequency_penalty.map(f64::from)
        );
        let extra = dynamo_sidecar_common::struct_to_json(
            native.extra.unwrap(),
            "TensorRT-LLM",
            "request extensions",
        )
        .unwrap();
        assert_eq!(extra["detokenize"], false);
        let response = native.response.as_ref().unwrap();
        assert_eq!(response.return_output_logprobs, Some(true));
        assert_eq!(response.return_prompt_logprobs, None);
        assert_eq!(
            response.output_candidates.as_ref().unwrap().selection,
            Some(pb::candidate_token_selection::Selection::TopN(2))
        );

        let native_responses = handle.native_responses();
        let tokens: Vec<_> = native_responses
            .iter()
            .filter_map(|response| match &response.event {
                Some(pb::generate_response::Event::Token(output)) => Some(output),
                _ => None,
            })
            .collect();
        let outputs: Vec<_> = outputs
            .iter()
            .map(|output| output.as_ref().unwrap())
            .collect();
        assert_eq!(outputs.len(), tokens.len() + 1);
        assert_eq!(
            tokens.len(),
            request.stop_conditions.max_tokens.unwrap() as usize
        );
        for (output, native) in outputs.iter().zip(tokens) {
            assert_eq!(
                output.token_ids,
                native
                    .tokens
                    .iter()
                    .map(|token| token.token_id)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                output.log_probs.as_ref().unwrap(),
                &native
                    .tokens
                    .iter()
                    .map(|token| token.logprob.unwrap())
                    .collect::<Vec<_>>()
            );
            let alternatives = output.top_logprobs.as_ref().unwrap();
            assert_eq!(alternatives.len(), native.tokens.len());
            for (actual, token) in alternatives.iter().zip(&native.tokens) {
                assert_eq!(actual.len(), 2);
                assert_eq!(actual.len(), token.candidates.len());
                for (actual, expected) in actual.iter().zip(&token.candidates) {
                    assert_eq!(actual.token_id, expected.token_id);
                    assert_eq!(actual.logprob, expected.logprob);
                    assert_eq!(actual.rank, expected.rank.unwrap());
                }
            }
        }
        assert!(outputs.last().unwrap().token_ids.is_empty());
    }

    async fn scheduler_active(&self) {
        wait_scheduler_active(self, self.service.metrics_receiver()).await;
    }
}

#[derive(Clone)]
struct ControlledEncoder {
    encoder: Arc<EncoderControls>,
}

#[tonic::async_trait]
impl Encoder for ControlledEncoder {
    async fn encode(
        &self,
        request: Request<pb::EncodeRequest>,
    ) -> Result<Response<pb::EncodeResponse>, Status> {
        let opened = self.encoder.encode.open(request.get_ref()).await?;
        let response = self
            .encoder
            .response
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| pb::EncodeResponse {
                request_id: request.get_ref().request_id.clone(),
                encoded_media: Some(pb::EncodedMediaRef {
                    opaque_state: vec![1, 2, 3],
                }),
            });
        Ok(Response::new(opened.finish(response)))
    }

    async fn release_encoded_media(
        &self,
        request: Request<pb::ReleaseEncodedMediaRequest>,
    ) -> Result<Response<pb::ReleaseEncodedMediaResponse>, Status> {
        let key = self
            .encoder
            .release_sequence
            .fetch_add(1, Ordering::SeqCst)
            .to_string();
        let request = ReleaseRequest {
            key,
            native: request.into_inner(),
        };
        let opened = self.encoder.release.open(&request).await?;
        Ok(Response::new(
            opened.finish(pb::ReleaseEncodedMediaResponse {}),
        ))
    }
}

#[derive(Clone)]
struct ControlledInference {
    scripted: Arc<Mutex<HashMap<String, Vec<pb::GenerateResponse>>>>,
    inner: TrtllmMockerService,
    control: Controller<Adapter>,
}

#[tonic::async_trait]
impl Inference for ControlledInference {
    type GenerateStream = BoxStream<'static, Result<pb::GenerateResponse, Status>>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let opened = self.control.open(request.get_ref()).await?;
        let scripted = self
            .scripted
            .lock()
            .unwrap()
            .remove(&request.get_ref().request_id);
        if let Some(responses) = scripted {
            return Ok(Response::new(opened.wrap(Box::pin(futures::stream::iter(
                responses.into_iter().map(Ok),
            )))));
        }
        let response = self.inner.generate(request).await?;
        Ok(Response::new(opened.wrap(response.into_inner())))
    }
}

#[derive(Clone, Copy)]
pub struct Adapter;

impl Protocol for Adapter {
    type Request = pb::GenerateRequest;
    type Response = pb::GenerateResponse;
    type Error = Status;

    fn request_id(request: &Self::Request) -> &str {
        &request.request_id
    }

    fn record_tokens(response: &Self::Response, tokens: &mut Vec<u32>) -> bool {
        if let Some(pb::generate_response::Event::Token(output)) = &response.event {
            tokens.extend(output.tokens.iter().map(|token| token.token_id));
            !output.tokens.is_empty()
        } else {
            false
        }
    }

    fn is_terminal(response: &Self::Response) -> bool {
        matches!(
            response.event,
            Some(
                pb::generate_response::Event::Finished(_)
                    | pb::generate_response::Event::PrefillReady(_)
                    | pb::generate_response::Event::Error(_)
            )
        )
    }

    fn injected_error(message: &'static str) -> Self::Error {
        Status::unavailable(message)
    }

    fn invalid_argument_error(message: &'static str) -> Self::Error {
        Status::invalid_argument(message)
    }
}

impl ProcessFixture for Fixture {
    fn endpoint(&self) -> String {
        self.server.endpoint()
    }

    fn command(model: &str) -> Command {
        let mut command = sidecar_command("dynamo-trtllm-sidecar", "DYNAMO_TRTLLM_SIDECAR");
        command.args(["--model-path", model]);
        command
    }

    fn configure_request(request: &mut PreprocessedRequest) {
        request.sampling_options.temperature = Some(0.125);
        request.sampling_options.presence_penalty = Some(0.25);
        request.sampling_options.frequency_penalty = Some(0.75);
        request.output_options.logprobs = Some(2);
    }

    fn assert_registration(card: &ModelDeploymentCard) {
        // Registration must preserve the Mocker's advertised cache capacity.
        assert_eq!(card.effective_context_length(), 32_768);
        assert_eq!(card.kv_cache_block_size, 4);
        assert_eq!(card.runtime_config.total_kv_blocks, Some(4_096));
        assert!(card.runtime_config.tool_call_parser.is_none());
        assert!(card.runtime_config.reasoning_parser.is_none());
    }
}

impl HandoffFixture for Fixture {
    const HAS_BOOTSTRAP: bool = false;
    const KV_BLOCK_SIZE: u32 = 4;

    fn assert_handoff(prefill: &RequestHandle<Adapter>, decode: &RequestHandle<Adapter>, id: &str) {
        let prefill_wire = prefill.native_request().unwrap();
        let decode_wire = decode.native_request().unwrap();
        assert_eq!(prefill_wire.request_id, id);
        assert_eq!(decode_wire.request_id, id);
        assert_eq!(prefill_wire.stopping.unwrap().max_tokens, Some(1));
        assert_eq!(decode_wire.stopping.unwrap().max_tokens, Some(3));
        let session = prefill
            .native_responses()
            .into_iter()
            .find_map(|response| match response.event {
                Some(pb::generate_response::Event::PrefillReady(ready)) => ready.kv_session,
                _ => None,
            })
            .expect("prefill must return a KV session");
        assert!(!session.session_id.is_empty());
        assert_eq!(session.transfer_backend, "MOCKER");
        assert_eq!(decode_wire.kv.unwrap().session.unwrap(), session);
    }
}
