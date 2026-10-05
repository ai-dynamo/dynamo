// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::process::Command;
use std::sync::{Arc, Mutex};

use dynamo_backend_common::{BackendError, DisaggregationMode, PreprocessedRequest};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::common::preprocessor::RoutingHints;
use dynamo_mocker::common::protocols::EngineType;
use dynamo_sglang_mocker::{MockerServerConfig, ServerMode, SglangMockerService};
use dynamo_sglang_sidecar::SglangSidecarEngine;
use dynamo_sglang_sidecar::proto::{
    self as pb,
    sglang_service_server::{SglangService, SglangServiceServer},
};
use dynamo_sidecar_testkit::control::{Controller, Protocol, RequestHandle};
use dynamo_sidecar_testkit::fixtures::Outputs;
use dynamo_sidecar_testkit::server::TestServer;
use futures::stream::BoxStream;
use serde_json::Value;
use tokio::sync::watch;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use super::{
    FixtureConfig, GenerateOpening, HandoffFixture, ProcessFixture, SidecarFixture, WireFixture,
    fast_engine_args, sidecar_command, wait_scheduler_idle,
};

pub struct Fixture {
    config: FixtureConfig,
    service: SglangMockerService,
    pub(crate) server: TestServer,
    scripted: Arc<Mutex<HashMap<String, Vec<pb::GenerateResponse>>>>,
    model_info: Arc<Mutex<Value>>,
    server_info: Arc<Mutex<VecDeque<Value>>>,
    health: watch::Sender<Option<bool>>,
    health_calls: watch::Receiver<usize>,
}

impl Fixture {
    pub fn respond(&self, request_id: &str, responses: Vec<pb::GenerateResponse>) {
        assert!(
            self.scripted
                .lock()
                .unwrap()
                .insert(request_id.to_owned(), responses)
                .is_none()
        );
    }

    pub fn override_discovery(&self, model_info: Value, server_info: Vec<Value>) {
        *self.model_info.lock().unwrap() = model_info;
        *self.server_info.lock().unwrap() = server_info.into();
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
        let model_info = Arc::new(Mutex::new(Value::Null));
        let server_info = Arc::new(Mutex::new(VecDeque::new()));
        let (health, health_rx) = watch::channel(Some(true));
        let (health_tx, health_calls) = watch::channel(0);
        let controlled = ControlledService {
            inner: service.clone(),
            control,
            scripted: scripted.clone(),
            model_info: model_info.clone(),
            server_info: server_info.clone(),
            health: health_rx,
            health_calls: health_tx,
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
        }
    }

    async fn engine(&self) -> Self::Engine {
        let argv = vec![
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
        // SGLang's tokenized generation RPC has no model selector.
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
            native.input_ids,
            request
                .token_ids
                .iter()
                .map(|&token| i32::try_from(token).unwrap())
                .collect::<Vec<_>>()
        );
        let sampling = native.sampling_params.as_ref().unwrap();
        assert_eq!(
            sampling.max_new_tokens,
            request.stop_conditions.max_tokens.map(|value| value as i32)
        );
        assert_eq!(sampling.temperature, request.sampling_options.temperature);
        assert_eq!(sampling.top_p, request.sampling_options.top_p);
        assert_eq!(sampling.top_k, request.sampling_options.top_k);
        assert_eq!(sampling.min_p, request.sampling_options.min_p);
        assert_eq!(
            sampling.presence_penalty,
            request.sampling_options.presence_penalty
        );
        assert_eq!(
            sampling.frequency_penalty,
            request.sampling_options.frequency_penalty
        );
        assert_eq!(
            sampling.repetition_penalty,
            request.sampling_options.repetition_penalty
        );
        assert_eq!(
            sampling.min_new_tokens,
            request.stop_conditions.min_tokens.map(|value| value as i32)
        );
        assert_eq!(
            sampling.stop,
            request.stop_conditions.stop.clone().unwrap_or_default()
        );
        let stop_ids = request
            .stop_conditions
            .stop_token_ids
            .iter()
            .chain(&request.stop_conditions.stop_token_ids_hidden)
            .flatten()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(sampling.stop_token_ids.len(), stop_ids.len());
        assert_eq!(
            sampling
                .stop_token_ids
                .iter()
                .map(|&value| value as u32)
                .collect::<std::collections::BTreeSet<_>>(),
            stop_ids
        );
        assert_eq!(sampling.ignore_eos, request.stop_conditions.ignore_eos);
        assert_eq!(native.stream, Some(true));
        assert_eq!(native.return_logprob, Some(true));
        assert_eq!(native.top_logprobs_num, Some(2));
        assert_eq!(native.logprob_start_len, Some(0));
        assert_eq!(native.routing_key, request.mdc_sum);
        assert_eq!(
            native.lora_path,
            request
                .routing
                .as_ref()
                .and_then(|routing| routing.lora_name.clone())
        );
        assert_eq!(
            native.routed_dp_rank,
            request
                .routing
                .as_ref()
                .and_then(|routing| routing.dp_rank)
                .map(|rank| rank as i32)
        );

        let native_outputs = handle.native_responses();
        let outputs: Vec<_> = outputs
            .iter()
            .map(|output| output.as_ref().unwrap())
            .collect();
        assert_eq!(outputs.len(), native_outputs.len());
        assert_eq!(
            outputs.len(),
            request.stop_conditions.max_tokens.unwrap() as usize
        );
        assert!(outputs.iter().all(|output| output.token_ids.len() == 1));
        for (output, native) in outputs.iter().zip(&native_outputs) {
            assert_eq!(
                output.token_ids,
                native
                    .output_ids
                    .iter()
                    .map(|&token| u32::try_from(token).unwrap())
                    .collect::<Vec<_>>()
            );
            assert!(output.text.is_none());
            let selected: Vec<serde_json::Value> =
                serde_json::from_str(&native.meta_info["output_token_logprobs"]).unwrap();
            assert_eq!(
                output.log_probs.as_ref().unwrap(),
                &selected
                    .iter()
                    .map(|entry| entry[0].as_f64().unwrap())
                    .collect::<Vec<_>>()
            );
            let alternatives: Vec<Vec<serde_json::Value>> =
                serde_json::from_str(&native.meta_info["output_top_logprobs"]).unwrap();
            let actual = output.top_logprobs.as_ref().unwrap();
            assert_eq!(actual.len(), native.output_ids.len());
            assert_eq!(actual.len(), alternatives.len());
            for (actual, expected) in actual.iter().zip(&alternatives) {
                assert_eq!(actual.len(), 2);
                assert_eq!(actual.len(), expected.len());
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert_eq!(actual.token_id as u64, expected[1].as_u64().unwrap());
                    assert_eq!(actual.logprob, expected[0].as_f64().unwrap());
                    assert_eq!(actual.rank, (index + 1) as u32);
                }
            }
        }
        assert!(
            outputs[..outputs.len() - 1]
                .iter()
                .all(|output| output.engine_data.is_none())
        );
        let prompts = &outputs.last().unwrap().engine_data.as_ref().unwrap()["prompt_logprobs"];
        let terminal = native_outputs.last().unwrap();
        let selected: Vec<serde_json::Value> =
            serde_json::from_str(&terminal.meta_info["input_token_logprobs"]).unwrap();
        let alternatives: Vec<serde_json::Value> =
            serde_json::from_str(&terminal.meta_info["input_top_logprobs"]).unwrap();
        assert_eq!(prompts.as_array().unwrap().len(), request.token_ids.len());
        assert_eq!(selected.len(), request.token_ids.len());
        assert_eq!(alternatives.len(), request.token_ids.len());
        assert!(prompts[0].is_null());
        for index in 1..request.token_ids.len() {
            for entry in
                std::iter::once(&selected[index]).chain(alternatives[index].as_array().unwrap())
            {
                let token = entry[1].as_u64().unwrap().to_string();
                assert_eq!(prompts[index][token]["logprob"], entry[0]);
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
    scripted: Arc<Mutex<HashMap<String, Vec<pb::GenerateResponse>>>>,
    model_info: Arc<Mutex<Value>>,
    server_info: Arc<Mutex<VecDeque<Value>>>,
    health: watch::Receiver<Option<bool>>,
    health_calls: watch::Sender<usize>,
}

fn patch_metadata(raw: &mut String, patch: &Value) {
    if let Some(patch) = patch.as_object() {
        let mut metadata: Value = serde_json::from_str(raw).unwrap();
        metadata.as_object_mut().unwrap().extend(patch.clone());
        *raw = metadata.to_string();
    }
}

macro_rules! delegate_service {
    ($($method:ident($request:ty) -> $response:ty;)*) => {
        #[tonic::async_trait]
        impl SglangService for ControlledService {
            type TextGenerateStream = <SglangMockerService as SglangService>::TextGenerateStream;
            type GenerateStream = BoxStream<'static, Result<pb::GenerateResponse, Status>>;
            type ChatCompleteStream = <SglangMockerService as SglangService>::ChatCompleteStream;
            type CompleteStream = <SglangMockerService as SglangService>::CompleteStream;

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
                if let Some(responses) = scripted {
                    return Ok(Response::new(opened.wrap(Box::pin(futures::stream::iter(
                        responses.into_iter().map(Ok),
                    )))));
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
                Ok(Response::new(pb::HealthCheckResponse { healthy }))
            }

            async fn get_model_info(
                &self,
                request: Request<pb::GetModelInfoRequest>,
            ) -> Result<Response<pb::GetModelInfoResponse>, Status> {
                let mut response = self.inner.get_model_info(request).await?.into_inner();
                patch_metadata(&mut response.json_info, &self.model_info.lock().unwrap());
                Ok(Response::new(response))
            }

            async fn get_server_info(
                &self,
                request: Request<pb::GetServerInfoRequest>,
            ) -> Result<Response<pb::GetServerInfoResponse>, Status> {
                let mut response = self.inner.get_server_info(request).await?.into_inner();
                let mut scripts = self.server_info.lock().unwrap();
                if let Some(patch) = scripts.front() {
                    patch_metadata(&mut response.json_info, patch);
                }
                if scripts.len() > 1 {
                    scripts.pop_front();
                }
                Ok(Response::new(response))
            }

            $(
                async fn $method(
                    &self,
                    request: Request<$request>,
                ) -> Result<Response<$response>, Status> {
                    self.inner.$method(request).await
                }
            )*
        }
    };
}

delegate_service! {
    text_generate(pb::TextGenerateRequest) -> Self::TextGenerateStream;
    text_embed(pb::TextEmbedRequest) -> pb::TextEmbedResponse;
    embed(pb::EmbedRequest) -> pb::EmbedResponse;
    classify(pb::ClassifyRequest) -> pb::ClassifyResponse;
    tokenize(pb::TokenizeRequest) -> pb::TokenizeResponse;
    detokenize(pb::DetokenizeRequest) -> pb::DetokenizeResponse;
    list_models(pb::ListModelsRequest) -> pb::ListModelsResponse;
    get_load(pb::GetLoadRequest) -> pb::GetLoadResponse;
    abort(pb::AbortRequest) -> pb::AbortResponse;
    flush_cache(pb::FlushCacheRequest) -> pb::FlushCacheResponse;
    pause_generation(pb::PauseGenerationRequest) -> pb::PauseGenerationResponse;
    continue_generation(pb::ContinueGenerationRequest) -> pb::ContinueGenerationResponse;
    chat_complete(pb::OpenAiRequest) -> Self::ChatCompleteStream;
    complete(pb::OpenAiRequest) -> Self::CompleteStream;
    open_ai_embed(pb::OpenAiRequest) -> pb::OpenAiResponse;
    open_ai_classify(pb::OpenAiRequest) -> pb::OpenAiResponse;
    score(pb::OpenAiRequest) -> pb::OpenAiResponse;
    rerank(pb::OpenAiRequest) -> pb::OpenAiResponse;
    start_profile(pb::StartProfileRequest) -> pb::StartProfileResponse;
    stop_profile(pb::StopProfileRequest) -> pb::StopProfileResponse;
    update_weights_from_disk(pb::UpdateWeightsRequest) -> pb::UpdateWeightsResponse;
}

#[derive(Clone, Copy)]
pub struct Adapter;

impl Protocol for Adapter {
    type Request = pb::GenerateRequest;
    type Response = pb::GenerateResponse;
    type Error = Status;

    fn request_id(request: &Self::Request) -> &str {
        request.rid.as_deref().expect("sidecar request ID")
    }

    fn record_tokens(response: &Self::Response, tokens: &mut Vec<u32>) -> bool {
        tokens.extend(
            response
                .output_ids
                .iter()
                .map(|&id| u32::try_from(id).unwrap()),
        );
        !response.output_ids.is_empty()
    }

    fn is_terminal(response: &Self::Response) -> bool {
        response.finished
    }

    fn injected_error(message: &'static str) -> Self::Error {
        Status::unavailable(message)
    }
}

impl ProcessFixture for Fixture {
    fn set_served_model_name(&self, name: &str) {
        self.override_discovery(
            Value::Null,
            vec![serde_json::json!({"served_model_name": name})],
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
    }

    fn endpoint(&self) -> String {
        self.server.endpoint()
    }

    fn command() -> Command {
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
        request.routing = Some(RoutingHints {
            lora_name: Some("test-adapter".into()),
            dp_rank: Some(3),
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
    }
}

impl HandoffFixture for Fixture {
    const HAS_BOOTSTRAP: bool = true;

    fn assert_handoff(
        prefill: &RequestHandle<Self::Protocol>,
        decode: &RequestHandle<Self::Protocol>,
        id: &str,
    ) {
        let prefill_wire = prefill.native_request().unwrap();
        let decode_wire = decode.native_request().unwrap();
        assert_eq!(prefill_wire.rid.as_deref(), Some(id));
        assert_eq!(decode_wire.rid.as_deref(), Some(id));
        assert_eq!(
            prefill_wire.sampling_params.unwrap().max_new_tokens,
            Some(1)
        );
        assert_eq!(decode_wire.sampling_params.unwrap().max_new_tokens, Some(3));
        let bootstrap = prefill_wire.disaggregated_params.unwrap();
        assert_eq!(decode_wire.disaggregated_params.as_ref(), Some(&bootstrap));
        assert!(!bootstrap.bootstrap_host.is_empty());
        assert_eq!(bootstrap.bootstrap_port, 8998);
        assert!(bootstrap.bootstrap_room >= 0);
    }
}
