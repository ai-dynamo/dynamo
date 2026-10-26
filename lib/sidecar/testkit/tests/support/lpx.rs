// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An LPX engine's OpenEngine (`openengine.v1`) server, played by the
//! TensorRT-LLM Mocker server behind a wrapper that answers like the engine.
//!
//! The engine speaks the same contract and the sidecar uses the same
//! generated types, so the Mocker serves generation unchanged. The wrapper
//! owns what differs: the engine's identity and role, and the decode-role KV
//! session, which it records and then strips because the aggregated Mocker
//! refuses sessions.

use std::process::Command;

use dynamo_backend_common::{BackendError, DisaggregationMode, PreprocessedRequest};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_lpx_sidecar::LpxSidecarEngine;
use dynamo_mocker::common::protocols::EngineType;
use dynamo_sidecar_common::{DEFAULT_MAX_GRPC_MESSAGE_SIZE, struct_to_json, struct_to_json_v14};
use dynamo_sidecar_testkit::control::{Controller, RequestHandle};
use dynamo_sidecar_testkit::fixtures::Outputs;
use dynamo_sidecar_testkit::server::TestServer;
use dynamo_trtllm_mocker::{MockerServerConfig, ServerMode, TrtllmMockerService};
use dynamo_trtllm_sidecar::proto::{
    self as pb,
    control_server::{Control, ControlServer},
    inference_server::{Inference, InferenceServer},
};
use futures::stream::BoxStream;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use super::trtllm::Adapter;
use super::{
    FixtureConfig, GenerateOpening, HandoffFixture, ProcessFixture, Rejection, SidecarFixture,
    WireFixture, fast_engine_args, sidecar_command, vllm, wait_scheduler_active,
    wait_scheduler_idle,
};

pub struct Fixture {
    config: FixtureConfig,
    service: TrtllmMockerService,
    server: TestServer,
}

impl SidecarFixture for Fixture {
    type Engine = LpxSidecarEngine;
    type Protocol = Adapter;
    const GENERATE_OPENING: GenerateOpening = GenerateOpening::WaitsForHeaders;
    // A refusal is deterministic, so the sidecar delivers it as the stream's
    // first item, which the frontend never migrates to another worker.
    const REJECTION: Rejection = Rejection::FirstStreamItem;

    async fn start(control: Controller<Adapter>, config: FixtureConfig) -> Self {
        let role = match config.disaggregation_mode {
            DisaggregationMode::Aggregated => pb::EngineRole::Aggregated,
            DisaggregationMode::Decode => pb::EngineRole::Decode,
            mode => panic!("the LPX sidecar serves aggregated and decode, not {mode}"),
        };
        let mut args = fast_engine_args(EngineType::Trtllm);
        args.speedup_ratio = config.speedup_ratio;
        // Aggregated in every role: the wrapper carries the decode role on
        // the wire and removes the session before a request reaches here.
        let service = TrtllmMockerService::new(
            MockerServerConfig {
                // Also the default ModelInfo.served_model_name, as the
                // engine's configured served name would be.
                model: config.model.clone(),
                mode: ServerMode::Aggregated,
                ..Default::default()
            },
            args,
        )
        .unwrap();
        let controlled = ControlledService {
            inner: service.clone(),
            control,
            role,
        };
        let control_service = controlled.clone();
        let server = TestServer::start(move |listener, shutdown| async move {
            tonic::transport::Server::builder()
                .add_service(
                    InferenceServer::new(controlled)
                        .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
                        .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE),
                )
                .add_service(ControlServer::new(control_service))
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
        }
    }

    async fn engine(&self) -> Self::Engine {
        let argv = vec![
            "dynamo-lpx-sidecar".into(),
            "--grpc-endpoint".into(),
            self.server.endpoint(),
            "--model-path".into(),
            self.config.model.clone(),
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
        let (_, bootstrap) = LpxSidecarEngine::try_from_args_async(argv).unwrap();
        bootstrap.await.unwrap().0
    }

    fn eof_error() -> BackendError {
        BackendError::Unknown
    }

    /// An engine ends open streams with UNAVAILABLE. Another worker can
    /// serve the request, so the sidecar reports a migratable shutdown.
    fn read_error() -> BackendError {
        BackendError::EngineShutdown
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

impl WireFixture for Fixture {
    /// The sidecar names the model the engine serves and sends the prompt as
    /// token IDs. Every other request field is the LPX crate's conversion
    /// tests' to pin; `terminal` checks the tokens and usage that come back.
    fn assert_stream(
        handle: &RequestHandle<Adapter>,
        request: &PreprocessedRequest,
        _outputs: &Outputs,
    ) {
        let native = handle.native_request().unwrap();
        assert_eq!(native.model, request.model);
        assert_eq!(
            native.input,
            Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                ids: request.token_ids.as_ref().clone()
            }))
        );
    }

    async fn scheduler_active(&self) {
        wait_scheduler_active(self, self.service.metrics_receiver()).await;
    }
}

#[derive(Clone)]
struct ControlledService {
    inner: TrtllmMockerService,
    control: Controller<Adapter>,
    role: pb::EngineRole,
}

#[tonic::async_trait]
impl Inference for ControlledService {
    type GenerateStream = BoxStream<'static, Result<pb::GenerateResponse, Status>>;

    async fn generate(
        &self,
        mut request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        // Observed as the sidecar sent it, KV session included.
        let opened = self.control.open(request.get_ref()).await?;
        // A real decode-role engine pulls the prefill's KV with this session;
        // the aggregated Mocker would refuse it, so it stops here.
        if let Some(kv) = request.get_mut().kv.as_mut() {
            kv.session = None;
        }
        let response = self.inner.generate(request).await?;
        Ok(Response::new(opened.wrap(response.into_inner())))
    }
}

#[tonic::async_trait]
impl Control for ControlledService {
    async fn get_server_info(
        &self,
        request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        let mut response = self.inner.get_server_info(request).await?;
        let info = response.get_mut();
        info.engine_name = "lpx-test".to_string();
        info.engine_role = self.role as i32;
        Ok(response)
    }

    async fn get_model_info(
        &self,
        request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        self.inner.get_model_info(request).await
    }

    async fn get_load(
        &self,
        request: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
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

impl ProcessFixture for Fixture {
    const QUARANTINES_ON_READ_ERROR: bool = true;

    fn endpoint(&self) -> String {
        self.server.endpoint()
    }

    /// The engine reports no tokenizer source the sidecar can read, so the
    /// model directory comes from the command line.
    fn command(model: &str) -> Command {
        let mut command = sidecar_command("dynamo-lpx-sidecar", "DYNAMO_LPX_SIDECAR");
        command.args(["--model-path", model]);
        command
    }

    /// The scenarios' default requests are what the engine admits; the LPX
    /// crate's conversion tests own every other request field.
    fn configure_request(_request: &mut PreprocessedRequest) {}

    /// What the engine reports reaches the card through the production Worker.
    fn assert_registration(card: &ModelDeploymentCard) {
        assert_eq!(card.kv_cache_block_size, 4);
        assert_eq!(card.runtime_config.max_num_seqs, Some(64));
        assert_eq!(card.runtime_config.max_num_batched_tokens, Some(1024));
        assert_eq!(card.runtime_config.data_parallel_size, 1);
        assert_eq!(card.effective_context_length(), 32_768);
    }
}

/// A vLLM prefill worker handing off to an LPX decode worker.
impl HandoffFixture<Fixture> for vllm::Fixture {
    const HAS_BOOTSTRAP: bool = false;
    // Both cards carry the Mockers' block size: vLLM's own, and the one the
    // LPX engine reports.
    const KV_BLOCK_SIZE: u32 = 4;

    /// The Rust vLLM sidecar's flat `kv_transfer_params` cross Dynamo as
    /// opaque JSON and reach the engine as exactly the attributes of the decode
    /// leg's KV session. The LPX crate's handoff tests own the other fields.
    fn assert_handoff(
        prefill: &RequestHandle<vllm::Adapter>,
        decode: &RequestHandle<Adapter>,
        _id: &str,
    ) {
        let native = prefill
            .native_responses()
            .into_iter()
            .find_map(|response| response.outputs?.finish_info?.kv_transfer_params)
            .expect("native prefill handoff");
        let kv_transfer_params = struct_to_json_v14(native, "vLLM", "kv_transfer_params").unwrap();
        let session = decode
            .native_request()
            .and_then(|wire| wire.kv?.session)
            .expect("decode KV session at the engine");
        let attributes = struct_to_json(
            session.attributes_struct.expect("session attributes"),
            "LPX",
            "attributes_struct",
        )
        .unwrap();
        assert_eq!(attributes, kv_transfer_params);
    }
}
