// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tonic_health_v14 as tonic_health;
use tonic_v14 as tonic;

use dynamo_backend_common::BackendError;
use dynamo_mocker::common::protocols::EngineType;
use dynamo_sidecar_testkit::control::{Controller, Protocol};
use dynamo_sidecar_testkit::server::TestServer;
use dynamo_vllm_mocker::{MockerServerConfig, VllmMockerService};
use dynamo_vllm_sidecar::VllmSidecarEngine;
use dynamo_vllm_sidecar::proto::{
    self as pb,
    control_server::ControlServer,
    inference_server::{Inference, InferenceServer},
};
use futures::stream::BoxStream;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use super::{FixtureConfig, SidecarFixture, fast_engine_args};

pub struct Fixture {
    config: FixtureConfig,
    service: VllmMockerService,
    server: TestServer,
}

impl SidecarFixture for Fixture {
    type Engine = VllmSidecarEngine;
    type Protocol = Adapter;

    async fn start(control: Controller<Adapter>, config: FixtureConfig) -> Self {
        let service = VllmMockerService::new(
            MockerServerConfig {
                model: config.model.clone(),
                ..Default::default()
            },
            fast_engine_args(EngineType::Vllm),
        )
        .unwrap();
        let controlled = ControlledService {
            inner: service.clone(),
            control,
        };
        let control_service = service.clone();
        let (health, health_service) = tonic_health::server::health_reporter();
        health
            .set_serving::<ControlServer<VllmMockerService>>()
            .await;
        health
            .set_serving::<InferenceServer<ControlledService>>()
            .await;
        let server = TestServer::start(move |listener, shutdown| async move {
            tonic::transport::Server::builder()
                .add_service(InferenceServer::new(controlled))
                .add_service(ControlServer::new(control_service))
                .add_service(health_service)
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
            "dynamo-vllm-sidecar".into(),
            "--grpc-endpoint".into(),
            self.server.endpoint(),
            "--grpc-connections".into(),
            self.config.connections.to_string(),
            "--grpc-startup-deadline-secs".into(),
            "5".into(),
            "--grpc-connect-attempt-timeout-secs".into(),
            "1".into(),
        ];
        tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(argv)).unwrap().0)
            .await
            .unwrap()
    }

    fn eof_error() -> BackendError {
        BackendError::Unknown
    }

    fn active_request_count(&self) -> usize {
        self.service.active_request_count()
    }

    async fn shutdown(&mut self) {
        self.server.shutdown().await.unwrap();
    }
}

#[derive(Clone)]
struct ControlledService {
    inner: VllmMockerService,
    control: Controller<Adapter>,
}

#[tonic::async_trait]
impl Inference for ControlledService {
    type GenerateStreamStream = BoxStream<'static, Result<pb::GenerateResponse, Status>>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        self.inner.generate(request).await
    }

    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let opened = self.control.open(request.get_ref()).await?;
        let response = self.inner.generate_stream(request).await?;
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
        let Some(output) = &response.outputs else {
            return false;
        };
        tokens.extend_from_slice(&output.token_ids);
        !output.token_ids.is_empty()
    }

    fn is_terminal(response: &Self::Response) -> bool {
        response
            .outputs
            .as_ref()
            .is_some_and(|output| output.finish_info.is_some())
    }

    fn injected_error(message: &'static str) -> Self::Error {
        Status::unavailable(message)
    }
}
