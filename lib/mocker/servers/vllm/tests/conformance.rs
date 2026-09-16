// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::BackendError;
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_vllm_mocker::{MockerServerConfig, VllmMockerService};
use dynamo_vllm_sidecar::VllmSidecarEngine;
use dynamo_vllm_sidecar::proto::{
    self as pb,
    generate_server::{Generate, GenerateServer},
};
use futures::stream::BoxStream;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

#[path = "../../tests/common/mod.rs"]
mod common;

struct Fixture {
    endpoint: String,
    service: VllmMockerService,
    server: JoinHandle<()>,
}

impl common::SidecarFixture for Fixture {
    type Engine = VllmSidecarEngine;

    async fn start(control: common::Control) -> Self {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(4_096)
            .max_num_seqs(Some(64))
            .max_num_batched_tokens(Some(1_024))
            .speedup_ratio(0.0)
            .dp_size(1)
            .build()
            .unwrap();
        let service = VllmMockerService::new(MockerServerConfig::default(), args).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let controlled = ControlledService {
            inner: service.clone(),
            control,
        };
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(GenerateServer::new(controlled))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        Self {
            endpoint: format!("http://{address}"),
            service,
            server,
        }
    }

    async fn engine(&self) -> Self::Engine {
        VllmSidecarEngine::from_args(Some(vec![
            "dynamo-vllm-sidecar".into(),
            "--vllm-endpoint".into(),
            self.endpoint.clone(),
            "--model-path".into(),
            "mocker-model".into(),
            "--grpc-connections".into(),
            "1".into(),
            "--grpc-startup-deadline-secs".into(),
            "5".into(),
            "--grpc-connect-attempt-timeout-secs".into(),
            "1".into(),
        ]))
        .unwrap()
        .0
    }

    fn eof_error() -> BackendError {
        BackendError::Unknown
    }

    fn active_request_count(&self) -> usize {
        self.service.active_request_count()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

#[derive(Clone)]
struct ControlledService {
    inner: VllmMockerService,
    control: common::Control,
}

#[tonic::async_trait]
impl Generate for ControlledService {
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
        let guard = self.control.open(&request.get_ref().request_id).await?;
        let response = self.inner.generate_stream(request).await?;
        Ok(Response::new(
            self.control.stream(response.into_inner(), guard),
        ))
    }
}

impl common::NativeResponse for pb::GenerateResponse {
    fn record_tokens(&self, tokens: &mut Vec<u32>) -> bool {
        let Some(output) = &self.outputs else {
            return false;
        };
        tokens.extend_from_slice(&output.token_ids);
        !output.token_ids.is_empty()
    }

    fn is_terminal(&self) -> bool {
        self.outputs
            .as_ref()
            .is_some_and(|output| output.finish_info.is_some())
    }
}
