// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use dynamo_vllm_mocker::VllmMockerService;
use dynamo_vllm_sidecar::proto::{self as pb, control_server::Control};
use tonic_v14::{Request, Response, Status};

#[derive(Clone)]
pub struct DiscoveryControl {
    pub inner: VllmMockerService,
    pub sources: Arc<Mutex<Option<Vec<pb::KvEventSource>>>>,
}

macro_rules! delegate_controls {
    ($($method:ident($request:ty) -> $response:ty;)*) => {
        #[tonic_v14::async_trait]
        impl Control for DiscoveryControl {
            async fn get_server_info(
                &self,
                request: Request<pb::GetServerInfoRequest>,
            ) -> Result<Response<pb::ServerInfo>, Status> {
                let mut response = self.inner.get_server_info(request).await?.into_inner();
                if let Some(sources) = self.sources.lock().unwrap().as_ref() {
                    let parallelism = response.parallelism.as_mut().unwrap();
                    parallelism.data_parallel_size = sources.len() as u32;
                    parallelism.data_parallel_size_local = sources.len() as u32;
                }
                Ok(Response::new(response))
            }

            async fn get_kv_event_sources(
                &self,
                request: Request<pb::GetKvEventSourcesRequest>,
            ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
                let sources = self.sources.lock().unwrap().clone();
                if let Some(sources) = sources {
                    Ok(Response::new(pb::GetKvEventSourcesResponse { sources }))
                } else {
                    self.inner.get_kv_event_sources(request).await
                }
            }

            $(async fn $method(&self, request: Request<$request>) -> Result<Response<$response>, Status> {
                self.inner.$method(request).await
            })*
        }
    };
}

delegate_controls! {
    get_model_info(pb::GetModelInfoRequest) -> pb::ModelInfo;
    abort(pb::AbortRequest) -> pb::AbortResponse;
    load_lora(pb::LoadLoraRequest) -> pb::LoadLoraResponse;
    unload_lora(pb::UnloadLoraRequest) -> pb::UnloadLoraResponse;
    list_loras(pb::ListLorasRequest) -> pb::ListLorasResponse;
    pause_generation(pb::PauseGenerationRequest) -> pb::PauseGenerationResponse;
    resume_generation(pb::ResumeGenerationRequest) -> pb::ResumeGenerationResponse;
    is_paused(pb::IsPausedRequest) -> pb::IsPausedResponse;
    sleep(pb::SleepRequest) -> pb::SleepResponse;
    wake_up(pb::WakeUpRequest) -> pb::WakeUpResponse;
    is_sleeping(pb::IsSleepingRequest) -> pb::IsSleepingResponse;
    init_weight_transfer_engine(pb::InitWeightTransferEngineRequest) -> pb::InitWeightTransferEngineResponse;
    start_weight_update(pb::StartWeightUpdateRequest) -> pb::StartWeightUpdateResponse;
    start_draft_weight_update(pb::StartDraftWeightUpdateRequest) -> pb::StartDraftWeightUpdateResponse;
    update_weights(pb::UpdateWeightsRequest) -> pb::UpdateWeightsResponse;
    finish_weight_update(pb::FinishWeightUpdateRequest) -> pb::FinishWeightUpdateResponse;
    update_weight_version(pb::UpdateWeightVersionRequest) -> pb::UpdateWeightVersionResponse;
    get_weight_version(pb::GetWeightVersionRequest) -> pb::GetWeightVersionResponse;
}
