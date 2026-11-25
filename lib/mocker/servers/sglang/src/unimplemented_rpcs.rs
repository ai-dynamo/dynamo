// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `impl_sglang_service!`: implements `SglangService` for a test server.
//!
//! The published `sglang-api-types` service covers all of SGLang's RPCs, and a
//! Rust trait impl must define every one. A test server writes the RPCs it
//! serves; this macro adds the rest, each answering `Unimplemented`.
//!
//! The RPC list is SGLang's published service minus the eleven RPCs the test
//! servers implement (`generate`, `health_check`, `get_model_info`,
//! `get_server_info`, `list_models`, `detokenize`, `abort`, and the four HiCache
//! storage RPCs). A new `sglang-api-types` release that adds an RPC fails to
//! compile until it is added here.
//!
//! The macro wraps the whole impl because `async_trait` does not expand macro
//! calls written inside an impl.

/// Re-exports the macro's own paths so callers need no extra dependencies.
#[doc(hidden)]
pub mod __deps {
    pub use futures;
    pub use sglang_api_types;
    pub use tonic_v14;
}

/// `impl_sglang_service! { impl MyServer { /* the RPCs it serves */ } }`
#[doc(hidden)]
#[macro_export]
macro_rules! impl_sglang_service {
    (impl $ty:ty { $($body:tt)* }) => {
        #[$crate::__deps::tonic_v14::async_trait]
        impl $crate::__deps::sglang_api_types::api::v1::sglang_service_server::SglangService
            for $ty
        {
            $($body)*

            async fn set_internal_state(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::SetInternalStateRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::SetInternalStateResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "set_internal_state is not supported by the SGLang test servers",
                ))
            }

            async fn start_profile(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::StartProfileRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::ProfileResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "start_profile is not supported by the SGLang test servers",
                ))
            }

            async fn stop_profile(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::StopProfileRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::ProfileResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "stop_profile is not supported by the SGLang test servers",
                ))
            }

            async fn start_expert_distribution_record(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::ExpertDistributionRecordRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::ExpertDistributionRecordResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "start_expert_distribution_record is not supported by the SGLang test servers",
                ))
            }

            async fn stop_expert_distribution_record(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::ExpertDistributionRecordRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::ExpertDistributionRecordResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "stop_expert_distribution_record is not supported by the SGLang test servers",
                ))
            }

            async fn dump_expert_distribution_record(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::ExpertDistributionRecordRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::ExpertDistributionRecordResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "dump_expert_distribution_record is not supported by the SGLang test servers",
                ))
            }

            async fn slow_down(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::SlowDownRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::SlowDownResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "slow_down is not supported by the SGLang test servers",
                ))
            }

            async fn release_memory_occupation(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::ReleaseMemoryOccupationRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::MemoryOccupationResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "release_memory_occupation is not supported by the SGLang test servers",
                ))
            }

            async fn resume_memory_occupation(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::ResumeMemoryOccupationRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::MemoryOccupationResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "resume_memory_occupation is not supported by the SGLang test servers",
                ))
            }

            async fn get_weights_by_name(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::GetWeightsByNameRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::GetWeightsByNameResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "get_weights_by_name is not supported by the SGLang test servers",
                ))
            }

            async fn init_weights_update_group(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::InitWeightsUpdateGroupRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::WeightGroupResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "init_weights_update_group is not supported by the SGLang test servers",
                ))
            }

            async fn destroy_weights_update_group(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::DestroyWeightsUpdateGroupRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::WeightGroupResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "destroy_weights_update_group is not supported by the SGLang test servers",
                ))
            }

            async fn init_weights_send_group_for_remote_instance(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::InitWeightsSendGroupForRemoteInstanceRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::WeightGroupResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "init_weights_send_group_for_remote_instance is not supported by the SGLang test servers",
                ))
            }

            async fn send_weights_to_remote_instance(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::SendWeightsToRemoteInstanceRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::WeightGroupResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "send_weights_to_remote_instance is not supported by the SGLang test servers",
                ))
            }

            async fn update_weights_from_disk(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::UpdateWeightsFromDiskRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::UpdateWeightsFromDiskResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "update_weights_from_disk is not supported by the SGLang test servers",
                ))
            }

            async fn update_weights_from_distributed(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::UpdateWeightsFromDistributedRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::WeightUpdateResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "update_weights_from_distributed is not supported by the SGLang test servers",
                ))
            }

            async fn update_weights_from_tensor(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::UpdateWeightsFromTensorRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::WeightUpdateResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "update_weights_from_tensor is not supported by the SGLang test servers",
                ))
            }

            async fn update_weights_from_ipc(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::UpdateWeightsFromIpcRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::WeightUpdateResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "update_weights_from_ipc is not supported by the SGLang test servers",
                ))
            }

            async fn begin_weight_update(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::BeginWeightUpdateRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::WeightUpdateResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "begin_weight_update is not supported by the SGLang test servers",
                ))
            }

            async fn end_weight_update(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::EndWeightUpdateRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::WeightUpdateResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "end_weight_update is not supported by the SGLang test servers",
                ))
            }

            async fn plugin_weight_update(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::PluginWeightUpdateRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::PluginWeightUpdateResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "plugin_weight_update is not supported by the SGLang test servers",
                ))
            }

            async fn load_lora_adapter(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::LoadLoraAdapterRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::LoraUpdateResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "load_lora_adapter is not supported by the SGLang test servers",
                ))
            }

            async fn load_lora_adapter_from_tensors(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::LoadLoraAdapterFromTensorsRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::LoraUpdateResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "load_lora_adapter_from_tensors is not supported by the SGLang test servers",
                ))
            }

            async fn unload_lora_adapter(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::UnloadLoraAdapterRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::LoraUpdateResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "unload_lora_adapter is not supported by the SGLang test servers",
                ))
            }

            async fn get_remote_instance_transfer_engine_info(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::RemoteInstanceTransferEngineInfoRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::RemoteInstanceTransferEngineInfoResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "get_remote_instance_transfer_engine_info is not supported by the SGLang test servers",
                ))
            }

            async fn parse_function_call(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::ParseFunctionCallRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::ParseFunctionCallResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "parse_function_call is not supported by the SGLang test servers",
                ))
            }

            async fn separate_reasoning(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::SeparateReasoningRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::SeparateReasoningResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "separate_reasoning is not supported by the SGLang test servers",
                ))
            }

            async fn add_external_corpus(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::AddExternalCorpusRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::AddExternalCorpusResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "add_external_corpus is not supported by the SGLang test servers",
                ))
            }

            async fn remove_external_corpus(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::RemoveExternalCorpusRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::RemoveExternalCorpusResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "remove_external_corpus is not supported by the SGLang test servers",
                ))
            }

            async fn list_external_corpora(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::ListExternalCorporaRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::ListExternalCorporaResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "list_external_corpora is not supported by the SGLang test servers",
                ))
            }

            async fn freeze_gc(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::FreezeGcRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::FreezeGcResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "freeze_gc is not supported by the SGLang test servers",
                ))
            }

            async fn pause_generation(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::PauseGenerationRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::GenerationPauseResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "pause_generation is not supported by the SGLang test servers",
                ))
            }

            async fn continue_generation(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::ContinueGenerationRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::GenerationPauseResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "continue_generation is not supported by the SGLang test servers",
                ))
            }

            async fn pd_role_switch(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::PdRoleSwitchRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::PdRoleSwitchResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "pd_role_switch is not supported by the SGLang test servers",
                ))
            }

            async fn update_weight_version(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::UpdateWeightVersionRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::UpdateWeightVersionResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "update_weight_version is not supported by the SGLang test servers",
                ))
            }

            async fn open_session(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenSessionRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenSessionResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_session is not supported by the SGLang test servers",
                ))
            }

            async fn close_session(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::CloseSessionRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::CloseSessionResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "close_session is not supported by the SGLang test servers",
                ))
            }

            async fn get_hi_cache_l3_cache_layout(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::GetHiCacheL3CacheLayoutRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::GetHiCacheL3CacheLayoutResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "get_hi_cache_l3_cache_layout is not supported by the SGLang test servers",
                ))
            }

            async fn configure_logging(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::ConfigureLoggingRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::ConfigureLoggingResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "configure_logging is not supported by the SGLang test servers",
                ))
            }

            async fn dumper_control(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::DumperControlRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::DumperControlResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "dumper_control is not supported by the SGLang test servers",
                ))
            }

            async fn scale_elastic_ep(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::ScaleElasticEpRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::ScaleElasticEpResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "scale_elastic_ep is not supported by the SGLang test servers",
                ))
            }

            async fn get_elastic_ep_state(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::ElasticEpStateRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::ElasticEpStateResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "get_elastic_ep_state is not supported by the SGLang test servers",
                ))
            }

            type OllamaChatStream = $crate::__deps::futures::stream::BoxStream<
                'static,
                Result<$crate::__deps::sglang_api_types::api::v1::OllamaChatResponse, $crate::__deps::tonic_v14::Status>,
            >;

            async fn ollama_chat(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OllamaChatRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<Self::OllamaChatStream>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "ollama_chat is not supported by the SGLang test servers",
                ))
            }

            type OllamaGenerateStream = $crate::__deps::futures::stream::BoxStream<
                'static,
                Result<$crate::__deps::sglang_api_types::api::v1::OllamaGenerateResponse, $crate::__deps::tonic_v14::Status>,
            >;

            async fn ollama_generate(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OllamaGenerateRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<Self::OllamaGenerateStream>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "ollama_generate is not supported by the SGLang test servers",
                ))
            }

            async fn ollama_tags(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OllamaTagsRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OllamaTagsResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "ollama_tags is not supported by the SGLang test servers",
                ))
            }

            async fn ollama_show(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OllamaShowRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OllamaShowResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "ollama_show is not supported by the SGLang test servers",
                ))
            }

            async fn anthropic_messages(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::AnthropicMessagesRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::AnthropicMessagesResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "anthropic_messages is not supported by the SGLang test servers",
                ))
            }

            type AnthropicStreamMessagesStream = $crate::__deps::futures::stream::BoxStream<
                'static,
                Result<$crate::__deps::sglang_api_types::api::v1::AnthropicStreamEvent, $crate::__deps::tonic_v14::Status>,
            >;

            async fn anthropic_stream_messages(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::AnthropicMessagesRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<Self::AnthropicStreamMessagesStream>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "anthropic_stream_messages is not supported by the SGLang test servers",
                ))
            }

            async fn anthropic_count_tokens(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::AnthropicCountTokensRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::AnthropicCountTokensResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "anthropic_count_tokens is not supported by the SGLang test servers",
                ))
            }

            async fn open_ai_completions(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiCompletionRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenAiCompletionResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_completions is not supported by the SGLang test servers",
                ))
            }

            type OpenAiStreamCompletionsStream = $crate::__deps::futures::stream::BoxStream<
                'static,
                Result<$crate::__deps::sglang_api_types::api::v1::OpenAiCompletionStreamResponse, $crate::__deps::tonic_v14::Status>,
            >;

            async fn open_ai_stream_completions(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiCompletionRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<Self::OpenAiStreamCompletionsStream>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_stream_completions is not supported by the SGLang test servers",
                ))
            }

            async fn open_ai_chat_completions(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiChatCompletionRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenAiChatCompletionResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_chat_completions is not supported by the SGLang test servers",
                ))
            }

            type OpenAiStreamChatCompletionsStream = $crate::__deps::futures::stream::BoxStream<
                'static,
                Result<$crate::__deps::sglang_api_types::api::v1::OpenAiChatCompletionStreamResponse, $crate::__deps::tonic_v14::Status>,
            >;

            async fn open_ai_stream_chat_completions(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiChatCompletionRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<Self::OpenAiStreamChatCompletionsStream>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_stream_chat_completions is not supported by the SGLang test servers",
                ))
            }

            async fn open_ai_embeddings(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiEmbeddingRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenAiEmbeddingResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_embeddings is not supported by the SGLang test servers",
                ))
            }

            async fn open_ai_classify(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiClassifyRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenAiClassifyResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_classify is not supported by the SGLang test servers",
                ))
            }

            async fn open_ai_score(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiScoreRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenAiScoreResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_score is not supported by the SGLang test servers",
                ))
            }

            async fn open_ai_rerank(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiRerankRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenAiRerankResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_rerank is not supported by the SGLang test servers",
                ))
            }

            async fn open_ai_tokenize(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiTokenizeRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenAiTokenizeResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_tokenize is not supported by the SGLang test servers",
                ))
            }

            async fn open_ai_detokenize(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiDetokenizeRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenAiDetokenizeResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_detokenize is not supported by the SGLang test servers",
                ))
            }

            async fn open_ai_responses(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiResponsesRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenAiResponsesResponse>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_responses is not supported by the SGLang test servers",
                ))
            }

            async fn open_ai_list_models(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiListModelsRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenAiModelList>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_list_models is not supported by the SGLang test servers",
                ))
            }

            async fn open_ai_retrieve_model(
                &self,
                _: $crate::__deps::tonic_v14::Request<$crate::__deps::sglang_api_types::api::v1::OpenAiRetrieveModelRequest>,
            ) -> Result<$crate::__deps::tonic_v14::Response<$crate::__deps::sglang_api_types::api::v1::OpenAiModelCard>, $crate::__deps::tonic_v14::Status> {
                Err($crate::__deps::tonic_v14::Status::unimplemented(
                    "open_ai_retrieve_model is not supported by the SGLang test servers",
                ))
            }
        }
    };
}
