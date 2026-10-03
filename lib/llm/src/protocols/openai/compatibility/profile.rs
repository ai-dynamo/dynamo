// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline-derived compatibility identity, not a blanket support declaration.
//!
//! This descriptor adds the missing context to admission decisions without a
//! second field allowlist. Unknown facts stay explicit: a callback is not proof
//! of a particular Python processor, and legacy cards do not prove topology.

use std::fmt;

use serde::Serialize;

use crate::{entrypoint::ChatProcessorIdentity, model_type::ModelInput, worker_type::WorkerType};

use super::admission::TargetAdmission;
use super::rejection::{CompatibilityRejection, RejectionKind, RejectionStage};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub(crate) enum Endpoint {
    #[serde(rename = "/v1/chat/completions")]
    Chat,
    #[serde(rename = "/v1/completions")]
    Completion,
}

impl Endpoint {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "/v1/chat/completions",
            Self::Completion => "/v1/completions",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Processor {
    Rust,
    Vllm,
    Sglang,
    /// A custom callback without a built-in processor identity.
    ExternalFactory,
    Backend,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Transport {
    PreprocessedRpc,
    OpenaiRpc,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Deployment {
    Aggregated,
    Disaggregated,
    Unknown,
}

/// Facts about the pipeline that was actually constructed. No changes to MDC
/// identity, runtime metadata, admission cohorts, or routing policy are needed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub(crate) struct PipelineContext {
    pub processor: Processor,
    pub transport: Transport,
    /// Existing RPC messages have no whole-message protocol-version field.
    /// Extension-envelope v1 must not be confused with a version for this wire.
    pub transport_protocol_version: Option<u32>,
    pub deployment: Deployment,
}

impl PipelineContext {
    pub(crate) fn from_pipeline(
        endpoint: Endpoint,
        model_input: ModelInput,
        worker_type: Option<WorkerType>,
        chat_factory_identity: Option<ChatProcessorIdentity>,
    ) -> Self {
        let (processor, transport) = match model_input {
            ModelInput::Tokens => (
                if endpoint == Endpoint::Chat {
                    match chat_factory_identity {
                        Some(ChatProcessorIdentity::Vllm) => Processor::Vllm,
                        Some(ChatProcessorIdentity::Sglang) => Processor::Sglang,
                        Some(ChatProcessorIdentity::Custom) => Processor::ExternalFactory,
                        None => Processor::Rust,
                    }
                } else {
                    Processor::Rust
                },
                Transport::PreprocessedRpc,
            ),
            ModelInput::Text => (Processor::Backend, Transport::OpenaiRpc),
            _ => (Processor::Unknown, Transport::Unknown),
        };
        Self {
            processor,
            transport,
            transport_protocol_version: None,
            deployment: match worker_type {
                Some(WorkerType::Aggregated) => Deployment::Aggregated,
                Some(WorkerType::Decode | WorkerType::Prefill | WorkerType::Encode) => {
                    Deployment::Disaggregated
                }
                // Legacy routing's aggregated fallback does not prove that
                // there is no remote prefill stage. Do not reuse it as evidence.
                None => Deployment::Unknown,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptLogprobsAdmission {
    UnidentifiedTarget,
    UnverifiedTarget,
    /// Upstream request validation, independent of the downstream pipeline.
    /// This does not declare unary prompt-logprob support on every pipeline.
    RejectPositiveStreaming,
}

/// Machine-readable projection of the same identity and rule used at runtime.
/// Fields without an implemented rule are not implicitly supported. This is an
/// admission descriptor, not yet the full per-field conformance catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub(crate) struct CompatibilityProfile {
    pub descriptor_version: u32,
    pub target: TargetAdmission,
    /// Inspected upstream rule source, not a runtime conformance attestation.
    pub upstream_commit: Option<&'static str>,
    pub endpoint: Endpoint,
    pub pipeline: PipelineContext,
    pub prompt_logprobs_admission: PromptLogprobsAdmission,
}

impl CompatibilityProfile {
    pub(super) fn validate_full_vocab_pipeline(
        self,
        runtime: Option<&crate::local_model::runtime_config::ModelRuntimeConfig>,
    ) -> anyhow::Result<()> {
        use crate::protocols::common::{
            invalid_argument_error,
            prompt_logprobs::{FULL_VOCAB_COUNT, validate_full_vocab_count},
        };
        if self.pipeline.transport != Transport::PreprocessedRpc
            || !matches!(self.pipeline.processor, Processor::Rust | Processor::Vllm)
        {
            return Err(CompatibilityRejection::prompt_logprobs(
                self,
                RejectionKind::UnsupportedField,
                RejectionStage::Admission,
                &[
                    "Use a verified token pipeline with the Rust or vLLM processor",
                    "Request a finite count",
                ],
            )
            .attach(invalid_argument_error(
                "`prompt_logprobs=-1` has no verified conversion on the selected frontend pipeline",
            )));
        }
        validate_full_vocab_count(Some(FULL_VOCAB_COUNT), runtime).map_err(|error| {
            CompatibilityRejection::prompt_logprobs(
                self,
                RejectionKind::UnsupportedField,
                RejectionStage::BackendCapability,
                &["Use a worker advertising full-vocabulary support and a sufficient max_logprobs limit", "Request a finite count"],
            ).attach(error)
        })
    }

    pub(crate) fn new(
        target: TargetAdmission,
        endpoint: Endpoint,
        pipeline: PipelineContext,
    ) -> Self {
        Self {
            descriptor_version: 1,
            target,
            upstream_commit: match target {
                TargetAdmission::Vllm029 => Some("98dff2a81d747d1dba01a47f939f48c3526d4206"),
                TargetAdmission::Vllm030 => Some("ced6857afa0ea7b2e3f0846a62e1394e90f15607"),
                TargetAdmission::Unidentified | TargetAdmission::Unverified => None,
            },
            endpoint,
            pipeline,
            prompt_logprobs_admission: match target {
                TargetAdmission::Unidentified => PromptLogprobsAdmission::UnidentifiedTarget,
                TargetAdmission::Unverified => PromptLogprobsAdmission::UnverifiedTarget,
                TargetAdmission::Vllm029 | TargetAdmission::Vllm030 => {
                    PromptLogprobsAdmission::RejectPositiveStreaming
                }
            },
        }
    }
}

impl fmt::Display for CompatibilityProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Bounded labels only: never include user model names, callback names,
        // worker addresses, or arbitrary advertised version strings.
        write!(
            f,
            "{} at {} [processor={:?}, transport={:?}, protocol_version=",
            self.target.label(),
            self.endpoint.as_str(),
            self.pipeline.processor,
            self.pipeline.transport,
        )?;
        match self.pipeline.transport_protocol_version {
            Some(version) => write!(f, "{version}")?,
            None => f.write_str("unversioned")?,
        }
        write!(f, ", deployment={:?}]", self.pipeline.deployment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pipeline_identity_follows_real_endpoint_ownership() {
        for endpoint in [Endpoint::Chat, Endpoint::Completion] {
            for (factory, expected) in [
                (None, Processor::Rust),
                (
                    Some(ChatProcessorIdentity::Custom),
                    Processor::ExternalFactory,
                ),
                (Some(ChatProcessorIdentity::Vllm), Processor::Vllm),
                (Some(ChatProcessorIdentity::Sglang), Processor::Sglang),
            ] {
                let context = PipelineContext::from_pipeline(
                    endpoint,
                    ModelInput::Tokens,
                    Some(WorkerType::Aggregated),
                    factory,
                );
                assert_eq!(
                    context.processor,
                    if endpoint == Endpoint::Chat {
                        expected
                    } else {
                        Processor::Rust
                    }
                );
                assert_eq!(context.transport, Transport::PreprocessedRpc);
                let backend = PipelineContext::from_pipeline(
                    endpoint,
                    ModelInput::Text,
                    Some(WorkerType::Decode),
                    factory,
                );
                assert_eq!(backend.processor, Processor::Backend);
                assert_eq!(backend.transport, Transport::OpenaiRpc);
                assert_eq!(backend.deployment, Deployment::Disaggregated);
            }
        }
    }

    #[test]
    fn missing_topology_and_unversioned_wire_are_not_invented() {
        let context = PipelineContext::from_pipeline(
            Endpoint::Chat,
            ModelInput::Tokens,
            None,
            Some(ChatProcessorIdentity::Custom),
        );
        assert_eq!(context.deployment, Deployment::Unknown);
        assert_eq!(context.transport_protocol_version, None);
        let profile = CompatibilityProfile::new(TargetAdmission::Vllm030, Endpoint::Chat, context);
        let value = serde_json::to_value(profile).unwrap();
        assert_eq!(value["endpoint"], "/v1/chat/completions");
        assert_eq!(
            value["pipeline"],
            json!({
                "processor":"external_factory", "transport":"preprocessed_rpc",
                "transport_protocol_version":null, "deployment":"unknown",
            })
        );
        assert_eq!(
            value["prompt_logprobs_admission"],
            "reject_positive_streaming"
        );
        assert!(!value.as_object().unwrap().contains_key("compatible"));
    }
}
