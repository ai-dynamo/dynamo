// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only projection of pipeline admission, not another policy allowlist.
//! Descriptors are captured with the engines they describe and disappear with
//! their committed WorkerSet. Live per-hop checks remain authoritative.

use serde::Serialize;

use super::profile::{CompatibilityProfile, Endpoint, Transport};
use crate::{
    local_model::runtime_config::ModelRuntimeConfig,
    protocols::common::{
        backend_extensions::{SAMPLING_FIELDS, SamplingTarget},
        legacy_vllm::LegacyVllmRelease,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SamplingTransport {
    V1WithLegacyCopy,
    LegacyOnly,
    Rejected,
    OutsideThisAdmissionRule,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub(crate) struct SamplingFieldRule {
    pub field: &'static str,
    pub request_location: &'static str,
    pub transport: SamplingTransport,
}

/// Partial admission coverage only. A missing descriptor on an embedded/custom
/// engine is explicit, never reconstructed from a default or guessed model card.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub(crate) struct PipelineAdmissionCatalog {
    pub endpoint: Endpoint,
    pub admission: Option<CompatibilityProfile>,
    pub sampling_fields: Vec<SamplingFieldRule>,
    /// Registered-pipeline admission only, not end-to-end conformance or live
    /// downstream availability. Streaming -1 is always rejected.
    pub full_vocab_prompt_logprobs_unary_admitted: Option<bool>,
}

impl PipelineAdmissionCatalog {
    pub(crate) fn unavailable(endpoint: Endpoint) -> Self {
        Self {
            endpoint,
            admission: None,
            sampling_fields: Vec::new(),
            full_vocab_prompt_logprobs_unary_admitted: None,
        }
    }

    pub(crate) fn new(
        admission: CompatibilityProfile,
        runtime: &ModelRuntimeConfig,
        legacy: Option<LegacyVllmRelease>,
    ) -> Self {
        let target = SamplingTarget::resolve(runtime, legacy);
        let sampling_fields = SAMPLING_FIELDS
            .iter()
            .map(|field| {
                let transport = if admission.pipeline.transport != Transport::PreprocessedRpc {
                    SamplingTransport::OutsideThisAdmissionRule
                } else {
                    match target.as_ref() {
                        Ok(target) if target.validate_field(field).is_ok() => match target {
                            SamplingTarget::Current(_) => SamplingTransport::V1WithLegacyCopy,
                            SamplingTarget::Legacy(_) => SamplingTransport::LegacyOnly,
                            SamplingTarget::Unavailable => SamplingTransport::Rejected,
                        },
                        _ => SamplingTransport::Rejected,
                    }
                };
                SamplingFieldRule {
                    field,
                    request_location: "root",
                    transport,
                }
            })
            .collect();
        Self {
            endpoint: admission.endpoint,
            admission: Some(admission),
            sampling_fields,
            full_vocab_prompt_logprobs_unary_admitted: Some(
                admission
                    .validate_full_vocab_pipeline(Some(runtime))
                    .is_ok(),
            ),
        }
    }
}

#[derive(Serialize, utoipa::ToSchema)]
pub(crate) struct ModelCompatibilityCatalog {
    pub schema_version: u32,
    pub scope: &'static str,
    pub model: String,
    pub coverage_complete: bool,
    pub unlisted_fields: &'static str,
    pub end_to_end_conformance: &'static str,
    pub profiles: Vec<PipelineAdmissionCatalog>,
}

impl ModelCompatibilityCatalog {
    pub(crate) fn new(model: String, mut profiles: Vec<PipelineAdmissionCatalog>) -> Self {
        // Stable output without exposing namespaces, worker IDs, or weights.
        profiles.sort_by_cached_key(|profile| {
            serde_json::to_string(profile)
                .expect("catalog contains only serializable enums, strings, and integers")
        });
        profiles.dedup();
        Self {
            schema_version: 1,
            scope: "registered_pipeline_admission",
            model,
            coverage_complete: false,
            unlisted_fields: "not_catalogued",
            end_to_end_conformance: "unverified",
            profiles,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{admission::TargetAdmission, profile::PipelineContext};
    use super::*;
    use crate::protocols::common::backend_extensions::{
        VLLM_PROTOCOL_EXTENSIONS_CAPABILITY, lower_sampling_passthrough_for_target,
    };
    use crate::{
        entrypoint::ChatProcessorIdentity, model_type::ModelInput, worker_type::WorkerType,
    };
    use serde_json::json;

    #[test]
    fn catalog_transport_is_the_actual_lowering_policy_not_a_second_allowlist() {
        let mut capable = ModelRuntimeConfig::default();
        capable
            .set_engine_specific(
                VLLM_PROTOCOL_EXTENSIONS_CAPABILITY,
                json!({
                    "schema_version":1, "target":"vllm", "engine_version":"0.30.0",
                    "sampling_fields":["allowed_token_ids", "logprob_token_ids"],
                }),
            )
            .unwrap();
        let mut invalid = capable.clone();
        invalid
            .set_engine_specific(
                VLLM_PROTOCOL_EXTENSIONS_CAPABILITY,
                json!({"schema_version":2}),
            )
            .unwrap();
        for runtime in [&ModelRuntimeConfig::default(), &capable, &invalid] {
            for legacy in [
                None,
                Some(LegacyVllmRelease::Dynamo14),
                Some(LegacyVllmRelease::Dynamo15),
            ] {
                for endpoint in [Endpoint::Chat, Endpoint::Completion] {
                    let profile = CompatibilityProfile::new(
                        TargetAdmission::from_runtime(runtime),
                        endpoint,
                        PipelineContext::from_pipeline(
                            endpoint,
                            ModelInput::Tokens,
                            Some(WorkerType::Aggregated),
                            Some(ChatProcessorIdentity::Vllm),
                        ),
                    );
                    let catalog = PipelineAdmissionCatalog::new(profile, runtime, legacy);
                    for rule in catalog.sampling_fields {
                        let value = if rule.field == "bad_words_token_ids" {
                            json!([[0]])
                        } else {
                            json!([0])
                        };
                        let actual = lower_sampling_passthrough_for_target(
                            serde_json::Map::from_iter([(rule.field.into(), value)]),
                            runtime,
                            legacy,
                        );
                        let expected = match actual {
                            Ok(fields) if fields.contains_key("backend_extensions") => {
                                SamplingTransport::V1WithLegacyCopy
                            }
                            Ok(_) => SamplingTransport::LegacyOnly,
                            Err(_) => SamplingTransport::Rejected,
                        };
                        assert_eq!(rule.transport, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn text_rpc_and_unregistered_profiles_do_not_inherit_token_admission_claims() {
        let profile = CompatibilityProfile::new(
            TargetAdmission::Vllm030,
            Endpoint::Chat,
            PipelineContext::from_pipeline(Endpoint::Chat, ModelInput::Text, None, None),
        );
        let entry = PipelineAdmissionCatalog::new(profile, &ModelRuntimeConfig::default(), None);
        assert!(
            entry
                .sampling_fields
                .iter()
                .all(|field| field.transport == SamplingTransport::OutsideThisAdmissionRule)
        );
        let unknown = PipelineAdmissionCatalog::unavailable(Endpoint::Chat);
        assert!(unknown.admission.is_none());
        assert!(unknown.sampling_fields.is_empty());
        let catalog =
            ModelCompatibilityCatalog::new("model".into(), vec![entry.clone(), unknown, entry]);
        assert_eq!(catalog.profiles.len(), 2);
        assert!(!catalog.coverage_complete);
        assert_eq!(catalog.end_to_end_conformance, "unverified");
    }
}
