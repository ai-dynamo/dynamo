// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::ModelDeploymentCard;
use crate::{
    local_model::runtime_config::{
        SGLANG_GENERATE_CAPABILITY, SPEC_DECODE_RUNTIME_KEY, VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
        VLLM_NATIVE_DECISION_MAX_CANDIDATES,
    },
    model_type::ModelInput,
    worker_type::WorkerType,
};

use crate::decision_profile::{
    DecisionExecutionProfile, DecisionProfileKind, NativeExecutorKind, TokenBackend,
};
use DecisionExecutionProfile as Profile;

fn transport_capability(backend: TokenBackend) -> &'static str {
    match backend {
        TokenBackend::Sglang => SGLANG_GENERATE_CAPABILITY,
        TokenBackend::VllmCausal => VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
    }
}

fn worker_shape_eligible(card: &ModelDeploymentCard) -> bool {
    card.worker_type == Some(WorkerType::Aggregated)
        && card.runtime_config.data_parallel_size == 1
        && card.needs.is_empty()
        && card.lora.is_none()
        && !card
            .runtime_config
            .runtime_data
            .contains_key(SPEC_DECODE_RUNTIME_KEY)
}

pub(super) const ACTIVE_DECISION_PROFILES: &[Profile] =
    &[Profile::SglangSerialV1, Profile::VllmCausalV1];

pub(super) fn classify(card: &ModelDeploymentCard) -> Option<Profile> {
    let mut profiles = ACTIVE_DECISION_PROFILES.iter().copied().filter(|profile| {
        card.runtime_config
            .supports_runtime_capability(profile.decision_capability())
    });
    let profile = profiles.next()?;
    if profiles.next().is_some() || !worker_shape_eligible(card) {
        return None;
    }
    let text_generation = card.model_type.supports_chat() || card.model_type.supports_completions();
    // Private token pipelines fix migration to zero; native PushRouters never add migration.
    // Public-route migration settings therefore do not gate any decision profile.
    let eligible = match profile.kind() {
        DecisionProfileKind::Token(backend) => {
            card.runtime_config.supports_runtime_capability(transport_capability(backend))
                && card.runtime_config.max_num_seqs == Some(1)
                // Preserve SGLang's existing private-router contract independently of public surfaces.
                && (backend == TokenBackend::Sglang || (card.model_input == ModelInput::Tokens && text_generation))
        }
        DecisionProfileKind::Native(NativeExecutorKind::Head) => {
            card.model_input == ModelInput::Text
                && card.model_type.supports_classify()
                && card.model_type.supports_pooling()
                && native_max_candidates(card, profile).is_some()
        }
        DecisionProfileKind::Native(NativeExecutorKind::Read) => {
            matches!(card.model_input, ModelInput::Text | ModelInput::Tokens)
                && text_generation
                && native_max_candidates(card, profile).is_some()
        }
    };
    eligible.then_some(profile)
}

pub(super) fn native_max_candidates(card: &ModelDeploymentCard, profile: Profile) -> Option<usize> {
    card.runtime_config
        .runtime_data
        .get(VLLM_NATIVE_DECISION_MAX_CANDIDATES)?
        .as_u64()
        .and_then(|count| usize::try_from(count).ok())
        .filter(|count| (2..=profile.limits().1).contains(count))
}

pub(super) fn append_checksum(bytes: &mut Vec<u8>, profile: Profile, max_candidates: usize) {
    match profile.kind() {
        DecisionProfileKind::Token(TokenBackend::Sglang) => {
            bytes.extend_from_slice(b"\0sglang_systemone_serial_v1\0true")
        }
        DecisionProfileKind::Token(TokenBackend::VllmCausal) => {
            bytes.extend_from_slice(b"\0vllm_causal_decision_v1\0true")
        }
        DecisionProfileKind::Native(kind) => {
            bytes.extend_from_slice(match kind {
                NativeExecutorKind::Head => b"\0decision_head\0",
                NativeExecutorKind::Read => b"\0decision_read\0",
            });
            bytes.extend_from_slice(profile.decision_capability().as_bytes());
            bytes.extend_from_slice(b"\0vllm_native_decision_max_candidates\0");
            bytes.extend_from_slice(&(max_candidates as u64).to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_model::runtime_config::{SGLANG_SYSTEMONE_SERIAL_V1, VLLM_CAUSAL_DECISION_V1};

    #[test]
    fn mixed_token_decision_flags_are_unsupported_without_a_checksum_suffix() {
        let mut card = ModelDeploymentCard::with_name_only("mixed-model");
        card.worker_type = Some(WorkerType::Aggregated);
        card.runtime_config.max_num_seqs = Some(1);
        for capability in [
            SGLANG_SYSTEMONE_SERIAL_V1,
            transport_capability(TokenBackend::Sglang),
            VLLM_CAUSAL_DECISION_V1,
        ] {
            card.runtime_config
                .runtime_data
                .insert(capability.into(), true.into());
        }
        let mut no_decision = card.clone();
        no_decision
            .runtime_config
            .runtime_data
            .remove(SGLANG_SYSTEMONE_SERIAL_V1);
        no_decision
            .runtime_config
            .runtime_data
            .remove(VLLM_CAUSAL_DECISION_V1);
        assert!(classify(&card).is_none());
        assert_eq!(card.mdcsum(), no_decision.mdcsum());
    }

    #[test]
    fn decision_checksum_suffixes_preserve_existing_bytes() {
        for (profile, expected) in [
            (
                Profile::SglangSerialV1,
                "\0sglang_systemone_serial_v1\0true",
            ),
            (Profile::VllmCausalV1, "\0vllm_causal_decision_v1\0true"),
            (Profile::VllmD3V1, "\0decision_head\0vllm_d3_decision_v1"),
            (
                Profile::VllmClefFlashV1,
                "\0decision_head\0vllm_clef_flash_joint_decision_v1",
            ),
            (
                Profile::VllmClefOmniV1,
                "\0decision_head\0vllm_clef_omni_joint_decision_v1",
            ),
            (
                Profile::VllmLiquidV1,
                "\0decision_read\0vllm_liquid_decision_v1",
            ),
            (
                Profile::VllmDiffusionV1,
                "\0decision_read\0vllm_diffusion_canvas_decision_v1",
            ),
        ] {
            let mut bytes = Vec::new();
            append_checksum(&mut bytes, profile, profile.limits().1);
            let mut expected = expected.as_bytes().to_vec();
            if matches!(profile.kind(), DecisionProfileKind::Native(_)) {
                expected.extend_from_slice(b"\0vllm_native_decision_max_candidates\0");
                expected.extend_from_slice(&(profile.limits().1 as u64).to_le_bytes());
            }
            assert_eq!(bytes, expected);
            assert_eq!(
                serde_json::to_value(profile).unwrap(),
                profile.decision_capability()
            );
            assert_eq!(
                serde_json::from_value::<Profile>(serde_json::to_value(profile).unwrap()).unwrap(),
                profile
            );
        }
    }
}
