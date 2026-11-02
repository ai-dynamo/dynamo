// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::ModelDeploymentCard;
use crate::{
    decision_profile::{
        DecisionExecutionProfile, DecisionProfileKind, NativeExecutorKind, TokenBackend,
        VLLM_NATIVE_DECISION_MAX_CANDIDATES, VLLM_NATIVE_DECISION_QUALIFICATION_FAILURE_V1,
    },
    local_model::runtime_config::{
        SGLANG_GENERATE_CAPABILITY, SPEC_DECODE_RUNTIME_KEY, VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
    },
    model_type::ModelInput,
    worker_type::WorkerType,
};

use DecisionExecutionProfile as Profile;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DecisionKinds {
    pub(crate) token: bool,
    pub(crate) head: bool,
    pub(crate) read: bool,
}

impl DecisionKinds {
    pub(crate) fn eligible(card: &ModelDeploymentCard) -> Self {
        let text_generation =
            card.model_type.supports_chat() || card.model_type.supports_completions();
        Self {
            token: card.model_input == ModelInput::Tokens && text_generation,
            head: card.model_input == ModelInput::Text
                && card.model_type.supports_classify()
                && card.model_type.supports_pooling(),
            read: matches!(card.model_input, ModelInput::Text | ModelInput::Tokens)
                && text_generation,
        }
    }

    pub(crate) fn intersection(self, other: Self) -> Self {
        Self {
            token: self.token && other.token,
            head: self.head && other.head,
            read: self.read && other.read,
        }
    }

    pub(crate) fn supports(self, kind: DecisionProfileKind) -> bool {
        match kind {
            DecisionProfileKind::Token(_) => self.token,
            DecisionProfileKind::Native(NativeExecutorKind::Head) => self.head,
            DecisionProfileKind::Native(NativeExecutorKind::Read) => self.read,
        }
    }

    pub(crate) fn any(self) -> bool {
        self.token || self.head || self.read
    }

    pub(crate) fn advertised(card: &ModelDeploymentCard) -> Self {
        let mut kinds = Self::default();
        // Failed qualification retains structural intent, never a success capability.
        for profile in card
            .advertised_decision_profiles()
            .chain(card.decision_qualification_failure())
        {
            match profile.kind() {
                DecisionProfileKind::Token(_) => kinds.token = true,
                DecisionProfileKind::Native(NativeExecutorKind::Head) => kinds.head = true,
                DecisionProfileKind::Native(NativeExecutorKind::Read) => kinds.read = true,
            }
        }
        kinds
    }

    pub(crate) fn include(&mut self, other: Self) {
        self.token |= other.token;
        self.head |= other.head;
        self.read |= other.read;
    }
}

pub(crate) fn transport_capability(backend: TokenBackend) -> &'static str {
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

pub(super) fn qualification_failure(card: &ModelDeploymentCard) -> Option<Profile> {
    #[derive(Deserialize)]
    struct Failure {
        profile: Profile,
    }
    let failure = Failure::deserialize(
        card.runtime_config
            .runtime_data
            .get(VLLM_NATIVE_DECISION_QUALIFICATION_FAILURE_V1)?,
    )
    .ok()?;
    (worker_shape_eligible(card)
        && matches!(failure.profile.kind(), DecisionProfileKind::Native(_))
        && DecisionKinds::eligible(card).supports(failure.profile.kind()))
    .then_some(failure.profile)
}

pub(super) const ACTIVE_DECISION_PROFILES: &[Profile] = &[
    Profile::SglangSerialV1,
    Profile::VllmCausalV1,
    Profile::VllmD3V1,
    Profile::VllmClefFlashV1,
    Profile::VllmClefOmniV1,
    Profile::VllmLiquidV1,
    Profile::VllmDiffusionV1,
];

pub(super) fn classify(card: &ModelDeploymentCard) -> Option<Profile> {
    let mut profiles = card.advertised_decision_profiles();
    let profile = profiles.next()?;
    if profiles.next().is_some()
        || !worker_shape_eligible(card)
        || !DecisionKinds::eligible(card).supports(profile.kind())
    {
        return None;
    }
    // Private score transports and native PushRouters do not add generation migration.
    // Public-route migration settings therefore do not gate any decision profile.
    let eligible = match profile.kind() {
        DecisionProfileKind::Token(backend) => {
            card.runtime_config
                .supports_runtime_capability(transport_capability(backend))
                && card.runtime_config.max_num_seqs == Some(1)
        }
        DecisionProfileKind::Native(_) => native_max_candidates(card, profile).is_some(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_profile::{SGLANG_SYSTEMONE_SERIAL_V1, VLLM_CAUSAL_DECISION_V1};

    #[test]
    fn qualification_failure_requires_a_recognized_native_worker_profile() {
        let mut card = ModelDeploymentCard::with_name_only("failure-shape");
        card.model_input = ModelInput::Text;
        card.model_type =
            crate::model_type::ModelType::Classify | crate::model_type::ModelType::Pooling;
        card.worker_type = Some(WorkerType::Aggregated);
        let ordinary_checksum = card.mdcsum().to_owned();
        for (marker, expected) in [
            (
                serde_json::json!({"profile":"vllm_d3_decision_v1"}),
                Some(Profile::VllmD3V1),
            ),
            (
                serde_json::json!({"profile":"sglang_systemone_serial_v1"}),
                None,
            ),
            (
                serde_json::json!({"profile":"vllm_liquid_decision_v1"}),
                None,
            ),
            (serde_json::json!({"profile":"unknown"}), None),
            (serde_json::json!(true), None),
        ] {
            card.runtime_config
                .runtime_data
                .insert(VLLM_NATIVE_DECISION_QUALIFICATION_FAILURE_V1.into(), marker);
            assert_eq!(qualification_failure(&card), expected);
            let reparsed: ModelDeploymentCard =
                serde_json::from_value(serde_json::to_value(&card).unwrap()).unwrap();
            assert_eq!(reparsed.mdcsum(), ordinary_checksum);
        }
    }

    #[test]
    fn token_backends_share_structural_card_eligibility() {
        for profile in [Profile::SglangSerialV1, Profile::VllmCausalV1] {
            let mut card = ModelDeploymentCard::with_name_only("token-shape");
            card.worker_type = Some(WorkerType::Aggregated);
            card.runtime_config.max_num_seqs = Some(1);
            for capability in [
                profile.decision_capability(),
                match profile.kind() {
                    DecisionProfileKind::Token(backend) => transport_capability(backend),
                    _ => unreachable!(),
                },
            ] {
                card.runtime_config
                    .runtime_data
                    .insert(capability.into(), true.into());
            }
            for (input, model_type, expected) in [
                (ModelInput::Tokens, crate::model_type::ModelType::Chat, true),
                (ModelInput::Text, crate::model_type::ModelType::Chat, false),
                (
                    ModelInput::Tokens,
                    crate::model_type::ModelType::Embedding,
                    false,
                ),
            ] {
                card.model_input = input;
                card.model_type = model_type;
                assert_eq!(
                    DecisionKinds::eligible(&card).supports(profile.kind()),
                    expected
                );
                assert_eq!(classify(&card), expected.then_some(profile));
            }
        }
    }

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
    fn decision_profiles_preserve_wire_names() {
        for &profile in ACTIVE_DECISION_PROFILES {
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
