// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::fmt;
use strum::Display;

bitflags! {
    /// Represents the set of model capabilities (endpoints) a model can support.
    ///
    /// This type is implemented using `bitflags` instead of a plain `enum`
    /// so that multiple capabilities can be combined in a single value:
    ///
    /// - `ModelType::Chat`
    /// - `ModelType::Completions`
    /// - `ModelType::Embedding`
    /// - `ModelType::TensorBased`
    ///
    /// For example, a model that supports both chat and completions can be
    /// expressed as:
    ///
    /// ```rust
    /// use dynamo_llm::model_type::ModelType;
    /// let mt = ModelType::Chat | ModelType::Completions;
    /// assert!(mt.supports_chat());
    /// assert!(mt.supports_completions());
    /// ```
    ///
    /// Using bitflags avoids deep branching on a single enum variant,
    /// simplifies checks like `supports_chat()`, and enables efficient,
    /// type-safe combinations of multiple endpoint types.
    #[derive(Copy, Debug, Default, Clone, Serialize, Deserialize, Eq, PartialEq)]
    pub struct ModelType: u16 {
        const Chat = 1 << 0;
        const Completions = 1 << 1;
        const Embedding = 1 << 2;
        const TensorBased = 1 << 3;
        // `Prefill` (bit 1 << 4) is a legacy *marker*, not an OpenAI surface.
        // The processing-stage role (prefill / decode / encode / aggregated)
        // is expressed via [`crate::worker_type::WorkerType`]; `ModelType`
        // otherwise only describes the OpenAI-style endpoints a model exposes.
        //
        // It is retained for **cross-version compatibility**: a *new* prefill
        // worker dual-emits this bit so an
        // *old* frontend (which detects prefill via `supports_prefill()`)
        // still routes disaggregated traffic, and a *new* frontend still
        // recognizes an *old* prefill card. It grants no `supports_*` surface.
        // TODO(compat follow-up): remove once the compat window closes.
        const Prefill = 1 << 4;
        const Images = 1 << 5;
        const Audios = 1 << 6;
        const Videos = 1 << 7;
        const Realtime = 1 << 8;
    }
}

impl ModelType {
    pub fn as_str(&self) -> String {
        self.as_vec().join(",")
    }

    pub fn supports_chat(&self) -> bool {
        self.contains(ModelType::Chat)
    }
    pub fn supports_completions(&self) -> bool {
        self.contains(ModelType::Completions)
    }
    pub fn supports_embedding(&self) -> bool {
        self.contains(ModelType::Embedding)
    }
    pub fn supports_tensor(&self) -> bool {
        self.contains(ModelType::TensorBased)
    }
    /// Legacy prefill *marker* (not an OpenAI surface). True when the card
    /// carries the retained `ModelType::Prefill` compat bit. New code should
    /// prefer `WorkerType::Prefill`; this exists for cross-version detection
    /// of legacy prefill cards during the cross-version rollout.
    pub fn supports_prefill(&self) -> bool {
        self.contains(ModelType::Prefill)
    }
    pub fn supports_images(&self) -> bool {
        self.contains(ModelType::Images)
    }
    pub fn supports_audios(&self) -> bool {
        self.contains(ModelType::Audios)
    }
    pub fn supports_videos(&self) -> bool {
        self.contains(ModelType::Videos)
    }
    pub fn supports_realtime(&self) -> bool {
        self.contains(ModelType::Realtime)
    }

    pub fn as_vec(&self) -> Vec<&'static str> {
        let mut result = Vec::new();
        if self.supports_chat() {
            result.push("chat");
        }
        if self.supports_completions() {
            result.push("completions");
        }
        if self.supports_embedding() {
            result.push("embedding");
        }
        if self.supports_tensor() {
            result.push("tensor");
        }
        if self.supports_prefill() {
            result.push("prefill");
        }
        if self.supports_images() {
            result.push("images");
        }
        if self.supports_audios() {
            result.push("audios");
        }
        if self.supports_videos() {
            result.push("videos");
        }
        if self.supports_realtime() {
            result.push("realtime");
        }
        result
    }

    /// Decompose the bitflag into it's component units:
    /// Chat | Completion -> [Chat, Completion]
    pub fn units(&self) -> Vec<ModelType> {
        let mut result = Vec::new();
        if self.supports_chat() {
            result.push(ModelType::Chat);
        }
        if self.supports_completions() {
            result.push(ModelType::Completions);
        }
        if self.supports_embedding() {
            result.push(ModelType::Embedding);
        }
        if self.supports_tensor() {
            result.push(ModelType::TensorBased);
        }
        if self.supports_prefill() {
            result.push(ModelType::Prefill);
        }
        if self.supports_images() {
            result.push(ModelType::Images);
        }
        if self.supports_audios() {
            result.push(ModelType::Audios);
        }
        if self.supports_videos() {
            result.push(ModelType::Videos);
        }
        if self.supports_realtime() {
            result.push(ModelType::Realtime);
        }
        result
    }

    /// Returns all endpoint types supported by this model type.
    /// This properly handles combinations like Chat | Completions.
    pub fn as_endpoint_types(&self) -> Vec<crate::endpoint_type::EndpointType> {
        self.as_endpoint_types_with_anthropic(dynamo_runtime::config::env_is_truthy(
            dynamo_runtime::config::environment_names::llm::DYN_ENABLE_ANTHROPIC_API,
        ))
    }

    /// Returns all endpoint types supported by this model type using an explicit
    /// Anthropic API gate.
    pub fn as_endpoint_types_with_anthropic(
        &self,
        enable_anthropic_api: bool,
    ) -> Vec<crate::endpoint_type::EndpointType> {
        let mut endpoint_types = Vec::new();
        if self.contains(Self::Chat) {
            endpoint_types.push(crate::endpoint_type::EndpointType::Chat);
            // Translation layers over chat completions
            endpoint_types.push(crate::endpoint_type::EndpointType::Responses);
            if enable_anthropic_api {
                endpoint_types.push(crate::endpoint_type::EndpointType::AnthropicMessages);
            }
        }
        if self.contains(Self::Completions) {
            endpoint_types.push(crate::endpoint_type::EndpointType::Completion);
        }
        // Token-in/token-out Generate endpoint is available for any
        // generate-capable model (chat or completions), matching the
        // default-on posture of the `/inference/v1/generate` route.
        if self.supports_chat() || self.supports_completions() {
            endpoint_types.push(crate::endpoint_type::EndpointType::Generate);
        }
        if self.contains(Self::Embedding) {
            endpoint_types.push(crate::endpoint_type::EndpointType::Embedding);
        }
        if self.contains(Self::Images) {
            endpoint_types.push(crate::endpoint_type::EndpointType::Images);
        }
        if self.contains(Self::Audios) {
            endpoint_types.push(crate::endpoint_type::EndpointType::Audios);
        }
        if self.contains(Self::Videos) {
            endpoint_types.push(crate::endpoint_type::EndpointType::Videos);
        }
        if self.contains(Self::Realtime) {
            endpoint_types.push(crate::endpoint_type::EndpointType::Realtime);
        }
        // [gluo NOTE] ModelType::Tensor doesn't map to any endpoint type,
        // current use of endpoint type is LLM specific and so does the HTTP
        // server that uses it.
        endpoint_types
    }
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Copy, Debug, Default, Clone, Display, Serialize, Deserialize, Eq, PartialEq)]
pub enum ModelInput {
    /// Raw text input
    #[default]
    Text,
    /// Pre-processed input
    Tokens,
    /// Tensor input
    Tensor,
}

impl ModelInput {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text => "text",
            Self::Tokens => "tokens",
            Self::Tensor => "tensor",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoint_type::EndpointType;

    #[test]
    fn realtime_bit_position() {
        assert_eq!(ModelType::Realtime.bits(), 1 << 8);
    }

    #[test]
    fn prefill_bit_position_unchanged() {
        // Retained for cross-version compat; must stay at 1 << 4 so old and
        // new frontends/workers agree on the legacy prefill marker.
        assert_eq!(ModelType::Prefill.bits(), 1 << 4);
    }

    #[test]
    fn prefill_is_marker_not_a_surface() {
        // The Prefill bit must not advertise any OpenAI surface.
        let p = ModelType::Prefill;
        assert!(p.supports_prefill());
        assert!(!p.supports_chat());
        assert!(!p.supports_completions());
        assert!(!p.supports_embedding());
        assert!(!p.supports_tensor());
        assert!(!p.supports_images());
        assert!(!p.supports_audios());
        assert!(!p.supports_videos());
        assert!(!p.supports_realtime());
    }

    #[test]
    fn prefill_in_as_vec_and_units() {
        assert_eq!(ModelType::Prefill.as_vec(), vec!["prefill"]);
        assert_eq!(ModelType::Prefill.units(), vec![ModelType::Prefill]);
    }

    #[test]
    fn prefill_serde_round_trip() {
        // ModelType serializes as a `|`-separated string of flag names. An old
        // prefill card carries model_type="prefill"; a NEW frontend must be
        // able to deserialize it (the whole reason the Prefill bit is retained
        // — without it, "prefill" is an unknown flag and the card is rejected).
        // serde uses the capitalized bitflags identifier ("Prefill"); as_vec()
        // uses lowercase ("prefill") for ws_keys. Both are stable across
        // versions, which is what cross-version compat relies on.
        let json = serde_json::to_string(&ModelType::Prefill).unwrap();
        assert_eq!(json, "\"Prefill\"");
        let back: ModelType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ModelType::Prefill);
        // The legacy wire string deserializes to the Prefill marker.
        let from_legacy: ModelType = serde_json::from_str("\"Prefill\"").unwrap();
        assert_eq!(from_legacy, ModelType::Prefill);
    }

    #[test]
    fn realtime_supports_realtime() {
        assert!(ModelType::Realtime.supports_realtime());
        assert!(!ModelType::Chat.supports_realtime());
    }

    #[test]
    fn realtime_in_as_vec() {
        assert_eq!(ModelType::Realtime.as_vec(), vec!["realtime"]);
    }

    #[test]
    fn realtime_in_units() {
        let combined = ModelType::Chat | ModelType::Realtime;
        assert_eq!(combined.units(), vec![ModelType::Chat, ModelType::Realtime]);
    }

    #[test]
    fn realtime_endpoint_mapping() {
        assert_eq!(
            ModelType::Realtime.as_endpoint_types(),
            vec![EndpointType::Realtime]
        );
    }

    #[test]
    fn realtime_combines_with_other_endpoints() {
        let endpoints = (ModelType::Chat | ModelType::Realtime).as_endpoint_types();
        assert!(endpoints.contains(&EndpointType::Chat));
        assert!(endpoints.contains(&EndpointType::Realtime));
    }
}
