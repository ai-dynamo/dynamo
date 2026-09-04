// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serving behavior layered on top of a worker's processing stage.

use anyhow::ensure;
use dynamo_runtime::protocols::EndpointId;
use serde::{Deserialize, Serialize};

use crate::{
    protocols::external_speculation::{validate_endpoint_id, validate_protocol},
    worker_type::WorkerType,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalDraftBinding {
    pub endpoint: EndpointId,
    pub protocol: String,
    pub router_hint_schema_version: u16,
}

impl ExternalDraftBinding {
    pub const ROUTER_HINT_SCHEMA_VERSION: u16 = 1;

    pub fn validate(&self) -> anyhow::Result<()> {
        validate_endpoint_id(&self.endpoint).map_err(anyhow::Error::msg)?;
        validate_protocol(&self.protocol)?;
        ensure!(
            self.router_hint_schema_version == Self::ROUTER_HINT_SCHEMA_VERSION,
            "unsupported speculative router-hint schema version {}; expected {}",
            self.router_hint_schema_version,
            Self::ROUTER_HINT_SCHEMA_VERSION
        );
        Ok(())
    }
}

/// A worker's serving contract. The default standard role preserves legacy wire behavior.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "role", content = "binding", rename_all = "snake_case")]
pub enum WorkerRole {
    #[default]
    Standard,
    SpeculativeTarget(ExternalDraftBinding),
    SpeculativeDraft,
}

impl WorkerRole {
    pub fn is_standard(&self) -> bool {
        matches!(self, Self::Standard)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::SpeculativeTarget(_) => "speculative_target",
            Self::SpeculativeDraft => "speculative_draft",
        }
    }

    pub fn target_binding(&self) -> Option<&ExternalDraftBinding> {
        match self {
            Self::SpeculativeTarget(binding) => Some(binding),
            Self::Standard | Self::SpeculativeDraft => None,
        }
    }

    pub fn validate(
        &self,
        worker_type: WorkerType,
        has_public_surface: bool,
        own_endpoint: Option<&EndpointId>,
    ) -> anyhow::Result<()> {
        match self {
            Self::Standard => Ok(()),
            Self::SpeculativeTarget(binding) => {
                ensure!(
                    worker_type == WorkerType::Aggregated,
                    "speculative targets must use worker_type=aggregated"
                );
                ensure!(
                    has_public_surface,
                    "speculative targets must expose a public model surface"
                );
                binding.validate()?;
                if let Some(own_endpoint) = own_endpoint {
                    ensure!(
                        &binding.endpoint != own_endpoint,
                        "speculative target cannot bind its own endpoint as the draft"
                    );
                }
                Ok(())
            }
            Self::SpeculativeDraft => {
                ensure!(
                    worker_type == WorkerType::Aggregated,
                    "speculative drafts must use worker_type=aggregated"
                );
                ensure!(
                    !has_public_surface,
                    "speculative drafts must not expose a public model surface"
                );
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> ExternalDraftBinding {
        ExternalDraftBinding {
            endpoint: EndpointId::from("specdec/draft/generate"),
            protocol: "mock-specdec-zmq-v1".into(),
            router_hint_schema_version: 1,
        }
    }

    #[test]
    fn standard_is_default_and_has_tagged_wire_form() {
        assert_eq!(WorkerRole::default(), WorkerRole::Standard);
        assert_eq!(
            serde_json::to_value(WorkerRole::SpeculativeTarget(binding())).unwrap(),
            serde_json::json!({
                "role": "speculative_target",
                "binding": {
                    "endpoint": {
                        "namespace": "specdec",
                        "component": "draft",
                        "name": "generate"
                    },
                    "protocol": "mock-specdec-zmq-v1",
                    "router_hint_schema_version": 1
                }
            })
        );
    }

    #[test]
    fn nonstandard_roles_are_aggregated_only() {
        assert!(
            WorkerRole::SpeculativeTarget(binding())
                .validate(WorkerType::Decode, true, None)
                .is_err()
        );
        assert!(
            WorkerRole::SpeculativeDraft
                .validate(WorkerType::Prefill, false, None)
                .is_err()
        );
    }

    #[test]
    fn target_and_draft_surface_rules_are_strict() {
        assert!(
            WorkerRole::SpeculativeTarget(binding())
                .validate(WorkerType::Aggregated, false, None)
                .is_err()
        );
        assert!(
            WorkerRole::SpeculativeDraft
                .validate(WorkerType::Aggregated, true, None)
                .is_err()
        );
    }

    #[test]
    fn target_cannot_bind_to_itself() {
        let own_endpoint = EndpointId::from("specdec/draft/generate");
        assert!(
            WorkerRole::SpeculativeTarget(binding())
                .validate(WorkerType::Aggregated, true, Some(&own_endpoint))
                .is_err()
        );
    }
}
