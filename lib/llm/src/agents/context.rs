// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Deserializer, Serialize};
use utoipa::ToSchema;

fn deserialize_non_empty_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if value.trim().is_empty() {
        return Err(serde::de::Error::custom(
            "agent_context required identifiers must be non-empty",
        ));
    }
    Ok(value)
}

/// Identity metadata for agentic workloads.
#[derive(ToSchema, Serialize, Deserialize, Builder, Debug, Clone, PartialEq, Eq)]
pub struct AgentContext {
    /// Reusable session/profile class.
    #[serde(deserialize_with = "deserialize_non_empty_string")]
    pub session_type_id: String,

    /// Top-level agent run/session identifier.
    #[serde(deserialize_with = "deserialize_non_empty_string")]
    pub session_id: String,

    /// Schedulable reasoning/tool trajectory identifier.
    #[serde(deserialize_with = "deserialize_non_empty_string")]
    pub trajectory_id: String,

    /// Optional parent trajectory for subagents.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_trajectory_id: Option<String>,
}

impl AgentContext {
    pub fn builder() -> AgentContextBuilder {
        AgentContextBuilder::default()
    }
}

#[cfg(test)]
mod tests {
    use super::AgentContext;

    #[test]
    fn test_agent_context_deserialize_rejects_empty_required_ids() {
        let err = serde_json::from_value::<AgentContext>(serde_json::json!({
            "session_type_id": "deep_research",
            "session_id": "",
            "trajectory_id": "trajectory-1"
        }))
        .expect_err("empty session_id should fail deserialization");

        assert!(
            err.to_string()
                .contains("agent_context required identifiers must be non-empty")
        );
    }
}
