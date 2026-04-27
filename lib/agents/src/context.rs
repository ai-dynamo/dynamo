// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Identity metadata for agentic workloads.
#[derive(ToSchema, Serialize, Deserialize, Builder, Debug, Clone, PartialEq, Eq)]
pub struct AgentContext {
    /// Reusable workflow/profile class.
    pub workflow_type_id: String,

    /// Top-level workflow/run identifier.
    pub workflow_id: String,

    /// Schedulable reason/tool trajectory identifier.
    pub program_id: String,

    /// Optional parent program for subagents.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_program_id: Option<String>,
}

impl AgentContext {
    pub fn builder() -> AgentContextBuilder {
        AgentContextBuilder::default()
    }
}
