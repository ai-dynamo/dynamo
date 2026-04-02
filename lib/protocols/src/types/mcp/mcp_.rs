// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::error::OpenAIError;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum McpToolConnectorId {
    ConnectorDropbox,
    ConnectorGmail,
    ConnectorGooglecalendar,
    ConnectorGoogledrive,
    ConnectorMicrosoftteams,
    ConnectorOutlookcalendar,
    ConnectorOutlookemail,
    ConnectorSharepoint,
}

#[derive(Debug, Serialize, Deserialize, Clone, Builder, PartialEq, Default, ToSchema)]
#[builder(
    name = "MCPToolArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct MCPTool {
    /// A label for this MCP server, used to identify it in tool calls.
    pub server_label: String,

    /// List of allowed tool names or a filter object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<MCPToolAllowedTools>,

    /// An OAuth access token that can be used with a remote MCP server, either with a custom MCP
    /// server URL or a service connector. Your application must handle the OAuth authorization
    /// flow and provide the token here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization: Option<String>,

    /// Identifier for service connectors, like those available in ChatGPT. One of `server_url` or
    /// `connector_id` must be provided. Learn more about service connectors [here](https://platform.openai.com/docs/guides/tools-remote-mcp#connectors).
    ///
    /// Currently supported `connector_id` values are:
    /// - Dropbox: `connector_dropbox`
    /// - Gmail: `connector_gmail`
    /// - Google Calendar: `connector_googlecalendar`
    /// - Google Drive: `connector_googledrive`
    /// - Microsoft Teams: `connector_microsoftteams`
    /// - Outlook Calendar: `connector_outlookcalendar`
    /// - Outlook Email: `connector_outlookemail`
    /// - SharePoint: `connector_sharepoint`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector_id: Option<McpToolConnectorId>,

    /// Optional HTTP headers to send to the MCP server. Use for authentication or other purposes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<serde_json::Value>,

    /// Specify which of the MCP server's tools require approval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_approval: Option<MCPToolRequireApproval>,

    /// Optional description of the MCP server, used to provide more context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_description: Option<String>,

    /// The URL for the MCP server. One of `server_url` or `connector_id` must be provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
#[serde(untagged)]
pub enum MCPToolAllowedTools {
    /// A string array of allowed tool names
    List(Vec<String>),
    /// A filter object to specify which tools are allowed.
    Filter(MCPToolFilter),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct MCPToolFilter {
    /// Indicates whether or not a tool modifies data or is read-only.
    /// If an MCP server is annotated with [readOnlyHint](https://modelcontextprotocol.io/specification/2025-06-18/schema#toolannotations-readonlyhint),
    /// it will match this filter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,
    /// List of allowed tool names.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_names: Option<Vec<String>>,
}

/// Approval policy or filter for MCP tools.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
#[serde(untagged)]
pub enum MCPToolRequireApproval {
    /// Specify which of the MCP server's tools require approval. Can be
    /// `always`, `never`, or a filter object associated with tools
    /// that require approval.
    Filter(MCPToolApprovalFilter),
    /// Specify a single approval policy for all tools. One of `always` or
    /// `never`. When set to `always`, all tools will require approval. When
    /// set to `never`, all tools will not require approval.
    ApprovalSetting(MCPToolApprovalSetting),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum MCPToolApprovalSetting {
    Always,
    Never,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct MCPToolApprovalFilter {
    /// A list of tools that always require approval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub always: Option<MCPToolFilter>,
    /// A list of tools that never require approval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub never: Option<MCPToolFilter>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct MCPListToolsTool {
    /// The JSON schema describing the tool's input.
    pub input_schema: serde_json::Value,
    /// The name of the tool.
    pub name: String,
    /// Additional annotations about the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<serde_json::Value>,
    /// The description of the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}
