// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::error::OpenAIError;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::types::responses::{IncludeParam, ListOrder};

/// Query parameters for listing conversation items.
#[derive(Clone, Serialize, Default, Debug, Deserialize, Builder, PartialEq, ToSchema)]
#[builder(name = "ListConversationItemsQueryArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ListConversationItemsQuery {
    /// A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// The order to return the input items in. Default is `desc`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<ListOrder>,
    /// An item ID to list items after, used in pagination.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
    /// Specify additional output data to include in the model response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeParam>>,
}

/// Query parameters for getting a response.
#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq, ToSchema)]
#[builder(name = "GetResponseQueryArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct GetResponseQuery {
    /// Additional fields to include in the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    /// If set to true, the model response data will be streamed to the client as it is generated using server-sent events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// The sequence number of the event after which to start streaming.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub starting_after: Option<u32>,
    /// When true, stream obfuscation will be enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
}

/// Query parameters for listing input items.
#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq, ToSchema)]
#[builder(name = "ListInputItemsQueryArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ListInputItemsQuery {
    /// A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// The order to return the input items in. Default is `desc`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<ListOrder>,
    /// An item ID to list items after, used in pagination.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
    /// Additional fields to include in the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
}

/// Query parameters for getting a conversation item.
#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq, ToSchema)]
#[builder(name = "GetConversationItemQueryArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct GetConversationItemQuery {
    /// Additional fields to include in the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeParam>>,
}

/// Query parameters for creating conversation items.
#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq, ToSchema)]
#[builder(name = "CreateConversationItemsQueryArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CreateConversationItemsQuery {
    /// Additional fields to include in the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeParam>>,
}
