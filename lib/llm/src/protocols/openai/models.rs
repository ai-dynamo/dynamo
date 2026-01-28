// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelInfo {
    object: String, // "list"
    data: Vec<ModelMetaData>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelMetaData {
    /// Model's ID - must be unique
    id: String,

    /// Always "model"
    object: String, // "model"

    /// Unix timestamp of when the model was created
    /// See <https://en.wikipedia.org/wiki/Unix_time>
    created: u64,

    /// Name of user or group that owns the model
    /// Defaults to "skynet" if not provided
    owned_by: String,

    /// Per-Org permissions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    permission: Option<Vec<Permission>>,

    // TODO(#30): docstring needed
    #[serde(default, skip_serializing_if = "Option::is_none")]
    root: Option<String>,

    // TODO(#30): docstring needed
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Permission {
    /// Unique identifier for the permission
    /// Example: modelperm-8ca99ea29964429ab51e968892b2b708"
    id: String,

    /// Always "model_permission"
    object: String, // "model_permission"

    /// Unix timestamp of when the permission was created
    /// See <https://en.wikipedia.org/wiki/Unix_time>
    created: u64,

    /// Name of the organization for which the permission is applicable
    organization: String,

    /// The name of the group that this permission belongs to
    #[serde(default, skip_serializing_if = "Option::is_none")]
    group: Option<String>,

    /// Whether the organization can create engines with this model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allow_create_engine: Option<bool>,

    /// Whether the organization can perform sampling/inference with this model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allow_sampling: Option<bool>,

    /// Whether the organization can request log probabilities for model outputs
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allow_logprobs: Option<bool>,

    /// Whether the organization can perform search operations with this model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allow_search_indices: Option<bool>,

    /// Whether the organization can view this model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allow_view: Option<bool>,

    /// Whether the organization can fine-tune this model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allow_fine_tuning: Option<bool>,

    /// Whether this permission blocks access to the model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    is_blocking: Option<bool>,
}
