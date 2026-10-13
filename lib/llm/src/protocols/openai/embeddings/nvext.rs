// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

pub trait NvExtProvider {
    fn nvext(&self) -> Option<&NvExt>;
}

/// NVIDIA LLM extensions to the OpenAI API
#[derive(ToSchema, Serialize, Deserialize, Default, Debug, Clone)]
pub struct NvExt {
    /// Annotations
    /// User requests triggers which result in the request issue back out-of-band information in the SSE
    /// stream using the `event:` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<String>>,
}
