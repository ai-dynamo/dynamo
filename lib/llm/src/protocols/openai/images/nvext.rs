// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

/// NVIDIA extensions to the OpenAI Images API
#[derive(ToSchema, Serialize, Deserialize, Validate, Default, Debug, Clone)]
#[schema(as = ImageNvExt)]
pub struct NvExt {
    /// Annotations
    /// User requests triggers which result in the request issue back out-of-band information in the SSE
    /// stream using the `event:` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<String>>,

    /// A text description of the undesired image(s).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,

    /// The number of denoising steps, at least 1. More denoising steps usually lead to a higher quality image at the expense of slower inference.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(minimum = 1)]
    #[validate(range(min = 1, message = "num_inference_steps must be at least 1"))]
    pub num_inference_steps: Option<u8>,

    /// The CFG scale. Higher values usually lead to more coherent images.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guidance_scale: Option<f32>,

    /// The seed for the random number generator.
    /// i64 to match PyTorch's torch.manual_seed() accepted range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}
