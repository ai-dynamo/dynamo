// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// NVIDIA extensions to the OpenAI Videos API
#[derive(ToSchema, Serialize, Deserialize, Default, Debug, Clone)]
pub struct NvExt {
    /// Annotations
    /// User requests triggers which result in the request issue back out-of-band information in the SSE
    /// stream using the `event:` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<String>>,

    /// Frames per second (default: 24)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<i32>,

    /// Number of frames to generate (overrides fps * seconds if set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_frames: Option<i32>,

    /// A text description of the undesired video content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,

    /// The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_inference_steps: Option<i32>,

    /// The CFG scale. Higher values usually lead to more coherent output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guidance_scale: Option<f32>,

    /// The seed for the random number generator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// MoE expert switching boundary as a fraction of the denoising schedule (vLLM-Omni I2V).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boundary_ratio: Option<f32>,

    /// CFG scale for the low-noise expert (vLLM-Omni I2V dual-guidance).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guidance_scale_2: Option<f32>,
}
