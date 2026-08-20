// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::{Validate, ValidationError};

pub trait NvExtProvider {
    fn nvext(&self) -> Option<&NvExt>;
}

#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum StartTimeSeconds {
    Scalar(f32),
    List(Vec<f32>),
}

/// NVIDIA extensions to the OpenAI Videos API
#[derive(ToSchema, Serialize, Deserialize, Builder, Validate, Debug, Clone)]
#[validate(schema(function = "validate_nv_ext"))]
pub struct NvExt {
    /// Annotations
    /// User requests triggers which result in the request issue back out-of-band information in the SSE
    /// stream using the `event:` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub annotations: Option<Vec<String>>,

    /// Frames per second (default: 24)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub fps: Option<i32>,

    /// Number of frames to generate (overrides fps * seconds if set)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub num_frames: Option<i32>,

    /// A text description of the undesired video content.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub negative_prompt: Option<String>,

    /// The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub num_inference_steps: Option<i32>,

    /// The CFG scale. Higher values usually lead to more coherent output.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub guidance_scale: Option<f32>,

    /// The seed for the random number generator.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub seed: Option<i64>,

    /// MoE expert switching boundary as a fraction of the denoising schedule (vLLM-Omni I2V).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub boundary_ratio: Option<f32>,

    /// CFG scale for the low-noise expert (vLLM-Omni I2V dual-guidance).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub guidance_scale_2: Option<f32>,

    /// MiniMax-H3 task routed to its FL2VA or Ref2VA transformer.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub task: Option<String>,

    /// Requested MiniMax-H3 duration in seconds (4 through 15).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub duration: Option<f32>,

    /// MiniMax-H3 video sigma shift.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub flow_shift: Option<f32>,

    /// MiniMax-H3 audio sigma shift.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub audio_flow_shift: Option<f32>,

    /// MiniMax-H3 output aspect ratio.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub aspect_ratio: Option<String>,

    /// MiniMax-H3 output canvas short edge.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub short_edge: Option<i32>,

    /// FL2VA keyframe positions: [0], [-1], or [0, -1].
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub frame_indices: Option<Vec<i32>>,

    /// Start offset for one reference video, or one offset per video.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub start_time_seconds: Option<StartTimeSeconds>,

    /// Number of generated videos (MiniMax-H3 supports 1 through 10).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub num_outputs_per_prompt: Option<i32>,

    /// MiniMax-H3 request-scoped quality policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub quality: Option<String>,
}

impl Default for NvExt {
    fn default() -> Self {
        NvExt::builder().build().unwrap()
    }
}

impl NvExt {
    pub fn builder() -> NvExtBuilder {
        NvExtBuilder::default()
    }
}

fn validate_nv_ext(nv_ext: &NvExt) -> Result<(), ValidationError> {
    if nv_ext
        .task
        .as_deref()
        .is_some_and(|task| !matches!(task, "t2va" | "fl2va" | "ref2va"))
    {
        return Err(ValidationError::new("invalid_h3_task"));
    }
    if nv_ext
        .duration
        .is_some_and(|duration| !duration.is_finite() || !(4.0..=15.0).contains(&duration))
    {
        return Err(ValidationError::new("invalid_h3_duration"));
    }
    if nv_ext
        .num_outputs_per_prompt
        .is_some_and(|count| !(1..=10).contains(&count))
    {
        return Err(ValidationError::new("invalid_h3_output_count"));
    }
    if nv_ext.task.is_some() && nv_ext.fps.is_some_and(|fps| fps != 24) {
        return Err(ValidationError::new("invalid_h3_fps"));
    }
    if nv_ext
        .quality
        .as_deref()
        .is_some_and(|quality| !matches!(quality, "lossless" | "high"))
    {
        return Err(ValidationError::new("invalid_h3_quality"));
    }
    Ok(())
}

impl NvExtBuilder {
    pub fn add_annotation(&mut self, annotation: impl Into<String>) -> &mut Self {
        self.annotations
            .get_or_insert_with(|| Some(vec![]))
            .as_mut()
            .expect("annotations should always be Some(Vec)")
            .push(annotation.into());
        self
    }
}
