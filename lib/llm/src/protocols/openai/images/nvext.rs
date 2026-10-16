// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// NVIDIA extensions to the OpenAI Images API
#[derive(ToSchema, Serialize, Deserialize, Default, Debug, Clone)]
pub struct NvExt {
    /// Annotations
    /// User requests triggers which result in the request issue back out-of-band information in the SSE
    /// stream using the `event:` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<String>>,

    /// A text description of the undesired image(s).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,

    /// The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_inference_steps: Option<u8>,

    /// The CFG scale. Higher values usually lead to more coherent images.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guidance_scale: Option<f32>,

    /// The seed for the random number generator.
    /// i64 to match PyTorch's torch.manual_seed() accepted range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

/// NVIDIA extensions to the OpenAI Images API response. Omitted from the
/// wire when the worker reports nothing.
#[derive(ToSchema, Serialize, Deserialize, Default, Debug, Clone, PartialEq, Eq)]
pub struct ImagesResponseNvExt {
    /// Image-based accounting for models that have no token usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ImageAccounting>,
}

/// Per-request accounting in image units.
///
/// `output_images` counts what the response carries, not the requested `n`.
/// The image counts are always present; only `num_inference_steps` can be
/// omitted, and an unknown step count is omitted, never written as 0. See
/// [`ImageAccounting::merge`] for how a multi-chunk response treats each field.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageAccounting {
    /// Images generated and returned in `data`.
    pub output_images: u32,

    /// Reference images the request consumed: 0 for text-to-image, 1 for an
    /// `input_reference` or an `/v1/images/edits` request.
    #[serde(default)]
    pub input_images: u32,

    /// The step count the backend ran. Omitted when the backend does not
    /// know it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_inference_steps: Option<u8>,
}

impl ImageAccounting {
    /// Fold a later chunk's accounting into this one: `output_images` adds
    /// up, `input_images` keeps the first reported value, and
    /// `num_inference_steps` is cleared when the chunks disagree or one does
    /// not report it.
    pub fn merge(&mut self, next: Self) {
        self.output_images = self.output_images.saturating_add(next.output_images);
        if self.num_inference_steps != next.num_inference_steps {
            self.num_inference_steps = None;
        }
    }
}

impl ImagesResponseNvExt {
    /// Fold a later chunk's extension into this one; see [`ImageAccounting::merge`].
    pub fn merge(&mut self, next: Self) {
        match (self.usage.as_mut(), next.usage) {
            (Some(usage), Some(next)) => usage.merge(next),
            (None, next) => self.usage = next,
            (Some(_), None) => {}
        }
    }
}
