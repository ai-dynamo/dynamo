// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::{Validate, ValidationError};

/// Provides access to the optional NVIDIA extensions on a request.
pub trait NvExtProvider {
    /// Returns a reference to the NVIDIA extensions, if present.
    fn nvext(&self) -> Option<&NvExt>;
}

/// NVIDIA extensions to the OpenAI Audio Speech API
#[derive(ToSchema, Serialize, Deserialize, Builder, Validate, Debug, Clone)]
#[validate(schema(function = "validate_nv_ext"))]
pub struct NvExt {
    /// Annotations
    /// User requests triggers which result in the request issue back out-of-band information in the SSE
    /// stream using the `event:` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub annotations: Option<Vec<String>>,

    /// Task type (e.g. "CustomVoice", "Base")
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub task_type: Option<String>,

    /// Language code (e.g. "en", "zh")
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub language: Option<String>,

    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub max_new_tokens: Option<u32>,

    /// Base64-encoded reference audio for voice cloning
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub ref_audio: Option<String>,

    /// Reference text corresponding to ref_audio
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub ref_text: Option<String>,

    /// If True, the model uses only the speaker's acoustic features (embedding) without considering the ref_text.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub x_vector_only_mode: Option<bool>,

    /// A pre-computed vector (1024 or 2048 dimensions) representing a voice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub speaker_embedding: Option<Vec<f32>>,

    /// Controls the initial buffering size for the audio codec.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub initial_codec_chunk_frames: Option<u32>,
}

impl Default for NvExt {
    fn default() -> Self {
        NvExt::builder().build().unwrap()
    }
}

impl NvExt {
    /// Creates a new [`NvExtBuilder`].
    pub fn builder() -> NvExtBuilder {
        NvExtBuilder::default()
    }
}

fn validate_nv_ext(_nv_ext: &NvExt) -> Result<(), ValidationError> {
    Ok(())
}

impl NvExtBuilder {
    /// Appends an annotation to the builder's annotation list.
    pub fn add_annotation(&mut self, annotation: impl Into<String>) -> &mut Self {
        self.annotations
            .get_or_insert_with(|| Some(vec![]))
            .as_mut()
            .expect("annotations should always be Some(Vec)")
            .push(annotation.into());
        self
    }
}
