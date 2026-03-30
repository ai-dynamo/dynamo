
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

mod aggregator;
mod nvext;

pub use aggregator::DeltaAggregator;
pub use nvext::{NvExt, NvExtProvider};

/// Request for audio speech generation (/v1/audio/speech endpoint)
///
/// Follows the OpenAI TTS API convention.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateAudioRequest {
    /// The text to generate audio for
    pub input: String,

    /// The model to use for audio generation
    pub model: String,

    /// The voice to use for generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,

    /// Control the voice of your generated audio with additional instructions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// The audio format to return: mp3, wav, opus, aac, flac, pcm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    /// Playback speed (0.25 to 4.0, default 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,

    /// The format to stream the audio in
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_format: Option<String>,

    /// NVIDIA extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// Response structure for audio speech generation.
///
/// Internal transport uses base64-encoded audio. The HTTP handler decodes
/// this to return raw binary audio to clients (matching OpenAI convention).
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvAudiosResponse {
    /// Base64-encoded audio bytes
    pub audio_b64: String,

    /// MIME content type (e.g. "audio/mpeg", "audio/wav")
    pub content_type: String,

    /// Model used for generation
    pub model: String,

    /// Unix timestamp of creation
    pub created: i64,
}

impl NvAudiosResponse {
    /// Returns an empty response with default values.
    pub fn empty() -> Self {
        Self {
            audio_b64: String::new(),
            content_type: "audio/mpeg".to_string(),
            model: String::new(),
            created: 0,
        }
    }
}

/// Implements `NvExtProvider` for `NvCreateAudioRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateAudioRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `AnnotationsProvider` for `NvCreateAudioRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateAudioRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}
