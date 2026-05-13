// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

mod aggregator;
mod nvext;

pub use aggregator::DeltaAggregator;
pub use nvext::{NvExt, NvExtProvider};

/// Request for audio speech generation (/v1/audio/speech endpoint).
///
/// Follows vLLM-Omni's OpenAICreateSpeechRequest format with TTS-specific
/// parameters as top-level fields.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateAudioSpeechRequest {
    /// The text to synthesize into speech (required)
    pub input: String,

    /// The TTS model to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Voice/speaker name (e.g., "vivian", "ryan", "aiden")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,

    /// Output format: "wav", "mp3", "pcm", "flac", "aac", "opus"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    /// Speed factor (0.25-4.0, default: 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,

    // Qwen3-TTS specific parameters (top-level, matching vLLM-Omni)
    /// TTS task type: "CustomVoice", "VoiceDesign", or "Base"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,

    /// Language: "Auto", "Chinese", "English", "Japanese", etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Voice style/emotion instructions (for VoiceDesign)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Reference audio URL or base64 (for voice cloning with Base task)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_audio: Option<String>,

    /// Reference transcript (for voice cloning with Base task)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_text: Option<String>,

    /// Maximum tokens to generate (default: 2048)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<i32>,

    /// Optional user identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// NVIDIA extensions (reserved for future use)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// Audio data in response
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AudioData {
    /// URL of the generated audio (if response_format is "url")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Base64-encoded audio data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
}

/// Response structure for audio speech generation
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvAudioSpeechResponse {
    /// Unique identifier for the response
    pub id: String,

    /// Object type (always "audio.speech")
    #[serde(default = "default_object_type")]
    pub object: String,

    /// Model used for generation
    pub model: String,

    /// Status of the generation ("completed", "failed", etc.)
    #[serde(default = "default_status")]
    pub status: String,

    /// Progress percentage (0-100)
    #[serde(default = "default_progress")]
    pub progress: i32,

    /// Unix timestamp of creation
    pub created: i64,

    /// Generated audio data
    #[serde(default)]
    pub data: Vec<AudioData>,

    /// Error message if generation failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// Inference time in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_time_s: Option<f64>,
}

fn default_object_type() -> String {
    "audio.speech".to_string()
}

fn default_status() -> String {
    "completed".to_string()
}

fn default_progress() -> i32 {
    100
}

impl NvAudioSpeechResponse {
    pub fn empty() -> Self {
        Self {
            id: String::new(),
            object: "audio.speech".to_string(),
            model: String::new(),
            status: "completed".to_string(),
            progress: 100,
            created: 0,
            data: vec![],
            error: None,
            inference_time_s: None,
        }
    }
}

/// Implements `NvExtProvider` for `NvCreateAudioSpeechRequest`.
impl NvExtProvider for NvCreateAudioSpeechRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `AnnotationsProvider` for `NvCreateAudioSpeechRequest`.
impl AnnotationsProvider for NvCreateAudioSpeechRequest {
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}
