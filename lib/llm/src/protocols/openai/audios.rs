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

    /// How the generated data should be returned: "url" or "b64_json" (default: "b64_json")
    /// Note that in image and video generation, the 'response_format' is the equivalent of
    /// this field. However, in audio generation, OpenAI specifies the 'response_format'
    /// to be used for output format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_source: Option<String>,

    /// Output codec: "wav", "mp3", "pcm", "flac", "aac", "opus" (default: "wav")
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
    /// Actual codec used for this audio: "wav", "mp3", "pcm", "flac", "aac", "opus"
    pub output_format: String,

    /// URL of the generated audio (if data_source is "url")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Base64-encoded audio data (if data_source is "b64_json")
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

#[cfg(test)]
mod tests {
    use super::*;

    // --- NvCreateAudioSpeechRequest ---

    #[test]
    fn audio_request_data_source_optional_absent_is_none() {
        let json = r#"{"input":"hello"}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.data_source, None);
    }

    #[test]
    fn audio_request_data_source_url_round_trips() {
        let json = r#"{"input":"hello","data_source":"url"}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.data_source.as_deref(), Some("url"));

        let out = serde_json::to_string(&req).unwrap();
        assert!(out.contains("\"data_source\":\"url\""));
    }

    #[test]
    fn audio_request_data_source_b64_json_round_trips() {
        let json = r#"{"input":"hi","data_source":"b64_json"}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.data_source.as_deref(), Some("b64_json"));
    }

    #[test]
    fn audio_request_data_source_and_response_format_coexist() {
        let json = r#"{"input":"hi","data_source":"url","response_format":"mp3"}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.data_source.as_deref(), Some("url"));
        assert_eq!(req.response_format.as_deref(), Some("mp3"));
    }

    #[test]
    fn audio_request_data_source_none_omitted_from_serialization() {
        let req = NvCreateAudioSpeechRequest {
            input: "hi".into(),
            model: None,
            voice: None,
            data_source: None,
            response_format: None,
            speed: None,
            task_type: None,
            language: None,
            instructions: None,
            ref_audio: None,
            ref_text: None,
            max_new_tokens: None,
            user: None,
            nvext: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("data_source"));
    }

    // --- AudioData ---

    #[test]
    fn audio_data_output_format_required_present() {
        let json = r#"{"output_format":"mp3","b64_json":"abc=="}"#;
        let d: AudioData = serde_json::from_str(json).unwrap();
        assert_eq!(d.output_format, "mp3");
        assert_eq!(d.b64_json.as_deref(), Some("abc=="));
    }

    #[test]
    fn audio_data_output_format_required_missing_fails() {
        let json = r#"{"b64_json":"abc=="}"#;
        assert!(serde_json::from_str::<AudioData>(json).is_err());
    }

    #[test]
    fn audio_data_url_omitted_when_none() {
        let d = AudioData {
            output_format: "wav".into(),
            url: None,
            b64_json: Some("xyz==".into()),
        };
        let json = serde_json::to_string(&d).unwrap();
        assert!(!json.contains("\"url\""));
        assert!(json.contains("b64_json"));
    }

    #[test]
    fn audio_data_round_trip_url_path() {
        let d = AudioData {
            output_format: "opus".into(),
            url: Some("http://x/a.ogg".into()),
            b64_json: None,
        };
        let json = serde_json::to_string(&d).unwrap();
        let d2: AudioData = serde_json::from_str(&json).unwrap();
        assert_eq!(d2.output_format, "opus");
        assert_eq!(d2.url.as_deref(), Some("http://x/a.ogg"));
        assert!(d2.b64_json.is_none());
    }

    #[test]
    fn audio_data_all_codec_values_deserialize() {
        for fmt in ["wav", "mp3", "pcm", "flac", "aac", "opus"] {
            let json = format!(r#"{{"output_format":"{}"}}"#, fmt);
            let d: AudioData = serde_json::from_str(&json).unwrap();
            assert_eq!(d.output_format, fmt);
        }
    }
}
