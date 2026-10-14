// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::engines::ValidateRequest;

mod aggregator;
mod nvext;

pub use nvext::NvExt;

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

    /// Delivery mode of the generated audio. Absent means [`AudioDataSource::B64Json`].
    /// Image and video generation use `response_format` for this choice. The
    /// OpenAI audio API uses `response_format` for the codec. Audio uses a
    /// separate field for the delivery mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_source: Option<AudioDataSource>,

    /// Output codec: "wav", "mp3", "pcm", "flac", "aac", "opus" (default: "wav")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    /// Speed factor. The frontend rejects a value outside 0.25 to 4.0.
    /// Absent means 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0.25, max = 4.0, message = "speed must be between 0.25 and 4.0"))]
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

    /// Worker-boundary contract, not a public field: the frontend moves
    /// `passthrough` under `extra_args["media_passthrough"]` before
    /// dispatch (see [`Self::nest_passthrough`]) so workers read one
    /// explicit nested entry. A client-sent `extra_args` lands in
    /// `passthrough` like any other unknown field.
    #[serde(default, skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub extra_args: Option<serde_json::Map<String, serde_json::Value>>,

    /// Unknown top-level fields are retained here and forwarded to the
    /// backend without strict validation. This matches the OpenAI client's
    /// extra_body option, which merges into the top level of the body.
    /// Stable knobs can be promoted to typed fields over time.
    #[serde(default, flatten)]
    pub passthrough: serde_json::Map<String, serde_json::Value>,
}

/// Delivery mode of the generated audio.
///
/// The frontend reads this field to select the delivery mode. The set has two
/// values. A request with an unknown value fails to parse.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AudioDataSource {
    /// The response carries a URL to the audio file.
    Url,
    /// The response carries the audio bytes as base64 text.
    B64Json,
}

impl NvCreateAudioSpeechRequest {
    /// Nest captured top-level unknowns under `extra_args["media_passthrough"]`
    /// for dispatch to a worker.
    pub fn nest_passthrough(&mut self) {
        super::nest_media_passthrough(&mut self.passthrough, &mut self.extra_args);
    }
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
#[derive(Serialize, Deserialize, Debug, Clone)]
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

impl ValidateRequest for NvCreateAudioSpeechRequest {
    fn validate(&self) -> Result<(), anyhow::Error> {
        // `Validate` and `ValidateRequest` share the method name, so the
        // call names the trait.
        Validate::validate(self).map_err(anyhow::Error::from)
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
        assert_eq!(req.data_source, Some(AudioDataSource::Url));

        let out = serde_json::to_string(&req).unwrap();
        assert!(out.contains("\"data_source\":\"url\""));
    }

    #[test]
    fn audio_request_data_source_b64_json_round_trips() {
        let json = r#"{"input":"hi","data_source":"b64_json"}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.data_source, Some(AudioDataSource::B64Json));
    }

    #[test]
    fn audio_request_data_source_and_response_format_coexist() {
        let json = r#"{"input":"hi","data_source":"url","response_format":"mp3"}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.data_source, Some(AudioDataSource::Url));
        assert_eq!(req.response_format.as_deref(), Some("mp3"));
    }

    #[test]
    fn audio_request_unknown_data_source_is_rejected() {
        let json = r#"{"input":"hi","data_source":"ftp"}"#;
        let err = serde_json::from_str::<NvCreateAudioSpeechRequest>(json).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("url") && message.contains("b64_json"),
            "expected the parse error to list the valid values; got: {message}"
        );
    }

    #[test]
    fn audio_request_speed_in_range_passes_validation() {
        // The bounds are inclusive, and an absent speed means 1.0.
        for json in [
            r#"{"input":"hi"}"#,
            r#"{"input":"hi","speed":0.25}"#,
            r#"{"input":"hi","speed":1.0}"#,
            r#"{"input":"hi","speed":4.0}"#,
        ] {
            let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
            assert!(
                ValidateRequest::validate(&req).is_ok(),
                "expected {json} to pass validation"
            );
        }
    }

    #[test]
    fn audio_request_speed_out_of_range_fails_validation() {
        for json in [
            r#"{"input":"hi","speed":0.1}"#,
            r#"{"input":"hi","speed":5.0}"#,
        ] {
            let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
            let err = ValidateRequest::validate(&req).unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("speed"),
                "expected the error for {json} to name the field; got: {message}"
            );
        }
    }

    #[test]
    fn audio_request_chunking_capability_round_trips() {
        let json = r#"{"input":"hi","nvext":{"frontend_accepts_audio_chunks":true}}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.nvext
                .and_then(|nvext| nvext.frontend_accepts_audio_chunks),
            Some(true)
        );
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
            extra_args: None,
            passthrough: serde_json::Map::new(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("data_source"));
    }

    #[test]
    fn audio_request_cfg_scale_reaches_the_worker() {
        // Unknown nvext keys are dropped silently, so a missing cfg_scale field
        // would disable guidance without any error surfacing to the client.
        let json = r#"{"input":"hi","nvext":{"cfg_scale":1.5}}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.nvext.as_ref().and_then(|n| n.cfg_scale), Some(1.5));

        let out = serde_json::to_string(&req).unwrap();
        assert!(out.contains("\"cfg_scale\":1.5"));
    }

    #[test]
    fn audio_request_cfg_scale_omitted_when_absent() {
        let json = r#"{"input":"hi","nvext":{}}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.nvext.as_ref().and_then(|n| n.cfg_scale), None);
        assert!(!serde_json::to_string(&req).unwrap().contains("cfg_scale"));
    }

    #[test]
    fn audio_request_top_level_cfg_scale_is_not_a_typed_field() {
        // A top-level cfg_scale is a client mistake: it lands in passthrough
        // (and so in extra_args for the worker), never in the Audex contract.
        let json = r#"{"input":"hi","cfg_scale":1.5}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert!(req.nvext.is_none());
        assert_eq!(req.passthrough["cfg_scale"], serde_json::json!(1.5));
    }

    #[test]
    fn audio_request_captures_unknown_top_level_fields() {
        // The OpenAI client's extra_body option merges into the top level of
        // the body, so that is where backend knobs arrive.
        let json = r#"{"input":"hello","emotion":"calm","pitch":1.2}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.passthrough["emotion"], serde_json::json!("calm"));
        assert_eq!(req.passthrough["pitch"], serde_json::json!(1.2));
        assert!(!req.passthrough.contains_key("input"));

        let out = serde_json::to_string(&req).unwrap();
        let back: NvCreateAudioSpeechRequest = serde_json::from_str(&out).unwrap();
        assert_eq!(back.passthrough, req.passthrough);
        assert!(out.contains("\"emotion\":\"calm\""));
    }

    #[test]
    fn audio_request_empty_passthrough_adds_nothing() {
        let json = r#"{"input":"hello"}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert!(req.passthrough.is_empty());
        let out: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap();
        assert_eq!(out, serde_json::json!({"input":"hello"}));
    }

    #[test]
    fn audio_request_nests_passthrough_for_workers() {
        let json = r#"{"input":"hello","emotion":"calm"}"#;
        let mut req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        req.nest_passthrough();
        assert!(req.passthrough.is_empty());
        let out = serde_json::to_value(&req).unwrap();
        assert_eq!(
            out["extra_args"]["media_passthrough"]["emotion"],
            serde_json::json!("calm")
        );
        assert!(out.get("emotion").is_none());
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
