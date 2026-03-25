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
pub struct NvCreateAudioSpeechRequest {
    /// The text to generate audio for
    pub input: String,

    /// The model to use for audio generation
    pub model: String,

    /// The voice to use for generation
    pub voice: String,

    /// The audio format to return: mp3, wav, opus, aac, flac, pcm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    /// Playback speed (0.25 to 4.0, default 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,

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
    pub fn empty() -> Self {
        Self {
            audio_b64: String::new(),
            content_type: "audio/mpeg".to_string(),
            model: String::new(),
            created: 0,
        }
    }
}

/// Implements `NvExtProvider` for `NvCreateAudioSpeechRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateAudioSpeechRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `AnnotationsProvider` for `NvCreateAudioSpeechRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateAudioSpeechRequest {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization_roundtrip() {
        let req = NvCreateAudioSpeechRequest {
            input: "Hello world".to_string(),
            model: "tts-model".to_string(),
            voice: "vivian".to_string(),
            response_format: Some("wav".to_string()),
            speed: Some(1.5),
            nvext: Some(NvExt {
                seed: Some(42),
                language: Some("en".to_string()),
                ..Default::default()
            }),
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: NvCreateAudioSpeechRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.input, "Hello world");
        assert_eq!(deserialized.model, "tts-model");
        assert_eq!(deserialized.voice, "vivian");
        assert_eq!(deserialized.response_format.as_deref(), Some("wav"));
        assert_eq!(deserialized.speed, Some(1.5));
        assert_eq!(deserialized.nvext.as_ref().unwrap().seed, Some(42));
        assert_eq!(
            deserialized.nvext.as_ref().unwrap().language.as_deref(),
            Some("en")
        );
    }

    #[test]
    fn test_request_minimal_deserialization() {
        let json = r#"{"input":"Hi","model":"tts","voice":"alloy"}"#;
        let req: NvCreateAudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.input, "Hi");
        assert_eq!(req.model, "tts");
        assert_eq!(req.voice, "alloy");
        assert!(req.response_format.is_none());
        assert!(req.speed.is_none());
        assert!(req.nvext.is_none());
    }

    #[test]
    fn test_response_serialization_roundtrip() {
        let resp = NvAudiosResponse {
            audio_b64: "dGVzdA==".to_string(),
            content_type: "audio/wav".to_string(),
            model: "tts-model".to_string(),
            created: 1234567890,
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: NvAudiosResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.audio_b64, "dGVzdA==");
        assert_eq!(deserialized.content_type, "audio/wav");
        assert_eq!(deserialized.model, "tts-model");
        assert_eq!(deserialized.created, 1234567890);
    }

    #[test]
    fn test_empty_response() {
        let resp = NvAudiosResponse::empty();
        assert!(resp.audio_b64.is_empty());
        assert_eq!(resp.content_type, "audio/mpeg");
        assert!(resp.model.is_empty());
        assert_eq!(resp.created, 0);
    }

    #[test]
    fn test_annotations_provider() {
        let req = NvCreateAudioSpeechRequest {
            input: "test".to_string(),
            model: "m".to_string(),
            voice: "v".to_string(),
            response_format: None,
            speed: None,
            nvext: Some(NvExt {
                annotations: Some(vec!["request_id".to_string()]),
                ..Default::default()
            }),
        };

        assert!(req.has_annotation("request_id"));
        assert!(!req.has_annotation("other"));
        assert_eq!(req.annotations().unwrap().len(), 1);
    }

    #[test]
    fn test_annotations_provider_no_nvext() {
        let req = NvCreateAudioSpeechRequest {
            input: "test".to_string(),
            model: "m".to_string(),
            voice: "v".to_string(),
            response_format: None,
            speed: None,
            nvext: None,
        };

        assert!(!req.has_annotation("request_id"));
        assert!(req.annotations().is_none());
    }

    #[test]
    fn test_nvext_builder() {
        let nvext = NvExt::builder()
            .seed(42_i64)
            .language("zh".to_string())
            .add_annotation("test_ann")
            .build()
            .unwrap();

        assert_eq!(nvext.seed, Some(42));
        assert_eq!(nvext.language.as_deref(), Some("zh"));
        assert_eq!(nvext.annotations.as_ref().unwrap(), &["test_ann"]);
    }

    #[test]
    fn test_nvext_default() {
        let nvext = NvExt::default();
        assert!(nvext.seed.is_none());
        assert!(nvext.language.is_none());
        assert!(nvext.annotations.is_none());
        assert!(nvext.task_type.is_none());
        assert!(nvext.instructions.is_none());
        assert!(nvext.ref_audio.is_none());
        assert!(nvext.ref_text.is_none());
        assert!(nvext.max_new_tokens.is_none());
    }
}
