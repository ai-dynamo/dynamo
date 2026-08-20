// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

mod aggregator;

/// Transport-neutral form of an OpenAI audio transcription request.
///
/// The HTTP frontend converts the multipart upload to base64 so the request can
/// cross any Dynamo request plane without relying on HTTP-specific types.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateAudioTranscriptionRequest {
    /// Base64-encoded contents of the uploaded audio file.
    pub audio_b64: String,

    /// Original upload filename, used by decoders for format detection.
    pub filename: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_granularities: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NvAudioTranscriptionUsage {
    #[serde(rename = "type")]
    pub usage_type: String,
    pub seconds: f64,
}

/// JSON response for the OpenAI audio transcription endpoint.
#[derive(Serialize, Deserialize, Validate, Debug, Clone, PartialEq)]
pub struct NvAudioTranscriptionResponse {
    pub text: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<NvAudioTranscriptionUsage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<serde_json::Value>,
}

impl NvAudioTranscriptionResponse {
    pub fn empty() -> Self {
        Self {
            text: String::new(),
            usage: None,
            language: None,
            duration: None,
            segments: None,
            words: None,
        }
    }
}

impl AnnotationsProvider for NvCreateAudioTranscriptionRequest {
    fn annotations(&self) -> Option<Vec<String>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips_openai_fields() {
        let request: NvCreateAudioTranscriptionRequest =
            serde_json::from_value(serde_json::json!({
                "audio_b64": "UklGRg==",
                "filename": "sample.wav",
                "model": "openai/whisper-tiny",
                "language": "en",
                "response_format": "verbose_json",
                "timestamp_granularities": ["word"]
            }))
            .unwrap();

        assert_eq!(request.filename, "sample.wav");
        assert_eq!(request.language.as_deref(), Some("en"));
        assert_eq!(
            request.timestamp_granularities,
            Some(vec!["word".to_string()])
        );
    }

    #[test]
    fn response_uses_openai_usage_field_name() {
        let response = NvAudioTranscriptionResponse {
            text: "hello".to_string(),
            usage: Some(NvAudioTranscriptionUsage {
                usage_type: "duration".to_string(),
                seconds: 1.25,
            }),
            ..NvAudioTranscriptionResponse::empty()
        };

        let value = serde_json::to_value(response).unwrap();
        assert_eq!(value["usage"]["type"], "duration");
        assert_eq!(value["usage"]["seconds"], 1.25);
    }
}
