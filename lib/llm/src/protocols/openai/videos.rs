// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

mod aggregator;
mod nvext;

pub use nvext::{NvExt, NvExtProvider};

/// Request for video generation (/v1/videos endpoint)
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateVideoRequest {
    /// The text prompt for video generation
    pub prompt: String,

    /// The model to use for video generation
    pub model: String,

    /// Optional image reference that guides generation (for I2V)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_reference: Option<String>,

    /// Clip duration in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds: Option<i32>,

    /// Video size in WxH format (default: "832x480")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    /// Optional user identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// How the generated data should be returned: "url" or "b64_json" (default: "url")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    /// Output container format: "mp4", "webm", "gif", etc.
    /// This field is used as model hint and the model may not
    /// return the requested format, should check with output_format
    /// field in the response data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,

    /// Whether to stream the video generation (default: false)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// NVIDIA extensions
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

impl NvCreateVideoRequest {
    /// Nest captured top-level unknowns under `extra_args["media_passthrough"]`
    /// for dispatch to a worker.
    pub fn nest_passthrough(&mut self) {
        super::nest_media_passthrough(&mut self.passthrough, &mut self.extra_args);
    }
}

/// Video data in response
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VideoData {
    /// Actual container format of this video: "mp4", "webm", "gif"
    pub output_format: String,

    /// URL of the generated video (if response_format is "url")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Base64-encoded video (if response_format is "b64_json")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
}

/// Response structure for video generation
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvVideosResponse {
    /// Unique identifier for the response
    pub id: String,

    /// Object type (always "video")
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

    /// Generated video data
    #[serde(default)]
    pub data: Vec<VideoData>,

    /// Error message if generation failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// Inference time in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_time_s: Option<f64>,
}

fn default_object_type() -> String {
    "video".to_string()
}

fn default_status() -> String {
    "completed".to_string()
}

fn default_progress() -> i32 {
    100
}

impl NvVideosResponse {
    pub fn empty() -> Self {
        Self {
            id: String::new(),
            object: "video".to_string(),
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

/// Implements `NvExtProvider` for `NvCreateVideoRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateVideoRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `AnnotationsProvider` for `NvCreateVideoRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateVideoRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    ///
    /// # Arguments
    /// * `annotation` - A string slice representing the annotation to check.
    ///
    /// # Returns
    /// `true` if the annotation exists, `false` otherwise.
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

    // --- NvCreateVideoRequest ---

    #[test]
    fn video_request_stream_field_round_trips() {
        let json = r#"{"prompt":"cat","model":"wan","stream":true}"#;
        let req: NvCreateVideoRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.stream, Some(true));

        let out = serde_json::to_string(&req).unwrap();
        assert!(out.contains("\"stream\":true"));
    }

    #[test]
    fn video_request_stream_false_round_trips() {
        let json = r#"{"prompt":"cat","model":"wan","stream":false}"#;
        let req: NvCreateVideoRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.stream, Some(false));
    }

    #[test]
    fn video_request_stream_absent_deserializes_as_none() {
        let json = r#"{"prompt":"cat","model":"wan"}"#;
        let req: NvCreateVideoRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.stream, None);
    }

    #[test]
    fn video_request_stream_none_omitted_from_serialization() {
        let req = NvCreateVideoRequest {
            prompt: "cat".into(),
            model: "wan".into(),
            input_reference: None,
            seconds: None,
            size: None,
            user: None,
            response_format: None,
            output_format: None,
            stream: None,
            nvext: None,
            extra_args: None,
            passthrough: serde_json::Map::new(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("stream"));
    }

    #[test]
    fn video_request_captures_unknown_top_level_fields() {
        // The OpenAI client's extra_body option merges into the top level of
        // the body, so that is where backend knobs arrive.
        let json = r#"{"prompt":"cat","model":"cosmos","generate_sound":true,"strength":0.7}"#;
        let req: NvCreateVideoRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.passthrough["generate_sound"], serde_json::json!(true));
        assert_eq!(req.passthrough["strength"], serde_json::json!(0.7));

        let out = serde_json::to_string(&req).unwrap();
        let back: NvCreateVideoRequest = serde_json::from_str(&out).unwrap();
        assert_eq!(back.passthrough, req.passthrough);
        assert!(out.contains("\"generate_sound\":true"));
    }

    #[test]
    fn video_request_typed_fields_stay_out_of_passthrough() {
        let json = r#"{"prompt":"cat","model":"wan","stream":true,"custom_knob":1}"#;
        let req: NvCreateVideoRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.stream, Some(true));
        assert!(!req.passthrough.contains_key("prompt"));
        assert!(!req.passthrough.contains_key("stream"));
        assert_eq!(req.passthrough["custom_knob"], serde_json::json!(1));
    }

    #[test]
    fn video_request_empty_passthrough_adds_nothing() {
        let json = r#"{"prompt":"cat","model":"wan"}"#;
        let req: NvCreateVideoRequest = serde_json::from_str(json).unwrap();
        assert!(req.passthrough.is_empty());
        let out: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap();
        assert_eq!(out, serde_json::json!({"prompt":"cat","model":"wan"}));
    }

    #[test]
    fn video_request_nests_passthrough_for_workers() {
        let json = r#"{"prompt":"cat","model":"cosmos","generate_sound":true}"#;
        let mut req: NvCreateVideoRequest = serde_json::from_str(json).unwrap();
        req.nest_passthrough();
        assert!(req.passthrough.is_empty());
        let out = serde_json::to_value(&req).unwrap();
        assert_eq!(
            out["extra_args"]["media_passthrough"]["generate_sound"],
            serde_json::json!(true)
        );
        assert!(out.get("generate_sound").is_none());
    }

    #[test]
    fn video_request_nest_without_unknowns_adds_nothing() {
        let json = r#"{"prompt":"cat","model":"wan"}"#;
        let mut req: NvCreateVideoRequest = serde_json::from_str(json).unwrap();
        req.nest_passthrough();
        let out = serde_json::to_value(&req).unwrap();
        assert!(out.get("extra_args").is_none());
    }

    #[test]
    fn video_request_client_extra_args_is_not_the_worker_field() {
        let json = r#"{"prompt":"cat","model":"wan","extra_args":{"x":1}}"#;
        let req: NvCreateVideoRequest = serde_json::from_str(json).unwrap();
        assert!(req.extra_args.is_none());
        assert_eq!(req.passthrough["extra_args"]["x"], serde_json::json!(1));
    }

    #[test]
    fn video_request_output_format_optional_absent_is_none() {
        let json = r#"{"prompt":"cat","model":"wan"}"#;
        let req: NvCreateVideoRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.output_format, None);
    }

    #[test]
    fn video_request_output_format_mp4_round_trips() {
        let json = r#"{"prompt":"cat","model":"wan","output_format":"mp4"}"#;
        let req: NvCreateVideoRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.output_format.as_deref(), Some("mp4"));
    }

    // --- VideoData ---

    #[test]
    fn video_data_output_format_required_present() {
        let json = r#"{"output_format":"mp4","url":"http://example.com/v.mp4"}"#;
        let d: VideoData = serde_json::from_str(json).unwrap();
        assert_eq!(d.output_format, "mp4");
        assert_eq!(d.url.as_deref(), Some("http://example.com/v.mp4"));
    }

    #[test]
    fn video_data_output_format_required_missing_fails() {
        let json = r#"{"url":"http://example.com/v.mp4"}"#;
        assert!(serde_json::from_str::<VideoData>(json).is_err());
    }

    #[test]
    fn video_data_url_omitted_when_none() {
        let d = VideoData {
            output_format: "mp4".into(),
            url: None,
            b64_json: Some("abc==".into()),
        };
        let json = serde_json::to_string(&d).unwrap();
        assert!(!json.contains("url"));
        assert!(json.contains("b64_json"));
    }

    #[test]
    fn video_data_round_trip_with_both_fields() {
        let d = VideoData {
            output_format: "webm".into(),
            url: Some("http://x/v.webm".into()),
            b64_json: None,
        };
        let json = serde_json::to_string(&d).unwrap();
        let d2: VideoData = serde_json::from_str(&json).unwrap();
        assert_eq!(d2.output_format, "webm");
        assert_eq!(d2.url.as_deref(), Some("http://x/v.webm"));
        assert!(d2.b64_json.is_none());
    }
}
