// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::MediaDelivery;

mod aggregator;
mod nvext;

pub use nvext::NvExt;

/// Request for image generation (/v1/images/generations and /v1/images/edits).
///
/// The OpenAI fields keep the wire format of the OpenAI `CreateImageRequest`.
/// `model` and `size` are free text, because the OpenAI type accepts any
/// string there.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
pub struct NvCreateImageRequest {
    pub prompt: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Number of images to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u8>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<ImageQuality>,

    /// Delivery mode of the generated images
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<MediaDelivery>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<ImageOutputFormat>,

    /// Compression level (0-100%) of jpeg and webp output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<u8>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Number of partial images to stream before the final image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_images: Option<u8>,

    /// Image size in WxH format, or "auto"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub moderation: Option<ImageModeration>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<ImageBackground>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<ImageStyle>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Optional image reference that guides generation (for I2I/TI2I).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_reference: Option<String>,

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
    #[schema(ignore)]
    pub passthrough: serde_json::Map<String, serde_json::Value>,
}

/// Quality of the generated images
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageQuality {
    Standard,
    Hd,
    High,
    Medium,
    Low,
    Auto,
}

/// File format of the generated images
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageOutputFormat {
    Png,
    Jpeg,
    Webp,
}

/// Content-moderation level of the generation
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageModeration {
    Auto,
    Low,
}

/// Requested background of the generated images
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageBackground {
    Auto,
    Transparent,
    Opaque,
}

/// Style of the generated images
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageStyle {
    Vivid,
    Natural,
}

impl NvCreateImageRequest {
    /// Nest captured top-level unknowns under `extra_args["media_passthrough"]`
    /// for dispatch to a worker.
    pub fn nest_passthrough(&mut self) {
        super::nest_media_passthrough(&mut self.passthrough, &mut self.extra_args);
    }
}

/// Response for image generation.
///
/// Keeps the wire format of the OpenAI `ImagesResponse`, which writes every
/// absent optional field as null.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
pub struct NvImagesResponse {
    /// Unix timestamp of creation
    pub created: u32,

    pub data: Vec<ImageData>,

    pub background: Option<ImageResponseBackground>,

    pub output_format: Option<ImageOutputFormat>,

    /// Image size in WxH format
    pub size: Option<String>,

    pub quality: Option<ImageQuality>,

    /// Token usage of the generation, when the model reports it
    pub usage: Option<serde_json::Map<String, serde_json::Value>>,
}

/// One generated image. The worker sets one of `url` and `b64_json`.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
pub struct ImageData {
    /// URL of the generated image (if response_format is "url")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Base64-encoded image (if response_format is "b64_json")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,

    /// The prompt the model used, when it rewrote the original prompt
    pub revised_prompt: Option<String>,
}

/// Actual background of the generated images
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageResponseBackground {
    Transparent,
    Opaque,
}

impl NvImagesResponse {
    pub fn empty() -> Self {
        Self {
            created: 0,
            data: vec![],
            background: None,
            output_format: None,
            size: None,
            quality: None,
            usage: None,
        }
    }
}

/// Implements `AnnotationsProvider` for `NvCreateImageRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateImageRequest {
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
    use dynamo_protocols::types::{CreateImageRequest, ImagesResponse};
    use serde::de::DeserializeOwned;

    use super::*;

    /// Parses `json` as the OpenAI type `O` and as the Dynamo type `D`.
    /// Both must reject it, or both must serialize it to the same JSON.
    fn assert_same_wire_format<O, D>(json: &str)
    where
        O: DeserializeOwned + Serialize,
        D: DeserializeOwned + Serialize,
    {
        let openai = serde_json::from_str::<O>(json).map(|v| serde_json::to_value(v).unwrap());
        let dynamo = serde_json::from_str::<D>(json).map(|v| serde_json::to_value(v).unwrap());
        match (openai, dynamo) {
            (Ok(openai), Ok(dynamo)) => assert_eq!(dynamo, openai, "wire format of {json}"),
            (Err(_), Err(_)) => {}
            (openai, dynamo) => panic!("{json}: OpenAI type {openai:?}, Dynamo type {dynamo:?}"),
        }
    }

    // --- Wire compatibility with the OpenAI types ---

    #[test]
    fn image_request_wire_format_matches_the_openai_type() {
        for json in [
            r#"{"prompt":"a cat","model":"gpt-image-1","n":2,"quality":"high","response_format":"b64_json","output_format":"webp","output_compression":80,"stream":false,"partial_images":2,"size":"1536x1024","moderation":"low","background":"transparent","style":"natural","user":"user-1"}"#,
            // The OpenAI type accepts any model and size string.
            r#"{"prompt":"a cat","model":"black-forest-labs/FLUX.1-dev","size":"768x512"}"#,
            r#"{"prompt":"a cat","model":null,"n":null}"#,
            // Rejected: an unknown quality, an n above u8, no prompt.
            r#"{"prompt":"a cat","quality":"ultra"}"#,
            r#"{"prompt":"a cat","n":300}"#,
            r#"{"model":"gpt-image-1"}"#,
        ] {
            assert_same_wire_format::<CreateImageRequest, NvCreateImageRequest>(json);
        }
    }

    #[test]
    fn image_request_enum_values_match_the_openai_type() {
        let fields: [(&str, &[&str]); 6] = [
            (
                "quality",
                &["standard", "hd", "high", "medium", "low", "auto"],
            ),
            ("response_format", &["url", "b64_json"]),
            ("output_format", &["png", "jpeg", "webp"]),
            ("moderation", &["auto", "low"]),
            ("background", &["auto", "transparent", "opaque"]),
            ("style", &["vivid", "natural"]),
        ];
        for (field, values) in fields {
            for value in values {
                let json = format!(r#"{{"prompt":"a cat","{field}":"{value}"}}"#);
                assert_same_wire_format::<CreateImageRequest, NvCreateImageRequest>(&json);
            }
        }
    }

    #[test]
    fn image_response_wire_format_matches_the_openai_type() {
        for json in [
            r#"{"created":1700000000,"data":[{"url":"http://x/1.png","revised_prompt":"a cat, photo"},{"b64_json":"aGVsbG8="}],"background":"opaque","output_format":"png","size":"1024x1024","quality":"high","usage":{"input_tokens":10,"total_tokens":30,"output_tokens":20,"output_token_details":{"text_tokens":0,"image_tokens":20},"input_tokens_details":{"text_tokens":10,"image_tokens":0}}}"#,
            // The OpenAI type writes every absent optional field as null.
            r#"{"created":1,"data":[]}"#,
            r#"{"created":1,"data":[{"b64_json":"aGVsbG8=","url":null}],"size":"768x512"}"#,
            // Rejected: an unknown background, a negative created, no data.
            r#"{"created":1,"data":[],"background":"auto"}"#,
            r#"{"created":-1,"data":[]}"#,
            r#"{"created":1}"#,
        ] {
            assert_same_wire_format::<ImagesResponse, NvImagesResponse>(json);
        }
    }

    #[test]
    fn image_response_enum_values_match_the_openai_type() {
        let fields: [(&str, &[&str]); 3] = [
            ("background", &["transparent", "opaque"]),
            ("output_format", &["png", "jpeg", "webp"]),
            (
                "quality",
                &["standard", "hd", "high", "medium", "low", "auto"],
            ),
        ];
        for (field, values) in fields {
            for value in values {
                let json = format!(r#"{{"created":1,"data":[],"{field}":"{value}"}}"#);
                assert_same_wire_format::<ImagesResponse, NvImagesResponse>(&json);
            }
        }
    }

    // --- NvCreateImageRequest ---

    #[test]
    fn image_request_captures_unknown_top_level_fields() {
        // The OpenAI client's extra_body option merges into the top level of
        // the body, so that is where backend knobs arrive.
        let json = r#"{"prompt":"a cat","think_mode":true,"size_override":{"h":512,"w":768}}"#;
        let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "a cat");
        assert_eq!(req.passthrough["think_mode"], serde_json::json!(true));
        assert_eq!(
            req.passthrough["size_override"]["h"],
            serde_json::json!(512)
        );

        let out = serde_json::to_string(&req).unwrap();
        let back: NvCreateImageRequest = serde_json::from_str(&out).unwrap();
        assert_eq!(back.prompt, "a cat");
        assert_eq!(back.passthrough, req.passthrough);
    }

    #[test]
    fn image_request_typed_fields_stay_out_of_passthrough() {
        let json = r#"{"prompt":"a cat","n":2,"input_reference":"ref.png","nvext":{"seed":7},"custom_knob":1}"#;
        let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "a cat");
        assert_eq!(req.input_reference.as_deref(), Some("ref.png"));
        assert_eq!(req.nvext.as_ref().and_then(|n| n.seed), Some(7));
        assert_eq!(req.passthrough["custom_knob"], serde_json::json!(1));
        for consumed in ["prompt", "n", "input_reference", "nvext"] {
            assert!(!req.passthrough.contains_key(consumed), "{consumed}");
        }
    }

    #[test]
    fn image_request_round_trips_all_field_kinds_at_top_level() {
        let json =
            r#"{"prompt":"a cat","input_reference":"ref.png","nvext":{"seed":7},"knob":"x"}"#;
        let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
        let out: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap();
        assert_eq!(out["prompt"], serde_json::json!("a cat"));
        assert_eq!(out["input_reference"], serde_json::json!("ref.png"));
        assert_eq!(out["nvext"]["seed"], serde_json::json!(7));
        assert_eq!(out["knob"], serde_json::json!("x"));
        assert!(out.get("passthrough").is_none());
        assert!(out.get("inner").is_none());
    }

    #[test]
    fn image_request_empty_passthrough_stays_empty() {
        let json = r#"{"prompt":"a cat"}"#;
        let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
        assert!(req.passthrough.is_empty());
    }

    #[test]
    fn image_request_nests_passthrough_for_workers() {
        let json = r#"{"prompt":"a cat","think_mode":true}"#;
        let mut req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
        req.nest_passthrough();
        assert!(req.passthrough.is_empty());
        let out = serde_json::to_value(&req).unwrap();
        assert_eq!(
            out["extra_args"]["media_passthrough"]["think_mode"],
            serde_json::json!(true)
        );
        assert!(out.get("think_mode").is_none());
        assert_eq!(out["prompt"], serde_json::json!("a cat"));
    }

    #[test]
    fn image_request_client_extra_args_is_not_the_worker_field() {
        let json = r#"{"prompt":"a cat","extra_args":{"x":1}}"#;
        let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
        assert!(req.extra_args.is_none());
        assert_eq!(req.passthrough["extra_args"]["x"], serde_json::json!(1));
    }
}
