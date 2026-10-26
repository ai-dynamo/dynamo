// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Image generation protocol: `/v1/images/generations` and `/v1/images/edits`.
//!
//! These types re-declare the fields of the OpenAI `CreateImageRequest` and
//! `ImagesResponse` instead of wrapping `dynamo_protocols::types::CreateImageRequest`,
//! which re-exports the `async-openai` type. The Python worker models are
//! generated from a `utoipa` schema of these types (see `media_schemas.rs`),
//! and no schema exists upstream: `async-openai` has no schema derive, and
//! the `protocol-schema` placeholders in `dynamo-protocols` carry a type name
//! with no field shape. Owning the fields also replaces the hand-written
//! serde that the wrapper needed to flatten the OpenAI fields next to the
//! passthrough map, and lets an explicit `null` for a typed field read as
//! absent instead of landing in the passthrough.
//!
//! The right home for the schema derive is upstream, in `async-openai` or in
//! `dynamo-protocols`. When it exists there, these types can shrink back to
//! a wrapper. Until then, the tests in this module hold the wire format to
//! the OpenAI type: every field, every enum value, and every rejected input
//! is checked against `dynamo_protocols::types::CreateImageRequest`, and a
//! field or an enum value that a later `async-openai` adds fails to compile
//! there until the owned type has it.

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

use super::MediaDelivery;
use crate::engines::ValidateRequest;

mod aggregator;
mod nvext;

pub use nvext::{ImageAccounting, ImagesResponseNvExt, NvExt};

/// Request for image generation (/v1/images/generations and /v1/images/edits).
///
/// The OpenAI fields keep the wire format of the OpenAI `CreateImageRequest`;
/// the module documentation says why they are declared here and not wrapped.
/// `model` and `size` are free text, because the OpenAI type accepts any
/// string there.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateImageRequest {
    pub prompt: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Number of images to generate, 1 to 10 as the OpenAI API documents.
    /// The frontend rejects a value outside the range, so a worker never
    /// sees one and needs no range check of its own.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(minimum = 1, maximum = 10)]
    #[validate(range(min = 1, max = 10, message = "n must be between 1 and 10"))]
    pub n: Option<u8>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<ImageQuality>,

    /// Delivery mode of the generated images
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<MediaDelivery>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<ImageOutputFormat>,

    /// Compression level of jpeg and webp output, 0 to 100 percent as the
    /// OpenAI API documents. The frontend rejects a value outside the range,
    /// so a worker never sees one.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(minimum = 0, maximum = 100)]
    #[validate(range(
        min = 0,
        max = 100,
        message = "output_compression must be between 0 and 100"
    ))]
    pub output_compression: Option<u8>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Number of partial images to stream before the final image, 0 to 3 as
    /// the OpenAI API documents. The frontend rejects a value outside the
    /// range, so a worker never sees one.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(minimum = 0, maximum = 3)]
    #[validate(range(min = 0, max = 3, message = "partial_images must be between 0 and 3"))]
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
    #[validate(nested)]
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
    #[serde(rename = "xhigh")]
    XHigh,
    Max,
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
/// absent optional field as null. `nvext` is Dynamo's response extension;
/// the key is omitted when the worker reports nothing.
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
    pub usage: Option<ImageUsage>,

    /// Image accounting for models without token usage
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nvext: Option<ImagesResponseNvExt>,
}

/// Token usage of an image generation. Keeps the wire format of the OpenAI
/// `ImageGenUsage`, including the singular `output_token_details`.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ImageUsage {
    /// Tokens (image and text) in the input prompt
    pub input_tokens: u32,

    /// Tokens (image and text) used for the generation
    pub total_tokens: u32,

    /// Tokens the model generated
    pub output_tokens: u32,

    /// The output tokens by kind
    pub output_token_details: Option<ImageTokenDetails>,

    /// The input tokens by kind
    pub input_tokens_details: ImageTokenDetails,
}

/// A token count split by kind.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageTokenDetails {
    pub text_tokens: u32,

    pub image_tokens: u32,
}

/// One generated image. The worker sets one of `url` and `b64_json`.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
#[serde(try_from = "ImageDataWire")]
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

/// The wire shape of [`ImageData`]. The OpenAI `Image` type rejects an entry
/// with neither `url` nor `b64_json`, and so does this one.
#[derive(Deserialize)]
struct ImageDataWire {
    url: Option<String>,
    b64_json: Option<String>,
    revised_prompt: Option<String>,
}

impl TryFrom<ImageDataWire> for ImageData {
    type Error = &'static str;

    fn try_from(wire: ImageDataWire) -> Result<Self, Self::Error> {
        if wire.url.is_none() && wire.b64_json.is_none() {
            return Err("an image needs `url` or `b64_json`");
        }
        Ok(Self {
            url: wire.url,
            b64_json: wire.b64_json,
            revised_prompt: wire.revised_prompt,
        })
    }
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
            nvext: None,
        }
    }
}

/// Implements `AnnotationsProvider` for `NvCreateImageRequest`,
/// enabling retrieval and management of request annotations.
impl ValidateRequest for NvCreateImageRequest {
    fn validate(&self) -> Result<(), anyhow::Error> {
        // `Validate` and `ValidateRequest` share the method name, so the
        // derived one is called by trait path.
        Validate::validate(self).map_err(anyhow::Error::from)
    }
}

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
    use dynamo_protocols::types::{self as openai, CreateImageRequest, ImagesResponse};
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

    /// The wire strings of the listed variants of the OpenAI enum `$ty`.
    ///
    /// The match has no wildcard arm, so a variant that a new async-openai
    /// adds fails to compile here until it is listed, and the enum-value
    /// tests then check that the Dynamo type accepts it.
    macro_rules! openai_wire_values {
        ($ty:ty { $($variant:ident),+ $(,)? }) => {{
            type OpenAi = $ty;
            let _exhaustive = |value: OpenAi| match value {
                $(OpenAi::$variant)|+ => {}
            };
            [$(OpenAi::$variant),+]
                .into_iter()
                .map(|value| serde_json::to_value(value).unwrap().as_str().unwrap().to_owned())
                .collect::<Vec<_>>()
        }};
    }

    fn openai_image_qualities() -> Vec<String> {
        openai_wire_values!(openai::ImageQuality {
            Standard,
            HD,
            High,
            Medium,
            Low,
            XHigh,
            Max,
            Auto
        })
    }

    /// A request that sets every OpenAI field.
    const FULL_IMAGE_REQUEST: &str = r#"{"prompt":"a cat","model":"gpt-image-1","n":2,"quality":"high","response_format":"b64_json","output_format":"webp","output_compression":80,"stream":false,"partial_images":2,"size":"1536x1024","moderation":"low","background":"transparent","style":"natural","user":"user-1"}"#;

    /// A response that sets every OpenAI field, with one image of each kind.
    const FULL_IMAGE_RESPONSE: &str = r#"{"created":1700000000,"data":[{"url":"http://x/1.png","revised_prompt":"a cat, photo"},{"b64_json":"aGVsbG8="}],"background":"opaque","output_format":"png","size":"1024x1024","quality":"high","usage":{"input_tokens":10,"total_tokens":30,"output_tokens":20,"output_token_details":{"text_tokens":0,"image_tokens":20},"input_tokens_details":{"text_tokens":10,"image_tokens":0}}}"#;

    // --- Wire compatibility with the OpenAI types ---

    #[test]
    fn image_request_wire_format_matches_the_openai_type() {
        for json in [
            FULL_IMAGE_REQUEST,
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
        let fields = [
            ("quality", openai_image_qualities()),
            (
                "response_format",
                openai_wire_values!(openai::ImageResponseFormat { Url, B64Json }),
            ),
            (
                "output_format",
                openai_wire_values!(openai::ImageOutputFormat { Png, Jpeg, Webp }),
            ),
            (
                "moderation",
                openai_wire_values!(openai::ImageModeration { Auto, Low }),
            ),
            (
                "background",
                openai_wire_values!(openai::ImageBackground {
                    Auto,
                    Transparent,
                    Opaque
                }),
            ),
            (
                "style",
                openai_wire_values!(openai::ImageStyle { Vivid, Natural }),
            ),
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
            FULL_IMAGE_RESPONSE,
            // The OpenAI type writes every absent optional field as null.
            r#"{"created":1,"data":[]}"#,
            r#"{"created":1,"data":[{"b64_json":"aGVsbG8=","url":null}],"size":"768x512"}"#,
            // Rejected: an unknown background, a negative created, no data,
            // a usage without its input token details, and an image with
            // neither url nor b64_json.
            r#"{"created":1,"data":[],"background":"auto"}"#,
            r#"{"created":1,"data":[],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}"#,
            r#"{"created":-1,"data":[]}"#,
            r#"{"created":1}"#,
            r#"{"created":1,"data":[{}]}"#,
            r#"{"created":1,"data":[{"revised_prompt":"a cat, photo"}]}"#,
        ] {
            assert_same_wire_format::<ImagesResponse, NvImagesResponse>(json);
        }
    }

    #[test]
    fn image_response_enum_values_match_the_openai_type() {
        let fields = [
            (
                "background",
                openai_wire_values!(openai::ImageResponseBackground {
                    Transparent,
                    Opaque
                }),
            ),
            (
                "output_format",
                openai_wire_values!(openai::ImageOutputFormat { Png, Jpeg, Webp }),
            ),
            ("quality", openai_image_qualities()),
        ];
        for (field, values) in fields {
            for value in values {
                let json = format!(r#"{{"created":1,"data":[],"{field}":"{value}"}}"#);
                assert_same_wire_format::<ImagesResponse, NvImagesResponse>(&json);
            }
        }
    }

    // --- Field drift against the OpenAI types ---
    //
    // The patterns below name every field and have no `..`, and the match
    // on `Image` has no wildcard arm, so a field or a variant that a later
    // async-openai adds fails to compile here until the owned type has it.
    // The OpenAI structs are not `non_exhaustive`.

    #[test]
    fn image_request_fields_match_the_openai_type() {
        let CreateImageRequest {
            prompt,
            model,
            n,
            quality,
            response_format,
            output_format,
            output_compression,
            stream,
            partial_images,
            size,
            moderation,
            background,
            style,
            user,
        } = serde_json::from_str(FULL_IMAGE_REQUEST).unwrap();
        assert!(!prompt.is_empty());
        for (field, set) in [
            ("model", model.is_some()),
            ("n", n.is_some()),
            ("quality", quality.is_some()),
            ("response_format", response_format.is_some()),
            ("output_format", output_format.is_some()),
            ("output_compression", output_compression.is_some()),
            ("stream", stream.is_some()),
            ("partial_images", partial_images.is_some()),
            ("size", size.is_some()),
            ("moderation", moderation.is_some()),
            ("background", background.is_some()),
            ("style", style.is_some()),
            ("user", user.is_some()),
        ] {
            assert!(set, "FULL_IMAGE_REQUEST does not set {field}");
        }

        // The flatten re-emits an undeclared field, so the wire-format
        // comparison alone would not see one: it must not be in here.
        let NvCreateImageRequest {
            prompt: _,
            model: _,
            n: _,
            quality: _,
            response_format: _,
            output_format: _,
            output_compression: _,
            stream: _,
            partial_images: _,
            size: _,
            moderation: _,
            background: _,
            style: _,
            user: _,
            input_reference: _,
            nvext: _,
            extra_args: _,
            passthrough,
        } = serde_json::from_str(FULL_IMAGE_REQUEST).unwrap();
        assert!(
            passthrough.is_empty(),
            "an OpenAI field is not declared on NvCreateImageRequest: {passthrough:?}"
        );
    }

    #[test]
    fn image_response_fields_match_the_openai_type() {
        let ImagesResponse {
            created,
            data,
            background,
            output_format,
            size,
            quality,
            usage,
        } = serde_json::from_str(FULL_IMAGE_RESPONSE).unwrap();
        assert!(created > 0);
        assert_eq!(
            data.len(),
            2,
            "FULL_IMAGE_RESPONSE has one image of each kind"
        );
        for image in &data {
            match &**image {
                openai::Image::Url {
                    url,
                    revised_prompt: _,
                } => assert!(!url.is_empty()),
                openai::Image::B64Json {
                    b64_json,
                    revised_prompt: _,
                } => assert!(!b64_json.is_empty()),
            }
        }
        for (field, set) in [
            ("background", background.is_some()),
            ("output_format", output_format.is_some()),
            ("size", size.is_some()),
            ("quality", quality.is_some()),
        ] {
            assert!(set, "FULL_IMAGE_RESPONSE does not set {field}");
        }
        let openai::ImageGenUsage {
            input_tokens: _,
            total_tokens: _,
            output_tokens: _,
            output_token_details,
            input_tokens_details,
        } = usage.expect("FULL_IMAGE_RESPONSE does not set usage");
        let openai::ImageGenOutputTokensDetails {
            text_tokens: _,
            image_tokens: _,
        } = output_token_details.expect("FULL_IMAGE_RESPONSE does not set output_token_details");
        let openai::ImageGenInputUsageDetails {
            text_tokens: _,
            image_tokens: _,
        } = input_tokens_details;

        let NvImagesResponse {
            created: _,
            data,
            background: _,
            output_format: _,
            size: _,
            quality: _,
            usage,
            nvext,
        } = serde_json::from_str(FULL_IMAGE_RESPONSE).unwrap();
        for ImageData {
            url,
            b64_json,
            revised_prompt: _,
        } in &data
        {
            assert!(url.is_some() || b64_json.is_some());
        }
        let ImageUsage {
            input_tokens: _,
            total_tokens: _,
            output_tokens: _,
            output_token_details,
            input_tokens_details,
        } = usage.unwrap();
        let ImageTokenDetails {
            text_tokens: _,
            image_tokens: _,
        } = output_token_details.unwrap();
        let ImageTokenDetails {
            text_tokens: _,
            image_tokens: _,
        } = input_tokens_details;
        assert!(
            nvext.is_none(),
            "FULL_IMAGE_RESPONSE has no Dynamo extension"
        );
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
    fn image_request_rejects_jpg_output_format_alias() {
        let json = r#"{"prompt":"a cat","output_format":"jpg"}"#;
        assert!(serde_json::from_str::<NvCreateImageRequest>(json).is_err());
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

    #[test]
    fn image_request_n_in_range_passes_validation() {
        // The bounds are inclusive. An absent `n` is the worker's default.
        for json in [
            r#"{"prompt":"a cat"}"#,
            r#"{"prompt":"a cat","n":1}"#,
            r#"{"prompt":"a cat","n":10}"#,
        ] {
            let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
            assert!(
                ValidateRequest::validate(&req).is_ok(),
                "expected {json} to pass validation"
            );
        }
    }

    #[test]
    fn image_request_n_out_of_range_fails_validation() {
        for json in [
            r#"{"prompt":"a cat","n":0}"#,
            r#"{"prompt":"a cat","n":11}"#,
        ] {
            let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
            let message = ValidateRequest::validate(&req).unwrap_err().to_string();
            assert!(
                message.contains("n must be between 1 and 10"),
                "expected the error for {json} to state the rule; got: {message}"
            );
        }
    }

    #[test]
    fn image_request_num_inference_steps_bound() {
        // Zero denoising steps is no generation. An absent value is the
        // worker's default; 1 is the smallest valid value.
        for (json, rule) in [
            (r#"{"prompt":"a cat"}"#, None),
            (
                r#"{"prompt":"a cat","nvext":{"num_inference_steps":1}}"#,
                None,
            ),
            (
                r#"{"prompt":"a cat","nvext":{"num_inference_steps":0}}"#,
                Some("num_inference_steps must be at least 1"),
            ),
        ] {
            let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
            match (ValidateRequest::validate(&req), rule) {
                (Ok(()), None) => {}
                (Err(err), Some(rule)) => assert!(
                    err.to_string().contains(rule),
                    "expected the error for {json} to state the rule; got: {err}"
                ),
                (result, rule) => panic!("{json}: expected {rule:?}, got {result:?}"),
            }
        }
    }

    #[test]
    fn image_request_output_compression_and_partial_images_bounds() {
        // 0 to 100 and 0 to 3 as the OpenAI API documents; the bounds are
        // inclusive. serde already rejects a negative value for a u8.
        for (json, rule) in [
            (r#"{"prompt":"a cat","output_compression":0}"#, None),
            (r#"{"prompt":"a cat","output_compression":100}"#, None),
            (
                r#"{"prompt":"a cat","output_compression":101}"#,
                Some("output_compression must be between 0 and 100"),
            ),
            (r#"{"prompt":"a cat","partial_images":0}"#, None),
            (r#"{"prompt":"a cat","partial_images":3}"#, None),
            (
                r#"{"prompt":"a cat","partial_images":4}"#,
                Some("partial_images must be between 0 and 3"),
            ),
        ] {
            let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
            match (ValidateRequest::validate(&req), rule) {
                (Ok(()), None) => {}
                (Err(err), Some(rule)) => assert!(
                    err.to_string().contains(rule),
                    "expected the error for {json} to state the rule; got: {err}"
                ),
                (result, _) => panic!("unexpected result for {json}: {result:?}"),
            }
        }
    }

    // --- NvImagesResponse ---

    #[test]
    fn image_response_wire_shape_with_and_without_accounting() {
        // What every worker sent before the extension existed: no `nvext`
        // key on the way out, and `usage` stays null.
        let legacy: NvImagesResponse =
            serde_json::from_str(r#"{"created":1,"data":[{"b64_json":"YQ=="}]}"#).unwrap();
        assert!(legacy.nvext.is_none());
        let out = serde_json::to_value(&legacy).unwrap();
        assert!(out.get("nvext").is_none(), "{out}");
        assert_eq!(out["usage"], serde_json::Value::Null);

        // Only `output_images` is required; an unknown step count is omitted,
        // not written as 0.
        let partial: NvImagesResponse = serde_json::from_str(
            r#"{"created":1,"data":[],"nvext":{"usage":{"output_images":1}}}"#,
        )
        .unwrap();
        let usage = partial.nvext.as_ref().unwrap().usage.unwrap();
        assert_eq!((usage.output_images, usage.input_images), (1, 0));
        assert_eq!(usage.num_inference_steps, None);
        let out = serde_json::to_value(&partial).unwrap();
        assert_eq!(
            out["nvext"]["usage"],
            serde_json::json!({"output_images": 1, "input_images": 0})
        );

        // The full extension round-trips next to the OpenAI fields.
        let json = r#"{"created":1,"data":[{"b64_json":"YQ=="},{"b64_json":"Yg=="}],"size":"768x512","output_format":"png","nvext":{"usage":{"output_images":2,"input_images":1,"num_inference_steps":4}}}"#;
        let full: NvImagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            full.nvext.as_ref().unwrap().usage.unwrap(),
            ImageAccounting {
                output_images: 2,
                input_images: 1,
                num_inference_steps: Some(4),
            }
        );
        let out = serde_json::to_value(&full).unwrap();
        assert_eq!(
            out["nvext"],
            serde_json::json!({"usage": {"output_images": 2, "input_images": 1, "num_inference_steps": 4}})
        );
        assert_eq!(out["size"], serde_json::json!("768x512"));
        assert_eq!(out["output_format"], serde_json::json!("png"));
        assert_eq!(out["usage"], serde_json::Value::Null);
    }
}
