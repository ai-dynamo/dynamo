// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};

mod aggregator;
mod nvext;

pub use nvext::{ImageAccounting, ImagesResponseNvExt, NvExt};

/// Image generation request with NVIDIA extensions.
///
/// Serde is hand-rolled: `inner` occupies the top level of the wire body,
/// and a derived flattened map next to a flattened struct would capture the
/// struct's keys too. The manual impls keep one external contract, typed
/// fields at the top level plus unknown top-level fields retained in
/// [`Self::passthrough`].
#[derive(Debug, Clone)]
pub struct NvCreateImageRequest {
    pub inner: dynamo_protocols::types::CreateImageRequest,

    /// Optional image reference that guides generation (for I2I/TI2I).
    pub input_reference: Option<String>,

    pub nvext: Option<NvExt>,

    /// Worker-boundary contract, not a public field: the frontend moves
    /// `passthrough` under `extra_args["media_passthrough"]` before
    /// dispatch (see [`Self::nest_passthrough`]) so workers read one
    /// explicit nested entry. A client-sent `extra_args` lands in
    /// `passthrough` like any other unknown field.
    pub extra_args: Option<serde_json::Map<String, serde_json::Value>>,

    /// Unknown top-level fields are retained here and forwarded to the
    /// backend without strict validation. This matches the OpenAI client's
    /// extra_body option, which merges into the top level of the body.
    /// Stable knobs can be promoted to typed fields over time.
    pub passthrough: serde_json::Map<String, serde_json::Value>,
}

impl NvCreateImageRequest {
    /// Nest captured top-level unknowns under `extra_args["media_passthrough"]`
    /// for dispatch to a worker.
    pub fn nest_passthrough(&mut self) {
        super::nest_media_passthrough(&mut self.passthrough, &mut self.extra_args);
    }
}

impl<'de> Deserialize<'de> for NvCreateImageRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut body = serde_json::Map::deserialize(deserializer)?;
        let input_reference = match body.remove("input_reference") {
            Some(value) => serde_json::from_value(value).map_err(serde::de::Error::custom)?,
            None => None,
        };
        let nvext = match body.remove("nvext") {
            Some(value) => serde_json::from_value(value).map_err(serde::de::Error::custom)?,
            None => None,
        };
        let inner: dynamo_protocols::types::CreateImageRequest =
            serde_json::from_value(serde_json::Value::Object(body.clone()))
                .map_err(serde::de::Error::custom)?;
        // Keys the typed request consumed stay out of the passthrough.
        if let serde_json::Value::Object(consumed) =
            serde_json::to_value(&inner).map_err(serde::de::Error::custom)?
        {
            for key in consumed.keys() {
                body.remove(key);
            }
        }
        Ok(Self {
            inner,
            input_reference,
            nvext,
            extra_args: None,
            passthrough: body,
        })
    }
}

impl Serialize for NvCreateImageRequest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut body = match serde_json::to_value(&self.inner).map_err(serde::ser::Error::custom)? {
            serde_json::Value::Object(map) => map,
            _ => {
                return Err(serde::ser::Error::custom(
                    "image request must serialize to an object",
                ));
            }
        };
        // Typed fields win over passthrough entries of the same name.
        for (key, value) in &self.passthrough {
            body.entry(key.clone()).or_insert_with(|| value.clone());
        }
        if let Some(input_reference) = &self.input_reference {
            body.insert(
                "input_reference".to_string(),
                serde_json::Value::String(input_reference.clone()),
            );
        }
        if let Some(nvext) = &self.nvext {
            body.insert(
                "nvext".to_string(),
                serde_json::to_value(nvext).map_err(serde::ser::Error::custom)?,
            );
        }
        if let Some(extra_args) = &self.extra_args {
            body.insert(
                "extra_args".to_string(),
                serde_json::Value::Object(extra_args.clone()),
            );
        }
        body.serialize(serializer)
    }
}

/// A response structure for image generation responses, embedding OpenAI's
/// `ImagesResponse`.
///
/// # Fields
/// - `inner`: The base OpenAI image response, embedded using `serde(flatten)`.
///   Its `usage` is the backend's token counts, or `null`.
/// - `nvext`: Dynamo's response extension, carrying image accounting for
///   models without token usage. Omitted when the worker reports nothing.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NvImagesResponse {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::ImagesResponse,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nvext: Option<ImagesResponseNvExt>,
}

impl NvImagesResponse {
    pub fn empty() -> Self {
        Self {
            inner: dynamo_protocols::types::ImagesResponse {
                created: 0,
                data: vec![],
                background: None,
                output_format: None,
                quality: None,
                size: None,
                usage: None,
            },
            nvext: None,
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
    use super::*;

    // --- NvCreateImageRequest ---

    #[test]
    fn image_request_captures_unknown_top_level_fields() {
        // The OpenAI client's extra_body option merges into the top level of
        // the body, so that is where backend knobs arrive.
        let json = r#"{"prompt":"a cat","think_mode":true,"size_override":{"h":512,"w":768}}"#;
        let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.inner.prompt, "a cat");
        assert_eq!(req.passthrough["think_mode"], serde_json::json!(true));
        assert_eq!(
            req.passthrough["size_override"]["h"],
            serde_json::json!(512)
        );

        let out = serde_json::to_string(&req).unwrap();
        let back: NvCreateImageRequest = serde_json::from_str(&out).unwrap();
        assert_eq!(back.inner.prompt, "a cat");
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
        assert_eq!(req.inner.prompt, "a cat");
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
        assert!(out.get("inner").is_none());
    }
}
