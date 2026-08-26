// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

mod aggregator;
mod nvext;

pub use nvext::{NvExt, NvExtProvider};

/// Image generation request with NVIDIA extensions.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateImageRequest {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateImageRequest,

    /// Optional image reference that guides generation (for I2I/TI2I).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_reference: Option<String>,

    /// Extra parameters passed through to the backend without strict
    /// validation. Meant for backend-specific knobs that have no typed
    /// field yet. Stable ones can be promoted to typed fields over time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// A response structure for image generation responses, embedding OpenAI's
/// `ImagesResponse`.
///
/// # Fields
/// - `inner`: The base OpenAI image response, embedded using `serde(flatten)`.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvImagesResponse {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::ImagesResponse,
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
        }
    }
}

/// Implements `NvExtProvider` for `NvCreateImageRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateImageRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
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
    fn image_request_extra_body_round_trips_beside_flattened_inner() {
        let json = r#"{"prompt":"a cat","extra_body":{"think_mode":true,"size_override":{"h":512,"w":768}}}"#;
        let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.inner.prompt, "a cat");
        let extra = req.extra_body.as_ref().unwrap();
        assert_eq!(extra["think_mode"], serde_json::json!(true));
        assert_eq!(extra["size_override"]["h"], serde_json::json!(512));

        let out = serde_json::to_string(&req).unwrap();
        let back: NvCreateImageRequest = serde_json::from_str(&out).unwrap();
        assert_eq!(back.extra_body, req.extra_body);
        assert_eq!(back.inner.prompt, "a cat");
    }

    #[test]
    fn image_request_extra_body_absent_is_none_and_omitted() {
        let json = r#"{"prompt":"a cat"}"#;
        let req: NvCreateImageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.extra_body, None);
        assert!(!serde_json::to_string(&req).unwrap().contains("extra_body"));
    }
}
