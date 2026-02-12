// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

mod aggregator;
mod nvext;

pub use aggregator::DeltaAggregator;
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

    /// Response format: "url" or "b64_json" (default: "url")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    /// NVIDIA extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// Video data in response
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VideoData {
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
