// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

mod aggregator;
mod nvext;

pub use aggregator::DeltaAggregator;
pub use nvext::{NvExt, NvExtProvider};

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateImageRequest {
    #[serde(flatten)]
    pub inner: dynamo_async_openai::types::CreateImageRequest,

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
    pub inner: dynamo_async_openai::types::ImagesResponse,
}

impl NvImagesResponse {
    pub fn empty() -> Self {
        Self {
            inner: dynamo_async_openai::types::ImagesResponse {
                created: 0,
                data: vec![],
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
