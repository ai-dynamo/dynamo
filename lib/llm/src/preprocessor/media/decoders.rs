// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::common::EncodedMediaData;
use super::rdma::DecodedMediaData;
pub mod image;
#[cfg(feature = "media-ffmpeg")]
pub mod video;

pub use image::{ImageDecoder, ImageMetadata};
#[cfg(feature = "media-ffmpeg")]
pub use video::{VideoDecoder, VideoMetadata};

#[async_trait::async_trait]
pub trait Decoder: Clone + Send + 'static {
    fn decode(&self, data: EncodedMediaData) -> Result<DecodedMediaData>;

    // Merges this decoder with an optional runtime override.
    // Limits should always be enforced from the MDC config
    fn with_runtime(&self, runtime: Option<&Self>) -> Self;

    async fn decode_async(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        // light clone (only config params)
        let decoder = self.clone();
        // compute heavy -> rayon
        let result = tokio_rayon::spawn(move || decoder.decode(data)).await?;
        Ok(result)
    }
}

/// Media decoder configuration.
/// Used both for MDC server config and runtime `media_io_kwargs`.
/// When used at runtime, limits are enforced from MDC and cannot be overridden.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, ToSchema)]
pub struct MediaDecoder {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageDecoder>,
    #[cfg(feature = "media-ffmpeg")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoDecoder>,
    // TODO: audio decoder
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DecodedMediaMetadata {
    Image(ImageMetadata),
    #[cfg(feature = "media-ffmpeg")]
    Video(VideoMetadata),
}
