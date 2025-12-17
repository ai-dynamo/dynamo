// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde::{Deserialize, Serialize};

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

    async fn decode_async(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        // light clone (only config params)
        let decoder = self.clone();
        // compute heavy -> rayon
        let result = tokio_rayon::spawn(move || decoder.decode(data)).await?;
        Ok(result)
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct MediaDecoder {
    #[serde(default)]
    pub image_decoder: ImageDecoder,
    #[cfg(feature = "media-ffmpeg")]
    #[serde(default)]
    pub video_decoder: VideoDecoder,
    // TODO: audio decoder
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DecodedMediaMetadata {
    Image(ImageMetadata),
    #[cfg(feature = "media-ffmpeg")]
    Video(VideoMetadata),
}
