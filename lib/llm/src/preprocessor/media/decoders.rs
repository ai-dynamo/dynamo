// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::common::EncodedMediaData;
use super::rdma::DecodedMediaData;
use dynamo_model_card::media_config::MediaDecoder;
pub mod image;
#[cfg(feature = "media-ffmpeg")]
pub mod video;

pub use image::ImageMetadata;
#[cfg(feature = "media-ffmpeg")]
pub use video::VideoMetadata;

#[async_trait::async_trait]
pub trait Decoder: Clone + Send + 'static {
    fn decode(&self, data: EncodedMediaData) -> Result<DecodedMediaData>;

    // Merges this decoder with an optional runtime override.
    // Limits should always be enforced from the MDC config
    fn with_runtime(&self, runtime: Option<&Self>) -> Self;

    async fn decode_async(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        self.decode_async_with_video_hash(data, false).await
    }

    async fn decode_async_with_video_hash(
        &self,
        data: EncodedMediaData,
        hash_video: bool,
    ) -> Result<DecodedMediaData> {
        // light clone (only config params)
        let decoder = self.clone();
        // compute heavy -> rayon
        let result = tokio_rayon::spawn(move || {
            let mut decoded = decoder.decode(data)?;
            decoded.compute_content_hash(hash_video);
            Ok::<_, anyhow::Error>(decoded)
        })
        .await?;
        Ok(result)
    }
}

pub(crate) fn warn_if_unavailable_backends(decoder: &MediaDecoder) {
    if let Some(image) = &decoder.image {
        image::warn_if_libjpeg_unavailable(image);
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DecodedMediaMetadata {
    Image(ImageMetadata),
    #[cfg(feature = "media-ffmpeg")]
    Video(VideoMetadata),
}
