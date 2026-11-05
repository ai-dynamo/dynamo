// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    /// Rotate each decoded image by its EXIF orientation. A worker turns it on
    /// when its own image loader does the same, so that the pixels match. It
    /// is not in `image`: older frontends reject unknown `image` fields, but
    /// ignore unknown fields here.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub apply_exif_orientation: bool,
}

impl MediaDecoder {
    /// Decode images with the default limits. The frontend leaves video and
    /// audio parts as URLs for the backend.
    pub fn images_only() -> Self {
        Self {
            image: Some(ImageDecoder::default()),
            #[cfg(feature = "media-ffmpeg")]
            video: None,
            apply_exif_orientation: false,
        }
    }

    pub(crate) fn warn_if_unavailable_backends(&self) {
        if let Some(image) = &self.image {
            image.warn_if_libjpeg_unavailable();
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DecodedMediaMetadata {
    Image(ImageMetadata),
    #[cfg(feature = "media-ffmpeg")]
    Video(VideoMetadata),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A card without `apply_exif_orientation` keeps the stored orientation,
    /// and a card that sets it keeps an `image` section that older frontends
    /// read.
    #[test]
    fn exif_orientation_is_off_unless_the_card_sets_it() {
        let old_card: MediaDecoder =
            serde_json::from_value(serde_json::json!({"image": {}})).unwrap();
        assert!(!old_card.apply_exif_orientation);
        let default = serde_json::to_value(MediaDecoder::images_only()).unwrap();
        assert!(default.get("apply_exif_orientation").is_none());

        let card = serde_json::to_value(MediaDecoder {
            apply_exif_orientation: true,
            ..MediaDecoder::images_only()
        })
        .unwrap();
        assert_eq!(card["apply_exif_orientation"], true);
        // Older frontends reject unknown fields in `image`.
        assert!(card["image"].get("apply_exif_orientation").is_none());
        assert!(
            serde_json::from_value::<MediaDecoder>(card)
                .unwrap()
                .apply_exif_orientation
        );
    }
}
