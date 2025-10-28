// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;

use super::common::EncodedMediaData;
use super::rdma::DecodedMediaData;

mod image;
mod video;

pub use image::ImageDecoder;
pub use video::VideoDecoder;

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
    #[serde(default)]
    pub video_decoder: VideoDecoder,
}
