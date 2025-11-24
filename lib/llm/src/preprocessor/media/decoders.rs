// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;

use super::common::EncodedMediaData;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
mod image;

pub use image::{ImageDecoder, ImageMetadata};

#[derive(Debug)]
pub enum DecodedMediaMetadata {
    #[allow(dead_code)] // used in followup MR
    Image(ImageMetadata),
}

#[derive(Debug, PartialEq, Eq)]
pub enum DataType {
    UINT8,
}

// Decoded media data (image RGB, video frames pixels, ...)
#[derive(Debug)]
pub struct DecodedMediaData {
    #[allow(dead_code)] // used in followup MR
    pub(crate) data: Vec<u8>,
    #[allow(dead_code)] // used in followup MR
    pub(crate) shape: Vec<usize>,
    #[allow(dead_code)] // used in followup MR
    pub(crate) dtype: DataType,
    #[allow(dead_code)] // used in followup MR
    pub(crate) metadata: Option<DecodedMediaMetadata>,
}

// convert Array{N}<u8> to DecodedMediaData
// TODO: Array1<f32> for audio
impl<D: Dimension> From<ArrayBase<OwnedRepr<u8>, D>> for DecodedMediaData {
    fn from(array: ArrayBase<OwnedRepr<u8>, D>) -> Self {
        let shape = array.shape().to_vec();
        let (data, _) = array.into_raw_vec_and_offset();
        Self {
            data,
            shape,
            dtype: DataType::UINT8,
            metadata: None,
        }
    }
}

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
    // TODO: video, audio decoders
}
