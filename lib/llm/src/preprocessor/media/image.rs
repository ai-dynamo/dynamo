// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use image::GenericImageView;
use ndarray::Array3;

use super::common::{DecodedMediaData, Decoder, EncodedMediaData};

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageDecoder {
    // maximum total size of the image in pixels
    #[serde(default)]
    pub max_pixels: Option<usize>,
}

impl Decoder for ImageDecoder {
    fn decode(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        let bytes = data.into_bytes()?;
        let img = image::load_from_memory(&bytes)?;
        let (width, height) = img.dimensions();
        let n_channels = img.color().channel_count();

        let max_pixels = self.max_pixels.unwrap_or(usize::MAX);
        anyhow::ensure!(
            (width as usize) * (height as usize) <= max_pixels,
            "Image dimensions {width}x{height} exceed max pixels {max_pixels}"
        );
        let data = match n_channels {
            1 => img.to_luma8().into_raw(),
            2 => img.to_luma_alpha8().into_raw(),
            3 => img.to_rgb8().into_raw(),
            4 => img.to_rgba8().into_raw(),
            other => anyhow::bail!("Unsupported channel count {other}"),
        };
        let shape = (height as usize, width as usize, n_channels as usize);
        let array = Array3::from_shape_vec(shape, data)?;
        Ok(array.into())
    }
}
