// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Cursor;

use anyhow::Result;
use image::{ColorType, GenericImageView, ImageFormat, ImageReader};
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::super::common::EncodedMediaData;
use super::super::rdma::DecodedMediaData;
use super::{DecodedMediaMetadata, Decoder};

const DEFAULT_MAX_ALLOC: u64 = 128 * 1024 * 1024; // 128 MB

/// Image decoder limits - can only be set via server config, not runtime kwargs.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageDecoderLimits {
    #[serde(default)]
    pub max_image_width: Option<u32>,
    #[serde(default)]
    pub max_image_height: Option<u32>,
    /// Maximum allowed total allocation of the decoder in bytes
    #[serde(default)]
    pub max_alloc: Option<u64>,
}

impl Default for ImageDecoderLimits {
    fn default() -> Self {
        Self {
            max_image_width: None,
            max_image_height: None,
            max_alloc: Some(DEFAULT_MAX_ALLOC),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageDecoder {
    #[serde(default)]
    pub(crate) limits: ImageDecoderLimits,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum ImageLayout {
    HWC,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct ImageMetadata {
    pub(crate) format: Option<ImageFormat>,
    pub(crate) color_type: ColorType,
    pub(crate) layout: ImageLayout,
}

impl Decoder for ImageDecoder {
    fn with_runtime(&self, runtime: Option<&Self>) -> Self {
        match runtime {
            Some(r) => {
                let mut d = r.clone();
                d.limits.clone_from(&self.limits);
                d
            }
            None => self.clone(),
        }
    }

    fn decode(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        let bytes = data.into_bytes()?;

        let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
        let mut limits = image::Limits::no_limits();
        limits.max_image_width = self.limits.max_image_width;
        limits.max_image_height = self.limits.max_image_height;
        limits.max_alloc = self.limits.max_alloc;
        reader.limits(limits);

        let format = reader.format();

        let img = reader.decode()?;
        let n_channels = img.color().channel_count();

        let (data, color_type) = match n_channels {
            1 => (img.to_luma8().into_raw(), ColorType::L8),
            2 => (img.to_luma_alpha8().into_raw(), ColorType::La8),
            3 => (img.to_rgb8().into_raw(), ColorType::Rgb8),
            4 => (img.to_rgba8().into_raw(), ColorType::Rgba8),
            other => anyhow::bail!("Unsupported channel count {other}"),
        };

        let (width, height) = img.dimensions();
        let shape = (height as usize, width as usize, n_channels as usize);
        let array = Array3::from_shape_vec(shape, data)?;
        let mut decoded: DecodedMediaData = array.try_into()?;
        decoded.tensor_info.metadata = Some(DecodedMediaMetadata::Image(ImageMetadata {
            format,
            color_type,
            layout: ImageLayout::HWC,
        }));
        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::rdma::DataType;
    use super::*;
    use image::{DynamicImage, ImageBuffer};
    use rstest::rstest;
    use std::io::Cursor;

    fn create_encoded_media_data(bytes: Vec<u8>) -> EncodedMediaData {
        EncodedMediaData {
            bytes,
            b64_encoded: false,
        }
    }

    fn create_test_image(
        width: u32,
        height: u32,
        channels: u32,
        format: image::ImageFormat,
    ) -> Vec<u8> {
        // Create dynamic image based on number of channels with constant values
        let pixels = vec![128u8; channels as usize].repeat((width * height) as usize);
        let dynamic_image = match channels {
            1 => DynamicImage::ImageLuma8(
                ImageBuffer::from_vec(width, height, pixels).expect("Failed to create image"),
            ),
            3 => DynamicImage::ImageRgb8(
                ImageBuffer::from_vec(width, height, pixels).expect("Failed to create image"),
            ),
            4 => DynamicImage::ImageRgba8(
                ImageBuffer::from_vec(width, height, pixels).expect("Failed to create image"),
            ),
            _ => unreachable!("Already validated channel count above"),
        };

        // Encode to bytes
        let mut bytes = Vec::new();
        dynamic_image
            .write_to(&mut Cursor::new(&mut bytes), format)
            .expect("Failed to encode test image");
        bytes
    }

    #[rstest]
    #[case(3, image::ImageFormat::Png, 10, 10, 3, "RGB PNG")]
    #[case(4, image::ImageFormat::Png, 25, 30, 4, "RGBA PNG")]
    #[case(1, image::ImageFormat::Png, 8, 12, 1, "Grayscale PNG")]
    #[case(3, image::ImageFormat::Jpeg, 15, 20, 3, "RGB JPEG")]
    #[case(3, image::ImageFormat::Bmp, 12, 18, 3, "RGB BMP")]
    #[case(3, image::ImageFormat::WebP, 8, 8, 3, "RGB WebP")]
    fn test_image_decode(
        #[case] input_channels: u32,
        #[case] format: image::ImageFormat,
        #[case] width: u32,
        #[case] height: u32,
        #[case] expected_channels: u32,
        #[case] description: &str,
    ) {
        let decoder = ImageDecoder::default();
        let image_bytes = create_test_image(width, height, input_channels, format);
        let encoded_data = create_encoded_media_data(image_bytes);

        let result = decoder.decode(encoded_data);
        assert!(result.is_ok(), "Failed to decode {}", description);

        let decoded = result.unwrap();
        assert_eq!(
            decoded.tensor_info.shape,
            vec![height as usize, width as usize, expected_channels as usize]
        );
        assert_eq!(decoded.tensor_info.dtype, DataType::UINT8);
    }

    #[rstest]
    #[case(Some(100), None, 50, 50, ImageFormat::Png, true, "width ok")]
    #[case(Some(50), None, 100, 50, ImageFormat::Jpeg, false, "width too large")]
    #[case(None, Some(100), 50, 100, ImageFormat::Png, true, "height ok")]
    #[case(None, Some(50), 50, 100, ImageFormat::Png, false, "height too large")]
    #[case(None, None, 2000, 2000, ImageFormat::Png, true, "no limits")]
    #[case(None, None, 8000, 8000, ImageFormat::Png, false, "alloc too large")]
    fn test_limits(
        #[case] max_width: Option<u32>,
        #[case] max_height: Option<u32>,
        #[case] width: u32,
        #[case] height: u32,
        #[case] format: image::ImageFormat,
        #[case] should_succeed: bool,
        #[case] test_case: &str,
    ) {
        let decoder = ImageDecoder {
            limits: ImageDecoderLimits {
                max_image_width: max_width,
                max_image_height: max_height,
                max_alloc: Some(DEFAULT_MAX_ALLOC),
            },
        };
        let image_bytes = create_test_image(width, height, 3, format); // RGB
        let encoded_data = create_encoded_media_data(image_bytes);

        let result = decoder.decode(encoded_data);

        if should_succeed {
            assert!(
                result.is_ok(),
                "Should decode successfully for case: {} with format {:?}",
                test_case,
                format
            );
            let decoded = result.unwrap();
            assert_eq!(
                decoded.tensor_info.shape,
                vec![height as usize, width as usize, 3]
            );
            assert_eq!(
                decoded.tensor_info.dtype,
                DataType::UINT8,
                "dtype should be uint8 for case: {}",
                test_case
            );
        } else {
            assert!(
                result.is_err(),
                "Should fail for case: {} with format {:?}",
                test_case,
                format
            );
            let error_msg = result.unwrap_err().to_string();
            assert!(
                error_msg.contains("dimensions") || error_msg.contains("limit"),
                "Error should mention dimension limits, got: {} for case: {}",
                error_msg,
                test_case
            );
        }
    }

    #[rstest]
    #[case(3, image::ImageFormat::Png)]
    fn test_decode_1x1_image(#[case] input_channels: u32, #[case] format: image::ImageFormat) {
        let decoder = ImageDecoder::default();
        let image_bytes = create_test_image(1, 1, input_channels, format);
        let encoded_data = create_encoded_media_data(image_bytes);

        let result = decoder.decode(encoded_data);
        assert!(
            result.is_ok(),
            "Should decode 1x1 image with {} channels in {:?} format successfully",
            input_channels,
            format
        );

        let decoded = result.unwrap();
        assert_eq!(
            decoded.tensor_info.shape.len(),
            3,
            "Should have 3 dimensions"
        );
        assert_eq!(decoded.tensor_info.shape[0], 1, "Height should be 1");
        assert_eq!(decoded.tensor_info.shape[1], 1, "Width should be 1");
        assert_eq!(
            decoded.tensor_info.dtype,
            DataType::UINT8,
            "dtype should be uint8 for {} channels {:?}",
            input_channels,
            format
        );
    }

    #[test]
    fn test_with_runtime_limit_enforcement() {
        let server_limits = ImageDecoderLimits {
            max_image_width: Some(100),
            max_image_height: Some(100),
            max_alloc: Some(1024),
        };
        let server_config = ImageDecoder {
            limits: server_limits.clone(),
        };

        // Runtime config tries to override limits (should be ignored)
        let runtime_limits = ImageDecoderLimits {
            max_image_width: Some(9999),
            max_image_height: Some(9999),
            max_alloc: Some(999999),
        };
        let runtime_config = ImageDecoder {
            limits: runtime_limits,
        };

        let merged = server_config.with_runtime(Some(&runtime_config));

        // Check that server limits are preserved
        assert_eq!(merged.limits.max_image_width, Some(100));
        assert_eq!(merged.limits.max_image_height, Some(100));
        assert_eq!(merged.limits.max_alloc, Some(1024));
    }
}
