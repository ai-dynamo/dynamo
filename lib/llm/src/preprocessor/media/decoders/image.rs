// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use image::GenericImageView;
use ndarray::Array3;

use super::super::common::EncodedMediaData;
use super::super::decoders::DecodedMediaData;
use super::Decoder;

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
        let pixel_count = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| anyhow::anyhow!("Image dimensions {width}x{height} overflow usize"))?;
        anyhow::ensure!(
            pixel_count <= max_pixels,
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

#[cfg(test)]
mod tests {
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
            decoded.shape,
            vec![height as usize, width as usize, expected_channels as usize]
        );
        assert_eq!(decoded.dtype, "uint8");
    }

    #[rstest]
    #[case(Some(200), 10, 10, image::ImageFormat::Png, true, "within limit")]
    #[case(Some(50), 10, 10, image::ImageFormat::Jpeg, false, "exceeds limit")]
    #[case(None, 200, 300, image::ImageFormat::Png, true, "no limit")]
    fn test_pixel_limits(
        #[case] max_pixels: Option<usize>,
        #[case] width: u32,
        #[case] height: u32,
        #[case] format: image::ImageFormat,
        #[case] should_succeed: bool,
        #[case] test_case: &str,
    ) {
        let decoder = ImageDecoder { max_pixels };
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
            assert_eq!(decoded.shape, vec![height as usize, width as usize, 3]);
            assert_eq!(
                decoded.dtype, "uint8",
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
                error_msg.contains("exceed max pixels"),
                "Error should mention exceeding max pixels for case: {}",
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
        assert_eq!(decoded.shape.len(), 3, "Should have 3 dimensions");
        assert_eq!(decoded.shape[0], 1, "Height should be 1");
        assert_eq!(decoded.shape[1], 1, "Width should be 1");
        assert_eq!(
            decoded.dtype, "uint8",
            "dtype should be uint8 for {} channels {:?}",
            input_channels, format
        );
    }
}
