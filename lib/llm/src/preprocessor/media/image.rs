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
        Ok(array.try_into()?)
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
    #[case(4, image::ImageFormat::Bmp, 7, 9, 4, "RGBA BMP")]
    #[case(1, image::ImageFormat::Bmp, 5, 6, 3, "Grayscale BMP")] // BMP converts grayscale to RGB
    #[case(3, image::ImageFormat::Gif, 10, 10, 4, "RGB GIF")] // GIF may add alpha channel
    #[case(3, image::ImageFormat::WebP, 8, 8, 3, "RGB WebP")]
    #[case(4, image::ImageFormat::WebP, 9, 11, 4, "RGBA WebP")]
    #[case(1, image::ImageFormat::WebP, 6, 7, 3, "Grayscale WebP")] // WebP converts grayscale to RGB
    fn test_decode_image_formats(
        #[case] input_channels: u32,
        #[case] format: image::ImageFormat,
        #[case] width: u32,
        #[case] height: u32,
        #[case] expected_channels: u32,
        #[case] description: &str,
    ) {
        // Skip JPEG for non-RGB formats (JPEG doesn't support transparency or pure grayscale)
        if format == image::ImageFormat::Jpeg && input_channels != 3 {
            return;
        }

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
    #[case(Some(100), 8, 10, image::ImageFormat::Png, true, "within limit")] // 80 pixels < 100
    #[case(Some(50), 10, 10, image::ImageFormat::Jpeg, false, "exceeds limit")] // 100 pixels > 50
    #[case(Some(25), 5, 5, image::ImageFormat::Bmp, true, "exactly at limit")] // 25 pixels = 25
    #[case(None, 200, 300, image::ImageFormat::Png, true, "no limit")] // 60,000 pixels, no limit
    #[case(Some(100), 9, 10, image::ImageFormat::WebP, true, "webp within limit")] // 90 pixels < 100
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

    #[test]
    fn test_invalid_image_data() {
        let decoder = ImageDecoder::default();
        // Random bytes that are not a valid image
        let invalid_bytes = vec![0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8];
        let encoded_data = create_encoded_media_data(invalid_bytes);

        let result = decoder.decode(encoded_data);
        assert!(
            result.is_err(),
            "Should fail when decoding invalid image data"
        );
    }

    #[test]
    fn test_empty_image_data() {
        let decoder = ImageDecoder::default();
        let empty_bytes = vec![];
        let encoded_data = create_encoded_media_data(empty_bytes);

        let result = decoder.decode(encoded_data);
        assert!(result.is_err(), "Should fail when decoding empty data");
    }

    #[rstest]
    #[case(3, image::ImageFormat::Png)]
    #[case(4, image::ImageFormat::Png)]
    #[case(1, image::ImageFormat::Png)]
    #[case(3, image::ImageFormat::Bmp)]
    #[case(1, image::ImageFormat::Bmp)]
    #[case(3, image::ImageFormat::Jpeg)]
    #[case(3, image::ImageFormat::WebP)]
    #[case(4, image::ImageFormat::WebP)]
    #[case(1, image::ImageFormat::WebP)]
    #[case(3, image::ImageFormat::Gif)]
    fn test_small_edge_case(#[case] input_channels: u32, #[case] format: image::ImageFormat) {
        let decoder = ImageDecoder::default();
        // Test with 1x1 image (smallest possible)
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
