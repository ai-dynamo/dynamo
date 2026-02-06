// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ColorType, ImageEncoder as _};

use super::Encoder;

/// PNG image encoder optimized for speed.
///
/// Uses fast compression and adaptive filtering for a good balance
/// between output size and encoding speed.
#[derive(Clone, Debug, Default)]
pub struct ImageEncoder;

impl Encoder for ImageEncoder {
    fn encode(&self, data: &[u8], width: u32, height: u32, channels: u8) -> Result<Vec<u8>> {
        let color_type = match channels {
            1 => ColorType::L8,
            3 => ColorType::Rgb8,
            4 => ColorType::Rgba8,
            other => bail!("Unsupported channel count: {other}. Expected 1 (L), 3 (RGB), or 4 (RGBA)."),
        };

        let expected_len = width as usize * height as usize * channels as usize;
        if data.len() != expected_len {
            bail!(
                "Data length mismatch: expected {} bytes ({}x{}x{}), got {}",
                expected_len,
                width,
                height,
                channels,
                data.len()
            );
        }

        // Pre-allocate output buffer. PNG compressed output is typically smaller
        // than raw pixels, so half the input size is a reasonable initial capacity.
        let mut buf = Vec::with_capacity(expected_len / 2);

        let encoder = PngEncoder::new_with_quality(&mut buf, CompressionType::Fast, FilterType::Adaptive);
        encoder.write_image(data, width, height, color_type.into())?;

        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_png_rgb() {
        let encoder = ImageEncoder;
        let width = 64;
        let height = 64;
        let channels = 3u8;
        let data = vec![128u8; width as usize * height as usize * channels as usize];

        let result = encoder.encode(&data, width, height, channels);
        assert!(result.is_ok(), "PNG encoding should succeed for RGB");

        let png_bytes = result.unwrap();
        // Check PNG magic bytes
        assert!(png_bytes.len() >= 8);
        assert_eq!(&png_bytes[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn test_encode_png_rgba() {
        let encoder = ImageEncoder;
        let width = 32;
        let height = 32;
        let channels = 4u8;
        let data = vec![200u8; width as usize * height as usize * channels as usize];

        let result = encoder.encode(&data, width, height, channels);
        assert!(result.is_ok(), "PNG encoding should succeed for RGBA");

        let png_bytes = result.unwrap();
        assert_eq!(&png_bytes[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn test_encode_png_grayscale() {
        let encoder = ImageEncoder;
        let width = 16;
        let height = 16;
        let channels = 1u8;
        let data = vec![100u8; width as usize * height as usize * channels as usize];

        let result = encoder.encode(&data, width, height, channels);
        assert!(result.is_ok(), "PNG encoding should succeed for grayscale");

        let png_bytes = result.unwrap();
        assert_eq!(&png_bytes[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn test_encode_png_1024x1024() {
        let encoder = ImageEncoder;
        let width = 1024;
        let height = 1024;
        let channels = 3u8;
        let data = vec![128u8; width as usize * height as usize * channels as usize];

        let result = encoder.encode(&data, width, height, channels);
        assert!(result.is_ok(), "PNG encoding should succeed for 1024x1024");

        let png_bytes = result.unwrap();
        assert_eq!(&png_bytes[..8], b"\x89PNG\r\n\x1a\n");
        // Compressed size should be significantly less than raw
        assert!(png_bytes.len() < data.len());
    }

    #[test]
    fn test_encode_data_length_mismatch() {
        let encoder = ImageEncoder;
        let data = vec![0u8; 100]; // Wrong length for 64x64x3

        let result = encoder.encode(&data, 64, 64, 3);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Data length mismatch"),
            "Error should mention data length mismatch, got: {}",
            err
        );
    }

    #[test]
    fn test_encode_unsupported_channels() {
        let encoder = ImageEncoder;
        let data = vec![0u8; 64 * 64 * 2]; // 2 channels not supported

        let result = encoder.encode(&data, 64, 64, 2);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Unsupported channel count"),
            "Error should mention unsupported channels, got: {}",
            err
        );
    }

    #[test]
    fn test_encode_roundtrip() {
        // Verify that encoded PNG can be decoded back to the same pixel data
        let encoder = ImageEncoder;
        let width = 8u32;
        let height = 8u32;
        let channels = 3u8;

        // Create a test pattern with varying pixel values
        let mut data = Vec::with_capacity(width as usize * height as usize * channels as usize);
        for y in 0..height {
            for x in 0..width {
                data.push((x * 32) as u8);       // R
                data.push((y * 32) as u8);       // G
                data.push(((x + y) * 16) as u8); // B
            }
        }

        let png_bytes = encoder.encode(&data, width, height, channels).unwrap();

        // Decode with the image crate
        let img = image::load_from_memory(&png_bytes).unwrap();
        let decoded = img.to_rgb8().into_raw();
        assert_eq!(decoded, data, "Decoded pixels should match original");
    }
}
