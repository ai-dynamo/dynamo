// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use base64::Engine as _;

pub mod image;
#[cfg(feature = "media-ffmpeg")]
pub mod video;

pub use image::ImageEncoder;
#[cfg(feature = "media-ffmpeg")]
pub use video::encode_video;

/// Trait for encoding raw pixel data into a specific format (e.g., PNG).
pub trait Encoder {
    /// Encode raw pixel bytes (HWC layout, u8) into an image format.
    ///
    /// # Arguments
    /// * `data` - Raw pixel bytes in HWC (height, width, channels) layout
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `channels` - Number of color channels (1=L, 3=RGB, 4=RGBA)
    fn encode(&self, data: &[u8], width: u32, height: u32, channels: u8) -> Result<Vec<u8>>;
}

/// Encode arbitrary bytes as base64 using the standard alphabet.
pub fn encode_base64(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_base64_empty() {
        assert_eq!(encode_base64(b""), "");
    }

    #[test]
    fn test_encode_base64_hello() {
        assert_eq!(encode_base64(b"hello"), "aGVsbG8=");
    }

    #[test]
    fn test_encode_base64_roundtrip() {
        let data = vec![0u8; 1024 * 1024]; // 1MB
        let encoded = encode_base64(&data);
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
        assert_eq!(decoded, data);
    }
}
