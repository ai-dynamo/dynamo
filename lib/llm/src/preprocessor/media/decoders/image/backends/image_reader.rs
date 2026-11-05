// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Cursor;

use anyhow::{Context as _, Result};
use image::{
    DynamicImage, GenericImageView, GrayAlphaImage, GrayImage, ImageFormat, ImageReader, RgbImage,
    RgbaImage,
};

use super::{
    BackendAvailability, DecodedImage, ImageDecodeBackend, ImageDecodeOutcome, ImageDecodeRequest,
    PixelFormat,
};

pub(super) struct ImageReaderBackend;

impl ImageDecodeBackend for ImageReaderBackend {
    fn name(&self) -> &'static str {
        "image_reader"
    }

    fn availability(&self) -> BackendAvailability {
        BackendAvailability::Available
    }

    fn supports(&self, _format: ImageFormat) -> bool {
        true
    }

    fn try_decode(&self, request: ImageDecodeRequest<'_>) -> Result<ImageDecodeOutcome> {
        let mut reader = ImageReader::with_format(Cursor::new(request.bytes), request.format);
        let mut limits = image::Limits::no_limits();
        limits.max_image_width = request.limits.max_image_width;
        limits.max_image_height = request.limits.max_image_height;
        limits.max_alloc = request.limits.max_alloc;
        reader.limits(limits);

        let image = narrow_sixteen_bit(reader.decode()?, request.bytes)?;
        let (width, height) = image.dimensions();
        let channels = image.color().channel_count();
        let (pixels, pixel_format) = match channels {
            1 => (image.into_luma8().into_raw(), PixelFormat::L8),
            2 => (image.into_luma_alpha8().into_raw(), PixelFormat::La8),
            3 => (image.into_rgb8().into_raw(), PixelFormat::Rgb8),
            4 => (image.into_rgba8().into_raw(), PixelFormat::Rgba8),
            other => anyhow::bail!("Unsupported channel count {other}"),
        };

        Ok(ImageDecodeOutcome::Decoded(DecodedImage::new(
            request.format,
            width,
            height,
            pixel_format,
            pixels,
            request.limits,
        )?))
    }
}

/// PNG colour type 0: greyscale without an alpha channel.
const PNG_COLOUR_TYPE_GREYSCALE: u8 = 0;

/// Narrow 16-bit samples to 8 bits as vLLM's loaders do (Pillow in Python, and
/// llm-multimodal in vllm-rs), so frontend-decoded pixels match theirs:
/// greyscale samples are clipped at 255, and colour samples keep their high
/// byte. A greyscale PNG with a `tRNS` key decodes with an alpha channel here,
/// which those loaders drop.
fn narrow_sixteen_bit(image: DynamicImage, bytes: &[u8]) -> Result<DynamicImage> {
    let clipped = |width: u32, height: u32, samples: Vec<u8>| {
        GrayImage::from_raw(width, height, samples).map(DynamicImage::ImageLuma8)
    };
    let high_byte = |samples: Vec<u16>| samples.into_iter().map(|sample| (sample >> 8) as u8);
    let (width, height) = image.dimensions();
    let narrowed = match image {
        DynamicImage::ImageLuma16(wide) => clipped(
            width,
            height,
            wide.into_raw()
                .into_iter()
                .map(|sample| sample.min(255) as u8)
                .collect(),
        ),
        DynamicImage::ImageLumaA16(wide)
            if png_colour_type(bytes) == Some(PNG_COLOUR_TYPE_GREYSCALE) =>
        {
            clipped(
                width,
                height,
                wide.into_raw()
                    .chunks_exact(2)
                    .map(|pixel| pixel[0].min(255) as u8)
                    .collect(),
            )
        }
        DynamicImage::ImageLumaA16(wide) => {
            GrayAlphaImage::from_raw(width, height, high_byte(wide.into_raw()).collect())
                .map(DynamicImage::ImageLumaA8)
        }
        DynamicImage::ImageRgb16(wide) => {
            RgbImage::from_raw(width, height, high_byte(wide.into_raw()).collect())
                .map(DynamicImage::ImageRgb8)
        }
        DynamicImage::ImageRgba16(wide) => {
            RgbaImage::from_raw(width, height, high_byte(wide.into_raw()).collect())
                .map(DynamicImage::ImageRgba8)
        }
        other => return Ok(other),
    };
    narrowed.context("8-bit samples do not fit the image dimensions")
}

/// The colour type in a PNG's `IHDR` chunk, or `None` for other data.
fn png_colour_type(bytes: &[u8]) -> Option<u8> {
    const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
    (bytes.starts_with(SIGNATURE) && bytes.get(12..16) == Some(b"IHDR".as_slice()))
        .then(|| bytes.get(25).copied())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixteen_bit_samples_narrow_as_vllm_loaders_do() {
        let grey =
            image::ImageBuffer::<image::Luma<u16>, _>::from_raw(4, 1, vec![0, 255, 256, 0xFFFF])
                .unwrap();
        match narrow_sixteen_bit(DynamicImage::ImageLuma16(grey), &[]).unwrap() {
            DynamicImage::ImageLuma8(narrow) => assert_eq!(narrow.as_raw(), &[0, 255, 255, 255]),
            other => panic!("expected grey8, got {:?}", other.color()),
        }
        let rgb =
            image::ImageBuffer::<image::Rgb<u16>, _>::from_raw(1, 1, vec![0x01FF, 0xFF00, 0x00FF])
                .unwrap();
        match narrow_sixteen_bit(DynamicImage::ImageRgb16(rgb), &[]).unwrap() {
            DynamicImage::ImageRgb8(narrow) => assert_eq!(narrow.as_raw(), &[0x01, 0xFF, 0x00]),
            other => panic!("expected rgb8, got {:?}", other.color()),
        }
    }
}
