// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Cursor;

use anyhow::Result;
use image::{DynamicImage, GenericImageView, ImageFormat, ImageReader};

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

        let image = reader.decode()?;
        let (width, height) = image.dimensions();
        let channels = image.color().channel_count();
        let (pixels, pixel_format) = match narrow_sixteen_bit(&image, request.bytes) {
            Some(narrowed) => narrowed,
            None => match channels {
                1 => (image.into_luma8().into_raw(), PixelFormat::L8),
                2 => (image.into_luma_alpha8().into_raw(), PixelFormat::La8),
                3 => (image.into_rgb8().into_raw(), PixelFormat::Rgb8),
                4 => (image.into_rgba8().into_raw(), PixelFormat::Rgba8),
                other => anyhow::bail!("Unsupported channel count {other}"),
            },
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

const PNG_COLOUR_TYPE_GREYSCALE: u8 = 0;

/// Narrows like Pillow's `convert("RGB")` on a 16-bit PNG and like llm-multimodal (vllm-rs) on any
/// format. A greyscale PNG with a `tRNS` key decodes with alpha here, but those loaders keep it grey.
fn narrow_sixteen_bit(image: &DynamicImage, bytes: &[u8]) -> Option<(Vec<u8>, PixelFormat)> {
    let clip = |sample: &u16| (*sample).min(255) as u8;
    let high_byte = |sample: &u16| (*sample >> 8) as u8;
    Some(match image {
        DynamicImage::ImageLuma16(wide) => (wide.iter().map(clip).collect(), PixelFormat::L8),
        DynamicImage::ImageLumaA16(wide)
            if png_colour_type(bytes) == Some(PNG_COLOUR_TYPE_GREYSCALE) =>
        {
            (wide.iter().step_by(2).map(clip).collect(), PixelFormat::L8)
        }
        DynamicImage::ImageLumaA16(wide) => {
            (wide.iter().map(high_byte).collect(), PixelFormat::La8)
        }
        DynamicImage::ImageRgb16(wide) => (wide.iter().map(high_byte).collect(), PixelFormat::Rgb8),
        DynamicImage::ImageRgba16(wide) => {
            (wide.iter().map(high_byte).collect(), PixelFormat::Rgba8)
        }
        _ => return None,
    })
}

fn png_colour_type(bytes: &[u8]) -> Option<u8> {
    const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
    (bytes.starts_with(SIGNATURE) && bytes.get(12..16) == Some(b"IHDR".as_slice()))
        .then(|| bytes.get(25).copied())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, ImageEncoder as _, Luma, LumaA, Rgb};

    #[test]
    fn sixteen_bit_samples_narrow_as_vllm_loaders_do() {
        let mut grey_png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut grey_png)
            .write_image(&[0; 2], 1, 1, image::ExtendedColorType::L16)
            .unwrap();
        let grey = ImageBuffer::<Luma<u16>, _>::from_raw(4, 1, vec![0, 255, 256, 0xFFFF]).unwrap();
        let rgb = ImageBuffer::<Rgb<u16>, _>::from_raw(1, 1, vec![0x01FF, 0xFF00, 0x00FF]).unwrap();
        let grey_alpha =
            ImageBuffer::<LumaA<u16>, _>::from_raw(2, 1, vec![30000, 0, 200, 0xFFFF]).unwrap();
        let cases = [
            (
                DynamicImage::ImageLuma16(grey),
                &[][..],
                vec![0, 255, 255, 255],
                PixelFormat::L8,
            ),
            (
                DynamicImage::ImageRgb16(rgb),
                &[][..],
                vec![0x01, 0xFF, 0x00],
                PixelFormat::Rgb8,
            ),
            // A greyscale PNG with a `tRNS` key decodes as grey+alpha.
            (
                DynamicImage::ImageLumaA16(grey_alpha.clone()),
                grey_png.as_slice(),
                vec![255, 200],
                PixelFormat::L8,
            ),
            (
                DynamicImage::ImageLumaA16(grey_alpha),
                &[][..],
                vec![0x75, 0x00, 0x00, 0xFF],
                PixelFormat::La8,
            ),
        ];
        for (image, bytes, pixels, pixel_format) in cases {
            assert_eq!(
                narrow_sixteen_bit(&image, bytes),
                Some((pixels, pixel_format))
            );
        }
    }
}
