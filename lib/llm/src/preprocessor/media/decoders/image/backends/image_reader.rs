// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Cursor;

use anyhow::Result;
use image::{GenericImageView, ImageFormat, ImageReader};

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
