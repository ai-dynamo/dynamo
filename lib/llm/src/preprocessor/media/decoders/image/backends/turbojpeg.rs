// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use image::ImageFormat;

use super::{
    BackendAvailability, BackendDecline, DecodedImage, ImageDecodeBackend, ImageDecodeOutcome,
    ImageDecodeRequest, PixelFormat,
};
use crate::preprocessor::media::jpeg_turbo;

pub(super) struct TurboJpegBackend;

impl ImageDecodeBackend for TurboJpegBackend {
    fn name(&self) -> &'static str {
        "libjpeg_turbo"
    }

    fn availability(&self) -> BackendAvailability {
        if jpeg_turbo::available() {
            BackendAvailability::Available
        } else {
            BackendAvailability::Unavailable
        }
    }

    fn supports(&self, format: ImageFormat) -> bool {
        format == ImageFormat::Jpeg
    }

    fn try_decode(&self, request: ImageDecodeRequest<'_>) -> Result<ImageDecodeOutcome> {
        if !self.supports(request.format) {
            return Ok(ImageDecodeOutcome::NotHandled(
                BackendDecline::UnsupportedFormat(request.format),
            ));
        }
        if self.availability() == BackendAvailability::Unavailable {
            return Ok(ImageDecodeOutcome::NotHandled(BackendDecline::Unavailable));
        }

        let Some(jpeg) = jpeg_turbo::decode_jpeg(
            request.bytes,
            request.limits.max_image_width,
            request.limits.max_image_height,
            request.limits.max_alloc,
        )?
        else {
            return Ok(ImageDecodeOutcome::NotHandled(BackendDecline::DecodeFailed));
        };

        let pixel_format = match jpeg.channels {
            1 => PixelFormat::L8,
            3 => PixelFormat::Rgb8,
            other => anyhow::bail!("Unsupported TurboJPEG channel count {other}"),
        };
        Ok(ImageDecodeOutcome::Decoded(DecodedImage::new(
            request.format,
            jpeg.width,
            jpeg.height,
            pixel_format,
            jpeg.data,
            request.limits,
        )?))
    }
}
