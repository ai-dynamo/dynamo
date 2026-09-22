// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use anyhow::Result;
use image::{ColorType, ImageFormat};

use super::ImageDecoderLimits;

mod image_reader;
mod turbojpeg;

use image_reader::ImageReaderBackend;
use turbojpeg::TurboJpegBackend;

static IMAGE_READER_BACKEND: ImageReaderBackend = ImageReaderBackend;
static TURBO_JPEG_BACKEND: TurboJpegBackend = TurboJpegBackend;

pub(super) fn image_reader_backend() -> &'static dyn ImageDecodeBackend {
    &IMAGE_READER_BACKEND
}

pub(super) fn turbojpeg_backend() -> &'static dyn ImageDecodeBackend {
    &TURBO_JPEG_BACKEND
}

#[derive(Clone, Copy)]
pub(super) struct ImageDecodeRequest<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) format: ImageFormat,
    pub(super) limits: &'a ImageDecoderLimits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BackendAvailability {
    Available,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PixelFormat {
    L8,
    La8,
    Rgb8,
    Rgba8,
}

impl PixelFormat {
    pub(super) fn channels(self) -> usize {
        match self {
            Self::L8 => 1,
            Self::La8 => 2,
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
        }
    }

    pub(super) fn color_type(self) -> ColorType {
        match self {
            Self::L8 => ColorType::L8,
            Self::La8 => ColorType::La8,
            Self::Rgb8 => ColorType::Rgb8,
            Self::Rgba8 => ColorType::Rgba8,
        }
    }
}

#[derive(Debug)]
pub(super) struct DecodedImage {
    pub(super) source_format: ImageFormat,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pixel_format: PixelFormat,
    pub(super) pixels: Vec<u8>,
}

impl DecodedImage {
    pub(super) fn new(
        source_format: ImageFormat,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        pixels: Vec<u8>,
        limits: &ImageDecoderLimits,
    ) -> Result<Self> {
        let expected_len = limits.validate_output(width, height, pixel_format.channels())?;
        anyhow::ensure!(
            pixels.len() == expected_len,
            "Decoded image buffer has {} bytes, expected {expected_len}",
            pixels.len()
        );
        Ok(Self {
            source_format,
            width,
            height,
            pixel_format,
            pixels,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BackendDecline {
    Unavailable,
    UnsupportedFormat(ImageFormat),
    DecodeFailed,
}

impl fmt::Display for BackendDecline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => write!(f, "backend is unavailable"),
            Self::UnsupportedFormat(format) => {
                write!(f, "format {format:?} is not supported")
            }
            Self::DecodeFailed => write!(f, "backend could not decode the input"),
        }
    }
}

pub(super) enum ImageDecodeOutcome {
    Decoded(DecodedImage),
    // Only this outcome is eligible for fallback. Resource-limit and allocation
    // failures must be returned as errors so another backend cannot bypass them.
    NotHandled(BackendDecline),
}

/// Decodes one encoded image without owning selection, fallback, or metadata policy.
pub(super) trait ImageDecodeBackend: Sync {
    fn name(&self) -> &'static str;

    fn availability(&self) -> BackendAvailability;

    fn supports(&self, format: ImageFormat) -> bool;

    fn try_decode(&self, request: ImageDecodeRequest<'_>) -> Result<ImageDecodeOutcome>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_image_validates_buffer_length_and_limits() {
        let limits = ImageDecoderLimits {
            max_image_width: Some(4),
            max_image_height: Some(4),
            max_alloc: Some(12),
        };

        DecodedImage::new(
            ImageFormat::Jpeg,
            2,
            2,
            PixelFormat::Rgb8,
            vec![0; 12],
            &limits,
        )
        .unwrap();

        let wrong_len = DecodedImage::new(
            ImageFormat::Jpeg,
            2,
            2,
            PixelFormat::Rgb8,
            vec![0; 11],
            &limits,
        )
        .unwrap_err();
        assert!(wrong_len.to_string().contains("expected 12"));

        let too_large = DecodedImage::new(
            ImageFormat::Jpeg,
            3,
            2,
            PixelFormat::Rgb8,
            vec![0; 18],
            &limits,
        )
        .unwrap_err();
        assert!(too_large.to_string().contains("configured limit"));
    }

    #[test]
    fn turbojpeg_declines_unsupported_format_before_availability_check() {
        let limits = ImageDecoderLimits::default();
        let outcome = turbojpeg_backend()
            .try_decode(ImageDecodeRequest {
                bytes: b"not a PNG",
                format: ImageFormat::Png,
                limits: &limits,
            })
            .unwrap();

        assert!(matches!(
            outcome,
            ImageDecodeOutcome::NotHandled(BackendDecline::UnsupportedFormat(ImageFormat::Png))
        ));
    }
}
