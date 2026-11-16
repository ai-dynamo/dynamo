// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Cursor;
use std::sync::Once;

use anyhow::{Context as _, Result};
use image::metadata::Orientation;
use image::{ColorType, DynamicImage, ImageDecoder as _, ImageFormat, ImageReader};
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::super::common::EncodedMediaData;
use super::super::rdma::DecodedMediaData;
use super::{DecodedMediaMetadata, Decoder};
use backends::{
    BackendAvailability, BackendDecline, DecodedImage, ImageDecodeBackend, ImageDecodeOutcome,
    ImageDecodeRequest, PixelFormat, image_reader_backend, turbojpeg_backend,
};

mod backends;

const DEFAULT_MAX_ALLOC: u64 = 128 * 1024 * 1024; // 128 MB
const ENABLE_LIBJPEG_ENV: &str = "DYN_MM_ENABLE_LIBJPEG";
// CI-only guard: an enabled JPEG test must exercise TurboJPEG, never its fallback.
const REQUIRE_LIBJPEG_TURBO_TEST_ENV: &str = "DYNAMO_REQUIRE_LIBJPEG_TURBO_TEST";
static LIBJPEG_TURBO_UNAVAILABLE_WARNING: Once = Once::new();

fn default_enable_libjpeg() -> bool {
    let value = std::env::var(ENABLE_LIBJPEG_ENV).ok();
    libjpeg_enabled(value.as_deref())
}

fn libjpeg_enabled(value: Option<&str>) -> bool {
    value
        .and_then(dynamo_runtime::config::parse_bool_opt)
        .unwrap_or(true)
}

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

impl ImageDecoderLimits {
    fn validate_output(&self, width: u32, height: u32, channels: usize) -> Result<usize> {
        if self.max_image_width.is_some_and(|limit| width > limit)
            || self.max_image_height.is_some_and(|limit| height > limit)
        {
            anyhow::bail!("Image dimensions exceed configured limits: {width}x{height}");
        }

        let nbytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(channels as u64))
            .ok_or_else(|| {
                anyhow::anyhow!("Image allocation size overflow for dimensions: {width}x{height}")
            })?;
        if let Some(limit) = self.max_alloc
            && nbytes > limit
        {
            anyhow::bail!("Image allocation {nbytes} bytes exceeds configured limit {limit} bytes");
        }
        usize::try_from(nbytes)
            .map_err(|_| anyhow::anyhow!("Image allocation does not fit in usize: {nbytes} bytes"))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageDecoder {
    #[serde(default)]
    pub(crate) limits: ImageDecoderLimits,
    /// Frontend-local backend selection; never part of model or request configuration.
    #[serde(skip, default = "default_enable_libjpeg")]
    #[schema(ignore)]
    pub(crate) enable_libjpeg: bool,
    /// Copied from [`super::MediaDecoder::apply_exif_orientation`] by the media loader.
    #[serde(skip)]
    #[schema(ignore)]
    pub(crate) apply_exif_orientation: bool,
}

impl Default for ImageDecoder {
    fn default() -> Self {
        Self {
            limits: ImageDecoderLimits::default(),
            enable_libjpeg: default_enable_libjpeg(),
            apply_exif_orientation: false,
        }
    }
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
                d.enable_libjpeg = self.enable_libjpeg;
                d.apply_exif_orientation = self.apply_exif_orientation;
                d
            }
            None => self.clone(),
        }
    }

    fn decode(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        let bytes = data.into_bytes()?;
        self.warn_if_libjpeg_unavailable();

        let format = image::guess_format(&bytes)?;
        let request = ImageDecodeRequest {
            bytes: &bytes,
            format,
            limits: &self.limits,
        };
        let turbojpeg = turbojpeg_backend();
        let image_reader = image_reader_backend();
        let image = if self.enable_libjpeg && turbojpeg.supports(format) {
            match turbojpeg.try_decode(request)? {
                ImageDecodeOutcome::Decoded(image) => image,
                ImageDecodeOutcome::NotHandled(reason) => {
                    ensure_fallback_allowed(
                        turbojpeg.name(),
                        reason,
                        std::env::var_os(REQUIRE_LIBJPEG_TURBO_TEST_ENV).is_some(),
                    )?;
                    decode_required_backend(image_reader, request)?
                }
            }
        } else {
            decode_required_backend(image_reader, request)?
        };

        let image = if self.apply_exif_orientation {
            oriented(image, &bytes, &self.limits)?
        } else {
            image
        };
        decoded_image_to_media_data(image)
    }
}

fn oriented(
    image: DecodedImage,
    bytes: &[u8],
    limits: &ImageDecoderLimits,
) -> Result<DecodedImage> {
    let orientation = match image.source_format {
        ImageFormat::WebP => webp_exif(bytes).and_then(Orientation::from_exif_chunk),
        format => ImageReader::with_format(Cursor::new(bytes), format)
            .into_decoder()
            .and_then(|mut decoder| decoder.orientation())
            .ok(),
    }
    .unwrap_or(Orientation::NoTransforms);
    if orientation == Orientation::NoTransforms {
        return Ok(image);
    }
    let DecodedImage {
        source_format,
        width,
        height,
        pixel_format,
        pixels,
    } = image;
    let mut image = match pixel_format {
        PixelFormat::L8 => {
            image::GrayImage::from_raw(width, height, pixels).map(DynamicImage::from)
        }
        PixelFormat::La8 => {
            image::GrayAlphaImage::from_raw(width, height, pixels).map(DynamicImage::from)
        }
        PixelFormat::Rgb8 => {
            image::RgbImage::from_raw(width, height, pixels).map(DynamicImage::from)
        }
        PixelFormat::Rgba8 => {
            image::RgbaImage::from_raw(width, height, pixels).map(DynamicImage::from)
        }
    }
    .context("decoded pixels do not fit their dimensions")?;
    image.apply_orientation(orientation);
    DecodedImage::new(
        source_format,
        image.width(),
        image.height(),
        pixel_format,
        image.into_bytes(),
        limits,
    )
}

// The EXIF chunk that image-webp records, read without its allocation of the declared size (up to
// 4 GiB). The scan bound is image-webp's: the end of the VP8X chunk plus the RIFF size minus 12.
fn webp_exif(bytes: &[u8]) -> Option<&[u8]> {
    let le32 = |at: usize| {
        let b = bytes.get(at..at + 4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    if bytes.get(12..16) != Some(b"VP8X".as_slice()) {
        return None;
    }
    let mut offset = 20 + le32(16)?.next_multiple_of(2);
    let end = offset + le32(4)?.saturating_sub(12);
    while offset < end {
        let size = le32(offset + 4)?;
        if bytes.get(offset..offset + 4) == Some(b"EXIF".as_slice()) {
            return bytes.get(offset + 8..offset + 8 + size);
        }
        offset += 8 + size.next_multiple_of(2);
    }
    None
}

fn ensure_fallback_allowed(
    backend: &str,
    reason: BackendDecline,
    backend_required: bool,
) -> Result<()> {
    if backend_required {
        anyhow::bail!(
            "{backend} was required by {REQUIRE_LIBJPEG_TURBO_TEST_ENV}, but the input would have \
             fallen back to image::ImageReader: {reason}"
        );
    }
    Ok(())
}

fn decode_required_backend(
    backend: &dyn ImageDecodeBackend,
    request: ImageDecodeRequest<'_>,
) -> Result<DecodedImage> {
    match backend.try_decode(request)? {
        ImageDecodeOutcome::Decoded(image) => Ok(image),
        ImageDecodeOutcome::NotHandled(reason) => {
            anyhow::bail!("{} did not handle the input: {reason}", backend.name())
        }
    }
}

fn decoded_image_to_media_data(image: DecodedImage) -> Result<DecodedMediaData> {
    let channels = image.pixel_format.channels();
    let color_type = image.pixel_format.color_type();
    let shape = (image.height as usize, image.width as usize, channels);
    let array = Array3::from_shape_vec(shape, image.pixels)?;
    let mut decoded: DecodedMediaData = array.try_into()?;
    decoded.tensor_info.metadata = Some(DecodedMediaMetadata::Image(ImageMetadata {
        format: Some(image.source_format),
        color_type,
        layout: ImageLayout::HWC,
    }));
    Ok(decoded)
}

impl ImageDecoder {
    #[doc(hidden)]
    pub fn with_libjpeg_for_benchmark(mut self, enabled: bool) -> Self {
        self.enable_libjpeg = enabled;
        self
    }

    pub(crate) fn warn_if_libjpeg_unavailable(&self) {
        if self.enable_libjpeg
            && turbojpeg_backend().availability() == BackendAvailability::Unavailable
        {
            LIBJPEG_TURBO_UNAVAILABLE_WARNING.call_once(|| {
                if std::env::var_os(REQUIRE_LIBJPEG_TURBO_TEST_ENV).is_some() {
                    tracing::warn!(
                        "libjpeg-turbo image decoding is required by \
                         {REQUIRE_LIBJPEG_TURBO_TEST_ENV}, but libturbojpeg could not be loaded; \
                         JPEG requests will fail"
                    );
                } else {
                    tracing::warn!(
                        "libjpeg-turbo image decoding is enabled, but libturbojpeg could not be \
                         loaded; falling back to image::ImageReader for JPEG inputs"
                    );
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::rdma::DataType;
    use super::*;
    use crate::preprocessor::media::jpeg_turbo;
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

    /// EXIF with orientation 6 (rotate 90 degrees clockwise), as a camera stores a portrait photo.
    const EXIF_ROTATE_90: &[u8] =
        b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";

    fn encode_with_exif<E: image::ImageEncoder>(mut encoder: E, image: &image::RgbImage) {
        encoder.set_exif_metadata(EXIF_ROTATE_90.to_vec()).unwrap();
        encoder
            .write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
    }

    #[rstest]
    fn test_exif_orientation_is_applied_only_when_enabled(
        #[values(ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::WebP)] format: ImageFormat,
        #[values(false, true)] apply_exif_orientation: bool,
    ) {
        use dynamo_memory::actions::Slice as _;

        let mut source = image::RgbImage::new(3, 2);
        source.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        let mut bytes = Vec::new();
        match format {
            ImageFormat::Png => {
                encode_with_exif(image::codecs::png::PngEncoder::new(&mut bytes), &source)
            }
            ImageFormat::WebP => encode_with_exif(
                image::codecs::webp::WebPEncoder::new_lossless(&mut bytes),
                &source,
            ),
            _ => encode_with_exif(image::codecs::jpeg::JpegEncoder::new(&mut bytes), &source),
        }

        let decoded = ImageDecoder {
            apply_exif_orientation,
            ..Default::default()
        }
        .decode(create_encoded_media_data(bytes))
        .unwrap();
        // Upright: 2 wide, 3 tall, and the red pixel moved to the top right.
        // Stored: 3 wide, 2 tall, and the red pixel stays at the top left.
        let (shape, first_pixels) = if apply_exif_orientation {
            (vec![3, 2, 3], [0, 0, 0, 255, 0, 0])
        } else {
            (vec![2, 3, 3], [255, 0, 0, 0, 0, 0])
        };
        assert_eq!(decoded.tensor_info.shape, shape);
        if format != ImageFormat::Jpeg {
            // SAFETY: nothing else holds the storage.
            let pixels = unsafe { decoded.data.as_slice() }.unwrap();
            assert_eq!(&pixels[..6], &first_pixels);
        }
    }

    #[test]
    fn test_webp_exif_is_read_where_image_webp_reads_it() {
        let mut stored = Vec::new();
        let source = image::RgbImage::new(3, 2);
        encode_with_exif(
            image::codecs::webp::WebPEncoder::new_lossless(&mut stored),
            &source,
        );
        let exif = stored.windows(4).position(|w| w == b"EXIF").unwrap();
        // The RIFF size ends image-webp's scan just before the EXIF chunk; VP8X no longer flags EXIF.
        let mut past_riff = stored.clone();
        past_riff[4..8].copy_from_slice(&(exif as u32 - 18).to_le_bytes());
        past_riff[20] &= !0x08;
        let mut oversized = stored.clone();
        oversized[exif + 4..exif + 8].copy_from_slice(&(1u32 << 30).to_le_bytes());
        assert_eq!(webp_exif(&stored), Some(EXIF_ROTATE_90));
        assert_eq!(webp_exif(&past_riff), None);
        assert_eq!(webp_exif(&oversized), None);
    }

    #[test]
    fn test_exif_rotation_is_checked_against_limits() {
        let mut bytes = Vec::new();
        let source = image::RgbImage::new(3, 2);
        encode_with_exif(image::codecs::png::PngEncoder::new(&mut bytes), &source);
        let decoder = ImageDecoder {
            limits: ImageDecoderLimits {
                max_image_height: Some(2),
                ..Default::default()
            },
            apply_exif_orientation: true,
            ..Default::default()
        };
        let error = decoder
            .decode(create_encoded_media_data(bytes))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Image dimensions exceed configured limits: 2x3"
        );
    }

    #[rstest]
    #[case(3, ImageFormat::Png, 10, 10, 3, ColorType::Rgb8, "RGB PNG")]
    #[case(4, ImageFormat::Png, 25, 30, 4, ColorType::Rgba8, "RGBA PNG")]
    #[case(1, ImageFormat::Png, 8, 12, 1, ColorType::L8, "Grayscale PNG")]
    #[case(3, ImageFormat::Jpeg, 15, 20, 3, ColorType::Rgb8, "RGB JPEG")]
    #[case(3, ImageFormat::Bmp, 12, 18, 3, ColorType::Rgb8, "RGB BMP")]
    #[case(3, ImageFormat::WebP, 8, 8, 3, ColorType::Rgb8, "RGB WebP")]
    fn test_image_decode(
        #[case] input_channels: u32,
        #[case] format: image::ImageFormat,
        #[case] width: u32,
        #[case] height: u32,
        #[case] expected_channels: u32,
        #[case] expected_color_type: ColorType,
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
        match decoded.tensor_info.metadata {
            Some(DecodedMediaMetadata::Image(metadata)) => {
                assert_eq!(metadata.format, Some(format));
                assert_eq!(metadata.color_type, expected_color_type);
            }
            other => panic!("expected image metadata, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_async_image_decode_precomputes_content_hash() {
        let decoder = ImageDecoder::default();
        let image_bytes = create_test_image(4, 4, 3, image::ImageFormat::Png);
        let encoded_data = create_encoded_media_data(image_bytes);

        let decoded = decoder.decode_async(encoded_data).await.unwrap();

        assert!(decoded.content_hash.is_some());
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
            ..Default::default()
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
    fn test_with_runtime_preserves_server_config() {
        let server_limits = ImageDecoderLimits {
            max_image_width: Some(100),
            max_image_height: Some(100),
            max_alloc: Some(1024),
        };
        let server_config = ImageDecoder {
            limits: server_limits.clone(),
            apply_exif_orientation: true,
            ..Default::default()
        }
        .with_libjpeg_for_benchmark(false);

        // Runtime config tries to override limits (should be ignored)
        let runtime_limits = ImageDecoderLimits {
            max_image_width: Some(9999),
            max_image_height: Some(9999),
            max_alloc: Some(999999),
        };
        let runtime_config = ImageDecoder {
            limits: runtime_limits,
            ..Default::default()
        }
        .with_libjpeg_for_benchmark(true);

        let merged = server_config.with_runtime(Some(&runtime_config));

        // Check that server limits are preserved
        assert_eq!(merged.limits.max_image_width, Some(100));
        assert_eq!(merged.limits.max_image_height, Some(100));
        assert_eq!(merged.limits.max_alloc, Some(1024));
        assert!(!merged.enable_libjpeg);
        assert!(merged.apply_exif_orientation);
    }

    // parse_bool_opt owns the accepted spellings; this locks the default-on
    // behavior and explicit opt-out used by the frontend-local setting.
    #[test]
    fn test_libjpeg_defaults_on_unless_explicitly_disabled() {
        assert!(libjpeg_enabled(None));
        assert!(libjpeg_enabled(Some("")));
        assert!(libjpeg_enabled(Some("invalid")));
        assert!(!libjpeg_enabled(Some("0")));
    }

    #[test]
    fn test_libjpeg_selection_is_not_media_config() {
        let config =
            serde_json::to_value(ImageDecoder::default().with_libjpeg_for_benchmark(false))
                .unwrap();
        assert!(config.get("enable_libjpeg").is_none());

        let decoder: ImageDecoder = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(decoder.enable_libjpeg, default_enable_libjpeg());

        let error =
            serde_json::from_value::<ImageDecoder>(serde_json::json!({"enable_libjpeg": false}))
                .unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn test_required_libjpeg_rejects_fallback() {
        let error = ensure_fallback_allowed(
            turbojpeg_backend().name(),
            BackendDecline::DecodeFailed,
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains(REQUIRE_LIBJPEG_TURBO_TEST_ENV));
        assert!(
            ensure_fallback_allowed(
                turbojpeg_backend().name(),
                BackendDecline::DecodeFailed,
                false,
            )
            .is_ok()
        );
    }

    #[test]
    fn test_libjpeg_enforces_configured_limits() {
        if !jpeg_turbo::available() {
            eprintln!("skipping libjpeg-turbo limit test: libturbojpeg not available");
            return;
        }

        let decoder = ImageDecoder {
            limits: ImageDecoderLimits {
                max_image_width: Some(4),
                max_image_height: None,
                max_alloc: Some(DEFAULT_MAX_ALLOC),
            },
            ..Default::default()
        }
        .with_libjpeg_for_benchmark(true);
        let image_bytes = create_test_image(8, 8, 3, ImageFormat::Jpeg);
        let error = decoder
            .decode(create_encoded_media_data(image_bytes))
            .expect_err("width limit must reject before fallback");
        let error_msg = error.to_string();
        assert!(
            error_msg.contains("dimensions") || error_msg.contains("limit"),
            "Error should mention dimension limits, got: {error_msg}"
        );
    }

    #[test]
    fn test_libjpeg_turbo_pixels_match_pil_vllm_decode_when_available() {
        let require = std::env::var_os(REQUIRE_LIBJPEG_TURBO_TEST_ENV).is_some();
        if !jpeg_turbo::available() {
            if require {
                panic!("{REQUIRE_LIBJPEG_TURBO_TEST_ENV} is set but libturbojpeg is unavailable");
            }
            eprintln!("skipping PIL parity test: libturbojpeg not available");
            return;
        }

        let (jpeg_bytes, expected_rgb, width, height) = pil_parity_fixture();

        let decoded = jpeg_turbo::decode_jpeg(&jpeg_bytes, None, None, Some(DEFAULT_MAX_ALLOC))
            .unwrap()
            .expect("libjpeg-turbo should decode the generated JPEG");

        assert_eq!((decoded.width, decoded.height), (width, height));
        assert_eq!(decoded.channels, 3);
        assert_eq!(decoded.data, expected_rgb);

        let decoder = ImageDecoder::default().with_libjpeg_for_benchmark(true);
        let decoded_media = decoder
            .decode(create_encoded_media_data(jpeg_bytes))
            .unwrap();
        assert_eq!(
            decoded_media.tensor_info.shape,
            vec![height as usize, width as usize, 3]
        );
        match decoded_media.tensor_info.metadata {
            Some(DecodedMediaMetadata::Image(metadata)) => {
                assert_eq!(metadata.format, Some(ImageFormat::Jpeg));
                assert_eq!(metadata.color_type, ColorType::Rgb8);
            }
            other => panic!("expected image metadata, got {other:?}"),
        }
    }

    #[test]
    fn test_libjpeg_turbo_declines_cmyk_before_output_allocation() {
        let require = std::env::var_os(REQUIRE_LIBJPEG_TURBO_TEST_ENV).is_some();
        if !jpeg_turbo::available() {
            if require {
                panic!("{REQUIRE_LIBJPEG_TURBO_TEST_ENV} is set but libturbojpeg is unavailable");
            }
            eprintln!("skipping CMYK JPEG test: libturbojpeg not available");
            return;
        }

        let jpeg_bytes = cmyk_jpeg_fixture();
        let decoded = jpeg_turbo::decode_jpeg(&jpeg_bytes, None, None, Some(0))
            .expect("CMYK JPEG should decline before applying the RGB output allocation limit");
        assert!(decoded.is_none());
    }

    #[test]
    fn test_libjpeg_turbo_gray_jpeg_shape() {
        let require = std::env::var_os(REQUIRE_LIBJPEG_TURBO_TEST_ENV).is_some();
        if !jpeg_turbo::available() {
            if require {
                panic!("{REQUIRE_LIBJPEG_TURBO_TEST_ENV} is set but libturbojpeg is unavailable");
            }
            eprintln!("skipping grayscale JPEG test: libturbojpeg not available");
            return;
        }

        let decoder = ImageDecoder::default().with_libjpeg_for_benchmark(true);
        let image_bytes = create_test_image(8, 9, 1, ImageFormat::Jpeg);
        let decoded = decoder
            .decode(create_encoded_media_data(image_bytes))
            .unwrap();
        assert_eq!(decoded.tensor_info.shape, vec![9, 8, 1]);
        match decoded.tensor_info.metadata {
            Some(DecodedMediaMetadata::Image(metadata)) => {
                assert_eq!(metadata.format, Some(ImageFormat::Jpeg));
                assert_eq!(metadata.color_type, ColorType::L8);
            }
            other => panic!("expected image metadata, got {other:?}"),
        }
    }

    /// Encoded 16-bit PNGs decode to the pixels that Pillow's `convert("RGB")` gives. The PNGs
    /// come from pypng; the expected bytes from Pillow 11.3.0 `Image.open(png).convert("RGB")`.
    #[rstest]
    #[case::grey(
        "iVBORw0KGgoAAAANSUhEUgAAAAUAAAABEAAAAABjBedsAAAAE0lEQVR4nGNgYGD4z8hQavD/PwAMawOk1Qn3MgAAAABJRU5ErkJggg==",
        1,
        &[0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
    )]
    #[case::grey_with_trns_key(
        "iVBORw0KGgoAAAANSUhEUgAAAAMAAAABEAAAAABuG5crAAAAAnRSTlMAyOMsh7oAAAAPSURBVHicYyg1YDjx/z8ACQoDbMNVpCIAAAAASUVORK5CYII=",
        1,
        &[255, 255, 255, 200, 200, 200, 255, 255, 255]
    )]
    #[case::grey_alpha(
        "iVBORw0KGgoAAAANSUhEUgAAAAMAAAABEAQAAADheQB8AAAAFUlEQVR4nGMoNWBgYDjx/z8jQwMDABkKA+2nh8FBAAAAAElFTkSuQmCC",
        2,
        &[117, 117, 117, 0, 0, 0, 1, 1, 1]
    )]
    #[case::rgb(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABEAIAAAAr0DSeAAAAFUlEQVR4nGNg/P+fgeH/q4T6/wwNACjSBkc6FpyfAAAAAElFTkSuQmCC",
        3,
        &[1, 255, 0, 234, 127, 0]
    )]
    #[case::rgba(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABEAYAAACksqPJAAAAGklEQVR4nGNg/P+fgeF/A8OrhPr/DA3//wMAQ1sIxQ74P4cAAAAASUVORK5CYII=",
        4,
        &[1, 255, 0, 234, 127, 0]
    )]
    fn test_sixteen_bit_png_matches_pillow_rgb(
        #[case] png_b64: &str,
        #[case] channels: usize,
        #[case] pillow_rgb: &[u8],
    ) {
        use base64::{Engine as _, engine::general_purpose};
        use dynamo_memory::actions::Slice as _;

        let png = general_purpose::STANDARD.decode(png_b64).unwrap();
        let decoded = ImageDecoder::default()
            .decode(create_encoded_media_data(png))
            .unwrap();
        assert_eq!(decoded.tensor_info.shape[2], channels);
        // SAFETY: nothing else holds the storage.
        let pixels = unsafe { decoded.data.as_slice() }.unwrap();
        // Pillow's convert("RGB") repeats a grey sample and drops alpha.
        let rgb: Vec<u8> = pixels
            .chunks(channels)
            .flat_map(|pixel| match channels {
                1 | 2 => [pixel[0]; 3],
                _ => [pixel[0], pixel[1], pixel[2]],
            })
            .collect();
        assert_eq!(rgb, pillow_rgb);
    }

    fn pil_parity_fixture() -> (Vec<u8>, Vec<u8>, u32, u32) {
        use base64::{Engine as _, engine::general_purpose};

        const WIDTH: u32 = 17;
        const HEIGHT: u32 = 11;
        // JPEG generated from a deterministic RGB gradient with Pillow
        // quality=87/subsampling=2; RGB fixture is Pillow convert("RGB").
        const JPEG_B64: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/media/pil_parity_17x11.jpg.b64"
        ));
        const RGB_B64: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/media/pil_parity_17x11.rgb.b64"
        ));

        let jpeg_bytes = general_purpose::STANDARD.decode(JPEG_B64.trim()).unwrap();
        let expected_rgb = general_purpose::STANDARD.decode(RGB_B64.trim()).unwrap();
        assert_eq!(expected_rgb.len(), (WIDTH * HEIGHT * 3) as usize);

        (jpeg_bytes, expected_rgb, WIDTH, HEIGHT)
    }

    fn cmyk_jpeg_fixture() -> Vec<u8> {
        use base64::{Engine as _, engine::general_purpose};

        // Pillow-generated 2x2 CMYK JPEG; TurboJPEG reports TJCS_CMYK.
        const JPEG_B64: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/media/cmyk_2x2.jpg.b64"
        ));
        general_purpose::STANDARD.decode(JPEG_B64.trim()).unwrap()
    }
}
