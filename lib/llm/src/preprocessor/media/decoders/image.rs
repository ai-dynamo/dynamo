// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{io::Cursor, sync::Once};

use anyhow::Result;
use image::{ColorType, GenericImageView, ImageFormat, ImageReader};
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::super::common::EncodedMediaData;
use super::super::jpeg_turbo;
use super::super::rdma::DecodedMediaData;
use super::{DecodedMediaMetadata, Decoder};

const DEFAULT_MAX_ALLOC: u64 = 128 * 1024 * 1024; // 128 MB
// CI-only guard: an enabled JPEG test must exercise TurboJPEG, never its fallback.
const REQUIRE_LIBJPEG_TURBO_TEST_ENV: &str = "DYNAMO_REQUIRE_LIBJPEG_TURBO_TEST";
static LIBJPEG_TURBO_UNAVAILABLE_WARNING: Once = Once::new();

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

#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageDecoder {
    #[serde(default)]
    pub(crate) limits: ImageDecoderLimits,
    /// Enable libjpeg-turbo for JPEG inputs. Defaults to `false`.
    #[serde(default)]
    pub(crate) enable_libjpeg: bool,
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
                d
            }
            None => self.clone(),
        }
    }

    fn decode(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        let bytes = data.into_bytes()?;
        self.warn_if_libjpeg_unavailable();

        if self.enable_libjpeg && jpeg_turbo::is_jpeg(&bytes) {
            let jpeg = require_libjpeg_result(
                jpeg_turbo::decode_jpeg(
                    &bytes,
                    self.limits.max_image_width,
                    self.limits.max_image_height,
                    self.limits.max_alloc,
                )?,
                std::env::var_os(REQUIRE_LIBJPEG_TURBO_TEST_ENV).is_some(),
            )?;
            if let Some(jpeg) = jpeg {
                let color_type = match jpeg.channels {
                    1 => ColorType::L8,
                    3 => ColorType::Rgb8,
                    other => anyhow::bail!("Unsupported TurboJPEG channel count {other}"),
                };
                let shape = (jpeg.height as usize, jpeg.width as usize, jpeg.channels);
                let array = Array3::from_shape_vec(shape, jpeg.data)?;
                let mut decoded: DecodedMediaData = array.try_into()?;
                decoded.tensor_info.metadata = Some(DecodedMediaMetadata::Image(ImageMetadata {
                    format: Some(ImageFormat::Jpeg),
                    color_type,
                    layout: ImageLayout::HWC,
                }));
                return Ok(decoded);
            }
        }

        let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
        let mut limits = image::Limits::no_limits();
        limits.max_image_width = self.limits.max_image_width;
        limits.max_image_height = self.limits.max_image_height;
        limits.max_alloc = self.limits.max_alloc;
        reader.limits(limits);

        let format = reader.format();

        let img = reader.decode()?;
        let n_channels = img.color().channel_count();

        let (data, color_type) = match n_channels {
            1 => (img.to_luma8().into_raw(), ColorType::L8),
            2 => (img.to_luma_alpha8().into_raw(), ColorType::La8),
            3 => (img.to_rgb8().into_raw(), ColorType::Rgb8),
            4 => (img.to_rgba8().into_raw(), ColorType::Rgba8),
            other => anyhow::bail!("Unsupported channel count {other}"),
        };

        let (width, height) = img.dimensions();
        let shape = (height as usize, width as usize, n_channels as usize);
        let array = Array3::from_shape_vec(shape, data)?;
        let mut decoded: DecodedMediaData = array.try_into()?;
        decoded.tensor_info.metadata = Some(DecodedMediaMetadata::Image(ImageMetadata {
            format,
            color_type,
            layout: ImageLayout::HWC,
        }));
        Ok(decoded)
    }
}

fn require_libjpeg_result<T>(decoded: Option<T>, required: bool) -> Result<Option<T>> {
    if required && decoded.is_none() {
        anyhow::bail!(
            "libjpeg-turbo was required by {REQUIRE_LIBJPEG_TURBO_TEST_ENV}, but the JPEG would \
             have fallen back to image::ImageReader"
        );
    }
    Ok(decoded)
}

impl ImageDecoder {
    pub(crate) fn warn_if_libjpeg_unavailable(&self) {
        if self.enable_libjpeg && !jpeg_turbo::available() {
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
            decoded.tensor_info.shape,
            vec![height as usize, width as usize, expected_channels as usize]
        );
        assert_eq!(decoded.tensor_info.dtype, DataType::UINT8);
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
            enable_libjpeg: false,
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
    fn test_with_runtime_limit_enforcement() {
        let server_limits = ImageDecoderLimits {
            max_image_width: Some(100),
            max_image_height: Some(100),
            max_alloc: Some(1024),
        };
        let server_config = ImageDecoder {
            limits: server_limits.clone(),
            enable_libjpeg: false,
        };

        // Runtime config tries to override limits (should be ignored)
        let runtime_limits = ImageDecoderLimits {
            max_image_width: Some(9999),
            max_image_height: Some(9999),
            max_alloc: Some(999999),
        };
        let runtime_config = ImageDecoder {
            limits: runtime_limits,
            enable_libjpeg: true,
        };

        let merged = server_config.with_runtime(Some(&runtime_config));

        // Check that server limits are preserved
        assert_eq!(merged.limits.max_image_width, Some(100));
        assert_eq!(merged.limits.max_image_height, Some(100));
        assert_eq!(merged.limits.max_alloc, Some(1024));
        assert!(merged.enable_libjpeg);
    }

    #[test]
    fn test_libjpeg_is_disabled_by_default() {
        let decoder = ImageDecoder::default();
        assert!(!decoder.enable_libjpeg);

        let decoder: ImageDecoder = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(!decoder.enable_libjpeg);
    }

    #[test]
    fn test_config_can_enable_libjpeg() {
        let decoder: ImageDecoder =
            serde_json::from_value(serde_json::json!({"enable_libjpeg": true})).unwrap();
        assert!(decoder.enable_libjpeg);
    }

    #[test]
    fn test_required_libjpeg_rejects_fallback() {
        let error = require_libjpeg_result::<()>(None, true).unwrap_err();
        assert!(error.to_string().contains(REQUIRE_LIBJPEG_TURBO_TEST_ENV));

        assert!(require_libjpeg_result::<()>(None, false).unwrap().is_none());
        assert_eq!(require_libjpeg_result(Some(1_u8), true).unwrap(), Some(1));
    }

    #[test]
    fn test_libjpeg_falls_back_for_non_jpeg() {
        let decoder = ImageDecoder {
            enable_libjpeg: true,
            ..Default::default()
        };
        let image_bytes = create_test_image(8, 9, 3, ImageFormat::Png);
        let decoded = decoder
            .decode(create_encoded_media_data(image_bytes))
            .unwrap();
        assert_eq!(decoded.tensor_info.shape, vec![9, 8, 3]);
        match decoded.tensor_info.metadata {
            Some(DecodedMediaMetadata::Image(metadata)) => {
                assert_eq!(metadata.format, Some(ImageFormat::Png));
                assert_eq!(metadata.color_type, ColorType::Rgb8);
            }
            other => panic!("expected image metadata, got {other:?}"),
        }
    }

    #[test]
    fn test_libjpeg_falls_back_for_rgba_png() {
        let decoder = ImageDecoder {
            enable_libjpeg: true,
            ..Default::default()
        };
        let image_bytes = create_test_image(8, 9, 4, ImageFormat::Png);
        let decoded = decoder
            .decode(create_encoded_media_data(image_bytes))
            .unwrap();
        assert_eq!(decoded.tensor_info.shape, vec![9, 8, 4]);
        match decoded.tensor_info.metadata {
            Some(DecodedMediaMetadata::Image(metadata)) => {
                assert_eq!(metadata.format, Some(ImageFormat::Png));
                assert_eq!(metadata.color_type, ColorType::Rgba8);
            }
            other => panic!("expected image metadata, got {other:?}"),
        }
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
            enable_libjpeg: true,
        };
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

        let decoder = ImageDecoder {
            enable_libjpeg: true,
            ..Default::default()
        };
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
    fn test_libjpeg_turbo_gray_jpeg_shape() {
        let require = std::env::var_os(REQUIRE_LIBJPEG_TURBO_TEST_ENV).is_some();
        if !jpeg_turbo::available() {
            if require {
                panic!("{REQUIRE_LIBJPEG_TURBO_TEST_ENV} is set but libturbojpeg is unavailable");
            }
            eprintln!("skipping grayscale JPEG test: libturbojpeg not available");
            return;
        }

        let decoder = ImageDecoder {
            enable_libjpeg: true,
            ..Default::default()
        };
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
}
