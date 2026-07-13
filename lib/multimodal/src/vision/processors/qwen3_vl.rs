// SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Qwen3-VL family image processors.
//!
//! This module provides the Qwen3-VL processor which wraps the shared
//! `QwenVLProcessorBase` with Qwen3-VL specific default parameters.
//!
//! # Key Differences from Qwen2-VL
//!
//! - **Patch Size**: 16 (vs 14 in Qwen2-VL)
//! - **Factor**: 32 (patch_size * merge_size) (vs 28 in Qwen2-VL)
//! - **Normalization**: [0.5, 0.5, 0.5] mean/std (vs CLIP in Qwen2-VL)
//!
//! # Qwen3-VL Parameters
//!
//! - patch_size: 16
//! - merge_size: 2
//! - factor: 32 (patch_size * merge_size)
//! - normalization: [0.5, 0.5, 0.5] mean/std

use std::ops::Deref;

use image::DynamicImage;

use super::qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase};
use crate::{
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::TransformError,
    },
};

/// Qwen3-VL normalization mean values (simple [0.5, 0.5, 0.5]).
pub const QWEN3_MEAN: [f64; 3] = [0.5, 0.5, 0.5];

/// Qwen3-VL normalization std values (simple [0.5, 0.5, 0.5]).
pub const QWEN3_STD: [f64; 3] = [0.5, 0.5, 0.5];

/// Default minimum pixels for Qwen3-VL
/// This corresponds to shortest_edge = 65536 from HF config
pub const DEFAULT_MIN_PIXELS: usize = 65536;

/// Default maximum pixels for Qwen3-VL
/// This corresponds to longest_edge = 16777216 from HF config
pub const DEFAULT_MAX_PIXELS: usize = 16777216;

/// Default patch size for Qwen3-VL (16, vs 14 in Qwen2-VL)
pub const DEFAULT_PATCH_SIZE: usize = 16;

/// Default merge size for token reduction
pub const DEFAULT_MERGE_SIZE: usize = 2;

/// Default temporal patch size (for video frames)
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;

/// Qwen3-VL image processor.
///
/// This is a thin wrapper around `QwenVLProcessorBase` with Qwen3-VL
/// specific default parameters:
/// - patch_size: 16
/// - merge_size: 2
/// - [0.5, 0.5, 0.5] normalization mean/std
#[derive(Debug, Clone)]
pub struct Qwen3VLProcessor {
    inner: QwenVLProcessorBase,
}

impl Default for Qwen3VLProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen3VLProcessor {
    /// Create a new Qwen3-VL processor with default settings.
    ///
    /// Defaults:
    /// - patch_size: 16
    /// - merge_size: 2
    /// - min_pixels: 65,536
    /// - max_pixels: 16,777,216
    /// - temporal_patch_size: 2
    /// - normalization: [0.5, 0.5, 0.5] mean/std
    pub fn new() -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size: DEFAULT_PATCH_SIZE,
                merge_size: DEFAULT_MERGE_SIZE,
                min_pixels: DEFAULT_MIN_PIXELS,
                max_pixels: DEFAULT_MAX_PIXELS,
                temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
                mean: QWEN3_MEAN,
                std: QWEN3_STD,
                model_name: "qwen3-vl",
            }),
        }
    }

    /// Create a processor with custom settings.
    pub fn with_config(
        patch_size: usize,
        merge_size: usize,
        min_pixels: usize,
        max_pixels: usize,
        temporal_patch_size: usize,
    ) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels,
                max_pixels,
                temporal_patch_size,
                mean: QWEN3_MEAN,
                std: QWEN3_STD,
                model_name: "qwen3-vl",
            }),
        }
    }

    /// Create a processor from preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE),
                merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
                min_pixels: config
                    .min_pixels
                    .or_else(|| config.get_shortest_edge())
                    .unwrap_or(DEFAULT_MIN_PIXELS),
                max_pixels: config
                    .max_pixels
                    .or_else(|| config.get_longest_edge())
                    .unwrap_or(DEFAULT_MAX_PIXELS),
                temporal_patch_size: config
                    .temporal_patch_size
                    .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
                mean: QWEN3_MEAN,
                std: QWEN3_STD,
                model_name: "qwen3-vl",
            }),
        }
    }

    fn with_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        if config.patch_size.is_some()
            || config.merge_size.is_some()
            || config.min_pixels.is_some()
            || config.max_pixels.is_some()
            || config.temporal_patch_size.is_some()
            || config.size.is_some()
        {
            Self::from_preprocessor_config(config)
        } else {
            self.clone()
        }
    }

    /// Preprocess RGB video using this instance's validated structural
    /// parameters and the supplied per-pixel transform configuration.
    ///
    /// Callers that build and retain a processor should use this method. The
    /// trait method also accepts structural overrides for legacy one-shot
    /// callers and therefore may construct a temporary processor.
    pub fn preprocess_configured_video_rgb(
        &self,
        frames: &[crate::types::RgbFrameRef<'_>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.inner.preprocess_video_rgb(frames, config)
    }

    /// Get the patch size.
    pub fn patch_size(&self) -> usize {
        self.inner.patch_size()
    }

    /// Get the merge size.
    pub fn merge_size(&self) -> usize {
        self.inner.merge_size()
    }

    /// Get the minimum pixels.
    pub fn min_pixels(&self) -> usize {
        self.inner.min_pixels()
    }

    /// Get the maximum pixels.
    pub fn max_pixels(&self) -> usize {
        self.inner.max_pixels()
    }

    /// Get the temporal patch size.
    pub fn temporal_patch_size(&self) -> usize {
        self.inner.temporal_patch_size()
    }

    /// Get the factor for dimension alignment.
    #[inline]
    pub fn get_factor(&self) -> usize {
        self.inner.get_factor()
    }

    /// Smart resize algorithm for Qwen3-VL.
    pub fn smart_resize(
        &self,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        self.inner.smart_resize(height, width)
    }

    /// Calculate the grid dimensions (T, H, W) for an image.
    pub fn calculate_grid_thw(
        &self,
        height: usize,
        width: usize,
        num_frames: usize,
    ) -> (usize, usize, usize) {
        self.inner.calculate_grid_thw(height, width, num_frames)
    }

    /// Calculate the number of image tokens after merge.
    pub fn calculate_tokens_from_grid(&self, grid_t: usize, grid_h: usize, grid_w: usize) -> usize {
        self.inner
            .calculate_tokens_from_grid(grid_t, grid_h, grid_w)
    }
}

impl Deref for Qwen3VLProcessor {
    type Target = QwenVLProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl VisionPreProcessor for Qwen3VLProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.inner.default_mean()
    }

    fn default_std(&self) -> [f64; 3] {
        self.inner.default_std()
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let processor = self.with_preprocessor_config(config);
        processor.inner.preprocess(images, config)
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let processor = self.with_preprocessor_config(config);
        processor.inner.preprocess_video(frames, config)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let processor = self.with_preprocessor_config(config);
        processor.inner.preprocess_video_rgb(frames, config)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        let processor = self.with_preprocessor_config(config);
        processor.inner.calculate_num_tokens(width, height, config)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        self.inner.get_processed_size(config)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use image::{Rgb, RgbImage};

    use super::*;
    use crate::vision::{preprocessor_config::PatchSize, processor::ModelSpecificValue};

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_qwen3_vl_processor_default() {
        let processor = Qwen3VLProcessor::new();
        assert_eq!(processor.patch_size(), 16);
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.min_pixels(), DEFAULT_MIN_PIXELS);
        assert_eq!(processor.max_pixels(), DEFAULT_MAX_PIXELS);
        assert_eq!(processor.get_factor(), 32); // 16 * 2
    }

    #[test]
    fn test_smart_resize_within_bounds() {
        let processor = Qwen3VLProcessor::new();

        // Image that's within bounds
        let (h, w) = processor.smart_resize(500, 500).unwrap();

        // Should be aligned to factor (32)
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);

        // Should be within bounds
        assert!(h * w >= processor.min_pixels());
        assert!(h * w <= processor.max_pixels());
    }

    #[test]
    fn test_smart_resize_aspect_ratio_preserved() {
        let processor = Qwen3VLProcessor::new();

        // 2:1 aspect ratio
        let (h, w) = processor.smart_resize(400, 800).unwrap();

        // Aspect ratio should be approximately preserved
        let original_ratio = 800.0 / 400.0;
        let new_ratio = w as f64 / h as f64;
        assert!((new_ratio - original_ratio).abs() < 0.5);
    }

    #[test]
    fn test_smart_resize_extreme_aspect_ratio_error() {
        let processor = Qwen3VLProcessor::new();

        // 300:1 aspect ratio - should fail
        let result = processor.smart_resize(100, 30000);
        assert!(result.is_err());
    }

    #[test]
    fn test_smart_resize_small_dimension_clamps_to_factor() {
        let processor = Qwen3VLProcessor::new();

        // Dimension smaller than factor (32) should be clamped up, not rejected
        let (h, w) = processor.smart_resize(10, 100).unwrap();
        assert!(h >= 32);
        assert!(w >= 32);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn test_calculate_grid_thw_image() {
        let processor = Qwen3VLProcessor::new();

        // 480x640 image
        let (t, h, w) = processor.calculate_grid_thw(480, 640, 1);

        assert_eq!(t, 1); // Single image
        assert_eq!(h, 480 / 16); // 30
        assert_eq!(w, 640 / 16); // 40
    }

    #[test]
    fn test_calculate_tokens() {
        let processor = Qwen3VLProcessor::new();

        // With merge_size=2, tokens = (t * h * w) / 4
        let tokens = processor.calculate_tokens_from_grid(1, 30, 40);
        assert_eq!(tokens, (30 * 40) / 4); // 300
    }

    #[test]
    fn test_qwen3_vl_preprocess() {
        let processor = Qwen3VLProcessor::new();
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_normalize: Some(true),
            image_mean: Some(QWEN3_MEAN.to_vec()),
            image_std: Some(QWEN3_STD.to_vec()),
            patch_size: Some(PatchSize {
                height: Some(16),
                width: Some(16),
            }),
            merge_size: Some(2),
            min_pixels: Some(DEFAULT_MIN_PIXELS),
            max_pixels: Some(DEFAULT_MAX_PIXELS),
            ..Default::default()
        };

        let image = create_test_image(640, 480, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        // encoder_input is patchified: [total_patches, patch_features]
        assert_eq!(result.encoder_input.ndim(), 2);
        assert!(result.encoder_input.shape()[0] > 0); // total_patches > 0

        // Check pixel values are normalized
        let flat = result.encoder_input_flat();
        // After normalization with [0.5, 0.5, 0.5] mean/std:
        // (0.5 - 0.5) / 0.5 = 0.0 for gray
        // Values should be in [-1, 1] range
        assert!(flat.iter().all(|&v| (-1.5..=1.5).contains(&v)));

        // Check image_grid_thw and patches_per_image are present
        assert!(result.model_specific.contains_key("image_grid_thw"));
        assert!(result.model_specific.contains_key("patches_per_image"));

        // Verify token count is reasonable
        assert!(result.feature_token_counts[0] > 0);
    }

    #[test]
    fn test_qwen3_vl_preprocess_multiple() {
        let processor = Qwen3VLProcessor::new();
        let config = PreProcessorConfig {
            image_mean: Some(QWEN3_MEAN.to_vec()),
            image_std: Some(QWEN3_STD.to_vec()),
            ..Default::default()
        };

        let images = vec![
            create_test_image(640, 480, Rgb([100, 100, 100])),
            create_test_image(480, 640, Rgb([150, 150, 150])),
        ];

        let result = processor.preprocess(&images, &config).unwrap();

        // Both images processed
        assert_eq!(result.item_sizes.len(), 2);
        assert_eq!(result.feature_token_counts.len(), 2);

        // encoder_input is 2D [total_patches, patch_features]
        assert_eq!(result.encoder_input.ndim(), 2);

        // Check grid_thw shape
        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("image_grid_thw")
        {
            assert_eq!(shape, &[2, 3]); // 2 images, 3 values (T, H, W) each
            assert_eq!(data.len(), 6);
        } else {
            panic!("Expected image_grid_thw to be IntTensor");
        }

        // Check patches_per_image
        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("patches_per_image")
        {
            assert_eq!(shape, &[2]); // 2 images
            assert_eq!(data.len(), 2);
            // Total patches should match encoder_input first dim
            let total: i64 = data.iter().sum();
            assert_eq!(total as usize, result.encoder_input.shape()[0]);
        } else {
            panic!("Expected patches_per_image to be IntTensor");
        }
    }

    #[test]
    fn test_qwen3_vl_preprocess_video() {
        let processor = Qwen3VLProcessor::new();
        let config = PreProcessorConfig {
            image_mean: Some(QWEN3_MEAN.to_vec()),
            image_std: Some(QWEN3_STD.to_vec()),
            ..Default::default()
        };

        let frames = vec![
            create_test_image(640, 480, Rgb([100, 100, 100])),
            create_test_image(640, 480, Rgb([150, 150, 150])),
            create_test_image(640, 480, Rgb([200, 200, 200])),
        ];

        let result = processor.preprocess_video(&frames, &config).unwrap();
        assert_eq!(result.encoder_input.ndim(), 2);
        assert_eq!(result.feature_token_counts.len(), 1);
        assert!(result.model_specific.contains_key("video_grid_thw"));

        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("video_grid_thw")
        {
            assert_eq!(shape, &[1, 3]);
            assert_eq!(data[0], 2); // 3 frames padded to 4, temporal_patch_size=2
        } else {
            panic!("Expected video_grid_thw to be IntTensor");
        }
    }

    #[test]
    fn test_qwen3_vl_preprocess_video_rgb_applies_config() {
        let processor = Qwen3VLProcessor::new();
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(8),
                width: Some(8),
            }),
            merge_size: Some(1),
            temporal_patch_size: Some(4),
            min_pixels: Some(1),
            max_pixels: Some(4096),
            ..Default::default()
        };

        let frames = vec![
            create_test_image(32, 32, Rgb([100, 100, 100])),
            create_test_image(32, 32, Rgb([150, 150, 150])),
            create_test_image(32, 32, Rgb([200, 200, 200])),
        ];
        let rgb_images: Vec<RgbImage> = frames.iter().map(|frame| frame.to_rgb8()).collect();
        let rgb_frames: Vec<RgbFrameRef<'_>> = rgb_images
            .iter()
            .map(|frame| RgbFrameRef {
                width: frame.width(),
                height: frame.height(),
                data: frame.as_raw(),
            })
            .collect();

        let dynamic = processor.preprocess_video(&frames, &config).unwrap();
        let rgb = processor
            .preprocess_video_rgb(&rgb_frames, &config)
            .unwrap();

        assert_eq!(rgb.encoder_input.shape(), dynamic.encoder_input.shape());
        assert_eq!(rgb.feature_token_counts, dynamic.feature_token_counts);
        let Some(ModelSpecificValue::IntTensor {
            data: rgb_grid,
            shape: rgb_shape,
        }) = rgb.model_specific.get("video_grid_thw")
        else {
            panic!("Expected RGB video_grid_thw to be IntTensor");
        };
        let Some(ModelSpecificValue::IntTensor {
            data: dynamic_grid,
            shape: dynamic_shape,
        }) = dynamic.model_specific.get("video_grid_thw")
        else {
            panic!("Expected dynamic video_grid_thw to be IntTensor");
        };
        assert_eq!(rgb_shape, dynamic_shape);
        assert_eq!(rgb_grid, dynamic_grid);
    }

    #[test]
    fn test_qwen3_vl_from_config() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(16),
                width: Some(16),
            }),
            merge_size: Some(4),
            min_pixels: Some(100000),
            max_pixels: Some(500000),
            temporal_patch_size: Some(4),
            ..Default::default()
        };

        let processor = Qwen3VLProcessor::from_preprocessor_config(&config);

        assert_eq!(processor.patch_size(), 16);
        assert_eq!(processor.merge_size(), 4);
        assert_eq!(processor.min_pixels(), 100000);
        assert_eq!(processor.max_pixels(), 500000);
        assert_eq!(processor.temporal_patch_size(), 4);
    }

    #[test]
    fn test_qwen3_vl_video_config_uses_size_edges() {
        let config = PreProcessorConfig {
            size: Some(HashMap::from([
                ("shortest_edge".to_string(), 4096),
                ("longest_edge".to_string(), 25165824),
            ])),
            ..Default::default()
        };

        let processor = Qwen3VLProcessor::from_preprocessor_config(&config);

        assert_eq!(processor.min_pixels(), 4096);
        assert_eq!(processor.max_pixels(), 25165824);
        assert_eq!(
            processor.inner.smart_resize_video(239, 720, 1280).unwrap(),
            (224, 416)
        );
    }

    #[test]
    fn test_model_name() {
        let processor = Qwen3VLProcessor::new();
        assert_eq!(processor.model_name(), "qwen3-vl");
    }

    #[test]
    fn test_default_mean_std() {
        let processor = Qwen3VLProcessor::new();
        assert_eq!(processor.default_mean(), QWEN3_MEAN);
        assert_eq!(processor.default_std(), QWEN3_STD);
    }

    #[test]
    fn test_qwen3_vs_qwen2_differences() {
        // Verify the key differences from Qwen2-VL
        let processor = Qwen3VLProcessor::new();

        // Qwen3-VL uses patch_size=16 (vs 14 in Qwen2)
        assert_eq!(processor.patch_size(), 16);

        // Factor is 32 (vs 28 in Qwen2)
        assert_eq!(processor.get_factor(), 32);

        // Mean/std are [0.5, 0.5, 0.5] (vs CLIP values in Qwen2)
        assert_eq!(processor.default_mean(), [0.5, 0.5, 0.5]);
        assert_eq!(processor.default_std(), [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_smart_resize_grayscale_400x300() {
        // grayscale.jpg is 400x300
        // 400/32 = 12.5 -> rounds to 12 (banker's rounding) -> 384
        // 300/32 = 9.375 -> rounds to 9 -> 288
        // Expected: 384x288, giving grid [1, 18, 24]
        let processor = Qwen3VLProcessor::new();

        // smart_resize takes (height, width)
        let (h, w) = processor.smart_resize(300, 400).unwrap();

        // Expected from HuggingFace: 288x384 -> grid [1, 18, 24]
        assert_eq!(h, 288, "Height should be 288");
        assert_eq!(w, 384, "Width should be 384");
    }
}
