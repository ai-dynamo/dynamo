//! Inkling image processor.
//!
//! Implements `InklingImageProcessor`: optional aspect-preserving
//! resize, RGB HWC input, CLIP normalization, and one extra patch column on
//! every row. The output tensor is
//! `[sum_patches, temporal_patch_size, patch_size, patch_size, 3]`.

use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::{ArrayD, IxDyn};

use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::{self, TransformError},
};

pub const INKLING_IMAGE_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const INKLING_IMAGE_STD: [f64; 3] = [0.26862954, 0.2613026, 0.2757771];
pub const DEFAULT_PATCH_SIZE: usize = 40;
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;
pub const DEFAULT_RESCALE_IMAGE_MAX_UPSCALED_LONG_EDGE: u32 = 2048;
const PAD_RAW_VALUE: f32 = -1.0 / 255.0;

#[derive(Debug, Clone)]
pub struct InklingImageProcessor {
    patch_size: usize,
    temporal_patch_size: usize,
    rescale_image_frac: Option<f64>,
    rescale_image_max_upscaled_long_edge: Option<u32>,
}

impl Default for InklingImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl InklingImageProcessor {
    pub fn new() -> Self {
        Self {
            patch_size: DEFAULT_PATCH_SIZE,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
            // Count patches at the authored image dimensions. Resizing remains
            // an explicit preprocessor override.
            rescale_image_frac: None,
            rescale_image_max_upscaled_long_edge: Some(
                DEFAULT_RESCALE_IMAGE_MAX_UPSCALED_LONG_EDGE,
            ),
        }
    }

    fn with_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        let mut processor = self.clone();
        if config.patch_size.is_some() {
            processor.patch_size = config.get_patch_size(self.patch_size);
        }
        if let Some(temporal_patch_size) = config.temporal_patch_size {
            processor.temporal_patch_size = temporal_patch_size;
        }
        if let Some(value) = config.extra.get("rescale_image_frac") {
            processor.rescale_image_frac = value.as_f64();
        }
        if let Some(value) = config.extra.get("rescale_image_max_upscaled_long_edge") {
            processor.rescale_image_max_upscaled_long_edge =
                value.as_u64().and_then(|v| u32::try_from(v).ok());
        }
        processor
    }

    fn patch_grid(&self, width: usize, height: usize) -> (usize, usize, usize) {
        let nph = height.div_ceil(self.patch_size);
        let npw = width / self.patch_size + 1;
        let num_patches = nph * npw;
        (nph, npw, num_patches)
    }

    fn num_patch_features(&self) -> usize {
        self.temporal_patch_size * self.patch_size * self.patch_size * 3
    }

    fn validate(&self) -> Result<(), TransformError> {
        if self.patch_size == 0 || self.temporal_patch_size == 0 {
            return Err(TransformError::ShapeError(
                "Inkling patch_size and temporal_patch_size must be positive".to_string(),
            ));
        }
        if let Some(frac) = self.rescale_image_frac {
            if !frac.is_finite() || frac <= 0.0 {
                return Err(TransformError::ShapeError(format!(
                    "Inkling rescale_image_frac must be positive and finite, got {frac}"
                )));
            }
        }
        if matches!(self.rescale_image_max_upscaled_long_edge, Some(0)) {
            return Err(TransformError::ShapeError(
                "Inkling rescale_image_max_upscaled_long_edge must be positive".to_string(),
            ));
        }
        Ok(())
    }

    fn scaled_image_dimensions(&self, width: u32, height: u32) -> (u32, u32) {
        let Some(frac) = self.rescale_image_frac else {
            return (width, height);
        };
        let long_edge = width.max(height);
        if long_edge == 0 {
            return (width, height);
        }

        let mut target_long_edge = long_edge as f64 * frac;
        if let Some(max_upscaled_long_edge) = self.rescale_image_max_upscaled_long_edge {
            let effective_cap = max_upscaled_long_edge.max(long_edge);
            target_long_edge = target_long_edge.min(effective_cap as f64);
        }
        let ratio = target_long_edge / long_edge as f64;
        if (ratio - 1.0).abs() < f64::EPSILON {
            return (width, height);
        }

        let scale_dim = |dim: u32| -> u32 { ((dim as f64 * ratio + 0.5).floor() as u32).max(1) };
        (scale_dim(width), scale_dim(height))
    }

    fn prepare_rgb_image(&self, image: &DynamicImage) -> RgbImage {
        let (width, height) = image.dimensions();
        let (scaled_width, scaled_height) = self.scaled_image_dimensions(width, height);
        if (scaled_width, scaled_height) == (width, height) {
            image.to_rgb8()
        } else {
            transforms::resize_lanczos_pil(image, scaled_width, scaled_height).to_rgb8()
        }
    }

    fn append_image_patches(
        &self,
        image: &DynamicImage,
        mean: &[f64; 3],
        std: &[f64; 3],
        output: &mut Vec<f32>,
    ) {
        let rgb = self.prepare_rgb_image(image);
        let width = rgb.width() as usize;
        let height = rgb.height() as usize;
        let raw = rgb.as_raw();
        let (_, npw, _) = self.patch_grid(width, height);
        // Fused: (raw/255 - mean) / std = raw * (1/(255*std)) - mean/std.
        let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c] as f32));
        let bias: [f32; 3] = std::array::from_fn(|c| -(mean[c] as f32) / (std[c] as f32));
        // PAD_RAW_VALUE is in the [0,1] domain; recover the raw byte for scale/bias.
        let pad_byte = PAD_RAW_VALUE * 255.0;
        let pad_norm: [f32; 3] = std::array::from_fn(|c| pad_byte * scale[c] + bias[c]);

        for patch_idx in 0..height.div_ceil(self.patch_size) * npw {
            let patch_y = patch_idx / npw;
            let patch_x = patch_idx - patch_y * npw;
            let y_base = patch_y * self.patch_size;
            let x_base = patch_x * self.patch_size;
            let patch_start = output.len();

            for y in 0..self.patch_size {
                let iy = y_base + y;
                for x in 0..self.patch_size {
                    let ix = x_base + x;
                    if iy < height && ix < width {
                        let offset = (iy * width + ix) * 3;
                        for c in 0..3 {
                            output.push(raw[offset + c] as f32 * scale[c] + bias[c]);
                        }
                    } else {
                        output.extend_from_slice(&pad_norm);
                    }
                }
            }

            let single_temporal_patch_len = self.patch_size * self.patch_size * 3;
            for _ in 1..self.temporal_patch_size {
                output.extend_from_within(patch_start..patch_start + single_temporal_patch_len);
            }
        }
    }
}

impl VisionPreProcessor for InklingImageProcessor {
    fn default_mean(&self) -> [f64; 3] {
        INKLING_IMAGE_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        INKLING_IMAGE_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let processor = self.with_preprocessor_config(config);
        processor.validate()?;

        let mean = config
            .image_mean
            .as_ref()
            .map_or(INKLING_IMAGE_MEAN, |values| {
                if values.len() >= 3 {
                    [values[0], values[1], values[2]]
                } else {
                    INKLING_IMAGE_MEAN
                }
            });
        let std = config
            .image_std
            .as_ref()
            .map_or(INKLING_IMAGE_STD, |values| {
                if values.len() >= 3 {
                    [values[0], values[1], values[2]]
                } else {
                    INKLING_IMAGE_STD
                }
            });

        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut tokens_per_item = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());
        let mut total_patches = 0usize;
        for image in images {
            let (width, height) = image.dimensions();
            let (scaled_width, scaled_height) = processor.scaled_image_dimensions(width, height);
            let (_, _, num_patches) =
                processor.patch_grid(scaled_width as usize, scaled_height as usize);
            feature_token_counts.push(num_patches);
            tokens_per_item.push(num_patches as i64);
            item_sizes.push((scaled_width, scaled_height));
            total_patches += num_patches;
        }

        let mut patches = Vec::with_capacity(total_patches * processor.num_patch_features());
        for image in images {
            processor.append_image_patches(image, &mean, &std, &mut patches);
        }

        let encoder_input = ArrayD::from_shape_vec(
            IxDyn(&[
                total_patches,
                processor.temporal_patch_size,
                processor.patch_size,
                processor.patch_size,
                3,
            ]),
            patches,
        )
        .map_err(|e| {
            TransformError::ShapeError(format!(
                "failed to create Inkling image encoder input [{total_patches}, {}, {}, {}, 3]: {e}",
                processor.temporal_patch_size, processor.patch_size, processor.patch_size
            ))
        })?;

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
                .with_extra(
                    "tokens_per_item",
                    ModelSpecificValue::int_1d(tokens_per_item),
                ),
        )
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        let processor = self.with_preprocessor_config(config);
        let (scaled_width, scaled_height) = processor.scaled_image_dimensions(width, height);
        processor
            .patch_grid(scaled_width as usize, scaled_height as usize)
            .2
    }

    fn model_name(&self) -> &'static str {
        "inkling"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::vision::{preprocessor_config::PatchSize, processor::ModelSpecificValue};

    fn test_image() -> DynamicImage {
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, Rgb([0, 127, 255]));
        img.put_pixel(1, 0, Rgb([255, 127, 0]));
        img.put_pixel(0, 1, Rgb([10, 20, 30]));
        img.put_pixel(1, 1, Rgb([40, 50, 60]));
        DynamicImage::ImageRgb8(img)
    }

    fn processor_without_rescale() -> InklingImageProcessor {
        InklingImageProcessor {
            patch_size: DEFAULT_PATCH_SIZE,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
            rescale_image_frac: None,
            rescale_image_max_upscaled_long_edge: None,
        }
    }

    fn norm(raw: f32, channel: usize) -> f32 {
        ((raw / 255.0) as f64 - INKLING_IMAGE_MEAN[channel]) as f32
            / INKLING_IMAGE_STD[channel] as f32
    }

    fn pad_norm(channel: usize) -> f32 {
        (PAD_RAW_VALUE as f64 - INKLING_IMAGE_MEAN[channel]) as f32
            / INKLING_IMAGE_STD[channel] as f32
    }

    #[test]
    fn inkling_image_shape_includes_extra_patch_column_and_temporal_duplication() {
        let processor = processor_without_rescale();
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(2),
                width: Some(2),
            }),
            temporal_patch_size: Some(2),
            ..PreProcessorConfig::default()
        };
        let result = processor.preprocess(&[test_image()], &config).unwrap();
        assert_eq!(result.encoder_input.shape(), &[2, 2, 2, 2, 3]);
        assert_eq!(result.feature_token_counts, vec![2]);

        let flat = result.encoder_input.as_slice().unwrap();
        assert!((flat[0] - norm(0.0, 0)).abs() < 1e-6);
        assert!((flat[1] - norm(127.0, 1)).abs() < 1e-6);
        assert!((flat[2] - norm(255.0, 2)).abs() < 1e-6);

        let single_temporal_len = 2 * 2 * 3;
        assert_eq!(
            &flat[0..single_temporal_len],
            &flat[single_temporal_len..2 * single_temporal_len]
        );

        let second_patch_start = 2 * single_temporal_len;
        assert!((flat[second_patch_start] - pad_norm(0)).abs() < 1e-6);
        assert!((flat[second_patch_start + 1] - pad_norm(1)).abs() < 1e-6);
        assert!((flat[second_patch_start + 2] - pad_norm(2)).abs() < 1e-6);
    }

    #[test]
    fn calculate_num_tokens_matches_expected_grid_rule() {
        let processor = processor_without_rescale();
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(40),
                width: Some(40),
            }),
            ..PreProcessorConfig::default()
        };
        assert_eq!(processor.calculate_num_tokens(80, 40, &config), 3);
        assert_eq!(processor.calculate_num_tokens(81, 41, &config), 6);
    }

    #[test]
    fn calculate_num_tokens_uses_authored_dimensions_by_default() {
        let processor = InklingImageProcessor::new();
        let config = PreProcessorConfig::default();

        assert_eq!(processor.calculate_num_tokens(640, 480, &config), 204);
        assert_eq!(processor.calculate_num_tokens(1200, 600, &config), 465);
        assert_eq!(processor.calculate_num_tokens(3000, 2000, &config), 3800);
    }

    #[test]
    fn config_can_explicitly_enable_image_rescale() {
        let processor = InklingImageProcessor::new();
        let config = PreProcessorConfig::from_json(
            r#"{"patch_size":40,"rescale_image_frac":2.0,"rescale_image_max_upscaled_long_edge":2048}"#,
        )
        .unwrap();

        assert_eq!(processor.calculate_num_tokens(640, 480, &config), 792);
    }

    #[test]
    fn preprocess_preserves_authored_size_by_default() {
        let processor = InklingImageProcessor::new();
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(2),
                width: Some(2),
            }),
            temporal_patch_size: Some(2),
            ..PreProcessorConfig::default()
        };

        let result = processor.preprocess(&[test_image()], &config).unwrap();

        assert_eq!(result.encoder_input.shape(), &[2, 2, 2, 2, 3]);
        assert_eq!(result.feature_token_counts, vec![2]);
        assert_eq!(result.item_sizes, vec![(2, 2)]);
        assert!(matches!(
            result.model_specific.get("tokens_per_item"),
            Some(ModelSpecificValue::IntTensor { data, shape })
                if data.as_slice() == [2] && shape.as_slice() == [1]
        ));
    }

    #[test]
    fn preprocess_applies_explicit_resize_before_patchifying() {
        let processor = InklingImageProcessor::new();
        let config = PreProcessorConfig::from_json(
            r#"{"patch_size":2,"temporal_patch_size":2,"rescale_image_frac":2.0}"#,
        )
        .unwrap();

        let result = processor.preprocess(&[test_image()], &config).unwrap();

        assert_eq!(result.encoder_input.shape(), &[6, 2, 2, 2, 3]);
        assert_eq!(result.feature_token_counts, vec![6]);
        assert_eq!(result.item_sizes, vec![(4, 4)]);
    }

    #[test]
    fn scaled_dimensions_match_expected_rounding_and_cap_rule() {
        let processor = InklingImageProcessor {
            patch_size: DEFAULT_PATCH_SIZE,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
            rescale_image_frac: Some(1.5),
            rescale_image_max_upscaled_long_edge: None,
        };

        assert_eq!(processor.scaled_image_dimensions(3, 2), (5, 3));

        let capped_processor = InklingImageProcessor {
            patch_size: DEFAULT_PATCH_SIZE,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
            rescale_image_frac: Some(2.0),
            rescale_image_max_upscaled_long_edge: Some(
                DEFAULT_RESCALE_IMAGE_MAX_UPSCALED_LONG_EDGE,
            ),
        };
        assert_eq!(
            capped_processor.scaled_image_dimensions(1200, 600),
            (2048, 1024)
        );
        assert_eq!(
            capped_processor.scaled_image_dimensions(3000, 2000),
            (3000, 2000)
        );
    }
}
