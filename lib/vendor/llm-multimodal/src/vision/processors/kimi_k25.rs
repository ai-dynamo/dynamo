//! Kimi-K2.5 (MoonViT) image processor.
//!
//! Matches the HuggingFace `KimiK25VisionProcessor` preprocessing pipeline; the
//! pipeline itself lives in [`super::moonvit`], which Kimi-K3 shares. K2.5
//! ships no `transparent_bg_config`, so alpha is dropped rather than
//! composited — the reference's `image.convert("RGB")` behavior.

use image::DynamicImage;

use super::moonvit::{self, MoonVitParams};
use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::TransformError,
};

pub const KIMI_K25_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
pub const KIMI_K25_STD: [f64; 3] = [0.5, 0.5, 0.5];

pub const DEFAULT_PATCH_SIZE: usize = 14;
pub const DEFAULT_MERGE_SIZE: usize = 2;
/// Maximum total patches before merge (from preprocessor_config.json in_patch_limit)
pub const DEFAULT_IN_PATCH_LIMIT: usize = 16384;
/// Maximum patches along one spatial dimension
pub const DEFAULT_PATCH_LIMIT_ON_ONE_SIDE: usize = 512;

#[derive(Debug, Clone)]
pub struct KimiK25Processor {
    params: MoonVitParams,
}

impl Default for KimiK25Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl KimiK25Processor {
    pub fn new() -> Self {
        Self {
            params: MoonVitParams {
                patch_size: DEFAULT_PATCH_SIZE,
                merge_size: DEFAULT_MERGE_SIZE,
                in_patch_limit: DEFAULT_IN_PATCH_LIMIT,
                patch_limit_on_one_side: DEFAULT_PATCH_LIMIT_ON_ONE_SIDE,
            },
        }
    }

    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            params: Self::new().params.resolved(config),
        }
    }

    pub fn patch_size(&self) -> usize {
        self.params.patch_size
    }

    pub fn merge_size(&self) -> usize {
        self.params.merge_size
    }
}

impl VisionPreProcessor for KimiK25Processor {
    fn default_mean(&self) -> [f64; 3] {
        KIMI_K25_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        KIMI_K25_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        // K2.5 ships no `transparent_bg_config`, so alpha is dropped.
        moonvit::preprocess(self.params.resolved(config), images, config, None)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        self.params
            .resolved(config)
            .compute_resize_config(width as usize, height as usize)
            .num_tokens
    }

    fn model_name(&self) -> &'static str {
        "kimi-k2.5"
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

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_defaults() {
        let p = KimiK25Processor::new();
        assert_eq!(p.patch_size(), 14);
        assert_eq!(p.merge_size(), 2);
        assert_eq!(p.params.factor(), 28);
        assert_eq!(p.params.in_patch_limit, DEFAULT_IN_PATCH_LIMIT);
    }

    #[test]
    fn test_mean_std() {
        let p = KimiK25Processor::new();
        assert_eq!(p.default_mean(), KIMI_K25_MEAN);
        assert_eq!(p.default_std(), KIMI_K25_STD);
    }

    #[test]
    fn test_model_name() {
        assert_eq!(KimiK25Processor::new().model_name(), "kimi-k2.5");
    }

    #[test]
    fn test_resize_config_no_upscale() {
        let p = KimiK25Processor::new();
        // Small image should NOT be upscaled (scale capped at 1.0)
        let cfg = p.params.compute_resize_config(100, 100);
        assert!(cfg.new_width <= 100);
        assert!(cfg.new_height <= 100);
        // Padded dimensions must be factor-aligned
        assert_eq!((cfg.new_height + cfg.pad_height) % 28, 0);
        assert_eq!((cfg.new_width + cfg.pad_width) % 28, 0);
    }

    #[test]
    fn test_resize_config_large_image_downscaled() {
        let p = KimiK25Processor::new();
        // Large image should be downscaled
        let cfg = p.params.compute_resize_config(4000, 3000);
        // Resized dimensions should be smaller than original
        assert!(cfg.new_width < 4000);
        assert!(cfg.new_height < 3000);
        // Per-side patch limit must be respected (HF assertion)
        let padded_h = cfg.new_height + cfg.pad_height;
        let padded_w = cfg.new_width + cfg.pad_width;
        assert!(padded_h / 14 <= DEFAULT_PATCH_LIMIT_ON_ONE_SIDE * 2);
        assert!(padded_w / 14 <= DEFAULT_PATCH_LIMIT_ON_ONE_SIDE * 2);
    }

    #[test]
    fn test_resize_config_matches_hf_reference() {
        let p = KimiK25Processor::new();
        // 600x400 image: scale=1.0 (small enough), resize to 600x400,
        // pad to (600+4=) → let's compute:
        // factor=28, 400 % 28 = 400 - 14*28 = 400-392 = 8, pad_h = 28-8 = 20
        // 600 % 28 = 600 - 21*28 = 600-588 = 12, pad_w = 28-12 = 16
        let cfg = p.params.compute_resize_config(600, 400);
        assert_eq!(cfg.new_width, 600);
        assert_eq!(cfg.new_height, 400);
        assert_eq!(cfg.pad_height, 20);
        assert_eq!(cfg.pad_width, 16);
        // Padded: 420 x 616, grid: 30 x 44, tokens: (30*44)/(2*2) = 330
        assert_eq!(cfg.num_tokens, 330);
    }

    #[test]
    fn test_preprocess_4d_output() {
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig {
            do_normalize: Some(true),
            image_mean: Some(KIMI_K25_MEAN.to_vec()),
            image_std: Some(KIMI_K25_STD.to_vec()),
            ..Default::default()
        };

        let image = create_test_image(600, 400, Rgb([128, 128, 128]));
        let result = p.preprocess(&[image], &config).unwrap();

        // 4D output: [total_patches, 3, 14, 14]
        assert_eq!(result.encoder_input.ndim(), 4);
        assert_eq!(result.encoder_input.shape()[1], 3);
        assert_eq!(result.encoder_input.shape()[2], 14);
        assert_eq!(result.encoder_input.shape()[3], 14);

        assert!(result.model_specific.contains_key("grid_thws"));
        assert!(result.model_specific.contains_key("patches_per_image"));
        assert!(result.feature_token_counts[0] > 0);
    }

    #[test]
    fn test_preprocess_multiple_images() {
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig::default();
        let images = vec![
            create_test_image(600, 400, Rgb([100, 100, 100])),
            create_test_image(400, 600, Rgb([150, 150, 150])),
        ];

        let result = p.preprocess(&images, &config).unwrap();

        assert_eq!(result.item_sizes.len(), 2);
        assert_eq!(result.feature_token_counts.len(), 2);
        assert_eq!(result.encoder_input.ndim(), 4);
        assert_eq!(result.encoder_input.shape()[1], 3);

        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("grid_thws")
        {
            assert_eq!(shape, &[2, 3]);
            assert_eq!(data.len(), 6);
        } else {
            panic!("Expected grid_thws to be IntTensor");
        }

        if let Some(ModelSpecificValue::IntTensor { data, .. }) =
            result.model_specific.get("patches_per_image")
        {
            let total: i64 = data.iter().sum();
            assert_eq!(total as usize, result.encoder_input.shape()[0]);
        }
    }

    #[test]
    fn test_calculate_num_tokens() {
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig::default();
        let tokens = p.calculate_num_tokens(600, 400, &config);
        assert_eq!(tokens, 330);
    }

    #[test]
    fn test_from_preprocessor_config() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            merge_size: Some(2),
            ..Default::default()
        };
        let p = KimiK25Processor::from_preprocessor_config(&config);
        assert_eq!(p.patch_size(), 14);
        assert_eq!(p.merge_size(), 2);
    }

    #[test]
    fn test_zero_padding_applied() {
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig {
            image_mean: Some(KIMI_K25_MEAN.to_vec()),
            image_std: Some(KIMI_K25_STD.to_vec()),
            ..Default::default()
        };

        // 100x100 white image — after normalization: (255/255 - 0.5) / 0.5 = 1.0
        // Padded region: (0/255 - 0.5) / 0.5 = -1.0
        let image = create_test_image(100, 100, Rgb([255, 255, 255]));
        let result = p.preprocess(&[image], &config).unwrap();

        let flat = result.encoder_input_flat();
        // Padded region should be normalized black (-1.0)
        let has_neg_ones = flat.iter().any(|&v| (v - (-1.0)).abs() < 1e-6);
        assert!(
            has_neg_ones,
            "Expected normalized-black padding (-1.0) in output"
        );

        // Image region should be normalized white (1.0)
        let has_ones = flat.iter().any(|&v| (v - 1.0).abs() < 1e-6);
        assert!(
            has_ones,
            "Expected normalized-white image values (1.0) in output"
        );
    }

    #[test]
    fn test_preprocess_tiny_image() {
        // 1x1 image should not panic — padded to 28x28
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig {
            image_mean: Some(KIMI_K25_MEAN.to_vec()),
            image_std: Some(KIMI_K25_STD.to_vec()),
            ..Default::default()
        };
        let image = create_test_image(1, 1, Rgb([128, 128, 128]));
        let result = p.preprocess(&[image], &config).unwrap();
        assert_eq!(result.encoder_input.ndim(), 4);
        assert!(result.encoder_input.shape()[0] > 0);
        assert!(result.feature_token_counts[0] > 0);
    }

    #[test]
    fn test_k25_drops_alpha_before_resizing() {
        // K2.5 ships no transparent_bg_config, so alpha is dropped and the
        // stored RGB kept. The reference does that at load time, before any
        // resize; resizing RGBA first would premultiply and discard the colour
        // under transparent pixels, so transparent red must survive as red.
        let p = KimiK25Processor::new();
        let config = PreProcessorConfig {
            image_mean: Some(KIMI_K25_MEAN.to_vec()),
            image_std: Some(KIMI_K25_STD.to_vec()),
            extra: [(
                "patch_limit_on_one_side".to_string(),
                serde_json::json!(2), // caps the long side at 2 * 14 px
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let hidden_red = DynamicImage::from(image::RgbaImage::from_pixel(
            112,
            112,
            image::Rgba([255, 0, 0, 0]),
        ));

        let result = p.preprocess(&[hidden_red], &config).unwrap();
        // [patches, 3, patch_size, patch_size]. 112px caps to 2 * 14 per side,
        // so every patch is content and none of it is padding.
        let &[patches, 3, ph, pw] = result.encoder_input.shape() else {
            panic!("unexpected shape {:?}", result.encoder_input.shape());
        };
        assert_eq!(patches, 4);
        for p in 0..patches {
            for y in 0..ph {
                for x in 0..pw {
                    let px = [0, 1, 2].map(|c| result.encoder_input[[p, c, y, x]]);
                    // mean = std = 0.5, so 255 -> +1.0 and 0 -> -1.0.
                    assert!(
                        (px[0] - 1.0).abs() < 1e-3 && px[1] < -0.99 && px[2] < -0.99,
                        "red under alpha=0 must survive as red at patch {p} ({x},{y}), got {px:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_in_patch_limit_resolved_from_config() {
        // A model shipping a larger budget must not be capped by the K2.5 default.
        let p = KimiK25Processor::new();
        let mut config = PreProcessorConfig::default();
        config
            .extra
            .insert("in_patch_limit".to_string(), serde_json::json!(65536));

        let default_tokens = p.calculate_num_tokens(4000, 3000, &PreProcessorConfig::default());
        let raised_tokens = p.calculate_num_tokens(4000, 3000, &config);
        assert!(
            raised_tokens > default_tokens,
            "raising in_patch_limit should raise the token count \
             ({raised_tokens} vs {default_tokens})"
        );
    }
}
