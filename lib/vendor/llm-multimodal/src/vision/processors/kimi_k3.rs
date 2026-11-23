//! Kimi-K3 (MoonViT) image processor.
//!
//! K3 runs the same MoonViT stack as K2.5 — see [`super::moonvit`] — but its
//! `preprocessor_config.json` differs in two ways that change the pixels the
//! encoder sees:
//!
//! * `in_patch_limit` is 65536, four times K2.5's 16384, so K3 keeps
//!   substantially more resolution before the downscale kicks in.
//! * It ships a `transparent_bg_config` (chessboard, 8px squares, 255/180) with
//!   `transparent_bg_fill_stage: "after_resize"`. Transparent pixels are
//!   composited over that board instead of having their alpha dropped, which
//!   is what `.convert("RGB")` — and a bare `to_rgb8()` — would do. A fully
//!   transparent pixel normally stores RGB `(0,0,0)`, so dropping alpha feeds
//!   the encoder solid black where the reference feeds a light checkerboard.
//!
//! Defaults here mirror the shipped config; anything the runtime loads from the
//! model's own `preprocessor_config.json` wins at call time.

use image::DynamicImage;
use serde_json::Value;

use super::moonvit::{self, MoonVitParams};
use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::{
        TransformError, TransparentBg, TransparentBgConfig, TransparentBgFillStage,
        TransparentBgPattern,
    },
};

pub const KIMI_K3_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
pub const KIMI_K3_STD: [f64; 3] = [0.5, 0.5, 0.5];

pub const DEFAULT_PATCH_SIZE: usize = 14;
pub const DEFAULT_MERGE_SIZE: usize = 2;
/// Maximum total patches before merge (`in_patch_limit`) — 4x K2.5's budget.
pub const DEFAULT_IN_PATCH_LIMIT: usize = 65536;
/// Maximum patches along one spatial dimension
pub const DEFAULT_PATCH_LIMIT_ON_ONE_SIDE: usize = 512;

/// The `transparent_bg_config` shipped with `moonshotai/Kimi-K3`, verbatim from
/// the checkpoint's `preprocessor_config.json`. These values describe K3 only —
/// note the stage is `"after_resize"` where the reference's fallback for a
/// missing key is `"before_resize"`.
fn default_transparent_bg() -> TransparentBg {
    TransparentBg {
        config: TransparentBgConfig {
            pattern: TransparentBgPattern::Chessboard,
            chessboard_square_size: 8,
            chessboard_square_on_top_left: true,
            chessboard_white_value: 255,
            chessboard_gray_value: 180,
        },
        stage: TransparentBgFillStage::AfterResize,
    }
}

#[derive(Debug, Clone)]
pub struct KimiK3Processor {
    params: MoonVitParams,
    transparent_bg: Option<TransparentBg>,
}

impl Default for KimiK3Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl KimiK3Processor {
    pub fn new() -> Self {
        Self {
            params: MoonVitParams {
                patch_size: DEFAULT_PATCH_SIZE,
                merge_size: DEFAULT_MERGE_SIZE,
                in_patch_limit: DEFAULT_IN_PATCH_LIMIT,
                patch_limit_on_one_side: DEFAULT_PATCH_LIMIT_ON_ONE_SIDE,
            },
            transparent_bg: Some(default_transparent_bg()),
        }
    }

    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        let base = Self::new();
        Self {
            params: base.params.resolved(config),
            transparent_bg: base.resolved_transparent_bg(config),
        }
    }

    pub fn patch_size(&self) -> usize {
        self.params.patch_size
    }

    pub fn merge_size(&self) -> usize {
        self.params.merge_size
    }

    /// Overlay the model's own transparency settings, if it ships any.
    ///
    /// The registry hands out one shared instance, so this is the only point at
    /// which a checkpoint's real config can take effect. An absent or malformed
    /// key keeps K3's shipped board rather than silently disabling compositing,
    /// because `preprocessor_config.json` is optional in this runtime (see
    /// `grpc::multimodal::config`). That diverges from the reference, which
    /// reads a missing key as "drop alpha", so a checkpoint wanting that path
    /// must say `"transparent_bg_config": null` explicitly.
    fn resolved_transparent_bg(&self, config: &PreProcessorConfig) -> Option<TransparentBg> {
        if config
            .extra
            .get("transparent_bg_config")
            .is_some_and(Value::is_null)
        {
            return None;
        }
        let default = self.transparent_bg?;
        Some(TransparentBg {
            config: config
                .get_extra::<TransparentBgConfig>("transparent_bg_config")
                .unwrap_or(default.config),
            stage: config
                .get_extra::<TransparentBgFillStage>("transparent_bg_fill_stage")
                .unwrap_or(default.stage),
        })
    }
}

impl VisionPreProcessor for KimiK3Processor {
    fn default_mean(&self) -> [f64; 3] {
        KIMI_K3_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        KIMI_K3_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        moonvit::preprocess(
            self.params.resolved(config),
            images,
            config,
            self.resolved_transparent_bg(config),
        )
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        self.params
            .resolved(config)
            .compute_resize_config(width as usize, height as usize)
            .num_tokens
    }

    fn model_name(&self) -> &'static str {
        "kimi-k3"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage, Rgba, RgbaImage};
    use serde_json::json;

    use super::*;
    use crate::vision::processors::kimi_k25::{
        KimiK25Processor, DEFAULT_IN_PATCH_LIMIT as K25_LIM,
    };

    fn norm_config() -> PreProcessorConfig {
        PreProcessorConfig {
            image_mean: Some(KIMI_K3_MEAN.to_vec()),
            image_std: Some(KIMI_K3_STD.to_vec()),
            ..Default::default()
        }
    }

    /// Undo the (x/255 - 0.5) / 0.5 normalization to get the byte back.
    fn denormalize(v: f32) -> f32 {
        (v * 0.5 + 0.5) * 255.0
    }

    #[test]
    fn test_defaults_differ_from_k25() {
        let p = KimiK3Processor::new();
        assert_eq!(p.patch_size(), 14);
        assert_eq!(p.merge_size(), 2);
        assert_eq!(p.params.in_patch_limit, 65536);
        assert_ne!(
            p.params.in_patch_limit, K25_LIM,
            "K3's patch budget must not inherit K2.5's"
        );
        assert_eq!(p.model_name(), "kimi-k3");
    }

    #[test]
    fn test_larger_patch_budget_keeps_more_resolution() {
        let k3 = KimiK3Processor::new();
        let k25 = KimiK25Processor::new();
        let config = PreProcessorConfig::default();

        // 4000x3000 is well past both budgets, so the difference shows up
        // directly in how far each model downscales.
        let k3_tokens = k3.calculate_num_tokens(4000, 3000, &config);
        let k25_tokens = k25.calculate_num_tokens(4000, 3000, &config);
        assert!(
            k3_tokens > k25_tokens,
            "K3 should keep more tokens than K2.5 ({k3_tokens} vs {k25_tokens})"
        );
    }

    #[test]
    fn test_transparent_pixels_composite_over_chessboard() {
        let p = KimiK3Processor::new();
        // 56x56 is factor-aligned (28*2), so there is no padding to confuse the
        // check, and 8px squares tile it exactly.
        let image = DynamicImage::from(RgbaImage::from_pixel(56, 56, Rgba([0, 0, 0, 0])));
        let result = p.preprocess(&[image], &norm_config()).unwrap();

        let values: Vec<f32> = result
            .encoder_input_flat()
            .iter()
            .map(|&v| denormalize(v))
            .collect();

        // Every pixel is fully transparent, so the output is the bare board:
        // only the two configured grey levels, and both must appear.
        assert!(
            values.iter().any(|&v| (v - 255.0).abs() < 0.5),
            "expected chessboard white (255)"
        );
        assert!(
            values.iter().any(|&v| (v - 180.0).abs() < 0.5),
            "expected chessboard grey (180)"
        );
        assert!(
            values
                .iter()
                .all(|&v| (v - 255.0).abs() < 0.5 || (v - 180.0).abs() < 0.5),
            "transparent input must not produce anything but board values"
        );
        // The bug this guards: dropping alpha would leave normalized -1.0.
        assert!(
            !result.encoder_input_flat().iter().any(|&v| v < -0.9),
            "transparent pixels must not read as solid black"
        );
    }

    #[test]
    fn test_chessboard_phase_matches_reference() {
        // The reference greys a square when `(y//s + x//s) % 2 == 1` for
        // chessboard_square_on_top_left=true, so (0,0) is *white* and the
        // neighbouring square is grey. An inverted board would still pass a
        // "both values present" check, hence this one.
        let p = KimiK3Processor::new();
        let image = DynamicImage::from(RgbaImage::from_pixel(56, 56, Rgba([0, 0, 0, 0])));
        let result = p.preprocess(&[image], &norm_config()).unwrap();

        // Patch 0 is the top-left 14x14 block, channel-first: element 0 is
        // R at (0,0), and element 8 is R at (8,0) — the next square over.
        let flat = result.encoder_input_flat();
        assert!((denormalize(flat[0]) - 255.0).abs() < 0.5, "(0,0) is white");
        assert!((denormalize(flat[8]) - 180.0).abs() < 0.5, "(8,0) is grey");
    }

    #[test]
    fn test_opaque_pixels_are_untouched() {
        let p = KimiK3Processor::new();
        let opaque = DynamicImage::from(RgbaImage::from_pixel(56, 56, Rgba([255, 255, 255, 255])));
        let result = p.preprocess(&[opaque], &norm_config()).unwrap();
        assert!(
            result
                .encoder_input_flat()
                .iter()
                .all(|&v| (v - 1.0).abs() < 1e-3),
            "fully opaque white must stay white"
        );
    }

    #[test]
    fn test_semi_transparent_blends_toward_background() {
        let p = KimiK3Processor::new();
        // Half-opaque black over the board: 0.5*0 + 0.5*bg → 90 or 127.
        let image = DynamicImage::from(RgbaImage::from_pixel(56, 56, Rgba([0, 0, 0, 128])));
        let result = p.preprocess(&[image], &norm_config()).unwrap();
        let values: Vec<f32> = result
            .encoder_input_flat()
            .iter()
            .map(|&v| denormalize(v))
            .collect();
        let alpha = 128.0 / 255.0;
        let expect_grey = (1.0 - alpha) * 180.0;
        let expect_white = (1.0 - alpha) * 255.0;
        assert!(
            values
                .iter()
                .all(|&v| (v - expect_grey).abs() < 1.5 || (v - expect_white).abs() < 1.5),
            "semi-transparent black should land between the board and black"
        );
    }

    #[test]
    fn test_rgb_input_matches_k25_pipeline() {
        // Compositing must be a no-op for images with no alpha channel, so an
        // RGB image goes through byte-identically to K2.5.
        let config = norm_config();
        let image = DynamicImage::from(RgbImage::from_pixel(196, 140, Rgb([37, 211, 102])));

        let k3 = KimiK3Processor::new()
            .preprocess(std::slice::from_ref(&image), &config)
            .unwrap();
        let k25 = KimiK25Processor::new()
            .preprocess(&[image], &config)
            .unwrap();

        assert_eq!(k3.encoder_input.shape(), k25.encoder_input.shape());
        assert_eq!(k3.encoder_input_flat(), k25.encoder_input_flat());
    }

    #[test]
    fn test_transparent_bg_config_overridden_by_model_config() {
        let mut config = norm_config();
        config.extra.insert(
            "transparent_bg_config".to_string(),
            json!({ "pattern": "white" }),
        );
        let p = KimiK3Processor::new();
        let image = DynamicImage::from(RgbaImage::from_pixel(56, 56, Rgba([0, 0, 0, 0])));
        let result = p.preprocess(&[image], &config).unwrap();
        assert!(
            result
                .encoder_input_flat()
                .iter()
                .all(|&v| (v - 1.0).abs() < 1e-3),
            "a white background config must flatten transparency to pure white"
        );
    }

    #[test]
    fn test_fill_stage_before_resize_changes_output() {
        // The stage is not cosmetic: the board is drawn at the resolution of
        // whatever it is painted onto, so a downscaled image differs.
        let p = KimiK3Processor::new();
        let mut config = norm_config();
        // Force a downscale so the two stages see different resolutions.
        config
            .extra
            .insert("patch_limit_on_one_side".to_string(), json!(4));

        let image = DynamicImage::from(RgbaImage::from_pixel(560, 560, Rgba([0, 0, 0, 0])));
        let after = p.preprocess(std::slice::from_ref(&image), &config).unwrap();

        config.extra.insert(
            "transparent_bg_fill_stage".to_string(),
            json!("before_resize"),
        );
        let before = p.preprocess(&[image], &config).unwrap();

        assert_eq!(after.encoder_input.shape(), before.encoder_input.shape());
        assert_ne!(
            after.encoder_input_flat(),
            before.encoder_input_flat(),
            "before_resize must not silently behave like after_resize"
        );
    }

    #[test]
    fn test_from_preprocessor_config_reads_limits() {
        let mut config = PreProcessorConfig::default();
        config
            .extra
            .insert("in_patch_limit".to_string(), json!(1024));
        config
            .extra
            .insert("patch_limit_on_one_side".to_string(), json!(64));
        let p = KimiK3Processor::from_preprocessor_config(&config);
        assert_eq!(p.params.in_patch_limit, 1024);
        assert_eq!(p.params.patch_limit_on_one_side, 64);
    }

    #[test]
    fn test_explicit_null_config_disables_compositing() {
        // Absence keeps K3's shipped board, so an explicit null is the only way
        // to reach the reference's alpha-dropping path.
        let mut config = norm_config();
        config
            .extra
            .insert("transparent_bg_config".to_string(), Value::Null);

        let p = KimiK3Processor::new();
        assert_eq!(p.resolved_transparent_bg(&config), None);

        let image = DynamicImage::from(RgbaImage::from_pixel(56, 56, Rgba([0, 0, 0, 0])));
        let result = p.preprocess(&[image], &config).unwrap();
        assert!(
            result
                .encoder_input_flat()
                .iter()
                .all(|&v| (v + 1.0).abs() < 1e-6),
            "an explicit null must fall back to dropping alpha"
        );
    }

    #[test]
    fn test_after_resize_ignores_colour_hidden_under_alpha() {
        // The reference resizes RGBA with PIL before compositing, and PIL
        // premultiplies (verified against Pillow 11.2.1), so colour under fully
        // transparent pixels cannot bleed into neighbours: swapping it must not
        // change the tensor.
        let p = KimiK3Processor::new();
        let mut config = norm_config();
        // Cap the long side at 2 * 14 px so a real downscale happens.
        config
            .extra
            .insert("patch_limit_on_one_side".to_string(), json!(2));

        let outputs = [[0, 0, 0], [255, 0, 255]].map(|hidden| {
            let mut img = RgbaImage::new(112, 112);
            for (_, y, px) in img.enumerate_pixels_mut() {
                *px = if y < 56 {
                    Rgba([255, 255, 255, 255])
                } else {
                    Rgba([hidden[0], hidden[1], hidden[2], 0])
                };
            }
            p.preprocess(&[DynamicImage::from(img)], &config).unwrap()
        });

        assert_eq!(
            outputs[0].encoder_input_flat(),
            outputs[1].encoder_input_flat(),
            "a straight-alpha resize would let the hidden magenta bleed through"
        );
    }

    #[test]
    fn test_empty_batch_errors() {
        let p = KimiK3Processor::new();
        assert!(p.preprocess(&[], &PreProcessorConfig::default()).is_err());
    }
}
