//! Qwen3-Omni image and video preprocessing.
//!
//! This processor shares patchification and normalization through
//! `QwenVLProcessorBase`, but cannot use `Qwen3VLProcessor` directly. Omni
//! applies video pixel limits per frame rather than across the sampled clip,
//! gives its video preprocessor config precedence for video-specific limits,
//! and emits timing metadata required by mixed-modality M-RoPE.

use image::DynamicImage;

use super::qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase, QwenVideoResizeMode};
use crate::{
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::TransformError,
    },
};

pub const QWEN3_OMNI_MEAN: [f64; 3] = [0.5; 3];
pub const QWEN3_OMNI_STD: [f64; 3] = [0.5; 3];
pub const DEFAULT_IMAGE_MIN_PIXELS: usize = 3_136;
pub const DEFAULT_IMAGE_MAX_PIXELS: usize = 12_845_056;
pub const DEFAULT_VIDEO_MIN_PIXELS: usize = 128 * 32 * 32;
pub const DEFAULT_VIDEO_MAX_PIXELS: usize = 768 * 32 * 32;
pub const DEFAULT_PATCH_SIZE: usize = 16;
pub const DEFAULT_MERGE_SIZE: usize = 2;
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;

#[derive(Debug, Clone)]
pub struct Qwen3OmniVisionProcessor {
    inner: QwenVLProcessorBase,
}

impl Default for Qwen3OmniVisionProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen3OmniVisionProcessor {
    pub fn new() -> Self {
        Self::with_limits(
            DEFAULT_IMAGE_MIN_PIXELS,
            DEFAULT_IMAGE_MAX_PIXELS,
            DEFAULT_VIDEO_MIN_PIXELS,
            DEFAULT_VIDEO_MAX_PIXELS,
            DEFAULT_PATCH_SIZE,
            DEFAULT_MERGE_SIZE,
            DEFAULT_TEMPORAL_PATCH_SIZE,
        )
    }

    fn with_limits(
        image_min_pixels: usize,
        image_max_pixels: usize,
        video_min_pixels: usize,
        video_max_pixels: usize,
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
    ) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels: image_min_pixels,
                max_pixels: image_max_pixels,
                video_min_pixels,
                video_max_pixels,
                video_resize_mode: QwenVideoResizeMode::PerFrame,
                temporal_patch_size,
                mean: QWEN3_OMNI_MEAN,
                std: QWEN3_OMNI_STD,
                model_name: "qwen3-omni",
            }),
        }
    }

    fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        let configured_min = config.min_pixels.or_else(|| config.get_shortest_edge());
        let configured_max = config.max_pixels.or_else(|| config.get_longest_edge());
        Self::with_limits(
            configured_min.unwrap_or(DEFAULT_IMAGE_MIN_PIXELS),
            configured_max.unwrap_or(DEFAULT_IMAGE_MAX_PIXELS),
            DEFAULT_VIDEO_MIN_PIXELS,
            DEFAULT_VIDEO_MAX_PIXELS,
            config.get_patch_size(DEFAULT_PATCH_SIZE),
            config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
            config
                .temporal_patch_size
                .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
        )
    }

    fn from_video_preprocessor_config(config: &PreProcessorConfig) -> Self {
        let configured_min = config.min_pixels.or_else(|| config.get_shortest_edge());
        let configured_max = config.max_pixels.or_else(|| config.get_longest_edge());
        Self::with_limits(
            DEFAULT_IMAGE_MIN_PIXELS,
            DEFAULT_IMAGE_MAX_PIXELS,
            configured_min.unwrap_or(DEFAULT_VIDEO_MIN_PIXELS),
            configured_max.unwrap_or(DEFAULT_VIDEO_MAX_PIXELS),
            config.get_patch_size(DEFAULT_PATCH_SIZE),
            config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
            config
                .temporal_patch_size
                .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
        )
    }

    fn with_image_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        if config.has_structural_overrides() {
            Self::from_preprocessor_config(config)
        } else {
            self.clone()
        }
    }

    fn with_video_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        if config.has_structural_overrides() {
            if config.is_image_only_processor_type() {
                // Qwen3-Omni's shared preprocessor_config.json carries image
                // limits. The HF processor supplies separate video defaults at
                // call time, so those image limits must not become a per-frame
                // video budget here.
                Self::from_preprocessor_config(config)
            } else {
                Self::from_video_preprocessor_config(config)
            }
        } else {
            self.clone()
        }
    }
}

impl VisionPreProcessor for Qwen3OmniVisionProcessor {
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
        self.with_image_preprocessor_config(config)
            .inner
            .preprocess(images, config)
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.with_video_preprocessor_config(config)
            .inner
            .preprocess_video(frames, config)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.with_video_preprocessor_config(config)
            .inner
            .preprocess_video_rgb(frames, config)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        self.with_image_preprocessor_config(config)
            .inner
            .calculate_num_tokens(width, height, config)
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
    use image::{DynamicImage, RgbImage};

    use super::*;
    use crate::vision::{processor::ModelSpecificValue, processors::Qwen3VLProcessor};

    #[test]
    fn omni_multiframe_resize_differs_from_qwen3_volume_budget() {
        let omni = Qwen3OmniVisionProcessor::new();
        let qwen3_vl = Qwen3VLProcessor::with_config(
            DEFAULT_PATCH_SIZE,
            DEFAULT_MERGE_SIZE,
            DEFAULT_VIDEO_MIN_PIXELS,
            DEFAULT_VIDEO_MAX_PIXELS,
            DEFAULT_TEMPORAL_PATCH_SIZE,
        );

        assert_eq!(omni.inner.min_pixels(), DEFAULT_IMAGE_MIN_PIXELS);
        assert_eq!(omni.inner.max_pixels(), DEFAULT_IMAGE_MAX_PIXELS);
        assert_eq!(omni.inner.video_min_pixels(), DEFAULT_VIDEO_MIN_PIXELS);
        assert_eq!(omni.inner.video_max_pixels(), DEFAULT_VIDEO_MAX_PIXELS);
        assert_eq!(
            omni.inner.video_resize_mode(),
            QwenVideoResizeMode::PerFrame
        );

        let omni_size = omni.inner.smart_resize_video(16, 720, 1280).unwrap();
        let volume_size = qwen3_vl.smart_resize_video(16, 720, 1280).unwrap();

        assert_eq!(omni_size, (640, 1152));
        assert_eq!(volume_size, (160, 288));
        assert_ne!(omni_size, volume_size);
    }

    #[test]
    fn video_preprocessor_config_overrides_per_frame_limits() {
        let config = PreProcessorConfig::from_json(
            r#"{"size":{"shortest_edge":65536,"longest_edge":262144},"temporal_patch_size":2}"#,
        )
        .unwrap();
        let processor = Qwen3OmniVisionProcessor::from_video_preprocessor_config(&config);

        assert_eq!(processor.inner.min_pixels(), DEFAULT_IMAGE_MIN_PIXELS);
        assert_eq!(processor.inner.max_pixels(), DEFAULT_IMAGE_MAX_PIXELS);
        assert_eq!(processor.inner.video_min_pixels(), 65_536);
        assert_eq!(processor.inner.video_max_pixels(), 262_144);
        assert_eq!(
            processor.inner.smart_resize_video(32, 720, 1280).unwrap(),
            (384, 672)
        );
    }

    #[test]
    fn shared_image_config_keeps_omni_video_defaults() {
        let config = PreProcessorConfig::from_json(
            r#"{"image_processor_type":"Qwen2VLImageProcessor","min_pixels":3136,"max_pixels":12845056,"patch_size":16,"merge_size":2,"temporal_patch_size":2}"#,
        )
        .unwrap();
        let processor = Qwen3OmniVisionProcessor::new().with_video_preprocessor_config(&config);

        assert_eq!(processor.inner.video_min_pixels(), DEFAULT_VIDEO_MIN_PIXELS);
        assert_eq!(processor.inner.video_max_pixels(), DEFAULT_VIDEO_MAX_PIXELS);
        assert_eq!(
            processor.inner.smart_resize_video(16, 720, 1280).unwrap(),
            (640, 1152)
        );
    }

    #[test]
    fn empty_config_uses_omni_half_normalization() {
        let image = DynamicImage::ImageRgb8(RgbImage::new(32, 32));
        let output = Qwen3OmniVisionProcessor::new()
            .preprocess(&[image], &PreProcessorConfig::default())
            .unwrap();

        assert!(output
            .encoder_input
            .iter()
            .all(|value| (*value + 1.0).abs() < 1e-6));
    }

    #[test]
    fn sampled_video_fps_controls_mrope_grid_timing() {
        let config = PreProcessorConfig::from_json(
            r#"{"size":{"shortest_edge":1024,"longest_edge":65536},"fps":4.0}"#,
        )
        .unwrap();
        let frames = vec![
            DynamicImage::ImageRgb8(RgbImage::new(32, 32)),
            DynamicImage::ImageRgb8(RgbImage::new(32, 32)),
        ];

        let output = Qwen3OmniVisionProcessor::new()
            .preprocess_video(&frames, &config)
            .unwrap();

        assert!(matches!(
            output.model_specific.get("video_second_per_grid"),
            Some(ModelSpecificValue::Tensor { data, shape })
                if data == &vec![0.5] && shape == &vec![1]
        ));
    }
}
