//! Vision processor trait and registry.
//!
//! Shared encoder output types live in [`crate::encoder_inputs`] and are re-exported
//! here for compatibility.

use std::collections::HashMap;

use image::DynamicImage;

use super::{preprocessor_config::PreProcessorConfig, transforms::TransformError};
pub use crate::encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs};
use crate::types::RgbFrameRef;

/// Helper to extract a dimension from encoder_input given an ndim-dependent axis index.
/// Returns `Err` if the ndim is not 4 or 5.
fn dim_for_ndim(
    ndim: usize,
    axis_4d: usize,
    axis_5d: usize,
    shape: &[usize],
) -> Result<usize, TransformError> {
    match ndim {
        4 => Ok(shape[axis_4d]),
        5 => Ok(shape[axis_5d]),
        _ => Err(TransformError::InvalidShape {
            expected: format!("4D or 5D encoder_input tensor, got {ndim}D"),
            actual: shape.to_vec(),
        }),
    }
}

impl PreprocessedEncoderInputs {
    /// Get the number of channels.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[1].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[2].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn channels(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 1, 2, self.encoder_input.shape())
    }

    /// Get the height of processed images.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[2].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[3].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn height(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 2, 3, self.encoder_input.shape())
    }

    /// Get the width of processed images.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[3].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[4].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn width(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 3, 4, self.encoder_input.shape())
    }
}

/// Trait for model-specific vision preprocessors.
///
/// Each vision model (LLaVA, Qwen-VL, Phi3-Vision, etc.) implements this trait
/// to provide the correct preprocessing pipeline.
pub trait VisionPreProcessor: Send + Sync {
    /// Default normalization mean for this model family.
    fn default_mean(&self) -> [f64; 3];

    /// Default normalization std for this model family.
    fn default_std(&self) -> [f64; 3];

    /// Preprocess a batch of images.
    ///
    /// # Arguments
    /// * `images` - Input images to preprocess
    /// * `config` - Preprocessor configuration from HuggingFace
    ///
    /// # Returns
    /// Preprocessed encoder inputs ready for the model, or an error.
    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError>;

    /// Preprocess one decoded video clip represented as sampled frames.
    ///
    /// Implementations that support video should emit the same primary
    /// `encoder_input` tensor shape used by the image path, plus video-specific
    /// model metadata such as `video_grid_thw`.
    fn preprocess_video(
        &self,
        _frames: &[DynamicImage],
        _config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError(format!(
            "{} does not support video preprocessing",
            self.model_name()
        )))
    }

    /// Preprocess one decoded video clip represented as borrowed RGB frame
    /// buffers. Implementations can override this to avoid materializing
    /// `DynamicImage` objects after media decode.
    fn preprocess_video_rgb(
        &self,
        _frames: &[RgbFrameRef<'_>],
        _config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError(format!(
            "{} does not support RGB video preprocessing",
            self.model_name()
        )))
    }

    /// Calculate the number of vision tokens for a given image size.
    ///
    /// This is used to determine how many placeholder tokens to insert
    /// in the text input before the image has been fully processed.
    ///
    /// # Arguments
    /// * `width` - Image width after preprocessing
    /// * `height` - Image height after preprocessing
    /// * `config` - Preprocessor configuration
    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize;

    /// Get the model family name for identification.
    fn model_name(&self) -> &'static str;

    /// Get the expected image size after preprocessing.
    ///
    /// Some models have fixed sizes, others are dynamic.
    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        config.get_target_size()
    }
}

/// Registry of available vision processors.
pub struct VisionProcessorRegistry {
    processors: HashMap<String, Box<dyn VisionPreProcessor>>,
}

impl VisionProcessorRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            processors: HashMap::new(),
        }
    }

    /// Register a processor for a model pattern.
    pub fn register(&mut self, pattern: impl Into<String>, processor: Box<dyn VisionPreProcessor>) {
        self.processors.insert(pattern.into(), processor);
    }

    /// Find a processor for the given model ID, falling back to model_type.
    ///
    /// Matches by substring containment (case-insensitive).
    pub fn find(
        &self,
        model_id: &str,
        model_type: Option<&str>,
    ) -> Option<&dyn VisionPreProcessor> {
        self.find_in_candidate(model_id)
            .or_else(|| model_type.and_then(|mt| self.find_in_candidate(mt)))
    }

    fn find_in_candidate(&self, candidate: &str) -> Option<&dyn VisionPreProcessor> {
        let candidate = candidate.to_lowercase();
        for (pattern, processor) in &self.processors {
            if candidate.contains(&pattern.to_lowercase()) {
                return Some(processor.as_ref());
            }
        }
        None
    }

    /// Get list of supported model patterns.
    pub fn supported_patterns(&self) -> Vec<&str> {
        self.processors.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for VisionProcessorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl VisionProcessorRegistry {
    /// Create a registry with all built-in processors registered.
    ///
    /// Currently registers:
    /// - `llava-next` -> LlavaNextProcessor
    /// - `llava-1.5` / `llava-v1.5` -> LlavaProcessor
    /// - `qwen2-vl` -> Qwen2VLProcessor
    /// - `qwen2.5-vl` -> Qwen2VLProcessor (same preprocessing as Qwen2-VL)
    /// - `qwen3-vl` -> Qwen3VLProcessor (patch_size=16, [0.5,0.5,0.5] normalization)
    /// - `qwen3.5` / `qwen3_5` -> Qwen3VLProcessor (Qwen3.5 reuses Qwen3-VL preprocessing)
    /// - `phi-3-vision` -> Phi3VisionProcessor (HD transform with 336x336 tiles)
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // LLaVA-NeXT (v1.6+, anyres multi-crop)
        registry.register(
            "llava-next",
            Box::new(super::processors::LlavaNextProcessor::new()),
        );
        registry.register(
            "llava_next",
            Box::new(super::processors::LlavaNextProcessor::new()),
        );
        registry.register(
            "llava-v1.6",
            Box::new(super::processors::LlavaNextProcessor::new()),
        );

        // Standard LLaVA (v1.5, single-patch).
        // Use specific patterns so they don't accidentally match LLaVA-NeXT
        // model IDs like "llava-v1.6-*".
        registry.register(
            "llava-1.5",
            Box::new(super::processors::LlavaProcessor::new()),
        );
        registry.register(
            "llava-v1.5",
            Box::new(super::processors::LlavaProcessor::new()),
        );

        // Register Qwen3-VL first (more specific pattern - must match before qwen2)
        registry.register(
            "qwen3-vl",
            Box::new(super::processors::Qwen3VLProcessor::new()),
        );
        registry.register(
            "qwen3_vl",
            Box::new(super::processors::Qwen3VLProcessor::new()),
        );

        // Qwen3-Omni uses the same patchification and normalization contract
        // as Qwen3-VL for its image and video towers.
        registry.register(
            "qwen3-omni",
            Box::new(super::processors::Qwen3OmniVisionProcessor::new()),
        );
        registry.register(
            "qwen3_omni",
            Box::new(super::processors::Qwen3OmniVisionProcessor::new()),
        );

        // Qwen3.5 family (and Qwen3.6: same arch) reuses Qwen3-VL preprocessing.
        registry.register(
            "qwen3.5",
            Box::new(super::processors::Qwen3VLProcessor::new()),
        );
        registry.register(
            "qwen3_5",
            Box::new(super::processors::Qwen3VLProcessor::new()),
        );
        registry.register(
            "qwen3.6",
            Box::new(super::processors::Qwen3VLProcessor::new()),
        );
        registry.register(
            "qwen3_6",
            Box::new(super::processors::Qwen3VLProcessor::new()),
        );

        // Inkling/TML multimodal model.
        registry.register(
            "inkling",
            Box::new(super::processors::InklingImageProcessor::new()),
        );
        // Register Qwen2-VL (matches Qwen/Qwen2-VL-*, etc.)
        registry.register(
            "qwen2-vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );
        registry.register(
            "qwen2_vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );

        // Register Qwen2.5-VL (uses identical preprocessing to Qwen2-VL)
        registry.register(
            "qwen2.5-vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );
        registry.register(
            "qwen2_5-vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );
        registry.register(
            "qwen2_5_vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );

        // Register Phi3-Vision
        registry.register(
            "phi-3-vision",
            Box::new(super::processors::Phi3VisionProcessor::new()),
        );
        registry.register(
            "phi3-vision",
            Box::new(super::processors::Phi3VisionProcessor::new()),
        );
        registry.register(
            "phi3_v",
            Box::new(super::processors::Phi3VisionProcessor::new()),
        );

        // Register LLaMA 4 Vision
        registry.register(
            "llama-4",
            Box::new(super::processors::Llama4VisionProcessor::new()),
        );
        registry.register(
            "llama4",
            Box::new(super::processors::Llama4VisionProcessor::new()),
        );

        // Register the Kimi MoonViT family. K3 shares the stack but ships a
        // larger patch budget and chessboard alpha compositing, so it gets its
        // own processor rather than reusing K2.5's.
        registry.register(
            "kimi-k2",
            Box::new(super::processors::KimiK25Processor::new()),
        );
        registry.register(
            "kimi_k2",
            Box::new(super::processors::KimiK25Processor::new()),
        );
        registry.register(
            "kimi-k3",
            Box::new(super::processors::KimiK3Processor::new()),
        );
        registry.register(
            "kimi_k3",
            Box::new(super::processors::KimiK3Processor::new()),
        );

        registry
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array4;

    use super::*;
    use crate::vision::processors::LlavaProcessor;

    #[test]
    fn test_preprocessed_encoder_inputs_geometry_accessors() {
        let encoder_input = Array4::<f32>::zeros((2, 3, 336, 336));
        let inputs = PreprocessedEncoderInputs::new(
            encoder_input,
            vec![576, 576],
            vec![(640, 480), (800, 600)],
        );

        assert_eq!(inputs.channels().unwrap(), 3);
        assert_eq!(inputs.height().unwrap(), 336);
        assert_eq!(inputs.width().unwrap(), 336);
    }

    #[test]
    fn test_registry_with_defaults() {
        let registry = VisionProcessorRegistry::with_defaults();

        // Should find LLaVA processor
        assert!(registry.find("llava-hf/llava-1.5-7b-hf", None).is_some());
        assert!(registry.find("liuhaotian/llava-v1.5-7b", None).is_some());

        // Should find LLaVA-NeXT processor
        assert!(registry
            .find("llava-hf/llava-v1.6-mistral-7b-hf", None)
            .is_some());
        assert!(registry
            .find("lmms-lab/llava-next-interleave-qwen-7b", None)
            .is_some());

        // Get the processor and check model name
        let processor = registry.find("llava-hf/llava-1.5-7b-hf", None).unwrap();
        assert_eq!(processor.model_name(), "llava");

        let processor = registry
            .find("org/inkling-chat", None)
            .expect("Inkling model family");
        assert_eq!(processor.model_name(), "inkling");
    }

    #[test]
    fn test_registry_separates_kimi_k25_and_k3() {
        // K3's pixel pipeline differs from K2.5's (larger patch budget, alpha
        // compositing), so resolving a K3 id to the K2.5 processor is a bug.
        let registry = VisionProcessorRegistry::with_defaults();

        for id in ["moonshotai/Kimi-K3", "moonshotai/Kimi_K3-Instruct"] {
            let processor = registry.find(id, None).expect("K3 vision processor");
            assert_eq!(processor.model_name(), "kimi-k3", "{id}");
        }
        for id in ["moonshotai/Kimi-K2.5", "moonshotai/Kimi_K2-VL"] {
            let processor = registry.find(id, None).expect("K2.5 vision processor");
            assert_eq!(processor.model_name(), "kimi-k2.5", "{id}");
        }

        // model_type fallback must split the same way.
        assert_eq!(
            registry
                .find("internal/checkpoint-final", Some("kimi_k3"))
                .expect("K3 by model_type")
                .model_name(),
            "kimi-k3"
        );
        assert_eq!(
            registry
                .find("internal/checkpoint-final", Some("kimi_k2"))
                .expect("K2.5 by model_type")
                .model_name(),
            "kimi-k2.5"
        );
    }

    #[test]
    fn test_registry_find() {
        let mut registry = VisionProcessorRegistry::new();

        // Create a mock processor using LlavaProcessor
        registry.register("test-model", Box::new(LlavaProcessor::new()));

        assert!(registry.find("test-model-7b", None).is_some());
        assert!(registry.find("TEST-MODEL", None).is_some());
        assert!(registry.find("other-model", None).is_none());
    }

    #[test]
    fn test_registry_find_falls_back_to_model_type() {
        let registry = VisionProcessorRegistry::with_defaults();

        assert!(registry.find("custom-model", None).is_none());

        let processor = registry
            .find("custom-model", Some("qwen3_vl"))
            .expect("qwen3 processor by model_type");
        assert_eq!(processor.model_name(), "qwen3-vl");
    }

    #[test]
    fn test_registry_find_preserves_fast_path() {
        let registry = VisionProcessorRegistry::with_defaults();

        let processor = registry
            .find("Qwen3-VL-30B-A3B-Instruct", Some("qwen2_vl"))
            .expect("qwen3 processor by model_id");
        assert_eq!(processor.model_name(), "qwen3-vl");
    }

    #[test]
    fn test_registry_find_phi3_model_type_fallback() {
        let registry = VisionProcessorRegistry::with_defaults();

        let processor = registry
            .find("custom-model", Some("phi3_v"))
            .expect("phi3 processor by model_type");
        assert_eq!(processor.model_name(), "phi3-vision");
    }
}
