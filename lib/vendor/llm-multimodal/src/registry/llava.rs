use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
};

pub(super) struct LlavaSpec;
pub(super) struct LlavaNextSpec;

impl ModelProcessorSpec for LlavaSpec {
    fn name(&self) -> &'static str {
        "llava"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        // Match plain "llava" but not "llava_next" (handled by LlavaNextSpec).
        let model_type = metadata.config_model_type();
        if model_type.is_some_and(|mt| mt == "llava_next") {
            return false;
        }
        let model_id_lower = metadata.model_id.to_ascii_lowercase();
        if model_id_lower.contains("llava-next") || model_id_lower.contains("llava_next") {
            return false;
        }
        model_id_lower.contains("llava") || model_type.is_some_and(|mt| mt == "llava")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<image>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        if let Some(value) = metadata.config_u32(&["image_token_index"]) {
            return Ok(value as TokenId);
        }
        metadata.token_id("<image>")
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 4)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let token_id = self.placeholder_token_id(metadata)?;
        let token = self.placeholder_token(metadata)?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| PromptReplacement::repeated(Modality::Image, &token, token_id, count))
            .collect())
    }
}

impl ModelProcessorSpec for LlavaNextSpec {
    fn name(&self) -> &'static str {
        "llava_next"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata
            .config_model_type()
            .is_some_and(|mt| mt == "llava_next")
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        LlavaSpec.placeholder_token(metadata)
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        LlavaSpec.placeholder_token_id(metadata)
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        LlavaSpec.modality_limits(metadata)
    }

    fn processor_kwargs(&self, metadata: &ModelMetadata) -> RegistryResult<Value> {
        LlavaSpec.processor_kwargs(metadata)
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        // LLaVA-Next token counts differ from plain LLaVA because of
        // anyres multi-crop + spatial_unpad.  The correct per-image counts
        // are already computed by LlavaNextProcessor::calculate_num_tokens
        // and stored in preprocessed.feature_token_counts.
        let token_id = LlavaSpec.placeholder_token_id(metadata)?;
        let token = LlavaSpec.placeholder_token(metadata)?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| PromptReplacement::repeated(Modality::Image, &token, token_id, count))
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // encoder_input is [num_images, max_patches, C, H, W] (5D, batched).
        // image_sizes is [num_images, 2] (batched), matching HF/vLLM kwargs.
        HashMap::from([
            ("pixel_values".to_string(), FieldLayout::Batched),
            ("image_sizes".to_string(), FieldLayout::Batched),
        ])
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::ImageSize,
    };

    #[test]
    fn llava_prompt_replacement_uses_preprocessed_tokens() {
        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({
            "model_type": "llava",
            "image_token_index": 32000,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "llava-v1.5",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llava spec");
        // Token count comes from preprocessed.feature_token_counts (set by
        // LlavaProcessor::calculate_num_tokens), not from image dimensions.
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(336, 336)], &[576]);
        let replacements = spec.prompt_replacements(&metadata, &preprocessed).unwrap();
        assert_eq!(replacements[0].tokens.len(), 576);
    }

    #[test]
    fn llava_prompt_replacement_uses_per_image_token_counts() {
        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({
            "model_type": "llava",
            "image_token_index": 32000,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "llava-v1.5",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llava spec");
        let preprocessed = test_preprocessed_with_tokens(
            &[ImageSize::new(336, 336), ImageSize::new(448, 448)],
            &[576, 1024],
        );

        let replacements = spec.prompt_replacements(&metadata, &preprocessed).unwrap();

        assert_eq!(replacements.len(), 2);
        assert_eq!(replacements[0].tokens.len(), 576);
        assert_eq!(replacements[1].tokens.len(), 1024);
    }

    #[test]
    fn llava_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({
            "model_type": "llava",
            "image_token_index": 32000,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llava alias");
        assert_eq!(spec.name(), "llava");
    }

    #[test]
    fn llava_spec_has_no_audio_processor() {
        use crate::vision::PreProcessorConfig;

        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({"model_type": "llava", "image_token_index": 32000});
        let metadata = ModelMetadata {
            model_id: "llava-v1.5",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llava spec");
        assert!(spec
            .audio_processor(&config, &PreProcessorConfig::default())
            .is_none());
    }
}
