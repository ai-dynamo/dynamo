use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
};

pub(super) struct Phi3VisionSpec;

impl ModelProcessorSpec for Phi3VisionSpec {
    fn name(&self) -> &'static str {
        "phi3_v"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("phi") && id.contains("vision")
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "phi3_v")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|image|>".to_owned())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id("<|image|>")
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

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            ("pixel_values".to_string(), FieldLayout::Batched),
            ("image_sizes".to_string(), FieldLayout::Batched),
        ])
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::ImageSize,
    };

    #[test]
    fn phi3_uses_num_img_tokens() {
        let tokenizer = TestTokenizer::new(&[("<|image|>", 555)]);
        let config = json!({
            "model_type": "phi3_v",
            "img_processor": {"num_img_tokens": 144}
        });
        let metadata = ModelMetadata {
            model_id: "Phi-3-vision",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("phi3 spec");
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(336, 336)], &[144]),
            )
            .unwrap();
        assert_eq!(replacements[0].tokens.len(), 144);
        assert_eq!(replacements[0].tokens[0], 555);
    }

    #[test]
    fn phi3_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image|>", 555)]);
        let config = json!({
            "model_type": "phi3_v",
            "img_processor": {"num_img_tokens": 144}
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("phi3 alias");
        assert_eq!(spec.name(), "phi3_v");
    }
}
