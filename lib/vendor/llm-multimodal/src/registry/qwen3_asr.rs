use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    audio::{AudioPreProcessor, Qwen3AudioProcessor},
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
    vision::PreProcessorConfig,
};

const AUDIO_PAD_TOKEN: &str = "<|audio_pad|>";

pub(super) struct Qwen3AsrSpec;

impl Qwen3AsrSpec {
    fn audio_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["thinker_config", "audio_token_id"])
            .or_else(|| metadata.config_u32(&["audio_token_id"]))
            .map(|value| value as TokenId)
            .map_or_else(|| metadata.token_id(AUDIO_PAD_TOKEN), Ok)
    }
}

impl ModelProcessorSpec for Qwen3AsrSpec {
    fn name(&self) -> &'static str {
        "qwen3_asr"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let model_id = metadata.model_id.to_ascii_lowercase();
        model_id.contains("qwen3-asr")
            || model_id.contains("qwen3_asr")
            || metadata
                .config_model_type()
                .is_some_and(|model_type| model_type == "qwen3_asr")
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        self.placeholder_token_for(metadata, Modality::Audio)
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        self.placeholder_token_id_for(metadata, Modality::Audio)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Audio => {
                let token_id = Self::audio_token_id(metadata)?;
                match metadata.tokenizer.id_to_token(token_id as u32) {
                    Some(token) => Ok(token),
                    None => {
                        metadata.token_id(AUDIO_PAD_TOKEN)?;
                        Ok(AUDIO_PAD_TOKEN.to_string())
                    }
                }
            }
            Modality::Image | Modality::Video | Modality::ImageEmbeds => {
                Err(ModelRegistryError::UnsupportedModality {
                    spec: self.name(),
                    modality,
                })
            }
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Audio => Self::audio_token_id(metadata),
            Modality::Image | Modality::Video | Modality::ImageEmbeds => {
                Err(ModelRegistryError::UnsupportedModality {
                    spec: self.name(),
                    modality,
                })
            }
        }
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Audio, 10)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn audio_processor(
        &self,
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
    ) -> Option<Box<dyn AudioPreProcessor>> {
        Some(Box::new(Qwen3AudioProcessor::from_configs(
            model_config,
            preprocessor_config,
        )))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        self.prompt_replacements_for(metadata, preprocessed, Modality::Audio)
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Audio => {
                let token_id = Self::audio_token_id(metadata)?;
                let token = self.placeholder_token_for(metadata, Modality::Audio)?;
                Ok(preprocessed
                    .feature_token_counts
                    .iter()
                    .map(|&count| {
                        PromptReplacement::repeated(Modality::Audio, &token, token_id, count)
                    })
                    .collect())
            }
            Modality::Image | Modality::Video | Modality::ImageEmbeds => {
                Err(ModelRegistryError::UnsupportedModality {
                    spec: self.name(),
                    modality,
                })
            }
        }
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            Modality::Audio => EncoderFieldLayouts::new(
                FieldLayout::Batched,
                HashMap::from([
                    ("feature_attention_mask".to_string(), FieldLayout::Batched),
                    ("audio_feature_lengths".to_string(), FieldLayout::Batched),
                ]),
            ),
            Modality::Image | Modality::Video | Modality::ImageEmbeds => {
                EncoderFieldLayouts::default()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        registry::{test_helpers::*, ModelRegistry},
        types::ImageSize,
    };

    #[test]
    fn asr_matches_and_expands_nested_audio_token() {
        let tokenizer = TestTokenizer::new(&[(AUDIO_PAD_TOKEN, 151676)]);
        let config = json!({
            "model_type": "qwen3_asr",
            "thinker_config": {"audio_token_id": 151676}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-ASR-1.7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).unwrap();
        assert_eq!(spec.name(), "qwen3_asr");
        assert_eq!(
            spec.placeholder_token(&metadata).unwrap(),
            spec.placeholder_token_for(&metadata, Modality::Audio)
                .unwrap()
        );
        assert_eq!(
            spec.placeholder_token_id(&metadata).unwrap(),
            spec.placeholder_token_id_for(&metadata, Modality::Audio)
                .unwrap()
        );

        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(128, 100)], &[13]),
                Modality::Audio,
            )
            .unwrap();
        assert_eq!(replacements[0].tokens, vec![151676; 13]);
        assert_eq!(
            spec.encoder_field_layouts_for(Modality::Audio)
                .encoder_input,
            FieldLayout::Batched
        );
        assert_eq!(
            spec.modality_limits(&metadata).unwrap(),
            HashMap::from([(Modality::Audio, 10)])
        );
    }

    #[test]
    fn asr_spec_builds_qwen_audio_processor() {
        use std::sync::Arc;

        use bytes::Bytes;

        use crate::{
            audio::DecodedAudio,
            types::{AudioClip, AudioSource},
        };

        let tokenizer = TestTokenizer::new(&[(AUDIO_PAD_TOKEN, 151676)]);
        let config = json!({"model_type": "qwen3_asr"});
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-ASR-1.7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).unwrap();

        let preprocessor_config = PreProcessorConfig::from_json(
            r#"{"feature_size": 16, "sampling_rate": 16000, "n_fft": 400, "hop_length": 160}"#,
        )
        .unwrap();
        let processor = spec
            .audio_processor(&config, &preprocessor_config)
            .expect("qwen3_asr spec must provide an audio processor");

        let clip = Arc::new(AudioClip::new(
            Bytes::from_static(b"audio"),
            DecodedAudio {
                samples: vec![0.0; 800],
                sample_rate: 16_000,
            },
            AudioSource::InlineBytes,
            "audio-hash".to_string(),
        ));
        let result = processor.preprocess(&[clip]).unwrap();
        assert_eq!(result.encoder_input.shape(), &[1, 16, 5]);
    }
}
