use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    audio::{AudioPreProcessor, Qwen3AudioProcessor},
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
    vision::PreProcessorConfig,
};

const IMAGE_PAD_TOKEN: &str = "<|image_pad|>";
const VIDEO_PAD_TOKEN: &str = "<|video_pad|>";
const AUDIO_PAD_TOKEN: &str = "<|audio_pad|>";

pub(super) struct Qwen3OmniSpec;

impl Qwen3OmniSpec {
    fn token_id(metadata: &ModelMetadata, field: &str) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["thinker_config", field])
            .or_else(|| metadata.config_u32(&[field]))
            .map(|value| value as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: format!("thinker_config.{field}"),
            })
    }

    fn token(metadata: &ModelMetadata, field: &str, fallback: &str) -> RegistryResult<String> {
        let token_id = Self::token_id(metadata, field)?;
        if let Some(token) = metadata.tokenizer.id_to_token(token_id as u32) {
            return Ok(token);
        }
        metadata.token_id(fallback)?;
        Ok(fallback.to_string())
    }

    fn replacements(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
        field: &str,
        fallback: &str,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let token_id = Self::token_id(metadata, field)?;
        let token = Self::token(metadata, field, fallback)?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| PromptReplacement::repeated(modality, &token, token_id, count))
            .collect())
    }
}

impl ModelProcessorSpec for Qwen3OmniSpec {
    fn name(&self) -> &'static str {
        "qwen3_omni"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let model_id = metadata.model_id.to_ascii_lowercase();
        model_id.contains("qwen3-omni")
            || model_id.contains("qwen3_omni")
            || metadata.config_model_type().is_some_and(|model_type| {
                model_type == "qwen3_omni_moe" || model_type == "qwen3_omni_moe_thinker"
            })
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        self.placeholder_token_for(metadata, Modality::Image)
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        self.placeholder_token_id_for(metadata, Modality::Image)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => Self::token(metadata, "image_token_id", IMAGE_PAD_TOKEN),
            Modality::Video => Self::token(metadata, "video_token_id", VIDEO_PAD_TOKEN),
            Modality::Audio => Self::token(metadata, "audio_token_id", AUDIO_PAD_TOKEN),
            Modality::ImageEmbeds => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image => Self::token_id(metadata, "image_token_id"),
            Modality::Video => Self::token_id(metadata, "video_token_id"),
            Modality::Audio => Self::token_id(metadata, "audio_token_id"),
            Modality::ImageEmbeds => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([
            (Modality::Image, 10),
            (Modality::Video, 1),
            (Modality::Audio, 10),
        ]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({"use_audio_in_video": false}))
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
        self.prompt_replacements_for(metadata, preprocessed, Modality::Image)
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => Self::replacements(
                metadata,
                preprocessed,
                modality,
                "image_token_id",
                IMAGE_PAD_TOKEN,
            ),
            Modality::Video => Self::replacements(
                metadata,
                preprocessed,
                modality,
                "video_token_id",
                VIDEO_PAD_TOKEN,
            ),
            Modality::Audio => Self::replacements(
                metadata,
                preprocessed,
                modality,
                "audio_token_id",
                AUDIO_PAD_TOKEN,
            ),
            Modality::ImageEmbeds => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            Modality::Image => EncoderFieldLayouts::new(
                FieldLayout::flat("patches_per_image"),
                HashMap::from([
                    ("image_grid_thw".to_string(), FieldLayout::Batched),
                    ("patches_per_image".to_string(), FieldLayout::Batched),
                ]),
            ),
            Modality::Video => EncoderFieldLayouts::new(
                FieldLayout::flat("patches_per_video"),
                HashMap::from([
                    ("video_grid_thw".to_string(), FieldLayout::Batched),
                    ("patches_per_video".to_string(), FieldLayout::Batched),
                    ("video_second_per_grid".to_string(), FieldLayout::Batched),
                ]),
            ),
            Modality::Audio => EncoderFieldLayouts::new(
                FieldLayout::Batched,
                HashMap::from([
                    ("feature_attention_mask".to_string(), FieldLayout::Batched),
                    ("audio_feature_lengths".to_string(), FieldLayout::Batched),
                ]),
            ),
            Modality::ImageEmbeds => EncoderFieldLayouts::default(),
        }
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string(), "video_grid_thw".to_string()]
    }

    fn keep_on_cpu_keys_for(&self, modality: Modality) -> Vec<String> {
        match modality {
            Modality::Image => vec!["image_grid_thw".to_string()],
            Modality::Video => vec!["video_grid_thw".to_string()],
            Modality::Audio | Modality::ImageEmbeds => vec![],
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

    fn omni_tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            (AUDIO_PAD_TOKEN, 151675),
            (IMAGE_PAD_TOKEN, 151655),
            (VIDEO_PAD_TOKEN, 151656),
        ])
    }

    #[test]
    fn omni_accepts_mixed_modalities_and_uses_nested_tokens() {
        let tokenizer = omni_tokenizer();
        let config = json!({
            "model_type": "qwen3_omni_moe",
            "thinker_config": {
                "audio_token_id": 151675,
                "image_token_id": 151655,
                "video_token_id": 151656
            }
        });
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-Omni-30B-A3B-Thinking",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).unwrap();
        assert_eq!(spec.name(), "qwen3_omni");
        assert_eq!(
            spec.placeholder_token(&metadata).unwrap(),
            spec.placeholder_token_for(&metadata, Modality::Image)
                .unwrap()
        );
        assert_eq!(
            spec.placeholder_token_id(&metadata).unwrap(),
            spec.placeholder_token_id_for(&metadata, Modality::Image)
                .unwrap()
        );
        spec.validate_media_request(
            &metadata,
            &[
                (Modality::Image, 2),
                (Modality::Video, 1),
                (Modality::Audio, 2),
            ],
        )
        .unwrap();

        for (modality, expected) in [
            (Modality::Image, 151655),
            (Modality::Video, 151656),
            (Modality::Audio, 151675),
        ] {
            let replacements = spec
                .prompt_replacements_for(
                    &metadata,
                    &test_preprocessed_with_tokens(&[ImageSize::new(32, 32)], &[3]),
                    modality,
                )
                .unwrap();
            assert_eq!(replacements[0].tokens, vec![expected; 3]);
        }
    }

    #[test]
    fn omni_layouts_are_modality_specific() {
        let image = Qwen3OmniSpec.encoder_field_layouts_for(Modality::Image);
        assert_eq!(image.encoder_input, FieldLayout::flat("patches_per_image"));
        assert!(image.model_specific.contains_key("image_grid_thw"));

        let video = Qwen3OmniSpec.encoder_field_layouts_for(Modality::Video);
        assert_eq!(video.encoder_input, FieldLayout::flat("patches_per_video"));
        assert!(video.model_specific.contains_key("video_grid_thw"));

        let audio = Qwen3OmniSpec.encoder_field_layouts_for(Modality::Audio);
        assert_eq!(audio.encoder_input, FieldLayout::Batched);
        assert!(audio.model_specific.contains_key("feature_attention_mask"));
    }

    #[test]
    fn omni_keep_on_cpu_keys_are_modality_specific() {
        assert_eq!(
            Qwen3OmniSpec.keep_on_cpu_keys_for(Modality::Image),
            vec!["image_grid_thw"]
        );
        assert_eq!(
            Qwen3OmniSpec.keep_on_cpu_keys_for(Modality::Video),
            vec!["video_grid_thw"]
        );
        assert!(Qwen3OmniSpec
            .keep_on_cpu_keys_for(Modality::Audio)
            .is_empty());
    }

    #[test]
    fn omni_spec_builds_qwen_audio_processor() {
        use std::sync::Arc;

        use bytes::Bytes;

        use crate::{
            audio::DecodedAudio,
            types::{AudioClip, AudioSource},
            vision::PreProcessorConfig,
        };

        let tokenizer = omni_tokenizer();
        let config = json!({"model_type": "qwen3_omni_moe"});
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-Omni-30B-A3B-Thinking",
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
            .expect("qwen3_omni spec must provide an audio processor");

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
