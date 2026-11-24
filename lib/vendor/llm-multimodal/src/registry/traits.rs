use std::collections::HashMap;

use llm_tokenizer::TokenizerTrait;
use serde_json::Value;
use thiserror::Error;

use crate::{
    audio::AudioPreProcessor,
    encoder_inputs::PreprocessedEncoderInputs,
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
    vision::PreProcessorConfig,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModelRegistryError {
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("token '{token}' not found in tokenizer vocabulary")]
    TokenNotFound { token: String },
    #[error("missing config field '{field}'")]
    MissingConfigField { field: String },
    #[error("model spec {spec} could not encode '{text}' with the model tokenizer")]
    TextEncodingFailed { spec: &'static str, text: String },
    #[error("modality {modality} is not supported by model spec {spec}")]
    UnsupportedModality {
        spec: &'static str,
        modality: Modality,
    },
    #[error("model spec {spec} supports at most {limit} {modality} inputs; got {requested}")]
    ModalityLimitExceeded {
        spec: &'static str,
        modality: Modality,
        limit: usize,
        requested: usize,
    },
    #[error("modality {modality} appears more than once in the request for model spec {spec}")]
    DuplicateModality {
        spec: &'static str,
        modality: Modality,
    },
}

pub type RegistryResult<T> = Result<T, ModelRegistryError>;

/// Ordering of media and text parts when rendering a multipart message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaPartOrder {
    /// Emit media parts before text (vLLM `interleave_mm_strings=false`).
    MediaFirst,
    /// Keep parts in request order; part order is protocol-visible.
    Authored,
}

/// Metadata about the current model used to derive tokenizer/config dependent fields.
pub struct ModelMetadata<'a> {
    pub model_id: &'a str,
    pub tokenizer: &'a dyn TokenizerTrait,
    pub config: &'a Value,
}

impl<'a> ModelMetadata<'a> {
    pub fn token_id(&self, token: &str) -> RegistryResult<TokenId> {
        self.tokenizer
            .token_to_id(token)
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::TokenNotFound {
                token: token.to_string(),
            })
    }

    pub fn config_u32(&self, path: &[&str]) -> Option<u32> {
        Self::find_value(self.config, path).and_then(|value| value.as_u64().map(|v| v as u32))
    }

    pub fn config_model_type(&self) -> Option<&str> {
        Self::find_value(self.config, &["model_type"]).and_then(Value::as_str)
    }

    fn find_value<'v>(value: &'v Value, path: &[&str]) -> Option<&'v Value> {
        let mut current = value;
        for key in path {
            current = current.get(*key)?;
        }
        Some(current)
    }
}

pub trait ModelProcessorSpec: Send + Sync {
    fn name(&self) -> &'static str;
    fn matches(&self, metadata: &ModelMetadata) -> bool;

    /// Ordering of media/text parts this model's chat template expects. Most
    /// models use vLLM-compatible media-first; families whose template renders
    /// parts positionally override to `Authored`.
    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::MediaFirst
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String>;
    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId>;
    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            _ => Err(ModelRegistryError::UnsupportedModality {
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
            Modality::Image => self.placeholder_token_id(metadata),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }
    fn modality_limits(&self, metadata: &ModelMetadata)
        -> RegistryResult<HashMap<Modality, usize>>;

    /// Validate the active modalities and item counts in one media request.
    ///
    /// Any subset of the modalities declared by [`Self::modality_limits`] is
    /// accepted. Each modality may appear once in `requested`; zero-count
    /// entries are ignored.
    fn validate_media_request(
        &self,
        metadata: &ModelMetadata,
        requested: &[(Modality, usize)],
    ) -> RegistryResult<()> {
        let limits = self.modality_limits(metadata)?;
        let mut active = Vec::with_capacity(requested.len());

        for &(modality, count) in requested {
            if count == 0 {
                continue;
            }
            if active.contains(&modality) {
                return Err(ModelRegistryError::DuplicateModality {
                    spec: self.name(),
                    modality,
                });
            }
            active.push(modality);

            let Some(&limit) = limits.get(&modality) else {
                return Err(ModelRegistryError::UnsupportedModality {
                    spec: self.name(),
                    modality,
                });
            };
            if count > limit {
                return Err(ModelRegistryError::ModalityLimitExceeded {
                    spec: self.name(),
                    modality,
                    limit,
                    requested: count,
                });
            }
        }

        Ok(())
    }

    fn processor_kwargs(&self, metadata: &ModelMetadata) -> RegistryResult<Value>;

    /// Build the audio preprocessor for this model, if it supports audio.
    ///
    /// This is the single source of truth for audio-processor selection: the
    /// same spec that owns a model's prompt/placeholder logic also owns its
    /// audio preprocessor factory, so there is no separate string-keyed
    /// registry to keep in sync. Audio-less specs use the default (`None`).
    ///
    /// The processor is built from the current model config because its feature
    /// shapes and quantization parameters can be checkpoint-specific.
    fn audio_processor(
        &self,
        _model_config: &Value,
        _preprocessor_config: &PreProcessorConfig,
    ) -> Option<Box<dyn AudioPreProcessor>> {
        None
    }

    /// Compute per-media prompt replacement token sequences.
    ///
    /// Receives the full preprocessed output so each model can extract whatever
    /// metadata it needs (e.g. aspect_ratios for tile-based models).  This
    /// mirrors vLLM's `_get_prompt_updates(out_mm_kwargs)` pattern.
    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>>;
    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => self.prompt_replacements(metadata, preprocessed),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    /// Declare how each tensor's first dimension maps to media items.
    ///
    /// Keys not listed are treated as shared (replicated across all media items).
    /// The `"pixel_values"` key mirrors HF/vLLM vision kwargs and should be
    /// included when the primary encoder input differs from batched layout.
    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // Default: encoder_input is batched (most models).
        HashMap::from([("pixel_values".to_string(), FieldLayout::Batched)])
    }

    /// Declare the neutral primary/side-tensor layout contract for one modality.
    ///
    /// The default converts the legacy HF/vLLM-shaped field map so existing
    /// vision specs remain source-compatible. New multimodal specs should
    /// override this method and keep backend-specific field names at adapters.
    fn encoder_field_layouts_for(&self, _modality: Modality) -> EncoderFieldLayouts {
        EncoderFieldLayouts::from_legacy_fields(self.field_layouts())
    }

    /// Tensor keys that should remain on CPU (not transferred to GPU).
    ///
    /// In vLLM, certain model-specific tensors are marked `keep_on_cpu=True`
    /// in their `MultiModalFieldConfig`.  This method mirrors that per-model
    /// knowledge so the router can send the hint via gRPC, avoiding the need
    /// for the backend to instantiate a Python processor just to query it.
    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![]
    }

    /// Tensor keys that should remain on CPU for one modality.
    ///
    /// The default preserves the legacy model-wide declaration.
    fn keep_on_cpu_keys_for(&self, _modality: Modality) -> Vec<String> {
        self.keep_on_cpu_keys()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::registry::test_helpers::TestTokenizer;

    struct TestSpec;

    impl ModelProcessorSpec for TestSpec {
        fn name(&self) -> &'static str {
            "test"
        }

        fn matches(&self, _metadata: &ModelMetadata) -> bool {
            true
        }

        fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
            Ok("<image>".to_string())
        }

        fn placeholder_token_id(&self, _metadata: &ModelMetadata) -> RegistryResult<TokenId> {
            Ok(1)
        }

        fn modality_limits(
            &self,
            _metadata: &ModelMetadata,
        ) -> RegistryResult<HashMap<Modality, usize>> {
            Ok(HashMap::from([(Modality::Image, 2), (Modality::Audio, 1)]))
        }

        fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
            Ok(json!({}))
        }

        fn prompt_replacements(
            &self,
            _metadata: &ModelMetadata,
            _preprocessed: &PreprocessedEncoderInputs,
        ) -> RegistryResult<Vec<PromptReplacement>> {
            Ok(vec![])
        }
    }

    fn validate(
        spec: &dyn ModelProcessorSpec,
        requested: &[(Modality, usize)],
    ) -> RegistryResult<()> {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({});
        let metadata = ModelMetadata {
            model_id: "test-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        spec.validate_media_request(&metadata, requested)
    }

    #[test]
    fn validation_accepts_any_declared_modality_subset() {
        assert_eq!(validate(&TestSpec, &[(Modality::Image, 2)]), Ok(()));
        assert_eq!(
            validate(&TestSpec, &[(Modality::Image, 1), (Modality::Audio, 1)]),
            Ok(())
        );
    }

    #[test]
    fn validation_rejects_undeclared_modality() {
        assert_eq!(
            validate(&TestSpec, &[(Modality::Video, 1)]),
            Err(ModelRegistryError::UnsupportedModality {
                spec: "test",
                modality: Modality::Video,
            })
        );
    }

    #[test]
    fn validation_rejects_count_above_limit() {
        assert_eq!(
            validate(&TestSpec, &[(Modality::Image, 3)]),
            Err(ModelRegistryError::ModalityLimitExceeded {
                spec: "test",
                modality: Modality::Image,
                limit: 2,
                requested: 3,
            })
        );
    }

    #[test]
    fn validation_rejects_duplicate_modality_counts() {
        assert_eq!(
            validate(&TestSpec, &[(Modality::Image, 1), (Modality::Image, 1)]),
            Err(ModelRegistryError::DuplicateModality {
                spec: "test",
                modality: Modality::Image,
            })
        );
    }
}
