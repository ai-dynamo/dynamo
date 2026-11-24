use std::collections::HashMap;

use llm_tokenizer::Encoding;
use serde_json::{json, Value};

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
};

/// Structural tokens wrapping one Kimi-K3 image, from the checkpoint's
/// `kimi_k3_vision_processing.py::make_image_prompt`:
/// `<|media_begin|>image {width}x{height}<|media_content|><|media_pad|><|media_end|>`.
const MEDIA_BEGIN: &str = "<|media_begin|>";
const MEDIA_CONTENT: &str = "<|media_content|>";
const MEDIA_END: &str = "<|media_end|>";

/// Kimi-K3.
///
/// Shares K2.5's MoonViT transport layout and `<|media_pad|>` fill token, but
/// not its prompt shape: K3 wraps each image in a block carrying the pre-resize
/// dimensions, while K2.5's chat template emits its own dimensionless wrapper.
///
/// That block cannot be built while rendering — the chat template runs before
/// any media is fetched, so the dimensions do not exist yet. It is built here
/// instead, from the sizes the preprocessor reports, as vLLM does in
/// `kimi_k3.py::_get_prompt_updates`.
pub(super) struct KimiK3VisionSpec;

impl KimiK3VisionSpec {
    /// The repeated pad token (`<|media_pad|>`) — `media_placeholder_token_id` in config.
    fn pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["media_placeholder_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "media_placeholder_token_id".to_string(),
            })
    }

    /// Encode ordinary text into token ids.
    ///
    /// The dimension text sits between two special tokens, which are hard
    /// segment boundaries for the encoder, so encoding it alone yields the same
    /// ids as the reference's one-shot encoding of the whole block.
    fn encode_plain_text(metadata: &ModelMetadata, text: &str) -> RegistryResult<Vec<TokenId>> {
        let encoding = metadata.tokenizer.encode(text, false).map_err(|_| {
            ModelRegistryError::TextEncodingFailed {
                spec: "kimi_k3",
                text: text.to_string(),
            }
        })?;
        Ok(match encoding {
            Encoding::Hf(inner) => inner.get_ids().iter().map(|&id| id as TokenId).collect(),
            Encoding::Plain(ids) | Encoding::Tiktoken(ids) => {
                ids.into_iter().map(|id| id as TokenId).collect()
            }
        })
    }
}

impl ModelProcessorSpec for KimiK3VisionSpec {
    fn name(&self) -> &'static str {
        "kimi_k3"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("kimi") && id.contains("k3")
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "kimi_k3")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|media_pad|>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::pad_token_id(metadata)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 10)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let pad_token_id = Self::pad_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        let media_begin = metadata.token_id(MEDIA_BEGIN)?;
        let media_content = metadata.token_id(MEDIA_CONTENT)?;
        let media_end = metadata.token_id(MEDIA_END)?;

        // `item_sizes` is the decoded `(width, height)` before any resize — the
        // pair the reference prints. The caller already checks both vectors
        // against the media count, so a short zip cannot reach here.
        preprocessed
            .feature_token_counts
            .iter()
            .zip(&preprocessed.item_sizes)
            .map(|(&num_tokens, &(width, height))| {
                let dims = Self::encode_plain_text(metadata, &format!("image {width}x{height}"))?;
                let mut tokens = Vec::with_capacity(dims.len() + num_tokens + 3);
                tokens.push(media_begin);
                tokens.extend(dims);
                tokens.push(media_content);
                // Only the pad run holds encoder features; the wrapper is text.
                let feature_offset = tokens.len();
                tokens.extend(std::iter::repeat_n(pad_token_id, num_tokens));
                tokens.push(media_end);

                Ok(
                    PromptReplacement::sequence(Modality::Image, &placeholder_token, tokens)
                        .with_feature_span(feature_offset, num_tokens),
                )
            })
            .collect()
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // MoonViT patchification, same transport layout as K2.5:
        // encoder_input is [total_patches, patch_features], split by patches_per_image.
        // grid_thws is [num_images, 3] with (temporal, height, width) grid dimensions.
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("grid_thws".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["grid_thws".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        encoder_inputs::PreprocessedEncoderInputs,
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::{Modality, PlaceholderRange, TokenId},
    };

    /// Wrapper token ids as the K3 checkpoint assigns them.
    const MEDIA_BEGIN_ID: u32 = 163602;
    const MEDIA_CONTENT_ID: u32 = 163603;
    const MEDIA_END_ID: u32 = 163604;
    const MEDIA_PAD_ID: u32 = 163605;
    /// Byte-encoder offset, chosen so text ids cannot collide with media ids.
    const TEXT_BASE: u32 = 1000;

    fn k3_tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            ("<|media_begin|>", MEDIA_BEGIN_ID),
            ("<|media_content|>", MEDIA_CONTENT_ID),
            ("<|media_end|>", MEDIA_END_ID),
            ("<|media_pad|>", MEDIA_PAD_ID),
        ])
        .with_byte_encoder(TEXT_BASE)
    }

    fn k3_config() -> serde_json::Value {
        json!({
            "model_type": "kimi_k3",
            "media_placeholder_token_id": MEDIA_PAD_ID,
        })
    }

    /// `(width, height)` per item, matching MoonViT's `item_sizes` contract.
    fn preprocessed(
        sizes: &[(u32, u32)],
        feature_token_counts: &[usize],
    ) -> PreprocessedEncoderInputs {
        PreprocessedEncoderInputs::new(
            ndarray::Array4::<f32>::zeros((1, 3, 14, 14)),
            feature_token_counts.to_vec(),
            sizes.to_vec(),
        )
    }

    fn text_ids(text: &str) -> Vec<TokenId> {
        text.bytes()
            .map(|b| (TEXT_BASE + u32::from(b)) as TokenId)
            .collect()
    }

    #[test]
    fn kimi_k3_matches_model_id_and_model_type() {
        let tokenizer = k3_tokenizer();
        let config = k3_config();
        let registry = ModelRegistry::new();

        let metadata = ModelMetadata {
            model_id: "moonshotai/Kimi-K3",
            tokenizer: &tokenizer,
            config: &config,
        };
        assert_eq!(
            registry.lookup(&metadata).expect("k3 spec").name(),
            "kimi_k3"
        );

        // Also match by model_type alone (id without a k3 hint).
        let metadata_by_type = ModelMetadata {
            model_id: "internal/checkpoint-final",
            tokenizer: &tokenizer,
            config: &config,
        };
        assert_eq!(
            registry.lookup(&metadata_by_type).expect("k3 spec").name(),
            "kimi_k3"
        );
    }

    #[test]
    fn kimi_k3_emits_the_reference_media_wrapper() {
        let tokenizer = k3_tokenizer();
        let config = k3_config();
        let metadata = ModelMetadata {
            model_id: "moonshotai/Kimi-K3",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("k3 spec");

        let replacements = spec
            .prompt_replacements(&metadata, &preprocessed(&[(1024, 768)], &[4]))
            .unwrap();

        assert_eq!(replacements.len(), 1);
        let rep = &replacements[0];
        assert_eq!(rep.modality, Modality::Image);
        assert_eq!(rep.placeholder_token, "<|media_pad|>");

        let mut expected = vec![MEDIA_BEGIN_ID as TokenId];
        expected.extend(text_ids("image 1024x768"));
        expected.push(MEDIA_CONTENT_ID as TokenId);
        expected.extend([MEDIA_PAD_ID as TokenId; 4]);
        expected.push(MEDIA_END_ID as TokenId);
        assert_eq!(rep.tokens, expected);

        // Only the pad run is an encoder-feature position; the wrapper is text.
        // Pads start after `<|media_begin|>`, the dims, and `<|media_content|>`.
        assert_eq!(
            rep.feature_ranges,
            Some(vec![PlaceholderRange {
                offset: 2 + "image 1024x768".len(),
                length: 4,
            }])
        );
    }

    #[test]
    fn kimi_k3_dimensions_are_per_image() {
        let tokenizer = k3_tokenizer();
        let config = k3_config();
        let metadata = ModelMetadata {
            model_id: "moonshotai/Kimi-K3",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("k3 spec");

        let replacements = spec
            .prompt_replacements(
                &metadata,
                &preprocessed(&[(4000, 3000), (224, 448)], &[8, 2]),
            )
            .unwrap();

        assert_eq!(replacements.len(), 2);
        for (rep, (text, pads)) in replacements
            .iter()
            .zip([("image 4000x3000", 8usize), ("image 224x448", 2)])
        {
            let mut expected = vec![MEDIA_BEGIN_ID as TokenId];
            expected.extend(text_ids(text));
            expected.push(MEDIA_CONTENT_ID as TokenId);
            expected.extend(std::iter::repeat_n(MEDIA_PAD_ID as TokenId, pads));
            expected.push(MEDIA_END_ID as TokenId);
            assert_eq!(rep.tokens, expected);
            assert_eq!(
                rep.feature_ranges,
                Some(vec![PlaceholderRange {
                    offset: 2 + text.len(),
                    length: pads,
                }])
            );
        }
    }

    #[test]
    fn kimi_k3_requires_the_media_tokens_in_the_vocabulary() {
        // A checkpoint without the structural tokens must fail loudly rather
        // than silently emit a bare pad run.
        let tokenizer = TestTokenizer::new(&[("<|media_pad|>", MEDIA_PAD_ID)]);
        let config = k3_config();
        let metadata = ModelMetadata {
            model_id: "moonshotai/Kimi-K3",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("k3 spec");

        let err = spec
            .prompt_replacements(&metadata, &preprocessed(&[(64, 64)], &[1]))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "token '<|media_begin|>' not found in tokenizer vocabulary"
        );
    }
}
