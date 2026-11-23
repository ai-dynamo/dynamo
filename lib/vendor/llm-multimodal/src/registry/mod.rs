mod inkling;
mod kimi_k25;
mod kimi_k3;
mod llama4;
mod llava;
mod phi3_v;
mod qwen3_asr;
mod qwen3_omni;
mod qwen3_vl;
mod qwen_vl;
mod traits;

use inkling::InklingSpec;
use kimi_k25::KimiK25VisionSpec;
use kimi_k3::KimiK3VisionSpec;
use llama4::Llama4Spec;
use llava::{LlavaNextSpec, LlavaSpec};
use once_cell::sync::Lazy;
use phi3_v::Phi3VisionSpec;
use qwen3_asr::Qwen3AsrSpec;
use qwen3_omni::Qwen3OmniSpec;
use qwen3_vl::Qwen3VLVisionSpec;
use qwen_vl::QwenVLVisionSpec;
// Re-export public API from traits.
pub use traits::{
    MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
};

pub struct ModelRegistry {
    specs: Vec<LazySpec>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            specs: vec![
                LazySpec::new(|| Box::new(InklingSpec)),
                // Kimi-K3 must be registered before Kimi-K2.5: the two families
                // share a transport layout but not a prompt shape.
                LazySpec::new(|| Box::new(KimiK3VisionSpec)),
                LazySpec::new(|| Box::new(KimiK25VisionSpec)),
                LazySpec::new(|| Box::new(Llama4Spec)),
                // LlavaNext must be registered before Llava so "llava_next" model_type matches first.
                LazySpec::new(|| Box::new(LlavaNextSpec)),
                LazySpec::new(|| Box::new(LlavaSpec)),
                LazySpec::new(|| Box::new(Qwen3AsrSpec)),
                LazySpec::new(|| Box::new(Qwen3OmniSpec)),
                // Qwen3-VL must be registered before QwenVL so "qwen3" matches first.
                LazySpec::new(|| Box::new(Qwen3VLVisionSpec)),
                LazySpec::new(|| Box::new(QwenVLVisionSpec)),
                LazySpec::new(|| Box::new(Phi3VisionSpec)),
            ],
        }
    }

    pub fn lookup<'a>(&'a self, metadata: &ModelMetadata) -> Option<&'a dyn ModelProcessorSpec> {
        for spec in &self.specs {
            let spec_ref = spec.get();
            if spec_ref.matches(metadata) {
                return Some(spec_ref);
            }
        }
        None
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

struct LazySpec {
    inner: Lazy<Box<dyn ModelProcessorSpec>>,
}

impl LazySpec {
    fn new(factory: fn() -> Box<dyn ModelProcessorSpec>) -> Self {
        Self {
            inner: Lazy::new(factory),
        }
    }

    fn get(&self) -> &dyn ModelProcessorSpec {
        self.inner.as_ref()
    }
}

#[cfg(test)]
pub(super) mod test_helpers {
    use std::collections::HashMap;

    use llm_tokenizer::{Decoder, Encoder, Encoding, SpecialTokens, TokenizerTrait};
    use once_cell::sync::Lazy;

    use crate::{
        encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
        types::ImageSize,
    };

    pub struct TestTokenizer {
        vocab: HashMap<String, u32>,
        text_base: Option<u32>,
    }

    impl TestTokenizer {
        pub fn new(pairs: &[(&str, u32)]) -> Self {
            let vocab = pairs
                .iter()
                .map(|(token, id)| ((*token).to_string(), *id))
                .collect();
            Self {
                vocab,
                text_base: None,
            }
        }

        /// Encode text as one id per byte, offset by `base`.
        ///
        /// Off by default so specs that only look up special tokens are
        /// unaffected; specs that splice encoded text into a replacement
        /// enable it to assert the exact layout.
        pub fn with_byte_encoder(mut self, base: u32) -> Self {
            self.text_base = Some(base);
            self
        }
    }

    impl Encoder for TestTokenizer {
        fn encode(&self, input: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
            let Some(base) = self.text_base else {
                return Ok(Encoding::Plain(Vec::new()));
            };
            Ok(Encoding::Plain(
                input.bytes().map(|b| base + u32::from(b)).collect(),
            ))
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
            add_special_tokens: bool,
        ) -> anyhow::Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|input| self.encode(input, add_special_tokens))
                .collect()
        }
    }

    impl Decoder for TestTokenizer {
        fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    impl TokenizerTrait for TestTokenizer {
        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            static TOKENS: Lazy<SpecialTokens> = Lazy::new(|| SpecialTokens {
                bos_token: None,
                eos_token: None,
                unk_token: None,
                sep_token: None,
                pad_token: None,
                cls_token: None,
                mask_token: None,
                additional_special_tokens: vec![],
            });
            &TOKENS
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.vocab
                .iter()
                .find(|(_, &v)| v == id)
                .map(|(k, _)| k.clone())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    pub fn test_preprocessed_with_tokens(
        item_sizes: &[ImageSize],
        feature_token_counts: &[usize],
    ) -> PreprocessedEncoderInputs {
        let sizes: Vec<(u32, u32)> = item_sizes.iter().map(|s| (s.height, s.width)).collect();
        PreprocessedEncoderInputs {
            encoder_input: ndarray::ArrayD::zeros(vec![1, 3, 336, 336]),
            feature_token_counts: feature_token_counts.to_vec(),
            item_sizes: sizes,
            model_specific: HashMap::new(),
        }
    }

    /// Build `PreprocessedEncoderInputs` with explicit aspect_ratios (for Llama4 tests).
    pub fn test_preprocessed_with_aspects(
        item_sizes: &[ImageSize],
        aspect_ratios: &[(i64, i64)],
    ) -> PreprocessedEncoderInputs {
        let sizes: Vec<(u32, u32)> = item_sizes.iter().map(|s| (s.height, s.width)).collect();
        let flat: Vec<i64> = aspect_ratios
            .iter()
            .flat_map(|&(h, w)| vec![h, w])
            .collect();
        let batch = aspect_ratios.len();
        let mut model_specific = HashMap::new();
        model_specific.insert(
            "aspect_ratios".to_string(),
            ModelSpecificValue::IntTensor {
                data: flat,
                shape: vec![batch, 2],
            },
        );
        PreprocessedEncoderInputs {
            encoder_input: ndarray::ArrayD::zeros(vec![1, 3, 336, 336]),
            feature_token_counts: vec![0; sizes.len()],
            item_sizes: sizes,
            model_specific,
        }
    }
}
