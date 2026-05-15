// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust per-image token-count and image-placeholder token-id resolution
//! via the `llm-multimodal` crate. Compiled only when the `lightseek-mm`
//! cargo feature is enabled.

use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow};
use llm_multimodal::{
    ImagePreProcessor, ImageProcessorRegistry, ModelMetadata, ModelRegistry, PreProcessorConfig,
};
use llm_tokenizer::traits::Tokenizer;
use llm_tokenizer::{Decoder, Encoder, Encoding, HuggingFaceTokenizer, SpecialTokens};

use crate::protocols::TokenIdType;

/// No-op `Tokenizer` impl used when a model directory has no `tokenizer.json`
/// (e.g. Kimi-K2.5 ships `tiktoken.model` instead of an HF fast tokenizer).
///
/// The lightseek `ModelMetadata` always expects a tokenizer reference, but
/// some `ModelProcessorSpec` impls — Kimi-K2.5 in particular — read the
/// image-placeholder token id straight out of `config.json` and never call
/// the tokenizer. Passing `NullTokenizer` lets those specs run; specs that
/// do need vocab access (Phi-3, LLaVA) just get `None` from
/// `token_to_id` and the resolver returns `None` gracefully.
struct NullTokenizer;

impl Encoder for NullTokenizer {
    fn encode(&self, _input: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
        Ok(Encoding::Plain(Vec::new()))
    }
    fn encode_batch(
        &self,
        inputs: &[&str],
        _add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        Ok(inputs.iter().map(|_| Encoding::Plain(Vec::new())).collect())
    }
}

impl Decoder for NullTokenizer {
    fn decode(&self, _ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
        Ok(String::new())
    }
}

impl Tokenizer for NullTokenizer {
    fn vocab_size(&self) -> usize {
        0
    }
    fn get_special_tokens(&self) -> &SpecialTokens {
        static EMPTY: LazyLock<SpecialTokens> = LazyLock::new(SpecialTokens::default);
        &EMPTY
    }
    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }
    fn id_to_token(&self, _id: u32) -> Option<String> {
        None
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Both registries borrow processor refs that callers hold across requests,
// so they must outlive every consumer — `LazyLock` gives them `'static`.
static REGISTRY: LazyLock<ImageProcessorRegistry> =
    LazyLock::new(ImageProcessorRegistry::with_defaults);
static MODEL_REGISTRY: LazyLock<ModelRegistry> = LazyLock::new(ModelRegistry::new);

/// Maps `(width, height) → num_image_tokens` for a single model using the
/// model's HF `preprocessor_config.json`.
pub struct LightseekMmCounter {
    processor: &'static dyn ImagePreProcessor,
    config: PreProcessorConfig,
    model_id: String,
}

impl LightseekMmCounter {
    /// Returns `Err` when `preprocessor_config.json` is missing or unparseable
    /// or no registered processor matches `model_id` / `model_type`. Callers
    /// should treat the error as "MM-aware routing disabled for this model"
    /// rather than failing the request.
    ///
    /// Uses sync filesystem I/O. This is intentional: `try_new` is called
    /// once per model during preprocessor construction (a startup-time path
    /// already guarded by sync setup like `PromptFormatter::from_mdc` and
    /// `ModelDeploymentCard::tokenizer`), not from a per-request hot path.
    /// Switching to async would cascade through `OpenAIPreprocessor::new`
    /// and every caller of it.
    pub fn try_new(model_id: &str, model_type: Option<&str>, model_dir: &Path) -> Result<Self> {
        let cfg_path = model_dir.join("preprocessor_config.json");
        let json = std::fs::read_to_string(&cfg_path).with_context(|| {
            format!(
                "lightseek: failed to read preprocessor_config.json at {}",
                cfg_path.display()
            )
        })?;
        let config = PreProcessorConfig::from_json(&json).with_context(|| {
            format!(
                "lightseek: failed to parse preprocessor_config.json at {}",
                cfg_path.display()
            )
        })?;

        let processor = REGISTRY.find(model_id, model_type).ok_or_else(|| {
            anyhow!(
                "lightseek: no image processor registered for model_id={:?} model_type={:?}",
                model_id,
                model_type
            )
        })?;

        Ok(Self {
            processor,
            config,
            model_id: model_id.to_string(),
        })
    }

    pub fn count_tokens(&self, width: u32, height: u32) -> usize {
        self.processor
            .calculate_num_tokens(width, height, &self.config)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Resolve the image-placeholder token id by delegating to lightseek's
/// per-model `ModelProcessorSpec`. Each registered model (Qwen3-VL,
/// Qwen2.5-VL, Qwen2-VL, LLaVA-NeXT, LLaVA-1.5, Phi-3-vision, Llama-4,
/// Kimi-K2.5) reads the right field of `config.json` (`image_token_id`,
/// `image_token_index`, `media_placeholder_token_id`) and falls back to the
/// tokenizer's vocab when only the placeholder string is known.
///
/// `model_id` is the HF id or local path; `model_dir` is the directory
/// containing `tokenizer.json` and `config.json`.
///
/// Returns `None` when:
/// - `tokenizer.json` or `config.json` is missing or unparseable, or
/// - no `ModelProcessorSpec` matches the model (caller should fall back to
///   text-prefix routing).
pub fn resolve_image_token_id(model_id: &str, model_dir: &Path) -> Option<TokenIdType> {
    // Try the HuggingFace fast tokenizer first; fall back to a no-op
    // tokenizer when `tokenizer.json` is missing (Kimi-K2.5 ships only
    // `tiktoken.model`, for example). Specs that read the placeholder
    // token id from `config.json` (Kimi) still resolve; specs that need
    // vocab access just return `None` here.
    let tokenizer_path = model_dir.join("tokenizer.json");
    let hf_tokenizer =
        tokenizer_path
            .to_str()
            .and_then(|p| match HuggingFaceTokenizer::from_file(p) {
                Ok(t) => Some(t),
                Err(e) => {
                    tracing::debug!(
                        target: "mm_routing",
                        model_dir = %model_dir.display(),
                        err = %e,
                        "lightseek: tokenizer.json not loaded; falling back to NullTokenizer"
                    );
                    None
                }
            });
    let null_tokenizer = NullTokenizer;
    let tokenizer: &dyn Tokenizer = match hf_tokenizer.as_ref() {
        Some(t) => t,
        None => &null_tokenizer,
    };

    let config_path = model_dir.join("config.json");
    let config_json = std::fs::read_to_string(&config_path)
        .map_err(|e| {
            tracing::warn!(
                target: "mm_routing",
                config = %config_path.display(),
                err = %e,
                "lightseek: failed to read config.json"
            );
            e
        })
        .ok()?;
    let config: serde_json::Value = serde_json::from_str(&config_json)
        .map_err(|e| {
            tracing::warn!(
                target: "mm_routing",
                config = %config_path.display(),
                err = %e,
                "lightseek: failed to parse config.json"
            );
            e
        })
        .ok()?;

    let metadata = ModelMetadata {
        model_id,
        tokenizer,
        config: &config,
    };

    let spec = MODEL_REGISTRY.lookup(&metadata)?;
    let id = spec
        .placeholder_token_id(&metadata)
        .map_err(|e| {
            tracing::warn!(
                target: "mm_routing",
                model_id = %model_id,
                err = %e,
                "lightseek: ModelProcessorSpec could not resolve placeholder_token_id"
            );
            e
        })
        .ok()?;
    tracing::debug!(
        target: "mm_routing",
        model_id = %model_id,
        image_token_id = id,
        spec = spec.name(),
        "resolved image-placeholder token id"
    );
    Some(id as TokenIdType)
}

#[cfg(test)]
mod tests {
    //! Contract tests against the upstream lightseek registry. Pin the
    //! behavior `OpenAIPreprocessor::new_with_parts` relies on so a future
    //! smg matcher change shows up here instead of as a silent runtime
    //! fallback to text-prefix-only routing.
    use super::*;

    #[test]
    fn image_processor_registry_resolves_qwen3vl_via_path_substring() {
        // HF id and any path containing "qwen3-vl" (or its underscore variant)
        // match without a model_type hint — the existing happy path.
        assert!(REGISTRY.find("Qwen/Qwen3-VL-2B-Instruct", None).is_some());
        assert!(REGISTRY.find("/models/Qwen3-VL-2B/", None).is_some());
    }

    #[test]
    fn image_processor_registry_uses_model_type_fallback() {
        // Custom dir without a family substring would fail substring match;
        // the model_type fallback parameter rescues those cases.
        assert!(REGISTRY.find("/models/my-finetune", None).is_none());
        assert!(
            REGISTRY
                .find("/models/my-finetune", Some("qwen3_vl"))
                .is_some()
        );
    }

    /// Coverage table for the VLM families we claim to support. Each row is
    /// a `(family_label, hf_id, model_type)` triple. A row "passes" when the
    /// upstream registry can match it via either the HF id substring OR the
    /// `model_type` config field. A failure here means either:
    ///   - the documented family lost coverage in a smg release (need to
    ///     pin or pick up the fix upstream), or
    ///   - we should remove that family from our supported-list claim.
    /// Update this list whenever we add a new supported family in docs.
    #[test]
    fn image_processor_registry_covers_documented_families() {
        // (family, hf_id, model_type)
        const FAMILIES: &[(&str, &str, &str)] = &[
            ("Qwen3-VL", "Qwen/Qwen3-VL-2B-Instruct", "qwen3_vl"),
            ("Qwen2-VL", "Qwen/Qwen2-VL-7B-Instruct", "qwen2_vl"),
            ("Qwen2.5-VL", "Qwen/Qwen2.5-VL-7B-Instruct", "qwen2_5_vl"),
            (
                "LLaVA-NeXT",
                "llava-hf/llava-v1.6-mistral-7b-hf",
                "llava_next",
            ),
            ("LLaVA-1.5", "llava-hf/llava-1.5-7b-hf", "llava"),
            (
                "Phi-3-vision",
                "microsoft/Phi-3-vision-128k-instruct",
                "phi3_v",
            ),
            ("Llama-4", "meta-llama/Llama-4-Scout-17B-16E", "llama4"),
            ("Kimi-K2.5", "moonshotai/Kimi-K2.5-Instruct", "kimi_k2_5"),
        ];

        let mut missing: Vec<&str> = Vec::new();
        for (family, hf_id, model_type) in FAMILIES {
            let by_id = REGISTRY.find(hf_id, None).is_some();
            let by_type = REGISTRY.find("/local/finetune", Some(model_type)).is_some();
            if !(by_id || by_type) {
                missing.push(family);
            }
        }
        assert!(
            missing.is_empty(),
            "lightseek registry has no processor for: {:?}. \
             Either pick up an smg release that registers these, or trim \
             the supported-families list in docs.",
            missing
        );
    }
}
