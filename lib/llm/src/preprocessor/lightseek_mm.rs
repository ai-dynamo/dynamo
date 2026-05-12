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
use llm_tokenizer::HuggingFaceTokenizer;

use crate::protocols::TokenIdType;

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
/// Qwen2-VL, LLaVA-NeXT, LLaVA-1.5, Phi-3-vision, Llama-4, Kimi-K2.5) reads
/// the right field of `config.json` (`image_token_id`, `image_token_index`,
/// `media_placeholder_token_id`) and falls back to the tokenizer's
/// vocab when only the placeholder string is known.
///
/// `model_id` is the HF id or local path; `model_dir` is the directory
/// containing `tokenizer.json` and `config.json`.
///
/// Returns `None` when:
/// - `tokenizer.json` or `config.json` is missing or unparseable, or
/// - no `ModelProcessorSpec` matches the model (caller should fall back to
///   text-prefix routing).
pub fn resolve_image_token_id(model_id: &str, model_dir: &Path) -> Option<TokenIdType> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str()?)
        .map_err(|e| {
            tracing::warn!(
                target: "mm_routing",
                model_dir = %model_dir.display(),
                err = %e,
                "lightseek: failed to load tokenizer.json"
            );
            e
        })
        .ok()?;

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
        tokenizer: &tokenizer,
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
}
