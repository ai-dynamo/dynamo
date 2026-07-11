// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust per-image token-count and image-placeholder token-id resolution
//! via the `llm-multimodal` crate. Compiled only when the `mm-routing`
//! cargo feature is enabled.

use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow};
use llm_multimodal::vision::{PreProcessorConfig, VisionPreProcessor, VisionProcessorRegistry};
use llm_multimodal::{ModelMetadata, ModelRegistry};
use llm_tokenizer::traits::Tokenizer;
use llm_tokenizer::{Decoder, Encoder, Encoding, HuggingFaceTokenizer, SpecialTokens};

use crate::protocols::TokenIdType;

/// No-op `Tokenizer` impl used when a model directory has no `tokenizer.json`
/// (e.g. Kimi-K2.5 ships `tiktoken.model` instead of an HF fast tokenizer).
///
/// `ModelMetadata` always expects a tokenizer reference, but
/// some `ModelProcessorSpec` impls — Kimi-K2.5 in particular — read the
/// image-placeholder token id straight out of `config.json` and never call
/// the tokenizer. Passing `NullTokenizer` lets those specs run; specs that
/// do need vocab access (LLaVA) just get `None` from
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
static REGISTRY: LazyLock<VisionProcessorRegistry> =
    LazyLock::new(VisionProcessorRegistry::with_defaults);
static MODEL_REGISTRY: LazyLock<ModelRegistry> = LazyLock::new(ModelRegistry::new);

/// Maps `(width, height) → num_image_tokens` for a single model using the
/// model's HF `preprocessor_config.json`.
pub struct LightseekMmCounter {
    processor: &'static dyn VisionPreProcessor,
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
                "mm-routing: failed to read preprocessor_config.json at {}",
                cfg_path.display()
            )
        })?;
        let config = PreProcessorConfig::from_json(&json).with_context(|| {
            format!(
                "mm-routing: failed to parse preprocessor_config.json at {}",
                cfg_path.display()
            )
        })?;

        let processor = REGISTRY.find(model_id, model_type).ok_or_else(|| {
            anyhow!(
                "mm-routing: no image processor registered for model_id={:?} model_type={:?}",
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

/// Resolve the image-placeholder token id by delegating to a per-model
/// `ModelProcessorSpec` from the registry. Each registered model (Qwen3-VL,
/// Qwen2.5-VL, Qwen2-VL, LLaVA-NeXT, LLaVA-1.5, Llama-4,
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
///
/// Standalone wrapper around [`resolve_image_token_id_with_config`]. Prefer
/// [`resolve_routing_tokens`] when also fetching the chat-template placeholder
/// or BOS token (one config-parse pass instead of two).
pub fn resolve_image_token_id(model_id: &str, model_dir: &Path) -> Option<TokenIdType> {
    let config = read_json(model_dir, "config.json")?;
    resolve_image_token_id_with_config(model_id, model_dir, &config)
}

fn resolve_image_token_id_with_config(
    model_id: &str,
    model_dir: &Path,
    config: &serde_json::Value,
) -> Option<TokenIdType> {
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
                        "mm-routing: tokenizer.json not loaded; falling back to NullTokenizer"
                    );
                    None
                }
            });
    let null_tokenizer = NullTokenizer;
    let tokenizer: &dyn Tokenizer = match hf_tokenizer.as_ref() {
        Some(t) => t,
        None => &null_tokenizer,
    };

    let metadata = ModelMetadata {
        model_id,
        tokenizer,
        config,
    };

    let spec = MODEL_REGISTRY.lookup(&metadata)?;
    let id = spec
        .placeholder_token_id(&metadata)
        .map_err(|e| {
            tracing::warn!(
                target: "mm_routing",
                model_id = %model_id,
                err = %e,
                "mm-routing: ModelProcessorSpec could not resolve placeholder_token_id"
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

/// Bundle of routing-side token info resolved from a model's HF JSON
/// configs. All fields default to `None` when the corresponding lookup
/// fails — callers disable the respective routing path without erroring.
///
/// Built by [`resolve_routing_tokens`]; reads `config.json` and
/// `tokenizer_config.json` at most once each.
pub struct RoutingTokens {
    /// Image-placeholder token id resolved via `ModelProcessorSpec`
    /// (per-family `config.json` field). `None` disables MM-aware routing.
    pub image_token_id: Option<TokenIdType>,
    /// Token id the chat template emits per image. Read from `config.json`'s
    /// literal `image_token_id` field, falling back to `image_token_id`
    /// above. Equals `image_token_id` for most VLMs; Qwen2-VL / Qwen2.5-VL
    /// emit `<|image_pad|>` here while the per-patch id is `<|vision_pad|>`.
    pub chat_placeholder_token_id: Option<TokenIdType>,
    /// `bos_token` string from `tokenizer_config.json` when
    /// `add_bos_token: true`. Caller encodes via its model tokenizer to
    /// produce the routing-side prepend id. `None` for models that don't
    /// prepend BOS.
    pub bos_token_string: Option<String>,
}

/// Resolve all routing-side token info from a model directory in a single
/// pass. Reads `config.json` once for the per-spec image id + chat-template
/// placeholder, and `tokenizer_config.json` once for BOS. Replaces the
/// in-`preprocessor.rs` `read_image_token_id_from_config` /
/// `read_bos_token_from_config` helpers so config parsing lives next to
/// the rest of the MM-routing token resolution.
pub fn resolve_routing_tokens(model_id: &str, model_dir: &Path) -> RoutingTokens {
    let config = read_json(model_dir, "config.json");
    let tokenizer_config = read_json(model_dir, "tokenizer_config.json");

    let image_token_id = config
        .as_ref()
        .and_then(|c| resolve_image_token_id_with_config(model_id, model_dir, c));
    let chat_placeholder_token_id = config
        .as_ref()
        .and_then(extract_chat_placeholder_from_config)
        .or(image_token_id);
    let bos_token_string = tokenizer_config
        .as_ref()
        .and_then(extract_bos_token_from_tokenizer_config);

    RoutingTokens {
        image_token_id,
        chat_placeholder_token_id,
        bos_token_string,
    }
}

/// Read and parse a model metadata file under `model_dir`.
/// `config.json` accepts JSON5 syntax; `tokenizer_config.json` remains strict JSON.
/// Warns on read or parse failure (missing files are silent — many models
/// legitimately lack `tokenizer_config.json`). Returns `None` on any error.
fn read_json(model_dir: &Path, filename: &str) -> Option<serde_json::Value> {
    let path = model_dir.join(filename);
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            tracing::warn!(
                target: "mm_routing",
                path = %path.display(),
                err = %e,
                "mm-routing: failed to read {filename}"
            );
            return None;
        }
    };
    let res = if filename == "config.json" {
        json_five::from_str(&raw).map_err(|e| e.to_string())
    } else {
        serde_json::from_str(&raw).map_err(|e| e.to_string())
    };
    match res {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::warn!(
                target: "mm_routing",
                path = %path.display(),
                err = %e,
                "mm-routing: failed to parse {filename}"
            );
            None
        }
    }
}

/// Read the literal `image_token_id` field from a pre-parsed `config.json`.
/// Used by Qwen2-VL / Qwen2.5-VL where the chat-template-emitted placeholder
/// differs from the per-patch expansion token returned by the spec.
fn extract_chat_placeholder_from_config(config: &serde_json::Value) -> Option<TokenIdType> {
    config
        .get("image_token_id")
        .and_then(|x| x.as_u64())
        .and_then(|id| u32::try_from(id).ok())
}

/// Return the `bos_token` string from a pre-parsed `tokenizer_config.json`
/// when `add_bos_token: true`. The routing-side sequence must prepend it to
/// match the backend's HF-processor output (LLaVA-1.5 and other
/// `LlamaTokenizer`-family models). Returns `None` otherwise.
fn extract_bos_token_from_tokenizer_config(cfg: &serde_json::Value) -> Option<String> {
    if !cfg
        .get("add_bos_token")
        .and_then(|x| x.as_bool())
        .unwrap_or(false)
    {
        return None;
    }
    // `bos_token` is usually a plain string ("<s>") but the HF schema also
    // allows it to be an `AddedToken` dict — handle both.
    cfg.get("bos_token").and_then(|x| match x {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o
            .get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_owned()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    //! Contract tests against the upstream `llm-multimodal` image-processor
    //! registry. Pin the behavior `OpenAIPreprocessor::new_with_parts`
    //! relies on so a future upstream matcher change shows up here instead
    //! of as a silent runtime fallback to text-prefix-only routing.
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
    ///
    /// - the documented family lost coverage in a smg release (need to
    ///   pin or pick up the fix upstream), or
    /// - we should remove that family from our supported-list claim.
    ///
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
            ("Llama-4", "meta-llama/Llama-4-Scout-17B-16E", "llama4"),
            ("Kimi-K2.5", "moonshotai/Kimi-K2.5-Instruct", "kimi_k2_5"),
            ("Kimi-K2.6", "moonshotai/Kimi-K2.6-Instruct", "kimi_k2_6"),
            ("Qwen3.5", "Qwen/Qwen3.5-0.8B", "qwen3_5"),
            ("Qwen3.6", "Qwen/Qwen3.6-35B-A3B", "qwen3_6"),
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
            "image-processor registry has no processor for: {:?}. \
             Either pick up an upstream release that registers these, or trim \
             the supported-families list in docs.",
            missing
        );
    }

    #[test]
    fn test_resolve_routing_tokens_with_infinity_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_content = r#"{
            "model_type": "qwen2_5_vl",
            "image_token_id": 151655,
            "vision_token_id": 151655,
            "time_step_limit": [0.0, Infinity]
        }"#;
        std::fs::write(dir.path().join("config.json"), config_content).unwrap();

        let tokenizer_content = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 151655,
                    "content": "<|image_pad|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "model": {
                "type": "BPE",
                "vocab": {
                    "<|image_pad|>": 151655
                },
                "merges": []
            }
        }"#;
        std::fs::write(dir.path().join("tokenizer.json"), tokenizer_content).unwrap();

        let tokens = resolve_routing_tokens("Qwen/Qwen2.5-VL-7B-Instruct", dir.path());
        assert_eq!(tokens.image_token_id, Some(151655));
    }
}
