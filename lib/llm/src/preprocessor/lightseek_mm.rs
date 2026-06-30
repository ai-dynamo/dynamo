// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo-owned image token counting and placeholder-token resolution.
//!
//! Compiled only with `mm-routing`. The count kernels intentionally preserve
//! the subset of `llm-multimodal` 1.7.0 behavior used by Dynamo while avoiding
//! its image-processing and tokenizer dependency graph.

#[path = "lightseek_mm/estimator.rs"]
mod estimator;

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use dynamo_tokenizers::{HuggingFaceTokenizer, traits::Tokenizer};

use self::estimator::{ImageTokenEstimator, ModelFamily};
use crate::protocols::TokenIdType;

/// Maps `(width, height) -> num_image_tokens` for one model.
pub struct LightseekMmCounter {
    estimator: ImageTokenEstimator,
    model_id: String,
}

impl LightseekMmCounter {
    /// Construct an estimator from the model's `preprocessor_config.json`.
    ///
    /// Returns `Err` when the file is missing/unparseable or the documented
    /// family cannot be identified. Callers should disable MM-aware routing
    /// rather than reject model startup.
    pub fn try_new(model_id: &str, model_type: Option<&str>, model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("preprocessor_config.json");
        let raw = std::fs::read_to_string(&config_path).with_context(|| {
            format!(
                "mm-routing: failed to read preprocessor_config.json at {}",
                config_path.display()
            )
        })?;
        let config: serde_json::Value = serde_json::from_str(&raw).with_context(|| {
            format!(
                "mm-routing: failed to parse preprocessor_config.json at {}",
                config_path.display()
            )
        })?;
        if !config.is_object() {
            return Err(anyhow!(
                "mm-routing: preprocessor_config.json root must be an object at {}",
                config_path.display()
            ));
        }

        let family = ModelFamily::identify(model_id, model_type).ok_or_else(|| {
            anyhow!(
                "mm-routing: no image-token estimator registered for model_id={:?} model_type={:?}",
                model_id,
                model_type
            )
        })?;

        Ok(Self {
            estimator: ImageTokenEstimator::from_config(family, &config).with_context(|| {
                format!(
                    "mm-routing: invalid image-token estimator config for model_id={model_id:?}"
                )
            })?,
            model_id: model_id.to_string(),
        })
    }

    #[inline]
    pub fn count_tokens(&self, width: u32, height: u32) -> usize {
        self.estimator.count_tokens(width, height)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Bundle of routing-side token information resolved from HF JSON configs.
pub struct RoutingTokens {
    /// Repeated image patch/pad token used by the model family.
    pub image_token_id: Option<TokenIdType>,
    /// Token emitted once per image by the chat template. This differs from
    /// `image_token_id` for Qwen2-VL and Qwen2.5-VL. `None` disables routing
    /// for families such as Llama4 that require structured expansion. This
    /// field is the MM-routing engagement gate.
    pub chat_placeholder_token_id: Option<TokenIdType>,
    /// BOS string to look up and prepend when `add_bos_token` is true.
    pub bos_token_string: Option<String>,
}

/// Resolve only the repeated image token ID, loading `tokenizer.json` when a
/// family needs vocabulary fallback.
pub fn resolve_image_token_id(model_id: &str, model_dir: &Path) -> Option<TokenIdType> {
    let config = read_json(model_dir, "config.json")?;
    let tokenizer = load_hf_tokenizer(model_dir);
    resolve_image_token_id_with_config(model_id, &config, tokenizer.as_ref().map(as_tokenizer))
}

/// Resolve routing tokens for standalone callers that do not already own a
/// tokenizer. Normal frontend startup should use
/// [`resolve_routing_tokens_with_tokenizer`] to avoid loading it twice.
pub fn resolve_routing_tokens(model_id: &str, model_dir: &Path) -> RoutingTokens {
    let tokenizer = load_hf_tokenizer(model_dir);
    resolve_routing_tokens_inner(model_id, model_dir, tokenizer.as_ref().map(as_tokenizer))
}

/// Resolve routing tokens using the tokenizer already loaded by the frontend.
pub(crate) fn resolve_routing_tokens_with_tokenizer(
    model_id: &str,
    model_dir: &Path,
    tokenizer: &dyn Tokenizer,
) -> RoutingTokens {
    resolve_routing_tokens_inner(model_id, model_dir, Some(tokenizer))
}

fn resolve_routing_tokens_inner(
    model_id: &str,
    model_dir: &Path,
    tokenizer: Option<&dyn Tokenizer>,
) -> RoutingTokens {
    let config = read_json(model_dir, "config.json");
    let tokenizer_config = read_json(model_dir, "tokenizer_config.json");

    let image_token_id = config
        .as_ref()
        .and_then(|config| resolve_image_token_id_with_config(model_id, config, tokenizer));
    let chat_placeholder_token_id = config.as_ref().and_then(|config| {
        resolve_chat_placeholder_token_id(model_id, config, tokenizer, image_token_id)
    });
    let bos_token_string = tokenizer_config
        .as_ref()
        .and_then(extract_bos_token_from_tokenizer_config);

    RoutingTokens {
        image_token_id,
        chat_placeholder_token_id,
        bos_token_string,
    }
}

fn resolve_image_token_id_with_config(
    model_id: &str,
    config: &serde_json::Value,
    tokenizer: Option<&dyn Tokenizer>,
) -> Option<TokenIdType> {
    let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    let family = ModelFamily::identify(model_id, model_type)?;

    let token_id = match family {
        ModelFamily::Qwen2 => u32_field(config, "vision_token_id"),
        ModelFamily::Qwen3 => u32_field(config, "image_token_id"),
        ModelFamily::Llava | ModelFamily::LlavaNext => u32_field(config, "image_token_index")
            .or_else(|| tokenizer.and_then(|tokenizer| tokenizer.token_to_id("<image>"))),
        ModelFamily::Llama4 => u32_field(config, "image_token_index")
            .or_else(|| tokenizer.and_then(|tokenizer| tokenizer.token_to_id("<|image|>"))),
        ModelFamily::KimiK2 => u32_field(config, "media_placeholder_token_id"),
    };

    if let Some(token_id) = token_id {
        tracing::debug!(
            target: "mm_routing",
            model_id,
            ?family,
            image_token_id = token_id,
            "resolved image-placeholder token id"
        );
    } else {
        tracing::warn!(
            target: "mm_routing",
            model_id,
            ?family,
            "mm-routing: model family matched but image-placeholder token id was unavailable"
        );
    }
    token_id
}

fn resolve_chat_placeholder_token_id(
    model_id: &str,
    config: &serde_json::Value,
    tokenizer: Option<&dyn Tokenizer>,
    image_token_id: Option<TokenIdType>,
) -> Option<TokenIdType> {
    let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    let family = ModelFamily::identify(model_id, model_type)?;
    let vocabulary_id = |token| tokenizer.and_then(|tokenizer| tokenizer.token_to_id(token));

    match family {
        // The Qwen2 chat template emits `<|image_pad|>`, while the model
        // config's `vision_token_id` names a different patch token.
        ModelFamily::Qwen2 => u32_field(config, "image_token_id")
            .or_else(|| vocabulary_id("<|image_pad|>"))
            .or(image_token_id),
        ModelFamily::Qwen3 => u32_field(config, "image_token_id").or(image_token_id),
        ModelFamily::Llava | ModelFamily::LlavaNext => vocabulary_id("<image>").or(image_token_id),
        // Llama4 needs structural start/end/separator tokens and 144 patch
        // positions per tile after pixel shuffle. The scalar compatibility
        // counter intentionally remains available, but the current routing
        // representation cannot encode that sequence. Preserve text-only
        // fallback instead of constructing hashes that cannot match vLLM.
        ModelFamily::Llama4 => None,
        ModelFamily::KimiK2 => vocabulary_id("<|media_pad|>").or(image_token_id),
    }
}

fn as_tokenizer(tokenizer: &HuggingFaceTokenizer) -> &dyn Tokenizer {
    tokenizer
}

fn load_hf_tokenizer(model_dir: &Path) -> Option<HuggingFaceTokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let path = tokenizer_path.to_str()?;
    match HuggingFaceTokenizer::from_file(path) {
        Ok(tokenizer) => Some(tokenizer),
        Err(error) => {
            tracing::debug!(
                target: "mm_routing",
                model_dir = %model_dir.display(),
                %error,
                "mm-routing: tokenizer.json unavailable; config-only token resolution remains enabled"
            );
            None
        }
    }
}

fn u32_field(config: &serde_json::Value, field: &str) -> Option<u32> {
    config
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

/// Read and parse a JSON file under `model_dir`. Missing optional files are
/// silent; other I/O and parse failures are warnings and degrade to `None`.
fn read_json(model_dir: &Path, filename: &str) -> Option<serde_json::Value> {
    let path = model_dir.join(filename);
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            tracing::warn!(
                target: "mm_routing",
                path = %path.display(),
                %error,
                "mm-routing: failed to read {filename}"
            );
            return None;
        }
    };
    match serde_json::from_str(&raw) {
        Ok(value) => Some(value),
        Err(error) => {
            tracing::warn!(
                target: "mm_routing",
                path = %path.display(),
                %error,
                "mm-routing: failed to parse {filename}"
            );
            None
        }
    }
}

fn extract_bos_token_from_tokenizer_config(config: &serde_json::Value) -> Option<String> {
    if !config
        .get("add_bos_token")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }

    config.get("bos_token").and_then(|token| match token {
        serde_json::Value::String(token) => Some(token.clone()),
        serde_json::Value::Object(token) => token
            .get("content")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_tokenizers::{
        Encoding,
        traits::{DecodeResult, Decoder, Encoder},
    };
    use serde::Deserialize;

    use super::*;

    #[derive(Deserialize)]
    struct CompatibilityFixture {
        source: String,
        algorithms: Vec<AlgorithmFixture>,
    }

    #[derive(Deserialize)]
    struct AlgorithmFixture {
        family: String,
        model_id: String,
        model_type: String,
        preprocessor_config: serde_json::Value,
        counts: Vec<CountFixture>,
    }

    #[derive(Deserialize)]
    struct CountFixture {
        width: u32,
        height: u32,
        tokens: usize,
    }

    struct TestTokenizer {
        vocabulary: HashMap<String, u32>,
    }

    impl TestTokenizer {
        fn new(entries: &[(&str, u32)]) -> Self {
            Self {
                vocabulary: entries
                    .iter()
                    .map(|(token, id)| ((*token).to_owned(), *id))
                    .collect(),
            }
        }
    }

    impl Encoder for TestTokenizer {
        fn encode(&self, _input: &str) -> anyhow::Result<Encoding> {
            Ok(Encoding::Sp(Vec::new()))
        }

        fn encode_batch(&self, inputs: &[&str]) -> anyhow::Result<Vec<Encoding>> {
            Ok(inputs.iter().map(|_| Encoding::Sp(Vec::new())).collect())
        }
    }

    impl Decoder for TestTokenizer {
        fn decode(
            &self,
            _token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> anyhow::Result<DecodeResult> {
            Ok(DecodeResult::Complete(String::new()))
        }
    }

    impl Tokenizer for TestTokenizer {
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocabulary.get(token).copied()
        }
    }

    #[test]
    fn compatibility_counts_match_llm_multimodal_1_7_0() {
        let fixture: CompatibilityFixture = serde_json::from_str(include_str!(
            "../../tests/data/mm_routing/llm_multimodal_1_7_0_counts.json"
        ))
        .expect("valid compatibility fixture");
        assert_eq!(fixture.source, "llm-multimodal 1.7.0");

        for algorithm in fixture.algorithms {
            let directory = tempfile::tempdir().expect("temporary model directory");
            std::fs::write(
                directory.path().join("preprocessor_config.json"),
                serde_json::to_vec(&algorithm.preprocessor_config).unwrap(),
            )
            .unwrap();
            let counter = LightseekMmCounter::try_new(
                &algorithm.model_id,
                Some(&algorithm.model_type),
                directory.path(),
            )
            .unwrap_or_else(|error| panic!("{} did not construct: {error}", algorithm.family));

            for count in algorithm.counts {
                assert_eq!(
                    counter.count_tokens(count.width, count.height),
                    count.tokens,
                    "{} at {}x{}",
                    algorithm.family,
                    count.width,
                    count.height
                );
            }
        }
    }

    #[test]
    fn every_documented_family_matches_by_id_and_model_type() {
        const FAMILIES: &[(&str, &str, ModelFamily)] = &[
            ("Qwen/Qwen2-VL-7B-Instruct", "qwen2_vl", ModelFamily::Qwen2),
            (
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "qwen2_5_vl",
                ModelFamily::Qwen2,
            ),
            ("Qwen/Qwen3-VL-2B-Instruct", "qwen3_vl", ModelFamily::Qwen3),
            ("Qwen/Qwen3.5-0.8B", "qwen3_5", ModelFamily::Qwen3),
            ("Qwen/Qwen3.6-35B-A3B", "qwen3_5_moe", ModelFamily::Qwen3),
            ("llava-hf/llava-1.5-7b-hf", "llava", ModelFamily::Llava),
            (
                "llava-hf/llava-v1.6-mistral-7b-hf",
                "llava_next",
                ModelFamily::LlavaNext,
            ),
            (
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "llama4",
                ModelFamily::Llama4,
            ),
            ("moonshotai/Kimi-K2.5", "kimi_k25", ModelFamily::KimiK2),
            ("moonshotai/Kimi-K2.6", "kimi_k25", ModelFamily::KimiK2),
        ];

        for &(model_id, model_type, expected) in FAMILIES {
            assert_eq!(
                ModelFamily::identify(model_id, None),
                Some(expected),
                "ID did not match: {model_id}"
            );
            assert_eq!(
                ModelFamily::identify("/models/custom-finetune", Some(model_type)),
                Some(expected),
                "model_type did not match: {model_type}"
            );
        }

        assert_eq!(
            ModelFamily::identify("/models/custom-finetune", Some("qwen3_5_moe")),
            Some(ModelFamily::Qwen3)
        );
        assert_eq!(
            ModelFamily::identify("/models/custom-finetune", Some("kimi_k25")),
            Some(ModelFamily::KimiK2)
        );
    }

    #[test]
    fn routing_tokens_resolve_config_vocab_dual_id_and_bos() {
        let directory = tempfile::tempdir().unwrap();
        let tokenizer = TestTokenizer::new(&[("<image>", 32_000), ("<|patch|>", 200_092)]);

        std::fs::write(
            directory.path().join("config.json"),
            r#"{"model_type":"qwen2_vl","vision_token_id":151654,"image_token_id":151655}"#,
        )
        .unwrap();
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"add_bos_token":true,"bos_token":{"content":"<s>"}}"#,
        )
        .unwrap();
        let tokens =
            resolve_routing_tokens_with_tokenizer("local-qwen", directory.path(), &tokenizer);
        assert_eq!(tokens.image_token_id, Some(151_654));
        assert_eq!(tokens.chat_placeholder_token_id, Some(151_655));
        assert_eq!(tokens.bos_token_string.as_deref(), Some("<s>"));

        std::fs::write(
            directory.path().join("config.json"),
            r#"{"model_type":"llava"}"#,
        )
        .unwrap();
        let tokens =
            resolve_routing_tokens_with_tokenizer("local-llava", directory.path(), &tokenizer);
        assert_eq!(tokens.image_token_id, Some(32_000));
        assert_eq!(tokens.chat_placeholder_token_id, Some(32_000));

        std::fs::write(
            directory.path().join("config.json"),
            r#"{"model_type":"llama4","image_token_index":200092}"#,
        )
        .unwrap();
        let tokens =
            resolve_routing_tokens_with_tokenizer("local-llama", directory.path(), &tokenizer);
        assert_eq!(tokens.image_token_id, Some(200_092));
        assert_eq!(tokens.chat_placeholder_token_id, None);
    }

    #[test]
    fn config_only_resolution_does_not_require_tokenizer_json() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            r#"{"model_type":"kimi_k25","media_placeholder_token_id":163605}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_image_token_id("/models/custom-kimi", directory.path()),
            Some(163_605)
        );
    }

    #[test]
    fn invalid_token_fields_do_not_truncate() {
        let directory = tempfile::tempdir().unwrap();
        let tokenizer = TestTokenizer::new(&[]);
        std::fs::write(
            directory.path().join("config.json"),
            r#"{"model_type":"llava","image_token_index":4294967296}"#,
        )
        .unwrap();
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"add_bos_token":false,"bos_token":"<s>"}"#,
        )
        .unwrap();

        let tokens =
            resolve_routing_tokens_with_tokenizer("local-llava", directory.path(), &tokenizer);
        assert_eq!(tokens.image_token_id, None);
        assert_eq!(tokens.chat_placeholder_token_id, None);
        assert_eq!(tokens.bos_token_string, None);
    }

    #[test]
    fn malformed_or_unsupported_configuration_degrades_cleanly() {
        let directory = tempfile::tempdir().unwrap();
        assert!(LightseekMmCounter::try_new("Qwen/Qwen3-VL", None, directory.path()).is_err());
        std::fs::write(
            directory.path().join("preprocessor_config.json"),
            "not json",
        )
        .unwrap();
        assert!(LightseekMmCounter::try_new("Qwen/Qwen3-VL", None, directory.path()).is_err());
        std::fs::write(directory.path().join("preprocessor_config.json"), "{}").unwrap();
        assert!(
            LightseekMmCounter::try_new("microsoft/Phi-3-vision", Some("phi3_v"), directory.path())
                .is_err()
        );

        let invalid_configs = [
            ("qwen2_vl", "[]"),
            ("qwen2_vl", r#"{"patch_size":0}"#),
            ("qwen2_vl", r#"{"merge_size":0}"#),
            ("qwen2_vl", r#"{"merge_size":"4"}"#),
            ("qwen2_vl", r#"{"do_resize":"yes"}"#),
            ("qwen2_vl", r#"{"image_mean":"bad"}"#),
            ("qwen2_vl", r#"{"image_mean":[0.5],"norm_mean":[0.5]}"#),
            ("qwen2_vl", r#"{"num_crops":1.5}"#),
            ("qwen2_vl", r#"{"temporal_patch_size":0}"#),
            ("qwen2_vl", r#"{"min_pixels":1024,"max_pixels":512}"#),
            (
                "qwen2_vl",
                r#"{"min_pixels":18446744073709551615,"max_pixels":18446744073709551615}"#,
            ),
            (
                "qwen2_vl",
                r#"{"patch_size":18446744073709551615,"merge_size":2}"#,
            ),
            ("llava", r#"{"patch_size":0}"#),
            ("llava", r#"{"patch_size":"14"}"#),
            ("llava_next", r#"{"size":{"shortest_edge":0}}"#),
            ("llava_next", r#"{"size":{"height":4294967296}}"#),
            ("llama4", r#"{"max_image_tiles":0}"#),
            ("llama4", r#"{"max_image_tiles":65}"#),
            ("llama4", r#"{"max_image_tiles":"4"}"#),
        ];
        for (model_type, config) in invalid_configs {
            std::fs::write(directory.path().join("preprocessor_config.json"), config).unwrap();
            assert!(
                LightseekMmCounter::try_new(
                    "/models/custom-finetune",
                    Some(model_type),
                    directory.path()
                )
                .is_err(),
                "invalid {model_type} config was accepted: {config}"
            );
        }
    }
}
