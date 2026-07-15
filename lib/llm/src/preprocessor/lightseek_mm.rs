// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust per-image token-count and image-placeholder token-id resolution
//! via the `llm-multimodal` crate. Compiled only when the `mm-routing`
//! cargo feature is enabled.

use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow};
use llm_multimodal::vision::{
    PreProcessorConfig, Qwen2VLProcessor, Qwen3VLProcessor, VisionPreProcessor,
    VisionProcessorRegistry,
};
use llm_multimodal::{ModelMetadata, ModelRegistry};
use llm_tokenizer::traits::Tokenizer;
use llm_tokenizer::{Decoder, Encoder, Encoding, HuggingFaceTokenizer, SpecialTokens};

use crate::{local_model::runtime_config::ImageTokenizationSpec, protocols::TokenIdType};

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

// Routing keeps the broad best-effort registry that predates the usage metric.
// Usage counting has a separate semantic binding below: tightening metric
// exactness must not remove MM-aware routing support for other families.
static ROUTING_REGISTRY: LazyLock<VisionProcessorRegistry> =
    LazyLock::new(VisionProcessorRegistry::with_defaults);
static MODEL_REGISTRY: LazyLock<ModelRegistry> = LazyLock::new(ModelRegistry::new);

/// Best-effort image-token expansion used only to build the router's synthetic
/// token sequence. This deliberately remains broader than the exact usage
/// counter and is never an authority for billing/usage metrics.
pub struct RoutingImageTokenCounter {
    processor: &'static dyn VisionPreProcessor,
    config: PreProcessorConfig,
    model_id: String,
}

impl RoutingImageTokenCounter {
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
        let processor = ROUTING_REGISTRY.find(model_id, model_type).ok_or_else(|| {
            anyhow!(
                "mm-routing: no image processor registered for model_id={model_id:?} model_type={model_type:?}"
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

/// Typed MoonViT-v1 parameters parsed from `media_proc_cfg`.
///
/// This is intentionally independent of `llm-multimodal`'s Kimi processor:
/// version 1.7.0 constructs that processor with crate defaults and ignores the
/// nested per-model values. Keeping the complete count-affecting schema here
/// lets the semantic contract work for non-default configs without an
/// artifact allowlist or file digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MoonvitV1Config {
    patch_size: usize,
    merge_kernel_size: usize,
    in_patch_limit: usize,
    patch_limit_on_one_side: usize,
    fixed_output_tokens: Option<usize>,
}

impl MoonvitV1Config {
    fn try_from_preprocessor_config(config: &PreProcessorConfig) -> Result<Self> {
        let media = config
            .extra
            .get("media_proc_cfg")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| anyhow!("moonvit_v1 requires an object media_proc_cfg"))?;

        let required_usize = |key: &str| -> Result<usize> {
            let value = media
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| anyhow!("moonvit_v1 requires integer media_proc_cfg.{key}"))?;
            let value = usize::try_from(value)
                .with_context(|| format!("media_proc_cfg.{key} does not fit usize"))?;
            if value == 0 {
                return Err(anyhow!("media_proc_cfg.{key} must be greater than zero"));
            }
            Ok(value)
        };

        let fixed_output_tokens = match media.get("fixed_output_tokens") {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => {
                let value = value.as_u64().ok_or_else(|| {
                    anyhow!("media_proc_cfg.fixed_output_tokens must be an integer or null")
                })?;
                Some(
                    usize::try_from(value)
                        .context("media_proc_cfg.fixed_output_tokens does not fit usize")?,
                )
            }
        };

        let result = Self {
            patch_size: required_usize("patch_size")?,
            merge_kernel_size: required_usize("merge_kernel_size")?,
            in_patch_limit: required_usize("in_patch_limit")?,
            patch_limit_on_one_side: required_usize("patch_limit_on_one_side")?,
            fixed_output_tokens,
        };
        result
            .patch_size
            .checked_mul(result.merge_kernel_size)
            .ok_or_else(|| anyhow!("moonvit_v1 patch/merge factor overflows usize"))?;
        result
            .patch_limit_on_one_side
            .checked_mul(result.patch_size)
            .ok_or_else(|| anyhow!("moonvit_v1 one-side pixel limit overflows usize"))?;
        Ok(result)
    }

    /// Exact port of MoonViT's versioned `navit_resize_image` token math.
    fn count_tokens(self, width: u32, height: u32) -> usize {
        if let Some(fixed) = self.fixed_output_tokens {
            return fixed;
        }

        let width = width as usize;
        let height = height as usize;
        let patches_w = (width / self.patch_size).max(1) as f64;
        let patches_h = (height / self.patch_size).max(1) as f64;
        let s1 = (self.in_patch_limit as f64 / (patches_w * patches_h)).sqrt();
        let max_side_pixels = self.patch_limit_on_one_side * self.patch_size;
        let s2 = max_side_pixels as f64 / width as f64;
        let s3 = max_side_pixels as f64 / height as f64;
        let scale = 1.0_f64.min(s1).min(s2).min(s3);
        let new_width = ((width as f64 * scale) as usize)
            .max(1)
            .min(max_side_pixels);
        let new_height = ((height as f64 * scale) as usize)
            .max(1)
            .min(max_side_pixels);
        let factor = self.patch_size * self.merge_kernel_size;

        new_width.div_ceil(factor) * new_height.div_ceil(factor)
    }
}

enum ExactCounterImpl {
    Qwen2(Qwen2VLProcessor),
    Qwen3(Qwen3VLProcessor),
    Moonvit(MoonvitV1Config),
}

/// A single, typed binding between the worker's declared semantic algorithm,
/// the effective model config, and the corresponding frontend implementation.
/// Selection and trust therefore cannot disagree.
struct ProcessorBinding {
    spec: ImageTokenizationSpec,
    config: PreProcessorConfig,
    counter: ExactCounterImpl,
}

impl ProcessorBinding {
    fn try_new(spec: ImageTokenizationSpec, config: PreProcessorConfig) -> Result<Self> {
        let counter = match spec {
            ImageTokenizationSpec::Qwen2VlV1 => {
                if config.do_resize == Some(false) {
                    return Err(anyhow!(
                        "qwen2_vl_v1 with do_resize=false is not implemented exactly"
                    ));
                }
                if config.min_pixels.is_none() || config.max_pixels.is_none() {
                    return Err(anyhow!(
                        "qwen2_vl_v1 requires explicit min_pixels and max_pixels"
                    ));
                }
                ExactCounterImpl::Qwen2(Qwen2VLProcessor::new())
            }
            ImageTokenizationSpec::Qwen3VlV1 => {
                if config.do_resize == Some(false) {
                    return Err(anyhow!(
                        "qwen3_vl_v1 with do_resize=false is not implemented exactly"
                    ));
                }
                ExactCounterImpl::Qwen3(Qwen3VLProcessor::new())
            }
            ImageTokenizationSpec::MoonvitV1 => {
                ExactCounterImpl::Moonvit(MoonvitV1Config::try_from_preprocessor_config(&config)?)
            }
        };

        Ok(Self {
            spec,
            config,
            counter,
        })
    }

    fn count_tokens(&self, width: u32, height: u32) -> usize {
        match &self.counter {
            ExactCounterImpl::Qwen2(processor) => {
                processor.calculate_num_tokens(width, height, &self.config)
            }
            ExactCounterImpl::Qwen3(processor) => {
                processor.calculate_num_tokens(width, height, &self.config)
            }
            ExactCounterImpl::Moonvit(config) => config.count_tokens(width, height),
        }
    }
}

/// Exact `(width, height) -> image tokens` counter for one worker-attested
/// semantic processor contract.
pub struct ExactImageTokenCounter {
    binding: ProcessorBinding,
    model_id: String,
}

impl ExactImageTokenCounter {
    /// Returns `Err` when `preprocessor_config.json` is missing or unparseable
    /// or the worker's semantic spec cannot represent the effective config.
    /// Callers should withhold exact image-token usage rather than failing the
    /// request.
    ///
    /// Uses sync filesystem I/O. This is intentional: `try_new` is called
    /// once per model during preprocessor construction (a startup-time path
    /// already guarded by sync setup like `PromptFormatter::from_mdc` and
    /// `ModelDeploymentCard::tokenizer`), not from a per-request hot path.
    /// Switching to async would cascade through `OpenAIPreprocessor::new`
    /// and every caller of it.
    pub fn try_new(model_id: &str, spec: ImageTokenizationSpec, model_dir: &Path) -> Result<Self> {
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

        let binding = ProcessorBinding::try_new(spec, config).with_context(|| {
            format!(
                "mm-routing: image tokenization spec {} is incompatible with {}",
                spec.as_str(),
                cfg_path.display()
            )
        })?;

        Ok(Self {
            binding,
            model_id: model_id.to_string(),
        })
    }

    pub fn count_tokens(&self, width: u32, height: u32) -> usize {
        self.binding.count_tokens(width, height)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn spec(&self) -> ImageTokenizationSpec {
        self.binding.spec
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

/// Read + parse a JSON file under `model_dir`. Warns on read or parse
/// failure (missing files are silent — many models legitimately lack
/// `tokenizer_config.json`). Returns `None` on any error.
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
    match serde_json::from_str(&raw) {
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
    use super::*;

    fn cfg(json: &str) -> PreProcessorConfig {
        PreProcessorConfig::from_json(json).unwrap()
    }

    fn moonvit_json(in_patch_limit: usize, fixed: &str) -> String {
        format!(
            r#"{{"media_proc_cfg":{{"in_patch_limit":{in_patch_limit},
                "patch_size":14,"merge_kernel_size":2,
                "patch_limit_on_one_side":512,"fixed_output_tokens":{fixed}}}}}"#
        )
    }

    #[test]
    fn routing_registry_remains_broad_and_model_type_aware() {
        assert!(
            ROUTING_REGISTRY
                .find("Qwen/Qwen3-VL-2B-Instruct", None)
                .is_some()
        );
        assert!(
            ROUTING_REGISTRY
                .find("/models/custom-finetune", Some("qwen3_vl"))
                .is_some()
        );
        assert!(
            ROUTING_REGISTRY
                .find("llava-hf/llava-1.5-7b-hf", Some("llava"))
                .is_some()
        );
    }

    #[test]
    fn semantic_spec_selects_counter_independent_of_model_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("preprocessor_config.json"),
            moonvit_json(16384, "null"),
        )
        .unwrap();

        // The arbitrary source path says Qwen, but the worker's semantic
        // capability is the sole selector. MoonViT 64x64 -> ceil(64/28)^2 = 9.
        let counter = ExactImageTokenCounter::try_new(
            "/models/qwen3-tests/custom",
            ImageTokenizationSpec::MoonvitV1,
            dir.path(),
        )
        .unwrap();
        assert_eq!(counter.spec(), ImageTokenizationSpec::MoonvitV1);
        assert_eq!(counter.count_tokens(64, 64), 9);
    }

    #[test]
    fn moonvit_v1_honors_effective_config_and_fixed_output() {
        let defaults = ProcessorBinding::try_new(
            ImageTokenizationSpec::MoonvitV1,
            cfg(&moonvit_json(16384, "null")),
        )
        .unwrap();
        let lower_limit = ProcessorBinding::try_new(
            ImageTokenizationSpec::MoonvitV1,
            cfg(&moonvit_json(4096, "null")),
        )
        .unwrap();
        let fixed = ProcessorBinding::try_new(
            ImageTokenizationSpec::MoonvitV1,
            cfg(&moonvit_json(4096, "256")),
        )
        .unwrap();

        assert_eq!(defaults.count_tokens(1024, 1024), 1369);
        assert_eq!(lower_limit.count_tokens(1024, 1024), 1089);
        assert_eq!(fixed.count_tokens(1024, 1024), 256);
    }

    #[test]
    fn semantic_bindings_fail_closed_on_unrepresented_config() {
        let qwen2_no_resize = cfg(r#"{"do_resize":false,"patch_size":14,"merge_size":2,
                "temporal_patch_size":2,"min_pixels":3136,
                "max_pixels":12845056}"#);
        assert!(
            ProcessorBinding::try_new(ImageTokenizationSpec::Qwen2VlV1, qwen2_no_resize)
                .err()
                .unwrap()
                .to_string()
                .contains("do_resize=false")
        );

        assert!(
            ProcessorBinding::try_new(ImageTokenizationSpec::Qwen2VlV1, cfg("{}"))
                .err()
                .unwrap()
                .to_string()
                .contains("explicit min_pixels")
        );

        assert!(
            ProcessorBinding::try_new(
                ImageTokenizationSpec::MoonvitV1,
                cfg(r#"{"media_proc_cfg":{"patch_size":14}}"#),
            )
            .err()
            .unwrap()
            .to_string()
            .contains("merge_kernel_size")
        );
    }

    #[test]
    fn qwen_specs_use_their_declared_algorithms() {
        let qwen2 = ProcessorBinding::try_new(
            ImageTokenizationSpec::Qwen2VlV1,
            cfg(r#"{"patch_size":14,"merge_size":2,"temporal_patch_size":2,
                    "min_pixels":3136,"max_pixels":12845056}"#),
        )
        .unwrap();
        let qwen3 = ProcessorBinding::try_new(
            ImageTokenizationSpec::Qwen3VlV1,
            cfg(r#"{"patch_size":16,"merge_size":2,"temporal_patch_size":2,
                    "size":{"shortest_edge":65536,"longest_edge":16777216}}"#),
        )
        .unwrap();

        assert_eq!(qwen2.spec, ImageTokenizationSpec::Qwen2VlV1);
        assert_eq!(qwen3.spec, ImageTokenizationSpec::Qwen3VlV1);
        assert_ne!(qwen2.count_tokens(64, 64), qwen3.count_tokens(64, 64));
    }
}
