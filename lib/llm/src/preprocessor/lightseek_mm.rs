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

/// Whether this model's analytic `count_tokens` output can be trusted as a
/// usage/billing figure — i.e. it provably equals what the backend tokenizes.
///
/// The `llm-multimodal` crate computes the count differently per family:
/// Qwen3 rebuilds its processor from the loaded config (honors it), but Kimi's
/// `calculate_num_tokens` *ignores* the passed config and uses the processor's
/// hardcoded defaults. So a count is trustworthy only when the crate's math is
/// actually driven by (or coincides with) the model's real config. See the
/// DIS-2313 scoping finding for the cross-version experiment behind these rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenCountTrust {
    /// `count_tokens` matches the backend's tokenization; safe to emit.
    Trusted,
    /// Family/config not verified against the backend; caller should skip
    /// emission and bump `image_tokens_skipped_total{reason="unverified_family"}`.
    Unverified,
}

// `llm-multimodal 1.7.0` KimiK25Processor defaults (private in the crate; the
// counter uses these, ignoring per-model config). A Kimi model is only trusted
// when its real `media_proc_cfg` equals every one of these — else the crate
// would silently count with the wrong values. Bump the crate => re-verify here.
const KIMI_CRATE_IN_PATCH_LIMIT: u64 = 16384;
const KIMI_CRATE_PATCH_LIMIT_ON_ONE_SIDE: u64 = 512;
const KIMI_CRATE_PATCH_SIZE: u64 = 14;
const KIMI_CRATE_MERGE_SIZE: u64 = 2;

/// Resolve token-count trust once at construction from the family + parsed config.
fn resolve_token_count_trust(
    model_id: &str,
    model_type: Option<&str>,
    config: &PreProcessorConfig,
) -> TokenCountTrust {
    // Classify by the authoritative `model_type` (from config.json) when present;
    // fall back to `model_id` only when it's absent. `model_id` can be an
    // arbitrary local path (`mdc.source_path()`), so its directory names must NOT
    // drive trust — e.g. a Kimi model under `/models/qwen3-tests/` must not inherit
    // Qwen3's blanket trust and skip the Kimi config check.
    let hint = match model_type {
        Some(mt) if !mt.trim().is_empty() => mt.to_ascii_lowercase(),
        _ => model_id.to_ascii_lowercase(),
    };

    // Kimi first — the fragile family: the crate IGNORES its config and counts with
    // hardcoded defaults, so trust iff the model's real `media_proc_cfg` equals
    // those defaults exactly. Verified for nvidia/Kimi-K2.5-NVFP4 and Kimi-K2.6-NVFP4
    // — both `model_type kimi_k25`, identical `media_proc_cfg` on all 4 params, and
    // byte-identical `media_utils.py` (the resize/token math). `media_proc_cfg` lands
    // in `config.extra` via serde(flatten); missing => Unverified (conservative).
    if hint.contains("kimi") {
        let media = config.extra.get("media_proc_cfg");
        let get = |k: &str| media.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
        let matches = get("in_patch_limit") == Some(KIMI_CRATE_IN_PATCH_LIMIT)
            && get("patch_limit_on_one_side") == Some(KIMI_CRATE_PATCH_LIMIT_ON_ONE_SIDE)
            && get("patch_size") == Some(KIMI_CRATE_PATCH_SIZE)
            && get("merge_kernel_size") == Some(KIMI_CRATE_MERGE_SIZE)
            // The crate has no `fixed_output_tokens` field, so a non-null value
            // would diverge. `null`/absent is the only trustworthy case.
            && media
                .and_then(|m| m.get("fixed_output_tokens"))
                .map(|v| v.is_null())
                .unwrap_or(true);
        return if matches {
            TokenCountTrust::Trusted
        } else {
            TokenCountTrust::Unverified
        };
    }
    // Qwen3 / Qwen3.5 (incl. qwen3_vl, qwen3_5_moe): the crate honors the config
    // AND its built-in defaults coincide with the shipped `size.{shortest,longest}_edge`,
    // so the count is robust even if a config strips the limits.
    if hint.contains("qwen3") {
        return TokenCountTrust::Trusted;
    }
    // Qwen2 / Qwen2.5: the crate honors the config, but its defaults differ from
    // HF's, so the count is right only when the config explicitly pins the limits.
    if hint.contains("qwen2") {
        return if config.min_pixels.is_some() && config.max_pixels.is_some() {
            TokenCountTrust::Trusted
        } else {
            TokenCountTrust::Unverified
        };
    }

    // LLaVA / Llama-4 / Phi-3 / unknown families: token-math parity not verified
    // for usage purposes — skip rather than emit a possibly-wrong number.
    TokenCountTrust::Unverified
}

/// Maps `(width, height) → num_image_tokens` for a single model using the
/// model's HF `preprocessor_config.json`.
pub struct LightseekMmCounter {
    processor: &'static dyn VisionPreProcessor,
    config: PreProcessorConfig,
    model_id: String,
    token_count_trust: TokenCountTrust,
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

        let token_count_trust = resolve_token_count_trust(model_id, model_type, &config);

        Ok(Self {
            processor,
            config,
            model_id: model_id.to_string(),
            token_count_trust,
        })
    }

    pub fn count_tokens(&self, width: u32, height: u32) -> usize {
        self.processor
            .calculate_num_tokens(width, height, &self.config)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Whether this model's `count_tokens` output is a trustworthy usage figure.
    pub fn token_count_trust(&self) -> TokenCountTrust {
        self.token_count_trust
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

    /// The image-token trust gate that PR2's usage metric depends on. Kimi is the
    /// sharp case: the crate ignores its config, so trust hinges on the real
    /// `media_proc_cfg` matching the crate defaults exactly.
    #[test]
    fn token_count_trust_gates_by_family_and_config() {
        let cfg = |json: &str| PreProcessorConfig::from_json(json).unwrap();

        // Kimi with config == crate defaults (nvidia/Kimi-K2.5-NVFP4) -> Trusted.
        let kimi_ok = cfg(
            r#"{"media_proc_cfg":{"in_patch_limit":16384,"patch_size":14,
                "merge_kernel_size":2,"patch_limit_on_one_side":512,
                "fixed_output_tokens":null}}"#,
        );
        assert_eq!(
            resolve_token_count_trust("moonshotai/Kimi-K2.5", Some("kimi_k25"), &kimi_ok),
            TokenCountTrust::Trusted
        );

        // k2.6 ships the identical config + processor (model_type still kimi_k25),
        // so the same matching config is trusted with no code change.
        assert_eq!(
            resolve_token_count_trust("nvidia/Kimi-K2.6-NVFP4", Some("kimi_k25"), &kimi_ok),
            TokenCountTrust::Trusted
        );

        // Kimi whose in_patch_limit drifts from the crate default -> Unverified
        // (the crate would silently count with 16384 anyway).
        let kimi_drift = cfg(r#"{"media_proc_cfg":{"in_patch_limit":8192,"patch_size":14,
                "merge_kernel_size":2,"patch_limit_on_one_side":512}}"#);
        assert_eq!(
            resolve_token_count_trust("moonshotai/Kimi-K2.6", Some("kimi_k25"), &kimi_drift),
            TokenCountTrust::Unverified
        );

        // A Kimi model whose local dir path contains "qwen3" must still be
        // Kimi-gated (classified by the authoritative model_type, not the
        // arbitrary source path) — a drifted config is NOT falsely trusted.
        assert_eq!(
            resolve_token_count_trust(
                "/models/qwen3-tests/kimi-k2.5",
                Some("kimi_k25"),
                &kimi_drift
            ),
            TokenCountTrust::Unverified
        );

        // Kimi that pins fixed_output_tokens -> Unverified (crate can't represent it).
        let kimi_fixed = cfg(
            r#"{"media_proc_cfg":{"in_patch_limit":16384,"patch_size":14,
                "merge_kernel_size":2,"patch_limit_on_one_side":512,
                "fixed_output_tokens":256}}"#,
        );
        assert_eq!(
            resolve_token_count_trust("kimi-k2", Some("kimi_k25"), &kimi_fixed),
            TokenCountTrust::Unverified
        );

        // Qwen3/3.5: crate honors config and defaults coincide -> always Trusted.
        assert_eq!(
            resolve_token_count_trust("Qwen/Qwen3.5-397B", Some("qwen3_5_moe"), &cfg("{}")),
            TokenCountTrust::Trusted
        );

        // Qwen2.5: Trusted only with explicit pixel limits.
        let qwen2_pixels = cfg(r#"{"min_pixels":3136,"max_pixels":12845056}"#);
        assert_eq!(
            resolve_token_count_trust("Qwen/Qwen2.5-VL-7B", Some("qwen2_5_vl"), &qwen2_pixels),
            TokenCountTrust::Trusted
        );
        assert_eq!(
            resolve_token_count_trust("Qwen/Qwen2.5-VL-7B", Some("qwen2_5_vl"), &cfg("{}")),
            TokenCountTrust::Unverified
        );

        // Unknown / unverified family -> Unverified.
        assert_eq!(
            resolve_token_count_trust("llava-hf/llava-1.5-7b-hf", Some("llava"), &cfg("{}")),
            TokenCountTrust::Unverified
        );
    }
}
