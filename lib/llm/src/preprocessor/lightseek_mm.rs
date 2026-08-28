// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust per-image token-count and image-placeholder token-id resolution
//! via the `llm-multimodal` crate. Compiled only when the `mm-routing`
//! cargo feature is enabled.

use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow, bail};
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
    processor: ImageTokenCounter,
    config: PreProcessorConfig,
    model_id: String,
}

enum ImageTokenCounter {
    Registry(&'static dyn VisionPreProcessor),
    NemotronNanoOmni(NemotronNanoOmniCounter),
}

/// Routing-only token counter matching vLLM's Nano Nemotron VL processor.
/// Image decoding and tensor preprocessing remain entirely backend-owned.
struct NemotronNanoOmniCounter {
    image_size: usize,
    patch_size: usize,
    downsample_ratio: f64,
    use_thumbnail: bool,
    max_num_tiles: usize,
    dynamic_resolution: Option<NemotronDynamicResolution>,
}

struct NemotronDynamicResolution {
    min_num_patches: usize,
    max_num_patches: usize,
}

impl NemotronNanoOmniCounter {
    fn try_from_configs(
        processor_config: &PreProcessorConfig,
        model_config: &serde_json::Value,
    ) -> Result<Self> {
        let image_size = processor_config
            .get_extra::<usize>("image_size")
            .or_else(|| json_usize(model_config, &["force_image_size"]))
            .ok_or_else(|| anyhow!("Nemotron Nano Omni image_size is missing"))?;
        let patch_size = processor_config.get_patch_size(0);
        let downsample_ratio = processor_config
            .get_extra::<f64>("downsample_ratio")
            .or_else(|| {
                model_config
                    .get("downsample_ratio")
                    .and_then(serde_json::Value::as_f64)
            })
            .ok_or_else(|| anyhow!("Nemotron Nano Omni downsample_ratio is missing"))?;
        let max_num_tiles = processor_config
            .get_extra::<usize>("max_num_tiles")
            .unwrap_or(12);
        let use_thumbnail = processor_config
            .get_extra::<bool>("use_thumbnail")
            .or_else(|| {
                model_config
                    .get("use_thumbnail")
                    .and_then(serde_json::Value::as_bool)
            })
            .unwrap_or(true);

        if image_size == 0
            || patch_size == 0
            || image_size % patch_size != 0
            || !(0.0..=1.0).contains(&downsample_ratio)
            || downsample_ratio == 0.0
            || max_num_tiles == 0
        {
            bail!(
                "invalid Nemotron Nano Omni image config: image_size={image_size}, \
                 patch_size={patch_size}, downsample_ratio={downsample_ratio}, \
                 max_num_tiles={max_num_tiles}"
            );
        }

        let min_num_patches =
            json_usize(model_config, &["vision_config", "args", "min_num_patches"]);
        let max_num_patches =
            json_usize(model_config, &["vision_config", "args", "max_num_patches"]);
        let dynamic_resolution = match (min_num_patches, max_num_patches) {
            (Some(min_num_patches), Some(max_num_patches))
                if min_num_patches > 0 && max_num_patches >= min_num_patches =>
            {
                Some(NemotronDynamicResolution {
                    min_num_patches,
                    max_num_patches,
                })
            }
            (None, None) => None,
            _ => bail!("invalid Nemotron Nano Omni dynamic-resolution patch limits"),
        };

        Ok(Self {
            image_size,
            patch_size,
            downsample_ratio,
            use_thumbnail,
            max_num_tiles,
            dynamic_resolution,
        })
    }

    fn count_tokens(&self, width: u32, height: u32) -> usize {
        if let Some(dynamic) = &self.dynamic_resolution {
            return dynamic.count_tokens_unconstrained(width, height, self.patch_size);
        }
        self.count_static_tokens(width, height)
    }

    fn count_tokens_for_images(
        &self,
        dimensions: &[(u32, u32)],
        max_model_len: usize,
        text_prompt_len: usize,
    ) -> Option<Vec<usize>> {
        match &self.dynamic_resolution {
            Some(dynamic) => dynamic.count_tokens_for_images(
                dimensions,
                max_model_len,
                text_prompt_len,
                self.patch_size,
            ),
            None => Some(
                dimensions
                    .iter()
                    .map(|&(width, height)| self.count_static_tokens(width, height))
                    .collect(),
            ),
        }
    }

    fn count_static_tokens(&self, width: u32, height: u32) -> usize {
        if width == 0 || height == 0 {
            return 0;
        }

        let mut target_ratios: Vec<(usize, usize)> = (1..=self.max_num_tiles)
            .flat_map(|tiles_w| {
                (1..=self.max_num_tiles)
                    .filter(move |tiles_h| tiles_w * tiles_h <= self.max_num_tiles)
                    .map(move |tiles_h| (tiles_w, tiles_h))
            })
            .collect();
        target_ratios.sort_by_key(|(tiles_w, tiles_h)| tiles_w * tiles_h);

        let aspect_ratio = f64::from(width) / f64::from(height);
        let area = f64::from(width) * f64::from(height);
        let mut best_ratio = (1usize, 1usize);
        let mut best_diff = f64::INFINITY;
        for ratio in target_ratios {
            let target_aspect_ratio = ratio.0 as f64 / ratio.1 as f64;
            let ratio_diff = (aspect_ratio - target_aspect_ratio).abs();
            if ratio_diff < best_diff {
                best_diff = ratio_diff;
                best_ratio = ratio;
            } else if ratio_diff == best_diff
                && area > 0.5 * (self.image_size * self.image_size * ratio.0 * ratio.1) as f64
            {
                best_ratio = ratio;
            }
        }

        let mut num_tiles = best_ratio.0 * best_ratio.1;
        if self.use_thumbnail && num_tiles > 1 {
            num_tiles += 1;
        }
        let patches_per_side = self.image_size / self.patch_size;
        let tokens_per_tile =
            ((patches_per_side * patches_per_side) as f64 * self.downsample_ratio.powi(2)) as usize;
        num_tiles * tokens_per_tile
    }
}

impl NemotronDynamicResolution {
    fn count_tokens_unconstrained(&self, width: u32, height: u32, patch_size: usize) -> usize {
        self.process_image(width, height, self.max_num_patches, patch_size)
            .map(|(_, embeddings)| embeddings)
            .unwrap_or(0)
    }

    /// Mirror vLLM's `DynamicResolutionImageTiler.compute_params`. The
    /// backend budgets in pre-pixel-shuffle patches, hence the factor of four.
    fn count_tokens_for_images(
        &self,
        dimensions: &[(u32, u32)],
        max_model_len: usize,
        text_prompt_len: usize,
        patch_size: usize,
    ) -> Option<Vec<usize>> {
        if dimensions.is_empty() {
            return Some(Vec::new());
        }
        let post_shuffle_budget = max_model_len.checked_sub(text_prompt_len)?.checked_sub(4)?;
        let mut total_patch_budget = post_shuffle_budget.checked_mul(4)?;
        total_patch_budget = total_patch_budget.max(self.min_num_patches * dimensions.len());
        let initial_budget = total_patch_budget.clamp(self.min_num_patches, self.max_num_patches);
        let mut per_image_budgets = vec![initial_budget; dimensions.len()];

        for _ in 0..10 {
            let processed: Option<Vec<(usize, usize)>> = dimensions
                .iter()
                .zip(&per_image_budgets)
                .map(|(&(width, height), &budget)| {
                    self.process_image(width, height, budget, patch_size)
                })
                .collect();
            let processed = processed?;
            let total_patches = processed
                .iter()
                .try_fold(0usize, |sum, (patches, _)| sum.checked_add(*patches))?;
            if total_patches <= total_patch_budget {
                return Some(
                    processed
                        .into_iter()
                        .map(|(_, embeddings)| embeddings)
                        .collect(),
                );
            }

            let scale = total_patch_budget as f64 / total_patches as f64;
            let scaled: Vec<usize> = processed
                .iter()
                .map(|(patches, _)| self.min_num_patches.max((*patches as f64 * scale) as usize))
                .collect();
            let scaled_down = scaled
                .iter()
                .zip(&per_image_budgets)
                .any(|(scaled, previous)| scaled < previous);
            per_image_budgets = if scaled_down {
                scaled
            } else {
                vec![self.min_num_patches; dimensions.len()]
            };
        }
        None
    }

    /// Return `(pre-shuffle patches, post-shuffle image tokens)`.
    fn process_image(
        &self,
        width: u32,
        height: u32,
        patch_budget: usize,
        patch_size: usize,
    ) -> Option<(usize, usize)> {
        if width == 0 || height == 0 || patch_size == 0 || patch_budget == 0 {
            return None;
        }
        // Python's round() is ties-to-even; vLLM deliberately adds 0.5 first.
        let closest_patch_height =
            (f64::from(height) / patch_size as f64 + 0.5).round_ties_even() as usize;
        let closest_patch_width =
            (f64::from(width) / patch_size as f64 + 0.5).round_ties_even() as usize;
        let closest_patch_height = closest_patch_height.max(1);
        let closest_patch_width = closest_patch_width.max(1);
        let patches = closest_patch_height.checked_mul(closest_patch_width)?;
        let factor = (patch_budget as f64 / patches as f64).sqrt().min(1.0);
        let mut target_height = (factor * closest_patch_height as f64).floor() as usize;
        let mut target_width = (factor * closest_patch_width as f64).floor() as usize;
        target_height = target_height.max(1);
        target_width = target_width.max(1);

        let target_patches = target_height.checked_mul(target_width)?;
        if patch_budget > self.min_num_patches && target_patches < self.min_num_patches {
            let up_factor = (self.min_num_patches as f64 / target_patches as f64).sqrt();
            target_height = (up_factor * target_height as f64).ceil() as usize;
            target_width = (up_factor * target_width as f64).ceil() as usize;
        }

        // Nano Nemotron uses one 2x pixel-shuffle reduction.
        round_patch_dimension(&mut target_height, target_width, patch_budget, 2);
        round_patch_dimension(&mut target_width, target_height, patch_budget, 2);

        let raw_patches = target_height.checked_mul(target_width)?;
        Some((raw_patches, raw_patches / 4))
    }
}

fn round_patch_dimension(
    dimension: &mut usize,
    other_dimension: usize,
    patch_budget: usize,
    divisor: usize,
) {
    let remainder = *dimension % divisor;
    if remainder == 0 {
        return;
    }
    let increase = divisor - remainder;
    if dimension
        .checked_add(increase)
        .and_then(|value| value.checked_mul(other_dimension))
        .is_some_and(|patches| patches <= patch_budget)
    {
        *dimension += increase;
    } else {
        *dimension = divisor.max(*dimension - remainder);
    }
}

fn json_usize(config: &serde_json::Value, path: &[&str]) -> Option<usize> {
    path.iter()
        .try_fold(config, |value, key| value.get(*key))
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
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

        let processor = if is_nemotron_nano_omni(model_id, model_type) {
            let model_config = read_json(model_dir, "config.json").ok_or_else(|| {
                anyhow!(
                    "mm-routing: failed to read Nemotron Nano Omni config.json at {}",
                    model_dir.display()
                )
            })?;
            ImageTokenCounter::NemotronNanoOmni(NemotronNanoOmniCounter::try_from_configs(
                &config,
                &model_config,
            )?)
        } else {
            ImageTokenCounter::Registry(REGISTRY.find(model_id, model_type).ok_or_else(|| {
                anyhow!(
                    "mm-routing: no image processor registered for model_id={:?} model_type={:?}",
                    model_id,
                    model_type
                )
            })?)
        };

        Ok(Self {
            processor,
            config,
            model_id: model_id.to_string(),
        })
    }

    pub fn count_tokens(&self, width: u32, height: u32) -> usize {
        match &self.processor {
            ImageTokenCounter::Registry(processor) => {
                processor.calculate_num_tokens(width, height, &self.config)
            }
            ImageTokenCounter::NemotronNanoOmni(counter) => counter.count_tokens(width, height),
        }
    }

    /// Return backend-exact counts for an entire image batch. Nemotron's
    /// dynamic tiler shares the remaining context budget across images, so
    /// counting each image independently would diverge for larger batches.
    pub fn count_tokens_for_images(
        &self,
        dimensions: &[(u32, u32)],
        max_model_len: usize,
        text_prompt_len: usize,
    ) -> Option<Vec<usize>> {
        match &self.processor {
            ImageTokenCounter::Registry(processor) => Some(
                dimensions
                    .iter()
                    .map(|&(width, height)| {
                        processor.calculate_num_tokens(width, height, &self.config)
                    })
                    .collect(),
            ),
            ImageTokenCounter::NemotronNanoOmni(counter) => {
                counter.count_tokens_for_images(dimensions, max_model_len, text_prompt_len)
            }
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

fn is_nemotron_nano_omni(model_id: &str, model_type: Option<&str>) -> bool {
    const ARCH: &str = "NemotronH_Nano_Omni_Reasoning_V3";
    model_type.is_some_and(|model_type| model_type.eq_ignore_ascii_case(ARCH))
        || model_id
            .to_ascii_lowercase()
            .contains("nemotron-3-nano-omni")
}

/// Resolve the image-placeholder token id from model config or by delegating
/// to a per-model `ModelProcessorSpec`. Nemotron Nano Omni publishes
/// `img_context_token_id`; registry-backed families read their corresponding
/// `image_token_id`, `image_token_index`, or `media_placeholder_token_id`.
///
/// `model_id` is the HF id or local path; `model_dir` is the directory
/// containing `tokenizer.json` and `config.json`.
///
/// Returns `None` when:
/// - `tokenizer.json` or `config.json` is missing or unparseable, or
/// - no `ModelProcessorSpec` matches the model (caller should fall back to
///   text-prefix routing).
///
/// Standalone token-only wrapper. Prefer [`resolve_routing_tokens`] when also
/// fetching the chat-template placeholder or BOS token (one config-parse pass
/// instead of two).
pub fn resolve_image_token_id(model_id: &str, model_dir: &Path) -> Option<TokenIdType> {
    let config = read_json(model_dir, "config.json")?;
    resolve_model_token_with_config(model_id, model_dir, &config).map(|resolved| resolved.token_id)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImagePromptKind {
    /// The rendered prompt already contains any model-specific wrapper; only
    /// the single image pad is replaced by the per-image feature-token run.
    RepeatedPad,
    /// Kimi-K3's renderer emits one structural `<|media_pad|>` per image, but
    /// the backend replaces it with a dimension-bearing media block.
    KimiK3,
    /// The backend wraps each repeated image-pad run in model-configured
    /// single-token delimiters. Nemotron Nano Omni uses `<img>` and `</img>`.
    WrappedRepeatedPad,
}

impl LightseekMmCounter {
    /// Return the exact-routing prompt shape for the processor selected by
    /// `VisionProcessorRegistry::find`.
    ///
    /// This deliberately maps the selected processor family rather than
    /// repeating its model-id / model-type aliases. New processor families
    /// fail closed until their worker prompt shape has been verified here.
    pub fn routing_prompt_kind(&self) -> Option<ImagePromptKind> {
        match &self.processor {
            ImageTokenCounter::NemotronNanoOmni(_) => Some(ImagePromptKind::WrappedRepeatedPad),
            ImageTokenCounter::Registry(processor) => match processor.model_name() {
                "kimi-k3" => Some(ImagePromptKind::KimiK3),
                "inkling" | "kimi-k2.5" | "llama4-vision" | "llava" | "llava-next"
                | "phi3-vision" | "qwen2-vl" | "qwen3-omni" | "qwen3-vl" => {
                    Some(ImagePromptKind::RepeatedPad)
                }
                _ => None,
            },
        }
    }
}

struct ResolvedModelToken {
    token_id: TokenIdType,
}

fn resolve_model_token_with_config(
    model_id: &str,
    model_dir: &Path,
    config: &serde_json::Value,
) -> Option<ResolvedModelToken> {
    if let Some(token_id) = extract_nemotron_omni_image_token_id(config) {
        tracing::debug!(
            target: "mm_routing",
            model_id = %model_id,
            image_token_id = token_id,
            "resolved Nemotron Nano Omni image-placeholder token id"
        );
        return Some(ResolvedModelToken { token_id });
    }

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
    Some(ResolvedModelToken {
        token_id: id as TokenIdType,
    })
}

/// vLLM expands every Nemotron image placeholder into `<img>`, a run of
/// `<image>` context tokens, and `</img>`. KV events therefore contain the
/// configured context-token id rather than a generic `image_token_id`.
fn extract_nemotron_omni_image_token_id(config: &serde_json::Value) -> Option<TokenIdType> {
    is_nemotron_nano_omni_config(config)
        .then(|| config.get("img_context_token_id"))
        .flatten()
        .and_then(serde_json::Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
}

fn is_nemotron_nano_omni_config(config: &serde_json::Value) -> bool {
    const ARCH: &str = "NemotronH_Nano_Omni_Reasoning_V3";
    config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|model_type| model_type.eq_ignore_ascii_case(ARCH))
        || config
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|architectures| {
                architectures.iter().any(|architecture| {
                    architecture
                        .as_str()
                        .is_some_and(|architecture| architecture.eq_ignore_ascii_case(ARCH))
                })
            })
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
    /// Model-specific shape of the routing-side image prompt, derived from
    /// the same selected vision processor used for image-token counting.
    pub image_prompt_kind: Option<ImagePromptKind>,
    /// `bos_token` string from `tokenizer_config.json` when
    /// `add_bos_token: true`. Caller encodes via its model tokenizer to
    /// produce the routing-side prepend id. `None` for models that don't
    /// prepend BOS.
    pub bos_token_string: Option<String>,
    /// Model-configured wrapper strings for
    /// [`ImagePromptKind::WrappedRepeatedPad`]. Both must encode atomically or
    /// exact routing fails closed.
    pub image_wrapper_strings: Option<(String, String)>,
}

impl RoutingTokens {
    /// Return the placeholder id when the static model prerequisites for exact
    /// MM routing are available. The frontend applies runtime tokenizer and
    /// prompt-validation gates separately.
    pub fn exact_routing_image_token_id(
        &self,
        image_token_counter_available: bool,
    ) -> Option<TokenIdType> {
        self.chat_placeholder_token_id
            .filter(|_| image_token_counter_available && self.image_prompt_kind.is_some())
    }
}

/// Resolve all routing-side token info from a model directory in a single
/// pass. Reads `config.json` once for the per-spec image id + chat-template
/// placeholder, and `tokenizer_config.json` once for BOS. Replaces the
/// in-`preprocessor.rs` `read_image_token_id_from_config` /
/// `read_bos_token_from_config` helpers so config parsing lives next to
/// the rest of the MM-routing token resolution. `counter` is the processor
/// already selected for image-token counting, so counting and layout
/// classification share one model-family selection.
pub fn resolve_routing_tokens(
    model_id: &str,
    model_dir: &Path,
    counter: Option<&LightseekMmCounter>,
) -> RoutingTokens {
    let config = read_json(model_dir, "config.json");
    let tokenizer_config = read_json(model_dir, "tokenizer_config.json");

    let resolved = config
        .as_ref()
        .and_then(|c| resolve_model_token_with_config(model_id, model_dir, c));
    let image_token_id = resolved.as_ref().map(|r| r.token_id);
    let chat_placeholder_token_id = config
        .as_ref()
        .and_then(extract_chat_placeholder_from_config)
        .or(image_token_id);
    // The counter's selected processor is the family authority for both
    // token counting and routing layout. The independent ModelRegistry lookup
    // above remains only a placeholder-token fallback; its alias set cannot
    // silently disable a layout selected by the vision processor.
    let image_prompt_kind =
        chat_placeholder_token_id.and(counter.and_then(LightseekMmCounter::routing_prompt_kind));
    let bos_token_string = tokenizer_config
        .as_ref()
        .and_then(extract_bos_token_from_tokenizer_config);
    let image_wrapper_strings = config
        .as_ref()
        .filter(|config| is_nemotron_nano_omni_config(config))
        .and_then(|config| {
            Some((
                config.get("img_start_token")?.as_str()?.to_owned(),
                config.get("img_end_token")?.as_str()?.to_owned(),
            ))
        });

    RoutingTokens {
        image_token_id,
        chat_placeholder_token_id,
        image_prompt_kind,
        bos_token_string,
        image_wrapper_strings,
    }
}

/// Resolve the worker-side placeholder id from static model prerequisites.
/// An explicit config token is not enough when the image-token counter or
/// prompt layout is unavailable. Request-time frontend readiness is carried
/// separately by frontend-issued canonical MM UUIDs in worker KV events.
pub fn resolve_exact_routing_image_token_id(
    model_id: &str,
    model_dir: &Path,
) -> Option<TokenIdType> {
    let config = read_json(model_dir, "config.json")?;
    let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    let counter = LightseekMmCounter::try_new(model_id, model_type, model_dir).ok()?;
    resolve_routing_tokens(model_id, model_dir, Some(&counter)).exact_routing_image_token_id(true)
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

    #[test]
    fn counter_loads_hf_config_and_counts_known_qwen3_vl_dimensions() {
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("preprocessor_config.json"),
            serde_json::json!({
                "patch_size": 16,
                "merge_size": 2,
                "min_pixels": 3136,
                "max_pixels": 12_845_056,
                "temporal_patch_size": 2
            })
            .to_string(),
        )
        .unwrap();

        let counter = LightseekMmCounter::try_new(
            "Qwen/Qwen3-VL-2B-Instruct",
            Some("qwen3_vl"),
            model_dir.path(),
        )
        .unwrap();

        // 640x480 is already aligned to patch_size * merge_size (32).
        // (640 / 16) * (480 / 16) / merge_size² = 300.
        assert_eq!(counter.count_tokens(640, 480), 300);
    }

    fn write_nemotron_omni_configs(model_dir: &Path) {
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::json!({
                "architectures": ["NemotronH_Nano_Omni_Reasoning_V3"],
                "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
                "force_image_size": 512,
                "patch_size": 16,
                "downsample_ratio": 0.5,
                "use_thumbnail": true,
                "img_context_token": "<image>",
                "img_context_token_id": 18,
                "img_start_token": "<img>",
                "img_end_token": "</img>",
                "vision_config": {
                    "args": {
                        "min_num_patches": 1024,
                        "max_num_patches": 13312
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            model_dir.join("preprocessor_config.json"),
            serde_json::json!({
                "image_processor_type": "NemotronH_Nano_Omni_Reasoning_V3ImageProcessor",
                "image_size": 512,
                "patch_size": 16,
                "downsample_ratio": 0.5,
                "max_num_tiles": 12,
                "use_thumbnail": true
            })
            .to_string(),
        )
        .unwrap();
    }

    #[test]
    fn routing_tokens_resolve_nemotron_nano_omni_prompt_shape() {
        let model_dir = tempfile::tempdir().unwrap();
        write_nemotron_omni_configs(model_dir.path());
        let model_id = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8";
        let counter = LightseekMmCounter::try_new(
            model_id,
            Some("NemotronH_Nano_Omni_Reasoning_V3"),
            model_dir.path(),
        )
        .unwrap();
        let resolved = resolve_routing_tokens(model_id, model_dir.path(), Some(&counter));

        assert_eq!(resolved.image_token_id, Some(18));
        assert_eq!(resolved.chat_placeholder_token_id, Some(18));
        assert_eq!(
            resolved.image_prompt_kind,
            Some(ImagePromptKind::WrappedRepeatedPad)
        );
        assert_eq!(
            resolved.image_wrapper_strings,
            Some(("<img>".to_string(), "</img>".to_string()))
        );
        assert_eq!(
            resolve_exact_routing_image_token_id(model_id, model_dir.path()),
            Some(18)
        );
    }

    #[test]
    fn counter_matches_vllm_nemotron_dynamic_resolution() {
        let model_dir = tempfile::tempdir().unwrap();
        write_nemotron_omni_configs(model_dir.path());
        let counter = LightseekMmCounter::try_new(
            "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
            Some("NemotronH_Nano_Omni_Reasoning_V3"),
            model_dir.path(),
        )
        .unwrap();

        // Golden values from vLLM 0.27.1's DynamicResolutionImageTiler.
        for (dimensions, expected) in [
            ((224, 224), 256),
            ((64, 32), 276),
            ((512, 512), 256),
            ((1000, 500), 512),
            ((500, 1000), 512),
            ((1024, 1024), 1024),
            ((1920, 1080), 2040),
        ] {
            assert_eq!(counter.count_tokens(dimensions.0, dimensions.1), expected);
        }

        let dimensions = vec![(1920, 1080); 3];
        assert_eq!(
            counter.count_tokens_for_images(&dimensions, 4096, 10),
            Some(vec![1344, 1344, 1344])
        );
    }

    fn write_model_config(model_dir: &Path, model_type: &str) {
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::json!({
                "model_type": model_type,
                "media_placeholder_token_id": 163605
            })
            .to_string(),
        )
        .unwrap();
    }

    #[test]
    fn routing_tokens_classify_kimi_k3_via_selected_vision_processor() {
        let model_dir = tempfile::tempdir().unwrap();
        write_model_config(model_dir.path(), "kimi_k3");
        std::fs::write(model_dir.path().join("preprocessor_config.json"), "{}").unwrap();
        let counter = LightseekMmCounter::try_new(
            "/models/internal-checkpoint",
            Some("kimi_k3"),
            model_dir.path(),
        )
        .unwrap();

        assert_eq!(counter.routing_prompt_kind(), Some(ImagePromptKind::KimiK3));
        let resolved = resolve_routing_tokens(
            "/models/internal-checkpoint",
            model_dir.path(),
            Some(&counter),
        );

        assert_eq!(resolved.image_token_id, Some(163605));
        assert_eq!(resolved.chat_placeholder_token_id, Some(163605));
        assert_eq!(resolved.image_prompt_kind, Some(ImagePromptKind::KimiK3));
    }

    #[test]
    fn routing_tokens_keep_kimi_k2_on_repeated_pad_prompt_shape() {
        let model_dir = tempfile::tempdir().unwrap();
        write_model_config(model_dir.path(), "kimi_k25");
        std::fs::write(model_dir.path().join("preprocessor_config.json"), "{}").unwrap();
        let counter =
            LightseekMmCounter::try_new("moonshotai/Kimi-K2.6", Some("kimi_k25"), model_dir.path())
                .unwrap();

        assert_eq!(
            counter.routing_prompt_kind(),
            Some(ImagePromptKind::RepeatedPad)
        );
        let resolved =
            resolve_routing_tokens("moonshotai/Kimi-K2.6", model_dir.path(), Some(&counter));

        assert_eq!(resolved.image_token_id, Some(163605));
        assert_eq!(
            resolved.image_prompt_kind,
            Some(ImagePromptKind::RepeatedPad)
        );
    }

    #[test]
    fn routing_tokens_keep_phi3_vision_on_repeated_pad_prompt_shape() {
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            serde_json::json!({
                "model_type": "phi3_v",
                "image_token_id": 32044
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(model_dir.path().join("preprocessor_config.json"), "{}").unwrap();
        let counter = LightseekMmCounter::try_new(
            "microsoft/Phi-3-vision-128k-instruct",
            Some("phi3_v"),
            model_dir.path(),
        )
        .unwrap();

        assert_eq!(
            counter.routing_prompt_kind(),
            Some(ImagePromptKind::RepeatedPad)
        );
        let resolved = resolve_routing_tokens(
            "microsoft/Phi-3-vision-128k-instruct",
            model_dir.path(),
            Some(&counter),
        );
        assert_eq!(
            resolved.image_prompt_kind,
            Some(ImagePromptKind::RepeatedPad)
        );
    }

    #[test]
    fn routing_tokens_keep_qwen3_omni_on_repeated_pad_prompt_shape() {
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            serde_json::json!({
                "model_type": "qwen3_omni_moe",
                "thinker_config": {
                    "image_token_id": 151655
                }
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(model_dir.path().join("preprocessor_config.json"), "{}").unwrap();
        let model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct";
        let counter =
            LightseekMmCounter::try_new(model_id, Some("qwen3_omni_moe"), model_dir.path())
                .unwrap();

        assert_eq!(
            counter.routing_prompt_kind(),
            Some(ImagePromptKind::RepeatedPad)
        );
        let resolved = resolve_routing_tokens(model_id, model_dir.path(), Some(&counter));
        assert_eq!(resolved.chat_placeholder_token_id, Some(151655));
        assert_eq!(
            resolved.image_prompt_kind,
            Some(ImagePromptKind::RepeatedPad)
        );
        assert_eq!(
            resolve_exact_routing_image_token_id(model_id, model_dir.path()),
            Some(151655)
        );
    }

    #[test]
    fn routing_prompt_classifies_inkling_as_repeated_pad() {
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(model_dir.path().join("preprocessor_config.json"), "{}").unwrap();

        let counter = LightseekMmCounter::try_new(
            "/models/internal-checkpoint",
            Some("inkling_mm_model"),
            model_dir.path(),
        )
        .unwrap();

        assert_eq!(
            counter.routing_prompt_kind(),
            Some(ImagePromptKind::RepeatedPad)
        );
    }

    #[test]
    fn explicit_placeholder_keeps_repeated_pad_for_generic_qwen_aliases() {
        for model_type in ["qwen2_5_vl", "qwen3_6"] {
            let model_dir = tempfile::tempdir().unwrap();
            std::fs::write(
                model_dir.path().join("config.json"),
                serde_json::json!({
                    "model_type": model_type,
                    "image_token_id": 151655
                })
                .to_string(),
            )
            .unwrap();
            std::fs::write(model_dir.path().join("preprocessor_config.json"), "{}").unwrap();

            let counter = LightseekMmCounter::try_new(
                "/models/vision-model",
                Some(model_type),
                model_dir.path(),
            )
            .unwrap();
            let resolved =
                resolve_routing_tokens("/models/vision-model", model_dir.path(), Some(&counter));

            assert_eq!(
                resolved.image_token_id, None,
                "{model_type} should reproduce the independent ModelRegistry alias gap"
            );
            assert_eq!(resolved.chat_placeholder_token_id, Some(151655));
            assert_eq!(
                counter.routing_prompt_kind(),
                Some(ImagePromptKind::RepeatedPad),
                "{model_type} counter selection should also choose its routing layout"
            );
            assert_eq!(
                resolved.image_prompt_kind,
                Some(ImagePromptKind::RepeatedPad),
                "{model_type} should retain exact routing through an explicit placeholder"
            );
            assert_eq!(
                resolve_exact_routing_image_token_id("/models/vision-model", model_dir.path()),
                Some(151655),
                "{model_type} should be fully ready through the counter's model_type fallback"
            );
        }
    }

    #[test]
    fn exact_worker_token_requires_counter_and_prompt_layout() {
        let model_dir = tempfile::tempdir().unwrap();
        write_model_config(model_dir.path(), "kimi_k3");

        assert_eq!(
            resolve_exact_routing_image_token_id("/models/internal-checkpoint", model_dir.path()),
            None,
            "a placeholder alone must not enable worker-side MM key rewriting"
        );

        std::fs::write(model_dir.path().join("preprocessor_config.json"), "{}").unwrap();
        assert_eq!(
            resolve_exact_routing_image_token_id("/models/internal-checkpoint", model_dir.path()),
            Some(163605)
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
            (
                "Qwen3-Omni",
                "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "qwen3_omni_moe",
            ),
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
            (
                "Phi-3 Vision",
                "microsoft/Phi-3-vision-128k-instruct",
                "phi3_v",
            ),
            ("Kimi-K2.5", "moonshotai/Kimi-K2.5-Instruct", "kimi_k2_5"),
            ("Kimi-K2.6", "moonshotai/Kimi-K2.6-Instruct", "kimi_k2_6"),
            ("Qwen3.5", "Qwen/Qwen3.5-0.8B", "qwen3_5"),
            ("Qwen3.6", "Qwen/Qwen3.6-35B-A3B", "qwen3_6"),
            ("Kimi-K3", "moonshotai/Kimi-K3", "kimi_k3"),
            ("Inkling", "/models/inkling", "inkling_mm_model"),
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
}
