// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Preprocessor consists of the following modules
//!
//! - `translation`: This module converts the allowed Ingress message types to the corresponding
//!   internal representation.
//! - `apply`: This module applies ModelConfig defaults to any empty optional fields specified
//! - `prompt`: This module applies any prompt template logic to the internal Request object.
//! - `tokenize`: This module tokenizes the formatted prompt string and returns the token ids.
//!
//! The Preprocessor will accept any IngressRequest and transform it to a BackendRequest.

#[cfg(feature = "mm-routing")]
pub mod lightseek_mm;
pub mod media;
pub mod prompt;
pub mod speculative_prefill;
mod structural_tag;
mod tool_choice;
pub mod tools;
use anyhow::Context;
use anyhow::{Result, bail};

use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionToolChoiceOption, EncodingFormat,
};
use dynamo_renderer::OAIPromptFormatter;
use dynamo_runtime::error::{DynamoError, ErrorType};
use futures::Stream;
use futures::stream::{self, StreamExt};
use std::time::{Duration, Instant};

use dynamo_runtime::dynamo_nvtx_range;
use dynamo_runtime::metrics::frontend_perf::{
    DETOKENIZE_TOKEN_COUNT, DETOKENIZE_TOTAL_US, STAGE_DURATION_SECONDS, STAGE_PREPROCESS,
    StageGuard, TEMPLATE_SECONDS, TOKENIZE_SECONDS,
};
use std::borrow::Cow;
use std::{collections::HashMap, pin::Pin, sync::Arc};
use tracing;

#[cfg(feature = "mm-routing")]
use crate::model_card::ModelInfoType;
use crate::model_card::{ModelDeploymentCard, ModelInfo};
use crate::preprocessor::media::MediaLoader;
use crate::protocols::common::preprocessor::{
    MultimodalData, MultimodalDataMap, PreprocessedRequestBuilder, RoutingHints,
};
use crate::protocols::common::timing::RequestTracker;
use crate::tokenizers::Encoding;

use dynamo_parsers::{ReasoningParser, ReasoningParserType};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{
    AsyncEngineContext, Error, ManyOut, Operator, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::{Annotated, AnnotationsProvider};

use crate::protocols::{
    common::{OutputOptionsProvider, SamplingOptionsProvider, StopConditionsProvider},
    openai::{
        DeltaGeneratorExt,
        chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse, jail::JailedStream,
        },
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
        embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
        nvext::NvExtProvider,
    },
};
use crate::tokenizers::traits::Tokenizer;

use crate::preprocessor::prompt::{MediaRequestExt, prompt_formatter_from_mdc};
use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter, PromptInput, TextInput, TokenInput};

pub use crate::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
pub use crate::protocols::common::preprocessor::PreprocessedEmbeddingRequest;

use crate::protocols::common::llm_backend::EmbeddingsEngineOutput;

/// Encode a slice of `f32` values as a base64 string per the OpenAI
/// `encoding_format=base64` spec: the raw little-endian byte
/// representation of each `f32` is concatenated and the resulting byte
/// buffer is base64-encoded with the standard alphabet.
fn encode_floats_to_base64(floats: &[f32]) -> String {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(floats));
    for f in floats {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    STANDARD.encode(&bytes)
}

pub const ANNOTATION_FORMATTED_PROMPT: &str = "formatted_prompt";
pub const ANNOTATION_TOKEN_IDS: &str = "token_ids";
pub const ANNOTATION_LLM_METRICS: &str = "llm_metrics";
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LLMMetricAnnotation {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub chunk_tokens: usize,
    pub cached_tokens: Option<usize>,
    /// Prefill worker ID (for TTFT attribution in disaggregated mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,
    /// Prefill worker DP rank
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_dp_rank: Option<u32>,
    /// Prefill worker type ("prefill" or "decode") for Prometheus metric labeling.
    /// Stored at routing time to avoid expensive MDC lookup when updating TTFT metrics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_worker_type: Option<String>,
    /// Decode worker ID (for ITL attribution in disaggregated mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,
    /// Decode worker DP rank
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_dp_rank: Option<u32>,
    /// Decode worker type ("prefill" or "decode") for Prometheus metric labeling.
    /// Stored at routing time to avoid expensive MDC lookup when updating ITL metrics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_worker_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenize_latency: Option<Duration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detokenize_total_latency: Option<Duration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detokenize_count: Option<u64>,
}

impl LLMMetricAnnotation {
    /// Convert this metrics struct to an Annotated event
    pub fn to_annotation<T>(&self) -> Result<Annotated<T>, serde_json::Error> {
        Annotated::from_annotation(ANNOTATION_LLM_METRICS, self)
    }

    /// Extract LLM metrics from an Annotated event, if present
    pub fn from_annotation<T>(
        annotation: &Annotated<T>,
    ) -> Result<Option<LLMMetricAnnotation>, Box<dyn std::error::Error>> {
        if annotation.event.is_none() {
            return Ok(None);
        }
        if annotation.event.as_ref().unwrap() != ANNOTATION_LLM_METRICS {
            return Ok(None);
        }
        let comments = annotation
            .comment
            .as_ref()
            .ok_or("missing comments block")?;
        if comments.len() != 1 {
            return Err("malformed comments block - expected exactly 1 comment".into());
        }
        let metrics: LLMMetricAnnotation = serde_json::from_str(&comments[0])?;
        Ok(Some(metrics))
    }
}

/// Take a standalone router's timing out of `routing_data` (see
/// `inject_timing_from_tracker`). Removing it keeps the field off the client wire.
fn take_router_timing(
    data: &mut Option<BackendOutput>,
) -> Option<crate::protocols::common::timing::TimingInfo> {
    data.as_mut()?.routing_data.take()?.timing
}

// Reasoning State for reasoning parsing transformation step
struct ReasoningState {
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>,
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
    bypass_bare_guided_json: bool,
    guided_json_bypass_decision: Option<bool>,
}

/// Per-image routing payload accumulated by `gather_multi_modal_data` and
/// consumed by `gather_mm_exact_routing_info`.
#[derive(Debug, Clone, Copy)]
pub struct MmImageEntry {
    pub mm_hash: u64,
    pub width: u32,
    pub height: u32,
}

/// Derive the model's local directory from the MDC. The directory is the
/// parent of `config.json` (which lives in `mdc.model_info` as `HfConfigJson`)
/// and contains the other artifacts MM-aware routing reads at startup
/// (`tokenizer.json`, `processor_config.json`, `preprocessor_config.json`).
/// Returns `None` for cards built from non-disk sources.
#[cfg(feature = "mm-routing")]
fn mdc_model_dir(mdc: &ModelDeploymentCard) -> Option<std::path::PathBuf> {
    let ModelInfoType::HfConfigJson(cf) = mdc.model_info.as_ref()?;
    cf.path()?.parent().map(std::path::PathBuf::from)
}

/// Shared SSRF-aware `MediaFetcher` + `reqwest::Client` for the dim-fetch
/// path used by MM-aware routing. Inherits the same policy contract as the
/// frontend-decode path (`MediaLoader`): blocklist DNS resolver, redirect
/// revalidation, hostname/IP blocklist, `DYN_MM_ALLOW_INTERNAL` opt-in.
///
/// **Lifecycle:** `LazyLock` so the closure runs on first access. For MM-
/// routable preprocessors, `OpenAIPreprocessor::new_with_parts` calls
/// `LazyLock::force(...)` at startup — that surfaces TLS-root / reqwest-
/// init / env-misconfig failures at deployment time, not on the first MM
/// request 20 minutes in. Text-only deployments skip the force, leaving
/// the LazyLock dormant.
#[cfg(feature = "mm-routing")]
static DIM_FETCH_MEDIA_FETCHER: std::sync::LazyLock<crate::preprocessor::media::MediaFetcher> =
    std::sync::LazyLock::new(crate::preprocessor::media::MediaFetcher::from_env);

#[cfg(feature = "mm-routing")]
static DIM_FETCH_HTTP_CLIENT: std::sync::LazyLock<reqwest::Client> =
    std::sync::LazyLock::new(|| {
        DIM_FETCH_MEDIA_FETCHER
            .build_http_client()
            .expect("dim-fetch http client construction failed")
    });

pub(crate) const PRESERVE_OMITTED_MAX_TOKENS_CONTEXT_KEY: &str =
    "dynamo.llm.preserve_omitted_max_tokens";

#[derive(Clone, Copy, Debug, Default)]
struct PreprocessRequestOptions {
    preserve_omitted_max_tokens: bool,
}

pub struct OpenAIPreprocessor {
    mdcsum: String,
    formatter: Arc<dyn OAIPromptFormatter>,
    tokenizer: Arc<dyn Tokenizer>,
    model_info: Arc<dyn ModelInfo>,
    lora_name: Option<String>,
    /// Per-model runtime configuration propagated to response generator (e.g., reasoning/tool parser)
    runtime_config: crate::local_model::runtime_config::ModelRuntimeConfig,
    /// KV cache block size published in the model deployment card.
    kv_cache_block_size: usize,
    tool_call_parser: Option<String>,
    media_loader: Option<MediaLoader>,
    /// Max context length (in tokens) this model can handle, from ModelDeploymentCard
    context_length: u32,
    /// Per-image token-count engine. `None` when the feature is disabled, the
    /// model isn't covered by the registry, or `preprocessor_config.json` is
    /// unreadable.
    #[cfg(feature = "mm-routing")]
    image_token_counter: Option<lightseek_mm::LightseekMmCounter>,
    /// Image-placeholder token id the routing-side sequence fills per image.
    /// Resolved from `config.json`'s `image_token_id` field when present,
    /// otherwise falls back to the `ModelProcessorSpec` registry value. This
    /// is the id the backend's HF processor emits in the expanded sequence
    /// (per-patch token for Qwen-VL families, the single placeholder for
    /// LLaVA/Phi-3), so block hashes align bit-for-bit with the worker.
    ///
    /// `None` disables MM-aware routing for this model and the router falls
    /// back to text-prefix routing.
    #[cfg(feature = "mm-routing")]
    routing_image_token_id: Option<crate::protocols::TokenIdType>,
    /// Per-family flatten-time image placeholder template (e.g.
    /// `"<|image_{n}|>"` for Phi-3, `"<image>"` for LLaVA-1.5). Threaded
    /// through from the formatter so the routing path can reverse the
    /// BPE-encoded numbered form (Phi-3) back into single placeholder
    /// tokens when the chat template uses numbered markers.
    #[cfg(feature = "mm-routing")]
    image_placeholder_template: Option<&'static str>,
    /// BOS token id to prepend to the routing-side sequence so per-block
    /// hashes match the backend's HF processor output on models with
    /// `add_bos_token: true` (Phi-3-vision and other `LlamaTokenizer`
    /// families). `None` when the model doesn't need it or `bos_token`
    /// doesn't round-trip to a single id.
    #[cfg(feature = "mm-routing")]
    routing_prepend_bos: Option<crate::protocols::TokenIdType>,
}

impl OpenAIPreprocessor {
    fn omitted_max_tokens_default(
        prompt_len: usize,
        context_length: u32,
        options: PreprocessRequestOptions,
    ) -> Option<u32> {
        if context_length == 0 || options.preserve_omitted_max_tokens {
            return None;
        }
        Some(context_length.saturating_sub(prompt_len as u32))
    }

    fn nvext_passthrough_args<R: NvExtProvider>(
        request: &R,
    ) -> Option<serde_json::Map<String, serde_json::Value>> {
        let mut nvext_passthrough = serde_json::Map::new();

        if let Some(nvext) = request.nvext() {
            if let Some(ref fields) = nvext.extra_fields {
                nvext_passthrough.insert("extra_fields".to_string(), serde_json::json!(fields));
            }
            if let Some(ref salt) = nvext.cache_salt {
                nvext_passthrough.insert("cache_salt".to_string(), serde_json::json!(salt));
            }
            if nvext.token_data.is_some() {
                nvext_passthrough.insert("token_in".to_string(), serde_json::Value::Bool(true));
            }
        }

        if !nvext_passthrough.contains_key("cache_salt")
            && let Some(salt) = request
                .unsupported_fields()
                .and_then(|fields| fields.get("cache_salt"))
                .and_then(|value| value.as_str())
        {
            nvext_passthrough.insert("cache_salt".to_string(), serde_json::json!(salt));
        }

        if nvext_passthrough.is_empty() {
            None
        } else {
            Some(nvext_passthrough)
        }
    }

    fn sampling_passthrough_args<R: NvExtProvider>(
        request: &R,
    ) -> Option<serde_json::Map<String, serde_json::Value>> {
        let mut sampling_passthrough = serde_json::Map::new();

        if let Some(fields) = request.unsupported_fields() {
            for key in ["detokenize", "allowed_token_ids", "bad_words_token_ids"] {
                if let Some(value) = fields.get(key) {
                    sampling_passthrough.insert(key.to_string(), value.clone());
                }
            }
        }

        if sampling_passthrough.is_empty() {
            None
        } else {
            Some(sampling_passthrough)
        }
    }

    fn backend_extra_args<R: NvExtProvider>(request: &R) -> Option<serde_json::Value> {
        let mut extra_args = serde_json::Map::new();

        if let Some(nvext_passthrough) = Self::nvext_passthrough_args(request) {
            extra_args.insert(
                "nvext".to_string(),
                serde_json::Value::Object(nvext_passthrough),
            );
        }

        if let Some(sampling_passthrough) = Self::sampling_passthrough_args(request) {
            extra_args.insert(
                "sampling_options".to_string(),
                serde_json::Value::Object(sampling_passthrough),
            );
        }

        if extra_args.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(extra_args))
        }
    }

    pub fn new(mdc: ModelDeploymentCard) -> Result<Arc<Self>> {
        let formatter = prompt_formatter_from_mdc(&mdc)?;
        let tokenizer = mdc.tokenizer()?;
        match formatter {
            PromptFormatter::OAI(formatter) => Self::new_with_parts(mdc, formatter, tokenizer),
        }
    }

    pub fn new_with_parts(
        mdc: ModelDeploymentCard,
        formatter: Arc<dyn OAIPromptFormatter>,
        tokenizer: crate::tokenizers::Tokenizer,
    ) -> Result<Arc<Self>> {
        let mdcsum = mdc.mdcsum().to_string();
        let tokenizer: Arc<dyn Tokenizer> = (*tokenizer).clone();
        let lora_name = mdc.lora.as_ref().map(|l| l.name.clone());
        let Some(ref model_info) = mdc.model_info else {
            anyhow::bail!(
                "Blank ModelDeploymentCard cannot be used for pre-processing, no model_info"
            );
        };
        let model_info = model_info.get_model_info()?;
        let tool_call_parser = mdc.runtime_config.tool_call_parser.clone();

        if let Some(ref lora_name) = lora_name {
            tracing::info!(model = %mdc.display_name, lora_name, "LoRA adapter detected in MDC");
        }

        // // Initialize runtime config from the ModelDeploymentCard
        let runtime_config = mdc.runtime_config.clone();
        let kv_cache_block_size = mdc.kv_cache_block_size as usize;

        // Capture MM-routing inputs before mdc is partially moved into MediaLoader.
        // model_type comes from config.json (e.g. "qwen3_vl") and lets the
        // image-processor registry resolve fine-tunes loaded from
        // custom-named directories where the family substring isn't in the path.
        #[cfg(feature = "mm-routing")]
        let model_dir_for_routing: Option<std::path::PathBuf> = mdc_model_dir(&mdc);
        // TODO(mm-routing): fastokens lacks a special-token mutator, so it
        // can't merge tokenizer_config.json specials and would BPE-shatter
        // placeholders (e.g. Qwen2-VL `<|image_pad|>`). Disable MM-routing
        // here; remove once fastokens upstream exposes the mutator.
        #[cfg(feature = "mm-routing")]
        let image_token_inputs: Option<(String, String, std::path::PathBuf)> = {
            let fastokens_active = std::env::var("DYN_TOKENIZER").as_deref() == Ok("fastokens");
            if fastokens_active && model_dir_for_routing.is_some() {
                tracing::warn!(
                    target: "mm_routing",
                    "DYN_TOKENIZER=fastokens is set; MM-aware KV routing disabled. \
                     Unset DYN_TOKENIZER (or set it to 'default') to re-enable."
                );
                None
            } else {
                model_dir_for_routing.as_ref().map(|p| {
                    (
                        mdc.source_path().to_string(),
                        model_info.model_type(),
                        p.clone(),
                    )
                })
            }
        };

        let media_loader = match mdc.media_decoder {
            Some(media_decoder) => Some(MediaLoader::new(media_decoder, mdc.media_fetcher)?),
            None => None,
        };

        let context_length = mdc.context_length;

        #[cfg(feature = "mm-routing")]
        let (image_token_counter, routing_image_token_id, bos_token_string) =
            match image_token_inputs {
                Some((model_id, model_type, model_dir)) => {
                    // Resolve counter + image-token id independently so the
                    // summary log can name which piece is missing.
                    let (counter, counter_err): (
                        Option<lightseek_mm::LightseekMmCounter>,
                        Option<String>,
                    ) = match lightseek_mm::LightseekMmCounter::try_new(
                        &model_id,
                        Some(&model_type),
                        &model_dir,
                    ) {
                        Ok(c) => (Some(c), None),
                        Err(e) => (None, Some(e.to_string())),
                    };
                    // One-shot config/tokenizer_config read for all
                    // routing-side token info. Parsing lives next to the
                    // spec resolution in the MM-routing module.
                    let routing_tokens =
                        lightseek_mm::resolve_routing_tokens(&model_id, &model_dir);
                    // `chat_placeholder_token_id` already prefers config.json's
                    // explicit field and falls back to the spec value, so it's
                    // the single id used both for the engagement gate and the
                    // routing-fill below.
                    let img_tok = routing_tokens.chat_placeholder_token_id;
                    let bos_tok_string = routing_tokens.bos_token_string;

                    match (counter.is_some(), img_tok.is_some()) {
                        (true, true) => tracing::info!(
                            target: "mm_routing",
                            model = %model_id,
                            model_dir = %model_dir.display(),
                            "MM-aware KV routing enabled"
                        ),
                        (counter_ok, img_ok) => {
                            let mut reasons: Vec<String> = Vec::new();
                            if !counter_ok {
                                reasons.push(format!(
                                    "model not supported by the MM-routing registry ({})",
                                    counter_err.as_deref().unwrap_or("unknown error")
                                ));
                            }
                            if !img_ok {
                                reasons.push(
                                    "image-placeholder token unresolvable from \
                                 config.json / processor_config.json / \
                                 tokenizer_config.json / vocab probe"
                                        .to_string(),
                                );
                            }
                            tracing::warn!(
                                target: "mm_routing",
                                model = %model_id,
                                reasons = %reasons.join("; "),
                                "{} is not supported for MM-aware KV routing ({}). \
                                 Falling back to KV routing without MM awareness — \
                                 text-prefix overlap still works but the router \
                                 cannot distinguish requests by image content.",
                                model_id,
                                reasons.join("; ")
                            );
                        }
                    }
                    (counter, img_tok, bos_tok_string)
                }
                None => {
                    tracing::debug!(
                        target: "mm_routing",
                        "model directory not derivable from MDC; MM-aware routing disabled"
                    );
                    (None, None, None)
                }
            };

        #[cfg(feature = "mm-routing")]
        let image_placeholder_template = formatter.image_placeholder_template();

        // Force the dim-fetch HTTP client to build at startup for any
        // MM-routable preprocessor, so TLS / env-var / reqwest-init
        // failures fail the deployment instead of crashing the first
        // MM request 20 minutes in. Text-only preprocessors skip the
        // force (both MM-routing hooks resolved to `None`) — no point
        // building a client they'll never use.
        #[cfg(feature = "mm-routing")]
        if image_token_counter.is_some() || routing_image_token_id.is_some() {
            std::sync::LazyLock::force(&DIM_FETCH_MEDIA_FETCHER);
            std::sync::LazyLock::force(&DIM_FETCH_HTTP_CLIENT);
        }

        // Resolve the routing-side BOS prepend for models with
        // `add_bos_token: true` (see `routing_prepend_bos` doc). Only kept
        // when the configured `bos_token` round-trips to a single id. The
        // BOS string was harvested above by `resolve_routing_tokens` from
        // the same `tokenizer_config.json` pass.
        #[cfg(feature = "mm-routing")]
        let routing_prepend_bos = match bos_token_string {
            Some(bos_text) => match tokenizer.encode(&bos_text) {
                Ok(enc) if enc.token_ids().len() == 1 => {
                    let id = enc.token_ids()[0];
                    tracing::debug!(
                        target: "mm_routing",
                        bos_token = %bos_text,
                        bos_token_id = id,
                        "routing-side BOS prepend enabled (tokenizer_config.json add_bos_token=true)"
                    );
                    Some(id)
                }
                Ok(enc) => {
                    tracing::debug!(
                        target: "mm_routing",
                        bos_token = %bos_text,
                        round_trip_ids = ?enc.token_ids(),
                        "BOS token does not round-trip to a single id; routing-side prepend disabled"
                    );
                    None
                }
                Err(e) => {
                    tracing::debug!(
                        target: "mm_routing",
                        bos_token = %bos_text,
                        error = %e,
                        "BOS token failed to re-encode; routing-side prepend disabled"
                    );
                    None
                }
            },
            None => None,
        };

        Ok(Arc::new(Self {
            formatter,
            tokenizer,
            model_info,
            mdcsum,
            lora_name,
            runtime_config,
            kv_cache_block_size,
            tool_call_parser,
            media_loader,
            context_length,
            #[cfg(feature = "mm-routing")]
            image_token_counter,
            #[cfg(feature = "mm-routing")]
            routing_image_token_id,
            #[cfg(feature = "mm-routing")]
            image_placeholder_template,
            #[cfg(feature = "mm-routing")]
            routing_prepend_bos,
        }))
    }

    /// Encode a string to it's tokens
    pub fn tokenize(&self, s: &str) -> anyhow::Result<Encoding> {
        self.tokenizer.encode(s)
    }

    /// Translate a [`NvCreateChatCompletionRequest`] request to a common completion request.
    /// Returns the common completion request, a hashmap of annotations, and a boolean
    /// indicating whether the rendered prompt ends with a reasoning start token (e.g.,
    /// `<think>`), meaning the model's completion will begin mid-reasoning.
    ///
    /// Annotations evaluated by this method include:
    /// - `formatted_prompt`
    /// - `token_ids`
    pub async fn preprocess_request<
        R: OAIChatLikeRequest
            + MediaRequestExt
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
        tracker: Option<&RequestTracker>,
    ) -> Result<(PreprocessedRequest, HashMap<String, String>, bool)> {
        self.preprocess_request_with_options(request, tracker, PreprocessRequestOptions::default())
            .await
    }

    async fn preprocess_request_with_options<
        R: OAIChatLikeRequest
            + MediaRequestExt
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
        tracker: Option<&RequestTracker>,
        options: PreprocessRequestOptions,
    ) -> Result<(PreprocessedRequest, HashMap<String, String>, bool)> {
        let _stage_guard = StageGuard::new(STAGE_PREPROCESS, "");
        let preprocess_start = Instant::now();
        let mut builder = self.builder(request)?;

        let template_start = Instant::now();
        let formatted_prompt = {
            let _nvtx = dynamo_nvtx_range!("preprocess.template");
            self.apply_template(request)
                .with_context(|| "Failed to apply prompt template")?
        };
        TEMPLATE_SECONDS.observe(template_start.elapsed().as_secs_f64());

        // Check if the chat template injected a reasoning start token at the end
        // of the prompt (e.g., Qwen3.5 appends `<think>\n` when enable_thinking
        // is not explicitly false). If so, the model's completion starts
        // mid-reasoning and the parser should begin in reasoning mode.
        let prompt_injected_reasoning = formatted_prompt
            .as_ref()
            .is_some_and(|p| p.trim_end().ends_with("<think>"));

        let tokenize_start = Instant::now();
        let (token_ids, annotations) = {
            let _nvtx = dynamo_nvtx_range!("preprocess.tokenize");
            self.gather_tokens(request, formatted_prompt.as_deref(), tracker)
                .with_context(|| "Failed to gather tokens")?
        };
        TOKENIZE_SECONDS.observe(tokenize_start.elapsed().as_secs_f64());

        let _mm_image_entries = self
            .gather_multi_modal_data(request, &mut builder, formatted_prompt.as_deref())
            .await
            .with_context(|| "Failed to gather multimodal data")?;

        // Build the MM-aware view (expanded routing_token_ids + per-block
        // mm_hashes) for the KV router. No-op when no images are present or
        // the model has no resolved image-placeholder.
        #[cfg(feature = "mm-routing")]
        self.gather_mm_exact_routing_info(
            &mut builder,
            &_mm_image_entries,
            &token_ids,
            formatted_prompt.as_deref(),
        )
        .with_context(|| "Failed to build MM routing info")?;

        // Install tokens on the builder. Done after MM routing built its
        // view so the routing-side borrow stays cheap and builder ownership
        // moves once.
        builder.token_ids(token_ids);

        STAGE_DURATION_SECONDS
            .with_label_values(&[STAGE_PREPROCESS])
            .observe(preprocess_start.elapsed().as_secs_f64());

        if let Some(nvext) = request.nvext()
            && let Some(router_params) = &nvext.router
        {
            builder.router(Some(router_params.clone()));
        }

        let mut preprocessed = builder.build()?;

        // If omitted, allow generation up to the remaining context length. Responses requests
        // preserve omission so backend adapters can compute the dynamic cap from their
        // effective prompt length/tokenization.
        if preprocessed.stop_conditions.max_tokens.is_none()
            && let Some(max_tokens) = Self::omitted_max_tokens_default(
                preprocessed.token_ids.len(),
                self.context_length,
                options,
            )
        {
            preprocessed.stop_conditions.max_tokens = Some(max_tokens);
        }

        Ok((preprocessed, annotations, prompt_injected_reasoning))
    }

    pub fn builder<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
    ) -> Result<PreprocessedRequestBuilder> {
        let mut builder = PreprocessedRequest::builder();
        builder.model(request.model());

        let mut stop_conditions = request.extract_stop_conditions()?;
        // Harmony's `<|call|>` is BOTH the tool-call terminator the parser needs in the
        // decoded text AND a gpt-oss EOS token. Hiding it like other EOS tokens strips the
        // terminator before the harmony parser sees it, so tool calls are silently dropped
        // (vLLM >=0.22 correctly stops on `<|call|>`, which is when this surfaces). Keep it
        // visible — out of the hidden stop set — when the harmony tool-call parser is active.
        // Generation still stops on it via the worker's own EOS; we only avoid *hiding* it.
        let visible_eos: Vec<u32> = if matches!(self.tool_call_parser.as_deref(), Some("harmony")) {
            dynamo_parsers::harmony_terminator_token_ids()
        } else {
            Vec::new()
        };
        if let Some(stop_tokens) = &mut stop_conditions.stop_token_ids_hidden {
            for eos_token in self.model_info.eos_token_ids() {
                if !visible_eos.contains(&eos_token) && !stop_tokens.contains(&eos_token) {
                    stop_tokens.push(eos_token);
                }
            }
        } else {
            stop_conditions.stop_token_ids_hidden = Some(
                self.model_info
                    .eos_token_ids()
                    .into_iter()
                    .filter(|t| !visible_eos.contains(t))
                    .collect(),
            );
        }

        // apply ignore eos if not already set
        stop_conditions.apply_ignore_eos();

        if !stop_conditions.ignore_eos.unwrap_or(false) {
            builder.eos_token_ids(self.model_info.eos_token_ids());
        }

        builder.stop_conditions(stop_conditions);
        builder.sampling_options(request.extract_sampling_options()?);

        // Some parsers rely on `<|tool_call>`, `<|channel>`, etc. being
        // visible in the decoded text. The default `skip_special_tokens=true`
        // strips them and silently bypasses parsing. Mirror upstream's
        // per-parser `adjust_request` hook by flipping the default to false
        // for parsers that need special tokens preserved, unless the caller
        // has explicitly set `skip_special_tokens`.
        let mut output_options = request.extract_output_options()?;
        if output_options.skip_special_tokens.is_none()
            && Self::parser_requires_special_tokens(
                self.tool_call_parser.as_deref(),
                self.runtime_config.reasoning_parser.as_deref(),
            )
        {
            output_options.skip_special_tokens = Some(false);
        } else if Self::special_tokens_will_be_stripped(
            output_options.skip_special_tokens,
            self.tool_call_parser.as_deref(),
            self.runtime_config.reasoning_parser.as_deref(),
        ) {
            // Caller forced `skip_special_tokens=true` while a special-token-
            // dependent parser is active. The engine's markers (e.g. harmony
            // `<|channel|>` / `<|message|>`) get stripped before the parser
            // runs, so tool_calls / reasoning_content come back empty and the
            // markup leaks into `content`. Warn once: this is a silent,
            // deterministic correctness loss that no parser fixture can catch.
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    tool_call_parser = ?self.tool_call_parser,
                    reasoning_parser = ?self.runtime_config.reasoning_parser,
                    "skip_special_tokens=true requested while a special-token-dependent parser is active; the engine's special tokens will be stripped before parsing, so tool_calls/reasoning_content will be empty and the markup will leak into content. Unset skip_special_tokens (Dynamo defaults it to false for these parsers) or set it to false."
                );
            });
        }
        builder.output_options(output_options);
        builder.annotations(request.annotations().unwrap_or_default());
        builder.mdc_sum(Some(self.mdcsum.clone()));
        let lora_name = self.lora_name.clone();

        // Extract routing hints from nvext if present
        if let Some(nvext) = request.nvext() {
            // Build routing hints from nvext fields
            let hints = nvext.agent_hints.as_ref();
            builder.request_timestamp_ms(nvext.request_timestamp_ms);
            builder.agent_context(nvext.agent_context.clone());
            let routing = RoutingHints {
                backend_instance_id: nvext.backend_instance_id,
                prefill_worker_id: nvext.prefill_worker_id,
                decode_worker_id: nvext.decode_worker_id,
                dp_rank: nvext.dp_rank,
                prefill_dp_rank: nvext.prefill_dp_rank,
                expected_output_tokens: hints.and_then(|h| h.osl),
                priority_jump: hints.and_then(|h| {
                    h.priority
                        .map(|priority| priority.max(0) as f64)
                        .or(h.latency_sensitivity)
                }),
                priority: hints.and_then(|h| h.priority),
                lora_name,
                allowed_worker_ids: None,
                session_control: nvext.session_control.clone(),
                routing_constraints: nvext.routing_constraints.clone(),
            };
            builder.routing(Some(routing));
        } else if lora_name.is_some() {
            // Ensure routing hints exist when we have LoRA,
            // even when nvext is absent.
            builder.routing(Some(RoutingHints {
                lora_name,
                ..Default::default()
            }));
        }

        if let Some(extra_args) = Self::backend_extra_args(request) {
            builder.extra_args(Some(extra_args));
        }

        // Forward mm_processor_kwargs (e.g. use_audio_in_video) to the backend.
        builder.mm_processor_kwargs(request.mm_processor_kwargs().cloned());

        Ok(builder)
    }

    pub fn apply_template<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
    ) -> Result<Option<String>> {
        if let PromptInput::Text(_) = request.prompt_input_type()
            && let Some(TextInput::Single(_)) = request.extract_text()
        {
            let use_raw_prompt = request
                .nvext()
                .is_some_and(|ext| ext.use_raw_prompt.unwrap_or(false));

            let formatted_prompt = if use_raw_prompt {
                match request.raw_prompt() {
                    Some(prompt) => prompt,
                    None => {
                        tracing::warn!("Raw prompt requested but not available");
                        self.formatter.render(request)?
                    }
                }
            } else {
                self.formatter.render(request)?
            };
            Ok(Some(formatted_prompt))
        } else {
            Ok(None)
        }
    }

    /// Replace inline `data:` URLs with empty strings in message content parts.
    /// Preserves HTTP(S) URLs, text content, and overall message structure.
    fn strip_inline_data_urls(messages: &mut serde_json::Value) {
        let Some(arr) = messages.as_array_mut() else {
            return;
        };
        for msg in arr {
            let Some(content) = msg.get_mut("content") else {
                continue;
            };
            let Some(parts) = content.as_array_mut() else {
                continue;
            };
            for part in parts {
                for key in ["image_url", "video_url", "audio_url"] {
                    if let Some(media) = part.get_mut(key)
                        && let Some(url) = media.get_mut("url")
                        && url.as_str().is_some_and(|s| s.starts_with("data:"))
                    {
                        *url = serde_json::Value::String(String::new());
                    }
                }
            }
        }
    }

    pub async fn gather_multi_modal_data<
        R: OAIChatLikeRequest + MediaRequestExt + NvExtProvider,
    >(
        &self,
        request: &R,
        builder: &mut PreprocessedRequestBuilder,
        formatted_prompt: Option<&str>,
    ) -> Result<Vec<MmImageEntry>> {
        let mut media_map: MultimodalDataMap = HashMap::new();
        let mut fetch_tasks: Vec<(String, &ChatCompletionRequestUserMessageContentPart)> =
            Vec::new();
        // Per-image (mm_hash, width, height) for the MM-routing path.
        // Accumulated in message order so we don't walk messages twice.
        // Cleared and returned to the caller; empty for non-image / text-only requests.
        #[cfg(feature = "mm-routing")]
        let mut mm_image_entries: Vec<MmImageEntry> = Vec::new();
        // Total `image_url` content parts in the request. Bumped at every
        // image part regardless of which fetch path handles it. Used at
        // `mm_hashes` forwarding time: if `mm_image_entries.len()` is
        // smaller, we omit `mm_hashes` for the whole request rather than
        // ship a partial / misaligned UUID list to vLLM.
        //
        // The mismatch is only reachable on the URL-passthrough path
        // (no media_loader): each `fetch_image_dims_uncached` failure logs
        // a warn and skips its `mm_image_entries.push`, but doesn't abort
        // the request. The decoded path (`has_media_loader`) propagates
        // any dim-fetch failure via `?`, so the request errors out before
        // mm_hashes forwarding is even considered.
        #[cfg(feature = "mm-routing")]
        let mut total_image_count: usize = 0;
        // For the URL-passthrough case (media_loader is None) we collect image
        // URLs here and resolve dims via header-only HTTP after the loop so we
        // can issue all fetches in parallel.
        #[cfg(feature = "mm-routing")]
        let mut url_passthrough_images: Vec<(u64, String)> = Vec::new();

        let Some(messages) = request.typed_messages() else {
            return Ok(Vec::new());
        };
        let has_media_loader = self.media_loader.is_some();

        for message in messages.iter() {
            let content_parts = match message {
                ChatCompletionRequestMessage::User(u) => match &u.content {
                    ChatCompletionRequestUserMessageContent::Array(parts) => parts,
                    _ => continue,
                },
                _ => continue,
            };
            for content_part in content_parts.iter() {
                if has_media_loader {
                    let type_str = match content_part {
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(_) => "image_url",
                        ChatCompletionRequestUserMessageContentPart::VideoUrl(_) => "video_url",
                        ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => "audio_url",
                        _ => continue,
                    };
                    #[cfg(feature = "mm-routing")]
                    if type_str == "image_url" {
                        total_image_count += 1;
                    }
                    fetch_tasks.push((type_str.to_string(), content_part));
                } else {
                    let (type_str, url) = match content_part {
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(p) => {
                            ("image_url", p.image_url.url.clone())
                        }
                        ChatCompletionRequestUserMessageContentPart::VideoUrl(p) => {
                            ("video_url", p.video_url.url.clone())
                        }
                        ChatCompletionRequestUserMessageContentPart::AudioUrl(p) => {
                            ("audio_url", p.audio_url.url.clone())
                        }
                        _ => continue,
                    };
                    #[cfg(feature = "mm-routing")]
                    if type_str == "image_url" {
                        total_image_count += 1;
                        let mm_hash = Self::hash_image_url(url.as_str());
                        url_passthrough_images.push((mm_hash, url.to_string()));
                    }
                    media_map
                        .entry(type_str.to_string())
                        .or_default()
                        .push(MultimodalData::Url(url));
                }
            }
        }

        // Execute all fetch tasks
        if !fetch_tasks.is_empty() {
            let loader = self.media_loader.as_ref().unwrap();
            let media_io_kwargs = request.media_io_kwargs();
            let results = futures::future::join_all(fetch_tasks.iter().map(|(_, content_part)| {
                loader.fetch_and_decode_media_part(content_part, media_io_kwargs)
            }))
            .await;

            for ((type_str, _content_part), result) in
                fetch_tasks.into_iter().zip(results.into_iter())
            {
                // if one item fails, errors the whole request, other items will be cleaned up by Drop
                let rdma_descriptor = result?;

                // Decoded RDMA descriptor carries shape `[H, W, C]`.
                // Image-only; MM-routing doesn't cover audio/video.
                #[cfg(feature = "mm-routing")]
                if type_str == "image_url" {
                    let shape = &rdma_descriptor.tensor_info.shape;
                    if shape.len() >= 2 {
                        let h = shape[0] as u32;
                        let w = shape[1] as u32;
                        let url_str = match _content_part {
                            ChatCompletionRequestUserMessageContentPart::ImageUrl(p) => {
                                p.image_url.url.as_str()
                            }
                            _ => unreachable!(
                                "rdma image_url descriptor only originates from ImageUrl content parts"
                            ),
                        };
                        // Frontend-decode path: hash the decoded RGB bytes so
                        // the same image reached via different (signed) URLs
                        // collides on the same `mm_hash` and routes to the
                        // worker that already has those KV blocks. Fall back
                        // to URL hashing only if the descriptor lost local
                        // storage (e.g. reconstructed from the wire), which
                        // shouldn't happen on the frontend.
                        let (mm_hash, hash_source) = match rdma_descriptor.content_hash() {
                            Some(h) => (h, "decoded_bytes"),
                            None => (Self::hash_image_url(url_str), "url_fallback"),
                        };
                        if let Some(counter) = self.image_token_counter.as_ref() {
                            let n = counter.count_tokens(w, h);
                            tracing::debug!(
                                target: "mm_routing",
                                model = counter.model_id(),
                                width = w,
                                height = h,
                                tokens = n,
                                mm_hash = mm_hash,
                                source = hash_source,
                                "image-token count"
                            );
                        }
                        mm_image_entries.push(MmImageEntry {
                            mm_hash,
                            width: w,
                            height: h,
                        });
                    }
                }

                media_map
                    .entry(type_str)
                    .or_default()
                    .push(MultimodalData::Decoded(rdma_descriptor));
            }
        }

        // URL-passthrough path (media_loader is None): fetch image headers in
        // parallel to get (W, H) per image without downloading the full bytes.
        // Enables MM-aware routing for backends that register
        // `media_decoder: null` and decode images on the worker.
        #[cfg(feature = "mm-routing")]
        if !url_passthrough_images.is_empty() {
            let dim_results = futures::future::join_all(
                url_passthrough_images
                    .iter()
                    .map(|(mm_hash, url)| Self::fetch_image_dims(*mm_hash, url.as_str())),
            )
            .await;
            for ((mm_hash, url), dim_res) in url_passthrough_images.into_iter().zip(dim_results) {
                match dim_res {
                    Ok((w, h)) => {
                        if let Some(counter) = self.image_token_counter.as_ref() {
                            let n = counter.count_tokens(w, h);
                            tracing::debug!(
                                target: "mm_routing",
                                model = counter.model_id(),
                                width = w,
                                height = h,
                                tokens = n,
                                mm_hash = mm_hash,
                                source = "url_passthrough_header_fetch",
                                "image-token count"
                            );
                        }
                        mm_image_entries.push(MmImageEntry {
                            mm_hash,
                            width: w,
                            height: h,
                        });
                    }
                    Err(e) => {
                        // Redact `data:` URIs to just the media-type prefix —
                        // the comma-separated payload is the entire (base64)
                        // image body and ships in logs would be log bloat /
                        // potential PII spillage if logs are aggregated.
                        let url_for_log = if url.starts_with("data:") {
                            url.split_once(',')
                                .map(|(p, _)| format!("{p},<redacted>"))
                                .unwrap_or_else(|| "data:<redacted>".to_string())
                        } else {
                            url.to_string()
                        };
                        tracing::warn!(
                            target: "mm_routing",
                            url = %url_for_log,
                            error = %e,
                            "mm-routing: failed to fetch image dims; MM routing entry skipped"
                        );
                    }
                }
            }
        }

        if !media_map.is_empty() {
            builder.multi_modal_data(Some(media_map));

            // Preserve original messages and formatted prompt in extra_args for multimodal
            // workers (e.g., TRT-LLM needs messages and the template-rendered prompt with
            // <image> placeholders for embedding-path / NIXL flows).
            let messages_json = serde_json::to_value(request.messages())?;
            let mut extra_args = serde_json::json!({
                "messages": messages_json
            });

            // Strip redundant inline data: URLs only when frontend decoding is active
            // (media_loader decoded the images into RDMA descriptors). TRT-LLM and
            // other backends that pass URLs through still need the original data: URIs.
            if self.media_loader.is_some() {
                Self::strip_inline_data_urls(&mut extra_args["messages"]);
            }

            if let Some(prompt) = formatted_prompt {
                // Clone here is the single owned allocation we actually need:
                // the prompt is inserted into the request's `extra_args` JSON.
                // The caller still holds the original `String`; passing
                // `Option<&str>` keeps text-only requests (no MM) clone-free.
                extra_args["formatted_prompt"] = serde_json::Value::String(prompt.to_string());
            }

            if let Some(serde_json::Value::Object(backend_extra_args)) =
                Self::backend_extra_args(request)
            {
                let extra_args_obj = extra_args
                    .as_object_mut()
                    .expect("multimodal extra_args must be an object");
                extra_args_obj.extend(backend_extra_args);
            }

            // Forward routing-side mm_hashes in `extra_args["mm_hashes"]` so the
            // backend's KV events publish under the same key the router computes.
            // Always the canonical 16-char hex (u64); each backend adapts it:
            // sglang reads `int(hex, 16)` as-is, vLLM pads to its 64-char
            // BlockStored form. No information is lost — both carry the same u64.
            //
            // Skip forwarding entirely if any image failed dim resolution —
            // a shorter `mm_hashes` list would misalign with the image
            // positions the backend derives from `multi_modal_data`, and
            // the wrong UUIDs would get injected onto the wrong images.
            #[cfg(feature = "mm-routing")]
            if self.routing_image_token_id.is_some()
                && !mm_image_entries.is_empty()
                && mm_image_entries.len() == total_image_count
            {
                let hexes: Vec<serde_json::Value> = mm_image_entries
                    .iter()
                    .map(|e| serde_json::Value::String(format!("{:016x}", e.mm_hash)))
                    .collect();
                extra_args["mm_hashes"] = serde_json::Value::Array(hexes);
            } else if !mm_image_entries.is_empty() && self.routing_image_token_id.is_some() {
                tracing::warn!(
                    target: "mm_routing",
                    resolved = mm_image_entries.len(),
                    expected = total_image_count,
                    "mm-routing: not all images resolved an MM-routing entry; skipping mm_hashes forwarding"
                );
            }

            builder.extra_args(Some(extra_args));
        }

        #[cfg(feature = "mm-routing")]
        return Ok(mm_image_entries);
        #[cfg(not(feature = "mm-routing"))]
        Ok(Vec::new())
    }

    /// Build `MmRoutingInfo` for exact MM-aware KV routing. The worker-bound
    /// `token_ids` are unchanged — only the routing-side view is expanded.
    /// `formatted_prompt` is only consumed for Phi-3-style numbered placeholder
    /// templates; single-special-token families (Qwen-VL, LLaVA) ignore it.
    /// Returns `Ok(())` with no work performed on any precondition miss
    /// (caller falls back to text-prefix routing).
    #[cfg(feature = "mm-routing")]
    pub fn gather_mm_exact_routing_info(
        &self,
        builder: &mut PreprocessedRequestBuilder,
        mm_image_entries: &[MmImageEntry],
        token_ids: &[crate::protocols::TokenIdType],
        formatted_prompt: Option<&str>,
    ) -> Result<()> {
        use crate::protocols::common::preprocessor::MmRoutingInfo;

        if mm_image_entries.is_empty() {
            return Ok(());
        }
        let Some(find_token_id) = self.routing_image_token_id else {
            tracing::debug!(
                target: "mm_routing",
                "routing_image_token_id unresolved; skipping MM routing info"
            );
            return Ok(());
        };
        let Some(counter) = self.image_token_counter.as_ref() else {
            tracing::debug!(
                target: "mm_routing",
                "image_token_counter unavailable; skipping MM routing info"
            );
            return Ok(());
        };
        let block_size = self.kv_cache_block_size;
        if block_size == 0 {
            tracing::debug!(
                target: "mm_routing",
                "kv_cache_block_size is 0; skipping MM routing info"
            );
            return Ok(());
        }

        // Single-special-token placeholders (Qwen-VL `<|image_pad|>`, LLaVA
        // `<image>`) emit one `find_token_id` per image in the tokenized
        // prompt and hit the fast path below. Numbered-text placeholders
        // (Phi-3 `<|image_N|>`) BPE-shatter and need the splice helper to
        // mirror what the worker hashes on.
        let placeholder_count = token_ids.iter().filter(|&&t| t == find_token_id).count();
        let normalized_token_ids: std::borrow::Cow<'_, [crate::protocols::TokenIdType]> =
            if placeholder_count == mm_image_entries.len() {
                std::borrow::Cow::Borrowed(token_ids)
            } else if let Some(tpl) = self.image_placeholder_template
                && tpl.contains("{n}")
                && let Some(prompt) = formatted_prompt
            {
                match self.splice_phi3_numbered_placeholders_at_token_level(
                    prompt,
                    tpl,
                    find_token_id,
                    mm_image_entries.len(),
                ) {
                    Some(spliced) => std::borrow::Cow::Owned(spliced),
                    None => {
                        tracing::warn!(
                            target: "mm_routing",
                            placeholder_count,
                            image_count = mm_image_entries.len(),
                            routing_image_token_id = find_token_id,
                            placeholder_template = tpl,
                            "splice failed for numbered placeholder; \
                             skipping MM routing info (text-prefix routing only)"
                        );
                        return Ok(());
                    }
                }
            } else {
                tracing::warn!(
                    target: "mm_routing",
                    placeholder_count,
                    image_count = mm_image_entries.len(),
                    routing_image_token_id = find_token_id,
                    "placeholder token count in tokenized prompt does not match image count; \
                     skipping MM routing info (text-prefix routing only)"
                );
                return Ok(());
            };

        // Compute per-image N via the registry + run the expansion.
        let n_tokens: Vec<usize> = mm_image_entries
            .iter()
            .map(|e| counter.count_tokens(e.width, e.height))
            .collect();
        let n_total: usize = n_tokens.iter().sum();

        // Canonical pad_value fill at image positions for ALL backends. sglang
        // consumes pad_value natively; vLLM events are normalized to pad_value
        // in the kv-router (see `create_stored_blocks`), so the frontend stays
        // backend-agnostic. pad_value formula pinned by
        // `pad_value_matches_sglang_protocol` in dynamo_kv_router.
        //
        // BOS ownership: when the Phi-3 splice helper produced
        // `normalized_token_ids` (Cow::Owned), it has already emitted the
        // leading BOS as part of its complete sequence. Only the non-splice
        // path (Cow::Borrowed = raw tokenized prompt) needs the caller to
        // prepend BOS here. Avoids the prior split-brain where the leading
        // push was here and the mid-segment pushes lived inside the helper.
        let splice_owns_bos = matches!(normalized_token_ids, std::borrow::Cow::Owned(_));
        let bos_extra = (!splice_owns_bos && self.routing_prepend_bos.is_some()) as usize;
        let mut expanded: Vec<crate::protocols::TokenIdType> =
            Vec::with_capacity(normalized_token_ids.len() + n_total + bos_extra);
        if !splice_owns_bos && let Some(bos) = self.routing_prepend_bos {
            expanded.push(bos);
        }
        let mut i = 0usize;
        for &t in normalized_token_ids.iter() {
            if t == find_token_id && i < mm_image_entries.len() {
                let fill_token =
                    dynamo_kv_router::protocols::pad_value_for_mm_hash(mm_image_entries[i].mm_hash);
                expanded.extend(std::iter::repeat_n(fill_token, n_tokens[i]));
                i += 1;
            } else {
                expanded.push(t);
            }
        }

        // Pad to a whole multiple of kv_cache_block_size. The router's
        // compute_block_hash_for_seq only hashes whole blocks, so the partial
        // tail block doesn't influence routing either way; aligning the length
        // keeps our routing_token_ids and `block_mm_infos` agreeing on count.
        // `div_ceil` guarantees `total_tokens >= expanded.len()`, so resize
        // only ever grows.
        let total_tokens = expanded.len().div_ceil(block_size) * block_size;
        if expanded.len() < total_tokens {
            expanded.resize(total_tokens, 0);
        }

        // pad_value already encodes mm_hash in the routing tokens the router
        // hashes, so block_mm_infos is always empty (the canonical scheme for
        // both backends). The mm identity rides in the token stream, not a
        // side channel.
        tracing::debug!(
            target: "mm_routing",
            n_images = mm_image_entries.len(),
            block_size,
            total_tokens,
            "MmRoutingInfo built (exact, pad_value)"
        );

        builder.mm_routing_info(Some(MmRoutingInfo {
            routing_token_ids: expanded,
            block_mm_infos: Vec::new(),
        }));
        Ok(())
    }

    /// Build routing-side tokens for Phi-3-vision's `<|image_N|>` numbered
    /// placeholder template so the router's per-block hashes match the
    /// worker's BlockStored events byte-for-byte.
    ///
    /// **Family contract — Phi-3-only.** The encode-decode roundtrip +
    /// per-segment BOS pattern here is specific to Phi-3's HF processor
    /// (and any future `LlamaTokenizer`-family with `<|image_{n}|>`
    /// placeholders): the processor splits the prompt at `<|image_N|>` and
    /// tokenizes each segment with `add_special_tokens=true`, prefixing
    /// every segment (including the first) with a fresh BOS, and decodes-
    /// then-re-encodes across special-token boundaries, which inserts
    /// whitespace after each special token and bumps the SentencePiece
    /// prefix token (e.g. 29871 `▁` -> 259 `▁▁` for `<|user|>\n`). Both
    /// effects are reproduced here.
    ///
    /// Caller is `gather_mm_exact_routing_info`, which gates on
    /// `placeholder_tpl.contains("{n}")`. The `routing_prepend_bos`
    /// requirement below acts as the family guard: a future model with a
    /// `{n}`-style placeholder but no BOS prepend (i.e. not a
    /// LlamaTokenizer-family) would not benefit from this roundtrip and
    /// could silently produce wrong block hashes, so we bail and let the
    /// caller fall back to text-prefix routing.
    ///
    /// Returns one `find_token_id` per image; the caller's expansion loop
    /// multiplies them to the per-image patch count. Leading BOS is
    /// emitted here as the first element of the returned vector — the
    /// caller does NOT prepend its own BOS when this helper succeeds.
    /// Returns `None` (caller falls back to text-prefix routing) when the
    /// family guard trips or on any tokenize/decode failure.
    #[cfg(feature = "mm-routing")]
    fn splice_phi3_numbered_placeholders_at_token_level(
        &self,
        prompt: &str,
        placeholder_tpl: &str,
        find_token_id: crate::protocols::TokenIdType,
        expected_count: usize,
    ) -> Option<Vec<crate::protocols::TokenIdType>> {
        // Family guard: the encode-decode roundtrip + per-segment BOS
        // semantics are LlamaTokenizer-family-specific. `routing_prepend_bos`
        // is set iff the model declares `add_bos_token: true` in
        // tokenizer_config.json, which is the marker for that family.
        let bos = self.routing_prepend_bos?;

        // Encode-decode roundtrip mirrors vLLM's text-substitute fallback,
        // which is what the Phi-3 worker actually hashes on.
        let prompt_owned: String = self
            .tokenizer
            .encode(prompt)
            .ok()
            .and_then(|enc| self.tokenizer.decode(enc.token_ids(), false).ok())
            .map(Into::into)?;
        let prompt: &str = &prompt_owned;

        let mut byte_ranges: Vec<(usize, usize)> = Vec::with_capacity(expected_count);
        let mut search_from = 0usize;
        for idx in 1..=expected_count {
            let pattern = placeholder_tpl.replace("{n}", &idx.to_string());
            let rel = prompt[search_from..].find(&pattern)?;
            let start = search_from + rel;
            let end = start + pattern.len();
            byte_ranges.push((start, end));
            search_from = end;
        }

        // Leading BOS is emitted here so the caller's general-purpose
        // BOS-prepend path can skip when this helper produced the
        // normalized tokens (avoids the prior split-brain where leading
        // BOS was pushed by the caller and mid/suffix BOS pushed here).
        let mut result: Vec<crate::protocols::TokenIdType> = vec![bos];
        let mut prev_end = 0usize;
        for &(ph_start, ph_end) in byte_ranges.iter() {
            let seg = &prompt[prev_end..ph_start];
            let seg_enc = self.tokenizer.encode(seg).ok()?;
            result.extend_from_slice(seg_enc.token_ids());
            result.push(find_token_id);
            // Mid-prompt segments get a fresh BOS — Phi-3's HF processor
            // tokenizes each segment with add_special_tokens=true.
            result.push(bos);
            prev_end = ph_end;
        }
        let suffix_enc = self.tokenizer.encode(&prompt[prev_end..]).ok()?;
        result.extend_from_slice(suffix_enc.token_ids());
        Some(result)
    }

    /// xxh3-64 of the raw URL bytes. Used as the routing `mm_hash` in the
    /// URL-passthrough path: two requests with byte-identical URLs route to
    /// the same worker, anything else routes independently.
    ///
    /// We deliberately do NOT strip cache-buster / signed-URL query
    /// parameters — a query string like `?v=2` could mean either "new
    /// cache-busted fetch of the same image" or "version 2 of a different
    /// image", and the URL alone doesn't tell us which. Keeping the hash
    /// URL-identical avoids the heuristic and the false-positive collisions
    /// that come with it. Workloads with rotating signed URLs (S3, GCS,
    /// Azure SAS) should use `--frontend-decoding`: that path hashes the
    /// decoded RGB bytes instead, so cross-URL cache reuse is restored
    /// without depending on URL conventions.
    #[cfg(feature = "mm-routing")]
    fn hash_image_url(url: &str) -> u64 {
        xxhash_rust::xxh3::xxh3_64(url.as_bytes())
    }

    /// Header-only image dim fetch. For HTTP/HTTPS we issue a Range request
    /// for the first 64 KB (covers PNG/WebP in <1 KB and JPEG SOF in worst
    /// case). For data: URIs we decode the base64 payload locally and parse
    /// the header. Caller treats Err as "MM routing entry unavailable for
    /// this image" — request still proceeds with text-prefix routing.
    ///
    /// Results are cached by `mm_hash` so repeated requests for the same image
    /// (typical of multi-turn / session workloads) hit the cache and skip the
    /// HTTP fetch entirely. Without this cache, sticky-routing workloads pay
    /// 4–5× HTTP Range fetches per request just to compute routing tokens.
    #[cfg(feature = "mm-routing")]
    async fn fetch_image_dims(mm_hash: u64, url: &str) -> Result<(u32, u32)> {
        use moka::future::Cache;
        use std::sync::LazyLock;

        // Bounded sharded LRU (moka uses TinyLFU internally — sharded write
        // locks, lock-free reads). Replaces an earlier hand-rolled DashMap +
        // tokio::sync::Notify singleflight; moka's `try_get_with` provides
        // both singleflight and bounded LRU eviction in one primitive.
        //
        //   max_capacity:  100k entries (~5 MB at ~50 B/entry incl. moka
        //                  bookkeeping). Caps memory under unbounded URL
        //                  pools (signed-URL refresh, image proxies).
        //   time_to_live:  24h. Bounds staleness if a URL is re-uploaded
        //                  with new content. Independent of capacity-based
        //                  eviction, which kicks in earlier under load.
        static DIM_CACHE: LazyLock<Cache<u64, (u32, u32)>> = LazyLock::new(|| {
            Cache::builder()
                .max_capacity(100_000)
                .time_to_live(Duration::from_secs(24 * 60 * 60))
                .build()
        });

        // Hot path: avoid allocating an owned URL on cache hit. moka's
        // `get` is async because it may do a small amount of bookkeeping
        // for the LRU/TinyLFU policy.
        if let Some(dims) = DIM_CACHE.get(&mm_hash).await {
            return Ok(dims);
        }

        // Cold path: take an owned String so the init future can be
        // 'static (moka may move waiters across executor threads). Because
        // try_get_with does built-in singleflight, concurrent callers for
        // the same `mm_hash` collapse into a single fetch.
        let url_owned = url.to_string();
        DIM_CACHE
            .try_get_with(mm_hash, async move {
                Self::fetch_image_dims_uncached(&url_owned)
                    .await
                    .map_err(|e| e.to_string())
            })
            .await
            .map_err(|e| anyhow::anyhow!("fetch_image_dims failed: {}", e))
    }

    #[cfg(feature = "mm-routing")]
    async fn fetch_image_dims_uncached(url: &str) -> Result<(u32, u32)> {
        use image::ImageReader;
        use std::io::Cursor;

        // Most JPEG SOF markers and PNG/WebP headers fit in the first 4 KB.
        // Start small and only escalate to 64 KB if the parser fails on the
        // truncated header.
        const SMALL_RANGE: usize = 4 * 1024 - 1;
        const LARGE_RANGE: usize = 64 * 1024 - 1;
        // Per-Range tighter bound than MediaFetcher's 30 s default — dim
        // fetch is best-effort; on a slow remote we'd rather skip MM
        // routing for this image than starve the request.
        const DIM_FETCH_TIMEOUT: Duration = Duration::from_secs(10);

        if let Some(rest) = url.strip_prefix("data:") {
            let comma = rest
                .find(',')
                .ok_or_else(|| anyhow::anyhow!("malformed data URI: no comma"))?;
            let prefix = &rest[..comma];
            let payload = &rest[comma + 1..];
            let bytes: Vec<u8> = if prefix.contains(";base64") {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(payload)
                    .map_err(|e| anyhow::anyhow!("data URI base64 decode: {}", e))?
            } else {
                payload.as_bytes().to_vec()
            };
            let (w, h) = ImageReader::new(Cursor::new(&bytes))
                .with_guessed_format()?
                .into_dimensions()?;
            return Ok((w, h));
        }

        if !(url.starts_with("http://") || url.starts_with("https://")) {
            anyhow::bail!("unsupported url scheme for dim fetch: {}", url);
        }

        // `DIM_FETCH_MEDIA_FETCHER` and `DIM_FETCH_HTTP_CLIENT` are
        // module-scope `LazyLock`s forced at startup in `new_with_parts`
        // for MM-routable preprocessors — see their definitions for the
        // lifecycle and policy contract.

        // Pre-flight SSRF check on the original URL. Redirect targets are
        // revalidated by the Client's redirect policy, and DNS-resolved
        // IPs are filtered by the resolver — so a URL that passes here
        // can't escape the contract on the wire either.
        let parsed = url::Url::parse(url)?;
        DIM_FETCH_MEDIA_FETCHER
            .check_if_url_allowed_with_dns(&parsed)
            .await?;

        let mut range_end = SMALL_RANGE;
        loop {
            let resp = DIM_FETCH_HTTP_CLIENT
                .get(url)
                .header("Range", format!("bytes=0-{}", range_end))
                .timeout(DIM_FETCH_TIMEOUT)
                .send()
                .await?;
            let status = resp.status();
            // Require 206 Partial Content — if the origin ignored the
            // Range header and answered 200 OK, `.bytes()` would buffer
            // the full image into memory. Bail in that case rather than
            // download an unbounded payload just to peek at dimensions.
            // The caller treats Err as "MM routing entry unavailable for
            // this image", which falls back to text-prefix routing.
            if status != reqwest::StatusCode::PARTIAL_CONTENT {
                anyhow::bail!(
                    "image dim fetch expected 206 Partial Content, got HTTP {}",
                    status
                );
            }
            let bytes = resp.bytes().await?;
            match ImageReader::new(Cursor::new(&bytes))
                .with_guessed_format()
                .and_then(|r| r.into_dimensions().map_err(std::io::Error::other))
            {
                Ok((w, h)) => return Ok((w, h)),
                Err(_) if range_end < LARGE_RANGE => {
                    range_end = LARGE_RANGE;
                    continue;
                }
                Err(e) => anyhow::bail!("image header parse failed after 64KB: {}", e),
            }
        }
    }

    /// Tokenize the request and return the token ids alongside any annotations
    /// the caller asked for. The caller owns the result and is responsible for
    /// installing it on the builder via `builder.token_ids(...)` once any
    /// downstream consumers (e.g. MM-routing) have borrowed it.
    pub fn gather_tokens<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
        formatted_prompt: Option<&str>,
        tracker: Option<&RequestTracker>,
    ) -> Result<(Vec<crate::protocols::TokenIdType>, HashMap<String, String>)> {
        let mut annotations = HashMap::new();
        let mut token_count: Option<usize> = None;
        let mut tokens_out: Vec<crate::protocols::TokenIdType> = Vec::new();
        // match request type before any conversion/processing
        match request.prompt_input_type() {
            PromptInput::Tokens(_) => {
                if let Some(token_input) = request.extract_tokens() {
                    match token_input {
                        TokenInput::Single(tokens) => {
                            token_count = Some(tokens.len());
                            tokens_out = tokens;
                        }
                        TokenInput::Batch(token_batches) => {
                            if token_batches.len() == 1 {
                                token_count = Some(token_batches[0].len());
                                tokens_out = token_batches[0].clone();
                            } else {
                                bail!(
                                    "Batch token input not supported for more than one token in requests (got {})",
                                    token_batches.len()
                                );
                            }
                        }
                    }
                }
            }
            PromptInput::Text(_) => {
                if let Some(text_input) = request.extract_text() {
                    match text_input {
                        TextInput::Single(raw_prompt) => {
                            if let Some(f) = formatted_prompt
                                && request.has_annotation(ANNOTATION_FORMATTED_PROMPT)
                            {
                                annotations
                                    .insert(ANNOTATION_FORMATTED_PROMPT.to_string(), f.to_string());
                            }

                            // Completions will use raw_prompt, no template.
                            // Borrow either input — no allocation needed; the
                            // tokenizer accepts `&str`.
                            let prompt: &str = formatted_prompt.unwrap_or(raw_prompt.as_str());

                            // If nvext.token_data is present, use the pre-computed tokens
                            // directly and skip tokenization.  This avoids redundant
                            // tokenization when an external component (e.g. the GAIE EPP
                            // KV-router) has already tokenized the prompt.
                            // When backend_instance_id is set without token_data, warn
                            // but fall back to tokenization (backward compat for non-GAIE
                            // routers that set the header without providing tokens).
                            let has_backend_instance_id = request
                                .nvext()
                                .and_then(|ext| ext.backend_instance_id)
                                .is_some();

                            let token_data =
                                request.nvext().and_then(|ext| ext.token_data.as_ref());

                            let (tokens_vec, skip_token_annotation) = if let Some(tokens) =
                                token_data
                            {
                                tracing::info!(
                                    token_count = tokens.len(),
                                    first_tokens = ?&tokens[..std::cmp::min(5, tokens.len())],
                                    "[SIDECAR-SKIP-TOKENIZE] Found nvext.token_data — using pre-computed tokens, SKIPPING tokenization"
                                );
                                (tokens.clone(), true)
                            } else if has_backend_instance_id {
                                tracing::warn!(
                                    "backend_instance_id provided but no token_data; tokenizing prompt"
                                );
                                let encoding = self.encode_with_timing(prompt, tracker)?;
                                (encoding.token_ids().to_vec(), false)
                            } else {
                                let encoding = self.encode_with_timing(prompt, tracker)?;
                                (encoding.token_ids().to_vec(), false)
                            };

                            if request.has_annotation(ANNOTATION_TOKEN_IDS)
                                && !skip_token_annotation
                            {
                                annotations.insert(
                                    ANNOTATION_TOKEN_IDS.to_string(),
                                    serde_json::to_string(&tokens_vec)?,
                                );
                            }

                            token_count = Some(tokens_vec.len());
                            tokens_out = tokens_vec;
                        }
                        TextInput::Batch(texts) => {
                            if texts.len() == 1 {
                                let encoding = self.encode_with_timing(&texts[0], tracker)?;
                                let tokens = encoding.token_ids().to_vec();
                                token_count = Some(tokens.len());
                                tokens_out = tokens;
                            } else {
                                bail!(
                                    "Batch text input not supported for more than one text in requests (got {})",
                                    texts.len()
                                );
                            }
                        }
                    }
                }
            }
        }

        // Validate prompt token count against model's context length
        if let Some(count) = token_count {
            Self::validate_token_count(count, self.context_length)?;
        }

        Ok((tokens_out, annotations))
    }

    /// Validate that the prompt token count does not consume the model's entire context length.
    /// Returns an error if the prompt leaves no room for output tokens.
    fn validate_token_count(token_count: usize, context_length: u32) -> Result<()> {
        let max_len = context_length as usize;
        // max_len == 0 means context_length was not configured (model_card.rs defaults
        // to 0 when max_position_embeddings is absent), so skip validation.
        // Use >= because context_length is the total budget (input + output): if the
        // prompt alone fills it, there is zero room for output tokens.
        if max_len > 0 && token_count >= max_len {
            return Err(DynamoError::builder()
                .error_type(ErrorType::InvalidArgument)
                .message(format!(
                    "This model's maximum context length is {} tokens. \
                     However, your messages resulted in {} tokens. \
                     Please reduce the length of the messages.",
                    max_len, token_count,
                ))
                .build()
                .into());
        }
        Ok(())
    }

    fn encode_with_timing(
        &self,
        prompt: &str,
        tracker: Option<&RequestTracker>,
    ) -> anyhow::Result<Encoding> {
        let encode_start = Instant::now();
        let prompt = if prompt.contains('\0') {
            tracing::debug!("Prompt contains null bytes; stripping to avoid tokenizer divergence");
            Cow::Owned(prompt.replace('\0', ""))
        } else {
            Cow::Borrowed(prompt)
        };
        let encoding = self.tokenizer.encode(prompt.as_ref())?;
        if let Some(t) = tracker {
            t.record_tokenize_latency(encode_start.elapsed());
        }
        Ok(encoding)
    }

    /// Preprocess an embedding request, handling both text and token ID inputs.
    ///
    /// For text inputs, tokenizes the text using the configured tokenizer.
    /// For token ID inputs, uses the provided token IDs directly and skips tokenization.
    ///
    /// Returns both the preprocessed request and a hashmap of annotations.
    pub async fn preprocess_embedding_request(
        &self,
        request: &NvCreateEmbeddingRequest,
    ) -> Result<(PreprocessedEmbeddingRequest, HashMap<String, String>)> {
        let _stage_guard = StageGuard::new(STAGE_PREPROCESS, "");
        let mut annotations = HashMap::new();
        let mut builder = PreprocessedEmbeddingRequest::builder();

        let all_token_ids = match &request.inner.input {
            dynamo_protocols::types::EmbeddingInput::String(s) => {
                let encoding = self.tokenizer.encode(s)?;
                vec![encoding.token_ids().to_vec()]
            }
            dynamo_protocols::types::EmbeddingInput::StringArray(arr) => {
                let input_strs: Vec<String> = arr.to_vec();
                let encodings = tokio::task::spawn_blocking({
                    let tokenizer = self.tokenizer.clone();
                    let strs = input_strs.clone();
                    move || {
                        tokenizer.encode_batch(&strs.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                    }
                })
                .await??;
                let token_arrays: Vec<Vec<u32>> = encodings
                    .into_iter()
                    .map(|encoding| encoding.token_ids().to_vec())
                    .collect();
                token_arrays
            }
            dynamo_protocols::types::EmbeddingInput::IntegerArray(token_ids) => {
                vec![token_ids.clone()]
            }
            dynamo_protocols::types::EmbeddingInput::ArrayOfIntegerArray(token_arrays) => {
                token_arrays.clone()
            }
        };

        // Handle annotations
        if request.has_annotation(ANNOTATION_TOKEN_IDS) {
            annotations.insert(
                ANNOTATION_TOKEN_IDS.to_string(),
                serde_json::to_string(&all_token_ids)?,
            );
        }

        builder.token_ids(all_token_ids);
        builder.model(request.inner.model.clone());
        builder.encoding_format(request.inner.encoding_format.as_ref().map(|f| match f {
            EncodingFormat::Float => "float".to_string(),
            EncodingFormat::Base64 => "base64".to_string(),
        }));
        builder.dimensions(request.inner.dimensions);

        builder.annotations(request.annotations().unwrap_or_default());
        builder.mdc_sum(Some(self.mdcsum.clone()));

        Ok((builder.build()?, annotations))
    }

    pub fn postprocessor_parsing_stream<S>(
        &self,
        stream: S,
        request: &NvCreateChatCompletionRequest,
        prompt_injected_reasoning: bool,
        uses_tool_call_structural_tag: bool,
    ) -> anyhow::Result<
        impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    >
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        // Kimi K2.5 tool-continuation turns produce the final user-facing
        // answer directly from the tool result. If the prompt happened to end
        // with `<think>`, starting the force-reasoning parser in reasoning mode
        // mislabels that answer as reasoning_content. DeepSeek V4 is the
        // opposite: its formatter can seed `<think>` for post-tool turns and
        // the model may emit only the closing `</think>`, so preserving the
        // injected-reasoning signal is required to avoid leaking the close tag.
        let last_is_tool = matches!(
            request.inner.messages.last(),
            Some(ChatCompletionRequestMessage::Tool(_))
        );
        let suppress_reasoning_after_tool = last_is_tool
            && matches!(
                self.runtime_config.reasoning_parser.as_deref(),
                Some("kimi_k25")
            );

        // Under guided-decoding (tool_choice=required/named), force-reasoning
        // parsers must skip because the backend emits bare JSON and the jail
        // needs to see that JSON as content. Some non-force parsers can also
        // hit that shape after prompt-injected reasoning, but they need a
        // stream-content check so `reasoning</think>JSON` still gets parsed.
        let is_guided_tool_choice = matches!(
            request.inner.tool_choice,
            Some(ChatCompletionToolChoiceOption::Required)
                | Some(ChatCompletionToolChoiceOption::Named(_))
        );
        let skip_reasoning_for_guided_json = is_guided_tool_choice
            && Self::is_force_reasoning_parser(self.runtime_config.reasoning_parser.as_deref());
        let bypass_reasoning_for_bare_guided_json = is_guided_tool_choice
            && prompt_injected_reasoning
            && !uses_tool_call_structural_tag
            && Self::skips_guided_json_when_prompt_injected(
                self.runtime_config.reasoning_parser.as_deref(),
            );

        let reasoning_disabled_by_request = Self::is_reasoning_disabled_by_request(
            self.runtime_config.reasoning_parser.as_deref(),
            request.chat_template_args.as_ref(),
        );

        // Try to parse reasoning content only if parser is configured.
        let should_parse_reasoning = self.runtime_config.reasoning_parser.is_some()
            && !reasoning_disabled_by_request
            && !suppress_reasoning_after_tool
            && !skip_reasoning_for_guided_json;
        let should_strip_disabled_reasoning_start = reasoning_disabled_by_request
            && Self::is_nemotron_force_reasoning(self.runtime_config.reasoning_parser.as_deref())
            && !suppress_reasoning_after_tool
            && !skip_reasoning_for_guided_json;

        // Reasoning Content Parsing Transformation Step
        // Current Solution:
        // This step operates on Deltas created by the transform_postprocessor_stream function
        // Only access to text and not token_ids - so can not support parsing based on token_ids for now
        // Future Solution:
        // To address the limitation if needed in future: move this step before transform_postprocessor_stream and add new field of reasoning_content to the backend output
        // Use backend_output.reasoning_content field to fill out the deltas.
        let stream: Pin<Box<dyn Stream<Item = _> + Send>> = if should_parse_reasoning {
            Box::pin(Self::parse_reasoning_content_from_stream_inner(
                stream,
                self.runtime_config.reasoning_parser.clone().unwrap(), // Safety: We already checked that parser is some, so gtg
                prompt_injected_reasoning,
                bypass_reasoning_for_bare_guided_json,
            ))
        } else if should_strip_disabled_reasoning_start {
            Box::pin(Self::strip_leading_reasoning_start_from_stream(
                stream, "<think>",
            ))
        } else {
            Box::pin(stream)
        };

        // Check if tools are present and if we should apply jail
        let has_tools = request
            .inner
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty());

        // Determine if we should apply jail (do this before moving request)
        let should_jail = Self::should_apply_tool_jail(
            self.tool_call_parser.as_ref(),
            request.inner.tool_choice.as_ref(),
            has_tools,
        )?;

        // Convert OpenAI tools to parser ToolDefinition format before applying jail
        let tool_definitions = request.inner.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| dynamo_parsers::tool_calling::ToolDefinition {
                    name: tool.function.name.clone(),
                    parameters: tool.function.parameters.clone(),
                    strict: tool.function.strict,
                })
                .collect()
        });

        // Apply jail conditionally
        let transformed_stream: Pin<Box<dyn Stream<Item = _> + Send>> = if should_jail {
            Box::pin(Self::apply_tool_calling_jail(
                self.tool_call_parser.clone(),
                request.inner.tool_choice.clone(),
                tool_definitions,
                uses_tool_call_structural_tag,
                stream,
            ))
        } else {
            Box::pin(stream)
        };

        Ok(transformed_stream)
    }

    pub fn transform_postprocessor_stream<S, Resp>(
        stream: S,
        generator: Box<dyn DeltaGeneratorExt<Resp>>,
        context: Arc<dyn AsyncEngineContext>,
        trace_tokens_enabled: bool,
        trace_finish_reason_metadata: Option<crate::agents::trace::SharedFinishReasonMetadata>,
    ) -> impl Stream<Item = Annotated<Resp>> + Send
    where
        S: Stream<Item = Annotated<BackendOutput>> + Send + 'static,
        Resp: Send + Sync + 'static + std::fmt::Debug,
    {
        struct State<Resp>
        where
            Resp: Send + Sync + 'static + std::fmt::Debug,
        {
            response_stream: Pin<Box<dyn Stream<Item = Annotated<BackendOutput>> + Send>>,
            response_generator: Box<dyn DeltaGeneratorExt<Resp>>,
            context: Arc<dyn AsyncEngineContext>,
            cancelled: bool,
            cumulative_output_tokens: usize,
            finish_reason_sent: bool,
            usage_chunk_sent: bool,
            finished: bool,
            trace_tokens_enabled: bool,
            trace_finish_reason_metadata: Option<crate::agents::trace::SharedFinishReasonMetadata>,
        }

        let state = State {
            response_stream: Box::pin(stream),
            response_generator: generator,
            context: context.clone(),
            cancelled: false,
            cumulative_output_tokens: 0,
            finish_reason_sent: false,
            usage_chunk_sent: false,
            finished: false,
            trace_tokens_enabled,
            trace_finish_reason_metadata,
        };

        // transform the common response stream into a chat response stream

        stream::unfold(state, |mut inner| {
            async move {
                // If already finished, return None immediately
                if inner.finished {
                    return None;
                }

                if let Some(mut response) = inner.response_stream.next().await {
                    // Split topology: overlay a standalone router's forwarded timing onto
                    // this request's tracker so the frontend's timing surfaces populate.
                    if let Some(timing) = take_router_timing(&mut response.data)
                        && let Some(tracker) = inner.response_generator.tracker()
                    {
                        tracker.set_external_timing(timing);
                    }

                    if inner.cancelled {
                        tracing::debug!(
                            request_id = inner.context.id(),
                            "Cancellation issued last message; closing stream"
                        );
                        // inner.finished = true; // Mark as finished
                        return None;
                    }

                    tracing::trace!(
                        request_id = inner.context.id(),
                        "Processing common response: {:?}",
                        response
                    );

                    // Check if this response has a finish_reason
                    let has_finish_reason = response
                        .data
                        .as_ref()
                        .map(|d| d.finish_reason.is_some())
                        .unwrap_or(false);

                    let (chunk_tokens, isl) = if let Some(ref backend_output) = response.data {
                        let chunk_tokens = backend_output.token_ids.len();
                        inner.cumulative_output_tokens += chunk_tokens;

                        let isl = inner.response_generator.get_isl().map(|isl| isl as usize);

                        crate::agents::trace::record_backend_finish_reason_metadata(
                            inner.trace_finish_reason_metadata.as_ref(),
                            backend_output.index,
                            backend_output.finish_reason.as_ref(),
                            backend_output.stop_reason.as_ref(),
                        );

                        (chunk_tokens, isl)
                    } else {
                        (0, None)
                    };

                    let current_osl = inner.cumulative_output_tokens;

                    let mut response = response.map_data(|data| {
                        inner
                            .response_generator
                            .choice_from_postprocessor(data)
                            .inspect_err(|e| {
                                tracing::error!(
                                    request_id = inner.context.id(),
                                    "Error processing common response: {:?}",
                                    e
                                );
                                inner.cancelled = true;
                                inner.context.stop_generating();
                            })
                            .map_err(|e| e.to_string())
                    });

                    // Create LLM metrics annotation with prefill/decode worker info from tracker.
                    // Worker types are stored at routing time to avoid expensive MDC lookup.
                    let tracker = inner.response_generator.tracker();
                    let prefill_worker_id = tracker.as_ref().and_then(|t| t.prefill_worker_id());
                    let prefill_dp_rank = tracker.as_ref().and_then(|t| t.prefill_dp_rank());
                    let prefill_worker_type = tracker
                        .as_ref()
                        .and_then(|t| t.prefill_worker_type())
                        .map(String::from);
                    let decode_worker_id = tracker.as_ref().and_then(|t| t.decode_worker_id());
                    let decode_dp_rank = tracker.as_ref().and_then(|t| t.decode_dp_rank());
                    let decode_worker_type = tracker
                        .as_ref()
                        .and_then(|t| t.decode_worker_type())
                        .map(String::from);
                    let llm_metrics = LLMMetricAnnotation {
                        input_tokens: isl.unwrap_or(0),
                        output_tokens: current_osl,
                        chunk_tokens,
                        cached_tokens: None,
                        prefill_worker_id,
                        prefill_dp_rank,
                        prefill_worker_type,
                        decode_worker_id,
                        decode_dp_rank,
                        decode_worker_type,
                        tokenize_latency: tracker.as_ref().and_then(|t| t.tokenize_latency()),
                        detokenize_total_latency: tracker.as_ref().and_then(|t| t.detokenize_total_latency()),
                        detokenize_count: tracker.as_ref().map(|t| t.detokenize_count()),
                    };
                    if inner.trace_tokens_enabled {
                        crate::agents::trace::record_llm_metric_tokens(
                            tracker.as_deref(),
                            isl,
                            current_osl,
                            None,
                        );
                    }

                    // Flush per-request detokenize accumulators to global Prometheus counters
                    // (once per request instead of per-token).
                    if let Some(t) = tracker.as_ref() {
                        if let Some(total) = t.detokenize_total_latency() {
                            DETOKENIZE_TOTAL_US.inc_by(total.as_micros() as f64);
                        }
                        DETOKENIZE_TOKEN_COUNT.inc_by(t.detokenize_count() as f64);
                    }

                    if let Ok(metrics_annotated) = llm_metrics.to_annotation::<()>() {
                        // Only set event if not already set to avoid overriding existing events (like errors)
                        if response.event.is_none() {
                            response.event = metrics_annotated.event;
                            response.comment = metrics_annotated.comment;
                        }
                    }

                    // Mark if we've seen a finish_reason
                    if has_finish_reason {
                        inner.finish_reason_sent = true;
                    }

                    tracing::trace!(
                        request_id = inner.context.id(),
                        "OpenAI NvCreateChatCompletionStreamResponse: {:?}",
                        response
                    );

                    Some((response, inner))
                } else {
                    // Stream has ended - must set finished to true to prevent unfold from polling
                    // again. The stream is exhausted and will panic if polled after None.
                    inner.finished = true;

                    if inner.finish_reason_sent && !inner.usage_chunk_sent {
                        inner.usage_chunk_sent = true;

                        let usage_chunk = inner.response_generator.create_usage_chunk();
                        let usage = inner.response_generator.get_usage();
                        let tracker = inner.response_generator.tracker();
                        let cached_tokens = usage
                            .prompt_tokens_details
                            .as_ref()
                            .and_then(|d| d.cached_tokens.map(|c| c as usize));
                        let prefill_worker_id =
                            tracker.as_ref().and_then(|t| t.prefill_worker_id());
                        let prefill_dp_rank = tracker.as_ref().and_then(|t| t.prefill_dp_rank());
                        let prefill_worker_type = tracker
                            .as_ref()
                            .and_then(|t| t.prefill_worker_type())
                            .map(String::from);
                        let decode_worker_id = tracker.as_ref().and_then(|t| t.decode_worker_id());
                        let decode_dp_rank = tracker.as_ref().and_then(|t| t.decode_dp_rank());
                        let decode_worker_type = tracker
                            .as_ref()
                            .and_then(|t| t.decode_worker_type())
                            .map(String::from);
                        let llm_metrics = LLMMetricAnnotation {
                            input_tokens: usage.prompt_tokens as usize,
                            output_tokens: usage.completion_tokens as usize,
                            chunk_tokens: 0,
                            cached_tokens,
                            prefill_worker_id,
                            prefill_dp_rank,
                            prefill_worker_type,
                            decode_worker_id,
                            decode_dp_rank,
                            decode_worker_type,
                            tokenize_latency: tracker.as_ref().and_then(|t| t.tokenize_latency()),
                            detokenize_total_latency: tracker
                                .as_ref()
                                .and_then(|t| t.detokenize_total_latency()),
                            detokenize_count: tracker.as_ref().map(|t| t.detokenize_count()),
                        };
                        if inner.trace_tokens_enabled {
                            crate::agents::trace::record_llm_metric_tokens(
                                tracker.as_deref(),
                                Some(usage.prompt_tokens as usize),
                                usage.completion_tokens as usize,
                                cached_tokens,
                            );
                        }

                        // Flush per-request detokenize accumulators to global Prometheus counters
                        // (once per request instead of per-token).
                        if let Some(t) = tracker.as_ref() {
                            if let Some(total) = t.detokenize_total_latency() {
                                DETOKENIZE_TOTAL_US.inc_by(total.as_micros() as f64);
                            }
                            DETOKENIZE_TOKEN_COUNT.inc_by(t.detokenize_count() as f64);
                        }

                        // Create annotation string
                        let annotation = llm_metrics.to_annotation::<()>().unwrap_or_else(|e| {
                            tracing::warn!("Failed to serialize metrics: {}", e);
                            Annotated::<()>::from_data(())
                        });

                        // Send the usage chunk if needed
                        let data = if inner.response_generator.is_usage_enabled() {
                            Some(usage_chunk)
                        } else {
                            None
                        };

                        let annotated_usage = Annotated::<Resp> {
                            id: None,
                            data,
                            event: Some(ANNOTATION_LLM_METRICS.to_string()),
                            comment: annotation.comment,
                            error: None,
                        };

                        tracing::trace!(
                            request_id = inner.context.id(),
                            "Sending final usage chunk for OpenAI compliance, annotated_usage: {:?}",
                            annotated_usage
                        );

                        Some((annotated_usage, inner))
                    } else {
                        // stream closed
                        None
                    }
                }
            }
        })
        .fuse()
    }

    /// Transform engine embedding output stream to OpenAI embedding response stream
    pub fn transform_embedding_postprocessor_stream<S>(
        stream: S,
        original_request: NvCreateEmbeddingRequest,
    ) -> impl Stream<Item = Annotated<NvCreateEmbeddingResponse>> + Send
    where
        S: Stream<Item = Annotated<EmbeddingsEngineOutput>> + Send + 'static,
    {
        // Honor the OpenAI `encoding_format` field. The default is `Float`;
        // `Base64` encodes the raw little-endian f32 bytes of each
        // per-input vector. The engine always returns floats, so the
        // base64 path runs at the postprocessor seam where we still have
        // the original request shape in scope.
        let encode_base64 = matches!(
            original_request.inner.encoding_format,
            Some(dynamo_protocols::types::EncodingFormat::Base64)
        );
        stream.map(move |output| {
            output.map_data(|engine_output| {
                // Convert engine output to OpenAI response format
                let embeddings: Vec<dynamo_protocols::types::Embedding> = engine_output
                    .embeddings
                    .into_iter()
                    .enumerate()
                    .map(|(index, embedding)| {
                        let floats: Vec<f32> = embedding.into_iter().map(|f| f as f32).collect();
                        let value = if encode_base64 {
                            dynamo_protocols::types::EmbeddingVector::Base64(
                                encode_floats_to_base64(&floats),
                            )
                        } else {
                            dynamo_protocols::types::EmbeddingVector::Float(floats)
                        };
                        dynamo_protocols::types::Embedding {
                            index: index as u32,
                            object: "embedding".to_string(),
                            embedding: value,
                        }
                    })
                    .collect();

                let response = NvCreateEmbeddingResponse {
                    inner: dynamo_protocols::types::CreateEmbeddingResponse {
                        object: "list".to_string(),
                        model: original_request.inner.model.clone(),
                        data: embeddings,
                        usage: dynamo_protocols::types::EmbeddingUsage {
                            prompt_tokens: engine_output.prompt_tokens,
                            total_tokens: engine_output.total_tokens,
                        },
                    },
                };

                Ok(response)
            })
        })
    }

    /// Determine if we should apply the tool calling jail based on configuration
    /// Returns Ok(true) if jail should be applied, Ok(false) if not, or Err if invalid config
    pub fn should_apply_tool_jail(
        tool_call_parser: Option<&String>,
        tool_choice: Option<&ChatCompletionToolChoiceOption>,
        has_tools: bool,
    ) -> std::result::Result<bool, Error> {
        match (tool_call_parser, tool_choice, has_tools) {
            // tool_choice=required/named work without parser (use Immediate jail mode)
            (None, Some(ChatCompletionToolChoiceOption::Required), true) => Ok(true),
            (None, Some(ChatCompletionToolChoiceOption::Named(_)), true) => Ok(true),

            // tool_choice=auto requires a parser
            (None, Some(ChatCompletionToolChoiceOption::Auto), true) => {
                tracing::warn!(
                    "Tool choice 'auto' specified but no tool parser configured; proceeding without jailing"
                );
                Ok(false)
            }

            // Parser exists and tools might be called
            (Some(_), Some(ChatCompletionToolChoiceOption::None), _) => {
                Ok(false) // Explicitly disabled
            }
            (Some(_), Some(_), true) => Ok(true), // Any other tool_choice with tools
            (Some(_), None, true) => Ok(true),    // Default behavior when tools present

            // No tools or no parser
            _ => Ok(false),
        }
    }

    /// Apply tool calling jail to the stream if needed
    pub fn apply_tool_calling_jail<S>(
        tool_call_parser: Option<String>,
        tool_choice: Option<dynamo_protocols::types::ChatCompletionToolChoiceOption>,
        tool_definitions: Option<Vec<dynamo_parsers::tool_calling::ToolDefinition>>,
        uses_tool_call_structural_tag: bool,
        stream: S,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        use dynamo_protocols::types::ChatCompletionToolChoiceOption;

        let mut builder = JailedStream::builder();

        // Set tool definitions if provided
        if let Some(tool_definitions) = tool_definitions
            && !tool_definitions.is_empty()
        {
            builder = builder.tool_definitions(tool_definitions);
        }

        // When structural_tag is active, the model output is already constrained by
        // guided decoding into a model-specific format. Always use the marker-based
        // parser to extract tool calls from that format.
        if uses_tool_call_structural_tag {
            if let Some(parser) = tool_call_parser {
                builder = builder.tool_call_parser(parser);
            }
        } else {
            // Configure jail based on tool_choice
            //
            // For tool_choice=required or named we mirror SGLang / vLLM: assume the
            // backend applied guided decoding and emit a bare JSON shape, so parse
            // via the JSON array parser (base_json_parser) rather than the model's
            // native-format parser.  If a parser is also configured we still carry
            // it so the Immediate branch can fall back to marker-based parsing for
            // backends that do not honor guided decoding (e.g. XML-native models
            // like qwen3_coder — see regression test_tool_choice_required_with_
            // qwen3_coder_parser).
            match tool_choice {
                Some(ChatCompletionToolChoiceOption::Named(named)) => {
                    builder = builder
                        .tool_choice_named(named.function.name.clone())
                        .named_tool_filter(named.function.name.clone());
                    if let Some(parser) = tool_call_parser {
                        builder = builder.tool_call_parser(parser);
                    }
                }
                Some(ChatCompletionToolChoiceOption::Required) => {
                    builder = builder.tool_choice_required();
                    if let Some(parser) = tool_call_parser {
                        builder = builder.tool_call_parser(parser);
                    }
                }
                Some(ChatCompletionToolChoiceOption::Auto)
                | Some(ChatCompletionToolChoiceOption::None)
                | None => {
                    // Traditional marker-based jail for auto/none/unspecified
                    if let Some(parser) = tool_call_parser {
                        builder = builder.tool_call_parser(parser);
                    }
                }
            }
        }

        let jail = builder.build();
        jail.apply_with_finish_reason(stream)
    }

    /// Whether the selected tool-call or reasoning parser depends on the
    /// engine emitting special tokens (e.g. Gemma 4's `<|tool_call>` /
    /// `<|channel>`). Mirrors upstream vLLM's per-parser `adjust_request`
    /// hooks. Used to flip the request default for `skip_special_tokens`
    /// from `true` to `false` so the parsers actually see the markers
    /// they're matching on.
    fn parser_requires_special_tokens(
        tool_call_parser: Option<&str>,
        reasoning_parser: Option<&str>,
    ) -> bool {
        // Parsers in this allow-list match against special tokens that the
        // tokenizer would otherwise strip when `skip_special_tokens=true`
        // (the OpenAI-API default). Without the tokens preserved through
        // decode the parsers silently produce empty reasoning_content /
        // tool_calls.
        //
        // - gemma4: `<|think|>` prompt trigger plus parser-visible
        //   `<|channel>` / `<channel|>` reasoning markers and tool-call markers.
        // - harmony / gpt_oss: `<|channel|>analysis<|message|>...<|end|>`.
        // - kimi_k2: `<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>`.
        // - kimi_k25: `</think>` (special token id 163607).
        matches!(
            tool_call_parser,
            Some("gemma4") | Some("gemma-4") | Some("harmony") | Some("kimi_k2")
        ) || matches!(
            reasoning_parser,
            Some("gemma4") | Some("gemma-4") | Some("gpt_oss") | Some("kimi_k25")
        )
    }

    /// Whether the resolved `skip_special_tokens` will strip the special-token
    /// markers an active parser depends on — guaranteeing the parser silently
    /// emits empty tool_calls / reasoning_content and the markup leaks into
    /// `content`. Only true when the caller explicitly forced
    /// `skip_special_tokens=true` while a special-token-dependent parser is
    /// active; otherwise the default is flipped to false before this check.
    fn special_tokens_will_be_stripped(
        skip_special_tokens: Option<bool>,
        tool_call_parser: Option<&str>,
        reasoning_parser: Option<&str>,
    ) -> bool {
        skip_special_tokens == Some(true)
            && Self::parser_requires_special_tokens(tool_call_parser, reasoning_parser)
    }

    fn is_nemotron_force_reasoning(reasoning_parser: Option<&str>) -> bool {
        matches!(
            reasoning_parser,
            Some("nemotron_nano" | "nemotron3" | "nemotron_v3")
        )
    }

    /// Parsers that begin streaming in reasoning mode (force_reasoning=true).
    /// These swallow any leading text without an open `<think>` tag as
    /// reasoning_content, so they cannot run on guided-decoding output where
    /// the model emits bare JSON from token 0.
    fn is_force_reasoning_parser(reasoning_parser: Option<&str>) -> bool {
        matches!(
            reasoning_parser,
            Some(
                "deepseek_r1"
                    | "step3"
                    | "kimi_k25"
                    | "mistral"
                    | "minimax_append_think"
                    | "nemotron_nano"
                    | "nemotron3"
                    | "nemotron_v3"
            )
        )
    }

    fn skips_guided_json_when_prompt_injected(reasoning_parser: Option<&str>) -> bool {
        matches!(
            reasoning_parser,
            Some("deepseek_v4" | "deepseek-v4" | "deepseekv4" | "glm45")
        )
    }

    /// Check if reasoning parsing should be disabled based on per-request parameters.
    /// For kimi_k25: disabled when chat_template_args contains "thinking": false.
    /// For Nemotron force-reasoning aliases: disabled when chat_template_args
    ///   contains "enable_thinking": false or "force_nonempty_content": true.
    /// For deepseek_r1 / deepseek_v4: disabled when chat_template_args contains
    ///   "thinking": false or "thinking_mode": "chat" — matches the V4 formatter's
    ///   `resolve_thinking_mode` convention, so the parser and the prompt stay in sync.
    /// For gemma4: disabled when chat_template_args contains "enable_thinking": false.
    ///   Gemma 4's chat template injects `<|think|>` only when `enable_thinking is
    ///   defined and enable_thinking` (truthy), so when callers explicitly set the
    ///   flag false the model emits no `<|channel>` markers and the parser would
    ///   only ever fall through.
    fn is_reasoning_disabled_by_request(
        reasoning_parser: Option<&str>,
        chat_template_args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> bool {
        match reasoning_parser {
            Some("kimi_k25") => {
                if let Some(args) = chat_template_args
                    && let Some(thinking) = args.get("thinking")
                {
                    return thinking == &serde_json::Value::Bool(false);
                }
                false
            }
            parser if Self::is_nemotron_force_reasoning(parser) => {
                if let Some(args) = chat_template_args {
                    if let Some(enable_thinking) = args.get("enable_thinking")
                        && enable_thinking == &serde_json::Value::Bool(false)
                    {
                        return true;
                    }
                    if let Some(force_nonempty) = args.get("force_nonempty_content")
                        && force_nonempty == &serde_json::Value::Bool(true)
                    {
                        return true;
                    }
                }
                false
            }
            Some("deepseek_r1") | Some("deepseek_v4") | Some("deepseek-v4")
            | Some("deepseekv4") => {
                if let Some(enabled) = dynamo_renderer::thinking_bool_from_args(chat_template_args)
                {
                    return !enabled;
                }
                if let Some(args) = chat_template_args
                    && let Some(mode) = args.get("thinking_mode").and_then(|v| v.as_str())
                {
                    return mode == "chat";
                }
                false
            }
            Some("gemma4") | Some("gemma-4") => {
                if let Some(enabled) = dynamo_renderer::thinking_bool_from_args(chat_template_args)
                {
                    return !enabled;
                }
                false
            }
            _ => false,
        }
    }

    // Motivation: Each transformation on the stream should be a separate step to allow for more flexibility
    // Earlier reasoning parser logic was nested under delta generation logic in choice_from_postprocessor
    // Since we have tool calling parsing as separate step, it makes sense to have reasoning parser as separate step as well
    /// Apply reasoning parsing to the output stream, splitting content into
    /// `reasoning_content` and normal `content` based on think tags.
    ///
    /// When `prompt_injected_reasoning` is `true`, the parser starts in reasoning
    /// mode immediately — use this when the chat template already appended the
    /// reasoning start token (e.g., `<think>`) to the prompt, so the model's
    /// completion begins with thinking content without an explicit start tag.
    pub fn parse_reasoning_content_from_stream<S>(
        stream: S,
        parser_name: String,
        prompt_injected_reasoning: bool,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        Self::parse_reasoning_content_from_stream_inner(
            stream,
            parser_name,
            prompt_injected_reasoning,
            false,
        )
    }

    fn parse_reasoning_content_from_stream_inner<S>(
        stream: S,
        parser_name: String,
        prompt_injected_reasoning: bool,
        bypass_bare_guided_json: bool,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        // Initialize reasoning parser from parser_name
        let mut reasoning_parser = Box::new(ReasoningParserType::get_reasoning_parser_from_name(
            parser_name.as_ref(),
        )) as Box<dyn ReasoningParser>;

        if prompt_injected_reasoning {
            reasoning_parser.set_in_reasoning(true);
        }

        let state = ReasoningState {
            stream: Box::pin(stream),
            reasoning_parser: Some(reasoning_parser),
            bypass_bare_guided_json,
            guided_json_bypass_decision: None,
        };

        stream::unfold(state, |mut state| async move {
            if let Some(response) = state.stream.next().await {
                let should_bypass_reasoning = if state.bypass_bare_guided_json {
                    match state.guided_json_bypass_decision {
                        Some(decision) => decision,
                        None => {
                            let decision = response.data.as_ref().and_then(|data| {
                                data.inner.choices.iter().find_map(|choice| {
                                    if let Some(ChatCompletionMessageContent::Text(text)) =
                                        choice.delta.content.as_ref()
                                    {
                                        let text = text.trim_start();
                                        if text.is_empty() {
                                            return None;
                                        }
                                        return Some(matches!(text.as_bytes()[0], b'[' | b'{'));
                                    }
                                    None
                                })
                            });
                            if let Some(decision) = decision {
                                state.guided_json_bypass_decision = Some(decision);
                            }
                            decision.unwrap_or(false)
                        }
                    }
                } else {
                    false
                };

                // Process the response through reasoning parser if available
                let processed_response = if should_bypass_reasoning {
                    response
                } else if let Some(ref mut parser) = state.reasoning_parser {
                    response.map_data(|mut data| {
                        // Process all choices, not just the first one
                        for choice in data.inner.choices.iter_mut() {
                            // Reasoning parsing only applies to text content
                            if let Some(ChatCompletionMessageContent::Text(text)) =
                                choice.delta.content.as_ref()
                            {
                                let parser_result =
                                    parser.parse_reasoning_streaming_incremental(text, &[]);

                                // Update this specific choice with parsed content
                                choice.delta.content = parser_result
                                    .get_some_normal_text()
                                    .map(ChatCompletionMessageContent::Text);
                                choice.delta.reasoning_content = parser_result.get_some_reasoning();
                            }
                            // For multimodal content, pass through unchanged
                        }
                        Ok(data)
                    })
                } else {
                    // No reasoning parser configured, pass through unchanged
                    response
                };

                Some((processed_response, state))
            } else {
                None
            }
        })
        .fuse()
    }

    // Motivation: when Nemotron reasoning is disabled by request flags, the
    // backend may still emit a leading <think>. Buffer the initial stream
    // bytes so split chunks like "<thi" + "nk>answer" are stripped cleanly.
    fn strip_leading_reasoning_start_from_stream<S>(
        stream: S,
        think_start_token: &'static str,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        struct StripReasoningStartState {
            stream:
                Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>,
            think_start_token: &'static str,
            choices: HashMap<u32, StripChoiceState>,
            last_response: Option<Annotated<NvCreateChatCompletionStreamResponse>>,
            eof_flushed: bool,
        }

        #[derive(Default)]
        struct StripChoiceState {
            buffer: String,
            decided: bool,
        }

        fn take_undecided_buffer(choice_state: &mut StripChoiceState) -> Option<String> {
            if choice_state.decided || choice_state.buffer.is_empty() {
                return None;
            }

            choice_state.decided = true;
            Some(std::mem::take(&mut choice_state.buffer))
        }

        fn drain_undecided_buffers(
            choices: &mut HashMap<u32, StripChoiceState>,
        ) -> HashMap<u32, String> {
            choices
                .iter_mut()
                .filter_map(|(index, choice_state)| {
                    take_undecided_buffer(choice_state).map(|buffer| (*index, buffer))
                })
                .collect()
        }

        let state = StripReasoningStartState {
            stream: Box::pin(stream),
            think_start_token,
            choices: HashMap::new(),
            last_response: None,
            eof_flushed: false,
        };

        stream::unfold(state, |mut state| async move {
            if let Some(mut response) = state.stream.next().await {
                let Some(mut data) = response.data.take() else {
                    return Some((response, state));
                };

                for choice in data.inner.choices.iter_mut() {
                    let choice_state = state.choices.entry(choice.index).or_default();
                    let text = match choice.delta.content.take() {
                        Some(ChatCompletionMessageContent::Text(text)) => text,
                        other => {
                            if let Some(buffer) = take_undecided_buffer(choice_state) {
                                choice.delta.content =
                                    Some(ChatCompletionMessageContent::Text(buffer));
                            } else {
                                choice.delta.content = other;
                            }
                            continue;
                        }
                    };

                    let output = if choice_state.decided {
                        text
                    } else {
                        choice_state.buffer.push_str(&text);
                        if state.think_start_token.starts_with(&choice_state.buffer)
                            && choice_state.buffer.len() < state.think_start_token.len()
                        {
                            choice.delta.content = None;
                            continue;
                        }

                        choice_state.decided = true;
                        if choice_state.buffer.starts_with(state.think_start_token) {
                            choice_state.buffer[state.think_start_token.len()..].to_string()
                        } else {
                            choice_state.buffer.clone()
                        }
                    };

                    choice_state.buffer.clear();
                    choice.delta.content = if output.is_empty() {
                        None
                    } else {
                        Some(ChatCompletionMessageContent::Text(output))
                    };
                }

                response.data = Some(data);
                state.last_response = Some(response.clone());

                Some((response, state))
            } else if state.eof_flushed {
                None
            } else {
                state.eof_flushed = true;
                let mut flushed = drain_undecided_buffers(&mut state.choices);
                if flushed.is_empty() {
                    None
                } else {
                    let mut response = state.last_response.clone()?;
                    let data = response.data.as_mut()?;
                    data.inner.usage = None;
                    data.inner.choices.retain_mut(|choice| {
                        if let Some(buffer) = flushed.remove(&choice.index) {
                            choice.delta.role = None;
                            choice.delta.content = Some(ChatCompletionMessageContent::Text(buffer));
                            choice.delta.tool_calls = None;
                            choice.delta.function_call = None;
                            choice.delta.refusal = None;
                            choice.delta.reasoning_content = None;
                            choice.finish_reason = None;
                            choice.logprobs = None;
                            true
                        } else {
                            false
                        }
                    });

                    if data.inner.choices.is_empty() {
                        None
                    } else {
                        Some((response, state))
                    }
                }
            }
        })
        .fuse()
    }
}

// for pals, we do not want to add the generation prompt to the formatted prompt
// we also need to know if the template support this add_generation_prompt bool
// any prompt template that does not support this should return an error
// oob - we should update any prompt template that does not support this to support it

#[async_trait]
impl
    Operator<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
    > for OpenAIPreprocessor
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
        next: Arc<
            dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        // unpack the request
        let (mut request, context) = request.into_parts();

        // Preserve original inbound streaming flag before any internal overrides
        let request_id = context.id().to_string();
        let original_stream_flag = request.inner.stream.unwrap_or(false);

        // Build audit handle (None if no DYN_AUDIT_SINKS)
        let mut audit_handle = crate::audit::handle::create_handle(&request, &request_id);

        if let Some(ref mut h) = audit_handle {
            h.set_request(std::sync::Arc::new(request.clone()));
        }

        // For non-streaming requests (stream=false), enable usage by default
        // This ensures compliance with OpenAI API spec where non-streaming responses
        // always include usage statistics
        request.enable_usage_for_nonstreaming(original_stream_flag);

        // Set stream=true for internal processing (after audit capture)
        request.inner.stream = Some(true);

        // create a response generator
        let response_generator = request.response_generator(context.id().to_string());
        let tracker = Some(response_generator.tracker());
        let preprocess_options = PreprocessRequestOptions {
            preserve_omitted_max_tokens: context
                .get::<bool>(PRESERVE_OMITTED_MAX_TOKENS_CONTEXT_KEY)
                .ok()
                .is_some_and(|flag| *flag),
        };

        // convert the chat completion request to a common completion request
        let (mut common_request, annotations, prompt_injected_reasoning) = self
            .preprocess_request_with_options(&request, tracker.as_deref(), preprocess_options)
            .await?;

        let uses_tool_call_structural_tag = self.apply_tool_choice_guided_decoding(
            &request,
            &mut common_request,
            prompt_injected_reasoning,
        )?;

        tracing::trace!(request = ?common_request, prompt_injected_reasoning, "Pre-processed request");
        let trace_state = crate::request_trace::build_request_end_trace_state(
            &common_request,
            &tracker,
            &context,
            self.kv_cache_block_size,
        );
        let trace_tokens_enabled = trace_state.is_some();
        let trace_finish_reason_metadata =
            crate::request_trace::finish_reason_metadata_handle(&trace_state);

        // Attach the timing tracker to the request so downstream components can record metrics
        common_request.tracker = tracker;

        let mut response_generator = Box::new(response_generator);

        // Update ISL only for text prompts (embeddings get sequence length from tensor shape)
        if common_request.prompt_embeds.is_none() {
            let isl = common_request.token_ids.len() as u32;
            response_generator.update_isl(isl);
        }

        // repack the common completion request
        let common_request = context.map(|_| common_request);

        // create a stream of annotations this will be prepend to the response stream
        let annotations: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = annotations
            .into_iter()
            .flat_map(|(k, v)| Annotated::from_annotation(k, &v))
            .collect();
        let annotations_stream = stream::iter(annotations);

        // forward the common completion request to the next operator
        let response_stream = next.generate(common_request).await?;
        // Extract context once
        let context = response_stream.context();

        // transform the postprocessor stream (no boxing yet) - detokenize
        let stream = Self::transform_postprocessor_stream(
            response_stream,
            response_generator,
            context.clone(),
            trace_tokens_enabled,
            trace_finish_reason_metadata,
        );

        let transformed_stream = self.postprocessor_parsing_stream(
            stream,
            &request,
            prompt_injected_reasoning,
            uses_tool_call_structural_tag,
        )?;

        // Apply audit aggregation strategy.
        // The audit branch already returns Pin<Box<...>> from scan/fold_aggregate_with_future,
        // while the non-audit branch boxes the impl Stream from postprocessor_parsing_stream.
        let final_stream = if let Some(mut audit) = audit_handle {
            let (stream, agg_fut) = if audit.streaming() {
                // Streaming: apply scan (pass-through + parallel aggregation)
                crate::audit::stream::scan_aggregate_with_future(transformed_stream)
            } else {
                // Non-streaming: apply fold (collect all, then emit single chunk)
                crate::audit::stream::fold_aggregate_with_future(transformed_stream)
            };

            // Spawn audit task
            tokio::spawn(async move {
                let final_resp = agg_fut.await;
                audit.set_response(Arc::new(final_resp));
                audit.emit();
            });

            stream
        } else {
            Box::pin(transformed_stream)
        };

        // Step 5: Speculative next-turn prefill
        let final_stream = speculative_prefill::maybe_wrap_stream(
            final_stream,
            &request,
            &next,
            &self.formatter,
            &self.tokenizer,
        );

        let final_stream = crate::request_trace::wrap_chat_request_end_stream(
            final_stream,
            trace_state,
            request_id,
        );

        // prepend the annotations to the response stream
        let stream = annotations_stream.chain(final_stream);

        // return the response stream - single boxing at the end
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<NvCreateCompletionRequest>,
        ManyOut<Annotated<NvCreateCompletionResponse>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
    > for OpenAIPreprocessor
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateCompletionRequest>,
        next: Arc<
            dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        let _stage_guard = StageGuard::new(STAGE_PREPROCESS, "");

        // unpack the request
        let (mut request, context) = request.into_parts();
        let request_id = context.id().to_string();

        // Preserve original streaming flag
        let original_stream_flag = request.inner.stream.unwrap_or(false);

        // For non-streaming requests (stream=false), enable usage by default
        // This ensures compliance with OpenAI API spec where non-streaming responses
        // always include usage statistics
        request.enable_usage_for_nonstreaming(original_stream_flag);

        request.inner.stream = Some(true);

        // create a response generator
        let response_generator = request.response_generator(request_id.clone());
        let mut response_generator = Box::new(response_generator);
        let tracker = Some(response_generator.tracker());
        // convert the chat completion request to a common completion request
        let mut builder = self.builder(&request)?;

        // Check if embeddings are provided - skip tokenization path
        let annotations = if let Some(ref prompt_embeds) = request.inner.prompt_embeds {
            // Skip tokenization for embeddings
            builder.token_ids(vec![]); // Empty token IDs
            builder.prompt_embeds(Some(prompt_embeds.clone()));
            // No token annotations
            HashMap::new()
        } else {
            // Normal path: tokenize the prompt; embeddings don't need MM routing,
            // so install tokens on the builder right away.
            let (token_ids, ann) = self.gather_tokens(&request, None, tracker.as_deref())?;
            builder.token_ids(token_ids);
            ann
        };

        // Gather multimodal data (works with both embeddings and text prompts)
        // Returned MM entries are unused on the embeddings path; routing info is
        // not built here.
        let _ = self
            .gather_multi_modal_data(&request, &mut builder, None)
            .await?;

        let mut common_request = builder.build()?;

        let trace_state = crate::request_trace::build_request_end_trace_state(
            &common_request,
            &tracker,
            &context,
            self.kv_cache_block_size,
        );
        let trace_tokens_enabled = trace_state.is_some();
        let trace_finish_reason_metadata =
            crate::request_trace::finish_reason_metadata_handle(&trace_state);

        // Attach the timing tracker to the request so downstream components can record metrics
        common_request.tracker = tracker;

        // Update ISL only for text prompts (embeddings get sequence length from tensor shape)
        if common_request.prompt_embeds.is_none() {
            let isl = common_request.token_ids.len() as u32;
            response_generator.update_isl(isl);
        }

        // repack the common completion request
        let common_request = context.map(|_| common_request);

        // create a stream of annotations this will be prepend to the response stream
        let annotations: Vec<Annotated<NvCreateCompletionResponse>> = annotations
            .into_iter()
            .flat_map(|(k, v)| Annotated::from_annotation(k, &v))
            .collect();
        let annotations_stream = stream::iter(annotations);

        // End preprocess stage before handing off to downstream (route/dispatch).
        drop(_stage_guard);

        // forward the common completion request to the next operator
        let response_stream = next.generate(common_request).await?;

        // Extract context once
        let context = response_stream.context();

        // transform the postprocessor stream
        let stream = Self::transform_postprocessor_stream(
            response_stream,
            response_generator,
            context.clone(),
            trace_tokens_enabled,
            trace_finish_reason_metadata,
        );

        let stream = crate::request_trace::wrap_completion_request_end_stream(
            Box::pin(stream),
            trace_state,
            request_id,
        );

        // prepend the annotations to the response stream
        let stream = annotations_stream.chain(stream);

        // return the response stream
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<NvCreateEmbeddingRequest>,
        ManyOut<Annotated<NvCreateEmbeddingResponse>>,
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
    > for OpenAIPreprocessor
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateEmbeddingRequest>,
        next: Arc<
            dyn AsyncEngine<
                    SingleIn<PreprocessedEmbeddingRequest>,
                    ManyOut<Annotated<EmbeddingsEngineOutput>>,
                    Error,
                >,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateEmbeddingResponse>>, Error> {
        // Unpack request
        let (request, context) = request.into_parts();

        // Preprocess the embedding request
        let (preprocessed_request, annotations) =
            self.preprocess_embedding_request(&request).await?;

        // Forward to next stage
        let preprocessed_request = context.map(|_| preprocessed_request);
        let response_stream = next.generate(preprocessed_request).await?;

        // Extract context once
        let context = response_stream.context();

        // Transform response stream back to OpenAI format
        let stream = Self::transform_embedding_postprocessor_stream(response_stream, request);

        // Prepend annotations
        let annotations_stream = stream::iter(
            annotations
                .into_iter()
                .flat_map(|(k, v)| Annotated::from_annotation(k, &v))
                .collect::<Vec<_>>(),
        );

        let combined_stream = annotations_stream.chain(stream);
        Ok(ResponseStream::new(Box::pin(combined_stream), context))
    }
}

// Note: tests for jailing and parser detection live in `lib/llm/tests/test_jail.rs`

#[cfg(test)]
mod strip_tests {
    use super::OpenAIPreprocessor;

    #[test]
    fn test_strip_inline_data_urls_replaces_data_urls() {
        let mut messages = serde_json::json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR...longdata..."}},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
            ]
        }]);
        OpenAIPreprocessor::strip_inline_data_urls(&mut messages);
        let parts = messages[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "What is this?");
        assert_eq!(parts[1]["image_url"]["url"], "");
        assert_eq!(parts[2]["image_url"]["url"], "https://example.com/img.png");
    }

    #[test]
    fn test_strip_inline_data_urls_handles_video_audio() {
        let mut messages = serde_json::json!([{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA..."}},
                {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.wav"}}
            ]
        }]);
        OpenAIPreprocessor::strip_inline_data_urls(&mut messages);
        let parts = messages[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["video_url"]["url"], "");
        assert_eq!(
            parts[1]["audio_url"]["url"],
            "https://example.com/audio.wav"
        );
    }

    #[test]
    fn test_strip_inline_data_urls_preserves_text_only() {
        let mut messages = serde_json::json!([{
            "role": "user",
            "content": "plain text message"
        }]);
        let original = messages.clone();
        OpenAIPreprocessor::strip_inline_data_urls(&mut messages);
        assert_eq!(messages, original);
    }

    #[test]
    fn test_strip_inline_data_urls_empty_messages() {
        let mut messages = serde_json::json!([]);
        OpenAIPreprocessor::strip_inline_data_urls(&mut messages);
        assert_eq!(messages, serde_json::json!([]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PRE.1 — `skip_special_tokens` default. See `lib/llm/PREPROCESSOR_CASES.md`.
    #[test]
    fn test_parser_requires_special_tokens() {
        let cases: &[(Option<&str>, Option<&str>, bool, &str)] = &[
            (
                Some("gemma4"),
                None,
                true,
                "gemma4 tool-call only → required",
            ),
            (
                None,
                Some("gemma4"),
                true,
                "gemma4 reasoning only → required",
            ),
            (
                Some("gemma-4"),
                None,
                true,
                "gemma-4 hyphen alias (tool) → required",
            ),
            (
                None,
                Some("gemma-4"),
                true,
                "gemma-4 hyphen alias (reasoning) → required",
            ),
            (
                Some("gemma4"),
                Some("gemma4"),
                true,
                "gemma4 paired → required",
            ),
            (Some("hermes"), None, false, "hermes → not required"),
            (
                Some("harmony"),
                None,
                true,
                "harmony tool-call only → required",
            ),
            (
                None,
                Some("gpt_oss"),
                true,
                "gpt_oss reasoning only → required",
            ),
            (
                Some("harmony"),
                Some("gpt_oss"),
                true,
                "harmony + gpt_oss paired → required",
            ),
            (
                Some("kimi_k2"),
                Some("kimi_k25"),
                true,
                "kimi_k2 + kimi_k25 paired → required \
                 (tool-call markers `<|tool_calls_section_*|>` and reasoning \
                  marker `</think>` are special tokens that get stripped under \
                  the default skip_special_tokens=true)",
            ),
            (
                None,
                Some("kimi_k25"),
                true,
                "kimi_k25 reasoning only → required (`</think>` is special token id 163607)",
            ),
            (
                Some("kimi_k2"),
                None,
                true,
                "kimi_k2 tool-call only → required \
                 (`<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>` are special)",
            ),
            (None, None, false, "no parsers → not required"),
        ];
        for (tool, reasoning, expected, desc) in cases {
            assert_eq!(
                OpenAIPreprocessor::parser_requires_special_tokens(*tool, *reasoning),
                *expected,
                "FAILED: {desc}",
            );
        }
    }

    /// Guard: a caller forcing `skip_special_tokens=true` while a
    /// special-token-dependent parser is active strips the markers before
    /// parsing → silent empty tool_calls / leaked markup. Only that exact
    /// combination should trip the warning condition.
    #[test]
    fn test_special_tokens_will_be_stripped() {
        let f = OpenAIPreprocessor::special_tokens_will_be_stripped;
        // forced-true + marker-dependent parser → will be stripped (warn)
        assert!(f(Some(true), Some("harmony"), Some("gpt_oss")));
        assert!(f(Some(true), Some("harmony"), None));
        assert!(f(Some(true), None, Some("gpt_oss")));
        assert!(f(Some(true), Some("kimi_k2"), None));
        // false / unset → never (default path keeps the markers)
        assert!(!f(Some(false), Some("harmony"), None));
        assert!(!f(None, Some("harmony"), None));
        // forced-true but parser doesn't need special tokens → fine
        assert!(!f(Some(true), Some("hermes"), None));
        assert!(!f(Some(true), None, None));
    }

    #[test]
    fn test_backend_extra_args_preserves_nvext_and_sampling_extensions() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "detokenize": false,
            "allowed_token_ids": [10, 11],
            "bad_words_token_ids": [[12, 13]],
            "nvext": {
                "cache_salt": "step_7",
                "extra_fields": ["completion_token_ids"]
            }
        }))
        .unwrap();

        let extra_args = OpenAIPreprocessor::backend_extra_args(&request).unwrap();

        assert_eq!(extra_args["nvext"]["cache_salt"], "step_7");
        assert_eq!(
            extra_args["nvext"]["extra_fields"],
            serde_json::json!(["completion_token_ids"])
        );
        assert_eq!(extra_args["sampling_options"]["detokenize"], false);
        assert_eq!(
            extra_args["sampling_options"]["allowed_token_ids"],
            serde_json::json!([10, 11])
        );
        assert_eq!(
            extra_args["sampling_options"]["bad_words_token_ids"],
            serde_json::json!([[12, 13]])
        );
    }

    #[test]
    fn test_internal_preserve_omitted_max_tokens_option() {
        assert_eq!(
            OpenAIPreprocessor::omitted_max_tokens_default(
                10,
                100,
                PreprocessRequestOptions::default()
            ),
            Some(90)
        );
        assert_eq!(
            OpenAIPreprocessor::omitted_max_tokens_default(
                10,
                100,
                PreprocessRequestOptions {
                    preserve_omitted_max_tokens: true,
                },
            ),
            None
        );
        assert_eq!(
            OpenAIPreprocessor::omitted_max_tokens_default(
                10,
                0,
                PreprocessRequestOptions::default()
            ),
            None
        );
    }

    /// PRE.2 — Per-request reasoning gate. See `lib/llm/PREPROCESSOR_CASES.md`.
    #[test]
    fn test_is_reasoning_disabled_by_request() {
        let thinking_true = {
            let mut m = std::collections::HashMap::new();
            m.insert("thinking".to_string(), serde_json::Value::Bool(true));
            m
        };
        let thinking_false = {
            let mut m = std::collections::HashMap::new();
            m.insert("thinking".to_string(), serde_json::Value::Bool(false));
            m
        };
        let enable_thinking_true = {
            let mut m = std::collections::HashMap::new();
            m.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
            m
        };
        let enable_thinking_false = {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "enable_thinking".to_string(),
                serde_json::Value::Bool(false),
            );
            m
        };
        let force_nonempty_content_true = {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "force_nonempty_content".to_string(),
                serde_json::Value::Bool(true),
            );
            m
        };
        let thinking_mode_chat = {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "thinking_mode".to_string(),
                serde_json::Value::String("chat".to_string()),
            );
            m
        };
        let thinking_mode_thinking = {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "thinking_mode".to_string(),
                serde_json::Value::String("thinking".to_string()),
            );
            m
        };
        let empty_args = std::collections::HashMap::new();

        // (parser, args, expected_disabled, description)
        let cases = [
            (
                Some("kimi_k25"),
                Some(&thinking_false),
                true,
                "kimi_k25 + thinking=false → disabled",
            ),
            (
                Some("kimi_k25"),
                Some(&thinking_true),
                false,
                "kimi_k25 + thinking=true → enabled",
            ),
            (
                Some("kimi_k25"),
                None,
                false,
                "kimi_k25 + no args → enabled",
            ),
            (
                Some("kimi_k25"),
                Some(&empty_args),
                false,
                "kimi_k25 + empty args → enabled",
            ),
            // deepseek_r1 uses "thinking" bool or "thinking_mode" string
            (
                Some("deepseek_r1"),
                Some(&thinking_false),
                true,
                "deepseek_r1 + thinking=false → disabled",
            ),
            (
                Some("deepseek_r1"),
                Some(&thinking_true),
                false,
                "deepseek_r1 + thinking=true → enabled",
            ),
            (
                Some("deepseek_r1"),
                Some(&thinking_mode_chat),
                true,
                "deepseek_r1 + thinking_mode=chat → disabled",
            ),
            (
                Some("deepseek_r1"),
                Some(&thinking_mode_thinking),
                false,
                "deepseek_r1 + thinking_mode=thinking → enabled",
            ),
            (
                Some("deepseek_r1"),
                None,
                false,
                "deepseek_r1 + no args → enabled",
            ),
            (
                Some("deepseek_r1"),
                Some(&empty_args),
                false,
                "deepseek_r1 + empty args → enabled",
            ),
            (
                Some("basic"),
                Some(&thinking_false),
                false,
                "basic → never disabled",
            ),
            (
                None,
                Some(&thinking_false),
                false,
                "no parser → never disabled",
            ),
            // nemotron_nano uses "enable_thinking" key
            (
                Some("nemotron_nano"),
                Some(&enable_thinking_false),
                true,
                "nemotron_nano + enable_thinking=false → disabled",
            ),
            (
                Some("nemotron_nano"),
                Some(&enable_thinking_true),
                false,
                "nemotron_nano + enable_thinking=true → enabled",
            ),
            (
                Some("nemotron_nano"),
                None,
                false,
                "nemotron_nano + no args → enabled",
            ),
            (
                Some("nemotron_nano"),
                Some(&empty_args),
                false,
                "nemotron_nano + empty args → enabled",
            ),
            (
                Some("nemotron3"),
                Some(&force_nonempty_content_true),
                true,
                "nemotron3 + force_nonempty_content=true → disabled",
            ),
            (
                Some("nemotron_v3"),
                Some(&enable_thinking_false),
                true,
                "nemotron_v3 + enable_thinking=false → disabled",
            ),
            (
                Some("nemotron_v3"),
                Some(&force_nonempty_content_true),
                true,
                "nemotron_v3 + force_nonempty_content=true → disabled",
            ),
            // deepseek_v4 — same convention as deepseek_r1; verify all three aliases
            // (deepseek_v4 / deepseek-v4 / deepseekv4) plus both signal keys.
            (
                Some("deepseek_v4"),
                Some(&thinking_false),
                true,
                "deepseek_v4 + thinking=false → disabled",
            ),
            (
                Some("deepseek_v4"),
                Some(&thinking_true),
                false,
                "deepseek_v4 + thinking=true → enabled",
            ),
            (
                Some("deepseek_v4"),
                Some(&thinking_mode_chat),
                true,
                "deepseek_v4 + thinking_mode=chat → disabled",
            ),
            (
                Some("deepseek_v4"),
                Some(&thinking_mode_thinking),
                false,
                "deepseek_v4 + thinking_mode=thinking → enabled",
            ),
            (
                Some("deepseek_v4"),
                None,
                false,
                "deepseek_v4 + no args → enabled",
            ),
            (
                Some("deepseek-v4"),
                Some(&thinking_false),
                true,
                "deepseek-v4 (hyphen alias) + thinking=false → disabled",
            ),
            (
                Some("deepseekv4"),
                Some(&thinking_mode_chat),
                true,
                "deepseekv4 (joined alias) + thinking_mode=chat → disabled",
            ),
            (
                Some("deepseek_v4"),
                Some(&enable_thinking_false),
                true,
                "deepseek_v4 + enable_thinking=false → disabled (vLLM alias)",
            ),
            (
                Some("deepseek_v4"),
                Some(&enable_thinking_true),
                false,
                "deepseek_v4 + enable_thinking=true → enabled (vLLM alias)",
            ),
            (
                Some("gemma4"),
                Some(&enable_thinking_false),
                true,
                "gemma4 + enable_thinking=false → disabled",
            ),
            (
                Some("gemma4"),
                Some(&enable_thinking_true),
                false,
                "gemma4 + enable_thinking=true → enabled",
            ),
            (
                Some("gemma4"),
                None,
                false,
                "gemma4 + no args → enabled (parser still runs but is a no-op when no markers arrive)",
            ),
            (
                Some("gemma-4"),
                Some(&enable_thinking_false),
                true,
                "gemma-4 (hyphen alias) + enable_thinking=false → disabled",
            ),
        ];

        for (parser, args, expected, desc) in cases {
            assert_eq!(
                OpenAIPreprocessor::is_reasoning_disabled_by_request(parser, args),
                expected,
                "FAILED: {desc}",
            );
        }
    }

    /// Different query strings must produce different hashes. `?v=1` and
    /// `?v=2` may look like cache-busters, but they could equally be a
    /// content selector ("version 2 of the image"). The URL alone doesn't
    /// tell us which, so we keep the hash URL-identical and let the URL be
    /// the identity. For signed-URL workloads where rotation actually
    /// hides a stable object, `--frontend-decoding` hashes the decoded
    /// bytes instead.
    #[cfg(feature = "mm-routing")]
    #[test]
    fn hash_image_url_distinguishes_query_strings() {
        let base = "https://cdn.example.com/img.jpg";
        let v1 = OpenAIPreprocessor::hash_image_url(&format!("{base}?v=1"));
        let v2 = OpenAIPreprocessor::hash_image_url(&format!("{base}?v=2"));
        let no_q = OpenAIPreprocessor::hash_image_url(base);
        assert_ne!(v1, v2, "different query values must hash differently");
        assert_ne!(v1, no_q, "presence of a query string must change the hash");
    }

    /// Rotating S3 / GCS / Azure SAS signatures change the URL and
    /// therefore the hash. This is a known limitation of URL-passthrough
    /// routing for signed-URL workloads — `--frontend-decoding` is the
    /// recommended mode there because it hashes the decoded image bytes
    /// regardless of how the URL was signed.
    #[cfg(feature = "mm-routing")]
    #[test]
    fn hash_image_url_distinguishes_rotating_signatures() {
        let base = "https://bucket.s3.amazonaws.com/img.jpg";
        let a = OpenAIPreprocessor::hash_image_url(&format!(
            "{base}?X-Amz-Signature=AAA&X-Amz-Date=20260101T000000Z&X-Amz-Expires=600"
        ));
        let b = OpenAIPreprocessor::hash_image_url(&format!(
            "{base}?X-Amz-Signature=BBB&X-Amz-Date=20260101T010000Z&X-Amz-Expires=900"
        ));
        assert_ne!(
            a, b,
            "rotating presign params produce a different URL and must hash differently"
        );
    }

    /// Identical URLs must hash to the same value (the basic identity
    /// guarantee that makes URL-passthrough routing useful at all).
    #[cfg(feature = "mm-routing")]
    #[test]
    fn hash_image_url_is_deterministic_for_identical_urls() {
        let url = "https://cdn.example.com/img.jpg?width=256";
        assert_eq!(
            OpenAIPreprocessor::hash_image_url(url),
            OpenAIPreprocessor::hash_image_url(url),
        );
    }

    /// data: URIs hash the entire URI string. Same payload → same hash;
    /// different payload → different hash.
    #[cfg(feature = "mm-routing")]
    #[test]
    fn hash_image_url_data_uri_content_addressed() {
        let same = "data:image/png;base64,AAAA";
        let other = "data:image/png;base64,BBBB";
        assert_eq!(
            OpenAIPreprocessor::hash_image_url(same),
            OpenAIPreprocessor::hash_image_url(same)
        );
        assert_ne!(
            OpenAIPreprocessor::hash_image_url(same),
            OpenAIPreprocessor::hash_image_url(other),
            "different data URI payloads must hash differently"
        );
    }

    /// Non-HTTP / non-data schemes (s3://, gs://, file://) hash as-is.
    #[cfg(feature = "mm-routing")]
    #[test]
    fn hash_image_url_other_schemes_passthrough() {
        let s3a = OpenAIPreprocessor::hash_image_url("s3://bucket/key?v=1");
        let s3b = OpenAIPreprocessor::hash_image_url("s3://bucket/key?v=2");
        assert_ne!(
            s3a, s3b,
            "s3:// query params identify objects and must not collide"
        );
    }
}
