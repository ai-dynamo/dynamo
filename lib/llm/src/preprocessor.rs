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
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPart,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionToolChoiceOption, EncodingFormat,
};
use dynamo_renderer::{OAIPromptFormatter, PromptRenderError, RenderedPrompt};
use dynamo_runtime::config::{is_truthy, parse_bool_opt};
use dynamo_runtime::error::{DynamoError, ErrorType};
use either::Either;
use futures::Stream;
use futures::stream::{self, StreamExt};
use std::borrow::Cow;
use std::time::Instant;

use dynamo_runtime::dynamo_nvtx_range;
use dynamo_runtime::metrics::frontend_perf::{
    DETOKENIZE_TOKEN_COUNT, DETOKENIZE_TOTAL_US, STAGE_DURATION_SECONDS, STAGE_PREPROCESS,
    StageGuard, TEMPLATE_SECONDS, TOKENIZE_SECONDS,
};
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::Arc,
};
use tracing;

use crate::local_model::runtime_config::{TOKEN_BUDGET_RUNTIME_KEY, TokenBudget};
#[cfg(feature = "mm-routing")]
use crate::model_card::ModelInfoType;
use crate::model_card::{ModelDeploymentCard, ModelInfo};
use crate::preprocessor::media::MediaLoader;
use crate::protocols::common::preprocessor::{
    MultimodalData, MultimodalDataMap, MultimodalUuidMap, PreprocessedRequestBuilder, RoutingHints,
};
use crate::protocols::common::timing::RequestTracker;
use crate::tokenizers::Encoding;

use dynamo_parsers::{
    ReasoningParser, ReasoningParserType, reasoning::ParserResult,
    tool_calling::parsers::get_tool_parser_map,
};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{
    AsyncEngineContext, Context as PipelineContext, Error, ManyOut, Operator, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::{Annotated, AnnotationsProvider};

use crate::protocols::{
    TokenIdType,
    common::{
        OutputOptionsProvider, SamplingOptionsProvider, StopConditionsProvider,
        extensions::{AgentHints, NvExtProvider, request_cache_salt, routing_constraints_to_kv},
    },
    openai::{
        DeltaGeneratorExt,
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
        embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
    },
};
use crate::tokenizers::traits::Tokenizer;

use crate::preprocessor::prompt::{MediaRequestExt, prompt_formatter_from_mdc};
use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter, PromptInput, TextInput, TokenInput};

pub use crate::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
pub use crate::protocols::common::metrics::{
    ANNOTATION_LLM_METRICS, ANNOTATION_PAYLOAD_USAGE, LLMMetricAnnotation,
};
pub use crate::protocols::common::preprocessor::PreprocessedEmbeddingRequest;

use crate::protocols::common::llm_backend::EmbeddingsEngineOutput;

fn routing_priorities(hints: Option<&AgentHints>) -> (Option<f64>, Option<u32>, Option<i32>) {
    let priority_jump = hints.and_then(|h| {
        h.priority
            .map(|priority| priority.max(0) as f64)
            .or(h.latency_sensitivity)
    });
    let strict_priority = hints.and_then(|h| h.strict_priority);
    let priority = hints.and_then(|h| h.priority);
    (priority_jump, strict_priority, priority)
}

fn invalid_argument_error(message: impl Into<String>) -> anyhow::Error {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message.into())
        .build()
        .into()
}

fn tool_content_part_as_user(
    part: &ChatCompletionRequestToolMessageContentPart,
) -> Cow<'_, ChatCompletionRequestUserMessageContentPart> {
    Cow::Owned(match part {
        ChatCompletionRequestToolMessageContentPart::Text(part) => {
            ChatCompletionRequestUserMessageContentPart::Text(part.clone())
        }
        ChatCompletionRequestToolMessageContentPart::ImageUrl(part) => {
            ChatCompletionRequestUserMessageContentPart::ImageUrl(part.clone())
        }
        ChatCompletionRequestToolMessageContentPart::VideoUrl(part) => {
            ChatCompletionRequestUserMessageContentPart::VideoUrl(part.clone())
        }
        ChatCompletionRequestToolMessageContentPart::AudioUrl(part) => {
            ChatCompletionRequestUserMessageContentPart::AudioUrl(part.clone())
        }
    })
}

enum MultimodalContentPart<'a> {
    User(&'a ChatCompletionRequestUserMessageContentPart),
    Tool(&'a ChatCompletionRequestToolMessageContentPart),
}

impl<'a> MultimodalContentPart<'a> {
    fn as_user(&self) -> Cow<'a, ChatCompletionRequestUserMessageContentPart> {
        match self {
            Self::User(part) => Cow::Borrowed(part),
            Self::Tool(part) => tool_content_part_as_user(part),
        }
    }

    fn media_info(&self) -> Option<(&'static str, Option<url::Url>, Option<String>)> {
        match self {
            Self::User(part) => match *part {
                ChatCompletionRequestUserMessageContentPart::ImageUrl(part) => Some((
                    "image_url",
                    part.image_url.as_ref().map(|media| media.url.clone()),
                    part.uuid.clone(),
                )),
                ChatCompletionRequestUserMessageContentPart::VideoUrl(part) => Some((
                    "video_url",
                    part.video_url.as_ref().map(|media| media.url.clone()),
                    part.uuid.clone(),
                )),
                ChatCompletionRequestUserMessageContentPart::AudioUrl(part) => Some((
                    "audio_url",
                    part.audio_url.as_ref().map(|media| media.url.clone()),
                    part.uuid.clone(),
                )),
                _ => None,
            },
            Self::Tool(part) => match *part {
                ChatCompletionRequestToolMessageContentPart::ImageUrl(part) => Some((
                    "image_url",
                    part.image_url.as_ref().map(|media| media.url.clone()),
                    part.uuid.clone(),
                )),
                ChatCompletionRequestToolMessageContentPart::VideoUrl(part) => Some((
                    "video_url",
                    part.video_url.as_ref().map(|media| media.url.clone()),
                    part.uuid.clone(),
                )),
                ChatCompletionRequestToolMessageContentPart::AudioUrl(part) => Some((
                    "audio_url",
                    part.audio_url.as_ref().map(|media| media.url.clone()),
                    part.uuid.clone(),
                )),
                _ => None,
            },
        }
    }
}

fn multimodal_content_parts(
    message: &ChatCompletionRequestMessage,
) -> Option<impl Iterator<Item = MultimodalContentPart<'_>>> {
    match message {
        ChatCompletionRequestMessage::User(user) => match &user.content {
            ChatCompletionRequestUserMessageContent::Array(parts) => {
                Some(Either::Left(parts.iter().map(MultimodalContentPart::User)))
            }
            ChatCompletionRequestUserMessageContent::Text(_) => None,
        },
        ChatCompletionRequestMessage::Tool(tool) => match &tool.content {
            ChatCompletionRequestToolMessageContent::Array(parts) => {
                Some(Either::Right(parts.iter().map(MultimodalContentPart::Tool)))
            }
            ChatCompletionRequestToolMessageContent::Text(_) => None,
        },
        _ => None,
    }
}

#[cfg(feature = "mm-routing")]
fn image_content_part_url(
    content_part: &ChatCompletionRequestUserMessageContentPart,
) -> Option<&str> {
    let ChatCompletionRequestUserMessageContentPart::ImageUrl(part) = content_part else {
        return None;
    };
    part.image_url.as_ref().map(|image| image.url.as_str())
}

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
const DEFAULT_THINKING_MODE_RUNTIME_KEY: &str = "default_thinking_mode";

/// Drain a standalone router's forwarded `routing_data` onto this request's tracker so the
/// frontend's timing/worker/token surfaces populate, then drop the field to keep it off the
/// client wire. Worker attribution is drained so `build_response_nvext` can surface it on the
/// split-router query-only (`query_instance_id`) path; it is first-write-wins, so the
/// frontend's own recordings (when present) take precedence over the forwarded values.
fn drain_router_routing_data(
    data: &mut Option<BackendOutput>,
    tracker: Option<&crate::protocols::common::timing::RequestTracker>,
) {
    let Some(routing_data) = data.as_mut().and_then(|d| d.routing_data.take()) else {
        return;
    };
    let Some(tracker) = tracker else {
        return;
    };
    if let Some(timing) = routing_data.timing {
        tracker.set_external_timing(timing);
    }
    if let Some(worker_id) = routing_data.worker_id {
        tracker.set_external_worker_info(worker_id);
    }
    if let Some(token_ids) = routing_data.token_ids {
        tracker.set_external_query_token_ids(token_ids);
    }
}

fn attach_metrics_annotation<Resp>(response: &mut Annotated<Resp>, metrics: &LLMMetricAnnotation) {
    if let Ok(metrics_annotated) = metrics.to_annotation::<()>() {
        response.event = metrics_annotated.event;
        response.comment = metrics_annotated.comment;
    } else {
        tracing::warn!("Failed to serialize LLM metrics annotation");
    }
}

fn attach_llm_metrics<Resp>(response: &mut Annotated<Resp>, metrics: LLMMetricAnnotation)
where
    Resp: 'static,
{
    if response.event.is_some() {
        return;
    }

    if let Some(data) = response.data.as_mut()
        && let Some(chat_response) =
            (data as &mut dyn Any).downcast_mut::<NvCreateChatCompletionStreamResponse>()
    {
        chat_response.llm_metrics = Some(metrics);
        return;
    }

    attach_metrics_annotation(response, &metrics);
}

#[inline]
fn build_llm_metric_annotation(
    tracker: Option<&RequestTracker>,
    input_tokens: usize,
    output_tokens: usize,
    chunk_tokens: usize,
    cached_tokens: Option<usize>,
    mm_counts: MultimodalCounts,
    image_tokens: Option<usize>,
) -> LLMMetricAnnotation {
    LLMMetricAnnotation {
        input_tokens,
        output_tokens,
        chunk_tokens,
        cached_tokens,
        image_count: mm_counts.image,
        video_count: mm_counts.video,
        audio_count: mm_counts.audio,
        image_tokens,
        prefill_worker_id: tracker.and_then(|t| t.prefill_worker_id()),
        prefill_dp_rank: tracker.and_then(|t| t.prefill_dp_rank()),
        prefill_worker_type: tracker
            .and_then(|t| t.prefill_worker_type())
            .map(String::from),
        decode_worker_id: tracker.and_then(|t| t.decode_worker_id()),
        decode_dp_rank: tracker.and_then(|t| t.decode_dp_rank()),
        decode_worker_type: tracker
            .and_then(|t| t.decode_worker_type())
            .map(String::from),
        tokenize_latency: tracker.and_then(|t| t.tokenize_latency()),
        detokenize_total_latency: tracker.and_then(|t| t.detokenize_total_latency()),
        detokenize_count: tracker.map(|t| t.detokenize_count()),
    }
}

// Reasoning State for reasoning parsing transformation step.
//
// The reasoning parser and the guided-JSON bypass decision are kept per
// `choice.index` so that with `n > 1` one choice's bare-JSON bypass cannot
// suppress another choice's reasoning split. This mirrors the per-choice state
// already used by the tool-call jail and the leading-`<think>` strip stage.
struct ChoiceReasoningState {
    parser: Box<dyn ReasoningParser>,
    guided_json_bypass_decision: Option<bool>,
    // Reasoning text held back while it is still ambiguous. Only used on the
    // `defer_reasoning_for_nonempty_content` path — see the buffering note on
    // `ReasoningState`.
    pending_reasoning: String,
    // Whitespace-only normal text seen while still ambiguous. Held rather than
    // emitted so the streaming path agrees with the aggregator, which counts
    // whitespace-only content as empty.
    pending_content: String,
    // Set once this choice's parser has produced normal text, which for these
    // force-reasoning parsers means it left the reasoning block. From then on
    // the ambiguity is gone and deltas pass straight through.
    left_reasoning: bool,
    // Set once this choice has been drained, so the terminal-chunk drain and the
    // end-of-stream fallback cannot both emit the same bytes. Reopened if the
    // backend keeps sending content after `finish_reason`, so those bytes still
    // get a destination.
    drained: bool,
    // Set the first time this choice's parser is finalized and NEVER reset, even
    // when `drained` is reopened. `ReasoningParser::finish_reasoning_stream` is
    // not documented to be idempotent, so calling it twice could re-emit text a
    // parser had already handed over. Draining again is fine; finalizing again
    // is not.
    parser_finished: bool,
}

/// Strip every per-chunk field from a response that is being reused as the
/// envelope for a synthetic end-of-stream chunk.
///
/// Both end-of-stream flushes build their chunk by cloning the last chunk they
/// saw, because that is the only way to carry the buffered bytes out on a
/// correctly-shaped response. The clone is not a new generation: it produced no
/// tokens and re-channels bytes the parser was already holding. So everything
/// describing the *original* chunk's generation has to go, or it is reported
/// twice:
///
/// - `event` / `comment` — the annotation channel carries per-chunk payloads
///   such as generated `token_ids`.
/// - `error` — an error annotation must not be replayed on a later chunk.
/// - `usage` / `llm_metrics` — `metrics.rs` sums `chunk_tokens` and samples an
///   ITL point per chunk carrying `llm_metrics`.
/// - `nvext` — per-chunk NVIDIA extensions. `merge_response_nvext` append-merges
///   `completion_token_ids`, so a repeat turns `[42]` into `[42, 42]`, and
///   fields it does not append (such as `prompt_logprobs`) are overwritten.
///
/// Kept as one function so the two call sites cannot drift apart: they already
/// did, which is how `nvext` survived on both.
fn scrub_synthetic_chunk_metadata(
    response: &mut Annotated<NvCreateChatCompletionStreamResponse>,
) -> Option<()> {
    response.event = None;
    response.comment = None;
    response.error = None;
    let data = response.data.as_mut()?;
    data.inner.usage = None;
    data.llm_metrics = None;
    data.nvext = None;
    Some(())
}

/// Drain what a choice still holds on the `defer_reasoning_for_nonempty_content`
/// path, returning `(content, reasoning_content)` to add to its delta. Shared by
/// the terminal-chunk drain and the end-of-stream fallback so the two cannot
/// disagree about which channel the leftover bytes belong to.
fn drain_deferred_reasoning(state: &mut ChoiceReasoningState) -> (Option<String>, Option<String>) {
    if state.drained {
        return (None, None);
    }
    state.drained = true;
    let pending = std::mem::take(&mut state.pending_reasoning);
    let pending_content = std::mem::take(&mut state.pending_content);
    // Finalize at most once per choice; see `parser_finished`.
    let result = if state.parser_finished {
        ParserResult::default()
    } else {
        state.parser_finished = true;
        state.parser.finish_reasoning_stream()
    };

    if state.left_reasoning {
        // An answer already streamed as content, so the non-empty-content
        // promise is met and the parser's own split stands: a later truncated
        // reasoning block stays reasoning instead of being appended to the
        // answer. `pending_content` is empty here — it was released with the
        // answer when the choice left reasoning.
        let mut content = pending_content;
        content.push_str(&result.normal_text);
        (
            (!content.is_empty()).then_some(content),
            (!result.reasoning_text.is_empty()).then_some(result.reasoning_text),
        )
    } else {
        // No real answer ever arrived, so everything held is all the response
        // has and must land in content — including a truncated `<think>` prefix,
        // which the parser reports as reasoning or normal text depending on
        // where it stopped. Held whitespace is dropped rather than prepended when
        // there is reasoning to surface, matching the aggregator, which replaces
        // whitespace-only content with the reasoning text instead of
        // concatenating.
        let mut text = pending;
        text.push_str(&result.reasoning_text);
        text.push_str(&result.normal_text);
        if text.is_empty() {
            // ...but if whitespace is genuinely all the turn produced, it is the
            // whole response. Returning nothing here would hand back an empty
            // `content` to a request that explicitly asked for non-empty content.
            text = pending_content;
        }
        ((!text.is_empty()).then_some(text), None)
    }
}

struct ReasoningState {
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>,
    parser_name: String,
    prompt_injected_reasoning: bool,
    bypass_bare_guided_json: bool,
    choices: HashMap<u32, ChoiceReasoningState>,
    // Last emitted content-bearing response, reused as the envelope to carry any
    // text the parsers are still buffering when the upstream stream ends. Only
    // retained when `defer_reasoning_for_nonempty_content` is set, to keep the
    // per-token clone off the common reasoning hot path. Chunks with no choices
    // (the trailing usage-only chunk) are never retained — that envelope has no
    // delta slot to attach the flushed bytes to, so the flush would be dropped.
    last_response: Option<Annotated<NvCreateChatCompletionStreamResponse>>,
    // Nemotron force-reasoning parsers start inside the reasoning block, so
    // leading model output with no `<think>` is reported as reasoning even when
    // it is really the whole answer. Under `force_nonempty_content=true` the
    // chat template promises non-empty `content`, so emitting that text as
    // `reasoning_content` would leave `content` empty — the exact failure the
    // flag exists to prevent.
    //
    // When this is set, reasoning deltas are held in each choice's
    // `pending_reasoning` until the ambiguity resolves:
    //   - the parser produces normal text (it left the reasoning block), so the
    //     buffer really was reasoning: emit it as `reasoning_content` and stream
    //     the answer as `content` from then on; or
    //   - the stream ends first, so no answer is coming: emit the buffer (plus
    //     anything `finish_reasoning_stream` still holds, e.g. a truncated
    //     `<think>` prefix) as `content`.
    //
    // The cost is real and worth stating plainly: everything held is delivered
    // in one delta rather than token by token. For a turn that does reason, that
    // is the `reasoning_content` up to `</think>`, after which the answer streams
    // normally. For a turn with no `</think>` at all the held text IS the answer,
    // so the whole answer lands in a single delta on the terminal chunk — on
    // `main` (parser off, leading `<think>` stripped) that case streamed
    // incrementally.
    //
    // There is no cheaper split available: these parsers emit reasoning with no
    // opening `<think>` (see the `nemotron_nano` mapping in dynamo-parsers), so
    // until `</think>` arrives the same bytes are equally consistent with
    // reasoning and with a plain answer. Emitting eagerly is what leaked reasoning
    // into `content` in the first place. Scoped to the flag so no other parser
    // pays it.
    defer_reasoning_for_nonempty_content: bool,
    // Set once the backend has sent an error annotation. The end-of-stream
    // flush must not run after that: an error is terminal, and emitting
    // recovered reasoning as ordinary `content` after it would put text on the
    // wire that the generation never successfully produced.
    //
    // This matters because the flush is the only place bytes can arrive after
    // the last upstream chunk. `/v1/chat/completions` ends its SSE on the error
    // and never sees it, but `/v1/responses` and `/v1/messages` both record the
    // error and keep consuming (`saw_error = true; continue;`), so without this
    // latch they emit the synthetic content chunk and only then report failure.
    saw_terminal_error: bool,
}

/// Per-image routing payload accumulated by `gather_multi_modal_data` and
/// consumed by `gather_mm_exact_routing_info`.
#[derive(Debug, Clone, Copy)]
pub struct MmImageEntry {
    pub mm_hash: u64,
    pub width: u32,
    pub height: u32,
}

struct MediaFetchTask<'a> {
    modality: &'static str,
    slot_idx: usize,
    content_part: Cow<'a, ChatCompletionRequestUserMessageContentPart>,
}

/// Per-request media content-part counts, carried to the metrics annotation.
/// Derived from `multi_modal_data`, so independent of the `mm-routing` feature.
#[derive(Debug, Clone, Copy, Default)]
pub struct MultimodalCounts {
    pub image: usize,
    pub video: usize,
    pub audio: usize,
}

impl MultimodalCounts {
    /// Count `image_url` / `video_url` / `audio_url` parts (vec length per modality).
    fn from_preprocessed(request: &PreprocessedRequest) -> Self {
        let count = |key: &str| {
            request
                .multi_modal_data
                .as_ref()
                .and_then(|m| m.get(key))
                .map_or(0, |v| v.len())
        };
        Self {
            image: count("image_url"),
            video: count("video_url"),
            audio: count("audio_url"),
        }
    }
}

#[cfg(feature = "mm-routing")]
fn checked_add_image_tokens(total: Option<usize>, next: usize) -> Option<usize> {
    total?.checked_add(next)
}

#[cfg(feature = "mm-routing")]
fn has_image_token_processor_override(value: Option<&serde_json::Value>) -> bool {
    value.is_some_and(|value| match value {
        serde_json::Value::Null => false,
        serde_json::Value::Object(map) => !map.is_empty(),
        // The schema expects an object. Fail closed for every other JSON shape
        // because its processor semantics are unknown here.
        _ => true,
    })
}

#[cfg(feature = "mm-routing")]
fn aggregate_image_tokens(
    image_tokens: Option<usize>,
    resolved_image_count: usize,
    total_image_count: usize,
    has_processor_override: bool,
) -> Option<usize> {
    if total_image_count == 0 || has_processor_override || resolved_image_count != total_image_count
    {
        return None;
    }

    image_tokens
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
/// path used by MM-aware routing and image-token metrics. Inherits the same
/// policy contract as the frontend-decode path (`MediaLoader`): blocklist DNS
/// resolver, redirect revalidation, hostname/IP blocklist,
/// `DYN_MM_ALLOW_INTERNAL` opt-in.
///
/// **Lifecycle:** `LazyLock` so the closure runs on first access. For MM-
/// countable or routable preprocessors, `OpenAIPreprocessor::new_with_parts` calls
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

fn attach_agent_context_from_context(
    request: &mut PreprocessedRequest,
    context: &PipelineContext<()>,
) {
    if let Ok(agent_context) = context.get::<crate::protocols::common::extensions::AgentContext>(
        crate::protocols::common::extensions::AGENT_CONTEXT_CONTEXT_KEY,
    ) {
        request.agent_context = Some(agent_context.as_ref().clone());
    }
}

/// Thin wrapper that normalizes `function.arguments` in historical tool-calls
/// from a JSON string to an object before passing messages to a MiniJinja
/// template.  Only used when `OpenAIPreprocessor::normalize_tool_call_args` is
/// true (e.g. GLM-5.2); all other trait methods delegate directly to the inner
/// request so routing, sampling, and annotation behavior is unchanged.
struct NormalizedArgsRequest<'a, R>(&'a R);

impl<R: OAIChatLikeRequest> OAIChatLikeRequest for NormalizedArgsRequest<'_, R> {
    fn model(&self) -> String {
        self.0.model()
    }

    fn messages(&self) -> minijinja::value::Value {
        let mut json =
            serde_json::to_value(self.0.typed_messages().unwrap_or_default()).unwrap_or_default();
        if let Err(e) = crate::preprocessor::prompt::normalize_tool_call_arguments(&mut json) {
            tracing::error!(
                error = %e,
                "tool_call arguments normalization failed; template rendering may fail \
                 if it calls .items() on a string"
            );
        }
        minijinja::value::Value::from_serialize(&json)
    }

    fn typed_messages(&self) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
        self.0.typed_messages()
    }

    fn tools(&self) -> Option<minijinja::value::Value> {
        self.0.tools()
    }

    fn tool_choice(&self) -> Option<minijinja::value::Value> {
        self.0.tool_choice()
    }

    fn response_format(&self) -> Option<minijinja::value::Value> {
        self.0.response_format()
    }

    fn should_add_generation_prompt(&self) -> bool {
        self.0.should_add_generation_prompt()
    }

    fn extract_text(&self) -> Option<TextInput> {
        self.0.extract_text()
    }

    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        self.0.chat_template_args()
    }

    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.0.mm_processor_kwargs()
    }
}

impl<R: AnnotationsProvider> AnnotationsProvider for NormalizedArgsRequest<'_, R> {
    fn annotations(&self) -> Option<Vec<String>> {
        self.0.annotations()
    }
}

impl<R: SamplingOptionsProvider> SamplingOptionsProvider for NormalizedArgsRequest<'_, R> {
    fn extract_sampling_options(
        &self,
    ) -> anyhow::Result<crate::protocols::common::SamplingOptions> {
        self.0.extract_sampling_options()
    }
}

impl<R: StopConditionsProvider> StopConditionsProvider for NormalizedArgsRequest<'_, R> {
    fn extract_stop_conditions(&self) -> anyhow::Result<crate::protocols::common::StopConditions> {
        self.0.extract_stop_conditions()
    }
}

impl<R: OutputOptionsProvider> OutputOptionsProvider for NormalizedArgsRequest<'_, R> {
    fn extract_output_options(&self) -> anyhow::Result<crate::protocols::common::OutputOptions> {
        self.0.extract_output_options()
    }
}

impl<R: NvExtProvider> NvExtProvider for NormalizedArgsRequest<'_, R> {
    fn nvext(&self) -> Option<&crate::protocols::common::extensions::NvExt> {
        self.0.nvext()
    }

    fn raw_prompt(&self) -> Option<String> {
        self.0.raw_prompt()
    }

    fn unsupported_fields(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        self.0.unsupported_fields()
    }
}

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
    /// Normalize historical tool-call `function.arguments` from a JSON string
    /// to an object before MiniJinja rendering.  Enabled for GLM-5.2 (glm47
    /// parser) and any model that sets `normalize_tool_call_args: true` in its
    /// ModelRuntimeConfig; disabled for all other models.
    normalize_tool_call_args: bool,
    media_loader: Option<MediaLoader>,
    /// Engine-published request-token admission policy.
    token_budget: Option<TokenBudget>,
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
    /// LLaVA), so block hashes align bit-for-bit with the worker.
    ///
    /// `None` disables MM-aware routing for this model and the router falls
    /// back to text-prefix routing.
    #[cfg(feature = "mm-routing")]
    routing_image_token_id: Option<crate::protocols::TokenIdType>,
    /// BOS token id to prepend to the routing-side sequence so per-block
    /// hashes match the backend's HF processor output on models with
    /// `add_bos_token: true` (LLaVA-1.5 and other `LlamaTokenizer`
    /// families). `None` when the model doesn't need it or `bos_token`
    /// doesn't round-trip to a single id.
    #[cfg(feature = "mm-routing")]
    routing_prepend_bos: Option<crate::protocols::TokenIdType>,
}

pub(crate) const LORA_NAME_CONTEXT_KEY: &str = "discovery.lora_name";

impl OpenAIPreprocessor {
    fn omitted_max_tokens_default(
        prompt_len: usize,
        token_limit: Option<u32>,
        options: PreprocessRequestOptions,
    ) -> Option<u32> {
        if options.preserve_omitted_max_tokens {
            return None;
        }
        token_limit.map(|limit| limit.saturating_sub(prompt_len as u32))
    }

    /// Return the exact prompt length when the frontend can prove it.
    ///
    /// MM routing currently expands image placeholders only. Therefore any
    /// other non-empty media modality makes even an image-expanded length
    /// incomplete and forces backend validation.
    fn exact_prompt_len(
        expanded_image_prompt_len: Option<usize>,
        multi_modal_data: Option<&MultimodalDataMap>,
        token_ids_len: usize,
    ) -> Option<usize> {
        let has_images = multi_modal_data
            .and_then(|media| media.get("image_url"))
            .is_some_and(|items| !items.is_empty());
        let has_other_media = multi_modal_data.is_some_and(|media| {
            media
                .iter()
                .any(|(kind, items)| kind != "image_url" && !items.is_empty())
        });

        if has_other_media {
            None
        } else if let Some(expanded_prompt_len) =
            expanded_image_prompt_len.filter(|length| *length > 0)
        {
            Some(expanded_prompt_len)
        } else if has_images {
            None
        } else {
            Some(token_ids_len)
        }
    }

    /// Apply the engine-published rejection policy to an exact request length.
    fn validate_requested_token_budget(
        prompt_len: usize,
        max_tokens: Option<u32>,
        token_budget: Option<&TokenBudget>,
    ) -> Result<()> {
        let Some(token_budget) = token_budget else {
            return Ok(());
        };
        let combined_limit = token_budget.combined_limit as usize;

        // A prompt that fills the combined budget leaves no room for generation.
        // When prompt overflow is backend-owned, its effective length may change.
        if prompt_len >= combined_limit {
            if !token_budget.reject_prompt_overflow {
                return Ok(());
            }
            return Err(Self::prompt_overflow_error(prompt_len, combined_limit).into());
        }

        // Generation requires at least one output token when the cap is omitted.
        let requested_tokens = prompt_len.saturating_add(max_tokens.unwrap_or(1) as usize);
        if requested_tokens > combined_limit && token_budget.reject_total_overflow {
            let request_description = match max_tokens {
                Some(max_tokens) => format!(
                    "your request has {prompt_len} input tokens and asks for {max_tokens} output \
                     tokens ({requested_tokens} tokens total)"
                ),
                None => format!(
                    "your request has {prompt_len} input tokens and requires room for at least one \
                     output token ({requested_tokens} tokens minimum)"
                ),
            };
            return Err(DynamoError::builder()
                .error_type(ErrorType::InvalidArgument)
                .message(format!(
                    "This model configuration accepts at most {} combined input and output \
                     tokens. However, {}. Please reduce the input length or requested output \
                     length.",
                    combined_limit, request_description,
                ))
                .build()
                .into());
        }

        Ok(())
    }

    /// Validate a preprocessed request when its frontend-visible prompt length
    /// is exact, returning that length for other context-budget decisions.
    fn validate_preprocessed_token_budget(
        request: &PreprocessedRequest,
        token_budget: Option<&TokenBudget>,
    ) -> Result<Option<usize>> {
        if request.prompt_embeds.is_some() {
            return Ok(None);
        }

        let exact_prompt_len = Self::exact_prompt_len(
            request
                .mm_routing_info
                .as_ref()
                .map(|mm| mm.expanded_prompt_len),
            request.multi_modal_data.as_ref(),
            request.token_ids.len(),
        );
        if let Some(prompt_len) = exact_prompt_len {
            Self::validate_requested_token_budget(
                prompt_len,
                request.stop_conditions.max_tokens,
                token_budget,
            )?;
        }

        Ok(exact_prompt_len)
    }

    fn nvext_passthrough_args<R: NvExtProvider>(
        request: &R,
    ) -> Option<serde_json::Map<String, serde_json::Value>> {
        let mut nvext_passthrough = serde_json::Map::new();

        if let Some(nvext) = request.nvext() {
            if let Some(ref fields) = nvext.extra_fields {
                nvext_passthrough.insert("extra_fields".to_string(), serde_json::json!(fields));
            }
            if let Some(ref metadata_upload) = nvext.metadata_upload {
                nvext_passthrough.insert(
                    "metadata_upload".to_string(),
                    serde_json::json!(metadata_upload),
                );
            }
            if nvext.token_data.is_some() {
                nvext_passthrough.insert("token_in".to_string(), serde_json::Value::Bool(true));
            }
        }

        if let Some(salt) = request_cache_salt(request) {
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
            for key in [
                "detokenize",
                "allowed_token_ids",
                "bad_words_token_ids",
                "logprob_token_ids",
            ] {
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

    fn backend_extra_args<R: OAIChatLikeRequest + NvExtProvider>(
        request: &R,
        reasoning_parser_configured: bool,
        reasoning_ended: Option<bool>,
    ) -> Option<serde_json::Value> {
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

        // vLLM constructs some native reasoning parsers per request. Forward
        // the accepted public template arguments through the internal request
        // so its guided-decoding gate observes the same thinking mode used to
        // render the prompt.
        if reasoning_parser_configured
            && let Some(chat_template_args) = request.chat_template_args()
        {
            extra_args.insert(
                "reasoning_parser_kwargs".to_string(),
                serde_json::json!({ "chat_template_kwargs": chat_template_args }),
            );
        }
        if reasoning_parser_configured && let Some(reasoning_ended) = reasoning_ended {
            extra_args.insert("reasoning_ended".to_string(), reasoning_ended.into());
        }

        if extra_args.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(extra_args))
        }
    }

    fn has_request_thinking_control(
        chat_template_args: Option<&HashMap<String, serde_json::Value>>,
    ) -> bool {
        chat_template_args.is_some_and(|args| {
            [
                "thinking",
                "enable_thinking",
                "thinking_mode",
                "reasoning_effort",
            ]
            .iter()
            .any(|key| args.contains_key(*key))
        })
    }

    fn request_has_client_thinking_control(request: &NvCreateChatCompletionRequest) -> bool {
        request.thinking.is_some()
            || Self::has_request_thinking_control(request.chat_template_args.as_ref())
    }

    fn apply_default_thinking_mode_from_runtime_config(
        runtime_config: &crate::local_model::runtime_config::ModelRuntimeConfig,
        request: &mut NvCreateChatCompletionRequest,
    ) {
        if Self::request_has_client_thinking_control(request) {
            return;
        }

        let Some(default_mode) = runtime_config
            .runtime_data
            .get(DEFAULT_THINKING_MODE_RUNTIME_KEY)
            .and_then(|value| value.as_str())
        else {
            return;
        };

        let enabled = match default_mode {
            "enabled" => true,
            "disabled" => false,
            other => {
                tracing::warn!(
                    default_thinking_mode = other,
                    "Ignoring invalid runtime_config default_thinking_mode; expected 'enabled' or 'disabled'"
                );
                return;
            }
        };

        let args = request.chat_template_args.get_or_insert_with(HashMap::new);
        args.insert("thinking".to_string(), serde_json::Value::Bool(enabled));
        args.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(enabled),
        );
        args.insert(
            "thinking_mode".to_string(),
            serde_json::Value::String(if enabled { "enabled" } else { "disabled" }.to_string()),
        );
    }

    fn apply_default_thinking_mode(&self, request: &mut NvCreateChatCompletionRequest) {
        Self::apply_default_thinking_mode_from_runtime_config(&self.runtime_config, request);
    }

    fn guided_output_requires_reasoning<R: OAIChatLikeRequest>(
        request: &R,
        reasoning_parser: Option<&str>,
    ) -> bool {
        if reasoning_parser.is_none() {
            return false;
        }

        let is_guided_tool_choice = Self::has_guided_tool_choice(request);
        let is_structured_response = Self::has_structured_response_format(request);
        let structured_response_requires_reasoning = is_structured_response
            && Self::structured_response_supports_sglang_reasoning_gate(reasoning_parser);

        (is_guided_tool_choice || structured_response_requires_reasoning)
            && Self::sglang_effective_reasoning_enabled(
                reasoning_parser,
                request.chat_template_args(),
            )
    }

    fn structured_response_supports_sglang_reasoning_gate(reasoning_parser: Option<&str>) -> bool {
        // GPT-OSS/Harmony must skip SGLang's `require_reasoning + json_schema`
        // path for structured output until upstream fixes malformed Harmony:
        // https://github.com/sgl-project/sglang/issues/31019
        // Tool calling still uses `require_reasoning`.
        !matches!(reasoning_parser, Some("gpt_oss"))
    }

    fn has_structured_response_format<R: OAIChatLikeRequest>(request: &R) -> bool {
        request.response_format().is_some_and(|format| {
            format
                .get_attr("type")
                .ok()
                .is_some_and(|kind| kind.as_str().is_some_and(|kind| kind != "text"))
        })
    }

    /// Match the rendered prompt's reasoning mode before selecting SGLang NativeGrammar.
    fn sglang_effective_reasoning_enabled(
        reasoning_parser: Option<&str>,
        chat_template_args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> bool {
        let thinking_enabled = dynamo_renderer::thinking_bool_from_args(chat_template_args);

        match reasoning_parser {
            // These SGLang reasoners are enabled unless the request opts out.
            Some("qwen3" | "glm45" | "nemotron_nano" | "nemotron3" | "nemotron_v3") => {
                thinking_enabled != Some(false)
            }
            Some("kimi_k25" | "kimi_k3" | "kimi-k3") => thinking_enabled != Some(false),
            Some("minimax_m2") => {
                Self::deepseek_renderer_reasoning_enabled(chat_template_args, true)
            }

            // DeepSeek V3/V3.1 templates are opt-in. The native V3.2/V4
            // renderers default to thinking; all honor the same aliases.
            Some("deepseek_v3" | "deepseek_v3_1") => {
                Self::deepseek_renderer_reasoning_enabled(chat_template_args, false)
            }
            Some("deepseek_v3_2" | "deepseek_v4" | "deepseek-v4" | "deepseekv4") => {
                Self::deepseek_renderer_reasoning_enabled(chat_template_args, true)
            }
            Some("gemma4" | "gemma-4") => thinking_enabled == Some(true),

            // SGLang's Mistral reasoner is active only for a concrete effort.
            Some("mistral") => Self::mistral_reasoning_enabled(chat_template_args),

            // MiniMax M3 defaults to adaptive reasoning unless disabled.
            Some("minimax_m3" | "minimax-m3") => chat_template_args
                .and_then(|args| args.get("thinking_mode"))
                .and_then(serde_json::Value::as_str)
                .is_none_or(|mode| mode != "disabled"),

            // These SGLang reasoners do not expose a per-request off mode.
            Some("deepseek_r1" | "step3" | "gpt_oss" | "kimi") => true,
            _ => false,
        }
    }

    /// Mirror dynamo-renderer's DeepSeek reasoning-toggle precedence.
    fn deepseek_renderer_reasoning_enabled(
        chat_template_args: Option<&std::collections::HashMap<String, serde_json::Value>>,
        default_enabled: bool,
    ) -> bool {
        if let Some(enabled) = dynamo_renderer::thinking_bool_from_args(chat_template_args) {
            return enabled;
        }
        if let Some(mode) = chat_template_args
            .and_then(|args| args.get("thinking_mode"))
            .and_then(serde_json::Value::as_str)
        {
            return match mode {
                "chat" => false,
                "thinking" => true,
                _ => default_enabled,
            };
        }
        default_enabled
    }

    #[cfg(test)]
    fn normalize_thinking_arg(
        request: &mut NvCreateChatCompletionRequest,
        reasoning_parser: Option<&str>,
        tool_call_parser: Option<&str>,
    ) {
        let thinking_control_from_client = Self::request_has_client_thinking_control(request);
        Self::normalize_thinking_arg_with_source(
            request,
            reasoning_parser,
            tool_call_parser,
            thinking_control_from_client,
        );
    }

    fn normalize_thinking_arg_with_source(
        request: &mut NvCreateChatCompletionRequest,
        reasoning_parser: Option<&str>,
        tool_call_parser: Option<&str>,
        thinking_control_from_client: bool,
    ) {
        let normalized = Self::normalize_thinking_aliases(request, reasoning_parser);

        if Self::is_minimax_m3_family(reasoning_parser, tool_call_parser) {
            Self::normalize_minimax_m3_thinking_mode(
                request,
                normalized,
                thinking_control_from_client,
            );
            return;
        }

        Self::apply_normalized_thinking_aliases(request, normalized);
    }

    fn normalize_thinking_aliases(
        request: &NvCreateChatCompletionRequest,
        reasoning_parser: Option<&str>,
    ) -> Option<bool> {
        request
            .chat_template_args
            .as_ref()
            .and_then(|args| {
                // Normalize public aliases in array order: `thinking`,
                // `enable_thinking`, then `thinking_mode`.
                // Preserve "adaptive" as tri-state by returning None.
                for key in ["thinking", "enable_thinking", "thinking_mode"] {
                    match args.get(key) {
                        Some(serde_json::Value::Bool(b)) => {
                            // `thinking_bool_from_args` ignores `thinking_mode`,
                            // so read that bool directly.
                            if key == "thinking_mode" {
                                return Some(*b);
                            }
                            return dynamo_renderer::thinking_bool_from_args(Some(args));
                        }
                        Some(serde_json::Value::String(value)) => {
                            // Parse `thinking_mode` explicitly so
                            // "enabled"/"disabled" and boolean spellings
                            // behave predictably. Preserve unrecognized strings
                            // such as "adaptive" or DeepSeek's "chat"/"thinking".
                            if key == "thinking_mode" {
                                if value.eq_ignore_ascii_case("enabled") {
                                    return Some(true);
                                }
                                if value.eq_ignore_ascii_case("disabled") {
                                    return Some(false);
                                }
                                if let Some(b) = parse_bool_opt(value) {
                                    return Some(b);
                                }
                                return None;
                            }
                            return Some(is_truthy(value));
                        }
                        Some(serde_json::Value::Number(value)) => {
                            if let Some(value) = value.as_f64() {
                                return Some(value != 0.0);
                            }
                        }
                        _ => {}
                    }
                }
                None
            })
            .or_else(|| {
                matches!(reasoning_parser, Some("kimi_k25" | "kimi_k3" | "kimi-k3")).then_some(true)
            })
    }

    fn normalize_minimax_m3_thinking_mode(
        request: &mut NvCreateChatCompletionRequest,
        normalized: Option<bool>,
        thinking_control_from_client: bool,
    ) {
        // MiniMax M3 defaults `thinking_mode` to "adaptive", which breaks
        // constrained JSON/schema and forced-tool generation. If the client
        // did not set thinking controls, force "disabled" for those requests.
        let explicit_thinking_mode_is_adaptive = request
            .chat_template_args
            .as_ref()
            .and_then(|args| args.get("thinking_mode"))
            .and_then(|v| v.as_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("adaptive"));
        if !thinking_control_from_client && Self::has_minimax_m3_constrained_generation(request) {
            let args = request.chat_template_args.get_or_insert_default();
            args.insert(
                "thinking_mode".to_string(),
                serde_json::Value::String("disabled".to_string()),
            );
            args.insert("thinking".to_string(), serde_json::Value::Bool(false));
            args.insert(
                "enable_thinking".to_string(),
                serde_json::Value::Bool(false),
            );
            return;
        }

        Self::apply_normalized_thinking_aliases(request, normalized);

        let Some(normalized) = normalized else {
            return;
        };

        // Canonicalize for M3 because its template only treats exact
        // "disabled" as off. Preserve explicit "adaptive" as client intent.
        if !explicit_thinking_mode_is_adaptive {
            let args = request.chat_template_args.get_or_insert_default();
            args.insert(
                "thinking_mode".to_string(),
                serde_json::Value::String(
                    if normalized { "enabled" } else { "disabled" }.to_string(),
                ),
            );
        }
    }

    fn apply_normalized_thinking_aliases(
        request: &mut NvCreateChatCompletionRequest,
        normalized: Option<bool>,
    ) {
        let Some(normalized) = normalized else {
            return;
        };
        let args = request.chat_template_args.get_or_insert_default();
        args.insert("thinking".to_string(), serde_json::Value::Bool(normalized));
        args.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(normalized),
        );
    }

    /// True when the deployment renders MiniMax M3's `thinking_mode` template.
    /// Either parser is sufficient, and tool-call aliases include `_nom`.
    fn is_minimax_m3_family(
        reasoning_parser: Option<&str>,
        tool_call_parser: Option<&str>,
    ) -> bool {
        matches!(reasoning_parser, Some("minimax_m3") | Some("minimax-m3"))
            || matches!(
                tool_call_parser,
                Some("minimax_m3")
                    | Some("minimax-m3")
                    | Some("minimax_m3_nom")
                    | Some("minimax-m3-nom")
            )
    }

    fn has_minimax_m3_constrained_generation<R: OAIChatLikeRequest>(request: &R) -> bool {
        Self::has_structured_response_format(request) || Self::has_guided_tool_choice(request)
    }

    /// True when `tool_choice` forces a tool call.
    /// Shared by guided-output reasoning checks and the M3 default.
    fn has_guided_tool_choice<R: OAIChatLikeRequest>(request: &R) -> bool {
        request.tool_choice().is_some_and(|tool_choice| {
            match tool_choice.as_str() {
                Some("required") => true,
                Some(_) => false,
                // The only supported non-string tool choice is a named function.
                None => true,
            }
        })
    }

    /// Apply Moonshot's named-tool exception for Kimi K3.
    ///
    /// The public K3 API treats a specified function as incompatible with
    /// thinking. Normalize that rule onto the request before prompt rendering
    /// so every downstream consumer observes the same effective mode:
    ///
    /// - the K3 renderer omits current and preserved thinking channels,
    /// - prompt-injected reasoning detection stays false,
    /// - backend reasoning-parser kwargs carry the disabled state, and
    /// - response postprocessing does not try to parse a reasoning stream.
    fn normalize_kimi_k3_named_tool_choice(
        request: &mut NvCreateChatCompletionRequest,
        tool_call_parser: Option<&str>,
    ) {
        let uses_kimi_k3_parser =
            tool_call_parser.is_some_and(|parser| matches!(parser, "kimi_k3" | "kimi-k3"));
        let is_named = matches!(
            request.inner.tool_choice.as_ref(),
            Some(ChatCompletionToolChoiceOption::Named(_))
        );
        if !uses_kimi_k3_parser || !is_named {
            return;
        }

        let args = request.chat_template_args.get_or_insert_default();
        args.insert("thinking".to_string(), serde_json::Value::Bool(false));
        args.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
    }

    fn mistral_reasoning_enabled(
        chat_template_args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> bool {
        chat_template_args
            .and_then(|args| args.get("reasoning_effort"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|effort| effort != "none")
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
        let normalize_tool_call_args = mdc.runtime_config.tool_call_arguments_format
            == crate::local_model::runtime_config::ToolCallArgumentsFormat::JsonObject
            || mdc.runtime_config.tool_call_parser.as_deref() == Some("glm47");

        if let Some(ref lora_name) = lora_name {
            tracing::info!(model = %mdc.display_name, lora_name, "LoRA adapter detected in MDC");
        }

        // // Initialize runtime config from the ModelDeploymentCard
        let runtime_config = mdc.runtime_config.clone();
        let token_budget = match runtime_config
            .get_engine_specific::<TokenBudget>(TOKEN_BUDGET_RUNTIME_KEY)
        {
            Ok(token_budget) => token_budget,
            Err(error) => {
                tracing::warn!(
                    %error,
                    key = TOKEN_BUDGET_RUNTIME_KEY,
                    "Ignoring invalid runtime metadata; token overflow handling will be delegated to the backend"
                );
                None
            }
        };
        let kv_cache_block_size = mdc.kv_cache_block_size as usize;

        // Capture MM-routing inputs before mdc is partially moved into MediaLoader.
        // model_type comes from config.json (e.g. "qwen3_vl") and lets the
        // image-processor registry resolve fine-tunes loaded from
        // custom-named directories where the family substring isn't in the path.
        #[cfg(feature = "mm-routing")]
        let model_dir_for_routing: Option<std::path::PathBuf> = mdc_model_dir(&mdc);
        #[cfg(feature = "mm-routing")]
        let fastokens_active = runtime_config.effective_tokenizer_backend().is_fastokens();
        // TODO(mm-routing): fastokens lacks a special-token mutator, so it
        // can't merge tokenizer_config.json specials and would BPE-shatter
        // placeholders (e.g. Qwen2-VL `<|image_pad|>`). Disable MM-routing
        // token resolution here, but keep SMG image-token counting enabled.
        // Remove this split once fastokens upstream exposes the mutator.
        #[cfg(feature = "mm-routing")]
        if fastokens_active && model_dir_for_routing.is_some() {
            tracing::warn!(
                target: "mm_routing",
                "fastokens tokenizer backend is active; MM-aware KV routing disabled. \
                 Image-token metrics remain enabled when SMG supports the model."
            );
        }
        #[cfg(feature = "mm-routing")]
        let image_token_inputs: Option<(String, String, std::path::PathBuf)> = {
            model_dir_for_routing.as_ref().map(|p| {
                (
                    mdc.source_path().to_string(),
                    model_info.model_type(),
                    p.clone(),
                )
            })
        };

        let media_loader = match mdc.media_decoder {
            Some(media_decoder) => Some(MediaLoader::new(media_decoder, mdc.media_fetcher)?),
            None => None,
        };

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
                    let (img_tok, bos_tok_string) = if fastokens_active {
                        (None, None)
                    } else {
                        // One-shot config/tokenizer_config read for all
                        // routing-side token info. Parsing lives next to the
                        // spec resolution in the MM-routing module.
                        let routing_tokens =
                            lightseek_mm::resolve_routing_tokens(&model_id, &model_dir);
                        // `chat_placeholder_token_id` already prefers
                        // config.json's explicit field and falls back to the
                        // spec value, so it is used for both the engagement
                        // gate and the routing fill.
                        (
                            routing_tokens.chat_placeholder_token_id,
                            routing_tokens.bos_token_string,
                        )
                    };

                    match (counter.is_some(), img_tok.is_some()) {
                        (true, true) => tracing::info!(
                            target: "mm_routing",
                            model = %model_id,
                            model_dir = %model_dir.display(),
                            "MM-aware KV routing enabled"
                        ),
                        _ if fastokens_active => {}
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

        // Force the dim-fetch HTTP client to build at startup for any
        // MM-countable or routable preprocessor, so TLS / env-var / reqwest-init
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
            normalize_tool_call_args,
            media_loader,
            token_budget,
            #[cfg(feature = "mm-routing")]
            image_token_counter,
            #[cfg(feature = "mm-routing")]
            routing_image_token_id,
            #[cfg(feature = "mm-routing")]
            routing_prepend_bos,
        }))
    }

    /// Encode a string to it's tokens
    pub fn tokenize(&self, s: &str) -> anyhow::Result<Encoding> {
        self.tokenizer.encode(s)
    }

    /// Encode a rendered prompt while preserving model-specific special-token
    /// boundaries.
    pub fn tokenize_rendered_prompt(&self, prompt: &RenderedPrompt) -> anyhow::Result<Encoding> {
        match prompt.encode_segments() {
            Some(segments) => self.tokenizer.encode_segments(&segments),
            None => self.tokenize(prompt.as_str()),
        }
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
        let (request, annotations, prompt_injected_reasoning, _image_tokens) = self
            .preprocess_request_with_options(
                request,
                tracker,
                PreprocessRequestOptions::default(),
                None,
            )
            .await?;
        Ok((request, annotations, prompt_injected_reasoning))
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
        lora_name: Option<String>,
    ) -> Result<(
        PreprocessedRequest,
        HashMap<String, String>,
        bool,
        Option<usize>,
    )> {
        let _stage_guard = StageGuard::new(STAGE_PREPROCESS, "");
        let preprocess_start = Instant::now();
        let mut builder = self.builder_with_lora(request, lora_name)?;

        let template_start = Instant::now();
        let formatted_prompt = {
            let _nvtx = dynamo_nvtx_range!("preprocess.template");
            self.apply_template(request)
                .with_context(|| "Failed to apply prompt template")?
        };
        TEMPLATE_SECONDS.observe(template_start.elapsed().as_secs_f64());

        // Generic reasoning parsers start from `<think>`; MiniMax M3 starts
        // from `<mm:think>`. If the chat template injected that opener at the
        // end of the prompt, the model completion starts mid-reasoning.
        let prompt_injected_reasoning = Self::prompt_injected_reasoning_start(
            self.runtime_config.reasoning_parser.as_deref(),
            formatted_prompt.as_ref().map(RenderedPrompt::as_str),
        );

        let tokenize_start = Instant::now();
        let (token_ids, annotations) = {
            let _nvtx = dynamo_nvtx_range!("preprocess.tokenize");
            self.gather_tokens(request, formatted_prompt.as_ref(), tracker)
                .await
                .with_context(|| "Failed to gather tokens")?
        };
        TOKENIZE_SECONDS.observe(tokenize_start.elapsed().as_secs_f64());

        let (_mm_image_entries, image_tokens) = self
            .gather_multi_modal_data_with_image_tokens(
                request,
                &mut builder,
                formatted_prompt.as_ref().map(RenderedPrompt::as_str),
                &token_ids,
            )
            .await
            .with_context(|| "Failed to gather multimodal data")?;

        // Build the MM-aware view (expanded routing_token_ids + per-block
        // mm_hashes) for the KV router. No-op when no images are present or
        // the model has no resolved image-placeholder.
        #[cfg(feature = "mm-routing")]
        self.gather_mm_exact_routing_info(&mut builder, &_mm_image_entries, &token_ids)
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
        if let Some(reasoning_ended) = Self::prompt_injected_reasoning_ended_arg(
            self.runtime_config.reasoning_parser.as_deref(),
            formatted_prompt.as_ref().map(RenderedPrompt::as_str),
        ) {
            let extra_args = preprocessed
                .extra_args
                .get_or_insert_with(|| serde_json::json!({}));
            let extra_args = extra_args
                .as_object_mut()
                .context("preprocessed extra_args must be an object")?;
            extra_args.insert("reasoning_ended".to_string(), reasoning_ended.into());
        }

        // If omitted, allow generation up to the remaining context length. Responses requests
        // preserve omission so backend adapters can compute the dynamic cap from their
        // effective prompt length/tokenization.
        //
        // Multimodal `token_ids` carry unexpanded image placeholders, so prefer
        // the MM-expanded length when available, else defer to the backend.
        let exact_prompt_len =
            Self::validate_preprocessed_token_budget(&preprocessed, self.token_budget.as_ref())?;
        if preprocessed.stop_conditions.max_tokens.is_none()
            && let Some(prompt_len) = exact_prompt_len
            // Preserve omission when the prompt itself is backend-owned. In
            // particular, setting a saturating zero here would interfere with
            // SGLang's auto-truncation before the backend sees the request.
            && !self.token_budget.is_some_and(|token_budget| {
                prompt_len >= token_budget.combined_limit as usize
                    && !token_budget.reject_prompt_overflow
            })
            && let Some(max_tokens) = Self::omitted_max_tokens_default(
                prompt_len,
                self.token_budget
                    .map(|token_budget| token_budget.combined_limit),
                options,
            )
        {
            preprocessed.stop_conditions.max_tokens = Some(max_tokens);
            // A strict limit may be smaller than the raw context window because
            // the engine reserves tokens. Revalidate the derived budget so an
            // already-over-limit prompt fails before streaming begins.
            Self::validate_preprocessed_token_budget(&preprocessed, self.token_budget.as_ref())?;
        }

        Ok((
            preprocessed,
            annotations,
            prompt_injected_reasoning,
            image_tokens,
        ))
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
        self.builder_with_lora(request, None)
    }

    fn builder_with_lora<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
        lora_name_override: Option<String>,
    ) -> Result<PreprocessedRequestBuilder> {
        let mut builder = PreprocessedRequest::builder();
        builder.model(request.model());

        let mut stop_conditions = request.extract_stop_conditions()?;
        let eos_token_ids = self.model_info.eos_token_ids();
        let hidden_eos_token_ids = eos_token_ids.clone();
        if let Some(stop_tokens) = &mut stop_conditions.stop_token_ids_hidden {
            for eos_token_id in hidden_eos_token_ids {
                if !stop_tokens.contains(&eos_token_id) {
                    stop_tokens.push(eos_token_id);
                }
            }
        } else {
            stop_conditions.stop_token_ids_hidden = Some(hidden_eos_token_ids);
        }
        // Some tool-call parsers terminate on a token that is also a model
        // EOS (e.g. Harmony's `<|call|>` for gpt-oss). Left in the hidden
        // set, the engine stops AND strips it, so the parser sees a
        // truncated envelope and drops the call. Move such tokens to the
        // visible set so the engine still stops on them but the token
        // survives into output for the parser to consume. See PR #9778.
        let mut visible_tool_parser_end_token_ids = Vec::new();
        if let Some(stop_tokens) = &mut stop_conditions.stop_token_ids_hidden {
            visible_tool_parser_end_token_ids =
                self.remove_tool_parser_end_tokens_from_hidden_stops(request, stop_tokens)?;
        }
        if !visible_tool_parser_end_token_ids.is_empty() {
            let visible_stops = stop_conditions
                .stop_token_ids_visible
                .get_or_insert_with(Vec::new);
            for token_id in visible_tool_parser_end_token_ids {
                if !visible_stops.contains(&token_id) {
                    visible_stops.push(token_id);
                }
            }
        }

        // apply ignore eos if not already set
        stop_conditions.apply_ignore_eos();

        if !stop_conditions.ignore_eos.unwrap_or(false) {
            builder.eos_token_ids(eos_token_ids);
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
        let lora_name = self.lora_name.clone().or(lora_name_override);
        let cache_namespace = request_cache_salt(request).map(str::to_owned);

        // Extract routing hints from nvext if present
        if let Some(nvext) = request.nvext() {
            // Build routing hints from nvext fields
            let hints = nvext.agent_hints.as_ref();
            let (priority_jump, strict_priority, priority) = routing_priorities(hints);
            builder.request_timestamp_ms(nvext.request_timestamp_ms);
            let routing = RoutingHints {
                backend_instance_id: nvext.backend_instance_id,
                prefill_worker_id: nvext.prefill_worker_id,
                decode_worker_id: nvext.decode_worker_id,
                dp_rank: nvext.dp_rank,
                prefill_dp_rank: nvext.prefill_dp_rank,
                expected_output_tokens: hints.and_then(|h| h.osl),
                priority_jump,
                strict_priority,
                priority,
                lora_name,
                cache_namespace: cache_namespace.clone(),
                allowed_worker_ids: None,
                routing_constraints: nvext
                    .routing_constraints
                    .clone()
                    .map(routing_constraints_to_kv),
            };
            builder.routing(Some(routing));
        } else if lora_name.is_some() || cache_namespace.is_some() {
            // Ensure routing hints exist when we have LoRA or a legacy
            // top-level cache_salt, even when nvext is absent.
            builder.routing(Some(RoutingHints {
                lora_name,
                cache_namespace,
                ..Default::default()
            }));
        }

        if let Some(extra_args) = Self::backend_extra_args(
            request,
            self.runtime_config.reasoning_parser.is_some(),
            None,
        ) {
            builder.extra_args(Some(extra_args));
        }

        // SGLang needs this request-scoped signal in addition to its native
        // reasoning parser so guided JSON starts after the reasoning boundary.
        builder.require_reasoning(Self::guided_output_requires_reasoning(
            request,
            self.runtime_config.reasoning_parser.as_deref(),
        ));

        // Forward mm_processor_kwargs (e.g. use_audio_in_video) to the backend.
        builder.mm_processor_kwargs(request.mm_processor_kwargs().cloned());

        Ok(builder)
    }

    fn remove_tool_parser_end_tokens_from_hidden_stops<R: OAIChatLikeRequest>(
        &self,
        request: &R,
        hidden_stop_token_ids: &mut Vec<TokenIdType>,
    ) -> Result<Vec<TokenIdType>> {
        let has_tools = request
            .tools()
            .as_ref()
            .and_then(|tools| tools.len())
            .is_some_and(|len| len > 0);
        let tool_choice_none = request
            .tool_choice()
            .as_ref()
            .and_then(|tool_choice| tool_choice.as_str())
            == Some("none");

        if !Self::should_keep_tool_parser_end_tokens_visible(has_tools, tool_choice_none) {
            return Ok(Vec::new());
        }

        let Some(tool_call_parser) = self.tool_call_parser.as_deref().filter(|p| !p.is_empty())
        else {
            return Ok(Vec::new());
        };
        let Some(tool_call_config) = get_tool_parser_map().get(tool_call_parser) else {
            return Ok(Vec::new());
        };

        let mut visible_stop_token_ids = Vec::new();
        for end_token in tool_call_config.parser_config.tool_call_end_tokens() {
            if end_token.is_empty() {
                continue;
            }
            let encoded = self.tokenizer.encode(&end_token).with_context(|| {
                format!(
                    "Failed to encode tool-call end token {end_token:?} for parser {tool_call_parser:?}"
                )
            })?;
            let was_hidden_eos =
                Self::remove_single_token_marker(hidden_stop_token_ids, encoded.token_ids());
            if !was_hidden_eos {
                tracing::debug!(
                    token_ids = ?encoded.token_ids(),
                    end_token,
                    parser = tool_call_parser,
                    "Tool-call end token was not a single hidden EOS token"
                );
                continue;
            }
            if let [token_id] = encoded.token_ids()
                && !visible_stop_token_ids.contains(token_id)
            {
                visible_stop_token_ids.push(*token_id);
            }
        }

        Ok(visible_stop_token_ids)
    }

    fn should_keep_tool_parser_end_tokens_visible(has_tools: bool, tool_choice_none: bool) -> bool {
        has_tools && !tool_choice_none
    }

    fn remove_single_token_marker(
        hidden_eos_token_ids: &mut Vec<TokenIdType>,
        marker_token_ids: &[TokenIdType],
    ) -> bool {
        let [marker_token_id] = marker_token_ids else {
            return false;
        };
        let before = hidden_eos_token_ids.len();
        hidden_eos_token_ids.retain(|token_id| token_id != marker_token_id);
        hidden_eos_token_ids.len() != before
    }

    /// Rendering is driven by the request, so its failures are reported as 400 rather than
    /// 500, matching vLLM. A misconfigured template can also fail here, for instance a
    /// `chat_template` map that omits the `tool_use` key, so log the cause chain before it
    /// is flattened into the client-facing message.
    fn map_prompt_render_error(error: anyhow::Error) -> anyhow::Error {
        tracing::debug!(?error, "Chat template rendering failed");
        let message = match error.downcast_ref::<PromptRenderError>() {
            Some(PromptRenderError::InvalidRequest(message)) => message.clone(),
            None => format!("{error:#}"),
        };
        invalid_argument_error(message)
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
    ) -> Result<Option<RenderedPrompt>> {
        if self.normalize_tool_call_args {
            return self.apply_template_inner(&NormalizedArgsRequest(request));
        }
        self.apply_template_inner(request)
    }

    fn apply_template_inner<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
    ) -> Result<Option<RenderedPrompt>> {
        if let PromptInput::Text(_) = request.prompt_input_type()
            && let Some(TextInput::Single(_)) = request.extract_text()
        {
            let use_raw_prompt = request
                .nvext()
                .is_some_and(|ext| ext.use_raw_prompt.unwrap_or(false));

            let formatted_prompt = if use_raw_prompt {
                match request.raw_prompt() {
                    Some(prompt) => RenderedPrompt::text(prompt),
                    None => {
                        tracing::warn!("Raw prompt requested but not available");
                        self.formatter
                            .render_prompt(request)
                            .map_err(Self::map_prompt_render_error)?
                    }
                }
            } else {
                self.formatter
                    .render_prompt(request)
                    .map_err(Self::map_prompt_render_error)?
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

    fn replace_reserved_media_slot(
        media_map: &mut MultimodalDataMap,
        modality: &str,
        slot_idx: usize,
        value: MultimodalData,
    ) -> Result<()> {
        let slot = media_map
            .get_mut(modality)
            .and_then(|slots| slots.get_mut(slot_idx))
            .with_context(|| {
                format!(
                    "missing reserved multimodal slot {modality}[{slot_idx}] during media decode"
                )
            })?;
        *slot = value;
        Ok(())
    }

    pub async fn gather_multi_modal_data<
        R: OAIChatLikeRequest + MediaRequestExt + NvExtProvider,
    >(
        &self,
        request: &R,
        builder: &mut PreprocessedRequestBuilder,
        formatted_prompt: Option<&str>,
        // Worker-bound token ids; used (mm-routing only) to gate `mm_hashes`
        // forwarding on the same placeholder-count precondition as
        // `gather_mm_exact_routing_info`, so the two never diverge.
        token_ids: &[crate::protocols::TokenIdType],
    ) -> Result<Vec<MmImageEntry>> {
        let (entries, _image_tokens) = self
            .gather_multi_modal_data_with_image_tokens(
                request,
                builder,
                formatted_prompt,
                token_ids,
            )
            .await?;
        Ok(entries)
    }

    async fn gather_multi_modal_data_with_image_tokens<
        R: OAIChatLikeRequest + MediaRequestExt + NvExtProvider,
    >(
        &self,
        request: &R,
        builder: &mut PreprocessedRequestBuilder,
        formatted_prompt: Option<&str>,
        token_ids: &[crate::protocols::TokenIdType],
    ) -> Result<(Vec<MmImageEntry>, Option<usize>)> {
        // `token_ids` is only consumed by the mm-routing `mm_hashes` gate below.
        #[cfg(not(feature = "mm-routing"))]
        let _ = token_ids;

        let mut media_map: MultimodalDataMap = HashMap::new();
        let mut uuid_map: MultimodalUuidMap = HashMap::new();
        let mut has_user_uuid = false;
        // Decoded results are written back into these reserved modality slots so
        // URL-backed and UUID-only inputs retain request order.
        let mut fetch_tasks: Vec<MediaFetchTask<'_>> = Vec::new();
        // Per-image (mm_hash, width, height) for the MM-routing path.
        // Accumulated in message order so we don't walk messages twice.
        // Cleared and returned to the caller; empty for non-image / text-only requests.
        #[cfg(feature = "mm-routing")]
        let mut mm_image_entries: Vec<MmImageEntry> = Vec::new();
        // Private per-request total for frontend metrics. `None` means the SMG
        // counter is unavailable or checked addition overflowed.
        #[cfg(feature = "mm-routing")]
        let mut image_tokens = self.image_token_counter.as_ref().map(|_| 0usize);
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
            return Ok((Vec::new(), None));
        };
        let has_media_loader = self.media_loader.is_some();

        for message in messages.iter() {
            let Some(content_parts) = multimodal_content_parts(message) else {
                continue;
            };
            for content_part in content_parts {
                let Some((type_str, url, uuid)) = content_part.media_info() else {
                    continue;
                };

                if type_str != "image_url" && uuid.is_some() {
                    return Err(invalid_argument_error(
                        "multimodal cache UUIDs are supported only for image_url parts with vLLM",
                    ));
                }

                #[cfg(feature = "mm-routing")]
                if type_str == "image_url" {
                    total_image_count += 1;
                }

                if uuid.as_deref().is_some_and(str::is_empty) {
                    return Err(invalid_argument_error(format!(
                        "{type_str} uuid must be a non-empty string"
                    )));
                }

                let slots = media_map.entry(type_str.to_string()).or_default();
                let slot_idx = slots.len();
                has_user_uuid |= uuid.is_some();
                if type_str == "image_url" {
                    uuid_map
                        .entry(type_str.to_string())
                        .or_default()
                        .push(uuid.clone());
                }

                match (url, uuid) {
                    (Some(url), _) => {
                        if has_media_loader {
                            fetch_tasks.push(MediaFetchTask {
                                modality: type_str,
                                slot_idx,
                                content_part: content_part.as_user(),
                            });
                        } else {
                            #[cfg(feature = "mm-routing")]
                            if type_str == "image_url" {
                                let mm_hash = Self::hash_image_url(url.as_str());
                                url_passthrough_images.push((mm_hash, url.as_str().to_string()));
                            }
                        }
                        slots.push(MultimodalData::Url(url));
                    }
                    (None, Some(uuid)) => {
                        slots.push(MultimodalData::UuidOnly(uuid));
                    }
                    (None, None) => {
                        return Err(invalid_argument_error(format!(
                            "{type_str} part has neither `url` nor `uuid`; at least one is required"
                        )));
                    }
                }
            }
        }

        // Execute all fetch tasks
        if !fetch_tasks.is_empty() {
            let loader = self.media_loader.as_ref().unwrap();
            let media_io_kwargs = request.media_io_kwargs();
            let results = futures::future::join_all(fetch_tasks.iter().map(|task| {
                loader.fetch_and_decode_media_part(task.content_part.as_ref(), media_io_kwargs)
            }))
            .await;

            for (task, result) in fetch_tasks.into_iter().zip(results) {
                // if one item fails, errors the whole request, other items will be cleaned up by Drop
                let rdma_descriptor = result?;

                // Decoded RDMA descriptor carries shape `[H, W, C]`.
                // Image-only; MM-routing doesn't cover audio/video.
                #[cfg(feature = "mm-routing")]
                if task.modality == "image_url" {
                    let shape = &rdma_descriptor.tensor_info.shape;
                    if shape.len() >= 2 {
                        let h = shape[0] as u32;
                        let w = shape[1] as u32;
                        // Frontend-decode path: hash the decoded RGB bytes so
                        // the same image reached via different (signed) URLs
                        // collides on the same `mm_hash` and routes to the
                        // worker that already has those KV blocks. Fall back
                        // to URL hashing only if the descriptor lost local
                        // storage (e.g. reconstructed from the wire), which
                        // shouldn't happen on the frontend.
                        let (mm_hash, hash_source) = match rdma_descriptor.content_hash() {
                            Some(h) => (h, "decoded_bytes"),
                            None => {
                                let source_url = image_content_part_url(task.content_part.as_ref())
                                    .ok_or_else(|| {
                                        invalid_argument_error(
                                            "image_url task must contain an image URL",
                                        )
                                    })?;
                                (Self::hash_image_url(source_url), "url_fallback")
                            }
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
                            image_tokens = checked_add_image_tokens(image_tokens, n);
                        }
                        mm_image_entries.push(MmImageEntry {
                            mm_hash,
                            width: w,
                            height: h,
                        });
                    }
                }

                Self::replace_reserved_media_slot(
                    &mut media_map,
                    task.modality,
                    task.slot_idx,
                    MultimodalData::Decoded(rdma_descriptor),
                )?;
            }
        }

        // URL-passthrough path (media_loader is None): fetch image headers in
        // parallel to get (W, H) per image without downloading the full bytes.
        // Enables MM-aware routing for backends that register
        // `media_decoder: null` and decode images on the worker.
        #[cfg(feature = "mm-routing")]
        if !has_user_uuid && !url_passthrough_images.is_empty() {
            let dim_results = futures::future::join_all(
                url_passthrough_images
                    .iter()
                    .map(|(mm_hash, url)| Self::fetch_image_dims(*mm_hash, url)),
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
                            image_tokens = checked_add_image_tokens(image_tokens, n);
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
            if has_user_uuid {
                builder.multi_modal_uuids(Some(uuid_map));
            }

            // User cache identities are opaque and cannot be converted into the
            // router's canonical image hashes. Fall back to text-prefix routing
            // instead of routing and publishing under different cache keys.
            #[cfg(feature = "mm-routing")]
            if has_user_uuid {
                mm_image_entries.clear();
            }

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

            if let Some(serde_json::Value::Object(backend_extra_args)) = Self::backend_extra_args(
                request,
                self.runtime_config.reasoning_parser.is_some(),
                Self::prompt_injected_reasoning_ended_arg(
                    self.runtime_config.reasoning_parser.as_deref(),
                    formatted_prompt,
                ),
            ) {
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
            //
            // Also gate on the single-token placeholder count matching the
            // image count — the same precondition `gather_mm_exact_routing_info`
            // uses to build routing info. Forwarding `mm_hashes` makes the
            // worker pad_value-key its KV blocks; if the router then falls back
            // to plain `token_ids` (e.g. numbered-placeholder models like Phi-3,
            // where the count won't match), the keys diverge and overlap is
            // lost. Keeping both gated together leaves that fallback as clean
            // text-prefix routing.
            #[cfg(feature = "mm-routing")]
            if let Some(find_token_id) = self.routing_image_token_id
                && !has_user_uuid
                && !mm_image_entries.is_empty()
                && mm_image_entries.len() == total_image_count
                && token_ids.iter().filter(|&&t| t == find_token_id).count()
                    == mm_image_entries.len()
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
                    "mm-routing: exact MM routing info not built (dim resolution or placeholder-count mismatch); skipping mm_hashes forwarding"
                );
            }

            builder.extra_args(Some(extra_args));
        }

        #[cfg(feature = "mm-routing")]
        let has_processor_override =
            has_image_token_processor_override(request.mm_processor_kwargs());
        #[cfg(feature = "mm-routing")]
        let image_tokens = aggregate_image_tokens(
            image_tokens,
            mm_image_entries.len(),
            total_image_count,
            has_processor_override,
        );
        #[cfg(feature = "mm-routing")]
        return Ok((mm_image_entries, image_tokens));
        #[cfg(not(feature = "mm-routing"))]
        Ok((Vec::new(), None))
    }

    /// Build `MmRoutingInfo` for exact MM-aware KV routing. The worker-bound
    /// `token_ids` are unchanged — only the routing-side view is expanded.
    /// Supports single-special-token placeholder families (Qwen-VL, LLaVA).
    /// Returns `Ok(())` with no work performed on any precondition miss
    /// (caller falls back to text-prefix routing).
    #[cfg(feature = "mm-routing")]
    pub fn gather_mm_exact_routing_info(
        &self,
        builder: &mut PreprocessedRequestBuilder,
        mm_image_entries: &[MmImageEntry],
        token_ids: &[crate::protocols::TokenIdType],
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
        // `<image>`) emit exactly one `find_token_id` per image in the
        // tokenized prompt. Any other shape (e.g. numbered-text placeholders
        // that BPE-shatter) can't be aligned to images here, so we skip MM
        // routing and let the caller fall back to text-prefix routing.
        let placeholder_count = token_ids.iter().filter(|&&t| t == find_token_id).count();
        if placeholder_count != mm_image_entries.len() {
            tracing::warn!(
                target: "mm_routing",
                placeholder_count,
                image_count = mm_image_entries.len(),
                routing_image_token_id = find_token_id,
                "placeholder token count in tokenized prompt does not match image count; \
                 skipping MM routing info (text-prefix routing only)"
            );
            return Ok(());
        }
        let normalized_token_ids = token_ids;

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
        // Prepend the routing-side BOS for `add_bos_token: true` models
        // (LlamaTokenizer family, e.g. LLaVA-1.5) so per-block hashes match
        // the backend's HF processor output.
        let bos_extra = self.routing_prepend_bos.is_some() as usize;
        let mut expanded: Vec<crate::protocols::TokenIdType> =
            Vec::with_capacity(normalized_token_ids.len() + n_total + bos_extra);
        if let Some(bos) = self.routing_prepend_bos {
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

        // Unpadded expanded length, before the block-padding added below.
        let expanded_prompt_len = expanded.len();

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
            expanded_prompt_len,
        }));
        Ok(())
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
                .time_to_live(std::time::Duration::from_secs(24 * 60 * 60))
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
        const DIM_FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

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
    pub async fn gather_tokens<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
        formatted_prompt: Option<&RenderedPrompt>,
        tracker: Option<&RequestTracker>,
    ) -> Result<(Vec<crate::protocols::TokenIdType>, HashMap<String, String>)> {
        let mut annotations = HashMap::new();
        let mut tokens_out: Vec<crate::protocols::TokenIdType> = Vec::new();
        // match request type before any conversion/processing
        match request.prompt_input_type() {
            PromptInput::Tokens(_) => {
                if let Some(token_input) = request.extract_tokens() {
                    match token_input {
                        TokenInput::Single(tokens) => {
                            tokens_out = tokens;
                        }
                        TokenInput::Batch(token_batches) => {
                            if token_batches.len() == 1 {
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
                                annotations.insert(
                                    ANNOTATION_FORMATTED_PROMPT.to_string(),
                                    f.as_str().to_string(),
                                );
                            }

                            // Completions will use raw_prompt, no template.
                            // K3 keeps control-token spans distinct from user text until
                            // tokenization; all other renderers use the plain text path.

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
                                let encoding = self
                                    .encode_prompt_with_timing(
                                        formatted_prompt,
                                        raw_prompt.as_str(),
                                        tracker,
                                    )
                                    .await?;
                                (encoding.token_ids().to_vec(), false)
                            } else {
                                let encoding = self
                                    .encode_prompt_with_timing(
                                        formatted_prompt,
                                        raw_prompt.as_str(),
                                        tracker,
                                    )
                                    .await?;
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

                            tokens_out = tokens_vec;
                        }
                        TextInput::Batch(texts) => {
                            if texts.len() == 1 {
                                let encoding = self.encode_with_timing(&texts[0], tracker).await?;
                                let tokens = encoding.token_ids().to_vec();
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

        Ok((tokens_out, annotations))
    }

    fn prompt_overflow_error(token_count: usize, combined_limit: usize) -> DynamoError {
        DynamoError::builder()
            .error_type(ErrorType::InvalidArgument)
            .message(format!(
                "This model's maximum context length is {} tokens. \
                 However, your messages resulted in {} tokens. \
                 Please reduce the length of the messages.",
                combined_limit, token_count,
            ))
            .build()
    }

    async fn encode_with_timing(
        &self,
        prompt: &str,
        tracker: Option<&RequestTracker>,
    ) -> anyhow::Result<Encoding> {
        let encode_start = Instant::now();
        // Offload the CPU-heavy BPE encode to the bounded blocking pool instead of running it on
        // the async event loop. For long prompts at high concurrency, a synchronous encode here
        // stalls the frontend tokio runtime for seconds, starving the I/O tasks that share the
        // runtime. Own the prompt + clone the tokenizer (Arc) so the closure is 'static + Send;
        // mirrors the embedding path's spawn_blocking offload.
        let owned = if prompt.contains('\0') {
            tracing::debug!("Prompt contains null bytes; stripping to avoid tokenizer divergence");
            prompt.replace('\0', "")
        } else {
            prompt.to_string()
        };
        let tokenizer = self.tokenizer.clone();
        let encoding = tokio::task::spawn_blocking(move || tokenizer.encode(&owned)).await??;
        if let Some(t) = tracker {
            t.record_tokenize_latency(encode_start.elapsed());
        }
        Ok(encoding)
    }

    async fn encode_prompt_with_timing(
        &self,
        formatted_prompt: Option<&RenderedPrompt>,
        raw_prompt: &str,
        tracker: Option<&RequestTracker>,
    ) -> anyhow::Result<Encoding> {
        let Some(prompt) = formatted_prompt
            .filter(|prompt| prompt.segments().is_some())
            .cloned()
        else {
            return self
                .encode_with_timing(
                    formatted_prompt
                        .map(RenderedPrompt::as_str)
                        .unwrap_or(raw_prompt),
                    tracker,
                )
                .await;
        };

        let encode_start = Instant::now();
        let tokenizer = self.tokenizer.clone();
        let encoding = tokio::task::spawn_blocking(move || {
            let segments = prompt
                .encode_segments()
                .expect("prompt was checked for rendered segments");
            tokenizer.encode_segments(&segments)
        })
        .await??;
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
        // Guided output may be bare JSON or `reasoning</think>JSON`. Supported
        // parsers inspect the stream shape before deciding whether to parse it.
        let is_guided_tool_choice = matches!(
            request.inner.tool_choice,
            Some(ChatCompletionToolChoiceOption::Required)
                | Some(ChatCompletionToolChoiceOption::Named(_))
        );
        let is_structured_response = Self::has_structured_response_format(request);
        let is_guided_output = is_guided_tool_choice || is_structured_response;
        let reasoning_parser = self.runtime_config.reasoning_parser.as_deref();
        // Force parsers opt in by capability; prompt-seeded parsers opt in per request.
        // Structural-tag tool formats do not use this guided-JSON detection path.
        let inspect_force_reasoning_guided_output = is_guided_output
            && !uses_tool_call_structural_tag
            && Self::supports_reasoning_before_guided_json(reasoning_parser);
        let inspect_prompt_injected_guided_output = is_guided_output
            && prompt_injected_reasoning
            && !uses_tool_call_structural_tag
            && (Self::skips_guided_json_when_prompt_injected(reasoning_parser)
                || (is_structured_response
                    && Self::skips_structured_response_when_prompt_injected(reasoning_parser)));
        let inspect_unsupported_structured_response_reasoning_gate = is_structured_response
            && !uses_tool_call_structural_tag
            && !Self::structured_response_supports_sglang_reasoning_gate(reasoning_parser);
        let bypass_reasoning_for_bare_guided_json = inspect_force_reasoning_guided_output
            || inspect_prompt_injected_guided_output
            || inspect_unsupported_structured_response_reasoning_gate;
        // Preserve the legacy bypass for force-reasoning parsers not yet opted in.
        let skip_reasoning_for_guided_json = is_guided_output
            && !uses_tool_call_structural_tag
            && Self::is_force_reasoning_parser(reasoning_parser)
            && !inspect_force_reasoning_guided_output;

        let reasoning_disabled_by_request = Self::is_reasoning_disabled_by_request(
            self.runtime_config.reasoning_parser.as_deref(),
            request.chat_template_args.as_ref(),
        );

        // Try to parse reasoning content only if parser is configured.
        let should_parse_reasoning = self.runtime_config.reasoning_parser.is_some()
            && !reasoning_disabled_by_request
            && !skip_reasoning_for_guided_json;
        let should_strip_disabled_reasoning_start = reasoning_disabled_by_request
            && Self::is_nemotron_force_reasoning(self.runtime_config.reasoning_parser.as_deref())
            && !skip_reasoning_for_guided_json;
        let guided_reasoning_start_token =
            if should_parse_reasoning && bypass_reasoning_for_bare_guided_json {
                Self::guided_json_reasoning_start_token(reasoning_parser)
            } else {
                None
            };

        // Reasoning Content Parsing Transformation Step
        // Current Solution:
        // This step operates on Deltas created by the transform_postprocessor_stream function
        // Only access to text and not token_ids - so can not support parsing based on token_ids for now
        // Future Solution:
        // To address the limitation if needed in future: move this step before transform_postprocessor_stream and add new field of reasoning_content to the backend output
        // Use backend_output.reasoning_content field to fill out the deltas.
        // A bracket-prefixed reasoning opener is indistinguishable from JSON
        // until enough bytes arrive. Strip only a configured complete marker;
        // nonmatching bare JSON is restored unchanged for shape detection.
        let stream: Pin<Box<dyn Stream<Item = _> + Send>> =
            if let Some(start_token) = guided_reasoning_start_token {
                Box::pin(Self::strip_leading_reasoning_start_from_stream(
                    stream,
                    start_token,
                ))
            } else {
                Box::pin(stream)
            };

        // Only a force_nonempty_content request needs the deferral and the EOF
        // flush of a truncated `<think>` prefix; gating on the request keeps the
        // per-token clone and the finish_reasoning_stream() flush off every
        // other request's path. Same predicate the aggregator uses, so the
        // streaming and non-streaming paths cannot disagree.
        let defer_reasoning_for_nonempty_content =
            Self::wants_reasoning_as_content_when_empty(request.chat_template_args.as_ref());
        let stream: Pin<Box<dyn Stream<Item = _> + Send>> = if should_parse_reasoning {
            Box::pin(Self::parse_reasoning_content_from_stream_inner(
                stream,
                self.runtime_config.reasoning_parser.clone().unwrap(), // Safety: We already checked that parser is some, so gtg
                prompt_injected_reasoning,
                bypass_reasoning_for_bare_guided_json,
                defer_reasoning_for_nonempty_content,
            ))
        } else if should_strip_disabled_reasoning_start {
            Box::pin(Self::strip_leading_reasoning_start_from_stream(
                stream, "<think>",
            ))
        } else {
            Box::pin(stream)
        };
        let stream: Pin<Box<dyn Stream<Item = _> + Send>> =
            if defer_reasoning_for_nonempty_content || should_strip_disabled_reasoning_start {
                Box::pin(Self::hold_usage_until_stream_end(stream))
            } else {
                stream
            };

        // Check if tools are present and if we should apply jail
        let has_tools = request
            .inner
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty());

        // K3's reasoning-only path still emits XTML response/message wrappers.
        // vLLM strips those in its K3 reasoner when no tool parser is active;
        // reuse the Rust K3 tool parser as the wrapper decoder so configuring
        // only `--dyn-reasoning-parser kimi_k3` remains safe too.
        let effective_tool_call_parser = self.tool_call_parser.clone().or_else(|| {
            self.runtime_config
                .reasoning_parser
                .as_deref()
                .filter(|parser| matches!(*parser, "kimi_k3" | "kimi-k3"))
                .map(str::to_string)
        });

        // A parser describes model syntax, but request semantics decide whether
        // tool calls are allowed. Keep K3's wrapper decoder active because that
        // model wraps ordinary assistant content in XTML even without tools.
        let parser_unwraps_all_kimi_k3_responses = effective_tool_call_parser
            .as_deref()
            .is_some_and(|parser| matches!(parser, "kimi_k3" | "kimi-k3"));
        let tool_call_parsing_enabled = Self::tool_call_parsing_enabled(request);
        let should_jail = if tool_call_parsing_enabled || parser_unwraps_all_kimi_k3_responses {
            Self::should_apply_tool_jail(
                effective_tool_call_parser.as_ref(),
                request.inner.tool_choice.as_ref(),
                has_tools,
            )?
        } else {
            false
        };

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

        // When DYN_ENABLE_EXPERIMENTAL_PARSERS_V2 is set, supported families
        // (Qwen3-Coder, DeepSeek-V4) stream through the dynamo-parsers-v2 parser
        // instead of the jail, which is never built for them on this path.
        // tool_choice=required/named and structural-tag still use the jail's
        // Immediate mode, since those rely on guided-decoded JSON rather than the
        // native markup the v2 parser reads. See tool_parser_v2::apply_stream.
        use crate::protocols::openai::chat_completions::tool_parser_v2;
        let parser_name = effective_tool_call_parser.as_deref();
        let use_parsers_v2 = tool_parser_v2::enabled()
            && parser_name.is_some_and(tool_parser_v2::supports_family)
            && !uses_tool_call_structural_tag
            && matches!(
                request.inner.tool_choice.as_ref(),
                None | Some(dynamo_protocols::types::ChatCompletionToolChoiceOption::Auto)
            );

        // Apply jail conditionally
        let transformed_stream: Pin<Box<dyn Stream<Item = _> + Send>> =
            if should_jail && use_parsers_v2 {
                Box::pin(tool_parser_v2::apply_stream(
                    stream,
                    tool_definitions,
                    parser_name
                        .expect("use_parsers_v2 implies a parser name")
                        .to_string(),
                ))
            } else if should_jail {
                Box::pin(Self::apply_tool_calling_jail(
                    effective_tool_call_parser,
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

    /// Ensure the first emitted delta for each choice carries the assistant role.
    ///
    /// This runs after reasoning and tool-call parsing so a parser that buffers the
    /// original role-bearing delta cannot leave downstream consumers without a role.
    fn normalize_chat_stream_roles<S>(
        stream: S,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        let mut role_emitted_choices = HashSet::new();

        stream.map(move |mut response| {
            if let Some(data) = response.data.as_mut() {
                for choice in &mut data.inner.choices {
                    choice.delta.role = role_emitted_choices
                        .insert(choice.index)
                        .then_some(dynamo_protocols::types::Role::Assistant);
                }
            }
            response
        })
    }

    pub fn transform_postprocessor_stream<S, Resp>(
        stream: S,
        generator: Box<dyn DeltaGeneratorExt<Resp>>,
        context: Arc<dyn AsyncEngineContext>,
        emit_payload_usage_chunk: bool,
        trace_tokens_enabled: bool,
        trace_finish_reason_metadata: Option<crate::request_trace::SharedFinishReasonMetadata>,
        mm_counts: MultimodalCounts,
    ) -> impl Stream<Item = Annotated<Resp>> + Send
    where
        S: Stream<Item = Annotated<BackendOutput>> + Send + 'static,
        Resp: Send + Sync + Clone + 'static + std::fmt::Debug,
    {
        Self::transform_postprocessor_stream_with_image_tokens(
            stream,
            generator,
            context,
            emit_payload_usage_chunk,
            trace_tokens_enabled,
            trace_finish_reason_metadata,
            mm_counts,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_postprocessor_stream_with_image_tokens<S, Resp>(
        stream: S,
        generator: Box<dyn DeltaGeneratorExt<Resp>>,
        context: Arc<dyn AsyncEngineContext>,
        emit_payload_usage_chunk: bool,
        trace_tokens_enabled: bool,
        trace_finish_reason_metadata: Option<crate::request_trace::SharedFinishReasonMetadata>,
        mm_counts: MultimodalCounts,
        image_tokens: Option<usize>,
    ) -> impl Stream<Item = Annotated<Resp>> + Send
    where
        S: Stream<Item = Annotated<BackendOutput>> + Send + 'static,
        Resp: Send + Sync + Clone + 'static + std::fmt::Debug,
    {
        struct DetokenizeMetricsGuard {
            tracker: Option<Arc<RequestTracker>>,
        }

        impl DetokenizeMetricsGuard {
            #[inline]
            fn tracker(&self) -> Option<&RequestTracker> {
                self.tracker.as_deref()
            }
        }

        impl Drop for DetokenizeMetricsGuard {
            fn drop(&mut self) {
                let Some(tracker) = self.tracker() else {
                    return;
                };
                if let Some(total) = tracker.detokenize_total_latency() {
                    DETOKENIZE_TOTAL_US.inc_by(total.as_micros() as f64);
                }
                DETOKENIZE_TOKEN_COUNT.inc_by(tracker.detokenize_count() as f64);
            }
        }

        struct State<Resp>
        where
            Resp: Send + Sync + Clone + 'static + std::fmt::Debug,
        {
            response_stream: Pin<Box<dyn Stream<Item = Annotated<BackendOutput>> + Send>>,
            response_generator: Box<dyn DeltaGeneratorExt<Resp>>,
            detokenize_metrics: DetokenizeMetricsGuard,
            context: Arc<dyn AsyncEngineContext>,
            cancelled: bool,
            cumulative_output_tokens: usize,
            finish_reason_sent: bool,
            usage_chunk_sent: bool,
            /// Buffered plain usage chunk to send to the client after the payload
            /// chunk (ANNOTATION_PAYLOAD_USAGE). Only Some when is_usage_enabled().
            pending_client_usage: Option<Annotated<Resp>>,
            finished: bool,
            emit_payload_usage_chunk: bool,
            trace_tokens_enabled: bool,
            trace_finish_reason_metadata: Option<crate::request_trace::SharedFinishReasonMetadata>,
            mm_counts: MultimodalCounts,
            image_tokens: Option<usize>,
        }

        let tracker = generator.tracker();
        let state = State {
            response_stream: Box::pin(stream),
            response_generator: generator,
            detokenize_metrics: DetokenizeMetricsGuard { tracker },
            context: context.clone(),
            cancelled: false,
            cumulative_output_tokens: 0,
            finish_reason_sent: false,
            usage_chunk_sent: false,
            pending_client_usage: None,
            finished: false,
            emit_payload_usage_chunk,
            trace_tokens_enabled,
            trace_finish_reason_metadata,
            mm_counts,
            image_tokens,
        };

        // transform the common response stream into a chat response stream

        stream::unfold(state, |mut inner| {
            async move {
                // Drain the buffered client-facing plain usage chunk first.
                // This MUST come before the `finished` guard: the stream-end
                // handler sets inner.finished = true before returning the payload
                // chunk, so on the very next iteration the finished guard would
                // terminate before we ever emit the client chunk.
                if let Some(client_chunk) = inner.pending_client_usage.take() {
                    inner.finished = true;
                    // Emit unconditionally to match the non-payload path below; the
                    // chunk is only buffered after a finish_reason, so payload capture must
                    // not alter the client SSE tail.
                    return Some((client_chunk, inner));
                }

                // If already finished (and no pending client chunk), stop.
                if inner.finished {
                    return None;
                }

                if let Some(mut response) = inner.response_stream.next().await {
                    // Split topology: overlay a standalone router's forwarded routing_data
                    // (timing, query-only token_ids) onto this request's tracker so the
                    // frontend's nvext/timing surfaces populate.
                    drain_router_routing_data(
                        &mut response.data,
                        inner.detokenize_metrics.tracker(),
                    );

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

                        crate::request_trace::record_backend_finish_reason_metadata(
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
                    let tracker = inner.detokenize_metrics.tracker();
                    let llm_metrics = build_llm_metric_annotation(
                        tracker,
                        isl.unwrap_or(0),
                        current_osl,
                        chunk_tokens,
                        None,
                        inner.mm_counts,
                        inner.image_tokens,
                    );
                    if inner.trace_tokens_enabled {
                        crate::request_trace::record_llm_metric_tokens(
                            tracker,
                            isl,
                            current_osl,
                            None,
                        );
                    }

                    attach_llm_metrics(&mut response, llm_metrics);

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
                        let tracker = inner.detokenize_metrics.tracker();
                        let cached_tokens = usage
                            .prompt_tokens_details
                            .as_ref()
                            .and_then(|d| d.cached_tokens.map(|c| c as usize));
                        let llm_metrics = build_llm_metric_annotation(
                            tracker,
                            usage.prompt_tokens as usize,
                            usage.completion_tokens as usize,
                            0,
                            cached_tokens,
                            inner.mm_counts,
                            inner.image_tokens,
                        );
                        if inner.trace_tokens_enabled {
                            crate::request_trace::record_llm_metric_tokens(
                                tracker,
                                Some(usage.prompt_tokens as usize),
                                usage.completion_tokens as usize,
                                cached_tokens,
                            );
                        }

                        let usage_requested = inner.response_generator.is_usage_enabled();

                        if inner.emit_payload_usage_chunk {
                            // Payload capture on: emit a dedicated payload-usage chunk that
                            // always carries usage for the payload DeltaAggregator
                            // (the EventConverter strips it entirely from the
                            // client), and buffer the plain client usage chunk only
                            // when include_usage was requested.
                            if usage_requested {
                                inner.pending_client_usage = Some(Annotated::<Resp> {
                                    id: None,
                                    data: Some(usage_chunk.clone()),
                                    event: None,
                                    comment: None,
                                    error: None,
                                });
                            }

                            let annotation =
                                llm_metrics.to_annotation::<()>().unwrap_or_else(|e| {
                                    tracing::warn!("Failed to serialize metrics: {}", e);
                                    Annotated::<()>::from_data(())
                                });
                            let payload_usage = Annotated::<Resp> {
                                id: None,
                                data: Some(usage_chunk),
                                event: Some(ANNOTATION_PAYLOAD_USAGE.to_string()),
                                comment: annotation.comment,
                                error: None,
                            };
                            Some((payload_usage, inner))
                        } else {
                            // Payload capture off: a single usage chunk; data is present only
                            // when include_usage was requested. Metrics ride via the
                            // typed (serde-skip) llm_metrics field for internal
                            // observation, never reaching the client.
                            let data = if usage_requested {
                                Some(usage_chunk)
                            } else {
                                None
                            };
                            let mut annotated_usage = Annotated::<Resp> {
                                id: None,
                                data,
                                event: None,
                                comment: None,
                                error: None,
                            };
                            attach_llm_metrics(&mut annotated_usage, llm_metrics);
                            Some((annotated_usage, inner))
                        }
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
        // K3 wraps every assistant response in XTML, even when the request has
        // no tools. Keep its parser active so response/message wrappers never
        // leak into OpenAI `content`.
        if tool_call_parser.is_some_and(|parser| matches!(parser.as_str(), "kimi_k3" | "kimi-k3")) {
            return Ok(true);
        }

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

    /// Apply tool calling jail to the stream if needed.
    ///
    /// The jail itself now lives in `dynamo-parsers`
    /// (`dynamo_parsers::tool_calling::jail`), where it operates on the shared
    /// `CreateChatCompletionStreamResponse` — dynamo-parsers cannot depend on
    /// dynamo-runtime, so it does not know about `Annotated` or the `Nv`
    /// newtype. This method is the boundary adapter: it unwraps the dynamo
    /// `Annotated<Nv{inner, nvext}>` stream into the jail's
    /// `Annotated<CreateChatCompletionStreamResponse>`, runs the moved jail, and
    /// re-wraps the result.
    ///
    /// `nvext` is not populated on the streaming tool-call path (only the unary
    /// aggregator/anthropic paths set it), so the jail never needs to preserve
    /// it and re-wrapped chunks carry `nvext: None`.
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
        use dynamo_parsers::tool_calling::jail::{
            Annotated as JailAnnotated, apply_tool_calling_jail as jail_apply,
        };
        use std::sync::{Arc, Mutex};

        // The jail operates on the shared `Create` payload and never touches the
        // dynamo-only typed `llm_metrics`, which `transform_postprocessor_stream`
        // stamps *upstream* of the jail. `Create` has no slot for it, so buffer it
        // here on the way in and re-attach on the way out — keeping the metrics off
        // the shared type while preserving them across the boundary.
        //
        // `llm_metrics` is cumulative (`output_tokens`) plus per-chunk
        // (`chunk_tokens`), and the jail may fold N buffered input chunks into one
        // emitted chunk. So accumulate `chunk_tokens` and stamp the running total,
        // with the latest cumulative fields, onto the next emitted data chunk. That
        // preserves what `metrics.rs` records: `observe_response` sums `chunk_tokens`
        // and `observe_current_osl` takes the latest `output_tokens`. (The
        // annotation form on data-less usage chunks rides through untouched via
        // `event`/`comment`.)
        #[derive(Default)]
        struct PendingMetrics {
            template: Option<LLMMetricAnnotation>,
            chunk_tokens: usize,
        }
        let pending = Arc::new(Mutex::new(PendingMetrics::default()));
        let pending_in = Arc::clone(&pending);

        // Per-choice recovery state — allocated only for glm47 since only that
        // parser emits <tool_call> XML that can be truncated at max_tokens.
        // Buffers raw input and tracks what the jail emitted per choice.index
        // so n > 1 is handled correctly and double-emit is avoided.
        #[derive(Default)]
        struct ChoiceRecovery {
            input_text: String,
            emitted_text: String,
            recovered: bool,
        }
        // Named so the bool in the recoveries tuple is legible at each use site.
        struct PendingRecovery {
            choice_idx: u32,
            tail: String,
            tail_already_emitted: bool,
        }
        let is_glm47 = tool_call_parser.as_deref() == Some("glm47");
        // Token strings from the parser config so recovery matches the parser
        // even if the defaults are ever overridden.
        let glm47_cfg = dynamo_parsers::tool_calling::config::Glm47ParserConfig::default();
        let glm47_start = glm47_cfg.tool_call_start;
        let glm47_end = glm47_cfg.tool_call_end;
        let choice_recovery: Arc<Mutex<std::collections::HashMap<u32, ChoiceRecovery>>> =
            Arc::new(Mutex::new(std::collections::HashMap::new()));
        let choice_recovery_in = Arc::clone(&choice_recovery);
        let glm47_start_jail = glm47_start.clone();

        // dynamo `Annotated<Nv>` -> jail `Annotated<Create>` (buffer llm_metrics)
        let jail_input = stream.map(move |mut a| {
            if let Some(metrics) = a.data.as_mut().and_then(|nv| nv.llm_metrics.take()) {
                let mut p = pending_in.lock().expect("jail metrics buffer poisoned");
                p.chunk_tokens = p.chunk_tokens.saturating_add(metrics.chunk_tokens);
                p.template = Some(metrics);
            }
            // Buffer input content only for glm47 (truncation recovery).
            // Only retain from the last <tool_call> marker onward to bound
            // memory on long responses.
            if is_glm47 && let Some(data) = &a.data {
                let mut cr = choice_recovery_in.lock().expect("choice recovery poisoned");
                for choice in &data.inner.choices {
                    if let Some(ChatCompletionMessageContent::Text(content)) = &choice.delta.content
                    {
                        let state = cr.entry(choice.index).or_default();
                        state.input_text.push_str(content);
                        // Drop everything before the last marker to keep
                        // the buffer small. Walk back to a char boundary
                        // before draining so a multi-byte char split
                        // across chunks never triggers a panic.
                        let mut keep_from = match state.input_text.rfind(glm47_start_jail.as_str())
                        {
                            Some(pos) => pos,
                            // No marker yet — keep enough tail to
                            // catch a marker split across two chunks.
                            None => state
                                .input_text
                                .len()
                                .saturating_sub(glm47_start_jail.len() - 1),
                        };
                        while keep_from > 0 && !state.input_text.is_char_boundary(keep_from) {
                            keep_from -= 1;
                        }
                        state.input_text.drain(..keep_from);
                    }
                }
            }
            JailAnnotated {
                data: a.data.map(|nv| nv.inner),
                id: a.id,
                event: a.event,
                comment: a.comment,
                error: a.error.map(|e| e.to_string()),
            }
        });

        // jail `Annotated<Create>` -> dynamo `Annotated<Nv>` (re-attach llm_metrics)
        jail_apply(
            tool_call_parser,
            tool_choice,
            tool_definitions,
            uses_tool_call_structural_tag,
            jail_input,
        )
        .flat_map(move |a| {
            // Stamp the accumulated metrics onto the next emitted data chunk;
            // data-less/synthesized chunks carry it forward (or `None`).
            let llm_metrics = a.data.as_ref().and_then(|_| {
                let mut p = pending.lock().expect("jail metrics buffer poisoned");
                let chunk_tokens = p.chunk_tokens;
                p.chunk_tokens = 0;
                p.template.take().map(|mut metrics| {
                    metrics.chunk_tokens = chunk_tokens;
                    metrics
                })
            });
            let mut nv_chunk = Annotated {
                data: a.data.map(|inner| NvCreateChatCompletionStreamResponse {
                    inner,
                    nvext: None,
                    llm_metrics,
                }),
                id: a.id,
                event: a.event,
                comment: a.comment,
                error: a.error.map(DynamoError::msg),
            };

            // glm47: on finish_reason=length, recover the last incomplete
            // <tool_call> block. rfind skips complete blocks so earlier parsed
            // calls are never duplicated. Recovered content WILL contain raw
            // markup — callers that require "no tool tags in content" must
            // filter on finish_reason=length.
            //
            // TODO: this recovery runs inside apply_tool_calling_jail, which
            // the v2 path bypasses (use_parsers_v2 branch above). Adding
            // "glm47" to V2_FAMILIES in tool_parser_v2.rs silently disables
            // streaming recovery while aggregator.rs keeps running. At that
            // point hoist this above the jail/v2 branch — it only needs
            // buffered input text + finish_reason, both available there.
            // Pass 1 (immutable): compute the recovery tail per choice and
            // whether the jail already released it as content on this chunk.
            // We collect into a Vec so we can release the immutable borrow on
            // nv_chunk before mutating it in pass 2.
            let recoveries: Vec<PendingRecovery> = if is_glm47 {
                let mut cr = choice_recovery.lock().expect("choice recovery poisoned");
                nv_chunk
                    .data
                    .iter()
                    .flat_map(|data| data.inner.choices.iter())
                    .filter_map(|choice| {
                        let state = cr.entry(choice.index).or_default();
                        if !state.recovered
                            && let Some(ChatCompletionMessageContent::Text(t)) =
                                &choice.delta.content
                        {
                            state.emitted_text.push_str(t);
                            // Bound like input_text: retain only the suffix from
                            // the last marker onward — all the contains(&tail)
                            // check needs.
                            let mut keep_from = match state.emitted_text.rfind(glm47_start.as_str())
                            {
                                Some(pos) => pos,
                                None => state
                                    .emitted_text
                                    .len()
                                    .saturating_sub(glm47_start.len() - 1),
                            };
                            while keep_from > 0 && !state.emitted_text.is_char_boundary(keep_from) {
                                keep_from -= 1;
                            }
                            state.emitted_text.drain(..keep_from);
                        }
                        if state.recovered
                            || !matches!(
                                choice.finish_reason,
                                Some(dynamo_protocols::types::FinishReason::Length)
                            )
                        {
                            return None;
                        }
                        let tail =
                            state
                                .input_text
                                .rfind(glm47_start.as_str())
                                .and_then(|pos| {
                                    let t = &state.input_text[pos..];
                                    if !t.contains(glm47_end.as_str()) {
                                        Some(t.to_string())
                                    } else {
                                        None
                                    }
                                })?;
                        let tail_already_emitted = state.emitted_text.contains(&tail);
                        state.recovered = true;
                        Some(PendingRecovery {
                            choice_idx: choice.index,
                            tail,
                            tail_already_emitted,
                        })
                    })
                    .collect()
            } else {
                vec![]
            };

            // Pass 2 (mutable): when the jail already released the tail verbatim,
            // suppress the finish chunk's content entirely. The recovery chunk
            // carries just the marker-onwards tail, matching the non-streaming
            // path (rfind result only, no post-call prose).
            for pr in &recoveries {
                if !pr.tail_already_emitted {
                    continue;
                }
                if let Some(ref mut data) = nv_chunk.data {
                    for rc in data
                        .inner
                        .choices
                        .iter_mut()
                        .filter(|c| c.index == pr.choice_idx)
                    {
                        // The jail released the truncated block verbatim as content
                        // on this chunk, potentially preceded by post-call prose.
                        // glm47's parser drops post-call prose deliberately, so
                        // suppress the whole content here and let the recovery
                        // chunk carry just the marker-onwards tail — matching batch.
                        rc.delta.content = None;
                    }
                }
            }

            // Pass 3: emit a recovery chunk per affected choice carrying just
            // the truncated tail (marker onwards, no post-call prose).
            let recovery_chunks: Vec<_> = recoveries
                .into_iter()
                .filter_map(|pr| {
                    let PendingRecovery {
                        choice_idx, tail, ..
                    } = pr;
                    tracing::warn!(
                        choice_index = choice_idx,
                        recovered_bytes = tail.len(),
                        "glm47 streaming: partial <tool_call> emitted as content \
                         on length finish"
                    );
                    let mut rec = nv_chunk.clone();
                    rec.id = None;
                    rec.event = None;
                    rec.comment = None;
                    rec.error = None;
                    let rd = rec.data.as_mut()?;
                    rd.inner.usage = None;
                    rd.llm_metrics = None;
                    rd.inner.choices.retain(|c| c.index == choice_idx);
                    for rc in &mut rd.inner.choices {
                        rc.delta.content = Some(ChatCompletionMessageContent::Text(tail.clone()));
                        rc.delta.tool_calls = None;
                        rc.finish_reason = None;
                        rc.logprobs = None;
                    }
                    Some(rec)
                })
                .collect();

            futures::stream::iter(recovery_chunks.into_iter().chain(std::iter::once(nv_chunk)))
        })
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
        // - kimi_k3: `<|open|>` / `<|close|>` / `<|sep|>` XTML markers.
        // - mistral: `[THINK]` / `[/THINK]` reasoning markers.
        // - minimax_m3: `]<]minimax[>[` tool-call namespace tokens and
        //   `<mm:think>` reasoning markers.
        // - inkling: `<|message_model|>` / `<|content_thinking|>` /
        //   `<|content_text|>` / `<|content_invoke_tool_json|>` / `<|end_message|>`
        //   channel markers, consumed by both the tool-call and reasoning parsers.
        matches!(
            tool_call_parser,
            Some("gemma4")
                | Some("gemma-4")
                | Some("harmony")
                | Some("kimi_k2")
                | Some("kimi_k3")
                | Some("kimi-k3")
                | Some("minimax_m3")
                | Some("minimax-m3")
                | Some("minimax_m3_nom")
                | Some("minimax-m3-nom")
                | Some("inkling")
        ) || matches!(
            reasoning_parser,
            Some("gemma4")
                | Some("gemma-4")
                | Some("gpt_oss")
                | Some("kimi_k25")
                | Some("kimi_k3")
                | Some("kimi-k3")
                | Some("mistral")
                | Some("minimax_m3")
                | Some("minimax-m3")
                | Some("inkling")
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

    /// Whether a request should surface parsed `reasoning_content` as `content`
    /// when no content was generated: any request carrying
    /// `force_nonempty_content=true`.
    ///
    /// Deliberately NOT keyed on the model or its reasoning parser. The flag is a
    /// request-level contract — "this response will have non-empty content" — so
    /// it is honored generically after parsing and before the response is sent,
    /// rather than being reimplemented inside each model-specific parser. Keying
    /// it on a parser allow-list means every new alias silently loses the
    /// behavior until someone edits the list.
    ///
    /// Note it rides in on `chat_template_args` but is NOT consumed by the chat
    /// template: the Nemotron template never reads `force_nonempty_content`, and
    /// rendering with it set produces a byte-identical prompt. It is a
    /// serving-layer flag that upstream happens to transport through the template
    /// kwargs, which is why the check belongs in postprocessing and not in a
    /// parser. So a client can set it on any model, and doing so is an explicit
    /// request for non-empty content — honoring it generically is the intent, not
    /// an accident of where the flag is declared.
    ///
    /// Drives both paths. The chat and Anthropic HTTP handlers pass it to the
    /// aggregator via `ParsingOptions::move_reasoning_to_content_when_empty` for
    /// non-streaming, and `postprocessor_parsing_stream` passes it as
    /// `defer_reasoning_for_nonempty_content` so the streaming path can hold
    /// reasoning back and reach the same answer at the terminal chunk. Both must
    /// use this one predicate or the two paths would disagree on the same input.
    ///
    /// `enable_thinking` is intentionally not consulted here: when thinking is
    /// off, reasoning parsing is disabled, so `reasoning_content` is always empty
    /// and the move is vacuous rather than suppressed.
    pub(crate) fn wants_reasoning_as_content_when_empty(
        chat_template_args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> bool {
        chat_template_args.is_some_and(|args| {
            args.get("force_nonempty_content") == Some(&serde_json::Value::Bool(true))
        })
    }

    /// Whether this request's stream can withhold every data frame while it
    /// buffers, which is the only reason to force SSE keep-alive frames on.
    ///
    /// The HTTP handlers used to gate the heartbeat on
    /// `wants_reasoning_as_content_when_empty && reasoning_parser.is_some()`,
    /// but that is broader than the deferral it describes: the buffering only
    /// runs when `postprocessor_parsing_stream` takes its `should_parse_reasoning`
    /// branch. A `force_nonempty_content=true` request with reasoning disabled
    /// (`enable_thinking=false`) defers nothing, yet still got heartbeats — and
    /// the configured interval is opt-in precisely because some
    /// OpenAI-compatible clients do not ignore SSE comment frames.
    ///
    /// Known gap: `skip_reasoning_for_guided_json` also suppresses the deferral,
    /// but it is derived from guided-output inspection that is not available at
    /// the HTTP layer, so a guided-JSON bypass still enables the heartbeat when
    /// nothing is withheld. That direction is conservative — extra comment
    /// frames on an opt-in path rather than a silent stream — and closing it
    /// needs the guided-output derivation lifted out of the stream builder.
    pub(crate) fn stream_can_defer_all_output(
        reasoning_parser: Option<&str>,
        chat_template_args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> bool {
        reasoning_parser.is_some()
            && Self::wants_reasoning_as_content_when_empty(chat_template_args)
            && !Self::is_reasoning_disabled_by_request(reasoning_parser, chat_template_args)
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
                    | "deepseek_v3"
                    | "deepseek_v3_1"
                    | "deepseek_v3_2"
                    | "step3"
                    | "kimi_k25"
                    | "mistral"
                    | "minimax_m2"
                    | "minimax_append_think"
                    | "nemotron_nano"
                    | "nemotron3"
                    | "nemotron_v3"
            )
        )
    }

    /// Force-reasoning parsers proven to receive both bare guided JSON and
    /// native-reasoner-gated `reasoning</think>JSON`. These use the stream-shape
    /// detector instead of the historical unconditional guided-JSON bypass.
    fn supports_reasoning_before_guided_json(reasoning_parser: Option<&str>) -> bool {
        matches!(
            reasoning_parser,
            Some(
                "deepseek_r1"
                    | "deepseek_v3"
                    | "deepseek_v3_1"
                    | "deepseek_v3_2"
                    | "step3"
                    | "kimi_k25"
                    | "mistral"
                    | "minimax_m2"
                    | "nemotron_nano"
                    | "nemotron3"
                    | "nemotron_v3"
            )
        )
    }

    /// Reasoning openers that overlap with a valid guided-JSON prefix.
    fn guided_json_reasoning_start_token(reasoning_parser: Option<&str>) -> Option<&'static str> {
        match reasoning_parser {
            Some("mistral") => Some("[THINK]"),
            _ => None,
        }
    }

    fn skips_guided_json_when_prompt_injected(reasoning_parser: Option<&str>) -> bool {
        matches!(
            reasoning_parser,
            Some(
                "deepseek_v4"
                    | "deepseek-v4"
                    | "deepseekv4"
                    | "glm45"
                    | "minimax_m3"
                    | "minimax-m3"
            )
        )
    }

    fn skips_structured_response_when_prompt_injected(reasoning_parser: Option<&str>) -> bool {
        matches!(reasoning_parser, Some("qwen3" | "kimi_k3" | "kimi-k3"))
            || Self::skips_guided_json_when_prompt_injected(reasoning_parser)
    }

    fn prompt_injected_reasoning_start(
        reasoning_parser: Option<&str>,
        formatted_prompt: Option<&str>,
    ) -> bool {
        let Some(prompt) = formatted_prompt.map(str::trim_end) else {
            return false;
        };

        match reasoning_parser {
            Some("minimax_m3") | Some("minimax-m3") => prompt.ends_with("<mm:think>"),
            Some("kimi_k3") | Some("kimi-k3") => prompt.ends_with("<|open|>think<|sep|>"),
            _ => prompt.ends_with("<think>"),
        }
    }

    fn prompt_injected_reasoning_ended_arg(
        reasoning_parser: Option<&str>,
        formatted_prompt: Option<&str>,
    ) -> Option<bool> {
        let should_forward = matches!(
            reasoning_parser,
            Some("minimax_m2" | "minimax_m3" | "minimax-m3" | "kimi_k3" | "kimi-k3")
        );
        if should_forward
            && Self::prompt_injected_reasoning_start(reasoning_parser, formatted_prompt)
        {
            Some(false)
        } else {
            None
        }
    }

    /// Check if reasoning parsing should be disabled based on per-request parameters.
    /// For kimi_k25/K3: disabled when chat_template_args contains "thinking": false.
    /// For Nemotron force-reasoning aliases: disabled when chat_template_args
    ///   contains "enable_thinking": false. "force_nonempty_content": true does
    ///   NOT disable parsing (streaming or non-streaming): the parser stays on so
    ///   reasoning is split from the answer, and reasoning is surfaced as content
    ///   only when no content was generated (non-streaming, in the aggregator).
    /// For DeepSeek: follows the same effective mode used by the prompt renderer.
    /// For Mistral: disabled unless `reasoning_effort` is present and not `none`.
    /// For Gemma 4: reasoning is opt-in and disabled unless chat_template_args
    ///   explicitly enables thinking. Gemma 4's chat template injects reasoning
    ///   markers only when enable_thinking=true.
    /// For MiniMax M3: disabled when chat_template_args contains
    ///   "thinking_mode": "disabled", matching SGLang's MiniMax M3 request
    ///   convention.
    fn is_reasoning_disabled_by_request(
        reasoning_parser: Option<&str>,
        chat_template_args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> bool {
        match reasoning_parser {
            Some("kimi_k25" | "kimi_k3" | "kimi-k3") => {
                dynamo_renderer::thinking_bool_from_args(chat_template_args) == Some(false)
            }
            parser if Self::is_nemotron_force_reasoning(parser) => {
                // `enable_thinking=false` turns reasoning off entirely (streaming
                // and non-streaming). `force_nonempty_content=true` does NOT
                // disable parsing: keeping the parser on splits reasoning from the
                // answer, so a reasoning+answer turn no longer leaks the reasoning
                // text and `</think>` into `content`.
                //
                // The reasoning-*only* move (surface reasoning as content when the
                // answer is empty) happens on both paths, by different means.
                // Non-streaming uses the aggregator flag
                // ParsingOptions::move_reasoning_to_content_when_empty. Streaming
                // cannot retract a reasoning_content delta already sent, so it
                // instead holds reasoning back until it knows whether an answer
                // follows — see `defer_reasoning_for_nonempty_content` and
                // `drain_deferred_reasoning`. Both end with the same contract: a
                // reasoning-only turn surfaces its text as `content`.
                dynamo_renderer::thinking_bool_from_args(chat_template_args) == Some(false)
            }
            Some("deepseek_v3" | "deepseek_v3_1") => {
                !Self::deepseek_renderer_reasoning_enabled(chat_template_args, false)
            }
            Some(
                "deepseek_r1" | "deepseek_v3_2" | "deepseek_v4" | "deepseek-v4" | "deepseekv4"
                | "minimax_m2",
            ) => !Self::deepseek_renderer_reasoning_enabled(chat_template_args, true),
            Some("gemma4") | Some("gemma-4") => {
                dynamo_renderer::thinking_bool_from_args(chat_template_args) != Some(true)
            }
            Some("mistral") => !Self::mistral_reasoning_enabled(chat_template_args),
            Some("minimax_m3") | Some("minimax-m3") => {
                if let Some(args) = chat_template_args
                    && let Some(mode) = args.get("thinking_mode").and_then(|v| v.as_str())
                {
                    return mode == "disabled";
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
            false,
        )
    }

    fn parse_reasoning_content_from_stream_inner<S>(
        stream: S,
        parser_name: String,
        prompt_injected_reasoning: bool,
        bypass_bare_guided_json: bool,
        defer_reasoning_for_nonempty_content: bool,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        // Parsers and bypass decisions are created lazily per `choice.index`
        // inside the unfold loop, so `n > 1` choices never share state.
        let state = ReasoningState {
            stream: Box::pin(stream),
            parser_name,
            prompt_injected_reasoning,
            bypass_bare_guided_json,
            choices: HashMap::new(),
            last_response: None,
            defer_reasoning_for_nonempty_content,
            saw_terminal_error: false,
        };

        stream::unfold(state, |mut state| async move {
            if let Some(response) = state.stream.next().await {
                // An error is terminal for the flush: latch it so the
                // end-of-stream branch below stays quiet. The chunk still takes
                // the normal path — `is_error()` keys on the annotation event,
                // not on `data`, so short-circuiting here would change how a
                // data-carrying error chunk is processed.
                if response.is_error() {
                    state.saw_terminal_error = true;
                }
                // Split disjoint field borrows so the per-choice map and the
                // parser-factory inputs can be used together inside map_data.
                // Scoped in a block so the borrows end before `state` moves.
                let processed_response = {
                    let ReasoningState {
                        parser_name,
                        prompt_injected_reasoning,
                        bypass_bare_guided_json,
                        choices,
                        defer_reasoning_for_nonempty_content,
                        ..
                    } = &mut state;
                    let parser_name = &*parser_name;
                    let prompt_injected_reasoning = *prompt_injected_reasoning;
                    let bypass_bare_guided_json = *bypass_bare_guided_json;
                    let defer_reasoning = *defer_reasoning_for_nonempty_content;

                    response.map_data(|mut data| {
                        for choice in data.inner.choices.iter_mut() {
                            let choice_state = choices.entry(choice.index).or_insert_with(|| {
                                let mut parser =
                                    Box::new(ReasoningParserType::get_reasoning_parser_from_name(
                                        parser_name,
                                    ))
                                        as Box<dyn ReasoningParser>;
                                if prompt_injected_reasoning {
                                    parser.set_in_reasoning(true);
                                }
                                ChoiceReasoningState {
                                    parser,
                                    guided_json_bypass_decision: (!bypass_bare_guided_json)
                                        .then_some(false),
                                    pending_reasoning: String::new(),
                                    pending_content: String::new(),
                                    left_reasoning: false,
                                    drained: false,
                                    parser_finished: false,
                                }
                            });

                            // Decide once per choice, from ITS OWN first
                            // non-whitespace content, whether the backend
                            // emitted bare guided JSON (`[`/`{`) that must reach
                            // the tool jail unparsed.
                            let bypass_decision = if bypass_bare_guided_json {
                                match choice_state.guided_json_bypass_decision {
                                    Some(decision) => Some(decision),
                                    None => {
                                        let decision = match choice.delta.content.as_ref() {
                                            Some(ChatCompletionMessageContent::Text(text)) => {
                                                let text = text.trim_start();
                                                if text.is_empty() {
                                                    None
                                                } else {
                                                    Some(matches!(text.as_bytes()[0], b'[' | b'{'))
                                                }
                                            }
                                            _ => None,
                                        };
                                        if let Some(decision) = decision {
                                            choice_state.guided_json_bypass_decision =
                                                Some(decision);
                                        }
                                        decision
                                    }
                                }
                            } else {
                                Some(false)
                            };

                            // Only a choice decided NOT to bypass is parsed. A
                            // bare-JSON or still-undecided (whitespace-only)
                            // choice keeps its content untouched for the jail.
                            // Reasoning parsing only applies to text content;
                            // multimodal content passes through unchanged.
                            if bypass_decision == Some(false)
                                && let Some(ChatCompletionMessageContent::Text(text)) =
                                    choice.delta.content.as_ref()
                            {
                                let parser_result = choice_state
                                    .parser
                                    .parse_reasoning_streaming_incremental(text, &[]);

                                // A backend that keeps sending content after this
                                // choice's `finish_reason` is out of protocol, but
                                // it has still fed bytes to the parser. Reopen the
                                // drain so they have somewhere to go at EOF instead
                                // of being stranded in parser state.
                                choice_state.drained = false;

                                if defer_reasoning && !choice_state.left_reasoning {
                                    // Still ambiguous: this text may be reasoning
                                    // followed by an answer, or the answer itself
                                    // reported as reasoning because the parser
                                    // starts inside the reasoning block. Hold it.
                                    choice_state
                                        .pending_reasoning
                                        .push_str(&parser_result.reasoning_text);
                                    // Whitespace-only normal text does not settle
                                    // anything: the aggregator treats such content
                                    // as empty (matching vLLM's
                                    // `not final_content.strip()`), so releasing
                                    // here would make streaming and non-streaming
                                    // disagree on the same turn. Hold it with the
                                    // reasoning until real answer text arrives.
                                    if parser_result.normal_text.trim().is_empty() {
                                        choice_state
                                            .pending_content
                                            .push_str(&parser_result.normal_text);
                                        choice.delta.content = None;
                                        choice.delta.reasoning_content = None;
                                    } else {
                                        // Real answer text means the parser left
                                        // the reasoning block, so everything held
                                        // so far really was reasoning.
                                        choice_state.left_reasoning = true;
                                        choice.delta.reasoning_content = (!choice_state
                                            .pending_reasoning
                                            .is_empty())
                                        .then(|| {
                                            std::mem::take(&mut choice_state.pending_reasoning)
                                        });
                                        let mut text =
                                            std::mem::take(&mut choice_state.pending_content);
                                        text.push_str(&parser_result.normal_text);
                                        choice.delta.content =
                                            Some(ChatCompletionMessageContent::Text(text));
                                    }
                                } else {
                                    choice.delta.content = parser_result
                                        .get_some_normal_text()
                                        .map(ChatCompletionMessageContent::Text);
                                    choice.delta.reasoning_content =
                                        parser_result.get_some_reasoning();
                                }
                            }

                            // This choice is finishing, so drain what it holds
                            // onto THIS delta. Emitting it as a later chunk
                            // would put content after `finish_reason` (and
                            // after the trailing usage chunk), where a client
                            // that stops at the terminal chunk never sees it.
                            // A parts delta has no text slot to append to, so
                            // draining onto it would leave the recovered text
                            // nowhere to go. Leave the choice undrained and let
                            // the end-of-stream fallback emit it as its own
                            // chunk instead of dropping it.
                            let terminal_carries_parts = matches!(
                                choice.delta.content,
                                Some(ChatCompletionMessageContent::Parts(_))
                            );
                            // Also require that this choice was actually parsed.
                            // A guided-JSON choice that bypassed the parser never
                            // fed it anything, so finishing that parser could only
                            // contribute text the choice never generated.
                            if defer_reasoning
                                && choice.finish_reason.is_some()
                                && !terminal_carries_parts
                                && bypass_decision == Some(false)
                            {
                                let (content, reasoning) = drain_deferred_reasoning(choice_state);
                                if let Some(content) = content {
                                    let merged = match choice.delta.content.take() {
                                        Some(ChatCompletionMessageContent::Text(existing)) => {
                                            existing + &content
                                        }
                                        Some(other) => {
                                            // Unreachable: guarded above.
                                            choice.delta.content = Some(other);
                                            content
                                        }
                                        None => content,
                                    };
                                    if choice.delta.content.is_none() {
                                        choice.delta.content =
                                            Some(ChatCompletionMessageContent::Text(merged));
                                    }
                                }
                                if let Some(reasoning) = reasoning {
                                    choice.delta.reasoning_content = Some(
                                        choice.delta.reasoning_content.take().unwrap_or_default()
                                            + &reasoning,
                                    );
                                }
                            }
                        }
                        Ok(data)
                    })
                };

                // Retain a spare envelope only when an EOF flush may follow, so
                // the common reasoning path avoids a full per-token clone. Skip
                // chunks with no choices (the trailing usage-only chunk): they
                // carry no delta slot, so using one as the flush envelope would
                // drop the very bytes the flush exists to preserve.
                if state.defer_reasoning_for_nonempty_content
                    && processed_response
                        .data
                        .as_ref()
                        .is_some_and(|data| !data.inner.choices.is_empty())
                {
                    state.last_response = Some(processed_response.clone());
                }
                Some((processed_response, state))
            } else if !state.defer_reasoning_for_nonempty_content || state.saw_terminal_error {
                // After a backend error the buffered bytes are dropped rather
                // than surfaced: the request failed, so there is no answer to
                // complete. See `saw_terminal_error`.
                None
            } else {
                // Upstream ended with no answer for the choices still holding a
                // pending buffer. Normally the terminal chunk already drained
                // them, so this is the fallback for a stream that ends without
                // any `finish_reason` — an aborted or truncated generation.
                // The synthetic chunk it builds is the only case where recovered
                // bytes arrive after the last upstream chunk, which is
                // unavoidable when there was no terminal chunk to attach them
                // to. Only the force_nonempty_content path reaches this branch;
                // every other parser took the `None` branch above and keeps its
                // original no-flush EOF behavior. Taking the envelope below
                // rather than cloning it makes this branch one-shot: once it is
                // gone the next poll ends the stream.
                // Sorted so the emitted choice order is deterministic rather
                // than following HashMap iteration order.
                let mut indices: Vec<u32> = state.choices.keys().copied().collect();
                indices.sort_unstable();
                #[allow(clippy::type_complexity)]
                let flushed: Vec<(u32, Option<String>, Option<String>)> = indices
                    .into_iter()
                    .filter_map(|index| {
                        let choice_state = state.choices.get_mut(&index)?;
                        if choice_state.guided_json_bypass_decision != Some(false) {
                            return None;
                        }
                        let (content, reasoning) = drain_deferred_reasoning(choice_state);
                        (content.is_some() || reasoning.is_some())
                            .then_some((index, content, reasoning))
                    })
                    .collect();
                if flushed.is_empty() {
                    return None;
                }
                let mut response = state.last_response.take()?;
                // See `scrub_synthetic_chunk_metadata`: this chunk produced no
                // tokens, so every per-chunk field from the envelope it was
                // cloned from has to be dropped rather than reported twice.
                scrub_synthetic_chunk_metadata(&mut response)?;
                let data = response.data.as_mut()?;
                // Rebuild the choice list from the flushed indices rather than
                // reusing the envelope's own choices: with `n > 1` the last
                // content-bearing chunk carries only the choices that happened
                // to be interleaved into it, which need not be the set that has
                // buffered bytes.
                let template = data.inner.choices.first()?.clone();
                data.inner.choices = flushed
                    .into_iter()
                    .map(|(index, content, reasoning)| {
                        let mut choice = template.clone();
                        choice.index = index;
                        choice.delta.role = None;
                        choice.delta.tool_calls = None;
                        choice.delta.function_call = None;
                        choice.delta.refusal = None;
                        choice.finish_reason = None;
                        choice.logprobs = None;
                        choice.delta.content = content.map(ChatCompletionMessageContent::Text);
                        choice.delta.reasoning_content = reasoning;
                        choice
                    })
                    .collect();
                Some((response, state))
            }
        })
        .fuse()
    }

    /// Hold the trailing usage-only chunk until every parser recovery chunk has
    /// been emitted. A truncated upstream stream can end without
    /// `finish_reason`, so reasoning recovery happens at EOF; forwarding usage
    /// immediately would put that recovered content after the chunk clients
    /// treat as the stream trailer.
    fn hold_usage_until_stream_end<S>(
        stream: S,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        async_stream::stream! {
            tokio::pin!(stream);
            let mut pending_usage = None;
            while let Some(response) = stream.next().await {
                let is_usage_only = response.data.as_ref().is_some_and(|data| {
                    data.inner.choices.is_empty() && data.inner.usage.is_some()
                });
                if is_usage_only {
                    if let Some(previous) = pending_usage.replace(response) {
                        yield previous;
                    }
                } else {
                    yield response;
                }
            }
            if let Some(usage) = pending_usage {
                yield usage;
            }
        }
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
                        let trimmed = choice_state.buffer.trim_start();
                        if trimmed.is_empty()
                            || (state.think_start_token.starts_with(trimmed)
                                && trimmed.len() < state.think_start_token.len())
                        {
                            choice.delta.content = None;
                            continue;
                        }

                        choice_state.decided = true;
                        if let Some(remainder) = trimmed.strip_prefix(state.think_start_token) {
                            remainder.to_string()
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
                if response
                    .data
                    .as_ref()
                    .is_some_and(|data| !data.inner.choices.is_empty())
                {
                    state.last_response = Some(response.clone());
                }

                Some((response, state))
            } else if state.eof_flushed {
                None
            } else {
                state.eof_flushed = true;
                let flushed = drain_undecided_buffers(&mut state.choices);
                if flushed.is_empty() {
                    None
                } else {
                    let mut response = state.last_response.clone()?;
                    // Same envelope problem as the reasoning-stream flush: this
                    // is a clone of an already-counted chunk, so it goes through
                    // the shared scrub rather than repeating a partial copy of
                    // it here.
                    scrub_synthetic_chunk_metadata(&mut response)?;
                    let data = response.data.as_mut()?;
                    let mut template = data.inner.choices.first()?.clone();
                    template.delta.role = None;
                    template.delta.tool_calls = None;
                    template.delta.function_call = None;
                    template.delta.refusal = None;
                    template.delta.reasoning_content = None;
                    template.finish_reason = None;
                    template.logprobs = None;
                    let mut flushed: Vec<_> = flushed.into_iter().collect();
                    flushed.sort_unstable_by_key(|(index, _)| *index);
                    data.inner.choices = flushed
                        .into_iter()
                        .map(|(index, buffer)| {
                            let mut choice = template.clone();
                            choice.index = index;
                            choice.delta.content = Some(ChatCompletionMessageContent::Text(buffer));
                            choice
                        })
                        .collect();

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

        // Build request payload handle (None if request trace is disabled / not eligible).
        // The handle snapshots the pristine request and its arrival time here;
        // the single payload record is published once at stream completion
        // (or with an empty response on cancel/timeout), off the request path.
        let payload_http_headers = if crate::request_trace::payload::http_header_capture_active() {
            context
                .get_optional::<std::collections::BTreeMap<String, String>>(
                    crate::request_trace::payload::HTTP_HEADERS_CONTEXT_KEY,
                )
                .ok()
                .flatten()
        } else {
            None
        };
        let payload_handle = crate::request_trace::payload::create_handle(
            &request,
            &request_id,
            payload_http_headers,
        );

        // For non-streaming requests (stream=false), enable usage by default
        // This ensures compliance with OpenAI API spec where non-streaming responses
        // always include usage statistics
        request.enable_usage_for_nonstreaming(original_stream_flag);

        // Set stream=true for internal processing (after request payload capture)
        request.inner.stream = Some(true);
        // Apply the deployment default before parser-specific normalization so
        // it can override an implicit model default (for example Kimi K2.5),
        // while explicit request controls still take precedence.
        let thinking_control_from_client = Self::request_has_client_thinking_control(&request);
        self.apply_default_thinking_mode(&mut request);
        Self::normalize_thinking_arg_with_source(
            &mut request,
            self.runtime_config.reasoning_parser.as_deref(),
            self.tool_call_parser.as_deref(),
            thinking_control_from_client,
        );
        Self::normalize_kimi_k3_named_tool_choice(&mut request, self.tool_call_parser.as_deref());

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
        let (mut common_request, annotations, prompt_injected_reasoning, image_tokens) = self
            .preprocess_request_with_options(
                &request,
                tracker.as_deref(),
                preprocess_options,
                context
                    .get_optional::<String>(LORA_NAME_CONTEXT_KEY)
                    .ok()
                    .flatten()
                    .map(|name| name.as_ref().clone()),
            )
            .await?;
        attach_agent_context_from_context(&mut common_request, &context);

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

        // Capture media counts before `common_request` is moved into the context.
        let mm_counts = MultimodalCounts::from_preprocessed(&common_request);

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
        let stream = Self::transform_postprocessor_stream_with_image_tokens(
            response_stream,
            response_generator,
            context.clone(),
            payload_handle.is_some(),
            trace_tokens_enabled,
            trace_finish_reason_metadata,
            mm_counts,
            image_tokens,
        );

        let transformed_stream = self.postprocessor_parsing_stream(
            stream,
            &request,
            prompt_injected_reasoning,
            uses_tool_call_structural_tag,
        )?;
        let transformed_stream = Self::normalize_chat_stream_roles(transformed_stream);

        // Apply request payload aggregation strategy.
        // The payload branch already returns Pin<Box<...>> from scan/fold_aggregate_with_future,
        // while the non-payload branch boxes the impl Stream from postprocessor_parsing_stream.
        let final_stream = if let Some(payload) = payload_handle {
            let (stream, agg_fut) = if payload.streaming() {
                // Streaming: apply scan (pass-through + parallel aggregation)
                crate::request_trace::payload_stream::scan_aggregate_with_future(transformed_stream)
            } else {
                // Non-streaming: apply fold (collect all, then emit single chunk)
                crate::request_trace::payload_stream::fold_aggregate_with_future(transformed_stream)
            };

            // Spawn the payload emit off the request path. `agg_fut` resolves to
            // None on client cancel / gateway timeout / aggregation failure; we
            // still emit the payload record with an empty response so those
            // cases remain inspectable. The record carries the request snapshot
            // and arrival time captured at handle creation.
            tokio::spawn(async move {
                match agg_fut.await {
                    Some(final_resp) => payload.emit(Some(Arc::new(final_resp))),
                    None => {
                        tracing::debug!(
                            request_id = %payload.request_id(),
                            "request payload: response aggregation incomplete (client cancel / timeout); emitting request-only record"
                        );
                        payload.emit(None);
                    }
                }
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
        let mut builder = self.builder_with_lora(
            &request,
            context
                .get_optional::<String>(LORA_NAME_CONTEXT_KEY)
                .ok()
                .flatten()
                .map(|name| name.as_ref().clone()),
        )?;

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
            let (token_ids, ann) = self
                .gather_tokens(&request, None, tracker.as_deref())
                .await?;
            builder.token_ids(token_ids);
            ann
        };

        // Gather multimodal data (works with both embeddings and text prompts)
        // Returned MM entries are unused on the embeddings path; routing info is
        // not built here.
        let _ = self
            .gather_multi_modal_data(&request, &mut builder, None, &[])
            .await?;

        let mut common_request = builder.build()?;
        Self::validate_preprocessed_token_budget(&common_request, self.token_budget.as_ref())?;
        attach_agent_context_from_context(&mut common_request, &context);

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

        // transform the postprocessor stream. Legacy `/v1/completions` is
        // text-only, so multimodal counts are always zero here.
        let stream = Self::transform_postprocessor_stream(
            response_stream,
            response_generator,
            context.clone(),
            false,
            trace_tokens_enabled,
            trace_finish_reason_metadata,
            MultimodalCounts::default(),
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
    use crate::protocols::common::extensions::{
        AGENT_CONTEXT_CONTEXT_KEY, AgentCompaction, AgentContext,
    };
    use crate::protocols::common::preprocessor::MultimodalData;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
        Role,
    };

    fn chat_stream_chunk(
        index: u32,
        role: Option<Role>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                role,
                content: Some(ChatCompletionMessageContent::Text("content".to_string())),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        };
        Annotated::from_data(NvCreateChatCompletionStreamResponse {
            inner: CreateChatCompletionStreamResponse {
                id: "test".to_string(),
                choices: vec![choice],
                created: 0,
                model: "test".to_string(),
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            },
            nvext: None,
            llm_metrics: None,
        })
    }

    #[tokio::test]
    async fn test_normalize_chat_stream_roles_recovers_missing_first_role_per_choice() {
        let input = stream::iter(vec![
            // Mirrors a parser releasing buffered content without the role that
            // arrived on the original, swallowed chunk.
            chat_stream_chunk(0, None),
            chat_stream_chunk(1, None),
            chat_stream_chunk(0, Some(Role::Assistant)),
            chat_stream_chunk(1, Some(Role::Assistant)),
        ]);

        let output: Vec<_> = OpenAIPreprocessor::normalize_chat_stream_roles(input)
            .collect()
            .await;
        let roles: Vec<_> = output
            .iter()
            .map(|response| response.data.as_ref().unwrap().inner.choices[0].delta.role)
            .collect();

        assert_eq!(
            roles,
            vec![Some(Role::Assistant), Some(Role::Assistant), None, None,]
        );
    }

    #[test]
    fn prompt_invalid_request_maps_to_invalid_argument() {
        let error = PromptRenderError::invalid_request("unsupported model parameter").into();
        let mapped = OpenAIPreprocessor::map_prompt_render_error(error);
        let mapped = mapped
            .downcast_ref::<DynamoError>()
            .expect("prompt validation should map to a DynamoError");

        assert!(matches!(mapped.error_type(), ErrorType::InvalidArgument));
        assert_eq!(mapped.message(), "unsupported model parameter");
    }

    #[test]
    fn ordinary_prompt_error_maps_to_invalid_argument() {
        let mapped = OpenAIPreprocessor::map_prompt_render_error(anyhow::anyhow!(
            "template configuration failed"
        ));
        let mapped = mapped
            .downcast_ref::<DynamoError>()
            .expect("any prompt render failure should map to a DynamoError");

        assert!(matches!(mapped.error_type(), ErrorType::InvalidArgument));
        assert_eq!(mapped.message(), "template configuration failed");
    }

    fn url_entry(u: &str) -> MultimodalData {
        MultimodalData::Url(url::Url::parse(u).unwrap())
    }

    fn preprocessed_with_media(media: Option<MultimodalDataMap>) -> PreprocessedRequest {
        let mut b = PreprocessedRequest::builder();
        b.model("m".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default());
        if let Some(m) = media {
            b.multi_modal_data(Some(m));
        }
        b.build().unwrap()
    }

    #[test]
    fn test_multimodal_counts_from_preprocessed_mixed() {
        // 2 images, 1 video, 0 audio -> counts reflect vec lengths per modality.
        let mut map: MultimodalDataMap = HashMap::new();
        map.insert(
            "image_url".to_string(),
            vec![url_entry("http://x/a.png"), url_entry("http://x/b.png")],
        );
        map.insert("video_url".to_string(), vec![url_entry("http://x/c.mp4")]);

        let req = preprocessed_with_media(Some(map));
        let counts = MultimodalCounts::from_preprocessed(&req);
        assert_eq!(counts.image, 2);
        assert_eq!(counts.video, 1);
        assert_eq!(counts.audio, 0);
    }

    #[test]
    fn test_multimodal_counts_from_preprocessed_text_only() {
        // No multi_modal_data -> all zero.
        let req = preprocessed_with_media(None);
        let counts = MultimodalCounts::from_preprocessed(&req);
        assert_eq!(counts.image, 0);
        assert_eq!(counts.video, 0);
        assert_eq!(counts.audio, 0);
    }

    #[test]
    fn tool_message_exposes_multimodal_content_parts() {
        let message: ChatCompletionRequestMessage = serde_json::from_value(serde_json::json!({
            "role": "tool",
            "tool_call_id": "call_media",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aGVsbG8="
                    }
                },
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://example.com/clip.mp4"
                    }
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://example.com/audio.wav"
                    }
                }
            ]
        }))
        .unwrap();

        let parts: Vec<_> = multimodal_content_parts(&message).unwrap().collect();
        assert!(matches!(
            parts[0].as_user().as_ref(),
            ChatCompletionRequestUserMessageContentPart::ImageUrl(_)
        ));
        assert!(matches!(
            parts[1].as_user().as_ref(),
            ChatCompletionRequestUserMessageContentPart::VideoUrl(_)
        ));
        assert!(matches!(
            parts[2].as_user().as_ref(),
            ChatCompletionRequestUserMessageContentPart::AudioUrl(_)
        ));
    }

    #[cfg(feature = "mm-routing")]
    #[test]
    fn image_token_aggregate_requires_complete_trustworthy_counts() {
        assert_eq!(aggregate_image_tokens(Some(300), 2, 2, false), Some(300));
        assert_eq!(aggregate_image_tokens(Some(300), 2, 3, false), None);
        assert_eq!(aggregate_image_tokens(Some(300), 2, 2, true), None);
        assert_eq!(aggregate_image_tokens(None, 2, 2, false), None);
        assert_eq!(aggregate_image_tokens(Some(0), 0, 0, false), None);
        assert_eq!(
            checked_add_image_tokens(Some(usize::MAX), 1),
            None,
            "overflow must omit the aggregate rather than wrap"
        );
    }

    #[cfg(feature = "mm-routing")]
    #[test]
    fn image_token_processor_override_is_conservative() {
        assert!(!has_image_token_processor_override(None));
        assert!(!has_image_token_processor_override(Some(
            &serde_json::Value::Null
        )));
        assert!(!has_image_token_processor_override(Some(
            &serde_json::json!({})
        )));
        assert!(has_image_token_processor_override(Some(
            &serde_json::json!([])
        )));
        assert!(has_image_token_processor_override(Some(
            &serde_json::json!({"min_pixels": 64})
        )));
    }

    #[test]
    fn replace_reserved_media_slot_preserves_alignment_and_returns_errors() {
        let mut map = HashMap::from([(
            "image_url".to_string(),
            vec![
                url_entry("http://x/a.png"),
                MultimodalData::UuidOnly("cached-b".to_string()),
                url_entry("http://x/c.png"),
            ],
        )]);

        OpenAIPreprocessor::replace_reserved_media_slot(
            &mut map,
            "image_url",
            2,
            MultimodalData::RawUrl("decoded-c".to_string()),
        )
        .unwrap();

        let images = &map["image_url"];
        assert!(matches!(images[0], MultimodalData::Url(_)));
        assert!(matches!(
            &images[1],
            MultimodalData::UuidOnly(uuid) if uuid == "cached-b"
        ));
        assert!(matches!(
            &images[2],
            MultimodalData::RawUrl(value) if value == "decoded-c"
        ));

        let error = OpenAIPreprocessor::replace_reserved_media_slot(
            &mut map,
            "image_url",
            3,
            MultimodalData::RawUrl("out-of-range".to_string()),
        )
        .expect_err("an out-of-range reserved slot must return an error");
        assert!(error.to_string().contains("image_url[3]"));
    }

    #[test]
    fn routing_priorities_keep_strict_tier_independent() {
        let hints = crate::protocols::common::extensions::AgentHints {
            priority: Some(-3),
            strict_priority: Some(7),
            ..Default::default()
        };

        assert_eq!(
            routing_priorities(Some(&hints)),
            (Some(0.0), Some(7), Some(-3))
        );
        assert_eq!(routing_priorities(None), (None, None, None));
    }

    fn test_llm_metrics_annotation() -> LLMMetricAnnotation {
        LLMMetricAnnotation {
            input_tokens: 10,
            output_tokens: 20,
            chunk_tokens: 3,
            cached_tokens: Some(4),
            image_tokens: Some(512),
            prefill_worker_id: Some(1),
            prefill_dp_rank: Some(2),
            prefill_worker_type: Some("prefill".to_string()),
            decode_worker_id: Some(3),
            decode_dp_rank: Some(4),
            decode_worker_type: Some("decode".to_string()),
            tokenize_latency: Some(std::time::Duration::from_millis(5)),
            detokenize_total_latency: Some(std::time::Duration::from_micros(50)),
            detokenize_count: Some(6),
            ..Default::default()
        }
    }

    #[test]
    fn llm_metrics_from_annotation_recognizes_both_metric_event_tags() {
        // Both the per-chunk `llm_metrics` event and the payload-only `payload_usage`
        // event carry the serialized LLMMetricAnnotation as their comment and must
        // be observed by the metrics collector.
        let base = test_llm_metrics_annotation()
            .to_annotation::<()>()
            .expect("metrics annotation serializes");
        for tag in [ANNOTATION_LLM_METRICS, ANNOTATION_PAYLOAD_USAGE] {
            let tagged = Annotated::<()> {
                id: None,
                data: None,
                event: Some(tag.to_string()),
                comment: base.comment.clone(),
                error: None,
            };
            let metrics = LLMMetricAnnotation::from_annotation(&tagged)
                .expect("metrics annotation parses")
                .unwrap_or_else(|| panic!("metrics recognized for tag {tag}"));
            assert_eq!(metrics.input_tokens, 10);
            assert_eq!(metrics.output_tokens, 20);
            assert_eq!(metrics.image_tokens, Some(512));
            assert_eq!(metrics.detokenize_count, Some(6));
        }
    }

    #[test]
    fn llm_metrics_from_annotation_ignores_untagged_and_other_events() {
        // No event → not metrics (per-chunk metrics are event-tagged again).
        let untagged = Annotated::<()> {
            id: None,
            data: None,
            event: None,
            comment: Some(vec!["{\"input_tokens\":1}".to_string()]),
            error: None,
        };
        assert!(
            LLMMetricAnnotation::from_annotation(&untagged)
                .expect("untagged chunk is not an error")
                .is_none()
        );

        // A different event tag → not metrics.
        let other = Annotated::<()> {
            id: None,
            data: None,
            event: Some(ANNOTATION_TOKEN_IDS.to_string()),
            comment: None,
            error: None,
        };
        assert!(
            LLMMetricAnnotation::from_annotation(&other)
                .expect("other event is not an error")
                .is_none()
        );
    }

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
                Some("kimi_k3"),
                Some("kimi_k3"),
                true,
                "kimi_k3 paired XTML parsers → required",
            ),
            (
                Some("kimi-k3"),
                Some("kimi-k3"),
                true,
                "kimi-k3 aliases → required",
            ),
            (
                None,
                Some("mistral"),
                true,
                "mistral reasoning only → required (`[THINK]` / `[/THINK]` are special tokens)",
            ),
            (
                Some("kimi_k2"),
                None,
                true,
                "kimi_k2 tool-call only → required \
                 (`<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>` are special)",
            ),
            (
                Some("minimax_m3"),
                Some("minimax_m3"),
                true,
                "minimax_m3 paired → required",
            ),
            (
                Some("minimax-m3-nom"),
                Some("minimax-m3"),
                true,
                "MiniMax M3 SGLang aliases → required",
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
        assert!(f(Some(true), Some("kimi_k3"), Some("kimi_k3")));
        assert!(f(Some(true), None, Some("mistral")));
        // false / unset → never (default path keeps the markers)
        assert!(!f(Some(false), Some("harmony"), None));
        assert!(!f(None, Some("harmony"), None));
        // forced-true but parser doesn't need special tokens → fine
        assert!(!f(Some(true), Some("hermes"), None));
        assert!(!f(Some(true), None, None));
    }

    #[test]
    fn test_kimi_k3_jail_always_unwraps_xtml_response_channels() {
        let parser = "kimi_k3".to_string();
        assert!(OpenAIPreprocessor::should_apply_tool_jail(Some(&parser), None, false).unwrap());

        let alias = "kimi-k3".to_string();
        assert!(
            OpenAIPreprocessor::should_apply_tool_jail(
                Some(&alias),
                Some(&ChatCompletionToolChoiceOption::None),
                true,
            )
            .unwrap()
        );
    }

    /// Verifies which force-reasoning parsers use guided-output shape detection.
    #[test]
    fn test_reasoning_before_guided_json_parser_allowlist() {
        for parser in [
            "deepseek_r1",
            "deepseek_v3",
            "deepseek_v3_1",
            "deepseek_v3_2",
            "step3",
            "kimi_k25",
            "mistral",
            "minimax_m2",
            "nemotron_nano",
            "nemotron3",
            "nemotron_v3",
        ] {
            assert!(
                OpenAIPreprocessor::supports_reasoning_before_guided_json(Some(parser)),
                "{parser} should inspect guided output shape"
            );
        }

        assert!(
            !OpenAIPreprocessor::supports_reasoning_before_guided_json(Some(
                "minimax_append_think"
            )),
            "minimax_append_think must retain the guided-JSON bypass"
        );
    }

    /// Verifies parser-specific openers that can be confused with guided JSON.
    #[test]
    fn test_guided_json_reasoning_start_tokens() {
        assert_eq!(
            OpenAIPreprocessor::guided_json_reasoning_start_token(Some("mistral")),
            Some("[THINK]")
        );
        assert_eq!(
            OpenAIPreprocessor::guided_json_reasoning_start_token(Some("nemotron_v3")),
            None
        );
    }

    #[test]
    fn test_prompt_injected_reasoning_start_by_parser() {
        let cases = [
            (
                Some("minimax_m3"),
                Some("...<mm:think>\n"),
                true,
                "MiniMax M3 starts from <mm:think>",
            ),
            (
                Some("minimax-m3"),
                Some("...</mm:think>\n"),
                false,
                "MiniMax M3 prefilled end marker means not in reasoning",
            ),
            (
                Some("minimax_m3"),
                Some("...<think>\n"),
                false,
                "MiniMax M3 must not use generic <think>",
            ),
            (
                Some("qwen3"),
                Some("...<think>\n"),
                true,
                "Qwen-style templated <think> behavior remains",
            ),
            (
                Some("kimi_k3"),
                Some("...<|open|>think<|sep|>\n"),
                true,
                "Kimi K3 starts inside its XTML think channel",
            ),
            (
                Some("kimi-k3"),
                Some("...<think>\n"),
                false,
                "Kimi K3 must not use the generic think marker",
            ),
            (
                Some("deepseek_v4"),
                Some("...<think>\n"),
                true,
                "existing <think> parser behavior remains",
            ),
            (
                None,
                Some("...<think>\n"),
                true,
                "legacy no-parser detection",
            ),
            (Some("minimax_m3"), None, false, "no prompt"),
        ];

        for (parser, prompt, expected, desc) in cases {
            assert_eq!(
                OpenAIPreprocessor::prompt_injected_reasoning_start(parser, prompt),
                expected,
                "FAILED: {desc}",
            );
        }
    }

    #[test]
    fn test_prompt_injected_reasoning_ended_backend_arg_by_parser() {
        let cases = [
            (
                Some("minimax_m2"),
                Some("...<think>\n"),
                Some(false),
                "MiniMax M2 needs native backend reasoning state aligned with the prompt",
            ),
            (
                Some("minimax_m3"),
                Some("...<mm:think>\n"),
                Some(false),
                "MiniMax M3 needs native backend reasoning state aligned with the prompt",
            ),
            (
                Some("kimi_k3"),
                Some("...<|open|>think<|sep|>\n"),
                Some(false),
                "Kimi K3 guided decoding must start after its prompt-opened think channel",
            ),
            (
                Some("deepseek_v4"),
                Some("...<think>\n"),
                None,
                "DeepSeek V4 native guided JSON must not be forced into reasoning mode",
            ),
            (
                Some("deepseek-v4"),
                Some("...<think>\n"),
                None,
                "DeepSeek V4 alias must not receive reasoning_ended=false",
            ),
            (
                Some("qwen3"),
                Some("...<think>\n"),
                None,
                "Qwen-style prompt injection is handled by Dynamo postprocessing only",
            ),
            (
                Some("minimax_m2"),
                Some("plain prompt"),
                None,
                "no injected reasoning opener means no backend state override",
            ),
        ];

        for (parser, prompt, expected, desc) in cases {
            assert_eq!(
                OpenAIPreprocessor::prompt_injected_reasoning_ended_arg(parser, prompt),
                expected,
                "FAILED: {desc}",
            );
        }
    }

    #[test]
    fn test_backend_extra_args_preserves_nvext_and_sampling_extensions() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "detokenize": false,
            "allowed_token_ids": [10, 11],
            "bad_words_token_ids": [[12, 13]],
            "logprob_token_ids": [14, 15],
            "nvext": {
                "cache_salt": "step_7",
                "extra_fields": ["completion_token_ids"],
                "metadata_upload": {
                    "url": "s3://bucket/root/rollouts"
                }
            }
        }))
        .unwrap();

        let extra_args = OpenAIPreprocessor::backend_extra_args(&request, false, None).unwrap();

        assert_eq!(extra_args["nvext"]["cache_salt"], "step_7");
        assert_eq!(
            extra_args["nvext"]["extra_fields"],
            serde_json::json!(["completion_token_ids"])
        );
        assert_eq!(
            extra_args["nvext"]["metadata_upload"],
            serde_json::json!({
                "url": "s3://bucket/root/rollouts"
            })
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
        assert_eq!(
            extra_args["sampling_options"]["logprob_token_ids"],
            serde_json::json!([14, 15])
        );
    }

    fn chat_request_with_args(
        chat_template_args: Option<HashMap<String, serde_json::Value>>,
    ) -> NvCreateChatCompletionRequest {
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .unwrap();
        request.chat_template_args = chat_template_args;
        request
    }

    fn runtime_config_with_default_thinking_mode(
        mode: &str,
    ) -> crate::local_model::runtime_config::ModelRuntimeConfig {
        let mut runtime_config = crate::local_model::runtime_config::ModelRuntimeConfig::new();
        runtime_config
            .set_engine_specific(DEFAULT_THINKING_MODE_RUNTIME_KEY, mode)
            .unwrap();
        runtime_config
    }

    #[test]
    fn test_default_thinking_mode_disabled_adds_template_args() {
        let runtime_config = runtime_config_with_default_thinking_mode("disabled");
        let mut request = chat_request_with_args(None);

        OpenAIPreprocessor::apply_default_thinking_mode_from_runtime_config(
            &runtime_config,
            &mut request,
        );

        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("disabled"))
        );
    }

    #[test]
    fn test_default_thinking_mode_enabled_adds_template_args() {
        let runtime_config = runtime_config_with_default_thinking_mode("enabled");
        let mut request = chat_request_with_args(None);

        OpenAIPreprocessor::apply_default_thinking_mode_from_runtime_config(
            &runtime_config,
            &mut request,
        );

        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("enabled"))
        );
    }

    #[test]
    fn test_gemma4_default_thinking_mode_controls_reasoning_parser() {
        for (mode, expected_disabled) in [("enabled", false), ("disabled", true)] {
            let runtime_config = runtime_config_with_default_thinking_mode(mode);
            let mut request = chat_request_with_args(None);

            OpenAIPreprocessor::apply_default_thinking_mode_from_runtime_config(
                &runtime_config,
                &mut request,
            );

            for parser in ["gemma4", "gemma-4"] {
                assert_eq!(
                    OpenAIPreprocessor::is_reasoning_disabled_by_request(
                        Some(parser),
                        request.chat_template_args.as_ref(),
                    ),
                    expected_disabled,
                    "parser={parser}, default_thinking_mode={mode}",
                );
            }
        }
    }

    #[test]
    fn test_default_thinking_mode_does_not_override_request() {
        let runtime_config = runtime_config_with_default_thinking_mode("disabled");
        let mut request = chat_request_with_args(Some(HashMap::from([(
            "thinking_mode".to_string(),
            serde_json::json!("enabled"),
        )])));

        OpenAIPreprocessor::apply_default_thinking_mode_from_runtime_config(
            &runtime_config,
            &mut request,
        );

        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("enabled"))
        );
        assert!(!args.contains_key("thinking"));
        assert!(!args.contains_key("enable_thinking"));
    }

    #[test]
    fn test_default_thinking_mode_precedes_parser_implicit_default() {
        let runtime_config = runtime_config_with_default_thinking_mode("disabled");
        let mut request = chat_request_with_args(None);

        OpenAIPreprocessor::apply_default_thinking_mode_from_runtime_config(
            &runtime_config,
            &mut request,
        );
        OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("kimi_k25"), None);

        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("disabled"))
        );
    }

    #[test]
    fn test_default_thinking_mode_does_not_override_reasoning_effort() {
        let runtime_config = runtime_config_with_default_thinking_mode("disabled");
        let mut request = chat_request_with_args(Some(HashMap::from([(
            "reasoning_effort".to_string(),
            serde_json::json!("high"),
        )])));

        OpenAIPreprocessor::apply_default_thinking_mode_from_runtime_config(
            &runtime_config,
            &mut request,
        );

        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("reasoning_effort"),
            Some(&serde_json::json!("high"))
        );
        assert!(!args.contains_key("thinking"));
        assert!(!args.contains_key("enable_thinking"));
        assert!(!args.contains_key("thinking_mode"));
    }

    /// Verifies template reasoning controls are forwarded to a configured parser.
    #[test]
    fn test_backend_extra_args_forwards_reasoning_template_args() {
        for enable_thinking in [true, false] {
            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking,
                        "reasoning_effort": "high"
                    }
                }))
                .unwrap();

            let extra_args =
                OpenAIPreprocessor::backend_extra_args(&request, true, Some(false)).unwrap();

            assert_eq!(
                extra_args["reasoning_parser_kwargs"]["chat_template_kwargs"],
                serde_json::json!({
                    "enable_thinking": enable_thinking,
                    "reasoning_effort": "high"
                })
            );
            assert_eq!(extra_args["reasoning_ended"], false);
        }
    }

    /// Verifies parser metadata is omitted when no reasoning parser is configured.
    #[test]
    fn test_backend_extra_args_omits_reasoning_metadata_without_configured_parser() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "detokenize": false,
            "chat_template_kwargs": {
                "enable_thinking": true
            }
        }))
        .unwrap();

        let extra_args =
            OpenAIPreprocessor::backend_extra_args(&request, false, Some(false)).unwrap();

        assert_eq!(extra_args["sampling_options"]["detokenize"], false);
        assert!(extra_args.get("reasoning_parser_kwargs").is_none());
        assert!(extra_args.get("reasoning_ended").is_none());
    }

    /// Verifies no parser metadata is added when template arguments are absent.
    #[test]
    fn test_backend_extra_args_omits_reasoning_metadata_without_template_args() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();

        assert!(OpenAIPreprocessor::backend_extra_args(&request, true, None).is_none());
    }

    /// Verifies the SGLang reasoning gate covers forced tool JSON and
    /// structured assistant output while honoring per-request thinking controls.
    #[test]
    fn test_guided_output_requires_reasoning() {
        let request = |tool_choice: serde_json::Value, enable_thinking: Option<bool>| {
            let mut value = serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "use the tool"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }],
                "tool_choice": tool_choice
            });
            if let Some(enabled) = enable_thinking {
                value["chat_template_kwargs"] = serde_json::json!({
                    "enable_thinking": enabled
                });
            }
            serde_json::from_value::<NvCreateChatCompletionRequest>(value).unwrap()
        };

        let required = request(serde_json::json!("required"), Some(true));
        assert!(OpenAIPreprocessor::guided_output_requires_reasoning(
            &required,
            Some("nemotron_v3")
        ));

        let named = request(
            serde_json::json!({
                "type": "function",
                "function": {"name": "lookup"}
            }),
            None,
        );
        assert!(OpenAIPreprocessor::guided_output_requires_reasoning(
            &named,
            Some("nemotron_v3")
        ));

        let disabled = request(serde_json::json!("required"), Some(false));
        assert!(!OpenAIPreprocessor::guided_output_requires_reasoning(
            &disabled,
            Some("nemotron_v3")
        ));
        assert!(!OpenAIPreprocessor::guided_output_requires_reasoning(
            &required, None
        ));

        let automatic = request(serde_json::json!("auto"), Some(true));
        assert!(!OpenAIPreprocessor::guided_output_requires_reasoning(
            &automatic,
            Some("nemotron_v3")
        ));

        let gemma_without_opt_in = request(serde_json::json!("required"), None);
        assert!(!OpenAIPreprocessor::guided_output_requires_reasoning(
            &gemma_without_opt_in,
            Some("gemma4")
        ));
        let gemma_with_opt_in = request(serde_json::json!("required"), Some(true));
        assert!(OpenAIPreprocessor::guided_output_requires_reasoning(
            &gemma_with_opt_in,
            Some("gemma4")
        ));

        let deepseek_default = request(serde_json::json!("required"), None);
        assert!(OpenAIPreprocessor::guided_output_requires_reasoning(
            &deepseek_default,
            Some("deepseek_v4")
        ));
        let deepseek_disabled = request(serde_json::json!("required"), Some(false));
        assert!(!OpenAIPreprocessor::guided_output_requires_reasoning(
            &deepseek_disabled,
            Some("deepseek_v4")
        ));

        let structured_request = |enable_thinking: bool| {
            serde_json::from_value::<NvCreateChatCompletionRequest>(serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "return json"}],
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "result",
                        "schema": {"type": "object"}
                    }
                }
            }))
            .unwrap()
        };
        let json_object_request = |enable_thinking: bool| {
            serde_json::from_value::<NvCreateChatCompletionRequest>(serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "return json"}],
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                "response_format": {"type": "json_object"}
            }))
            .unwrap()
        };
        assert!(OpenAIPreprocessor::guided_output_requires_reasoning(
            &structured_request(true),
            Some("qwen3")
        ));
        assert!(OpenAIPreprocessor::guided_output_requires_reasoning(
            &json_object_request(true),
            Some("qwen3")
        ));
        assert!(!OpenAIPreprocessor::guided_output_requires_reasoning(
            &structured_request(false),
            Some("qwen3")
        ));
        assert!(!OpenAIPreprocessor::guided_output_requires_reasoning(
            &structured_request(true),
            Some("gpt_oss")
        ));
        assert!(!OpenAIPreprocessor::guided_output_requires_reasoning(
            &json_object_request(true),
            Some("gpt_oss")
        ));
        assert!(!OpenAIPreprocessor::guided_output_requires_reasoning(
            &structured_request(false),
            Some("gpt_oss")
        ));
        assert!(OpenAIPreprocessor::guided_output_requires_reasoning(
            &request(serde_json::json!("required"), None),
            Some("gpt_oss")
        ));
    }

    /// Verifies SGLang's effective reasoning mode for each parser family.
    #[test]
    fn test_sglang_effective_reasoning_enabled() {
        let cases = [
            ("qwen3", serde_json::json!({}), true),
            (
                "qwen3",
                serde_json::json!({"enable_thinking": false}),
                false,
            ),
            ("nemotron_v3", serde_json::json!({}), true),
            ("gemma4", serde_json::json!({}), false),
            ("gemma4", serde_json::json!({"enable_thinking": true}), true),
            ("deepseek_v4", serde_json::json!({}), true),
            ("deepseek_v4", serde_json::json!({"thinking": true}), true),
            (
                "deepseek_v4",
                serde_json::json!({"enable_thinking": true}),
                true,
            ),
            (
                "deepseek_v4",
                serde_json::json!({"thinking_mode": "thinking"}),
                true,
            ),
            (
                "deepseek_v4",
                serde_json::json!({"enable_thinking": false}),
                false,
            ),
            (
                "deepseek_v4",
                serde_json::json!({"thinking_mode": "chat"}),
                false,
            ),
            (
                "deepseek_v4",
                serde_json::json!({"thinking": false, "thinking_mode": "thinking"}),
                false,
            ),
            ("deepseek_v3_2", serde_json::json!({}), true),
            ("deepseek_v3_1", serde_json::json!({}), false),
            (
                "deepseek_v3_1",
                serde_json::json!({"enable_thinking": true}),
                true,
            ),
            ("kimi_k25", serde_json::json!({"thinking": false}), false),
            ("kimi_k3", serde_json::json!({}), true),
            ("kimi-k3", serde_json::json!({"thinking": false}), false),
            ("minimax_m2", serde_json::json!({}), true),
            ("minimax_m2", serde_json::json!({"thinking": false}), false),
            ("mistral", serde_json::json!({}), false),
            (
                "mistral",
                serde_json::json!({"reasoning_effort": "none"}),
                false,
            ),
            (
                "mistral",
                serde_json::json!({"reasoning_effort": "high"}),
                true,
            ),
            ("minimax_m3", serde_json::json!({}), true),
            (
                "minimax_m3",
                serde_json::json!({"thinking_mode": "disabled"}),
                false,
            ),
            ("gpt_oss", serde_json::json!({}), true),
            (
                "gpt_oss",
                serde_json::json!({"enable_thinking": true}),
                true,
            ),
            (
                "gpt_oss",
                serde_json::json!({"enable_thinking": false}),
                true,
            ),
            ("deepseek_r1", serde_json::json!({}), true),
            ("minimax_append_think", serde_json::json!({}), false),
            ("basic", serde_json::json!({}), false),
        ];

        for (parser, request_args, expected) in cases {
            let request_args = serde_json::from_value(request_args).unwrap();
            assert_eq!(
                OpenAIPreprocessor::sglang_effective_reasoning_enabled(
                    Some(parser),
                    Some(&request_args),
                ),
                expected,
                "parser={parser}, args={request_args:?}",
            );
        }
        assert!(!OpenAIPreprocessor::sglang_effective_reasoning_enabled(
            None, None
        ));
    }

    #[test]
    fn test_internal_preserve_omitted_max_tokens_option() {
        assert_eq!(
            OpenAIPreprocessor::omitted_max_tokens_default(
                10,
                Some(100),
                PreprocessRequestOptions::default()
            ),
            Some(90)
        );
        assert_eq!(
            OpenAIPreprocessor::omitted_max_tokens_default(
                10,
                Some(100),
                PreprocessRequestOptions {
                    preserve_omitted_max_tokens: true,
                },
            ),
            None
        );
        assert_eq!(
            OpenAIPreprocessor::omitted_max_tokens_default(
                10,
                None,
                PreprocessRequestOptions::default()
            ),
            None
        );
        assert_eq!(
            OpenAIPreprocessor::omitted_max_tokens_default(
                10,
                Some(0),
                PreprocessRequestOptions::default()
            ),
            Some(0)
        );
    }

    #[test]
    fn test_exact_prompt_len() {
        let images = MultimodalDataMap::from([(
            "image_url".to_string(),
            vec![MultimodalData::Url(
                url::Url::parse("https://example.com/image.png").unwrap(),
            )],
        )]);
        let videos = MultimodalDataMap::from([(
            "video_url".to_string(),
            vec![MultimodalData::Url(
                url::Url::parse("https://example.com/video.mp4").unwrap(),
            )],
        )]);
        let mixed = MultimodalDataMap::from([
            (
                "image_url".to_string(),
                vec![MultimodalData::Url(
                    url::Url::parse("https://example.com/image.png").unwrap(),
                )],
            ),
            (
                "audio_url".to_string(),
                vec![MultimodalData::Url(
                    url::Url::parse("https://example.com/audio.wav").unwrap(),
                )],
            ),
        ]);

        // Image-expanded length present: use it instead of placeholder tokens.
        assert_eq!(
            OpenAIPreprocessor::exact_prompt_len(Some(500), Some(&images), 12),
            Some(500)
        );
        // Expanded length 0 (serde-default / absent) with images: defer to backend.
        assert_eq!(
            OpenAIPreprocessor::exact_prompt_len(Some(0), Some(&images), 12),
            None
        );
        // No routing info but images present: defer to backend.
        assert_eq!(
            OpenAIPreprocessor::exact_prompt_len(None, Some(&images), 12),
            None
        );
        // Video/audio expansion is backend-owned, including mixed requests
        // whose routing metadata expands only the image portion.
        assert_eq!(
            OpenAIPreprocessor::exact_prompt_len(None, Some(&videos), 12),
            None
        );
        assert_eq!(
            OpenAIPreprocessor::exact_prompt_len(Some(500), Some(&mixed), 12),
            None
        );
        // Text-only: use the token count.
        assert_eq!(
            OpenAIPreprocessor::exact_prompt_len(None, None, 12),
            Some(12)
        );
        // Expanded length 0 without media: fall back to the token count.
        assert_eq!(
            OpenAIPreprocessor::exact_prompt_len(Some(0), None, 12),
            Some(12)
        );
    }

    fn token_budget(
        combined_limit: u32,
        reject_prompt_overflow: bool,
        reject_total_overflow: bool,
    ) -> TokenBudget {
        TokenBudget {
            combined_limit,
            reject_prompt_overflow,
            reject_total_overflow,
        }
    }

    #[test]
    fn test_requested_token_budget_policy_matrix() {
        let cases = [
            // Exact combined limit is accepted.
            (3, Some(7), true, true, false),
            // One token beyond the combined limit is rejected.
            (3, Some(8), true, true, true),
            // A full prompt leaves no room for generation.
            (10, None, true, true, true),
            // Each rejection dimension can be delegated independently.
            (3, Some(8), true, false, false),
            (10, Some(8), false, true, false),
        ];

        for (
            prompt_len,
            max_tokens,
            reject_prompt_overflow,
            reject_total_overflow,
            should_reject,
        ) in cases
        {
            let budget = token_budget(10, reject_prompt_overflow, reject_total_overflow);
            assert_eq!(
                OpenAIPreprocessor::validate_requested_token_budget(
                    prompt_len,
                    max_tokens,
                    Some(&budget),
                )
                .is_err(),
                should_reject,
            );
        }
    }

    fn preprocessed_budget_request(max_tokens: Option<u32>) -> PreprocessedRequest {
        let stop_conditions = crate::protocols::common::StopConditions {
            max_tokens,
            ..Default::default()
        };

        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(stop_conditions)
            .sampling_options(crate::protocols::common::SamplingOptions::default())
            .output_options(crate::protocols::common::OutputOptions::default())
            .build()
            .unwrap()
    }

    #[test]
    fn attach_agent_context_forwards_compaction() {
        let agent_context = AgentContext {
            session_id: "codex-thread".to_string(),
            parent_session_id: None,
            session_final: None,
            compaction: Some(AgentCompaction {
                trigger: Some("manual".to_string()),
                ..Default::default()
            }),
            kv_hints: None,
            input_trigger: None,
        };
        let mut context = PipelineContext::new(());
        context.insert(AGENT_CONTEXT_CONTEXT_KEY, agent_context.clone());
        let mut request = preprocessed_budget_request(None);

        attach_agent_context_from_context(&mut request, &context);

        assert_eq!(request.agent_context.as_ref(), Some(&agent_context));
        let wire = serde_json::to_value(&request).unwrap();
        assert_eq!(
            wire["agent_context"]["compaction"]["trigger"],
            serde_json::json!("manual")
        );
    }

    #[test]
    fn test_preprocessed_completion_budget_validation_and_deferral() {
        let text_request = preprocessed_budget_request(Some(8));
        let reject = token_budget(10, true, true);
        assert!(
            OpenAIPreprocessor::validate_preprocessed_token_budget(&text_request, Some(&reject))
                .is_err()
        );

        let defer_total = token_budget(10, true, false);
        assert_eq!(
            OpenAIPreprocessor::validate_preprocessed_token_budget(
                &text_request,
                Some(&defer_total),
            )
            .unwrap(),
            Some(3)
        );

        // Prompt embeddings do not expose their sequence length as token_ids.
        let mut embeddings_request = text_request.clone();
        embeddings_request.prompt_embeds = Some("opaque-tensor".to_string());
        assert_eq!(
            OpenAIPreprocessor::validate_preprocessed_token_budget(
                &embeddings_request,
                Some(&reject),
            )
            .unwrap(),
            None
        );
    }

    struct UnreachableCompletionBackend;

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>
        for UnreachableCompletionBackend
    {
        async fn generate(
            &self,
            _request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<BackendOutput>>, Error> {
            panic!("over-budget completion must be rejected before backend dispatch")
        }
    }

    #[tokio::test]
    async fn test_completion_operator_rejects_token_budget_overflow() {
        let mut mdc = ModelDeploymentCard::load_from_disk(
            "tests/data/sample-models/mock-llama-3.1-8b-instruct",
            None,
        )
        .unwrap();
        mdc.runtime_config.context_length = Some(100);
        mdc.runtime_config
            .set_engine_specific(TOKEN_BUDGET_RUNTIME_KEY, token_budget(10, true, true))
            .unwrap();
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();

        let request = NvCreateCompletionRequest {
            inner: dynamo_protocols::types::CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: dynamo_protocols::types::Prompt::IntegerArray(vec![1, 2, 3]),
                max_tokens: Some(8),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        };
        let next: Arc<
            dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>,
        > = Arc::new(UnreachableCompletionBackend);

        let result =
            Operator::generate(preprocessor.as_ref(), PipelineContext::new(request), next).await;
        let Err(err) = result else {
            panic!("over-budget completion should fail admission");
        };
        let dynamo_err = err
            .downcast_ref::<DynamoError>()
            .expect("error should preserve the DynamoError type");
        assert_eq!(dynamo_err.error_type(), ErrorType::InvalidArgument);
    }

    fn test_prompt_formatter(template: &str) -> Arc<dyn OAIPromptFormatter> {
        let template: dynamo_renderer::ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": template
        }))
        .unwrap();
        match dynamo_renderer::PromptFormatter::from_parts(
            template,
            dynamo_renderer::ContextMixins::default(),
            false,
        )
        .unwrap()
        {
            dynamo_renderer::PromptFormatter::OAI(formatter) => formatter,
        }
    }

    fn assistant_only_request() -> NvCreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "assistant", "content": "prefill"}]
        }))
        .unwrap()
    }

    const REQUIRES_USER_TEMPLATE: &str = "\
        {% set ns = namespace(has_user=false) %}\
        {% for message in messages %}\
            {% if message['role'] == 'user' %}{% set ns.has_user = true %}{% endif %}\
        {% endfor %}\
        {% if not ns.has_user %}{{ raise_exception('No user query found in messages.') }}{% endif %}\
        {{ messages[0]['content'] }}";

    fn render_through_preprocessor(
        formatter: &dyn OAIPromptFormatter,
        request: &dyn OAIChatLikeRequest,
    ) -> Result<RenderedPrompt> {
        formatter
            .render_prompt(request)
            .map_err(OpenAIPreprocessor::map_prompt_render_error)
    }

    #[test]
    fn test_assistant_only_request_accepted_when_template_accepts_it() {
        let formatter = test_prompt_formatter(
            "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}{% endfor %}",
        );

        let rendered =
            render_through_preprocessor(formatter.as_ref(), &assistant_only_request()).unwrap();

        assert_eq!(rendered.as_str(), "assistant:prefill");
    }

    #[test]
    fn test_assistant_only_template_error_is_invalid_argument() {
        let formatter = test_prompt_formatter(REQUIRES_USER_TEMPLATE);

        let error = render_through_preprocessor(formatter.as_ref(), &assistant_only_request())
            .context("Failed to apply prompt template")
            .unwrap_err();
        let dynamo_error = error
            .chain()
            .find_map(|cause| cause.downcast_ref::<DynamoError>())
            .expect("template render error should be classified as a DynamoError");

        assert_eq!(dynamo_error.error_type(), ErrorType::InvalidArgument);
        assert!(
            dynamo_error
                .message()
                .contains("No user query found in messages.")
        );
    }

    #[test]
    fn test_restrictive_template_accepts_request_with_user_message() {
        let formatter = test_prompt_formatter(REQUIRES_USER_TEMPLATE);
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let rendered = render_through_preprocessor(formatter.as_ref(), &request).unwrap();

        assert_eq!(rendered.as_str(), "hello");
    }

    #[test]
    fn test_kimi_thinking_normalization_keeps_template_and_gates_in_sync() {
        let template: dynamo_renderer::ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": "{% if thinking == true and enable_thinking == true %}<think>{% elif thinking == false and enable_thinking == false %}<chat>{% else %}<mismatch>{% endif %}"
        }))
        .unwrap();
        let formatter = dynamo_renderer::PromptFormatter::from_parts(
            template,
            dynamo_renderer::ContextMixins::default(),
            false,
        )
        .unwrap();
        let cases = [
            (serde_json::json!(true), true, "true"),
            (serde_json::json!("true"), true, "string true"),
            (serde_json::json!("TRUE"), true, "uppercase true"),
            (serde_json::json!("1"), true, "string one"),
            (serde_json::json!("yes"), true, "yes"),
            (serde_json::json!("on"), true, "on"),
            (serde_json::json!(1), true, "one"),
            (serde_json::json!(false), false, "false"),
            (serde_json::json!("false"), false, "string false"),
            (serde_json::json!("no"), false, "no"),
            (serde_json::json!("off"), false, "off"),
            (serde_json::json!(0), false, "zero"),
        ];

        let assert_case =
            |key: Option<&str>, value: Option<&serde_json::Value>, expected, description| {
                let mut request: NvCreateChatCompletionRequest =
                    serde_json::from_value(serde_json::json!({
                        "messages": [{"role": "user", "content": "hello"}],
                        "model": "moonshotai/Kimi-K2.5-Instruct"
                    }))
                    .unwrap();
                if let (Some(key), Some(value)) = (key, value) {
                    request.chat_template_args = Some(std::collections::HashMap::from([(
                        key.to_string(),
                        value.clone(),
                    )]));
                }

                OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("kimi_k25"), None);
                let args = request.chat_template_args.as_ref();
                assert_eq!(
                    dynamo_renderer::thinking_bool_from_args(args),
                    Some(expected),
                    "renderer helper mismatch for {description}"
                );
                assert_eq!(
                    OpenAIPreprocessor::sglang_effective_reasoning_enabled(Some("kimi_k25"), args),
                    expected,
                    "SGLang gate mismatch for {description}"
                );
                assert_eq!(
                    !OpenAIPreprocessor::is_reasoning_disabled_by_request(Some("kimi_k25"), args,),
                    expected,
                    "postprocessor gate mismatch for {description}"
                );

                let rendered = match &formatter {
                    dynamo_renderer::PromptFormatter::OAI(formatter) => {
                        formatter.render(&request).unwrap()
                    }
                };
                assert_eq!(
                    rendered,
                    if expected { "<think>" } else { "<chat>" },
                    "rendered prompt mismatch for {description}"
                );
            };

        for key in ["thinking", "enable_thinking"] {
            for (value, expected, description) in &cases {
                assert_case(Some(key), Some(value), *expected, *description);
            }
        }
        assert_case(None, None, true, "omitted Kimi default");

        for parser in ["kimi_k3", "kimi-k3"] {
            let mut request: NvCreateChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "messages": [{"role": "user", "content": "hello"}],
                    "model": "moonshotai/Kimi-K3"
                }))
                .unwrap();
            OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some(parser), None);
            assert_eq!(
                dynamo_renderer::thinking_bool_from_args(request.chat_template_args.as_ref()),
                Some(true),
                "{parser} should default to thinking mode",
            );
        }

        let mut conflicting_request: NvCreateChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "messages": [{"role": "user", "content": "hello"}],
                "model": "moonshotai/Kimi-K2.5-Instruct",
                "chat_template_kwargs": {
                    "thinking": "true",
                    "enable_thinking": false
                }
            }))
            .unwrap();
        OpenAIPreprocessor::normalize_thinking_arg(
            &mut conflicting_request,
            Some("kimi_k25"),
            None,
        );
        let args = conflicting_request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&serde_json::Value::Bool(true)));
        assert_eq!(
            args.get("enable_thinking"),
            Some(&serde_json::Value::Bool(true))
        );
        let rendered = match &formatter {
            dynamo_renderer::PromptFormatter::OAI(formatter) => {
                formatter.render(&conflicting_request).unwrap()
            }
        };
        assert_eq!(rendered, "<think>");
    }

    #[test]
    fn test_named_kimi_k3_normalizes_every_reasoning_consumer_to_disabled() {
        for parser in ["kimi_k3", "kimi-k3"] {
            let mut request: NvCreateChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "messages": [{"role": "user", "content": "Weather in Berlin?"}],
                    "model": "moonshotai/Kimi-K3",
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}}
                            }
                        }
                    }],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "get_weather"}
                    },
                    "reasoning_effort": "high",
                    "chat_template_args": {
                        "thinking": true,
                        "enable_thinking": true
                    }
                }))
                .unwrap();

            request.normalize_reasoning_template_args().unwrap();
            OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some(parser), None);
            OpenAIPreprocessor::normalize_kimi_k3_named_tool_choice(&mut request, Some(parser));

            let args = request.chat_template_args.as_ref().unwrap();
            assert_eq!(args.get("thinking"), Some(&serde_json::Value::Bool(false)));
            assert_eq!(
                args.get("enable_thinking"),
                Some(&serde_json::Value::Bool(false))
            );
            assert_eq!(
                args.get("reasoning_effort"),
                Some(&serde_json::Value::String("high".to_string())),
                "the public effort value is preserved while the named-call exception disables thinking"
            );
            assert!(OpenAIPreprocessor::is_reasoning_disabled_by_request(
                Some(parser),
                Some(args)
            ));
        }
    }

    #[test]
    fn test_named_kimi_k3_override_is_scoped_to_k3_tool_parser() {
        let request_json = serde_json::json!({
            "messages": [{"role": "user", "content": "Weather?"}],
            "model": "test",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"}
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"}
            },
            "chat_template_args": {"thinking": true}
        });

        for parser in [None, Some("hermes"), Some("kimi_k2")] {
            let mut request: NvCreateChatCompletionRequest =
                serde_json::from_value(request_json.clone()).unwrap();
            OpenAIPreprocessor::normalize_kimi_k3_named_tool_choice(&mut request, parser);
            assert_eq!(
                dynamo_renderer::thinking_bool_from_args(request.chat_template_args.as_ref()),
                Some(true),
                "parser {parser:?} must retain its existing policy"
            );
        }
    }

    fn minimax_m3_request(body: serde_json::Value) -> NvCreateChatCompletionRequest {
        let mut base = serde_json::json!({
            "model": "MiniMaxAI/MiniMax-M3",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let base_obj = base.as_object_mut().unwrap();
        for (k, v) in body.as_object().unwrap() {
            base_obj.insert(k.clone(), v.clone());
        }
        serde_json::from_value(base).unwrap()
    }

    #[test]
    fn test_normalize_thinking_arg_m3_defaults_disabled_for_json_schema() {
        for parser in ["minimax_m3", "minimax-m3"] {
            let mut request = minimax_m3_request(serde_json::json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "s", "schema": {"type": "object"}}
                }
            }));
            OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some(parser), None);
            let args = request.chat_template_args.as_ref().unwrap();
            assert_eq!(
                args.get("thinking_mode"),
                Some(&serde_json::json!("disabled"))
            );
            assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
            assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
        }
    }

    #[test]
    fn test_normalize_thinking_arg_m3_defaults_disabled_for_tool_choice_required() {
        let mut request = minimax_m3_request(serde_json::json!({
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "tool_choice": "required"
        }));
        OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("disabled"))
        );
    }

    #[test]
    fn test_normalize_thinking_arg_m3_defaults_disabled_for_named_tool_choice() {
        let mut request = minimax_m3_request(serde_json::json!({
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "f"}}
        }));
        OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("disabled"))
        );
    }

    #[test]
    fn test_normalize_thinking_arg_m3_tool_call_only_deployment_gets_default() {
        // Tool-call-only M3 deployments still render the M3 template.
        // Guided-output requests therefore still need "adaptive" overridden.
        for tool_call_parser in [
            "minimax_m3",
            "minimax-m3",
            "minimax_m3_nom",
            "minimax-m3-nom",
        ] {
            let mut request = minimax_m3_request(serde_json::json!({
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                "tool_choice": "required"
            }));
            OpenAIPreprocessor::normalize_thinking_arg(&mut request, None, Some(tool_call_parser));
            let args = request.chat_template_args.as_ref().unwrap();
            assert_eq!(
                args.get("thinking_mode"),
                Some(&serde_json::json!("disabled")),
                "tool_call_parser={tool_call_parser:?} + tool_choice=required with no reasoning parser must still default thinking_mode=disabled"
            );
        }
    }

    #[test]
    fn test_normalize_thinking_arg_m3_preserves_client_adaptive() {
        let mut request = minimax_m3_request(serde_json::json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "s", "schema": {"type": "object"}}
            },
            "chat_template_kwargs": {"thinking_mode": "adaptive"}
        }));
        OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("adaptive"))
        );
        assert!(!args.contains_key("thinking"));
        assert!(!args.contains_key("enable_thinking"));
    }

    #[test]
    fn test_normalize_thinking_arg_m3_bridges_thinking_bool_to_mode() {
        let mut request = minimax_m3_request(serde_json::json!({
            "chat_template_kwargs": {"thinking": true}
        }));
        OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("enabled"))
        );
    }

    #[test]
    fn test_normalize_thinking_arg_m3_bridges_enable_thinking_to_mode() {
        let mut request = minimax_m3_request(serde_json::json!({
            "chat_template_kwargs": {"enable_thinking": false}
        }));
        OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("disabled"))
        );
    }

    #[test]
    fn test_normalize_thinking_arg_thinking_mode_enabled_is_thinking_on() {
        // `thinking_mode` strings must bypass `is_truthy`, which would map
        // "enabled" to false. M3 inputs are canonicalized to exact values.
        for value in ["enabled", "ENABLED", "Enabled"] {
            let mut request = minimax_m3_request(serde_json::json!({
                "chat_template_kwargs": {"thinking_mode": value}
            }));
            OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
            let args = request.chat_template_args.as_ref().unwrap();
            assert_eq!(
                args.get("thinking"),
                Some(&serde_json::json!(true)),
                "thinking_mode={value:?} must map to thinking=true"
            );
            assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
            assert_eq!(
                args.get("thinking_mode"),
                Some(&serde_json::json!("enabled"))
            );
        }
    }

    #[test]
    fn test_normalize_thinking_arg_thinking_mode_disabled_is_thinking_off() {
        for value in ["disabled", "DISABLED"] {
            let mut request = minimax_m3_request(serde_json::json!({
                "chat_template_kwargs": {"thinking_mode": value}
            }));
            OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
            let args = request.chat_template_args.as_ref().unwrap();
            assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
            assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
            assert_eq!(
                args.get("thinking_mode"),
                Some(&serde_json::json!("disabled"))
            );
        }
    }

    #[test]
    fn test_normalize_thinking_arg_m3_canonicalizes_non_canonical_thinking_mode() {
        // M3 only treats exact "disabled" as off.
        // Canonicalize falsy aliases so `thinking=false` and the template gate agree.
        for value in [
            serde_json::json!(false),
            serde_json::json!(0),
            serde_json::json!("false"),
            serde_json::json!("no"),
            serde_json::json!("off"),
        ] {
            let mut request = minimax_m3_request(serde_json::json!({
                "chat_template_kwargs": {"thinking_mode": value.clone()}
            }));
            OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
            let args = request.chat_template_args.as_ref().unwrap();
            assert_eq!(
                args.get("thinking_mode"),
                Some(&serde_json::json!("disabled")),
                "input={value} must canonicalize to \"disabled\""
            );
            assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
            assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
        }

        // Truthy non-canonical values must canonicalize to "enabled".
        for value in [
            serde_json::json!(true),
            serde_json::json!(1),
            serde_json::json!("true"),
            serde_json::json!("yes"),
            serde_json::json!("on"),
        ] {
            let mut request = minimax_m3_request(serde_json::json!({
                "chat_template_kwargs": {"thinking_mode": value.clone()}
            }));
            OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
            let args = request.chat_template_args.as_ref().unwrap();
            assert_eq!(
                args.get("thinking_mode"),
                Some(&serde_json::json!("enabled")),
                "input={value} must canonicalize to \"enabled\""
            );
            assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
            assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
        }
    }

    #[test]
    fn test_normalize_thinking_arg_deepseek_thinking_mode_words_preserved() {
        // DeepSeek V3.1 uses `thinking_mode` values "chat" and "thinking".
        // Preserve them verbatim without writing boolean aliases.
        for value in ["thinking", "chat"] {
            let mut request: NvCreateChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": "deepseek/DeepSeek-V3.1",
                    "messages": [{"role": "user", "content": "hi"}],
                    "chat_template_kwargs": {"thinking_mode": value}
                }))
                .unwrap();
            OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("deepseek_v3_1"), None);
            let args = request.chat_template_args.as_ref().unwrap();
            assert_eq!(
                args.get("thinking_mode").and_then(|v| v.as_str()),
                Some(value),
                "DeepSeek {value:?} must survive verbatim"
            );
            assert!(
                !args.contains_key("thinking"),
                "no boolean coercion for DeepSeek {value:?}"
            );
            assert!(!args.contains_key("enable_thinking"));
        }
    }

    #[test]
    fn test_normalize_thinking_arg_thinking_mode_not_written_for_non_m3_parsers() {
        // Only M3-family templates use the "enabled"/"disabled" vocabulary.
        // Other parsers may treat an injected value as out-of-vocabulary.
        for parser in [
            "deepseek_v3_1",
            "deepseek_v3_2",
            "minimax_m2",
            "qwen3",
            "kimi_k25",
            "kimi_k3",
        ] {
            let mut request: NvCreateChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "chat_template_kwargs": {"thinking": true}
                }))
                .unwrap();
            OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some(parser), None);
            let args = request.chat_template_args.as_ref().unwrap();
            assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
            assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
            assert!(
                !args.contains_key("thinking_mode"),
                "parser {parser:?} must not receive an injected thinking_mode value"
            );
        }
    }

    #[test]
    fn test_normalize_thinking_arg_kimi_or_else_does_not_write_thinking_mode() {
        // Kimi parsers synthesize `Some(true)` with no client input.
        // Write boolean aliases only; Kimi does not read `thinking_mode`.
        for parser in ["kimi_k25", "kimi_k3", "kimi-k3"] {
            let mut request: NvCreateChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}]
                }))
                .unwrap();
            OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some(parser), None);
            let args = request.chat_template_args.as_ref().unwrap();
            assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
            assert!(!args.contains_key("thinking_mode"));
        }
    }

    #[test]
    fn test_normalize_thinking_arg_m3_leaves_plain_chat_untouched() {
        let mut request = minimax_m3_request(serde_json::json!({}));
        OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        assert!(request.chat_template_args.is_none());
    }

    #[test]
    fn test_normalize_thinking_arg_non_m3_adaptive_is_no_op() {
        // Preserve non-M3 `thinking_mode: "adaptive"` verbatim by leaving
        // normalization as `None`.
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"thinking_mode": "adaptive"}
            }))
            .unwrap();
        OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some("qwen3"), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("adaptive"))
        );
        assert!(!args.contains_key("thinking"));
        assert!(!args.contains_key("enable_thinking"));
    }

    #[test]
    fn test_normalize_thinking_arg_operator_default_forces_m3_constrained_disabled() {
        // Runtime defaults are operator policy, not client intent. When the
        // client sends no thinking control, constrained M3 requests force
        // "disabled" even if the operator default is "enabled" — otherwise
        // the deployment default silently propagates a mode M3 cannot honor
        // with a constrained decoder attached.
        let runtime_config = runtime_config_with_default_thinking_mode("enabled");
        let mut request = minimax_m3_request(serde_json::json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "s", "schema": {"type": "object"}}
            }
        }));
        let thinking_control_from_client =
            OpenAIPreprocessor::request_has_client_thinking_control(&request);
        assert!(
            !thinking_control_from_client,
            "no chat_template_kwargs → not client intent"
        );
        OpenAIPreprocessor::apply_default_thinking_mode_from_runtime_config(
            &runtime_config,
            &mut request,
        );
        OpenAIPreprocessor::normalize_thinking_arg_with_source(
            &mut request,
            Some("minimax_m3"),
            None,
            thinking_control_from_client,
        );
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("disabled"))
        );
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
    }

    #[test]
    fn test_normalize_thinking_arg_m3_preserves_explicit_thinking_for_constrained() {
        // Explicit client thinking intent is respected for constrained M3
        // requests. The runtime attempts the combination even though the
        // template's adaptive/enabled path is known to conflict with the
        // constrained decoder — this is the client's opt-in, and returning
        // possibly-degraded content is preferred to a hard reject at the
        // API boundary (which would break provider-compatibility probes
        // that expect the request to be attempted).
        for (body, expected_mode, desc) in [
            (
                serde_json::json!({
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "s", "schema": {"type": "object"}}
                    },
                    "chat_template_kwargs": {"thinking_mode": "enabled"}
                }),
                "enabled",
                "json_schema + explicit thinking_mode=enabled",
            ),
            (
                serde_json::json!({
                    "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                    "tool_choice": "required",
                    "chat_template_kwargs": {"thinking_mode": "adaptive"}
                }),
                "adaptive",
                "required tool + explicit thinking_mode=adaptive",
            ),
            (
                serde_json::json!({
                    "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                    "tool_choice": {"type": "function", "function": {"name": "f"}},
                    "chat_template_kwargs": {"thinking": true}
                }),
                "enabled",
                "named tool + explicit thinking=true",
            ),
        ] {
            let mut request = minimax_m3_request(body);
            let thinking_control_from_client =
                OpenAIPreprocessor::request_has_client_thinking_control(&request);
            assert!(
                thinking_control_from_client,
                "{desc} must be classified as client thinking intent"
            );
            OpenAIPreprocessor::normalize_thinking_arg_with_source(
                &mut request,
                Some("minimax_m3"),
                None,
                thinking_control_from_client,
            );
            let args = request.chat_template_args.as_ref().unwrap();
            assert_eq!(
                args.get("thinking_mode"),
                Some(&serde_json::json!(expected_mode)),
                "{desc} must preserve explicit client thinking"
            );
        }
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
        let thinking_mode_disabled = {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "thinking_mode".to_string(),
                serde_json::Value::String("disabled".to_string()),
            );
            m
        };
        let reasoning_effort_none = {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "reasoning_effort".to_string(),
                serde_json::Value::String("none".to_string()),
            );
            m
        };
        let reasoning_effort_high = {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "reasoning_effort".to_string(),
                serde_json::Value::String("high".to_string()),
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
            (
                Some("kimi_k3"),
                Some(&thinking_false),
                true,
                "kimi_k3 + thinking=false → disabled",
            ),
            (
                Some("kimi-k3"),
                Some(&thinking_true),
                false,
                "kimi-k3 + thinking=true → enabled",
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
                Some("deepseek_v3"),
                None,
                true,
                "deepseek_v3 + no args → disabled",
            ),
            (
                Some("deepseek_v3"),
                Some(&thinking_true),
                false,
                "deepseek_v3 + thinking=true → enabled",
            ),
            (
                Some("deepseek_v3_1"),
                Some(&enable_thinking_true),
                false,
                "deepseek_v3_1 + enable_thinking=true → enabled",
            ),
            (
                Some("deepseek_v3_2"),
                None,
                false,
                "deepseek_v3_2 + no args → enabled",
            ),
            (
                Some("deepseek_v3_2"),
                Some(&thinking_false),
                true,
                "deepseek_v3_2 + thinking=false → disabled",
            ),
            (
                Some("minimax_m2"),
                Some(&thinking_false),
                true,
                "minimax_m2 + thinking=false → disabled",
            ),
            (
                Some("minimax_m2"),
                Some(&thinking_true),
                false,
                "minimax_m2 + thinking=true → enabled",
            ),
            (
                Some("minimax_m2"),
                None,
                false,
                "minimax_m2 + no args → enabled",
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
                false,
                "nemotron3 + force_nonempty_content=true → NOT disabled (parser stays on)",
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
                false,
                "nemotron_v3 + force_nonempty_content=true → NOT disabled (parser stays on)",
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
                true,
                "gemma4 + no args → disabled (reasoning is opt-in)",
            ),
            (
                Some("gemma-4"),
                Some(&enable_thinking_false),
                true,
                "gemma-4 (hyphen alias) + enable_thinking=false → disabled",
            ),
            (
                Some("gemma-4"),
                None,
                true,
                "gemma-4 (hyphen alias) + no args → disabled (reasoning is opt-in)",
            ),
            (Some("mistral"), None, true, "mistral + no args → disabled"),
            (
                Some("mistral"),
                Some(&reasoning_effort_none),
                true,
                "mistral + reasoning_effort=none → disabled",
            ),
            (
                Some("mistral"),
                Some(&reasoning_effort_high),
                false,
                "mistral + reasoning_effort=high → enabled",
            ),
            (
                Some("minimax_m3"),
                Some(&thinking_mode_disabled),
                true,
                "minimax_m3 + thinking_mode=disabled → disabled",
            ),
            (
                Some("minimax-m3"),
                Some(&thinking_mode_disabled),
                true,
                "minimax-m3 + thinking_mode=disabled → disabled",
            ),
            (
                Some("minimax_m3"),
                Some(&thinking_mode_thinking),
                false,
                "minimax_m3 + thinking_mode=thinking → enabled",
            ),
            (
                Some("minimax_m3"),
                None,
                false,
                "minimax_m3 + no args → enabled",
            ),
        ];

        // The disable decision no longer depends on streaming — `enable_thinking`
        // and the per-family signals decide it identically for both paths.
        for (parser, args, expected, desc) in cases {
            assert_eq!(
                OpenAIPreprocessor::is_reasoning_disabled_by_request(parser, args),
                expected,
                "FAILED: {desc}",
            );
        }

        // force_nonempty_content=true does NOT disable reasoning parsing for
        // either path: the parser stays on so reasoning is split from the answer
        // (no leak). Non-streaming additionally moves reasoning into content when
        // no content was generated (the aggregator); streaming skips that move.
        assert!(
            !OpenAIPreprocessor::is_reasoning_disabled_by_request(
                Some("nemotron3"),
                Some(&force_nonempty_content_true),
            ),
            "nemotron3 + force_nonempty_content=true → NOT disabled",
        );
        assert!(
            !OpenAIPreprocessor::is_reasoning_disabled_by_request(
                Some("nemotron_v3"),
                Some(&force_nonempty_content_true),
            ),
            "nemotron_v3 + force_nonempty_content=true → NOT disabled",
        );
        // enable_thinking=false disables entirely (user turned thinking off).
        assert!(
            OpenAIPreprocessor::is_reasoning_disabled_by_request(
                Some("nemotron3"),
                Some(&enable_thinking_false),
            ),
            "nemotron3 + enable_thinking=false → disabled",
        );

        // The force_nonempty_content=true → NOT disabled behavior is what lets a
        // reasoning-only non-streaming turn surface reasoning as content; verify
        // the aggregator half in test_move_reasoning_to_content_when_empty.
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
