// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool calls routed through the `dynamo-parsers-v2` streaming parser, bypassing the jail.
//!
//! Gated behind
//! [`DYN_ENABLE_EXPERIMENTAL_PARSERS_V2`](dynamo_runtime::config::environment_names::llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2).
//! When enabled, the families in `V2_FAMILIES` (Qwen3-Coder, DeepSeek-V4) stream
//! straight through their `dynamo_parsers_v2` parser instead of
//! `JailedStream`: the v2 parser owns incremental
//! tool-call emission and drops a parameter value truncated at EOF rather than
//! guessing it. The jail is never built for these families in either path
//! (`apply_stream` for streaming, `parse_complete` for batch). Other families, and
//! non-`auto` tool_choice, keep the v1 jail / aggregate-finalize path. The parser is
//! selected by family name via `dynamo_parsers_v2::create_tool_parser_for_family`, so
//! adding a family is a one-line change here plus support in that crate.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use async_stream::stream;
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
    ChatCompletionToolChoiceOption, FinishReason, FunctionCallStream, FunctionType,
};
use dynamo_runtime::config::{env_is_truthy, environment_names::llm as env_llm};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use uuid::Uuid;

use dynamo_parsers::tool_calling::{
    CalledFunction, ToolCallResponse, ToolCallType, ToolDefinition,
};
use dynamo_parsers_v2::{
    Tool as ToolV2, ToolCallDelta, ToolParser, UnifiedEvent, UnifiedParser, UnifiedParserEvent,
    UnifiedParserExt, create_tool_parser_for_family, create_unified_parser_for_family,
};

use super::{NvCreateChatCompletionStreamResponse, stream_choice_chunk_from_template};

// TODO: when glm47 is added here AND DYN_ENABLE_EXPERIMENTAL_PARSERS_V2 is set,
// port the streaming <tool_call> truncation recovery from apply_tool_calling_jail
// (preprocessor.rs) to tool_parser_v2::apply_stream. The v2 path skips the jail
// entirely, so the ChoiceRecovery buffer and finish_reason=length synthetic-chunk
// logic will not run. The aggregator.rs (non-streaming) half is parser-agnostic
// and keeps working on both paths — only the streaming side needs porting.
/// Tool-call families with a `dynamo-parsers-v2` parser wired into both the batch and
/// the streaming path. Must stay a subset of the families
/// `dynamo_parsers_v2::create_tool_parser_for_family` accepts; the strings match
/// dynamo's `tool_call_parser` names so a parser name maps straight to a v2 family.
pub(crate) const V2_FAMILIES: &[&str] = &["qwen3_coder", "deepseek_v4"];

/// Whether the experimental v2 tool-parser routing is enabled. Read once from
/// [`DYN_ENABLE_EXPERIMENTAL_PARSERS_V2`](env_llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2) —
/// env vars are fixed for the process lifetime, so the result is cached.
pub(crate) fn enabled() -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| env_is_truthy(env_llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2));
    *ENABLED
}

/// Whether `family` has a v2 parser and should bypass the v1 jail when [`enabled`].
pub(crate) fn supports_family(family: &str) -> bool {
    V2_FAMILIES.contains(&family)
}

/// Families served by the v2 UNIFIED parser (reasoning + content + tool calls in
/// ONE ordered pass), default-on — no `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` gate.
/// Muse has no usable v1 reasoning parser (the v1 crate dropped the variant, so
/// `get_reasoning_parser_from_name` falls back to `Basic`, which cannot read the
/// `to=self<|message|>` grammar), so the unified pass is the only correct path.
/// Strings match dynamo's parser names.
pub(crate) const UNIFIED_FAMILIES: &[&str] = &["muse_glimmer", "muse-glimmer"];

/// The parser names that route through the muse unified pass. Public accessor for
/// [`UNIFIED_FAMILIES`] so the Python bindings can add muse to the selectable
/// tool-call and reasoning parser names — fc's v1 registries dropped muse, so this
/// is the only source of truth for the unified names.
pub fn unified_family_names() -> &'static [&'static str] {
    UNIFIED_FAMILIES
}

/// The unified family for a request, or `None`. Keyed on EITHER the tool-call or
/// the reasoning parser name being muse, so a card that sets only
/// `--dyn-reasoning-parser muse_glimmer` still routes here. Canonicalizes the
/// hyphen alias to the crate's family key `muse_glimmer`. Default-on: reads no
/// environment variable.
pub(crate) fn unified_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<String> {
    let is_muse = |p: Option<&str>| UNIFIED_FAMILIES.contains(&p.unwrap_or_default());
    (is_muse(tool_call_parser) || is_muse(reasoning_parser)).then(|| "muse_glimmer".to_string())
}

/// Request-side gate for routing the **batch** finalize through v2, mirroring the
/// streaming gate's tool_choice clause (see `preprocessor.rs`): only an unset or `auto`
/// tool_choice is eligible. `tool_choice=required`/named and structural-tag mode are
/// guided-decoded JSON, not the native markup the v2 parser reads, so they stay on the
/// v1 finalize path.
///
/// The streaming gate's `!uses_tool_call_structural_tag` clause is intentionally not
/// re-checked here. That flag is computed in the worker preprocessor and isn't visible
/// to the frontend aggregator; but any request that gets structural-tag/guided decoding
/// is parsed upstream (the worker jails it and emits `tool_calls`), so it never reaches
/// the batch finalize with raw text. The request's tool_choice is the reachable guard.
pub(crate) fn batch_tool_choice_eligible(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
) -> bool {
    matches!(
        tool_choice,
        None | Some(ChatCompletionToolChoiceOption::Auto)
    )
}

/// Map dynamo's v1 `ToolDefinition`s onto the v2 parser's `Tool` shape.
fn to_v2_tools(tools: Option<&[ToolDefinition]>) -> Vec<ToolV2> {
    tools
        .unwrap_or(&[])
        .iter()
        .map(|t| ToolV2 {
            name: t.name.clone(),
            description: None,
            parameters: t.parameters.clone().unwrap_or(serde_json::Value::Null),
            strict: t.strict,
        })
        .collect()
}

/// Batch (non-streaming) path: run the whole response text through the `family` v2
/// parser's complete lifecycle and map the coalesced calls back onto the v1
/// `(tool_calls, normal_text)` tuple the aggregator consumes. No jail involved; a
/// call truncated mid parameter value is dropped (returns zero calls, empty text).
pub(crate) fn parse_complete(
    content: &str,
    tools: Option<&[ToolDefinition]>,
    family: &str,
) -> anyhow::Result<(Vec<ToolCallResponse>, String)> {
    let v2_tools = to_v2_tools(tools);
    let mut parser = create_tool_parser_for_family(family, &v2_tools)?;
    let result = parser.parse_complete(content)?;

    let tool_calls = result
        .calls
        .into_iter()
        .map(|call| ToolCallResponse {
            id: format!("call-{}", Uuid::new_v4()),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: call.name.unwrap_or_default(),
                arguments: call.arguments,
            },
        })
        .collect();

    Ok((tool_calls, result.normal_text))
}

/// Batch (non-streaming) unified finalize: run the whole response text through the
/// `family` UNIFIED parser and split the assembled events into
/// `(tool_calls, reasoning, content)`. Mirrors [`parse_complete`] but adds the
/// reasoning channel the unified parser owns. Used by the (B)-topology aggregator
/// path where raw model text reaches the frontend un-split.
pub(crate) fn parse_complete_unified(
    content: &str,
    tools: Option<&[ToolDefinition]>,
    family: &str,
) -> anyhow::Result<(Vec<ToolCallResponse>, String, String)> {
    let v2_tools = to_v2_tools(tools);
    let mut parser = create_unified_parser_for_family(family, &v2_tools)?;
    let events = parser.parse_complete(content)?;

    let mut tool_calls = Vec::new();
    let mut reasoning = String::new();
    let mut text = String::new();
    for event in events {
        match event {
            UnifiedEvent::Reasoning { text: t } => reasoning.push_str(&t),
            UnifiedEvent::Text { text: t } => text.push_str(&t),
            UnifiedEvent::ToolCall { name, arguments } => tool_calls.push(ToolCallResponse {
                id: format!("call-{}", Uuid::new_v4()),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name,
                    // `UnifiedEvent` carries a typed `Value`; the wire wants a string.
                    arguments: serde_json::to_string(&arguments).unwrap_or_default(),
                },
            }),
        }
    }

    Ok((tool_calls, reasoning, text))
}

/// Map v2 per-chunk tool deltas onto OpenAI streaming tool-call chunks. The first
/// delta for a given tool index carries the minted id, type and function name;
/// later deltas for that index carry only argument fragments (`id`/`type`/`name`
/// `None`), matching the OpenAI streaming tool-call contract. `opened` tracks the
/// indices already opened; shared by the tool-only and unified streaming states.
fn emit_tool_chunks(
    opened: &mut HashSet<usize>,
    calls: Vec<ToolCallDelta>,
) -> Option<Vec<ChatCompletionMessageToolCallChunk>> {
    if calls.is_empty() {
        return None;
    }
    let chunks = calls
        .into_iter()
        .map(|delta| {
            let first = opened.insert(delta.tool_index);
            ChatCompletionMessageToolCallChunk {
                index: delta.tool_index as u32,
                id: first.then(|| format!("call-{}", Uuid::new_v4())),
                r#type: first.then_some(FunctionType::Function),
                function: Some(FunctionCallStream {
                    name: delta.name,
                    arguments: Some(delta.arguments),
                }),
            }
        })
        .collect();
    Some(chunks)
}

/// Split one push/finish batch of unified deltas into the three OpenAI delta
/// channels. Each channel is a single concatenated string on the wire, so
/// collapsing one batch is lossless w.r.t. the schema while still emitting
/// reasoning WHERE it occurred rather than hoisting it ahead of a call.
fn fold_unified_deltas(deltas: Vec<UnifiedParserEvent>) -> (String, String, Vec<ToolCallDelta>) {
    let mut reasoning = String::new();
    let mut text = String::new();
    let mut calls = Vec::new();
    for delta in deltas {
        match delta {
            UnifiedParserEvent::Reasoning(t) => reasoning.push_str(&t),
            UnifiedParserEvent::Text(t) => text.push_str(&t),
            UnifiedParserEvent::ToolCall(call) => calls.push(call),
        }
    }
    (reasoning, text, calls)
}

/// Per-choice streaming state: one parser plus the set of tool indices whose
/// opening delta (id + type + function name) has already been emitted.
struct ChoiceState {
    parser: Box<dyn ToolParser>,
    opened: HashSet<usize>,
}

impl ChoiceState {
    fn new(family: &str, tools: &[ToolV2]) -> anyhow::Result<Self> {
        Ok(Self {
            parser: create_tool_parser_for_family(family, tools)?,
            opened: HashSet::new(),
        })
    }

    /// Map v2 per-chunk deltas onto OpenAI streaming tool-call chunks.
    fn emit_chunks(
        &mut self,
        calls: Vec<ToolCallDelta>,
    ) -> Option<Vec<ChatCompletionMessageToolCallChunk>> {
        emit_tool_chunks(&mut self.opened, calls)
    }
}

/// Per-choice UNIFIED streaming state: one unified parser plus the same
/// tool-index opened-set as [`ChoiceState`], so the id-minting contract is
/// shared verbatim.
struct UnifiedChoiceState {
    parser: Box<dyn UnifiedParser>,
    opened: HashSet<usize>,
}

impl UnifiedChoiceState {
    fn new(family: &str, tools: &[ToolV2]) -> anyhow::Result<Self> {
        Ok(Self {
            parser: create_unified_parser_for_family(family, tools)?,
            opened: HashSet::new(),
        })
    }

    fn emit_chunks(
        &mut self,
        calls: Vec<ToolCallDelta>,
    ) -> Option<Vec<ChatCompletionMessageToolCallChunk>> {
        emit_tool_chunks(&mut self.opened, calls)
    }
}

/// Finish every choice that has not received an upstream finish reason. This is
/// called before a usage-only chunk when one exists, with EOF as a fallback.
fn finish_unterminated_choices(
    states: &mut HashMap<u32, ChoiceState>,
    finished: &mut HashSet<u32>,
    tool_emitted: &mut HashSet<u32>,
    template: &NvCreateChatCompletionStreamResponse,
) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
    let mut indices: Vec<_> = states
        .keys()
        .copied()
        .filter(|index| !finished.contains(index))
        .collect();
    indices.sort_unstable();

    let mut responses = Vec::new();
    for index in indices {
        finished.insert(index);
        let state = states
            .get_mut(&index)
            .expect("choice index came from parser state map");
        let result = match state.parser.finish() {
            Ok(result) => result,
            Err(error) => {
                tracing::warn!(error = %error, choice_index = index, "v2 stream finish failed");
                dynamo_parsers_v2::ToolParseResult::default()
            }
        };
        let tool_calls = state.emit_chunks(result.calls);
        if tool_calls.is_some() {
            tool_emitted.insert(index);
        }
        // A choice that produced tool calls during the stream must terminate
        // with `ToolCalls` even when the backend never sent a finish_reason.
        // Text-only output without an upstream finish reason stays `None`.
        let finish_reason = if tool_emitted.contains(&index) {
            Some(FinishReason::ToolCalls)
        } else {
            None
        };
        let content = (!result.normal_text.is_empty())
            .then_some(ChatCompletionMessageContent::Text(result.normal_text));
        if content.is_none() && tool_calls.is_none() && finish_reason.is_none() {
            continue;
        }
        responses.push(stream_choice_chunk_from_template(
            template,
            index,
            content,
            None,
            tool_calls,
            finish_reason,
        ));
    }
    responses
}

/// Unified counterpart of [`finish_unterminated_choices`]: flush each unfinished
/// UNIFIED parser and build the trailing chunk, which here can also carry
/// `reasoning_content` (open reasoning is promoted at `finish`).
fn finish_unterminated_choices_unified(
    states: &mut HashMap<u32, UnifiedChoiceState>,
    finished: &mut HashSet<u32>,
    tool_emitted: &mut HashSet<u32>,
    template: &NvCreateChatCompletionStreamResponse,
) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
    let mut indices: Vec<_> = states
        .keys()
        .copied()
        .filter(|index| !finished.contains(index))
        .collect();
    indices.sort_unstable();

    let mut responses = Vec::new();
    for index in indices {
        finished.insert(index);
        let state = states
            .get_mut(&index)
            .expect("choice index came from parser state map");
        let deltas = match state.parser.finish() {
            Ok(output) => output.events,
            Err(error) => {
                tracing::warn!(error = %error, choice_index = index, "muse unified stream finish failed");
                Vec::new()
            }
        };
        let (reasoning, text, calls) = fold_unified_deltas(deltas);
        let tool_calls = state.emit_chunks(calls);
        if tool_calls.is_some() {
            tool_emitted.insert(index);
        }
        let finish_reason = if tool_emitted.contains(&index) {
            Some(FinishReason::ToolCalls)
        } else {
            None
        };
        let content = (!text.is_empty()).then_some(ChatCompletionMessageContent::Text(text));
        let reasoning_content = (!reasoning.is_empty()).then_some(reasoning);
        if content.is_none()
            && reasoning_content.is_none()
            && tool_calls.is_none()
            && finish_reason.is_none()
        {
            continue;
        }
        responses.push(stream_choice_chunk_from_template(
            template,
            index,
            content,
            reasoning_content,
            tool_calls,
            finish_reason,
        ));
    }
    responses
}

/// Streaming path: replace the jail with the `family` v2 parser. Each upstream text
/// delta is pushed into the parser; the parser's `normal_text` becomes the emitted
/// content and its tool-call deltas become OpenAI tool-call chunks. The jail is never
/// built. `finish()` runs on a choice's terminating chunk (and again at stream end as
/// a backstop) so a value truncated mid-stream is dropped instead of leaking markup.
pub(crate) fn apply_stream<S>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    family: String,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let v2_tools = to_v2_tools(tool_definitions.as_deref());
    stream! {
        // The caller only routes supported families here, but if a parser cannot be
        // built we pass every chunk through untouched rather than dropping output.
        if create_tool_parser_for_family(&family, &v2_tools).is_err() {
            tracing::warn!(family = %family, "no dynamo-parsers-v2 parser for family; passing stream through unchanged");
            tokio::pin!(stream_in);
            while let Some(response) = stream_in.next().await {
                yield response;
            }
            return;
        }

        let mut states: HashMap<u32, ChoiceState> = HashMap::new();
        // Choice indices whose finish() has already run (terminating chunk seen).
        let mut finished: HashSet<u32> = HashSet::new();
        // Choice indices that have emitted at least one tool-call chunk; used to flip a
        // `Stop` terminating reason to `ToolCalls` (OpenAI contract — see below).
        let mut tool_emitted: HashSet<u32> = HashSet::new();
        // Last data response, kept (with choices cleared) as a template for the
        // end-of-stream flush when no finish_reason chunk arrived.
        let mut template: Option<NvCreateChatCompletionStreamResponse> = None;

        tokio::pin!(stream_in);

        while let Some(mut response) = stream_in.next().await {
            let Some(chat_response) = response.data.as_mut() else {
                // Non-data annotations (errors, comments) pass through untouched.
                yield response;
                continue;
            };

            {
                let mut t = chat_response.clone();
                t.inner.choices.clear();
                template = Some(t);
            }
            let is_empty_choices = chat_response.inner.choices.is_empty();

            for choice in chat_response.inner.choices.iter_mut() {
                let state = states.entry(choice.index).or_insert_with(|| {
                    // Family validated above; construction is deterministic in-process.
                    ChoiceState::new(&family, &v2_tools)
                        .expect("dynamo-parsers-v2 parser construction validated above")
                });

                // Only text content feeds the parser; multimodal parts pass through.
                let text = match choice.delta.content.as_ref() {
                    Some(ChatCompletionMessageContent::Text(t)) => Some(t.clone()),
                    _ => None,
                };

                let mut result = dynamo_parsers_v2::ToolParseResult::default();
                let mut parsed_any = false;
                if let Some(text) = text.as_deref() {
                    match state.parser.push(text) {
                        Ok(r) => {
                            result.append(r);
                            parsed_any = true;
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, family = %family, "v2 stream push failed; passing chunk through");
                        }
                    }
                }
                // Flush on the terminating chunk so a value truncated at EOF is dropped.
                if choice.finish_reason.is_some() && finished.insert(choice.index) {
                    match state.parser.finish() {
                        Ok(r) => {
                            result.append(r);
                            parsed_any = true;
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, family = %family, "v2 stream finish failed");
                        }
                    }
                }

                if parsed_any {
                    let tool_calls = state.emit_chunks(result.calls);
                    if tool_calls.is_some() {
                        tool_emitted.insert(choice.index);
                    }
                    // The parser consumed text input, so replace content with its
                    // normal_text (None when the input was all tool markup) — raw tool
                    // markup must never reach the client. Role, reasoning and logprobs
                    // are preserved as-is.
                    choice.delta.content = if result.normal_text.is_empty() {
                        None
                    } else {
                        Some(ChatCompletionMessageContent::Text(result.normal_text))
                    };
                    choice.delta.tool_calls = tool_calls;
                }

                // OpenAI streaming contract: once a choice has emitted tool calls, a
                // `Stop` terminating reason must be reported as `ToolCalls` (mirrors the
                // v1 jail's fix_finish_reason). Length/ContentFilter are preserved as-is.
                // Runs regardless of parsed_any so a role-only terminating chunk that
                // still carries finish_reason gets fixed.
                if choice.finish_reason == Some(FinishReason::Stop)
                    && tool_emitted.contains(&choice.index)
                {
                    choice.finish_reason = Some(FinishReason::ToolCalls);
                }
            }

            // OpenAI stream ordering requires a terminal finish_reason before the
            // usage-only chunk. Finish every unterminated choice before yielding an
            // empty-choices response; EOF below remains the fallback when no such
            // response arrives.
            if is_empty_choices && let Some(template) = &template {
                for terminal in finish_unterminated_choices(
                    &mut states,
                    &mut finished,
                    &mut tool_emitted,
                    template,
                ) {
                    yield terminal;
                }
            }

            yield response;
        }

        // Backstop: the stream ended without a finish_reason for some choice. Flush
        // each unfinished parser; emit a trailing chunk when the flush yields output
        // or when the choice already emitted tool calls and still needs a terminal
        // `ToolCalls` reason.
        if let Some(template) = &template {
            for terminal in finish_unterminated_choices(
                &mut states,
                &mut finished,
                &mut tool_emitted,
                template,
            ) {
                yield terminal;
            }
        }
    }
}

/// Streaming path for the UNIFIED families (muse), default-on. One parser per
/// choice owns reasoning + content + tool calls in ONE ordered pass, so this
/// replaces BOTH the v1 reasoning stage and the tool jail. Each upstream text
/// delta is pushed; the parser's ordered [`UnifiedParserEvent`]s fold into
/// `reasoning_content` / `content` / tool-call chunks on the same output chunk.
/// `finish()` runs on a choice's terminating chunk (and again at stream end as a
/// backstop) so open reasoning is promoted and a value truncated at EOF is
/// dropped instead of leaking markup.
pub(crate) fn apply_unified_stream<S>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    family: String,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let v2_tools = to_v2_tools(tool_definitions.as_deref());
    stream! {
        tracing::debug!(family = %family, "muse unified stream engaged");

        // The caller only routes supported families here, but if a parser cannot be
        // built we pass every chunk through untouched rather than dropping output.
        if create_unified_parser_for_family(&family, &v2_tools).is_err() {
            tracing::warn!(family = %family, "no dynamo-parsers-v2 unified parser for family; passing stream through unchanged");
            tokio::pin!(stream_in);
            while let Some(response) = stream_in.next().await {
                yield response;
            }
            return;
        }

        let mut states: HashMap<u32, UnifiedChoiceState> = HashMap::new();
        // Choice indices whose finish() has already run (terminating chunk seen).
        let mut finished: HashSet<u32> = HashSet::new();
        // Choice indices that have emitted at least one tool-call chunk; used to flip
        // a `Stop` terminating reason to `ToolCalls` (OpenAI contract).
        let mut tool_emitted: HashSet<u32> = HashSet::new();
        // Last data response, kept (with choices cleared) as a template for the
        // end-of-stream flush when no finish_reason chunk arrived.
        let mut template: Option<NvCreateChatCompletionStreamResponse> = None;

        tokio::pin!(stream_in);

        while let Some(mut response) = stream_in.next().await {
            let Some(chat_response) = response.data.as_mut() else {
                // Non-data annotations (errors, comments) pass through untouched.
                yield response;
                continue;
            };

            {
                let mut t = chat_response.clone();
                t.inner.choices.clear();
                template = Some(t);
            }
            let is_empty_choices = chat_response.inner.choices.is_empty();

            for choice in chat_response.inner.choices.iter_mut() {
                let state = states.entry(choice.index).or_insert_with(|| {
                    // Family validated above; construction is deterministic in-process.
                    UnifiedChoiceState::new(&family, &v2_tools)
                        .expect("dynamo-parsers-v2 unified parser construction validated above")
                });

                // Only text content feeds the parser; multimodal parts pass through.
                let text = match choice.delta.content.as_ref() {
                    Some(ChatCompletionMessageContent::Text(t)) => Some(t.clone()),
                    _ => None,
                };

                let mut deltas: Vec<UnifiedParserEvent> = Vec::new();
                let mut parsed_any = false;
                if let Some(text) = text.as_deref() {
                    match state.parser.push(text) {
                        Ok(mut pushed) => {
                            deltas.append(&mut pushed);
                            parsed_any = true;
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, family = %family, "muse unified push failed; passing chunk through");
                        }
                    }
                }
                // Flush on the terminating chunk so a value truncated at EOF is dropped
                // and open reasoning is promoted.
                if choice.finish_reason.is_some() && finished.insert(choice.index) {
                    match state.parser.finish() {
                        Ok(mut flushed) => {
                            deltas.append(&mut flushed.events);
                            parsed_any = true;
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, family = %family, "muse unified finish failed");
                        }
                    }
                }

                if parsed_any {
                    let (reasoning, out_text, calls) = fold_unified_deltas(deltas);
                    let tool_calls = state.emit_chunks(calls);
                    if tool_calls.is_some() {
                        tool_emitted.insert(choice.index);
                    }
                    // The parser consumed the text input, so replace every channel
                    // from its output — raw markup must never reach the client. Role
                    // and logprobs are preserved as-is.
                    choice.delta.reasoning_content =
                        (!reasoning.is_empty()).then_some(reasoning);
                    choice.delta.content = (!out_text.is_empty())
                        .then_some(ChatCompletionMessageContent::Text(out_text));
                    choice.delta.tool_calls = tool_calls;
                }

                // OpenAI streaming contract: once a choice has emitted tool calls, a
                // `Stop` terminating reason must be reported as `ToolCalls`.
                if choice.finish_reason == Some(FinishReason::Stop)
                    && tool_emitted.contains(&choice.index)
                {
                    choice.finish_reason = Some(FinishReason::ToolCalls);
                }
            }

            // OpenAI stream ordering requires a terminal finish_reason before the
            // usage-only chunk. Finish every unterminated choice before yielding an
            // empty-choices response; EOF below remains the fallback.
            if is_empty_choices && let Some(template) = &template {
                for terminal in finish_unterminated_choices_unified(
                    &mut states,
                    &mut finished,
                    &mut tool_emitted,
                    template,
                ) {
                    yield terminal;
                }
            }

            yield response;
        }

        // Backstop: the stream ended without a finish_reason for some choice.
        if let Some(template) = &template {
            for terminal in finish_unterminated_choices_unified(
                &mut states,
                &mut finished,
                &mut tool_emitted,
                template,
            ) {
                yield terminal;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, CompletionUsage, FinishReason, Role,
    };
    use futures::stream;

    struct FinishErrorParser;

    impl ToolParser for FinishErrorParser {
        fn create(_tools: &[ToolV2]) -> anyhow::Result<Box<dyn ToolParser>> {
            Ok(Box::new(Self))
        }

        fn push(&mut self, _chunk: &str) -> anyhow::Result<dynamo_parsers_v2::ToolParseResult> {
            Ok(dynamo_parsers_v2::ToolParseResult::default())
        }

        fn finish(&mut self) -> anyhow::Result<dynamo_parsers_v2::ToolParseResult> {
            anyhow::bail!("intentional finish failure")
        }
    }

    const QWEN3_GET_WEATHER: &str = "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>";

    // DeepSeek-V4 DSML: one get_weather(location="NYC") call. The `｜` glyphs are the
    // fullwidth vertical bars the DSML markers use (see dynamo_parsers_v2::dsml).
    const DSV4_GET_WEATHER: &str = "<｜DSML｜tool_calls> <｜DSML｜invoke name=\"get_weather\"> <｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter> </｜DSML｜invoke> </｜DSML｜tool_calls>";

    // A full muse turn: a `to=self` thought, one `get_weather` call, then the
    // visible `to=user` answer. Grammar mirrors the frontend-crates unified corpus.
    const MUSE_REASONING: &str = "<|start|>assistant to=self<|message|>Look it up.<|eom|>";
    const MUSE_TOOL: &str = "<|start|>assistant to=get_weather<|message|><atem:invoke name=\"get_weather\"><atem:parameter name=\"location\">Paris</atem:parameter></atem:invoke><|eom|>";
    const MUSE_ANSWER: &str = "<|start|>assistant to=user<|message|>It's 18C.<|eot|>";

    fn muse_turn() -> String {
        format!("{MUSE_REASONING}{MUSE_TOOL}{MUSE_ANSWER}")
    }

    fn chunk(text: &str, finish: bool) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let response = NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "test".to_string(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        role: Some(Role::Assistant),
                        content: Some(ChatCompletionMessageContent::Text(text.to_string())),
                        tool_calls: None,
                        function_call: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: finish.then_some(FinishReason::Stop),
                    logprobs: None,
                }],
                created: 0,
                model: "test".to_string(),
                system_fingerprint: None,
                service_tier: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
            },
            nvext: None,
            llm_metrics: None,
        };
        Annotated {
            data: Some(response),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    fn usage_chunk() -> Annotated<NvCreateChatCompletionStreamResponse> {
        let mut chunk = chunk("", false);
        let data = chunk.data.as_mut().expect("usage chunk response data");
        data.inner.choices.clear();
        data.inner.usage = Some(CompletionUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        });
        data.llm_metrics = Some(crate::protocols::common::metrics::LLMMetricAnnotation {
            input_tokens: 10,
            output_tokens: 5,
            chunk_tokens: 0,
            cached_tokens: None,
            ..Default::default()
        });
        chunk
    }

    /// Reassemble the streamed tool-call deltas into (name, arguments) per index and
    /// collect all emitted content, mirroring how an OpenAI client reconstructs a
    /// streamed tool call.
    fn reassemble(
        responses: &[Annotated<NvCreateChatCompletionStreamResponse>],
    ) -> (Vec<(String, String)>, String) {
        let mut calls: Vec<(String, String)> = Vec::new();
        let mut content = String::new();
        for r in responses {
            let Some(data) = r.data.as_ref() else {
                continue;
            };
            for choice in &data.inner.choices {
                if let Some(ChatCompletionMessageContent::Text(t)) = &choice.delta.content {
                    content.push_str(t);
                }
                let Some(tcs) = &choice.delta.tool_calls else {
                    continue;
                };
                for tc in tcs {
                    let idx = tc.index as usize;
                    if calls.len() <= idx {
                        calls.resize(idx + 1, (String::new(), String::new()));
                    }
                    if let Some(f) = &tc.function {
                        if let Some(name) = &f.name {
                            calls[idx].0 = name.clone();
                        }
                        if let Some(args) = &f.arguments {
                            calls[idx].1.push_str(args);
                        }
                    }
                }
            }
        }
        (calls, content)
    }

    /// The last `finish_reason` emitted across the stream (the terminating reason a
    /// client sees). `None` if the stream never carried one.
    fn final_finish_reason(
        responses: &[Annotated<NvCreateChatCompletionStreamResponse>],
    ) -> Option<FinishReason> {
        responses
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| d.inner.choices.iter())
            .filter_map(|c| c.finish_reason)
            .next_back()
    }

    // Feed a Qwen3-Coder tool call split across many small chunks (incremental
    // streaming) and confirm the bypass reconstructs exactly one call with the
    // right arguments and never leaks raw markup into content.
    #[tokio::test]
    async fn qwen3_bypass_streams_incrementally_without_leaking_markup() {
        // Split into 8-char chunks to force partial markers across push() calls.
        let mut chunks: Vec<_> = QWEN3_GET_WEATHER
            .as_bytes()
            .chunks(8)
            .map(|b| chunk(std::str::from_utf8(b).unwrap(), false))
            .collect();
        chunks.push(chunk("", true));

        let out: Vec<_> = apply_stream(stream::iter(chunks), None, "qwen3_coder".to_string())
            .collect::<Vec<_>>()
            .await;

        let (calls, content) = reassemble(&out);
        assert_eq!(calls.len(), 1, "expected exactly one tool call: {calls:?}");
        assert_eq!(calls[0].0, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["location"], "Paris");
        for marker in ["<tool_call>", "<function=", "<parameter="] {
            assert!(
                !content.contains(marker),
                "raw markup {marker:?} leaked into content: {content:?}"
            );
        }
        // A choice that emitted tool calls must report finish_reason=ToolCalls, not the
        // upstream Stop (OpenAI streaming contract; mirrors the v1 jail).
        assert_eq!(
            final_finish_reason(&out),
            Some(FinishReason::ToolCalls),
            "finish_reason must flip Stop->ToolCalls when tool calls are emitted"
        );
    }

    // A parameter value truncated at EOF must be dropped (no guessed argument),
    // and no markup may leak — the reason qwen3 uses the v2 parser, not the jail.
    #[tokio::test]
    async fn qwen3_bypass_drops_value_truncated_at_eof() {
        let truncated = "<tool_call>\n<function=get_weather>\n<parameter=location>\nPar";
        let chunks = vec![chunk(truncated, false), chunk("", true)];

        let out: Vec<_> = apply_stream(stream::iter(chunks), None, "qwen3_coder".to_string())
            .collect::<Vec<_>>()
            .await;

        let (calls, content) = reassemble(&out);
        let complete: Vec<_> = calls.iter().filter(|(n, _)| !n.is_empty()).collect();
        assert!(
            complete.is_empty(),
            "truncated value must not produce a finished call: {calls:?}"
        );
        assert!(
            !content.contains("<function="),
            "raw markup leaked into content: {content:?}"
        );
        // No call was emitted (truncated value dropped), so the terminating reason must
        // stay Stop — the flip only fires when tool calls were actually emitted.
        assert_eq!(
            final_finish_reason(&out),
            Some(FinishReason::Stop),
            "no tool calls emitted -> finish_reason stays Stop"
        );
    }

    // Same family-agnostic bypass for DeepSeek-V4 DSML: incremental streaming
    // reconstructs exactly one call and never leaks DSML markup into content.
    #[tokio::test]
    async fn dsv4_bypass_streams_incrementally_without_leaking_markup() {
        // DSML markers contain the multibyte `｜` glyph; chunk by chars (not bytes) so
        // a small chunk size still splits markers across push() calls without slicing
        // a UTF-8 character.
        let glyphs: Vec<char> = DSV4_GET_WEATHER.chars().collect();
        let mut chunks: Vec<_> = glyphs
            .chunks(6)
            .map(|c| chunk(&c.iter().collect::<String>(), false))
            .collect();
        chunks.push(chunk("", true));

        let out: Vec<_> = apply_stream(stream::iter(chunks), None, "deepseek_v4".to_string())
            .collect::<Vec<_>>()
            .await;

        let (calls, content) = reassemble(&out);
        let complete: Vec<_> = calls.iter().filter(|(n, _)| !n.is_empty()).collect();
        assert_eq!(
            complete.len(),
            1,
            "expected exactly one tool call: {calls:?}"
        );
        assert_eq!(complete[0].0, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&complete[0].1).unwrap();
        assert_eq!(args["location"], "NYC");
        assert!(
            !content.contains("DSML"),
            "raw DSML markup leaked into content: {content:?}"
        );
        assert_eq!(
            final_finish_reason(&out),
            Some(FinishReason::ToolCalls),
            "finish_reason must flip Stop->ToolCalls when tool calls are emitted"
        );
    }

    // Missing-finish-reason regression: the stream emits a complete tool call but
    // ends without any finish_reason chunk (e.g. speculative decoding folded EOS
    // into content, or the engine dropped the terminal signal). A strict OpenAI
    // client waits for a non-null finish_reason before considering the tool call
    // complete; the end-of-stream backstop must synthesize `ToolCalls` so the
    // client doesn't hang until its timeout.
    #[tokio::test]
    async fn qwen3_bypass_synthesizes_tool_calls_when_stream_lacks_finish_reason() {
        // Same call as the incremental test, but the final chunk carries NO
        // finish_reason — the stream simply ends after the tool markup.
        let mut chunks: Vec<_> = QWEN3_GET_WEATHER
            .as_bytes()
            .chunks(8)
            .map(|b| chunk(std::str::from_utf8(b).unwrap(), false))
            .collect();
        // A usage-only chunk arrives without any terminating choice.
        chunks.push(usage_chunk());

        let out: Vec<_> = apply_stream(stream::iter(chunks), None, "qwen3_coder".to_string())
            .collect::<Vec<_>>()
            .await;

        let (calls, _content) = reassemble(&out);
        assert_eq!(calls.len(), 1, "expected exactly one tool call: {calls:?}");
        assert_eq!(calls[0].0, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["location"], "Paris");
        assert_eq!(
            final_finish_reason(&out),
            Some(FinishReason::ToolCalls),
            "backstop must synthesize ToolCalls when the stream ended without a finish_reason"
        );
        let finish_positions: Vec<_> = out
            .iter()
            .enumerate()
            .filter_map(|(position, response)| {
                response.data.as_ref().and_then(|data| {
                    data.inner
                        .choices
                        .iter()
                        .any(|choice| choice.finish_reason == Some(FinishReason::ToolCalls))
                        .then_some(position)
                })
            })
            .collect();
        assert_eq!(
            finish_positions.len(),
            1,
            "expected exactly one synthesized finish chunk"
        );
        let usage_position =
            out.iter()
                .position(|response| {
                    response.data.as_ref().is_some_and(|data| {
                        data.inner.choices.is_empty() && data.inner.usage.is_some()
                    })
                })
                .expect("usage-only response");
        assert!(
            finish_positions[0] < usage_position,
            "synthesized finish chunk must precede usage"
        );
        let terminal = out[finish_positions[0]]
            .data
            .as_ref()
            .expect("synthesized terminal response");
        assert!(
            terminal.inner.usage.is_none(),
            "synthesized terminal chunk must not repeat usage"
        );
        assert!(
            terminal.llm_metrics.is_none(),
            "synthesized terminal chunk must not repeat LLM metrics"
        );
    }

    // Text-only corollary: when the stream ends without a finish_reason and no
    // tool call was emitted, the backstop must not invent a finish_reason. There
    // is no signal to synthesize one from. A trailing content chunk may be
    // emitted, but its finish_reason stays None.
    #[tokio::test]
    async fn qwen3_bypass_does_not_synthesize_finish_reason_for_text_only_stream() {
        let chunks = vec![chunk("hello world", false), chunk("", false)];

        let out: Vec<_> = apply_stream(stream::iter(chunks), None, "qwen3_coder".to_string())
            .collect::<Vec<_>>()
            .await;

        let (calls, _content) = reassemble(&out);
        assert!(calls.is_empty(), "no tool calls expected: {calls:?}");
        assert_eq!(
            final_finish_reason(&out),
            None,
            "text-only stream with no upstream finish_reason must not get a synthetic one"
        );
    }

    #[test]
    fn finish_error_still_terminates_a_choice_that_emitted_tools() {
        let mut states = HashMap::from([(
            3,
            ChoiceState {
                parser: Box::new(FinishErrorParser),
                opened: HashSet::new(),
            },
        )]);
        let mut finished = HashSet::new();
        let mut tool_emitted = HashSet::from([3]);
        let template = usage_chunk().data.expect("usage response data");

        let responses =
            finish_unterminated_choices(&mut states, &mut finished, &mut tool_emitted, &template);

        assert_eq!(
            responses.len(),
            1,
            "the choice still needs a terminal chunk"
        );
        let response = responses[0].data.as_ref().expect("terminal response data");
        assert!(
            response.inner.usage.is_none(),
            "terminal chunk must not repeat usage"
        );
        assert!(
            response.llm_metrics.is_none(),
            "terminal chunk must not repeat LLM metrics"
        );
        assert_eq!(response.inner.choices.len(), 1);
        assert_eq!(response.inner.choices[0].index, 3);
        assert_eq!(
            response.inner.choices[0].finish_reason,
            Some(FinishReason::ToolCalls)
        );
    }

    // ── UNIFIED (muse) streaming ──────────────────────────────────────────────

    /// Reassemble the unified stream: reuse [`reassemble`] for calls + content and
    /// concatenate every `reasoning_content` delta.
    fn reassemble_unified(
        responses: &[Annotated<NvCreateChatCompletionStreamResponse>],
    ) -> (Vec<(String, String)>, String, String) {
        let (calls, content) = reassemble(responses);
        let mut reasoning = String::new();
        for r in responses {
            let Some(data) = r.data.as_ref() else {
                continue;
            };
            for choice in &data.inner.choices {
                if let Some(rc) = &choice.delta.reasoning_content {
                    reasoning.push_str(rc);
                }
            }
        }
        (calls, content, reasoning)
    }

    // The one pass yields reasoning_content, content AND a tool call, split into
    // small chunks so markers straddle push() boundaries. This is what the split
    // v1-reasoning + v2-tool path structurally cannot do (it would hoist the
    // post-call thought), and why muse routes through the unified parser.
    #[tokio::test]
    async fn muse_unified_streams_reasoning_content_and_tool_calls() {
        let turn = muse_turn();
        let mut chunks: Vec<_> = turn
            .as_bytes()
            .chunks(8)
            .map(|b| chunk(std::str::from_utf8(b).unwrap(), false))
            .collect();
        chunks.push(chunk("", true));

        let out: Vec<_> =
            apply_unified_stream(stream::iter(chunks), None, "muse_glimmer".to_string())
                .collect::<Vec<_>>()
                .await;

        let (calls, content, reasoning) = reassemble_unified(&out);
        assert_eq!(reasoning, "Look it up.", "reasoning_content mismatch");
        assert_eq!(content, "It's 18C.", "content mismatch");
        assert_eq!(calls.len(), 1, "expected exactly one tool call: {calls:?}");
        assert_eq!(calls[0].0, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["location"], "Paris");
        for marker in ["<|start|>", "<|message|>", "<|eom|>", "<atem:invoke"] {
            assert!(
                !content.contains(marker) && !reasoning.contains(marker),
                "raw markup {marker:?} leaked: content={content:?} reasoning={reasoning:?}"
            );
        }
        assert_eq!(
            final_finish_reason(&out),
            Some(FinishReason::ToolCalls),
            "finish_reason must flip Stop->ToolCalls when a tool call is emitted"
        );
    }

    // A call truncated mid parameter value at EOF is dropped by finish() (policy
    // P2), and no `<atem:invoke` opener may leak into content.
    #[tokio::test]
    async fn muse_unified_drops_value_truncated_at_eof() {
        let truncated = "<|start|>assistant to=get_weather<|message|><atem:invoke name=\"get_weather\"><atem:parameter name=\"location\">Par";
        let chunks = vec![chunk(truncated, false), chunk("", true)];

        let out: Vec<_> =
            apply_unified_stream(stream::iter(chunks), None, "muse_glimmer".to_string())
                .collect::<Vec<_>>()
                .await;

        let (calls, content, _reasoning) = reassemble_unified(&out);
        let complete: Vec<_> = calls.iter().filter(|(n, _)| !n.is_empty()).collect();
        assert!(
            complete.is_empty(),
            "truncated value must not produce a finished call: {calls:?}"
        );
        assert!(
            !content.contains("<atem:invoke"),
            "raw markup leaked into content: {content:?}"
        );
        assert_eq!(
            final_finish_reason(&out),
            Some(FinishReason::Stop),
            "no tool call emitted -> finish_reason stays Stop"
        );
    }

    // Missing-finish-reason backstop: the stream ends with a usage-only chunk and
    // no terminating finish_reason. The end-of-stream flush must synthesize
    // `ToolCalls` before the usage chunk so a strict client does not hang.
    #[tokio::test]
    async fn muse_unified_synthesizes_tool_calls_when_stream_lacks_finish_reason() {
        let turn = muse_turn();
        let mut chunks: Vec<_> = turn
            .as_bytes()
            .chunks(8)
            .map(|b| chunk(std::str::from_utf8(b).unwrap(), false))
            .collect();
        chunks.push(usage_chunk());

        let out: Vec<_> =
            apply_unified_stream(stream::iter(chunks), None, "muse_glimmer".to_string())
                .collect::<Vec<_>>()
                .await;

        let (calls, _content, _reasoning) = reassemble_unified(&out);
        assert_eq!(calls.len(), 1, "expected exactly one tool call: {calls:?}");
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(
            final_finish_reason(&out),
            Some(FinishReason::ToolCalls),
            "backstop must synthesize ToolCalls when the stream lacks a finish_reason"
        );
        let finish_position = out
            .iter()
            .position(|response| {
                response.data.as_ref().is_some_and(|data| {
                    data.inner
                        .choices
                        .iter()
                        .any(|c| c.finish_reason == Some(FinishReason::ToolCalls))
                })
            })
            .expect("synthesized finish chunk");
        let usage_position =
            out.iter()
                .position(|response| {
                    response.data.as_ref().is_some_and(|data| {
                        data.inner.choices.is_empty() && data.inner.usage.is_some()
                    })
                })
                .expect("usage-only response");
        assert!(
            finish_position < usage_position,
            "synthesized finish chunk must precede usage"
        );
    }

    // Reasoning + content, no tools: the backstop must NOT invent a finish_reason
    // (nothing to synthesize one from), and both channels reassemble.
    #[tokio::test]
    async fn muse_unified_text_only_no_synthetic_finish() {
        let turn = format!(
            "{}{}",
            "<|start|>assistant to=self<|message|>Just thinking.<|eom|>",
            "<|start|>assistant to=user<|message|>Hello.<|eot|>"
        );
        let chunks = vec![chunk(&turn, false), chunk("", false)];

        let out: Vec<_> =
            apply_unified_stream(stream::iter(chunks), None, "muse_glimmer".to_string())
                .collect::<Vec<_>>()
                .await;

        let (calls, content, reasoning) = reassemble_unified(&out);
        assert!(calls.is_empty(), "no tool calls expected: {calls:?}");
        assert_eq!(reasoning, "Just thinking.");
        assert_eq!(content, "Hello.");
        assert_eq!(
            final_finish_reason(&out),
            None,
            "text-only stream with no upstream finish_reason must not get a synthetic one"
        );
    }

    #[test]
    fn unified_family_default_on_and_alias() {
        for parser in ["muse_glimmer", "muse-glimmer"] {
            assert_eq!(
                unified_family(Some(parser), None).as_deref(),
                Some("muse_glimmer"),
                "tool-call name {parser:?} must route to muse unified"
            );
            assert_eq!(
                unified_family(None, Some(parser)).as_deref(),
                Some("muse_glimmer"),
                "reasoning name {parser:?} must route to muse unified"
            );
        }
        assert_eq!(unified_family(Some("qwen3_coder"), None), None);
        assert_eq!(unified_family(None, None), None);
        // Default-on: `unified_family` reads no environment variable. muse routes to
        // the unified parser whether or not the experimental v2 gate is set — proven
        // here by routing while `enabled()` (DYN_ENABLE_EXPERIMENTAL_PARSERS_V2) is off.
        assert!(
            !enabled(),
            "test env should not set DYN_ENABLE_EXPERIMENTAL_PARSERS_V2"
        );
        assert!(
            unified_family(Some("muse_glimmer"), None).is_some(),
            "muse must route with the experimental v2 gate OFF (default-on)"
        );
    }

    #[test]
    fn parse_complete_unified_splits_reasoning_content_calls() {
        let (calls, reasoning, content) =
            parse_complete_unified(&muse_turn(), None, "muse_glimmer").unwrap();
        assert_eq!(reasoning, "Look it up.");
        assert_eq!(content, "It's 18C.");
        assert_eq!(calls.len(), 1, "expected exactly one tool call: {calls:?}");
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "Paris");
    }

    // A single push carrying reasoning, a tool call and text folds each onto its own
    // channel (no cross-leak), and empty-string deltas stay empty (dropped downstream
    // by `then_some`).
    #[test]
    fn fold_unified_deltas_routes_each_channel_and_drops_empties() {
        let deltas = vec![
            UnifiedParserEvent::Reasoning("think".into()),
            UnifiedParserEvent::ToolCall(ToolCallDelta {
                tool_index: 0,
                name: Some("f".into()),
                arguments: "{}".into(),
            }),
            UnifiedParserEvent::Text("ans".into()),
        ];
        let (r, t, c) = fold_unified_deltas(deltas);
        assert_eq!(
            r, "think",
            "Reasoning must land on the reasoning channel only"
        );
        assert_eq!(t, "ans", "Text must land on the content channel only");
        assert_eq!(c.len(), 1, "ToolCall must land on the calls channel only");
        // Empty-string deltas accumulate to empty strings (dropped by then_some).
        let (r2, t2, c2) = fold_unified_deltas(vec![
            UnifiedParserEvent::Reasoning(String::new()),
            UnifiedParserEvent::Text(String::new()),
        ]);
        assert!(r2.is_empty() && t2.is_empty() && c2.is_empty());
    }

    // Topology A: the worker already stripped markers, then the frontend aggregator
    // re-runs parse_complete_unified on the clean content. The re-run must be a no-op
    // (plain text stays content, never reclassified as reasoning), so the aggregator's
    // unconditional `choice.text = content` cannot wipe already-split output.
    #[test]
    fn parse_complete_unified_is_idempotent_on_stripped_text() {
        let (calls, reasoning, content) =
            parse_complete_unified("It's 18C.", None, "muse_glimmer").unwrap();
        assert!(calls.is_empty(), "no calls from plain text: {calls:?}");
        assert_eq!(
            content, "It's 18C.",
            "stripped text must round-trip as content, got reasoning={reasoning:?} content={content:?}"
        );
        assert!(
            reasoning.is_empty(),
            "stripped text must NOT be reclassified as reasoning: {reasoning:?}"
        );
    }

    // Every other family (tool-call or reasoning name) must return None so the guard
    // leaves its byte-for-byte original path untouched.
    #[test]
    fn unified_family_returns_none_for_other_families() {
        for other in ["deepseek_v4", "qwen3", "glm47", "harmony", "nemotron_deci"] {
            assert_eq!(unified_family(Some(other), None), None, "{other} tool");
            assert_eq!(unified_family(None, Some(other)), None, "{other} reasoning");
        }
    }

    fn two_choice_chunk(
        t0: &str,
        t1: &str,
        finish: bool,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let mut c = chunk(t0, finish);
        let data = c.data.as_mut().unwrap();
        let mut second = data.inner.choices[0].clone();
        second.index = 1;
        second.delta.content = Some(ChatCompletionMessageContent::Text(t1.to_string()));
        data.inner.choices.push(second);
        c
    }

    // n>1: each choice owns its own UnifiedChoiceState. Choice 0 emits a tool call;
    // choice 1 (reasoning + answer, no tool) must not inherit it, and the Stop->ToolCalls
    // flip must apply per choice.
    #[tokio::test]
    async fn muse_unified_isolates_state_across_choices() {
        let c1_turn = "<|start|>assistant to=self<|message|>C1 thought.<|eom|><|start|>assistant to=user<|message|>C1 answer.<|eot|>";
        let chunks = vec![
            two_choice_chunk(&muse_turn(), c1_turn, false),
            two_choice_chunk("", "", true),
        ];
        let out: Vec<_> =
            apply_unified_stream(stream::iter(chunks), None, "muse_glimmer".to_string())
                .collect::<Vec<_>>()
                .await;

        // Per-choice extraction.
        let per = |idx: u32| {
            let mut reasoning = String::new();
            let mut content = String::new();
            let mut names: Vec<String> = Vec::new();
            let mut finish = None;
            for r in &out {
                let Some(d) = r.data.as_ref() else { continue };
                for ch in &d.inner.choices {
                    if ch.index != idx {
                        continue;
                    }
                    if let Some(rc) = &ch.delta.reasoning_content {
                        reasoning.push_str(rc);
                    }
                    if let Some(ChatCompletionMessageContent::Text(t)) = &ch.delta.content {
                        content.push_str(t);
                    }
                    if let Some(tcs) = &ch.delta.tool_calls {
                        for tc in tcs {
                            if let Some(n) = tc.function.as_ref().and_then(|f| f.name.clone()) {
                                names.push(n);
                            }
                        }
                    }
                    if ch.finish_reason.is_some() {
                        finish = ch.finish_reason;
                    }
                }
            }
            (reasoning, content, names, finish)
        };
        let (r0, c0, n0, f0) = per(0);
        let (r1, c1, n1, f1) = per(1);
        assert_eq!(r0, "Look it up.", "choice0 reasoning");
        assert_eq!(c0, "It's 18C.", "choice0 content");
        assert_eq!(n0, vec!["get_weather".to_string()], "choice0 call");
        assert_eq!(f0, Some(FinishReason::ToolCalls), "choice0 finish flips");
        assert_eq!(r1, "C1 thought.", "choice1 reasoning");
        assert_eq!(c1, "C1 answer.", "choice1 content");
        assert!(
            n1.is_empty(),
            "choice1 must NOT inherit choice0's call: {n1:?}"
        );
        assert_eq!(f1, Some(FinishReason::Stop), "choice1 stays Stop (no flip)");
    }
}
