// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool calls routed through the `dynamo-parsers-v2` streaming parser, bypassing the jail.
//!
//! Gated behind
//! [`DYN_ENABLE_EXPERIMENTAL_PARSERS_V2`](dynamo_runtime::config::environment_names::llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2).
//! When enabled, the families in [`V2_FAMILIES`] (Qwen3-Coder, DeepSeek-V4) stream
//! straight through their `dynamo_parsers_v2` parser instead of
//! [`JailedStream`](super::jail::JailedStream): the v2 parser owns incremental
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
use dynamo_parsers_v2::{Tool as ToolV2, ToolCallDelta, ToolParser, create_tool_parser_for_family};

use super::NvCreateChatCompletionStreamResponse;

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

    /// Map v2 per-chunk deltas onto OpenAI streaming tool-call chunks. The first
    /// delta for a given tool index carries the minted id, type and function name;
    /// later deltas for that index carry only argument fragments (`id`/`type`/`name`
    /// `None`), matching the OpenAI streaming tool-call contract.
    fn emit_chunks(
        &mut self,
        calls: Vec<ToolCallDelta>,
    ) -> Option<Vec<ChatCompletionMessageToolCallChunk>> {
        if calls.is_empty() {
            return None;
        }
        let chunks = calls
            .into_iter()
            .map(|delta| {
                let first = self.opened.insert(delta.tool_index);
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

            yield response;
        }

        // Backstop: the stream ended without a finish_reason for some choice. Flush
        // each unfinished parser; emit a trailing chunk when the flush yields output
        // or when the choice already emitted tool calls and still needs a terminal
        // `ToolCalls` reason.
        if let Some(template) = template {
            for (index, state) in states.iter_mut() {
                if finished.contains(index) {
                    continue;
                }
                let Ok(result) = state.parser.finish() else {
                    continue;
                };
                let tool_calls = state.emit_chunks(result.calls);
                if tool_calls.is_some() {
                    tool_emitted.insert(*index);
                }
                // A choice that produced tool calls during the stream must terminate
                // with `ToolCalls` even when the backend never sent a finish_reason
                // chunk (e.g. speculative decoding folded EOS into content, or the
                // engine dropped the terminal signal). Strict OpenAI clients wait for
                // a non-null finish_reason before considering a tool call complete;
                // emitting `None` here left them hanging until their client-side
                // timeout (DGH-967). Text-only output without an upstream finish
                // reason stays `None` — we have no signal to invent one from.
                let finish_reason = if tool_emitted.contains(index) {
                    Some(FinishReason::ToolCalls)
                } else {
                    None
                };
                if result.normal_text.is_empty()
                    && tool_calls.is_none()
                    && finish_reason.is_none()
                {
                    continue;
                }
                let mut response = template.clone();
                #[allow(deprecated)]
                let choice = dynamo_protocols::types::ChatChoiceStream {
                    index: *index,
                    delta: dynamo_protocols::types::ChatCompletionStreamResponseDelta {
                        role: None,
                        content: (!result.normal_text.is_empty())
                            .then_some(ChatCompletionMessageContent::Text(result.normal_text)),
                        tool_calls,
                        function_call: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason,
                    logprobs: None,
                };
                response.inner.choices = vec![choice];
                yield Annotated {
                    data: Some(response),
                    id: None,
                    event: None,
                    comment: None,
                    error: None,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, FinishReason, Role,
    };
    use futures::stream;

    const QWEN3_GET_WEATHER: &str = "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>";

    // DeepSeek-V4 DSML: one get_weather(location="NYC") call. The `｜` glyphs are the
    // fullwidth vertical bars the DSML markers use (see dynamo_parsers_v2::dsml).
    const DSV4_GET_WEATHER: &str = "<｜DSML｜tool_calls> <｜DSML｜invoke name=\"get_weather\"> <｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter> </｜DSML｜invoke> </｜DSML｜tool_calls>";

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

    // DGH-967: the stream emits a complete tool call but ends WITHOUT any
    // finish_reason chunk (e.g. speculative decoding folded EOS into content, or
    // the engine dropped the terminal signal). A strict OpenAI client waits for a
    // non-null finish_reason before considering the tool call complete; the
    // end-of-stream backstop must synthesize `ToolCalls` so the client doesn't
    // hang until its timeout.
    #[tokio::test]
    async fn qwen3_bypass_synthesizes_tool_calls_when_stream_lacks_finish_reason() {
        // Same call as the incremental test, but the final chunk carries NO
        // finish_reason — the stream simply ends after the tool markup.
        let mut chunks: Vec<_> = QWEN3_GET_WEATHER
            .as_bytes()
            .chunks(8)
            .map(|b| chunk(std::str::from_utf8(b).unwrap(), false))
            .collect();
        // Trailing empty chunk with finish=false — no finish_reason at all.
        chunks.push(chunk("", false));

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
    }

    // DGH-967 corollary: when the stream ends without a finish_reason AND no tool
    // call was emitted (text-only output), the backstop must NOT invent a
    // finish_reason — there is no signal to synthesize one from. A trailing
    // content chunk may be emitted, but its finish_reason stays None.
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
}
