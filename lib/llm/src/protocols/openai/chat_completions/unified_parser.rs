// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in ordered reasoning, text, and tool-call parsing through one state machine.
//!
//! Gated behind
//! [`DYN_ENABLE_EXPERIMENTAL_PARSERS_V2`](dynamo_runtime::config::environment_names::llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2).
//!
//! # What this replaces
//!
//! Dynamo serves a Qwen3 turn today by chaining two independent parsers: the reasoning
//! parser strips `<think>...</think>` across the whole stream into one assembled
//! `reasoning_content`, and the tool-call jail then scans whatever is left. That chain
//! cannot represent WHERE a thought happened. Given
//!
//! ```text
//! <think>Look it up.</think><tool_call>…</tool_call><think>Now answer.</think>It's 18C.
//! ```
//!
//! it serves `reasoning("Look it up.Now answer.")`, then the call, then `text("It's 18C.")`:
//! the second thought moved ahead of the call it followed and fused with the first. A
//! client rendering thoughts inline puts them in the wrong place, and a client counting
//! reasoning turns sees one where there were two.
//!
//! Ordering is not a field the split can add — it is lost at the seam between the two
//! parsers. So when this path is enabled, ONE [`UnifiedParser`] owns the whole grammar
//! and emits deltas in the order the model produced them, and this module maps those
//! deltas onto the OpenAI streaming/batch wire shapes.
//!
//! # Why one chunk per delta
//!
//! [`ChatCompletionStreamResponseDelta`] carries `content`, `reasoning_content` and
//! `tool_calls` side by side with no way to say which came first. Packing a whole
//! `push` into one delta object would throw away exactly the ordering this path exists
//! to preserve, so every [`UnifiedDelta`] becomes its own chunk.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use async_stream::stream;
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers_v2::{
    Tool, UnifiedDelta, UnifiedEvent, UnifiedParser, UnifiedParserPrefill, UnifiedToolOutputMode,
    create_unified_parser_for_family,
};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    ChatCompletionToolChoiceOption, FinishReason, FunctionCall, FunctionCallStream, FunctionType,
};
use dynamo_runtime::config::{env_is_truthy, environment_names::llm as env_llm};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use uuid::Uuid;

use super::NvCreateChatCompletionStreamResponse;

/// The `dynamo-parsers-v2` unified family that serves Qwen3.
///
/// `REGISTERED_UNIFIED_FAMILIES` accepts both `qwen3` and the `qwen3_coder` alias for
/// the same XML grammar; `qwen3` is the canonical registry name and the one the
/// conformance corpus uses, so it is what this module passes and logs.
pub(crate) const QWEN3_UNIFIED_FAMILY: &str = "qwen3";

/// Dynamo's `--dyn-tool-call-parser` name that pairs into [`QWEN3_UNIFIED_FAMILY`].
const QWEN3_TOOL_CALL_PARSER: &str = "qwen3_coder";

/// Dynamo's `--dyn-reasoning-parser` name that pairs into [`QWEN3_UNIFIED_FAMILY`].
const QWEN3_REASONING_PARSER: &str = "qwen3";

/// Whether the experimental v2 parser path is enabled. Read once — env vars are fixed
/// for the process lifetime, so re-reading per request would only add syscalls.
///
/// This reuses `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` rather than adding a second switch.
/// That flag already means "route this family through `dynamo-parsers-v2` instead of the
/// v1 jail, for BOTH the batch and the streaming path". The unified parser is the same
/// intent carried one step further: it also takes over reasoning, so the family stops
/// needing a separate reasoning parser at all. Two flags would have to define what
/// setting only one of them means for a family that has both, and the answer is not
/// interesting — so there is one flag, and the parser PAIR decides which v2 shape a
/// request gets (see `configured_family`).
fn experimental_parsers_v2_enabled() -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| env_is_truthy(env_llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2));
    *ENABLED
}

/// The unified family this parser pair names, ignoring whether it is switched on.
///
/// Both halves must be set and must agree: a unified parser owns reasoning AND tool
/// calls, so taking over a request configured with only one of them, or with a
/// reasoning parser from a different family, would silently change what the operator
/// asked for. Split out from [`selected_family`] so the pairing rule can be tested
/// without the process-wide env flag.
pub(crate) fn configured_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    match (tool_call_parser, reasoning_parser) {
        (Some(QWEN3_TOOL_CALL_PARSER), Some(QWEN3_REASONING_PARSER)) => Some(QWEN3_UNIFIED_FAMILY),
        _ => None,
    }
}

/// The unified family to actually use for this parser pair, or `None` to keep the
/// existing split reasoning-parser + tool-call-jail path.
pub(crate) fn selected_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    // A request that silently fell back to the split path is otherwise
    // indistinguishable from one the unified parser handled — the two produce the
    // same shape of response. One INFO line per stream lets an operator answer
    // "did v2 actually run?" from the log instead of inferring it from a build
    // pin, which is exactly the ambiguity that cost real debugging time here.
    let configured = configured_family(tool_call_parser, reasoning_parser);
    tracing::info!(
        target: "dynamo_unified",
        ?tool_call_parser,
        ?reasoning_parser,
        ?configured,
        flag_on = experimental_parsers_v2_enabled(),
        "unified parser path decision"
    );
    configured.filter(|family| match *family {
        QWEN3_UNIFIED_FAMILY => experimental_parsers_v2_enabled(),
        // A family with no opt-in flag stays off; adding one here is what turns it on.
        _ => false,
    })
}

/// Which channel the rendered generation prompt already opened, for the streaming path.
///
/// `prompt_injected_reasoning` is the per-request fact the preprocessor already
/// computes: the rendered prompt ended with the family's reasoning opener, so generated
/// output starts inside a thought the model will close without ever emitting the
/// opener. Absent that, the answer is per family.
pub(crate) fn stream_prefill(
    family: &str,
    prompt_injected_reasoning: bool,
) -> UnifiedParserPrefill {
    if prompt_injected_reasoning {
        return UnifiedParserPrefill::Reasoning;
    }
    match family {
        // Qwen3's generation prompt ends at the assistant header with no channel open,
        // so the model emits `<think>` itself when it thinks.
        QWEN3_UNIFIED_FAMILY => UnifiedParserPrefill::None,
        // A family whose non-thinking prompt ends INSIDE the visible response channel
        // would return `Response` here. None exists yet; an unknown family gets the
        // conservative answer, which is to assume the model opens its own channels.
        _ => UnifiedParserPrefill::None,
    }
}

/// Which channel the prompt opened, inferred from complete output text.
///
/// The batch path has no prompt in hand, only what the model produced, so the channel
/// state is read back off the text: a `<think>` opener means the model opened reasoning
/// itself; a bare `</think>` with no opener means the prompt had already opened it; and
/// neither marker means reasoning never ran for this turn.
fn detect_prefill(family: &str, content: &str) -> anyhow::Result<UnifiedParserPrefill> {
    match family {
        QWEN3_UNIFIED_FAMILY => Ok(if content.contains("<think>") {
            UnifiedParserPrefill::None
        } else if content.contains("</think>") {
            UnifiedParserPrefill::Reasoning
        } else {
            UnifiedParserPrefill::Response
        }),
        other => anyhow::bail!("no prefill detector for unified parser family '{other}'"),
    }
}

/// Map dynamo's v1 [`ToolDefinition`]s onto the v2 parser's [`Tool`] shape.
fn to_v2_tools(tools: Option<&[ToolDefinition]>) -> Vec<Tool> {
    tools
        .unwrap_or(&[])
        .iter()
        .map(|tool| Tool {
            name: tool.name.clone(),
            description: None,
            parameters: tool.parameters.clone().unwrap_or(serde_json::Value::Null),
            strict: tool.strict,
        })
        .collect()
}

/// Map the request's `tool_choice` onto the wire format the backend will produce.
///
/// A named or `required` choice is served by guided decoding, which constrains the
/// model to bare JSON instead of Qwen's native `<tool_call>` XML: a named choice to
/// that one tool's argument object, `required` to a call object or an array of them.
/// Unset / `auto` / `none` leave the model in native markup.
///
/// Callers must not consult this when a structural tag is active — see
/// [`apply_stream`]'s `uses_tool_call_structural_tag`.
fn tool_output_mode(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
) -> UnifiedToolOutputMode<'_> {
    match tool_choice {
        Some(ChatCompletionToolChoiceOption::Named(named)) => UnifiedToolOutputMode::GuidedJson {
            named_tool: Some(&named.function.name),
        },
        Some(ChatCompletionToolChoiceOption::Required) => {
            UnifiedToolOutputMode::GuidedJson { named_tool: None }
        }
        None
        | Some(ChatCompletionToolChoiceOption::Auto)
        | Some(ChatCompletionToolChoiceOption::None) => UnifiedToolOutputMode::Native,
    }
}

/// Merge adjacent same-kind text/reasoning deltas so one `push` does not become three
/// chunks that say the same thing.
///
/// Tool-call deltas never merge: two fragments belonging to different `tool_index`es
/// would fuse into one call, and even two fragments of the SAME call must keep their
/// `name`-carrying first delta distinct from later argument-only ones.
fn coalesce(deltas: Vec<UnifiedDelta>) -> Vec<UnifiedDelta> {
    let mut out: Vec<UnifiedDelta> = Vec::with_capacity(deltas.len());
    for delta in deltas {
        match (out.last_mut(), delta) {
            (Some(UnifiedDelta::Text { text: prev }), UnifiedDelta::Text { text }) => {
                prev.push_str(&text)
            }
            (Some(UnifiedDelta::Reasoning { text: prev }), UnifiedDelta::Reasoning { text }) => {
                prev.push_str(&text)
            }
            (_, delta) => out.push(delta),
        }
    }
    out
}

/// An empty streaming choice for `index`, used as the base every emitted chunk is
/// filled in from.
///
/// `logprobs` is dropped on purpose: once parsing rewrites a choice, the emitted text
/// no longer lines up token-for-token with the backend's raw stream, so per-token
/// logprobs would be attached to the wrong characters.
fn empty_choice(index: u32) -> ChatChoiceStream {
    #[allow(deprecated)]
    ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role: None,
            content: None,
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    }
}

/// Per-choice streaming state: one parser instance plus the bookkeeping the OpenAI
/// streaming tool-call contract needs. One instance parses exactly one choice of one
/// request, which is what gives per-stream isolation by construction.
struct ChoiceState {
    family: &'static str,
    parser: Box<dyn UnifiedParser>,
    /// Tool indices whose opening chunk (id + type + name) has already gone out.
    opened_calls: HashSet<usize>,
    /// Whether any tool-call chunk was emitted; flips a terminal `Stop` to `ToolCalls`.
    tool_emitted: bool,
    /// The parser errored. Later chunks pass through as plain text instead of failing
    /// the request — a parser bug must not turn a served answer into a 500.
    failed: bool,
}

impl ChoiceState {
    fn new(
        family: &'static str,
        tools: &[Tool],
        prefill: UnifiedParserPrefill,
        tool_output_mode: UnifiedToolOutputMode<'_>,
    ) -> anyhow::Result<Self> {
        let mut parser = create_unified_parser_for_family(family, tools)?;
        parser.initialize_with_output_mode(prefill, tool_output_mode)?;
        Ok(Self {
            family,
            parser,
            opened_calls: HashSet::new(),
            tool_emitted: false,
            failed: false,
        })
    }

    /// Feed one decoded text delta through the parser.
    fn push(&mut self, text: &str) -> Vec<UnifiedDelta> {
        if self.failed {
            return text_delta(text.to_string());
        }
        match self.parser.push(text) {
            Ok(deltas) => deltas,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    family = self.family,
                    "unified parser push failed; falling back to plain text for this choice"
                );
                self.give_up(text)
            }
        }
    }

    /// Flush buffered partial state at end of stream.
    fn finish(&mut self) -> Vec<UnifiedDelta> {
        if self.failed {
            return Vec::new();
        }
        match self.parser.finish() {
            Ok(deltas) => deltas,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    family = self.family,
                    "unified parser finish failed; recovering buffered text"
                );
                self.give_up("")
            }
        }
    }

    /// Stop using the parser and surface whatever it was holding.
    ///
    /// `reset()` hands back the bytes the parser had buffered but not yet emitted.
    /// Dropping them would silently delete model output — the caller would see a
    /// truncated answer with no indication anything was lost — so they go out as
    /// visible text. When the parser had nothing buffered, the chunk that broke it
    /// does instead, so that chunk is not lost either.
    fn give_up(&mut self, fallback: &str) -> Vec<UnifiedDelta> {
        self.failed = true;
        let recovered = self.parser.reset();
        if recovered.is_empty() {
            text_delta(fallback.to_string())
        } else {
            text_delta(recovered)
        }
    }

    /// Convert one ordered delta into a streaming choice for `index`.
    fn delta_to_choice(&mut self, index: u32, delta: UnifiedDelta) -> ChatChoiceStream {
        let mut choice = empty_choice(index);
        match delta {
            UnifiedDelta::Text { text } => {
                choice.delta.content = Some(ChatCompletionMessageContent::Text(text));
            }
            UnifiedDelta::Reasoning { text } => {
                choice.delta.reasoning_content = Some(text);
            }
            UnifiedDelta::ToolCall(call) => {
                self.tool_emitted = true;
                // The OpenAI streaming tool-call contract: the FIRST chunk for a tool
                // index carries id + type + name, later chunks carry only argument
                // fragments. `dynamo-parsers-v2` mints no ids (serving layers own them),
                // so one is minted here per call, exactly once.
                let first = self.opened_calls.insert(call.tool_index);
                choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
                    index: call.tool_index as u32,
                    id: first.then(|| format!("call-{}", Uuid::new_v4())),
                    r#type: first.then_some(FunctionType::Function),
                    function: Some(FunctionCallStream {
                        name: first.then_some(call.name).flatten(),
                        arguments: Some(call.arguments),
                    }),
                }]);
            }
        }
        choice
    }

    /// Convert an ordered delta run into the streaming choices it becomes.
    ///
    /// `role` / `refusal` ride on the first emitted choice and the terminating
    /// `finish_reason` on the last, so a client that reassembles the stream sees the
    /// same envelope it would have without this path.
    fn choices_for(
        &mut self,
        original: &ChatChoiceStream,
        deltas: Vec<UnifiedDelta>,
        finish_reason: Option<FinishReason>,
    ) -> Vec<ChatChoiceStream> {
        let deltas = coalesce(deltas);
        let index = original.index;
        let count = deltas.len();
        let mut choices = Vec::with_capacity(count.max(1));

        for (position, delta) in deltas.into_iter().enumerate() {
            let mut choice = self.delta_to_choice(index, delta);
            if position == 0 {
                choice.delta.role = original.delta.role;
                choice.delta.refusal = original.delta.refusal.clone();
            }
            if position + 1 == count {
                choice.finish_reason = self.normalize_finish_reason(finish_reason);
            }
            choices.push(choice);
        }

        // The parser produced nothing, but the chunk still carried envelope state that
        // has to reach the client (the opening role chunk, a refusal, or the terminal
        // finish_reason).
        if choices.is_empty()
            && (original.delta.role.is_some()
                || original.delta.refusal.is_some()
                || finish_reason.is_some())
        {
            let mut choice = empty_choice(index);
            choice.delta.role = original.delta.role;
            choice.delta.refusal = original.delta.refusal.clone();
            choice.finish_reason = self.normalize_finish_reason(finish_reason);
            choices.push(choice);
        }

        choices
    }

    /// OpenAI streaming contract: once a choice has emitted tool calls, a `Stop`
    /// terminating reason must be reported as `ToolCalls`. `Length` / `ContentFilter`
    /// describe why generation stopped and are preserved as-is.
    fn normalize_finish_reason(&self, finish_reason: Option<FinishReason>) -> Option<FinishReason> {
        if finish_reason == Some(FinishReason::Stop) && self.tool_emitted {
            Some(FinishReason::ToolCalls)
        } else {
            finish_reason
        }
    }
}

/// One text delta, or nothing at all when the text is empty — an empty content chunk
/// carries no information and clients render it as a stray empty string.
fn text_delta(text: String) -> Vec<UnifiedDelta> {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![UnifiedDelta::Text { text }]
    }
}

/// The aggregated result of parsing one complete (non-streaming) output.
pub(crate) struct CompleteOutput {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<ChatCompletionMessageToolCall>,
}

/// Batch (non-streaming) path: run the whole output through the same parser lifecycle
/// and fold the assembled events into the final message.
///
/// Routing batch through `push`/`finish` is what makes stream/batch parity structural
/// rather than a property two code paths have to agree on.
///
/// Reasoning spans are concatenated because the non-streaming message schema has ONE
/// `reasoning_content` string — the ordering the unified parser recovered survives only
/// on the streaming path, which is where a client can act on it.
pub(crate) fn parse_complete(family: &str, content: &str) -> anyhow::Result<CompleteOutput> {
    let mut parser = create_unified_parser_for_family(family, &[])?;
    parser.initialize(detect_prefill(family, content)?)?;

    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for event in parser.parse_complete(content)? {
        match event {
            UnifiedEvent::Text { text: chunk } => text.push_str(&chunk),
            UnifiedEvent::Reasoning { text: chunk } => reasoning.push_str(&chunk),
            UnifiedEvent::ToolCall { name, arguments } => {
                tool_calls.push(ChatCompletionMessageToolCall {
                    id: format!("call-{}", Uuid::new_v4()),
                    r#type: FunctionType::Function,
                    // `assemble` already parsed the argument fragments into a typed
                    // object, so this re-serializes rather than passing the model's
                    // bytes through. Formatting is normalized as a result.
                    function: FunctionCall {
                        name,
                        arguments: serde_json::to_string(&arguments)?,
                    },
                });
            }
        }
    }

    Ok(CompleteOutput {
        text,
        reasoning,
        tool_calls,
    })
}

/// Finish every choice that never received a terminating chunk, in index order.
fn finish_unterminated_choices(
    states: &mut HashMap<u32, ChoiceState>,
    finished: &mut HashSet<u32>,
) -> Vec<ChatChoiceStream> {
    let mut indices: Vec<_> = states
        .keys()
        .copied()
        .filter(|index| !finished.contains(index))
        .collect();
    indices.sort_unstable();

    let mut choices = Vec::new();
    for index in indices {
        finished.insert(index);
        let state = states
            .get_mut(&index)
            .expect("index came from this map's keys");
        let deltas = state.finish();
        // A choice that emitted tool calls must terminate with `ToolCalls` even when
        // the backend never sent a finish_reason: a strict client waits for a non-null
        // one before considering the call complete, and would otherwise hang.
        let finish_reason = state.tool_emitted.then_some(FinishReason::ToolCalls);
        let base = empty_choice(index);
        choices.extend(state.choices_for(&base, deltas, finish_reason));
    }
    choices
}

/// Wrap one rewritten choice in a response built from `template`.
///
/// Usage, nvext and metrics are cleared: they belong to the chunk that carried them,
/// and repeating them on a synthesized chunk would double-count.
fn response_with_choice(
    template: &NvCreateChatCompletionStreamResponse,
    choice: ChatChoiceStream,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    let mut data = template.clone();
    data.inner.choices = vec![choice];
    data.inner.usage = None;
    data.nvext = None;
    data.llm_metrics = None;
    Annotated::from_data(data)
}

/// Whether a choice arrived already parsed by something upstream and must be passed
/// through untouched rather than re-parsed.
fn already_parsed(choice: &ChatChoiceStream) -> bool {
    matches!(
        choice.delta.content,
        Some(ChatCompletionMessageContent::Parts(_))
    ) || choice.delta.tool_calls.is_some()
        || choice.delta.reasoning_content.is_some()
}

/// Streaming path: one unified parser per response choice, replacing both the reasoning
/// parser and the tool-call jail for this request.
///
/// `uses_tool_call_structural_tag` reports that the backend was given a structural tag
/// constraining generation to the family's NATIVE grammar. When it is set the output is
/// native markup no matter what `tool_choice` says, so `tool_choice` is not consulted;
/// this path only parses guided JSON, it never builds the grammar that produces it.
pub(crate) fn apply_stream<S>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
    prefill: UnifiedParserPrefill,
    family: &'static str,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let tools = to_v2_tools(tool_definitions.as_deref());
    stream! {
        let mut states: HashMap<u32, ChoiceState> = HashMap::new();
        let mut finished: HashSet<u32> = HashSet::new();
        // Last data response with its choices cleared, so an end-of-stream flush has an
        // envelope (id, model, created) to attach synthesized chunks to.
        let mut template: Option<NvCreateChatCompletionStreamResponse> = None;

        tokio::pin!(stream_in);

        while let Some(mut response) = stream_in.next().await {
            let Some(chat) = response.data.as_mut() else {
                // Non-data annotations (errors, comments) pass through untouched.
                yield response;
                continue;
            };

            {
                let mut next = chat.clone();
                next.inner.choices.clear();
                next.inner.usage = None;
                next.nvext = None;
                next.llm_metrics = None;
                template = Some(next);
            }

            if chat.inner.choices.is_empty() {
                // A usage-only chunk. OpenAI stream ordering requires every choice's
                // terminal finish_reason to precede it, so flush first.
                if let Some(template) = &template {
                    for choice in finish_unterminated_choices(&mut states, &mut finished) {
                        yield response_with_choice(template, choice);
                    }
                }
                yield response;
                continue;
            }

            let originals = std::mem::take(&mut chat.inner.choices);
            let mut emitted: Vec<ChatChoiceStream> = Vec::new();
            for original in originals {
                if already_parsed(&original) {
                    if original.finish_reason.is_some() {
                        finished.insert(original.index);
                    }
                    emitted.push(original);
                    continue;
                }

                let state = match states.entry(original.index) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        // A structural tag pins generation to native markup; only
                        // without one does tool_choice select guided JSON.
                        let mode = if uses_tool_call_structural_tag {
                            UnifiedToolOutputMode::Native
                        } else {
                            tool_output_mode(tool_choice.as_ref())
                        };
                        match ChoiceState::new(family, &tools, prefill, mode) {
                            Ok(state) => entry.insert(state),
                            Err(error) => {
                                tracing::warn!(
                                    error = %error,
                                    family,
                                    choice = original.index,
                                    "unified parser construction failed; passing choice through"
                                );
                                emitted.push(original);
                                continue;
                            }
                        }
                    }
                };

                let mut deltas = Vec::new();
                if let Some(ChatCompletionMessageContent::Text(text)) =
                    original.delta.content.as_ref()
                {
                    deltas.extend(state.push(text));
                }
                let terminal = original.finish_reason;
                if terminal.is_some() && finished.insert(original.index) {
                    deltas.extend(state.finish());
                }

                let mut parsed = state.choices_for(&original, deltas, terminal);
                if parsed.is_empty() {
                    // A marker-only chunk produced no deltas. Keep it as an empty
                    // choice so the typed llm_metrics and annotation metadata it
                    // carries still reach the client.
                    parsed.push(empty_choice(original.index));
                }
                emitted.extend(parsed);
            }

            if emitted.is_empty() {
                continue;
            }

            // One upstream chunk can fan out into several. Only the LAST carries the
            // usage / nvext / metrics and the annotation fields, so nothing is counted
            // or reported twice.
            let last = emitted.len() - 1;
            for (position, choice) in emitted.into_iter().enumerate() {
                let is_last = position == last;
                let mut data = chat.clone();
                data.inner.choices = vec![choice];
                if !is_last {
                    data.inner.usage = None;
                    data.nvext = None;
                    data.llm_metrics = None;
                }
                yield Annotated {
                    data: Some(data),
                    id: if is_last { response.id.take() } else { None },
                    event: if is_last { response.event.take() } else { None },
                    comment: if is_last { response.comment.take() } else { None },
                    error: if is_last { response.error.take() } else { None },
                };
            }
        }

        // Backstop: the stream ended without a terminating chunk for some choice.
        if let Some(template) = &template {
            for choice in finish_unterminated_choices(&mut states, &mut finished) {
                yield response_with_choice(template, choice);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers_v2::ToolCallDelta;
    use dynamo_protocols::types::{CreateChatCompletionStreamResponse, Role};
    use futures::stream;

    fn chunk(text: &str, finish: bool) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let response = NvCreateChatCompletionStreamResponse {
            inner: CreateChatCompletionStreamResponse {
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
                model: "qwen3".to_string(),
                system_fingerprint: None,
                service_tier: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
            },
            nvext: None,
            llm_metrics: None,
        };
        Annotated::from_data(response)
    }

    fn weather_tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            })),
            strict: None,
        }]
    }

    fn collect_choices(
        responses: &[Annotated<NvCreateChatCompletionStreamResponse>],
    ) -> Vec<&ChatChoiceStream> {
        responses
            .iter()
            .filter_map(|response| response.data.as_ref())
            .flat_map(|data| data.inner.choices.iter())
            .collect()
    }

    fn named_choice(name: &str) -> ChatCompletionToolChoiceOption {
        serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {"name": name}
        }))
        .expect("named tool choice")
    }

    // --- selected_family / configured_family -------------------------------------

    #[test]
    fn pairs_only_qwen3_coder_with_qwen3() {
        assert_eq!(
            configured_family(Some("qwen3_coder"), Some("qwen3")),
            Some(QWEN3_UNIFIED_FAMILY)
        );
        // A unified parser owns BOTH halves, so half a pair must not opt in.
        assert_eq!(configured_family(Some("qwen3_coder"), None), None);
        assert_eq!(configured_family(None, Some("qwen3")), None);
        assert_eq!(configured_family(None, None), None);
        // Right tool parser, wrong reasoning family.
        assert_eq!(
            configured_family(Some("qwen3_coder"), Some("deepseek_r1")),
            None
        );
        // Right reasoning parser, wrong tool family.
        assert_eq!(configured_family(Some("kimi_k2"), Some("qwen3")), None);
        // The pairing is on dynamo's parser names, not the unified family name.
        assert_eq!(configured_family(Some("qwen3"), Some("qwen3")), None);
    }

    #[test]
    fn selected_family_needs_the_env_flag() {
        // The env flag is process-wide and read once, so this asserts the relationship
        // between the two functions rather than mutating the environment: whatever the
        // flag says, `selected_family` never selects a pair `configured_family` rejects,
        // and it agrees with `configured_family` exactly when the flag is on.
        let pair = (Some("qwen3_coder"), Some("qwen3"));
        assert_eq!(
            configured_family(pair.0, pair.1),
            Some(QWEN3_UNIFIED_FAMILY)
        );
        if experimental_parsers_v2_enabled() {
            assert_eq!(
                selected_family(pair.0, pair.1),
                Some(QWEN3_UNIFIED_FAMILY),
                "flag on: the configured pair must be selected"
            );
        } else {
            assert_eq!(
                selected_family(pair.0, pair.1),
                None,
                "flag off: the configured pair must NOT be selected"
            );
        }
        // Never selected regardless of the flag.
        assert_eq!(selected_family(Some("qwen3_coder"), None), None);
        assert_eq!(selected_family(None, Some("qwen3")), None);
    }

    // --- tool_choice -> UnifiedToolOutputMode ------------------------------------

    #[test]
    fn maps_tool_choice_onto_the_output_mode() {
        let named = named_choice("get_weather");
        assert_eq!(
            tool_output_mode(Some(&named)),
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather")
            }
        );
        assert_eq!(
            tool_output_mode(Some(&ChatCompletionToolChoiceOption::Required)),
            UnifiedToolOutputMode::GuidedJson { named_tool: None }
        );
        for native in [
            None,
            Some(&ChatCompletionToolChoiceOption::Auto),
            Some(&ChatCompletionToolChoiceOption::None),
        ] {
            assert_eq!(
                tool_output_mode(native),
                UnifiedToolOutputMode::Native,
                "{native:?} must stay on native markup"
            );
        }
    }

    #[tokio::test]
    async fn structural_tag_keeps_required_on_native_markup() {
        // A structural tag constrains generation to Qwen's XML, so `required` must NOT
        // put the parser into guided-JSON mode; doing so would surface the whole call
        // as text.
        let output = concat!(
            "<think>reason</think>",
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let responses = apply_stream(
            stream::iter([chunk(output, true)]),
            Some(weather_tools()),
            Some(ChatCompletionToolChoiceOption::Required),
            true,
            UnifiedParserPrefill::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        let call = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .next()
            .expect("a native tool call");
        let function = call.function.as_ref().expect("function");
        assert_eq!(function.name.as_deref(), Some("get_weather"));
        assert_eq!(function.arguments.as_deref(), Some(r#"{"city":"Tokyo"}"#));
    }

    // --- prefill -----------------------------------------------------------------

    #[test]
    fn prompt_injected_reasoning_selects_the_reasoning_prefill() {
        assert_eq!(
            stream_prefill(QWEN3_UNIFIED_FAMILY, true),
            UnifiedParserPrefill::Reasoning
        );
        // Qwen3's template pre-opens no channel, so the model emits `<think>` itself.
        assert_eq!(
            stream_prefill(QWEN3_UNIFIED_FAMILY, false),
            UnifiedParserPrefill::None
        );
    }

    #[test]
    fn detects_batch_prefill_from_complete_output() {
        assert_eq!(
            detect_prefill(QWEN3_UNIFIED_FAMILY, "<think>reason</think>answer").unwrap(),
            UnifiedParserPrefill::None
        );
        assert_eq!(
            detect_prefill(QWEN3_UNIFIED_FAMILY, "reason</think>answer").unwrap(),
            UnifiedParserPrefill::Reasoning
        );
        assert_eq!(
            detect_prefill(QWEN3_UNIFIED_FAMILY, "answer").unwrap(),
            UnifiedParserPrefill::Response
        );
        assert!(detect_prefill("kimi_k3", "answer").is_err());
    }

    // --- delta -> chunk conversion -----------------------------------------------

    fn call_delta(tool_index: usize, name: Option<&str>, arguments: &str) -> UnifiedDelta {
        UnifiedDelta::ToolCall(ToolCallDelta {
            tool_index,
            name: name.map(str::to_string),
            arguments: arguments.to_string(),
        })
    }

    fn test_state() -> ChoiceState {
        ChoiceState::new(
            QWEN3_UNIFIED_FAMILY,
            &[],
            UnifiedParserPrefill::None,
            UnifiedToolOutputMode::Native,
        )
        .expect("qwen3 unified parser")
    }

    #[test]
    fn each_delta_becomes_its_own_chunk_in_order() {
        // The whole point of this path: a thought that followed a call stays after it.
        let mut state = test_state();
        let base = empty_choice(0);
        let choices = state.choices_for(
            &base,
            vec![
                UnifiedDelta::Reasoning {
                    text: "look it up".into(),
                },
                call_delta(0, Some("get_weather"), r#"{"city":"Tokyo"}"#),
                UnifiedDelta::Reasoning {
                    text: "now answer".into(),
                },
                UnifiedDelta::Text {
                    text: "It's 18C.".into(),
                },
            ],
            Some(FinishReason::Stop),
        );

        assert_eq!(choices.len(), 4, "one chunk per delta: {choices:?}");
        assert_eq!(
            choices[0].delta.reasoning_content.as_deref(),
            Some("look it up")
        );
        assert!(choices[1].delta.tool_calls.is_some());
        assert_eq!(
            choices[2].delta.reasoning_content.as_deref(),
            Some("now answer"),
            "the second thought must stay AFTER the call, not fuse with the first"
        );
        assert_eq!(
            choices[3].delta.content,
            Some(ChatCompletionMessageContent::Text("It's 18C.".into()))
        );
        // Only the last chunk terminates, and Stop became ToolCalls.
        assert_eq!(choices[3].finish_reason, Some(FinishReason::ToolCalls));
        assert!(choices[..3].iter().all(|c| c.finish_reason.is_none()));
    }

    #[test]
    fn first_tool_chunk_opens_the_call_and_later_ones_only_add_arguments() {
        let mut state = test_state();
        let base = empty_choice(7);
        let choices = state.choices_for(
            &base,
            vec![
                call_delta(0, Some("get_weather"), r#"{"city":"#),
                call_delta(0, None, r#""Tokyo"}"#),
            ],
            None,
        );

        assert_eq!(choices.len(), 2);
        let first = &choices[0].delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(choices[0].index, 7, "the choice index is preserved");
        assert_eq!(first.index, 0, "the tool index is preserved");
        assert!(first.id.is_some(), "the opening chunk mints an id");
        assert_eq!(first.r#type, Some(FunctionType::Function));
        assert_eq!(
            first.function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );

        let second = &choices[1].delta.tool_calls.as_ref().unwrap()[0];
        assert!(second.id.is_none(), "only the first chunk carries an id");
        assert!(second.r#type.is_none());
        assert!(second.function.as_ref().unwrap().name.is_none());
        assert_eq!(
            second.function.as_ref().unwrap().arguments.as_deref(),
            Some(r#""Tokyo"}"#)
        );
    }

    #[test]
    fn two_calls_get_distinct_ids_and_keep_their_indices() {
        let mut state = test_state();
        let base = empty_choice(0);
        let choices = state.choices_for(
            &base,
            vec![
                call_delta(0, Some("a"), "{}"),
                call_delta(1, Some("b"), "{}"),
            ],
            None,
        );

        let ids: Vec<_> = choices
            .iter()
            .map(|choice| {
                let call = &choice.delta.tool_calls.as_ref().unwrap()[0];
                (call.index, call.id.clone().expect("id"))
            })
            .collect();
        assert_eq!(ids[0].0, 0);
        assert_eq!(ids[1].0, 1);
        assert_ne!(ids[0].1, ids[1].1, "each call gets its own id");
    }

    #[test]
    fn adjacent_same_kind_deltas_coalesce_but_calls_never_do() {
        let merged = coalesce(vec![
            UnifiedDelta::Text { text: "he".into() },
            UnifiedDelta::Text { text: "llo".into() },
            call_delta(0, Some("f"), "{"),
            call_delta(0, None, "}"),
            UnifiedDelta::Reasoning { text: "a".into() },
            UnifiedDelta::Reasoning { text: "b".into() },
        ]);
        assert_eq!(merged.len(), 4);
        assert_eq!(
            merged[0],
            UnifiedDelta::Text {
                text: "hello".into()
            }
        );
        assert_eq!(merged[1], call_delta(0, Some("f"), "{"));
        assert_eq!(merged[2], call_delta(0, None, "}"));
        assert_eq!(merged[3], UnifiedDelta::Reasoning { text: "ab".into() });
    }

    #[test]
    fn role_rides_the_first_chunk_and_finish_reason_the_last() {
        let mut state = test_state();
        let mut base = empty_choice(0);
        base.delta.role = Some(Role::Assistant);
        let choices = state.choices_for(
            &base,
            vec![
                UnifiedDelta::Text { text: "a".into() },
                call_delta(0, Some("f"), "{}"),
            ],
            Some(FinishReason::Stop),
        );

        assert_eq!(choices[0].delta.role, Some(Role::Assistant));
        assert!(choices[1].delta.role.is_none());
        assert!(choices[0].finish_reason.is_none());
        assert_eq!(choices[1].finish_reason, Some(FinishReason::ToolCalls));
    }

    #[test]
    fn an_empty_run_still_carries_the_terminating_envelope() {
        let mut state = test_state();
        let mut base = empty_choice(0);
        base.delta.role = Some(Role::Assistant);
        let choices = state.choices_for(&base, Vec::new(), Some(FinishReason::Length));

        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0].delta.role, Some(Role::Assistant));
        assert_eq!(
            choices[0].finish_reason,
            Some(FinishReason::Length),
            "Length is not rewritten"
        );
    }

    #[test]
    fn an_empty_run_with_no_envelope_emits_nothing() {
        let mut state = test_state();
        let base = empty_choice(0);
        assert!(state.choices_for(&base, Vec::new(), None).is_empty());
    }

    // --- end-to-end streaming ----------------------------------------------------

    #[tokio::test]
    async fn streams_ordered_reasoning_text_and_tool_calls() {
        let output = concat!(
            "<think>reason</think>answer ",
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        // Split into small chunks so markers straddle push() boundaries.
        let mut chunks: Vec<_> = output
            .as_bytes()
            .chunks(7)
            .map(|bytes| chunk(std::str::from_utf8(bytes).unwrap(), false))
            .collect();
        chunks.push(chunk("", true));

        let responses = apply_stream(
            stream::iter(chunks),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserPrefill::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        let reasoning: String = choices
            .iter()
            .filter_map(|choice| choice.delta.reasoning_content.as_deref())
            .collect();
        let content: String = choices
            .iter()
            .filter_map(|choice| match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let arguments: String = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();

        assert_eq!(reasoning, "reason");
        assert_eq!(content, "answer ");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments).unwrap()["city"],
            "Tokyo"
        );
        for marker in ["<think>", "<tool_call>", "<function=", "<parameter="] {
            assert!(
                !content.contains(marker),
                "raw markup {marker:?} leaked into content: {content:?}"
            );
        }
        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .next_back(),
            Some(FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn named_guided_json_becomes_a_tool_call() {
        let responses = apply_stream(
            stream::iter([chunk("reason</think>{\"city\": \"Tokyo\"}", true)]),
            Some(weather_tools()),
            Some(named_choice("get_weather")),
            false,
            UnifiedParserPrefill::Reasoning,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        assert_eq!(
            choices[0].delta.reasoning_content.as_deref(),
            Some("reason")
        );
        let call = &choices[1].delta.tool_calls.as_ref().unwrap()[0];
        let function = call.function.as_ref().unwrap();
        assert_eq!(function.name.as_deref(), Some("get_weather"));
        assert_eq!(
            function.arguments.as_deref(),
            Some("{\"city\": \"Tokyo\"}"),
            "a named choice passes the model's argument bytes through verbatim"
        );
        assert_eq!(choices[1].finish_reason, Some(FinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn required_guided_json_becomes_a_tool_call() {
        let responses = apply_stream(
            stream::iter([chunk(
                r#"reason</think>[{"name":"get_weather","parameters":{"city":"Tokyo"}}]"#,
                true,
            )]),
            Some(weather_tools()),
            Some(ChatCompletionToolChoiceOption::Required),
            false,
            UnifiedParserPrefill::Reasoning,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        let call = &choices[1].delta.tool_calls.as_ref().unwrap()[0];
        let function = call.function.as_ref().unwrap();
        assert_eq!(function.name.as_deref(), Some("get_weather"));
        assert_eq!(function.arguments.as_deref(), Some(r#"{"city":"Tokyo"}"#));
    }

    #[tokio::test]
    async fn already_parsed_choices_pass_through_untouched() {
        // A chunk that already carries reasoning_content was parsed upstream; running
        // it through the parser again would double it.
        let mut pre_parsed = chunk("", false);
        let choice = &mut pre_parsed.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("already".to_string());

        let responses = apply_stream(
            stream::iter([pre_parsed]),
            None,
            None,
            false,
            UnifiedParserPrefill::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        assert_eq!(choices.len(), 1);
        assert_eq!(
            choices[0].delta.reasoning_content.as_deref(),
            Some("already")
        );
    }

    #[tokio::test]
    async fn a_stream_without_a_finish_reason_still_terminates_a_tool_call() {
        let output = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        // No terminating chunk at all — the stream just ends.
        let responses = apply_stream(
            stream::iter([chunk(output, false)]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserPrefill::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .next_back(),
            Some(FinishReason::ToolCalls),
            "the backstop must synthesize a terminal reason or a strict client hangs"
        );
    }

    #[tokio::test]
    async fn text_only_stream_gets_no_synthetic_finish_reason() {
        let responses = apply_stream(
            stream::iter([chunk("hello world", false)]),
            None,
            None,
            false,
            UnifiedParserPrefill::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        assert!(
            choices.iter().all(|choice| choice.finish_reason.is_none()),
            "there is no signal to synthesize a finish_reason from"
        );
    }

    // --- error recovery ----------------------------------------------------------

    #[test]
    fn a_failed_parser_surfaces_buffered_text_and_stops_parsing() {
        let mut state = test_state();
        // The parser releases the settled text immediately and holds back only the
        // bytes that could still turn out to be a `<tool_call>` opener.
        assert_eq!(
            state.push("hello <tool_c"),
            vec![UnifiedDelta::Text {
                text: "hello ".into()
            }]
        );
        // Now force the failure path with those held-back bytes still buffered.
        let recovered = state.give_up("");
        assert_eq!(
            recovered,
            vec![UnifiedDelta::Text {
                text: "<tool_c".into()
            }],
            "buffered bytes must be surfaced, not silently dropped"
        );
        assert!(state.failed);
        // Every later chunk now passes through as plain text.
        assert_eq!(
            state.push("more"),
            vec![UnifiedDelta::Text {
                text: "more".into()
            }]
        );
        assert!(state.finish().is_empty());
    }

    #[test]
    fn give_up_falls_back_to_the_chunk_when_nothing_was_buffered() {
        let mut state = test_state();
        assert_eq!(
            state.give_up("the chunk that broke it"),
            vec![UnifiedDelta::Text {
                text: "the chunk that broke it".into()
            }]
        );
    }

    // --- batch -------------------------------------------------------------------

    #[test]
    fn parses_complete_output_with_reasoning_and_a_tool_call() {
        let parsed = parse_complete(
            QWEN3_UNIFIED_FAMILY,
            concat!(
                "<think>reason</think>answer ",
                "<tool_call>\n<function=get_weather>\n",
                "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
            ),
        )
        .unwrap();

        assert_eq!(parsed.reasoning, "reason");
        assert_eq!(parsed.text, "answer ");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            parsed.tool_calls[0].function.arguments,
            r#"{"city":"Tokyo"}"#
        );
        assert!(parsed.tool_calls[0].id.starts_with("call-"));
    }

    #[test]
    fn parses_complete_output_with_a_reasoning_prefill() {
        let parsed = parse_complete(QWEN3_UNIFIED_FAMILY, "hidden</think>visible").unwrap();
        assert_eq!(parsed.reasoning, "hidden");
        assert_eq!(parsed.text, "visible");
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn plain_text_passes_through_the_batch_path_unchanged() {
        let parsed = parse_complete(QWEN3_UNIFIED_FAMILY, "just an answer").unwrap();
        assert_eq!(parsed.text, "just an answer");
        assert!(parsed.reasoning.is_empty());
        assert!(parsed.tool_calls.is_empty());
    }
}
