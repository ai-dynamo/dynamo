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
//! to preserve, so every [`UnifiedParserEvent`] becomes its own chunk.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use async_stream::stream;
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Tool, UnifiedEvent, UnifiedParser, UnifiedParserEvent,
    UnifiedParserExt, UnifiedParserInit, UnifiedParserOutput, UnifiedParserStartingState,
    UnifiedToolOutputMode, create_unified_parser_for_family,
};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta, FinishReason,
    FunctionCall, FunctionCallStream, FunctionType,
};
use dynamo_runtime::config::{env_is_truthy, environment_names::llm as env_llm};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use uuid::Uuid;

use crate::protocols::openai::GuidedToolConstraint;

use super::NvCreateChatCompletionStreamResponse;

#[cfg(test)]
use dynamo_protocols::types::ChatCompletionToolChoiceOption;

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

/// The configured family eligible to own raw aggregate text.
///
/// Every request for the configured pair opts into the v2 batch parser. Requests that
/// suppress calls still need the unified decoder to strip native markup before the
/// calls are discarded, while named/required raw text is decoded using the installed
/// constraint plus the observed native-marker fallback in [`batch_tool_output_mode`].
pub(crate) fn configured_batch_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    configured_family(tool_call_parser, reasoning_parser)
}

pub(crate) fn selected_batch_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    configured_batch_family(tool_call_parser, reasoning_parser).filter(|family| match *family {
        QWEN3_UNIFIED_FAMILY => experimental_parsers_v2_enabled(),
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
) -> UnifiedParserStartingState {
    if prompt_injected_reasoning {
        return UnifiedParserStartingState::Reasoning;
    }
    match family {
        // Qwen3's generation prompt ends at the assistant header with no channel open,
        // so the model emits `<think>` itself when it thinks.
        QWEN3_UNIFIED_FAMILY => UnifiedParserStartingState::None,
        // A family whose non-thinking prompt ends INSIDE the visible response channel
        // would return `Response` here. None exists yet; an unknown family gets the
        // conservative answer, which is to assume the model opens its own channels.
        _ => UnifiedParserStartingState::None,
    }
}

/// Which channel the prompt opened, inferred from complete output text.
///
/// The batch path has no prompt in hand, only what the model produced, so the channel
/// state is read back off the text: a `<think>` opener means the model opened reasoning
/// itself; a bare `</think>` with no opener means the prompt had already opened it; and
/// neither marker means reasoning never ran for this turn.
fn detect_prefill(family: &str, content: &str) -> anyhow::Result<UnifiedParserStartingState> {
    match family {
        QWEN3_UNIFIED_FAMILY => Ok(if contains_unquoted_marker(content, "<think>") {
            UnifiedParserStartingState::None
        } else if contains_unquoted_marker(content, "</think>") {
            UnifiedParserStartingState::Reasoning
        } else {
            UnifiedParserStartingState::Response
        }),
        other => anyhow::bail!("no prefill detector for unified parser family '{other}'"),
    }
}

/// Marker-looking text quoted as prose is visible content, not parser control syntax.
pub(crate) fn contains_unquoted_marker(content: &str, marker: &str) -> bool {
    let chars: Vec<(usize, char)> = content.char_indices().collect();
    // Whether an unescaped `"`, `'`, or `` ` `` appears anywhere at or after each
    // position, precomputed once in O(n) so the scan below doesn't rescan the
    // remainder of the string for every candidate quote character (which would be
    // O(n^2) on adversarial text with many non-closing quote-like characters).
    let closes_later = later_unescaped_quote_closes(&chars);

    let mut quote = None;
    let mut escaped = false;
    for (position, &(offset, character)) in chars.iter().enumerate() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == active_quote {
                quote = None;
            }
            continue;
        }
        let apostrophe = character == '\''
            && content[..offset]
                .chars()
                .next_back()
                .is_some_and(char::is_alphanumeric)
            && content[offset + character.len_utf8()..]
                .chars()
                .next()
                .is_some_and(char::is_alphanumeric);
        // A quote that never closes before end-of-string was never really quoting
        // anything; treating it as an open span would blind the rest of the scan to
        // a real marker that follows (e.g. a stray `"` before a genuine `</think>`).
        if let Some(slot) = quote_slot(character)
            && !apostrophe
            && closes_later[position + 1][slot]
        {
            quote = Some(character);
            continue;
        }
        if content[offset..].starts_with(marker) {
            return true;
        }
    }
    false
}

/// Index into the fixed-size "which quote characters close later" arrays used by
/// [`later_unescaped_quote_closes`].
fn quote_slot(character: char) -> Option<usize> {
    match character {
        '"' => Some(0),
        '\'' => Some(1),
        '`' => Some(2),
        _ => None,
    }
}

/// For each position in `chars`, whether an unescaped `"`, `'`, or `` ` `` appears
/// anywhere later in the string — `result[i][quote_slot(c)]` answers "does an
/// unescaped `c` exist in `chars[i..]`". Built in one right-to-left O(n) pass: the
/// answer at `i` depends only on `chars[i]` and the answer at `i + 1` (or `i + 2` for
/// a backslash, which escapes exactly the next character), so this is an exact
/// backward re-expression of the left-to-right escaped-flag scan that used to be
/// re-run from scratch for every candidate quote character.
fn later_unescaped_quote_closes(chars: &[(usize, char)]) -> Vec<[bool; 3]> {
    let n = chars.len();
    let mut closes = vec![[false; 3]; n + 1];
    let mut i = n;
    while i > 0 {
        i -= 1;
        let character = chars[i].1;
        closes[i] = if character == '\\' {
            closes.get(i + 2).copied().unwrap_or([false; 3])
        } else {
            let mut next = closes[i + 1];
            if let Some(slot) = quote_slot(character) {
                next[slot] = true;
            }
            next
        };
    }
    closes
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
fn tool_output_mode(constraint: &GuidedToolConstraint) -> UnifiedToolOutputMode {
    match constraint {
        GuidedToolConstraint::GuidedJsonNamed { tool_name } => UnifiedToolOutputMode::GuidedJson {
            named_tool: Some(tool_name.clone()),
        },
        GuidedToolConstraint::GuidedJsonRequired => {
            UnifiedToolOutputMode::GuidedJson { named_tool: None }
        }
        GuidedToolConstraint::None | GuidedToolConstraint::StructuralTag => {
            UnifiedToolOutputMode::Native
        }
    }
}

/// Select the complete-output grammar from the bytes that actually arrived.
///
/// A remote frontend can reconstruct a forced request as guided JSON without seeing
/// that the worker installed a structural tag. Native Qwen markup is unambiguous once
/// generated, so prefer that observed shape over the reconstructed request policy.
fn batch_tool_output_mode(
    content: &str,
    constraint: &GuidedToolConstraint,
) -> UnifiedToolOutputMode {
    if constraint.installs_guided_json() && contains_unquoted_marker(content, "<tool_call>") {
        UnifiedToolOutputMode::Native
    } else {
        tool_output_mode(constraint)
    }
}

/// Merge adjacent same-kind text/reasoning deltas so one `push` does not become three
/// chunks that say the same thing.
///
/// Tool-call deltas never merge: two fragments belonging to different `tool_index`es
/// would fuse into one call, and even two fragments of the SAME call must keep their
/// `name`-carrying first delta distinct from later argument-only ones.
fn coalesce(deltas: Vec<UnifiedParserEvent>) -> Vec<UnifiedParserEvent> {
    let mut out: Vec<UnifiedParserEvent> = Vec::with_capacity(deltas.len());
    for delta in deltas {
        match (out.last_mut(), delta) {
            (Some(UnifiedParserEvent::Text(prev)), UnifiedParserEvent::Text(text)) => {
                prev.push_str(&text)
            }
            (Some(UnifiedParserEvent::Reasoning(prev)), UnifiedParserEvent::Reasoning(text)) => {
                prev.push_str(&text)
            }
            (_, delta) => out.push(delta),
        }
    }
    out
}

/// Pick the one fanout child that keeps the source chunk's token metrics.
///
/// The reasoning-usage estimator runs after unified parsing and attributes the source
/// chunk's tokens to reasoning when any emitted child carries `reasoning_content`.
/// Keeping metrics on the last child unconditionally loses that classification whenever
/// reasoning is followed by visible text, a tool call, or a later response choice.
pub(crate) fn fanout_llm_metrics_position(choices: &[ChatChoiceStream]) -> Option<usize> {
    choices
        .iter()
        .position(|choice| choice.delta.reasoning_content.is_some())
        .or_else(|| choices.len().checked_sub(1))
}

/// An empty streaming choice for `index`, used as the base every emitted chunk is
/// filled in from.
///
/// `logprobs` is dropped on purpose: once parsing rewrites a choice, the emitted text
/// no longer lines up token-for-token with the backend's raw stream, so per-token
/// logprobs would be attached to the wrong characters.
pub(crate) fn empty_choice(index: u32) -> ChatChoiceStream {
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
pub(crate) struct ChoiceState {
    family: String,
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
    pub(crate) fn new(
        family: &str,
        tools: &[Tool],
        prefill: UnifiedParserStartingState,
        tool_output_mode: UnifiedToolOutputMode,
    ) -> anyhow::Result<Self> {
        let mut parser = create_unified_parser_for_family(family, tools)?;
        // `prompt_token_ids` stays empty: this path establishes the starting state from
        // the rendered prompt text (see `stream_prefill` / `detect_prefill`), not from
        // token IDs, which the preprocessor has already consumed by this point.
        //
        // Streaming guided calls commit per call once the name and argument object opener
        // are unambiguous. A committed fragment cannot later be rejected or recovered as
        // text, so this path opts into that contract explicitly.
        parser.initialize_request(UnifiedParserInit {
            prompt_token_ids: Vec::new(),
            starting_state: prefill,
            tool_output_mode,
            invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
        })?;
        Ok(Self {
            family: family.to_string(),
            parser,
            opened_calls: HashSet::new(),
            tool_emitted: false,
            failed: false,
        })
    }

    pub(crate) fn new_default(family: &str, tools: &[Tool]) -> anyhow::Result<Self> {
        Ok(Self {
            family: family.to_string(),
            parser: create_unified_parser_for_family(family, tools)?,
            opened_calls: HashSet::new(),
            tool_emitted: false,
            failed: false,
        })
    }

    /// Feed one decoded text delta through the parser.
    pub(crate) fn push(&mut self, text: &str) -> Vec<UnifiedParserEvent> {
        if self.failed {
            return text_delta(text.to_string());
        }
        let mut output = UnifiedParserOutput::default();
        match self.parser.parse_into(text, &mut output) {
            Ok(()) => output.events,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    family = self.family,
                    "unified parser push failed; falling back to plain text for this choice"
                );
                output.events.extend(self.give_up(text));
                output.events
            }
        }
    }

    /// Flush buffered partial state at end of stream.
    pub(crate) fn finish(&mut self) -> Vec<UnifiedParserEvent> {
        if self.failed {
            return Vec::new();
        }
        match self.parser.finish() {
            // `finish` now hands back the whole `UnifiedParserOutput`; this path only ever
            // wants the ordered events out of it.
            Ok(output) => output.events,
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
    fn give_up(&mut self, fallback: &str) -> Vec<UnifiedParserEvent> {
        self.failed = true;
        let recovered = self.parser.reset();
        if recovered.is_empty() {
            text_delta(fallback.to_string())
        } else {
            text_delta(recovered)
        }
    }

    /// Convert one ordered delta into a streaming choice for `index`.
    fn delta_to_choice(&mut self, index: u32, delta: UnifiedParserEvent) -> ChatChoiceStream {
        let mut choice = empty_choice(index);
        match delta {
            UnifiedParserEvent::Text(text) => {
                choice.delta.content = Some(ChatCompletionMessageContent::Text(text));
            }
            UnifiedParserEvent::Reasoning(text) => {
                choice.delta.reasoning_content = Some(text);
            }
            UnifiedParserEvent::ToolCall(call) => {
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
    pub(crate) fn choices_for(
        &mut self,
        original: &ChatChoiceStream,
        deltas: Vec<UnifiedParserEvent>,
        emit_tool_calls: bool,
        finish_reason: Option<FinishReason>,
    ) -> Vec<ChatChoiceStream> {
        let deltas = coalesce(
            deltas
                .into_iter()
                .filter(|delta| {
                    emit_tool_calls || !matches!(delta, UnifiedParserEvent::ToolCall(_))
                })
                .collect(),
        );
        let index = original.index;
        let count = deltas.len();
        let mut choices = Vec::with_capacity(count.max(1));

        for (position, delta) in deltas.into_iter().enumerate() {
            let mut choice = self.delta_to_choice(index, delta);
            if position == 0 {
                choice.delta.role = original.delta.role;
                choice.delta.refusal = original.delta.refusal.clone();
                choice.delta.function_call = original.delta.function_call.clone();
                if let Some(reasoning) = original.delta.reasoning_content.as_deref() {
                    choice
                        .delta
                        .reasoning_content
                        .get_or_insert_default()
                        .insert_str(0, reasoning);
                }
                if let Some(mut original_calls) = original.delta.tool_calls.clone() {
                    if !original_calls.is_empty() {
                        self.tool_emitted = true;
                    }
                    if let Some(parsed_calls) = choice.delta.tool_calls.take() {
                        original_calls.extend(parsed_calls);
                    }
                    choice.delta.tool_calls = Some(original_calls);
                }
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
                || original.delta.reasoning_content.is_some()
                || original.delta.tool_calls.is_some()
                || original.delta.function_call.is_some()
                || finish_reason.is_some())
        {
            let mut choice = empty_choice(index);
            choice.delta.role = original.delta.role;
            choice.delta.refusal = original.delta.refusal.clone();
            choice.delta.reasoning_content = original.delta.reasoning_content.clone();
            choice.delta.tool_calls = original.delta.tool_calls.clone();
            choice.delta.function_call = original.delta.function_call.clone();
            if choice
                .delta
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
            {
                self.tool_emitted = true;
            }
            choice.finish_reason = self.normalize_finish_reason(finish_reason);
            choices.push(choice);
        }

        choices
    }

    pub(crate) fn unterminated_finish_reason(&self) -> Option<FinishReason> {
        self.tool_emitted.then_some(FinishReason::ToolCalls)
    }

    /// Whether this choice has emitted a tool call so far.
    pub(crate) fn tool_emitted(&self) -> bool {
        self.tool_emitted
    }

    /// Seed tool-call history from before an already-parsed detour into a freshly
    /// constructed parser instance for the same choice index.
    pub(crate) fn mark_tool_emitted(&mut self) {
        self.tool_emitted = true;
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
fn text_delta(text: String) -> Vec<UnifiedParserEvent> {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![UnifiedParserEvent::Text(text)]
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
pub(crate) fn parse_complete(
    family: &str,
    content: &str,
    guided_tool_constraint: &GuidedToolConstraint,
) -> anyhow::Result<CompleteOutput> {
    let mut parser = create_unified_parser_for_family(family, &[])?;
    // Batch replays already-generated output. Prefer its observable native marker when
    // topology B could not carry the worker's installed structural tag to the frontend;
    // otherwise use the reconstructed guided-JSON constraint.
    parser.initialize_request(UnifiedParserInit {
        starting_state: detect_prefill(family, content)?,
        tool_output_mode: batch_tool_output_mode(content, guided_tool_constraint),
        ..UnifiedParserInit::default()
    })?;

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

/// Per-choice bookkeeping that outlives any single `ChoiceState` instance for that
/// index: whether it has ever emitted a tool call, and whether its terminal chunk has
/// already been sent. See the field comment at its use site in
/// `apply_stream_with_constraint` for why this can't just live inside `ChoiceState`.
#[derive(Default)]
struct ChoiceRecord {
    tool_emitted: bool,
    finished: bool,
}

/// Finish every choice that never received a terminating chunk, in index order.
///
/// A choice can reach this point two ways: still holding a live `ChoiceState` (the
/// common case — nothing special happened, it just never got an explicit terminal),
/// or "history-only" — its `ChoiceState` was removed by an already-parsed detour and
/// never rebuilt, so all that remains is its `ChoiceRecord`. The second case has
/// nothing left to flush, but a tool-emitting choice still needs a synthesized
/// `ToolCalls` terminal so a strict client doesn't hang waiting for one; a choice that
/// never called a tool gets no synthetic chunk at all, matching the no-signal,
/// text-only contract a live `ChoiceState` already has via `tool_emitted` above.
fn finish_unterminated_choices(
    states: &mut HashMap<u32, ChoiceState>,
    records: &mut HashMap<u32, ChoiceRecord>,
) -> Vec<ChatChoiceStream> {
    let mut indices: Vec<u32> = states
        .keys()
        .copied()
        .chain(records.keys().copied())
        .collect();
    indices.sort_unstable();
    indices.dedup();

    let mut choices = Vec::new();
    for index in indices {
        if records.get(&index).is_some_and(|record| record.finished) {
            continue;
        }
        records.entry(index).or_default().finished = true;
        let base = empty_choice(index);
        match states.get_mut(&index) {
            Some(state) => {
                let deltas = state.finish();
                // A choice that emitted tool calls must terminate with `ToolCalls`
                // even when the backend never sent a finish_reason: a strict client
                // waits for a non-null one before considering the call complete, and
                // would otherwise hang.
                let finish_reason = state.tool_emitted().then_some(FinishReason::ToolCalls);
                choices.extend(state.choices_for(&base, deltas, true, finish_reason));
            }
            None => {
                if records
                    .get(&index)
                    .is_some_and(|record| record.tool_emitted)
                {
                    let mut choice = base;
                    choice.finish_reason = Some(FinishReason::ToolCalls);
                    choices.push(choice);
                }
            }
        }
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
    let has_raw_text = matches!(
        choice.delta.content,
        Some(ChatCompletionMessageContent::Text(_))
    );
    matches!(
        choice.delta.content,
        Some(ChatCompletionMessageContent::Parts(_))
    ) || (!has_raw_text
        && (choice.delta.tool_calls.is_some()
            || choice.delta.function_call.is_some()
            || choice.delta.reasoning_content.is_some()))
}

/// Streaming path: one unified parser per response choice, replacing both the reasoning
/// parser and the tool-call jail for this request.
///
/// `uses_tool_call_structural_tag` reports that the backend was given a structural tag
/// constraining generation to the family's NATIVE grammar. When it is set the output is
/// native markup no matter what `tool_choice` says, so `tool_choice` is not consulted;
/// this path only parses guided JSON, it never builds the grammar that produces it.
#[cfg(test)]
pub(crate) fn apply_stream<S>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
    prefill: UnifiedParserStartingState,
    family: &'static str,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let guided_tool_constraint = if uses_tool_call_structural_tag {
        GuidedToolConstraint::StructuralTag
    } else {
        match tool_choice {
            Some(ChatCompletionToolChoiceOption::Named(named)) => {
                GuidedToolConstraint::GuidedJsonNamed {
                    tool_name: named.function.name,
                }
            }
            Some(ChatCompletionToolChoiceOption::Required) => {
                GuidedToolConstraint::GuidedJsonRequired
            }
            None
            | Some(ChatCompletionToolChoiceOption::Auto)
            | Some(ChatCompletionToolChoiceOption::None) => GuidedToolConstraint::None,
        }
    };
    apply_stream_with_constraint(
        stream_in,
        tool_definitions,
        guided_tool_constraint,
        prefill,
        family,
    )
}

pub(crate) fn apply_stream_with_constraint<S>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    guided_tool_constraint: GuidedToolConstraint,
    prefill: UnifiedParserStartingState,
    family: &'static str,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let tools = to_v2_tools(tool_definitions.as_deref());
    stream! {
        let mut states: HashMap<u32, ChoiceState> = HashMap::new();
        // Per-choice bookkeeping that must outlive a single `ChoiceState`: a
        // `ChoiceState` is dropped whenever an already-parsed chunk interrupts a
        // choice (its buffered parser state would be wrong to keep across the
        // detour), but whether that choice ever called a tool and whether its
        // terminal has already gone out must still be known afterward — by a
        // resumed raw run, an already-parsed terminal, or the EOF backstop below,
        // even when no `ChoiceState` for that index exists any more (or never did:
        // an index whose first-ever chunk is already-parsed never gets one either).
        let mut records: HashMap<u32, ChoiceRecord> = HashMap::new();
        // Last data response with its choices cleared, so an end-of-stream flush has an
        // envelope (id, model, created) to attach synthesized chunks to.
        let mut template: Option<NvCreateChatCompletionStreamResponse> = None;

        tokio::pin!(stream_in);

        while let Some(mut response) = stream_in.next().await {
            if response.is_error() {
                yield response;
                return;
            }
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
                    for choice in finish_unterminated_choices(&mut states, &mut records) {
                        yield response_with_choice(template, choice);
                    }
                }
                yield response;
                continue;
            }

            let originals = std::mem::take(&mut chat.inner.choices);
            let mut emitted: Vec<ChatChoiceStream> = Vec::new();
            for mut original in originals {
                if already_parsed(&original) {
                    // Record this index exists even if no `ChoiceState` is ever
                    // built for it (e.g. its first-ever chunk is already-parsed),
                    // so it is not invisible to `finish_unterminated_choices`.
                    let record = records.entry(original.index).or_default();
                    // An already-parsed chunk carries its tool calls verbatim in its
                    // own delta rather than through a `ChoiceState`, so that history
                    // has to be observed here directly — a state may never exist for
                    // this index at all (its very first chunk can be already-parsed).
                    if original
                        .delta
                        .tool_calls
                        .as_ref()
                        .is_some_and(|calls| !calls.is_empty())
                    {
                        record.tool_emitted = true;
                    }
                    if let Some(mut state) = states.remove(&original.index) {
                        // A prior raw chunk on this choice may have emitted a tool
                        // call before this already-parsed terminal replaced it; that
                        // history lives only in the state being discarded here, so
                        // fold it into the record before it's gone (also needed by a
                        // later chunk or the EOF backstop after this `ChoiceState` is
                        // gone). Normalize against `record.tool_emitted`, not the
                        // state's own flag: this chunk's own `delta.tool_calls` (set
                        // into `record` above) can carry a tool call the state itself
                        // never saw, e.g. an already-parsed terminal chunk that is
                        // itself the first and only tool-call signal for this choice.
                        if state.tool_emitted() {
                            record.tool_emitted = true;
                        }
                        original.finish_reason =
                            if original.finish_reason == Some(FinishReason::Stop) && record.tool_emitted {
                                Some(FinishReason::ToolCalls)
                            } else {
                                original.finish_reason
                            };
                        let deltas = state.finish();
                        emitted.extend(state.choices_for(
                            &empty_choice(original.index),
                            deltas,
                            true,
                            None,
                        ));
                    } else if record.tool_emitted
                        && original.finish_reason == Some(FinishReason::Stop)
                    {
                        // No live state for this chunk (an earlier already-parsed
                        // chunk already discarded it), but this choice emitted a
                        // tool call before that gap.
                        original.finish_reason = Some(FinishReason::ToolCalls);
                    }
                    if original.finish_reason.is_some() {
                        record.finished = true;
                    }
                    emitted.push(original);
                    continue;
                }

                let state = match states.entry(original.index) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let mode = tool_output_mode(&guided_tool_constraint);
                        match ChoiceState::new(family, &tools, prefill, mode) {
                            Ok(mut state) => {
                                // A raw run resuming after an already-parsed detour
                                // gets a brand-new parser instance; seed it with any
                                // tool-call history from before the gap so a Stop at
                                // the end of this run still normalizes correctly.
                                if records.get(&original.index).is_some_and(|record| record.tool_emitted) {
                                    state.mark_tool_emitted();
                                }
                                entry.insert(state)
                            }
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
                let already_finished = records
                    .get(&original.index)
                    .is_some_and(|record| record.finished);
                if terminal.is_some() {
                    if !already_finished {
                        deltas.extend(state.finish());
                    }
                    records.entry(original.index).or_default().finished = true;
                }

                let mut parsed = state.choices_for(&original, deltas, true, terminal);
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

            // One upstream chunk can fan out into several. Usage, nvext and annotation
            // fields stay on the last child. Token metrics stay on one reasoning child
            // when present so the downstream reasoning-usage estimator preserves the
            // source chunk's classification without counting it twice.
            let last = emitted.len() - 1;
            let Some(llm_metrics_position) = fanout_llm_metrics_position(&emitted) else {
                continue;
            };
            for (position, choice) in emitted.into_iter().enumerate() {
                let is_last = position == last;
                let mut data = chat.clone();
                data.inner.choices = vec![choice];
                if !is_last {
                    data.inner.usage = None;
                    data.nvext = None;
                }
                if position != llm_metrics_position {
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
            for choice in finish_unterminated_choices(&mut states, &mut records) {
                yield response_with_choice(template, choice);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers_v2::ToolCallDelta;
    use dynamo_protocols::types::{
        ChatCompletionStreamResponseDeltaFunctionCall, CreateChatCompletionStreamResponse, Role,
    };
    use futures::stream;

    struct PartialCommitParser {
        recovered: String,
    }

    impl UnifiedParser for PartialCommitParser {
        fn parse_into(
            &mut self,
            _delta: &str,
            output: &mut UnifiedParserOutput,
        ) -> anyhow::Result<()> {
            output
                .events
                .push(UnifiedParserEvent::Text("committed".to_string()));
            anyhow::bail!("intentional failure after a committed event")
        }

        fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
            Ok(UnifiedParserOutput::default())
        }

        fn reset(&mut self) -> String {
            std::mem::take(&mut self.recovered)
        }
    }

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

    #[derive(Debug, PartialEq, Eq)]
    enum LogicalEvent {
        Reasoning(String),
        Text(String),
        Tool {
            index: u32,
            name: Option<String>,
            arguments: String,
        },
    }

    fn logical_events(
        responses: &[Annotated<NvCreateChatCompletionStreamResponse>],
    ) -> Vec<LogicalEvent> {
        let mut events = Vec::new();
        for choice in collect_choices(responses) {
            if let Some(reasoning) = &choice.delta.reasoning_content {
                match events.last_mut() {
                    Some(LogicalEvent::Reasoning(existing)) => existing.push_str(reasoning),
                    _ => events.push(LogicalEvent::Reasoning(reasoning.clone())),
                }
            }
            if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                match events.last_mut() {
                    Some(LogicalEvent::Text(existing)) => existing.push_str(text),
                    _ => events.push(LogicalEvent::Text(text.clone())),
                }
            }
            for call in choice.delta.tool_calls.iter().flatten() {
                let name = call.function.as_ref().and_then(|f| f.name.clone());
                let arguments = call
                    .function
                    .as_ref()
                    .and_then(|f| f.arguments.clone())
                    .unwrap_or_default();
                match events.last_mut() {
                    Some(LogicalEvent::Tool {
                        index,
                        name: existing_name,
                        arguments: existing_arguments,
                    }) if *index == call.index => {
                        if existing_name.is_none() {
                            *existing_name = name;
                        }
                        existing_arguments.push_str(&arguments);
                    }
                    _ => events.push(LogicalEvent::Tool {
                        index: call.index,
                        name,
                        arguments,
                    }),
                }
            }
        }
        events
    }

    async fn parse_at_split(
        input: &str,
        split: usize,
        tool_choice: Option<ChatCompletionToolChoiceOption>,
    ) -> Vec<LogicalEvent> {
        let (first, second) = input.split_at(split);
        let responses = apply_stream(
            stream::iter([chunk(first, false), chunk(second, true)]),
            Some(weather_tools()),
            tool_choice,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        logical_events(&responses)
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

    #[test]
    fn batch_routing_uses_the_configured_pair_for_every_output_mode() {
        let pair = (Some("qwen3_coder"), Some("qwen3"));
        assert_eq!(
            configured_batch_family(pair.0, pair.1),
            Some(QWEN3_UNIFIED_FAMILY),
            "the carried constraint selects native versus guided parsing after routing"
        );
    }

    // --- tool_choice -> UnifiedToolOutputMode ------------------------------------

    #[test]
    fn maps_tool_choice_onto_the_output_mode() {
        assert_eq!(
            tool_output_mode(&GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            }),
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".to_string())
            }
        );
        assert_eq!(
            tool_output_mode(&GuidedToolConstraint::GuidedJsonRequired),
            UnifiedToolOutputMode::GuidedJson { named_tool: None }
        );
        for native in [
            GuidedToolConstraint::None,
            GuidedToolConstraint::StructuralTag,
        ] {
            assert_eq!(
                tool_output_mode(&native),
                UnifiedToolOutputMode::Native,
                "{native:?} must stay on native markup"
            );
        }
    }

    #[test]
    fn batch_native_marker_overrides_a_reconstructed_guided_constraint() {
        for constraint in [
            GuidedToolConstraint::GuidedJsonRequired,
            GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            },
        ] {
            assert_eq!(
                batch_tool_output_mode(
                    "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
                    &constraint,
                ),
                UnifiedToolOutputMode::Native,
            );
        }
        assert_eq!(
            batch_tool_output_mode(
                r#"[{"name":"get_weather","parameters":{"literal":"<tool_call>"}}]"#,
                &GuidedToolConstraint::GuidedJsonRequired,
            ),
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            "a marker inside JSON arguments remains payload data",
        );
    }

    #[tokio::test]
    async fn every_valid_utf8_split_matches_whole_input_across_output_modes() {
        let cases = [
            (
                concat!(
                    "<think>理由は東京です。</think>",
                    "<tool_call>\n<function=get_weather>\n",
                    "<parameter=city>東京</parameter>\n</function>\n</tool_call>"
                ),
                Some(ChatCompletionToolChoiceOption::Auto),
            ),
            (
                r#"{"city":"東</think>京"}"#,
                Some(named_choice("get_weather")),
            ),
            (
                r#"[{"name":"get_weather","parameters":{"city":"東京"}}]"#,
                Some(ChatCompletionToolChoiceOption::Required),
            ),
        ];

        for (input, tool_choice) in cases {
            let expected = parse_at_split(input, 0, tool_choice.clone()).await;
            for split in (0..=input.len()).filter(|split| input.is_char_boundary(*split)) {
                assert_eq!(
                    parse_at_split(input, split, tool_choice.clone()).await,
                    expected,
                    "output changed at UTF-8 split {split} for {input:?}"
                );
            }
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
            UnifiedParserStartingState::None,
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
            UnifiedParserStartingState::Reasoning
        );
        // Qwen3's template pre-opens no channel, so the model emits `<think>` itself.
        assert_eq!(
            stream_prefill(QWEN3_UNIFIED_FAMILY, false),
            UnifiedParserStartingState::None
        );
    }

    #[test]
    fn detects_batch_prefill_from_complete_output() {
        assert_eq!(
            detect_prefill(QWEN3_UNIFIED_FAMILY, "<think>reason</think>answer").unwrap(),
            UnifiedParserStartingState::None
        );
        assert_eq!(
            detect_prefill(QWEN3_UNIFIED_FAMILY, "reason</think>answer").unwrap(),
            UnifiedParserStartingState::Reasoning
        );
        assert_eq!(
            detect_prefill(QWEN3_UNIFIED_FAMILY, "answer").unwrap(),
            UnifiedParserStartingState::Response
        );
        assert_eq!(
            detect_prefill(QWEN3_UNIFIED_FAMILY, "It's hidden</think>visible").unwrap(),
            UnifiedParserStartingState::Reasoning,
            "an apostrophe in prose must not hide a later control marker"
        );
        assert!(detect_prefill("kimi_k3", "answer").is_err());
    }

    // --- delta -> chunk conversion -----------------------------------------------

    fn call_delta(tool_index: usize, name: Option<&str>, arguments: &str) -> UnifiedParserEvent {
        UnifiedParserEvent::ToolCall(ToolCallDelta {
            tool_index,
            name: name.map(str::to_string),
            arguments: arguments.to_string(),
        })
    }

    fn test_state() -> ChoiceState {
        ChoiceState::new(
            QWEN3_UNIFIED_FAMILY,
            &[],
            UnifiedParserStartingState::None,
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
                UnifiedParserEvent::Reasoning("look it up".into()),
                call_delta(0, Some("get_weather"), r#"{"city":"Tokyo"}"#),
                UnifiedParserEvent::Reasoning("now answer".into()),
                UnifiedParserEvent::Text("It's 18C.".into()),
            ],
            true,
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
            true,
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
            true,
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
            UnifiedParserEvent::Text("he".into()),
            UnifiedParserEvent::Text("llo".into()),
            call_delta(0, Some("f"), "{"),
            call_delta(0, None, "}"),
            UnifiedParserEvent::Reasoning("a".into()),
            UnifiedParserEvent::Reasoning("b".into()),
        ]);
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[0], UnifiedParserEvent::Text("hello".into()));
        assert_eq!(merged[1], call_delta(0, Some("f"), "{"));
        assert_eq!(merged[2], call_delta(0, None, "}"));
        assert_eq!(merged[3], UnifiedParserEvent::Reasoning("ab".into()));
    }

    #[test]
    fn role_rides_the_first_chunk_and_finish_reason_the_last() {
        let mut state = test_state();
        let mut base = empty_choice(0);
        base.delta.role = Some(Role::Assistant);
        let choices = state.choices_for(
            &base,
            vec![
                UnifiedParserEvent::Text("a".into()),
                call_delta(0, Some("f"), "{}"),
            ],
            true,
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
        let choices = state.choices_for(&base, Vec::new(), true, Some(FinishReason::Length));

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
        assert!(state.choices_for(&base, Vec::new(), true, None).is_empty());
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
            UnifiedParserStartingState::None,
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
    async fn qwen_fanout_keeps_source_metrics_on_reasoning() {
        let mut source = chunk("<think>reason</think>answer", false);
        source.data.as_mut().unwrap().llm_metrics =
            Some(crate::protocols::common::metrics::LLMMetricAnnotation {
                chunk_tokens: 4,
                ..Default::default()
            });

        let responses = apply_stream(
            stream::iter([source]),
            None,
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let metric_choices: Vec<_> = responses
            .iter()
            .filter_map(|response| response.data.as_ref())
            .filter(|data| data.llm_metrics.is_some())
            .flat_map(|data| data.inner.choices.iter())
            .collect();

        assert_eq!(
            metric_choices.len(),
            1,
            "source metrics must be emitted once"
        );
        assert_eq!(
            metric_choices[0].delta.reasoning_content.as_deref(),
            Some("reason"),
            "the metrics owner must retain the source chunk's reasoning classification"
        );
    }

    #[tokio::test]
    async fn named_guided_json_becomes_a_tool_call() {
        let responses = apply_stream(
            stream::iter([
                chunk("reason</think>{\"city\": ", false),
                chunk("\"Tokyo\"}", true),
            ]),
            Some(weather_tools()),
            Some(named_choice("get_weather")),
            false,
            UnifiedParserStartingState::Reasoning,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        assert_eq!(
            choices[0].delta.reasoning_content.as_deref(),
            Some("reason")
        );
        // A named choice now STREAMS: the name rides the first delta and the argument
        // bytes follow, so collect across deltas exactly as the required case does.
        let calls: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .collect();
        assert_eq!(
            calls
                .iter()
                .find_map(|call| call.function.as_ref()?.name.as_deref()),
            Some("get_weather")
        );
        let arguments: String = calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        assert_eq!(
            arguments, "{\"city\": \"Tokyo\"}",
            "a named choice passes the model's argument bytes through verbatim"
        );
        // The argument bytes must arrive as they are generated, not in one terminal
        // burst: the caller split the payload across two chunks, so a client that
        // renders each delta sees the call grow instead of waiting for the whole
        // object. One frame here means the wiring buffered what the parser streamed.
        let frames = calls
            .iter()
            .filter(|call| {
                call.function
                    .as_ref()
                    .and_then(|f| f.arguments.as_deref())
                    .is_some_and(|a| !a.is_empty())
            })
            .count();
        assert!(
            frames >= 2,
            "arguments arrived in {frames} frame(s) - that is a burst, not a stream"
        );
        assert!(
            choices
                .iter()
                .any(|c| c.finish_reason == Some(FinishReason::ToolCalls)),
            "the stream must terminate with ToolCalls"
        );
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
            UnifiedParserStartingState::Reasoning,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        let calls: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .collect();
        assert_eq!(
            calls
                .iter()
                .find_map(|call| call.function.as_ref()?.name.as_deref()),
            Some("get_weather")
        );
        let arguments: String = calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        assert_eq!(arguments, r#"{"city":"Tokyo"}"#);
    }

    #[tokio::test]
    async fn required_guided_emits_name_before_polling_call_completion() {
        let prefix = r#"reason</think>[{"name":"get_weather","arguments":{"#;
        let upstream = stream::iter([chunk(prefix, false)]).chain(stream::poll_fn(|_| {
            panic!("apply_stream polled for call-completing bytes before emitting the tool name")
        }));
        let responses = apply_stream(
            upstream,
            Some(weather_tools()),
            Some(ChatCompletionToolChoiceOption::Required),
            false,
            UnifiedParserStartingState::Reasoning,
            QWEN3_UNIFIED_FAMILY,
        );
        tokio::pin!(responses);

        let reasoning = responses.next().await.expect("reasoning delta");
        assert_eq!(
            collect_choices(&[reasoning])[0]
                .delta
                .reasoning_content
                .as_deref(),
            Some("reason")
        );

        let tool = responses.next().await.expect("tool-name delta");
        let tool_responses = [tool];
        let tool_choices = collect_choices(&tool_responses);
        let call = &tool_choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("tool call before completion")[0];
        assert_eq!(
            call.function
                .as_ref()
                .and_then(|function| function.name.as_deref()),
            Some("get_weather")
        );
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
            UnifiedParserStartingState::None,
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
    async fn mixed_preparsed_reasoning_still_parses_raw_content() {
        let output = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let mut mixed = chunk(output, true);
        mixed.data.as_mut().unwrap().inner.choices[0]
            .delta
            .reasoning_content = Some("already parsed".to_string());

        let responses = apply_stream(
            stream::iter([mixed]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        let visible: String = choices
            .iter()
            .filter_map(|choice| match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.delta.reasoning_content.as_deref())
                .collect::<String>(),
            "already parsed"
        );
        assert!(
            choices
                .iter()
                .any(|choice| choice.delta.tool_calls.is_some()),
            "raw native tool markup must still become a structured call"
        );
        assert!(
            !visible.contains("<tool_call>"),
            "raw markup leaked: {visible:?}"
        );
    }

    #[tokio::test]
    async fn mixed_preparsed_tool_call_still_parses_raw_content() {
        let output = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let mut mixed = chunk(output, true);
        let choice = &mut mixed.data.as_mut().unwrap().inner.choices[0];
        choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 7,
            id: Some("existing-call".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("already_parsed".to_string()),
                arguments: Some("{}".to_string()),
            }),
        }]);

        let responses = apply_stream(
            stream::iter([mixed]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        let visible: String = choices
            .iter()
            .filter_map(|choice| match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let names: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .filter_map(|call| call.function.as_ref()?.name.as_deref())
            .collect();

        assert!(names.contains(&"already_parsed"));
        assert!(names.contains(&"get_weather"));
        assert!(
            !visible.contains("<tool_call>"),
            "raw markup leaked: {visible:?}"
        );
    }

    #[tokio::test]
    async fn preparsed_terminal_choice_does_not_drop_buffered_bytes() {
        let mut terminal = chunk("", true);
        let choice = &mut terminal.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("already".to_string());

        let responses = apply_stream(
            stream::iter([chunk("hello <tool_c", false), terminal]),
            None,
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        let visible: String = choices
            .iter()
            .filter_map(|choice| match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(visible, "hello <tool_c");
        let terminal_position = choices
            .iter()
            .position(|choice| choice.finish_reason.is_some())
            .expect("terminal choice");
        assert!(
            choices[terminal_position + 1..]
                .iter()
                .all(|choice| choice.delta.content.is_none()),
            "buffered content appeared after the terminal choice"
        );
    }

    #[tokio::test]
    async fn preparsed_terminal_normalizes_stop_after_prior_raw_tool_call() {
        let tool_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let mut terminal = chunk("", true);
        let choice = &mut terminal.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("already parsed".to_string());

        let responses = apply_stream(
            stream::iter([chunk(tool_call, false), terminal]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        assert!(
            choices
                .iter()
                .any(|choice| choice.delta.tool_calls.is_some()),
            "the raw first chunk must emit a tool call"
        );
        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .next_back(),
            Some(FinishReason::ToolCalls),
            "an already-parsed terminal Stop must retain the earlier tool-call state"
        );
    }

    #[tokio::test]
    async fn raw_preparsed_raw_sequence_preserves_order_and_buffered_bytes() {
        let mut pre_parsed = chunk("", false);
        let choice = &mut pre_parsed.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("middle".to_string());

        let responses = apply_stream(
            stream::iter([
                chunk("before <tool_c", false),
                pre_parsed,
                chunk(" after", true),
            ]),
            None,
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        let ordered: Vec<_> = choices
            .iter()
            .filter_map(|choice| {
                if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                    Some(("content", text.as_str()))
                } else {
                    choice
                        .delta
                        .reasoning_content
                        .as_deref()
                        .map(|reasoning| ("reasoning", reasoning))
                }
            })
            .collect();

        assert_eq!(
            ordered,
            vec![
                ("content", "before "),
                ("content", "<tool_c"),
                ("reasoning", "middle"),
                ("content", " after"),
            ]
        );
    }

    #[tokio::test]
    async fn raw_preparsed_raw_terminal_normalizes_stop_after_tool_call_before_gap() {
        let tool_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let mut pre_parsed = chunk("", false);
        let choice = &mut pre_parsed.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("already parsed".to_string());

        let responses = apply_stream(
            stream::iter([
                chunk(tool_call, false),
                pre_parsed,
                chunk(" trailing text", true),
            ]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        assert!(
            choices
                .iter()
                .any(|choice| choice.delta.tool_calls.is_some()),
            "the raw first chunk must emit a tool call"
        );
        let finish_reasons: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect();
        assert_eq!(
            finish_reasons,
            vec![FinishReason::ToolCalls],
            "a fresh parser instance for the resumed raw run must not lose the \
             tool-call history from before the already-parsed gap, and exactly \
             one terminal must be emitted"
        );
    }

    #[tokio::test]
    async fn two_consecutive_already_parsed_chunks_after_tool_call_normalize_once() {
        let tool_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let already_parsed_chunk = |reasoning: &str, finish: bool| {
            let mut c = chunk("", finish);
            let choice = &mut c.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some(reasoning.to_string());
            c
        };

        let responses = apply_stream(
            stream::iter([
                chunk(tool_call, false),
                already_parsed_chunk("first already-parsed", false),
                already_parsed_chunk("second already-parsed", false),
                already_parsed_chunk("third already-parsed (terminal)", true),
            ]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        assert!(
            choices
                .iter()
                .any(|choice| choice.delta.tool_calls.is_some()),
            "the raw first chunk must emit a tool call"
        );
        let finish_reasons: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect();
        assert_eq!(
            finish_reasons,
            vec![FinishReason::ToolCalls],
            "two already-parsed chunks in a row after the tool call (the second \
             with no live ChoiceState left to remove) must still normalize to \
             exactly one ToolCalls terminal, not Stop and not more than one terminal"
        );
    }

    #[tokio::test]
    async fn tool_history_is_isolated_per_choice_index() {
        fn two_choice_response(
            index0: (&str, bool),
            index1: (&str, bool),
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            let make = |index: u32, text: &str, finish: bool| ChatChoiceStream {
                index,
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
            };
            #[allow(deprecated)]
            let response = NvCreateChatCompletionStreamResponse {
                inner: CreateChatCompletionStreamResponse {
                    id: "test".to_string(),
                    choices: vec![make(0, index0.0, index0.1), make(1, index1.0, index1.1)],
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
        let already_parsed_pair = || {
            let mut response = two_choice_response(("", false), ("", false));
            for choice in &mut response.data.as_mut().unwrap().inner.choices {
                choice.delta.content = None;
                choice.delta.reasoning_content = Some("already parsed".to_string());
            }
            response
        };

        let tool_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let responses = apply_stream(
            stream::iter([
                two_choice_response((tool_call, false), ("hello", false)),
                already_parsed_pair(),
                two_choice_response((" trailing0", true), (" trailing1", true)),
            ]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        let finish_for = |index: u32| {
            choices
                .iter()
                .filter(|choice| choice.index == index)
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            finish_for(0),
            vec![FinishReason::ToolCalls],
            "choice 0 emitted a tool call before the gap and must normalize to ToolCalls"
        );
        assert_eq!(
            finish_for(1),
            vec![FinishReason::Stop],
            "choice 1 never emitted a tool call and must not be contaminated by \
             choice 0's tool history"
        );
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn legacy_function_call_passes_through_untouched() {
        let mut legacy = chunk("", true);
        let choice = &mut legacy.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.function_call = Some(ChatCompletionStreamResponseDeltaFunctionCall {
            name: Some("get_weather".to_string()),
            arguments: Some(r#"{"city":"Tokyo"}"#.to_string()),
        });

        let responses = apply_stream(
            stream::iter([legacy]),
            None,
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        assert_eq!(choices.len(), 1);
        assert_eq!(
            choices[0]
                .delta
                .function_call
                .as_ref()
                .and_then(|call| call.name.as_deref()),
            Some("get_weather")
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
            UnifiedParserStartingState::None,
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
    async fn already_parsed_gap_then_immediate_eof_still_terminates_a_tool_call() {
        let tool_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let mut already_parsed = chunk("", false);
        let choice = &mut already_parsed.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("already parsed".to_string());

        // Raw tool call, then an already-parsed chunk that removes the live
        // ChoiceState (no finish_reason on it), then the stream just ends — no
        // further chunks, no usage-only chunk, no explicit terminal at all.
        let responses = apply_stream(
            stream::iter([chunk(tool_call, false), already_parsed]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        assert!(
            choices
                .iter()
                .any(|choice| choice.delta.tool_calls.is_some()),
            "the raw first chunk must emit a tool call"
        );
        let finish_reasons: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect();
        assert_eq!(
            finish_reasons,
            vec![FinishReason::ToolCalls],
            "a choice whose live state was removed by an already-parsed gap must \
             still get exactly one terminal at end-of-stream, not zero — the EOF \
             backstop must not silently drop it"
        );
    }

    #[tokio::test]
    async fn already_parsed_gap_terminal_precedes_a_usage_only_chunk() {
        let tool_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let mut already_parsed = chunk("", false);
        let choice = &mut already_parsed.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("already parsed".to_string());
        let mut usage_only = chunk("unused", false);
        usage_only.data.as_mut().unwrap().inner.choices.clear();

        let responses = apply_stream(
            stream::iter([chunk(tool_call, false), already_parsed, usage_only]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let usage_position = responses
            .iter()
            .position(|response| {
                response
                    .data
                    .as_ref()
                    .is_some_and(|data| data.inner.choices.is_empty())
            })
            .expect("the usage-only chunk must still be forwarded");
        let terminal_position = responses
            .iter()
            .position(|response| {
                response.data.as_ref().is_some_and(|data| {
                    data.inner
                        .choices
                        .iter()
                        .any(|choice| choice.finish_reason == Some(FinishReason::ToolCalls))
                })
            })
            .expect("a history-only choice must still get exactly one ToolCalls terminal");
        assert!(
            terminal_position < usage_position,
            "the terminal must precede the usage-only chunk, per OpenAI stream ordering"
        );
    }

    #[tokio::test]
    async fn non_tool_choice_gets_no_synthetic_terminal_after_a_gap() {
        let mut already_parsed = chunk("", false);
        let choice = &mut already_parsed.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("already parsed".to_string());

        // No tool call anywhere: raw plain text, then a gap, then immediate EOF.
        let responses = apply_stream(
            stream::iter([chunk("plain text, no tool call", false), already_parsed]),
            None,
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        assert!(
            choices.iter().all(|choice| choice.finish_reason.is_none()),
            "a choice that never called a tool must not get a synthesized terminal \
             after a gap, matching the text-only, no-signal contract a live \
             ChoiceState already has"
        );
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn legacy_function_call_choice_gets_no_synthetic_terminal_after_a_gap() {
        // The legacy `function_call` field is a distinct signal from `tool_calls`
        // (it has its own `FinishReason::FunctionCall`, never `ToolCalls`) and, like
        // plain text, is never treated as `tool_emitted` anywhere in this file (only
        // actual parsed tool-call events and non-empty `delta.tool_calls` are). An
        // already-parsed chunk that only carries `function_call` must therefore get
        // the same "no synthetic terminal" treatment after a gap as plain text does,
        // matching the pre-existing live-`ChoiceState` precedent, not a new gap.
        let mut already_parsed = chunk("", false);
        let choice = &mut already_parsed.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.function_call = Some(ChatCompletionStreamResponseDeltaFunctionCall {
            name: Some("get_weather".to_string()),
            arguments: Some(r#"{"city": "Tokyo"}"#.to_string()),
        });

        let responses = apply_stream(
            stream::iter([chunk("plain text, no tool call", false), already_parsed]),
            None,
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        assert!(
            choices.iter().all(|choice| choice.finish_reason.is_none()),
            "a choice whose only already-parsed signal is the legacy function_call \
             field must not get a synthesized ToolCalls terminal after a gap"
        );
    }

    #[tokio::test]
    async fn already_parsed_terminal_carrying_its_own_tool_call_normalizes_stop() {
        // The live ChoiceState never emits a tool call itself (raw reasoning-only
        // content), but the already-parsed TERMINAL chunk that follows carries its
        // own `delta.tool_calls` together with `finish_reason: Stop` in the same
        // chunk. The state's own `tool_emitted()` is false, but the record must still
        // pick up this chunk's own tool call and normalize Stop -> ToolCalls.
        let mut terminal = chunk("", true);
        let choice = &mut terminal.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call-terminal".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some("{}".to_string()),
            }),
        }]);

        let responses = apply_stream(
            stream::iter([chunk("reasoning only, no tool call", false), terminal]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::ToolCalls],
            "an already-parsed terminal chunk that itself carries a tool call must \
             normalize Stop to ToolCalls even when the live state never saw one"
        );
    }

    #[tokio::test]
    async fn already_parsed_first_chunk_with_no_prior_raw_still_terminates_correctly() {
        let mut already_parsed = chunk("", false);
        let choice = &mut already_parsed.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call-preexisting".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some("{}".to_string()),
            }),
        }]);

        // This choice index never has a raw chunk at all — no ChoiceState is ever
        // constructed for it — then the stream ends immediately.
        let responses = apply_stream(
            stream::iter([already_parsed]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::ToolCalls],
            "a choice whose only-ever chunk was already-parsed and carried a tool \
             call must still get exactly one ToolCalls terminal at EOF"
        );
    }

    #[tokio::test]
    async fn multi_choice_interleave_finishes_each_unfinished_index_once() {
        fn choice(index: u32, text: &str, finish: bool) -> ChatChoiceStream {
            ChatChoiceStream {
                index,
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
            }
        }
        fn response(
            choices: Vec<ChatChoiceStream>,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            #[allow(deprecated)]
            let response = NvCreateChatCompletionStreamResponse {
                inner: CreateChatCompletionStreamResponse {
                    id: "test".to_string(),
                    choices,
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
        let tool_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        // Index 0: live the whole time, never removed, never explicitly finished —
        // the ordinary backstop case. Index 1: raw tool call, then a gap that removes
        // its state and is never touched again — history-only at EOF. Index 2: raw
        // tool call, explicitly finished right away, never touched again.
        let first = response(vec![
            choice(0, tool_call, false),
            choice(1, tool_call, false),
            choice(2, tool_call, false),
        ]);
        let mut second = response(vec![choice(0, " more", false), choice(2, " done", true)]);
        {
            let mut gap = choice(1, "", false);
            gap.delta.content = None;
            gap.delta.reasoning_content = Some("already parsed".to_string());
            second.data.as_mut().unwrap().inner.choices.push(gap);
        }
        // Index 1 gets no further chunks after this — it must stay history-only
        // through to EOF for this to actually exercise the bug.

        let responses = apply_stream(
            stream::iter([first, second]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        let finish_for = |index: u32| {
            choices
                .iter()
                .filter(|choice| choice.index == index)
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            finish_for(0),
            vec![FinishReason::ToolCalls],
            "the ordinary live choice must get exactly one terminal via the backstop"
        );
        assert_eq!(
            finish_for(1),
            vec![FinishReason::ToolCalls],
            "the history-only choice must get exactly one terminal, not zero"
        );
        assert_eq!(
            finish_for(2),
            vec![FinishReason::ToolCalls],
            "the already-explicitly-finished choice must not be finished a second \
             time by the EOF backstop"
        );
    }

    #[tokio::test]
    async fn three_consecutive_already_parsed_gaps_then_eof_finishes_once() {
        let tool_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let already_parsed_chunk = |reasoning: &str| {
            let mut c = chunk("", false);
            let choice = &mut c.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some(reasoning.to_string());
            c
        };

        // Three already-parsed chunks in a row, no raw chunk between them, no
        // explicit terminal ever, then the stream just ends.
        let responses = apply_stream(
            stream::iter([
                chunk(tool_call, false),
                already_parsed_chunk("first"),
                already_parsed_chunk("second"),
                already_parsed_chunk("third"),
            ]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::ToolCalls],
            "N consecutive already-parsed gaps with no explicit terminal must still \
             produce exactly one ToolCalls terminal at EOF, not zero and not N"
        );
    }

    #[tokio::test]
    async fn raw_gap_resume_second_gap_then_eof_finishes_once_in_order() {
        // Distinct from `three_consecutive_already_parsed_gaps_then_eof_finishes_once`
        // (which never resumes with a raw chunk in between): this drives raw -> gap ->
        // raw-resumed (a fresh `ChoiceState`, `tool_emitted` reseeded from the
        // `ChoiceRecord`) -> a SECOND gap on that resumed instance -> EOF, so the
        // fold-into-record step at removal runs twice on two different `ChoiceState`
        // instances for the same choice index.
        let tool_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let already_parsed_chunk = |reasoning: &str| {
            let mut c = chunk("", false);
            let choice = &mut c.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some(reasoning.to_string());
            c
        };

        let responses = apply_stream(
            stream::iter([
                chunk(tool_call, false),
                already_parsed_chunk("first gap"),
                chunk(" resumed text", false),
                already_parsed_chunk("second gap"),
            ]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        let ordered: Vec<_> = choices
            .iter()
            .filter_map(|choice| {
                if choice.delta.tool_calls.is_some() {
                    Some("tool_call")
                } else if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content
                {
                    if text.is_empty() {
                        None
                    } else {
                        Some("content")
                    }
                } else {
                    choice
                        .delta
                        .reasoning_content
                        .as_deref()
                        .map(|_| "reasoning")
                }
            })
            .collect();
        assert_eq!(
            ordered,
            vec!["tool_call", "reasoning", "content", "reasoning"],
            "delivery order across the raw/gap/resume/gap sequence must be preserved: \
             tool call, first gap's reasoning, resumed content, second gap's reasoning"
        );

        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::ToolCalls],
            "a second gap on a resumed (fresh) ChoiceState must still fold into the \
             same choice's record and produce exactly one ToolCalls terminal at EOF, \
             not zero and not two"
        );
    }

    #[tokio::test]
    async fn text_only_stream_gets_no_synthetic_finish_reason() {
        let responses = apply_stream(
            stream::iter([chunk("hello world", false)]),
            None,
            None,
            false,
            UnifiedParserStartingState::None,
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
            vec![UnifiedParserEvent::Text("hello ".into())]
        );
        // Now force the failure path with those held-back bytes still buffered.
        let recovered = state.give_up("");
        assert_eq!(
            recovered,
            vec![UnifiedParserEvent::Text("<tool_c".into())],
            "buffered bytes must be surfaced, not silently dropped"
        );
        assert!(state.failed);
        // Every later chunk now passes through as plain text.
        assert_eq!(
            state.push("more"),
            vec![UnifiedParserEvent::Text("more".into())]
        );
        assert!(state.finish().is_empty());
    }

    #[test]
    fn a_failed_push_keeps_committed_events_before_recovered_bytes() {
        let mut state = ChoiceState {
            family: "test".to_string(),
            parser: Box::new(PartialCommitParser {
                recovered: "<broken>".to_string(),
            }),
            opened_calls: HashSet::new(),
            tool_emitted: false,
            failed: false,
        };

        assert_eq!(
            state.push("ignored"),
            vec![
                UnifiedParserEvent::Text("committed".to_string()),
                UnifiedParserEvent::Text("<broken>".to_string()),
            ]
        );
    }

    #[tokio::test]
    async fn terminal_error_suppresses_eof_recovery() {
        let responses = apply_stream(
            stream::iter([
                chunk("hello <tool_c", false),
                Annotated::from_error("backend exploded"),
            ]),
            None,
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let error_position = responses
            .iter()
            .position(Annotated::is_error)
            .expect("terminal error");

        assert!(
            responses[error_position + 1..]
                .iter()
                .all(|response| response.data.is_none()),
            "no parser data may be emitted after a terminal error"
        );
    }

    #[test]
    fn give_up_falls_back_to_the_chunk_when_nothing_was_buffered() {
        let mut state = test_state();
        assert_eq!(
            state.give_up("the chunk that broke it"),
            vec![UnifiedParserEvent::Text("the chunk that broke it".into())]
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
            &GuidedToolConstraint::None,
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
        let parsed = parse_complete(
            QWEN3_UNIFIED_FAMILY,
            "hidden</think>visible",
            &GuidedToolConstraint::None,
        )
        .unwrap();
        assert_eq!(parsed.reasoning, "hidden");
        assert_eq!(parsed.text, "visible");
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn unmatched_quote_before_real_reasoning_close_does_not_expose_reasoning() {
        let parsed = parse_complete(
            QWEN3_UNIFIED_FAMILY,
            r#"hidden "unfinished</think>visible"#,
            &GuidedToolConstraint::None,
        )
        .unwrap();

        assert_eq!(parsed.reasoning, r#"hidden "unfinished"#);
        assert_eq!(parsed.text, "visible");
    }

    #[test]
    fn unmatched_single_quote_before_real_reasoning_close_does_not_expose_reasoning() {
        // Leading with a space keeps this a genuine quote-open candidate rather
        // than the in-word apostrophe exception (e.g. "it's").
        let parsed = parse_complete(
            QWEN3_UNIFIED_FAMILY,
            "hidden 'unfinished</think>visible",
            &GuidedToolConstraint::None,
        )
        .unwrap();

        assert_eq!(parsed.reasoning, "hidden 'unfinished");
        assert_eq!(parsed.text, "visible");
    }

    #[test]
    fn unmatched_backtick_before_real_reasoning_close_does_not_expose_reasoning() {
        let parsed = parse_complete(
            QWEN3_UNIFIED_FAMILY,
            "hidden `unfinished</think>visible",
            &GuidedToolConstraint::None,
        )
        .unwrap();

        assert_eq!(parsed.reasoning, "hidden `unfinished");
        assert_eq!(parsed.text, "visible");
    }

    #[test]
    fn contains_unquoted_marker_finds_real_marker_past_many_unmatched_escaped_quotes() {
        // Every `"` here is immediately preceded by an escaping backslash, so none of
        // them closes any other: each is individually a "does an unescaped close exist
        // later" candidate. `later_unescaped_quote_closes` answers all of these in one
        // O(n) pass; a naive per-candidate rescan (the pre-fix `has_unescaped_close`
        // called fresh per candidate) would instead be O(n^2) on input shaped like
        // this. This case is a deterministic correctness check, not a timing
        // assertion — see the stage evidence ledger for measured old-vs-new behavior.
        let unit = "a".repeat(50) + "\\\"";
        let filler: String = unit.repeat(4000);
        let content = format!("{filler}</think>visible");

        assert!(
            contains_unquoted_marker(&content, "</think>"),
            "the real marker after {} chars of adversarial unmatched-escaped-quote \
             filler must still be found",
            content.len()
        );
    }

    #[test]
    fn marker_looking_visible_text_stays_visible_in_batch() {
        let literal = "The literal \"</think>\" closes reasoning.";
        let parsed =
            parse_complete(QWEN3_UNIFIED_FAMILY, literal, &GuidedToolConstraint::None).unwrap();
        assert_eq!(parsed.text, literal);
        assert!(parsed.reasoning.is_empty());

        let embedded = "The string \"a </think> token\" stays visible.";
        let parsed =
            parse_complete(QWEN3_UNIFIED_FAMILY, embedded, &GuidedToolConstraint::None).unwrap();
        assert_eq!(parsed.text, embedded);
        assert!(parsed.reasoning.is_empty());
    }

    #[test]
    fn plain_text_passes_through_the_batch_path_unchanged() {
        let parsed = parse_complete(
            QWEN3_UNIFIED_FAMILY,
            "just an answer",
            &GuidedToolConstraint::None,
        )
        .unwrap();
        assert_eq!(parsed.text, "just an answer");
        assert!(parsed.reasoning.is_empty());
        assert!(parsed.tool_calls.is_empty());
    }
}
