// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered reasoning, text, and tool-call parsing through one state machine.
//!
//! Qwen3 is gated behind
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

use std::collections::{HashMap, HashSet, VecDeque};
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
pub(crate) const DEEPSEEK_V41_UNIFIED_FAMILY: &str = "deepseek_v41";

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
        (Some(DEEPSEEK_V41_UNIFIED_FAMILY), Some(DEEPSEEK_V41_UNIFIED_FAMILY)) => {
            Some(DEEPSEEK_V41_UNIFIED_FAMILY)
        }
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
        DEEPSEEK_V41_UNIFIED_FAMILY => true,
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
        DEEPSEEK_V41_UNIFIED_FAMILY => true,
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
        DEEPSEEK_V41_UNIFIED_FAMILY => UnifiedParserStartingState::Response,
        _ => UnifiedParserStartingState::None,
    }
}

/// Resolve a `Reasoning`-prefill choice's ACTUAL starting state when guided JSON is
/// installed, from the shape of that choice's first content-bearing chunk.
///
/// Guided decoding forbids the model from emitting the reasoning closer, so a request
/// whose rendered prompt already opened reasoning (`prompt_injected_reasoning`) would,
/// if started unconditionally at `Reasoning`, have its ENTIRE guided JSON payload
/// swallowed as `reasoning_content` with zero tool calls ever extracted — the parser
/// waits forever for a closer guided decoding will never produce. But committing to
/// `None` unconditionally instead is equally wrong: a turn that genuinely reasons before
/// its guided JSON (`reasoning</think>JSON`) would then be misread as pure content, and
/// the legitimate `reasoning_content`/`content` split would be lost.
///
/// Mirrors `bypass_bare_guided_json`'s shape check in
/// `Preprocessor::parse_reasoning_content_from_stream_inner` (preprocessor.rs), which
/// makes the identical call for the non-unified reasoning-parser path: a payload whose
/// first non-whitespace byte is `[` or `{` is bare guided JSON with no reasoning to
/// split out (start at `None`, so the whole payload reaches the tool/response path
/// untouched); anything else is ordinary text, meaning this really is a reasoning block
/// that will close normally (start at `Reasoning`).
///
/// An empty or whitespace-only first chunk (e.g. a role-only opening delta) is
/// inconclusive by this single-chunk check — unlike the legacy path, which re-evaluates
/// per subsequent chunk, the unified parser commits to a starting state once at
/// `ChoiceState` creation, so an inconclusive first chunk conservatively keeps the
/// `Reasoning` default rather than risking a genuine reasoning turn being misclassified.
fn bare_guided_json_prefill(
    first_content: Option<&ChatCompletionMessageContent>,
) -> UnifiedParserStartingState {
    let Some(ChatCompletionMessageContent::Text(text)) = first_content else {
        return UnifiedParserStartingState::Reasoning;
    };
    let trimmed = text.trim_start();
    if trimmed.is_empty() {
        return UnifiedParserStartingState::Reasoning;
    }
    match trimmed.as_bytes()[0] {
        b'[' | b'{' => UnifiedParserStartingState::None,
        _ => UnifiedParserStartingState::Reasoning,
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
        QWEN3_UNIFIED_FAMILY | DEEPSEEK_V41_UNIFIED_FAMILY => {
            // Compare FIRST-occurrence positions, not mere presence: a prompt that
            // pre-opened reasoning produces a leading `</think>` with no opener before
            // it, but a later `<think>...</think>` pair from the model can still follow
            // in the same output (e.g. after a tool-call gap). Testing "does an opener
            // exist anywhere" before "does a closer exist anywhere" would misclassify
            // that case as `None` instead of `Reasoning`.
            let opener = first_unquoted_marker_position(content, "<think>");
            let closer = first_unquoted_marker_position(content, "</think>");
            Ok(match (opener, closer) {
                // No opener before it (or no opener at all): the prompt opened reasoning.
                (None, Some(_)) => UnifiedParserStartingState::Reasoning,
                (Some(open_at), Some(close_at)) if close_at < open_at => {
                    UnifiedParserStartingState::Reasoning
                }
                // An opener exists (and any closer does not precede it): the model will
                // open reasoning itself later, currently visible as ordinary text.
                (Some(_), _) => UnifiedParserStartingState::None,
                (None, None) => UnifiedParserStartingState::Response,
            })
        }
        other => anyhow::bail!("no prefill detector for unified parser family '{other}'"),
    }
}

/// Marker-looking text quoted as prose is visible content, not parser control syntax.
pub(crate) fn contains_unquoted_marker(content: &str, marker: &str) -> bool {
    first_unquoted_marker_position(content, marker).is_some()
}

/// The byte offset of the first unquoted occurrence of `marker` in `content`, or `None`
/// if it never appears outside a quoted span. Shared scanner behind
/// [`contains_unquoted_marker`] and [`detect_prefill`], which needs the position (not
/// just presence) to compare two markers' first occurrences.
pub(crate) fn first_unquoted_marker_position(content: &str, marker: &str) -> Option<usize> {
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
        // A contraction/possessive apostrophe is exempt from quote-opening whenever
        // EITHER side is alphanumeric (`James'`, `isn't`, `'twas`) — only an apostrophe
        // with non-alphanumeric on both sides (`' quoted text '`) is a real quote-open.
        // Requiring alphanumeric on both sides would treat a word-final possessive like
        // `James'` as opening a quote, and a later contraction's apostrophe would then
        // wrongly close that phantom span, hiding every real marker in between.
        let apostrophe = character == '\''
            && (content[..offset]
                .chars()
                .next_back()
                .is_some_and(char::is_alphanumeric)
                || content[offset + character.len_utf8()..]
                    .chars()
                    .next()
                    .is_some_and(char::is_alphanumeric));
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
            return Some(offset);
        }
    }
    None
}

/// Return the first unquoted native tool-call marker configured for `parser`.
///
/// Native markers quoted as prose remain visible text in both streaming and
/// aggregated responses.
pub(crate) fn first_unquoted_native_tool_call_marker(content: &str, parser: &str) -> Option<usize> {
    native_tool_call_start_tokens(parser)?
        .iter()
        .filter_map(|marker| first_unquoted_marker_position(content, marker))
        .min()
}

/// Like [`first_unquoted_native_tool_call_marker`], but only for markers that cannot be
/// ordinary prose.
///
/// Truncating a response at a marker is safe only when the marker is control syntax. Some
/// parsers open a call with a plain word — `phi4` uses `functools` — and an answer about
/// the Python module of that name would lose everything from the word onward. A marker
/// carrying a punctuation character (`<`, `|`, `[`) cannot be written by accident.
pub(crate) fn first_unquoted_structural_tool_call_marker(
    content: &str,
    parser: &str,
) -> Option<usize> {
    native_tool_call_start_tokens(parser)?
        .iter()
        .filter(|marker| !marker.chars().all(char::is_alphanumeric))
        .filter_map(|marker| first_unquoted_marker_position(content, marker))
        .min()
}

/// Return the start of an unquoted native marker, including an unquoted
/// partial marker suffix that must remain buffered for the next stream chunk.
pub(crate) fn unquoted_native_tool_call_marker_or_prefix_start(
    content: &str,
    parser: &str,
) -> Option<usize> {
    let markers = native_tool_call_start_tokens(parser)?;
    if let Some(marker_start) = markers
        .iter()
        .filter_map(|marker| first_unquoted_marker_position(content, marker))
        .min()
    {
        return Some(marker_start);
    }

    markers
        .iter()
        .flat_map(|marker| {
            marker
                .char_indices()
                .skip(1)
                .map(move |(end, _)| &marker[..end])
        })
        .filter(|prefix| {
            let before = content.strip_suffix(prefix).unwrap_or(content);
            first_unquoted_marker_position(before, prefix).is_none()
        })
        .filter_map(|prefix| {
            let before = content.strip_suffix(prefix)?;
            let position = first_unquoted_marker_position(content, prefix)?;
            (position == before.len()).then_some(position)
        })
        .min()
}

fn native_tool_call_start_tokens(parser: &str) -> Option<Vec<String>> {
    let parser_key = if parser.is_empty() { "default" } else { parser };
    Some(
        dynamo_parsers::tool_calling::parsers::get_tool_parser_map()
            .get(parser_key)?
            .parser_config
            .tool_call_start_tokens()
            .iter()
            .filter(|marker| !marker.is_empty())
            .cloned()
            .collect(),
    )
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

/// The malformed-guided-payload contract for one request, selected by the
/// guided tool-call streaming rollback lever.
///
/// Only [`InvalidGuidedPayloadPolicy::StreamBestEffort`] may emit a call before its
/// payload closes; the other two contracts buffer to completion (see the policy's
/// own doc table). So this IS the streaming decision on the unified path, the same
/// way `guided_streaming` is on the jail path — the caller passes down one already-
/// made decision rather than re-deriving it here.
///
/// Rolling back picks `RecoverAsText`, not `Reject`: the lever exists to stop early
/// emission, not to convert a malformed payload from recovered text into a typed
/// error, which would be a second, louder behaviour change riding along with it.
fn invalid_guided_payload_policy(guided_streaming: bool) -> InvalidGuidedPayloadPolicy {
    if guided_streaming {
        InvalidGuidedPayloadPolicy::StreamBestEffort
    } else {
        InvalidGuidedPayloadPolicy::RecoverAsText
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
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum QwenNativeLexicalMode {
    #[default]
    OutsideBlock,
    InsideBlock,
    ParameterHeader,
    ParameterValue,
}

#[derive(Clone, Default)]
struct QwenNativeLexicalState {
    mode: QwenNativeLexicalMode,
    marker_tail: String,
}

impl QwenNativeLexicalState {
    fn observe(&mut self, text: &str) {
        const BLOCK_START: &str = "<tool_call>";
        const BLOCK_END: &str = "</tool_call>";
        const PARAMETER_START: &str = "<parameter=";
        const PARAMETER_END: &str = "</parameter>";
        const FUNCTION_END: &str = "</function>";

        let mut input = std::mem::take(&mut self.marker_tail);
        input.push_str(text);
        loop {
            match self.mode {
                QwenNativeLexicalMode::OutsideBlock => {
                    if let Some(start) = input.find(BLOCK_START) {
                        input.drain(..start + BLOCK_START.len());
                        self.mode = QwenNativeLexicalMode::InsideBlock;
                    } else {
                        self.retain_partial_suffix(&input, &[BLOCK_START]);
                        return;
                    }
                }
                QwenNativeLexicalMode::InsideBlock => {
                    let parameter = input.find(PARAMETER_START);
                    let block_end = input.find(BLOCK_END);
                    match (parameter, block_end) {
                        (Some(parameter), Some(block_end)) if block_end < parameter => {
                            input.drain(..block_end + BLOCK_END.len());
                            self.mode = QwenNativeLexicalMode::OutsideBlock;
                        }
                        (Some(parameter), _) => {
                            input.drain(..parameter + PARAMETER_START.len());
                            self.mode = QwenNativeLexicalMode::ParameterHeader;
                        }
                        (None, Some(block_end)) => {
                            input.drain(..block_end + BLOCK_END.len());
                            self.mode = QwenNativeLexicalMode::OutsideBlock;
                        }
                        (None, None) => {
                            self.retain_partial_suffix(&input, &[PARAMETER_START, BLOCK_END]);
                            return;
                        }
                    }
                }
                QwenNativeLexicalMode::ParameterHeader => {
                    let header_end = input.find('>');
                    if let Some(function_end) = input.find(FUNCTION_END)
                        && header_end.is_none_or(|end| function_end < end)
                    {
                        input.drain(..function_end + FUNCTION_END.len());
                        self.mode = QwenNativeLexicalMode::InsideBlock;
                    } else if let Some(header_end) = header_end {
                        input.drain(..header_end + 1);
                        self.mode = QwenNativeLexicalMode::ParameterValue;
                    } else {
                        self.retain_partial_suffix(&input, &[FUNCTION_END]);
                        return;
                    }
                }
                QwenNativeLexicalMode::ParameterValue => {
                    // The native scanner can recover a function whose last
                    // parameter omitted its close. Mirror that literal boundary so
                    // later partial block openers still retain their event order.
                    let close = [PARAMETER_END, FUNCTION_END]
                        .into_iter()
                        .filter_map(|marker| input.find(marker).map(|at| (at, marker.len())))
                        .min_by_key(|(at, _)| *at);
                    if let Some((at, length)) = close {
                        input.drain(..at + length);
                        self.mode = QwenNativeLexicalMode::InsideBlock;
                    } else {
                        self.retain_partial_suffix(&input, &[PARAMETER_END, FUNCTION_END]);
                        return;
                    }
                }
            }
        }
    }

    fn retain_partial_suffix(&mut self, input: &str, markers: &[&str]) {
        let suffix_length = markers
            .iter()
            .flat_map(|marker| (1..marker.len()).map(move |length| &marker[..length]))
            .filter(|prefix| input.ends_with(prefix))
            .map(str::len)
            .max()
            .unwrap_or_default();
        self.marker_tail = input[input.len() - suffix_length..].to_string();
    }

    fn can_start_call(&self) -> bool {
        self.mode != QwenNativeLexicalMode::ParameterValue
    }
}

fn qwen_opener_resolution(held: &str, text: &str) -> Option<(usize, bool)> {
    const BLOCK_START: &str = "<tool_call>";
    for (start, character) in text.char_indices() {
        let end = start + character.len_utf8();
        let candidate = format!("{held}{}", &text[..end]);
        if candidate == BLOCK_START {
            return Some((end, true));
        }
        if !BLOCK_START.starts_with(&candidate) {
            return Some((end, false));
        }
    }
    None
}

fn qwen_partial_opener_suffix_start(
    lexical_state: &QwenNativeLexicalState,
    text: &str,
) -> Option<usize> {
    const BLOCK_START: &str = "<tool_call>";
    let start = (1..BLOCK_START.len())
        .rev()
        .find(|length| text.ends_with(&BLOCK_START[..*length]))
        .map(|length| text.len() - length)?;
    let mut before_candidate = lexical_state.clone();
    before_candidate.observe(&text[..start]);
    (before_candidate.mode == QwenNativeLexicalMode::OutsideBlock).then_some(start)
}

pub(crate) struct ChoiceState {
    family: String,
    parser: Box<dyn UnifiedParser>,
    tool_output_mode: UnifiedToolOutputMode,
    qwen_native_lexical_state: QwenNativeLexicalState,
    /// Qwen's native tool opener is held until it is complete or disproven.
    pending_qwen_opener_text: String,
    /// Reasoning chunks arriving while a split Qwen block opener is ambiguous.
    /// They must follow recovered opener text on mismatch and precede call events
    /// when the opener completes.
    pending_qwen_opener_reasoning: Vec<String>,
    qwen_reasoning_event_positions: HashSet<usize>,
    qwen_reasoning_choice_positions: HashSet<usize>,
    /// Tool indices that have emitted at least one `tool_calls` delta.
    opened_calls: HashSet<usize>,
    /// Upstream IDs and types may arrive after an index opens. Names remain fragments.
    tool_ids_emitted: HashSet<usize>,
    tool_types_emitted: HashSet<usize>,
    /// Stable externally visible indices for parser-local and pass-through calls.
    parser_tool_indices: HashMap<usize, usize>,
    passthrough_tool_indices: HashMap<usize, usize>,
    /// Offset applied to tool indices from a parser instance rebuilt after a detour.
    tool_index_offset: usize,
    /// Next externally visible tool index, retained across parser-instance detours.
    next_tool_index: usize,
    /// Tool indices whose UnifiedParser events have not emitted an explicit completion.
    pending_tool_calls: HashSet<usize>,
    /// Pass-through OpenAI call fragments awaiting a terminal finish reason.
    pending_parsed_tool_calls: HashSet<usize>,
    /// Whether any tool-call delta was emitted, including a streamed fragment.
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
        guided_streaming: bool,
    ) -> anyhow::Result<Self> {
        let mut parser = create_unified_parser_for_family(family, tools)?;
        // `prompt_token_ids` stays empty: this path establishes the starting state from
        // the rendered prompt text (see `stream_prefill` / `detect_prefill`), not from
        // token IDs, which the preprocessor has already consumed by this point.
        //
        // Streaming guided calls commit per call once the name and argument object opener
        // are unambiguous. A committed fragment cannot later be rejected or recovered as
        // text, so this path opts into that contract explicitly — unless the request
        // rolled guided tool streaming back, in which case it buffers to completion.
        // `guided_streaming` is decided once per request by
        // `OpenAIPreprocessor::guided_tool_streaming_release` and threaded down here;
        // it is inert in `UnifiedToolOutputMode::Native`, where the parser builds no
        // guided state at all.
        parser.initialize_request(UnifiedParserInit {
            prompt_token_ids: Vec::new(),
            starting_state: prefill,
            tool_output_mode: tool_output_mode.clone(),
            invalid_guided_payload: invalid_guided_payload_policy(guided_streaming),
        })?;
        Ok(Self {
            family: family.to_string(),
            parser,
            tool_output_mode,
            qwen_native_lexical_state: QwenNativeLexicalState::default(),
            pending_qwen_opener_text: String::new(),
            pending_qwen_opener_reasoning: Vec::new(),
            qwen_reasoning_event_positions: HashSet::new(),
            qwen_reasoning_choice_positions: HashSet::new(),
            opened_calls: HashSet::new(),
            tool_ids_emitted: HashSet::new(),
            tool_types_emitted: HashSet::new(),
            parser_tool_indices: HashMap::new(),
            passthrough_tool_indices: HashMap::new(),
            tool_index_offset: 0,
            next_tool_index: 0,
            pending_tool_calls: HashSet::new(),
            pending_parsed_tool_calls: HashSet::new(),
            tool_emitted: false,
            failed: false,
        })
    }

    pub(crate) fn new_default(family: &str, tools: &[Tool]) -> anyhow::Result<Self> {
        Ok(Self {
            family: family.to_string(),
            parser: create_unified_parser_for_family(family, tools)?,
            tool_output_mode: UnifiedToolOutputMode::Native,
            qwen_native_lexical_state: QwenNativeLexicalState::default(),
            pending_qwen_opener_text: String::new(),
            pending_qwen_opener_reasoning: Vec::new(),
            qwen_reasoning_event_positions: HashSet::new(),
            qwen_reasoning_choice_positions: HashSet::new(),
            opened_calls: HashSet::new(),
            tool_ids_emitted: HashSet::new(),
            tool_types_emitted: HashSet::new(),
            parser_tool_indices: HashMap::new(),
            passthrough_tool_indices: HashMap::new(),
            tool_index_offset: 0,
            next_tool_index: 0,
            pending_tool_calls: HashSet::new(),
            pending_parsed_tool_calls: HashSet::new(),
            tool_emitted: false,
            failed: false,
        })
    }

    /// Feed one decoded text delta through the parser.
    pub(crate) fn push(&mut self, text: &str) -> Vec<UnifiedParserEvent> {
        if self.failed {
            return text_delta(text.to_string());
        }
        if self.is_qwen_native_outside_block() {
            let held = std::mem::take(&mut self.pending_qwen_opener_text);
            if !self.pending_qwen_opener_reasoning.is_empty() && !held.is_empty() {
                if let Some((_resolution_end, opener_completed)) =
                    qwen_opener_resolution(&held, text)
                {
                    let mut combined = held.clone();
                    combined.push_str(text);
                    let mut events = self.parse_segment(&combined);
                    if opener_completed {
                        self.prepend_pending_qwen_reasoning(&mut events);
                    } else {
                        self.insert_pending_qwen_reasoning_after_recovered_opener(
                            &held,
                            &mut events,
                        );
                    }
                    return events;
                }
                self.pending_qwen_opener_text = held;
                self.pending_qwen_opener_text.push_str(text);
                return Vec::new();
            }

            let mut combined = held;
            combined.push_str(text);
            if let Some(start) =
                qwen_partial_opener_suffix_start(&self.qwen_native_lexical_state, &combined)
            {
                let safe_text = combined[..start].to_string();
                self.pending_qwen_opener_text = combined[start..].to_string();
                return if safe_text.is_empty() {
                    Vec::new()
                } else {
                    self.parse_segment(&safe_text)
                };
            }
            return self.parse_segment(&combined);
        }
        self.parse_segment(text)
    }

    fn is_qwen_native_outside_block(&self) -> bool {
        self.family == QWEN3_UNIFIED_FAMILY
            && matches!(&self.tool_output_mode, UnifiedToolOutputMode::Native)
            && self.qwen_native_lexical_state.mode == QwenNativeLexicalMode::OutsideBlock
    }

    fn parse_segment(&mut self, text: &str) -> Vec<UnifiedParserEvent> {
        if self.family == QWEN3_UNIFIED_FAMILY
            && matches!(&self.tool_output_mode, UnifiedToolOutputMode::Native)
        {
            self.qwen_native_lexical_state.observe(text);
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

    fn queue_qwen_opener_reasoning(&mut self, reasoning: String) {
        self.pending_qwen_opener_reasoning.push(reasoning);
    }

    fn prepend_pending_qwen_reasoning(&mut self, events: &mut Vec<UnifiedParserEvent>) {
        let mut reasoning = std::mem::take(&mut self.pending_qwen_opener_reasoning)
            .into_iter()
            .map(UnifiedParserEvent::Reasoning)
            .collect::<Vec<_>>();
        self.qwen_reasoning_event_positions
            .extend(0..reasoning.len());
        reasoning.append(events);
        *events = reasoning;
    }

    fn append_pending_qwen_reasoning(&mut self, events: &mut Vec<UnifiedParserEvent>) {
        self.qwen_reasoning_event_positions
            .extend(events.len()..events.len() + self.pending_qwen_opener_reasoning.len());
        events.extend(
            std::mem::take(&mut self.pending_qwen_opener_reasoning)
                .into_iter()
                .map(UnifiedParserEvent::Reasoning),
        );
    }

    fn insert_pending_qwen_reasoning_after_recovered_opener(
        &mut self,
        held: &str,
        events: &mut Vec<UnifiedParserEvent>,
    ) {
        let pending = std::mem::take(&mut self.pending_qwen_opener_reasoning);
        let mut ordered = Vec::new();
        let mut inserted = false;
        for event in std::mem::take(events) {
            if !inserted
                && let UnifiedParserEvent::Text(text) = &event
                && let Some(remainder) = text.strip_prefix(held)
            {
                if !held.is_empty() {
                    ordered.push(UnifiedParserEvent::Text(held.to_string()));
                }
                self.qwen_reasoning_event_positions
                    .extend(ordered.len()..ordered.len() + pending.len());
                ordered.extend(pending.iter().cloned().map(UnifiedParserEvent::Reasoning));
                if !remainder.is_empty() {
                    ordered.push(UnifiedParserEvent::Text(remainder.to_string()));
                }
                inserted = true;
                continue;
            }
            ordered.push(event);
        }
        if !inserted {
            self.qwen_reasoning_event_positions
                .extend(ordered.len()..ordered.len() + pending.len());
            ordered.extend(pending.into_iter().map(UnifiedParserEvent::Reasoning));
        }
        *events = ordered;
    }

    /// Flush buffered partial state at end of stream.
    pub(crate) fn finish(&mut self) -> Vec<UnifiedParserEvent> {
        if self.failed {
            return Vec::new();
        }
        let pending_qwen_opener = std::mem::take(&mut self.pending_qwen_opener_text);
        let mut events = if pending_qwen_opener.is_empty() {
            Vec::new()
        } else {
            self.parse_segment(&pending_qwen_opener)
        };
        events.extend(match self.parser.finish() {
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
        });
        self.append_pending_qwen_reasoning(&mut events);
        events
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
        let mut recovered = self.parser.reset();
        if recovered.is_empty() {
            recovered.push_str(fallback);
        }
        recovered.push_str(&std::mem::take(&mut self.pending_qwen_opener_text));
        text_delta(recovered)
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
                let parser_index = call.tool_index + self.tool_index_offset;
                let tool_index = stable_tool_index(
                    parser_index,
                    &mut self.parser_tool_indices,
                    &self.opened_calls,
                    &mut self.next_tool_index,
                );
                self.tool_emitted = true;
                if call.complete {
                    self.pending_tool_calls.remove(&tool_index);
                    self.pending_parsed_tool_calls.remove(&tool_index);
                } else {
                    self.pending_tool_calls.insert(tool_index);
                }
                // The OpenAI streaming tool-call contract: the FIRST chunk for a tool
                // index carries id + type + name, later chunks carry only argument
                // fragments. `dynamo-parsers-v2` mints no ids (serving layers own them),
                // so one is minted here per call, exactly once.
                let first = self.opened_calls.insert(tool_index);
                if first {
                    self.tool_ids_emitted.insert(tool_index);
                    self.tool_types_emitted.insert(tool_index);
                }
                choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
                    index: tool_index as u32,
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
    /// `role` / `refusal` ride on the first choice owned by the current source
    /// response; queued reasoning keeps its earlier source envelope. The terminating
    /// `finish_reason` stays after every emitted delta.
    pub(crate) fn choices_for(
        &mut self,
        original: &ChatChoiceStream,
        deltas: Vec<UnifiedParserEvent>,
        emit_tool_calls: bool,
        finish_reason: Option<FinishReason>,
    ) -> Vec<ChatChoiceStream> {
        let terminal_tool_call_delta = finish_reason_closes_tool_calls(finish_reason.as_ref());
        if terminal_tool_call_delta {
            self.pending_parsed_tool_calls.clear();
        }
        // Queued reasoning belongs to its earlier source response. Coalescing it
        // with parser reasoning would erase the source envelope's event boundary.
        let source_positions = std::mem::take(&mut self.qwen_reasoning_event_positions);
        self.qwen_reasoning_choice_positions.clear();
        let mut run = Vec::new();
        let mut projected = Vec::new();
        for (position, delta) in deltas.into_iter().enumerate() {
            if !emit_tool_calls && matches!(delta, UnifiedParserEvent::ToolCall(_)) {
                continue;
            }
            if source_positions.contains(&position) {
                projected.extend(coalesce(std::mem::take(&mut run)));
                self.qwen_reasoning_choice_positions.insert(projected.len());
                projected.push(delta);
            } else {
                run.push(delta);
            }
        }
        projected.extend(coalesce(run));
        let deltas = projected;
        let envelope_position = (0..deltas.len())
            .find(|position| !self.qwen_reasoning_choice_positions.contains(position));
        let mut original = original.clone();
        if let Some(original_calls) = original.delta.tool_calls.as_mut() {
            for call in original_calls {
                let tool_index = normalize_passthrough_tool_call(
                    call,
                    &mut self.passthrough_tool_indices,
                    &mut self.opened_calls,
                    &mut self.tool_ids_emitted,
                    &mut self.tool_types_emitted,
                    &mut self.next_tool_index,
                );
                if !terminal_tool_call_delta {
                    self.pending_parsed_tool_calls.insert(tool_index);
                }
                self.tool_emitted = true;
            }
        }
        let index = original.index;
        let count = deltas.len();
        let mut choices = Vec::with_capacity(count.max(1));

        for (position, delta) in deltas.into_iter().enumerate() {
            let mut choice = self.delta_to_choice(index, delta);
            if Some(position) == envelope_position {
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
                    if let Some(parsed_calls) = choice.delta.tool_calls.take() {
                        original_calls.extend(parsed_calls);
                    }
                    choice.delta.tool_calls = Some(original_calls);
                }
            }
            if position + 1 == count && !self.qwen_reasoning_choice_positions.contains(&position) {
                choice.finish_reason = self.normalize_finish_reason(finish_reason);
            }
            choices.push(choice);
        }

        if !choices.is_empty()
            && (envelope_position.is_none()
                || (finish_reason.is_some()
                    && self.qwen_reasoning_choice_positions.contains(&(count - 1))))
        {
            let mut choice = empty_choice(index);
            if envelope_position.is_none() {
                choice.delta.role = original.delta.role;
                choice.delta.refusal = original.delta.refusal.clone();
                choice.delta.reasoning_content = original.delta.reasoning_content.clone();
                choice.delta.tool_calls = original.delta.tool_calls.clone();
                choice.delta.function_call = original.delta.function_call.clone();
            }
            choice.finish_reason = self.normalize_finish_reason(finish_reason);
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
        self.has_terminal_tool_calls()
            .then_some(FinishReason::ToolCalls)
    }

    /// Whether this choice has emitted a tool-call delta so far.
    pub(crate) fn tool_emitted(&self) -> bool {
        self.tool_emitted
    }

    fn has_terminal_tool_calls(&self) -> bool {
        self.tool_emitted
            && self.pending_tool_calls.is_empty()
            && self.pending_parsed_tool_calls.is_empty()
    }

    fn is_guided_named(&self) -> bool {
        matches!(
            &self.tool_output_mode,
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some(_)
            }
        )
    }

    fn should_preserve_reasoning_only_state(&self) -> bool {
        self.is_guided_named()
            || self.tool_emitted()
            // Reasoning is an out-of-band event even when malformed native bytes
            // look closed to the opener tracker. Only the parser owns recovery.
            || (self.family == QWEN3_UNIFIED_FAMILY
                && matches!(&self.tool_output_mode, UnifiedToolOutputMode::Native))
            || self.family == "kimi_k2"
    }

    fn qwen_resumed_marker_can_start_call(&self) -> bool {
        self.qwen_native_lexical_state.can_start_call()
    }

    /// Seed tool-call history from before an already-parsed detour into a freshly
    /// constructed parser instance for the same choice index.
    pub(crate) fn mark_tool_emitted(&mut self) {
        self.tool_emitted = true;
    }

    /// OpenAI streaming contract: `Stop` becomes `ToolCalls` only when every emitted
    /// call index is complete. `Length` / `ContentFilter` stay unchanged.
    fn normalize_finish_reason(&self, finish_reason: Option<FinishReason>) -> Option<FinishReason> {
        if finish_reason == Some(FinishReason::Stop) && self.has_terminal_tool_calls() {
            Some(FinishReason::ToolCalls)
        } else {
            finish_reason
        }
    }
}

// OpenAI tool-call chunks have no per-delta completion bit, so only a terminal
// finish reason can close a pass-through call fragment.
fn finish_reason_closes_tool_calls(finish_reason: Option<&FinishReason>) -> bool {
    matches!(
        finish_reason,
        Some(FinishReason::Stop | FinishReason::ToolCalls)
    )
}

fn stable_tool_index(
    source_index: usize,
    indices: &mut HashMap<usize, usize>,
    opened_calls: &HashSet<usize>,
    next_tool_index: &mut usize,
) -> usize {
    if let Some(tool_index) = indices.get(&source_index) {
        return *tool_index;
    }
    let tool_index = if opened_calls.contains(&source_index) {
        let mut candidate = *next_tool_index;
        while opened_calls.contains(&candidate) {
            candidate += 1;
        }
        candidate
    } else {
        source_index
    };
    indices.insert(source_index, tool_index);
    *next_tool_index = (*next_tool_index).max(tool_index + 1);
    tool_index
}

fn normalize_passthrough_tool_call(
    call: &mut ChatCompletionMessageToolCallChunk,
    indices: &mut HashMap<usize, usize>,
    opened_calls: &mut HashSet<usize>,
    ids_emitted: &mut HashSet<usize>,
    types_emitted: &mut HashSet<usize>,
    next_tool_index: &mut usize,
) -> usize {
    let tool_index = stable_tool_index(call.index as usize, indices, opened_calls, next_tool_index);
    if call.id.is_some() && !ids_emitted.insert(tool_index) {
        call.id = None;
    }
    if call.r#type.is_some() && !types_emitted.insert(tool_index) {
        call.r#type = None;
    }
    call.index = tool_index as u32;
    opened_calls.insert(tool_index);
    tool_index
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
    tool_definitions: &[ToolDefinition],
) -> anyhow::Result<CompleteOutput> {
    let tools = to_v2_tools(Some(tool_definitions));
    let mut parser = create_unified_parser_for_family(family, &tools)?;
    // Batch replays already-generated output. Prefer its observable native marker when
    // topology B could not carry the worker's installed structural tag to the frontend;
    // otherwise use the reconstructed guided-JSON constraint.
    let effective_mode = batch_tool_output_mode(content, guided_tool_constraint);
    // `batch_tool_output_mode` drops to `Native` on observed markup even when the
    // constraint reconstructed a `GuidedJsonNamed` pin. That native fallback recovers
    // whatever tool name the markup happens to contain, so it must be checked against
    // the forced name below instead of trusted unfiltered — a malformed guided output
    // that embeds a *different* tool's markup must not be handed back as the pinned
    // tool's call.
    let forced_tool_name = match (guided_tool_constraint, &effective_mode) {
        (GuidedToolConstraint::GuidedJsonNamed { tool_name }, UnifiedToolOutputMode::Native) => {
            Some(tool_name.as_str())
        }
        _ => None,
    };
    parser.initialize_request(UnifiedParserInit {
        starting_state: detect_prefill(family, content)?,
        tool_output_mode: effective_mode,
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
                if forced_tool_name.is_some_and(|forced| forced != name) {
                    tracing::warn!(
                        forced_tool_name = forced_tool_name,
                        recovered_tool_name = %name,
                        "dropped native-fallback tool call whose name did not match the tool_choice-forced tool_name"
                    );
                    continue;
                }
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
/// index: whether it has emitted a tool call, which indices remain provisional, and whether its terminal chunk has
/// already been sent. See the field comment at its use site in
/// `apply_stream_with_constraint` for why this can't just live inside `ChoiceState`.
#[derive(Default)]
struct ChoiceRecord {
    opened_calls: HashSet<usize>,
    tool_ids_emitted: HashSet<usize>,
    tool_types_emitted: HashSet<usize>,
    parser_tool_indices: HashMap<usize, usize>,
    passthrough_tool_indices: HashMap<usize, usize>,
    next_tool_index: usize,
    /// Incomplete events from the live UnifiedParser instance.
    pending_tool_calls: HashSet<usize>,
    /// Pass-through OpenAI call fragments awaiting a terminal finish reason.
    pending_parsed_tool_calls: HashSet<usize>,
    /// Source envelopes for Qwen reasoning held until a split native opener resolves.
    pending_qwen_opener_responses: VecDeque<PendingQwenReasoningResponse>,
    tool_emitted: bool,
    finished: bool,
    /// Whether an already-parsed chunk has EVER interrupted this choice. An
    /// already-parsed chunk implies the model has already emitted structured output
    /// (reasoning_content / tool_calls / function_call / `Parts` content), which per
    /// `already_parsed`'s own trigger set only happens after any reasoning phase would
    /// have concluded. So a raw run resuming after such a detour must rebuild its
    /// `ChoiceState` starting at `Response`, never at the outer request-level `prefill`
    /// (which only describes what the PROMPT opened, before this choice ever ran) — see
    /// the `Vacant` arm in the main loop below.
    detoured: bool,
}

struct PendingQwenReasoningResponse {
    text: String,
    choice: ChatChoiceStream,
    response: Annotated<NvCreateChatCompletionStreamResponse>,
}

enum FinishedStreamChunk {
    Choice(ChatChoiceStream),
    Source(Annotated<Box<NvCreateChatCompletionStreamResponse>>),
}

fn take_qwen_reasoning_response(
    record: &mut ChoiceRecord,
) -> Annotated<Box<NvCreateChatCompletionStreamResponse>> {
    let pending = record
        .pending_qwen_opener_responses
        .pop_front()
        .expect("queued reasoning event must have a source response");
    let mut response = pending.response;
    let mut source_reasoning = pending.choice;
    source_reasoning.delta.reasoning_content = Some(pending.text);
    source_reasoning.finish_reason = None;
    response
        .data
        .as_mut()
        .expect("queued source response has data")
        .inner
        .choices = vec![source_reasoning];
    response.map_data(|data| Ok(Box::new(data)))
}

fn qwen_chunks_for_choices(
    state: &ChoiceState,
    record: &mut ChoiceRecord,
    choices: Vec<ChatChoiceStream>,
) -> Vec<FinishedStreamChunk> {
    choices
        .into_iter()
        .enumerate()
        .map(|(position, choice)| {
            if state.qwen_reasoning_choice_positions.contains(&position) {
                FinishedStreamChunk::Source(take_qwen_reasoning_response(record))
            } else {
                FinishedStreamChunk::Choice(choice)
            }
        })
        .collect()
}

impl ChoiceRecord {
    fn remember_state(&mut self, state: &ChoiceState) {
        self.opened_calls.extend(state.opened_calls.iter().copied());
        self.tool_ids_emitted
            .extend(state.tool_ids_emitted.iter().copied());
        self.tool_types_emitted
            .extend(state.tool_types_emitted.iter().copied());
        self.parser_tool_indices
            .extend(state.parser_tool_indices.iter().map(|(k, v)| (*k, *v)));
        self.passthrough_tool_indices
            .extend(state.passthrough_tool_indices.iter().map(|(k, v)| (*k, *v)));
        self.next_tool_index = self.next_tool_index.max(state.next_tool_index);
        self.pending_tool_calls
            .clone_from(&state.pending_tool_calls);
        self.pending_parsed_tool_calls
            .clone_from(&state.pending_parsed_tool_calls);
        self.tool_emitted |= state.tool_emitted();
    }
}

/// Finish every choice that never received a terminating chunk, in index order.
///
/// A choice can reach this point two ways: still holding a live `ChoiceState` (the
/// common case — nothing special happened, it just never got an explicit terminal),
/// or "history-only" — its `ChoiceState` was removed by an already-parsed detour and
/// never rebuilt, so all that remains is its `ChoiceRecord`. The second case has
/// nothing left to flush, but a choice with a completed tool call still needs a synthesized
/// `ToolCalls` terminal so a strict client doesn't hang waiting for one; a choice that
/// never called a tool gets no synthetic chunk at all, matching the no-signal,
/// text-only contract a live `ChoiceState` already has via `tool_emitted` above.
fn finish_unterminated_choices(
    states: &mut HashMap<u32, ChoiceState>,
    records: &mut HashMap<u32, ChoiceRecord>,
) -> Vec<FinishedStreamChunk> {
    let mut indices: Vec<u32> = states
        .keys()
        .copied()
        .chain(records.keys().copied())
        .collect();
    indices.sort_unstable();
    indices.dedup();

    let mut chunks = Vec::new();
    for index in indices {
        if records.get(&index).is_some_and(|record| record.finished) {
            continue;
        }
        let record = records.entry(index).or_default();
        record.finished = true;
        let base = empty_choice(index);
        match states.get_mut(&index) {
            Some(state) => {
                let deltas = state.finish();
                // Only complete calls get a synthetic `ToolCalls` terminal at EOF;
                // provisional parser or pass-through fragments must stay provisional.
                let mut flushed = state.choices_for(&base, deltas, true, None);
                if state.has_terminal_tool_calls() {
                    if let Some(last_position) = flushed.len().checked_sub(1) {
                        if state
                            .qwen_reasoning_choice_positions
                            .contains(&last_position)
                        {
                            let mut terminal = empty_choice(index);
                            terminal.finish_reason = Some(FinishReason::ToolCalls);
                            flushed.push(terminal);
                        } else {
                            flushed[last_position].finish_reason = Some(FinishReason::ToolCalls);
                        }
                    } else {
                        flushed.extend(state.choices_for(
                            &base,
                            Vec::new(),
                            true,
                            Some(FinishReason::ToolCalls),
                        ));
                    }
                }
                chunks.extend(qwen_chunks_for_choices(state, record, flushed));
            }
            None => {
                while !record.pending_qwen_opener_responses.is_empty() {
                    chunks.push(FinishedStreamChunk::Source(take_qwen_reasoning_response(
                        record,
                    )));
                }
                if record.tool_emitted
                    && record.pending_tool_calls.is_empty()
                    && record.pending_parsed_tool_calls.is_empty()
                {
                    let mut choice = base;
                    choice.finish_reason = Some(FinishReason::ToolCalls);
                    chunks.push(FinishedStreamChunk::Choice(choice));
                }
            }
        }
    }
    chunks
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
        // Streaming released, matching the production default when the rollback lever
        // is unset. Tests that exercise the rolled-back contract call
        // `apply_stream_with_constraint` directly with `false`.
        true,
    )
}

/// `guided_streaming` is the request's already-made guided tool-call streaming
/// decision (`OpenAIPreprocessor::guided_tool_streaming_release`), not a second
/// derivation of it: `true` releases each call as soon as its name and argument
/// object opener are unambiguous, `false` buffers every call to completion. It is
/// inert unless `guided_tool_constraint` installed guided JSON.
pub(crate) fn apply_stream_with_constraint<S>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    guided_tool_constraint: GuidedToolConstraint,
    prefill: UnifiedParserStartingState,
    family: &'static str,
    guided_streaming: bool,
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
            let response_id = response.id.clone();
            let response_event = response.event.clone();
            let response_comment = response.comment.clone();
            let response_error = response.error.clone();
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
                    for output in finish_unterminated_choices(&mut states, &mut records) {
                        match output {
                            FinishedStreamChunk::Choice(choice) => {
                                yield response_with_choice(template, choice);
                            }
                            FinishedStreamChunk::Source(response) => yield response.map_data(|data| Ok(*data)),
                        }
                    }
                }
                yield response;
                continue;
            }

            let originals = std::mem::take(&mut chat.inner.choices);
            let mut emitted: Vec<ChatChoiceStream> = Vec::new();
            let mut deferred_qwen_responses = Vec::new();
            let mut qwen_source_metadata_deferred = false;
            for mut original in originals {
                if already_parsed(&original) {
                    // Record this index exists even if no `ChoiceState` is ever
                    // built for it (e.g. its first-ever chunk is already-parsed),
                    // so it is not invisible to `finish_unterminated_choices`.
                    let record = records.entry(original.index).or_default();
                    // Any raw run resuming this choice after this point must rebuild
                    // starting at `Response`, not the outer request-level `prefill` —
                    // see the `Vacant` arm below and the field doc on `ChoiceRecord`.
                    record.detoured = true;
                    if let Some(state) = states.get(&original.index) {
                        record.remember_state(state);
                    }
                    let parsed_calls_are_terminal =
                        finish_reason_closes_tool_calls(original.finish_reason.as_ref());
                    if parsed_calls_are_terminal {
                        record.pending_parsed_tool_calls.clear();
                    }
                    // An already-parsed chunk carries its tool calls verbatim in its
                    // own delta rather than through a `ChoiceState`, so that history
                    // has to be observed here directly — a state may never exist for
                    // this index at all (its very first chunk can be already-parsed).
                    if let Some(calls) = original
                        .delta
                        .tool_calls
                        .as_mut()
                        .filter(|calls| !calls.is_empty())
                    {
                        record.tool_emitted = true;
                        for call in calls {
                            let tool_index = normalize_passthrough_tool_call(
                                call,
                                &mut record.passthrough_tool_indices,
                                &mut record.opened_calls,
                                &mut record.tool_ids_emitted,
                                &mut record.tool_types_emitted,
                                &mut record.next_tool_index,
                            );
                            if parsed_calls_are_terminal {
                                record.pending_tool_calls.remove(&tool_index);
                            } else {
                                record.pending_parsed_tool_calls.insert(tool_index);
                            }
                        }
                    }
                    let reported_tool_indices: HashSet<usize> = if parsed_calls_are_terminal {
                        original
                            .delta
                            .tool_calls
                            .as_ref()
                            .into_iter()
                            .flatten()
                            .map(|call| call.index as usize)
                            .collect()
                    } else {
                        HashSet::new()
                    };
                    let preserve_reasoning_only_state = original.finish_reason.is_none()
                        && original.delta.reasoning_content.is_some()
                        && original.delta.content.is_none()
                        && original.delta.tool_calls.is_none()
                        && original.delta.function_call.is_none()
                        && original.delta.refusal.is_none()
                        && states
                            .get(&original.index)
                            .is_some_and(ChoiceState::should_preserve_reasoning_only_state);
                    let queue_qwen_opener_reasoning = preserve_reasoning_only_state
                        && original
                            .delta
                            .reasoning_content
                            .as_ref()
                            .is_some_and(|reasoning| !reasoning.is_empty())
                        && family == QWEN3_UNIFIED_FAMILY
                        && states.get(&original.index).is_some_and(|state| {
                            state.qwen_native_lexical_state.mode
                                == QwenNativeLexicalMode::OutsideBlock
                                && !state.pending_qwen_opener_text.is_empty()
                        });
                    if preserve_reasoning_only_state {
                        // Reasoning is an ordered event between generated bytes, not a
                        // replacement parser input. Keep the parser state alive so a
                        // later raw chunk can finish the same native call or guided
                        // argument object. Kimi may still be buffering an open JSON
                        // string before its first tool-call event.
                        if queue_qwen_opener_reasoning {
                            let reasoning = original
                                .delta
                                .reasoning_content
                                .take()
                                .expect("reasoning-only chunk has reasoning content");
                            states
                                .get_mut(&original.index)
                                .expect("preserved parser state must still exist")
                                .queue_qwen_opener_reasoning(reasoning.clone());
                            let mut source_data = chat.clone();
                            source_data.inner.choices.clear();
                            let keep_source_metadata = !qwen_source_metadata_deferred;
                            qwen_source_metadata_deferred = true;
                            if !keep_source_metadata {
                                source_data.inner.usage = None;
                                source_data.nvext = None;
                                source_data.llm_metrics = None;
                            }
                            record.pending_qwen_opener_responses.push_back(
                                PendingQwenReasoningResponse {
                                    text: reasoning,
                                    choice: original.clone(),
                                    response: Annotated {
                                        data: Some(source_data),
                                        id: if keep_source_metadata {
                                            response_id.clone()
                                        } else {
                                            None
                                        },
                                        event: if keep_source_metadata {
                                            response_event.clone()
                                        } else {
                                            None
                                        },
                                        comment: if keep_source_metadata {
                                            response_comment.clone()
                                        } else {
                                            None
                                        },
                                        error: if keep_source_metadata {
                                            response_error.clone()
                                        } else {
                                            None
                                        },
                                    },
                                },
                            );
                        }
                    } else if let Some(mut state) = states.remove(&original.index) {
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
                        record
                            .pending_tool_calls
                            .extend(
                                state
                                    .pending_tool_calls
                                    .iter()
                                    .filter(|index| !reported_tool_indices.contains(*index))
                                    .copied(),
                            );
                        record.opened_calls.extend(state.opened_calls.iter().copied());
                        record
                            .tool_ids_emitted
                            .extend(state.tool_ids_emitted.iter().copied());
                        record
                            .tool_types_emitted
                            .extend(state.tool_types_emitted.iter().copied());
                        record
                            .parser_tool_indices
                            .extend(state.parser_tool_indices.iter().map(|(k, v)| (*k, *v)));
                        record.passthrough_tool_indices.extend(
                            state
                                .passthrough_tool_indices
                                .iter()
                                .map(|(k, v)| (*k, *v)),
                        );
                        record.next_tool_index =
                            record.next_tool_index.max(state.next_tool_index);
                        state
                            .pending_tool_calls
                            .retain(|index| !reported_tool_indices.contains(index));
                        state
                            .pending_tool_calls
                            .extend(record.pending_tool_calls.iter().copied());
                        if parsed_calls_are_terminal {
                            state.pending_parsed_tool_calls.clear();
                        }
                        state
                            .pending_parsed_tool_calls
                            .extend(record.pending_parsed_tool_calls.iter().copied());
                        let deltas = state.finish();
                        let flushed = state.choices_for(
                            &empty_choice(original.index),
                            deltas,
                            true,
                            None,
                        );
                        for output in qwen_chunks_for_choices(&state, record, flushed) {
                            match output {
                                FinishedStreamChunk::Source(response) => {
                                    deferred_qwen_responses.push((emitted.len(), response));
                                }
                                FinishedStreamChunk::Choice(choice) => emitted.push(choice),
                            }
                        }
                        record.remember_state(&state);
                        if original.finish_reason == Some(FinishReason::Stop)
                            && record.tool_emitted
                            && record.pending_tool_calls.is_empty()
                            && record.pending_parsed_tool_calls.is_empty()
                        {
                            original.finish_reason = Some(FinishReason::ToolCalls);
                        }
                    } else {
                        if record.tool_emitted
                            && record.pending_tool_calls.is_empty()
                            && record.pending_parsed_tool_calls.is_empty()
                            && original.finish_reason == Some(FinishReason::Stop)
                        {
                            // No live state for this chunk (an earlier already-parsed
                            // chunk already discarded it), but this choice emitted a
                            // tool call before that gap.
                            original.finish_reason = Some(FinishReason::ToolCalls);
                        }
                    }
                    if original.finish_reason.is_some() {
                        record.finished = true;
                    }
                    if !queue_qwen_opener_reasoning {
                        emitted.push(original);
                    }
                    continue;
                }

                // Keep a provisional native call's parser across a reasoning-only
                // detour so its argument tail can finish. If the resumed bytes start
                // a fresh call instead, finish the old parser first; the record keeps
                // its emitted index pending while the next parser gets a new index.
                // Qwen's lexical state distinguishes a new block marker from the same
                // text inside a parameter value; JSON grammars use their own quoted
                // strings and may legally contain another parser marker there.
                let new_native_call_after_detour = records
                    .get(&original.index)
                    .is_some_and(|record| record.detoured)
                    && family == QWEN3_UNIFIED_FAMILY
                    && states
                        .get(&original.index)
                        .is_some_and(|state| {
                            matches!(&state.tool_output_mode, UnifiedToolOutputMode::Native)
                                && !state.pending_tool_calls.is_empty()
                                && state.qwen_resumed_marker_can_start_call()
                        })
                    && match original.delta.content.as_ref() {
                        Some(ChatCompletionMessageContent::Text(text)) => {
                            first_unquoted_structural_tool_call_marker(
                                text,
                                QWEN3_TOOL_CALL_PARSER,
                            )
                                .is_some_and(|marker| text[..marker].trim().is_empty())
                        }
                        _ => false,
                    };
                if new_native_call_after_detour
                    && let Some(mut state) = states.remove(&original.index)
                {
                        let deltas = state.finish();
                        let flushed = state.choices_for(
                            &empty_choice(original.index), deltas, true, None,
                        );
                        let record = records.entry(original.index).or_default();
                        for output in qwen_chunks_for_choices(&state, record, flushed) {
                            match output {
                                FinishedStreamChunk::Source(response) => {
                                    deferred_qwen_responses.push((emitted.len(), response));
                                }
                                FinishedStreamChunk::Choice(choice) => emitted.push(choice),
                            }
                        }
                        record.remember_state(&state);
                }

                let state = match states.entry(original.index) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let mode = tool_output_mode(&guided_tool_constraint);
                        // `Vacant` covers both a brand-new choice AND one whose
                        // `ChoiceState` was just dropped by an already-parsed detour.
                        // Only the former may start at the request-level `prefill`
                        // (what the PROMPT opened); a choice that has already been
                        // through a detour has, by definition, already emitted
                        // structured output, so its reasoning phase (if any) already
                        // concluded and a resumed raw run must start at `Response`.
                        let choice_prefill = if records
                            .get(&original.index)
                            .is_some_and(|record| record.detoured)
                        {
                            UnifiedParserStartingState::Response
                        } else if prefill == UnifiedParserStartingState::Reasoning
                            && guided_tool_constraint.installs_guided_json()
                        {
                            bare_guided_json_prefill(original.delta.content.as_ref())
                        } else {
                            prefill
                        };
                        match ChoiceState::new(family, &tools, choice_prefill, mode, guided_streaming) {
                            Ok(mut state) => {
                                // A raw run resuming after an already-parsed detour
                                // gets a brand-new parser instance; seed it with any
                                // tool-call history from before the gap so a Stop at
                                // the end of this run still normalizes correctly.
                                if records
                                    .get(&original.index)
                                    .is_some_and(|record| record.tool_emitted)
                                {
                                    state.mark_tool_emitted();
                                }
                                if let Some(record) = records.get(&original.index) {
                                    state.tool_index_offset = record.next_tool_index;
                                    state.next_tool_index = record.next_tool_index;
                                    state.opened_calls.extend(record.opened_calls.iter().copied());
                                    state
                                        .tool_ids_emitted
                                        .extend(record.tool_ids_emitted.iter().copied());
                                    state
                                        .tool_types_emitted
                                        .extend(record.tool_types_emitted.iter().copied());
                                    state
                                        .parser_tool_indices
                                        .extend(record.parser_tool_indices.iter().map(|(k, v)| (*k, *v)));
                                    state.passthrough_tool_indices.extend(
                                        record
                                            .passthrough_tool_indices
                                            .iter()
                                            .map(|(k, v)| (*k, *v)),
                                    );
                                    state
                                        .pending_tool_calls
                                        .extend(record.pending_tool_calls.iter().copied());
                                    state.pending_parsed_tool_calls.extend(
                                        record.pending_parsed_tool_calls.iter().copied(),
                                    );
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

                let choices = state.choices_for(&original, deltas, true, terminal);
                let mut parsed = Vec::new();
                let record = records.entry(original.index).or_default();
                for output in qwen_chunks_for_choices(state, record, choices) {
                    match output {
                        FinishedStreamChunk::Source(response) => {
                            deferred_qwen_responses.push((emitted.len() + parsed.len(), response));
                        }
                        FinishedStreamChunk::Choice(choice) => parsed.push(choice),
                    }
                }
                if parsed.is_empty() {
                    // A marker-only chunk produced no deltas. Keep it as an empty
                    // choice so the typed llm_metrics and annotation metadata it
                    // carries still reach the client.
                    parsed.push(empty_choice(original.index));
                }
                emitted.extend(parsed);
            }

            if qwen_source_metadata_deferred {
                chat.inner.usage = None;
                chat.nvext = None;
                chat.llm_metrics = None;
                response.id = None;
                response.event = None;
                response.comment = None;
                response.error = None;
            }

            if emitted.is_empty() && deferred_qwen_responses.is_empty() {
                continue;
            }

            // One upstream chunk can fan out into several. Usage, nvext and annotation
            // fields stay on the last child. Token metrics stay on one reasoning child
            // when present so the downstream reasoning-usage estimator preserves the
            // source chunk's classification without counting it twice.
            let last = emitted.len().checked_sub(1);
            let llm_metrics_position = fanout_llm_metrics_position(&emitted);
            let mut deferred_qwen_responses = deferred_qwen_responses.into_iter().peekable();
            for position in 0..=emitted.len() {
                while deferred_qwen_responses
                    .peek()
                    .is_some_and(|(deferred_position, _)| *deferred_position == position)
                {
                    let (_, response) = deferred_qwen_responses
                        .next()
                        .expect("peeked Qwen response should be present");
                    yield response.map_data(|data| Ok(*data));
                }
                let Some(choice) = emitted.get(position).cloned() else {
                    continue;
                };
                let is_last = Some(position) == last;
                let mut data = chat.clone();
                data.inner.choices = vec![choice];
                if !is_last {
                    data.inner.usage = None;
                    data.nvext = None;
                }
                if Some(position) != llm_metrics_position {
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
            for output in finish_unterminated_choices(&mut states, &mut records) {
                match output {
                    FinishedStreamChunk::Choice(choice) => {
                        yield response_with_choice(template, choice);
                    }
                    FinishedStreamChunk::Source(response) => yield response.map_data(|data| Ok(*data)),
                }
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
    use futures::FutureExt;
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

    fn count_tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "count_items".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"]
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
    fn deepseek_v41_uses_native_parsing_without_an_experimental_flag() {
        let pair = (Some("deepseek_v41"), Some("deepseek_v41"));
        assert_eq!(configured_family(pair.0, pair.1), Some("deepseek_v41"));
        assert_eq!(selected_family(pair.0, pair.1), Some("deepseek_v41"));
        assert_eq!(selected_batch_family(pair.0, pair.1), Some("deepseek_v41"));
        assert_eq!(configured_family(Some("deepseek_v41"), Some("qwen3")), None);
        assert_eq!(configured_family(Some("deepseek_v41"), None), None);
        assert_eq!(configured_family(None, Some("deepseek_v41")), None);
    }

    #[tokio::test]
    async fn deepseek_v41_preserves_calls_across_utf8_chunk_boundaries() {
        let call = concat!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"get_weather\">",
            "<｜DSML｜ parameter name=\"city\" string=\"true\">東京</｜DSML｜ parameter>",
            "</｜DSML｜ invoke></｜DSML｜ calls>"
        );
        for (prefix, prefill, reasoning) in [
            ("", UnifiedParserStartingState::Response, ""),
            (
                "plan</think>",
                UnifiedParserStartingState::Reasoning,
                "plan",
            ),
            (
                "<think>plan</think>",
                UnifiedParserStartingState::None,
                "plan",
            ),
        ] {
            let input = format!("{prefix}{call}");
            let batch = parse_complete(
                DEEPSEEK_V41_UNIFIED_FAMILY,
                &input,
                &GuidedToolConstraint::None,
                &[],
            )
            .unwrap();
            assert_eq!(batch.reasoning, reasoning);
            assert_eq!(batch.tool_calls.len(), 1);
            assert_eq!(batch.tool_calls[0].function.arguments, r#"{"city":"東京"}"#);
            for split in input.char_indices().map(|(index, _)| index) {
                let (first, second) = input.split_at(split);
                let responses = apply_stream(
                    stream::iter([chunk(first, false), chunk(second, true)]),
                    Some(weather_tools()),
                    None,
                    false,
                    prefill,
                    DEEPSEEK_V41_UNIFIED_FAMILY,
                )
                .collect::<Vec<_>>()
                .await;
                let mut expected = Vec::new();
                if !reasoning.is_empty() {
                    expected.push(LogicalEvent::Reasoning(reasoning.to_string()));
                }
                expected.push(LogicalEvent::Tool {
                    index: 0,
                    name: Some("get_weather".to_string()),
                    arguments: r#"{"city":"東京"}"#.to_string(),
                });
                let mut actual = logical_events(&responses);
                for event in &mut actual {
                    if let LogicalEvent::Tool { arguments, .. } = event {
                        *arguments = serde_json::from_str::<serde_json::Value>(arguments)
                            .unwrap()
                            .to_string();
                    }
                }
                assert_eq!(actual, expected, "split={split}");
                assert_eq!(
                    collect_choices(&responses).last().unwrap().finish_reason,
                    Some(FinishReason::ToolCalls)
                );
            }
        }
    }

    #[tokio::test]
    async fn deepseek_v41_complete_call_survives_missing_outer_closer() {
        // Model the frontend hiding an EOS token that is also the outer DSML
        // closer. A complete invocation is committed at `</｜DSML｜ invoke>`, so
        // removing only `</｜DSML｜ calls>` must not drop the call.
        let input = concat!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"get_weather\">",
            "<｜DSML｜ parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜ parameter>",
            "</｜DSML｜ invoke>"
        );

        let batch = parse_complete(
            DEEPSEEK_V41_UNIFIED_FAMILY,
            input,
            &GuidedToolConstraint::None,
            &weather_tools(),
        )
        .unwrap();
        assert_eq!(batch.tool_calls.len(), 1);
        assert_eq!(batch.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            batch.tool_calls[0].function.arguments,
            r#"{"city":"Tokyo"}"#
        );

        let responses = apply_stream(
            stream::iter([chunk(input, true)]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::Response,
            DEEPSEEK_V41_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        assert!(responses.iter().all(|response| !response.is_error()));
        assert_eq!(
            logical_events(&responses),
            vec![LogicalEvent::Tool {
                index: 0,
                name: Some("get_weather".to_string()),
                arguments: r#"{"city":"Tokyo"}"#.to_string(),
            }]
        );
        assert_eq!(
            collect_choices(&responses).last().unwrap().finish_reason,
            Some(FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn deepseek_v41_recovers_malformed_closed_arguments_as_text() {
        let text = concat!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"get_weather\">",
            "<｜DSML｜ parameter name=\"count\" string=\"false\">not-json</｜DSML｜ parameter>",
            "</｜DSML｜ invoke></｜DSML｜ calls>"
        );
        let responses = apply_stream(
            stream::iter([chunk(text, true)]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::Response,
            DEEPSEEK_V41_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        assert!(responses.iter().all(|response| !response.is_error()));
        assert_eq!(
            logical_events(&responses),
            vec![LogicalEvent::Text(text.to_string())]
        );
        assert_eq!(
            collect_choices(&responses).last().unwrap().finish_reason,
            Some(FinishReason::Stop)
        );
    }

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

    #[test]
    fn detect_prefill_compares_first_marker_positions_not_mere_presence() {
        // A prompt-opened reasoning span that the model closes, followed by the model
        // opening (and closing) a SECOND thought later in the same output. The leading
        // `</think>` has no opener before it, so the prompt pre-opened reasoning: this
        // must return `Reasoning`, not `None`. Presence-only testing ("does <think>
        // exist anywhere") wrongly answers `None` because a `<think>` exists somewhere
        // in the string, even though it comes AFTER the first `</think>`.
        assert_eq!(
            detect_prefill(
                QWEN3_UNIFIED_FAMILY,
                "secret</think>answer<think>second thought</think>tail"
            )
            .unwrap(),
            UnifiedParserStartingState::Reasoning,
            "a leading closer with no prior opener means the prompt pre-opened reasoning, \
             regardless of a later model-opened thought"
        );
    }

    #[test]
    fn apostrophe_exemption_needs_only_one_alphanumeric_side() {
        // A possessive apostrophe that ends a word (`James'`) has non-alphanumeric on
        // the RIGHT. Requiring alphanumeric on both sides treats it as a real
        // quote-open, and the later contraction apostrophe in `isn't` then closes that
        // phantom span, hiding the real `</think>` marker in between.
        assert_eq!(
            detect_prefill(
                QWEN3_UNIFIED_FAMILY,
                "James' secret </think>answer isn't final"
            )
            .unwrap(),
            UnifiedParserStartingState::Reasoning,
            "a trailing possessive apostrophe must not be treated as a real quote-open"
        );
        // Mirror case: a leading contraction (`'twas`) has non-alphanumeric on the LEFT
        // and alphanumeric only on the right — the OR-based fix must exempt this too.
        assert!(
            contains_unquoted_marker("'twas </think> hidden", "</think>"),
            "a leading-apostrophe contraction must not be treated as a real quote-open"
        );
    }

    // --- delta -> chunk conversion -----------------------------------------------

    fn call_delta(tool_index: usize, name: Option<&str>, arguments: &str) -> UnifiedParserEvent {
        UnifiedParserEvent::ToolCall(ToolCallDelta {
            tool_index,
            name: name.map(str::to_string),
            arguments: arguments.to_string(),
            complete: true,
        })
    }

    fn partial_call_delta(
        tool_index: usize,
        name: Option<&str>,
        arguments: &str,
    ) -> UnifiedParserEvent {
        UnifiedParserEvent::ToolCall(ToolCallDelta {
            tool_index,
            name: name.map(str::to_string),
            arguments: arguments.to_string(),
            complete: false,
        })
    }

    fn test_state() -> ChoiceState {
        ChoiceState::new(
            QWEN3_UNIFIED_FAMILY,
            &[],
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::Native,
            true,
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
    fn mixed_raw_content_and_upstream_tool_call_do_not_reuse_tool_index() {
        let mut state = test_state();
        let mut original = empty_choice(0);
        original.delta.content = Some(ChatCompletionMessageContent::Text(
            "raw parser input".to_string(),
        ));
        original.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("upstream-call".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("upstream".to_string()),
                arguments: Some("{}".to_string()),
            }),
        }]);

        let choices = state.choices_for(
            &original,
            vec![call_delta(0, Some("parsed"), "{}")],
            true,
            None,
        );
        let mut indices: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .map(|call| call.index)
            .collect();
        indices.sort_unstable();

        assert_eq!(indices, vec![0, 1]);

        let mut upstream_continuation = empty_choice(0);
        upstream_continuation.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: None,
            r#type: None,
            function: Some(FunctionCallStream {
                name: None,
                arguments: Some("more".to_string()),
            }),
        }]);
        let choices = state.choices_for(&upstream_continuation, Vec::new(), true, None);
        assert_eq!(
            choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
            0,
            "the pass-through call keeps its remapped index across chunks"
        );

        let choices = state.choices_for(
            &empty_choice(0),
            vec![call_delta(0, None, "more")],
            true,
            None,
        );
        assert_eq!(
            choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
            1,
            "the parser call keeps its index while the pass-through call continues"
        );
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
    fn incomplete_tool_call_fragments_do_not_mark_the_choice_terminal() {
        let mut state = test_state();
        let partial = state.choices_for(
            &empty_choice(0),
            vec![partial_call_delta(
                0,
                Some("get_weather"),
                r#"{"city":"Tok"#,
            )],
            true,
            Some(FinishReason::Stop),
        );

        assert_eq!(partial.len(), 1);
        assert!(partial[0].delta.tool_calls.is_some());
        assert_eq!(
            partial[0].finish_reason,
            Some(FinishReason::Stop),
            "an unfinished parser fragment must not rewrite Stop to ToolCalls"
        );
        assert!(state.tool_emitted());
        assert_eq!(state.unterminated_finish_reason(), None);

        let completed = state.choices_for(
            &empty_choice(0),
            vec![call_delta(0, None, "yo\"}")],
            true,
            Some(FinishReason::Stop),
        );
        assert_eq!(completed[0].finish_reason, Some(FinishReason::ToolCalls));
        assert!(state.tool_emitted());
        assert_eq!(
            state.unterminated_finish_reason(),
            Some(FinishReason::ToolCalls)
        );
    }

    #[test]
    fn completed_call_does_not_hide_another_incomplete_call() {
        let mut state = test_state();
        let partial = state.choices_for(
            &empty_choice(0),
            vec![partial_call_delta(0, Some("first"), r#"{"city":"Tok"#)],
            true,
            None,
        );
        assert_eq!(partial.len(), 1);

        let completed = state.choices_for(
            &empty_choice(0),
            vec![call_delta(1, Some("second"), r#"{"city":"Paris"}"#)],
            true,
            Some(FinishReason::Stop),
        );
        assert_eq!(completed[0].finish_reason, Some(FinishReason::Stop));
        assert_eq!(state.unterminated_finish_reason(), None);

        let final_fragment = state.choices_for(
            &empty_choice(0),
            vec![call_delta(0, None, "yo\"}")],
            true,
            Some(FinishReason::Stop),
        );
        assert_eq!(
            final_fragment[0].finish_reason,
            Some(FinishReason::ToolCalls)
        );
        assert_eq!(
            state.unterminated_finish_reason(),
            Some(FinishReason::ToolCalls)
        );
    }

    #[test]
    fn separate_terminal_finish_closes_all_parsed_call_fragments() {
        let mut state = test_state();
        let mut parsed_fragments = empty_choice(0);
        parsed_fragments.delta.tool_calls = Some(vec![
            ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some("call-first".to_string()),
                r#type: Some(FunctionType::Function),
                function: Some(FunctionCallStream {
                    name: Some("first".to_string()),
                    arguments: Some(r#"{"city":"Par"#.to_string()),
                }),
            },
            ChatCompletionMessageToolCallChunk {
                index: 1,
                id: Some("call-second".to_string()),
                r#type: Some(FunctionType::Function),
                function: Some(FunctionCallStream {
                    name: Some("second".to_string()),
                    arguments: Some(r#"{"city":"Tok"#.to_string()),
                }),
            },
        ]);
        let partials = state.choices_for(&parsed_fragments, Vec::new(), true, None);
        assert_eq!(partials.len(), 1);
        assert_eq!(state.unterminated_finish_reason(), None);

        let terminal =
            state.choices_for(&empty_choice(0), Vec::new(), true, Some(FinishReason::Stop));

        assert_eq!(terminal.len(), 1);
        assert_eq!(terminal[0].finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(
            state.unterminated_finish_reason(),
            Some(FinishReason::ToolCalls)
        );
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
    async fn qwen_delayed_reasoning_keeps_source_envelope_and_continuation_metadata() {
        for (opening, continuation_prefix) in [
            ("before <tool_c", "all>"),
            ("before <", "<think>raw</think><tool_call>"),
        ] {
            for reasoning_text in ["", "held thought"] {
                let mut reasoning = chunk("", false);
                let choice = &mut reasoning.data.as_mut().unwrap().inner.choices[0];
                choice.delta.content = None;
                choice.delta.reasoning_content = Some(reasoning_text.to_string());
                let source = reasoning.data.as_mut().unwrap();
                source.nvext = Some(serde_json::json!({"source": "reasoning"}));
                source.llm_metrics = Some(crate::protocols::common::metrics::LLMMetricAnnotation {
                    chunk_tokens: 7,
                    ..Default::default()
                });
                reasoning.id = Some("reasoning-source-id".to_string());
                reasoning.event = Some("reasoning-source-event".to_string());
                reasoning.comment = Some(vec!["reasoning-source-comment".to_string()]);

                let continuation_text = format!(
                    "{continuation_prefix}\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>answer"
                );
                let mut continuation = chunk(&continuation_text, true);
                let continuation_data = continuation.data.as_mut().unwrap();
                continuation_data.nvext = Some(serde_json::json!({"source": "continuation"}));
                continuation.id = Some("continuation-id".to_string());
                continuation.event = Some("continuation-event".to_string());
                continuation.comment = Some(vec!["continuation-comment".to_string()]);

                let responses = apply_stream(
                    stream::iter([chunk(opening, false), reasoning, continuation]),
                    Some(weather_tools()),
                    None,
                    false,
                    UnifiedParserStartingState::None,
                    QWEN3_UNIFIED_FAMILY,
                )
                .collect::<Vec<_>>()
                .await;

                let delayed_reasoning = responses
                    .iter()
                    .find(|response| {
                        response.data.as_ref().is_some_and(|data| {
                            data.inner.choices.iter().any(|choice| {
                                choice.delta.reasoning_content.as_deref() == Some(reasoning_text)
                            })
                        })
                    })
                    .expect("queued reasoning should be released when the opener resolves");
                let delayed_data = delayed_reasoning.data.as_ref().unwrap();
                assert_eq!(delayed_reasoning.id.as_deref(), Some("reasoning-source-id"));
                assert_eq!(
                    delayed_reasoning.event.as_deref(),
                    Some("reasoning-source-event")
                );
                assert_eq!(
                    delayed_reasoning.comment.as_deref(),
                    Some(["reasoning-source-comment".to_string()].as_slice())
                );
                assert_eq!(
                    delayed_data.nvext,
                    Some(serde_json::json!({"source": "reasoning"}))
                );
                assert_eq!(delayed_data.llm_metrics.as_ref().unwrap().chunk_tokens, 7);
                assert_eq!(delayed_data.inner.choices.len(), 1);
                assert!(delayed_data.inner.choices[0].delta.tool_calls.is_none());

                let continuation_answer = responses
                .iter()
                .find(|response| {
                    response.data.as_ref().is_some_and(|data| {
                        data.inner.choices.iter().any(|choice| {
                            matches!(&choice.delta.content, Some(ChatCompletionMessageContent::Text(text)) if text == "answer")
                        })
                    })
                })
                .expect("continuation answer should keep the resolving chunk envelope");
                let continuation_data = continuation_answer.data.as_ref().unwrap();
                assert_eq!(continuation_answer.id.as_deref(), Some("continuation-id"));
                assert_eq!(
                    continuation_answer.event.as_deref(),
                    Some("continuation-event")
                );
                assert_eq!(
                    continuation_answer.comment.as_deref(),
                    Some(["continuation-comment".to_string()].as_slice())
                );
                assert_eq!(
                    continuation_data.nvext,
                    Some(serde_json::json!({"source": "continuation"}))
                );
                let content_chunks: Vec<_> = responses
                    .iter()
                    .flat_map(|response| response.data.iter())
                    .flat_map(|data| data.inner.choices.iter())
                    .filter_map(|choice| match &choice.delta.content {
                        Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                assert!(
                    content_chunks.iter().all(|text| !text.contains("<tool_c")),
                    "a partial native opener must not leak into visible content: {content_chunks:?}"
                );
            }
        }
    }

    // Unified corpus input is raw model text; it cannot express upstream parsed
    // reasoning or source response annotations, whose ownership this adapter adds.
    #[tokio::test]
    async fn qwen_queued_sources_keep_identity_beside_mixed_and_raw_reasoning() {
        for finish in [false, true] {
            let mut inputs = vec![chunk("before <", false)];
            for (source_id, tokens) in [("first-source", 3), ("second-source", 5)] {
                let mut response = chunk("", false);
                let data = response.data.as_mut().unwrap();
                data.inner.choices[0].delta.content = None;
                data.inner.choices[0].delta.reasoning_content = Some("same".to_string());
                data.nvext = Some(serde_json::json!({"source": source_id}));
                data.llm_metrics = Some(crate::protocols::common::metrics::LLMMetricAnnotation {
                    chunk_tokens: tokens,
                    ..Default::default()
                });
                response.id = Some(source_id.to_string());
                inputs.push(response);
            }
            let mut continuation = chunk("<think>raw</think>answer", finish);
            let data = continuation.data.as_mut().unwrap();
            data.inner.choices[0].delta.reasoning_content = Some("mixed".to_string());
            data.nvext = Some(serde_json::json!({"source": "continuation"}));
            continuation.id = Some("continuation".to_string());
            inputs.push(continuation);
            let responses = apply_stream(
                stream::iter(inputs),
                Some(weather_tools()),
                None,
                false,
                UnifiedParserStartingState::None,
                QWEN3_UNIFIED_FAMILY,
            )
            .collect::<Vec<_>>()
            .await;
            for (source_id, tokens) in [("first-source", 3), ("second-source", 5)] {
                let sources: Vec<_> = responses
                    .iter()
                    .filter(|response| response.id.as_deref() == Some(source_id))
                    .collect();
                assert_eq!(sources.len(), 1);
                let data = sources[0].data.as_ref().unwrap();
                assert_eq!(
                    data.inner.choices[0].delta.reasoning_content.as_deref(),
                    Some("same")
                );
                assert!(data.inner.choices[0].delta.content.is_none());
                assert_eq!(data.llm_metrics.as_ref().unwrap().chunk_tokens, tokens);
                assert_eq!(data.nvext, Some(serde_json::json!({"source": source_id})));
            }
            let reasoning: Vec<_> = collect_choices(&responses)
                .into_iter()
                .filter_map(|choice| choice.delta.reasoning_content.clone())
                .collect();
            assert_eq!(reasoning, ["mixed", "same", "same", "raw"]);
            assert_eq!(
                responses
                    .iter()
                    .filter(|response| response.id.as_deref() == Some("continuation"))
                    .count(),
                1,
            );
            assert_eq!(
                collect_choices(&responses)
                    .iter()
                    .filter(|choice| choice.finish_reason.is_some())
                    .count(),
                usize::from(finish),
            );
        }
    }

    #[tokio::test]
    async fn qwen_eof_flushes_held_opener_before_source_annotated_reasoning() {
        let mut reasoning = chunk("", false);
        let choice = &mut reasoning.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("eof thought".to_string());
        let source = reasoning.data.as_mut().unwrap();
        source.nvext = Some(serde_json::json!({"source": "reasoning"}));
        source.llm_metrics = Some(crate::protocols::common::metrics::LLMMetricAnnotation {
            chunk_tokens: 9,
            ..Default::default()
        });
        reasoning.id = Some("eof-reasoning-id".to_string());
        reasoning.event = Some("eof-reasoning-event".to_string());
        reasoning.comment = Some(vec!["eof-reasoning-comment".to_string()]);

        let responses = apply_stream(
            stream::iter([chunk("before <tool_c", false), reasoning]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let ordered: Vec<_> = responses
            .iter()
            .filter_map(|response| {
                let choice = response.data.as_ref()?.inner.choices.first()?;
                if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                    Some(("text", text.as_str()))
                } else {
                    choice
                        .delta
                        .reasoning_content
                        .as_deref()
                        .map(|text| ("reasoning", text))
                }
            })
            .collect();
        assert_eq!(
            ordered,
            vec![
                ("text", "before "),
                ("text", "<tool_c"),
                ("reasoning", "eof thought"),
            ],
            "EOF must release each held byte once, before queued reasoning"
        );

        let delayed_reasoning = responses
            .iter()
            .find(|response| {
                response.data.as_ref().is_some_and(|data| {
                    data.inner.choices.iter().any(|choice| {
                        choice.delta.reasoning_content.as_deref() == Some("eof thought")
                    })
                })
            })
            .expect("EOF must release the queued reasoning chunk");
        let delayed_data = delayed_reasoning.data.as_ref().unwrap();
        assert_eq!(delayed_reasoning.id.as_deref(), Some("eof-reasoning-id"));
        assert_eq!(
            delayed_reasoning.event.as_deref(),
            Some("eof-reasoning-event")
        );
        assert_eq!(
            delayed_reasoning.comment.as_deref(),
            Some(["eof-reasoning-comment".to_string()].as_slice())
        );
        assert_eq!(
            delayed_data.nvext,
            Some(serde_json::json!({"source": "reasoning"}))
        );
        assert_eq!(delayed_data.llm_metrics.as_ref().unwrap().chunk_tokens, 9);
        assert_eq!(delayed_data.inner.choices.len(), 1);
        assert!(delayed_data.inner.choices[0].delta.tool_calls.is_none());

        let held_text_count = responses
            .iter()
            .flat_map(|response| response.data.iter())
            .flat_map(|data| data.inner.choices.iter())
            .filter(|choice| {
                matches!(&choice.delta.content, Some(ChatCompletionMessageContent::Text(text)) if text == "<tool_c")
            })
            .count();
        assert_eq!(held_text_count, 1);
    }

    #[tokio::test]
    async fn qwen_mismatched_opener_releases_source_reasoning_between_recovery_and_suffix() {
        let (input_tx, input_rx) = futures::channel::mpsc::unbounded();
        let mut output = Box::pin(apply_stream(
            input_rx,
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        ));

        input_tx
            .unbounded_send(chunk("before <tool_c", false))
            .unwrap();
        let prefix_response = output.next().await.expect("visible prefix should stream");
        let prefix_choice = &prefix_response.data.as_ref().unwrap().inner.choices[0];
        assert!(matches!(
            &prefix_choice.delta.content,
            Some(ChatCompletionMessageContent::Text(text)) if text == "before "
        ));

        let mut reasoning = chunk("", false);
        let choice = &mut reasoning.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("middle".to_string());
        let source = reasoning.data.as_mut().unwrap();
        source.nvext = Some(serde_json::json!({"source": "reasoning"}));
        source.llm_metrics = Some(crate::protocols::common::metrics::LLMMetricAnnotation {
            chunk_tokens: 5,
            ..Default::default()
        });
        reasoning.id = Some("mismatch-reasoning-id".to_string());
        reasoning.event = Some("mismatch-reasoning-event".to_string());
        reasoning.comment = Some(vec!["mismatch-reasoning-comment".to_string()]);
        input_tx.unbounded_send(reasoning).unwrap();

        assert!(
            output.next().now_or_never().is_none(),
            "reasoning must remain held while the opener is ambiguous"
        );

        let mut continuation = chunk("Xtail", true);
        let continuation_data = continuation.data.as_mut().unwrap();
        continuation_data.nvext = Some(serde_json::json!({"source": "continuation"}));
        continuation.id = Some("mismatch-continuation-id".to_string());
        continuation.event = Some("mismatch-continuation-event".to_string());
        continuation.comment = Some(vec!["mismatch-continuation-comment".to_string()]);
        input_tx.unbounded_send(continuation).unwrap();
        drop(input_tx);
        let mut responses = vec![prefix_response];
        responses.extend(output.collect::<Vec<_>>().await);

        let ordered: Vec<_> = responses
            .iter()
            .filter_map(|response| {
                let choice = response.data.as_ref()?.inner.choices.first()?;
                if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                    Some(("text", text.as_str()))
                } else {
                    choice
                        .delta
                        .reasoning_content
                        .as_deref()
                        .map(|text| ("reasoning", text))
                }
            })
            .collect();
        assert_eq!(
            ordered,
            vec![
                ("text", "before "),
                ("text", "<tool_c"),
                ("reasoning", "middle"),
                ("text", "Xtail"),
            ]
        );

        let delayed_reasoning = responses
            .iter()
            .find(|response| {
                response.data.as_ref().is_some_and(|data| {
                    data.inner
                        .choices
                        .iter()
                        .any(|choice| choice.delta.reasoning_content.as_deref() == Some("middle"))
                })
            })
            .unwrap();
        let delayed_data = delayed_reasoning.data.as_ref().unwrap();
        assert_eq!(
            delayed_reasoning.id.as_deref(),
            Some("mismatch-reasoning-id")
        );
        assert_eq!(
            delayed_reasoning.event.as_deref(),
            Some("mismatch-reasoning-event")
        );
        assert_eq!(
            delayed_reasoning.comment.as_deref(),
            Some(["mismatch-reasoning-comment".to_string()].as_slice())
        );
        assert_eq!(
            delayed_data.nvext,
            Some(serde_json::json!({"source": "reasoning"}))
        );
        assert_eq!(delayed_data.llm_metrics.as_ref().unwrap().chunk_tokens, 5);

        let suffix = responses
            .iter()
            .find(|response| {
                response.data.as_ref().is_some_and(|data| {
                    data.inner.choices.iter().any(|choice| {
                        matches!(&choice.delta.content, Some(ChatCompletionMessageContent::Text(text)) if text == "Xtail")
                    })
                })
            })
            .unwrap();
        assert_eq!(suffix.id.as_deref(), Some("mismatch-continuation-id"));
        assert_eq!(suffix.event.as_deref(), Some("mismatch-continuation-event"));
        assert_eq!(
            suffix.comment.as_deref(),
            Some(["mismatch-continuation-comment".to_string()].as_slice())
        );
        assert_eq!(
            suffix.data.as_ref().unwrap().nvext,
            Some(serde_json::json!({"source": "continuation"}))
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
    async fn truncated_named_guided_json_stays_pending_at_stop() {
        let responses = apply_stream(
            stream::iter([
                chunk("reason</think>{\"city\": ", false),
                chunk("\"Tokyo\"", true),
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
        assert!(
            choices
                .iter()
                .any(|choice| choice.delta.tool_calls.is_some()),
            "the guided parser may have already streamed the provisional call"
        );
        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::Stop],
            "an incomplete argument object must not be reported as a completed tool call"
        );
    }

    #[tokio::test]
    async fn named_guided_json_continues_across_reasoning_detour() {
        let mut reasoning_detour = chunk("", false);
        let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("interleaved thought".to_string());

        let responses = apply_stream(
            stream::iter([
                chunk("reason</think>{\"city\": ", false),
                reasoning_detour,
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
        let arguments: String = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        assert_eq!(arguments, "{\"city\": \"Tokyo\"}");
        assert!(
            choices.iter().any(|choice| {
                choice.delta.reasoning_content.as_deref() == Some("interleaved thought")
            }),
            "the reasoning detour must remain visible in its original position"
        );
        assert!(
            choices
                .iter()
                .any(|choice| choice.finish_reason == Some(FinishReason::ToolCalls)),
            "the resumed argument object must complete the same guided tool call"
        );
    }

    #[tokio::test]
    async fn native_qwen_call_continues_across_reasoning_detour() {
        let mut reasoning_detour = chunk("", false);
        let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("interleaved thought".to_string());

        let responses = apply_stream(
            stream::iter([
                chunk("<tool_call><function=get_weather><parameter=city>NY", false),
                reasoning_detour,
                chunk("C</parameter></function></tool_call>", true),
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
        let calls: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .collect();
        let arguments: String = calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        assert_eq!(arguments, r#"{"city":"NYC"}"#);
        assert!(calls.iter().any(|call| call.index == 0));
        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::ToolCalls]
        );
        let reasoning_position = choices
            .iter()
            .position(|choice| choice.delta.reasoning_content.is_some())
            .expect("the already-parsed reasoning delta must remain visible");
        assert!(
            choices[..reasoning_position]
                .iter()
                .any(|choice| choice.delta.tool_calls.is_some()),
            "the provisional native call must be released before the reasoning detour"
        );
        assert!(
            choices[reasoning_position + 1..]
                .iter()
                .any(|choice| choice.delta.tool_calls.is_some()),
            "the resumed native call fragment must follow the reasoning detour"
        );
    }

    #[tokio::test]
    async fn native_qwen_partial_call_opener_survives_reasoning_detour_at_every_split() {
        let opener = "<tool_call>";
        let body_and_answer = concat!(
            "\n<function=get_weather>\n",
            "<parameter=city>Paris</parameter>\n",
            "</function>\n</tool_call>answer"
        );

        for split in 1..opener.len() {
            let mut reasoning_detour = chunk("", false);
            let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some("interleaved thought".to_string());

            let continuation = format!("{}{}", &opener[split..], body_and_answer);
            let responses = apply_stream(
                stream::iter([
                    chunk(&opener[..split], false),
                    reasoning_detour,
                    chunk(&continuation, true),
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
            let calls: Vec<_> = choices
                .iter()
                .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
                .collect();
            let arguments: String = calls
                .iter()
                .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
                .collect();
            assert_eq!(arguments, r#"{"city":"Paris"}"#, "split {split}");
            assert_eq!(
                calls
                    .iter()
                    .filter_map(|call| call.function.as_ref()?.name.as_deref())
                    .collect::<String>(),
                "get_weather",
                "split {split}"
            );
            let mut indices: Vec<_> = calls.iter().map(|call| call.index).collect();
            indices.sort_unstable();
            indices.dedup();
            assert_eq!(indices, vec![0], "split {split}");
            assert_eq!(
                calls
                    .iter()
                    .filter_map(|call| call.id.as_deref())
                    .collect::<HashSet<_>>()
                    .len(),
                1,
                "split {split} must keep one call ID"
            );
            assert!(choices.iter().any(|choice| {
                choice.delta.reasoning_content.as_deref() == Some("interleaved thought")
            }));
            let content: String = choices
                .iter()
                .filter_map(|choice| match &choice.delta.content {
                    Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(content, "answer", "split {split}");
            assert_eq!(
                choices
                    .iter()
                    .filter_map(|choice| choice.finish_reason)
                    .collect::<Vec<_>>(),
                vec![FinishReason::ToolCalls],
                "split {split}"
            );

            let reasoning_position = choices
                .iter()
                .position(|choice| choice.delta.reasoning_content.is_some())
                .expect("reasoning detour must be emitted");
            let call_position = choices
                .iter()
                .position(|choice| choice.delta.tool_calls.is_some())
                .expect("tool call must follow the detour");
            let answer_position = choices
                .iter()
                .position(|choice| choice.delta.content.is_some())
                .expect("trailing answer must be emitted");
            assert!(
                reasoning_position < call_position && call_position < answer_position,
                "raw/parsed/raw order changed at split {split}"
            );
        }
    }

    #[tokio::test]
    async fn native_qwen_mismatched_partial_opener_orders_recovery_before_reasoning_at_every_split()
    {
        let opener = "<tool_call>";

        for split in 1..opener.len() {
            let mismatch_position = opener.len() - 1;
            let continuation = format!("{}Xpost", &opener[split..mismatch_position]);
            let mut reasoning_detour = chunk("", false);
            let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some("interleaved thought".to_string());

            let responses = apply_stream(
                stream::iter([
                    chunk(&opener[..split], false),
                    reasoning_detour,
                    chunk(&continuation, true),
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
                    if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                        Some(("content", text.as_str()))
                    } else {
                        choice
                            .delta
                            .reasoning_content
                            .as_deref()
                            .map(|text| ("reasoning", text))
                    }
                })
                .collect();

            assert_eq!(
                ordered,
                vec![
                    ("content", &opener[..split]),
                    ("reasoning", "interleaved thought"),
                    ("content", &continuation[..]),
                ],
                "mismatch at opener split {split}"
            );
        }
    }

    #[tokio::test]
    async fn native_qwen_partial_opener_eof_flushes_recovered_text_before_reasoning() {
        let mut reasoning_detour = chunk("", false);
        let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("interleaved thought".to_string());

        let responses = apply_stream(
            stream::iter([chunk("before <tool_c", false), reasoning_detour]),
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
                if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                    Some(("content", text.as_str()))
                } else {
                    choice
                        .delta
                        .reasoning_content
                        .as_deref()
                        .map(|text| ("reasoning", text))
                }
            })
            .collect();

        assert_eq!(
            ordered,
            vec![
                ("content", "before "),
                ("content", "<tool_c"),
                ("reasoning", "interleaved thought"),
            ]
        );
    }

    #[tokio::test]
    async fn native_qwen_buffered_non_string_argument_survives_reasoning_detour() {
        let mut reasoning_detour = chunk("", false);
        let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("interleaved thought".to_string());

        let responses = apply_stream(
            stream::iter([
                chunk("<tool_call><function=count_items><parameter=count>1", false),
                reasoning_detour,
                chunk("2</parameter></function></tool_call>", true),
            ]),
            Some(count_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        let calls: Vec<_> = choices
            .iter()
            .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
            .collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0]
                .function
                .as_ref()
                .and_then(|function| function.name.as_deref()),
            Some("count_items")
        );
        assert_eq!(
            calls[0]
                .function
                .as_ref()
                .and_then(|function| function.arguments.as_deref()),
            Some(r#"{"count":12}"#)
        );
        assert!(choices.iter().any(|choice| {
            choice.delta.reasoning_content.as_deref() == Some("interleaved thought")
        }));
        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::ToolCalls]
        );
    }

    #[tokio::test]
    async fn truncated_named_guided_json_stays_pending_across_reasoning_detour_at_eof() {
        let mut reasoning_detour = chunk("", false);
        let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("interleaved thought".to_string());

        let responses = apply_stream(
            stream::iter([chunk("reason</think>{\"city\": ", false), reasoning_detour]),
            Some(weather_tools()),
            Some(named_choice("get_weather")),
            false,
            UnifiedParserStartingState::Reasoning,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        assert!(
            choices
                .iter()
                .any(|choice| choice.delta.tool_calls.is_some()),
            "the cursor may have streamed provisional name and argument fragments"
        );
        assert!(
            choices
                .iter()
                .all(|choice| choice.finish_reason != Some(FinishReason::ToolCalls)),
            "EOF must not complete an invalid or truncated argument object"
        );
    }

    #[tokio::test]
    async fn bare_guided_json_under_prompt_injected_reasoning_still_becomes_a_tool_call() {
        // Regression for the guided-JSON-swallowed-as-reasoning bug: a Qwen3-Thinking
        // style template appends `<think>` to the generation prompt
        // (`prompt_injected_reasoning`, modeled here by starting the parser at
        // `Reasoning`), and guided decoding forbids the model from ever emitting
        // `</think>`. Unconditionally honoring the `Reasoning` prefill would make the
        // parser wait forever for a closer that will never arrive, classifying the
        // ENTIRE guided JSON payload as `reasoning_content` with zero tool calls
        // extracted. The payload here has no `reason</think>` prefix at all — its first
        // byte is `[` — so `bare_guided_json_prefill` must reclassify this choice's
        // starting state to `None` and let the array parse as a tool call instead.
        let responses = apply_stream(
            stream::iter([chunk(
                r#"[{"name":"get_weather","parameters":{"city":"Paris"}}]"#,
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
        assert!(
            choices.iter().all(|c| c.delta.reasoning_content.is_none()),
            "a bare guided JSON payload with no reasoning prefix must not be classified \
             as reasoning_content: {choices:?}"
        );
        let calls: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .collect();
        assert_eq!(
            calls
                .iter()
                .find_map(|call| call.function.as_ref()?.name.as_deref()),
            Some("get_weather"),
            "the guided JSON payload must still be recovered as a tool call: {choices:?}"
        );
        let arguments: String = calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        assert_eq!(arguments, r#"{"city":"Paris"}"#);
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

    /// Drive one guided-`required` stream whose payload closes only in the LAST
    /// upstream chunk, and report how many emitted responses carried a tool-call
    /// delta plus the assembled logical events.
    ///
    /// Anything counted above one is an argument fragment that went out before the
    /// payload closed, which is exactly the difference the guided tool-call
    /// streaming rollback lever is supposed to make. Nothing here touches process
    /// env, so it is safe under the default parallel test runner.
    async fn guided_required_tool_chunk_count(
        guided_streaming: bool,
    ) -> (usize, Vec<LogicalEvent>) {
        let head = r#"reason</think>[{"name":"get_weather","arguments":{"city":"Tok"#;
        let tail = r#"yo"}}]"#;
        let responses = apply_stream_with_constraint(
            stream::iter([chunk(head, false), chunk(tail, true)]),
            Some(weather_tools()),
            GuidedToolConstraint::GuidedJsonRequired,
            UnifiedParserStartingState::Reasoning,
            QWEN3_UNIFIED_FAMILY,
            guided_streaming,
        )
        .collect::<Vec<_>>()
        .await;
        let tool_chunks = responses
            .iter()
            .filter(|response| {
                response.data.as_ref().is_some_and(|data| {
                    data.inner.choices.iter().any(|choice| {
                        choice
                            .delta
                            .tool_calls
                            .as_ref()
                            .is_some_and(|calls| !calls.is_empty())
                    })
                })
            })
            .count();
        (tool_chunks, logical_events(&responses))
    }

    #[tokio::test]
    async fn guided_streaming_rollback_buffers_the_call_to_completion() {
        let expected_call = LogicalEvent::Tool {
            index: 0,
            name: Some("get_weather".to_string()),
            arguments: r#"{"city":"Tokyo"}"#.to_string(),
        };

        // Released (production default when DYN_ENABLE_GUIDED_TOOL_STREAMING is unset):
        // the call goes out in pieces as the argument object streams in.
        let (released_chunks, released_events) = guided_required_tool_chunk_count(true).await;
        assert!(
            released_chunks > 1,
            "guided streaming must emit argument fragments before the payload closes, \
             got {released_chunks} tool-call chunk(s)"
        );

        // Rolled back (DYN_ENABLE_GUIDED_TOOL_STREAMING=0): nothing may go out until the
        // payload closes, so the whole call arrives as one chunk.
        let (rolled_back_chunks, rolled_back_events) =
            guided_required_tool_chunk_count(false).await;
        assert_eq!(
            rolled_back_chunks, 1,
            "DYN_ENABLE_GUIDED_TOOL_STREAMING=0 must buffer the guided call to completion \
             on the unified path, but {rolled_back_chunks} tool-call chunk(s) went out"
        );

        // The lever changes WHEN the call is emitted, never WHAT is emitted.
        assert!(
            released_events.contains(&expected_call),
            "released events lost the call: {released_events:?}"
        );
        assert!(
            rolled_back_events.contains(&expected_call),
            "rolled-back events lost the call: {rolled_back_events:?}"
        );
    }

    #[test]
    fn only_stream_best_effort_may_emit_before_the_payload_closes() {
        // The two buffering contracts are interchangeable for WHEN, but not for WHAT a
        // malformed payload becomes, so the rollback must pick `RecoverAsText` (text)
        // and not `Reject` (typed error).
        assert_eq!(
            invalid_guided_payload_policy(true),
            InvalidGuidedPayloadPolicy::StreamBestEffort
        );
        assert_eq!(
            invalid_guided_payload_policy(false),
            InvalidGuidedPayloadPolicy::RecoverAsText
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
    async fn resumed_raw_tool_call_gets_a_new_index_after_an_already_parsed_detour() {
        let first_partial = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Paris</parameter>"
        );
        let already_parsed = {
            let mut value = chunk("", false);
            let choice = &mut value.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some("already parsed".to_string());
            value
        };
        let second_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let responses = apply_stream(
            stream::iter([
                chunk(first_partial, false),
                already_parsed,
                chunk(second_call, true),
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
        let calls: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .collect();
        assert!(calls.iter().any(|call| call.index == 0));
        assert!(calls.iter().any(|call| call.index == 1));
        let first_ids: HashSet<_> = calls
            .iter()
            .filter(|call| call.index == 0)
            .filter_map(|call| call.id.as_deref())
            .collect();
        let second_ids: HashSet<_> = calls
            .iter()
            .filter(|call| call.index == 1)
            .filter_map(|call| call.id.as_deref())
            .collect();
        assert_eq!(
            first_ids.len(),
            1,
            "the first call identity must stay stable"
        );
        assert_eq!(second_ids.len(), 1, "the resumed call needs one stable ID");
        assert_ne!(first_ids, second_ids, "separate calls must not reuse an ID");
        let names: Vec<_> = calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.name.as_deref())
            .collect();
        assert_eq!(names, vec!["get_weather", "get_weather"]);
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
    async fn already_parsed_detour_allocates_index_after_parser_index_collision() {
        let raw_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let mut upstream_call = chunk("", false);
        let choice = &mut upstream_call.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("upstream-call".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("upstream".to_string()),
                arguments: Some("{\"city\":\"Paris\"}".to_string()),
            }),
        }]);

        let mut continuation = chunk("", false);
        let choice = &mut continuation.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: None,
            r#type: None,
            function: Some(FunctionCallStream {
                name: None,
                arguments: Some("more".to_string()),
            }),
        }]);

        let responses = apply_stream(
            stream::iter([chunk(raw_call, false), upstream_call, continuation]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
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
        let parser_call = calls
            .iter()
            .find(|call| {
                call.function
                    .as_ref()
                    .and_then(|function| function.name.as_deref())
                    == Some("get_weather")
            })
            .expect("raw parser call must be emitted");
        let upstream_open = calls
            .iter()
            .find(|call| {
                call.function
                    .as_ref()
                    .and_then(|function| function.name.as_deref())
                    == Some("upstream")
            })
            .expect("the first upstream call chunk must keep its name");
        assert_eq!(parser_call.index, 0);
        assert_eq!(upstream_open.index, 1);
        assert_eq!(upstream_open.id.as_deref(), Some("upstream-call"));
        let passthrough_continuation = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .find(|call| {
                call.function
                    .as_ref()
                    .and_then(|function| function.arguments.as_deref())
                    == Some("more")
            })
            .expect("upstream continuation must be emitted");

        assert_eq!(
            passthrough_continuation.index, 1,
            "continuation must retain the upstream call's remapped index"
        );
        assert_eq!(passthrough_continuation.id, None);
        assert_eq!(passthrough_continuation.r#type, None);
        assert_eq!(
            passthrough_continuation
                .function
                .as_ref()
                .and_then(|function| function.name.as_deref()),
            None
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
    async fn raw_preparsed_raw_resumes_at_response_not_original_prefill() {
        // Request-level prefill is `Reasoning` (prompt-injected reasoning): the first
        // raw run closes reasoning mid-run. An already-parsed chunk then detours the
        // choice, dropping its live `ChoiceState`. The next raw chunk must NOT rebuild
        // starting at `Reasoning` again — an already-parsed chunk implies the model has
        // already emitted structured output, which only happens after any reasoning
        // phase concluded, so the resumed run's text belongs to `content`.
        let mut pre_parsed = chunk("", false);
        let choice = &mut pre_parsed.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("already parsed".to_string());

        let responses = apply_stream(
            stream::iter([
                chunk("secret</think>visible-before-gap ", false),
                pre_parsed,
                chunk("visible-after-gap", true),
            ]),
            None,
            None,
            false,
            UnifiedParserStartingState::Reasoning,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        let reasoning: String = choices
            .iter()
            .filter_map(|choice| choice.delta.reasoning_content.as_deref())
            .collect();
        let visible: String = choices
            .iter()
            .filter_map(|choice| match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(
            visible, "visible-before-gap visible-after-gap",
            "text after the already-parsed gap must resume as content, not reasoning"
        );
        assert_eq!(reasoning, "secretalready parsed");
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
    async fn late_tool_metadata_and_fragmented_name_survive_index_open() {
        fn call_chunk(
            id: Option<&str>,
            call_type: Option<FunctionType>,
            name: Option<&str>,
            arguments: Option<&str>,
            finish: Option<FinishReason>,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            let mut response = chunk("", false);
            let choice = &mut response.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
                index: 0,
                id: id.map(str::to_string),
                r#type: call_type,
                function: Some(FunctionCallStream {
                    name: name.map(str::to_string),
                    arguments: arguments.map(str::to_string),
                }),
            }]);
            choice.finish_reason = finish;
            response
        }

        let responses = apply_stream(
            stream::iter([
                call_chunk(None, None, None, Some(""), None),
                call_chunk(
                    Some("call-late"),
                    Some(FunctionType::Function),
                    Some("get_"),
                    None,
                    None,
                ),
                call_chunk(
                    None,
                    None,
                    Some("weather"),
                    Some("{}"),
                    Some(FinishReason::ToolCalls),
                ),
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
        let calls: Vec<_> = choices
            .iter()
            .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
            .collect();

        assert_eq!(
            calls
                .iter()
                .filter_map(|call| call.id.as_deref())
                .collect::<Vec<_>>(),
            vec!["call-late"],
            "an ID supplied after its tool index opens must be forwarded once"
        );
        assert_eq!(
            calls
                .iter()
                .filter_map(|call| call.r#type.clone())
                .collect::<Vec<_>>(),
            vec![FunctionType::Function],
            "a type supplied after its tool index opens must be forwarded once"
        );
        assert_eq!(
            calls
                .iter()
                .filter_map(|call| call.function.as_ref()?.name.as_deref())
                .collect::<Vec<_>>(),
            vec!["get_", "weather"],
            "name fragments must remain separate deltas so clients can reassemble them"
        );
        assert_eq!(
            calls
                .iter()
                .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
                .collect::<Vec<_>>(),
            vec!["", "{}"]
        );
    }

    #[tokio::test]
    async fn late_name_fragments_survive_raw_content_and_reasoning_detour() {
        let mut first = chunk("visible", false);
        first.data.as_mut().unwrap().inner.choices[0]
            .delta
            .tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call-once".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_".to_string()),
                arguments: Some(r#"{"city":"Pa"#.to_string()),
            }),
        }]);
        let mut gap = chunk("", false);
        let gap_choice = &mut gap.data.as_mut().unwrap().inner.choices[0];
        gap_choice.delta.content = None;
        gap_choice.delta.reasoning_content = Some("reasoning gap".to_string());

        let mut resumed = chunk("", true);
        let resumed_choice = &mut resumed.data.as_mut().unwrap().inner.choices[0];
        resumed_choice.delta.content = None;
        resumed_choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("duplicate-id".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("weather".to_string()),
                arguments: Some("ris\"}".to_string()),
            }),
        }]);
        resumed_choice.finish_reason = Some(FinishReason::ToolCalls);

        let responses = apply_stream(
            stream::iter([first, gap, resumed]),
            None,
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        let calls: Vec<_> = choices
            .iter()
            .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
            .collect();

        assert_eq!(
            calls
                .iter()
                .filter_map(|call| call.id.as_deref())
                .collect::<Vec<_>>(),
            vec!["call-once"],
            "an already emitted ID must not be repeated after a detour"
        );
        assert_eq!(
            calls
                .iter()
                .filter_map(|call| call.r#type.clone())
                .collect::<Vec<_>>(),
            vec![FunctionType::Function],
            "an already emitted type must not be repeated after a detour"
        );
        assert_eq!(
            calls
                .iter()
                .filter_map(|call| call.function.as_ref()?.name.as_deref())
                .collect::<Vec<_>>(),
            vec!["get_", "weather"],
            "a later name fragment must survive the parsed detour"
        );
        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::ToolCalls]
        );
    }

    #[tokio::test]
    async fn incomplete_upstream_call_stays_pending_across_reasoning_detour_at_eof() {
        let mut first = chunk("visible", false);
        first.data.as_mut().unwrap().inner.choices[0]
            .delta
            .tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call-pending".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Pa"#.to_string()),
            }),
        }]);
        let mut gap = chunk("", false);
        let gap_choice = &mut gap.data.as_mut().unwrap().inner.choices[0];
        gap_choice.delta.content = None;
        gap_choice.delta.reasoning_content = Some("reasoning gap".to_string());

        let responses = apply_stream(
            stream::iter([first, gap]),
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
            choices.iter().any(|choice| choice
                .delta
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())),
            "the partial upstream tool-call delta must remain visible"
        );
        assert!(
            choices.iter().all(|choice| choice.finish_reason.is_none()),
            "EOF must not claim tool-call completion when upstream arguments remain partial"
        );
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn legacy_function_call_choice_gets_no_synthetic_terminal_after_a_gap() {
        // The legacy `function_call` field is a distinct signal from `tool_calls`
        // (it has its own `FinishReason::FunctionCall`, never `ToolCalls`) and, like
        // plain text, is never treated as a completed tool call anywhere in this file
        // (only complete parser events and terminal parsed deltas are). An
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
        // chunk. The state's own completed-call state is false, but the record must still
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
    async fn terminal_already_parsed_call_keeps_index_separate_from_raw_call() {
        let raw_call = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        );
        let mut terminal = chunk("", true);
        let choice = &mut terminal.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call-complete".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("upstream".to_string()),
                arguments: Some("{\"city\":\"Paris\"}".to_string()),
            }),
        }]);

        let responses = apply_stream(
            stream::iter([chunk(raw_call, false), terminal]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        let tool_call_chunks: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .collect();
        assert_eq!(tool_call_chunks.len(), 2);
        assert_eq!(tool_call_chunks[0].index, 0);
        assert!(tool_call_chunks[0].id.is_some());
        assert_eq!(tool_call_chunks[1].index, 1);
        assert_eq!(tool_call_chunks[1].id.as_deref(), Some("call-complete"));
        assert_eq!(
            tool_call_chunks[1]
                .function
                .as_ref()
                .and_then(|function| function.name.as_deref()),
            Some("upstream")
        );
        let finish_reasons: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect();
        assert_eq!(
            finish_reasons,
            vec![FinishReason::ToolCalls],
            "the already-parsed completion must clear the matching raw provisional call"
        );
    }

    #[tokio::test]
    async fn finish_recovery_during_reasoning_detour_clears_pending_tool_call() {
        let complete_json_without_call_end = concat!(
            "<|tool_calls_section_begin|><|tool_call_begin|>",
            "functions.get_weather:0<|tool_call_argument_begin|>",
            r#"{"city":"Paris"}"#
        );
        for terminal in [false, true] {
            let mut parsed_reasoning = chunk("", terminal);
            let choice = &mut parsed_reasoning.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some("already parsed reasoning".to_string());

            let responses = apply_stream(
                stream::iter([
                    chunk(complete_json_without_call_end, false),
                    parsed_reasoning,
                ]),
                Some(weather_tools()),
                None,
                false,
                UnifiedParserStartingState::None,
                "kimi_k2",
            )
            .collect::<Vec<_>>()
            .await;
            let choices = collect_choices(&responses);

            assert!(
                choices
                    .iter()
                    .any(|choice| choice.delta.tool_calls.is_some()),
                "finish must recover the valid JSON body when Kimi omits the call-end marker"
            );
            assert_eq!(
                choices
                    .iter()
                    .filter_map(|choice| choice.finish_reason)
                    .collect::<Vec<_>>(),
                vec![FinishReason::ToolCalls],
                "finish-recovered calls must clear the pending index before EOF (terminal={terminal})"
            );
        }
    }

    #[tokio::test]
    async fn qwen_call_start_marker_inside_parameter_value_survives_reasoning_detour_at_every_split()
     {
        async fn run(
            split: usize,
            continuation: &str,
        ) -> (
            String,
            String,
            bool,
            String,
            Vec<FinishReason>,
            Vec<u32>,
            usize,
            bool,
        ) {
            let mut reasoning_detour = chunk("", false);
            let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some("interleaved thought".to_string());

            let responses = apply_stream(
                stream::iter([
                    chunk("<tool_call><function=get_weather><parameter=city>Pa", false),
                    reasoning_detour,
                    chunk(&continuation[..split], false),
                    chunk(&continuation[split..], true),
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
            let calls: Vec<_> = choices
                .iter()
                .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
                .collect();
            let arguments = calls
                .iter()
                .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
                .collect();
            let name = calls
                .iter()
                .filter_map(|call| call.function.as_ref()?.name.as_deref())
                .collect();
            let reasoning = choices.iter().any(|choice| {
                choice.delta.reasoning_content.as_deref() == Some("interleaved thought")
            });
            let content = choices
                .iter()
                .filter_map(|choice| match &choice.delta.content {
                    Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            let finish_reasons = choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect();
            let mut indices: Vec<_> = calls.iter().map(|call| call.index).collect();
            indices.sort_unstable();
            indices.dedup();
            let call_ids: HashSet<_> = calls.iter().filter_map(|call| call.id.as_deref()).collect();
            let reasoning_position = choices
                .iter()
                .position(|choice| choice.delta.reasoning_content.is_some())
                .expect("the reasoning detour must be emitted");
            let streamed_before_reasoning = choices[..reasoning_position].iter().any(|choice| {
                choice.delta.tool_calls.as_ref().is_some_and(|calls| {
                    calls.iter().any(|call| {
                        call.function
                            .as_ref()
                            .and_then(|function| function.arguments.as_deref())
                            .is_some_and(|arguments| !arguments.is_empty())
                    })
                })
            });
            (
                arguments,
                name,
                reasoning,
                content,
                finish_reasons,
                indices,
                call_ids.len(),
                streamed_before_reasoning,
            )
        }

        let continuation = "<tool_call>after</parameter></function></tool_call>";
        let whole = run(continuation.len(), continuation).await;
        assert_eq!(whole.0, r#"{"city":"Pa<tool_call>after"}"#);
        assert_eq!(whole.1, "get_weather");
        assert!(whole.2, "the reasoning detour must stay visible");
        assert_eq!(whole.3, "", "native markup must not leak into content");
        assert_eq!(whole.4, vec![FinishReason::ToolCalls]);
        assert_eq!(whole.5, vec![0]);
        assert_eq!(whole.6, 1, "exactly one call ID must be emitted");
        assert!(whole.7, "argument bytes must stream before the detour");

        for split in 0..=continuation.len() {
            assert_eq!(
                run(split, continuation).await,
                whole,
                "splitting the continuation at byte {split} changed the result"
            );
        }
    }

    #[tokio::test]
    async fn kimi_call_start_marker_inside_json_string_survives_reasoning_detour_at_every_split() {
        async fn run(
            split: usize,
            continuation: &str,
        ) -> (String, bool, String, Vec<FinishReason>) {
            let mut reasoning_detour = chunk("", false);
            let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some("interleaved thought".to_string());

            let responses = apply_stream(
                stream::iter([
                    chunk(
                        "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\":\"Pa",
                        false,
                    ),
                    reasoning_detour,
                    chunk(&continuation[..split], false),
                    chunk(&continuation[split..], true),
                ]),
                Some(weather_tools()),
                None,
                false,
                UnifiedParserStartingState::None,
                "kimi_k2",
            )
            .collect::<Vec<_>>()
            .await;

            let choices = collect_choices(&responses);
            let arguments = choices
                .iter()
                .filter_map(|choice| choice.delta.tool_calls.as_ref())
                .flatten()
                .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
                .collect();
            let reasoning = choices.iter().any(|choice| {
                choice.delta.reasoning_content.as_deref() == Some("interleaved thought")
            });
            let content = choices
                .iter()
                .filter_map(|choice| match &choice.delta.content {
                    Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            let finish_reasons = choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect();
            (arguments, reasoning, content, finish_reasons)
        }

        let continuation = "<|tool_call_begin|>after\"}<|tool_call_end|><|tool_calls_section_end|>";
        let whole = run(continuation.len(), continuation).await;
        assert_eq!(whole.0, "{\"city\":\"Pa<|tool_call_begin|>after\"}");
        assert!(whole.1, "the reasoning detour must stay visible");
        assert_eq!(whole.2, "", "no native markup may leak into content");
        assert_eq!(whole.3, vec![FinishReason::ToolCalls]);

        for split in 0..=continuation.len() {
            assert_eq!(
                run(split, continuation).await,
                whole,
                "splitting the continuation at byte {split} changed the result"
            );
        }
    }

    #[tokio::test]
    async fn nonterminal_already_parsed_tool_fragment_does_not_complete_pending_call() {
        let first_partial = concat!(
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Paris"
        );
        let mut parsed_fragment = chunk("", false);
        let choice = &mut parsed_fragment.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call-parsed".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Paris""#.to_string()),
            }),
        }]);

        let responses = apply_stream(
            stream::iter([chunk(first_partial, false), parsed_fragment]),
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
            "both the raw and already-parsed partial fragments should be emitted"
        );
        assert!(
            choices.iter().all(|choice| choice.finish_reason.is_none()),
            "nonterminal parsed argument fragments must not synthesize a completed-call terminal"
        );
    }

    #[tokio::test]
    async fn qwen_queued_reasoning_is_emitted_once_when_parsed_tool_call_interrupts() {
        #[derive(Debug, PartialEq)]
        enum OutputEvent {
            Content(String),
            Reasoning(String),
            ToolCall {
                index: u32,
                id: Option<String>,
                name: Option<String>,
                arguments: Option<String>,
            },
            Finish(FinishReason),
        }

        for terminal in [false, true] {
            let mut reasoning = chunk("", false);
            let choice = &mut reasoning.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some("held thought".to_string());
            let source = reasoning.data.as_mut().unwrap();
            source.nvext = Some(serde_json::json!({"source": "reasoning"}));
            source.llm_metrics = Some(crate::protocols::common::metrics::LLMMetricAnnotation {
                chunk_tokens: 7,
                ..Default::default()
            });
            reasoning.id = Some("reasoning-source-id".to_string());
            reasoning.event = Some("reasoning-source-event".to_string());
            reasoning.comment = Some(vec!["reasoning-source-comment".to_string()]);

            let mut parsed_call = chunk("", terminal);
            let choice = &mut parsed_call.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some("parsed-call".to_string()),
                r#type: Some(FunctionType::Function),
                function: Some(FunctionCallStream {
                    name: Some("get_weather".to_string()),
                    arguments: Some(r#"{"city":"NY"}"#.to_string()),
                }),
            }]);

            let responses = apply_stream(
                stream::iter([chunk("before <tool_c", false), reasoning, parsed_call]),
                Some(weather_tools()),
                None,
                false,
                UnifiedParserStartingState::None,
                QWEN3_UNIFIED_FAMILY,
            )
            .collect::<Vec<_>>()
            .await;
            let choices = collect_choices(&responses);
            let observed: Vec<_> = choices
                .iter()
                .flat_map(|choice| {
                    let mut events = Vec::new();
                    if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                        events.push(OutputEvent::Content(text.clone()));
                    }
                    if let Some(reasoning) = &choice.delta.reasoning_content {
                        events.push(OutputEvent::Reasoning(reasoning.clone()));
                    }
                    if let Some(calls) = &choice.delta.tool_calls {
                        events.extend(calls.iter().map(|call| {
                            let function = call.function.as_ref();
                            OutputEvent::ToolCall {
                                index: call.index,
                                id: call.id.clone(),
                                name: function.and_then(|function| function.name.clone()),
                                arguments: function.and_then(|function| function.arguments.clone()),
                            }
                        }));
                    }
                    if let Some(reason) = &choice.finish_reason {
                        events.push(OutputEvent::Finish(reason.clone()));
                    }
                    events
                })
                .collect();
            let mut expected = vec![
                OutputEvent::Content("before ".to_string()),
                OutputEvent::Content("<tool_c".to_string()),
                OutputEvent::Reasoning("held thought".to_string()),
                OutputEvent::ToolCall {
                    index: 0,
                    id: Some("parsed-call".to_string()),
                    name: Some("get_weather".to_string()),
                    arguments: Some(r#"{"city":"NY"}"#.to_string()),
                },
            ];
            if terminal {
                expected.push(OutputEvent::Finish(FinishReason::ToolCalls));
            }
            assert_eq!(
                observed, expected,
                "content, held reasoning, the parsed call, and terminal metadata must stay ordered and appear once"
            );
            let source_reasoning: Vec<_> = responses
                .iter()
                .filter(|response| {
                    response.data.as_ref().is_some_and(|data| {
                        data.inner.choices.iter().any(|choice| {
                            choice.delta.reasoning_content.as_deref() == Some("held thought")
                        })
                    })
                })
                .collect();
            assert_eq!(source_reasoning.len(), 1);
            assert_eq!(
                source_reasoning[0].id.as_deref(),
                Some("reasoning-source-id")
            );
            assert_eq!(
                source_reasoning[0].event.as_deref(),
                Some("reasoning-source-event")
            );
            assert_eq!(
                source_reasoning[0].comment.as_deref(),
                Some(["reasoning-source-comment".to_string()].as_slice())
            );
            let source_data = source_reasoning[0].data.as_ref().unwrap();
            assert_eq!(
                source_data.nvext,
                Some(serde_json::json!({"source": "reasoning"}))
            );
            assert_eq!(source_data.llm_metrics.as_ref().unwrap().chunk_tokens, 7);
        }
    }

    #[tokio::test]
    async fn qwen_multi_choice_queued_reasoning_emits_source_metadata_once() {
        fn choice(index: u32, content: Option<&str>, reasoning: Option<&str>) -> ChatChoiceStream {
            ChatChoiceStream {
                index,
                delta: ChatCompletionStreamResponseDelta {
                    role: Some(Role::Assistant),
                    content: content
                        .map(|text| ChatCompletionMessageContent::Text(text.to_string())),
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: reasoning.map(str::to_string),
                },
                finish_reason: None,
                logprobs: None,
            }
        }

        for (reasoning_zero, reasoning_one) in [
            ("thought zero", "thought one"),
            ("", ""),
            ("", "thought one"),
            ("thought zero", ""),
        ] {
            let mut opening = chunk("", false);
            opening.data.as_mut().unwrap().inner.choices = vec![
                choice(0, Some("before <tool_c"), None),
                choice(1, Some("before <tool_c"), None),
            ];
            let mut reasoning = chunk("", false);
            let data = reasoning.data.as_mut().unwrap();
            data.inner.choices = vec![
                choice(0, None, Some(reasoning_zero)),
                choice(1, None, Some(reasoning_one)),
            ];
            data.nvext = Some(serde_json::json!({"source": "reasoning"}));
            data.llm_metrics = Some(crate::protocols::common::metrics::LLMMetricAnnotation {
                chunk_tokens: 11,
                ..Default::default()
            });
            reasoning.id = Some("shared-reasoning-id".to_string());
            reasoning.event = Some("shared-reasoning-event".to_string());
            reasoning.comment = Some(vec!["shared-reasoning-comment".to_string()]);

            let mut continuation = chunk("", true);
            continuation.data.as_mut().unwrap().inner.choices = vec![
                choice(
                    0,
                    Some(
                        "all>\n<function=get_weather>\n<parameter=city>NY</parameter>\n</function>\n</tool_call>",
                    ),
                    None,
                ),
                choice(
                    1,
                    Some(
                        "all>\n<function=get_weather>\n<parameter=city>SF</parameter>\n</function>\n</tool_call>",
                    ),
                    None,
                ),
            ];
            let responses = apply_stream(
                stream::iter([opening, reasoning, continuation]),
                Some(weather_tools()),
                None,
                false,
                UnifiedParserStartingState::None,
                QWEN3_UNIFIED_FAMILY,
            )
            .collect::<Vec<_>>()
            .await;

            let choices = collect_choices(&responses);
            for (index, expected) in [(0, reasoning_zero), (1, reasoning_one)] {
                assert_eq!(
                    choices
                        .iter()
                        .filter(|choice| {
                            choice.index == index
                                && choice.delta.reasoning_content.as_deref() == Some(expected)
                        })
                        .count(),
                    1,
                    "choice {index} reasoning must remain attached to the correct choice once"
                );
            }
            assert_eq!(
                responses
                    .iter()
                    .filter(|response| response.id.as_deref() == Some("shared-reasoning-id"))
                    .count(),
                1,
                "the source response id must be emitted once across the two queued choices"
            );
            assert_eq!(
                responses
                    .iter()
                    .filter(|response| response.event.as_deref() == Some("shared-reasoning-event"))
                    .count(),
                1
            );
            assert_eq!(
                responses
                    .iter()
                    .filter(|response| response.comment.is_some())
                    .count(),
                1
            );
            assert_eq!(
                responses
                    .iter()
                    .filter_map(|response| response.data.as_ref())
                    .filter(|data| data.nvext.is_some())
                    .count(),
                1
            );
            assert_eq!(
                responses
                    .iter()
                    .filter_map(|response| response.data.as_ref())
                    .filter(|data| data.llm_metrics.is_some())
                    .count(),
                1
            );
        }
    }

    #[tokio::test]
    async fn separate_terminal_chunk_closes_parsed_fragments_for_all_indices() {
        for tool_calls_finish in [false, true] {
            let mut parsed_fragments = chunk("", false);
            let choice = &mut parsed_fragments.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.tool_calls = Some(vec![
                ChatCompletionMessageToolCallChunk {
                    index: 0,
                    id: Some("call-first".to_string()),
                    r#type: Some(FunctionType::Function),
                    function: Some(FunctionCallStream {
                        name: Some("get_weather".to_string()),
                        arguments: Some(r#"{"city":"Par"#.to_string()),
                    }),
                },
                ChatCompletionMessageToolCallChunk {
                    index: 1,
                    id: Some("call-second".to_string()),
                    r#type: Some(FunctionType::Function),
                    function: Some(FunctionCallStream {
                        name: Some("get_weather".to_string()),
                        arguments: Some(r#"{"city":"Tok"#.to_string()),
                    }),
                },
            ]);

            let mut terminal = chunk("", false);
            let choice = &mut terminal.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some("terminal reasoning".to_string());
            choice.finish_reason = Some(if tool_calls_finish {
                FinishReason::ToolCalls
            } else {
                FinishReason::Stop
            });

            let responses = apply_stream(
                stream::iter([parsed_fragments, terminal]),
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
                "a separate terminal reason must close all parsed fragments (tool_calls_finish={tool_calls_finish})"
            );
        }
    }

    #[tokio::test]
    async fn already_parsed_first_terminal_chunk_normalizes_stop() {
        let mut already_parsed = chunk("", true);
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
        // constructed for it — but the parsed chunk carries the terminal Stop signal.
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
            "an already-parsed tool call on the terminal chunk must normalize Stop to ToolCalls"
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
        // raw-resumed (a fresh `ChoiceState`, completed-call state reseeded from the
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
            tool_output_mode: UnifiedToolOutputMode::Native,
            qwen_native_lexical_state: QwenNativeLexicalState::default(),
            pending_qwen_opener_text: String::new(),
            pending_qwen_opener_reasoning: Vec::new(),
            qwen_reasoning_event_positions: HashSet::new(),
            qwen_reasoning_choice_positions: HashSet::new(),
            opened_calls: HashSet::new(),
            tool_ids_emitted: HashSet::new(),
            tool_types_emitted: HashSet::new(),
            parser_tool_indices: HashMap::new(),
            passthrough_tool_indices: HashMap::new(),
            tool_index_offset: 0,
            next_tool_index: 0,
            pending_tool_calls: HashSet::new(),
            pending_parsed_tool_calls: HashSet::new(),
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
            &[],
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
            &[],
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
            &[],
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
            &[],
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
            &[],
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
        let parsed = parse_complete(
            QWEN3_UNIFIED_FAMILY,
            literal,
            &GuidedToolConstraint::None,
            &[],
        )
        .unwrap();
        assert_eq!(parsed.text, literal);
        assert!(parsed.reasoning.is_empty());

        let embedded = "The string \"a </think> token\" stays visible.";
        let parsed = parse_complete(
            QWEN3_UNIFIED_FAMILY,
            embedded,
            &GuidedToolConstraint::None,
            &[],
        )
        .unwrap();
        assert_eq!(parsed.text, embedded);
        assert!(parsed.reasoning.is_empty());
    }

    #[test]
    fn plain_text_passes_through_the_batch_path_unchanged() {
        let parsed = parse_complete(
            QWEN3_UNIFIED_FAMILY,
            "just an answer",
            &GuidedToolConstraint::None,
            &[],
        )
        .unwrap();
        assert_eq!(parsed.text, "just an answer");
        assert!(parsed.reasoning.is_empty());
        assert!(parsed.tool_calls.is_empty());
    }

    // Regression for a batch/streaming argument-type divergence: `parse_complete`
    // used to hardcode an empty tool slice, so it had no schema to type-coerce
    // arguments against and always emitted them as JSON strings. The streaming
    // path (`apply_stream_with_constraint`) builds its parser from the request's
    // real `tool_definitions` and correctly coerces `count` to an integer for the
    // same input. Passing the real tool schema through here must match that.
    #[test]
    fn batch_path_coerces_argument_types_from_the_real_tool_schema() {
        let tool_definitions = vec![ToolDefinition {
            name: "set_count".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            })),
            strict: None,
        }];

        let parsed = parse_complete(
            QWEN3_UNIFIED_FAMILY,
            concat!(
                "<tool_call>\n<function=set_count>\n",
                "<parameter=count>42</parameter>\n</function>\n</tool_call>"
            ),
            &GuidedToolConstraint::None,
            &tool_definitions,
        )
        .unwrap();

        assert_eq!(
            parsed.tool_calls[0].function.arguments, r#"{"count":42}"#,
            "batch path must type-coerce `count` to an integer using the real tool \
             schema, matching what the streaming path already produces for the same \
             input"
        );
    }
    #[tokio::test]
    async fn qwen_reasoning_detours_preserve_native_grammar_at_every_utf8_split() {
        async fn observe(
            input: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
            tools: Vec<ToolDefinition>,
        ) -> (
            String,
            Vec<(u32, String, String)>,
            Vec<FinishReason>,
            String,
        ) {
            let responses = apply_stream(
                stream::iter(input),
                Some(tools),
                None,
                false,
                UnifiedParserStartingState::None,
                QWEN3_UNIFIED_FAMILY,
            )
            .collect::<Vec<_>>()
            .await;
            let choices = collect_choices(&responses);
            let mut calls = std::collections::BTreeMap::<u32, (String, String)>::new();
            let mut text = String::new();
            let mut reasoning = String::new();
            let mut finishes = Vec::new();
            for choice in choices {
                if let Some(ChatCompletionMessageContent::Text(value)) = &choice.delta.content {
                    text.push_str(&value);
                }
                if let Some(value) = &choice.delta.reasoning_content {
                    reasoning.push_str(&value);
                }
                if let Some(value) = choice.finish_reason {
                    finishes.push(value);
                }
                for call in choice.delta.tool_calls.iter().flatten() {
                    let entry = calls.entry(call.index).or_default();
                    if let Some(function) = &call.function {
                        if let Some(name) = &function.name {
                            entry.0.push_str(&name);
                        }
                        if let Some(arguments) = &function.arguments {
                            entry.1.push_str(&arguments);
                        }
                    }
                }
            }
            (
                text,
                calls
                    .into_iter()
                    .map(|(index, (name, arguments))| (index, name, arguments))
                    .collect(),
                finishes,
                reasoning,
            )
        }
        let cases = [
            (
                serde_json::json!({"type":"string"}),
                "quote \\\" é <tool_call> literal",
            ),
            (
                serde_json::json!({"type":"object"}),
                r#"{"a":"é <tool_call> literal"}"#,
            ),
            (
                serde_json::json!({"type":"array"}),
                r#"["é <tool_call>", {"a":true}]"#,
            ),
            (serde_json::json!({"type":"integer"}), "42"),
            (serde_json::json!({"$ref":"#/$defs/value"}), r#"{"a":"é"}"#),
            (
                serde_json::json!({"type":["object","null"]}),
                r#"{"a":"é"}"#,
            ),
            (serde_json::json!({}), r#"{"a":"é"}"#),
            // Qwen closes parameters literally even inside JSON-looking text.
            // Pin its whole-input recovery rather than inventing a JSON boundary.
            (
                serde_json::json!({"type":"object"}),
                r#"{"a":"</parameter></tool_call><tool_call>literal"}"#,
            ),
        ];
        for (schema, value) in cases {
            let tools = vec![ToolDefinition {
                name: "f".into(),
                parameters: Some(
                    serde_json::json!({"type":"object","properties":{"x":schema},"$defs":{"value":{"type":"object"}}}),
                ),
                strict: None,
            }];
            let input = format!(
                "<tool_call><function=f><parameter=x>{value}</parameter></function></tool_call>"
            );
            let whole = observe(vec![chunk(&input, true)], tools.clone()).await;
            assert_eq!(
                whole.1.len(),
                1,
                "whole-input control must emit f: {value:?}"
            );
            assert_eq!(whole.1[0].1, "f");
            if value.contains("</parameter>") {
                assert_eq!(whole.1[0].2, r#"{"x":"{\"a\":\""}"#);
            }
            for split in (1..input.len()).filter(|at| input.is_char_boundary(*at)) {
                let mut detour = chunk("", false);
                let delta = &mut detour.data.as_mut().unwrap().inner.choices[0].delta;
                delta.content = None;
                delta.reasoning_content = Some("detour".into());
                let actual = observe(
                    vec![
                        chunk(&input[..split], false),
                        detour,
                        chunk(&input[split..], true),
                    ],
                    tools.clone(),
                )
                .await;
                assert_eq!(
                    (&actual.0, &actual.1, &actual.2),
                    (&whole.0, &whole.1, &whole.2),
                    "value={value:?} split={split}"
                );
                assert_eq!(
                    actual.3, "detour",
                    "reasoning must survive once at split={split}"
                );
            }
        }
    }

    #[tokio::test]
    async fn qwen_queued_reasoning_preserves_choice_envelope_on_resolution_and_eof() {
        for suffix in [
            Some(
                "all><function=get_weather><parameter=city>Paris</parameter></function></tool_call>",
            ),
            Some("at literal"),
            None,
        ] {
            for refusal in [None, Some("declined")] {
                let mut opening = chunk("<tool_c", false);
                opening.data.as_mut().unwrap().inner.choices[0].delta.role = None;
                let mut source = chunk("", false);
                let original = &mut source.data.as_mut().unwrap().inner.choices[0];
                original.delta.content = None;
                original.delta.reasoning_content = Some("detour".into());
                original.delta.refusal = refusal.map(str::to_string);
                original.logprobs = Some(serde_json::from_value(serde_json::json!({"content":[{"token":"detour","logprob":-0.5,"bytes":[100],"top_logprobs":[]}],"refusal":null})).unwrap());
                let expected = original.clone();
                let mut records = HashMap::from([(
                    0,
                    ChoiceRecord {
                        pending_qwen_opener_responses: VecDeque::from([
                            PendingQwenReasoningResponse {
                                text: "detour".into(),
                                choice: expected.clone(),
                                response: source.clone(),
                            },
                        ]),
                        ..ChoiceRecord::default()
                    },
                )]);
                let fallback = finish_unterminated_choices(&mut HashMap::new(), &mut records);
                assert_eq!(fallback.len(), 1);
                let FinishedStreamChunk::Source(fallback) = &fallback[0] else {
                    panic!("queued source must retain its source response");
                };
                assert_eq!(
                    fallback.data.as_ref().unwrap().inner.choices,
                    [expected.clone()]
                );
                let mut input = vec![opening, source];
                if let Some(suffix) = suffix {
                    let mut closing = chunk(suffix, true);
                    closing.data.as_mut().unwrap().inner.choices[0].delta.role = None;
                    input.push(closing);
                }
                let responses = apply_stream(
                    stream::iter(input),
                    Some(weather_tools()),
                    None,
                    false,
                    UnifiedParserStartingState::None,
                    QWEN3_UNIFIED_FAMILY,
                )
                .collect::<Vec<_>>()
                .await;
                let choices = collect_choices(&responses);
                let reasoning: Vec<_> = choices
                    .iter()
                    .filter(|choice| choice.delta.reasoning_content.as_deref() == Some("detour"))
                    .collect();
                assert_eq!(reasoning.len(), 1, "suffix={suffix:?} refusal={refusal:?}");
                assert_eq!(
                    *reasoning[0], &expected,
                    "queued or immediate reasoning must retain its untouched choice envelope"
                );
                assert_eq!(
                    choices
                        .iter()
                        .filter(|choice| choice.delta.role.is_some())
                        .count(),
                    1
                );
                assert_eq!(
                    choices
                        .iter()
                        .filter(|choice| choice.logprobs.is_some())
                        .count(),
                    1
                );
            }
        }
    }
    #[tokio::test]
    async fn qwen_recovered_parameter_close_restores_opener_ordering_at_every_split() {
        for first in [
            "<tool_call><function=get_weather><parameter=city>Paris</function></tool_call>",
            "<tool_call><function=get_weather><parameter=city</function></tool_call>",
        ] {
            for split in 1..first.len() {
                let mut reasoning = chunk("", false);
                let delta = &mut reasoning.data.as_mut().unwrap().inner.choices[0].delta;
                delta.content = None;
                delta.reasoning_content = Some("thought".into());
                let responses = apply_stream(
                    stream::iter([
                        chunk(&first[..split], false),
                        chunk(&first[split..], false),
                        chunk("before <tool_c", false),
                        reasoning,
                        chunk("XYZanswer", true),
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
                let mut sides = [String::new(), String::new()];
                let mut seen_reasoning = false;
                for choice in choices {
                    if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                        sides[usize::from(seen_reasoning)].push_str(text);
                    }
                    if choice.delta.reasoning_content.as_deref() == Some("thought") {
                        seen_reasoning = true;
                    }
                }
                assert!(seen_reasoning);
                assert_eq!(
                    sides,
                    ["before <tool_c".to_string(), "XYZanswer".to_string()],
                    "split={split}"
                );
            }
        }
    }
}
