// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered reasoning, text, and tool-call parsing through one state machine.
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

const DYNAMO_FAMILY_ALIASES: &[(&str, &str)] = &[
    ("deepseek-v4", "deepseek_v4"),
    ("deepseekv4", "deepseek_v4"),
    ("muse", "muse_glimmer"),
    ("gemma-4", "gemma4"),
    ("glm45", "glm47"),
    ("kimi_k25", "kimi_k2"),
];

/// Canonical family names and aliases share one route for registration, stream, and batch.
pub(crate) fn canonical_family(parser: &str) -> Option<&'static str> {
    let parser = DYNAMO_FAMILY_ALIASES
        .iter()
        .find_map(|(alias, family)| (*alias == parser).then_some(*family))
        .unwrap_or(parser);
    dynamo_parsers_v2::canonical_unified_family(parser)
}

pub(crate) static FAMILY_NAMES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let mut names = dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES.to_vec();
    names.extend(DYNAMO_FAMILY_ALIASES.iter().map(|(alias, _)| *alias));
    names.sort_unstable();
    names.dedup();
    names
});

/// A configured half may select a unified family, but two configured halves must agree.
pub(crate) fn configured_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    match (tool_call_parser, reasoning_parser) {
        (Some(tool), Some(reasoning)) => {
            let family = canonical_family(tool)?;
            (canonical_family(reasoning)? == family).then_some(family)
        }
        (Some(parser), None) | (None, Some(parser)) => canonical_family(parser),
        (None, None) => None,
    }
}

pub(crate) fn is_v2_configured_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> bool {
    configured_family(tool_call_parser, reasoning_parser).is_some()
}

/// The unified family to actually use for this parser pair, or `None` to keep the
/// existing split reasoning-parser + tool-call-jail path.
pub(crate) fn selected_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    let configured = configured_family(tool_call_parser, reasoning_parser);
    let selection = super::tool_parser_v2::selected_version()
        .unwrap_or_else(|error| panic!("invalid parser routing configuration: {error:#}"));
    let selected =
        selected_family_for_selection(configured, selection, tool_call_parser, reasoning_parser);
    tracing::debug!(
        target: "dynamo_unified",
        ?tool_call_parser,
        ?reasoning_parser,
        ?configured,
        ?selection,
        "unified parser path decision"
    );
    selected
}

fn selected_family_for_selection(
    configured: Option<&'static str>,
    selection: dynamo_runtime::config::ParserVersion,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    match selection {
        dynamo_runtime::config::ParserVersion::Auto => configured.filter(|family| {
            *family == DEEPSEEK_V41_UNIFIED_FAMILY
                && tool_call_parser == Some(DEEPSEEK_V41_UNIFIED_FAMILY)
                && reasoning_parser == Some(DEEPSEEK_V41_UNIFIED_FAMILY)
        }),
        dynamo_runtime::config::ParserVersion::V1 => None,
        dynamo_runtime::config::ParserVersion::V2 => configured,
    }
}

/// Whether the configured unified family is needed to decode content even when
/// request-level routing may keep tool calls on the legacy path. Muse's auto
/// route is request-mode dependent, but its decoder must remain available for
/// reasoning-only streams and when tool-call output is suppressed.
pub(crate) fn selected_content_decoder_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    let configured = configured_family(tool_call_parser, reasoning_parser);
    let selection = super::tool_parser_v2::selected_version()
        .unwrap_or_else(|error| panic!("invalid parser routing configuration: {error:#}"));
    let selected = selected_batch_family_for_selection(
        configured,
        selection,
        tool_call_parser,
        reasoning_parser,
    );
    tracing::debug!(
        target: "dynamo_unified",
        ?tool_call_parser,
        ?reasoning_parser,
        ?configured,
        ?selection,
        ?selected,
        "unified content decoder decision"
    );
    selected
}

fn selected_batch_family_for_selection(
    configured: Option<&'static str>,
    selection: dynamo_runtime::config::ParserVersion,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    match selection {
        dynamo_runtime::config::ParserVersion::Auto if configured == Some("muse_glimmer") => {
            configured
        }
        _ => {
            selected_family_for_selection(configured, selection, tool_call_parser, reasoning_parser)
        }
    }
}

/// The configured family eligible to own raw aggregate text.
///
/// Return the compatible unified family configured on the model card, before applying
/// the process-wide generation policy. Batch requests that suppress calls still need
/// the unified decoder to strip native markup before the calls are discarded.
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
    let configured = configured_batch_family(tool_call_parser, reasoning_parser);
    let selection = super::tool_parser_v2::selected_version()
        .unwrap_or_else(|error| panic!("invalid parser routing configuration: {error:#}"));
    let selected = selected_batch_family_for_selection(
        configured,
        selection,
        tool_call_parser,
        reasoning_parser,
    );
    tracing::debug!(
        target: "dynamo_unified",
        ?tool_call_parser,
        ?reasoning_parser,
        ?configured,
        ?selection,
        "unified batch parser path decision"
    );
    selected
}

/// Request facts that affect channel interpretation before any bytes are consumed.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UnifiedRequestPolicy {
    pub reasoning_disabled: bool,
    pub structured_response: bool,
}

impl UnifiedRequestPolicy {
    fn starting_state(
        self,
        family: &str,
        inferred: UnifiedParserStartingState,
    ) -> UnifiedParserStartingState {
        if self.reasoning_disabled && family == "gemma4" {
            // Gemma's Response state preserves literal thought delimiters and still parses tools.
            UnifiedParserStartingState::Response
        } else {
            inferred
        }
    }

    fn inspects_json_prefill(self, constraint: &GuidedToolConstraint) -> bool {
        self.structured_response || constraint.installs_guided_json()
    }

    fn batch_starting_state(
        self,
        family: &str,
        content: &str,
    ) -> anyhow::Result<UnifiedParserStartingState> {
        let inferred = if self.structured_response
            && structured_json_response_prefill(family, content, true)?
                == Some(UnifiedParserStartingState::Response)
        {
            UnifiedParserStartingState::Response
        } else {
            detect_prefill(family, content)?
        };
        Ok(self.starting_state(family, inferred))
    }
}

/// Scalar and container roots can also begin genuine reasoning. Hold only that
/// ambiguous prefix; process each incoming byte once and release it when a suffix
/// disproves JSON.
#[derive(Default)]
struct JsonPrefillProbe {
    text: String,
    state: JsonPrefixState,
    starts_json: Option<bool>,
    inspected_bytes: usize,
    container_start_validated: bool,
    marker_probe: Option<QuotedControlMarkerProbe>,
    marker_suffix: String,
    marker_decision: Option<UnifiedParserStartingState>,
    marker_inspected_bytes: usize,
}

#[derive(Default)]
enum JsonPrefixState {
    #[default]
    Start,
    ContainerStart {
        opener: u8,
        validated: bool,
    },
    String {
        escaped: bool,
        unicode_left: u8,
    },
    Literal {
        expected: &'static [u8],
        position: usize,
    },
    Number(NumberPrefixState),
    Done,
    Invalid,
}

#[derive(Clone, Copy)]
enum NumberPrefixState {
    Minus,
    Zero,
    Digits,
    Dot,
    Fraction,
    Exponent,
    ExponentSign,
    ExponentDigits,
}

impl JsonPrefillProbe {
    fn push(&mut self, text: &str) {
        self.text.push_str(text);
        for byte in text.bytes() {
            self.inspected_bytes += 1;
            let next_state = match &self.state {
                JsonPrefixState::Start => match byte {
                    b' ' | b'\r' | b'\n' | b'\t' => JsonPrefixState::Start,
                    b'{' | b'[' => JsonPrefixState::ContainerStart {
                        opener: byte,
                        validated: false,
                    },
                    b'"' => JsonPrefixState::String {
                        escaped: false,
                        unicode_left: 0,
                    },
                    b'-' => JsonPrefixState::Number(NumberPrefixState::Minus),
                    b'0' => JsonPrefixState::Number(NumberPrefixState::Zero),
                    b'1'..=b'9' => JsonPrefixState::Number(NumberPrefixState::Digits),
                    b't' => JsonPrefixState::Literal {
                        expected: b"true",
                        position: 1,
                    },
                    b'f' => JsonPrefixState::Literal {
                        expected: b"false",
                        position: 1,
                    },
                    b'n' => JsonPrefixState::Literal {
                        expected: b"null",
                        position: 1,
                    },
                    _ => JsonPrefixState::Invalid,
                },
                JsonPrefixState::String {
                    escaped,
                    unicode_left,
                } => {
                    if *unicode_left > 0 {
                        if byte.is_ascii_hexdigit() {
                            JsonPrefixState::String {
                                escaped: false,
                                unicode_left: unicode_left - 1,
                            }
                        } else {
                            JsonPrefixState::Invalid
                        }
                    } else if *escaped {
                        match byte {
                            b'u' => JsonPrefixState::String {
                                escaped: false,
                                unicode_left: 4,
                            },
                            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                                JsonPrefixState::String {
                                    escaped: false,
                                    unicode_left: 0,
                                }
                            }
                            _ => JsonPrefixState::Invalid,
                        }
                    } else {
                        match byte {
                            b'"' => JsonPrefixState::Done,
                            b'\\' => JsonPrefixState::String {
                                escaped: true,
                                unicode_left: 0,
                            },
                            0..=31 => JsonPrefixState::Invalid,
                            _ => JsonPrefixState::String {
                                escaped: false,
                                unicode_left: 0,
                            },
                        }
                    }
                }
                JsonPrefixState::Literal { expected, position } => {
                    if byte != expected[*position] {
                        JsonPrefixState::Invalid
                    } else if position + 1 == expected.len() {
                        JsonPrefixState::Done
                    } else {
                        JsonPrefixState::Literal {
                            expected,
                            position: position + 1,
                        }
                    }
                }
                JsonPrefixState::Number(number) => {
                    use NumberPrefixState::*;
                    let next = match (*number, byte) {
                        (Minus, b'0') => Some(Zero),
                        (Minus, b'1'..=b'9') => Some(Digits),
                        (Digits, b'0'..=b'9') => Some(Digits),
                        (Zero | Digits, b'.') => Some(Dot),
                        (Dot | Fraction, b'0'..=b'9') => Some(Fraction),
                        (Zero | Digits | Fraction, b'e' | b'E') => Some(Exponent),
                        (Exponent, b'+' | b'-') => Some(ExponentSign),
                        (Exponent | ExponentSign | ExponentDigits, b'0'..=b'9') => {
                            Some(ExponentDigits)
                        }
                        _ => None,
                    };
                    if let Some(next) = next {
                        JsonPrefixState::Number(next)
                    } else if matches!(number, Zero | Digits | Fraction | ExponentDigits)
                        && byte.is_ascii_whitespace()
                    {
                        JsonPrefixState::Done
                    } else {
                        JsonPrefixState::Invalid
                    }
                }
                JsonPrefixState::Done => {
                    if matches!(byte, b' ' | b'\r' | b'\n' | b'\t') {
                        JsonPrefixState::Done
                    } else {
                        JsonPrefixState::Invalid
                    }
                }
                JsonPrefixState::ContainerStart { opener, validated } => {
                    if *validated {
                        JsonPrefixState::ContainerStart {
                            opener: *opener,
                            validated: true,
                        }
                    } else if matches!(byte, b' ' | b'\r' | b'\n' | b'\t') {
                        JsonPrefixState::ContainerStart {
                            opener: *opener,
                            validated: false,
                        }
                    } else {
                        let valid_first_value = matches!(
                            (*opener, byte),
                            (b'{', b'"' | b'}')
                                | (
                                    b'[',
                                    b']' | b'{' | b'[' | b'"' | b'-' | b'0'
                                        ..=b'9' | b't' | b'f' | b'n',
                                )
                        );
                        if valid_first_value {
                            JsonPrefixState::ContainerStart {
                                opener: *opener,
                                validated: true,
                            }
                        } else {
                            JsonPrefixState::Invalid
                        }
                    }
                }
                JsonPrefixState::Invalid => JsonPrefixState::Invalid,
            };
            if self.starts_json.is_none() && !matches!(next_state, JsonPrefixState::Start) {
                self.starts_json = Some(!matches!(next_state, JsonPrefixState::Invalid));
            }
            if matches!(
                &next_state,
                JsonPrefixState::ContainerStart {
                    validated: true,
                    ..
                }
            ) {
                self.container_start_validated = true;
            }
            self.state = next_state;
        }
    }

    fn unopened_channel_decision(&self, terminal: bool) -> Option<UnifiedParserStartingState> {
        // JSON roots are unambiguous when the prompt has not opened reasoning;
        // waiting for a complete scalar would unnecessarily buffer its stream.
        match self.starts_json {
            Some(true) => Some(UnifiedParserStartingState::Response),
            Some(false) => Some(UnifiedParserStartingState::None),
            None if terminal => Some(UnifiedParserStartingState::Response),
            None => None,
        }
    }

    fn decision(&self, terminal: bool) -> Option<UnifiedParserStartingState> {
        if self.container_start_validated {
            return Some(UnifiedParserStartingState::None);
        }
        match self.state {
            JsonPrefixState::ContainerStart {
                validated: true, ..
            } if terminal => Some(UnifiedParserStartingState::None),
            JsonPrefixState::ContainerStart {
                validated: true, ..
            } => None,
            JsonPrefixState::ContainerStart { .. } if terminal => {
                Some(UnifiedParserStartingState::None)
            }
            JsonPrefixState::ContainerStart { .. } => None,
            JsonPrefixState::Invalid => Some(UnifiedParserStartingState::Reasoning),
            JsonPrefixState::Done
            | JsonPrefixState::Number(
                NumberPrefixState::Zero
                | NumberPrefixState::Digits
                | NumberPrefixState::Fraction
                | NumberPrefixState::ExponentDigits,
            ) if terminal => Some(UnifiedParserStartingState::None),
            _ if terminal => Some(UnifiedParserStartingState::Reasoning),
            _ => None,
        }
    }

    fn decision_for_family(
        &mut self,
        family: &str,
        terminal: bool,
        latest_text: &str,
    ) -> anyhow::Result<Option<UnifiedParserStartingState>> {
        if let Some(decision) = self.marker_decision {
            return Ok(Some(decision));
        }
        let first_push = self.marker_probe.is_none();
        let shield = self
            .marker_probe
            .get_or_insert_with(|| QuotedControlMarkerProbe::new(family));
        let incoming = if first_push { &self.text } else { latest_text };
        let mut unquoted = shield.push(incoming);
        if terminal {
            unquoted.push_str(&shield.finish());
        }
        self.marker_suffix.push_str(&unquoted);
        self.marker_inspected_bytes += self.marker_suffix.len();
        let has_candidate = shield
            .markers
            .iter()
            .any(|marker| self.marker_suffix.contains(marker));
        let suffix_limit = shield.markers.iter().map(String::len).max().unwrap_or(0);
        let mut keep_from = self.marker_suffix.len().saturating_sub(suffix_limit);
        while !self.marker_suffix.is_char_boundary(keep_from) {
            keep_from += 1;
        }
        self.marker_suffix.drain(..keep_from);

        // Only complete unquoted controls require raw-prefix classification;
        // rescanning growing JSON for every punctuation-bearing chunk is quadratic.
        if has_candidate && matches!(self.state, JsonPrefixState::ContainerStart { .. }) {
            self.marker_inspected_bytes += self.text.len();
            if let Some(state) = reasoning_marker_prefill(family, &self.text)? {
                self.marker_decision = Some(state);
                return Ok(Some(state));
            }
        }
        Ok(self.decision(terminal))
    }
}

fn structured_json_prefill(
    family: &str,
    content: &str,
    terminal: bool,
) -> anyhow::Result<Option<UnifiedParserStartingState>> {
    let mut probe = JsonPrefillProbe::default();
    probe.push(content);
    probe.decision_for_family(family, terminal, content)
}

fn structured_json_response_prefill(
    family: &str,
    content: &str,
    terminal: bool,
) -> anyhow::Result<Option<UnifiedParserStartingState>> {
    if let Some(state) = reasoning_marker_prefill(family, content)? {
        return Ok(Some(state));
    }
    let json_state = structured_json_prefill(family, content, terminal)?;
    if json_state == Some(UnifiedParserStartingState::None) {
        return Ok(Some(UnifiedParserStartingState::Response));
    }
    Ok(json_state)
}

fn reasoning_marker_prefill(
    family: &str,
    content: &str,
) -> anyhow::Result<Option<UnifiedParserStartingState>> {
    let has_marker = if let Some((opener, closer)) = reasoning_marker_pair(family) {
        contains_unquoted_marker(content, opener) || contains_unquoted_marker(content, closer)
    } else {
        match family {
            "kimi_k3" => {
                kimi_channel_marker_position(content, "<|open|>", "think").is_some()
                    || kimi_channel_marker_position(content, "<|close|>", "think").is_some()
            }
            "muse_glimmer" => {
                contains_unquoted_marker(content, "<|start|>")
                    || contains_unquoted_marker(content, "<|eom|>")
            }
            _ => false,
        }
    };
    has_marker
        .then(|| detect_prefill(family, content))
        .transpose()
}

fn starts_bare_json(content: &str) -> bool {
    matches!(content.trim_start().as_bytes().first(), Some(b'[' | b'{'))
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
        DEEPSEEK_V41_UNIFIED_FAMILY => UnifiedParserStartingState::None,
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
/// The stream adapter buffers empty and whitespace-only prefixes until a decisive
/// content chunk arrives. At EOF it preserves an undecided prefix as reasoning.
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
    if starts_bare_json(trimmed) {
        UnifiedParserStartingState::None
    } else {
        UnifiedParserStartingState::Reasoning
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
        "qwen3" | "deepseek_v4" | "deepseek_v41" | "glm47" | "kimi_k2" => {
            let (opener_marker, closer_marker) = reasoning_marker_pair(family)
                .expect("think-marker families have a configured reasoning pair");
            // Compare FIRST-occurrence positions, not mere presence: a prompt that
            // pre-opened reasoning produces a leading `</think>` with no opener before
            // it, but a later `<think>...</think>` pair from the model can still follow
            // in the same output (e.g. after a tool-call gap). Testing "does an opener
            // exist anywhere" before "does a closer exist anywhere" would misclassify
            // that case as `None` instead of `Reasoning`.
            let opener = first_unquoted_marker_position(content, opener_marker);
            let closer = first_unquoted_marker_position(content, closer_marker);
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
        "gemma4" => {
            let (opener, closer) = reasoning_marker_pair(family)
                .expect("Gemma 4 has a configured reasoning marker pair");
            Ok(infer_marker_prefill(content, opener, closer))
        }
        "kimi_k3" => Ok(
            match (
                kimi_channel_marker_position(content, "<|open|>", "think"),
                kimi_channel_marker_position(content, "<|close|>", "think"),
            ) {
                (None, Some(_)) => UnifiedParserStartingState::Reasoning,
                (Some(open), Some(close)) if close < open => UnifiedParserStartingState::Reasoning,
                _ => UnifiedParserStartingState::None,
            },
        ),
        "muse_glimmer" => Ok(infer_marker_prefill(content, "<|start|>", "<|eom|>")),
        other => anyhow::bail!("no prefill detector for unified parser family '{other}'"),
    }
}

fn reasoning_marker_pair(family: &str) -> Option<(&'static str, &'static str)> {
    match family {
        "qwen3" | "deepseek_v4" | "deepseek_v41" | "glm47" | "kimi_k2" => {
            Some(("<think>", "</think>"))
        }
        "gemma4" => Some(("<|channel>", "<channel|>")),
        _ => None,
    }
}

fn kimi_channel_marker_position(content: &str, prefix: &str, channel: &str) -> Option<usize> {
    content
        .match_indices(prefix)
        .filter_map(|(offset, _)| {
            let after = offset + prefix.len();
            let end = after + content[after..].find("<|sep|>")?;
            if content[after..end].split_whitespace().next() != Some(channel) {
                return None;
            }
            let marker = &content[offset..end + "<|sep|>".len()];
            first_unquoted_marker_position(content, marker)
        })
        .min()
}

fn infer_marker_prefill(content: &str, opener: &str, closer: &str) -> UnifiedParserStartingState {
    match (
        first_unquoted_marker_position(content, opener),
        first_unquoted_marker_position(content, closer),
    ) {
        (None, Some(_)) => UnifiedParserStartingState::Reasoning,
        (Some(open), Some(close)) if close < open => UnifiedParserStartingState::Reasoning,
        _ => UnifiedParserStartingState::None,
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
                let previous = content[..offset].chars().next_back();
                let next = content[offset + character.len_utf8()..].chars().next();
                if active_quote == '\'' && is_internal_apostrophe(previous, next) {
                    continue;
                }
                quote = None;
            }
            continue;
        }
        let previous = content[..offset].chars().next_back();
        let next = content[offset + character.len_utf8()..].chars().next();
        let apostrophe = character == '\''
            && (previous.is_some_and(char::is_alphanumeric)
                || next.is_some_and(char::is_alphanumeric) && !closes_later[position + 1][1]);
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

fn is_internal_apostrophe(previous: Option<char>, next: Option<char>) -> bool {
    previous.is_some_and(char::is_alphanumeric) && next.is_some_and(char::is_alphanumeric)
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
            let internal_apostrophe = character == '\''
                && is_internal_apostrophe(
                    i.checked_sub(1)
                        .and_then(|previous| chars.get(previous).map(|(_, c)| *c)),
                    chars.get(i + 1).map(|(_, c)| *c),
                );
            if let Some(slot) = quote_slot(character)
                && !internal_apostrophe
            {
                next[slot] = true;
            }
            next
        };
    }
    closes
}

/// Map dynamo's v1 [`ToolDefinition`]s onto the v2 parser's [`Tool`] shape.
pub(super) fn to_v2_tools(tools: Option<&[ToolDefinition]>) -> Vec<Tool> {
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
/// that the worker installed a structural tag. Each family's native tool marker
/// takes precedence over the reconstructed guided-JSON policy.
fn batch_tool_output_mode(
    family: &str,
    content: &str,
    constraint: &GuidedToolConstraint,
) -> UnifiedToolOutputMode {
    if constraint.installs_guided_json()
        && (native_family_marker(family)
            .is_some_and(|marker| contains_unquoted_marker(content, marker))
            || family == "kimi_k3"
                && kimi_channel_marker_position(content, "<|open|>", "call").is_some())
    {
        UnifiedToolOutputMode::Native
    } else {
        tool_output_mode(constraint)
    }
}

fn contains_native_envelope(content: &str, family: &str) -> bool {
    unified_parser_control_markers(family)
        .iter()
        .any(|marker| contains_unquoted_marker(content, marker))
}

fn unified_parser_control_markers(family: &str) -> Vec<String> {
    let mut markers = reasoning_marker_pair(family)
        .map(|(opener, closer)| vec![opener.to_string(), closer.to_string()])
        .unwrap_or_default();
    if family == "kimi_k3" {
        for channel in ["think", "response", "tools", "call"] {
            markers.push(format!("<|open|>{channel}"));
            markers.push(format!("<|close|>{channel}"));
        }
        markers.push("<|sep|>".to_string());
    }
    if family == "muse_glimmer" {
        markers.extend(
            ["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"]
                .into_iter()
                .map(str::to_string),
        );
    }
    if let Some(marker) = native_family_marker(family) {
        markers.push(marker.to_string());
    }
    if let Some(native_markers) = native_tool_call_start_tokens(family) {
        markers.extend(native_markers);
    }
    markers.sort_by_key(|marker| std::cmp::Reverse(marker.len()));
    markers.dedup();
    markers
}

fn native_family_marker(family: &str) -> Option<&'static str> {
    match family {
        "qwen3" | "glm47" => Some("<tool_call>"),
        "deepseek_v4" => Some("<｜DSML｜tool_calls>"),
        "deepseek_v41" => Some("<｜DSML｜ calls>"),
        "gemma4" => Some("<|tool_call>"),
        "kimi_k2" => Some("<|tool_call_begin|>"),
        "kimi_k3" => Some("<|open|>call"),
        "muse_glimmer" => Some("<atem:invoke"),
        _ => None,
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
                    if let Some(header_end) = input.find('>') {
                        input.drain(..header_end + 1);
                        self.mode = QwenNativeLexicalMode::ParameterValue;
                    } else {
                        input.clear();
                        return;
                    }
                }
                QwenNativeLexicalMode::ParameterValue => {
                    if let Some(parameter_end) = input.find(PARAMETER_END) {
                        input.drain(..parameter_end + PARAMETER_END.len());
                        self.mode = QwenNativeLexicalMode::InsideBlock;
                    } else {
                        self.retain_partial_suffix(&input, &[PARAMETER_END]);
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

    fn has_open_call(&self) -> bool {
        self.mode != QwenNativeLexicalMode::OutsideBlock || !self.marker_tail.is_empty()
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
        reasoning.append(events);
        *events = reasoning;
    }

    fn append_pending_qwen_reasoning(&mut self, events: &mut Vec<UnifiedParserEvent>) {
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
            if !inserted {
                if let UnifiedParserEvent::Text(text) = &event {
                    if let Some(remainder) = text.strip_prefix(held) {
                        if !held.is_empty() {
                            ordered.push(UnifiedParserEvent::Text(held.to_string()));
                        }
                        ordered.extend(pending.iter().cloned().map(UnifiedParserEvent::Reasoning));
                        if !remainder.is_empty() {
                            ordered.push(UnifiedParserEvent::Text(remainder.to_string()));
                        }
                        inserted = true;
                        continue;
                    }
                }
            }
            ordered.push(event);
        }
        if !inserted {
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
            Ok(output) => {
                deltas.extend(output.events);
                deltas
            }
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    family = self.family,
                    "unified parser finish failed; recovering buffered text"
                );
                deltas.extend(self.give_up(""));
                deltas
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
            // TODO: Preserve the parser's consumed offset so a partial-commit error cannot replay input.
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
                        name: first.then_some(name).flatten(),
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
        let terminal_tool_call_delta = finish_reason_closes_tool_calls(finish_reason.as_ref());
        if terminal_tool_call_delta {
            self.pending_parsed_tool_calls.clear();
        }
        let deltas = coalesce(
            deltas
                .into_iter()
                .filter(|delta| {
                    emit_tool_calls || !matches!(delta, UnifiedParserEvent::ToolCall(_))
                })
                .collect(),
        );
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

    #[cfg(test)]
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
            || (self.family == QWEN3_UNIFIED_FAMILY
                && (self.qwen_native_lexical_state.has_open_call()
                    || !self.pending_qwen_opener_text.is_empty()))
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
#[cfg(test)]
pub(crate) fn parse_complete(
    family: &str,
    content: &str,
    guided_tool_constraint: &GuidedToolConstraint,
    tool_definitions: &[ToolDefinition],
) -> anyhow::Result<CompleteOutput> {
    parse_complete_with_policy(
        family,
        content,
        guided_tool_constraint,
        tool_definitions,
        UnifiedRequestPolicy::default(),
    )
}

pub(crate) fn parse_complete_with_policy(
    family: &str,
    content: &str,
    guided_tool_constraint: &GuidedToolConstraint,
    tool_definitions: &[ToolDefinition],
    policy: UnifiedRequestPolicy,
) -> anyhow::Result<CompleteOutput> {
    let tools = to_v2_tools(Some(tool_definitions));
    let mut parser = create_unified_parser_for_family(family, &tools)?;
    // Batch replays already-generated output. Prefer its observable native marker when
    // topology B could not carry the worker's installed structural tag to the frontend;
    // otherwise use the reconstructed guided-JSON constraint.
    let mut effective_mode = batch_tool_output_mode(family, content, guided_tool_constraint);
    // `batch_tool_output_mode` drops to `Native` on observed markup even when the
    // constraint reconstructed a `GuidedJsonNamed` pin. That native fallback recovers
    // whatever tool name the markup happens to contain, so it must be checked against
    // the forced name below instead of trusted unfiltered — a malformed guided output
    // that embeds a *different* tool's markup must not be handed back as the pinned
    // tool's call.
    parser.initialize_request(UnifiedParserInit {
        starting_state: policy.batch_starting_state(family, content)?,
        tool_output_mode: effective_mode.clone(),
        ..UnifiedParserInit::default()
    })?;

    let events = match parser.parse_complete(content) {
        Ok(events) => events,
        Err(error)
            if matches!(effective_mode, UnifiedToolOutputMode::GuidedJson { .. })
                && error.is::<dynamo_parsers_v2::InvalidGuidedPayload>()
                && contains_native_envelope(content, family) =>
        {
            // A reconstructed JSON constraint can mask native channel output with no calls.
            effective_mode = UnifiedToolOutputMode::Native;
            // parse_complete has consumed bytes; request initialization requires a fresh parser.
            parser = create_unified_parser_for_family(family, &tools)?;
            parser.initialize_request(UnifiedParserInit {
                starting_state: policy.batch_starting_state(family, content)?,
                tool_output_mode: effective_mode.clone(),
                ..UnifiedParserInit::default()
            })?;
            parser.parse_complete(content)?
        }
        Err(error) => return Err(error),
    };
    let forced_tool_name = match (guided_tool_constraint, &effective_mode) {
        (GuidedToolConstraint::GuidedJsonNamed { tool_name }, UnifiedToolOutputMode::Native) => {
            Some(tool_name.as_str())
        }
        _ => None,
    };
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for event in events {
        match event {
            UnifiedEvent::Text { text: chunk } => text.push_str(&chunk),
            UnifiedEvent::Reasoning { text: chunk } => {
                if policy.reasoning_disabled {
                    text.push_str(&chunk);
                } else {
                    reasoning.push_str(&chunk);
                }
            }
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
    response: Annotated<NvCreateChatCompletionStreamResponse>,
}

enum FinishedStreamChunk {
    Choice(ChatChoiceStream),
    Source(Annotated<NvCreateChatCompletionStreamResponse>),
}

fn take_qwen_reasoning_responses(
    record: &mut ChoiceRecord,
    reasoning: &ChatChoiceStream,
) -> Option<Vec<Annotated<NvCreateChatCompletionStreamResponse>>> {
    let pending_reasoning: String = record
        .pending_qwen_opener_responses
        .iter()
        .map(|pending| pending.text.as_str())
        .collect();
    if pending_reasoning.is_empty()
        || reasoning.delta.reasoning_content.as_deref() != Some(pending_reasoning.as_str())
    {
        return None;
    }

    Some(
        std::iter::from_fn(|| record.pending_qwen_opener_responses.pop_front())
            .map(|pending| {
                let mut response = pending.response;
                let mut source_reasoning = reasoning.clone();
                source_reasoning.delta.reasoning_content = Some(pending.text);
                source_reasoning.finish_reason = None;
                response
                    .data
                    .as_mut()
                    .expect("queued source response has data")
                    .inner
                    .choices = vec![source_reasoning];
                response
            })
            .collect(),
    )
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
                    if let Some(last) = flushed.last_mut() {
                        last.finish_reason = Some(FinishReason::ToolCalls);
                    } else {
                        flushed.extend(state.choices_for(
                            &base,
                            Vec::new(),
                            true,
                            Some(FinishReason::ToolCalls),
                        ));
                    }
                }
                for choice in flushed {
                    if let Some(responses) = take_qwen_reasoning_responses(record, &choice) {
                        chunks.extend(responses.into_iter().map(FinishedStreamChunk::Source));
                    } else {
                        chunks.push(FinishedStreamChunk::Choice(choice));
                    }
                }
            }
            None => {
                while let Some(pending) = record.pending_qwen_opener_responses.pop_front() {
                    let mut choice = base.clone();
                    choice.delta.reasoning_content = Some(pending.text);
                    let mut response = pending.response;
                    response
                        .data
                        .as_mut()
                        .expect("queued source response has data")
                        .inner
                        .choices = vec![choice];
                    chunks.push(FinishedStreamChunk::Source(response));
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

fn resolve_guided_prefix(
    index: u32,
    probe: JsonPrefillProbe,
    policy: UnifiedRequestPolicy,
    prefill: UnifiedParserStartingState,
) -> Option<ChatChoiceStream> {
    let decision = probe.decision(true);
    if probe.text.is_empty() {
        return None;
    }
    let mut choice = empty_choice(index);
    if policy.structured_response
        && (prefill == UnifiedParserStartingState::None
            || decision == Some(UnifiedParserStartingState::None))
    {
        choice.delta.content = Some(ChatCompletionMessageContent::Text(probe.text));
    } else {
        choice.delta.reasoning_content = Some(probe.text);
    }
    Some(choice)
}

fn flush_undecided_guided_prefixes(
    prefixes: &mut HashMap<u32, JsonPrefillProbe>,
    policy: UnifiedRequestPolicy,
    prefill: UnifiedParserStartingState,
) -> Vec<ChatChoiceStream> {
    prefixes
        .drain()
        .filter_map(|(index, probe)| resolve_guided_prefix(index, probe, policy, prefill))
        .collect()
}

/// `guided_streaming` is the request's already-made guided tool-call streaming
/// decision (`OpenAIPreprocessor::guided_tool_streaming_release`), not a second
/// derivation of it: `true` releases each call as soon as its name and argument
/// object opener are unambiguous, `false` buffers every call to completion. It is
/// inert unless `guided_tool_constraint` installed guided JSON.
#[cfg(test)]
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
    apply_stream_with_policy(
        stream_in,
        tool_definitions,
        guided_tool_constraint,
        prefill,
        family,
        guided_streaming,
        UnifiedRequestPolicy::default(),
    )
}

pub(crate) fn apply_stream_with_policy<S>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    guided_tool_constraint: GuidedToolConstraint,
    prefill: UnifiedParserStartingState,
    family: &'static str,
    guided_streaming: bool,
    policy: UnifiedRequestPolicy,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let tools = to_v2_tools(tool_definitions.as_deref());
    let prefill = policy.starting_state(family, prefill);
    stream! {
            let mut states: HashMap<u32, ChoiceState> = HashMap::new();
            let mut undecided_guided_prefix: HashMap<u32, JsonPrefillProbe> = HashMap::new();
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
                        for choice in flush_undecided_guided_prefixes(&mut undecided_guided_prefix, policy, prefill) {
                            yield response_with_choice(template, choice);
                        }
                        for output in finish_unterminated_choices(&mut states, &mut records) {
                            match output {
                                FinishedStreamChunk::Choice(choice) => {
                                    yield response_with_choice(template, choice);
                                }
                                FinishedStreamChunk::Source(response) => yield response,
                            }
                        }
                    }
                    yield response;
                    continue;
                }

                let originals = std::mem::take(&mut chat.inner.choices);
                let mut emitted: Vec<ChatChoiceStream> = Vec::new();
                let mut deferred_qwen_responses = Vec::new();
                for mut original in originals {
                    if already_parsed(&original) {
                        if let Some(prefix) = undecided_guided_prefix.remove(&original.index)
                            && let Some(choice) = resolve_guided_prefix(original.index, prefix, policy, prefill)
                        { emitted.push(choice); }
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
                                record.pending_qwen_opener_responses.push_back(
                                    PendingQwenReasoningResponse {
                                        text: reasoning,
                                        response: Annotated {
                                            data: Some(source_data),
                                            id: response_id.clone(),
                                            event: response_event.clone(),
                                            comment: response_comment.clone(),
                                            error: response_error.clone(),
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
                            emitted.extend(state.choices_for(
                                &empty_choice(original.index),
                                deltas,
                                true,
                                None,
                            ));
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

                    let mut detected_prefill = prefill;
                    if !states.contains_key(&original.index)
                        && (prefill == UnifiedParserStartingState::Reasoning
                            || policy.structured_response && prefill == UnifiedParserStartingState::None)
                        && policy.inspects_json_prefill(&guided_tool_constraint)
                        && !records.get(&original.index).is_some_and(|record| record.detoured)
                    {
                        let text = match original.delta.content.as_ref() {
                            Some(ChatCompletionMessageContent::Text(text)) => text.as_str(),
                            _ => "",
                        };
                        let mut probe = undecided_guided_prefix.remove(&original.index).unwrap_or_default();
                        probe.push(text);
                        let probe_decision = if prefill == UnifiedParserStartingState::None {
                            // Without a prompt-opened channel, non-JSON may contain an explicit
                            // reasoning opener. Let the native parser recognize it.
                            probe.unopened_channel_decision(original.finish_reason.is_some())
                        } else {
                            match probe.decision_for_family(
                                family,
                                original.finish_reason.is_some(),
                                text,
                            ) {
                                Ok(decision) => decision,
                                Err(error) => {
                                    tracing::warn!(
                                        error = %error,
                                        family,
                                        choice = original.index,
                                        "structured response prefill detection failed; treating buffered output as reasoning"
                                    );
                                    Some(UnifiedParserStartingState::Reasoning)
                                }
                            }
                        };
                        let awaiting_reasoning_boundary = prefill == UnifiedParserStartingState::Reasoning
                            && policy.structured_response
                            && !policy.reasoning_disabled
                            && probe_decision == Some(UnifiedParserStartingState::None)
                            && original.finish_reason.is_none();
                        if (probe.text.trim().is_empty() && original.finish_reason.is_none())
                            || policy.structured_response
                                && (probe_decision.is_none() || awaiting_reasoning_boundary)
                        {
                            undecided_guided_prefix.insert(original.index, probe);
                            original.delta.content = None;
                            emitted.push(original);
                            continue;
                        }
                        if prefill == UnifiedParserStartingState::None
                            && probe_decision == Some(UnifiedParserStartingState::Response)
                        {
                            detected_prefill = UnifiedParserStartingState::Response;
                        }
                        original.delta.content = Some(ChatCompletionMessageContent::Text(probe.text));
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
                    if new_native_call_after_detour {
                        if let Some(mut state) = states.remove(&original.index) {
                            let deltas = state.finish();
                            records
                                .entry(original.index)
                                .or_default()
                                .remember_state(&state);
                            emitted.extend(state.choices_for(
                                &empty_choice(original.index),
                                deltas,
                                true,
                                None,
                            ));
                        }
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
                                && policy.inspects_json_prefill(&guided_tool_constraint)
                            {
                                if policy.structured_response {
                                    match original.delta.content.as_ref() {
                                        Some(ChatCompletionMessageContent::Text(text)) => match structured_json_response_prefill(family, text, original.finish_reason.is_some()) {
                                            Ok(Some(state)) => state,
                                            Ok(None) => UnifiedParserStartingState::Reasoning,
                                            Err(error) => {
                                                tracing::warn!(
                                                    error = %error,
                                                    family,
                                                    choice = original.index,
                                                    "structured response prefill detection failed; treating output as reasoning"
                                                );
                                                UnifiedParserStartingState::Reasoning
                                            }
                                        },
                                        _ => UnifiedParserStartingState::Reasoning,
                                    }
                                } else {
                                    bare_guided_json_prefill(original.delta.content.as_ref())
                                }
                            } else {
                                detected_prefill
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

                    let mut parsed = state.choices_for(&original, deltas, true, terminal);
                    if let Some(record) = records.get_mut(&original.index) {
                        if let Some(position) = parsed.iter().position(|choice| {
                            let queued_text: String = record
                                .pending_qwen_opener_responses
                                .iter()
                                .map(|pending| pending.text.as_str())
                                .collect();
                            !queued_text.is_empty()
                                && choice.delta.reasoning_content.as_deref()
                                    == Some(queued_text.as_str())
                        }) {
                            let reasoning = parsed.remove(position);
                            if let Some(responses) = take_qwen_reasoning_responses(record, &reasoning) {
                                deferred_qwen_responses.extend(
                                    responses
                                        .into_iter()
                                        .map(|response| (emitted.len() + position, response)),
                                );
                            }
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
                        yield response;
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
                        for choice in flush_undecided_guided_prefixes(&mut undecided_guided_prefix, policy, prefill) {
                            yield response_with_choice(template, choice);
                        }
                for output in finish_unterminated_choices(&mut states, &mut records) {
                    match output {
                        FinishedStreamChunk::Choice(choice) => {
                            yield response_with_choice(template, choice);
                        }
                        FinishedStreamChunk::Source(response) => yield response,
                    }
                }
            }
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers_v2::{ToolCallDelta, UnifiedParserExt};
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
            prompt_logprobs: None,
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

    fn all_family_native_calls() -> [(&'static str, &'static str); 8] {
        [
            (
                "deepseek_v4",
                r#"<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather"><｜DSML｜parameter name="city" string="true">Tokyo</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"#,
            ),
            (
                "deepseek_v41",
                r#"<｜DSML｜ calls><｜DSML｜ invoke name="get_weather"><｜DSML｜ parameter name="city" string="true">Tokyo</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"#,
            ),
            (
                "gemma4",
                "<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>",
            ),
            (
                "glm47",
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>",
            ),
            (
                "kimi_k2",
                r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city":"Tokyo"}<|tool_call_end|><|tool_calls_section_end|>"#,
            ),
            (
                "kimi_k3",
                r#"<|open|>tools<|sep|><|open|>call tool="get_weather" index="0"<|sep|><|open|>argument key="city" type="string"<|sep|>Tokyo<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"#,
            ),
            (
                "muse_glimmer",
                r#"<|start|>assistant to=get_weather<|message|><atem:function_calls><atem:invoke name="get_weather"><atem:parameter name="city">Tokyo</atem:parameter></atem:invoke></atem:function_calls><|eom|>"#,
            ),
            (
                "qwen3",
                "<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>",
            ),
        ]
    }

    fn assert_stream_calls_match_batch(
        events: &[LogicalEvent],
        batch: &CompleteOutput,
        context: &str,
    ) {
        let mut calls = std::collections::BTreeMap::<u32, (Option<String>, String)>::new();
        for event in events {
            if let LogicalEvent::Tool {
                index,
                name,
                arguments,
            } = event
            {
                let (current_name, current_arguments) = calls.entry(*index).or_default();
                if let Some(name) = name {
                    assert!(current_name.is_none(), "{context}: repeated tool name");
                    *current_name = Some(name.clone());
                }
                current_arguments.push_str(arguments);
            }
        }
        let actual: Vec<_> = calls
            .into_iter()
            .map(|(index, (name, arguments))| {
                (
                    index,
                    name.expect("streamed tool name"),
                    serde_json::from_str::<serde_json::Value>(&arguments)
                        .expect("streamed arguments"),
                )
            })
            .collect();
        let expected: Vec<_> = batch
            .tool_calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                (
                    index as u32,
                    call.function.name.clone(),
                    serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                        .expect("batch arguments"),
                )
            })
            .collect();
        assert_eq!(actual, expected, "{context}");
    }

    #[tokio::test]
    async fn every_family_native_and_guided_batch_stream_split_parity() {
        for (family, native) in all_family_native_calls() {
            for (input, constraint) in [
                (native, GuidedToolConstraint::None),
                (
                    r#"{"city":"Tokyo"}"#,
                    GuidedToolConstraint::GuidedJsonNamed {
                        tool_name: "get_weather".to_string(),
                    },
                ),
                (
                    r#"[{"name":"get_weather","arguments":{"city":"Tokyo"}}]"#,
                    GuidedToolConstraint::GuidedJsonRequired,
                ),
            ] {
                let batch = parse_complete(family, input, &constraint, &weather_tools()).unwrap();
                assert_eq!(batch.tool_calls.len(), 1, "{family} {constraint:?}");
                assert_eq!(batch.tool_calls[0].function.name, "get_weather", "{family}");
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(
                        &batch.tool_calls[0].function.arguments
                    )
                    .unwrap(),
                    serde_json::json!({"city":"Tokyo"}),
                    "{family}"
                );
                assert!(batch.text.is_empty(), "{family}: {:?}", batch.text);
                for split in input
                    .char_indices()
                    .map(|(offset, _)| offset)
                    .chain(std::iter::once(input.len()))
                {
                    let (first, second) = input.split_at(split);
                    let output = apply_stream_with_constraint(
                        stream::iter([chunk(first, false), chunk(second, true)]),
                        Some(weather_tools()),
                        constraint.clone(),
                        UnifiedParserStartingState::None,
                        family,
                        true,
                    )
                    .collect::<Vec<_>>()
                    .await;
                    assert!(
                        output.iter().all(|response| !response.is_error()),
                        "{family} split {split}"
                    );
                    let events = logical_events(&output);
                    assert_stream_calls_match_batch(
                        &events,
                        &batch,
                        &format!("{family} split {split}"),
                    );
                    assert_eq!(
                        collect_choices(&output).last().unwrap().finish_reason,
                        Some(FinishReason::ToolCalls),
                        "{family} split {split}"
                    );
                    assert!(!events.iter().any(|event| matches!(event, LogicalEvent::Text(t) | LogicalEvent::Reasoning(t) if !t.is_empty())), "{family} split {split}: {events:?}");
                }
            }
        }
    }

    #[tokio::test]
    async fn all_family_reasoning_prefill_and_native_batch_stream_parity() {
        for (family, native) in all_family_native_calls() {
            let (opener, closer, answer) = match family {
                "gemma4" => ("<|channel>thought\n", "<channel|>", "answer"),
                "kimi_k3" => (
                    "<|open|>think<|sep|>",
                    "<|close|>think<|sep|>",
                    "<|open|>response<|sep|>answer<|close|>response<|sep|>",
                ),
                "muse_glimmer" => (
                    "<|start|>assistant to=self<|message|>",
                    "<|eom|>",
                    "<|start|>assistant to=user<|message|>answer<|eot|>",
                ),
                _ => ("<think>", "</think>", "answer"),
            };
            for prefilling in [false, true] {
                let input = format!(
                    "{}plan{closer}{native}{answer}",
                    if prefilling { "" } else { opener }
                );
                let batch = parse_complete(
                    family,
                    &input,
                    &GuidedToolConstraint::None,
                    &weather_tools(),
                )
                .unwrap();
                assert_eq!(batch.reasoning, "plan", "{family} prefilled={prefilling}");
                assert_eq!(batch.text, "answer", "{family} prefilled={prefilling}");
                assert_eq!(batch.tool_calls.len(), 1, "{family} prefilled={prefilling}");
                for split in input
                    .char_indices()
                    .map(|(offset, _)| offset)
                    .chain(std::iter::once(input.len()))
                {
                    let (first, second) = input.split_at(split);
                    let output = apply_stream_with_constraint(
                        stream::iter([chunk(first, false), chunk(second, true)]),
                        Some(weather_tools()),
                        GuidedToolConstraint::None,
                        stream_prefill(family, prefilling),
                        family,
                        true,
                    )
                    .collect::<Vec<_>>()
                    .await;
                    let events = logical_events(&output);
                    assert_stream_calls_match_batch(
                        &events,
                        &batch,
                        &format!("{family} prefilled={prefilling} split={split}"),
                    );
                    let reasoning: String = events
                        .iter()
                        .filter_map(|event| match event {
                            LogicalEvent::Reasoning(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect();
                    let text: String = events
                        .iter()
                        .filter_map(|event| match event {
                            LogicalEvent::Text(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect();
                    assert_eq!(
                        reasoning, batch.reasoning,
                        "{family} prefilled={prefilling} split={split}"
                    );
                    assert_eq!(
                        text, batch.text,
                        "{family} prefilled={prefilling} split={split}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn all_family_guided_streaming_rollback_preserves_complete_calls() {
        for (family, _) in all_family_native_calls() {
            let constraint = GuidedToolConstraint::GuidedJsonRequired;
            let input = r#"[{"name":"get_weather","arguments":{"city":"Tokyo"}}]"#;
            let released = apply_stream_with_constraint(
                stream::iter([chunk(&input[..35], false), chunk(&input[35..], true)]),
                Some(weather_tools()),
                constraint.clone(),
                UnifiedParserStartingState::None,
                family,
                true,
            )
            .collect::<Vec<_>>()
            .await;
            let buffered = apply_stream_with_constraint(
                stream::iter([chunk(&input[..35], false), chunk(&input[35..], true)]),
                Some(weather_tools()),
                constraint,
                UnifiedParserStartingState::None,
                family,
                false,
            )
            .collect::<Vec<_>>()
            .await;
            assert_eq!(
                logical_events(&released),
                logical_events(&buffered),
                "{family}"
            );
        }
    }

    #[tokio::test]
    async fn kimi_named_guided_call_does_not_leak_native_end_marker() {
        let tools = vec![ToolDefinition {
            name: "send_email".to_string(),
            parameters: Some(
                serde_json::json!({"type":"object", "properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}}}),
            ),
            strict: None,
        }];
        let input = r#"{"to":"user@example.com","subject":"Hello","body":"Hi"}<|tool_call_end|>"#;
        for by_character in [false, true] {
            for finish_reason in [FinishReason::Stop, FinishReason::Length] {
                let mut chunks: Vec<_> = if by_character {
                    input
                        .char_indices()
                        .map(|(start, c)| chunk(&input[start..start + c.len_utf8()], false))
                        .collect()
                } else {
                    vec![chunk(input, false)]
                };
                let mut terminal = chunk("", true);
                terminal.data.as_mut().unwrap().inner.choices[0].finish_reason =
                    Some(finish_reason);
                chunks.push(terminal);
                let output = apply_stream_with_constraint(
                    stream::iter(chunks),
                    Some(tools.clone()),
                    GuidedToolConstraint::GuidedJsonNamed {
                        tool_name: "send_email".to_string(),
                    },
                    UnifiedParserStartingState::None,
                    "kimi_k2",
                    true,
                )
                .collect::<Vec<_>>()
                .await;
                let events = logical_events(&output);
                let text: String = events
                    .iter()
                    .filter_map(|event| match event {
                        LogicalEvent::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .collect();
                let names: Vec<_> = events
                    .iter()
                    .filter_map(|event| match event {
                        LogicalEvent::Tool { name: Some(n), .. } => Some(n.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    names,
                    vec!["send_email"],
                    "{by_character} {finish_reason:?}"
                );
                assert!(
                    text.is_empty(),
                    "{by_character} {finish_reason:?}: {text:?}"
                );
            }
        }
    }

    #[test]
    fn invalid_parser_version_cannot_enable_unified_routing() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::invalid_parser_version_cannot_enable_unified_routing"
            ),
            &[(
                dynamo_runtime::config::environment_names::llm::DYN_PARSER_VERSION,
                "v3",
            )],
        ) {
            return;
        }
        assert!(std::panic::catch_unwind(|| selected_family(Some("qwen3"), None)).is_err());
    }

    #[test]
    fn explicit_v1_rolls_back_unified_routes_in_an_isolated_process() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::explicit_v1_rolls_back_unified_routes_in_an_isolated_process"
            ),
            &[(
                dynamo_runtime::config::environment_names::llm::DYN_PARSER_VERSION,
                "1",
            )],
        ) {
            return;
        }
        for (family, _) in all_family_native_calls() {
            assert_eq!(
                selected_family(Some(family), Some(family)),
                None,
                "{family}"
            );
            assert_eq!(
                selected_batch_family(Some(family), Some(family)),
                None,
                "{family}"
            );
        }
    }

    #[test]
    fn spaced_kimi_think_prefill_and_response_closer_are_distinct() {
        assert_eq!(detect_prefill("kimi_k3", "hidden<|close|> think <|sep|><|open|> response <|sep|>answer<|close|> response <|sep|>").unwrap(), UnifiedParserStartingState::Reasoning);
        assert_eq!(
            detect_prefill("kimi_k3", "answer<|close|> response <|sep|>").unwrap(),
            UnifiedParserStartingState::None
        );
    }

    #[test]
    fn every_family_batch_native_markup_overrides_reconstructed_guided_json() {
        for (family, input) in all_family_native_calls() {
            for constraint in [
                GuidedToolConstraint::GuidedJsonNamed {
                    tool_name: "get_weather".to_string(),
                },
                GuidedToolConstraint::GuidedJsonRequired,
            ] {
                let parsed = parse_complete(family, input, &constraint, &weather_tools()).unwrap();
                assert_eq!(parsed.tool_calls.len(), 1, "{family}");
                assert_eq!(
                    parsed.tool_calls[0].function.name, "get_weather",
                    "{family}"
                );
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(
                        &parsed.tool_calls[0].function.arguments
                    )
                    .unwrap(),
                    serde_json::json!({"city":"Tokyo"}),
                    "{family}"
                );
            }
        }
        let (_, input) = all_family_native_calls()
            .into_iter()
            .find(|(family, _)| *family == "kimi_k3")
            .unwrap();
        let spaced = input.replace("<|open|>call", "<|open|> call");
        let parsed = parse_complete(
            "kimi_k3",
            &spaced,
            &GuidedToolConstraint::GuidedJsonRequired,
            &weather_tools(),
        )
        .unwrap();
        assert_eq!(parsed.tool_calls.len(), 1);
    }

    #[tokio::test]
    async fn guided_prefill_whitespace_is_decided_before_json_and_flushed_before_detours() {
        let constraint = GuidedToolConstraint::GuidedJsonNamed {
            tool_name: "get_weather".to_string(),
        };
        for (family, _) in all_family_native_calls() {
            let output = apply_stream_with_constraint(
                stream::iter([
                    chunk("", false),
                    chunk("  ", false),
                    chunk(r#"{"city":"Tokyo"}"#, true),
                ]),
                Some(weather_tools()),
                constraint.clone(),
                UnifiedParserStartingState::Reasoning,
                family,
                true,
            )
            .collect::<Vec<_>>()
            .await;
            assert!(logical_events(&output).iter().any(|event| matches!(event, LogicalEvent::Tool { name: Some(name), .. } if name == "get_weather")), "{family}");
        }
        let mut terminal = chunk("", true);
        terminal.data.as_mut().unwrap().inner.choices[0]
            .delta
            .content = None;
        terminal.data.as_mut().unwrap().inner.choices[0]
            .delta
            .reasoning_content = Some("done".to_string());
        let output = apply_stream_with_constraint(
            stream::iter([chunk(" ", false), terminal]),
            Some(weather_tools()),
            constraint,
            UnifiedParserStartingState::Reasoning,
            "qwen3",
            true,
        )
        .collect::<Vec<_>>()
        .await;
        assert_eq!(
            logical_events(&output),
            vec![LogicalEvent::Reasoning(" done".to_string())]
        );
        let terminal_position = output
            .iter()
            .position(|r| {
                r.data
                    .as_ref()
                    .is_some_and(|d| d.inner.choices.iter().any(|c| c.finish_reason.is_some()))
            })
            .unwrap();
        assert!(
            !output[terminal_position + 1..]
                .iter()
                .any(|r| r.data.as_ref().is_some_and(|d| d
                    .inner
                    .choices
                    .iter()
                    .any(|c| c.delta.reasoning_content.is_some())))
        );
    }

    #[test]
    fn all_registered_families_route_to_one_unified_owner() {
        for (family, _) in all_family_native_calls() {
            assert_eq!(configured_family(Some(family), Some(family)), Some(family));
            assert!(
                create_unified_parser_for_family(family, &[])
                    .unwrap()
                    .preserve_special_tokens()
            );
        }
        assert_eq!(
            configured_family(Some("kimi_k2"), Some("kimi_k25")),
            Some("kimi_k2")
        );
        assert_eq!(configured_family(Some("qwen3_coder"), Some("glm47")), None);
    }

    // --- selected_family / configured_family -------------------------------------

    #[test]
    fn deepseek_v41_uses_native_parsing_without_an_experimental_flag() {
        let pair = (Some("deepseek_v41"), Some("deepseek_v41"));
        assert_eq!(configured_family(pair.0, pair.1), Some("deepseek_v41"));
        assert_eq!(selected_family(pair.0, pair.1), Some("deepseek_v41"));
        assert_eq!(selected_batch_family(pair.0, pair.1), Some("deepseek_v41"));
        assert_eq!(configured_family(Some("deepseek_v41"), Some("qwen3")), None);
        assert_eq!(
            configured_family(Some("deepseek_v41"), None),
            Some("deepseek_v41")
        );
        assert_eq!(
            configured_family(None, Some("deepseek_v41")),
            Some("deepseek_v41")
        );
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
        // A single configured half selects the same family.
        assert_eq!(configured_family(Some("qwen3_coder"), None), Some("qwen3"));
        assert_eq!(configured_family(None, Some("qwen3")), Some("qwen3"));
        assert_eq!(configured_family(None, None), None);
        // Right tool parser, wrong reasoning family.
        assert_eq!(
            configured_family(Some("qwen3_coder"), Some("deepseek_r1")),
            None
        );
        // Right reasoning parser, wrong tool family.
        assert_eq!(configured_family(Some("kimi_k2"), Some("qwen3")), None);
        // The canonical name and Dynamo tool alias select the same owner.
        assert_eq!(
            configured_family(Some("qwen3"), Some("qwen3")),
            Some("qwen3")
        );
    }

    #[test]
    fn muse_content_decoder_preserves_auto_routing_without_changing_v1() {
        let muse = configured_family(Some("muse_glimmer"), None);
        assert_eq!(
            selected_batch_family_for_selection(
                muse,
                dynamo_runtime::config::ParserVersion::Auto,
                Some("muse_glimmer"),
                None,
            ),
            Some("muse_glimmer")
        );
        assert_eq!(
            selected_batch_family_for_selection(
                muse,
                dynamo_runtime::config::ParserVersion::V1,
                Some("muse_glimmer"),
                None,
            ),
            None
        );
        assert_eq!(
            selected_batch_family_for_selection(
                muse,
                dynamo_runtime::config::ParserVersion::V2,
                Some("muse_glimmer"),
                None,
            ),
            Some("muse_glimmer")
        );
        assert_eq!(
            selected_batch_family_for_selection(
                configured_family(Some("qwen3"), None),
                dynamo_runtime::config::ParserVersion::Auto,
                Some("qwen3"),
                None,
            ),
            None
        );
    }

    #[test]
    fn family_defaults_keep_deepseek_v41_on_v2_and_muse_on_original_modes() {
        const CHILD: &str = "DYNAMO_PARSER_FAMILY_DEFAULTS_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = crate::test_utils::isolated_command(
                "protocols::openai::chat_completions::unified_parser::tests::family_defaults_keep_deepseek_v41_on_v2_and_muse_on_original_modes",
            )
            .arg("--test-threads=1")
            .env(CHILD, "1")
            .env_remove(dynamo_runtime::config::environment_names::llm::DYN_PARSER_VERSION)
            .env_remove(
                dynamo_runtime::config::environment_names::llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2,
            )
            .output()
            .expect("run isolated family-default routing test");
            crate::test_utils::assert_isolated_success(&output);
            return;
        }

        let (tool, reasoning) = (Some("deepseek_v41"), Some("deepseek_v41"));
        let expected = configured_family(tool, reasoning);
        assert!(expected.is_some());
        assert_eq!(selected_family(tool, reasoning), expected);
        assert_eq!(selected_batch_family(tool, reasoning), expected);
        for (tool, reasoning) in [(Some("muse_glimmer"), None), (None, Some("muse_glimmer"))] {
            assert!(configured_family(tool, reasoning).is_some());
            assert_eq!(selected_family(tool, reasoning), None);
            assert_eq!(selected_batch_family(tool, reasoning), Some("muse_glimmer"));
        }
        for (tool, reasoning) in [
            (Some("qwen3_coder"), Some("qwen3")),
            (Some("deepseek_v41"), None),
        ] {
            assert_eq!(selected_family(tool, reasoning), None);
            assert_eq!(selected_batch_family(tool, reasoning), None);
        }
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
                    QWEN3_UNIFIED_FAMILY,
                    "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
                    &constraint,
                ),
                UnifiedToolOutputMode::Native,
            );
        }
        assert_eq!(
            batch_tool_output_mode(
                QWEN3_UNIFIED_FAMILY,
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
        assert_eq!(
            detect_prefill("kimi_k3", "answer").unwrap(),
            UnifiedParserStartingState::None
        );
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
    fn apostrophe_handling_preserves_quotes_contractions_and_possessives() {
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
        assert!(
            contains_unquoted_marker("'twas didn't </think> hidden", "</think>"),
            "an apostrophe inside a later contraction must not close a leading contraction"
        );
        assert!(
            !contains_unquoted_marker("'The literal </think> marker' stays visible", "</think>"),
            "a leading quote before a word must shield markers through its closing quote"
        );
        assert!(
            !contains_unquoted_marker("'The word doesn't contain </think> as syntax'", "</think>"),
            "an internal contraction must not close a quoted span"
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

        let mut continuation = chunk(
            "all>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>answer",
            true,
        );
        let continuation_data = continuation.data.as_mut().unwrap();
        continuation_data.nvext = Some(serde_json::json!({"source": "continuation"}));
        continuation.id = Some("continuation-id".to_string());
        continuation.event = Some("continuation-event".to_string());
        continuation.comment = Some(vec!["continuation-comment".to_string()]);

        let responses = apply_stream(
            stream::iter([chunk("before <tool_c", false), reasoning, continuation]),
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
                        choice.delta.reasoning_content.as_deref() == Some("held thought")
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
                prompt_logprobs: None,
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
                prompt_logprobs: None,
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

    #[tokio::test]
    async fn unmatched_quote_does_not_hide_real_reasoning_close_at_any_stream_split() {
        let raw = r#"<think>I considered "Paris.</think>The answer is Paris."#;
        let batch =
            parse_complete(QWEN3_UNIFIED_FAMILY, raw, &GuidedToolConstraint::None, &[]).unwrap();
        assert_eq!(batch.reasoning, r#"I considered "Paris."#);
        assert_eq!(batch.text, "The answer is Paris.");

        for split in 0..=raw.len() {
            let responses = apply_stream(
                stream::iter([chunk(&raw[..split], false), chunk(&raw[split..], true)]),
                None,
                None,
                false,
                UnifiedParserStartingState::None,
                QWEN3_UNIFIED_FAMILY,
            )
            .collect::<Vec<_>>()
            .await;
            let choices = collect_choices(&responses);
            let reasoning = choices
                .iter()
                .filter_map(|choice| choice.delta.reasoning_content.as_deref())
                .collect::<String>();
            let text = choices
                .iter()
                .filter_map(|choice| choice.delta.content.as_ref())
                .filter_map(|content| match content {
                    ChatCompletionMessageContent::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            assert_eq!(reasoning, batch.reasoning, "split {split}");
            assert_eq!(text, batch.text, "split {split}");
        }
    }

    #[tokio::test]
    async fn structured_json_then_reasoning_close_matches_batch_and_streaming() {
        let policy = UnifiedRequestPolicy {
            structured_response: true,
            ..Default::default()
        };
        let raw = r#"{"plan":"check"}</think>{"answer":42}"#;
        let batch = parse_complete_with_policy(
            QWEN3_UNIFIED_FAMILY,
            raw,
            &GuidedToolConstraint::None,
            &[],
            policy,
        )
        .unwrap();
        assert_eq!(batch.reasoning, r#"{"plan":"check"}"#);
        assert_eq!(batch.text, r#"{"answer":42}"#);

        for split in 0..=raw.len() {
            let responses = apply_stream_with_policy(
                stream::iter([chunk(&raw[..split], false), chunk(&raw[split..], true)]),
                None,
                GuidedToolConstraint::None,
                UnifiedParserStartingState::Reasoning,
                QWEN3_UNIFIED_FAMILY,
                false,
                policy,
            )
            .collect::<Vec<_>>()
            .await;
            let choices = collect_choices(&responses);
            let reasoning = choices
                .iter()
                .filter_map(|choice| choice.delta.reasoning_content.as_deref())
                .collect::<String>();
            let text = choices
                .iter()
                .filter_map(|choice| choice.delta.content.as_ref())
                .filter_map(|content| match content {
                    ChatCompletionMessageContent::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            assert_eq!(reasoning, batch.reasoning, "split {split}");
            assert_eq!(text, batch.text, "split {split}");
        }
    }

    // TODO: Extend early-progress checks to literal native tool-call opener cases.
    #[test]
    fn quoted_marker_in_tool_arguments_releases_following_content_before_close() {
        assert_quoted_marker_tool_arguments(true);
    }

    #[test]
    fn quoted_marker_tool_arguments_preserve_completed_output_at_every_split() {
        assert_quoted_marker_tool_arguments(false);
    }

    fn assert_quoted_marker_tool_arguments(require_early_progress: bool) {
        let tools = vec![Tool {
            name: "write_file".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }),
            strict: None,
        }];
        let body = "a".repeat(8192);
        for (mode, prefix, suffix) in [
            (
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("write_file".to_string()),
                },
                r#"{"path":"x","content":"prefix <think>"#,
                r#" suffix"}"#,
            ),
            (
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
                r#"[{"name":"write_file","arguments":{"path":"x","content":"prefix <think>"#,
                r#" suffix"}}]"#,
            ),
            (
                UnifiedToolOutputMode::Native,
                r#"<tool_call><function=write_file><parameter=path>x</parameter><parameter=content>"prefix <think>"#,
                r#" suffix"</parameter></function></tool_call>"#,
            ),
        ] {
            let mut state = ChoiceState::new(
                QWEN3_UNIFIED_FAMILY,
                &tools,
                UnifiedParserStartingState::Response,
                mode.clone(),
                true,
            )
            .unwrap();
            let mut events = state.push(prefix);
            let progress = state.push(&body);
            let arguments: String = progress
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::ToolCall(call) => Some(call.arguments.as_str()),
                    _ => None,
                })
                .collect();
            if require_early_progress {
                assert!(
                    arguments.contains(&body),
                    "{mode:?}: argument body must stream before its quote closes"
                );
            }
            events.extend(progress);
            events.extend(state.push(suffix));
            events.extend(state.finish());

            let mut batch = create_unified_parser_for_family(QWEN3_UNIFIED_FAMILY, &tools).unwrap();
            let init = UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                tool_output_mode: mode.clone(),
                invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
                ..Default::default()
            };
            batch.initialize_request(init.clone()).unwrap();
            let raw = format!("{prefix}{body}{suffix}");
            let completed = dynamo_parsers_v2::assemble(&events);
            assert_eq!(
                completed,
                vec![UnifiedEvent::ToolCall {
                    name: "write_file".to_string(),
                    arguments: serde_json::json!({
                        "path": "x",
                        "content": if matches!(mode, UnifiedToolOutputMode::Native) {
                            // XML parameter quotes are payload; JSON string quotes are syntax.
                            format!("\"prefix <think>{body} suffix\"")
                        } else {
                            format!("prefix <think>{body} suffix")
                        },
                    }),
                }],
                "{mode:?}: completed arguments must preserve every body byte"
            );
            assert_eq!(completed, batch.parse_complete(&raw).unwrap());

            let short_raw = format!(r#"{prefix}é \"quoted\" <think>{suffix}"#);
            batch.reset();
            batch.initialize_request(init).unwrap();
            let expected = batch.parse_complete(&short_raw).unwrap();
            for split in (0..=short_raw.len()).filter(|split| short_raw.is_char_boundary(*split)) {
                let mut split_state = ChoiceState::new(
                    QWEN3_UNIFIED_FAMILY,
                    &tools,
                    UnifiedParserStartingState::Response,
                    mode.clone(),
                    true,
                )
                .unwrap();
                let mut split_events = split_state.push(&short_raw[..split]);
                split_events.extend(split_state.push(&short_raw[split..]));
                split_events.extend(split_state.finish());
                assert_eq!(
                    dynamo_parsers_v2::assemble(&split_events),
                    expected,
                    "{mode:?} split {split}"
                );
            }
        }
    }

    #[test]
    fn repeated_quoted_markers_are_excluded_from_prefill_classification() {
        let literal = "The literal \"<think>private</think>\" stays visible. ".repeat(1024);
        let mut shield = QuotedControlMarkerProbe::new(QWEN3_UNIFIED_FAMILY);
        let mut masked = shield.push(&literal);
        masked.push_str(&shield.finish());
        assert!(!masked.contains("</think>"));
        assert_eq!(masked.len(), literal.len());
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

    async fn assert_quoted_literal_streams_as_text(family: &'static str, literal: &str) {
        for split in literal
            .char_indices()
            .map(|(offset, _)| offset)
            .chain(std::iter::once(literal.len()))
        {
            let responses = apply_stream(
                stream::iter([
                    chunk(&literal[..split], false),
                    chunk(&literal[split..], false),
                ]),
                None,
                None,
                false,
                UnifiedParserStartingState::None,
                family,
            )
            .collect::<Vec<_>>()
            .await;
            let choices = collect_choices(&responses);
            let streamed = choices
                .iter()
                .filter_map(|choice| choice.delta.content.as_ref())
                .filter_map(|content| match content {
                    ChatCompletionMessageContent::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            let reasoning = choices
                .iter()
                .filter_map(|choice| choice.delta.reasoning_content.as_deref())
                .collect::<String>();
            assert_eq!(streamed, literal, "{family} split {split}");
            assert!(
                reasoning.is_empty(),
                "{family} split {split}: {reasoning:?}"
            );
            assert!(
                choices
                    .iter()
                    .all(|choice| choice.delta.tool_calls.is_none()),
                "{family} split {split}: {choices:?}"
            );
        }
    }

    // TODO: Cover literal native tool-call opener text separately from reasoning controls.
    #[tokio::test]
    async fn quoted_control_markers_stay_visible_in_batch_and_streaming() {
        assert_quoted_control_markers(true, true).await;
    }

    #[tokio::test]
    async fn quoted_control_markers_stay_visible_in_batch() {
        assert_quoted_control_markers(false, false).await;
    }

    // TODO: Cover literal native tool-call opener text in native-channel batch responses.
    #[tokio::test]
    async fn native_channel_quoted_control_markers_stay_visible_in_batch() {
        assert_quoted_control_markers(false, true).await;
    }

    async fn assert_quoted_control_markers(check_streaming: bool, include_deferred: bool) {
        for family in [
            "deepseek_v4",
            "deepseek_v41",
            "qwen3",
            "glm47",
            "kimi_k2",
            "gemma4",
            "kimi_k3",
            "muse_glimmer",
        ] {
            if matches!(family, "gemma4" | "kimi_k3" | "muse_glimmer") && !include_deferred {
                continue;
            }
            let (opener, closer) = match family {
                "kimi_k3" => ("<|open|>think<|sep|>", "<|close|>think<|sep|>"),
                "muse_glimmer" => ("<|start|>assistant to=self<|message|>", "<|eom|>"),
                _ => reasoning_marker_pair(family).unwrap(),
            };
            for quote in ['"', '\'', '`'] {
                let literal =
                    format!("{quote}The literal {opener}private{closer}{quote} stays visible.");
                let parsed =
                    parse_complete(family, &literal, &GuidedToolConstraint::None, &[]).unwrap();
                assert_eq!(parsed.text, literal, "{family} quote {quote}");
                assert!(
                    parsed.reasoning.is_empty(),
                    "{family} quote {quote}: {:?}",
                    parsed.reasoning
                );

                if check_streaming {
                    assert_quoted_literal_streams_as_text(family, &literal).await;
                }
            }
        }

        for quote in ['\'', '`'] {
            let literal =
                format!("The literal {quote}<think>private</think>{quote} stays visible.");
            let parsed = parse_complete(
                QWEN3_UNIFIED_FAMILY,
                &literal,
                &GuidedToolConstraint::None,
                &[],
            )
            .unwrap();
            assert_eq!(parsed.text, literal, "quote {quote}");
            assert!(parsed.reasoning.is_empty(), "quote {quote}");
            if check_streaming {
                assert_quoted_literal_streams_as_text(QWEN3_UNIFIED_FAMILY, &literal).await;
            }
        }
    }

    #[test]
    fn quoted_control_marker_probe_masks_closed_quotes_and_flushes_unclosed_quotes() {
        for family in [
            "deepseek_v4",
            "deepseek_v41",
            "gemma4",
            "glm47",
            "kimi_k2",
            "kimi_k3",
            "muse_glimmer",
            "qwen3",
        ] {
            let marker = unified_parser_control_markers(family)
                .into_iter()
                .next()
                .unwrap();
            for quote in ['"', '\'', '`'] {
                let literal = format!("The literal {quote}{marker}{quote} stays visible.");
                for split in literal
                    .char_indices()
                    .map(|(offset, _)| offset)
                    .chain(std::iter::once(literal.len()))
                {
                    let mut shield = QuotedControlMarkerProbe::new(family);
                    let mut masked = shield.push(&literal[..split]);
                    masked.push_str(&shield.push(&literal[split..]));
                    masked.push_str(&shield.finish());
                    assert!(
                        !masked.contains(&marker),
                        "{family} quote {quote} split {split}"
                    );
                    assert_eq!(masked.len(), literal.len(), "{family} split {split}");
                }
            }

            let literal = format!("The literal \"{marker}");
            let mut unfinished = QuotedControlMarkerProbe::new(family);
            let mut masked = unfinished.push(&literal);
            masked.push_str(&unfinished.finish());
            assert!(masked.contains(&marker), "{family} EOF");
            assert_eq!(masked, literal, "{family} EOF");
        }

        let literal = r#"The literal "escaped \" closer </think>" remains visible."#;
        for split in literal
            .char_indices()
            .map(|(offset, _)| offset)
            .chain(std::iter::once(literal.len()))
        {
            let mut shield = QuotedControlMarkerProbe::new(QWEN3_UNIFIED_FAMILY);
            let mut masked = shield.push(&literal[..split]);
            masked.push_str(&shield.push(&literal[split..]));
            masked.push_str(&shield.finish());
            assert!(!masked.contains("</think>"), "split {split}");
            assert_eq!(masked.len(), literal.len(), "split {split}");
        }
    }

    #[test]
    fn single_quoted_prose_shields_a_marker_after_its_opening_word() {
        for family in [
            "deepseek_v4",
            "deepseek_v41",
            "gemma4",
            "glm47",
            "kimi_k2",
            "kimi_k3",
            "muse_glimmer",
            "qwen3",
        ] {
            for marker in unified_parser_control_markers(family) {
                for (literal, should_be_shielded) in [
                    (
                        format!("'The literal {marker} marker' stays visible."),
                        true,
                    ),
                    (
                        format!("'The word doesn't contain {marker} as syntax'"),
                        true,
                    ),
                    (format!("'twas {marker} must remain unshielded"), false),
                ] {
                    for split in literal
                        .char_indices()
                        .map(|(offset, _)| offset)
                        .chain(std::iter::once(literal.len()))
                    {
                        let mut shield = QuotedControlMarkerProbe::new(family);
                        let mut masked = shield.push(&literal[..split]);
                        let second_output = shield.push(&literal[split..]);
                        masked.push_str(&second_output);
                        masked.push_str(&shield.finish());
                        assert_eq!(
                            masked.contains(&marker),
                            !should_be_shielded,
                            "{family} marker {marker:?} split {split}, literal {literal:?}, masked {masked:?}"
                        );
                        assert_eq!(masked.len(), literal.len(), "{family} split {split}");
                    }
                }
            }
        }
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
    async fn structured_response_json_stays_content_at_every_split_for_all_families_and_choices() {
        let policy = UnifiedRequestPolicy {
            structured_response: true,
            ..Default::default()
        };
        for prefill in [
            UnifiedParserStartingState::None,
            UnifiedParserStartingState::Reasoning,
        ] {
            for family in dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES {
                for raw in [
                    " \n{\"country\":\"France\"}",
                    "[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
                    "true",
                    "false",
                    "null",
                    "42",
                    "-1.25e2",
                    "\"literal <think> text\"",
                ] {
                    let batch = parse_complete_with_policy(
                        family,
                        raw,
                        &GuidedToolConstraint::None,
                        &weather_tools(),
                        policy,
                    )
                    .unwrap();
                    assert_eq!(batch.text, raw, "{family} batch");
                    assert!(batch.reasoning.is_empty());
                    assert!(batch.tool_calls.is_empty());
                    for split in 0..=raw.len() {
                        let mut other_prefix = chunk(" \n", false);
                        other_prefix.data.as_mut().unwrap().inner.choices[0].index = 1;
                        let mut other_body = chunk(raw, true);
                        other_body.data.as_mut().unwrap().inner.choices[0].index = 1;
                        let outputs: Vec<_> = apply_stream_with_policy(
                            futures::stream::iter([
                                chunk(&raw[..split], false),
                                other_prefix,
                                chunk(&raw[split..], true),
                                other_body,
                            ]),
                            Some(weather_tools()),
                            GuidedToolConstraint::None,
                            prefill,
                            family,
                            true,
                            policy,
                        )
                        .collect()
                        .await;
                        for index in [0, 1] {
                            let choices: Vec<_> = collect_choices(&outputs)
                                .into_iter()
                                .filter(|choice| choice.index == index)
                                .collect();
                            let text: String = choices
                                .iter()
                                .filter_map(|choice| match &choice.delta.content {
                                    Some(ChatCompletionMessageContent::Text(text)) => {
                                        Some(text.as_str())
                                    }
                                    _ => None,
                                })
                                .collect();
                            assert_eq!(
                                text,
                                if index == 0 {
                                    raw.to_string()
                                } else {
                                    format!(" \n{raw}")
                                },
                                "{family} split {split} choice {index}"
                            );
                            assert!(
                                choices.iter().all(|choice| choice
                                    .delta
                                    .reasoning_content
                                    .is_none()
                                    && choice.delta.tool_calls.is_none()),
                                "{family} split {split} choice {index}"
                            );
                        }
                    }
                    let outputs: Vec<_> = apply_stream_with_policy(
                        futures::stream::iter(
                            raw.chars()
                                .map(|c| chunk(&c.to_string(), false))
                                .collect::<Vec<_>>(),
                        ),
                        None,
                        GuidedToolConstraint::None,
                        prefill,
                        family,
                        true,
                        policy,
                    )
                    .collect()
                    .await;
                    let text: String = collect_choices(&outputs)
                        .iter()
                        .filter_map(|choice| match &choice.delta.content {
                            Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                            _ => None,
                        })
                        .collect();
                    assert_eq!(text, raw, "{family} EOF without finish");
                    assert!(
                        collect_choices(&outputs)
                            .iter()
                            .all(|choice| choice.delta.reasoning_content.is_none())
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn structured_unopened_channel_prefix_flush_and_progress() {
        let policy = UnifiedRequestPolicy {
            structured_response: true,
            ..Default::default()
        };
        for raw in [" \n", "\"unterminated <think>", "{", "[", "42"] {
            for ending in 0..4 {
                let mut input = vec![chunk(raw, false)];
                match ending {
                    0 => {} // Raw EOF.
                    1 => input.push(chunk("", true)),
                    2 => {
                        let mut usage = chunk("", false);
                        usage.data.as_mut().unwrap().inner.choices.clear();
                        input.push(usage);
                    }
                    _ => {
                        let mut parsed = chunk("", true);
                        let delta = &mut parsed.data.as_mut().unwrap().inner.choices[0].delta;
                        delta.content = None;
                        delta.reasoning_content = Some("upstream".into());
                        input.push(parsed);
                    }
                }
                let output = apply_stream_with_policy(
                    futures::stream::iter(input),
                    None,
                    GuidedToolConstraint::None,
                    UnifiedParserStartingState::None,
                    "qwen3",
                    true,
                    policy,
                )
                .collect::<Vec<_>>()
                .await;
                let choices = collect_choices(&output);
                let text: String = choices
                    .iter()
                    .filter_map(|c| match &c.delta.content {
                        Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                let reasoning: String = choices
                    .iter()
                    .filter_map(|c| c.delta.reasoning_content.as_deref())
                    .collect();
                assert_eq!(text, raw, "ending {ending}");
                assert_eq!(reasoning, if ending == 3 { "upstream" } else { "" });
            }
        }
        for raw in ["{", "[", "\"partial", "4", "t", "n", "f"] {
            let output = apply_stream_with_policy(
                futures::stream::iter([chunk(raw, false), chunk("", true)]),
                None,
                GuidedToolConstraint::None,
                UnifiedParserStartingState::None,
                "qwen3",
                true,
                policy,
            )
            .take(1)
            .collect::<Vec<_>>()
            .await;
            let text: String = collect_choices(&output)
                .iter()
                .filter_map(|c| match &c.delta.content {
                    Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(text, raw, "JSON prefix must progress before terminal");
        }
    }

    #[test]
    fn scalar_probe_inspects_each_byte_once_and_resolves_all_release_paths() {
        let raw = format!("\"{}\"", "x".repeat(65536));
        let mut probe = JsonPrefillProbe::default();
        for byte in raw.bytes() {
            probe.push(std::str::from_utf8(&[byte]).unwrap());
            assert!(probe.decision(false).is_none());
        }
        assert_eq!(probe.inspected_bytes, raw.len());
        assert_eq!(probe.decision(true), Some(UnifiedParserStartingState::None));
        let policy = UnifiedRequestPolicy {
            structured_response: true,
            ..Default::default()
        };
        let resolved =
            resolve_guided_prefix(7, probe, policy, UnifiedParserStartingState::Reasoning).unwrap();
        assert_eq!(
            resolved.delta.content,
            Some(ChatCompletionMessageContent::Text(raw))
        );
        assert!(resolved.delta.reasoning_content.is_none());
        for raw in ["42", "true", "null", "\"answer\""] {
            let mut probe = JsonPrefillProbe::default();
            probe.push(raw);
            assert!(probe.decision(false).is_none());
            let resolved =
                resolve_guided_prefix(0, probe, policy, UnifiedParserStartingState::Reasoning)
                    .unwrap();
            assert_eq!(
                resolved.delta.content,
                Some(ChatCompletionMessageContent::Text(raw.into()))
            );
        }
        let mut thought = JsonPrefillProbe::default();
        thought.push("42");
        assert!(thought.decision(false).is_none());
        thought.push(" reasons");
        assert_eq!(
            thought.decision(false),
            Some(UnifiedParserStartingState::Reasoning)
        );
        assert_eq!(thought.inspected_bytes, 10);
    }

    #[test]
    fn json_prefill_marker_detection_has_linear_work_for_quoted_controls() {
        for payload in ["|", "<", ">", "<think>", "</think>"] {
            let raw = format!("[\"{}\"]", payload.repeat(4096));
            let mut probe = JsonPrefillProbe::default();
            for character in raw.chars() {
                let text = character.to_string();
                probe.push(&text);
                assert_ne!(
                    probe
                        .decision_for_family(QWEN3_UNIFIED_FAMILY, false, &text)
                        .unwrap(),
                    Some(UnifiedParserStartingState::Reasoning),
                );
            }
            assert_eq!(
                probe
                    .decision_for_family(QWEN3_UNIFIED_FAMILY, true, "")
                    .unwrap(),
                Some(UnifiedParserStartingState::None),
            );
            let shield_work = probe.marker_probe.as_ref().unwrap().inspected_bytes;
            assert!(
                probe.inspected_bytes + probe.marker_inspected_bytes + shield_work
                    <= raw.len() * 64,
                "payload {payload:?}: JSON {}, marker {}, shield {}, input {}",
                probe.inspected_bytes,
                probe.marker_inspected_bytes,
                shield_work,
                raw.len(),
            );
        }
    }

    #[test]
    fn structured_scalar_prefix_keeps_real_reasoning_before_json() {
        for thought in ["42 reasons", "\"a thought\"", "true", "null hypothesis"] {
            let full = format!("{thought}</think>\"answer\"");
            for split in 0..=full.len() {
                let prefix = &full[..split];
                if prefix.contains("</think>") {
                    assert_eq!(
                        structured_json_prefill(QWEN3_UNIFIED_FAMILY, prefix, false).unwrap(),
                        Some(UnifiedParserStartingState::Reasoning),
                        "split {split}"
                    );
                }
            }
            assert_eq!(
                structured_json_prefill(QWEN3_UNIFIED_FAMILY, &full, true).unwrap(),
                Some(UnifiedParserStartingState::Reasoning)
            );
        }
    }

    #[tokio::test]
    async fn structured_response_literal_native_markers_match_batch_and_stream() {
        let policy = UnifiedRequestPolicy {
            structured_response: true,
            ..Default::default()
        };
        let tools = weather_tools();
        let literal_marker_json = r#"{"note":"<think>literal</think>"}"#;
        let batch = parse_complete_with_policy(
            "qwen3",
            literal_marker_json,
            &GuidedToolConstraint::None,
            &tools,
            policy,
        )
        .unwrap();
        assert_eq!(batch.text, literal_marker_json);
        assert!(batch.reasoning.is_empty());

        let streamed = apply_stream_with_policy(
            futures::stream::iter(vec![chunk(literal_marker_json, true)]),
            Some(tools.clone()),
            GuidedToolConstraint::None,
            stream_prefill("qwen3", false),
            "qwen3",
            false,
            policy,
        )
        .collect::<Vec<_>>()
        .await;
        let streamed_choices = collect_choices(&streamed);
        let streamed_text: String = streamed_choices
            .iter()
            .filter_map(|choice| match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let streamed_reasoning: String = streamed_choices
            .iter()
            .filter_map(|choice| choice.delta.reasoning_content.as_deref())
            .collect();
        assert_eq!(streamed_text, literal_marker_json);
        assert!(streamed_reasoning.is_empty());
    }

    #[tokio::test]
    async fn structured_response_reasoning_prefill_starting_with_container_matches_batch_and_stream()
     {
        let policy = UnifiedRequestPolicy {
            structured_response: true,
            ..Default::default()
        };
        let tools = weather_tools();
        let reasoning_then_json = "{private thought}</think>{\"answer\":42}";
        let batch = parse_complete_with_policy(
            "qwen3",
            reasoning_then_json,
            &GuidedToolConstraint::None,
            &tools,
            policy,
        )
        .unwrap();
        assert_eq!(batch.reasoning, "{private thought}");
        assert_eq!(batch.text, "{\"answer\":42}");

        let streamed = apply_stream_with_policy(
            futures::stream::iter(vec![chunk(reasoning_then_json, true)]),
            Some(tools.clone()),
            GuidedToolConstraint::None,
            stream_prefill("qwen3", true),
            "qwen3",
            false,
            policy,
        )
        .collect::<Vec<_>>()
        .await;
        let streamed_choices = collect_choices(&streamed);
        let streamed_text: String = streamed_choices
            .iter()
            .filter_map(|choice| match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let streamed_reasoning: String = streamed_choices
            .iter()
            .filter_map(|choice| choice.delta.reasoning_content.as_deref())
            .collect();
        assert_eq!(
            streamed_reasoning, "{private thought}",
            "streamed content was {streamed_text:?}"
        );
        assert_eq!(streamed_text, "{\"answer\":42}");
    }

    #[test]
    fn malformed_guided_zero_call_native_channels_recover_for_every_family() {
        for (family, raw) in [
            ("deepseek_v4", "<think>private</think>answer"),
            ("deepseek_v41", "<think>private</think>answer"),
            ("qwen3", "<think>private</think>answer"),
            ("glm47", "<think>private</think>answer"),
            ("kimi_k2", "<think>private</think>answer"),
            ("gemma4", "<|channel>thought\nprivate<channel|>answer"),
            (
                "kimi_k3",
                "<|open|>think<|sep|>private<|close|>think<|sep|><|open|>response<|sep|>answer<|close|>response<|sep|>",
            ),
            (
                "muse_glimmer",
                "<|start|>assistant to=self<|message|>private<|eom|><|start|>assistant to=user<|message|>answer<|eot|>",
            ),
        ] {
            for constraint in [
                GuidedToolConstraint::GuidedJsonRequired,
                GuidedToolConstraint::GuidedJsonNamed {
                    tool_name: "get_weather".to_string(),
                },
            ] {
                let batch = parse_complete_with_policy(
                    family,
                    raw,
                    &constraint,
                    &weather_tools(),
                    UnifiedRequestPolicy::default(),
                )
                .unwrap();
                assert_eq!(batch.reasoning, "private", "{family} {constraint:?}");
                assert_eq!(batch.text, "answer", "{family} {constraint:?}");
                assert!(batch.tool_calls.is_empty());
                let quoted = serde_json::to_string(raw).unwrap();
                assert!(
                    !contains_native_envelope(&quoted, family),
                    "{family} quoted control"
                );
                assert!(
                    parse_complete_with_policy(
                        family,
                        &quoted,
                        &constraint,
                        &weather_tools(),
                        UnifiedRequestPolicy::default()
                    )
                    .is_err(),
                    "{family} malformed JSON string must keep its error"
                );
            }
        }
    }

    #[tokio::test]
    async fn disabled_gemma_preserves_thought_delimiters_around_native_calls_at_every_split() {
        let prefix = "<|channel>thought\nbefore<channel|>";
        let suffix = "after<|channel>thought\nlater<channel|>done";
        let raw = format!(
            "{prefix}<|tool_call>call:get_weather{{city:<|\"|>Paris<|\"|>}}<tool_call|>{suffix}"
        );
        let policy = UnifiedRequestPolicy {
            reasoning_disabled: true,
            ..Default::default()
        };
        let batch = parse_complete_with_policy(
            "gemma4",
            &raw,
            &GuidedToolConstraint::None,
            &weather_tools(),
            policy,
        )
        .unwrap();
        assert_eq!(batch.text, format!("{prefix}{suffix}"));
        assert!(batch.reasoning.is_empty());
        assert_eq!(batch.tool_calls.len(), 1);
        assert_eq!(batch.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&batch.tool_calls[0].function.arguments)
                .unwrap(),
            serde_json::json!({"city":"Paris"})
        );
        for split in 0..=raw.len() {
            let outputs: Vec<_> = apply_stream_with_policy(
                futures::stream::iter([chunk(&raw[..split], false), chunk(&raw[split..], true)]),
                Some(weather_tools()),
                GuidedToolConstraint::None,
                UnifiedParserStartingState::None,
                "gemma4",
                true,
                policy,
            )
            .collect()
            .await;
            let choices = collect_choices(&outputs);
            let text: String = choices
                .iter()
                .filter_map(|choice| match &choice.delta.content {
                    Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(text, format!("{prefix}{suffix}"), "split {split}");
            assert!(
                choices
                    .iter()
                    .all(|choice| choice.delta.reasoning_content.is_none()),
                "split {split}"
            );
            assert_eq!(
                choices
                    .iter()
                    .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
                    .filter(|call| call.id.is_some())
                    .count(),
                1,
                "split {split}"
            );
            let first_call = choices
                .iter()
                .position(|choice| choice.delta.tool_calls.is_some())
                .unwrap();
            let before_call: String = choices[..first_call]
                .iter()
                .filter_map(|choice| match &choice.delta.content {
                    Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(before_call, prefix, "split {split}: call ordering");
        }
    }
}
