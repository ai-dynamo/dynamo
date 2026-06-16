// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use crate::ParserResult;
use crate::ReasoningParser;

use openai_harmony::StreamableParser;
use openai_harmony::chat::{Content, Message, TextContent};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, chat::Role, load_harmony_encoding};
use regex::Regex;

///// Static initialization of harmony encoder to not affect performance every time a parser is created
/// This is because load_harmony_encoding downloads some tiktoken files into a directory and we don't want to do this every time we create a parser.
use std::sync::OnceLock;

static GLOBAL_HARMONY_GPTOSS_ENCODING: OnceLock<Result<HarmonyEncoding, anyhow::Error>> =
    OnceLock::new();
static HARMONY_CALL_TOKEN_IDS: OnceLock<Vec<u32>> = OnceLock::new();
static HARMONY_ANALYSIS_BLOCK_REGEX: OnceLock<Regex> = OnceLock::new();
const HARMONY_CALL_MARKER: &str = "<|call|>";
const HARMONY_SPECIAL_TOKENS: &[&str] = &[
    "<|start|>",
    "<|end|>",
    "<|return|>",
    "<|channel|>",
    "<|constrain|>",
    "<|message|>",
    "<|call|>",
];

fn get_harmony_encoding() -> &'static Result<HarmonyEncoding, anyhow::Error> {
    GLOBAL_HARMONY_GPTOSS_ENCODING.get_or_init(|| {
        // load_harmony_encoding internally constructs a reqwest::blocking::Client,
        // which builds and drops a Tokio Runtime. Dropping a Runtime from inside
        // an async context (e.g. when this is called for the first time during
        // an HTTP request handler) panics with "Cannot drop a runtime in a
        // context where blocking is not allowed". Run the load on a fresh OS
        // thread so the inner Runtime is dropped outside any async context.
        // The init runs at most once per process via OnceLock.
        std::thread::spawn(|| load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss))
            .join()
            .unwrap_or_else(|_| Err(anyhow::anyhow!("harmony encoding loader thread panicked")))
    })
}

pub struct GptOssReasoningParser {
    parser: StreamableParser,
    pending_text: String,
    pending_tool_call_text: String,
    emitted_reasoning_text: bool,
    insert_reasoning_separator: bool,
    emitted_normal_text: bool,
    insert_normal_separator: bool,
    /// Channel/recipient of the directed tool call currently being streamed.
    /// Persisted across chunks because the `<|call|>` terminator can arrive in a
    /// later chunk than the one that resolved the recipient (and `<|call|>` resets
    /// the live parser state), so the `<|call|>`-chunk envelope reconstruction
    /// would otherwise lose the channel/recipient. Cleared once the call is emitted.
    last_directed_channel: Option<String>,
    last_directed_recipient: Option<String>,
}

/// Implement Debug for GptOssReasoningParser separately because StreamableParser does not implement Debug
impl Debug for GptOssReasoningParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GptOssReasoningParser")
            .field("parser", &self.parser.state_json())
            .finish()
    }
}

impl GptOssReasoningParser {
    pub fn new() -> anyhow::Result<Self> {
        let parser = match get_harmony_encoding().as_ref() {
            Ok(enc) => match StreamableParser::new(enc.clone(), Some(Role::Assistant)) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("Harmony StreamableParser init failed for GPT OSS: {e}");
                    return Err(anyhow::anyhow!(
                        "Failed to load Harmony StreamableParser: {e}"
                    ));
                }
            },
            Err(e) => {
                tracing::warn!("Failed to load Harmony encoding for GPT OSS: {e}");
                return Err(anyhow::anyhow!("Failed to load Harmony encoding: {e}"));
            }
        };
        Ok(Self {
            parser,
            pending_text: String::new(),
            pending_tool_call_text: String::new(),
            emitted_reasoning_text: false,
            insert_reasoning_separator: false,
            emitted_normal_text: false,
            insert_normal_separator: false,
            last_directed_channel: None,
            last_directed_recipient: None,
        })
    }
}

fn encode_text_to_tokens(text: &str) -> anyhow::Result<Vec<u32>> {
    let enc = get_harmony_encoding()
        .as_ref()
        .map_err(|e| anyhow::anyhow!("Failed to get harmony encoding: {e}"))?;
    Ok(enc.tokenizer().encode_with_special_tokens(text))
}

fn harmony_call_token_ids() -> Option<&'static [u32]> {
    let ids = HARMONY_CALL_TOKEN_IDS.get_or_init(|| {
        get_harmony_encoding()
            .as_ref()
            .map(|enc| {
                enc.tokenizer()
                    .encode_with_special_tokens(HARMONY_CALL_MARKER)
            })
            .unwrap_or_default()
    });
    if ids.is_empty() {
        None
    } else {
        Some(ids.as_slice())
    }
}

/// Token id(s) for the harmony `<|call|>` terminator.
///
/// `<|call|>` is simultaneously the harmony tool-call terminator the parser
/// requires AND a gpt-oss EOS token. Pipelines that hide EOS tokens from the
/// decoded output (e.g. the preprocessor's stop-condition handling) must keep
/// these ids visible when the harmony tool-call parser is active — otherwise the
/// terminator is stripped before the parser sees it and tool calls are silently
/// dropped. Empty if the harmony encoding is unavailable.
pub fn harmony_terminator_token_ids() -> Vec<u32> {
    harmony_call_token_ids()
        .map(|ids| ids.to_vec())
        .unwrap_or_default()
}

fn token_ids_contain_sequence(token_ids: &[u32], marker_ids: &[u32]) -> bool {
    !marker_ids.is_empty()
        && marker_ids.len() <= token_ids.len()
        && token_ids
            .windows(marker_ids.len())
            .any(|window| window == marker_ids)
}

fn harmony_analysis_block_regex() -> &'static Regex {
    HARMONY_ANALYSIS_BLOCK_REGEX.get_or_init(|| {
        // Match ONLY recipientless analysis reasoning blocks (the CoT channel).
        // Requiring `<|message|>` to immediately follow `analysis` (no `.*?`
        // gap) ensures that `analysis to=functions.X code<|message|>…` tool-call
        // envelopes are NOT stripped — they must survive into normal_text so
        // the downstream harmony tool-call jail can recover them.
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>analysis<\|message\|>.*?(?:<\|call\|>|<\|end\|>|\z)",
        )
            .expect("harmony analysis block regex")
    })
}

fn input_contains_call_marker(
    text: &str,
    token_ids: &[u32],
    caller_supplied_token_ids: bool,
) -> bool {
    if !token_ids.is_empty()
        && let Some(call_ids) = harmony_call_token_ids()
    {
        return token_ids_contain_sequence(token_ids, call_ids);
    }

    !caller_supplied_token_ids && text.contains(HARMONY_CALL_MARKER)
}

fn raw_input_text_for_tool_parser(
    text: &str,
    token_ids: &[u32],
    caller_supplied_token_ids: bool,
) -> String {
    if caller_supplied_token_ids && let Ok(enc) = get_harmony_encoding() {
        return enc.tokenizer().decode_utf8(token_ids).unwrap_or_default();
    }

    text.to_string()
}

fn split_incomplete_harmony_suffix(text: &str) -> (&str, &str) {
    let hold_len = HARMONY_SPECIAL_TOKENS
        .iter()
        .flat_map(|token| (1..token.len()).map(|idx| &token[..idx]))
        .filter(|prefix| text.ends_with(*prefix))
        .map(str::len)
        .max()
        .unwrap_or(0);

    text.split_at(text.len() - hold_len)
}

fn contains_directed_harmony_function_call(text: &str) -> bool {
    (text.contains("<|channel|>commentary to=functions.")
        || text.contains("<|channel|>analysis to=functions."))
        && text.contains(HARMONY_CALL_MARKER)
}

fn strip_analysis_blocks_for_tool_handoff(text: &str) -> String {
    harmony_analysis_block_regex()
        .replace_all(text, "")
        .into_owned()
}

fn append_separated(target: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    if !target.is_empty() {
        target.push('\n');
    }
    target.push_str(text);
}

fn append_text_content(target: &mut String, content: &[Content]) {
    for item in content {
        if let Content::Text(TextContent { text }) = item {
            append_separated(target, text);
        }
    }
}

fn append_message_by_channel(reasoning_text: &mut String, normal_text: &mut String, msg: &Message) {
    match msg.channel.as_deref() {
        // Analysis WITHOUT a recipient is chain-of-thought reasoning.
        // Analysis WITH a functions recipient is a (malformed) tool call whose
        // payload is handed off via strip_analysis_blocks_for_tool_handoff —
        // do not also surface it as reasoning_content.
        Some("analysis") if msg.recipient.is_none() => {
            append_text_content(reasoning_text, &msg.content)
        }
        Some("final") => append_text_content(normal_text, &msg.content),
        Some("commentary") if msg.recipient.is_none() => {
            append_text_content(normal_text, &msg.content)
        }
        _ => {}
    }
}

fn append_current_by_channel(
    reasoning_text: &mut String,
    normal_text: &mut String,
    channel: Option<String>,
    current: String,
) {
    match channel.as_deref() {
        Some("analysis") => append_separated(reasoning_text, &current),
        Some("final") => append_separated(normal_text, &current),
        Some("commentary") => append_separated(normal_text, &current),
        _ => {}
    }
}

fn is_visible_normal_channel(channel: Option<&str>, recipient: Option<&str>) -> bool {
    matches!(channel, Some("final"))
        || (matches!(channel, Some("commentary")) && recipient.is_none())
}

fn starts_directed_harmony_tool_call(text: &str) -> bool {
    text.contains("<|channel|>commentary to=functions.")
        || text.contains("<|channel|>analysis to=functions.")
}

/// Reconstruct the complete harmony envelope for the directed tool call that
/// just terminated, by decoding the cumulative parser token buffer from its
/// last `<|channel|>` token to the end.
///
/// This is the canonical handoff to the downstream tool-call jail: it always
/// starts exactly at the tool call's `<|channel|>` token, so it (a) survives a
/// channel header that was split across streaming chunks, and (b) excludes any
/// leading `<|end|>` / text from a preceding reasoning message that raw
/// per-chunk accumulation would otherwise prepend (and leak).
///
/// Returns `None` when the captured channel is not a directed tool call, when
/// the harmony encoding is unavailable, or when no `<|channel|>` token is found.
fn reconstruct_directed_envelope(
    parser: &StreamableParser,
    channel: Option<&str>,
    recipient: Option<&str>,
) -> Option<String> {
    let ch = channel?;
    let is_tool_ch = ch == "commentary"
        || (ch == "analysis" && recipient.is_some_and(|r| r.starts_with("functions.")));
    if !is_tool_ch {
        return None;
    }
    let Ok(enc) = get_harmony_encoding() else {
        return None;
    };
    let tokens = parser.tokens();
    let channel_token_id = enc
        .tokenizer()
        .encode_with_special_tokens("<|channel|>")
        .into_iter()
        .next()?;
    let last_channel_idx = tokens.iter().rposition(|t| *t == channel_token_id)?;
    let decoded = enc
        .tokenizer()
        .decode_utf8(&tokens[last_channel_idx..])
        .unwrap_or_default();
    if decoded.is_empty() {
        None
    } else {
        Some(decoded)
    }
}

impl ReasoningParser for GptOssReasoningParser {
    fn finish_reasoning_stream(&mut self) -> ParserResult {
        self.pending_tool_call_text.clear();
        self.last_directed_channel = None;
        self.last_directed_recipient = None;
        let pending = std::mem::take(&mut self.pending_text);
        if pending.is_empty() {
            return ParserResult::default();
        }

        match self.parser.current_channel().as_deref() {
            Some("analysis") => ParserResult {
                normal_text: String::new(),
                reasoning_text: pending,
            },
            Some("final") | Some("commentary") => ParserResult {
                normal_text: pending,
                reasoning_text: String::new(),
            },
            _ => ParserResult::default(),
        }
    }

    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult {
        let caller_supplied_token_ids = !token_ids.is_empty();
        let encoded_tokens;
        let token_ids = if token_ids.is_empty() {
            // WAR: Since we are moving to just text based reasoning parsing, converting to token_ids now using harmony encoding
            encoded_tokens = match encode_text_to_tokens(text) {
                Ok(tokens) => tokens,
                Err(err) => {
                    tracing::warn!("Failed to encode Harmony tokens: {err}");
                    return ParserResult::default();
                }
            };
            encoded_tokens.as_slice()
        } else {
            token_ids
        };

        let parser = &mut self.parser;

        for (i, token_id) in token_ids.iter().enumerate() {
            tracing::debug!(
                "Processing token {} of {}: {}",
                i + 1,
                token_ids.len(),
                token_id
            );
            if let Err(e) = parser.process(*token_id) {
                tracing::warn!("Harmony parse error for token_id {token_id}: {e}");
                return ParserResult::default();
            }
        }

        let output_msgs = parser.messages();
        tracing::debug!("Parser has {} output messages", output_msgs.len());

        let mut reasoning_text = String::new();
        let mut normal_text = String::new();
        for msg in output_msgs {
            append_message_by_channel(&mut reasoning_text, &mut normal_text, msg);
        }

        let current = parser.current_content().unwrap_or_default();
        let current_channel = parser.current_channel();
        if current_channel.as_deref() != Some("commentary") || parser.current_recipient().is_none()
        {
            append_current_by_channel(
                &mut reasoning_text,
                &mut normal_text,
                current_channel,
                current,
            );
        }

        let raw_input_text =
            raw_input_text_for_tool_parser(text, token_ids, caller_supplied_token_ids);
        if contains_directed_harmony_function_call(&raw_input_text) {
            // The serving pipeline feeds reasoning normal_text into the Harmony
            // tool parser. Keep directed commentary envelopes intact so tool
            // calls are not silently dropped.
            normal_text = strip_analysis_blocks_for_tool_handoff(&raw_input_text);
        }

        tracing::debug!(
            "Final result - normal_text: {} chars, reasoning_text: {} chars",
            normal_text.len(),
            reasoning_text.len()
        );

        ParserResult {
            normal_text,
            reasoning_text,
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        let caller_supplied_token_ids = !token_ids.is_empty();
        let text_to_process;
        let text = if caller_supplied_token_ids {
            text
        } else {
            self.pending_text.push_str(text);
            let combined = std::mem::take(&mut self.pending_text);
            let (ready, pending) = split_incomplete_harmony_suffix(&combined);
            self.pending_text.push_str(pending);
            text_to_process = ready.to_string();
            text_to_process.as_str()
        };

        if text.is_empty() && token_ids.is_empty() {
            return ParserResult::default();
        }

        let encoded_tokens;
        let token_ids = if token_ids.is_empty() {
            // WAR: Since we are moving to just text based reasoning parsing, converting to token_ids now using harmony encoding
            encoded_tokens = match encode_text_to_tokens(text) {
                Ok(tokens) => tokens,
                Err(err) => {
                    tracing::warn!("Failed to encode Harmony tokens: {err}");
                    return ParserResult::default();
                }
            };
            encoded_tokens.as_slice()
        } else {
            token_ids
        };

        let parser: &mut StreamableParser = &mut self.parser;
        let mut normal_delta = String::new();
        let mut reasoning_delta = String::new();
        // Tracks whether at least one directed call envelope was reconstructed inside
        // the per-token loop below. When true the post-loop `has_call_marker` branch
        // must not reconstruct again (it would pick the wrong `<|channel|>` token
        // because the cumulative buffer by then includes post-`<|call|>` tokens).
        let mut call_reconstructed_in_loop = false;

        for (i, token_id) in token_ids.iter().enumerate() {
            tracing::debug!(
                "Processing streaming token {} of {}: {}",
                i + 1,
                token_ids.len(),
                token_id
            );
            let previous_channel = parser.current_channel();
            let previous_recipient = parser.current_recipient();
            if let Err(e) = parser.process(*token_id) {
                tracing::warn!("Harmony parse error for token_id {token_id}: {e}");
                return ParserResult::default();
            }
            let current_channel = parser.current_channel();
            let current_recipient = parser.current_recipient();

            // Persist directed tool-call context before `<|call|>` can reset it.
            // This is stored on `self` (not a local) because the `<|call|>`
            // terminator can arrive in a later chunk than the one that resolved
            // the recipient — e.g. for short responses split into tiny chunks the
            // `<|call|>` token can be the first token of its own chunk, by which
            // point the live parser recipient is already gone.
            if let (Some(ch), Some(rec)) = (&current_channel, &current_recipient)
                && (ch.as_str() == "commentary" || ch.as_str() == "analysis")
                && rec.starts_with("functions.")
            {
                self.last_directed_channel = current_channel.clone();
                self.last_directed_recipient = current_recipient.clone();
            }

            // Reconstruct each directed tool-call envelope the moment `<|call|>`
            // resets the parser channel to None (detected by the directed→None
            // channel transition while `last_directed_channel` is set).
            //
            // Doing this here — not in the post-loop section — is critical: at this
            // point `parser.tokens()` ends exactly at the `<|call|>` token, so
            // `reconstruct_directed_envelope`'s `rposition` scan finds the correct
            // `<|channel|>` token for this call.  If we waited until after all
            // tokens in the chunk were processed, subsequent tokens (e.g. a
            // `<|channel|>final` that arrived in the same chunk) would have been
            // appended to the buffer first, making `rposition` return the wrong
            // index and corrupting the reconstructed envelope.
            //
            // If reconstruction returns None (encoding unavailable), `last_directed_channel`
            // is left set so the post-loop fallback can flush `pending_tool_call_text`.
            if self.last_directed_channel.is_some()
                && previous_channel.is_some()
                && current_channel.is_none()
                && let Some(env) = reconstruct_directed_envelope(
                    parser,
                    self.last_directed_channel.as_deref(),
                    self.last_directed_recipient.as_deref(),
                )
            {
                self.pending_tool_call_text.clear();
                normal_delta.push_str(&env);
                self.last_directed_channel = None;
                self.last_directed_recipient = None;
                call_reconstructed_in_loop = true;
            }

            if previous_channel.as_deref() != Some("analysis")
                && current_channel.as_deref() == Some("analysis")
                && self.emitted_reasoning_text
            {
                self.insert_reasoning_separator = true;
            }

            if is_visible_normal_channel(current_channel.as_deref(), current_recipient.as_deref())
                && !is_visible_normal_channel(
                    previous_channel.as_deref(),
                    previous_recipient.as_deref(),
                )
                && self.emitted_normal_text
            {
                self.insert_normal_separator = true;
            }

            if let (Some(delta), Some(channel)) = (
                parser.last_content_delta().unwrap_or_default(),
                current_channel,
            ) {
                // `last_content_delta` only exposes the newest token slice, so directed
                // commentary still falls through to the raw handoff path for tool parsing.
                match channel.as_str() {
                    "final" => {
                        if self.insert_normal_separator {
                            normal_delta.push('\n');
                            self.insert_normal_separator = false;
                        }
                        normal_delta.push_str(&delta);
                        self.emitted_normal_text = true;
                    }
                    "analysis" => {
                        // Analysis WITHOUT a recipient is reasoning content.
                        // Analysis WITH a functions recipient is a (malformed)
                        // tool call — its payload is handed off via normal_delta
                        // below; do not also surface it as reasoning_content.
                        if current_recipient.is_none() {
                            if self.insert_reasoning_separator {
                                reasoning_delta.push('\n');
                                self.insert_reasoning_separator = false;
                            }
                            reasoning_delta.push_str(&delta);
                            self.emitted_reasoning_text = true;
                        }
                    }
                    "commentary" => {
                        if current_recipient.is_none() {
                            if self.insert_normal_separator {
                                normal_delta.push('\n');
                                self.insert_normal_separator = false;
                            }
                            normal_delta.push_str(&delta);
                            self.emitted_normal_text = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        let has_call_marker =
            input_contains_call_marker(text, token_ids, caller_supplied_token_ids);
        let raw_input_text = if has_call_marker
            || !self.pending_tool_call_text.is_empty()
            || starts_directed_harmony_tool_call(text)
        {
            Some(raw_input_text_for_tool_parser(
                text,
                token_ids,
                caller_supplied_token_ids,
            ))
        } else {
            None
        };

        if let Some(raw_input_text) = raw_input_text {
            // Streaming feeds the downstream Harmony tool parser through
            // `normal_text`. This is an internal handoff, not visible-content
            // semantics: directed commentary/analysis tool payloads must become
            // tool calls, not assistant `content`.
            if has_call_marker {
                if call_reconstructed_in_loop {
                    // The per-token loop already reconstructed every directed
                    // envelope in this chunk at the correct buffer position (right
                    // after each `<|call|>` token).  Nothing more to do here —
                    // attempting another reconstruction now would use a stale
                    // `rposition` over a buffer that includes post-`<|call|>` tokens
                    // and would produce the wrong envelope or duplicate content.
                } else {
                    // Fallback: in-loop reconstruction was not attempted (encoding
                    // unavailable) or did not complete.  Use the post-loop path,
                    // which may produce a slightly wrong result when trailing tokens
                    // follow `<|call|>` in the same chunk, but is still better than
                    // silently dropping the tool call.
                    let reconstructed = reconstruct_directed_envelope(
                        parser,
                        self.last_directed_channel.as_deref(),
                        self.last_directed_recipient.as_deref(),
                    );
                    match reconstructed {
                        Some(env) => {
                            self.pending_tool_call_text.clear();
                            normal_delta.push_str(&env);
                        }
                        None if !self.pending_tool_call_text.is_empty() => {
                            // Fallback: no clean reconstruction available; flush the
                            // accumulated raw text plus this chunk.
                            self.pending_tool_call_text.push_str(&raw_input_text);
                            normal_delta
                                .push_str(&std::mem::take(&mut self.pending_tool_call_text));
                        }
                        None => normal_delta.push_str(&raw_input_text),
                    }
                    // Clear the persisted context so the next call (or trailing
                    // reasoning) starts fresh.
                    self.last_directed_channel = None;
                    self.last_directed_recipient = None;
                }
            } else if !self.pending_tool_call_text.is_empty()
                || starts_directed_harmony_tool_call(&raw_input_text)
            {
                // Mid tool call, terminator not yet seen: keep accumulating as a
                // fallback (e.g. truncated streams that never emit `<|call|>`).
                self.pending_tool_call_text.push_str(&raw_input_text);
            }
        }

        if !normal_delta.is_empty() || !reasoning_delta.is_empty() {
            tracing::debug!(
                "Returning aggregated deltas: normal: {} chars, reasoning: {} chars",
                normal_delta.len(),
                reasoning_delta.len()
            );
            return ParserResult {
                normal_text: normal_delta,
                reasoning_text: reasoning_delta,
            };
        }

        if let Some(channel) = parser.current_channel() {
            let is_directed_tool_call = channel == "commentary"
                || (channel == "analysis"
                    && parser
                        .current_recipient()
                        .as_deref()
                        .is_some_and(|r| r.starts_with("functions.")));

            if is_directed_tool_call {
                // We are mid directed-tool-call (channel + functions recipient) but
                // produced no delta this chunk. Emit NOTHING here:
                //
                //  * A complete tool call is reconstructed in full — header, message
                //    and `<|call|>` together — from the cumulative token buffer on
                //    the `<|call|>` chunk (the `has_call_marker` branch above). Any
                //    emission here would be a redundant partial: a bare header (when
                //    a chunk boundary lands at `<|message|>`) or a header-less body
                //    fragment, neither of which the downstream jail can assemble into
                //    a tool call — they only leak raw harmony markup into
                //    `response_text`.
                //  * A truncated tool call that never emits `<|call|>` has no complete
                //    envelope to recover, so emitting its partial header would leak
                //    too. Suppressing loses nothing recoverable.
                tracing::debug!(
                    "In directed tool-call channel ({channel}); suppressing partial mid-stream \
                     content (complete envelope is recovered on the <|call|> chunk)"
                );
            } else if channel != "analysis" {
                // Pure analysis reasoning (no recipient) is expected to be silent here —
                // it was already emitted to reasoning_delta in the per-token loop.
                tracing::warn!("Shouldn't be delta content after in channel: {}", channel);
            }
        }
        tracing::debug!("No deltas to return, returning empty result");
        ParserResult::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] // REASONING.batch.2.c, TOOLCALLING.harmony.1
    fn test_gpt_oss_reasoning_parser() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let text = "<|channel|>analysis<|message|>The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed.<|end|><|start|>assistant<|channel|>final<|message|>The capital of Brazil is Brasília.";
        let result = parser.detect_and_parse_reasoning(text, &[]);
        assert!(result.normal_text == "The capital of Brazil is Brasília.");
        assert!(
            result.reasoning_text
                == "The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed."
        );
    }

    #[test] // REASONING.batch.4
    fn test_gpt_oss_final_channel_without_analysis_is_normal_text() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let result = parser.detect_and_parse_reasoning("<|channel|>final<|message|>answer", &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "answer");
    }

    #[test] // REASONING.batch.6.a
    fn test_gpt_oss_multiple_analysis_spans_join_before_final() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let text = "<|channel|>analysis<|message|>first<|end|><|start|>assistant<|channel|>analysis<|message|>second<|end|><|start|>assistant<|channel|>final<|message|>done";
        let result = parser.detect_and_parse_reasoning(text, &[]);
        assert_eq!(result.reasoning_text, "first\nsecond");
        assert_eq!(result.normal_text, "done");
    }

    #[test]
    fn test_gpt_oss_directed_commentary_is_tool_parser_handoff() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let text = "<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|><|start|>assistant<|channel|>final<|message|>It is sunny.";
        let result = parser.detect_and_parse_reasoning(text, &[]);
        assert_eq!(result.reasoning_text, "think");
        assert_eq!(
            result.normal_text,
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|><|start|>assistant<|channel|>final<|message|>It is sunny."
        );
    }

    #[test]
    fn test_gpt_oss_analysis_call_does_not_swallow_commentary_handoff() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let text = "<|channel|>analysis to=functions.search <|constrain|>json<|message|>{\"q\":\"x\"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|><|start|>assistant<|channel|>final<|message|>It is sunny.";
        let result = parser.detect_and_parse_reasoning(text, &[]);
        // An analysis-channel message WITH a `to=functions.` recipient is a
        // (malformed) directed tool call, not chain-of-thought: its args are NOT
        // leaked to reasoning_text.
        assert_eq!(result.reasoning_text, "");
        // Both directed tool-call envelopes (the analysis `search` call and the
        // commentary `get_weather` call) plus the final message are handed off
        // intact to the downstream tool-call jail — the analysis call no longer
        // swallows or strips the commentary handoff.
        assert_eq!(
            result.normal_text,
            "<|channel|>analysis to=functions.search <|constrain|>json<|message|>{\"q\":\"x\"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|><|start|>assistant<|channel|>final<|message|>It is sunny."
        );
    }

    #[test]
    fn test_gpt_oss_recipientless_commentary_is_normal_text() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let text = "<|channel|>commentary<|message|>I will check that.<|end|><|start|>assistant<|channel|>final<|message|>Done.";
        let result = parser.detect_and_parse_reasoning(text, &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "I will check that.\nDone.");
    }

    #[test] // REASONING.stream.2.a, REASONING.batch.2.c, TOOLCALLING.harmony.1
    fn test_gpt_oss_reasoning_parser_streaming() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>",
            "analysis<|message|>The user asks a simple factual question: capital of Brazil.",
            " The answer is Brasília. No additional explanation needed.",
            "<|end|><|start|>assistant<|channel|>final<|message|>",
            "The capital of Brazil is Brasília.",
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert!(normal_text_incr == "The capital of Brazil is Brasília.");
        assert!(
            reasoning_text_incr
                == "The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed."
        );
    }

    #[test] // REASONING.stream.3.b
    fn test_gpt_oss_reasoning_parser_streaming_split_end_marker() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>analysis<|message|>thinking<|e",
            "nd|><|start|>assistant<|channel|>final<|message|>answer",
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text_incr, "thinking");
        assert_eq!(normal_text_incr, "answer");
    }

    #[test]
    fn test_gpt_oss_reasoning_parser_streaming_flushes_pending_suffix_on_finish() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let result = parser.parse_reasoning_streaming_incremental(
            "<|channel|>analysis<|message|>thinking<|e",
            &[],
        );
        let finish = parser.finish_reasoning_stream();
        assert_eq!(result.reasoning_text, "thinking");
        assert_eq!(finish.reasoning_text, "<|e");
        assert_eq!(finish.normal_text, "");
        let finish = parser.finish_reasoning_stream();
        assert_eq!(finish.reasoning_text, "");
        assert_eq!(finish.normal_text, "");

        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let result = parser
            .parse_reasoning_streaming_incremental("<|channel|>final<|message|>answer<|e", &[]);
        let finish = parser.finish_reasoning_stream();
        assert_eq!(result.normal_text, "answer");
        assert_eq!(finish.normal_text, "<|e");
        assert_eq!(finish.reasoning_text, "");
    }

    #[test] // REASONING.stream.2.b
    fn test_gpt_oss_reasoning_parser_streaming_multiple_analysis_spans() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>analysis<|message|>first<|end|><|start|>assistant",
            "<|channel|>analysis<|message|>second<|end|><|start|>assistant",
            "<|channel|>final<|message|>done",
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text_incr, "first\nsecond");
        assert_eq!(normal_text_incr, "done");
    }

    #[test]
    fn test_gpt_oss_reasoning_parser_streaming_recipientless_commentary() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>commentary<|message|>I will check that.<|end|><|start|>assistant",
            "<|channel|>final<|message|>Done.",
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text_incr, "");
        assert_eq!(normal_text_incr, "I will check that.\nDone.");
    }

    #[test]
    fn test_gpt_oss_reasoning_parser_streaming_split_directed_commentary() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>commentary to=functions.get_",
            "weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|>",
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text_incr, "");
        assert_eq!(
            normal_text_incr,
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|>"
        );
    }

    #[test] // REASONING.stream.2.a, REASONING.batch.2.c, TOOLCALLING.harmony.1
    fn test_gpt_oss_reasoning_parser_streaming_chunked() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let enc = get_harmony_encoding()
            .as_ref()
            .expect("Failed to get encoding");
        let text = "<|channel|>analysis<|message|>The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed.<|end|><|start|>assistant<|channel|>final<|message|>The capital of Brazil is Brasília.";
        let token_ids = enc.tokenizer().encode_with_special_tokens(text);
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();

        let mut idx = 0;
        let chunk_size = 4;
        while idx < token_ids.len() {
            let end = (idx + chunk_size).min(token_ids.len());
            let result =
                parser.parse_reasoning_streaming_incremental("Test text", &token_ids[idx..end]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
            idx = end;
        }

        assert_eq!(normal_text_incr, "The capital of Brazil is Brasília.");
        assert_eq!(
            reasoning_text_incr,
            "The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed."
        );
    }

    #[test] // REASONING.batch.3.a, TOOLCALLING.harmony.1
    fn test_gpt_oss_reasoning_parser_streaming_variable_length_chunks() {
        let text = "<|channel|>analysis<|message|>User asks: \"Hey, quick check: is everything up and running?\" We should check system health using the provided function get_system_health. Use function.<|end|><|start|>assistant<|channel|>commentary to=functions.get_system_health <|constrain|>json<|message|>{}";
        let enc = get_harmony_encoding()
            .as_ref()
            .expect("Failed to get encoding");
        let token_ids = enc.tokenizer().encode_with_special_tokens(text);

        // Send token one by one
        {
            let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
            let mut reasoning_text_incr = String::new();
            let mut normal_text_incr = String::new();
            for token in token_ids.iter() {
                let result = parser.parse_reasoning_streaming_incremental("", &[(*token)]);
                normal_text_incr.push_str(&result.normal_text);
                reasoning_text_incr.push_str(&result.reasoning_text);
            }
            assert_eq!(
                reasoning_text_incr,
                "User asks: \"Hey, quick check: is everything up and running?\" We should check system health using the provided function get_system_health. Use function."
            );
            // This input is a TRUNCATED tool call: it ends at `<|message|>{}` with
            // no `<|call|>` terminator. A complete directed tool call is handed off
            // to the jail in one piece (header + body + `<|call|>`) on the `<|call|>`
            // chunk; without a terminator there is no complete envelope to recover,
            // so the streaming parser emits no normal_text rather than leaking a
            // partial `<|channel|>…<|message|>` header.
            assert_eq!(normal_text_incr, "");
        }

        // Send token in chunks (chunking obtained from actual model output)
        {
            let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
            let mut reasoning_text_incr = String::new();
            let mut normal_text_incr = String::new();
            let chunk_tokens = [
                vec![200005],
                vec![35644, 200008, 1844, 31064, 25, 392, 25216, 11, 4853],
                vec![2371, 25, 382, 5519, 869, 326, 6788, 16842, 1416, 1757],
                vec![2371, 2420, 3230, 2360, 290, 5181, 1114, 717, 39303, 126214],
                vec![
                    13, 7649, 1114, 13, 200007, 200006, 173781, 200005, 12606, 815,
                ],
                vec![
                    316, 28, 44580, 775, 39303, 126214, 220, 200003, 4108, 200008,
                ],
                vec![12083],
            ];
            // Concatenate chunk tokens and verify they match original token_ids
            let concatenated: Vec<u32> = chunk_tokens.iter().flatten().copied().collect();
            assert_eq!(concatenated, token_ids);

            for token in chunk_tokens.iter() {
                let result = parser.parse_reasoning_streaming_incremental("", token);
                normal_text_incr.push_str(&result.normal_text);
                reasoning_text_incr.push_str(&result.reasoning_text);
            }
            assert_eq!(
                reasoning_text_incr,
                "User asks: \"Hey, quick check: is everything up and running?\" We should check system health using the provided function get_system_health. Use function."
            );
            // Same truncated-call contract as the token-by-token case above:
            // no <|call|> terminator → no complete envelope → no normal_text.
            assert_eq!(normal_text_incr, "");
        }
    }

    /// `<|call|>` and the following `final` text arrive in the **same chunk**.
    /// Before the fix, `reconstruct_directed_envelope` ran after the whole token
    /// loop and `rposition` would find the `final` channel's `<|channel|>` token
    /// (the last one in the buffer) instead of the tool call's, producing the wrong
    /// envelope.  The in-loop reconstruction fixes this by calling
    /// `reconstruct_directed_envelope` the moment `<|call|>` resets the parser,
    /// before any subsequent tokens are appended to the buffer.
    #[test] // TOOLCALLING.harmony.same-chunk-call-and-final
    fn test_gpt_oss_streaming_call_and_final_in_same_chunk() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>analysis to=functions.search code<|message|>",
            // `<|call|>` and `<|channel|>final` land in one chunk — this is the
            // scenario that the original post-loop reconstruction mis-handled.
            r#"{"q":"foo"}<|call|><|start|>assistant<|channel|>final<|message|>done"#,
        ];
        let mut normal_text = String::new();
        let mut reasoning_text = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text.push_str(&result.normal_text);
            reasoning_text.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text, "");
        assert!(
            normal_text.contains("analysis to=functions.search"),
            "tool call envelope must be present: {normal_text:?}"
        );
        assert!(
            normal_text.contains("<|call|>"),
            "call terminator must be present: {normal_text:?}"
        );
        assert!(
            normal_text.ends_with("done"),
            "final text must follow tool call: {normal_text:?}"
        );
        let call_pos = normal_text.find("<|call|>").unwrap();
        let done_pos = normal_text.find("done").unwrap();
        assert!(
            call_pos < done_pos,
            "tool call must precede final text in output"
        );
    }

    /// Two directed tool calls arriving in a single chunk.  The original code had
    /// only one post-loop reconstruction path, so the second `<|call|>` clobbered
    /// the first.  The in-loop per-`<|call|>` reconstruction handles each
    /// terminator independently and preserves both envelopes.
    #[test] // TOOLCALLING.harmony.two-calls-one-chunk
    fn test_gpt_oss_streaming_two_calls_in_one_chunk() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            // Both tool calls arrive together.
            "<|channel|>analysis to=functions.search code<|message|>{\"q\":\"x\"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|>",
        ];
        let mut normal_text = String::new();
        let mut reasoning_text = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text.push_str(&result.normal_text);
            reasoning_text.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text, "");
        assert!(
            normal_text.contains("analysis to=functions.search"),
            "first tool call must be present: {normal_text:?}"
        );
        assert!(
            normal_text.contains("commentary to=functions.get_weather"),
            "second tool call must be present: {normal_text:?}"
        );
        // First call must precede second call in the output.
        let first_pos = normal_text.find("search").unwrap();
        let second_pos = normal_text.find("get_weather").unwrap();
        assert!(first_pos < second_pos, "calls must appear in stream order");
    }

    #[test] // REASONING.stream.4.d
    fn test_gpt_oss_reasoning_parser_streaming_split_directed_analysis() {
        // Header arrives in chunk 1; body + <|call|> arrive in chunk 2.
        // This is the exact case last_directed_channel / reconstruct_directed_envelope
        // is designed for: the parser state resets on <|call|>, so the channel
        // and recipient must be persisted from the earlier chunk.
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>analysis to=functions.grep code<|message|>",
            r#"{"pattern":"foo"}<|call|>"#,
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text_incr, "");
        assert!(
            normal_text_incr.contains("analysis to=functions.grep"),
            "channel header must survive into normal_text: {normal_text_incr:?}"
        );
        assert!(
            normal_text_incr.ends_with("<|call|>"),
            "terminator must be present: {normal_text_incr:?}"
        );
    }
}
