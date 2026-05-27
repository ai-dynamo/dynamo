// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use crate::ParserResult;
use crate::ReasoningParser;

use openai_harmony::StreamableParser;
use openai_harmony::chat::TextContent;
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, chat::Role, load_harmony_encoding};
use regex::Regex;

///// Static initialization of harmony encoder to not affect performance every time a parser is created
/// This is because load_harmony_encoding downloads some tiktoken files into a directory and we don't want to do this every time we create a parser.
use std::sync::OnceLock;

static GLOBAL_HARMONY_GPTOSS_ENCODING: OnceLock<Result<HarmonyEncoding, anyhow::Error>> =
    OnceLock::new();

static HARMONY_CONTROL_TOKEN_RE: OnceLock<Regex> = OnceLock::new();

/// Strip harmony control tokens (`<|start|>`, `<|channel|>`, `<|message|>`,
/// `<|constrain|>`, `<|end|>`, `<|call|>`, `<|return|>`) from raw chunk text
/// surfaced on the StreamableParser error path. Without this, malformed
/// harmony chunks that don't match the jail's `<|channel|>commentary` start
/// marker (e.g. bare `<|end|>` followed by prose, or analysis-only chunks)
/// would leak structural tokens to the client. Mirrors SGLang's
/// `_is_standalone_structural_token` filter in their tolerant HarmonyParser.
fn strip_harmony_control_tokens(text: &str) -> String {
    let re = HARMONY_CONTROL_TOKEN_RE.get_or_init(|| {
        Regex::new(r"<\|(?:start|channel|message|constrain|end|call|return)\|>")
            .expect("harmony control token regex")
    });
    re.replace_all(text, "").into_owned()
}

/// Decode the parser's tokens from the most recent `<|channel|>` marker to the
/// end. Used to surface the commentary header (`<|channel|>commentary
/// to=functions.X <|constrain|>json<|message|>`) that `StreamableParser`
/// silently consumes, so the downstream jail / tool-call parser sees a start
/// marker. Returns `None` if the harmony encoding failed to load or no
/// `<|channel|>` token is present.
fn reconstruct_from_last_channel(parser: &StreamableParser) -> Option<String> {
    let enc = get_harmony_encoding().as_ref().ok()?;
    let channel_token_id = enc
        .tokenizer()
        .encode_with_special_tokens("<|channel|>")
        .last()
        .copied()?;
    let tokens = parser.tokens();
    let last_idx = tokens.iter().rposition(|t| *t == channel_token_id)?;
    enc.tokenizer().decode_utf8(&tokens[last_idx..]).ok()
}

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
        Ok(Self { parser })
    }
}

fn encode_text_to_tokens(text: &str) -> anyhow::Result<Vec<u32>> {
    let enc = get_harmony_encoding()
        .as_ref()
        .map_err(|e| anyhow::anyhow!("Failed to get harmony encoding: {e}"))?;
    Ok(enc.tokenizer().encode_with_special_tokens(text))
}

impl ReasoningParser for GptOssReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult {
        let token_ids = if token_ids.is_empty() {
            // WAR: Since we are moving to just text based reasoning parsing, converting to token_ids now using harmony encoding
            let encoded_tokens = match encode_text_to_tokens(text) {
                Ok(tokens) => tokens,
                Err(err) => {
                    tracing::warn!("Failed to encode Harmony tokens: {err}");
                    return ParserResult::default();
                }
            };
            &encoded_tokens.to_vec()
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
                // Strict StreamableParser rejects malformed model output (e.g. natural
                // language in a `to=` recipient slot, or text after `<|end|>` without
                // `<|start|>`). Dropping the chunk silently loses content; surface the
                // raw text instead so downstream (harmony tool-call regex fallback,
                // jail) gets a shot. Once StreamableParser is in an error state, no
                // further deltas will materialize, so abandon the loop. Strip harmony
                // control tokens so bare markers (e.g. `<|end|>`, `<|channel|>analysis`)
                // don't leak past the jail, whose start markers only cover commentary.
                tracing::warn!("Harmony parse error for token_id {token_id}: {e}");
                return ParserResult {
                    normal_text: strip_harmony_control_tokens(text),
                    reasoning_text: String::new(),
                };
            }
        }

        let output_msgs = parser.messages();
        tracing::debug!("Parser has {} output messages", output_msgs.len());

        match output_msgs.len() {
            0 => {
                tracing::debug!("No output messages, using current content");
                let current = parser.current_content().unwrap_or_default();
                tracing::debug!("Current content length: {}", current.len());
                ParserResult {
                    normal_text: String::new(),
                    reasoning_text: current,
                }
            }
            1 => {
                tracing::debug!("Single output message detected");
                let mut reasoning_text = String::new();
                if let Some(openai_harmony::chat::Content::Text(TextContent { text })) =
                    output_msgs[0].content.first()
                {
                    reasoning_text.push_str(text);
                    tracing::debug!("Extracted reasoning text length: {}", reasoning_text.len());
                }
                let current = parser.current_content().unwrap_or_default();
                tracing::debug!("Current content length: {}", current.len());
                ParserResult {
                    normal_text: current,
                    reasoning_text,
                }
            }
            _ => {
                tracing::debug!("Multiple output messages detected: {}", output_msgs.len());
                let mut reasoning_text = String::new();
                let mut normal_text = String::new();

                // Loop until second last message
                for (i, parse_msg) in output_msgs.iter().take(output_msgs.len() - 1).enumerate() {
                    tracing::debug!("Processing reasoning message {}", i + 1);
                    if let Some(openai_harmony::chat::Content::Text(TextContent { text })) =
                        parse_msg.content.first()
                    {
                        reasoning_text.push_str(text);
                        tracing::debug!("Added {} chars to reasoning text", text.len());
                    }
                }

                let last_msg = &output_msgs[output_msgs.len() - 1];
                tracing::debug!("Processing final message");

                // Handle the last message
                if let Some(openai_harmony::chat::Content::Text(TextContent { text })) =
                    last_msg.content.first()
                {
                    normal_text.push_str(text);
                    tracing::debug!("Added {} chars to normal text", text.len());
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
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        let token_ids = if token_ids.is_empty() {
            // WAR: Since we are moving to just text based reasoning parsing, converting to token_ids now using harmony encoding
            let encoded_tokens = match encode_text_to_tokens(text) {
                Ok(tokens) => tokens,
                Err(err) => {
                    tracing::warn!("Failed to encode Harmony tokens: {err}");
                    return ParserResult::default();
                }
            };
            &encoded_tokens.to_vec()
        } else {
            token_ids
        };

        let parser: &mut StreamableParser = &mut self.parser;
        let mut normal_delta = String::new();
        let mut reasoning_delta = String::new();
        // Whether the parser was in the commentary channel at the start of
        // this chunk. Combined with `entered_commentary_this_chunk` below to
        // distinguish first-entry chunks (need header reconstruction) from
        // continuation chunks (pass through incremental text).
        let was_in_commentary_at_start = parser.current_channel().as_deref() == Some("commentary");
        // Set to true once the parser transitions into the commentary channel
        // during this chunk's token loop. Drives the commentary-header
        // reconstruction below; using a transition flag (rather than
        // `current_content.is_empty()`) ensures recovery still fires when
        // body tokens arrive in the same chunk as the `<|message|>` boundary.
        let mut entered_commentary_this_chunk = false;

        for (i, token_id) in token_ids.iter().enumerate() {
            tracing::debug!(
                "Processing streaming token {} of {}: {}",
                i + 1,
                token_ids.len(),
                token_id
            );
            if let Err(e) = parser.process(*token_id) {
                // See sibling fallback in `detect_and_parse_reasoning`: surface raw
                // chunk text on strict-parser failure so downstream parsers (harmony
                // tool-call regex fallback, jail) can still recover instead of the
                // chunk being dropped entirely. Drops any partial deltas accumulated
                // in this chunk before the failure; trade-off vs. losing the chunk.
                // Strip harmony control tokens to keep bare markers from leaking past
                // the jail (whose start markers only cover commentary).
                tracing::warn!("Harmony parse error for token_id {token_id}: {e}");
                return ParserResult {
                    normal_text: strip_harmony_control_tokens(text),
                    reasoning_text: String::new(),
                };
            }

            if !was_in_commentary_at_start
                && !entered_commentary_this_chunk
                && parser.current_channel().as_deref() == Some("commentary")
            {
                entered_commentary_this_chunk = true;
            }

            if let (Some(delta), Some(channel)) = (
                parser.last_content_delta().unwrap_or_default(),
                parser.current_channel(),
            ) {
                // `last_content_delta` only exposes the newest token slice, so we forward
                // `final`/`analysis` chunks immediately; commentary is reconstructed in the
                // fallback path below because it needs the stripped metadata.
                match channel.as_str() {
                    "final" => normal_delta.push_str(&delta),
                    "analysis" => reasoning_delta.push_str(&delta),
                    "commentary" => {}
                    _ => {}
                }
            }
        }

        // Commentary header recovery. `StreamableParser` consumes
        // `<|channel|>commentary to=functions.X <|constrain|>json<|message|>`
        // silently and emits no delta for commentary content. The jail keys
        // off `<|channel|>commentary`, so the consumed header must be surfaced
        // here on the chunk that enters commentary — including chunks that
        // also contain body tokens (current_content non-empty).
        //
        // Must run BEFORE the delta early-return: when a single chunk carries
        // both an analysis tail (`reasoning_delta` non-empty) AND entry into
        // commentary, returning only the delta would drop the header and leak
        // `<|constrain|>json<|message|>{...}` past the jail on subsequent
        // chunks (which is the dominant leak pattern observed in production).
        let commentary_header = if entered_commentary_this_chunk {
            reconstruct_from_last_channel(parser).unwrap_or_default()
        } else {
            String::new()
        };

        if !normal_delta.is_empty() || !reasoning_delta.is_empty() || !commentary_header.is_empty()
        {
            tracing::debug!(
                "Returning aggregated deltas: normal: {} chars, reasoning: {} chars, commentary_header: {} chars",
                normal_delta.len(),
                reasoning_delta.len(),
                commentary_header.len(),
            );
            return ParserResult {
                normal_text: format!("{}{}", normal_delta, commentary_header),
                reasoning_text: reasoning_delta,
            };
        }

        if let Some(channel) = parser.current_channel() {
            if channel == "commentary" {
                // Continuation chunk while in commentary (was already in
                // commentary at the start of this chunk and no deltas to
                // forward). Pass through the raw chunk text incrementally so
                // the jail's accumulator gets the body (and any trailing
                // `<|call|>` end marker).
                tracing::debug!("In commentary channel continuation, passing raw text");
                return ParserResult {
                    normal_text: text.to_string(),
                    reasoning_text: String::new(),
                };
            } else {
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

    #[test] // REASONING.batch.1, PARSER.harmony.1
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

    #[test] // REASONING.stream.3, REASONING.batch.1, PARSER.harmony.1
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

    #[test] // REASONING.stream.3, REASONING.batch.1, PARSER.harmony.1
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

    #[test] // REASONING.stream.3, REASONING.batch.1, PARSER.harmony.1
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
            // [gluo TODO] missing "<|start|>assistant" and "{}" from original message
            assert_eq!(
                normal_text_incr,
                "<|channel|>commentary to=functions.get_system_health <|constrain|>json<|message|>"
            );
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
            assert_eq!(
                normal_text_incr,
                "<|channel|>commentary to=functions.get_system_health <|constrain|>json<|message|>"
            );
        }
    }

    #[test]
    fn test_strip_harmony_control_tokens() {
        // Bare control tokens and channel markers removed; prose preserved.
        assert_eq!(
            strip_harmony_control_tokens("<|end|>regular text"),
            "regular text"
        );
        assert_eq!(
            strip_harmony_control_tokens("<|channel|>analysis<|message|>thinking<|end|>"),
            "analysisthinking"
        );
        assert_eq!(
            strip_harmony_control_tokens("plain text with no markers"),
            "plain text with no markers"
        );
        // Unknown `<|foo|>` markers are left alone — only the harmony control
        // set is stripped, so model-emitted XML-ish tokens aren't clobbered.
        assert_eq!(
            strip_harmony_control_tokens("<|foo|>kept<|return|>dropped"),
            "<|foo|>keptdropped"
        );
    }

    /// Regression: when a single chunk carries the tail of an analysis message
    /// AND completes entry into the commentary channel, the commentary header
    /// must be surfaced alongside the reasoning delta. Previously the delta
    /// early-return dropped the header, causing subsequent commentary chunks
    /// (e.g. `<|constrain|>json<|message|>{...}`) to leak past the jail as
    /// regular content.
    #[test]
    fn test_gpt_oss_reasoning_parser_analysis_then_commentary_in_one_chunk() {
        let text = "<|channel|>analysis<|message|>We should call get_system_health.<|end|><|start|>assistant<|channel|>commentary to=functions.get_system_health <|constrain|>json<|message|>{}";
        let enc = get_harmony_encoding()
            .as_ref()
            .expect("Failed to get encoding");
        let token_ids = enc.tokenizer().encode_with_special_tokens(text);

        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let mut normal_text_incr = String::new();
        let mut reasoning_text_incr = String::new();

        // Single chunk carrying everything from analysis tail through the
        // commentary header (and the empty body up to but not including the
        // body delta).
        let result = parser.parse_reasoning_streaming_incremental("", &token_ids);
        normal_text_incr.push_str(&result.normal_text);
        reasoning_text_incr.push_str(&result.reasoning_text);

        assert_eq!(reasoning_text_incr, "We should call get_system_health.");
        // The commentary header must be surfaced so the jail's
        // `<|channel|>commentary` start marker triggers.
        assert!(
            normal_text_incr.contains("<|channel|>commentary"),
            "expected commentary header in normal_text, got: {normal_text_incr:?}"
        );
        assert!(
            normal_text_incr.contains("to=functions.get_system_health"),
            "expected recipient in normal_text, got: {normal_text_incr:?}"
        );
        assert!(
            normal_text_incr.contains("<|constrain|>json<|message|>"),
            "expected constrain/message tokens in normal_text, got: {normal_text_incr:?}"
        );
    }
}
