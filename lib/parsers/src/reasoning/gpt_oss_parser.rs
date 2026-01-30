// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use crate::ParserResult;
use crate::ReasoningParser;

use openai_harmony::StreamableParser;
use openai_harmony::chat::TextContent;
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, chat::Role, load_harmony_encoding};

///// Static initialization of harmony encoder to not affect performance every time a parser is created
/// This is because load_harmony_encoding downloads some tiktoken files into a directory and we don't want to do this every time we create a parser.
use std::sync::OnceLock;

static GLOBAL_HARMONY_GPTOSS_ENCODING: OnceLock<Result<HarmonyEncoding, anyhow::Error>> =
    OnceLock::new();

fn get_harmony_encoding() -> &'static Result<HarmonyEncoding, anyhow::Error> {
    GLOBAL_HARMONY_GPTOSS_ENCODING
        .get_or_init(|| load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss))
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
                tracing::warn!("Harmony parse error for token_id {token_id}: {e}");
                return ParserResult::default();
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

        for (i, token_id) in token_ids.iter().enumerate() {
            tracing::debug!(
                "Processing streaming token {} of {}: {}",
                i + 1,
                token_ids.len(),
                token_id
            );
            if let Err(e) = parser.process(*token_id) {
                tracing::warn!("Harmony parse error for token_id {token_id}: {e}");
                return ParserResult::default();
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
            if channel == "commentary" {
                tracing::debug!("In commentary channel, recovering full content");
                // If we're in the commentary channel, we should return raw token content and recover content that has been consumed by the parser
                // so that the tool parser can process it properly
                if let Ok(enc) = get_harmony_encoding() {
                    let current_content = parser.current_content().unwrap_or_default();
                    let mut final_text = text.to_string();

                    // Restore commentary metadata consumed by the parser so the tool-call parser can
                    // process it correctly.
                    //
                    // Example:
                    //   Before parsing:
                    //   "<|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{\"format\":\"celsius\",\"location\":\"San Francisco\"}<|call|>"
                    //   After parsing, the header is stripped, so we must reconstruct it:
                    //   "<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>"
                    //
                    // This ensures downstream tool-call parsing receives the channel, target, and
                    // constraint metadata together with the message payload.

                    // Recovery should only happen once, and only when `current_content` is empty.
                    if current_content.is_empty() {
                        let tokens = parser.tokens();

                        // Get the token id for " <|channel|>"
                        let channel_token_id = enc
                            .tokenizer()
                            .encode_with_special_tokens("<|channel|>")
                            .last()
                            .copied();

                        // Find the last occurrence of the <|channel|> token (id 20005) in the tokens vector
                        let last_channel_token_idx = channel_token_id
                            .and_then(|token_id| {
                                tokens.iter().rposition(|token| *token == token_id)
                            })
                            .unwrap_or(0);

                        // Then get the generated text from the last <|channel|> to the end of parser.tokens()
                        let end_token_idx = parser.tokens().len();
                        // Use Harmony's decode_utf8 to decode tokens into text
                        let generated_text = enc
                            .tokenizer()
                            .decode_utf8(&parser.tokens()[last_channel_token_idx..end_token_idx])
                            .unwrap_or_default();

                        final_text = generated_text;
                    }

                    return ParserResult {
                        normal_text: final_text,
                        reasoning_text: String::new(),
                    };
                }
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

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
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
}
