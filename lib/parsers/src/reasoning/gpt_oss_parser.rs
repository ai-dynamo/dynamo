// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

impl ReasoningParser for GptOssReasoningParser {
    fn detect_and_parse_reasoning(&mut self, _text: &str, token_ids: &[u32]) -> ParserResult {
        tracing::debug!(
            "detect_and_parse_reasoning called with {} token_ids",
            token_ids.len()
        );

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
        tracing::debug!(
            "parse_reasoning_streaming_incremental called with {} token_ids",
            token_ids.len()
        );

        let parser: &mut StreamableParser = &mut self.parser;
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
        }

        if let Some(channel) = self.parser.current_channel() {
            tracing::debug!("Current channel {}", channel);
            if channel == "final" {
                // If we're in the final channel, we should not parse reasoning
                if let Some(current) = self.parser.last_content_delta().unwrap_or_default() {
                    tracing::debug!("Got normal text delta of {} chars", current.len());
                    return ParserResult {
                        normal_text: current,
                        reasoning_text: String::new(),
                    };
                }
                tracing::debug!("No content delta in final channel");
                ParserResult::default()
            } else if channel == "commentary" {
                // If we're in the commentary channel, we should return raw token content and recover content that has been consumed by the parser
                // so that the tool parser can process it properly
                if let Ok(enc) = get_harmony_encoding() {
                    let current_content = self.parser.current_content().unwrap_or_default();
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
                        let tokens = self.parser.tokens();

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

                        // Then get the generated text from the last <|channel|> to the end of self.parser.tokens()
                        let end_token_idx = self.parser.tokens().len();
                        // Use Harmony's decode_utf8 to decode tokens into text
                        let generated_text = enc
                            .tokenizer()
                            .decode_utf8(
                                &self.parser.tokens()[last_channel_token_idx..end_token_idx],
                            )
                            .unwrap_or_default();

                        final_text = generated_text;
                    }

                    ParserResult {
                        normal_text: final_text,
                        reasoning_text: String::new(),
                    }
                } else {
                    tracing::warn!("Failed to get harmony encoding for raw token decoding");
                    ParserResult::default()
                }
            } else {
                tracing::debug!("In reasoning channel: {}", channel);
                if let Some(current) = self.parser.last_content_delta().unwrap_or_default() {
                    tracing::debug!("Got reasoning text delta of {} chars", current.len());
                    return ParserResult {
                        normal_text: String::new(),
                        reasoning_text: current,
                    };
                }
                tracing::debug!("No content delta in reasoning channel");
                ParserResult::default()
            }
        } else {
            tracing::debug!("No current channel detected");
            ParserResult::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_oss_reasoning_parser() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let enc = get_harmony_encoding()
            .as_ref()
            .expect("Failed to get encoding");
        let text = "<|channel|>analysis<|message|>The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed.<|end|><|start|>assistant<|channel|>final<|message|>The capital of Brazil is Brasília.";
        let token_ids = enc.tokenizer().encode_with_special_tokens(text); // Example token IDs
        let result = parser.detect_and_parse_reasoning("Test text", &token_ids);
        assert!(result.normal_text == "The capital of Brazil is Brasília.");
        assert!(
            result.reasoning_text
                == "The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed."
        );
    }

    #[test]
    fn test_gpt_oss_reasoning_parser_streaming() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let enc = get_harmony_encoding()
            .as_ref()
            .expect("Failed to get encoding");
        let text = "<|channel|>analysis<|message|>The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed.<|end|><|start|>assistant<|channel|>final<|message|>The capital of Brazil is Brasília.";
        let token_ids = enc.tokenizer().encode_with_special_tokens(text); // Example token IDs
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for token in token_ids.iter() {
            let result = parser.parse_reasoning_streaming_incremental("Test text", &[(*token)]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert!(normal_text_incr == "The capital of Brazil is Brasília.");
        assert!(
            reasoning_text_incr
                == "The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed."
        );
    }
}
