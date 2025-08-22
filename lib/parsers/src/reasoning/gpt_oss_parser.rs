// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::ParserResult;
use crate::ReasoningParser;

use openai_harmony::chat::TextContent;
use openai_harmony::StreamableParser;
use openai_harmony::{chat::Role, load_harmony_encoding, HarmonyEncoding, HarmonyEncodingName};

#[derive(Debug)]
pub struct GptOssReasoningParser {
    enc: HarmonyEncoding,
}

impl GptOssReasoningParser {
    pub fn new() -> anyhow::Result<Self> {
        let enc = match load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss) {
            Ok(enc) => enc,
            Err(e) => {
                tracing::error!("Failed to load HarmonyGptOss encoding: {e}");
                // This leaves `enc` in an unusable state. Consider making `new()` return a Result in a follow-up.
                return Err(anyhow::anyhow!(
                    "Failed to load HarmonyGptOss encoding: {e}"
                ));
            }
        };
        Ok(Self { enc })
    }
}

impl Default for GptOssReasoningParser {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl GptOssReasoningParser {
    fn reason_parsing_wrapper(&self, _text: &str, token_ids: &[u32]) -> ParserResult {
        let mut parser = match StreamableParser::new(self.enc.clone(), Some(Role::Assistant)) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Harmony StreamableParser init failed for GPT OSS: {e}");
                return ParserResult::default();
            }
        };

        for token_id in token_ids {
            if let Err(e) = parser.process(*token_id) {
                tracing::warn!("Harmony parse error for token_id {token_id}: {e}");
                return ParserResult::default();
            }
        }
        let output_msgs = parser.messages();
        // let mut reasoning_token_ids = vec![];
        // let mut normal_token_ids = vec![];
        match output_msgs.len() {
            0 => {
                let current = parser.current_content().unwrap_or_default();
                ParserResult {
                    normal_text: String::new(),
                    reasoning_text: current,
                }
            }
            1 => {
                let mut reasoning_text = String::new();
                if let Some(openai_harmony::chat::Content::Text(TextContent { text })) =
                    output_msgs[0].content.first()
                {
                    reasoning_text.push_str(text);
                }
                let current = parser.current_content().unwrap_or_default();
                ParserResult {
                    normal_text: current,
                    reasoning_text,
                }
            }
            _ => {
                let mut reasoning_text = String::new();
                let mut normal_text = String::new();

                // Loop until second last message
                for parse_msg in output_msgs.iter().take(output_msgs.len() - 1) {
                    if let Some(openai_harmony::chat::Content::Text(TextContent { text })) =
                        parse_msg.content.first()
                    {
                        reasoning_text.push_str(text);
                    }
                }

                let last_msg = &output_msgs[output_msgs.len() - 1];

                // Handle the last message
                if let Some(openai_harmony::chat::Content::Text(TextContent { text })) =
                    last_msg.content.first()
                {
                    normal_text.push_str(text);
                }

                ParserResult {
                    normal_text,
                    reasoning_text,
                }
            }
        }
    }
}

impl ReasoningParser for GptOssReasoningParser {
    fn detect_and_parse_reasoning(&self, text: &str, token_ids: &[u32]) -> ParserResult {
        self.reason_parsing_wrapper(text, token_ids)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        self.reason_parsing_wrapper(text, token_ids)
    }
}
