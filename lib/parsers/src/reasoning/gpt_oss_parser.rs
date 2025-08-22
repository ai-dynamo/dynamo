// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::Deref;

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
    pub fn new() -> Self {
        let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
        Self { enc }
    }
}

impl Default for GptOssReasoningParser {
    fn default() -> Self {
        Self::new()
    }
}

impl GptOssReasoningParser {
    fn reason_parsing_wrapper(&self, _text: &str, token_ids: &[u32]) -> ParserResult {
        let mut parser = StreamableParser::new(self.enc.clone(), Some(Role::Assistant)).unwrap();
        for token_id in token_ids {
            parser.process(*token_id).unwrap();
        }
        let output_msgs = parser.messages();
        // let mut reasoning_token_ids = vec![];
        // let mut normal_token_ids = vec![];
        match output_msgs.len() {
            0 => {
                ParserResult {
                    normal_text: String::new(), // No normal text in this example
                    reasoning_text: parser.current_content().unwrap().deref().to_string(), // All text is reasoning
                }
            }
            1 => {
                let mut reasoning_text = String::new();
                if let Some(openai_harmony::chat::Content::Text(TextContent { text })) =
                    output_msgs[0].content.first()
                {
                    reasoning_text.push_str(text);
                }
                ParserResult {
                    normal_text: parser.current_content().unwrap().deref().to_string(),
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
