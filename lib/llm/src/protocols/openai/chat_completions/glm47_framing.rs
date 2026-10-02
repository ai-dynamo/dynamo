// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Quote-aware framing around the published GLM decoder.
//! Unclosed quotes/code remain literal prose rather than executable examples.

use dynamo_parsers::tool_calling::{ToolCallResponse, ToolDefinition, config::Glm47ParserConfig};

const TOOL_START: &str = "<tool_call>";
const TOOL_END: &str = "</tool_call>";
const ARG_MARKERS: [&str; 4] = ["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"];

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Glm47Frame {
    Text(String),
    ToolBlock(String),
}

#[derive(Debug, Default)]
pub(crate) struct Glm47Finish {
    pub(crate) frames: Vec<Glm47Frame>,
    pub(crate) incomplete_tool_call: bool,
}

#[derive(Debug, Default)]
pub(crate) struct Glm47Framer {
    tail: String,
    marker: String,
    block: Option<String>,
    in_arg_value: bool,
    quote: Option<Quote>,
    previous: Option<char>,
    escaped_delimiter: bool,
}

#[derive(Debug)]
enum Quote {
    ApostropheAfterWord,
    DoubleQuoteAfterDigit,
    String {
        delimiter: char,
        escaped: bool,
        pending_apostrophe: bool,
    },
    Backticks {
        width: usize,
        opening: bool,
        closing_run: usize,
    },
}

impl Glm47Framer {
    pub(crate) fn push(&mut self, text: &str) -> Vec<Glm47Frame> {
        let mut frames = Vec::new();
        for ch in text.chars() {
            self.consume(ch, &mut frames);
            self.previous = Some(ch);
        }
        frames
    }

    pub(crate) fn flush_prose(&mut self) -> Vec<Glm47Frame> {
        let mut frames = Vec::new();
        if self.block.is_none() && self.marker.is_empty() {
            append_text(&mut frames, &std::mem::take(&mut self.tail));
        }
        frames
    }

    pub(crate) fn finish(&mut self) -> Glm47Finish {
        // Short prefixes are ambiguous prose; the underscore makes a native opener
        // distinctive enough to suppress without turning ordinary `2 <` into loss.
        let partial_tool = self.marker.starts_with("<tool_");
        let partial_arg = self.marker.starts_with("<arg_") || self.marker.starts_with("</arg_");
        let incomplete_tool_call = self.block.take().is_some() || partial_tool || partial_arg;
        let mut frames = Vec::new();
        if !partial_arg {
            append_text(&mut frames, &std::mem::take(&mut self.tail));
        } else {
            self.tail.clear();
        }
        if !partial_tool && !partial_arg {
            append_text(&mut frames, &self.marker);
        }
        self.marker.clear();
        self.quote = None;
        self.previous = None;
        self.in_arg_value = false;
        self.escaped_delimiter = false;
        Glm47Finish {
            frames,
            incomplete_tool_call,
        }
    }

    fn consume(&mut self, ch: char, frames: &mut Vec<Glm47Frame>) {
        if let Some(block) = self.block.as_mut() {
            block.push(ch);
            if block.ends_with("<arg_value>") {
                self.in_arg_value = true;
            } else if block.ends_with("</arg_value>") {
                self.in_arg_value = false;
            }
            // Only the bounded suffix is inspected, never the accumulated response.
            if !self.in_arg_value
                && block.ends_with(TOOL_END)
                && let Some(block) = self.block.take()
            {
                frames.push(Glm47Frame::ToolBlock(block));
            }
            return;
        }
        if !self.marker.is_empty() {
            self.consume_marker(ch, frames);
            return;
        }
        if self.consume_quoted(ch, frames) {
            return;
        }
        let escaped = std::mem::take(&mut self.escaped_delimiter);
        if escaped && matches!(ch, '"' | '\'' | '`' | '\\') {
            self.consume_plain(ch, frames);
            return;
        }
        match ch {
            '\\' => {
                self.consume_plain(ch, frames);
                self.escaped_delimiter = true;
            }
            '<' => self.marker.push(ch),
            '\'' if self.previous.is_some_and(char::is_alphanumeric) => {
                append_text(frames, &std::mem::take(&mut self.tail));
                append_char(frames, ch);
                self.quote = Some(Quote::ApostropheAfterWord);
            }
            '"' if self
                .previous
                .is_some_and(|previous| previous.is_ascii_digit()) =>
            {
                append_text(frames, &std::mem::take(&mut self.tail));
                append_char(frames, ch);
                self.quote = Some(Quote::DoubleQuoteAfterDigit);
            }
            '"' | '\'' => {
                append_text(frames, &std::mem::take(&mut self.tail));
                append_char(frames, ch);
                self.quote = Some(Quote::String {
                    delimiter: ch,
                    escaped: false,
                    pending_apostrophe: false,
                });
            }
            '`' => {
                append_text(frames, &std::mem::take(&mut self.tail));
                append_char(frames, ch);
                self.quote = Some(Quote::Backticks {
                    width: 1,
                    opening: true,
                    closing_run: 0,
                });
            }
            _ => self.consume_plain(ch, frames),
        }
    }

    fn consume_marker(&mut self, ch: char, frames: &mut Vec<Glm47Frame>) {
        self.marker.push(ch);
        if self.marker == TOOL_START {
            append_text(frames, &std::mem::take(&mut self.tail));
            self.block = Some(std::mem::take(&mut self.marker));
        } else if ARG_MARKERS.contains(&self.marker.as_str()) {
            self.in_arg_value = self.marker == "<arg_value>";
            let mut block = std::mem::take(&mut self.tail);
            block.push_str(&std::mem::take(&mut self.marker));
            self.block = Some(block);
        } else if !std::iter::once(TOOL_START)
            .chain(ARG_MARKERS)
            .any(|marker| marker.starts_with(&self.marker))
        {
            append_text(frames, &std::mem::take(&mut self.tail));
            self.marker.pop();
            append_text(frames, &std::mem::take(&mut self.marker));
            self.consume(ch, frames);
        }
    }

    fn consume_plain(&mut self, ch: char, frames: &mut Vec<Glm47Frame>) {
        // Keep just the last identifier and following whitespace: GLM also supports
        // bare `function<arg_key>... </tool_call>` without an opening wrapper.
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
            if self.tail.ends_with(char::is_whitespace) {
                append_text(frames, &std::mem::take(&mut self.tail));
            }
            self.tail.push(ch);
        } else if ch.is_whitespace() && !self.tail.is_empty() {
            self.tail.push(ch);
        } else {
            append_text(frames, &std::mem::take(&mut self.tail));
            append_char(frames, ch);
        }
    }

    fn consume_quoted(&mut self, ch: char, frames: &mut Vec<Glm47Frame>) -> bool {
        let Some(quote) = self.quote.take() else {
            return false;
        };
        match quote {
            Quote::DoubleQuoteAfterDigit => {
                if ch.is_whitespace() || (ch.is_ascii_punctuation() && !matches!(ch, '<' | '"')) {
                    return false;
                }
                self.quote = Some(Quote::String {
                    delimiter: '"',
                    escaped: false,
                    pending_apostrophe: false,
                });
                return self.consume_quoted(ch, frames);
            }
            Quote::ApostropheAfterWord => {
                if ch.is_alphanumeric() || ch.is_whitespace() {
                    return false;
                }
                self.quote = Some(Quote::String {
                    delimiter: '\'',
                    escaped: false,
                    pending_apostrophe: false,
                });
                return self.consume_quoted(ch, frames);
            }
            Quote::String {
                delimiter,
                mut escaped,
                mut pending_apostrophe,
            } => {
                if pending_apostrophe && !ch.is_alphanumeric() {
                    return false;
                }
                pending_apostrophe = false;
                append_char(frames, ch);
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == delimiter {
                    if delimiter == '\'' && self.previous.is_some_and(char::is_alphanumeric) {
                        pending_apostrophe = true;
                    } else {
                        return true;
                    }
                }
                self.quote = Some(Quote::String {
                    delimiter,
                    escaped,
                    pending_apostrophe,
                });
            }
            Quote::Backticks {
                mut width,
                mut opening,
                mut closing_run,
            } => {
                if ch == '`' {
                    if opening {
                        width += 1;
                    } else {
                        closing_run += 1;
                    }
                } else {
                    if !opening && (closing_run == width || (width >= 3 && closing_run > width)) {
                        return false;
                    }
                    opening = false;
                    closing_run = 0;
                }
                append_char(frames, ch);
                self.quote = Some(Quote::Backticks {
                    width,
                    opening,
                    closing_run,
                });
            }
        }
        true
    }
}

fn append_text(frames: &mut Vec<Glm47Frame>, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(Glm47Frame::Text(previous)) = frames.last_mut() {
        previous.push_str(text);
    } else {
        frames.push(Glm47Frame::Text(text.to_owned()));
    }
}

fn append_char(frames: &mut Vec<Glm47Frame>, ch: char) {
    let mut bytes = [0; 4];
    append_text(frames, ch.encode_utf8(&mut bytes));
}

pub(crate) fn parse_block(
    block: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let tools = tools.filter(|definitions| !definitions.is_empty());
    let mut config = Glm47ParserConfig {
        allow_eof_recovery: false,
        ..Default::default()
    };
    let mut reframed = String::new();
    let input = if let Some(body) = block.strip_suffix(TOOL_END)
        && body.contains(TOOL_END)
    {
        // The decoder searches for its end token without argument context. Move
        // only the outer fence; literal argument bytes remain unchanged.
        loop {
            config.tool_call_end = format!("</dynamo_tool_call_{}>", uuid::Uuid::new_v4());
            if !block.contains(&config.tool_call_end) {
                break;
            }
        }
        reframed.push_str(body);
        reframed.push_str(&config.tool_call_end);
        &reframed
    } else {
        block
    };
    dynamo_parsers::tool_calling::xml::try_tool_call_parse_glm47(input, &config, tools)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(chunks: &[&str]) -> (String, Vec<String>, bool) {
        let mut framer = Glm47Framer::default();
        let mut frames: Vec<_> = chunks.iter().flat_map(|text| framer.push(text)).collect();
        let finish = framer.finish();
        frames.extend(finish.frames);
        let mut content = String::new();
        let mut calls = Vec::new();
        for frame in frames {
            match frame {
                Glm47Frame::Text(text) => content.push_str(&text),
                Glm47Frame::ToolBlock(block) => calls.push(block),
            }
        }
        (content, calls, finish.incomplete_tool_call)
    }

    fn assert_all_splits(
        text: &str,
        expected_content: &str,
        expected_calls: &[&str],
        incomplete: bool,
    ) {
        let expected_calls: Vec<_> = expected_calls.iter().map(|call| call.to_string()).collect();
        for boundary in 0..=text.len() {
            if !text.is_char_boundary(boundary) {
                continue;
            }
            assert_eq!(
                frame(&[&text[..boundary], &text[boundary..]]),
                (
                    expected_content.to_string(),
                    expected_calls.clone(),
                    incomplete
                ),
                "split at byte {boundary} in {text:?}"
            );
        }
        let chars: Vec<_> = text
            .char_indices()
            .map(|(idx, ch)| &text[idx..idx + ch.len_utf8()])
            .collect();
        assert_eq!(
            frame(&chars),
            (expected_content.to_string(), expected_calls, incomplete)
        );
    }

    #[test]
    fn complete_calls_keep_surrounding_prose() {
        let call =
            "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
        let input = format!("I'll check. {call} Done.");
        assert_all_splits(&input, "I'll check.  Done.", &[call], false);
    }

    #[test]
    fn distinctive_partial_openers_are_incomplete() {
        for suffix in [
            "<tool_",
            "<tool_cal",
            "<tool_call>",
            "<tool_call>weather<arg_key>city",
        ] {
            assert_all_splits(&format!("I'll check. {suffix}"), "I'll check. ", &[], true);
        }
    }

    #[test]
    fn ambiguous_short_prefixes_remain_plain_text() {
        for text in ["2 <", "a <t", "<tool", "<tool_box>", "Unicode π < 4"] {
            assert_all_splits(text, text, &[], false);
        }
    }

    #[test]
    fn quoted_and_code_markers_are_literal() {
        for text in [
            "The literal \"<tool_call>\" marker is prose.",
            "The literal '<tool_call>bad</tool_call>' is prose.",
            "The literal'<tool_call>bad</tool_call>' is prose.",
            "Literal `<tool_call>bad</tool_call>` is prose.",
            "Literal ``a `<tool_call>bad</tool_call>` b`` is prose.",
            "```xml\n<tool_call>bad</tool_call>\n``` remains prose.",
            "Unclosed \"<tool_call>bad</tool_call>",
            "Unclosed `<tool_call>",
        ] {
            assert_all_splits(text, text, &[], false);
        }
    }

    #[test]
    fn escaping_and_contractions_do_not_change_quote_state() {
        let call = "<tool_call>weather</tool_call>";
        let input = format!("I'm saying \"escaped \\\"<tool_call>\\\"\" then {call}");
        assert_all_splits(
            &input,
            "I'm saying \"escaped \\\"<tool_call>\\\"\" then ",
            &[call],
            false,
        );
        assert_all_splits(
            &format!("The users' request: {call}"),
            "The users' request: ",
            &[call],
            false,
        );
    }

    #[test]
    fn closing_code_fences_release_the_next_native_call() {
        let call = "<tool_call>weather</tool_call>";
        for code in [
            "`literal`",
            "``literal``",
            "```xml\n<tool_call>bad</tool_call>\n````",
        ] {
            assert_all_splits(&format!("{code}{call}"), code, &[call], false);
        }
    }

    #[test]
    fn bare_body_calls_keep_the_function_name() {
        let call = "weather <arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
        let input = format!("Checking now. {call} Done.");
        assert_all_splits(&input, "Checking now.  Done.", &[call], false);
    }

    #[test]
    fn complete_call_survives_an_incomplete_second_call() {
        let call = "<tool_call>weather</tool_call>";
        assert_all_splits(&format!("{call}<tool_cal"), "", &[call], true);
    }

    #[test]
    fn finish_is_idempotent() {
        let mut framer = Glm47Framer::default();
        framer.push("<tool_cal");
        assert!(framer.finish().incomplete_tool_call);
        let finish = framer.finish();
        assert!(!finish.incomplete_tool_call);
        assert!(finish.frames.is_empty());
    }

    #[test]
    fn completed_frames_use_the_published_decoder() {
        for call in [
            "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            "weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        ] {
            let (calls, content) = parse_block(call, None).unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].function.name, "weather");
            let arguments: serde_json::Value =
                serde_json::from_str(&calls[0].function.arguments).unwrap();
            assert_eq!(arguments["city"], "Paris");
            assert_eq!(content.as_deref(), Some(""));
        }
    }

    #[test]
    fn decoder_never_recovers_an_unclosed_call() {
        let (calls, _) = parse_block(
            "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value>",
            None,
        )
        .unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn empty_tool_definitions_do_not_disable_native_decoding() {
        let call = "<tool_call>weather</tool_call>";
        assert_eq!(parse_block(call, None).unwrap().0.len(), 1);
        assert_eq!(parse_block(call, Some(&[])).unwrap().0.len(), 1);
        let tools = [ToolDefinition {
            name: "search".to_string(),
            parameters: None,
            strict: None,
        }];
        assert!(parse_block(call, Some(&tools)).unwrap().0.is_empty());
    }

    #[test]
    fn escaped_prose_delimiters_do_not_hide_native_calls() {
        let call = "<tool_call>weather</tool_call>";
        for prose in [
            "Print \\` as text. ",
            "Print \\\" as text. ",
            "Print \\' as text. ",
        ] {
            assert_all_splits(&format!("{prose}{call}"), prose, &[call], false);
        }
        let prose = "Print \\\\`<tool_call>bad</tool_call>` as code. ";
        assert_all_splits(&format!("{prose}{call}"), prose, &[call], false);
    }

    #[test]
    fn inch_marks_do_not_hide_following_native_calls() {
        let call = "<tool_call>weather</tool_call>";
        for prose in [
            "Height is 6\". ",
            "Height is 6\" tall. ",
            "Height is 6\": checking. ",
        ] {
            assert_all_splits(&format!("{prose}{call}"), prose, &[call], false);
        }
        let literal = "Example 6\"<tool_call>bad</tool_call>\" stays prose.";
        assert_all_splits(literal, literal, &[], false);
    }

    #[test]
    fn literal_tool_closers_inside_arguments_are_not_outer_fences() {
        for value in [
            "Paris </tool_call> France",
            "\"Paris </tool_call> France\"",
            "Literal </dynamo_tool_call> and </tool_call>",
        ] {
            let call = format!(
                "<tool_call>weather<arg_key>city</arg_key><arg_value>{value}</arg_value></tool_call>"
            );
            assert_all_splits(&call, "", &[&call], false);
            let tools = [ToolDefinition {
                name: "weather".to_string(),
                parameters: Some(
                    serde_json::json!({"type":"object", "properties":{"city":{"type":"string"}}}),
                ),
                strict: None,
            }];
            let (calls, content) = parse_block(&call, Some(&tools)).unwrap();
            assert_eq!(calls.len(), 1);
            let arguments: serde_json::Value =
                serde_json::from_str(&calls[0].function.arguments).unwrap();
            assert_eq!(arguments["city"], value);
            assert_eq!(content.as_deref(), Some(""));
        }
    }

    #[test]
    fn emitted_prose_is_not_accumulated() {
        let mut framer = Glm47Framer::default();
        for _ in 0..400 {
            framer.push("Long ordinary explanatory prose. ");
            assert!(framer.tail.len() < 40);
            assert!(framer.marker.is_empty());
            assert!(framer.block.is_none());
        }
    }

    #[test]
    fn prose_flush_preserves_parser_context() {
        let mut framer = Glm47Framer::default();
        framer.push("Hello");
        assert_eq!(
            framer.flush_prose(),
            vec![Glm47Frame::Text("Hello".to_string())]
        );
        assert!(framer.flush_prose().is_empty());

        framer.push(" '<tool_call>literal");
        assert!(framer.flush_prose().is_empty());
        assert!(
            framer
                .push("</tool_call>'")
                .iter()
                .all(|frame| matches!(frame, Glm47Frame::Text(_)))
        );

        let mut framer = Glm47Framer::default();
        framer.push("hello <tool_");
        assert!(framer.flush_prose().is_empty());
        assert_eq!(
            framer.push("call>weather</tool_call>"),
            vec![
                Glm47Frame::Text("hello ".to_string()),
                Glm47Frame::ToolBlock("<tool_call>weather</tool_call>".to_string())
            ]
        );

        let mut framer = Glm47Framer::default();
        framer.push("<tool_call>wea");
        assert!(framer.flush_prose().is_empty());
        assert_eq!(
            framer.push("ther</tool_call>"),
            vec![Glm47Frame::ToolBlock(
                "<tool_call>weather</tool_call>".to_string()
            )]
        );

        let mut framer = Glm47Framer::default();
        framer.push("Print \\");
        framer.flush_prose();
        assert!(
            framer
                .push("` as text. <tool_call>weather</tool_call>")
                .iter()
                .any(|frame| matches!(frame, Glm47Frame::ToolBlock(_)))
        );
    }
}
