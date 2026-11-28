// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A toy grammar for trying out parser plugin designs:
//!
//! ```text
//! <dummy:think>reasoning</dummy:think>visible text
//! <dummy:call>{"name": "get_weather", "arguments": {"location": "SF"}}</dummy:call>
//! ```

use anyhow::Result;
use dynamo_parsers_v2::{
    Tool, ToolCallDelta, UnifiedParser, UnifiedParserInit, UnifiedParserOutput,
    UnifiedParserStartingState, UnifiedToolOutputMode,
};

pub const FAMILY: &str = "dummy";
pub const THINK_OPEN: &str = "<dummy:think>";
pub const THINK_CLOSE: &str = "</dummy:think>";
pub const CALL_OPEN: &str = "<dummy:call>";
pub const CALL_CLOSE: &str = "</dummy:call>";

/// What the host needs to know about this family, published next to the parser.
pub struct Dummy;

impl dynamo_parsers_v2::UnifiedFamily for Dummy {
    const INFO: dynamo_parsers_v2::UnifiedFamilyInfo =
        dynamo_parsers_v2::UnifiedFamilyInfo::new(FAMILY)
            .aliases(&["dummy-alias"])
            .reasoning(THINK_OPEN, THINK_CLOSE)
            .tool_call_opener(CALL_OPEN);

    fn create(tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
        factory(tools)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Mode {
    #[default]
    Text,
    Reasoning,
    Call,
}

#[derive(Debug, Default)]
pub struct DummyParser {
    mode: Mode,
    /// Bytes not yet safe to emit: a possible partial marker, or an open call body.
    pending: String,
    next_tool_index: usize,
    /// `Some` when guided decoding replaced the native grammar with bare JSON.
    guided: Option<UnifiedToolOutputMode>,
}

pub fn factory(_tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
    Ok(Box::new(DummyParser::default()))
}

/// Length of the longest suffix of `text` that is a proper prefix of any marker.
fn partial_marker_len(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .flat_map(|marker| (1..marker.len()).rev().map(move |k| &marker[..k]))
        .filter(|prefix| text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

impl DummyParser {
    /// Guided payloads are buffered and judged once complete. Named: the arguments
    /// object. Required: one `{name, parameters}` object or an array of them.
    fn finish_guided(&mut self, named_tool: Option<String>, output: &mut UnifiedParserOutput) {
        let payload = std::mem::take(&mut self.pending);
        let calls: Option<Vec<(String, serde_json::Value)>> =
            serde_json::from_str::<serde_json::Value>(&payload)
                .ok()
                .and_then(|value| match named_tool {
                    Some(name) => value.is_object().then(|| vec![(name, value)]),
                    None => {
                        let items = match value {
                            serde_json::Value::Array(items) => items,
                            object => vec![object],
                        };
                        items
                            .into_iter()
                            .map(|item| {
                                let name = item.get("name")?.as_str()?.to_string();
                                let args = item.get("parameters").or_else(|| item.get("arguments"));
                                Some((name, args?.clone()))
                            })
                            .collect()
                    }
                });
        match calls {
            Some(calls) => {
                for (name, arguments) in calls {
                    output.push_call(ToolCallDelta {
                        tool_index: self.next_tool_index,
                        name: Some(name),
                        arguments: arguments.to_string(),
                        complete: true,
                    });
                    self.next_tool_index += 1;
                }
            }
            None => output.push_text(payload),
        }
    }

    fn emit_call(&mut self, body: &str, output: &mut UnifiedParserOutput) {
        let parsed: Option<(String, String)> = serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(|value| {
                let name = value.get("name")?.as_str()?.to_string();
                let arguments = value
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                Some((name, arguments.to_string()))
            });
        match parsed {
            Some((name, arguments)) => {
                output.push_call(ToolCallDelta {
                    tool_index: self.next_tool_index,
                    name: Some(name),
                    arguments,
                    complete: true,
                });
                self.next_tool_index += 1;
            }
            // Recovery: a malformed call is shown to the user as the text the model wrote.
            None => output.push_text(format!("{CALL_OPEN}{body}{CALL_CLOSE}")),
        }
    }
}

impl UnifiedParser for DummyParser {
    fn initialize_request(&mut self, init: UnifiedParserInit) -> Result<()> {
        self.mode = match init.starting_state {
            UnifiedParserStartingState::Reasoning => Mode::Reasoning,
            UnifiedParserStartingState::None | UnifiedParserStartingState::Response => Mode::Text,
        };
        self.guided = match init.tool_output_mode {
            UnifiedToolOutputMode::Native => None,
            guided => Some(guided),
        };
        Ok(())
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        self.pending.push_str(delta);
        if self.guided.is_some() && self.mode == Mode::Text {
            return Ok(());
        }
        loop {
            match self.mode {
                Mode::Text => {
                    let next = [(THINK_OPEN, Mode::Reasoning), (CALL_OPEN, Mode::Call)]
                        .into_iter()
                        .filter_map(|(marker, mode)| {
                            self.pending.find(marker).map(|at| (at, marker, mode))
                        })
                        .min_by_key(|(at, _, _)| *at);
                    if let Some((at, marker, mode)) = next {
                        output.push_text(&self.pending[..at]);
                        self.pending.drain(..at + marker.len());
                        self.mode = mode;
                        continue;
                    }
                    let keep = partial_marker_len(&self.pending, &[THINK_OPEN, CALL_OPEN]);
                    let cut = self.pending.len() - keep;
                    output.push_text(&self.pending[..cut]);
                    self.pending.drain(..cut);
                    return Ok(());
                }
                Mode::Reasoning => {
                    if let Some(at) = self.pending.find(THINK_CLOSE) {
                        output.push_reasoning(&self.pending[..at]);
                        self.pending.drain(..at + THINK_CLOSE.len());
                        self.mode = Mode::Text;
                        if self.guided.is_some() {
                            return Ok(());
                        }
                        continue;
                    }
                    let keep = partial_marker_len(&self.pending, &[THINK_CLOSE]);
                    let cut = self.pending.len() - keep;
                    output.push_reasoning(&self.pending[..cut]);
                    self.pending.drain(..cut);
                    return Ok(());
                }
                Mode::Call => {
                    let Some(at) = self.pending.find(CALL_CLOSE) else {
                        return Ok(());
                    };
                    let body: String = self.pending.drain(..at + CALL_CLOSE.len()).collect();
                    self.emit_call(&body[..at], output);
                    self.mode = Mode::Text;
                }
            }
        }
    }

    /// Open reasoning is promoted to reasoning; an unterminated call is dropped.
    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        if let Some(UnifiedToolOutputMode::GuidedJson { named_tool }) = self.guided.clone()
            && self.mode == Mode::Text
        {
            self.finish_guided(named_tool, &mut output);
            return Ok(output);
        }
        let tail = std::mem::take(&mut self.pending);
        match std::mem::take(&mut self.mode) {
            Mode::Text => output.push_text(tail),
            Mode::Reasoning => output.push_reasoning(tail),
            Mode::Call => {}
        }
        Ok(output)
    }

    fn reset(&mut self) -> String {
        let guided = self.guided.take();
        let pending = std::mem::take(self).pending;
        self.guided = guided;
        pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers_v2::{UnifiedEvent, UnifiedParserExt, assemble};

    const SAMPLE: &str = concat!(
        "<dummy:think>user wants weather</dummy:think>Checking. ",
        r#"<dummy:call>{"name":"get_weather","arguments":{"location":"SF"}}</dummy:call>"#,
        " done"
    );

    fn run(chunks: &[&str]) -> Vec<UnifiedEvent> {
        let mut parser = DummyParser::default();
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            parser.parse_into(chunk, &mut output).unwrap();
        }
        output.events.extend(parser.finish().unwrap().events);
        assemble(&output.events)
    }

    #[test]
    fn parses_reasoning_text_and_call() {
        let events = DummyParser::default().parse_complete(SAMPLE).unwrap();
        assert_eq!(
            serde_json::to_value(&events).unwrap(),
            serde_json::json!([
                {"kind": "reasoning", "text": "user wants weather"},
                {"kind": "text", "text": "Checking. "},
                {"kind": "tool_call", "name": "get_weather", "arguments": {"location": "SF"}},
                {"kind": "text", "text": " done"},
            ])
        );
    }

    #[test]
    fn every_split_point_gives_the_same_output() {
        let whole = run(&[SAMPLE]);
        for cut in 1..SAMPLE.len() {
            assert_eq!(
                run(&[&SAMPLE[..cut], &SAMPLE[cut..]]),
                whole,
                "cut at {cut}"
            );
        }
        let bytes: Vec<String> = SAMPLE.chars().map(String::from).collect();
        let bytes: Vec<&str> = bytes.iter().map(String::as_str).collect();
        assert_eq!(run(&bytes), whole);
    }

    #[test]
    fn guided_required_json_becomes_calls() {
        let mut parser = DummyParser::default();
        parser
            .initialize_request(UnifiedParserInit {
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                ..Default::default()
            })
            .unwrap();
        let events = parser
            .parse_complete(r#"[{"name":"get_weather","parameters":{"location":"SF"}}]"#)
            .unwrap();
        assert_eq!(
            serde_json::to_value(&events).unwrap(),
            serde_json::json!([
                {"kind": "tool_call", "name": "get_weather", "arguments": {"location": "SF"}},
            ])
        );
    }

    #[test]
    fn malformed_call_is_recovered_as_text_and_open_reasoning_is_flushed() {
        let events = run(&["<dummy:call>nope</dummy:call><dummy:think>still thin"]);
        assert_eq!(
            serde_json::to_value(&events).unwrap(),
            serde_json::json!([
                {"kind": "text", "text": "<dummy:call>nope</dummy:call>"},
                {"kind": "reasoning", "text": "still thin"},
            ])
        );
    }
}
