// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parser anomaly detection (always on).
//!
//! The chat postprocessor pairs the RAW model completion text (pre-parse, captured
//! at stream entry) with the parsed result (`reasoning_content` / `content` /
//! `tool_calls` / `finish_reason`) per choice, and at stream end emits one
//! structured `warn` event for any choice whose pairing indicates a parse defect —
//! e.g. raw text containing the tool parser's start marker while zero tool calls
//! were extracted (a silently dropped call), or tool-call markup surviving into
//! `content` (a leak) or `reasoning_content` (an absorb). Healthy requests emit
//! nothing, so this stays quiet in production and fires exactly on the
//! small-percentage failures that are otherwise unreproducible from logs.
//!
//! The reasoning parser rewrites `choice.delta.content` in place per delta, so the
//! raw text only exists at stream entry while the parsed result is only complete
//! after the tool-call jail. Two taps therefore share a per-choice accumulator: an
//! entry tap captures raw text before any parsing, and an exit tap captures the
//! parsed deltas and evaluates the anomaly predicates at stream end.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::Stream;
use futures::stream::{self, StreamExt};

use dynamo_protocols::types::ChatCompletionMessageContent;
use dynamo_runtime::protocols::annotated::Annotated;

use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

/// Shared per-choice accumulator for the RAW (pre-parse) completion text, keyed by
/// choice index. Populated by the entry tap, read by the exit tap at stream end.
///
/// Each delta is stored as its own chunk: parse failures in the streaming parsers
/// are frequently chunk-boundary-dependent (a marker split across deltas parses
/// differently from the same bytes in one delta), so the boundaries are required
/// to *reproduce* a failure, not just to see it.
#[derive(Clone, Default)]
pub struct RawAccumulator(Arc<Mutex<HashMap<u32, Vec<String>>>>);

impl RawAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&self, index: u32, text: &str) {
        let mut guard = self
            .0
            .lock()
            .expect("parser-debug raw accumulator poisoned");
        guard.entry(index).or_default().push(text.to_string());
    }

    fn snapshot(&self) -> HashMap<u32, Vec<String>> {
        self.0
            .lock()
            .expect("parser-debug raw accumulator poisoned")
            .clone()
    }
}

/// One emitted debug record per choice: the raw completion next to the parsed result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParserDebugRecord {
    pub stream_id: String,
    pub choice_index: u32,
    pub reasoning_parser: String,
    pub tool_parser: String,
    /// Full pre-parse completion text (concatenated) — grep this for expected markup.
    pub raw: String,
    /// The same text split exactly as the deltas arrived — replay these chunks
    /// through the parser to reproduce chunk-boundary-dependent failures.
    pub raw_chunks: Vec<String>,
    pub reasoning_content: String,
    pub content: String,
    pub tool_call_count: usize,
    pub tool_call_names: Vec<String>,
    pub finish_reason: String,
}

/// Per-choice accumulator for the PARSED outputs, built by the exit tap.
#[derive(Default)]
struct ParsedAccum {
    reasoning_content: String,
    content: String,
    tool_call_names: Vec<String>,
    tool_call_count: usize,
    finish_reason: Option<String>,
}

/// Build one record per choice from the accumulated raw + parsed state.
///
/// Pure: takes the captured maps and parser names, returns the records. Tests assert
/// on the returned structs rather than on emitted log output.
fn build_records(
    stream_id: &str,
    reasoning_parser: &str,
    tool_parser: &str,
    raw: &HashMap<u32, Vec<String>>,
    parsed: &HashMap<u32, ParsedAccum>,
) -> Vec<ParserDebugRecord> {
    // Union of choice indices seen on either side. A choice with raw but no parsed
    // output (or vice versa) still gets a record so the gap is visible.
    let mut indices: Vec<u32> = raw.keys().chain(parsed.keys()).copied().collect();
    indices.sort_unstable();
    indices.dedup();

    indices
        .into_iter()
        .map(|index| {
            let p = parsed.get(&index);
            let raw_chunks = raw.get(&index).cloned().unwrap_or_default();
            ParserDebugRecord {
                stream_id: stream_id.to_string(),
                choice_index: index,
                reasoning_parser: reasoning_parser.to_string(),
                tool_parser: tool_parser.to_string(),
                raw: raw_chunks.concat(),
                raw_chunks,
                reasoning_content: p.map(|p| p.reasoning_content.clone()).unwrap_or_default(),
                content: p.map(|p| p.content.clone()).unwrap_or_default(),
                tool_call_count: p.map(|p| p.tool_call_count).unwrap_or(0),
                tool_call_names: p.map(|p| p.tool_call_names.clone()).unwrap_or_default(),
                finish_reason: p.and_then(|p| p.finish_reason.clone()).unwrap_or_default(),
            }
        })
        .collect()
}

impl ParserDebugRecord {
    /// Anomaly predicates over the raw-vs-parsed pairing. Returns the names of
    /// every predicate that fired (empty = healthy). `tool_start_tokens` are the
    /// configured tool parser's start markers (e.g. `<tool_call>`); marker-based
    /// predicates are skipped when no tool parser is configured.
    ///
    /// The three marker predicates map 1:1 to the production failure shapes from
    /// the Nemotron tool-call-leak incident: a start marker in `raw` with zero
    /// extracted calls is a silently dropped call (the model emitted a block the
    /// parser could not extract — e.g. a missing `<function=NAME>` line); a marker
    /// in `content` is a leak (parser-owned markup reached the client); a marker in
    /// `reasoning_content` is an absorb (the block was swallowed into reasoning).
    /// `all_output_lost` is marker-independent and catches total-loss shapes even
    /// for parsers whose markers are unknown here.
    pub fn anomalies(&self, tool_start_tokens: &[String]) -> Vec<&'static str> {
        let mut found = Vec::new();
        let contains_marker = |text: &str| {
            tool_start_tokens
                .iter()
                .any(|t| !t.is_empty() && text.contains(t.as_str()))
        };

        if !tool_start_tokens.is_empty() {
            if self.tool_call_count == 0 && contains_marker(&self.raw) {
                found.push("tool_call_dropped");
            }
            if contains_marker(&self.content) {
                found.push("tool_markup_in_content");
            }
            if contains_marker(&self.reasoning_content) {
                found.push("tool_markup_in_reasoning");
            }
        }

        if !self.raw.trim().is_empty()
            && self.content.trim().is_empty()
            && self.reasoning_content.trim().is_empty()
            && self.tool_call_count == 0
        {
            found.push("all_output_lost");
        }

        found
    }
}

/// Emit one structured `warn` event per ANOMALOUS record (healthy records emit
/// nothing). Each field is a separate `tracing` field (not a formatted string) so
/// the event is queryable in `DYN_LOGGING_JSONL` deployments.
fn emit_anomalous(records: &[ParserDebugRecord], tool_start_tokens: &[String]) {
    for r in records {
        let anomalies = r.anomalies(tool_start_tokens);
        if anomalies.is_empty() {
            continue;
        }
        tracing::warn!(
            target: "parser_debug",
            anomalies = ?anomalies,
            stream_id = %r.stream_id,
            choice_index = r.choice_index,
            reasoning_parser = %r.reasoning_parser,
            tool_parser = %r.tool_parser,
            raw = %r.raw,
            raw_chunks = ?r.raw_chunks,
            reasoning_content = %r.reasoning_content,
            content = %r.content,
            tool_call_count = r.tool_call_count,
            tool_call_names = ?r.tool_call_names,
            finish_reason = %r.finish_reason,
            "parser anomaly: raw completion vs parsed output disagree"
        );
    }
}

/// Entry tap: accumulate the RAW (pre-parse) text content per choice, then yield each
/// response unchanged. Multimodal `Parts` content is skipped (text-only deltas are
/// what the parsing seam operates on).
pub fn tap_raw<S>(
    stream: S,
    acc: RawAccumulator,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    stream::unfold((Box::pin(stream), acc), |(mut stream, acc)| async move {
        let response = stream.next().await?;
        if let Some(data) = response.data.as_ref() {
            for choice in &data.inner.choices {
                if let Some(ChatCompletionMessageContent::Text(text)) =
                    choice.delta.content.as_ref()
                {
                    acc.push(choice.index, text);
                }
            }
        }
        Some((response, (stream, acc)))
    })
    .fuse()
}

/// Exit tap: accumulate the PARSED outputs per choice as they flow, yield each
/// response unchanged, and at stream end emit a `warn` event for every choice whose
/// raw-vs-parsed pairing is anomalous (healthy choices emit nothing). Production
/// entry point — see [`tap_parsed_collecting`] for the callback-generic core that
/// tests drive with a capturing sink.
pub fn tap_parsed_and_emit<S>(
    stream: S,
    raw: RawAccumulator,
    reasoning_parser: Option<String>,
    tool_parser: Option<String>,
    tool_start_tokens: Vec<String>,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    tap_parsed_collecting(stream, raw, reasoning_parser, tool_parser, move |records| {
        emit_anomalous(&records, &tool_start_tokens)
    })
}

/// Callback-generic exit tap: same accumulation as [`tap_parsed_and_emit`], but
/// routes the built records to `on_end` at stream end instead of hardcoding `emit`.
/// Lets tests assert on `Vec<ParserDebugRecord>` directly (no log capture).
pub fn tap_parsed_collecting<S, F>(
    stream: S,
    raw: RawAccumulator,
    reasoning_parser: Option<String>,
    tool_parser: Option<String>,
    on_end: F,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    F: FnOnce(Vec<ParserDebugRecord>) + Send + 'static,
{
    struct ExitState<St, G> {
        stream: std::pin::Pin<Box<St>>,
        raw: RawAccumulator,
        reasoning_parser: String,
        tool_parser: String,
        stream_id: String,
        parsed: HashMap<u32, ParsedAccum>,
        on_end: Option<G>,
    }

    let state = ExitState {
        stream: Box::pin(stream),
        raw,
        reasoning_parser: reasoning_parser.unwrap_or_else(|| "-".to_string()),
        tool_parser: tool_parser.unwrap_or_else(|| "-".to_string()),
        stream_id: String::new(),
        parsed: HashMap::new(),
        on_end: Some(on_end),
    };

    stream::unfold(state, |mut state| async move {
        match state.stream.next().await {
            Some(response) => {
                if let Some(data) = response.data.as_ref() {
                    if !data.inner.id.is_empty() {
                        state.stream_id = data.inner.id.clone();
                    }
                    for choice in &data.inner.choices {
                        let acc = state.parsed.entry(choice.index).or_default();
                        if let Some(reasoning) = choice.delta.reasoning_content.as_ref() {
                            acc.reasoning_content.push_str(reasoning);
                        }
                        if let Some(ChatCompletionMessageContent::Text(text)) =
                            choice.delta.content.as_ref()
                        {
                            acc.content.push_str(text);
                        }
                        if let Some(tool_calls) = choice.delta.tool_calls.as_ref() {
                            for tc in tool_calls {
                                if let Some(name) =
                                    tc.function.as_ref().and_then(|f| f.name.clone())
                                {
                                    // Name arrives once per tool call (first delta).
                                    acc.tool_call_count += 1;
                                    acc.tool_call_names.push(name);
                                }
                            }
                        }
                        if let Some(fr) = choice.finish_reason {
                            acc.finish_reason = Some(format!("{fr:?}"));
                        }
                    }
                }
                Some((response, state))
            }
            None => {
                if let Some(on_end) = state.on_end.take() {
                    let records = build_records(
                        &state.stream_id,
                        &state.reasoning_parser,
                        &state.tool_parser,
                        &state.raw.snapshot(),
                        &state.parsed,
                    );
                    on_end(records);
                }
                None
            }
        }
    })
    .fuse()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accum(reasoning: &str, content: &str, names: &[&str], finish: Option<&str>) -> ParsedAccum {
        ParsedAccum {
            reasoning_content: reasoning.to_string(),
            content: content.to_string(),
            tool_call_names: names.iter().map(|s| s.to_string()).collect(),
            tool_call_count: names.len(),
            finish_reason: finish.map(|s| s.to_string()),
        }
    }

    #[test]
    fn one_record_per_choice() {
        let mut raw = HashMap::new();
        raw.insert(0, vec!["ra".to_string(), "w0".to_string()]);
        raw.insert(1, vec!["raw1".to_string()]);
        let mut parsed = HashMap::new();
        parsed.insert(0, accum("r0", "c0", &["get_weather"], Some("Stop")));
        parsed.insert(1, accum("", "c1", &[], Some("Stop")));

        let records = build_records("id-x", "qwen3", "harmony", &raw, &parsed);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].choice_index, 0);
        assert_eq!(records[0].raw, "raw0");
        assert_eq!(records[0].tool_call_names, vec!["get_weather"]);
        assert_eq!(records[0].tool_call_count, 1);
        assert_eq!(records[0].reasoning_parser, "qwen3");
        assert_eq!(records[1].choice_index, 1);
        assert_eq!(records[1].tool_call_count, 0);
    }

    #[test]
    fn raw_without_parsed_still_gets_record() {
        // A choice that produced raw text but no parsed output (parser dropped it).
        let mut raw = HashMap::new();
        raw.insert(0, vec!["<function=foo>{}".to_string()]);
        let parsed = HashMap::new();

        let records = build_records("id", "-", "harmony", &raw, &parsed);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].raw, "<function=foo>{}");
        assert_eq!(records[0].tool_call_count, 0);
        assert_eq!(records[0].content, "");
        assert_eq!(records[0].finish_reason, "");
    }

    #[test]
    fn empty_state_yields_no_records() {
        let records = build_records("id", "-", "-", &HashMap::new(), &HashMap::new());
        assert!(records.is_empty());
    }

    fn record(
        raw: &str,
        reasoning: &str,
        content: &str,
        tool_call_count: usize,
    ) -> ParserDebugRecord {
        ParserDebugRecord {
            stream_id: "sid".to_string(),
            choice_index: 0,
            reasoning_parser: "nemotron3".to_string(),
            tool_parser: "qwen3_coder".to_string(),
            raw: raw.to_string(),
            raw_chunks: vec![raw.to_string()],
            reasoning_content: reasoning.to_string(),
            content: content.to_string(),
            tool_call_count,
            tool_call_names: Vec::new(),
            finish_reason: "Stop".to_string(),
        }
    }

    fn markers() -> Vec<String> {
        vec!["<tool_call>".to_string()]
    }

    #[test]
    fn anomaly_dropped_call() {
        // The Nemotron empty-stop shape: raw has a tool block (sans <function=NAME>),
        // jail strips it, nothing extracted — content is just the stripped newlines.
        let r = record(
            "prose</think>\n<tool_call>\n<parameter=command>\nls\n</parameter>\n</tool_call>\n",
            "prose",
            "\n\n",
            0,
        );
        assert_eq!(r.anomalies(&markers()), vec!["tool_call_dropped"]);
    }

    #[test]
    fn anomaly_leak_into_content() {
        let r = record("raw", "", "before <tool_call>x</tool_call> after", 0);
        assert!(r.anomalies(&markers()).contains(&"tool_markup_in_content"));
    }

    #[test]
    fn anomaly_absorb_into_reasoning() {
        // The chunk-split absorb shape: whole tool block swallowed into reasoning.
        let r = record("raw", "plan <tool_call>x</tool_call>", "", 0);
        assert!(
            r.anomalies(&markers())
                .contains(&"tool_markup_in_reasoning")
        );
    }

    #[test]
    fn anomaly_all_output_lost_without_markers() {
        // Marker-independent: everything vanished even though raw had text.
        let r = record("some raw text", "", "  \n", 0);
        assert_eq!(r.anomalies(&[]), vec!["all_output_lost"]);
    }

    #[test]
    fn healthy_tool_call_is_not_anomalous() {
        // Marker in raw is fine when a call WAS extracted and markup stayed out of
        // content/reasoning.
        let r = record("<tool_call>...</tool_call>", "plan", "done", 1);
        assert!(r.anomalies(&markers()).is_empty());
    }

    #[test]
    fn healthy_plain_text_is_not_anomalous() {
        let r = record("hello world", "", "hello world", 0);
        assert!(r.anomalies(&markers()).is_empty());
    }

    #[test]
    fn raw_accumulator_concats_per_choice() {
        let acc = RawAccumulator::new();
        acc.push(0, "a");
        acc.push(0, "b");
        acc.push(1, "x");
        let snap = acc.snapshot();
        assert_eq!(
            snap.get(&0).unwrap(),
            &vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(snap.get(&1).unwrap(), &vec!["x".to_string()]);
    }

    // --- tap-driving tests (accumulation behavior, synthetic streams) ---

    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
        CreateChatCompletionStreamResponse, FinishReason, FunctionCallStream, Role,
    };
    use std::sync::Mutex as StdMutex;

    /// Per-choice test delta: (index, content, reasoning, tool_name, finish).
    type ChoiceSpec<'a> = (
        u32,
        Option<&'a str>,
        Option<&'a str>,
        Option<&'a str>,
        Option<FinishReason>,
    );

    /// One stream chunk built from per-choice specs.
    #[allow(deprecated)]
    fn chunk(choices: &[ChoiceSpec<'_>]) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let choices = choices
            .iter()
            .map(
                |(index, content, reasoning, tool_name, finish)| ChatChoiceStream {
                    index: *index,
                    delta: ChatCompletionStreamResponseDelta {
                        role: Some(Role::Assistant),
                        content: content.map(|c| ChatCompletionMessageContent::Text(c.to_string())),
                        tool_calls: tool_name.map(|n| {
                            vec![ChatCompletionMessageToolCallChunk {
                                index: 0,
                                function: Some(FunctionCallStream {
                                    name: Some(n.to_string()),
                                    arguments: None,
                                }),
                                ..Default::default()
                            }]
                        }),
                        function_call: None,
                        refusal: None,
                        reasoning_content: reasoning.map(|r| r.to_string()),
                    },
                    finish_reason: *finish,
                    logprobs: None,
                },
            )
            .collect();
        Annotated::from_data(NvCreateChatCompletionStreamResponse {
            inner: CreateChatCompletionStreamResponse {
                id: "sid".to_string(),
                choices,
                created: 0,
                model: "m".to_string(),
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            },
            nvext: None,
        })
    }

    /// Drive tap_raw -> tap_parsed_collecting over `chunks` and return the records.
    fn drive(
        chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
        reasoning_parser: Option<&str>,
        tool_parser: Option<&str>,
    ) -> Vec<ParserDebugRecord> {
        let sink = Arc::new(StdMutex::new(Vec::new()));
        let sink_w = sink.clone();
        let raw = RawAccumulator::new();
        // Entry tap captures raw, exit tap accumulates parsed. No parser in between
        // here: these tests assert the tap accumulation/record-build, not parsing.
        let entry = tap_raw(stream::iter(chunks), raw.clone());
        let exit = tap_parsed_collecting(
            entry,
            raw,
            reasoning_parser.map(str::to_string),
            tool_parser.map(str::to_string),
            move |records| *sink_w.lock().unwrap() = records,
        );
        futures::executor::block_on(exit.collect::<Vec<_>>());
        Arc::try_unwrap(sink).unwrap().into_inner().unwrap()
    }

    #[test]
    fn tap_finish_reason_takes_last_non_none() {
        let records = drive(
            vec![
                chunk(&[(0, Some("hi"), None, None, None)]),
                chunk(&[(0, Some(" there"), None, None, Some(FinishReason::Stop))]),
                chunk(&[(0, None, None, None, None)]),
            ],
            Some("qwen3"),
            None,
        );
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].content, "hi there");
        assert_eq!(records[0].finish_reason, "Stop");
        assert_eq!(records[0].reasoning_parser, "qwen3");
        assert_eq!(records[0].tool_parser, "-");
    }

    #[test]
    fn tap_collects_tool_call_names_and_count() {
        let records = drive(
            vec![
                chunk(&[(0, None, None, Some("get_weather"), None)]),
                chunk(&[(
                    0,
                    None,
                    None,
                    Some("get_time"),
                    Some(FinishReason::ToolCalls),
                )]),
            ],
            None,
            Some("harmony"),
        );
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].tool_call_count, 2);
        assert_eq!(records[0].tool_call_names, vec!["get_weather", "get_time"]);
        assert_eq!(records[0].finish_reason, "ToolCalls");
    }

    #[test]
    fn tap_two_choices_distinct_raw() {
        let records = drive(
            vec![
                chunk(&[
                    (0, Some("a0"), None, None, None),
                    (1, Some("b1"), None, None, None),
                ]),
                chunk(&[(0, Some("a0b"), None, None, None)]),
            ],
            None,
            None,
        );
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].choice_index, 0);
        assert_eq!(records[0].raw, "a0a0b");
        assert_eq!(records[1].choice_index, 1);
        assert_eq!(records[1].raw, "b1");
    }

    #[test]
    fn tap_reasoning_split_from_content() {
        // Simulates post-reasoning-parser deltas: reasoning_content + content set
        // separately. Raw (entry tap) only sees the text content here.
        let records = drive(
            vec![
                chunk(&[(0, None, Some("plan"), None, None)]),
                chunk(&[(0, Some("answer"), None, None, Some(FinishReason::Stop))]),
            ],
            Some("qwen3"),
            None,
        );
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].reasoning_content, "plan");
        assert_eq!(records[0].content, "answer");
        // Entry tap saw only the "answer" text delta as raw (reasoning delta has no content).
        assert_eq!(records[0].raw, "answer");
    }
}
