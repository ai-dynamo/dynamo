// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in ordered reasoning, text, and tool-call parsing for Kimi K3.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::LazyLock;

use async_stream::stream;
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers_v2::{
    KIMI_K3_FAMILY, Tool, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput,
    UnifiedParserPrefill, create_unified_parser_for_family,
};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta, FinishReason,
    FunctionCall, FunctionCallStream, FunctionType,
};
use dynamo_runtime::config::{env_is_truthy, environment_names::llm as env_llm};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use uuid::Uuid;

use super::NvCreateChatCompletionStreamResponse;

pub(crate) fn enabled() -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| env_is_truthy(env_llm::DYN_ENABLE_KIMI_K3_UNIFIED_PARSER));
    *ENABLED
}

pub(crate) fn configured(tool_call_parser: Option<&str>, reasoning_parser: Option<&str>) -> bool {
    tool_call_parser == Some(KIMI_K3_FAMILY) && reasoning_parser == Some(KIMI_K3_FAMILY)
}

pub(crate) struct CompleteOutput {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<ChatCompletionMessageToolCall>,
}

pub(crate) fn parse_complete(content: &str) -> anyhow::Result<CompleteOutput> {
    let mut parser = create_unified_parser_for_family(KIMI_K3_FAMILY, &[])?;
    parser.initialize(detect_prefill(content))?;
    let output = parser.parse_complete(content)?;

    let mut text = String::new();
    let mut reasoning = String::new();
    let mut calls = BTreeMap::<usize, (Option<String>, Option<String>, String)>::new();
    for event in output.events {
        match event {
            UnifiedParserEvent::Text(delta) => text.push_str(&delta),
            UnifiedParserEvent::Reasoning(delta) => reasoning.push_str(&delta),
            UnifiedParserEvent::ToolCall(call) => {
                let entry = calls.entry(call.tool_index).or_insert_with(|| {
                    (
                        parser.tool_call_id(call.tool_index).map(str::to_string),
                        None,
                        String::new(),
                    )
                });
                if entry.1.is_none() {
                    entry.1 = call.name;
                }
                entry.2.push_str(&call.arguments);
            }
        }
    }
    let tool_calls = calls
        .into_values()
        .filter_map(|(id, name, arguments)| {
            Some(ChatCompletionMessageToolCall {
                id: id.unwrap_or_else(|| format!("call-{}", Uuid::new_v4())),
                r#type: FunctionType::Function,
                function: FunctionCall {
                    name: name?,
                    arguments,
                },
            })
        })
        .collect();

    Ok(CompleteOutput {
        text,
        reasoning,
        tool_calls,
    })
}

fn detect_prefill(content: &str) -> UnifiedParserPrefill {
    const THINK_CLOSE: &str = "<|close|>think<|sep|>";
    const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
    const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
    const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";

    if content.starts_with("<|open|>") {
        return UnifiedParserPrefill::None;
    }
    if content.find(THINK_CLOSE).is_some_and(|think_end| {
        content
            .find(RESPONSE_OPEN)
            .is_none_or(|response_open| think_end < response_open)
    }) {
        return UnifiedParserPrefill::Reasoning;
    }
    if content.contains(RESPONSE_CLOSE) || content.contains(TOOLS_OPEN) {
        return UnifiedParserPrefill::Response;
    }
    UnifiedParserPrefill::None
}

fn to_v2_tools(tools: Option<&[ToolDefinition]>) -> Vec<Tool> {
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

struct ChoiceState {
    parser: Box<dyn UnifiedParser>,
    parser_failed: bool,
    opened_calls: HashSet<usize>,
    tool_emitted: bool,
}

impl ChoiceState {
    fn new(tools: &[Tool], prefill: UnifiedParserPrefill) -> anyhow::Result<Self> {
        let mut parser = create_unified_parser_for_family(KIMI_K3_FAMILY, tools)?;
        parser.initialize(prefill)?;
        Ok(Self {
            parser,
            parser_failed: false,
            opened_calls: HashSet::new(),
            tool_emitted: false,
        })
    }

    fn process_text(&mut self, delta: &str) -> UnifiedParserOutput {
        if self.parser_failed {
            let mut output = UnifiedParserOutput::default();
            output.push_text(delta);
            return output;
        }

        let mut output = UnifiedParserOutput::default();
        if let Err(error) = self.parser.parse_into(delta, &mut output) {
            tracing::warn!(
                error = %error,
                "Kimi K3 unified parser failed; falling back to plain text"
            );
            self.parser_failed = true;
            let recovered = self.parser.reset();
            if recovered.is_empty() && output.events.is_empty() {
                output.push_text(delta);
            } else {
                output.push_text(recovered);
            }
        }
        output
    }

    fn finish(&mut self) -> UnifiedParserOutput {
        if self.parser_failed {
            return UnifiedParserOutput::default();
        }

        match self.parser.finish() {
            Ok(output) => output,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "Kimi K3 unified parser finish failed; recovering buffered text"
                );
                self.parser_failed = true;
                let mut output = UnifiedParserOutput::default();
                output.push_text(self.parser.reset());
                output
            }
        }
    }

    fn event_choices(
        &mut self,
        original: &ChatChoiceStream,
        events: Vec<UnifiedParserEvent>,
        finish_reason: Option<FinishReason>,
    ) -> Vec<ChatChoiceStream> {
        let mut choices = Vec::with_capacity(events.len().max(1));
        let event_count = events.len();

        for (event_index, event) in events.into_iter().enumerate() {
            let mut choice = cleared_choice(original);
            if event_index == 0 {
                choice.delta.role = original.delta.role;
                choice.delta.refusal = original.delta.refusal.clone();
            }

            match event {
                UnifiedParserEvent::Text(text) => {
                    choice.delta.content = Some(ChatCompletionMessageContent::Text(text));
                }
                UnifiedParserEvent::Reasoning(reasoning) => {
                    choice.delta.reasoning_content = Some(reasoning);
                }
                UnifiedParserEvent::ToolCall(call) => {
                    self.tool_emitted = true;
                    let first = self.opened_calls.insert(call.tool_index);
                    choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
                        index: call.tool_index as u32,
                        id: first.then(|| {
                            self.parser
                                .tool_call_id(call.tool_index)
                                .map(str::to_string)
                                .unwrap_or_else(|| format!("call-{}", Uuid::new_v4()))
                        }),
                        r#type: first.then_some(FunctionType::Function),
                        function: Some(FunctionCallStream {
                            name: first.then_some(call.name).flatten(),
                            arguments: Some(call.arguments),
                        }),
                    }]);
                }
            }

            if event_index + 1 == event_count {
                choice.finish_reason = self.normalize_finish_reason(finish_reason);
            }
            choices.push(choice);
        }

        if choices.is_empty()
            && (original.delta.role.is_some()
                || original.delta.refusal.is_some()
                || finish_reason.is_some())
        {
            let mut choice = cleared_choice(original);
            choice.delta.role = original.delta.role;
            choice.delta.refusal = original.delta.refusal.clone();
            choice.finish_reason = self.normalize_finish_reason(finish_reason);
            choices.push(choice);
        }

        choices
    }

    fn normalize_finish_reason(&self, finish_reason: Option<FinishReason>) -> Option<FinishReason> {
        if finish_reason == Some(FinishReason::Stop) && self.tool_emitted {
            Some(FinishReason::ToolCalls)
        } else {
            finish_reason
        }
    }
}

#[allow(deprecated)]
fn cleared_choice(original: &ChatChoiceStream) -> ChatChoiceStream {
    ChatChoiceStream {
        index: original.index,
        delta: ChatCompletionStreamResponseDelta {
            role: None,
            content: None,
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        // Channel markers and output text no longer align with the backend's
        // raw token stream once parsing rewrites the choice.
        logprobs: None,
    }
}

fn finish_unterminated_choices(
    states: &mut HashMap<u32, ChoiceState>,
    finished: &mut HashSet<u32>,
) -> Vec<ChatChoiceStream> {
    let mut indices = states
        .keys()
        .copied()
        .filter(|index| !finished.contains(index))
        .collect::<Vec<_>>();
    indices.sort_unstable();

    let mut choices = Vec::new();
    for index in indices {
        finished.insert(index);
        let state = states
            .get_mut(&index)
            .expect("choice index came from unified parser state");
        let output = state.finish();
        #[allow(deprecated)]
        let base = ChatChoiceStream {
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
        };
        let finish_reason = state.tool_emitted.then_some(FinishReason::ToolCalls);
        choices.extend(state.event_choices(&base, output.events, finish_reason));
    }
    choices
}

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

/// Apply one Kimi K3 unified parser per response choice.
pub(crate) fn apply_stream<S>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    prefill: UnifiedParserPrefill,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let tools = to_v2_tools(tool_definitions.as_deref());
    stream! {
        if ChoiceState::new(&tools, prefill).is_err() {
            tracing::warn!("Kimi K3 unified parser construction failed; passing stream through");
            tokio::pin!(stream_in);
            while let Some(response) = stream_in.next().await {
                yield response;
            }
            return;
        }

        let mut states = HashMap::<u32, ChoiceState>::new();
        let mut finished = HashSet::<u32>::new();
        let mut template: Option<NvCreateChatCompletionStreamResponse> = None;
        tokio::pin!(stream_in);

        while let Some(mut response) = stream_in.next().await {
            let Some(chat) = response.data.as_mut() else {
                yield response;
                continue;
            };

            let mut next_template = chat.clone();
            next_template.inner.choices.clear();
            next_template.inner.usage = None;
            next_template.nvext = None;
            next_template.llm_metrics = None;
            template = Some(next_template);

            if chat.inner.choices.is_empty() {
                if let Some(template) = &template {
                    for choice in finish_unterminated_choices(&mut states, &mut finished) {
                        yield response_with_choice(template, choice);
                    }
                }
                yield response;
                continue;
            }

            let original_choices = std::mem::take(&mut chat.inner.choices);
            let mut emitted = Vec::new();
            for original in original_choices {
                if matches!(original.delta.content, Some(ChatCompletionMessageContent::Parts(_)))
                    || original.delta.tool_calls.is_some()
                    || original.delta.reasoning_content.is_some()
                {
                    if original.finish_reason.is_some() {
                        finished.insert(original.index);
                    }
                    emitted.push(original);
                    continue;
                }

                let state = states.entry(original.index).or_insert_with(|| {
                    ChoiceState::new(&tools, prefill)
                        .expect("Kimi K3 parser construction validated before stream")
                });
                let mut output = UnifiedParserOutput::default();
                if let Some(ChatCompletionMessageContent::Text(text)) =
                    original.delta.content.as_ref()
                {
                    output.append(state.process_text(text));
                }

                let terminal = original.finish_reason;
                if terminal.is_some() && finished.insert(original.index) {
                    output.append(state.finish());
                }
                let mut parsed = state.event_choices(&original, output.events, terminal);
                if parsed.is_empty() {
                    // Keep marker-only input chunks as empty deltas. Besides matching
                    // the existing v2 parser's stream shape, this preserves typed
                    // llm_metrics and annotation metadata carried by that chunk.
                    parsed.push(cleared_choice(&original));
                }
                emitted.extend(parsed);
            }

            if emitted.is_empty() {
                continue;
            }

            let last = emitted.len() - 1;
            for (position, choice) in emitted.into_iter().enumerate() {
                let is_last = position == last;
                let mut data = chat.clone();
                data.inner.choices = vec![choice];
                if !is_last {
                    data.inner.usage = None;
                    data.nvext = None;
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

        if let Some(template) = &template {
            for choice in finish_unterminated_choices(&mut states, &mut finished) {
                yield response_with_choice(template, choice);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::metrics::LLMMetricAnnotation;
    use dynamo_protocols::types::{CreateChatCompletionStreamResponse, Role};
    use futures::stream;

    #[allow(deprecated)]
    fn chunk(text: &str, finish: bool) -> Annotated<NvCreateChatCompletionStreamResponse> {
        Annotated::from_data(NvCreateChatCompletionStreamResponse {
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
                model: "kimi-k3".to_string(),
                system_fingerprint: None,
                service_tier: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
            },
            nvext: None,
            llm_metrics: None,
        })
    }

    #[tokio::test]
    async fn emits_ordered_reasoning_text_and_tool_chunks() {
        let output = concat!(
            "<|open|>think<|sep|>reason<|close|>think<|sep|>",
            "<|open|>response<|sep|>answer<|close|>response<|sep|>",
            "<|open|>tools<|sep|>",
            "<|open|>call tool=\"lookup\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"q\" type=\"string\"<|sep|>x",
            "<|close|>argument<|sep|><|close|>call<|sep|>",
            "<|close|>tools<|sep|><|close|>message<|sep|>"
        );
        let responses = apply_stream(
            stream::iter([chunk(output, true)]),
            None,
            UnifiedParserPrefill::None,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = responses
            .iter()
            .flat_map(|response| response.data.as_ref().unwrap().inner.choices.iter())
            .collect::<Vec<_>>();

        assert_eq!(choices.len(), 3);
        assert_eq!(
            choices[0].delta.reasoning_content.as_deref(),
            Some("reason")
        );
        assert_eq!(
            choices[1].delta.content,
            Some(ChatCompletionMessageContent::Text("answer".to_string()))
        );
        let tool = choices[2]
            .delta
            .tool_calls
            .as_ref()
            .unwrap()
            .first()
            .unwrap();
        assert_eq!(tool.id.as_deref(), Some("lookup:0"));
        assert_eq!(
            tool.function.as_ref().unwrap().arguments.as_deref(),
            Some(r#"{"q":"x"}"#)
        );
        assert_eq!(choices[2].finish_reason, Some(FinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn reasoning_prefill_parses_leading_hidden_text() {
        let responses = apply_stream(
            stream::iter([chunk(
                "hidden<|close|>think<|sep|><|open|>response<|sep|>visible<|close|>response<|sep|>",
                true,
            )]),
            None,
            UnifiedParserPrefill::Reasoning,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = responses
            .iter()
            .flat_map(|response| response.data.as_ref().unwrap().inner.choices.iter())
            .collect::<Vec<_>>();

        assert_eq!(
            choices[0].delta.reasoning_content.as_deref(),
            Some("hidden")
        );
        assert_eq!(
            choices[1].delta.content,
            Some(ChatCompletionMessageContent::Text("visible".to_string()))
        );
        assert_eq!(choices[1].finish_reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn marker_only_chunk_preserves_metrics() {
        let mut marker = chunk("<|open|>", false);
        let data = marker.data.as_mut().unwrap();
        data.inner.choices[0].delta.role = None;
        data.llm_metrics = Some(LLMMetricAnnotation {
            input_tokens: 10,
            output_tokens: 1,
            chunk_tokens: 1,
            ..Default::default()
        });

        let responses = apply_stream(
            stream::iter([
                marker,
                chunk("response<|sep|>visible<|close|>response<|sep|>", true),
            ]),
            None,
            UnifiedParserPrefill::None,
        )
        .collect::<Vec<_>>()
        .await;

        assert_eq!(
            responses[0]
                .data
                .as_ref()
                .unwrap()
                .llm_metrics
                .as_ref()
                .unwrap()
                .chunk_tokens,
            1
        );
        assert!(
            responses[0].data.as_ref().unwrap().inner.choices[0]
                .delta
                .content
                .is_none()
        );
    }

    #[test]
    fn requires_matching_explicit_parser_names() {
        assert!(configured(Some("kimi_k3"), Some("kimi_k3")));
        assert!(!configured(Some("kimi_k3"), Some("kimi")));
        assert!(!configured(None, Some("kimi_k3")));
    }

    #[test]
    fn parses_complete_reasoning_prefill_with_tool_call() {
        let parsed = parse_complete(concat!(
            "hidden<|close|>think<|sep|>",
            "<|open|>response<|sep|>visible<|close|>response<|sep|>",
            "<|open|>tools<|sep|>",
            "<|open|>call tool=\"lookup\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"q\" type=\"string\"<|sep|>x",
            "<|close|>argument<|sep|><|close|>call<|sep|>",
            "<|close|>tools<|sep|><|close|>message<|sep|>"
        ))
        .unwrap();

        assert_eq!(parsed.reasoning, "hidden");
        assert_eq!(parsed.text, "visible");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].id, "lookup:0");
        assert_eq!(parsed.tool_calls[0].function.name, "lookup");
        assert_eq!(parsed.tool_calls[0].function.arguments, r#"{"q":"x"}"#);
    }

    #[test]
    fn detects_response_prefill_for_aggregate_output() {
        assert_eq!(
            detect_prefill(
                "visible<|close|>response<|sep|><|open|>tools<|sep|><|close|>tools<|sep|>"
            ),
            UnifiedParserPrefill::Response
        );
    }
}
