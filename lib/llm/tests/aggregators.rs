// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::{
    Annotated, ContentProvider, DataStream,
    codec::{Message, SseCodecError, create_message_stream},
    openai::{
        GuidedToolConstraint, ParsingOptions,
        chat_completions::{
            NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse,
            aggregator::ChatCompletionAggregator,
        },
        completions::NvCreateCompletionResponse,
    },
};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, Role,
};
use dynamo_runtime::config::{env_is_truthy, environment_names::llm as env_llm};
use futures::StreamExt;

fn get_text(content: &ChatCompletionMessageContent) -> &str {
    match content {
        ChatCompletionMessageContent::Text(text) => text.as_str(),
        ChatCompletionMessageContent::Parts(_) => "",
    }
}

const CMPL_ROOT_PATH: &str = "tests/data/replays/meta/llama-3.1-8b-instruct/completions";
const CHAT_ROOT_PATH: &str = "tests/data/replays/meta/llama-3.1-8b-instruct/chat_completions";

fn create_stream(root_path: &str, file_name: &str) -> DataStream<Result<Message, SseCodecError>> {
    let data = std::fs::read_to_string(format!("{}/{}", root_path, file_name)).unwrap();
    create_message_stream(&data)
}

#[tokio::test]
async fn test_openai_chat_stream() {
    let data = std::fs::read_to_string("tests/data/replays/meta/llama-3.1-8b-instruct/chat_completions/chat-completion.streaming.1").unwrap();

    // note: we are only taking the first 16 messages to keep the size of the response small
    let stream = create_message_stream(&data).take(16);
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    // todo: provide a cleaner way to extract the content from choices
    assert_eq!(
        get_text(
            result
                .inner
                .choices
                .first()
                .unwrap()
                .message
                .content
                .as_ref()
                .expect("there to be content")
        ),
        "Deep learning is a subfield of machine learning that involves the use of artificial"
    );
}

#[tokio::test]
async fn test_openai_chat_edge_case_multi_line_data() {
    let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/valid-multi-line-data");
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        get_text(
            result
                .inner
                .choices
                .first()
                .unwrap()
                .message
                .content
                .as_ref()
                .expect("there to be content")
        ),
        "Deep learning"
    );
}

#[tokio::test]
async fn test_openai_chat_edge_case_comments_per_response() {
    let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/valid-comments_per_response");
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        get_text(
            result
                .inner
                .choices
                .first()
                .unwrap()
                .message
                .content
                .as_ref()
                .expect("there to be content")
        ),
        "Deep learning"
    );
}

#[tokio::test]
async fn test_openai_chat_edge_case_invalid_deserialize_error() {
    let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/invalid-deserialize_error");
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await;

    assert!(result.is_err());
    // insta::assert_debug_snapshot!(result);
}

// =============================
// Completions (/v1/completions)
// =============================

#[tokio::test]
async fn test_openai_cmpl_stream() {
    let stream = create_stream(CMPL_ROOT_PATH, "completion.streaming.1").take(16);
    let result =
        NvCreateCompletionResponse::from_sse_stream(Box::pin(stream), ParsingOptions::default())
            .await
            .unwrap();

    // todo: provide a cleaner way to extract the content from choices
    assert_eq!(
        result.inner.choices.first().unwrap().content(),
        " This is a question that is often asked by those outside of AI research and development"
    );
}

// ===================================
// nvext aggregation regression tests
// ===================================

#[allow(deprecated)]
fn make_stream_delta(
    content: Option<&str>,
    nvext: Option<serde_json::Value>,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    Annotated::from_data(NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices: if let Some(text) = content {
                vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        content: Some(ChatCompletionMessageContent::Text(text.to_string())),
                        function_call: None,
                        tool_calls: None,
                        role: Some(Role::Assistant),
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }]
            } else {
                vec![]
            },
            created: 1234567890,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        },
        nvext,
        llm_metrics: None,
    })
}

/// Verify that nvext set on a stream delta survives aggregation into the final response.
#[tokio::test]
async fn test_nvext_passthrough_aggregation() {
    let nvext_value = serde_json::json!({"custom_field": "test_value"});

    let deltas = vec![
        make_stream_delta(Some("Hello"), None),
        make_stream_delta(Some(" world"), Some(nvext_value.clone())),
        make_stream_delta(Some("!"), None),
    ];

    let stream = futures::stream::iter(deltas);
    let result =
        NvCreateChatCompletionResponse::from_annotated_stream(stream, ParsingOptions::default())
            .await
            .unwrap();

    assert_eq!(result.nvext, Some(nvext_value));
    assert_eq!(
        get_text(
            result
                .inner
                .choices
                .first()
                .unwrap()
                .message
                .content
                .as_ref()
                .unwrap()
        ),
        "Hello world!"
    );
}

/// Verify that the last non-None nvext wins when multiple deltas carry nvext.
#[tokio::test]
async fn test_nvext_last_value_wins() {
    let first_nvext = serde_json::json!({"version": 1});
    let last_nvext = serde_json::json!({"version": 2});

    let deltas = vec![
        make_stream_delta(Some("a"), Some(first_nvext)),
        make_stream_delta(Some("b"), None),
        make_stream_delta(Some("c"), Some(last_nvext.clone())),
    ];

    let stream = futures::stream::iter(deltas);
    let result =
        NvCreateChatCompletionResponse::from_annotated_stream(stream, ParsingOptions::default())
            .await
            .unwrap();

    assert_eq!(result.nvext, Some(last_nvext));
}

/// Verify that nvext remains None when no delta carries it.
#[tokio::test]
async fn test_nvext_none_when_absent() {
    let deltas = vec![make_stream_delta(Some("hello"), None)];

    let stream = futures::stream::iter(deltas);
    let result =
        NvCreateChatCompletionResponse::from_annotated_stream(stream, ParsingOptions::default())
            .await
            .unwrap();

    assert_eq!(result.nvext, None);
}

// ===================================
// Muse unified batch finalize (topology B)
// ===================================

async fn run_qwen_unified_batch_suppression_assertions() {
    let raw = concat!(
        "<think>Look it up.</think>",
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=location>Paris</parameter>\n</function>\n</tool_call>"
    );
    let options = ParsingOptions::new(Some("qwen3_coder".to_string()), Some("qwen3".to_string()))
        .with_tool_call_parsing_enabled(false);
    assert!(
        options.tool_call_parser.is_some(),
        "with the flag set, the qwen3 pair must retain the whole-response decoder \
         so it can still strip native markup even though tool calls are suppressed"
    );

    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        futures::stream::iter([make_stream_delta(Some(raw), None)]),
        options,
    )
    .await
    .unwrap();
    let choice = result.inner.choices.first().expect("one choice");
    let content = choice
        .message
        .content
        .as_ref()
        .map(get_text)
        .unwrap_or_default();

    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Look it up.")
    );
    assert!(choice.message.tool_calls.as_ref().is_none_or(Vec::is_empty));
    assert!(
        !content.contains("<tool_call>"),
        "raw markup leaked: {content:?}"
    );
}

#[tokio::test]
#[ignore = "only run as a child process spawned by \
            test_qwen_unified_batch_suppresses_forbidden_calls_and_strips_markup, \
            with the experimental flag set in that child's own environment; running \
            it directly outside that harness gives no guarantee the flag is set"]
async fn qwen_unified_batch_suppression_child() {
    assert!(
        env_is_truthy(env_llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2),
        "this child test must only ever run with the flag set by its parent"
    );
    run_qwen_unified_batch_suppression_assertions().await;
}

#[test]
fn test_qwen_unified_batch_suppresses_forbidden_calls_and_strips_markup() {
    // The real suppression gate lives inside `ChatCompletionAggregator::apply`'s
    // function body (`aggregator.rs`'s `qwen3_unified_family` block), re-checking
    // the process-wide flag itself rather than taking it as a parameter — so no
    // flag-independent call from this process can route through it. Mutating this
    // (the shared, multi-threaded) test process's own environment would leak into
    // every other test in this binary regardless of execution order, which is
    // forbidden. Instead, this parent test — itself flag-independent, so it always
    // actually runs and asserts something — re-executes this exact compiled test
    // binary, filtered to ONLY the `#[ignore]`d child test above, with the flag set
    // solely in that CHILD PROCESS's own environment. The child is a distinct,
    // `#[ignore]`d function name, not this same function, so there is no recursive
    // self-spawn: `--exact <child> --ignored` can only ever select that one child.
    let exe = std::env::current_exe().expect("test binary path for self re-exec");
    let status = std::process::Command::new(exe)
        .args([
            "--exact",
            "qwen_unified_batch_suppression_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(env_llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2, "1")
        .status()
        .expect("failed to spawn child test process");
    assert!(
        status.success(),
        "child aggregator suppression test failed (see its own output above): {status:?}"
    );
}

async fn run_qwen_unified_batch_flag_off_assertions() {
    let raw = concat!(
        "<think>Look it up.</think>",
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=location>Paris</parameter>\n</function>\n</tool_call>"
    );
    let options = ParsingOptions::new(Some("qwen3_coder".to_string()), Some("qwen3".to_string()))
        .with_tool_call_parsing_enabled(false);
    assert_eq!(
        options.tool_call_parser, None,
        "with the flag unset, a qwen3 pair must not retain a whole-response decoder \
         when tool-call parsing is disabled"
    );

    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        futures::stream::iter([make_stream_delta(Some(raw), None)]),
        options,
    )
    .await
    .unwrap();
    let choice = result.inner.choices.first().expect("one choice");
    let content = choice
        .message
        .content
        .as_ref()
        .map(get_text)
        .unwrap_or_default();

    assert_eq!(
        content, raw,
        "with no decoder retained (flag off), the raw markup must pass through \
         unparsed rather than be silently stripped by a decoder that shouldn't be \
         running"
    );
    assert_eq!(choice.message.reasoning_content, None);
    assert!(choice.message.tool_calls.as_ref().is_none_or(Vec::is_empty));
}

#[tokio::test]
#[ignore = "only run as a child process spawned by \
            test_qwen_unified_batch_stays_off_without_the_experimental_flag, with the \
            experimental flag explicitly cleared in that child's own environment; \
            running it directly outside that harness gives no guarantee the flag is \
            actually unset"]
async fn qwen_unified_batch_flag_off_child() {
    assert!(
        !env_is_truthy(env_llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2),
        "this child test must only ever run with the flag explicitly cleared by its parent"
    );
    run_qwen_unified_batch_flag_off_assertions().await;
}

#[test]
fn test_qwen_unified_batch_stays_off_without_the_experimental_flag() {
    // Mirrors the suppression parent/child design above: re-executes this exact
    // compiled test binary, filtered to ONLY the `#[ignore]`d child test, with the
    // flag explicitly CLEARED in that child process's own environment
    // (`env_remove`) rather than relying on the ambient absence of the flag. This
    // parent is flag-independent and always spawns and requires the real-assertion
    // child, so it passes correctly whether the surrounding test invocation (or
    // ladder lane) happens to run with the flag externally set or unset — unlike a
    // single-process test whose premise assertion would fail under a flag-on lane.
    let exe = std::env::current_exe().expect("test binary path for self re-exec");
    let status = std::process::Command::new(exe)
        .args([
            "--exact",
            "qwen_unified_batch_flag_off_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env_remove(env_llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2)
        .status()
        .expect("failed to spawn child test process");
    assert!(
        status.success(),
        "child aggregator flag-off test failed (see its own output above): {status:?}"
    );
}

fn assert_guided_batch_call(result: &NvCreateChatCompletionResponse) {
    let choice = result.inner.choices.first().expect("one choice");
    let calls = choice.message.tool_calls.as_ref().expect("tool_calls");
    assert_eq!(calls.len(), 1, "expected one guided tool call");
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
        serde_json::json!({"location": "Paris"})
    );
    assert!(
        choice
            .message
            .content
            .as_ref()
            .is_none_or(|content| get_text(content).is_empty()),
        "guided payload must not be returned as visible content: {:?}",
        choice.message.content
    );
}

fn muse_raw_batch_deltas() -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
    vec![
        make_stream_delta(
            Some("<|start|>assistant to=self<|message|>Look it up.<|eom|>"),
            None,
        ),
        make_stream_delta(
            Some(
                "<|start|>assistant to=get_weather<|message|><atem:invoke name=\"get_weather\"><atem:parameter name=\"location\">Paris</atem:parameter></atem:invoke><|eom|>",
            ),
            None,
        ),
        make_stream_delta(
            Some("<|start|>assistant to=user<|message|>It's 18C.<|eot|>"),
            None,
        ),
    ]
}

#[tokio::test]
async fn test_qwen_unified_batch_finalizes_raw_guided_json() {
    for (raw, constraint) in [
        (
            r#"[{"name":"get_weather","parameters":{"location":"Paris"}}]"#,
            GuidedToolConstraint::GuidedJsonRequired,
        ),
        (
            r#"{"location":"Paris"}"#,
            GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            },
        ),
    ] {
        let result = NvCreateChatCompletionResponse::from_annotated_stream(
            futures::stream::iter([make_stream_delta(Some(raw), None)]),
            ParsingOptions::new(Some("qwen3_coder".to_string()), Some("qwen3".to_string()))
                .with_guided_tool_constraint(constraint),
        )
        .await
        .unwrap();

        assert_guided_batch_call(&result);
    }
}

#[tokio::test]
async fn test_qwen_unified_batch_recovers_native_structural_tag_output() {
    if !env_is_truthy(env_llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2) {
        return;
    }

    let raw = concat!(
        "<think>Look it up.</think>",
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=location>Paris</parameter>\n</function>\n</tool_call>"
    );
    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        futures::stream::iter([make_stream_delta(Some(raw), None)]),
        ParsingOptions::new(Some("qwen3_coder".to_string()), Some("qwen3".to_string()))
            // Topology B does not carry the worker's installed structural tag back to
            // the frontend, so request-only reconstruction may classify this as JSON.
            .with_guided_tool_constraint(GuidedToolConstraint::GuidedJsonRequired),
    )
    .await
    .unwrap();

    let choice = result.inner.choices.first().expect("one choice");
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Look it up.")
    );
    assert_guided_batch_call(&result);
}

#[tokio::test]
async fn test_forced_batch_recovers_observed_native_tool_markup() {
    let native_cases = [
        (
            "minimax_m2",
            "<minimax:tool_call><invoke name=\"get_weather\"><parameter name=\"location\">Paris</parameter></invoke></minimax:tool_call>",
        ),
        (
            "kimi_k2",
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"Paris\"}<|tool_call_end|><|tool_calls_section_end|>",
        ),
        (
            "deepseek_v4",
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        ),
    ];

    for (parser, raw) in native_cases {
        for constraint in [
            GuidedToolConstraint::GuidedJsonRequired,
            GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            },
        ] {
            let result = NvCreateChatCompletionResponse::from_annotated_stream(
                futures::stream::iter([make_stream_delta(Some(raw), None)]),
                ParsingOptions::new(Some(parser.to_string()), None)
                    .with_guided_tool_constraint(constraint.clone()),
            )
            .await
            .unwrap();
            let choice = result.inner.choices.first().expect("one choice");
            let calls = choice.message.tool_calls.as_ref().unwrap_or_else(|| {
                panic!("{parser} {constraint:?} did not recover its native tool call")
            });

            assert_eq!(calls.len(), 1, "{parser} {constraint:?}");
            assert_eq!(calls[0].function.name, "get_weather");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
                serde_json::json!({"location": "Paris"}),
                "{parser} {constraint:?}",
            );
            assert!(
                choice
                    .message
                    .content
                    .as_ref()
                    .is_none_or(|content| get_text(content).is_empty()),
                "{parser} {constraint:?} leaked native markup as content: {:?}",
                choice.message.content,
            );
        }
    }
}

#[tokio::test]
async fn test_muse_unified_batch_finalizes_raw_guided_json() {
    for (raw, constraint) in [
        (
            r#"[{"name":"get_weather","parameters":{"location":"Paris"}}]"#,
            GuidedToolConstraint::GuidedJsonRequired,
        ),
        (
            r#"{"location":"Paris"}"#,
            GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            },
        ),
    ] {
        let result = NvCreateChatCompletionResponse::from_annotated_stream(
            futures::stream::iter([make_stream_delta(Some(raw), None)]),
            ParsingOptions::new(Some("muse_glimmer".to_string()), None)
                .with_guided_tool_constraint(constraint),
        )
        .await
        .unwrap();

        assert_guided_batch_call(&result);
    }
}

#[tokio::test]
async fn test_forced_muse_batch_recovers_observed_native_markup() {
    for constraint in [
        GuidedToolConstraint::GuidedJsonRequired,
        GuidedToolConstraint::GuidedJsonNamed {
            tool_name: "get_weather".to_string(),
        },
    ] {
        let result = NvCreateChatCompletionResponse::from_annotated_stream(
            futures::stream::iter(muse_raw_batch_deltas()),
            ParsingOptions::new(Some("muse_glimmer".to_string()), None)
                .with_guided_tool_constraint(constraint.clone()),
        )
        .await
        .unwrap();
        let choice = result.inner.choices.first().expect("one choice");

        assert_eq!(
            choice.message.reasoning_content.as_deref(),
            Some("Look it up.")
        );
        assert_eq!(
            get_text(choice.message.content.as_ref().expect("content")),
            "It's 18C."
        );
        let calls =
            choice.message.tool_calls.as_ref().unwrap_or_else(|| {
                panic!("Muse {constraint:?} did not recover its native tool call")
            });
        assert_eq!(calls.len(), 1, "{constraint:?}");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({"location": "Paris"}),
            "{constraint:?}",
        );
    }
}

/// Topology B: raw muse markup reaches the aggregator un-split (the frontend
/// holds the tool_call_parser, but the worker did not run the streaming hook, so
/// the deltas carry raw text). The finalize block must route muse through the
/// UNIFIED parser and populate reasoning_content + content + tool_calls on the
/// final message — the reasoning channel is the one thing the v2 tool-only
/// `parse_complete` cannot recover.
#[tokio::test]
async fn test_muse_unified_batch_finalize_splits_all_channels() {
    let stream = futures::stream::iter(muse_raw_batch_deltas());
    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        stream,
        ParsingOptions::new(Some("muse_glimmer".to_string()), None),
    )
    .await
    .unwrap();

    let choice = result.inner.choices.first().expect("one choice");
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Look it up."),
        "reasoning_content must come from the unified batch finalize"
    );
    assert_eq!(
        get_text(choice.message.content.as_ref().expect("content")),
        "It's 18C."
    );
    let tool_calls = choice.message.tool_calls.as_ref().expect("tool_calls");
    assert_eq!(tool_calls.len(), 1, "expected one tool call");
    assert_eq!(tool_calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
    assert_eq!(args, serde_json::json!({"location": "Paris"}));
}

/// Reasoning-only card (topology B): `reasoning_parser=muse_glimmer`, NO
/// `tool_call_parser` — the card shape the streaming guard, `unified_family`, and
/// `parser_requires_special_tokens` all support on either name. The batch finalize
/// must key on the reasoning name too, so raw channel markup never survives into the
/// non-streaming response.
#[tokio::test]
async fn test_muse_unified_batch_finalize_routes_on_reasoning_name_only() {
    let stream = futures::stream::iter(muse_raw_batch_deltas());
    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        stream,
        ParsingOptions::new(None, Some("muse_glimmer".to_string())),
    )
    .await
    .unwrap();

    let choice = result.inner.choices.first().expect("one choice");
    let content = get_text(choice.message.content.as_ref().expect("content"));
    assert!(
        !content.contains("<|start|>") && !content.contains("<atem:invoke"),
        "raw markup leaked into content: {content:?}"
    );
    assert_eq!(content, "It's 18C.");
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Look it up."),
        "reasoning_content must be split even when only the reasoning name is set"
    );
    let tool_calls = choice.message.tool_calls.as_ref().expect("tool_calls");
    assert_eq!(tool_calls.len(), 1, "expected one tool call");
    assert_eq!(tool_calls[0].function.name, "get_weather");
}

/// Batch counterpart of the streaming `none` pin. A caller that disabled tool calling
/// still needs the reasoning/content split and the marker stripping — muse has no v1
/// reasoning parser to fall back on — but must not receive `tool_calls`. The handlers
/// express that with the shared tool-call response policy.
#[tokio::test]
async fn test_muse_unified_batch_finalize_suppresses_calls_when_disabled() {
    let stream = futures::stream::iter(muse_raw_batch_deltas());
    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        stream,
        ParsingOptions::new(Some("muse_glimmer".to_string()), None)
            .with_tool_call_parsing_enabled(false),
    )
    .await
    .unwrap();

    let choice = result.inner.choices.first().expect("one choice");
    assert!(
        choice
            .message
            .tool_calls
            .as_ref()
            .is_none_or(|calls| calls.is_empty()),
        "a request that disabled tools must not return tool_calls: {:?}",
        choice.message.tool_calls
    );
    // The split and the stripping still have to happen.
    let content = get_text(choice.message.content.as_ref().expect("content"));
    assert_eq!(content, "It's 18C.");
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Look it up.")
    );
    for marker in ["<|start|>", "<|message|>", "<atem:invoke"] {
        assert!(
            !content.contains(marker),
            "marker {marker:?} leaked: {content:?}"
        );
    }
}

// ===================================
// DIS-2544 stage5b regression tests (Fix A/B/C)
// ===================================

/// Fix A: a guided-JSON response that legitimately contains zero tool calls
/// (`[]` under `tool_choice: "required"`) must not wipe the batch's original raw
/// text. Before the fix, the muse unified block's guided-JSON success arm
/// (`Ok((calls, String::new(), String::new()))`) fed into an unconditional
/// `choice.text = content`, which always blanked `choice.text` to empty
/// regardless of whether the guided parse actually produced any calls.
#[tokio::test]
async fn test_muse_unified_batch_guided_json_zero_calls_preserves_original_text() {
    let raw = "[]";
    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        futures::stream::iter([make_stream_delta(Some(raw), None)]),
        ParsingOptions::new(Some("muse_glimmer".to_string()), None)
            .with_guided_tool_constraint(GuidedToolConstraint::GuidedJsonRequired),
    )
    .await
    .unwrap();

    let choice = result.inner.choices.first().expect("one choice");
    assert!(
        choice.message.tool_calls.as_ref().is_none_or(Vec::is_empty),
        "guided JSON with zero calls must not produce any tool_calls: {:?}",
        choice.message.tool_calls
    );
    assert_eq!(
        get_text(choice.message.content.as_ref().expect("content")),
        raw,
        "a guided-JSON parse that returns zero calls must preserve the original raw \
         text instead of blanking choice.text to empty"
    );
}

/// Build a stream delta that carries ONLY `reasoning_content`, no visible text —
/// used to pre-populate `choice.reasoning_content` via the normal per-delta
/// aggregation path before the muse unified finalize block runs.
#[allow(deprecated)]
fn make_stream_delta_with_reasoning(
    reasoning_content: &str,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    Annotated::from_data(NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    content: None,
                    function_call: None,
                    tool_calls: None,
                    role: Some(Role::Assistant),
                    refusal: None,
                    reasoning_content: Some(reasoning_content.to_string()),
                },
                finish_reason: None,
                logprobs: None,
            }],
            created: 1234567890,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        },
        nvext: None,
        llm_metrics: None,
    })
}

/// Fix B: the muse unified finalize's reasoning merge must APPEND newly recovered
/// reasoning onto any `reasoning_content` the normal per-delta aggregation already
/// populated, not overwrite-only-if-`None`. Before the fix,
/// `if choice.reasoning_content.is_none() && !reasoning.is_empty()` meant any
/// pre-existing `reasoning_content` silently discarded whatever the unified parse
/// step went on to recover.
#[tokio::test]
async fn test_muse_unified_batch_finalize_appends_to_existing_reasoning_content() {
    let mut deltas = vec![make_stream_delta_with_reasoning("Pre-populated. ")];
    deltas.extend(muse_raw_batch_deltas());
    let stream = futures::stream::iter(deltas);
    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        stream,
        ParsingOptions::new(Some("muse_glimmer".to_string()), None),
    )
    .await
    .unwrap();

    let choice = result.inner.choices.first().expect("one choice");
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Pre-populated. Look it up."),
        "the unified finalize must APPEND its recovered reasoning onto any \
         reasoning_content already aggregated from prior deltas, not overwrite it \
         only when None"
    );
}

/// Fix C: a COMPLETE `<tool_call>...</tool_call>` span that only appears inside a
/// quoted parameter value must not be misclassified as real native tool-call
/// markup by the guided-JSON failure fallback. Before the fix,
/// `parse_complete_tool_output` used
/// `dynamo_parsers::tool_calling::detect_tool_call_start`, which does a naive
/// substring search: it would false-positive on `<tool_call>` appearing only as
/// quoted illustrative text, misrouting prose that never contained a real
/// tool-call request into the native hermes parser, which would then actually
/// extract and execute the quoted example as a bogus tool call.
#[tokio::test]
async fn test_hermes_batch_guided_json_failure_ignores_quoted_marker_substring() {
    let raw = "Please literally output the example call \
               '<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}</tool_call>' \
               verbatim and nothing else.";
    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        futures::stream::iter([make_stream_delta(Some(raw), None)]),
        ParsingOptions::new(Some("hermes".to_string()), None)
            .with_guided_tool_constraint(GuidedToolConstraint::GuidedJsonRequired),
    )
    .await
    .unwrap();

    let choice = result.inner.choices.first().expect("one choice");
    assert!(
        choice.message.tool_calls.as_ref().is_none_or(Vec::is_empty),
        "a complete <tool_call>...</tool_call> span that only appears inside a \
         quoted illustrative example must not be extracted as a real tool call: {:?}",
        choice.message.tool_calls
    );
    assert_eq!(
        get_text(choice.message.content.as_ref().expect("content")),
        raw,
        "content whose only marker-looking span is quoted illustrative prose must \
         pass through unparsed rather than be misclassified as native tool-call markup"
    );
}
