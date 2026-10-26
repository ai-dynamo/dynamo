// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::common::extensions::{
    InputTrigger, SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId,
};

use super::*;
use crate::picker::ResponseEvent;

fn preprocessor() -> Arc<OpenAIPreprocessor> {
    let card = ModelDeploymentCard::load_from_disk(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../lib/llm/tests/data/sample-models/TinyLlama_v1.1"
        ),
        None,
    )
    .expect("load fixture model card");
    OpenAIPreprocessor::new(card).expect("build preprocessor from fixture card")
}

fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect()
}

const CHAT: &str = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;

/// #16576 plus session affinity: the request reaching the routing host carries
/// the frontend's reading of the Envoy headers, not an EPP re-parse.
#[tokio::test]
async fn prepare_applies_the_frontend_header_policy() {
    let request = GatewayRouting::prepare(
        &preprocessor(),
        CHAT,
        &headers(&[
            ("x-dynamo-request-priority", "5"),
            ("x-dynamo-request-strict-priority", "2"),
            ("x-dynamo-session-id", "session-a"),
        ]),
        "req-1",
        Some(HashSet::from([7])),
    )
    .await
    .unwrap()
    .expect("a chat request takes the routing-host path");

    let context = &request.context;
    assert_eq!(context.id(), "req-1");
    let routing = context.content().routing.as_ref().unwrap();
    assert_eq!(routing.priority_jump, Some(5.0));
    assert_eq!(routing.strict_priority, Some(2));
    assert_eq!(routing.allowed_worker_ids, Some(HashSet::from([7])));
    assert!(!context.content().token_ids.is_empty());
    let session = context
        .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
        .expect("session affinity rides the request context, as in the frontend");
    assert_eq!(session.as_str(), "session-a");
}

/// Oversized metadata headers get the frontend's 431, not a generic 400.
#[tokio::test]
async fn prepare_rejects_oversized_metadata_with_431() {
    let metadata: Vec<(String, String)> = (0..65)
        .map(|i| (format!("x-dynamo-meta-key{i}"), "v".to_string()))
        .collect();

    let error = match GatewayRouting::prepare(&preprocessor(), CHAT, &metadata, "req-1", None).await
    {
        Err(error) => error,
        Ok(_) => panic!("65 metadata entries exceed the limit of 64"),
    };
    assert!(
        matches!(error, PickError::Rejected { status: 431, .. }),
        "{error:?}"
    );
}

/// An agent request continuing after a tool call carries the frontend's
/// tool-result classification, which trigger-aware policies select on.
#[tokio::test]
async fn prepare_classifies_a_tool_result_turn() {
    let tool_turn = r#"{"model":"m","messages":[
        {"role":"user","content":"weather?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call-1","type":"function","function":{"name":"weather","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"call-1","content":"sunny"}
    ]}"#;

    let request = GatewayRouting::prepare(
        &preprocessor(),
        tool_turn,
        &headers(&[("x-dynamo-session-id", "session-a")]),
        "req-1",
        None,
    )
    .await
    .unwrap()
    .expect("a chat request takes the routing-host path");

    let agent_context = request.context.content().agent_context.as_ref().unwrap();
    assert_eq!(agent_context.input_trigger, Some(InputTrigger::ToolResult));
}

#[tokio::test]
async fn prepare_leaves_multi_prompt_batches_to_the_existing_path() {
    let batch = r#"{"model":"m","prompt":["one","two"]}"#;
    let single = r#"{"model":"m","prompt":"one"}"#;
    let preprocessor = preprocessor();

    assert!(
        GatewayRouting::prepare(&preprocessor, batch, &[], "req-1", None)
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        GatewayRouting::prepare(&preprocessor, single, &[], "req-2", None)
            .await
            .unwrap()
            .is_some()
    );
}

fn sse(payload: &str) -> ResponseEvent {
    ResponseEvent::from_sse_data(payload.as_bytes()).expect("the event reports something")
}

fn finish_of(progress: &[GatewayProgress]) -> (Option<FinishReason>, Option<u32>) {
    match progress.last() {
        Some(GatewayProgress::Finish { reason, usage }) => (
            reason.clone(),
            usage.as_ref().map(|usage| usage.total_tokens),
        ),
        other => panic!("a finished response ends with its finish, got {other:?}"),
    }
}

/// #16587: streamed output is reported per event, and output the events did
/// not show is topped up from `usage` when the response finishes.
#[test]
fn streamed_output_is_reported_per_event_and_topped_up_from_usage() {
    let mut response = ResponseProgress::default();
    assert_eq!(
        response.observe(&sse(r#"{"choices":[{"delta":{"content":"a"}}]}"#)),
        1
    );
    assert_eq!(
        response.observe(&sse(r#"{"choices":[{"delta":{"content":"b"}}]}"#)),
        1
    );
    response.observe(&sse(
        r#"{"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":4,"completion_tokens":5,"total_tokens":9}}"#,
    ));

    let progress = response.finish();
    assert_eq!(progress.first(), Some(&GatewayProgress::Tokens(3)));
    assert_eq!(finish_of(&progress), (Some(FinishReason::Length), Some(9)));
    assert_eq!(response.outcome(), Ok(()));
}

/// A buffered body shows its output only through `usage`.
#[test]
fn a_buffered_response_reports_its_usage_as_output() {
    let mut response = ResponseProgress::default();
    let body = br#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":9,"total_tokens":13}}"#;
    assert_eq!(response.observe(&ResponseEvent::from_body(body)), 0);

    let progress = response.finish();
    assert_eq!(progress.first(), Some(&GatewayProgress::Tokens(9)));
    assert_eq!(finish_of(&progress), (Some(FinishReason::Stop), Some(13)));
}

/// A failed response (an error event here; a non-2xx status sets the same flag)
/// ends with an abort, not a completion, and reports no finish.
#[test]
fn a_failed_response_is_aborted_not_completed() {
    let mut response = ResponseProgress::default();
    response.observe(&sse(r#"{"error":{"message":"engine died","code":500}}"#));
    assert!(response.finish().is_empty());
    assert!(response.outcome().is_err());

    let mut early = ResponseProgress::default();
    early.observe(&sse(r#"{"choices":[{"delta":{"content":"a"}}]}"#));
    assert!(
        early.outcome().is_err(),
        "a stream that closes unfinished is aborted"
    );
}
