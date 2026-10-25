// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::common::extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId};

use super::*;

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

/// #16587: one token per content delta, across events Envoy split mid-line;
/// role-only and `[DONE]` events carry no output.
#[test]
fn sse_progress_counts_content_deltas_across_split_chunks() {
    let mut response = ResponseProgress::default();
    let events = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"lo\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}],",
        "\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
        "data: [DONE]\n\n",
    )
    .as_bytes();
    let split = 70;

    let first = response.ingest_sse(&events[..split], false);
    let rest = response.ingest_sse(&events[split..], true);
    let progress = [first, rest].concat();

    let tokens: usize = progress
        .iter()
        .map(|progress| match progress {
            GatewayProgress::Tokens(count) => *count,
            GatewayProgress::Finish { .. } => 0,
        })
        .sum();
    assert_eq!(tokens, 2);
    match progress.last() {
        Some(GatewayProgress::Finish { reason, usage }) => {
            assert_eq!(reason, &Some(FinishReason::Length));
            assert_eq!(usage.as_ref().map(|usage| usage.completion_tokens), Some(2));
        }
        other => panic!("the stream must end with its finish, got {other:?}"),
    }
    assert!(response.finished);
}

#[test]
fn unary_progress_reports_completion_tokens() {
    let mut response = ResponseProgress::default();
    let body = br#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":9,"total_tokens":13}}"#;

    let progress = response.ingest_unary(body);

    assert_eq!(progress.first(), Some(&GatewayProgress::Tokens(9)));
    assert!(matches!(
        progress.last(),
        Some(GatewayProgress::Finish {
            reason: Some(FinishReason::Stop),
            ..
        })
    ));
}

#[test]
fn an_unfinished_response_is_not_marked_finished() {
    let mut response = ResponseProgress::default();
    response.ingest_sse(
        b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        false,
    );
    assert!(!response.finished);
}
