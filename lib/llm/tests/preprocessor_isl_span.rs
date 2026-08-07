// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A request rejected for exceeding the model's context length never reaches a
//! worker, so no metrics annotation ever comes back to report its input size.
//! The preprocessor stamps ISL on the enclosing request span as soon as the
//! token count is known — before the check that rejects — so the rejection is
//! still diagnosable. This exercises that on the real tokenize path rather than
//! asserting it from the shape of the code.

use std::sync::{Arc, Mutex};

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use tracing::Instrument;
use tracing_subscriber::prelude::*;

/// Captures `span.record(...)` calls for integer fields.
#[derive(Clone, Default)]
struct CaptureRecords(Arc<Mutex<Vec<(String, u64)>>>);

impl CaptureRecords {
    fn get(&self, field: &str) -> Option<u64> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .find(|(name, _)| name == field)
            .map(|(_, value)| *value)
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CaptureRecords {
    fn on_record(
        &self,
        _id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        struct Visitor<'a>(&'a mut Vec<(String, u64)>);
        impl tracing::field::Visit for Visitor<'_> {
            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                self.0.push((field.name().to_string(), value));
            }
            fn record_debug(
                &mut self,
                _field: &tracing::field::Field,
                _value: &dyn std::fmt::Debug,
            ) {
            }
        }
        values.record(&mut Visitor(&mut self.0.lock().unwrap()));
    }
}

fn chat_request(content: &str) -> NvCreateChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "mock-llama",
        "messages": [{ "role": "user", "content": content }],
    }))
    .expect("chat request fixture should deserialize")
}

/// `input_tokens` is declared by `make_inference_request_span`; mirror that here
/// so `record` has a field to write into, as it does in the frontend.
fn request_span() -> tracing::Span {
    tracing::info_span!("http-request", input_tokens = tracing::field::Empty)
}

/// The sample model's tokenizer is a stub: token counts are small and unrelated
/// to prompt length (the fixture yields a handful of tokens no matter how long
/// the text is). Lengthening the prompt therefore cannot force a rejection —
/// the budget has to be driven below the token count instead. The assertions
/// below stay tied to this constant rather than to any literal count so they
/// hold if the fixture's tokenizer changes.
const REJECTING_CONTEXT_LENGTH: u32 = 1;

#[tokio::test]
async fn context_length_rejection_still_records_isl() {
    let mut mdc = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .expect("sample model card should load");
    mdc.runtime_config.context_length = Some(REJECTING_CONTEXT_LENGTH);
    let preprocessor = OpenAIPreprocessor::new(mdc).expect("preprocessor should build");

    let recorded = CaptureRecords::default();
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(recorded.clone()));

    let request = chat_request("the quick brown fox jumps over the lazy dog");
    let result = preprocessor
        .preprocess_request(&request, None)
        .instrument(request_span())
        .await;

    assert!(
        result.is_err(),
        "a prompt past the context budget must be rejected"
    );

    // The point of the fix: the rejection happens inside preprocessing, so
    // nothing downstream ever reports this request's size. The count has to be
    // on the span already, stamped before the check that returned the error.
    let isl = recorded.get("input_tokens");
    assert!(
        isl.is_some_and(|n| n >= REJECTING_CONTEXT_LENGTH as u64),
        "ISL must be stamped before the context-length check rejects the request, got {isl:?}"
    );
}

#[tokio::test]
async fn accepted_request_leaves_isl_to_the_response_path() {
    // The preprocessor must NOT stamp ISL when the request survives. The
    // response path records the identical count on drop, and two writes to one
    // span field are not idempotent: the fmt field formatter appends rather
    // than replaces, so the log line would read `input_tokens=N input_tokens=N`.
    let mut mdc = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .expect("sample model card should load");
    mdc.runtime_config.context_length = Some(4096);
    let preprocessor = OpenAIPreprocessor::new(mdc).expect("preprocessor should build");

    let recorded = CaptureRecords::default();
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(recorded.clone()));

    let request = chat_request("hello");
    let (preprocessed, _annotations, _reasoning) = preprocessor
        .preprocess_request(&request, None)
        .instrument(request_span())
        .await
        .expect("a short prompt should preprocess cleanly");

    assert!(
        !preprocessed.token_ids.is_empty(),
        "fixture should tokenize to something, else this asserts nothing"
    );
    assert_eq!(
        recorded.get("input_tokens"),
        None,
        "accepted requests must be stamped once, by the response path only"
    );
}
