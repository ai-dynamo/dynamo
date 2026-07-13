// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use base64::Engine as _;
use futures::{StreamExt, stream};
use ndarray::{Array, IxDyn};
use ndarray_npy::WriteNpyExt;
use serde_json::json;

use dynamo_runtime::pipeline::{ResponseStream, context::Controller};

use super::*;
use crate::protocols::inference::generate::{GenerateBackendMetadata, GenerateLogprob};

fn prefill_stream(items: Vec<Annotated<LLMEngineOutput>>) -> ManyOut<Annotated<LLMEngineOutput>> {
    ResponseStream::new(
        Box::pin(stream::iter(items)),
        Arc::new(Controller::default()),
    )
}

fn valid_prefill_output() -> Annotated<LLMEngineOutput> {
    Annotated::from_data(LLMEngineOutput {
        disaggregated_params: Some(json!({})),
        ..Default::default()
    })
}

fn routed_experts_payload(value: i16) -> String {
    let array = Array::from_shape_vec(IxDyn(&[1, 1]), vec![value]).unwrap();
    let mut bytes = Vec::new();
    array.write_npy(&mut bytes).unwrap();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[tokio::test]
async fn first_output_error_does_not_record_prefill_complete() {
    let tracker = Arc::new(RequestTracker::new());
    let result = PrefillRouter::consume_prefill_stream(
        prefill_stream(vec![Annotated::from_error("prefill failed")]),
        Some(tracker.clone()),
        None,
    )
    .await;

    assert!(result.is_err());
    assert!(tracker.record_prefill_complete());
}

#[tokio::test]
async fn later_output_error_is_propagated_after_prefill_arrival() {
    let tracker = Arc::new(RequestTracker::new());
    let result = PrefillRouter::consume_prefill_stream(
        prefill_stream(vec![
            valid_prefill_output(),
            Annotated::from_error("prefill stream failed"),
        ]),
        Some(tracker.clone()),
        None,
    )
    .await;

    assert!(result.is_err());
    assert!(!tracker.record_prefill_complete());
}

#[tokio::test]
async fn prefill_rejects_mismatched_request_level_handoff_frames() {
    let first = valid_prefill_output();
    let mut second = valid_prefill_output();
    second.data.as_mut().unwrap().disaggregated_params = Some(json!({"different": true}));

    let error =
        PrefillRouter::consume_prefill_stream(prefill_stream(vec![first, second]), None, Some(1))
            .await
            .err()
            .expect("request-level handoff mismatch must fail");

    assert!(error.to_string().contains("disaggregated_params"));
}

#[tokio::test]
async fn prefill_generate_metadata_is_typed_and_choice_aligned() {
    let prompt_logprobs = vec![Some(HashMap::from([(
        11,
        GenerateLogprob {
            logprob: -0.25,
            rank: Some(1),
            decoded_token: Some("a".to_string()),
        },
    )]))];
    let mut first = valid_prefill_output();
    first.data.as_mut().unwrap().index = Some(0);
    first.data.as_mut().unwrap().generate_metadata = Some(GenerateBackendMetadata {
        prompt_logprobs: Some(prompt_logprobs.clone()),
        routed_experts: Some(routed_experts_payload(10)),
        prefill_routed_experts: None,
        kv_transfer_params: None,
    });
    let mut second = valid_prefill_output();
    second.data.as_mut().unwrap().index = Some(1);
    second.data.as_mut().unwrap().generate_metadata = Some(GenerateBackendMetadata {
        prompt_logprobs: Some(prompt_logprobs.clone()),
        routed_experts: Some(routed_experts_payload(20)),
        prefill_routed_experts: None,
        kv_transfer_params: None,
    });

    let choice_zero = routed_experts_payload(10);
    let choice_one = routed_experts_payload(20);
    let completion =
        PrefillRouter::consume_prefill_stream(prefill_stream(vec![first, second]), None, Some(2))
            .await
            .unwrap();
    let metadata = completion
        .result
        .generate_metadata
        .expect("typed Generate prefill metadata");

    assert_eq!(metadata.prompt_logprobs, Some(prompt_logprobs));
    assert_eq!(
        metadata
            .routed_experts_by_choice
            .get(&0)
            .map(String::as_str),
        Some(choice_zero.as_str())
    );
    assert_eq!(
        metadata
            .routed_experts_by_choice
            .get(&1)
            .map(String::as_str),
        Some(choice_one.as_str())
    );
}

#[tokio::test]
async fn prefill_metadata_rejects_malformed_routed_experts() {
    let mut output = valid_prefill_output();
    output.data.as_mut().unwrap().generate_metadata = Some(GenerateBackendMetadata {
        routed_experts: Some("not-base64".to_string()),
        ..Default::default()
    });

    let error = PrefillRouter::consume_prefill_stream(prefill_stream(vec![output]), None, Some(1))
        .await
        .err()
        .expect("malformed routed experts must fail");

    assert!(error.to_string().contains("routed experts"));
}

#[tokio::test]
async fn prefill_metadata_rejects_unbounded_choice_and_frame_stream() {
    let payload = routed_experts_payload(1);
    let outputs = (0..20)
        .map(|index| {
            let mut output = valid_prefill_output();
            output.data.as_mut().unwrap().index = Some(index);
            output.data.as_mut().unwrap().generate_metadata = Some(GenerateBackendMetadata {
                routed_experts: Some(payload.clone()),
                ..Default::default()
            });
            output
        })
        .collect();

    let error = PrefillRouter::consume_prefill_stream(prefill_stream(outputs), None, Some(1))
        .await
        .err()
        .expect("unbounded metadata must fail");

    assert!(error.to_string().contains("choice") || error.to_string().contains("frame"));
}

#[tokio::test]
async fn prefill_metadata_rejects_partial_routed_expert_choices() {
    let mut output = valid_prefill_output();
    output.data.as_mut().unwrap().index = Some(0);
    output.data.as_mut().unwrap().generate_metadata = Some(GenerateBackendMetadata {
        routed_experts: Some(routed_experts_payload(1)),
        ..Default::default()
    });

    let error = PrefillRouter::consume_prefill_stream(prefill_stream(vec![output]), None, Some(2))
        .await
        .err()
        .expect("partial per-choice routed experts must fail");

    assert!(error.to_string().contains("1 of 2 choices"));
}

#[tokio::test]
async fn bootstrap_prefill_metadata_is_attached_to_matching_decode_choices() {
    let prompt_logprobs = vec![Some(HashMap::from([(
        11,
        GenerateLogprob {
            logprob: -0.25,
            rank: Some(1),
            decoded_token: None,
        },
    )]))];
    let prefill_metadata = crate::protocols::inference::generate::GeneratePrefillMetadata {
        prompt_logprobs: Some(prompt_logprobs.clone()),
        routed_experts_by_choice: std::collections::BTreeMap::from([
            (0, "prefill-zero".to_string()),
            (1, "prefill-one".to_string()),
        ]),
    };
    let completion = tokio::spawn(async move {
        Ok(PrefillCompletion {
            result: crate::protocols::common::preprocessor::PrefillResult {
                disaggregated_params: json!({}),
                prompt_tokens_details: None,
                generate_metadata: Some(prefill_metadata),
            },
            worker_link: None,
        })
    });
    let terminal = |index| {
        Annotated::from_data(LLMEngineOutput {
            index: Some(index),
            finish_reason: Some(crate::protocols::common::FinishReason::Stop),
            generate_metadata: Some(GenerateBackendMetadata {
                routed_experts: Some(format!("decode-{index}")),
                ..Default::default()
            }),
            ..Default::default()
        })
    };
    let stream = prefill_stream(vec![terminal(0), terminal(1)]);
    let outputs =
        PrefillRouter::attach_bootstrap_generate_metadata(stream, AbortOnDrop::new(completion))
            .collect::<Vec<_>>()
            .await;

    assert_eq!(outputs.len(), 2);
    for (index, output) in outputs.into_iter().enumerate() {
        let metadata = output.data.unwrap().generate_metadata.unwrap();
        assert_eq!(metadata.prompt_logprobs, Some(prompt_logprobs.clone()));
        assert_eq!(
            metadata.prefill_routed_experts.as_deref(),
            Some(if index == 0 {
                "prefill-zero"
            } else {
                "prefill-one"
            })
        );
        assert_eq!(
            metadata.routed_experts.as_deref(),
            Some(if index == 0 { "decode-0" } else { "decode-1" })
        );
    }
}

#[tokio::test]
async fn bootstrap_without_optional_metadata_handles_multiple_terminals() {
    let completion = tokio::spawn(async move {
        Ok(PrefillCompletion {
            result: crate::protocols::common::preprocessor::PrefillResult {
                disaggregated_params: json!({}),
                prompt_tokens_details: None,
                generate_metadata: None,
            },
            worker_link: None,
        })
    });
    let terminal = |index| {
        Annotated::from_data(LLMEngineOutput {
            index: Some(index),
            finish_reason: Some(crate::protocols::common::FinishReason::Stop),
            ..Default::default()
        })
    };
    let outputs = PrefillRouter::attach_bootstrap_generate_metadata(
        prefill_stream(vec![terminal(0), terminal(1)]),
        AbortOnDrop::new(completion),
    )
    .collect::<Vec<_>>()
    .await;

    assert_eq!(outputs.len(), 2);
    assert!(outputs.iter().all(|output| output.error.is_none()));
}

struct DropSignal(Arc<AtomicBool>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[tokio::test]
async fn dropping_owned_bootstrap_before_decode_attachment_aborts_task() {
    let dropped = Arc::new(AtomicBool::new(false));
    let task_dropped = dropped.clone();
    let completion = tokio::spawn(async move {
        let _signal = DropSignal(task_dropped);
        std::future::pending::<Result<PrefillCompletion, PrefillError>>().await
    });
    tokio::task::yield_now().await;

    drop(AbortOnDrop::new(completion));
    tokio::task::yield_now().await;

    assert!(dropped.load(Ordering::Acquire));
}

#[tokio::test]
async fn dropping_decode_wrapper_aborts_bootstrap_prefill_task() {
    let dropped = Arc::new(AtomicBool::new(false));
    let task_dropped = dropped.clone();
    let completion = tokio::spawn(async move {
        let _signal = DropSignal(task_dropped);
        std::future::pending::<Result<PrefillCompletion, PrefillError>>().await
    });
    tokio::task::yield_now().await;
    let wrapped = PrefillRouter::attach_bootstrap_generate_metadata(
        prefill_stream(Vec::new()),
        AbortOnDrop::new(completion),
    );

    drop(wrapped);
    tokio::task::yield_now().await;

    assert!(dropped.load(Ordering::Acquire));
}

#[tokio::test]
async fn request_cancellation_interrupts_bootstrap_metadata_wait() {
    let controller = Arc::new(Controller::default());
    let terminal = Annotated::from_data(LLMEngineOutput {
        finish_reason: Some(crate::protocols::common::FinishReason::Stop),
        ..Default::default()
    });
    let decode = ResponseStream::new(Box::pin(stream::iter(vec![terminal])), controller.clone());
    let completion = tokio::spawn(async move {
        std::future::pending::<Result<PrefillCompletion, PrefillError>>().await
    });
    let mut wrapped =
        PrefillRouter::attach_bootstrap_generate_metadata(decode, AbortOnDrop::new(completion));
    let next = tokio::spawn(async move { wrapped.next().await });
    tokio::task::yield_now().await;

    dynamo_runtime::pipeline::AsyncEngineContext::kill(controller.as_ref());
    let output = tokio::time::timeout(std::time::Duration::from_secs(1), next)
        .await
        .expect("cancellation should interrupt metadata wait")
        .unwrap()
        .expect("cancellation should produce an annotated error");

    assert!(output.error.is_some());
}

#[tokio::test(start_paused = true)]
async fn bootstrap_metadata_wait_has_a_deadline() {
    let terminal = Annotated::from_data(LLMEngineOutput {
        finish_reason: Some(crate::protocols::common::FinishReason::Stop),
        ..Default::default()
    });
    let completion = tokio::spawn(async move {
        std::future::pending::<Result<PrefillCompletion, PrefillError>>().await
    });
    let mut wrapped = PrefillRouter::attach_bootstrap_generate_metadata(
        prefill_stream(vec![terminal]),
        AbortOnDrop::new(completion),
    );
    let next = tokio::spawn(async move { wrapped.next().await });
    tokio::task::yield_now().await;

    tokio::time::advance(BOOTSTRAP_PREFILL_COMPLETION_TIMEOUT).await;
    let output = next
        .await
        .unwrap()
        .expect("deadline should produce an annotated error");

    assert!(
        output
            .error
            .as_ref()
            .is_some_and(|error| error.to_string().contains("timed out"))
    );
}
