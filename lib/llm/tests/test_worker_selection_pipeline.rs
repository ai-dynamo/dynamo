// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for worker selection pipeline functionality

use dynamo_async_openai::types::CreateChatCompletionRequest;
use dynamo_llm::entrypoint::input::worker_selection_pipeline::*;
use dynamo_llm::protocols::openai::{
    chat_completions::NvCreateChatCompletionRequest, common_ext::CommonExt, nvext::NvExt,
};
use dynamo_runtime::protocols::annotated::AnnotationsProvider;

#[tokio::test]
#[ignore] // Requires full distributed setup
async fn test_worker_selection_pipeline() {
    // This test would require:
    // - A real ModelDeploymentCard
    // - A Component client connected to workers
    // - A KvRouter with actual worker state

    // Example test structure:
    // let engine = build_worker_selection_pipeline(...).await.unwrap();
    //
    // // Create a request with query_instance_id annotation
    // let request = create_test_request_with_annotation("query_instance_id");
    // let response_stream = engine.generate(request).await.unwrap();
    //
    // // Use the helper function to extract worker selection information
    // let (worker_id, tokens) = extract_worker_selection_from_stream(response_stream).await.unwrap();
    //
    // assert!(worker_id > 0);
    // assert!(!tokens.is_empty());
}

#[test]
fn test_add_query_instance_id() {
    // Test adding annotation to request without nvext
    let mut request = NvCreateChatCompletionRequest {
        inner: CreateChatCompletionRequest::default(),
        common: CommonExt::default(),
        nvext: None,
        chat_template_args: None,
    };

    // Initially should not have the annotation
    assert!(!request.has_annotation("query_instance_id"));

    // Add the annotation
    add_query_instance_id(&mut request);

    // Now should have the annotation
    assert!(request.has_annotation("query_instance_id"));

    // Test adding annotation to request that already has nvext but no annotations
    let mut request2 = NvCreateChatCompletionRequest {
        inner: CreateChatCompletionRequest::default(),
        common: CommonExt::default(),
        nvext: Some(NvExt::builder().build().unwrap()),
        chat_template_args: None,
    };

    assert!(!request2.has_annotation("query_instance_id"));
    add_query_instance_id(&mut request2);
    assert!(request2.has_annotation("query_instance_id"));

    // Test adding annotation to request that already has annotations
    let mut request3 = NvCreateChatCompletionRequest {
        inner: CreateChatCompletionRequest::default(),
        common: CommonExt::default(),
        nvext: Some(
            NvExt::builder()
                .add_annotation("some_other_annotation")
                .build()
                .unwrap(),
        ),
        chat_template_args: None,
    };

    assert!(request3.has_annotation("some_other_annotation"));
    assert!(!request3.has_annotation("query_instance_id"));

    add_query_instance_id(&mut request3);

    assert!(request3.has_annotation("some_other_annotation"));
    assert!(request3.has_annotation("query_instance_id"));

    // Test that adding the same annotation twice doesn't duplicate it
    add_query_instance_id(&mut request3);

    let annotations = request3.annotations().unwrap();
    let query_instance_id_count = annotations
        .iter()
        .filter(|&ann| ann == "query_instance_id")
        .count();
    assert_eq!(
        query_instance_id_count, 1,
        "query_instance_id should appear only once"
    );
}

#[test]
fn test_annotation_helper_functions() {
    // Test adding worker_instance_id annotation
    let mut request = NvCreateChatCompletionRequest {
        inner: CreateChatCompletionRequest::default(),
        common: CommonExt::default(),
        nvext: None,
        chat_template_args: None,
    };

    add_worker_instance_id_annotation(&mut request, 42);

    let annotations = request.annotations().unwrap();
    assert!(annotations.contains(&"worker_instance_id:42".to_string()));

    // Test adding token_data annotation
    let tokens = vec![1, 2, 3, 4, 5];
    add_token_data_annotation(&mut request, &tokens);

    let annotations = request.annotations().unwrap();
    assert!(annotations.contains(&"token_data:[1,2,3,4,5]".to_string()));
    assert!(annotations.contains(&"worker_instance_id:42".to_string()));

    // Test updating worker_instance_id (should replace existing)
    add_worker_instance_id_annotation(&mut request, 99);

    let annotations = request.annotations().unwrap();
    assert!(annotations.contains(&"worker_instance_id:99".to_string()));
    assert!(!annotations.contains(&"worker_instance_id:42".to_string()));

    // Verify token_data is still there
    assert!(annotations.contains(&"token_data:[1,2,3,4,5]".to_string()));
}
