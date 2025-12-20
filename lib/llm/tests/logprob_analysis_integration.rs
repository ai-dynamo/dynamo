// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for logprob analysis functionality

use std::sync::Arc;
use std::time::Instant;

use dynamo_llm::perf::logprobs::analyze_logprob_sensitivity;
use dynamo_llm::perf::{RecordedStream, TimestampedResponse};
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

use dynamo_async_openai::types::{
    ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionStreamResponseDelta,
    ChatCompletionTokenLogprob, FinishReason, Role, TopLogprobs,
};

// Type aliases to simplify complex test data structures
type TokenAlternative = (&'static str, f32);
type TokenData = (&'static str, f32, Vec<TokenAlternative>);
type TokenDataVec = Vec<TokenData>;

// Type aliases for multi-choice test data (using String instead of &str)
type StringTokenAlternative = (String, f32);
type StringTokenData = (String, f32, Vec<StringTokenAlternative>);
type ChoiceTokenData = Vec<StringTokenData>;
type MultiChoiceData = Vec<ChoiceTokenData>;

/// Test full workflow with realistic streaming data
#[test]
fn test_realistic_streaming_analysis() {
    let stream = create_realistic_stream();
    let analysis = analyze_logprob_sensitivity(stream);

    // Verify basic structure
    assert_eq!(analysis.total_responses, 3);
    assert_eq!(analysis.choice_analyses.len(), 1);
    assert_eq!(
        analysis.choice_analyses.get(&0).unwrap().positions_analyzed,
        3
    );

    // Check that positions are sorted by closeness
    let positions = &analysis.choice_analyses.get(&0).unwrap().position_closeness;
    for i in 1..positions.len() {
        assert!(positions[i - 1].probability_difference <= positions[i].probability_difference);
    }

    // Test API methods
    let close_positions = analysis.get_close_positions_for_choice(0, 0.2);
    assert!(!close_positions.is_empty());

    let percentage = analysis.close_position_percentage_for_choice(0, 0.2);
    assert!((0.0..=100.0).contains(&percentage));
}

/// Test multiple choices analysis
#[test]
fn test_multiple_choices_independent_analysis() {
    let stream = create_multi_choice_stream();
    let analysis = analyze_logprob_sensitivity(stream);

    // Should have 2 choices
    assert_eq!(analysis.choice_analyses.len(), 2);

    // Each choice should be analyzed independently
    let choice0_count = analysis.choice_analyses.get(&0).unwrap().positions_analyzed;
    let choice1_count = analysis.choice_analyses.get(&1).unwrap().positions_analyzed;
    assert_eq!(choice0_count, 2);
    assert_eq!(choice1_count, 2);

    // Test that choices have different closeness patterns
    let choice0_close = analysis.get_close_positions_for_choice(0, 0.3);
    let choice1_close = analysis.get_close_positions_for_choice(1, 0.3);

    // Based on our test data, choice 1 should have closer logprobs
    assert!(choice1_close.len() >= choice0_close.len());
}

/// Test detection of multiple close tokens
#[test]
fn test_multiple_close_tokens_detection() {
    let stream = create_stream_with_multiple_close_tokens();
    let analysis = analyze_logprob_sensitivity(stream);

    // Should detect positions with 3+ close tokens
    let multiple_close = analysis.detect_multiple_close_tokens(0, 0.05);
    assert!(!multiple_close.is_empty());

    let first_multiple = &multiple_close[0];
    assert!(first_multiple.close_count >= 3);
    assert!(first_multiple.max_difference <= 0.05);

    // Verify the close tokens are actually close in probability space
    for i in 1..first_multiple.close_tokens.len() {
        let prob_top = first_multiple.close_tokens[0].logprob.exp();
        let prob_current = first_multiple.close_tokens[i].logprob.exp();
        let diff = prob_top - prob_current;
        assert!(diff <= 0.05);
    }
}

/// Test edge cases and error handling
#[test]
fn test_edge_cases() {
    // Empty stream
    let empty_stream = create_empty_stream();
    let analysis = analyze_logprob_sensitivity(empty_stream);
    assert_eq!(analysis.total_responses, 0);
    assert!(analysis.choice_analyses.is_empty());

    // Single token positions (no alternatives)
    let single_token_stream = create_single_token_stream();
    let analysis = analyze_logprob_sensitivity(single_token_stream);

    // Should have no close positions since there's only one token per position
    let close_positions = analysis.get_close_positions_for_choice(0, 1.0);
    assert!(close_positions.is_empty());
}

/// Test threshold sensitivity
#[test]
fn test_threshold_sensitivity() {
    let stream = create_graduated_closeness_stream();
    let analysis = analyze_logprob_sensitivity(stream);

    // Test different thresholds
    let strict_close = analysis.get_close_positions_for_choice(0, 0.01);
    let permissive_close = analysis.get_close_positions_for_choice(0, 0.1);
    let very_permissive_close = analysis.get_close_positions_for_choice(0, 0.5);

    // Should have increasing numbers of close positions
    assert!(strict_close.len() <= permissive_close.len());
    assert!(permissive_close.len() <= very_permissive_close.len());

    // Percentages should increase with threshold
    let strict_pct = analysis.close_position_percentage_for_choice(0, 0.01);
    let permissive_pct = analysis.close_position_percentage_for_choice(0, 0.1);
    assert!(strict_pct <= permissive_pct);
}

/// Test performance with larger datasets
#[test]
fn test_large_dataset_performance() {
    let stream = create_large_stream(100, 5); // 100 positions, 5 choices
    let start_time = Instant::now();
    let analysis = analyze_logprob_sensitivity(stream);
    let elapsed = start_time.elapsed();

    // Should complete quickly
    assert!(elapsed.as_millis() < 100);

    // Verify correctness
    assert_eq!(analysis.total_responses, 100);
    assert_eq!(analysis.choice_analyses.len(), 5);

    for i in 0..5 {
        let choice_analysis = analysis.choice_analyses.get(&(i as u32)).unwrap();
        assert_eq!(choice_analysis.choice_index, i as u32);
        assert_eq!(choice_analysis.positions_analyzed, 100);
    }
}

// Helper functions for creating test data

fn create_realistic_stream() -> Arc<RecordedStream<NvCreateChatCompletionStreamResponse>> {
    let start_time = Instant::now();
    let responses = vec![
        TimestampedResponse::new(
            create_response_with_linear_probs(
                "Hello",
                vec![("Hello", 0.6, vec![("Hi", 0.3), ("Hey", 0.1)])], // Moderate differences
            ),
            0,
        ),
        TimestampedResponse::new(
            create_response_with_linear_probs(
                " world",
                vec![(" world", 0.55, vec![(" there", 0.4), (" everyone", 0.05)])], // Close competition
            ),
            1,
        ),
        TimestampedResponse::new(
            create_response_with_linear_probs(
                "!",
                vec![("!", 0.8, vec![(".", 0.15), ("?", 0.05)])],
            ), // Clear winner
            2,
        ),
    ];

    let stream = RecordedStream::new(responses, start_time, Instant::now());
    Arc::new(stream)
}

fn create_multi_choice_stream() -> Arc<RecordedStream<NvCreateChatCompletionStreamResponse>> {
    let start_time = Instant::now();
    let responses = vec![
        TimestampedResponse::new(
            create_multi_choice_response(vec![
                // Choice 0: moderate closeness (65% vs 35%)
                vec![("token1".to_string(), 0.65, vec![("alt1".to_string(), 0.35)])],
                // Choice 1: very close logprobs (51% vs 49%)
                vec![("token2".to_string(), 0.51, vec![("alt2".to_string(), 0.49)])],
            ]),
            0,
        ),
        TimestampedResponse::new(
            create_multi_choice_response(vec![
                // Choice 0: not close (80% vs 20%)
                vec![("token3".to_string(), 0.8, vec![("alt3".to_string(), 0.2)])],
                // Choice 1: close (53% vs 47%)
                vec![("token4".to_string(), 0.53, vec![("alt4".to_string(), 0.47)])],
            ]),
            1,
        ),
    ];

    let stream = RecordedStream::new(responses, start_time, Instant::now());
    Arc::new(stream)
}

// fn create_stream_from_recorded_sse_stream(
//     file: &str,
// ) -> Arc<RecordedStream<NvCreateChatCompletionStreamResponse>> {
//     let data = std::fs::read_to_string(file).unwrap();
//     let sse_stream = create_message_stream(&data);
//     let response_stream =
//         convert_sse_stream::<NvCreateChatCompletionStreamResponse>(Box::pin(sse_stream));

//     let context = Arc::new(MockContext::new());
//     let response_stream = record_stream_with_context(response_stream, context, RecordingMode::Sink);
//     let filtered_stream = response_stream.filter_map(|annotated| async move { annotated.data });
//     let (recorded_stream, recording_rx) =
//         record_stream_with_context(Box::pin(filtered_stream), ctx, RecordingMode::Sink);
// }

fn create_stream_with_multiple_close_tokens()
-> Arc<RecordedStream<NvCreateChatCompletionStreamResponse>> {
    let start_time = Instant::now();
    let responses = vec![TimestampedResponse::new(
        create_response_with_linear_probs(
            "test",
            vec![(
                "test",
                0.27,
                vec![
                    ("best", 0.26), // diff = 0.01
                    ("rest", 0.25), // diff = 0.01 from best, 0.02 from test
                    ("nest", 0.22), // diff = 0.03 from rest, 0.05 from test (sum = 1.0)
                ],
            )],
        ),
        0,
    )];

    let stream = RecordedStream::new(responses, start_time, Instant::now());
    Arc::new(stream)
}

fn create_empty_stream() -> Arc<RecordedStream<NvCreateChatCompletionStreamResponse>> {
    let start_time = Instant::now();
    let stream = RecordedStream::new(vec![], start_time, Instant::now());
    Arc::new(stream)
}

fn create_single_token_stream() -> Arc<RecordedStream<NvCreateChatCompletionStreamResponse>> {
    let start_time = Instant::now();
    let responses = vec![TimestampedResponse::new(
        create_response_with_linear_probs(
            "only",
            vec![
                ("only", 1.0, vec![]), // 100% probability, no alternatives
            ],
        ),
        0,
    )];

    let stream = RecordedStream::new(responses, start_time, Instant::now());
    Arc::new(stream)
}

fn create_graduated_closeness_stream() -> Arc<RecordedStream<NvCreateChatCompletionStreamResponse>>
{
    let start_time = Instant::now();
    let responses = vec![TimestampedResponse::new(
        create_response_with_linear_probs(
            "test",
            vec![
                ("very_close", 0.501, vec![("alt1", 0.499)]), // diff = 0.002 (very close)
                ("close", 0.55, vec![("alt2", 0.45)]),        // diff = 0.1 (close)
                ("medium", 0.7, vec![("alt3", 0.3)]),         // diff = 0.4 (medium)
                ("far", 0.9, vec![("alt4", 0.1)]),            // diff = 0.8 (far)
            ],
        ),
        0,
    )];

    let stream = RecordedStream::new(responses, start_time, Instant::now());
    Arc::new(stream)
}

fn create_large_stream(
    positions: usize,
    choices: usize,
) -> Arc<RecordedStream<NvCreateChatCompletionStreamResponse>> {
    let start_time = Instant::now();
    let mut responses = Vec::new();

    for i in 0..positions {
        let mut choice_data = Vec::new();
        for j in 0..choices {
            let token = format!("token_{}_{}", i, j);
            let alt = format!("alt_{}_{}", i, j);

            // Create varied but realistic probability distributions
            let prob = 0.5 + (i as f32 * 0.001) + (j as f32 * 0.01); // Range: ~0.5-0.6
            let alt_prob = 1.0 - prob - 0.05; // Ensure sum < 1, remaining ~5-15% for other tokens
            let alt_prob = alt_prob.max(0.1); // Ensure alt_prob is reasonable

            choice_data.push(vec![(token, prob, vec![(alt, alt_prob)])]);
        }
        responses.push(TimestampedResponse::new(
            create_multi_choice_response(choice_data),
            i,
        ));
    }

    let stream = RecordedStream::new(responses, start_time, Instant::now());
    Arc::new(stream)
}

/// Helper function to create response with linear probabilities [0, 1]
/// This ensures realistic probability distributions that sum to ≤ 1
fn create_response_with_linear_probs(
    _content: &str,
    token_data: TokenDataVec,
) -> NvCreateChatCompletionStreamResponse {
    let token_logprobs = token_data
        .into_iter()
        .map(|(token, prob, alternatives)| {
            // Validate probabilities
            assert!(
                (0.0..=1.0).contains(&prob),
                "Probability must be in [0, 1]: {}",
                prob
            );
            let total_prob = prob + alternatives.iter().map(|(_, p)| p).sum::<f32>();
            assert!(
                total_prob <= 1.001,
                "Total probability mass exceeds 1: {}",
                total_prob
            );

            let top_logprobs = alternatives
                .into_iter()
                .map(|(alt_token, alt_prob)| {
                    assert!(
                        (0.0..=1.0).contains(&alt_prob),
                        "Probability must be in [0, 1]: {}",
                        alt_prob
                    );
                    TopLogprobs {
                        token: alt_token.to_string(),
                        logprob: alt_prob.ln(),
                        bytes: None,
                    }
                })
                .collect();

            ChatCompletionTokenLogprob {
                token: token.to_string(),
                logprob: prob.ln(),
                bytes: None,
                top_logprobs,
            }
        })
        .collect();

    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            content: Some(_content.to_string()),
            #[expect(deprecated)]
            function_call: None,
            tool_calls: None,
            role: Some(Role::Assistant),
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: Some(FinishReason::Stop),
        logprobs: Some(ChatChoiceLogprobs {
            content: Some(token_logprobs),
            refusal: None,
        }),
    };

    NvCreateChatCompletionStreamResponse {
        id: "test_id".to_string(),
        choices: vec![choice],
        created: 1234567890,
        model: "test-model".to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
        nvext: None,
    }
}

fn create_multi_choice_response(
    choices_data: MultiChoiceData,
) -> NvCreateChatCompletionStreamResponse {
    let choices = choices_data
        .into_iter()
        .enumerate()
        .map(|(choice_idx, token_data)| {
            let token_logprobs = token_data
                .into_iter()
                .map(|(token, prob, alternatives)| {
                    // Validate probabilities
                    assert!(
                        (0.0..=1.0).contains(&prob),
                        "Probability must be in [0, 1]: {}",
                        prob
                    );
                    let total_prob = prob + alternatives.iter().map(|(_, p)| p).sum::<f32>();
                    assert!(
                        total_prob <= 1.001,
                        "Total probability mass exceeds 1: {}",
                        total_prob
                    );

                    let top_logprobs = alternatives
                        .into_iter()
                        .map(|(alt_token, alt_prob)| {
                            assert!(
                                (0.0..=1.0).contains(&alt_prob),
                                "Probability must be in [0, 1]: {}",
                                alt_prob
                            );
                            TopLogprobs {
                                token: alt_token,
                                logprob: alt_prob.ln(),
                                bytes: None,
                            }
                        })
                        .collect();

                    ChatCompletionTokenLogprob {
                        token,
                        logprob: prob.ln(),
                        bytes: None,
                        top_logprobs,
                    }
                })
                .collect();

            ChatChoiceStream {
                index: choice_idx as u32,
                delta: ChatCompletionStreamResponseDelta {
                    content: Some("test".to_string()),
                    #[expect(deprecated)]
                    function_call: None,
                    tool_calls: None,
                    role: Some(Role::Assistant),
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: Some(ChatChoiceLogprobs {
                    content: Some(token_logprobs),
                    refusal: None,
                }),
            }
        })
        .collect();

    NvCreateChatCompletionStreamResponse {
        id: "test_id".to_string(),
        choices,
        created: 1234567890,
        model: "test-model".to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
        nvext: None,
    }
}
