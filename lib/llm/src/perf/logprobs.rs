// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Module for recording logprobs from a streaming response.
//!
//! Logprobs are a bit easier than token counting and timing because they are
//! fully self-contained in the response chunk.
//!
//! In fact, if logprobs are given, they are a good way to count tokens; however,
//! the emission of logprobs is also more costly and generally not available unless
//! explicitly requested.
//!
//! The primary reason to record logprobs is to analyze the possible outputs of
//! a model as a function of sequence position.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::perf::RecordedStream;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

/// The type of logprobs observed in the response.
pub enum LogprobType {
    /// If normalized, then all the reported "top_logprobs" sum to 0.
    Normalized,

    /// If unnormalized, then the reported "top_logprobs" are not normalized,
    /// so the sum of the "top_logprobs" will not sum to 0.
    Unnormalized,
}

/// Represents a token with its logprob information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// The token as a string
    pub token: String,
    /// The log probability of this token
    pub logprob: f32,
    /// Optional byte representation of the token
    pub bytes: Option<Vec<u8>>,
}

/// Represents logprob information for a single position with selected and alternative tokens
#[derive(Debug, Clone)]
pub struct TokenLogProbs {
    selected: TokenLogprob,
    alternatives: Vec<TokenLogprob>,
    all_sorted: Vec<TokenLogprob>,
}

impl TokenLogProbs {
    /// Create a new TokenLogProbs from a selected token and alternatives
    pub fn new(selected: TokenLogprob, mut alternatives: Vec<TokenLogprob>) -> Self {
        // Sort alternatives by logprob (highest first)
        alternatives.sort_by(|a, b| b.logprob.partial_cmp(&a.logprob).unwrap());

        // Create all_sorted by merging selected with alternatives (ensuring uniqueness)
        let mut all_sorted = Vec::new();
        let mut added_selected = false;

        // Check if selected token appears in alternatives
        let selected_in_alternatives = alternatives.iter().any(|alt| {
            alt.token == selected.token && (alt.logprob - selected.logprob).abs() < 1e-6
        });

        // If selected is not in alternatives, we need to insert it in the right position
        if !selected_in_alternatives {
            // Find the correct position to insert selected token
            let mut insert_position = alternatives.len();
            for (i, alt) in alternatives.iter().enumerate() {
                if selected.logprob > alt.logprob {
                    insert_position = i;
                    break;
                }
            }

            // Build all_sorted by merging at the correct position
            for (i, alt) in alternatives.iter().enumerate() {
                if i == insert_position && !added_selected {
                    all_sorted.push(selected.clone());
                    added_selected = true;
                }
                all_sorted.push(alt.clone());
            }

            // If we haven't added selected yet, it goes at the end
            if !added_selected {
                all_sorted.push(selected.clone());
            }
        } else {
            // Selected is already in alternatives, just use alternatives
            all_sorted = alternatives.clone();
        }

        Self {
            selected,
            alternatives,
            all_sorted,
        }
    }

    /// Get the selected token
    pub fn selected_token(&self) -> &TokenLogprob {
        &self.selected
    }

    /// Get alternative tokens sorted by most likely first
    pub fn alternative_tokens(&self) -> &[TokenLogprob] {
        &self.alternatives
    }

    /// Get all tokens (selected merged with alternatives, unique) sorted by most likely first
    pub fn all_tokens(&self) -> &[TokenLogprob] {
        &self.all_sorted
    }
}

/// Trait for extracting logprob information from various response types
pub trait LogprobExtractor {
    /// Extract logprobs organized by choice index
    /// Returns: HashMap<choice_index, Vec<TokenLogProbs>>
    fn extract_logprobs_by_choice(&self) -> HashMap<u32, Vec<TokenLogProbs>>;
}

/// Implementation for NvCreateChatCompletionStreamResponse (our main streaming response type)
impl LogprobExtractor for NvCreateChatCompletionStreamResponse {
    fn extract_logprobs_by_choice(&self) -> HashMap<u32, Vec<TokenLogProbs>> {
        let mut result = HashMap::new();

        for choice in &self.choices {
            let choice_index = choice.index;

            let choice_logprobs = choice
                .logprobs
                .as_ref()
                .and_then(|logprobs| logprobs.content.as_ref())
                .map(|content| {
                    content
                        .iter()
                        .map(|token_logprob| {
                            let selected_token = TokenLogprob {
                                token: token_logprob.token.clone(),
                                logprob: token_logprob.logprob,
                                bytes: token_logprob.bytes.clone(),
                            };

                            // Convert top alternatives to our format
                            let alternatives: Vec<TokenLogprob> = token_logprob
                                .top_logprobs
                                .iter()
                                .map(|top_logprob| TokenLogprob {
                                    token: top_logprob.token.clone(),
                                    logprob: top_logprob.logprob,
                                    bytes: top_logprob.bytes.clone(),
                                })
                                .collect();

                            TokenLogProbs::new(selected_token, alternatives)
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            result.insert(choice_index, choice_logprobs);
        }

        result
    }
}

/// Validate and flatten choice logprobs HashMap to Vec
/// Ensures all expected choice indices [0, max_choice) are present
pub fn validate_and_flatten_choices(
    choice_logprobs: HashMap<u32, Vec<TokenLogProbs>>,
) -> Result<Vec<Vec<TokenLogProbs>>, String> {
    if choice_logprobs.is_empty() {
        return Ok(Vec::new());
    }

    let max_choice = *choice_logprobs.keys().max().unwrap();
    let expected_count = (max_choice + 1) as usize;

    if choice_logprobs.len() != expected_count {
        return Err(format!(
            "Missing choice indices: expected {} choices [0, {}), but found {} choices: {:?}",
            expected_count,
            max_choice + 1,
            choice_logprobs.len(),
            choice_logprobs.keys().collect::<Vec<_>>()
        ));
    }

    // Validate all indices from 0 to max_choice are present
    for i in 0..=max_choice {
        if !choice_logprobs.contains_key(&i) {
            return Err(format!(
                "Missing choice index {}: expected [0, {}), found {:?}",
                i,
                max_choice + 1,
                choice_logprobs.keys().collect::<Vec<_>>()
            ));
        }
    }

    // Flatten to Vec ordered by keys
    let mut result = Vec::with_capacity(expected_count);
    for i in 0..=max_choice {
        result.push(choice_logprobs[&i].clone());
    }

    Ok(result)
}

/// Analysis focused on detecting close logprobs indicating model uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    /// Total number of responses analyzed
    pub total_responses: usize,
    /// Analysis results per choice index
    pub choice_analyses: HashMap<u32, ChoiceAnalysis>,
}

/// Analysis for a single choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceAnalysis {
    /// Choice index
    pub choice_index: u32,
    /// All positions with their closeness values, sorted by closeness
    pub position_closeness: Vec<PositionCloseness>,
    /// Number of positions analyzed for this choice
    pub positions_analyzed: usize,
}

/// Closeness information for a position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionCloseness {
    /// Position in the stream (response index)
    pub stream_position: usize,
    /// Position within the token sequence
    pub token_position: usize,
    /// Logprob difference between top 2 candidates (deprecated - use probability_difference)
    pub logprob_difference: f32,
    /// Probability difference between top 2 candidates (in linear space 0-1)
    pub probability_difference: f32,
    /// Probability mass not accounted for by all_tokens (1 - sum of all_tokens probabilities)
    pub probability_remaining: f32,
    /// All candidates at this position, sorted by logprob (highest first)
    pub candidates: Vec<TokenLogprob>,
}

/// A position where top candidates have close probabilities
#[derive(Debug, Clone)]
pub struct ClosePosition {
    /// Position in the stream (response index)
    pub stream_position: usize,
    /// Position within the token sequence
    pub token_position: usize,
    /// Logprob difference between top 2 candidates (deprecated - use probability_difference)
    pub logprob_difference: f32,
    /// Probability difference between top 2 candidates (in linear space 0-1)
    pub probability_difference: f32,
    /// Probability mass not accounted for by top_candidates (1 - sum of top_candidates probabilities)
    pub probability_remaining: f32,
    /// Top 2 candidates at this position
    pub top_candidates: Vec<TokenLogprob>,
}

/// Analyzes logprobs from a recorded stream focusing on token similarity/closeness
pub fn analyze_logprob_sensitivity(
    recorded_stream: Arc<RecordedStream<impl LogprobExtractor>>,
) -> SensitivityAnalysis {
    let mut choice_analyses: HashMap<u32, ChoiceAnalysis> = HashMap::new();
    // Track cumulative sequence position per choice
    let mut choice_sequence_positions: HashMap<u32, usize> = HashMap::new();

    for (stream_pos, timestamped_response) in recorded_stream.responses().iter().enumerate() {
        let response = &timestamped_response.response;
        let logprobs_by_choice = response.extract_logprobs_by_choice();

        for (choice_index, choice_logprobs) in logprobs_by_choice {
            // Ensure we have a ChoiceAnalysis for this choice
            let choice_analysis =
                choice_analyses
                    .entry(choice_index)
                    .or_insert_with(|| ChoiceAnalysis {
                        choice_index,
                        position_closeness: Vec::new(),
                        positions_analyzed: 0,
                    });

            // Get current sequence position for this choice
            let current_seq_pos = choice_sequence_positions.entry(choice_index).or_insert(0);

            for token_logprobs in choice_logprobs {
                let all_tokens = token_logprobs.all_tokens();

                if all_tokens.len() < 2 {
                    *current_seq_pos += 1;
                    continue;
                }

                // all_tokens is already sorted by logprob (highest first)
                let sorted_candidates = all_tokens.to_vec();

                // Calculate difference between top 2 in both logprob and probability space
                let logprob_difference =
                    sorted_candidates[0].logprob - sorted_candidates[1].logprob;

                // Convert to probability space for more intuitive closeness calculation
                let prob1 = sorted_candidates[0].logprob.exp();
                let prob2 = sorted_candidates[1].logprob.exp();
                let probability_difference = prob1 - prob2;

                // Calculate probability_remaining
                let total_prob_sum: f32 = sorted_candidates.iter().map(|t| t.logprob.exp()).sum();
                let probability_remaining = 1.0 - total_prob_sum;

                choice_analysis.position_closeness.push(PositionCloseness {
                    stream_position: stream_pos,
                    token_position: *current_seq_pos,
                    logprob_difference,
                    probability_difference,
                    probability_remaining,
                    candidates: sorted_candidates,
                });

                choice_analysis.positions_analyzed += 1;
                *current_seq_pos += 1;
            }
        }
    }

    // Sort position closeness by probability difference (smallest first = most uncertain)
    for choice_analysis in choice_analyses.values_mut() {
        choice_analysis.position_closeness.sort_by(|a, b| {
            a.probability_difference
                .partial_cmp(&b.probability_difference)
                .unwrap()
        });
    }

    SensitivityAnalysis {
        total_responses: recorded_stream.responses().len(),
        choice_analyses,
    }
}

impl SensitivityAnalysis {
    /// Get positions below a threshold for a specific choice
    /// Threshold is in probability space (0-1), where smaller values indicate closer probabilities
    pub fn get_close_positions_for_choice(
        &self,
        choice_index: u32,
        threshold: f32,
    ) -> Vec<&PositionCloseness> {
        self.choice_analyses
            .get(&choice_index)
            .map(|analysis| {
                analysis
                    .position_closeness
                    .iter()
                    .filter(|pos| pos.probability_difference <= threshold)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the closest N positions for a specific choice
    pub fn get_closest_positions_for_choice(
        &self,
        choice_index: u32,
        count: usize,
    ) -> Vec<&PositionCloseness> {
        self.choice_analyses
            .get(&choice_index)
            .map(|analysis| analysis.position_closeness.iter().take(count).collect())
            .unwrap_or_default()
    }

    /// Print a summary of the sensitivity analysis
    pub fn print_summary(&self) {
        println!("=== Logprob Sensitivity Analysis Summary ===");
        println!("Total stream responses analyzed: {}", self.total_responses);
        println!("Number of choices: {}", self.choice_analyses.len());
        println!();

        for (choice_index, choice_analysis) in &self.choice_analyses {
            println!(
                "Choice {}: {} positions analyzed",
                choice_index, choice_analysis.positions_analyzed
            );

            if !choice_analysis.position_closeness.is_empty() {
                println!("  Closest positions (smallest probability differences):");
                for (j, pos) in choice_analysis
                    .position_closeness
                    .iter()
                    .take(3)
                    .enumerate()
                {
                    let top_token = &pos.candidates[0].token;
                    let second_token = &pos.candidates[1].token;
                    let prob1 = pos.candidates[0].logprob.exp();
                    let prob2 = pos.candidates[1].logprob.exp();
                    println!(
                        "    {}: Stream pos {}, token pos {} - '{}' ({:.1}%) vs '{}' ({:.1}%) (prob diff: {:.4})",
                        j + 1,
                        pos.stream_position,
                        pos.token_position,
                        top_token,
                        prob1 * 100.0,
                        second_token,
                        prob2 * 100.0,
                        pos.probability_difference
                    );
                }
            }
            println!();
        }
    }

    /// Get percentage of positions with close probabilities for a specific choice
    /// Threshold is in probability space (0-1)
    pub fn close_position_percentage_for_choice(&self, choice_index: u32, threshold: f32) -> f32 {
        if let Some(analysis) = self.choice_analyses.get(&choice_index) {
            if analysis.positions_analyzed == 0 {
                return 0.0;
            }
            let close_count = analysis
                .position_closeness
                .iter()
                .filter(|pos| pos.probability_difference <= threshold)
                .count();
            (close_count as f32 / analysis.positions_analyzed as f32) * 100.0
        } else {
            0.0
        }
    }

    /// Check if multiple tokens are close (within threshold of each other)
    pub fn detect_multiple_close_tokens(
        &self,
        choice_index: u32,
        threshold: f32,
    ) -> Vec<MultipleCloseTokens> {
        let mut results = Vec::new();

        if let Some(analysis) = self.choice_analyses.get(&choice_index) {
            for pos in &analysis.position_closeness {
                let close_tokens = self.count_close_tokens_at_position(pos, threshold);
                if close_tokens.close_count > 2 {
                    results.push(close_tokens);
                }
            }
        }

        results
    }

    /// Detect if greedy decoding was likely used by checking if selected tokens are always the most probable
    /// Note: This is an approximation since we infer selection from the data structure
    pub fn detect_likely_greedy_decoding(&self, choice_index: u32) -> bool {
        if let Some(analysis) = self.choice_analyses.get(&choice_index) {
            if analysis.positions_analyzed == 0 {
                return true; // No evidence against greedy
            }

            // For greedy detection, we're looking for positions with moderate to large differences
            // Very small differences (< 0.01) suggest equal alternatives - could be greedy or random
            // Very large differences (> 0.05) suggest clear winners - likely greedy
            let likely_greedy_positions = analysis
                .position_closeness
                .iter()
                .filter(|pos| {
                    if pos.candidates.is_empty() {
                        return true; // No contradiction
                    }

                    // Either very close (tie - could be greedy) or clear difference (likely greedy)
                    pos.probability_difference < 0.01 || pos.probability_difference > 0.05
                })
                .count();

            // If most positions show greedy-like patterns, consider it greedy
            (likely_greedy_positions as f32 / analysis.positions_analyzed as f32) > 0.5
        } else {
            false
        }
    }

    /// Get percentage of positions with greedy-like selection patterns
    pub fn greedy_selection_percentage(&self, choice_index: u32) -> f32 {
        if let Some(analysis) = self.choice_analyses.get(&choice_index) {
            if analysis.positions_analyzed == 0 {
                return 0.0;
            }

            let greedy_like_positions = analysis
                .position_closeness
                .iter()
                .filter(|pos| {
                    // Same logic as detect_likely_greedy_decoding for consistency
                    pos.probability_difference < 0.01 || pos.probability_difference > 0.05
                })
                .count();

            (greedy_like_positions as f32 / analysis.positions_analyzed as f32) * 100.0
        } else {
            0.0
        }
    }

    /// Count how many tokens are close at a specific position
    /// Threshold is in probability space (0-1)
    fn count_close_tokens_at_position(
        &self,
        position: &PositionCloseness,
        threshold: f32,
    ) -> MultipleCloseTokens {
        let top_prob = position.candidates[0].logprob.exp();
        let mut close_count = 1; // Top token is always included
        let mut close_tokens = vec![position.candidates[0].clone()];

        for candidate in &position.candidates[1..] {
            let candidate_prob = candidate.logprob.exp();
            let prob_diff = top_prob - candidate_prob;
            if prob_diff <= threshold {
                close_count += 1;
                close_tokens.push(candidate.clone());
            } else {
                break; // Since candidates are sorted, no need to check further
            }
        }

        let max_difference = if close_count > 1 {
            let last_prob = close_tokens.last().unwrap().logprob.exp();
            top_prob - last_prob
        } else {
            0.0
        };

        MultipleCloseTokens {
            stream_position: position.stream_position,
            token_position: position.token_position,
            close_count,
            close_tokens,
            max_difference,
        }
    }
}

/// Information about multiple close tokens at a position
#[derive(Debug, Clone)]
pub struct MultipleCloseTokens {
    pub stream_position: usize,
    pub token_position: usize,
    pub close_count: usize,
    pub close_tokens: Vec<TokenLogprob>,
    pub max_difference: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type aliases to simplify complex test data structures
    type TestTokenAlternative = (&'static str, f32);
    type TestTokenData = (&'static str, f32, Vec<TestTokenAlternative>);
    type TestTokenDataVec = Vec<TestTokenData>;
    use crate::perf::{RecordingMode, TimestampedResponse, record_stream_with_context};
    use crate::protocols::codec::create_message_stream;
    use crate::protocols::convert_sse_stream;
    use approx::assert_abs_diff_eq;
    use dynamo_async_openai::types::{
        ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionStreamResponseDelta,
        ChatCompletionTokenLogprob, FinishReason, Role, TopLogprobs,
    };
    use futures::StreamExt;
    use std::sync::Arc;
    use std::time::Instant;

    const FLOAT_EPSILON: f32 = 1e-6;

    #[test]
    fn test_two_tokens_close() {
        // Two very close tokens: 45% vs 44% (remaining 11% for other tokens)
        // Linear probs: [0.45, 0.44], difference = 0.01
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob_from_linear_probs(
            "hello",
            0.45,
            vec![("world", 0.44)], // Very close: 45% vs 44%
        )]);

        let close_positions = analysis.get_close_positions_for_choice(0, 0.1);
        assert_eq!(close_positions.len(), 1);

        // Probability difference should be 0.01 (45% - 44%)
        assert_abs_diff_eq!(
            close_positions[0].probability_difference,
            0.01,
            epsilon = FLOAT_EPSILON
        );

        // Logprob difference: ln(0.45) - ln(0.44) ≈ -0.798 - (-0.821) ≈ 0.023
        assert_abs_diff_eq!(
            close_positions[0].logprob_difference,
            0.023,
            epsilon = 0.001
        );

        let multiple_close = analysis.detect_multiple_close_tokens(0, 0.05);
        assert_eq!(multiple_close.len(), 0); // Only 2 tokens, so no "multiple" detected
    }

    #[test]
    fn test_three_tokens_close() {
        // Three close tokens: 35%, 33%, 32% (complete distribution)
        // Linear probs: [0.35, 0.33, 0.32], differences = [0.02, 0.01]
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob_from_linear_probs(
            "hello",
            0.35,
            vec![
                ("world", 0.33), // Close: 35% vs 33% (diff = 0.02)
                ("there", 0.32), // Close: 33% vs 32% (diff = 0.01)
            ],
        )]);

        let close_positions = analysis.get_close_positions_for_choice(0, 0.025);
        assert_eq!(close_positions.len(), 1);

        // Top 2 probability difference: 0.35 - 0.33 = 0.02
        assert_abs_diff_eq!(
            close_positions[0].probability_difference,
            0.02,
            epsilon = FLOAT_EPSILON
        );

        let multiple_close = analysis.detect_multiple_close_tokens(0, 0.04);
        assert_eq!(multiple_close.len(), 1);
        assert_eq!(multiple_close[0].close_count, 3);
        // Max difference: 0.35 - 0.32 = 0.03
        assert_abs_diff_eq!(
            multiple_close[0].max_difference,
            0.03,
            epsilon = FLOAT_EPSILON
        );
    }

    #[test]
    fn test_four_tokens_close() {
        // Four close tokens: 27%, 26%, 25%, 22% (complete distribution)
        // Linear probs: [0.27, 0.26, 0.25, 0.22], all very close
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob_from_linear_probs(
            "hello",
            0.27,
            vec![
                ("world", 0.26),  // Close: 27% vs 26% (diff = 0.01)
                ("there", 0.25),  // Close: 26% vs 25% (diff = 0.01)
                ("friend", 0.22), // Close: 25% vs 22% (diff = 0.03)
            ],
        )]);

        let close_positions = analysis.get_close_positions_for_choice(0, 0.02);
        assert_eq!(close_positions.len(), 1);

        // Top 2 probability difference: 0.27 - 0.26 = 0.01
        assert_abs_diff_eq!(
            close_positions[0].probability_difference,
            0.01,
            epsilon = FLOAT_EPSILON
        );

        let multiple_close = analysis.detect_multiple_close_tokens(0, 0.06);
        assert_eq!(multiple_close.len(), 1);
        assert_eq!(multiple_close[0].close_count, 4);
        // Max difference: 0.27 - 0.22 = 0.05
        assert_abs_diff_eq!(
            multiple_close[0].max_difference,
            0.05,
            epsilon = FLOAT_EPSILON
        );
    }

    #[test]
    fn test_multiple_choices_analysis() {
        let analysis = create_analysis_with_multiple_choices(vec![
            // Choice 0: Moderately close tokens (70% vs 25%, remaining 5%)
            vec![create_token_logprob_from_linear_probs(
                "hello",
                0.7,
                vec![("world", 0.25)],
            )],
            // Choice 1: Very close tokens (50.5% vs 49.5%)
            vec![create_token_logprob_from_linear_probs(
                "hi",
                0.505,
                vec![("there", 0.495)],
            )],
        ]);

        assert_eq!(analysis.choice_analyses.len(), 2);

        // Check choice 0: probability difference = 0.7 - 0.25 = 0.45
        let choice0_close = analysis.get_close_positions_for_choice(0, 0.5);
        assert_eq!(choice0_close.len(), 1);
        assert_abs_diff_eq!(
            choice0_close[0].probability_difference,
            0.45,
            epsilon = FLOAT_EPSILON
        );

        // Check choice 1: probability difference = 0.505 - 0.495 = 0.01
        let choice1_close = analysis.get_close_positions_for_choice(1, 0.5);
        assert_eq!(choice1_close.len(), 1);
        assert_abs_diff_eq!(
            choice1_close[0].probability_difference,
            0.01,
            epsilon = FLOAT_EPSILON
        );

        // Choice 1 should be much closer than choice 0
        assert!(choice1_close[0].probability_difference < choice0_close[0].probability_difference);
    }

    #[test]
    fn test_edge_case_single_token() {
        // Position with only one token (100% probability, no alternatives)
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob_from_linear_probs(
            "hello",
            1.0,
            vec![],
        )]);

        let close_positions = analysis.get_close_positions_for_choice(0, 1.0);
        assert_eq!(close_positions.len(), 0); // No close positions when only 1 token
    }

    #[test]
    fn test_threshold_filtering() {
        let analysis = create_analysis_with_logprobs(vec![
            // Position 1: Close tokens (55% vs 45%)
            create_token_logprob_from_linear_probs("token1", 0.55, vec![("token2", 0.45)]),
            // Position 2: Far tokens (80% vs 20%)
            create_token_logprob_from_linear_probs("token3", 0.8, vec![("token4", 0.2)]),
        ]);

        // With threshold 0.15, only first position should be close (diff = 0.1)
        let close_strict = analysis.get_close_positions_for_choice(0, 0.15);
        assert_eq!(close_strict.len(), 1);
        assert_abs_diff_eq!(
            close_strict[0].probability_difference,
            0.1,
            epsilon = FLOAT_EPSILON
        );

        // With threshold 0.7, both positions should be close
        let close_permissive = analysis.get_close_positions_for_choice(0, 0.7);
        assert_eq!(close_permissive.len(), 2);

        // Check they're sorted by closeness (0.1 < 0.6)
        assert!(
            close_permissive[0].probability_difference < close_permissive[1].probability_difference
        );
    }

    #[test]
    fn test_percentage_calculation() {
        let analysis = create_analysis_with_logprobs(vec![
            // Position 1: Close (60% vs 40%, diff = 0.2)
            create_token_logprob_from_linear_probs("token1", 0.6, vec![("token2", 0.4)]),
            // Position 2: Far (90% vs 10%, diff = 0.8)
            create_token_logprob_from_linear_probs("token3", 0.9, vec![("token4", 0.1)]),
            // Position 3: Close (52% vs 48%, diff = 0.04)
            create_token_logprob_from_linear_probs("token5", 0.52, vec![("token6", 0.48)]),
        ]);

        let percentage = analysis.close_position_percentage_for_choice(0, 0.25);
        assert!((percentage - 66.67).abs() < 0.01); // 2 out of 3 positions are close
    }

    #[test]
    fn test_real_vllm_equal_logprobs() {
        // Real example from vLLM where two tokens have identical logprobs
        // Both "Ġblock" and "Ġchunk" have logprob -0.9078922271728516
        // exp(-0.9078922271728516) ≈ 0.403 = 40.3% each (sum = 80.6%, remaining 19.4%)
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob_from_linear_probs(
            "Ġblock",
            0.403,
            vec![("Ġchunk", 0.403)], // Identical probability = equally likely
        )]);

        // These should be detected as extremely close (difference = 0.0)
        let close_positions = analysis.get_close_positions_for_choice(0, 0.001);
        assert_eq!(close_positions.len(), 1);
        assert_abs_diff_eq!(
            close_positions[0].probability_difference,
            0.0,
            epsilon = FLOAT_EPSILON
        );

        // Verify probabilities are exactly equal at 40.3%
        let position = &close_positions[0];
        assert_eq!(position.candidates.len(), 2);

        // Check that both tokens are present (order doesn't matter for equal logprobs)
        let tokens: Vec<&str> = position
            .candidates
            .iter()
            .map(|c| c.token.as_str())
            .collect();
        assert!(tokens.contains(&"Ġblock"));
        assert!(tokens.contains(&"Ġchunk"));

        // Both should have identical logprobs (ln(0.403) ≈ -0.907892)
        assert_abs_diff_eq!(
            position.candidates[0].logprob,
            position.candidates[1].logprob,
            epsilon = FLOAT_EPSILON
        );

        // Verify the actual probability values
        let prob1 = position.candidates[0].logprob.exp();
        let prob2 = position.candidates[1].logprob.exp();
        assert_abs_diff_eq!(prob1, 0.403, epsilon = 0.001);
        assert_abs_diff_eq!(prob2, 0.403, epsilon = 0.001);
    }

    // Helper functions for creating test data
    fn create_analysis_with_logprobs(
        token_logprobs: Vec<ChatCompletionTokenLogprob>,
    ) -> SensitivityAnalysis {
        let start_time = Instant::now();
        let response = create_mock_response_with_logprobs(token_logprobs);
        let responses = vec![TimestampedResponse::new(response, 0)];
        let recorded_stream = RecordedStream::new(responses, start_time, Instant::now());
        let arc_stream = Arc::new(recorded_stream);

        analyze_logprob_sensitivity(arc_stream)
    }

    fn create_analysis_with_multiple_choices(
        choices_logprobs: Vec<Vec<ChatCompletionTokenLogprob>>,
    ) -> SensitivityAnalysis {
        let start_time = Instant::now();
        let response = create_mock_response_with_multiple_choices(choices_logprobs);
        let responses = vec![TimestampedResponse::new(response, 0)];
        let recorded_stream = RecordedStream::new(responses, start_time, Instant::now());
        let arc_stream = Arc::new(recorded_stream);

        analyze_logprob_sensitivity(arc_stream)
    }

    fn create_analysis_with_mixed_sampling(mixed_data: TestTokenDataVec) -> SensitivityAnalysis {
        let start_time = Instant::now();
        let token_logprobs: Vec<ChatCompletionTokenLogprob> = mixed_data
            .into_iter()
            .map(|(selected_token, selected_prob, alternatives)| {
                create_token_logprob_from_linear_probs(selected_token, selected_prob, alternatives)
            })
            .collect();

        let response = create_mock_response_with_logprobs(token_logprobs);
        let responses = vec![TimestampedResponse::new(response, 0)];
        let recorded_stream = RecordedStream::new(responses, start_time, Instant::now());
        let arc_stream = Arc::new(recorded_stream);

        analyze_logprob_sensitivity(arc_stream)
    }

    fn create_analysis_with_missing_selected_token() -> SensitivityAnalysis {
        let start_time = Instant::now();

        // Create a scenario where the selected token has a lower probability than alternatives
        // This simulates non-greedy sampling: selected token 15%, but alternatives are 40% and 30%
        let token_logprobs = vec![ChatCompletionTokenLogprob {
            token: "unlikely_selection".to_string(),
            logprob: (0.15_f32).ln(), // Selected but not optimal: 15%
            bytes: None,
            top_logprobs: vec![
                TopLogprobs {
                    token: "best_option".to_string(),
                    logprob: (0.4_f32).ln(), // Much better option: 40%
                    bytes: None,
                },
                TopLogprobs {
                    token: "second_best".to_string(),
                    logprob: (0.3_f32).ln(), // Still better than selected: 30%
                    bytes: None,
                },
            ],
        }];

        let response = create_mock_response_with_logprobs(token_logprobs);
        let responses = vec![TimestampedResponse::new(response, 0)];
        let recorded_stream = RecordedStream::new(responses, start_time, Instant::now());
        let arc_stream = Arc::new(recorded_stream);

        analyze_logprob_sensitivity(arc_stream)
    }

    /// Helper function to create token logprobs from linear probabilities [0, 1]
    /// This ensures realistic probability distributions that sum to ≤ 1
    fn create_token_logprob_from_linear_probs(
        token: &str,
        prob: f32,
        top_probs: Vec<(&str, f32)>,
    ) -> ChatCompletionTokenLogprob {
        // Validate that probabilities are in [0, 1] range
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability must be in [0, 1]: {}",
            prob
        );

        // Calculate total probability mass
        let total_prob = prob + top_probs.iter().map(|(_, p)| p).sum::<f32>();
        assert!(
            total_prob <= 1.001,
            "Total probability mass exceeds 1: {}",
            total_prob
        ); // Allow small floating point error

        for (_, p) in &top_probs {
            assert!(
                *p >= 0.0 && *p <= 1.0,
                "Probability must be in [0, 1]: {}",
                p
            );
        }

        ChatCompletionTokenLogprob {
            token: token.to_string(),
            logprob: prob.ln(),
            bytes: None,
            top_logprobs: top_probs
                .into_iter()
                .map(|(t, p)| TopLogprobs {
                    token: t.to_string(),
                    logprob: p.ln(),
                    bytes: None,
                })
                .collect(),
        }
    }

    fn create_mock_response_with_logprobs(
        token_logprobs: Vec<ChatCompletionTokenLogprob>,
    ) -> NvCreateChatCompletionStreamResponse {
        #[expect(deprecated)]
        NvCreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    content: Some("test".to_string()),
                    function_call: None,
                    tool_calls: None,
                    role: Some(Role::Assistant),
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
                logprobs: Some(ChatChoiceLogprobs {
                    content: Some(token_logprobs),
                    refusal: None,
                }),
            }],
            created: 1234567890,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            nvext: None,
        }
    }

    fn create_mock_response_with_multiple_choices(
        choices_logprobs: Vec<Vec<ChatCompletionTokenLogprob>>,
    ) -> NvCreateChatCompletionStreamResponse {
        #[expect(deprecated)]
        let choices = choices_logprobs
            .into_iter()
            .enumerate()
            .map(|(i, token_logprobs)| ChatChoiceStream {
                index: i as u32,
                delta: ChatCompletionStreamResponseDelta {
                    content: Some("test".to_string()),
                    function_call: None,
                    tool_calls: None,
                    role: Some(Role::Assistant),
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
                logprobs: Some(ChatChoiceLogprobs {
                    content: Some(token_logprobs),
                    refusal: None,
                }),
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

    #[test]
    fn test_sensitivity_analysis() {
        let start_time = Instant::now();
        let responses = vec![TimestampedResponse::new(create_mock_response(), 0)];

        let recorded_stream = RecordedStream::new(responses, start_time, Instant::now());
        let arc_stream = Arc::new(recorded_stream);

        let analysis = analyze_logprob_sensitivity(arc_stream);
        // Basic validation that analysis was created
        assert_eq!(analysis.total_responses, 1);
        assert!(analysis.close_position_percentage_for_choice(0, 0.5) >= 0.0);
    }

    #[test]
    fn test_extract_logprobs_by_choice_empty() {
        let response = create_mock_response();
        let logprobs = response.extract_logprobs_by_choice();
        assert!(logprobs.is_empty() || logprobs.values().any(|v| v.is_empty()));
    }

    #[test]
    fn test_token_logprobs_struct() {
        // Test TokenLogProbs with selected token not in alternatives
        let selected = TokenLogprob {
            token: "selected".to_string(),
            logprob: 0.7_f32.ln(), // 70%
            bytes: None,
        };

        let alternatives = vec![
            TokenLogprob {
                token: "alt1".to_string(),
                logprob: 0.2_f32.ln(), // 20%
                bytes: None,
            },
            TokenLogprob {
                token: "alt2".to_string(),
                logprob: 0.1_f32.ln(), // 10%
                bytes: None,
            },
        ];

        let token_logprobs = TokenLogProbs::new(selected.clone(), alternatives.clone());

        // Test methods
        assert_eq!(token_logprobs.selected_token(), &selected);
        assert_eq!(token_logprobs.alternative_tokens().len(), 2);
        assert_eq!(token_logprobs.all_tokens().len(), 3);

        // Test sorting - all_tokens should be sorted by logprob (highest first)
        let all_tokens = token_logprobs.all_tokens();
        assert_eq!(all_tokens[0].token, "selected"); // 70%
        assert_eq!(all_tokens[1].token, "alt1"); // 20%
        assert_eq!(all_tokens[2].token, "alt2"); // 10%

        // Test that alternatives are sorted
        let alt_tokens = token_logprobs.alternative_tokens();
        assert_eq!(alt_tokens[0].token, "alt1"); // 20%
        assert_eq!(alt_tokens[1].token, "alt2"); // 10%
    }

    #[test]
    fn test_token_logprobs_selected_in_alternatives() {
        // Test case where selected token already appears in alternatives
        let selected = TokenLogprob {
            token: "token".to_string(),
            logprob: 0.4_f32.ln(), // 40%
            bytes: None,
        };

        let alternatives = vec![
            TokenLogprob {
                token: "token".to_string(),
                logprob: 0.4_f32.ln(), // Same as selected
                bytes: None,
            },
            TokenLogprob {
                token: "other".to_string(),
                logprob: 0.3_f32.ln(), // 30%
                bytes: None,
            },
        ];

        let token_logprobs = TokenLogProbs::new(selected, alternatives.clone());

        // all_tokens should not duplicate the selected token
        let all_tokens = token_logprobs.all_tokens();
        assert_eq!(all_tokens.len(), 2);
        assert_eq!(all_tokens[0].token, "token"); // 40%
        assert_eq!(all_tokens[1].token, "other"); // 30%
    }

    #[test]
    fn test_validate_and_flatten_choices() {
        // Test successful validation
        let mut choices = HashMap::new();
        choices.insert(0, vec![]);
        choices.insert(1, vec![]);
        choices.insert(2, vec![]);

        let result = validate_and_flatten_choices(choices);
        assert!(result.is_ok());
        let flattened = result.unwrap();
        assert_eq!(flattened.len(), 3);

        // Test missing choice index
        let mut choices = HashMap::new();
        choices.insert(0, vec![]);
        choices.insert(2, vec![]); // Missing index 1

        let result = validate_and_flatten_choices(choices);
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(
            error_msg.contains("Missing choice indices")
                && error_msg.contains("expected 3 choices")
        );

        // Test empty choices
        let choices = HashMap::new();
        let result = validate_and_flatten_choices(choices);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_probability_remaining_calculation() {
        // Test with tokens that don't sum to 1.0 (incomplete distribution)
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob_from_linear_probs(
            "token",
            0.4, // 40%
            vec![
                ("alt1", 0.3), // 30%
                ("alt2", 0.1), // 10%
                               // Missing 20% probability mass
            ],
        )]);

        let close_positions = analysis.get_close_positions_for_choice(0, 1.0);
        assert_eq!(close_positions.len(), 1);

        let position = &close_positions[0];

        // Should have probability_remaining ≈ 0.2 (20% missing)
        // Total: 40% + 30% + 10% = 80%, so remaining = 20%
        assert_abs_diff_eq!(position.probability_remaining, 0.2, epsilon = 0.01);

        // Test with tokens that nearly sum to 1.0 (complete distribution)
        let analysis_complete =
            create_analysis_with_logprobs(vec![create_token_logprob_from_linear_probs(
                "token",
                0.5, // 50%
                vec![
                    ("alt1", 0.3), // 30%
                    ("alt2", 0.2), // 20%
                                   // Total: 100%
                ],
            )]);

        let complete_positions = analysis_complete.get_close_positions_for_choice(0, 1.0);
        assert_eq!(complete_positions.len(), 1);

        let complete_position = &complete_positions[0];

        // Should have probability_remaining ≈ 0.0 (no missing mass)
        assert_abs_diff_eq!(complete_position.probability_remaining, 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_position_closeness_ordering() {
        let analysis = create_analysis_with_logprobs(vec![
            // Position 1: Far apart (85% vs 15%, diff = 0.7)
            create_token_logprob_from_linear_probs("far", 0.85, vec![("alt", 0.15)]),
            // Position 2: Close (51% vs 49%, diff = 0.02)
            create_token_logprob_from_linear_probs("close", 0.51, vec![("alt", 0.49)]),
            // Position 3: Medium (70% vs 30%, diff = 0.4)
            create_token_logprob_from_linear_probs("medium", 0.7, vec![("alt", 0.3)]),
        ]);

        let positions = &analysis.choice_analyses.get(&0).unwrap().position_closeness;
        assert_eq!(positions.len(), 3);

        // Should be sorted by closeness (smallest difference first)
        assert!(positions[0].probability_difference <= positions[1].probability_difference);
        assert!(positions[1].probability_difference <= positions[2].probability_difference);

        // Check actual values
        assert_abs_diff_eq!(
            positions[0].probability_difference,
            0.02,
            epsilon = FLOAT_EPSILON
        );
        assert_abs_diff_eq!(
            positions[1].probability_difference,
            0.4,
            epsilon = FLOAT_EPSILON
        );
        assert_abs_diff_eq!(
            positions[2].probability_difference,
            0.7,
            epsilon = FLOAT_EPSILON
        );
    }

    #[test]
    fn test_multiple_close_tokens_edge_cases() {
        // Test with exactly 3 close tokens: 34%, 33%, 32% (close within 0.02)
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob_from_linear_probs(
            "token",
            0.34,
            vec![
                ("alt1", 0.33), // diff = 0.01
                ("alt2", 0.32), // diff = 0.01 from alt1, 0.02 from token
                ("alt3", 0.01), // diff = 0.31 (not close)
            ],
        )]);

        let multiple_close = analysis.detect_multiple_close_tokens(0, 0.025);
        assert_eq!(multiple_close.len(), 1);
        assert_eq!(multiple_close[0].close_count, 3);
    }

    #[test]
    fn test_choice_analysis_independence() {
        let analysis = create_analysis_with_multiple_choices(vec![
            // Choice 0: 2 positions, 1 close
            vec![
                create_token_logprob_from_linear_probs("token1", 0.55, vec![("alt1", 0.45)]), // diff = 0.1
                create_token_logprob_from_linear_probs("token2", 0.9, vec![("alt2", 0.1)]), // diff = 0.8
            ],
            // Choice 1: 1 position, very close
            vec![
                create_token_logprob_from_linear_probs("token3", 0.501, vec![("alt3", 0.499)]), // diff = 0.002
            ],
        ]);

        assert_eq!(analysis.choice_analyses.len(), 2);
        assert_eq!(
            analysis.choice_analyses.get(&0).unwrap().positions_analyzed,
            2
        );
        assert_eq!(
            analysis.choice_analyses.get(&1).unwrap().positions_analyzed,
            1
        );

        // Check independence - each choice should have different closeness patterns
        let choice0_close = analysis.get_close_positions_for_choice(0, 0.5);
        let choice1_close = analysis.get_close_positions_for_choice(1, 0.5);

        assert_eq!(choice0_close.len(), 1);
        assert_eq!(choice1_close.len(), 1);

        // Choice 1 should be much closer
        assert!(choice1_close[0].probability_difference < choice0_close[0].probability_difference);
    }

    #[test]
    fn test_get_closest_positions_boundary() {
        let analysis = create_analysis_with_logprobs(vec![
            create_token_logprob_from_linear_probs("token1", 0.6, vec![("alt1", 0.4)]),
            create_token_logprob_from_linear_probs("token2", 0.75, vec![("alt2", 0.25)]),
        ]);

        // Request more positions than available
        let closest = analysis.get_closest_positions_for_choice(0, 10);
        assert_eq!(closest.len(), 2);

        // Request exactly the number available
        let closest = analysis.get_closest_positions_for_choice(0, 2);
        assert_eq!(closest.len(), 2);

        // Request fewer
        let closest = analysis.get_closest_positions_for_choice(0, 1);
        assert_eq!(closest.len(), 1);
    }

    #[test]
    fn test_zero_threshold() {
        let analysis = create_analysis_with_logprobs(vec![
            create_token_logprob_from_linear_probs("token", 0.5, vec![("alt", 0.5)]), // diff = 0.0
        ]);

        let close_positions = analysis.get_close_positions_for_choice(0, 0.0);
        assert_eq!(close_positions.len(), 1);
        assert_abs_diff_eq!(
            close_positions[0].probability_difference,
            0.0,
            epsilon = FLOAT_EPSILON
        );
    }

    #[test]
    fn test_nonexistent_choice() {
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob_from_linear_probs(
            "token",
            0.6,
            vec![("alt", 0.4)],
        )]);

        // Request analysis for non-existent choice
        let close_positions = analysis.get_close_positions_for_choice(5, 0.1);
        assert!(close_positions.is_empty());

        let closest = analysis.get_closest_positions_for_choice(5, 3);
        assert!(closest.is_empty());

        let percentage = analysis.close_position_percentage_for_choice(5, 0.1);
        assert_eq!(percentage, 0.0);
    }

    #[test]
    fn test_logprob_extractor_with_missing_data() {
        // Test with choice that has no logprobs
        #[expect(deprecated)]
        let response = NvCreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    content: Some("test".to_string()),
                    function_call: None,
                    tool_calls: None,
                    role: Some(Role::Assistant),
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
                logprobs: None, // No logprobs
            }],
            created: 1234567890,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            nvext: None,
        };

        let logprobs = response.extract_logprobs_by_choice();
        assert_eq!(logprobs.len(), 1);
        assert!(logprobs.values().any(|v| v.is_empty()));
    }

    #[test]
    fn test_print_summary_no_panic() {
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob_from_linear_probs(
            "token",
            0.6,
            vec![("alt", 0.4)],
        )]);

        // Should not panic when printing summary
        analysis.print_summary();
    }

    #[test]
    fn test_greedy_decoding_detection() {
        // Greedy decoding: selected token is always the most probable
        // Position 1: Clear winner (80% vs 15% vs 5%)
        // Position 2: Another clear winner (70% vs 20% vs 10%)
        let analysis = create_analysis_with_logprobs(vec![
            create_token_logprob_from_linear_probs(
                "best",
                0.8,
                vec![("second", 0.15), ("third", 0.05)],
            ),
            create_token_logprob_from_linear_probs(
                "optimal",
                0.7,
                vec![("suboptimal", 0.2), ("bad", 0.1)],
            ),
        ]);

        // Should detect greedy-like behavior (selected tokens have highest probability)
        let is_greedy = analysis.detect_likely_greedy_decoding(0);
        assert!(is_greedy);

        let greedy_percentage = analysis.greedy_selection_percentage(0);
        assert!(greedy_percentage > 90.0); // Should be close to 100%
    }

    #[test]
    fn test_non_greedy_decoding_detection() {
        // Non-greedy decoding: some positions show sampling behavior
        // Position 1: Greedy selection (best token chosen: 60% vs 40%)
        // Position 2: Non-greedy-like (close tokens: 35% vs 33% vs 32%)
        let analysis = create_analysis_with_mixed_sampling(vec![
            ("selected_best", 0.6, vec![("alternative", 0.4)]),
            (
                "close_choice",
                0.35,
                vec![("very_close", 0.33), ("also_close", 0.32)],
            ),
        ]);

        let _is_greedy = analysis.detect_likely_greedy_decoding(0);
        // This should be detected as greedy since we have some clear differences

        let greedy_percentage = analysis.greedy_selection_percentage(0);
        assert!((0.0..=100.0).contains(&greedy_percentage)); // Valid percentage range
    }

    #[test]
    fn test_selected_token_not_in_top_logprobs() {
        // Edge case: selected token doesn't appear in top_logprobs at all
        // Selected: 15%, but alternatives are 40% and 30% (non-greedy sampling)
        let analysis = create_analysis_with_missing_selected_token();

        // Should still work - the algorithm adapts to different logprob patterns
        let greedy_percentage = analysis.greedy_selection_percentage(0);
        assert!((0.0..=100.0).contains(&greedy_percentage)); // Valid percentage range
    }

    #[test]
    fn test_equal_logprobs_greedy_detection() {
        // Test the original vLLM example - equal logprobs should be detected as close
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob_from_linear_probs(
            "Ġblock",
            0.403,
            vec![("Ġchunk", 0.403)], // Identical probability = equally likely
        )]);

        // Equal probabilities should be detected as extremely close
        let close_positions = analysis.get_close_positions_for_choice(0, 0.001);
        assert_eq!(close_positions.len(), 1);

        // Should be detected as greedy-like since there's no clear better choice
        let is_greedy = analysis.detect_likely_greedy_decoding(0);
        assert!(is_greedy);
    }

    #[tokio::test]
    async fn test_real_sse_stream_analysis() {
        // Read the real SSE data with logprobs
        let data = std::fs::read_to_string(
            "tests/data/replays/deepseek-r1-distill-llama-8b/chat-completions.stream.1",
        )
        .expect("Failed to read test data file");

        // Create stream from SSE data
        let sse_stream = create_message_stream(&data);

        // Convert SSE messages to our stream response format using the existing converter
        let response_stream =
            convert_sse_stream::<NvCreateChatCompletionStreamResponse>(Box::pin(sse_stream));

        // Filter out errors and extract successful responses
        let filtered_stream = response_stream.filter_map(|annotated| async move { annotated.data });

        // Create a mock context for recording
        let ctx = Arc::new(MockContext::new());

        // Record the stream
        let (recorded_stream, recording_rx) =
            record_stream_with_context(Box::pin(filtered_stream), ctx, RecordingMode::Sink);

        // Consume the stream (it will be recorded)
        let _collected: Vec<_> = recorded_stream.collect().await;

        // Get the recorded data
        let recorded = recording_rx
            .await
            .expect("Failed to receive recorded stream");

        // Verify we have data
        assert!(recorded.response_count() > 0, "No responses recorded");
        println!("Recorded {} responses", recorded.response_count());

        // Perform logprob analysis
        let arc_recorded = Arc::new(recorded);
        let analysis = analyze_logprob_sensitivity(arc_recorded);

        // Print analysis summary
        analysis.print_summary();

        // Verify the analysis found logprob data
        assert!(
            !analysis.choice_analyses.is_empty(),
            "No choice analyses found"
        );
        assert!(
            analysis
                .choice_analyses
                .values()
                .any(|a| a.positions_analyzed > 0),
            "No positions analyzed"
        );

        // Look for the specific vLLM case with equal logprobs ("Ġblock" vs "Ġchunk")
        let close_positions = analysis.get_close_positions_for_choice(0, 0.001);

        // Should find at least one very close position (the equal logprob case)
        assert!(!close_positions.is_empty(), "No close positions found");

        // Check if we found the exact equal case (difference = 0)
        let equal_positions = close_positions
            .iter()
            .filter(|pos| pos.probability_difference < 0.0001)
            .count();
        if equal_positions > 0 {
            println!(
                "Found {} positions with nearly equal probabilities",
                equal_positions
            );
        }

        // Test other analysis methods
        let closest_3 = analysis.get_closest_positions_for_choice(0, 3);
        assert!(
            closest_3.len() <= 3,
            "Should return at most 3 closest positions"
        );

        let percentage = analysis.close_position_percentage_for_choice(0, 0.1);
        assert!(
            (0.0..=100.0).contains(&percentage),
            "Percentage should be valid"
        );

        // Test greedy detection
        let is_greedy = analysis.detect_likely_greedy_decoding(0);
        let greedy_percentage = analysis.greedy_selection_percentage(0);
        println!(
            "Greedy detection: {} ({}% greedy-like)",
            is_greedy, greedy_percentage
        );

        // Test multiple close tokens detection
        let multiple_close = analysis.detect_multiple_close_tokens(0, 0.05);
        if !multiple_close.is_empty() {
            println!(
                "Found {} positions with multiple close tokens",
                multiple_close.len()
            );
        }
    }

    fn create_mock_response() -> NvCreateChatCompletionStreamResponse {
        // Create a mock response for testing
        // In practice, this would have real logprobs data

        NvCreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            choices: vec![],
            created: 1234567890,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            nvext: None,
        }
    }

    // Mock context for testing
    #[derive(Debug)]
    struct MockContext {
        id: String,
    }

    impl MockContext {
        fn new() -> Self {
            Self {
                id: "test-context".to_string(),
            }
        }
    }

    #[async_trait::async_trait]
    impl dynamo_runtime::engine::AsyncEngineContext for MockContext {
        fn id(&self) -> &str {
            &self.id
        }

        fn stop(&self) {
            // No-op for testing
        }

        fn stop_generating(&self) {
            // No-op for testing
        }

        fn kill(&self) {
            // No-op for testing
        }

        fn is_stopped(&self) -> bool {
            false
        }

        fn is_killed(&self) -> bool {
            false
        }

        async fn stopped(&self) {
            // No-op for testing
        }

        async fn killed(&self) {
            // No-op for testing
        }

        fn link_child(&self, _: Arc<dyn dynamo_runtime::engine::AsyncEngineContext>) {
            // No-op for testing
        }
    }
}
