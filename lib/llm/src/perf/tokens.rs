// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Module for counting tokens in a response stream and computing performance metrics.
//!
//! This module provides comprehensive token counting and analysis capabilities for streaming
//! chat completion responses. It supports multiple data sources for token counting with
//! different levels of accuracy and includes validation for completeness and consistency.

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::perf::RecordedStream;
use crate::protocols::TokenIdType;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use crate::tokenizers::Encoding;
use crate::tokenizers::traits::TokenCounter;

pub type TokenCount = usize;
pub type ForwardPassDuration = Duration;

/// Data source used for counting tokens in a specific response chunk.
/// Different sources have different accuracy levels and availability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokenDataSource {
    /// OpenAI's usage field
    /// If this is the data source, then every response chunk had a usage field,
    /// and that usage field was self-consistent between chunks.
    ///
    /// This is the preferred data source for counting tokens.
    Usage(TokenCount),

    /// If [`TokenDataSource::Usage`] is not available, but the tokenizer for the model
    /// is available, then we can directly decode the text per chunk to get the token count
    /// per chunk.
    ///
    /// Errors could result if the supplied tokenizer is not compatible with the model.
    Tokenizer(TokenCount),

    /// If neither [`TokenDataSource::Usage`] nor [`TokenDataSource::Tokenizer`] are available,
    /// then the presence of text in the content of the chunk is the signal for tokens being
    /// emitted; however, the number of tokens is not guaranteed to be accurate.
    ///
    /// In this scenario, we assume that each response chunk is a single token; however, this
    /// easily breaks down if the model performs some speculative decoding or batches multiple
    /// tokens into a single response chunk to reduce messaging overheads.
    SingleTokenApproximation,

    /// If this is provided, then we have a custom server-side mechanism for both timing and
    /// counting tokens.
    ///
    /// If this is provided, then we expect one entry per forward pass. This provides the most
    /// accurate timing and token count information as it directly breaks down the time by
    /// forward pass which disambiguates the time for multiple tokens (Usage) and potential
    /// response batching, e.g. [`TokenDataSource::Usage`] or [`TokenDataSource::Tokenizer`]
    /// might tell us that we received 10 tokens, but that could have been from 10 forward passes
    /// batched together or 1 forward pass that emitted 10 tokens via speculative decoding.
    ///
    /// Dynamo Annotations can provide this level of timing details; however, not all frameworks
    /// are fully integrated with Dynamo Annotations.
    ServerSideTiming(TokenCount, ForwardPassDuration),
}

impl TokenDataSource {
    /// Extract the token count from any data source variant
    pub fn token_count(&self) -> TokenCount {
        match self {
            TokenDataSource::Usage(count) => *count,
            TokenDataSource::Tokenizer(count) => *count,
            TokenDataSource::SingleTokenApproximation => 1,
            TokenDataSource::ServerSideTiming(count, _) => *count,
        }
    }

    /// Returns true if this data source is considered accurate
    pub fn is_accurate(&self) -> bool {
        match self {
            TokenDataSource::Usage(_) | TokenDataSource::ServerSideTiming(_, _) => true,
            TokenDataSource::Tokenizer(_) => true, // Assumed accurate if tokenizer matches model
            TokenDataSource::SingleTokenApproximation => false,
        }
    }

    /// Returns a human-readable description of the data source
    pub fn description(&self) -> &'static str {
        match self {
            TokenDataSource::Usage(_) => "OpenAI Usage Field",
            TokenDataSource::Tokenizer(_) => "Tokenizer-based Counting",
            TokenDataSource::SingleTokenApproximation => "Single Token Approximation",
            TokenDataSource::ServerSideTiming(_, _) => "Server-side Timing",
        }
    }
}

/// Token data for a specific choice in a response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceTokenData {
    /// The choice index
    pub choice_index: u32,
    /// Number of tokens in this chunk for this choice
    pub token_count: TokenCount,
    /// Token IDs generated in this chunk (if available)
    pub token_ids: Option<Vec<TokenIdType>>,
    /// Data source used for counting
    pub data_source: TokenDataSource,
    /// Whether this choice has finished (has a finish_reason)
    pub is_finished: bool,
    /// The finish reason if the choice is finished
    pub finish_reason: Option<String>,
    /// Text content that contributed to the token count (for debugging)
    pub content: Option<String>,
}

/// Trait for extracting token information from various response types
pub trait TokenExtractor {
    /// Extract token data organized by choice index
    /// Returns: HashMap<choice_index, ChoiceTokenData>
    fn extract_tokens_by_choice(
        &self,
        tokenizer: Option<&dyn TokenCounter>,
    ) -> HashMap<u32, ChoiceTokenData>;
}

/// Implementation for NvCreateChatCompletionStreamResponse (our main streaming response type)
impl TokenExtractor for NvCreateChatCompletionStreamResponse {
    fn extract_tokens_by_choice(
        &self,
        tokenizer: Option<&dyn TokenCounter>,
    ) -> HashMap<u32, ChoiceTokenData> {
        let mut result = HashMap::new();

        for choice in &self.choices {
            let choice_index = choice.index;
            let content = choice.delta.content.as_deref().unwrap_or("");
            let finish_reason = choice.finish_reason.as_ref().map(|fr| format!("{:?}", fr));
            let is_finished = choice.finish_reason.is_some();

            // Determine token count, token IDs, and data source
            let (token_count, token_ids, data_source) = if let Some(tokenizer) = tokenizer {
                // Use tokenizer if available and there's content
                if !content.is_empty() {
                    match tokenizer.count_tokens_with_ids(content) {
                        Ok((ids, count)) => (count, Some(ids), TokenDataSource::Tokenizer(count)),
                        Err(_) => {
                            // Fallback to single token approximation if tokenizer fails
                            (1, None, TokenDataSource::SingleTokenApproximation)
                        }
                    }
                } else {
                    (0, Some(Vec::new()), TokenDataSource::Tokenizer(0))
                }
            } else {
                // Use single token approximation
                if !content.is_empty() {
                    (1, None, TokenDataSource::SingleTokenApproximation)
                } else {
                    (0, None, TokenDataSource::SingleTokenApproximation)
                }
            };

            result.insert(
                choice_index,
                ChoiceTokenData {
                    choice_index,
                    token_count,
                    token_ids,
                    data_source,
                    is_finished,
                    finish_reason,
                    content: if content.is_empty() {
                        None
                    } else {
                        Some(content.to_string())
                    },
                },
            );
        }

        result
    }
}

/// Event representing token generation at a specific point in the stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEvent {
    /// Position in the stream (response index)
    pub stream_position: usize,
    /// Number of tokens in this chunk
    pub tokens_in_chunk: TokenCount,
    /// Cumulative tokens up to this point
    pub cumulative_tokens: TokenCount,
    /// Data source used for this chunk
    pub data_source: TokenDataSource,
    /// Content that generated these tokens (for debugging)
    pub content: Option<String>,
}

/// Validation results comparing chunk-by-chunk tokenization vs full text tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationValidation {
    /// Total tokens from chunk-by-chunk tokenization
    pub chunk_total_tokens: TokenCount,
    /// Total tokens from full text tokenization
    pub full_text_total_tokens: TokenCount,
    /// Whether the token counts match
    pub counts_match: bool,
    /// Detailed encoding information (when available)
    pub encoding_details: Option<EncodingDetails>,
    /// Any validation errors found (structured as JSON for detailed error information)
    pub validation_errors: Vec<serde_json::Value>,
}

/// Detailed encoding information from the tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingDetails {
    /// Token IDs from full text tokenization
    pub token_ids: Vec<crate::protocols::TokenIdType>,
    /// Token strings from full text tokenization
    pub token_strings: Vec<String>,
    /// Token offsets from full text tokenization
    pub token_offsets: Vec<(usize, usize)>,
    /// Whether special tokens were skipped
    pub skipped_special_tokens: bool,
}

/// Analysis for a single choice across the entire stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceTokenAnalysis {
    /// Choice index
    pub choice_index: u32,
    /// Total tokens generated for this choice
    pub total_tokens: TokenCount,
    /// Primary data source used (most common in the stream)
    pub primary_data_source: TokenDataSource,
    /// Whether this choice completed with a finish_reason
    pub is_complete: bool,
    /// The finish reason if completed
    pub finish_reason: Option<String>,
    /// Timeline of token generation events
    pub token_timeline: Vec<TokenEvent>,
    /// Number of responses where this choice appeared
    pub responses_with_choice: usize,
    /// Position where the choice finished (if it finished)
    pub finish_position: Option<usize>,
    /// Full concatenated content for this choice
    pub concatenated_content: String,
    /// BLAKE3 hash of the concatenated content
    pub content_blake3_hash: String,
    /// 12-character shortened hash
    pub content_hash_short: String,
    /// Validation results comparing chunk vs full text tokenization
    pub full_tokenization_validation: Option<TokenizationValidation>,
}

/// Comprehensive token analysis for all choices in a recorded stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAnalysis {
    /// Total number of responses analyzed
    pub total_responses: usize,
    /// Analysis results per choice index
    pub choice_analyses: HashMap<u32, ChoiceTokenAnalysis>,
    /// Validation errors found during analysis
    pub validation_errors: Vec<String>,
    /// Whether the stream appears to be complete (all choices finished)
    pub is_stream_complete: bool,
}

/// Validates and analyzes token counting from a recorded stream with enhanced validation
pub fn analyze_token_counting<T: TokenExtractor>(
    recorded_stream: Arc<RecordedStream<T>>,
    tokenizer: Option<&dyn TokenCounter>,
) -> TokenAnalysis {
    let mut choice_analyses: HashMap<u32, ChoiceTokenAnalysis> = HashMap::new();
    let mut validation_errors = Vec::new();

    // Track which choices have finished and when
    let mut finished_choices: HashMap<u32, usize> = HashMap::new();

    // Collect content for each choice for full-text validation
    let mut choice_content: HashMap<u32, Vec<String>> = HashMap::new();

    for (stream_pos, timestamped_response) in recorded_stream.responses().iter().enumerate() {
        let response = &timestamped_response.response;
        let tokens_by_choice = response.extract_tokens_by_choice(tokenizer);

        for (choice_index, choice_token_data) in tokens_by_choice {
            // Collect content for this choice
            if let Some(content) = &choice_token_data.content {
                choice_content
                    .entry(choice_index)
                    .or_default()
                    .push(content.clone());
            }

            // Get or create choice analysis
            let choice_analysis =
                choice_analyses
                    .entry(choice_index)
                    .or_insert_with(|| ChoiceTokenAnalysis {
                        choice_index,
                        total_tokens: 0,
                        primary_data_source: TokenDataSource::SingleTokenApproximation,
                        is_complete: false,
                        finish_reason: None,
                        token_timeline: Vec::new(),
                        responses_with_choice: 0,
                        finish_position: None,
                        concatenated_content: String::new(),
                        content_blake3_hash: String::new(),
                        content_hash_short: String::new(),
                        full_tokenization_validation: None,
                    });

            choice_analysis.responses_with_choice += 1;

            // Check if choice was already finished
            if let Some(finish_pos) = finished_choices.get(&choice_index)
                && choice_token_data.token_count > 0
            {
                validation_errors.push(format!(
                    "Choice {} generated {} tokens at position {} after finishing at position {}",
                    choice_index, choice_token_data.token_count, stream_pos, finish_pos
                ));
            }

            // Update total tokens
            choice_analysis.total_tokens += choice_token_data.token_count;

            // Record token event
            choice_analysis.token_timeline.push(TokenEvent {
                stream_position: stream_pos,
                tokens_in_chunk: choice_token_data.token_count,
                cumulative_tokens: choice_analysis.total_tokens,
                data_source: choice_token_data.data_source.clone(),
                content: choice_token_data.content,
            });

            // Update finish status
            if choice_token_data.is_finished {
                choice_analysis.is_complete = true;
                choice_analysis.finish_reason = choice_token_data.finish_reason;
                choice_analysis.finish_position = Some(stream_pos);
                finished_choices.insert(choice_index, stream_pos);
            }

            // Update primary data source (use the most accurate one seen)
            if choice_token_data.data_source.is_accurate()
                && !choice_analysis.primary_data_source.is_accurate()
            {
                choice_analysis.primary_data_source = choice_token_data.data_source;
            }
        }
    }

    // Perform full-text validation for each choice
    for (choice_index, analysis) in choice_analyses.iter_mut() {
        if let Some(content_parts) = choice_content.get(choice_index) {
            // Concatenate all content for this choice
            let concatenated = content_parts.join("");
            analysis.concatenated_content = concatenated.clone();

            // Compute BLAKE3 hash
            let hash = blake3::hash(concatenated.as_bytes());
            let hash_hex = hash.to_hex();
            analysis.content_blake3_hash = hash_hex.to_string();
            analysis.content_hash_short = hash_hex.chars().take(12).collect();

            // Perform full-text tokenization validation if tokenizer is available
            if let Some(tokenizer) = tokenizer {
                analysis.full_tokenization_validation = Some(validate_full_text_tokenization(
                    &concatenated,
                    analysis.total_tokens,
                    &analysis.token_timeline,
                    tokenizer,
                ));
            }
        }
    }

    // Validate that all choices are complete
    let incomplete_choices: Vec<u32> = choice_analyses
        .values()
        .filter(|analysis| !analysis.is_complete)
        .map(|analysis| analysis.choice_index)
        .collect();

    let is_stream_complete = incomplete_choices.is_empty();

    if !incomplete_choices.is_empty() {
        validation_errors.push(format!(
            "Stream incomplete: choices {:?} did not finish with a finish_reason",
            incomplete_choices
        ));
    }

    TokenAnalysis {
        total_responses: recorded_stream.responses().len(),
        choice_analyses,
        validation_errors,
        is_stream_complete,
    }
}

/// Validates that individual chunks align with the corresponding portions of full text tokenization
/// Adds structured error details to validation_errors if alignment fails
fn validate_chunk_alignment(
    full_text: &str,
    full_text_token_ids: &[TokenIdType],
    token_timeline: &[TokenEvent],
    tokenizer: &dyn TokenCounter,
    validation_errors: &mut Vec<serde_json::Value>,
) {
    let mut char_position = 0;
    let mut token_position = 0;
    let mut chunk_alignment_errors = Vec::new();

    // Only process chunks that have content
    let content_chunks: Vec<_> = token_timeline
        .iter()
        .enumerate()
        .filter_map(|(idx, event)| event.content.as_ref().map(|content| (idx, event, content)))
        .collect();

    if content_chunks.is_empty() {
        return; // No chunks to validate
    }

    let total_chunks = content_chunks.len();

    for (chunk_index, event, chunk_content) in content_chunks {
        let char_start = char_position;
        let char_end = char_position + chunk_content.len();

        // Tokenize this chunk in isolation
        let (chunk_token_ids, chunk_token_count) =
            match tokenizer.count_tokens_with_ids(chunk_content) {
                Ok(result) => result,
                Err(e) => {
                    chunk_alignment_errors.push(json!({
                        "type": "tokenization_failed",
                        "chunk_index": chunk_index,
                        "stream_position": event.stream_position,
                        "content": chunk_content,
                        "error": e.to_string()
                    }));
                    (Vec::new(), 0)
                }
            };

        // Verify chunk token count matches what was recorded in the timeline
        if chunk_token_count != event.tokens_in_chunk {
            chunk_alignment_errors.push(json!({
                "type": "chunk_token_count_mismatch",
                "chunk_index": chunk_index,
                "stream_position": event.stream_position,
                "content": chunk_content,
                "recorded_tokens": event.tokens_in_chunk,
                "retokenized_tokens": chunk_token_count
            }));
        }

        // Extract expected tokens from full text tokenization
        let token_start = token_position;
        let token_end = token_position + chunk_token_count;
        let expected_token_ids = if token_end <= full_text_token_ids.len() {
            full_text_token_ids[token_start..token_end].to_vec()
        } else {
            chunk_alignment_errors.push(json!({
                "type": "token_range_exceeded",
                "chunk_index": chunk_index,
                "stream_position": event.stream_position,
                "content": chunk_content,
                "token_range": [token_start, token_end],
                "full_text_length": full_text_token_ids.len()
            }));
            Vec::new()
        };

        // Check if tokens match
        let tokens_match = chunk_token_ids == expected_token_ids;
        if !tokens_match && !chunk_token_ids.is_empty() && !expected_token_ids.is_empty() {
            chunk_alignment_errors.push(json!({
                "type": "token_sequence_mismatch",
                "chunk_index": chunk_index,
                "stream_position": event.stream_position,
                "content": chunk_content,
                "chunk_token_ids": chunk_token_ids,
                "expected_token_ids": expected_token_ids,
                "char_range": [char_start, char_end],
                "token_range": [token_start, token_end]
            }));
        }

        // Update positions for next chunk
        char_position = char_end;
        token_position = token_end;
    }

    // Verify that we've consumed all the text
    if char_position != full_text.len() {
        chunk_alignment_errors.push(json!({
            "type": "character_position_mismatch",
            "processed_chars": char_position,
            "expected_chars": full_text.len()
        }));
    }

    // Verify that we've consumed all the tokens
    if token_position != full_text_token_ids.len() {
        chunk_alignment_errors.push(json!({
            "type": "token_position_mismatch",
            "processed_tokens": token_position,
            "expected_tokens": full_text_token_ids.len()
        }));
    }

    // Only add chunk alignment errors if there are any
    if !chunk_alignment_errors.is_empty() {
        validation_errors.push(json!({
            "type": "chunk_alignment_validation",
            "total_chunks_validated": total_chunks,
            "errors": chunk_alignment_errors
        }));
    }
}

/// Validates full-text tokenization against chunk-by-chunk tokenization with alignment
fn validate_full_text_tokenization(
    full_text: &str,
    chunk_total_tokens: TokenCount,
    token_timeline: &[TokenEvent],
    tokenizer: &dyn TokenCounter,
) -> TokenizationValidation {
    let mut validation_errors = Vec::new();

    // Perform full-text tokenization
    let (full_text_token_ids, full_text_total_tokens) =
        match tokenizer.count_tokens_with_ids(full_text) {
            Ok(result) => result,
            Err(e) => {
                validation_errors.push(json!({
                    "type": "full_text_tokenization_failed",
                    "error": e.to_string()
                }));
                return TokenizationValidation {
                    chunk_total_tokens,
                    full_text_total_tokens: 0,
                    counts_match: false,
                    encoding_details: None,
                    validation_errors,
                };
            }
        };

    // Check if counts match
    let counts_match = chunk_total_tokens == full_text_total_tokens;
    if !counts_match {
        validation_errors.push(json!({
            "type": "token_count_mismatch",
            "chunk_total_tokens": chunk_total_tokens,
            "full_text_total_tokens": full_text_total_tokens
        }));
    }

    // Try to get detailed encoding information if available
    let encoding_details = if let Ok(encoding) = tokenizer.encode_detailed(full_text) {
        extract_encoding_details(&encoding)
    } else {
        None
    };

    // Perform chunk alignment validation (only when we have a tokenizer)
    validate_chunk_alignment(
        full_text,
        &full_text_token_ids,
        token_timeline,
        tokenizer,
        &mut validation_errors,
    );

    TokenizationValidation {
        chunk_total_tokens,
        full_text_total_tokens,
        counts_match,
        encoding_details,
        validation_errors,
    }
}

/// Extracts detailed information from an Encoding object
fn extract_encoding_details(encoding: &Encoding) -> Option<EncodingDetails> {
    match encoding {
        Encoding::Hf(hf_encoding) => {
            Some(EncodingDetails {
                token_ids: hf_encoding.get_ids().to_vec(),
                token_strings: hf_encoding.get_tokens().to_vec(),
                token_offsets: hf_encoding.get_offsets().to_vec(),
                skipped_special_tokens: false, // TODO: Track this properly
            })
        }
        Encoding::Sp(token_ids) => {
            Some(EncodingDetails {
                token_ids: token_ids.clone(),
                token_strings: vec![], // SentencePiece doesn't provide token strings in our wrapper
                token_offsets: vec![], // SentencePiece doesn't provide offsets in our wrapper
                skipped_special_tokens: false, // TODO: Track this properly
            })
        }
    }
}

impl TokenAnalysis {
    /// Get total tokens for a specific choice
    pub fn total_tokens_for_choice(&self, choice_index: u32) -> Option<TokenCount> {
        self.choice_analyses
            .get(&choice_index)
            .map(|analysis| analysis.total_tokens)
    }

    /// Get all choice indices present in the analysis
    pub fn choice_indices(&self) -> Vec<u32> {
        let mut indices: Vec<u32> = self.choice_analyses.keys().copied().collect();
        indices.sort();
        indices
    }

    /// Check if the analysis has any validation errors
    pub fn has_errors(&self) -> bool {
        !self.validation_errors.is_empty()
    }

    /// Get a summary of data sources used across all choices
    pub fn data_source_summary(&self) -> HashMap<String, usize> {
        let mut summary = HashMap::new();
        for analysis in self.choice_analyses.values() {
            let source_name = analysis.primary_data_source.description();
            *summary.entry(source_name.to_string()).or_insert(0) += 1;
        }
        summary
    }

    /// Print a comprehensive summary of the token analysis
    pub fn print_summary(&self) {
        println!("=== Token Analysis Summary ===");
        println!("Total stream responses: {}", self.total_responses);
        println!("Number of choices: {}", self.choice_analyses.len());
        println!("Stream complete: {}", self.is_stream_complete);

        if self.has_errors() {
            println!("\n⚠️  Validation Errors:");
            for error in &self.validation_errors {
                println!("  - {}", error);
            }
        }

        println!("\nData Source Summary:");
        for (source, count) in self.data_source_summary() {
            println!("  {}: {} choices", source, count);
        }

        println!("\nPer-Choice Analysis:");
        for choice_index in self.choice_indices() {
            let analysis = &self.choice_analyses[&choice_index];
            let status = if analysis.is_complete {
                format!(
                    "✓ Complete ({})",
                    analysis.finish_reason.as_deref().unwrap_or("unknown")
                )
            } else {
                "❌ Incomplete".to_string()
            };

            println!(
                "  Choice {}: {} tokens, {} - {}",
                choice_index,
                analysis.total_tokens,
                analysis.primary_data_source.description(),
                status
            );

            // Print content hash information
            if !analysis.concatenated_content.is_empty() {
                println!(
                    "    Content: {} chars, hash: {} ({})",
                    analysis.concatenated_content.len(),
                    analysis.content_hash_short,
                    analysis.content_blake3_hash
                );
            }

            // Print tokenization validation results
            if let Some(validation) = &analysis.full_tokenization_validation {
                let validation_status = if validation.counts_match {
                    "✓ PASS"
                } else {
                    "❌ FAIL"
                };
                println!(
                    "    Tokenization validation: {} (chunk: {}, full: {})",
                    validation_status,
                    validation.chunk_total_tokens,
                    validation.full_text_total_tokens
                );

                if !validation.validation_errors.is_empty() {
                    for error in &validation.validation_errors {
                        match error.get("type").and_then(|v| v.as_str()) {
                            Some("chunk_alignment_validation") => {
                                let total_chunks = error
                                    .get("total_chunks_validated")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                let errors = error
                                    .get("errors")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| arr.len())
                                    .unwrap_or(0);

                                println!(
                                    "      Chunk alignment: ❌ MISALIGNED ({} chunks validated, {} errors)",
                                    total_chunks, errors
                                );
                            }
                            Some("token_count_mismatch") => {
                                let chunk_tokens = error
                                    .get("chunk_total_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                let full_tokens = error
                                    .get("full_text_total_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                println!(
                                    "      ⚠️  Token count mismatch: chunk-by-chunk={}, full-text={}",
                                    chunk_tokens, full_tokens
                                );
                            }
                            Some("full_text_tokenization_failed") => {
                                let err_msg = error
                                    .get("error")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown error");
                                println!("      ⚠️  Failed to tokenize full text: {}", err_msg);
                            }
                            _ => {
                                println!("      ⚠️  {}", error);
                            }
                        }
                    }
                } else {
                    // If no validation errors, check if we had chunks to validate
                    // (this indicates successful chunk alignment)
                    if let Some(analysis) = &self.choice_analyses.get(&choice_index) {
                        let content_chunks = analysis
                            .token_timeline
                            .iter()
                            .filter(|event| event.content.is_some())
                            .count();
                        if content_chunks > 0 {
                            println!(
                                "      Chunk alignment: ✓ ALL ALIGNED ({} chunks validated)",
                                content_chunks
                            );
                        }
                    }
                }

                if let Some(details) = &validation.encoding_details {
                    println!(
                        "      Encoding: {} token IDs, {} strings, {} offsets",
                        details.token_ids.len(),
                        details.token_strings.len(),
                        details.token_offsets.len()
                    );
                }
            }
        }
    }

    /// Get token generation rate (tokens per response that generated tokens) for a choice
    /// This excludes responses with 0 tokens from the calculation
    pub fn token_rate_for_choice(&self, choice_index: u32) -> Option<f64> {
        self.choice_analyses.get(&choice_index).map(|analysis| {
            // Count only responses that actually generated tokens (> 0)
            let responses_with_tokens = analysis
                .token_timeline
                .iter()
                .filter(|event| event.tokens_in_chunk > 0)
                .count();

            if responses_with_tokens > 0 {
                analysis.total_tokens as f64 / responses_with_tokens as f64
            } else {
                0.0
            }
        })
    }

    /// Check for consistency in data sources across the stream
    pub fn validate_data_source_consistency(&self) -> Vec<String> {
        let mut errors = Vec::new();

        for analysis in self.choice_analyses.values() {
            let mut sources: HashMap<String, usize> = HashMap::new();
            for event in &analysis.token_timeline {
                let source_name = event.data_source.description();
                *sources.entry(source_name.to_string()).or_insert(0) += 1;
            }

            if sources.len() > 1 {
                errors.push(format!(
                    "Choice {} used multiple data sources: {:?}",
                    analysis.choice_index, sources
                ));
            }
        }

        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::{RecordedStream, TimestampedResponse};
    use anyhow::Result;
    use dynamo_async_openai::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, FinishReason, Role,
    };
    use std::time::Instant;

    // Mock tokenizer for testing
    struct MockTokenizer;

    impl TokenCounter for MockTokenizer {
        fn count_tokens(&self, text: &str) -> crate::tokenizers::Result<TokenCount> {
            // Simple mock: count words as tokens
            Ok(text.split_whitespace().count())
        }

        fn count_tokens_with_ids(
            &self,
            text: &str,
        ) -> crate::tokenizers::Result<(Vec<TokenIdType>, usize)> {
            // More realistic mock: assign consistent token IDs based on word hash
            // This ensures the same word gets the same token ID regardless of context
            let words: Vec<&str> = text.split_whitespace().collect();
            let token_ids: Vec<TokenIdType> = words
                .iter()
                .map(|word| {
                    // Simple hash to get consistent IDs for same words
                    let mut hash = 0u32;
                    for byte in word.bytes() {
                        hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
                    }
                    hash % 10000 // Keep IDs reasonable
                })
                .collect();
            Ok((token_ids, words.len()))
        }

        fn encode_detailed(&self, text: &str) -> crate::tokenizers::Result<Encoding> {
            // Use the same logic as count_tokens_with_ids
            let (token_ids, _) = self.count_tokens_with_ids(text)?;
            Ok(Encoding::Sp(token_ids))
        }
    }

    #[test]
    fn test_single_token_approximation() {
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello", false, None),
                create_mock_response_with_content(0, " world", false, None),
                create_mock_response_with_content(0, "!", true, Some(FinishReason::Stop)),
            ],
            None,
        );

        assert_eq!(analysis.choice_analyses.len(), 1);
        let choice_analysis = &analysis.choice_analyses[&0];

        // Should count 3 tokens (one per response with content)
        assert_eq!(choice_analysis.total_tokens, 3);
        assert!(matches!(
            choice_analysis.primary_data_source,
            TokenDataSource::SingleTokenApproximation
        ));
        assert!(choice_analysis.is_complete);
        assert_eq!(choice_analysis.finish_reason.as_deref(), Some("Stop"));
    }

    #[test]
    fn test_tokenizer_based_counting() {
        let tokenizer = MockTokenizer;
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello world", false, None),
                create_mock_response_with_content(0, "How are you", false, None),
                create_mock_response_with_content(0, "today?", true, Some(FinishReason::Stop)),
            ],
            Some(&tokenizer),
        );

        assert_eq!(analysis.choice_analyses.len(), 1);
        let choice_analysis = &analysis.choice_analyses[&0];

        // Should count: 2 + 3 + 1 = 6 tokens (word count)
        assert_eq!(choice_analysis.total_tokens, 6);
        assert!(matches!(
            choice_analysis.primary_data_source,
            TokenDataSource::Tokenizer(_)
        ));
        assert!(choice_analysis.is_complete);
    }

    #[test]
    fn test_multiple_choices() {
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello", false, None),
                    (1, "Hi", false, None),
                ]),
                create_mock_response_with_multiple_choices(vec![
                    (0, " world", true, Some(FinishReason::Stop)),
                    (1, " there", true, Some(FinishReason::Stop)),
                ]),
            ],
            None,
        );

        assert_eq!(analysis.choice_analyses.len(), 2);
        assert!(analysis.is_stream_complete);

        for choice_index in [0, 1] {
            let choice_analysis = &analysis.choice_analyses[&choice_index];
            assert_eq!(choice_analysis.total_tokens, 2);
            assert!(choice_analysis.is_complete);
        }
    }

    #[test]
    fn test_incomplete_stream_validation() {
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello", false, None),
                create_mock_response_with_content(0, " world", false, None),
                // Missing finish reason
            ],
            None,
        );

        assert!(!analysis.is_stream_complete);
        assert!(analysis.has_errors());
        assert!(
            analysis
                .validation_errors
                .iter()
                .any(|err| err.contains("did not finish"))
        );
    }

    #[test]
    fn test_tokens_after_finish_validation() {
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello", false, None),
                create_mock_response_with_content(0, " world", true, Some(FinishReason::Stop)),
                create_mock_response_with_content(0, "!", false, None), // Invalid: after finish
            ],
            None,
        );

        assert!(analysis.has_errors());
        assert!(
            analysis
                .validation_errors
                .iter()
                .any(|err| err.contains("after finishing"))
        );
    }

    #[test]
    fn test_empty_content_handling() {
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "", false, None), // Empty content
                create_mock_response_with_content(0, "Hello", false, None),
                create_mock_response_with_content(0, "", true, Some(FinishReason::Stop)), // Empty finish
            ],
            None,
        );

        let choice_analysis = &analysis.choice_analyses[&0];
        // Should count only 1 token (for "Hello")
        assert_eq!(choice_analysis.total_tokens, 1);
        assert!(choice_analysis.is_complete);
    }

    #[test]
    fn test_multiple_choices_different_finish_times() {
        // Choice 0 finishes at response 2, Choice 1 finishes at response 4
        let analysis = create_analysis_with_responses(
            vec![
                // Response 0: Both choices generate tokens
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello", false, None),
                    (1, "Hi there", false, None),
                ]),
                // Response 1: Both choices continue
                create_mock_response_with_multiple_choices(vec![
                    (0, " world", false, None),
                    (1, " my friend", false, None),
                ]),
                // Response 2: Choice 0 finishes, Choice 1 continues
                create_mock_response_with_multiple_choices(vec![
                    (0, "!", true, Some(FinishReason::Stop)),
                    (1, " how", false, None),
                ]),
                // Response 3: Only Choice 1 continues (Choice 0 is finished)
                create_mock_response_with_multiple_choices(vec![(1, " are", false, None)]),
                // Response 4: Choice 1 finishes
                create_mock_response_with_multiple_choices(vec![(
                    1,
                    " you?",
                    true,
                    Some(FinishReason::Stop),
                )]),
            ],
            None,
        );

        // Verify basic structure
        assert_eq!(analysis.choice_analyses.len(), 2);
        assert!(analysis.is_stream_complete);
        assert!(!analysis.has_errors());

        // Check Choice 0: finished early at position 2
        let choice0 = &analysis.choice_analyses[&0];
        assert_eq!(choice0.total_tokens, 3); // "Hello", " world", "!"
        assert!(choice0.is_complete);
        assert_eq!(choice0.finish_position, Some(2));
        assert_eq!(choice0.finish_reason.as_deref(), Some("Stop"));
        assert_eq!(choice0.responses_with_choice, 3); // Appeared in responses 0, 1, 2

        // Check Choice 1: finished later at position 4
        let choice1 = &analysis.choice_analyses[&1];
        assert_eq!(choice1.total_tokens, 5); // "Hi there", " my friend", " how", " are", " you?"
        assert!(choice1.is_complete);
        assert_eq!(choice1.finish_position, Some(4));
        assert_eq!(choice1.finish_reason.as_deref(), Some("Stop"));
        assert_eq!(choice1.responses_with_choice, 5); // Appeared in all responses

        // Verify token timelines
        assert_eq!(choice0.token_timeline.len(), 3);
        assert_eq!(choice1.token_timeline.len(), 5);

        // Check that cumulative tokens are correct
        assert_eq!(choice0.token_timeline[0].cumulative_tokens, 1);
        assert_eq!(choice0.token_timeline[1].cumulative_tokens, 2);
        assert_eq!(choice0.token_timeline[2].cumulative_tokens, 3);

        assert_eq!(choice1.token_timeline[0].cumulative_tokens, 1);
        assert_eq!(choice1.token_timeline[1].cumulative_tokens, 2);
        assert_eq!(choice1.token_timeline[2].cumulative_tokens, 3);
        assert_eq!(choice1.token_timeline[3].cumulative_tokens, 4);
        assert_eq!(choice1.token_timeline[4].cumulative_tokens, 5);
    }

    #[test]
    fn test_choice_missing_from_middle_responses() {
        // Choice 0 appears in responses 0, 2, 4 (missing from 1, 3)
        // Choice 1 appears in responses 1, 3, 4 (missing from 0, 2)
        let analysis = create_analysis_with_responses(
            vec![
                // Response 0: Only Choice 0
                create_mock_response_with_multiple_choices(vec![(0, "Hello", false, None)]),
                // Response 1: Only Choice 1
                create_mock_response_with_multiple_choices(vec![(1, "Hi", false, None)]),
                // Response 2: Only Choice 0
                create_mock_response_with_multiple_choices(vec![(0, " world", false, None)]),
                // Response 3: Only Choice 1
                create_mock_response_with_multiple_choices(vec![(1, " there", false, None)]),
                // Response 4: Both choices finish
                create_mock_response_with_multiple_choices(vec![
                    (0, "!", true, Some(FinishReason::Stop)),
                    (1, "!", true, Some(FinishReason::Stop)),
                ]),
            ],
            None,
        );

        assert_eq!(analysis.choice_analyses.len(), 2);
        assert!(analysis.is_stream_complete);

        // Choice 0: appeared in responses 0, 2, 4
        let choice0 = &analysis.choice_analyses[&0];
        assert_eq!(choice0.total_tokens, 3);
        assert_eq!(choice0.responses_with_choice, 3);
        assert_eq!(choice0.finish_position, Some(4));

        // Choice 1: appeared in responses 1, 3, 4
        let choice1 = &analysis.choice_analyses[&1];
        assert_eq!(choice1.total_tokens, 3);
        assert_eq!(choice1.responses_with_choice, 3);
        assert_eq!(choice1.finish_position, Some(4));
    }

    #[test]
    fn test_validation_tokens_after_finish_multiple_choices() {
        // Choice 0 finishes early but then generates more tokens (invalid)
        // Choice 1 finishes properly
        let analysis = create_analysis_with_responses(
            vec![
                // Response 0: Both choices generate tokens
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello", false, None),
                    (1, "Hi", false, None),
                ]),
                // Response 1: Choice 0 finishes, Choice 1 continues
                create_mock_response_with_multiple_choices(vec![
                    (0, " world", true, Some(FinishReason::Stop)),
                    (1, " there", false, None),
                ]),
                // Response 2: Choice 0 generates more tokens (INVALID), Choice 1 finishes
                create_mock_response_with_multiple_choices(vec![
                    (0, "!", false, None), // This should trigger a validation error
                    (1, "!", true, Some(FinishReason::Stop)),
                ]),
            ],
            None,
        );

        // Should detect the validation error
        assert!(analysis.has_errors());
        assert!(
            analysis.validation_errors.iter().any(
                |err| err.contains("Choice 0") && err.contains("after finishing at position 1")
            )
        );

        // Choice 0 should still show as complete (finished at position 1)
        let choice0 = &analysis.choice_analyses[&0];
        assert!(choice0.is_complete);
        assert_eq!(choice0.finish_position, Some(1));
        // But total tokens includes the invalid token
        assert_eq!(choice0.total_tokens, 3);

        // Choice 1 should be fine
        let choice1 = &analysis.choice_analyses[&1];
        assert!(choice1.is_complete);
        assert_eq!(choice1.finish_position, Some(2));
        assert_eq!(choice1.total_tokens, 3);
    }

    #[test]
    fn test_stream_incomplete_when_some_choices_dont_finish() {
        // Choice 0 finishes, Choice 1 doesn't finish
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello", false, None),
                    (1, "Hi", false, None),
                ]),
                create_mock_response_with_multiple_choices(vec![
                    (0, " world!", true, Some(FinishReason::Stop)),
                    (1, " there", false, None), // No finish reason - incomplete
                ]),
            ],
            None,
        );

        // Stream should be incomplete
        assert!(!analysis.is_stream_complete);
        assert!(analysis.has_errors());
        assert!(
            analysis
                .validation_errors
                .iter()
                .any(|err| err.contains("choices [1]") && err.contains("did not finish"))
        );

        // Choice 0 should be complete
        let choice0 = &analysis.choice_analyses[&0];
        assert!(choice0.is_complete);

        // Choice 1 should be incomplete
        let choice1 = &analysis.choice_analyses[&1];
        assert!(!choice1.is_complete);
        assert_eq!(choice1.finish_position, None);
    }

    #[test]
    fn test_choice_appears_only_for_finish() {
        // Choice 0 generates tokens normally
        // Choice 1 only appears in the final response with finish_reason (edge case)
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_multiple_choices(vec![(0, "Hello", false, None)]),
                create_mock_response_with_multiple_choices(vec![(0, " world", false, None)]),
                create_mock_response_with_multiple_choices(vec![
                    (0, "!", true, Some(FinishReason::Stop)),
                    (1, "", true, Some(FinishReason::Stop)), // Only appears for finish
                ]),
            ],
            None,
        );

        assert_eq!(analysis.choice_analyses.len(), 2);
        assert!(analysis.is_stream_complete);

        // Choice 0: normal case
        let choice0 = &analysis.choice_analyses[&0];
        assert_eq!(choice0.total_tokens, 3);
        assert_eq!(choice0.responses_with_choice, 3);

        // Choice 1: only appeared once for finish
        let choice1 = &analysis.choice_analyses[&1];
        assert_eq!(choice1.total_tokens, 0); // No actual content tokens
        assert_eq!(choice1.responses_with_choice, 1);
        assert!(choice1.is_complete);
        assert_eq!(choice1.finish_position, Some(2));
    }

    #[test]
    fn test_different_finish_reasons() {
        // Test multiple choices with different finish reasons
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello", false, None),
                    (1, "Hi", false, None),
                    (2, "Hey", false, None),
                ]),
                create_mock_response_with_multiple_choices(vec![
                    (0, " world", true, Some(FinishReason::Stop)),
                    (1, " there", true, Some(FinishReason::Length)),
                    (2, " you", true, Some(FinishReason::ContentFilter)),
                ]),
            ],
            None,
        );

        assert!(analysis.is_stream_complete);
        assert_eq!(analysis.choice_analyses.len(), 3);

        // Check different finish reasons
        assert_eq!(
            analysis.choice_analyses[&0].finish_reason.as_deref(),
            Some("Stop")
        );
        assert_eq!(
            analysis.choice_analyses[&1].finish_reason.as_deref(),
            Some("Length")
        );
        assert_eq!(
            analysis.choice_analyses[&2].finish_reason.as_deref(),
            Some("ContentFilter")
        );

        // All should be complete
        assert!(analysis.choice_analyses[&0].is_complete);
        assert!(analysis.choice_analyses[&1].is_complete);
        assert!(analysis.choice_analyses[&2].is_complete);
    }

    #[test]
    fn test_token_rate_calculation_different_appearances() {
        // Choice 0 appears in 3 responses, generates 5 tokens
        // Choice 1 appears in 2 responses, generates 3 tokens
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello world", false, None), // 2 tokens with tokenizer
                    (1, "Hi there", false, None),    // 2 tokens with tokenizer
                ]),
                create_mock_response_with_multiple_choices(vec![
                    (0, "How are", false, None), // 2 tokens
                                                 // Choice 1 missing from this response
                ]),
                create_mock_response_with_multiple_choices(vec![
                    (0, "you?", true, Some(FinishReason::Stop)), // 1 token (single word), total = 5
                    (1, "today?", true, Some(FinishReason::Stop)), // 1 token (single word), total = 3
                ]),
            ],
            Some(&MockTokenizer),
        );

        // Choice 0: 5 tokens / 3 appearances = 1.6666... tokens per response
        assert!((analysis.token_rate_for_choice(0).unwrap() - (5.0 / 3.0)).abs() < 0.001);

        // Choice 1: 3 tokens / 2 appearances = 1.5 tokens per response
        assert_eq!(analysis.token_rate_for_choice(1), Some(1.5));

        // Non-existent choice
        assert_eq!(analysis.token_rate_for_choice(99), None);
    }

    #[test]
    fn test_chunk_alignment_validation() {
        let tokenizer = MockTokenizer;

        // Create a case where chunk alignment should work perfectly
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello world", false, None), // 2 tokens
                create_mock_response_with_content(0, " this is", false, None),    // 2 tokens
                create_mock_response_with_content(0, " a test", true, Some(FinishReason::Stop)), // 2 tokens
            ],
            Some(&tokenizer),
        );

        let choice_analysis = &analysis.choice_analyses[&0];
        assert_eq!(choice_analysis.total_tokens, 6); // 2 + 2 + 2

        // Check that we have tokenization validation with chunk alignments
        let validation = choice_analysis
            .full_tokenization_validation
            .as_ref()
            .unwrap();
        assert!(validation.counts_match);

        // Should have no validation errors (indicating successful chunk alignment)
        assert!(validation.validation_errors.is_empty());

        // Verify concatenated content matches
        let expected_full_text = "Hello world this is a test";
        assert_eq!(choice_analysis.concatenated_content, expected_full_text);
    }

    #[test]
    fn test_chunk_alignment_with_empty_chunks() {
        let tokenizer = MockTokenizer;

        // Test case with empty chunks mixed in
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello", false, None),
                create_mock_response_with_content(0, "", false, None), // Empty chunk
                create_mock_response_with_content(0, " world", false, None),
                create_mock_response_with_content(0, "", true, Some(FinishReason::Stop)), // Empty finish
            ],
            Some(&tokenizer),
        );

        let choice_analysis = &analysis.choice_analyses[&0];
        let validation = choice_analysis
            .full_tokenization_validation
            .as_ref()
            .unwrap();

        // Should have no validation errors (indicating successful chunk alignment)
        assert!(validation.validation_errors.is_empty());
    }

    #[test]
    fn test_chunk_alignment_errors_captured() {
        // Create a custom tokenizer that produces misaligned results
        struct MisalignedTokenizer;

        impl TokenCounter for MisalignedTokenizer {
            fn count_tokens(&self, text: &str) -> crate::tokenizers::Result<TokenCount> {
                Ok(text.split_whitespace().count())
            }

            fn count_tokens_with_ids(
                &self,
                text: &str,
            ) -> crate::tokenizers::Result<(Vec<TokenIdType>, usize)> {
                let words: Vec<&str> = text.split_whitespace().collect();
                // Create different token IDs for the same words when tokenized in different contexts
                let token_ids: Vec<TokenIdType> = words
                    .iter()
                    .enumerate()
                    .map(|(i, _)| i as u32 + if text.contains("isolated") { 1000 } else { 0 })
                    .collect();
                Ok((token_ids, words.len()))
            }

            fn encode_detailed(&self, text: &str) -> crate::tokenizers::Result<Encoding> {
                let (token_ids, _) = self.count_tokens_with_ids(text)?;
                Ok(Encoding::Sp(token_ids))
            }
        }

        let tokenizer = MisalignedTokenizer;

        // Create responses that will produce alignment errors
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "isolated word", false, None),
                create_mock_response_with_content(0, " test", true, Some(FinishReason::Stop)),
            ],
            Some(&tokenizer),
        );

        let choice_analysis = &analysis.choice_analyses[&0];
        let validation = choice_analysis
            .full_tokenization_validation
            .as_ref()
            .unwrap();

        // Should have validation errors due to misaligned tokenization
        assert!(!validation.validation_errors.is_empty());

        // Check that we have a chunk alignment validation error
        let has_chunk_alignment_error = validation.validation_errors.iter().any(|error| {
            error.get("type").and_then(|v| v.as_str()) == Some("chunk_alignment_validation")
        });
        assert!(
            has_chunk_alignment_error,
            "Should have chunk alignment validation errors"
        );
    }

    #[test]
    fn test_real_tokenizer_integration() {
        // This test shows how to use the real tokenizer infrastructure
        // Note: This test uses a mock since we don't have a real tokenizer file in tests

        // Demonstrate that our TokenCounter trait works with any Encoder
        let mock_tokenizer = MockTokenizer;

        // Test the TokenCounter methods directly
        let result = mock_tokenizer.count_tokens("Hello world test").unwrap();
        assert_eq!(result, 3); // "Hello", "world", "test"

        let (token_ids, count) = mock_tokenizer.count_tokens_with_ids("Hello world").unwrap();
        assert_eq!(count, 2);
        assert_eq!(token_ids.len(), 2); // Should have 2 token IDs

        // Test integration with our analysis system
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello world", false, None),
                create_mock_response_with_content(0, "How are you", false, None),
                create_mock_response_with_content(0, "today?", true, Some(FinishReason::Stop)),
            ],
            Some(&mock_tokenizer),
        );

        let choice_analysis = &analysis.choice_analyses[&0];
        assert_eq!(choice_analysis.total_tokens, 6); // 2 + 3 + 1

        // Verify that token IDs are captured
        let timeline = &choice_analysis.token_timeline;
        assert_eq!(timeline.len(), 3);
        assert!(timeline[0].content.is_some()); // Has content

        // Check that the data source is correctly identified as Tokenizer
        assert!(matches!(
            choice_analysis.primary_data_source,
            TokenDataSource::Tokenizer(_)
        ));
    }

    #[tokio::test]
    async fn test_token_stream_with_mock_tokenizer() -> Result<()> {
        use crate::tokenizers::Tokenizer;
        use std::sync::Arc;

        // Step 1: Load the mock tokenizer (no extraction needed)
        let tokenizer_path = "tests/data/sample-models/mock-llama-3.1-8b-instruct/tokenizer.json";
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        println!("✓ Mock tokenizer loaded successfully");

        // Step 2: Create a mock stream with known content
        let mock_responses = vec![
            create_mock_response_with_content(0, "Hello", false, None),
            create_mock_response_with_content(0, " world", false, None),
            create_mock_response_with_content(0, "!", false, None),
            create_mock_response_with_content(0, " How", false, None),
            create_mock_response_with_content(0, " are", false, None),
            create_mock_response_with_content(0, " you", false, None),
            create_mock_response_with_content(0, "?", false, Some(FinishReason::Stop)),
        ];

        // Step 3: Create recorded stream from mock responses
        let start_time = std::time::Instant::now();
        let timestamped_responses = mock_responses
            .into_iter()
            .enumerate()
            .map(|(i, response)| TimestampedResponse::new(response, i))
            .collect();

        let recorded_stream = RecordedStream::new(
            timestamped_responses,
            start_time,
            start_time + std::time::Duration::from_millis(100),
        );
        let arc_stream = Arc::new(recorded_stream);

        println!(
            "✓ Mock stream created with {} responses",
            arc_stream.response_count()
        );

        // Step 4: Analyze token counting with the mock tokenizer
        let analysis = analyze_token_counting(arc_stream, Some(&tokenizer));

        // Step 5: Validate the analysis results
        println!("=== Mock Stream Analysis Results ===");
        analysis.print_summary();

        // Basic validation
        assert!(
            !analysis.choice_analyses.is_empty(),
            "Should have at least one choice"
        );

        assert!(analysis.choice_analyses.len() == 1);

        // Check that we're using the tokenizer data source
        let choice_0 = analysis
            .choice_analyses
            .get(&0)
            .expect("Should have choice 0");

        assert!(
            matches!(choice_0.primary_data_source, TokenDataSource::Tokenizer(_)),
            "Should use tokenizer data source"
        );

        // Verify the stream was successfully completed
        assert!(
            analysis.is_stream_complete,
            "Stream should be complete with all choices finished"
        );

        // Verify the specific choice completed properly
        assert!(
            choice_0.is_complete,
            "Choice 0 should be complete with a finish_reason"
        );

        println!(
            "Choice 0: {} tokens, complete: {}",
            choice_0.total_tokens, choice_0.is_complete
        );

        // Validate that we have some token timeline events
        assert!(
            !choice_0.token_timeline.is_empty(),
            "Should have token events"
        );

        // Verify concatenated content is correct
        assert_eq!(
            choice_0.concatenated_content, "Hello world! How are you?",
            "Concatenated content should match expected text"
        );

        // Check data source consistency
        let consistency_errors = analysis.validate_data_source_consistency();
        if !consistency_errors.is_empty() {
            println!("Data source consistency warnings: {:?}", consistency_errors);
        }

        // Print token rate information
        if let Some(rate) = analysis.token_rate_for_choice(0) {
            println!(
                "Token generation rate for choice 0: {:.2} tokens per response (excluding empty responses)",
                rate
            );
        }

        println!("✓ Mock stream analysis completed successfully!");

        Ok(())
    }

    // Helper functions
    fn create_analysis_with_responses(
        responses: Vec<NvCreateChatCompletionStreamResponse>,
        tokenizer: Option<&dyn TokenCounter>,
    ) -> TokenAnalysis {
        let start_time = Instant::now();
        let timestamped_responses = responses
            .into_iter()
            .enumerate()
            .map(|(i, response)| TimestampedResponse::new(response, i))
            .collect();

        let recorded_stream =
            RecordedStream::new(timestamped_responses, start_time, Instant::now());
        let arc_stream = Arc::new(recorded_stream);

        analyze_token_counting(arc_stream, tokenizer)
    }

    fn create_mock_response_with_content(
        choice_index: u32,
        content: &str,
        _is_finished: bool,
        finish_reason: Option<FinishReason>,
    ) -> NvCreateChatCompletionStreamResponse {
        NvCreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            choices: vec![ChatChoiceStream {
                index: choice_index,
                delta: ChatCompletionStreamResponseDelta {
                    content: if content.is_empty() {
                        None
                    } else {
                        Some(content.to_string())
                    },
                    function_call: None,
                    tool_calls: None,
                    role: Some(Role::Assistant),
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason,
                logprobs: None,
            }],
            created: 1234567890,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        }
    }

    fn create_mock_response_with_multiple_choices(
        choices_data: Vec<(u32, &str, bool, Option<FinishReason>)>,
    ) -> NvCreateChatCompletionStreamResponse {
        #[expect(deprecated)]
        let choices = choices_data
            .into_iter()
            .map(
                |(choice_index, content, _is_finished, finish_reason)| ChatChoiceStream {
                    index: choice_index,
                    delta: ChatCompletionStreamResponseDelta {
                        content: if content.is_empty() {
                            None
                        } else {
                            Some(content.to_string())
                        },
                        function_call: None,
                        tool_calls: None,
                        role: Some(Role::Assistant),
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason,
                    logprobs: None,
                },
            )
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
        }
    }
}
