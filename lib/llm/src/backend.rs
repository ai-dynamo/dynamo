// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend
//!
//! An [`Backend`] is the final stage of the pipeline. It represents the execution of the LLM
//! on some processing hardware.
//!
//! At minimum, the Backend is split into two components, the [`Backend`] itself and a downstream [`ExecutionContext`].
//!
//! The [`ExecutionContext`] can be thought of as the core driver of the forward pass, whereas the [`Backend`] is the
//! manager of all resources and concurrent tasks surrounding the LLM execution context / forward pass.
//!
//! For almost every known scenario, detokenization and initial post processing must happen in the Backend.
//! Further post-processing can happen in the response stream. One example is the jailing mechanism for partial
//! hidden stop condition matches, which can be handled in the response stream rather than the backend.

use std::{collections::HashSet, sync::Arc, time::Instant};

use anyhow::Result;
use futures::stream::{self, StreamExt};

use crate::model_card::ModelDeploymentCard;
use dynamo_runtime::{
    pipeline::{
        AsyncEngineContextProvider, ManyOut, Operator, ResponseStream, ServerStreamingEngine,
        SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};

use crate::protocols::{
    TokenIdType,
    common::{
        StopConditions,
        llm_backend::{
            BackendOutput, EmbeddingsEngineOutput, FinishReason, LLMEngineOutput,
            PreprocessedRequest,
        },
        preprocessor::PreprocessedEmbeddingRequest,
        timing::RequestTracker,
    },
};
use crate::tokenizers::{DecodeStream, HuggingFaceTokenizer, Tokenizer};
use dynamo_async_openai::types::StopReason;
use tokenizers::Tokenizer as HfTokenizer;

/// Represents the output stream from the execution engine
pub type ExecutionOutputStream = Annotated<LLMEngineOutput>;

/// Context for executing LLM inference, engine consumes backend input and produces execution output stream
pub type ExecutionContext = ServerStreamingEngine<PreprocessedRequest, ExecutionOutputStream>;

/// Backend handles resource management and orchestrates LLM execution
#[allow(dead_code)]
pub struct Backend {
    pub tokenizer: Option<Tokenizer>, // Handles token encoding/decoding
    validate_engine_decode: bool,     // Enable validation of engine decoding
}

/// Internal state for managing token decoding and stream processing
#[allow(dead_code)]
struct DecoderUnfoldState {
    stream: ManyOut<ExecutionOutputStream>,
    decoder: Decoder,
    validate_engine_decode: bool,
    /// Set to true when a local stop condition is detected, causing the stream to end
    finished: bool,
    /// Tokenizer for detokenizing top_logprobs tokens
    tokenizer: Option<Tokenizer>,
}

impl Backend {
    pub fn from_tokenizer(tokenizer: HfTokenizer) -> Arc<Self> {
        let tokenizer = HuggingFaceTokenizer::from_tokenizer(tokenizer);
        let tokenizer = Tokenizer::from(Arc::new(tokenizer));

        Arc::new(Self {
            tokenizer: Some(tokenizer),
            validate_engine_decode: false,
        })
    }

    pub fn from_mdc(mdc: &ModelDeploymentCard) -> Arc<Self> {
        match mdc.tokenizer_hf() {
            Ok(tokenizer) => Self::from_tokenizer(tokenizer),
            Err(err) => {
                tracing::warn!(%err, "tokenizer_hf error converting ModelDeploymentCard to HF tokenizer");
                Arc::new(Self {
                    tokenizer: None,
                    validate_engine_decode: false,
                })
            }
        }
    }

    fn decoder(
        &self,
        stream: ManyOut<ExecutionOutputStream>,
        prompt_token_ids: &[TokenIdType],
        stop_conditions: StopConditions,
        skip_special_tokens: bool,
        include_stop_str_in_output: bool,
        tracker: Option<Arc<RequestTracker>>,
    ) -> anyhow::Result<DecoderUnfoldState> {
        let Some(tokenizer) = self.tokenizer.as_ref() else {
            anyhow::bail!("Backend built from blank ModelDeploymentCard, no tokenizer");
        };
        let decoder = Decoder::new(
            tokenizer.decode_stream(prompt_token_ids, skip_special_tokens),
            stop_conditions,
            include_stop_str_in_output,
            tracker,
        );

        Ok(DecoderUnfoldState {
            stream,
            decoder,
            validate_engine_decode: self.validate_engine_decode,
            finished: false,
            tokenizer: self.tokenizer.clone(),
        })
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for Backend
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<BackendOutput>>> {
        let stop_conditions = request.stop_conditions.clone();

        let prompt_token_ids = request.token_ids.clone();

        // TODO: Consider updating default to true to match behavior of other frameworks
        let skip_special_tokens = request.output_options.skip_special_tokens.unwrap_or(false);

        // Extract include_stop_str_in_output from sampling_options (defaults to false)
        let include_stop_str_in_output = request
            .sampling_options
            .include_stop_str_in_output
            .unwrap_or(false);
        let tracker = request.tracker.clone();

        let next_stream = next.generate(request).await?;

        let context = next_stream.context();
        let state = self.decoder(
            next_stream,
            &prompt_token_ids,
            stop_conditions,
            skip_special_tokens,
            include_stop_str_in_output,
            tracker,
        )?;

        let processed_stream = stream::unfold(state, |mut state| async move {
            // If we've already detected a local stop condition, end the stream
            if state.finished {
                return None;
            }

            match state.stream.next().await {
                Some(output) => {
                    // move to state.process_output
                    // handle any error conditions / unwraps here

                    // events are pass thru
                    if output.is_event() || output.data.is_none() {
                        return Some((output, state));
                    }

                    // if we have a data field without an event, then we might need to update the data
                    if let Some(data) = &output.data
                        && data.text.is_some()
                        && !state.validate_engine_decode
                    {
                        return Some((output, state));
                    }

                    let data = output.data.as_ref().unwrap();

                    let result = state.decoder.process_token_ids(&data.token_ids).unwrap();

                    // NOTE: the `finish_reason` is computed from the generated `token_ids` alone.
                    // The `data` field can have a `finish_reason` set, coming from the underlying
                    // LLM inference `Engine`, and empty `token_ids`. See comment below for more details.
                    //
                    // stop_reason is only set for user-provided stop sequences, not for system
                    // EOS tokens (HiddenStopTokenDetected). This matches OpenAI API behavior where
                    // stop_reason is only present when a user-specified stop sequence is matched.
                    let (finish_reason, stop_reason) = match &result.stop_trigger {
                        Some(StopTrigger::MaxTokensLimit) => (Some(FinishReason::Length), None),
                        Some(StopTrigger::HiddenStopTokenDetected(_)) => {
                            // System EOS token - no stop_reason (user didn't request this stop)
                            (Some(FinishReason::Stop), None)
                        }
                        Some(StopTrigger::HiddenStopSequenceDetected(seq)) => {
                            // User-provided stop sequence (hidden from output)
                            (
                                Some(FinishReason::Stop),
                                Some(StopReason::String(seq.clone())),
                            )
                        }
                        Some(StopTrigger::VisibleStopSequenceDetected(seq)) => {
                            // User-provided stop sequence (included in output)
                            (
                                Some(FinishReason::Stop),
                                Some(StopReason::String(seq.clone())),
                            )
                        }
                        None => (None, None),
                    };

                    // If we detected a local stop condition, mark stream as finished
                    // so we stop iterating (upstream may keep generating, but we ignore it)
                    if finish_reason.is_some() && data.finish_reason.is_none() {
                        state.stream.context().stop_generating();
                        state.finished = true;
                    }

                    let text = result.text;
                    let tokens = result.tokens;

                    if state.validate_engine_decode {
                        if data.finish_reason != finish_reason {
                            tracing::warn!(
                                "finish reason mismatch: expected {:?}, got {:?}",
                                data.finish_reason,
                                finish_reason
                            );
                        }

                        if data.text.is_some() && data.text != text {
                            tracing::warn!(
                                "text mismatch: expected {:?}, got {:?}",
                                data.text,
                                text
                            );
                        }
                    }

                    // update output in-place
                    let mut output = output;
                    let mut data = output.data.take().unwrap();

                    // NOTE: If `finish_reason.is_some()`, then one of the stop conditions was triggered
                    // by the token generation. We should update the `data.finish_reason` in that case.
                    // However, if `finish_reason.is_none()`, it is possible that we are in the case where
                    // `data.token_ids` is empty, and `data.finish_reason` is already correctly set.
                    // In that case, `process_token_ids` above will rewrite `finish_reason` to `None`,
                    // which we don't want to propagate to `data.finish_reason`.
                    if finish_reason.is_some() {
                        data.finish_reason = finish_reason;
                        data.stop_reason = stop_reason;
                    }
                    data.text = text;
                    data.tokens = Some(tokens);

                    // Detokenize top_logprobs tokens if needed
                    if let Some(ref mut top_logprobs) = data.top_logprobs {
                        if let Some(ref tokenizer) = state.tokenizer {
                            for position_logprobs in top_logprobs.iter_mut() {
                                for logprob in position_logprobs.iter_mut() {
                                    if logprob.token.is_none() {
                                        // Decode single token_id to get token text
                                        if let Ok(text) =
                                            tokenizer.decode(&[logprob.token_id], false)
                                        {
                                            logprob.token = Some(text);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    output.data = Some(data);

                    Some((output, state))
                }

                None => None,
            }
        })
        .fuse();

        // convert stream of processed Annotated<LLMEngineOutput> to Annotated<BackendOutput>
        //let mdcsum = self.mdcsum.clone();
        let stream = processed_stream.map(move |output| {
            output.map_data(|data| {
                Ok(BackendOutput {
                    token_ids: data.token_ids,
                    tokens: data.tokens.unwrap_or_default(),
                    text: data.text,
                    cum_log_probs: data.cum_log_probs,
                    log_probs: data.log_probs,
                    top_logprobs: data.top_logprobs,
                    finish_reason: data.finish_reason,
                    stop_reason: data.stop_reason,
                    //mdcsum: mdcsum.clone(),
                    index: data.index,
                    completion_usage: data.completion_usage,
                    disaggregated_params: data.disaggregated_params,
                })
            })
        });

        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
    > for Backend
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedEmbeddingRequest>,
        next: ServerStreamingEngine<
            PreprocessedEmbeddingRequest,
            Annotated<EmbeddingsEngineOutput>,
        >,
    ) -> Result<ManyOut<Annotated<EmbeddingsEngineOutput>>> {
        // For embeddings, we mostly pass through since no detokenization is needed
        // But we could add validation, logging, or other post-processing here
        let response_stream = next.generate(request).await?;

        // Could add embedding-specific post-processing here:
        // - Validation of embedding dimensions
        // - Normalization if requested
        // - Usage statistics validation

        Ok(response_stream)
    }
}

/// The [`Decoder`] object could be a member of either the internal LLM engine or part of the
/// postprocessor. If in the postprocessor, should be minimally in the same process or at very minimum
/// on the same physical machine connected by an IPC.
#[allow(dead_code)]
pub struct Decoder {
    decode_stream: DecodeStream,
    tracker: Option<Arc<RequestTracker>>,

    // do not trigger stop conditions until at least this many tokens have been generated
    min_tokens: u32,

    // single tokens that if found in the response will trigger a stop condition after the
    // minimum number of tokens have been generated
    hidden_stop_ids: HashSet<TokenIdType>,

    // text sequences that if found in the response will trigger a stop condition after the
    // minimum number of tokens have been generated (excluded from output)
    hidden_stop_sequences: Vec<String>,

    // text sequences that if found in the response will trigger a stop condition after the
    // minimum number of tokens have been generated (included in output)
    visible_stop_sequences: Vec<String>,

    // number of generated tokens
    generated_tokens: u32,

    // content jailed by partial hidden stop matches
    jail: String,

    // maximum number of bytes for the largest stop sequence
    jail_max_bytes: usize,

    // the number of bytes currently jailed
    jailed_bytes: usize,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum StopTrigger {
    MaxTokensLimit,
    HiddenStopTokenDetected(TokenIdType),
    HiddenStopSequenceDetected(String),
    VisibleStopSequenceDetected(String),
}

pub struct StepResult {
    pub token: Option<String>,
    pub stop_trigger: Option<StopTrigger>,
}

impl StepResult {
    fn ok(token: Option<String>) -> Self {
        Self {
            token,
            stop_trigger: None,
        }
    }

    fn with_stop_trigger(token: Option<String>, stop_trigger: StopTrigger) -> Self {
        Self {
            token,
            stop_trigger: Some(stop_trigger),
        }
    }
}

/// Result of processing a sequence of tokens
pub struct SeqResult {
    pub tokens: Vec<Option<String>>,       // Individual decoded tokens
    pub text: Option<String>,              // Combined decoded text
    pub stop_trigger: Option<StopTrigger>, // Reason for stopping generation, if any
}

#[allow(dead_code)]
impl Decoder {
    pub fn new(
        decode_stream: DecodeStream,
        stop_condition: StopConditions,
        include_stop_str_in_output: bool,
        tracker: Option<Arc<RequestTracker>>,
    ) -> Self {
        let hidden_stop_ids: HashSet<TokenIdType> = stop_condition
            .stop_token_ids_hidden
            .unwrap_or_default()
            .iter()
            .copied()
            .collect();

        // Categorize stop sequences based on include_stop_str_in_output:
        // - When true: user-provided stop sequences go to visible (included in output)
        // - When false: user-provided stop sequences go to hidden (excluded from output)
        let (hidden_stop_sequences, visible_stop_sequences) = if include_stop_str_in_output {
            (Vec::new(), stop_condition.stop.unwrap_or_default())
        } else {
            (stop_condition.stop.unwrap_or_default(), Vec::new())
        };

        // Calculate jail_max_bytes considering both hidden and visible stop sequences
        let jail_max_bytes = hidden_stop_sequences
            .iter()
            .chain(visible_stop_sequences.iter())
            .map(|x| x.len())
            .max()
            .unwrap_or(0);

        Self {
            decode_stream,
            tracker,
            hidden_stop_ids,
            hidden_stop_sequences,
            visible_stop_sequences,
            min_tokens: stop_condition.min_tokens.unwrap_or(0),
            generated_tokens: 0,
            jail: String::new(),
            jail_max_bytes,
            jailed_bytes: 0,
        }
    }

    /// Minimum amount of work to determine if a given generated/decoded sequence should be stopped
    /// This method can be called by the inner most loop of the LLM engine or minimally in the same
    /// process as the LLM engine.
    ///
    /// In the future, this method may kick off async cpu/tokio tasks and or async cuda tasks to
    /// handle logits post-processing and/or other tasks.
    pub fn step(&mut self, token_id: TokenIdType) -> Result<StepResult> {
        // increment the generated tokens
        self.generated_tokens += 1;

        // decode the token
        let detokenize_start = Instant::now();
        let token = self.decode_stream.step(token_id)?;
        if let Some(tracker) = &self.tracker {
            tracker.record_detokenize_latency(detokenize_start.elapsed());
        }

        // stop conditions to not apply until the minimum number of tokens have been generated
        if self.generated_tokens < self.min_tokens {
            return Ok(StepResult::ok(token));
        }

        // check for hidden stop tokens - eos takes precedence
        if self.hidden_stop_ids.contains(&token_id) {
            return Ok(StepResult::with_stop_trigger(
                None,
                StopTrigger::HiddenStopTokenDetected(token_id),
            ));
        }

        // check stop sequences - the jail will always hold at least the largest stop sequence
        // if jail_max_bytes is 0, then there are no stop sequences
        if self.jail_max_bytes > 0
            && let Some(token) = &token
        {
            let pre_append = self.jail.len();
            self.jail.push_str(token);

            // Check hidden stop sequences first (excluded from output)
            for seq in &self.hidden_stop_sequences {
                if let Some(offset) = galil_seiferas::gs_find(self.jail.as_bytes(), seq.as_bytes())
                {
                    // return only new bytes after pre_append .. offset (excluding stop sequence)
                    // example: seq = "ox", token = "boxes", return "b"
                    // note: this changes when we start jailing tokens for partial matches
                    // on the suffix of the jail with prefixes of the stop sequences
                    //
                    // we might have returned a partial match, if so, then offset < pre_append
                    // in that case, we return the empty string
                    let partial_token = if offset >= pre_append {
                        self.jail[pre_append..offset].to_string()
                    } else {
                        "".to_string()
                    };
                    return Ok(StepResult::with_stop_trigger(
                        Some(partial_token),
                        StopTrigger::HiddenStopSequenceDetected(seq.to_string()),
                    ));
                }
            }

            // Check visible stop sequences (included in output)
            for seq in &self.visible_stop_sequences {
                if let Some(offset) = galil_seiferas::gs_find(self.jail.as_bytes(), seq.as_bytes())
                {
                    // For visible stop sequences, include the stop string in the output
                    // Return all text from pre_append up to and including the stop sequence
                    let stop_end = offset + seq.len();
                    let token_with_stop = if stop_end > pre_append {
                        self.jail[pre_append..stop_end].to_string()
                    } else {
                        // Stop sequence was entirely in previously returned text
                        "".to_string()
                    };
                    return Ok(StepResult::with_stop_trigger(
                        Some(token_with_stop),
                        StopTrigger::VisibleStopSequenceDetected(seq.to_string()),
                    ));
                }
            }

            Self::maybe_drain_to_max_bytes(&mut self.jail, self.jail_max_bytes);
        }

        Ok(StepResult::ok(token))
    }

    pub fn process_token_ids(&mut self, token_ids: &[TokenIdType]) -> Result<SeqResult> {
        let mut text: Option<String> = None;
        let mut tokens = Vec::with_capacity(token_ids.len());

        for token_id in token_ids {
            let StepResult {
                token,
                stop_trigger,
            } = self.step(*token_id)?;

            // Always include token text (for visible stops, the stop string is already in the token)
            if let Some(token) = &token {
                text.get_or_insert_with(|| String::with_capacity(token_ids.len()))
                    .push_str(token);
            }
            tokens.push(token);

            if let Some(stop_trigger) = stop_trigger {
                return Ok(SeqResult {
                    tokens,
                    text,
                    stop_trigger: Some(stop_trigger),
                });
            }
        }

        Ok(SeqResult {
            tokens,
            text,
            stop_trigger: None,
        })
    }

    fn jailed_string(&self) -> Option<String> {
        if self.jailed_bytes > 0 {
            // get the last jailed_bytes from the jail
            Some(self.jail[self.jail.len() - self.jailed_bytes..].to_string())
        } else {
            None
        }
    }

    fn maybe_drain_to_max_bytes(s: &mut String, max_bytes: usize) {
        if s.len() > max_bytes {
            let mut drain_len = s.len() - max_bytes;
            while !s.is_char_boundary(drain_len) {
                drain_len -= 1;
            }
            s.drain(0..drain_len);
        }
    }
}

mod tests {

    #[test]
    fn test_char_boundary_drain() {
        use super::Decoder;
        let mut s = String::from("helloñworld"); // 12 bytes total ñ is 2 bytes
        let max_bytes = 6; // 12 - 6 = 6 which is inside ñ
        assert!(!s.is_char_boundary(s.len() - max_bytes)); // initially we are not on a char boundary
        Decoder::maybe_drain_to_max_bytes(&mut s, max_bytes);
        assert!(s.is_char_boundary(0)); // front of jail string on valid char boundary
        assert_eq!(s, "ñworld");
    }
}
