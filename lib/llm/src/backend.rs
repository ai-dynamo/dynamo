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

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use anyhow::Result;
use futures::stream::{self, StreamExt};

use crate::model_card::ModelDeploymentCard;
use dynamo_runtime::dynamo_nvtx_range;
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
            PreprocessedRequest, TopLogprobs,
        },
        preprocessor::PreprocessedEmbeddingRequest,
        timing::RequestTracker,
    },
};
use crate::tokenizers::{DecodeStream, Tokenizer};
use dynamo_protocols::types::StopReason;

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
/// Supports n>1 (multiple choices) by maintaining per-choice decoders.
#[allow(dead_code)]
struct DecoderUnfoldState {
    stream: ManyOut<ExecutionOutputStream>,
    decoders: HashMap<u32, Decoder>,
    finished_choices: HashSet<u32>,
    validate_engine_decode: bool,
    /// Set to true when all expected choices are finished locally, causing the stream to end
    finished: bool,
    /// Tokenizer used to decode top-logprob token_ids when the backend omits text
    /// (e.g. SGLang with --skip-tokenizer-init forcibly drops it).
    tokenizer: Tokenizer,
    skip_special_tokens: bool,
    /// Text flushed from a choice's decoder because the underlying engine stream ended
    /// without ever sending that choice a terminal `finish_reason`, queued here so each
    /// flushed choice can be emitted as its own synthetic final chunk.
    pending_flush: Vec<(u32, String)>,
    /// Set once `stream` has yielded `None`, so it is never polled again -- a `Stream` is
    /// not guaranteed to be safely pollable past its first `None`.
    stream_ended: bool,
    /// Whether this request could still be migrated (see `DecoderParams::migration_possible`).
    /// When `false`, `jailed_text` is left unset on every chunk rather than snapshotting the
    /// decoder's withheld state for a checkpoint nothing can ever consume.
    migration_possible: bool,
}

fn fill_missing_top_logprob_text(
    tokenizer: &Tokenizer,
    top_logprobs: &mut TopLogprobs,
    skip_special_tokens: bool,
) {
    for position in top_logprobs.iter_mut() {
        for entry in position.iter_mut() {
            if entry.token.is_none()
                && let Ok(decoded) = tokenizer.decode(&[entry.token_id], skip_special_tokens)
            {
                let token: String = decoded.into();
                if entry.bytes.is_none() && !token.is_empty() {
                    entry.bytes = Some(token.as_bytes().to_vec());
                }
                entry.token = Some(token);
            }
        }
    }
}

struct DecoderParams {
    prompt_token_ids: Vec<TokenIdType>,
    stop_conditions: StopConditions,
    skip_special_tokens: bool,
    include_stop_str_in_output: bool,
    tracker: Option<Arc<RequestTracker>>,
    n: u32,
    // Withheld hidden-stop-sequence prefix carried over from a migrated attempt's last
    // known-good chunk (see `PreprocessedRequest::jail_seed`). `None` on a first attempt.
    jail_seed: Option<String>,
    // Whether this request could still be migrated to another worker (i.e. a `RetryManager`
    // sits in front of this `Backend` and has retries configured). When `false`, no chunk
    // this Backend emits can ever be reseeded into a retry, so there is no point snapshotting
    // the decoder's withheld state onto every chunk via `jailed_text`.
    migration_possible: bool,
}

impl DecoderParams {
    fn from_request(request: &PreprocessedRequest) -> Self {
        Self {
            prompt_token_ids: request.token_ids.as_ref().clone(),
            stop_conditions: request.stop_conditions.clone(),
            // Default to true to match upstream framework behavior:
            //   vLLM/sgLang/TRT-LLM: SamplingParams.skip_special_tokens defaults True
            // Without this, models that occasionally emit a special token id
            // (e.g. DeepSeek-V4 producing token id 0 = `<｜begin▁of▁sentence｜>`
            // mid-output) leak the token's text into `content` / `reasoning_content`.
            skip_special_tokens: request.output_options.skip_special_tokens.unwrap_or(true),
            include_stop_str_in_output: request
                .sampling_options
                .include_stop_str_in_output
                .unwrap_or(false),
            tracker: request.tracker.clone(),
            n: request.sampling_options.n.unwrap_or(1) as u32,
            jail_seed: request.jail_seed.clone(),
            migration_possible: request.migration_state.is_some(),
        }
    }
}

impl Backend {
    pub fn from_tokenizer(tokenizer: Tokenizer) -> Arc<Self> {
        Arc::new(Self {
            tokenizer: Some(tokenizer),
            validate_engine_decode: false,
        })
    }

    pub fn from_mdc(mdc: &ModelDeploymentCard) -> Arc<Self> {
        match mdc.tokenizer() {
            Ok(tokenizer) => Self::from_tokenizer(tokenizer),
            Err(err) => {
                tracing::warn!(%err, "error loading tokenizer from ModelDeploymentCard");
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
        params: DecoderParams,
    ) -> anyhow::Result<DecoderUnfoldState> {
        let Some(tokenizer) = self.tokenizer.as_ref() else {
            anyhow::bail!("Backend built from blank ModelDeploymentCard, no tokenizer");
        };

        // Pre-create one decoder per expected choice so interleaved n>1
        // responses do not share tokenizer state.
        let n = params.n.max(1);
        let mut decoders = HashMap::with_capacity(n as usize);
        for idx in 0..n {
            let decoder = Decoder::new(
                tokenizer.decode_stream(&params.prompt_token_ids, params.skip_special_tokens),
                params.stop_conditions.clone(),
                params.include_stop_str_in_output,
                params.tracker.clone(),
                // Every choice starts from the same carried-over withheld prefix. This
                // only matters for `n == 1` migration retries in practice; a fresh
                // decoder normally starts unseeded (`None`).
                params.jail_seed.clone(),
            );
            decoders.insert(idx, decoder);
        }

        Ok(DecoderUnfoldState {
            stream,
            decoders,
            finished_choices: HashSet::new(),
            validate_engine_decode: self.validate_engine_decode,
            finished: false,
            tokenizer: tokenizer.clone(),
            skip_special_tokens: params.skip_special_tokens,
            pending_flush: Vec::new(),
            stream_ended: false,
            migration_possible: params.migration_possible,
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
        let decoder_params = DecoderParams::from_request(&request);

        let next_stream = next.generate(request).await?;

        let context = next_stream.context();
        let state = self.decoder(next_stream, decoder_params)?;

        let processed_stream = stream::unfold(state, |mut state| async move {
            // If we've already detected a local stop condition, end the stream
            if state.finished {
                return None;
            }

            // `stream` already yielded `None` once; do not poll it again (unspecified
            // behavior for most `Stream` impls). Only drain the flush queue from here on.
            if state.stream_ended {
                return state.pending_flush.pop().map(|(idx, flushed)| {
                    let output = Annotated::from_data(LLMEngineOutput {
                        index: Some(idx),
                        text: Some(flushed),
                        finish_reason: Some(FinishReason::Stop),
                        ..Default::default()
                    });
                    (output, state)
                });
            }

            match state.stream.next().await {
                Some(output) => {
                    // move to state.process_output
                    // handle any error conditions / unwraps here

                    // events are pass thru
                    if output.is_event() || output.data.is_none() {
                        // A genuine stream-level error ends generation for every choice at
                        // once -- `Annotated::from_err` carries no per-choice `index` the
                        // way `LLMEngineOutput` does, so there is no narrower target than
                        // "all of them". Mark them finished so the bare-EOF flush loop below
                        // does not synthesize a spurious successful `Stop` for any choice
                        // after a real error, the same hardening already applied to decode
                        // errors and the decoded-text bypass path.
                        if output.is_error() {
                            state.finished_choices.extend(state.decoders.keys().copied());
                        }
                        return Some((output, state));
                    }

                    // Top-logprob text is independent of the selected token text. An engine may
                    // decode the selected token while still omitting candidate text, so repair the
                    // candidates before the decoded-text fast path below.
                    //
                    // Per-entry decode is O(positions * top_k) per delta. Bounded in
                    // practice (streaming: 1 * top_k <= 20) and dwarfed by serialization
                    // on the same path, so we ship the simple version. Revisit if a
                    // streaming flamegraph with top_logprobs=20 puts this above ~1%:
                    // the cheapest win is a shared LRU on the Tokenizer keyed by
                    // (token_id, skip_special_tokens) — top-k entries repeat heavily
                    // across positions and requests. Do NOT batch as a single
                    // decode(&[ids..]) call: BPE merge / leading-space rules differ
                    // between single-token and sequence decode and will corrupt strings.
                    let mut output = output;
                    if let Some(top_logprobs) = output
                        .data
                        .as_mut()
                        .and_then(|data| data.top_logprobs.as_mut())
                    {
                        fill_missing_top_logprob_text(
                            &state.tokenizer,
                            top_logprobs,
                            state.skip_special_tokens,
                        );
                    }

                    // if we have a data field without an event, then we might need to update the data
                    if let Some(data) = &output.data
                        && data.text.is_some()
                        && !state.validate_engine_decode
                    {
                        // Text already decoded; track finish for this choice
                        let choice_idx = data.index.unwrap_or(0);
                        let has_finish = data.finish_reason.is_some();
                        if has_finish {
                            state.finished_choices.insert(choice_idx);
                        }
                        // This path bypasses the decoder entirely (the engine pre-decoded
                        // its own text for this chunk), but the decoder for this choice is
                        // still alive across chunks and may be holding text withheld by a
                        // *previous* chunk that did go through it. Losing track of that here
                        // would either drop it silently (on a terminal chunk) or make a
                        // migration checkpoint captured from this chunk (see `jail_seed`)
                        // look falsely resolved, so:
                        if let Some(decoder) = state.decoders.get_mut(&choice_idx) {
                            // On a terminal chunk, flush it and put it *before* this chunk's
                            // own text -- it is strictly older -- rather than appending,
                            // which would reorder output (e.g. "there" + withheld "o" must
                            // come out as "othere", not "thereo").
                            if has_finish
                                && let Some(flushed) = decoder.flush_jailed()
                                && let Some(data) = &mut output.data
                            {
                                let newer = data.text.take().unwrap_or_default();
                                data.text = Some(flushed + &newer);
                            }
                            // Mirror the decoder's current withheld state on every chunk
                            // (terminal or not), even though this chunk didn't change it, so
                            // a checkpoint captured here reflects reality instead of always
                            // reading as resolved. Skipped entirely when migration can't
                            // happen -- nothing will ever read this checkpoint, so there is
                            // no reason to allocate a snapshot of it.
                            if state.migration_possible
                                && let Some(data) = &mut output.data
                            {
                                data.jailed_text = decoder.peek_jailed();
                            }
                        }
                        return Some((output, state));
                    }

                    let data = output.data.as_ref().unwrap();
                    let choice_idx = data.index.unwrap_or(0);
                    // Snapshot the choice count before borrowing a specific decoder mutably
                    // below: that borrow stays alive until this choice's `peek_jailed()` call
                    // near the end of this arm, so `state.decoders` can't be read again
                    // (even just its length) in between. The count itself never changes
                    // after `Backend::decoder()` creates the map, so this is safe to cache.
                    let decoders_count = state.decoders.len();

                    let Some(decoder) = state.decoders.get_mut(&choice_idx) else {
                        tracing::error!(
                            "engine emitted choice index {choice_idx}, but only {decoders_count} choices were requested"
                        );
                        let mut output = output;
                        if let Some(data) = &mut output.data {
                            data.finish_reason = Some(FinishReason::Error(format!(
                                "invalid choice index {choice_idx}"
                            )));
                        }
                        return Some((output, state));
                    };

                    let mut result = match decoder.process_token_ids(&data.token_ids) {
                        Ok(result) => result,
                        Err(e) => {
                            tracing::error!("Failed to process token_ids for choice {choice_idx}: {e}");
                            state.finished_choices.insert(choice_idx);
                            if state.finished_choices.len() >= decoders_count {
                                state.stream.context().stop_generating();
                                state.finished = true;
                            }
                            let mut output = output;
                            if let Some(data) = &mut output.data {
                                data.finish_reason =
                                    Some(FinishReason::Error(format!("decode error: {e}")));
                            }
                            return Some((output, state));
                        }
                    };

                    // The engine can report its own completion (e.g. it hit `max_tokens`)
                    // without our decoder ever detecting a local stop condition. Any text
                    // still withheld as a partial hidden-stop-sequence match can never
                    // complete at that point, so flush it now rather than silently dropping
                    // it -- this is the last chance before the decoder for this choice is
                    // discarded.
                    if result.stop_trigger.is_none()
                        && data.finish_reason.is_some()
                        && let Some(flushed) = decoder.flush_jailed()
                    {
                        result.text.get_or_insert_with(String::new).push_str(&flushed);
                    }

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
                        Some(StopTrigger::UserStopTokenDetected(token_id)) => {
                            // User-provided token stop (hidden from output)
                            (
                                Some(FinishReason::Stop),
                                Some(StopReason::Int((*token_id).into())),
                            )
                        }
                        Some(StopTrigger::VisibleStopTokenDetected(token_id)) => {
                            // Token stop included in output.
                            (
                                Some(FinishReason::Stop),
                                Some(StopReason::Int((*token_id).into())),
                            )
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

                    // If we detected a local stop condition, mark this choice as finished.
                    // Once all expected choices are finished, stop the upstream generator.
                    if finish_reason.is_some() && data.finish_reason.is_none() {
                        state.finished_choices.insert(choice_idx);
                        if state.finished_choices.len() >= decoders_count {
                            state.stream.context().stop_generating();
                            state.finished = true;
                        }
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
                    let mut data = output.data.take().unwrap();

                    // NOTE: If `finish_reason.is_some()`, then one of the stop conditions was triggered
                    // by the token generation. We should update the `data.finish_reason` in that case.
                    // However, if `finish_reason.is_none()`, it is possible that we are in the case where
                    // `data.token_ids` is empty, and `data.finish_reason` is already correctly set.
                    // In that case, `process_token_ids` above will rewrite `finish_reason` to `None`,
                    // which we don't want to propagate to `data.finish_reason`.
                    if finish_reason.is_some() {
                        data.finish_reason = finish_reason;
                        data.stop_reason = stop_reason.or(data.stop_reason);
                    }
                    data.text = text;
                    data.tokens = Some(tokens);
                    // Snapshot of whatever this choice's decoder is still withholding as a
                    // possible hidden-stop-sequence prefix after this step -- `None` once
                    // resolved (matched, ruled out, or flushed). Carried so a migration
                    // retry's fresh decoder can be reseeded from the last known-good chunk
                    // instead of silently losing it (see `PreprocessedRequest::jail_seed`).
                    // Skipped when migration can't happen: nothing will ever read this
                    // checkpoint, so there is no reason to allocate a snapshot of it.
                    if state.migration_possible {
                        data.jailed_text = decoder.peek_jailed();
                    }

                    output.data = Some(data);

                    Some((output, state))
                }

                None => {
                    // The engine's stream ended without ever sending a terminal
                    // `finish_reason` to some choice -- no more tokens are coming for any
                    // of them, so flush whatever each decoder still holds back as an
                    // incomplete hidden-stop-sequence match before the decoders are
                    // dropped, and drain the flushed choices one synthetic chunk at a time.
                    state.stream_ended = true;
                    for (idx, decoder) in state.decoders.iter_mut() {
                        // A choice that already finished -- successfully or with an error --
                        // was already given its chance to flush (see the decoder-driven and
                        // decoded-text-passthrough paths above). Synthesizing another chunk
                        // for it here would, at best, be redundant and, at worst, turn a
                        // choice that ended in an error into a spurious extra `Stop`.
                        if state.finished_choices.contains(idx) {
                            continue;
                        }
                        if let Some(flushed) = decoder.flush_jailed() {
                            state.pending_flush.push((*idx, flushed));
                        }
                    }
                    state.pending_flush.pop().map(|(idx, flushed)| {
                        let output = Annotated::from_data(LLMEngineOutput {
                            index: Some(idx),
                            text: Some(flushed),
                            finish_reason: Some(FinishReason::Stop),
                            ..Default::default()
                        });
                        (output, state)
                    })
                }
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
                    encoder_result: data.encoder_result,
                    worker_trace_link: data.worker_trace_link,
                    engine_data: data.engine_data,
                    routing_data: data.routing_data,
                    jailed_text: data.jailed_text,
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

    // single tokens that if found in the response will trigger a stop condition and be returned
    visible_stop_ids: HashSet<TokenIdType>,

    // user-provided token stop IDs, kept separate from system/EOS stop IDs so
    // stop_reason can report user-triggered token stops without reporting EOS.
    user_stop_ids: HashSet<TokenIdType>,

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

    // Scratch buffers reused across `longest_hidden_prefix_suffix` calls (one per decoded
    // token, for the lifetime of the request) instead of allocating fresh ones every step.
    kmp_scratch: Vec<u8>,
    kmp_pi_scratch: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum StopTrigger {
    MaxTokensLimit,
    HiddenStopTokenDetected(TokenIdType),
    UserStopTokenDetected(TokenIdType),
    VisibleStopTokenDetected(TokenIdType),
    HiddenStopSequenceDetected(String),
    VisibleStopSequenceDetected(String),
}

pub struct StepResult {
    /// This step's own decoded text, independent of any stop-sequence withholding.
    /// `SeqResult.tokens[i]` must equal the text of `token_ids[i]` so that consumers
    /// zipping `tokens` with per-token logprobs (e.g. `create_logprobs`) stay aligned --
    /// logprobs describe what the model generated and must not change because a hidden
    /// stop sequence also happens to be in flight.
    pub token: Option<String>,
    /// Text to append to the caller-visible `text`/content this step. May be `None` on a
    /// step whose text is being withheld as a possible hidden-stop-sequence prefix, and may
    /// carry more than one prior step's worth of text on the step that releases a withheld
    /// backlog.
    pub released_text: Option<String>,
    pub stop_trigger: Option<StopTrigger>,
}

impl StepResult {
    /// No stop-sequence withholding in effect: the token's own text is released as-is.
    fn ok(token: Option<String>) -> Self {
        Self {
            token: token.clone(),
            released_text: token,
            stop_trigger: None,
        }
    }

    /// `token` and `released_text` differ, e.g. because `released_text` also carries a
    /// previously withheld backlog, or because a matched stop sequence is excluded from
    /// `released_text` but not from `token`.
    fn ok_split(token: Option<String>, released_text: Option<String>) -> Self {
        Self {
            token,
            released_text,
            stop_trigger: None,
        }
    }

    fn with_stop_trigger(
        token: Option<String>,
        released_text: Option<String>,
        stop_trigger: StopTrigger,
    ) -> Self {
        Self {
            token,
            released_text,
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
        // Withheld hidden-stop-sequence prefix to resume from, e.g. after a migration
        // retry (see `PreprocessedRequest::jail_seed`). `None` starts unseeded, as before.
        jail_seed: Option<String>,
    ) -> Self {
        let user_stop_ids: HashSet<TokenIdType> = stop_condition
            .stop_token_ids
            .unwrap_or_default()
            .iter()
            .copied()
            .collect();
        let system_stop_ids: HashSet<TokenIdType> = stop_condition
            .stop_token_ids_hidden
            .unwrap_or_default()
            .iter()
            .copied()
            .collect();
        let visible_stop_ids: HashSet<TokenIdType> = stop_condition
            .stop_token_ids_visible
            .unwrap_or_default()
            .iter()
            .copied()
            .collect();
        let hidden_stop_ids = user_stop_ids.union(&system_stop_ids).copied().collect();

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

        // The entire seed is, by construction, text a prior attempt withheld as a partial
        // hidden-stop-sequence match that had not yet resolved -- treat all of it as still
        // jailed so this attempt can either complete the match or release it exactly as the
        // original attempt would have, rather than leaking it or re-checking it as new text.
        let jail = jail_seed.unwrap_or_default();
        let jailed_bytes = jail.len();

        Self {
            decode_stream,
            tracker,
            hidden_stop_ids,
            visible_stop_ids,
            user_stop_ids,
            hidden_stop_sequences,
            visible_stop_sequences,
            min_tokens: stop_condition.min_tokens.unwrap_or(0),
            generated_tokens: 0,
            jail,
            jail_max_bytes,
            jailed_bytes,
            kmp_scratch: Vec::new(),
            kmp_pi_scratch: Vec::new(),
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
        let detokenize_start = self.tracker.as_ref().map(|_| Instant::now());
        let token = {
            let _nvtx = dynamo_nvtx_range!("detokenize");
            self.decode_stream.step(token_id)?
        };
        if let (Some(start), Some(tracker)) = (detokenize_start, &self.tracker) {
            tracker.record_detokenize_latency(start.elapsed());
        }

        // stop conditions to not apply until the minimum number of tokens have been generated
        if self.generated_tokens < self.min_tokens {
            return Ok(StepResult::ok(token));
        }

        // Check token stops. Visible token IDs are included in output.
        if self.visible_stop_ids.contains(&token_id) {
            // Any text still withheld as a partial hidden-stop-sequence match can never
            // complete now, so it goes out ahead of this (included) token's own text --
            // otherwise it would be silently dropped and, if it were released later, would
            // come out of order.
            let released = match (self.flush_jailed(), &token) {
                (Some(mut flushed), Some(t)) => {
                    flushed.push_str(t);
                    Some(flushed)
                }
                (Some(flushed), None) => Some(flushed),
                (None, t) => t.clone(),
            };
            return Ok(StepResult::with_stop_trigger(
                token,
                released,
                StopTrigger::VisibleStopTokenDetected(token_id),
            ));
        }

        // Check token stops. User-provided token IDs take precedence over
        // system/EOS IDs so stop_reason only reports stops the caller requested.
        if self.hidden_stop_ids.contains(&token_id) {
            let trigger = if self.user_stop_ids.contains(&token_id) {
                StopTrigger::UserStopTokenDetected(token_id)
            } else {
                StopTrigger::HiddenStopTokenDetected(token_id)
            };
            // This token's own text stays hidden, as before. But any text still withheld
            // from an earlier, never-completed hidden-stop-sequence prefix match is not
            // part of *this* stop and must still reach the caller.
            return Ok(StepResult::with_stop_trigger(
                None,
                self.flush_jailed(),
                trigger,
            ));
        }

        // check stop sequences - the jail will always hold at least the largest stop sequence
        // if jail_max_bytes is 0, then there are no stop sequences
        if self.jail_max_bytes > 0
            && let Some(token_text) = &token
        {
            // bytes at the tail of `jail` withheld by a previous step because they could
            // still grow into a complete hidden stop sequence; everything before this point
            // has already been released to the caller.
            let release_start = self.jail.len() - self.jailed_bytes;
            self.jail.push_str(token_text);

            // Check hidden stop sequences first (excluded from output)
            for seq in &self.hidden_stop_sequences {
                if let Some(offset) = galil_seiferas::gs_find(self.jail.as_bytes(), seq.as_bytes())
                {
                    // return only new bytes after release_start .. offset (excluding stop sequence)
                    // example: seq = "ox", token = "boxes", return "b"
                    //
                    // we might have returned a partial match, if so, then offset < release_start
                    // in that case, we return no text
                    let partial_token = (offset >= release_start)
                        .then(|| self.jail[release_start..offset].to_string())
                        .filter(|s| !s.is_empty());
                    self.jailed_bytes = 0;
                    // `token` (this step's own raw decoded text) is reported unchanged so
                    // that SeqResult.tokens[i] keeps describing token_ids[i] for logprobs;
                    // only the caller-visible `released_text` excludes the matched sequence.
                    // `token_text`'s last use was the `push_str` above, so `token` itself
                    // (not yet borrowed at this point) can move here instead of cloning.
                    return Ok(StepResult::with_stop_trigger(
                        token,
                        partial_token,
                        StopTrigger::HiddenStopSequenceDetected(seq.to_string()),
                    ));
                }
            }

            // Check visible stop sequences (included in output)
            for seq in &self.visible_stop_sequences {
                if let Some(offset) = galil_seiferas::gs_find(self.jail.as_bytes(), seq.as_bytes())
                {
                    // For visible stop sequences, include the stop string in the output
                    // Return all text from release_start up to and including the stop sequence
                    let stop_end = offset + seq.len();
                    let token_with_stop = (stop_end > release_start)
                        .then(|| self.jail[release_start..stop_end].to_string())
                        .filter(|s| !s.is_empty());
                    self.jailed_bytes = 0;
                    // Same reasoning as the hidden-sequence branch above: `token` can move
                    // here instead of cloning.
                    return Ok(StepResult::with_stop_trigger(
                        token,
                        token_with_stop,
                        StopTrigger::VisibleStopSequenceDetected(seq.to_string()),
                    ));
                }
            }

            // No complete match. Withhold the longest tail of `jail` that is still a viable
            // prefix of some hidden stop sequence -- a later token could complete it into a
            // full match. Everything else since the last release is now safe to hand back:
            // it cannot be part of a hidden stop sequence, complete or partial.
            self.jailed_bytes = self.longest_hidden_prefix_suffix();
            let release_end = self.jail.len() - self.jailed_bytes;
            let released = (release_end > release_start)
                .then(|| self.jail[release_start..release_end].to_string());

            Self::maybe_drain_to_max_bytes(&mut self.jail, self.jail_max_bytes);
            // `token` is this step's own raw decoded text (for logprobs); `released` is
            // what, if anything, newly clears the jail this step (for `text`/content).
            return Ok(StepResult::ok_split(token, released));
        }

        Ok(StepResult::ok(token))
    }

    /// Releases any text still withheld as a partial hidden-stop-sequence match. Call this
    /// once it is known no further tokens are coming (e.g. the engine reports completion
    /// without `Decoder` having detected a local stop condition) so a stop sequence that
    /// never completed is not silently dropped from the output.
    pub fn flush_jailed(&mut self) -> Option<String> {
        let flushed = self.jailed_string();
        self.jailed_bytes = 0;
        flushed
    }

    /// Non-consuming look at whatever is currently withheld as a possible hidden-stop-
    /// sequence prefix, without releasing it. Unlike [`Self::flush_jailed`], this does not
    /// end the withholding -- it exists so a caller (e.g. the streaming pipeline) can
    /// snapshot the in-flight jail state onto each chunk, so it can be recovered and used to
    /// reseed a fresh `Decoder` (via `jail_seed` in [`Self::new`]) if this attempt is
    /// abandoned partway through, e.g. by a migration retry.
    pub(crate) fn peek_jailed(&self) -> Option<String> {
        self.jailed_string()
    }

    /// Returns the length, in bytes, of the longest suffix of `self.jail` that is also a
    /// strict prefix of some hidden stop sequence -- text that might still grow into a
    /// complete hidden stop sequence and so must not be released to the caller yet. Only
    /// considers byte offsets that land on a `jail` char boundary, so the result is always
    /// safe to slice with. Reuses `self.kmp_scratch`/`self.kmp_pi_scratch` across calls
    /// (one per decoded token) instead of allocating fresh buffers every step.
    fn longest_hidden_prefix_suffix(&mut self) -> usize {
        let jail_bytes = self.jail.as_bytes();
        let mut best = 0;
        for seq in &self.hidden_stop_sequences {
            let seq_bytes = seq.as_bytes();
            // A full-length match would already have been caught as a complete stop;
            // only strictly shorter prefixes are candidates here. Sequences that cannot
            // possibly beat the current best are skipped outright.
            let max_k = seq_bytes.len().saturating_sub(1).min(jail_bytes.len());
            if max_k <= best {
                continue;
            }
            let pattern = &seq_bytes[..max_k];
            let tail_len = pattern.len().min(jail_bytes.len());
            let tail = &jail_bytes[jail_bytes.len() - tail_len..];

            // Longest suffix of `tail` that is also a prefix of `pattern`, found in
            // O(pattern.len() + tail.len()) via the standard KMP-border trick: build the
            // prefix (failure) function of `pattern + sep + tail` (`sep` = 0xFF, which
            // cannot occur in valid UTF-8, so no border can bridge across it) and read the
            // border length back from its last entry. Borders of `pattern + sep + tail`
            // strictly decrease along the classic `pi[k - 1]` chain, which is walked here
            // only as far as needed to find one that also lands on a `jail` char boundary.
            self.kmp_scratch.clear();
            self.kmp_scratch.extend_from_slice(pattern);
            self.kmp_scratch.push(0xFF);
            self.kmp_scratch.extend_from_slice(tail);
            Self::kmp_prefix_function_into(&self.kmp_scratch, &mut self.kmp_pi_scratch);

            let mut k = *self.kmp_pi_scratch.last().unwrap_or(&0);
            while k > best {
                if self.jail.is_char_boundary(jail_bytes.len() - k) {
                    best = k;
                    break;
                }
                k = self.kmp_pi_scratch[k - 1];
            }
        }
        best
    }

    /// Standard KMP prefix (failure) function, written into `pi` (cleared and reused
    /// rather than allocated fresh every call): `pi[i]` is the length of the longest
    /// proper prefix of `s[..=i]` that is also a suffix of it.
    fn kmp_prefix_function_into(s: &[u8], pi: &mut Vec<usize>) {
        pi.clear();
        pi.resize(s.len(), 0);
        let mut k = 0usize;
        for i in 1..s.len() {
            while k > 0 && s[i] != s[k] {
                k = pi[k - 1];
            }
            if s[i] == s[k] {
                k += 1;
            }
            pi[i] = k;
        }
    }

    pub fn process_token_ids(&mut self, token_ids: &[TokenIdType]) -> Result<SeqResult> {
        let mut text: Option<String> = None;
        let mut tokens = Vec::with_capacity(token_ids.len());

        for token_id in token_ids {
            let StepResult {
                token,
                released_text,
                stop_trigger,
            } = self.step(*token_id)?;

            // `text` accumulates the caller-visible content, which can lag behind and later
            // release more than one step's worth at once. `tokens[i]` always reports
            // token_ids[i]'s own decoded text, independent of that withholding, so per-token
            // consumers (logprobs) stay aligned with token_ids.
            if let Some(released_text) = &released_text {
                text.get_or_insert_with(|| String::with_capacity(token_ids.len()))
                    .push_str(released_text);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::llm_backend::TopLogprob;
    use crate::protocols::common::{OutputOptions, SamplingOptions};
    use crate::protocols::openai::{
        DeltaGeneratorExt, chat_completions::DeltaGenerator, delta_common::DeltaGeneratorOptions,
    };
    use crate::tokenizers::traits;
    use dynamo_runtime::pipeline::{AsyncEngine, Error, ResponseStream};
    use futures::StreamExt;
    use std::sync::Arc;

    #[test]
    fn test_char_boundary_drain() {
        let mut s = String::from("helloñworld"); // 12 bytes total ñ is 2 bytes
        let max_bytes = 6; // 12 - 6 = 6 which is inside ñ
        assert!(!s.is_char_boundary(s.len() - max_bytes)); // initially we are not on a char boundary
        Decoder::maybe_drain_to_max_bytes(&mut s, max_bytes);
        assert!(s.is_char_boundary(0)); // front of jail string on valid char boundary
        assert_eq!(s, "ñworld");
    }

    /// A mock tokenizer that always returns Err from decode().
    /// Used to test the error propagation path in Decoder::process_token_ids().
    struct FailingDecoder;

    impl traits::Encoder for FailingDecoder {
        fn encode(&self, _input: &str) -> anyhow::Result<crate::tokenizers::Encoding> {
            Ok(crate::tokenizers::Encoding::Sp(vec![]))
        }
        fn encode_batch(
            &self,
            _inputs: &[&str],
        ) -> anyhow::Result<Vec<crate::tokenizers::Encoding>> {
            Ok(vec![])
        }
    }

    impl traits::Decoder for FailingDecoder {
        fn decode(
            &self,
            _token_ids: &[TokenIdType],
            _skip_special_tokens: bool,
        ) -> anyhow::Result<traits::DecodeResult> {
            Err(anyhow::anyhow!(
                "Unable to decode into a valid UTF-8 string: incomplete utf-8 byte sequence from index 6"
            ))
        }
    }

    impl traits::Tokenizer for FailingDecoder {}

    struct CandidateDecoder;

    impl traits::Encoder for CandidateDecoder {
        fn encode(&self, _input: &str) -> anyhow::Result<crate::tokenizers::Encoding> {
            Ok(crate::tokenizers::Encoding::Sp(vec![]))
        }

        fn encode_batch(
            &self,
            _inputs: &[&str],
        ) -> anyhow::Result<Vec<crate::tokenizers::Encoding>> {
            Ok(vec![])
        }
    }

    impl traits::Decoder for CandidateDecoder {
        fn decode(
            &self,
            token_ids: &[TokenIdType],
            _skip_special_tokens: bool,
        ) -> anyhow::Result<traits::DecodeResult> {
            let token = match token_ids {
                [] => "",
                [101] => "Okay",
                [102] => " Okay",
                [103] => "<|user|>",
                [101, 103] => "Okay<|user|>",
                _ => anyhow::bail!("unexpected token IDs: {token_ids:?}"),
            };
            Ok(traits::DecodeResult::Complete(token.to_string()))
        }
    }

    impl traits::Tokenizer for CandidateDecoder {}

    struct SyntheticSglangEngine {
        engine_decodes_text: bool,
    }

    struct SyntheticSglangStopEngine;

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for SyntheticSglangStopEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let output = LLMEngineOutput {
                // SGLang's output_ids_through_stop() includes the matched stop
                // position. The frontend must hide its text without mutating
                // this raw token sequence.
                token_ids: vec![101, 103],
                finish_reason: Some(FinishReason::Stop),
                index: Some(0),
                ..Default::default()
            };

            Ok(ResponseStream::new(
                Box::pin(futures::stream::once(async move {
                    Annotated::from_data(output)
                })),
                request.context(),
            ))
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for SyntheticSglangEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let output = LLMEngineOutput {
                token_ids: vec![101],
                tokens: self
                    .engine_decodes_text
                    .then(|| vec![Some("Okay".to_string())]),
                text: self.engine_decodes_text.then(|| "Okay".to_string()),
                log_probs: Some(vec![-0.125]),
                top_logprobs: Some(vec![vec![
                    TopLogprob {
                        rank: 1,
                        token_id: 101,
                        token: None,
                        logprob: -0.125,
                        bytes: None,
                    },
                    TopLogprob {
                        rank: 2,
                        token_id: 102,
                        token: None,
                        logprob: -1.5,
                        bytes: None,
                    },
                ]]),
                index: Some(0),
                ..Default::default()
            };

            Ok(ResponseStream::new(
                Box::pin(futures::stream::once(async move {
                    Annotated::from_data(output)
                })),
                request.context(),
            ))
        }
    }

    #[test]
    fn test_fill_missing_top_logprob_text_without_worker_tokenizer() {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(CandidateDecoder);
        let tokenizer = Tokenizer::from(tokenizer);
        let mut top_logprobs = vec![vec![
            TopLogprob {
                rank: 1,
                token_id: 101,
                token: None,
                logprob: -0.1,
                bytes: None,
            },
            TopLogprob {
                rank: 2,
                token_id: 102,
                token: None,
                logprob: -0.2,
                bytes: None,
            },
        ]];

        fill_missing_top_logprob_text(&tokenizer, &mut top_logprobs, true);

        assert_eq!(top_logprobs[0][0].token.as_deref(), Some("Okay"));
        assert_eq!(top_logprobs[0][0].bytes, Some(b"Okay".to_vec()));
        assert_eq!(top_logprobs[0][1].token.as_deref(), Some(" Okay"));
        assert_eq!(top_logprobs[0][1].bytes, Some(b" Okay".to_vec()));
        assert_eq!(top_logprobs[0][0].logprob, -0.1);
        assert_eq!(top_logprobs[0][1].logprob, -0.2);
    }

    async fn assert_sglang_top_logprobs_are_decoded_in_openai_response(engine_decodes_text: bool) {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(CandidateDecoder);
        let backend = Backend::from_tokenizer(Tokenizer::from(tokenizer));
        let request = PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions {
                logprobs: Some(2),
                ..Default::default()
            })
            .build()
            .expect("valid preprocessed request");
        let engine: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(SyntheticSglangEngine {
                engine_decodes_text,
            });

        let mut stream = Operator::generate(backend.as_ref(), SingleIn::new(request), engine)
            .await
            .expect("backend generation succeeds");
        let output = stream
            .next()
            .await
            .expect("backend emits a response")
            .data
            .expect("response contains backend output");

        let options = DeltaGeneratorOptions::new(None, None, true, None);
        let mut generator = DeltaGenerator::new(
            "test-model".to_string(),
            options,
            "test-request".to_string(),
        );
        let response = generator
            .choice_from_postprocessor(output)
            .expect("OpenAI response conversion succeeds");
        let content = response.inner.choices[0]
            .logprobs
            .as_ref()
            .expect("client-visible logprobs")
            .content
            .as_ref()
            .expect("client-visible logprob content");
        let candidates = &content[0].top_logprobs;

        assert_eq!(candidates[0].token, "Okay");
        assert_eq!(candidates[0].bytes, Some(b"Okay".to_vec()));
        assert_eq!(candidates[0].logprob, -0.125);
        assert_eq!(candidates[1].token, " Okay");
        assert_eq!(candidates[1].bytes, Some(b" Okay".to_vec()));
        assert_eq!(candidates[1].logprob, -1.5);
    }

    #[tokio::test]
    async fn test_sglang_top_logprobs_are_decoded_in_openai_response() {
        assert_sglang_top_logprobs_are_decoded_in_openai_response(false).await;
    }

    #[tokio::test]
    async fn test_sglang_top_logprobs_are_decoded_before_engine_text_fast_path() {
        assert_sglang_top_logprobs_are_decoded_in_openai_response(true).await;
    }

    #[tokio::test]
    async fn test_sglang_hidden_stop_text_is_suppressed_without_mutating_tito_ids() {
        use crate::protocols::common::extensions::NvExt;

        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(CandidateDecoder);
        let backend = Backend::from_tokenizer(Tokenizer::from(tokenizer));
        let request = PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![])
            .stop_conditions(StopConditions {
                stop_token_ids_hidden: Some(vec![103]),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .expect("valid preprocessed request");
        let engine: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(SyntheticSglangStopEngine);

        let mut stream = Operator::generate(backend.as_ref(), SingleIn::new(request), engine)
            .await
            .expect("backend generation succeeds");
        let output = stream
            .next()
            .await
            .expect("backend emits a response")
            .data
            .expect("response contains backend output");

        assert_eq!(output.text.as_deref(), Some("Okay"));
        assert_eq!(output.tokens, vec![Some("Okay".to_string()), None]);
        assert_eq!(output.token_ids, vec![101, 103]);
        assert_eq!(output.finish_reason, Some(FinishReason::Stop));

        let nvext = NvExt::builder()
            .extra_fields(vec!["completion_token_ids".to_string()])
            .build()
            .expect("valid nvext selection");
        let options = DeltaGeneratorOptions::new(None, None, false, Some(&nvext));
        let mut generator = DeltaGenerator::new(
            "test-model".to_string(),
            options,
            "test-request".to_string(),
        );
        let response = generator
            .choice_from_postprocessor(output)
            .expect("OpenAI response conversion succeeds");

        assert_eq!(
            response
                .nvext
                .as_ref()
                .and_then(|fields| fields.get("completion_token_ids")),
            Some(&serde_json::json!([101, 103]))
        );
        assert_eq!(generator.get_usage().completion_tokens, 2);
    }

    /// When the tokenizer's decode() returns Err, Decoder::process_token_ids()
    /// should propagate the error. In the backend unfold closure, this error
    /// gets caught and converted to FinishReason::Error.
    #[test]
    fn test_decoder_process_token_ids_propagates_decode_error() {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(FailingDecoder);
        let decode_stream = crate::tokenizers::DecodeStream::new(tokenizer, &[], false);
        let stop_conditions = StopConditions::default();

        let mut decoder = Decoder::new(decode_stream, stop_conditions, false, None, None);

        let result = decoder.process_token_ids(&[42]);
        assert!(
            result.is_err(),
            "process_token_ids should propagate decode errors"
        );

        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("incomplete utf-8 byte sequence"),
            "error should contain the original decode error message, got: {err_msg}"
        );
    }

    /// Verify that the error message format matches what the backend unfold
    /// closure would wrap into FinishReason::Error.
    #[test]
    fn test_decoder_error_message_format_for_finish_reason() {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(FailingDecoder);
        let decode_stream = crate::tokenizers::DecodeStream::new(tokenizer, &[], false);
        let stop_conditions = StopConditions::default();

        let mut decoder = Decoder::new(decode_stream, stop_conditions, false, None, None);

        let result = decoder.process_token_ids(&[42]);
        let err = result.err().expect("should be Err");

        // This is what the backend unfold closure does:
        let finish_reason = FinishReason::Error(format!("decode error: {err}"));
        match &finish_reason {
            FinishReason::Error(msg) => {
                assert!(
                    msg.starts_with("decode error:"),
                    "FinishReason::Error should have 'decode error:' prefix, got: {msg}"
                );
                assert!(
                    msg.contains("incomplete utf-8 byte sequence"),
                    "FinishReason::Error should contain original error, got: {msg}"
                );
            }
            other => panic!("Expected FinishReason::Error, got: {:?}", other),
        }
    }

    /// A tokenizer whose per-token fragments are chosen so `1` then `2` leaves a partial
    /// hidden-stop-sequence match ("STOP", a strict prefix of the configured stop
    /// sequence "STOPPED") jailed, and `99` fails to decode -- used to exercise
    /// EOF-time flushing and error/EOF interaction at the `Backend` (not just `Decoder`)
    /// level, across multiple choices.
    struct JailingTokenizer;

    impl traits::Encoder for JailingTokenizer {
        fn encode(&self, _input: &str) -> anyhow::Result<crate::tokenizers::Encoding> {
            Ok(crate::tokenizers::Encoding::Sp(vec![]))
        }
        fn encode_batch(
            &self,
            _inputs: &[&str],
        ) -> anyhow::Result<Vec<crate::tokenizers::Encoding>> {
            Ok(vec![])
        }
    }

    impl traits::Decoder for JailingTokenizer {
        // `DecodeStream::step` (the real, pinned `dynamo-tokenizers` implementation) does
        // not call `decode` with a single fresh id at a time: it calls it once with the
        // previously-read slice and once with that slice plus the newest id, then diffs the
        // two to find the newly emitted text. Both calls must therefore accept *any*
        // contextual slice of already-seen ids, not just a single one, so this decodes
        // compositionally (concatenating each id's own fragment) instead of matching whole
        // slices -- except token 99, which fails deliberately wherever it appears, to give
        // tests a controlled decode-error boundary.
        fn decode(
            &self,
            token_ids: &[TokenIdType],
            _skip_special_tokens: bool,
        ) -> anyhow::Result<traits::DecodeResult> {
            if token_ids.contains(&99) {
                anyhow::bail!("simulated decode failure");
            }
            let mut text = String::new();
            for &id in token_ids {
                match id {
                    1 => text.push_str("abc"),
                    2 => text.push_str("STOP"),
                    other => anyhow::bail!("unexpected token id: {other}"),
                }
            }
            Ok(traits::DecodeResult::Complete(text))
        }
    }

    impl traits::Tokenizer for JailingTokenizer {}

    fn jailing_request(n: u8) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![])
            .stop_conditions(StopConditions {
                stop: Some(vec!["STOPPED".to_string()]),
                ..Default::default()
            })
            .sampling_options(SamplingOptions {
                n: Some(n),
                ..Default::default()
            })
            .output_options(OutputOptions::default())
            .build()
            .expect("valid preprocessed request")
    }

    /// Wraps an inner stream and flips a shared flag if it is ever polled again after
    /// already returning `None` once -- used to verify the backend's `stream_ended` guard
    /// actually prevents re-polling `stream`, which most `Stream` impls do not guarantee
    /// is safe.
    struct EndGuardStream {
        inner: std::pin::Pin<Box<dyn futures::Stream<Item = Annotated<LLMEngineOutput>> + Send>>,
        ended: bool,
        overpolled: Arc<std::sync::atomic::AtomicBool>,
    }

    impl futures::Stream for EndGuardStream {
        type Item = Annotated<LLMEngineOutput>;

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            if self.ended {
                self.overpolled
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            }
            let poll = self.inner.as_mut().poll_next(cx);
            if matches!(poll, std::task::Poll::Ready(None)) {
                self.ended = true;
            }
            poll
        }
    }

    struct BareEofMultiChoiceEngine {
        overpolled: Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for BareEofMultiChoiceEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            // Both choices withhold "STOP" as a never-completed partial match against
            // "STOPPED", then the stream ends with no `finish_reason` ever sent for
            // either -- a "bare EOF", as from an engine that simply drops the connection.
            let chunks = vec![
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![1],
                    index: Some(0),
                    ..Default::default()
                }),
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![2],
                    index: Some(0),
                    ..Default::default()
                }),
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![1],
                    index: Some(1),
                    ..Default::default()
                }),
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![2],
                    index: Some(1),
                    ..Default::default()
                }),
            ];
            let guarded = EndGuardStream {
                inner: Box::pin(futures::stream::iter(chunks)),
                ended: false,
                overpolled: self.overpolled.clone(),
            };
            Ok(ResponseStream::new(Box::pin(guarded), request.context()))
        }
    }

    #[tokio::test]
    async fn backend_flushes_multiple_jailed_choices_on_bare_eof_without_repolling() {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(JailingTokenizer);
        let backend = Backend::from_tokenizer(Tokenizer::from(tokenizer));
        let request = jailing_request(2);
        let overpolled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let engine: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(BareEofMultiChoiceEngine {
                overpolled: overpolled.clone(),
            });

        let stream = Operator::generate(backend.as_ref(), SingleIn::new(request), engine)
            .await
            .expect("backend generation succeeds");
        let outputs: Vec<_> = stream.collect().await;

        // 2 normal chunks per choice (releasing "abc", withholding "STOP") + one
        // synthetic EOF-flush chunk per choice releasing the withheld "STOP".
        assert_eq!(outputs.len(), 6, "unexpected output count: {outputs:?}");

        let mut flushed_by_index = std::collections::HashMap::new();
        for output in &outputs {
            let data = output.data.as_ref().expect("every chunk carries data");
            if data.finish_reason.is_some() {
                flushed_by_index.insert(data.index, data.text.clone());
            }
        }
        assert_eq!(
            flushed_by_index.get(&Some(0)).cloned().flatten().as_deref(),
            Some("STOP"),
            "choice 0's withheld text must be flushed on bare EOF"
        );
        assert_eq!(
            flushed_by_index.get(&Some(1)).cloned().flatten().as_deref(),
            Some("STOP"),
            "choice 1's withheld text must be flushed on bare EOF"
        );

        assert!(
            !overpolled.load(std::sync::atomic::Ordering::SeqCst),
            "backend must not poll the underlying stream again after it yields None"
        );
    }

    struct ErrorThenBareEofEngine;

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for ErrorThenBareEofEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            // Choice 0 withholds "STOP" (same as choice 1) and *then* hits a decode
            // error -- so it still has jailed text outstanding at the moment it
            // terminates via the error path rather than a real stop. Choice 1 withholds
            // "STOP" too but never gets a `finish_reason` -- the stream just ends (bare
            // EOF). Without excluding already-finished choices from the EOF flush, choice
            // 0's leftover jailed text would be flushed into a second, spurious `Stop`
            // chunk after its error.
            let chunks = vec![
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![1],
                    index: Some(0),
                    ..Default::default()
                }),
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![2],
                    index: Some(0),
                    ..Default::default()
                }),
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![99],
                    index: Some(0),
                    ..Default::default()
                }),
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![1],
                    index: Some(1),
                    ..Default::default()
                }),
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![2],
                    index: Some(1),
                    ..Default::default()
                }),
            ];
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter(chunks)),
                request.context(),
            ))
        }
    }

    /// A choice that already ended in a decode error must not be given a second,
    /// synthetic `Stop` chunk by the bare-EOF flush loop -- that would turn a genuine
    /// error into what looks like a normal completion downstream.
    #[tokio::test]
    async fn backend_does_not_synthesize_stop_after_choice_decode_error() {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(JailingTokenizer);
        let backend = Backend::from_tokenizer(Tokenizer::from(tokenizer));
        let request = jailing_request(2);
        let engine: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(ErrorThenBareEofEngine);

        let stream = Operator::generate(backend.as_ref(), SingleIn::new(request), engine)
            .await
            .expect("backend generation succeeds");
        let outputs: Vec<_> = stream.collect().await;

        let choice0_finishes: Vec<_> = outputs
            .iter()
            .filter_map(|o| o.data.as_ref())
            .filter(|d| d.index == Some(0) && d.finish_reason.is_some())
            .collect();
        assert_eq!(
            choice0_finishes.len(),
            1,
            "choice 0 must finish exactly once (its decode error), not again via EOF flush: {choice0_finishes:?}"
        );
        assert!(
            matches!(
                choice0_finishes[0].finish_reason,
                Some(FinishReason::Error(_))
            ),
            "choice 0's only finish must be its decode error, got: {:?}",
            choice0_finishes[0].finish_reason
        );

        let choice1_flush = outputs
            .iter()
            .filter_map(|o| o.data.as_ref())
            .find(|d| d.index == Some(1) && d.finish_reason.is_some());
        assert_eq!(
            choice1_flush.and_then(|d| d.text.as_deref()),
            Some("STOP"),
            "choice 1's withheld text must still be flushed on the same bare EOF"
        );
    }

    /// A tokenizer whose only token (`1`) decodes to `"o"` -- paired with hidden stop `"ozzy"`,
    /// this lets a single decoder-driven chunk withhold "o" before later chunks switch to the
    /// engine-pre-decoded-text fast path, which never calls back into this tokenizer.
    struct SingleOTokenizer;

    impl traits::Encoder for SingleOTokenizer {
        fn encode(&self, _input: &str) -> anyhow::Result<crate::tokenizers::Encoding> {
            Ok(crate::tokenizers::Encoding::Sp(vec![]))
        }
        fn encode_batch(
            &self,
            _inputs: &[&str],
        ) -> anyhow::Result<Vec<crate::tokenizers::Encoding>> {
            Ok(vec![])
        }
    }

    impl traits::Decoder for SingleOTokenizer {
        fn decode(
            &self,
            token_ids: &[TokenIdType],
            _skip_special_tokens: bool,
        ) -> anyhow::Result<traits::DecodeResult> {
            Ok(traits::DecodeResult::Complete(
                token_ids.iter().map(|_| "o").collect(),
            ))
        }
    }

    impl traits::Tokenizer for SingleOTokenizer {}

    struct MixedBypassEngine;

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for MixedBypassEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let chunks = vec![
                // Token-only: goes through the decoder, which withholds "o" as a viable
                // prefix of the hidden stop "ozzy".
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![1],
                    index: Some(0),
                    ..Default::default()
                }),
                // Engine-pre-decoded, non-terminal: takes the fast bypass path entirely
                // (never touches the decoder to produce this text), but must not make the
                // withheld "o" look resolved.
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![],
                    index: Some(0),
                    text: Some(String::new()),
                    ..Default::default()
                }),
                // Engine-pre-decoded, terminal: the withheld "o" must be flushed ahead of
                // this chunk's own (newer) text, not appended after it.
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![],
                    index: Some(0),
                    text: Some("there".to_string()),
                    finish_reason: Some(FinishReason::Length),
                    ..Default::default()
                }),
            ];
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter(chunks)),
                request.context(),
            ))
        }
    }

    /// Regression test for two related bugs in the engine-pre-decoded-text fast path: it
    /// must not silently erase a checkpoint of the decoder's still-withheld text just
    /// because a given chunk skipped the decoder (P2, `jailed_text` erasure), and when it
    /// does flush withheld text into a terminal chunk, that older text must come first, not
    /// be appended after the chunk's own (newer) text (P2, terminal ordering).
    #[tokio::test]
    async fn backend_bypass_path_preserves_and_orders_withheld_text() {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(SingleOTokenizer);
        let backend = Backend::from_tokenizer(Tokenizer::from(tokenizer));
        let request = PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![])
            .stop_conditions(StopConditions {
                stop: Some(vec!["ozzy".to_string()]),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            // `jailed_text` is only populated when migration is possible (see
            // `DecoderParams::migration_possible`); this test asserts on it.
            .migration_state(Some(
                crate::protocols::common::preprocessor::MigrationState::default(),
            ))
            .build()
            .expect("valid preprocessed request");
        let engine: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(MixedBypassEngine);

        let stream = Operator::generate(backend.as_ref(), SingleIn::new(request), engine)
            .await
            .expect("backend generation succeeds");
        let outputs: Vec<_> = stream.collect().await;
        assert_eq!(outputs.len(), 3, "unexpected output count: {outputs:?}");

        let data: Vec<_> = outputs
            .iter()
            .map(|o| o.data.as_ref().expect("every chunk carries data"))
            .collect();

        assert_eq!(
            data[0].jailed_text.as_deref(),
            Some("o"),
            "the decoder-driven chunk must publish what it withheld"
        );

        // The non-terminal bypass chunk did not touch the decoder, so the checkpoint must
        // still reflect the still-withheld "o" -- not look resolved just because this
        // particular chunk skipped the decoder.
        assert_eq!(
            data[1].jailed_text.as_deref(),
            Some("o"),
            "a non-terminal bypass chunk must not erase the withheld-text checkpoint"
        );
        assert_eq!(data[1].text.as_deref(), Some(""));

        // The terminal bypass chunk flushes the withheld "o" ahead of its own "there",
        // producing "othere" -- not "thereo" -- and resolves the checkpoint.
        assert_eq!(
            data[2].text.as_deref(),
            Some("othere"),
            "withheld text must be flushed before, not after, this chunk's own newer text"
        );
        assert_eq!(data[2].finish_reason, Some(FinishReason::Length));
        assert_eq!(data[2].jailed_text, None);
    }

    struct AnnotatedErrorThenBareEofEngine;

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for AnnotatedErrorThenBareEofEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            // Withholds "STOP" (same jailing setup as the other Backend-level tests), then
            // the stream reports a request-level error via `Annotated::from_error` --
            // unlike a decode failure, this carries no `data` and so no per-choice
            // `index` -- before ending.
            let chunks = vec![
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![1],
                    index: Some(0),
                    ..Default::default()
                }),
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![2],
                    index: Some(0),
                    ..Default::default()
                }),
                Annotated::from_error("engine reported a fatal error"),
            ];
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter(chunks)),
                request.context(),
            ))
        }
    }

    /// Regression test for the `Annotated`-error counterpart of
    /// `backend_does_not_synthesize_stop_after_choice_decode_error`: a request-level error
    /// annotation (no `data`, hence no per-choice `index`) must also mark every choice
    /// finished, not just choices that fail through a decode error, so the bare-EOF flush
    /// loop does not turn it into a spurious successful `Stop` afterward.
    #[tokio::test]
    async fn backend_does_not_synthesize_stop_after_annotated_error() {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(JailingTokenizer);
        let backend = Backend::from_tokenizer(Tokenizer::from(tokenizer));
        let request = jailing_request(1);
        let engine: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(AnnotatedErrorThenBareEofEngine);

        let stream = Operator::generate(backend.as_ref(), SingleIn::new(request), engine)
            .await
            .expect("backend generation succeeds");
        let outputs: Vec<_> = stream.collect().await;

        let finishes: Vec<_> = outputs.iter().filter(|o| o.is_error()).collect();
        assert_eq!(
            finishes.len(),
            1,
            "the error must be reported exactly once, not again via EOF flush: {outputs:?}"
        );

        let synthetic_stops: Vec<_> = outputs
            .iter()
            .filter(|o| {
                o.data
                    .as_ref()
                    .is_some_and(|d| d.finish_reason == Some(FinishReason::Stop))
            })
            .collect();
        assert!(
            synthetic_stops.is_empty(),
            "no choice may receive a synthetic successful Stop after a request-level error: {outputs:?}"
        );
    }
}
