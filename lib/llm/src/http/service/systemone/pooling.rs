// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP preflight and dispatch for noncausal pooling-classifier decisions (`ScoringBackend::VllmPooling`).

use std::sync::Arc;
use std::time::Instant;

use axum::{http::StatusCode, response::Response};
use dynamo_decisions::{CanonicalQuestion, CanonicalRequest, QuestionKind, QuestionOutcome, Usage};
use dynamo_http_server::metrics::{Metrics, SystemOnePhase};
use dynamo_runtime::{
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    pipeline::Context,
};
use futures::{StreamExt, stream};
use tokio::sync::OwnedSemaphorePermit;

use super::{
    DispatchError, MAX_CONCURRENT_SIBLINGS, MAX_TOKENIZATION_TEXT_BYTES, add_input_tokens,
    backend_error, dispatch_error, finish_decision_response, stream_error,
};
use crate::discovery::{SystemOneEngine, SystemOneExecutionSelection};
use crate::http::service::decision_lifecycle::run_until_killed;
use crate::protocols::systemone::{
    PoolingDecisionConfig, build_pooling_request, extract_marker_logits,
    format_chunked_question_text, format_joint_user_prompt, format_pooling_state_text,
    reduce_head_logits,
};

pub(super) struct PreparedPoolingBatch {
    pub(super) question_indices: Vec<usize>,
    pub(super) token_ids: Vec<u32>,
    pub(super) markers: Vec<usize>,
}

struct PoolingPrep<'a> {
    request: &'a CanonicalRequest,
    selection: &'a SystemOneExecutionSelection,
    config: &'a PoolingDecisionConfig,
    marker_id: u32,
    max_seq_len: usize,
    max_per_batch: usize,
    remaining_encoding_bytes: usize,
    batches: Vec<PreparedPoolingBatch>,
}

impl<'a> PoolingPrep<'a> {
    fn prepare_chunked(
        &mut self,
        questions: &[&CanonicalQuestion],
    ) -> Result<(), (StatusCode, String)> {
        let separator = self.config.separator.as_deref().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "chunked pooling decision config missing separator".to_string(),
            )
        })?;
        let separator_id = self
            .selection
            .preprocessor
            .systemone_single_token_id(
                separator,
                "pooling decision separator must encode to a single token",
            )
            .map_err(map_render_error)?;
        let state_text = format_pooling_state_text(
            &self.request.input,
            self.config,
            self.selection.card.special_tokens(),
        );
        let (context_ids, ctx_bytes) = self
            .selection
            .preprocessor
            .render_systemone_chunked_pooling_context(
                &self.selection.canonical_model,
                &state_text,
                Some(&self.request.chat_template_kwargs),
                separator_id,
                self.remaining_encoding_bytes,
            )
            .map_err(map_render_error)?;
        self.remaining_encoding_bytes = self.remaining_encoding_bytes.saturating_sub(ctx_bytes);

        self.batches.reserve(questions.len());
        for question in questions {
            let question_text = format_chunked_question_text(
                question,
                self.request.dialect,
                self.config,
                self.selection.card.special_tokens(),
            )
            .map_err(|cause| (StatusCode::UNPROCESSABLE_ENTITY, cause.to_string()))?;
            let (q_ids, q_bytes) = self
                .selection
                .preprocessor
                .encode_systemone_chunked_pooling_question(
                    &question_text,
                    self.marker_id,
                    self.remaining_encoding_bytes,
                )
                .map_err(map_render_error)?;
            self.remaining_encoding_bytes = self.remaining_encoding_bytes.saturating_sub(q_bytes);

            let mut token_ids = Vec::with_capacity(context_ids.len() + q_ids.len());
            token_ids.extend_from_slice(&context_ids);
            token_ids.extend_from_slice(&q_ids);

            let marker_count = token_ids.iter().filter(|&&t| t == self.marker_id).count();
            if marker_count != 1 || token_ids.last() != Some(&self.marker_id) {
                return Err((
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!(
                        "readout markers ({marker_count}) did not match expected question after tokenization"
                    ),
                ));
            }
            if token_ids.len() > self.max_seq_len {
                return Err(overlength_error(
                    question,
                    token_ids.len(),
                    self.max_seq_len,
                ));
            }
            let marker_pos = token_ids.len() - 1;
            self.batches.push(PreparedPoolingBatch {
                question_indices: vec![question.ordinal],
                token_ids,
                markers: vec![marker_pos],
            });
        }
        Ok(())
    }

    fn bisect(&mut self, questions: &[&CanonicalQuestion]) -> Result<(), (StatusCode, String)> {
        if questions.len() > self.max_per_batch {
            let mid = questions.len() / 2;
            self.bisect(&questions[..mid])?;
            return self.bisect(&questions[mid..]);
        }
        let user_content = format_joint_user_prompt(
            &self.request.input,
            questions,
            self.request.dialect,
            self.config,
            self.selection.card.special_tokens(),
        )
        .map_err(|cause| (StatusCode::UNPROCESSABLE_ENTITY, cause.to_string()))?;

        let (token_ids, markers, encoding_bytes) = self
            .selection
            .preprocessor
            .render_systemone_pooling_prompt(
                &self.selection.canonical_model,
                &user_content,
                Some(&self.request.chat_template_kwargs),
                self.marker_id,
                questions.len(),
                self.remaining_encoding_bytes,
            )
            .map_err(map_render_error)?;
        self.remaining_encoding_bytes =
            self.remaining_encoding_bytes.saturating_sub(encoding_bytes);

        if token_ids.len() <= self.max_seq_len {
            self.batches.push(PreparedPoolingBatch {
                question_indices: questions.iter().map(|q| q.ordinal).collect(),
                token_ids,
                markers,
            });
            return Ok(());
        }
        if questions.len() == 1 {
            return Err(overlength_error(
                questions[0],
                token_ids.len(),
                self.max_seq_len,
            ));
        }
        let mid = questions.len() / 2;
        self.bisect(&questions[..mid])?;
        self.bisect(&questions[mid..])
    }
}

pub(super) fn prepare_batches(
    request: &CanonicalRequest,
    selection: &SystemOneExecutionSelection,
    max_input_tokens: usize,
) -> Result<Vec<PreparedPoolingBatch>, (StatusCode, String)> {
    let config = selection.pooling_config.as_ref().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "missing pooling decision config for selected worker".to_string(),
        )
    })?;
    let marker_id = selection
        .preprocessor
        .systemone_single_token_id(
            &config.marker,
            "pooling decision readout marker must encode to a single token",
        )
        .map_err(map_render_error)?;
    let max_seq_len = match (
        selection.card.effective_context_length() as usize,
        config.max_length,
    ) {
        (0, Some(limit)) => limit,
        (0, None) => max_input_tokens,
        (ctx, Some(limit)) => ctx.min(limit),
        (ctx, None) => ctx,
    };
    let questions: Vec<&CanonicalQuestion> = request.questions.iter().collect();
    let mut prep = PoolingPrep {
        request,
        selection,
        config,
        marker_id,
        max_seq_len,
        max_per_batch: config.max_questions.unwrap_or(usize::MAX),
        remaining_encoding_bytes: MAX_TOKENIZATION_TEXT_BYTES,
        batches: Vec::new(),
    };
    if config.is_chunked() {
        prep.prepare_chunked(&questions)?;
    } else {
        prep.bisect(&questions)?;
    }

    let mut total_input_tokens = 0_usize;
    for batch in &prep.batches {
        total_input_tokens =
            add_input_tokens(total_input_tokens, batch.token_ids.len(), max_input_tokens)
                .map_err(|message| (StatusCode::UNPROCESSABLE_ENTITY, message))?;
    }
    Ok(prep.batches)
}

fn map_render_error(cause: anyhow::Error) -> (StatusCode, String) {
    if let Some(oversized) = cause.downcast_ref::<crate::preprocessor::SystemOneEncodingTooLarge>()
    {
        return (StatusCode::UNPROCESSABLE_ENTITY, oversized.to_string());
    }
    if let Some(incompatible) =
        cause.downcast_ref::<crate::preprocessor::SystemOneModelIncompatible>()
    {
        return (StatusCode::UNPROCESSABLE_ENTITY, incompatible.0.to_string());
    }
    if dynamo_error::http::find_canonical_error_in_chain(cause.as_ref())
        .is_some_and(|e| e.class() == dynamo_runtime::error::ErrorClass::InvalidRequest)
    {
        return (StatusCode::UNPROCESSABLE_ENTITY, cause.to_string());
    }
    tracing::error!(error = %cause, "pooling decision prompt rendering failed");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal server error".to_string(),
    )
}

fn overlength_error(
    question: &CanonicalQuestion,
    tokens: usize,
    max_seq_len: usize,
) -> (StatusCode, String) {
    let qid = question.id.as_deref().unwrap_or("unnamed");
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        format!(
            "question {qid:?} needs {tokens} tokens with the state, exceeding the {max_seq_len}-token limit; inputs are never truncated"
        ),
    )
}

fn link_child_or_cancelled(
    parent: &Arc<dyn AsyncEngineContext>,
    child: Arc<dyn AsyncEngineContext>,
) -> Result<(), DispatchError> {
    parent.link_child(child.clone());
    if parent.is_killed() {
        child.kill();
        return Err(DispatchError::cancelled());
    }
    Ok(())
}

pub(super) async fn dispatch(
    selection: SystemOneExecutionSelection,
    batches: Vec<PreparedPoolingBatch>,
    request_id: String,
    parent_context: Arc<dyn AsyncEngineContext>,
    _admission_permit: OwnedSemaphorePermit,
    metrics: Arc<Metrics>,
    request: Arc<CanonicalRequest>,
) -> Response {
    let dialect = request.dialect;
    let Some(config) = selection.pooling_config.clone() else {
        return dispatch_error(
            &request_id,
            &parent_context,
            DispatchError::malformed("missing pooling decision config"),
            dialect,
        );
    };
    let fanout_started = Instant::now();

    let mut futures_stream =
        stream::iter(batches.into_iter().enumerate().map(|(batch_idx, batch)| {
            run_batch(
                selection.clone(),
                config.clone(),
                batch_idx,
                batch,
                request_id.clone(),
                parent_context.clone(),
                request.clone(),
            )
        }))
        .buffer_unordered(MAX_CONCURRENT_SIBLINGS);

    let mut per_question = Vec::with_capacity(request.questions.len());
    let mut total_input_tokens: u64 = 0;
    let mut failure = None;
    while let Some(result) = futures_stream.next().await {
        if failure.is_some() {
            continue;
        }
        match result {
            Ok((input_tokens, batch_outcomes)) => {
                total_input_tokens += input_tokens as u64;
                per_question.extend(batch_outcomes);
            }
            Err(cause) => {
                parent_context.kill();
                failure = Some(cause);
            }
        }
    }
    drop(futures_stream);
    if let Some(cause) = failure {
        return dispatch_error(&request_id, &parent_context, cause, dialect);
    }

    metrics.observe_systemone_phase(
        &selection.canonical_model,
        SystemOnePhase::Fanout,
        fanout_started.elapsed().as_secs_f64(),
    );

    per_question.sort_by_key(|(ordinal, _)| *ordinal);
    let outcomes: Vec<QuestionOutcome> = per_question.into_iter().map(|(_, o)| o).collect();
    let usage = Usage {
        input_tokens: total_input_tokens,
        output_tokens: 0,
        cached_tokens: Some(0),
        reasoning_tokens: 0,
    };
    finish_decision_response(
        &request,
        &selection.canonical_model,
        &outcomes,
        &usage,
        &request_id,
    )
}

async fn run_batch(
    selection: SystemOneExecutionSelection,
    config: PoolingDecisionConfig,
    batch_idx: usize,
    batch: PreparedPoolingBatch,
    request_id: String,
    parent_context: Arc<dyn AsyncEngineContext>,
    request: Arc<CanonicalRequest>,
) -> Result<(usize, Vec<(usize, QuestionOutcome)>), DispatchError> {
    selection.validate_dispatch(None).map_err(|_| {
        DispatchError::unavailable("selected worker revision is no longer available")
    })?;
    let SystemOneEngine::Pooling(pooling_engine) = &selection.engine else {
        return Err(DispatchError::malformed(
            "generate engine cannot execute pooling decision batch",
        ));
    };
    let seq_len = batch.token_ids.len();
    let task = if config.uses_last_classify() {
        "classify"
    } else {
        "token_classify"
    };
    let pooling_req = build_pooling_request(&selection.canonical_model, batch.token_ids, task);
    let context = Context::with_id_and_metadata(
        pooling_req,
        format!("{request_id}-p{batch_idx}"),
        Default::default(),
    );
    link_child_or_cancelled(&parent_context, context.context())?;
    let mut stream = run_until_killed(parent_context.as_ref(), pooling_engine.generate(context))
        .await
        .ok_or_else(DispatchError::cancelled)?
        .map_err(|cause| backend_error::classify(cause.as_ref()))?;
    link_child_or_cancelled(&parent_context, stream.context())?;

    let mut response = None;
    while let Some(frame) = run_until_killed(parent_context.as_ref(), stream.next())
        .await
        .ok_or_else(DispatchError::cancelled)?
    {
        if let Some(payload) = frame.into_data().map_err(stream_error)? {
            if response.replace(payload).is_some() {
                return Err(DispatchError::malformed(
                    "pooling worker returned multiple payloads",
                ));
            }
        }
    }
    if parent_context.is_killed() {
        return Err(DispatchError::cancelled());
    }
    let response =
        response.ok_or_else(|| DispatchError::malformed("pooling worker returned no payload"))?;

    let rows = extract_marker_logits(
        &response,
        seq_len,
        config.num_classes(),
        &batch.markers,
        config.readout_offset,
    )
    .map_err(|cause| DispatchError::malformed(cause.to_string()))?;

    let effective_temperature = config.temperature * request.temperature;
    let mut outcomes = Vec::with_capacity(batch.question_indices.len());
    for (&ordinal, row) in batch.question_indices.iter().zip(&rows) {
        let question = request
            .questions
            .iter()
            .find(|q| q.ordinal == ordinal)
            .ok_or_else(|| DispatchError::malformed("unknown question ordinal in pooling batch"))?;
        let distribution = reduce_head_logits(
            row,
            question.candidates.len(),
            question.kind == QuestionKind::Predicate,
            effective_temperature,
        )
        .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
        outcomes.push((ordinal, QuestionOutcome::Answer(distribution)));
    }
    Ok((seq_len, outcomes))
}
