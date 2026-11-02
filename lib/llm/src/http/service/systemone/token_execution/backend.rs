// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{
    BranchResult, DispatchError, MAX_CONCURRENT_SIBLINGS, StartedBranch, backend_error,
    decision_lifecycle, run_until_killed,
};
use super::{TokenPolicy, TokenScoring, measured_cached_tokens, scoring_placement};
use crate::decision_profile::TokenBackend;
use crate::protocols::{
    response_merge::PREFILL_OUTPUT_ANNOTATION,
    systemone::{
        build_native_score_request, parse_candidate_scores,
        vllm::{VllmScoreCollector, causal_scoring_request},
    },
};
use dynamo_decisions::{QuestionOutcome, reduce_vocab_logprobs};
use dynamo_runtime::engine::AsyncEngineContext;
use futures::StreamExt;
use std::sync::Arc;

pub(in super::super) fn scoring_for(backend: TokenBackend) -> &'static dyn TokenScoring {
    match backend {
        TokenBackend::Sglang => &SglangScoring,
        TokenBackend::VllmCausal => &VllmScoring,
    }
}

fn stream_error(cause: dynamo_runtime::error::DynamoError) -> DispatchError {
    backend_error::classify(&cause)
}

fn accept_frame(
    frame: serde_json::Value,
    terminal: &mut Option<serde_json::Value>,
) -> Result<(), DispatchError> {
    let finish = frame
        .get("meta_info")
        .and_then(|meta| meta.get("finish_reason"));
    if let Some(cause) = finish.and_then(super::super::abort::classify) {
        return Err(cause);
    }
    if finish.is_some_and(|finish| !finish.is_null()) && terminal.replace(frame).is_some() {
        return Err(DispatchError::invalid_executor(
            "multiple terminal SGLang scoring frames",
        ));
    }
    Ok(())
}

struct SglangScoring;

#[async_trait::async_trait]
impl TokenScoring for SglangScoring {
    fn policy(&self) -> TokenPolicy {
        TokenPolicy {
            format_name: "sglang",
            output_tokens: 0,
            sibling_concurrency: MAX_CONCURRENT_SIBLINGS,
        }
    }
    fn build_request(
        &self,
        prompt_ids: Vec<u32>,
        label_ids: &[u32],
        cache_salt: &str,
    ) -> Result<serde_json::Value, crate::protocols::systemone::SystemOneError> {
        build_native_score_request(&prompt_ids, label_ids, cache_salt)
    }
    async fn read_frames(
        &self,
        started: StartedBranch,
        data_parallel_start_rank: u32,
        data_parallel_size: u32,
        parent_context: Arc<dyn AsyncEngineContext>,
    ) -> Result<BranchResult, DispatchError> {
        let StartedBranch {
            prepared,
            tracker,
            mut stream,
            mut exchange,
        } = started;
        let mut terminal = None;
        let mut frames = Vec::new();
        loop {
            let delta = run_until_killed(parent_context.as_ref(), stream.next())
                .await
                .ok_or_else(DispatchError::cancelled)?;
            let Some(delta) = delta else { break };
            let prefill = delta.event.as_deref() == Some(PREFILL_OUTPUT_ANNOTATION);
            if let Some(output) = delta.into_data().map_err(stream_error)? {
                exchange
                    .chunk(output, prefill, &mut |frame| frames.push(frame.clone()))
                    .map_err(stream_error)?;
            }
            for frame in frames.drain(..) {
                accept_frame(frame, &mut terminal)?;
            }
        }
        if let Some(frame) = exchange.finish().map_err(stream_error)? {
            accept_frame(frame, &mut terminal)?;
        }
        let terminal = terminal.ok_or_else(|| {
            DispatchError::invalid_executor("missing terminal SGLang scoring frame")
        })?;
        let placement = scoring_placement(&tracker, data_parallel_start_rank, data_parallel_size)?;
        let scores = parse_candidate_scores(&terminal, &prepared.label_ids)
            .map_err(|cause| DispatchError::invalid_executor(cause.to_string()))?;
        let answer = QuestionOutcome::Answer(
            reduce_vocab_logprobs(&scores, prepared.temperature)
                .map_err(|cause| DispatchError::invalid_executor(cause.to_string()))?,
        );
        let input_tokens = prepared.prompt_len;
        let cached_tokens = measured_cached_tokens(&terminal, input_tokens)?;
        Ok(BranchResult {
            index: prepared.index,
            answer,
            input_tokens,
            output_tokens: self.policy().output_tokens as usize,
            cached_tokens,
            placement,
        })
    }
}

struct VllmScoring;

#[async_trait::async_trait]
impl TokenScoring for VllmScoring {
    fn policy(&self) -> TokenPolicy {
        TokenPolicy {
            format_name: "vllm",
            output_tokens: 1,
            sibling_concurrency: 1,
        }
    }
    fn build_request(
        &self,
        prompt_ids: Vec<u32>,
        label_ids: &[u32],
        cache_salt: &str,
    ) -> Result<serde_json::Value, crate::protocols::systemone::SystemOneError> {
        causal_scoring_request(prompt_ids, label_ids, cache_salt)
    }
    async fn read_frames(
        &self,
        started: StartedBranch,
        data_parallel_start_rank: u32,
        data_parallel_size: u32,
        parent_context: Arc<dyn AsyncEngineContext>,
    ) -> Result<BranchResult, DispatchError> {
        let StartedBranch {
            prepared,
            tracker,
            mut stream,
            ..
        } = started;
        // vLLM format binding encodes the request only. Candidate-score frames and measured
        // usage are validated here, not by the generated-text response exchange.
        let mut collector = VllmScoreCollector::new(prepared.label_ids, prepared.prompt_len)
            .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
        while let Some(output) =
            decision_lifecycle::next_frame(parent_context.as_ref(), &mut stream)
                .await
                .map_err(backend_error::lifecycle_error)?
        {
            collector
                .observe(&output)
                .map_err(|cause| DispatchError::invalid_executor(cause.to_string()))?;
        }
        let result = collector
            .finish()
            .map_err(|cause| DispatchError::invalid_executor(cause.to_string()))?;
        let placement = scoring_placement(&tracker, data_parallel_start_rank, data_parallel_size)?;
        let answer = QuestionOutcome::Answer(
            reduce_vocab_logprobs(&result.scores, prepared.temperature)
                .map_err(|cause| DispatchError::invalid_executor(cause.to_string()))?,
        );
        Ok(BranchResult {
            index: prepared.index,
            answer,
            input_tokens: result.input_tokens,
            output_tokens: self.policy().output_tokens as usize,
            cached_tokens: Some(result.cached_tokens),
            placement,
        })
    }
}
