// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_decisions::{QuestionOutcome, reduce_vocab_logprobs};
use dynamo_runtime::engine::AsyncEngineContext;
use futures::StreamExt;

use super::super::super::decision_lifecycle;
use super::super::{
    BranchResult, DispatchError, MAX_CONCURRENT_SIBLINGS, Placement, StartedBranch, backend_error,
    resolve_dp_rank, run_until_killed,
};
use crate::{
    local_model::runtime_config::TokenBackend,
    protocols::{
        common::timing::RequestTracker,
        response_merge::PREFILL_OUTPUT_ANNOTATION,
        systemone::vllm::{VllmScoreCollector, causal_scoring_request},
        systemone::{build_native_score_request, parse_candidate_scores},
    },
};

pub(in super::super) enum PreparedScoringRequest {
    Sglang(Box<serde_json::Value>),
    Vllm(Box<serde_json::Value>),
}

pub(in super::super) fn measured_cached_tokens(
    response: &serde_json::Value,
    input_tokens: usize,
) -> Result<Option<usize>, DispatchError> {
    let Some(count) = response
        .get("meta_info")
        .and_then(|meta| meta.get("cached_tokens"))
    else {
        return Ok(None);
    };
    let count = count
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value <= input_tokens)
        .ok_or_else(|| DispatchError::malformed("invalid cache-read token count"))?;
    Ok(Some(count))
}

pub(in super::super) fn build(
    backend: TokenBackend,
    prompt_ids: Vec<u32>,
    label_ids: &[u32],
    cache_salt: &str,
) -> Result<PreparedScoringRequest, crate::protocols::systemone::SystemOneError> {
    match backend {
        TokenBackend::Sglang => PreparedScoringRequest::sglang(prompt_ids, label_ids, cache_salt),
        TokenBackend::VllmCausal => PreparedScoringRequest::vllm(prompt_ids, label_ids, cache_salt),
    }
}

pub(in super::super) fn max_concurrent_siblings(backend: TokenBackend) -> usize {
    match backend {
        TokenBackend::Sglang => MAX_CONCURRENT_SIBLINGS,
        TokenBackend::VllmCausal => 1,
    }
}

impl PreparedScoringRequest {
    pub(in super::super) fn backend(&self) -> TokenBackend {
        match self {
            Self::Sglang(_) => TokenBackend::Sglang,
            Self::Vllm(_) => TokenBackend::VllmCausal,
        }
    }
    pub(in super::super) fn sglang(
        prompt_ids: Vec<u32>,
        label_ids: &[u32],
        cache_salt: &str,
    ) -> Result<Self, crate::protocols::systemone::SystemOneError> {
        build_native_score_request(&prompt_ids, label_ids, cache_salt)
            .map(Box::new)
            .map(Self::Sglang)
    }
    pub(in super::super) fn vllm(
        prompt_ids: Vec<u32>,
        label_ids: &[u32],
        cache_salt: &str,
    ) -> Result<Self, crate::protocols::systemone::SystemOneError> {
        causal_scoring_request(prompt_ids, label_ids, cache_salt)
            .map(Box::new)
            .map(Self::Vllm)
    }
    pub(in super::super) fn compile(
        self,
        canonical_model: &str,
        kv_cache_block_size: u32,
        branch_id: &str,
    ) -> Result<crate::generate_format::Bound, DispatchError> {
        let (backend, native, prompt_len) = match self {
            Self::Sglang(native) => {
                let prompt_len = native["input_ids"].as_array().map_or(0, Vec::len);
                ("sglang", *native, prompt_len)
            }
            Self::Vllm(native) => {
                let prompt_len = native["token_ids"].as_array().map_or(0, Vec::len);
                ("vllm", *native, prompt_len)
            }
        };
        let mut bound = super::bind_native(
            backend,
            canonical_model,
            kv_cache_block_size,
            &native,
            branch_id,
        )
        .map_err(|cause| DispatchError::malformed(format!("native request: {cause:#}")))?;
        bound
            .request
            .routing
            .get_or_insert_default()
            .expected_output_tokens = Some(if backend == "sglang" { 0 } else { 1 });
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_isl(prompt_len, None);
        bound.request.tracker = Some(tracker);
        Ok(bound)
    }
}

pub(in super::super) fn scoring_placement(
    tracker: &RequestTracker,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
) -> Result<Placement, DispatchError> {
    tracker
        .get_worker_info()
        .and_then(|worker| {
            worker
                .decode_worker_id
                .or(worker.prefill_worker_id)
                .map(|worker_id| {
                    resolve_dp_rank(
                        worker.decode_dp_rank.or(worker.prefill_dp_rank),
                        data_parallel_start_rank,
                        data_parallel_size,
                    )
                    .map(|dp_rank| Placement { worker_id, dp_rank })
                })
        })
        .flatten()
        .ok_or_else(|| {
            DispatchError::unavailable("Dynamo did not report an unambiguous scoring worker rank")
        })
}

pub(in super::super) async fn finish_branch(
    backend: TokenBackend,
    started: StartedBranch,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
    parent_context: Arc<dyn AsyncEngineContext>,
) -> Result<BranchResult, DispatchError> {
    let result = match backend {
        TokenBackend::Sglang => {
            read_sglang_branch(
                started,
                data_parallel_start_rank,
                data_parallel_size,
                parent_context.clone(),
            )
            .await
        }
        TokenBackend::VllmCausal => {
            read_vllm_branch(
                started,
                data_parallel_start_rank,
                data_parallel_size,
                parent_context.clone(),
            )
            .await
        }
    };
    if parent_context.is_killed() {
        return Err(DispatchError::cancelled());
    }
    result
}

fn stream_error(cause: dynamo_runtime::error::DynamoError) -> DispatchError {
    if cause.class().normalized() == dynamo_runtime::error::ErrorClass::BackendProtocol {
        DispatchError::malformed("invalid native scoring stream")
    } else {
        backend_error::classify(&cause)
    }
}

fn accept_frame(
    frame: serde_json::Value,
    terminal: &mut Option<serde_json::Value>,
) -> Result<(), DispatchError> {
    let finish = frame
        .get("meta_info")
        .and_then(|meta| meta.get("finish_reason"));
    if let Some(cause) = finish.and_then(super::abort::classify) {
        return Err(cause);
    }
    if finish.is_some_and(|finish| !finish.is_null()) && terminal.replace(frame).is_some() {
        return Err(DispatchError::malformed(
            "multiple terminal SGLang scoring frames",
        ));
    }
    Ok(())
}

async fn read_sglang_branch(
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
    let terminal = terminal
        .ok_or_else(|| DispatchError::malformed("missing terminal SGLang scoring frame"))?;
    let placement = scoring_placement(&tracker, data_parallel_start_rank, data_parallel_size)?;
    let scores = parse_candidate_scores(&terminal, &prepared.label_ids)
        .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
    let answer = QuestionOutcome::Answer(
        reduce_vocab_logprobs(&scores, prepared.temperature)
            .map_err(|cause| DispatchError::malformed(cause.to_string()))?,
    );
    let input_tokens = prepared.prompt_len;
    let cached_tokens = measured_cached_tokens(&terminal, input_tokens)?;
    Ok(BranchResult {
        index: prepared.index,
        answer,
        input_tokens,
        output_tokens: 0,
        cached_tokens,
        placement,
    })
}

async fn read_vllm_branch(
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
    let mut collector = VllmScoreCollector::new(prepared.label_ids, prepared.prompt_len)
        .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
    while let Some(output) = decision_lifecycle::next_frame(parent_context.as_ref(), &mut stream)
        .await
        .map_err(backend_error::lifecycle_error)?
    {
        collector
            .observe(&output)
            .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
    }
    let result = collector
        .finish()
        .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
    let placement = scoring_placement(&tracker, data_parallel_start_rank, data_parallel_size)?;
    let answer = QuestionOutcome::Answer(
        reduce_vocab_logprobs(&result.scores, prepared.temperature)
            .map_err(|cause| DispatchError::malformed(cause.to_string()))?,
    );
    Ok(BranchResult {
        index: prepared.index,
        answer,
        input_tokens: result.input_tokens,
        output_tokens: 1,
        cached_tokens: result.cached_tokens,
        placement,
    })
}
