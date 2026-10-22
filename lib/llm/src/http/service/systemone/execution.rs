// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_decisions::{QuestionOutcome, reduce_vocab_logprobs};
use dynamo_runtime::engine::AsyncEngineContext;
use futures::StreamExt;

use super::super::{generate, sglang_generate};
use super::{
    BranchResult, DispatchError, Placement, StartedBranch, resolve_dp_rank, run_until_killed,
};
use crate::{
    discovery::SystemOneExecutionSelection,
    protocols::{
        common::{preprocessor::PreprocessedRequest, timing::RequestTracker},
        openai::generate::GenerateRequest,
        sglang::generate::SglangGenerateRequest,
        systemone::vllm::VllmScoreCollector,
    },
};

pub(super) enum PreparedScoringRequest {
    Sglang(SglangGenerateRequest),
    Vllm(GenerateRequest),
}

pub(super) fn measured_cached_tokens(
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

impl PreparedScoringRequest {
    pub(super) fn compile(
        self,
        selection: &SystemOneExecutionSelection,
        branch_id: &str,
        cache_salt: String,
        tracker: Arc<RequestTracker>,
    ) -> Result<PreprocessedRequest, DispatchError> {
        match self {
            Self::Sglang(mut native) => {
                native.cache_salt = Some(cache_salt);
                let mut request = sglang_generate::preprocessed_request(
                    native,
                    &selection.canonical_model,
                    None,
                    branch_id,
                );
                if let Ok(request) = &mut request {
                    request
                        .routing
                        .get_or_insert_default()
                        .expected_output_tokens = Some(0);
                }
                request
            }
            Self::Vllm(mut native) => {
                native.cache_salt = Some(cache_salt);
                generate::preprocessed_decision_request(
                    native,
                    &selection.canonical_model,
                    branch_id,
                    selection.card.kv_cache_block_size,
                    tracker,
                )
            }
        }
        .map_err(|cause| DispatchError::malformed(format!("native request: {cause:#}")))
    }
}

pub(super) fn scoring_placement(
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

pub(super) async fn read_vllm_branch(
    started: StartedBranch,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
    parent_context: Arc<dyn AsyncEngineContext>,
) -> Result<BranchResult, DispatchError> {
    let StartedBranch {
        prepared,
        tracker,
        mut stream,
    } = started;
    let mut collector = VllmScoreCollector::new(prepared.label_ids, prepared.prompt_ids.len());
    loop {
        let frame = run_until_killed(parent_context.as_ref(), stream.next())
            .await
            .ok_or_else(DispatchError::cancelled)?;
        let Some(frame) = frame else { break };
        if let Some(output) = frame
            .into_data()
            .map_err(|cause| DispatchError::backend(&cause))?
        {
            collector
                .observe(&output)
                .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
        }
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
        output_tokens: result.output_tokens,
        cached_tokens: result.cached_tokens,
        placement,
    })
}
