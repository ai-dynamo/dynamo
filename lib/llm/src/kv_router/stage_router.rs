// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The interface shared by single-stage routers: encoder, prefill and decode.

use anyhow::Result;
use dynamo_runtime::pipeline::{SingleIn, async_trait};

use crate::protocols::common::llm_backend::PreprocessedRequest;

/// One routing stage of a disaggregated request.
///
/// A stage router answers three questions, each as a separate call:
///
/// - [`schedule`](Self::schedule): where does this stage run? The plan holds
///   the chosen worker and any booking made for it, or a decision that the
///   stage does not run.
/// - [`preview`](Self::preview): what does the next stage look like for this
///   request? Conditional disaggregation uses it to decide whether this stage
///   is needed at all.
/// - [`dispatch`](Self::dispatch): send the request as the plan says.
///
/// The frontend operator calls all three in sequence. A host that does not
/// send requests itself, such as the inference-gateway EPP, can call
/// `schedule` and `preview` and hand the result to the frontend.
#[async_trait]
pub(crate) trait StageRouter: Send + Sync {
    /// Where the stage runs for one request.
    type Plan: Send;
    /// What the next stage looks like for one request.
    type Preview: Send;
    /// What dispatch produced: the stage's response stream and whatever the
    /// caller needs to continue to the next stage.
    type Dispatched: Send;

    /// Choose and book where this stage runs. A caller that does not dispatch
    /// the plan must give its booking back through the plan type.
    // No caller yet: `PrefillRouter::generate` runs schedule's steps one by
    // one to keep its error logging. The EPP is the intended first caller.
    #[allow(dead_code)]
    async fn schedule(&self, request: &SingleIn<PreprocessedRequest>) -> Result<Self::Plan>;

    /// Look at the next stage for this request without dispatching anything.
    async fn preview(&self, request: &SingleIn<PreprocessedRequest>) -> Result<Self::Preview>;

    /// Send `request` as `plan` says.
    async fn dispatch(
        &self,
        request: SingleIn<PreprocessedRequest>,
        plan: Self::Plan,
    ) -> Result<Self::Dispatched>;
}
