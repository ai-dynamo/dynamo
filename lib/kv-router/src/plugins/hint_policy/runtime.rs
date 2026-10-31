// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-side inputs for running hint policies after booking.

use super::{HintActionContext, HintRequirements, KvFetchPayload};
use crate::kv_hints::KvFetchSource;
use crate::protocols::WorkerWithDpRank;

/// Request values the selection core shares with every hint policy of one booking.
pub(crate) struct ActionInputs<'a> {
    pub(crate) request_id: &'a str,
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) prompt_tokens: usize,
    pub(crate) block_size: u32,
    /// The fetch the selection committed `worker` to, which the booking must deliver.
    pub(crate) committed_kv_fetch: Option<KvFetchSource>,
    /// `worker`'s eligible fetch source, committed or not.
    pub(crate) kv_fetch_source: Option<KvFetchSource>,
    /// `None` when the request has no transfer data, so no fetch can be built.
    pub(crate) kv_fetch_payload: Option<KvFetchPayload<'a>>,
}

impl<'a> ActionInputs<'a> {
    /// The context a policy with `requirements` sees: fetch values only when it declared them.
    pub(crate) fn context(&self, requirements: HintRequirements) -> HintActionContext<'a> {
        HintActionContext {
            request_id: self.request_id,
            worker: self.worker,
            prompt_tokens: self.prompt_tokens,
            block_size: self.block_size,
            committed_kv_fetch: self
                .committed_kv_fetch
                .filter(|_| requirements.delivers_committed_kv_fetch()),
            kv_fetch_source: self
                .kv_fetch_source
                .filter(|_| requirements.reads_kv_fetch_sources()),
            kv_fetch_payload: self.kv_fetch_payload,
        }
    }
}
