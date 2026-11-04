// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-side inputs for running hint policies after booking.

use super::HintActionContext;
use crate::protocols::WorkerWithDpRank;

/// Request values the selection core shares with every hint policy of one booking.
pub(crate) struct ActionInputs<'a> {
    pub(crate) request_id: &'a str,
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) prompt_tokens: usize,
    pub(crate) block_size: u32,
}

impl<'a> ActionInputs<'a> {
    /// The context every policy sees.
    pub(crate) fn context(&self) -> HintActionContext<'a> {
        HintActionContext {
            request_id: self.request_id,
            worker: self.worker,
            prompt_tokens: self.prompt_tokens,
            block_size: self.block_size,
        }
    }
}
