// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "console")]

use super::{
    Blocks, BlocksView, FinishedStatus, G1, KVConnectorOutput, LeaderRuntime, MatchResult, Request,
    SchedulerOutput,
};
use anyhow::Result;

pub fn is_enabled() -> bool {
    std::env::var("KVBM_CONSOLE").is_ok()
}

/// Wrapper that instruments a leader implementation with console hooks. Currently a stub.
pub struct InstrumentedLeader<T: LeaderRuntime> {
    inner: T,
}

impl<T: LeaderRuntime> InstrumentedLeader<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    fn pre_hook(&self, _label: &str) {
        // TODO: emit console events / breakpoints
    }

    fn post_hook(&self, _label: &str) {
        // TODO: emit console events / breakpoints
    }
}

impl<T: LeaderRuntime> LeaderRuntime for InstrumentedLeader<T> {
    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<MatchResult> {
        self.pre_hook("get_num_new_matched_tokens");
        let res = self
            .inner
            .get_num_new_matched_tokens(request_id, num_computed_tokens);
        self.post_hook("get_num_new_matched_tokens");
        res
    }

    fn update_state_after_alloc(
        &mut self,
        request_id: &str,
        block_ids: BlocksView<G1>,
        num_external_tokens: usize,
    ) -> Result<()> {
        self.pre_hook("update_state_after_alloc");
        let res = self
            .inner
            .update_state_after_alloc(request_id, block_ids, num_external_tokens);
        self.post_hook("update_state_after_alloc");
        res
    }

    fn request_finished(
        &mut self,
        request_id: &str,
        block_ids: Blocks<G1>,
    ) -> Result<FinishedStatus> {
        self.pre_hook("request_finished");
        let res = self.inner.request_finished(request_id, block_ids);
        self.post_hook("request_finished");
        res
    }

    fn build_connector_metadata(&mut self, output: &SchedulerOutput) -> Result<Vec<u8>> {
        self.pre_hook("build_connector_metadata");
        let res = self.inner.build_connector_metadata(output);
        self.post_hook("build_connector_metadata");
        res
    }

    fn update_connector_output(&mut self, connector_output: KVConnectorOutput) -> Result<()> {
        self.pre_hook("update_connector_output");
        let res = self.inner.update_connector_output(connector_output);
        self.post_hook("update_connector_output");
        res
    }

    fn engine_id(&self) -> &str {
        self.inner.engine_id()
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn wait_ready(&self) -> Result<()> {
        self.inner.wait_ready()
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.inner.has_slot(request_id)
    }

    fn create_slot(&mut self, request: Request, all_token_ids: Vec<Vec<i64>>) -> Result<()> {
        self.pre_hook("create_slot");
        let res = self.inner.create_slot(request, all_token_ids);
        self.post_hook("create_slot");
        res
    }
}
