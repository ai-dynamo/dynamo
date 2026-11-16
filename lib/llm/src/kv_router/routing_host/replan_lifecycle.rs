// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_runtime::pipeline::{Error, SingleIn};
use dynamo_scheduler::host::RequestLifecycle;
use parking_lot::Mutex;

use crate::protocols::common::{llm_backend::PreprocessedRequest, preprocessor::owned_abort_error};

const REPLAN_LIFECYCLE: &str = "conditional-replan-classifier";
type LifecycleSlot = Mutex<Option<Box<RequestLifecycle>>>;

/// Owns a declined conditional plan until Prefill finishes or Decode claims it.
/// Real migration retries keep their existing owner instead.
pub(crate) struct ReplanLifecycle(Option<Arc<LifecycleSlot>>);

impl ReplanLifecycle {
    pub(super) fn park(
        mut lifecycle: Box<RequestLifecycle>,
        request: &mut SingleIn<PreprocessedRequest>,
    ) -> Self {
        lifecycle.prepare_retry();
        if let Some(state) = &request.migration_state {
            state.store_request_lifecycle(lifecycle);
            return Self(None);
        }
        let slot = Arc::new(Mutex::new(Some(lifecycle)));
        request.insert_arc(REPLAN_LIFECYCLE, slot.clone());
        Self(Some(slot))
    }

    pub(super) fn take(
        request: &SingleIn<PreprocessedRequest>,
    ) -> Result<Option<Box<RequestLifecycle>>, Error> {
        let Some(slot) = request
            .get_optional::<LifecycleSlot>(REPLAN_LIFECYCLE)
            .map_err(|message| {
                anyhow::anyhow!("invalid conditional classifier context: {message}")
            })?
        else {
            return Ok(None);
        };
        Ok(slot.lock().take())
    }

    pub(crate) fn take_owned(&mut self) -> Option<Box<RequestLifecycle>> {
        self.0.take()?.lock().take()
    }

    pub(crate) fn abort(&mut self, error: Option<&Error>) {
        if let Some(slot) = self.0.take()
            && let Some(mut lifecycle) = slot.lock().take()
        {
            lifecycle.abort(error.map(|error| owned_abort_error(error.as_ref())));
        }
    }
}

impl Drop for ReplanLifecycle {
    fn drop(&mut self) {
        self.abort(None);
    }
}
