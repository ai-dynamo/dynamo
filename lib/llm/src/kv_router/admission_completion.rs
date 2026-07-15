// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use dynamo_kv_router::scheduling::AdmissionToolSummary;
use parking_lot::Mutex;

pub(crate) const ADMISSION_TOOL_SUMMARY_CONTEXT_KEY: &str = "kv_router.admission_tool_summary";

/// Request-local parsed tool-call state shared by the postprocessor and the
/// terminal router cleanup.
#[derive(Clone, Default)]
pub(crate) struct SharedAdmissionToolSummary {
    state: Arc<Mutex<AdmissionToolSummaryState>>,
}

#[derive(Default)]
struct AdmissionToolSummaryState {
    summary: AdmissionToolSummary,
    calls: HashSet<(u32, u32)>,
    named_calls: HashSet<(u32, u32)>,
}

impl SharedAdmissionToolSummary {
    pub(crate) fn record_tool_call(
        &self,
        choice_index: u32,
        tool_call_index: u32,
        name: Option<&str>,
    ) {
        let mut state = self.state.lock();
        let call = (choice_index, tool_call_index);
        if !state.calls.contains(&call) {
            if !state.summary.record_tool_call() {
                return;
            }
            state.calls.insert(call);
        }
        if let Some(name) = name
            && !state.named_calls.contains(&call)
            && state.summary.record_tool_name(name)
        {
            state.named_calls.insert(call);
        }
    }

    pub(crate) fn snapshot(&self) -> AdmissionToolSummary {
        self.state.lock().summary.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_calls_are_counted_once_and_names_are_bounded() {
        let summary = SharedAdmissionToolSummary::default();
        summary.record_tool_call(0, 0, None);
        summary.record_tool_call(0, 0, Some("shell"));
        for index in 1..6 {
            summary.record_tool_call(0, index, Some(&format!("tool-{index}")));
        }

        let snapshot = summary.snapshot();
        assert_eq!(snapshot.tool_call_count(), 4);
        assert_eq!(
            snapshot.tool_names(),
            ["shell", "tool-1", "tool-2", "tool-3"]
        );
    }
}
