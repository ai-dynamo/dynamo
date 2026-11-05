// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile an external hosted picker against the public `plugins` contracts.

use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
};
use dynamo_kv_router::selector::{HostedSelectionRequest, HostedWorkerPicker};
use dynamo_kv_router::{KvRouterConfig, WorkerSelectionPolicyError};

/// Reads the host's in-flight counts lazily, the way a least-loaded picker does.
struct ActiveRequestsPicker;

impl WorkerPicker for ActiveRequestsPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::ACTIVE_REQUESTS
    }

    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        (0..input.len())
            .min_by_key(|&row| input.active_requests(row).unwrap_or(u64::MAX))
            .ok_or_else(|| WorkerSelectionPolicyError::failed("empty candidate table"))
    }
}

#[test]
fn active_requests_pickers_select_through_the_public_hosted_api() {
    #[allow(deprecated)]
    let alias = WorkerInputs::OCCUPANCY;
    assert_eq!(alias, WorkerInputs::ACTIVE_REQUESTS);

    let Ok(picker) = WorkerSelectionPolicy::new(
        KvRouterConfig::default(),
        "decode",
        Vec::new(),
        Box::new(ActiveRequestsPicker),
    )
    .into_hosted_picker() else {
        panic!("a picker-only ACTIVE_REQUESTS policy runs hosted");
    };
    let picker: HostedWorkerPicker = picker;
    assert_eq!(picker.inputs(), WorkerInputs::ACTIVE_REQUESTS);
    let counts = |worker_id: u64| if worker_id == 4 { 0 } else { 3 };
    let selected = picker
        .select(&[2, 4, 6], Some(&counts), &HostedSelectionRequest::new(32))
        .unwrap();
    assert_eq!(selected, 4);
}
