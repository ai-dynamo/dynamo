// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A stand-in for the router-mode plugins Dynamo's hosts pass to `PushRouter`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_runtime::pipeline::{InstanceSelector, ModeSelectors, PushRouterOptions};

struct Rotation(AtomicUsize);

impl InstanceSelector for Rotation {
    fn reads_active_requests(&self) -> bool {
        false
    }

    fn select_instance(
        &self,
        instance_ids: &[u64],
        _active_requests: &dyn Fn(u64) -> u64,
        preview: bool,
    ) -> Option<u64> {
        let cursor = if preview {
            self.0.load(Ordering::Relaxed)
        } else {
            self.0.fetch_add(1, Ordering::Relaxed)
        };
        instance_ids.get(cursor % instance_ids.len()).copied()
    }
}

/// A rotation for every mode.
pub fn selectors() -> ModeSelectors {
    Arc::new(|_| Some(Arc::new(Rotation(AtomicUsize::new(0))) as Arc<dyn InstanceSelector>))
}

/// Router options carrying [`selectors`].
pub fn options<T, U>() -> PushRouterOptions<T, U> {
    PushRouterOptions {
        selectors: Some(selectors()),
        ..PushRouterOptions::default()
    }
}
