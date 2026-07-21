// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::RouterEvent;

use super::super::core::{EngineEventBatch, EngineObservation, NoEngineEvents};

pub(in crate::replay) trait ReplayEngineObservation:
    EngineObservation<Vec<RouterEvent>>
{
    fn as_router_events(batch: &Self::Batch) -> &[RouterEvent];
}

impl ReplayEngineObservation for NoEngineEvents {
    #[inline]
    fn as_router_events(_batch: &Self::Batch) -> &[RouterEvent] {
        &[]
    }
}

#[derive(Debug, Default)]
pub(in crate::replay) struct RouterEventBatch(pub Vec<RouterEvent>);

impl EngineEventBatch for RouterEventBatch {
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
    }
}

#[derive(Debug, Default)]
pub(in crate::replay) struct RouterEventObservation;

impl EngineObservation<Vec<RouterEvent>> for RouterEventObservation {
    type Batch = RouterEventBatch;

    const CAPTURE_RAW: bool = true;

    #[inline]
    fn observe(raw: Vec<RouterEvent>) -> Self::Batch {
        RouterEventBatch(raw)
    }
}

impl ReplayEngineObservation for RouterEventObservation {
    #[inline]
    fn as_router_events(batch: &Self::Batch) -> &[RouterEvent] {
        &batch.0
    }
}
