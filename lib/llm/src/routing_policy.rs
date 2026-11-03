// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{discovery::RuntimeConfigWatch, preprocessor::PreprocessedRequest};
use dynamo_kv_router::scheduling::SchedulingRequest;
use dynamo_runtime::pipeline::Error;
use std::{collections::HashSet, sync::Arc};

pub(crate) type AttemptLease = Box<dyn Send + Sync>;

pub(crate) struct CandidateSelection {
    pub target: u64,
    pub allowed_fallback: HashSet<u64>,
    pub lease: Option<AttemptLease>,
}

pub(crate) trait CandidatePolicy: Send + Sync {
    /// `Ok(None)` defers to normal selection; policy rejections return `Err`, never fallback.
    fn select(
        &self,
        request: &PreprocessedRequest,
        live: &dyn Fn() -> Vec<u64>,
        free: &dyn Fn() -> Vec<u64>,
    ) -> Result<Option<CandidateSelection>, Error>;
    /// `None` leaves availability unrestricted; `Some(empty)` excludes every worker.
    fn refresh_admission(
        &self,
        request: &SchedulingRequest,
        available: Option<Arc<HashSet<u64>>>,
        workers: &RuntimeConfigWatch,
    ) -> Option<Arc<HashSet<u64>>>;
}
