// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::protocols::MockEngineArgs;
use crate::replay::TraceCollector;
use crate::scheduler::{EngineCore, EnginePassResult, SglangCore, VllmCore};

pub(crate) struct ReplayWorkerCore {
    core: EngineCore,
}

impl ReplayWorkerCore {
    pub(crate) fn new(args: MockEngineArgs) -> Self {
        let core = match args.engine_type {
            crate::common::protocols::EngineType::Vllm => EngineCore::Vllm(VllmCore::new(args)),
            crate::common::protocols::EngineType::Sglang => {
                EngineCore::Sglang(SglangCore::new(args))
            }
        };
        Self { core }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.core.is_empty()
    }

    pub(crate) fn receive(
        &mut self,
        request: crate::common::protocols::DirectRequest,
    ) -> uuid::Uuid {
        self.core.receive(request)
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.core.num_requests()
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> EnginePassResult {
        self.core.execute_pass(collector, now_ms)
    }
}
