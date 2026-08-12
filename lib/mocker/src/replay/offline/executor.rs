// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-shot execution for an already prepared multi-worker replay runtime.

use anyhow::Result;

use super::agg::RoundRobinAggRuntime;
use super::disagg::RoundRobinDisaggRuntime;
use super::extensions::kv_router::{AggRuntime, DisaggRuntime};
use super::scaling::ReplayScalingPolicy;
use crate::replay::TraceCollector;

#[allow(clippy::large_enum_variant)]
pub(crate) enum PreparedOfflineReplay {
    AggRoundRobin(RoundRobinAggRuntime),
    AggKv(AggRuntime),
    DisaggRoundRobin(RoundRobinDisaggRuntime),
    DisaggKv(DisaggRuntime),
}

impl PreparedOfflineReplay {
    pub(crate) fn run(
        self,
        scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    ) -> Result<TraceCollector> {
        let collector = match (self, scaling_policy) {
            (Self::AggRoundRobin(runtime), Some(policy)) => {
                runtime.with_scaling_policy(policy).run()?.0
            }
            (Self::AggKv(runtime), Some(policy)) => runtime.with_scaling_policy(policy).run()?.0,
            (Self::DisaggRoundRobin(runtime), Some(policy)) => {
                runtime.with_scaling_policy(policy).run()?.0
            }
            (Self::DisaggKv(runtime), Some(policy)) => runtime.with_scaling_policy(policy).run()?.0,
            (Self::AggRoundRobin(runtime), None) => runtime.run()?.0,
            (Self::AggKv(runtime), None) => runtime.run()?.0,
            (Self::DisaggRoundRobin(runtime), None) => runtime.run()?.0,
            (Self::DisaggKv(runtime), None) => runtime.run()?.0,
        };
        Ok(collector)
    }
}
