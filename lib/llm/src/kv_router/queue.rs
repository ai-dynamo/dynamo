// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use dynamo_kv_router::queue::DEFAULT_MAX_BATCHED_TOKENS;

use crate::kv_router::sequence::RuntimeSequencePublisher;
use crate::local_model::runtime_config::ModelRuntimeConfig;

/// Concrete `SchedulerQueue` wired to the runtime publisher and config types.
pub type SchedulerQueue =
    dynamo_kv_router::queue::SchedulerQueue<RuntimeSequencePublisher, ModelRuntimeConfig>;
