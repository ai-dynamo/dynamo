// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod events;

use super::{DistributedRuntime, Runtime};
/// A trait for objects that proivde access to the [Runtime]
pub trait RuntimeProvider {
    fn rt(&self) -> &Runtime;
}

/// A trait for objects that provide access to the [DistributedRuntime].
pub trait DistributedRuntimeProvider {
    fn drt(&self) -> &DistributedRuntime;
}

impl RuntimeProvider for DistributedRuntime {
    fn rt(&self) -> &Runtime {
        &self.runtime
    }
}

// This implementation is required because:
// 1. MetricsRegistry has a supertrait bound: `MetricsRegistry: Send + Sync + DistributedRuntimeProvider`
// 2. DistributedRuntime implements MetricsRegistry (in distributed.rs)
// 3. Therefore, DistributedRuntime must implement DistributedRuntimeProvider to satisfy the trait bound
// 4. This enables DistributedRuntime to serve as both a provider (of itself) and a metrics registry
impl DistributedRuntimeProvider for DistributedRuntime {
    fn drt(&self) -> &DistributedRuntime {
        self
    }
}
