// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod events;

use super::DistributedRuntime;
use crate::runtime::Runtime;

/// A trait for objects that provide access to the [Runtime].
///
/// Note: This always returns the standard runtime type, even when the `loom-runtime`
/// feature is enabled. This ensures compatibility with code that stores the runtime
/// directly. For loom-specific features, use the methods on the loom-wrapped runtime.
pub trait RuntimeProvider {
    fn rt(&self) -> &Runtime;
}

/// A trait for objects that provide access to the [DistributedRuntime].
pub trait DistributedRuntimeProvider {
    fn drt(&self) -> &DistributedRuntime;
}

impl RuntimeProvider for DistributedRuntime {
    fn rt(&self) -> &Runtime {
        self.runtime()
    }
}

// This implementation allows DistributedRuntime to provide access to itself
// when used in contexts that require DistributedRuntimeProvider.
// Components, Namespaces, and Endpoints use this trait to access their DRT.
impl DistributedRuntimeProvider for DistributedRuntime {
    fn drt(&self) -> &DistributedRuntime {
        self
    }
}
