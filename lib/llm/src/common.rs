// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub mod dtype;
pub mod versioned;

/// Test utilities for integration tests
#[cfg(all(test, feature = "integration"))]
pub mod test_utils {
    use dynamo_runtime::{DistributedRuntime, Runtime};

    /// Creates a test DistributedRuntime for integration testing.
    /// Uses from_current to leverage the existing tokio runtime in tests.
    pub async fn create_test_drt_async() -> DistributedRuntime {
        let runtime = Runtime::from_current().expect("Failed to get current runtime");
        DistributedRuntime::from_settings_without_discovery(runtime)
            .await
            .expect("Failed to create test DRT")
    }
}
