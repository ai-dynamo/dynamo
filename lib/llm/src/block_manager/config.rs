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

use super::*;

#[derive(Debug, Clone)]
pub enum NixlOptions {
    /// Enable NIXL and create a new NIXL agent
    Enabled,

    /// Enable NIXL and use the provided NIXL agent
    EnabledWithAgent(NixlAgent),

    /// Disable NIXL
    Disabled,
}

#[derive(Debug, Clone, Builder, Validate)]
#[builder(pattern = "owned")]
pub struct KvManagerRuntimeConfig {
    pub worker_id: u64,

    #[builder(default)]
    pub cancellation_token: CancellationToken,

    #[builder(default = "NixlOptions::Enabled")]
    pub nixl: NixlOptions,
}

impl KvManagerRuntimeConfig {
    pub fn builder() -> KvManagerRuntimeConfigBuilder {
        KvManagerRuntimeConfigBuilder::default()
    }
}

impl KvManagerRuntimeConfigBuilder {
    pub fn enable_nixl(mut self) -> Self {
        self.nixl = Some(NixlOptions::Enabled);
        self
    }

    pub fn use_nixl_agent(mut self, agent: NixlAgent) -> Self {
        self.nixl = Some(NixlOptions::EnabledWithAgent(agent));
        self
    }

    pub fn disable_nixl(mut self) -> Self {
        self.nixl = Some(NixlOptions::Disabled);
        self
    }
}

#[derive(Debug, Clone, Builder, Validate)]
#[builder(pattern = "owned")]
pub struct KvManagerModelConfig {
    #[validate(range(min = 1))]
    pub num_layers: usize,

    #[validate(range(min = 1))]
    pub page_size: usize,

    #[validate(range(min = 1))]
    pub inner_dim: usize,

    #[builder(default = "DType::FP16")]
    pub dtype: DType,
}

impl KvManagerModelConfig {
    pub fn builder() -> KvManagerModelConfigBuilder {
        KvManagerModelConfigBuilder::default()
    }
}

#[derive(Builder, Validate)]
#[builder(pattern = "owned", build_fn(validate = "Self::validate"))]
pub struct KvManagerLayoutConfig<S: Storage + NixlRegisterableStorage> {
    /// The number of blocks to allocate
    #[validate(range(min = 1))]
    pub num_blocks: usize,

    /// The type of layout to use
    #[builder(default = "LayoutType::FullyContiguous")]
    pub layout_type: LayoutType,

    /// Storage for the blocks
    /// If provided, the blocks will be allocated from the provided storage
    /// Otherwise, the blocks will be allocated from
    #[builder(default)]
    pub storage: Option<Vec<S>>,

    /// If provided, the blocks will be allocated from the provided allocator
    /// This option is mutually exclusive with the `storage` option
    #[builder(default, setter(custom))]
    pub allocator: Option<Arc<dyn StorageAllocator<S>>>,
}

impl<S: Storage + NixlRegisterableStorage> KvManagerLayoutConfig<S> {
    /// Create a new builder for the KvManagerLayoutConfig
    pub fn builder() -> KvManagerLayoutConfigBuilder<S> {
        KvManagerLayoutConfigBuilder::default()
    }
}

// Implement the validation and build functions on the generated builder type
// Note: derive_builder generates KvManagerBlockConfigBuilder<S>
impl<S: Storage + NixlRegisterableStorage> KvManagerLayoutConfigBuilder<S> {
    /// Custom setter for the `allocator` field
    pub fn allocator(mut self, allocator: impl StorageAllocator<S> + 'static) -> Self {
        self.allocator = Some(Some(Arc::new(allocator)));
        self
    }

    // Validation function
    fn validate(&self) -> Result<(), String> {
        match (self.storage.is_some(), self.allocator.is_some()) {
            (true, false) | (false, true) => Ok(()), // XOR condition met
            (true, true) => Err("Cannot provide both `storage` and `allocator`.".to_string()),
            (false, false) => Err("Must provide either `storage` or `allocator`.".to_string()),
        }
    }
}

/// Configuration for the KvBlockManager
#[derive(Builder, Validate)]
#[builder(pattern = "owned")]
pub struct KvBlockManagerConfig {
    /// Runtime configuration
    ///
    /// This provides core runtime configuration for the KvBlockManager.
    pub runtime: KvManagerRuntimeConfig,

    /// Model configuration
    ///
    /// This provides model-specific configuration for the KvBlockManager, specifically,
    /// the number of layers and the size of the inner dimension which is directly related
    /// to the type of attention used by the model.
    ///
    /// Included in this configuration is also the page_size, i.e. the number of tokens that will
    /// be represented in each "paged" KV block.
    pub model: KvManagerModelConfig,

    /// Specific configuration for the device layout
    ///
    /// This includes the number of blocks and the layout of the data into the device memory/storage.
    #[builder(default, setter(strip_option))]
    pub device_layout: Option<KvManagerLayoutConfig<DeviceStorage>>,

    /// Specific configuration for the host layout
    ///
    /// This includes the number of blocks and the layout of the data into the host memory/storage.
    #[builder(default, setter(strip_option))]
    pub host_layout: Option<KvManagerLayoutConfig<PinnedStorage>>,

    // Specific configuration for the disk layout
    #[builder(default, setter(strip_option))]
    pub disk_layout: Option<KvManagerLayoutConfig<DiskStorage>>,
}

impl KvBlockManagerConfig {
    /// Create a new builder for the KvBlockManagerConfig
    pub fn builder() -> KvBlockManagerConfigBuilder {
        KvBlockManagerConfigBuilder::default()
    }
}
