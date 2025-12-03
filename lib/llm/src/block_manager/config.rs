// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::events::EventManager;
use super::*;
use dynamo_runtime::config::environment_names::kvbm::cpu_cache as env_cpu_cache;
use dynamo_runtime::config::environment_names::kvbm::disk_cache as env_disk_cache;
use prometheus::Registry;

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

    #[builder(default)]
    pub async_runtime: Option<Arc<tokio::runtime::Runtime>>,

    #[builder(default = "Arc::new(Registry::new())")]
    pub metrics_registry: Arc<Registry>,
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

    #[validate(range(min = 1, max = 2))]
    pub outer_dim: usize,

    #[validate(range(min = 1))]
    pub page_size: usize,

    #[validate(range(min = 1))]
    pub inner_dim: usize,

    #[builder(default = "2")]
    pub dtype_width_bytes: usize,
}

impl KvManagerModelConfig {
    pub fn builder() -> KvManagerModelConfigBuilder {
        KvManagerModelConfigBuilder::default()
    }
}

#[derive(Debug, Clone)]
pub enum BlockParallelismStrategy {
    /// KV blocks are sharded across all workers.
    /// This reduces the memory footprint and computational cost of each worker; however,
    /// requires extra communication between workers.
    LeaderWorkerSharded,
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

    /// The type of block parallelism strategy to use
    #[builder(default)]
    pub logical: Option<BlockParallelismStrategy>,

    /// The offload filter to use (if any).
    /// This dictates which blocks will be offloaded to the next-lowest cache level.
    #[builder(default = "None")]
    pub offload_filter: Option<Arc<dyn OffloadFilter>>,
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
        match (
            self.storage.is_some(),
            self.allocator.is_some(),
            self.logical.is_some(),
        ) {
            (true, false, false) | (false, true, false) | (false, false, true) => Ok(()), // XOR condition met
            (false, false, false) => {
                Err("Must provide either `storage` or `allocator` or `logical`.".to_string())
            }
            _ => Err(
                "Only one selection of either `storage` and `allocator` or `logical`.".to_string(),
            ),
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

    /// Event manager to handle block related events
    #[builder(default)]
    pub event_manager: Option<Arc<dyn EventManager>>,

    /// Channel to reset the block manager to a specific cache level
    #[builder(default)]
    pub block_reset_channel: Option<BlockResetChannel>,

    /// Optional KVBM-level metrics for tracking offload/onboard operations
    #[builder(default)]
    pub kvbm_metrics: Option<crate::block_manager::metrics_kvbm::KvbmMetrics>,

    /// Optional KV Event Consolidator Configuration
    ///
    /// If provided, KVBM will create a KV Event Consolidator that deduplicates
    /// KV cache events from vLLM (G1) and KVBM (G2/G3) before sending to the router.
    /// This is used when `--connector kvbm` is enabled with prefix caching.
    #[builder(default, setter(custom))]
    pub consolidator_config:
        Option<crate::block_manager::kv_consolidator::KvEventConsolidatorConfig>,
}

impl KvBlockManagerConfig {
    /// Create a new builder for the KvBlockManagerConfig
    pub fn builder() -> KvBlockManagerConfigBuilder {
        KvBlockManagerConfigBuilder::default()
    }
}

impl KvBlockManagerConfigBuilder {
    /// Set the consolidator config using individual parameters
    pub fn consolidator_config(
        mut self,
        engine_endpoint: String,
        output_endpoint: Option<String>,
        engine_source: crate::block_manager::kv_consolidator::EventSource,
    ) -> Self {
        let config = match engine_source {
            crate::block_manager::kv_consolidator::EventSource::Vllm => {
                let output_ep = output_endpoint.expect("output_endpoint is required for vLLM");
                crate::block_manager::kv_consolidator::KvEventConsolidatorConfig::new_vllm(
                    engine_endpoint,
                    output_ep,
                )
            }
            crate::block_manager::kv_consolidator::EventSource::Trtllm => {
                // output_endpoint is the ZMQ endpoint where consolidator publishes
                // Worker-side publishers subscribe to this and forward to NATS
                let output_ep = output_endpoint.expect(
                    "output_endpoint (consolidated_event_endpoint) is required for TensorRT-LLM",
                );
                crate::block_manager::kv_consolidator::KvEventConsolidatorConfig::new_trtllm(
                    engine_endpoint,
                    output_ep,
                )
            }
            crate::block_manager::kv_consolidator::EventSource::Kvbm => {
                // This case should never be reached - consolidator_config() is only called with
                // EventSource::Vllm or EventSource::Trtllm. EventSource::Kvbm is used when KVBM
                // sends events TO the consolidator (via DynamoEventManager), but KVBM is never
                // the engine_source that publishes events via ZMQ that the consolidator subscribes to.
                unreachable!(
                    "consolidator_config() should never be called with EventSource::Kvbm. \
                     KVBM events are sent directly to the consolidator handle, not via ZMQ."
                )
            }
        };
        // With setter(custom), the builder field is Option<Option<T>>, so we need Some(Some(...))
        self.consolidator_config = Some(Some(config));
        self
    }
}

/// Determines if CPU memory (G2) should be bypassed for direct G1->G3 (Device->Disk) offloading.
///
/// Returns `true` if:
/// - Disk cache env vars are set (`DYN_KVBM_DISK_CACHE_GB` or `DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS`)
///   AND their values are non-zero
/// - AND CPU cache env vars are NOT set (`DYN_KVBM_CPU_CACHE_GB` or `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS`)
///   OR their values are zero (treated as not set)
pub fn should_bypass_cpu_cache() -> bool {
    let cpu_cache_gb_set = std::env::var(env_cpu_cache::DYN_KVBM_CPU_CACHE_GB)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|v| v > 0)
        .unwrap_or(false);
    let cpu_cache_override_set =
        std::env::var(env_cpu_cache::DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v > 0)
            .unwrap_or(false);
    let disk_cache_gb_set = std::env::var(env_disk_cache::DYN_KVBM_DISK_CACHE_GB)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|v| v > 0)
        .unwrap_or(false);
    let disk_cache_override_set =
        std::env::var(env_disk_cache::DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v > 0)
            .unwrap_or(false);

    let cpu_cache_set = cpu_cache_gb_set || cpu_cache_override_set;
    let disk_cache_set = disk_cache_gb_set || disk_cache_override_set;

    disk_cache_set && !cpu_cache_set
}
