// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::events::EventManager;
use super::*;
use crate::block_manager::block::transfer::TransferContext;
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

#[derive(Clone, Debug)]
pub enum RemoteStorageConfig {
    Object {
        default_bucket: Option<String>,
        endpoint: Option<String>,
        region: Option<String>,
    },
    Disk {
        base_path: String,
        use_gds: bool,
    },
}

impl RemoteStorageConfig {
    pub fn object(bucket: impl Into<String>) -> Self {
        Self::Object {
            default_bucket: Some(bucket.into()),
            endpoint: None,
            region: None,
        }
    }

    pub fn object_with_options(
        bucket: Option<String>,
        endpoint: Option<String>,
        region: Option<String>,
    ) -> Self {
        Self::Object {
            default_bucket: bucket,
            endpoint,
            region,
        }
    }

    pub fn disk(base_path: impl Into<String>, use_gds: bool) -> Self {
        Self::Disk {
            base_path: base_path.into(),
            use_gds,
        }
    }
}

#[derive(Clone)]
pub struct RemoteTransferContext {
    base: Arc<TransferContext>,
    config: RemoteStorageConfig,
    worker_id: u64,
}

#[derive(Clone)]
pub struct RemoteContextConfig {
    pub remote_storage_config: RemoteStorageConfig,
    pub worker_id: u64,
}

impl RemoteTransferContext {
    pub fn for_object(base: Arc<TransferContext>, default_bucket: Option<String>) -> Self {
        Self {
            base,
            config: RemoteStorageConfig::Object {
                default_bucket,
                endpoint: None,
                region: None,
            },
            worker_id: 0,
        }
    }

    pub fn for_object_with_options(
        base: Arc<TransferContext>,
        default_bucket: Option<String>,
        endpoint: Option<String>,
        region: Option<String>,
        worker_id: u64,
    ) -> Self {
        Self {
            base,
            config: RemoteStorageConfig::Object {
                default_bucket,
                endpoint,
                region,
            },
            worker_id,
        }
    }

    pub fn for_disk(base: Arc<TransferContext>, base_path: String, use_gds: bool) -> Self {
        Self {
            base,
            config: RemoteStorageConfig::Disk { base_path, use_gds },
            worker_id: 0,
        }
    }

    pub fn new(base: Arc<TransferContext>, config: RemoteStorageConfig) -> Self {
        Self {
            base,
            config,
            worker_id: 0,
        }
    }

    pub fn with_worker_id(mut self, worker_id: u64) -> Self {
        self.worker_id = worker_id;
        self
    }

    pub fn base(&self) -> &Arc<TransferContext> {
        &self.base
    }

    pub fn config(&self) -> &RemoteStorageConfig {
        &self.config
    }

    pub fn nixl_agent(&self) -> Arc<Option<NixlAgent>> {
        self.base.nixl_agent()
    }

    pub fn async_rt_handle(&self) -> &tokio::runtime::Handle {
        self.base.async_rt_handle()
    }

    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    pub fn default_bucket(&self) -> Option<&str> {
        match &self.config {
            RemoteStorageConfig::Object { default_bucket, .. } => default_bucket.as_deref(),
            _ => None,
        }
    }

    pub fn base_path(&self) -> Option<&str> {
        match &self.config {
            RemoteStorageConfig::Disk { base_path, .. } => Some(base_path),
            _ => None,
        }
    }
}

impl std::fmt::Debug for RemoteTransferContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteTransferContext")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod remote_storage_config_tests {
        use super::*;

        #[test]
        fn test_object_with_bucket() {
            let config = RemoteStorageConfig::object("my-bucket");
            match config {
                RemoteStorageConfig::Object {
                    default_bucket,
                    endpoint,
                    region,
                } => {
                    assert_eq!(default_bucket, Some("my-bucket".to_string()));
                    assert!(endpoint.is_none());
                    assert!(region.is_none());
                }
                _ => panic!("Expected Object variant"),
            }
        }

        #[test]
        fn test_object_with_options() {
            let config = RemoteStorageConfig::object_with_options(
                Some("test-bucket".to_string()),
                Some("http://localhost:9000".to_string()),
                Some("us-west-2".to_string()),
            );
            match config {
                RemoteStorageConfig::Object {
                    default_bucket,
                    endpoint,
                    region,
                } => {
                    assert_eq!(default_bucket, Some("test-bucket".to_string()));
                    assert_eq!(endpoint, Some("http://localhost:9000".to_string()));
                    assert_eq!(region, Some("us-west-2".to_string()));
                }
                _ => panic!("Expected Object variant"),
            }
        }

        #[test]
        fn test_object_with_no_bucket() {
            let config = RemoteStorageConfig::object_with_options(None, None, None);
            match config {
                RemoteStorageConfig::Object {
                    default_bucket,
                    endpoint,
                    region,
                } => {
                    assert!(default_bucket.is_none());
                    assert!(endpoint.is_none());
                    assert!(region.is_none());
                }
                _ => panic!("Expected Object variant"),
            }
        }

        #[test]
        fn test_disk_config() {
            let config = RemoteStorageConfig::disk("/mnt/kv-cache", false);
            match config {
                RemoteStorageConfig::Disk { base_path, use_gds } => {
                    assert_eq!(base_path, "/mnt/kv-cache");
                    assert!(!use_gds);
                }
                _ => panic!("Expected Disk variant"),
            }
        }

        #[test]
        fn test_disk_config_with_gds() {
            let config = RemoteStorageConfig::disk("/mnt/nvme", true);
            match config {
                RemoteStorageConfig::Disk { base_path, use_gds } => {
                    assert_eq!(base_path, "/mnt/nvme");
                    assert!(use_gds);
                }
                _ => panic!("Expected Disk variant"),
            }
        }

        #[test]
        fn test_config_clone() {
            let config = RemoteStorageConfig::object("bucket");
            let cloned = config.clone();
            match (config, cloned) {
                (
                    RemoteStorageConfig::Object {
                        default_bucket: b1, ..
                    },
                    RemoteStorageConfig::Object {
                        default_bucket: b2, ..
                    },
                ) => {
                    assert_eq!(b1, b2);
                }
                _ => panic!("Clone should preserve variant"),
            }
        }

        #[test]
        fn test_config_debug() {
            let config = RemoteStorageConfig::object("debug-bucket");
            let debug_str = format!("{:?}", config);
            assert!(debug_str.contains("Object"));
            assert!(debug_str.contains("debug-bucket"));
        }
    }

    mod remote_context_config_tests {
        use super::*;

        #[test]
        fn test_remote_context_config_object() {
            let config = RemoteContextConfig {
                remote_storage_config: RemoteStorageConfig::object("test-bucket"),
                worker_id: 42,
            };
            assert_eq!(config.worker_id, 42);
            match config.remote_storage_config {
                RemoteStorageConfig::Object { default_bucket, .. } => {
                    assert_eq!(default_bucket, Some("test-bucket".to_string()));
                }
                _ => panic!("Expected Object variant"),
            }
        }

        #[test]
        fn test_remote_context_config_disk() {
            let config = RemoteContextConfig {
                remote_storage_config: RemoteStorageConfig::disk("/data/cache", true),
                worker_id: 7,
            };
            assert_eq!(config.worker_id, 7);
            match config.remote_storage_config {
                RemoteStorageConfig::Disk { base_path, use_gds } => {
                    assert_eq!(base_path, "/data/cache");
                    assert!(use_gds);
                }
                _ => panic!("Expected Disk variant"),
            }
        }

        #[test]
        fn test_remote_context_config_clone() {
            let config = RemoteContextConfig {
                remote_storage_config: RemoteStorageConfig::object("clone-bucket"),
                worker_id: 123,
            };
            let cloned = config.clone();
            assert_eq!(cloned.worker_id, 123);
        }
    }
}
