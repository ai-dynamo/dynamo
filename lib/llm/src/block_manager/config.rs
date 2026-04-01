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

/// Bit flags for RemoteDiskStorage (G4) transfer backend selection.
///
/// Bit 0: use GDS_MT for offload (write)
/// Bit 1: use GDS_MT for onboard (read); falls back to POSIX if GDS_MT unavailable
///
/// Common combinations:
///   `DISK_FLAGS_GDS_BOTH`       (0b11) — GDS for both directions (default)
///   `DISK_FLAGS_GDS_READS_ONLY` (0b10) — POSIX write + GDS read (hybrid, works on any FS)
///   `DISK_FLAGS_POSIX_BOTH`     (0b00) — POSIX for both directions
pub type DiskTransferFlags = u8;
pub const DISK_FLAG_GDS_WRITE: DiskTransferFlags = 0b01;
pub const DISK_FLAG_GDS_READ: DiskTransferFlags = 0b10;
pub const DISK_FLAGS_GDS_BOTH: DiskTransferFlags = DISK_FLAG_GDS_WRITE | DISK_FLAG_GDS_READ;
pub const DISK_FLAGS_POSIX_BOTH: DiskTransferFlags = 0b00;
pub const DISK_FLAGS_GDS_READS_ONLY: DiskTransferFlags = DISK_FLAG_GDS_READ;

#[derive(Clone, Debug)]
pub enum RemoteStorageConfig {
    /// Object storage (S3-compatible).
    ///
    /// `bucket_template` may contain `{worker_id}` which is resolved per-TP-rank
    /// at transfer / registry time.  It must NOT be resolved early so that
    /// `register_tp` can derive the correct bucket for every worker.
    Object {
        bucket_template: Option<String>,
        endpoint: Option<String>,
        region: Option<String>,
        access_key: Option<String>,
        secret_key: Option<String>,
        session_token: Option<String>,
        scheme: Option<String>,
        use_virtual_addressing: Option<bool>,
        req_checksum: Option<String>,
        ca_bundle: Option<String>,
    },
    Disk {
        base_path: String,
        transfer_flags: DiskTransferFlags,
    },
}

impl RemoteStorageConfig {
    pub fn object(bucket: impl Into<String>) -> Self {
        Self::Object {
            bucket_template: Some(bucket.into()),
            endpoint: None,
            region: None,
            access_key: None,
            secret_key: None,
            session_token: None,
            scheme: None,
            use_virtual_addressing: None,
            req_checksum: None,
            ca_bundle: None,
        }
    }

    pub fn object_with_options(
        bucket: Option<String>,
        endpoint: Option<String>,
        region: Option<String>,
    ) -> Self {
        Self::Object {
            bucket_template: bucket,
            endpoint,
            region,
            access_key: None,
            secret_key: None,
            session_token: None,
            scheme: None,
            use_virtual_addressing: None,
            req_checksum: None,
            ca_bundle: None,
        }
    }

    /// Resolve `{worker_id}` in the bucket template for a specific TP rank.
    pub fn resolve_bucket(&self, worker_id: usize) -> Option<String> {
        match self {
            Self::Object {
                bucket_template, ..
            } => bucket_template
                .as_deref()
                .map(|t| t.replace("{worker_id}", &worker_id.to_string())),
            _ => None,
        }
    }

    pub fn disk(base_path: impl Into<String>, transfer_flags: DiskTransferFlags) -> Self {
        Self::Disk {
            base_path: base_path.into(),
            transfer_flags,
        }
    }
}

#[derive(Clone)]
pub struct RemoteTransferContext {
    base: Arc<TransferContext>,
    config: RemoteStorageConfig,
    worker_id: u64,
    world_size: usize,
}

#[derive(Clone)]
pub struct RemoteContextConfig {
    pub remote_storage_config: RemoteStorageConfig,
    pub worker_id: u64,
}

impl RemoteTransferContext {
    pub fn for_object(base: Arc<TransferContext>, bucket_template: Option<String>) -> Self {
        Self {
            base,
            config: RemoteStorageConfig::object_with_options(bucket_template, None, None),
            worker_id: 0,
            world_size: 1,
        }
    }

    pub fn for_object_with_options(
        base: Arc<TransferContext>,
        bucket_template: Option<String>,
        endpoint: Option<String>,
        region: Option<String>,
        worker_id: u64,
    ) -> Self {
        Self {
            base,
            config: RemoteStorageConfig::object_with_options(bucket_template, endpoint, region),
            worker_id,
            world_size: 1,
        }
    }

    pub fn for_disk(
        base: Arc<TransferContext>,
        base_path: String,
        transfer_flags: DiskTransferFlags,
    ) -> Self {
        Self {
            base,
            config: RemoteStorageConfig::Disk {
                base_path,
                transfer_flags,
            },
            worker_id: 0,
            world_size: 1,
        }
    }

    pub fn new(base: Arc<TransferContext>, config: RemoteStorageConfig) -> Self {
        Self {
            base,
            config,
            worker_id: 0,
            world_size: 1,
        }
    }

    pub fn with_topology(mut self, worker_id: u64, world_size: usize) -> Self {
        self.worker_id = worker_id;
        self.world_size = world_size;
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

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn bucket_template(&self) -> Option<&str> {
        match &self.config {
            RemoteStorageConfig::Object {
                bucket_template, ..
            } => bucket_template.as_deref(),
            _ => None,
        }
    }

    pub fn base_path(&self) -> Option<&str> {
        match &self.config {
            RemoteStorageConfig::Disk { base_path, .. } => Some(base_path),
            _ => None,
        }
    }

    pub fn disk_transfer_flags(&self) -> DiskTransferFlags {
        match &self.config {
            RemoteStorageConfig::Disk { transfer_flags, .. } => *transfer_flags,
            _ => DISK_FLAGS_GDS_BOTH,
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
                    bucket_template,
                    endpoint,
                    region,
                    ..
                } => {
                    assert_eq!(bucket_template, Some("my-bucket".to_string()));
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
                    bucket_template,
                    endpoint,
                    region,
                    ..
                } => {
                    assert_eq!(bucket_template, Some("test-bucket".to_string()));
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
                    bucket_template,
                    endpoint,
                    region,
                    ..
                } => {
                    assert!(bucket_template.is_none());
                    assert!(endpoint.is_none());
                    assert!(region.is_none());
                }
                _ => panic!("Expected Object variant"),
            }
        }

        #[test]
        fn test_resolve_bucket_with_worker_id() {
            let config = RemoteStorageConfig::object("kvcache-{worker_id}");
            assert_eq!(config.resolve_bucket(0), Some("kvcache-0".to_string()));
            assert_eq!(config.resolve_bucket(3), Some("kvcache-3".to_string()));
        }

        #[test]
        fn test_resolve_bucket_without_template() {
            let config = RemoteStorageConfig::object("flat-bucket");
            assert_eq!(config.resolve_bucket(0), Some("flat-bucket".to_string()));
            assert_eq!(config.resolve_bucket(5), Some("flat-bucket".to_string()));
        }

        #[test]
        fn test_disk_config_posix() {
            let config = RemoteStorageConfig::disk("/mnt/kv-cache", DISK_FLAGS_POSIX_BOTH);
            match config {
                RemoteStorageConfig::Disk {
                    base_path,
                    transfer_flags,
                } => {
                    assert_eq!(base_path, "/mnt/kv-cache");
                    assert_eq!(transfer_flags, DISK_FLAGS_POSIX_BOTH);
                }
                _ => panic!("Expected Disk variant"),
            }
        }

        #[test]
        fn test_disk_config_gds() {
            let config = RemoteStorageConfig::disk("/mnt/nvme", DISK_FLAGS_GDS_BOTH);
            match config {
                RemoteStorageConfig::Disk {
                    base_path,
                    transfer_flags,
                } => {
                    assert_eq!(base_path, "/mnt/nvme");
                    assert_eq!(transfer_flags, DISK_FLAGS_GDS_BOTH);
                }
                _ => panic!("Expected Disk variant"),
            }
        }

        #[test]
        fn test_disk_config_gds_reads_only() {
            let config = RemoteStorageConfig::disk("/mnt/nfs", DISK_FLAGS_GDS_READS_ONLY);
            match config {
                RemoteStorageConfig::Disk {
                    base_path,
                    transfer_flags,
                } => {
                    assert_eq!(base_path, "/mnt/nfs");
                    assert_eq!(transfer_flags & DISK_FLAG_GDS_READ, DISK_FLAG_GDS_READ);
                    assert_eq!(transfer_flags & DISK_FLAG_GDS_WRITE, 0);
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
                        bucket_template: b1,
                        ..
                    },
                    RemoteStorageConfig::Object {
                        bucket_template: b2,
                        ..
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
                RemoteStorageConfig::Object {
                    bucket_template, ..
                } => {
                    assert_eq!(bucket_template, Some("test-bucket".to_string()));
                }
                _ => panic!("Expected Object variant"),
            }
        }

        #[test]
        fn test_remote_context_config_disk() {
            let config = RemoteContextConfig {
                remote_storage_config: RemoteStorageConfig::disk(
                    "/data/cache",
                    DISK_FLAGS_GDS_BOTH,
                ),
                worker_id: 7,
            };
            assert_eq!(config.worker_id, 7);
            match config.remote_storage_config {
                RemoteStorageConfig::Disk {
                    base_path,
                    transfer_flags,
                } => {
                    assert_eq!(base_path, "/data/cache");
                    assert_eq!(transfer_flags, DISK_FLAGS_GDS_BOTH);
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
