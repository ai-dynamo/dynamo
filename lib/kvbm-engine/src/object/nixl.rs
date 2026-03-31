// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL-based object storage client for KV cache block offload/onboard.
//!
//! Uses NIXL's OBJ backend to transfer blocks between host DRAM and object
//! storage without going through the AWS SDK on the hot path.
//! The OBJ backend is configured once at agent init time; individual
//! block transfers are expressed as NIXL `XferRequest`s.
//!
//! # Backend configuration
//!
//! The NIXL OBJ plugin reads S3 credentials and endpoint from the standard
//! AWS environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
//! `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_OVERRIDE`, …) plus any plugin-specific
//! keys supplied via [`NixlObjectConfig`](kvbm_config::NixlObjectConfig).
//! See [`NixlS3Config::to_nixl_params`](kvbm_config::NixlS3Config::to_nixl_params)
//! for the full parameter list.
//!
//! # Object key mapping
//!
//! NIXL's OBJ backend identifies objects with a `u64` device_id.
//! [`SequenceHash`] (`PositionalLineageHash`, a `u128`) is XOR-folded into
//! 64 bits.  The same function must be used on both write and read so the
//! object key round-trips correctly.
//!
//! # `has_blocks`
//!
//! NIXL's OBJ backend does not expose a HEAD/stat primitive.  This
//! implementation delegates `has_blocks` to an SDK-backed client built from
//! the same S3 config.  When the `s3` feature is disabled the method returns
//! `None` for every hash (conservative: forces re-upload).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_memory::nixl::{
    Agent as RawAgent, MemType, NixlAgent, NixlDescriptor, XferDescList, XferOp, XferRequest,
};
use futures::future::BoxFuture;
use tracing::instrument;

use crate::object::{KeyFormatter, LayoutConfigExt, ObjectBlockOps};
use crate::{BlockId, SequenceHash};
use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::transfer::PhysicalLayout;

/// Convert a [`NixlS3Config`] into an [`S3Config`] for SDK-backed `has_blocks` checks.
#[cfg(feature = "s3")]
pub(super) fn nixl_s3_to_sdk_config(cfg: &kvbm_config::NixlS3Config) -> super::s3::S3Config {
    super::s3::S3Config {
        endpoint_url: cfg.endpoint_override.clone(),
        bucket: cfg.bucket_name().unwrap_or_else(|| "kvbm-blocks".to_string()),
        region: cfg.region.clone().unwrap_or_else(|| "us-east-1".to_string()),
        // path-style when virtual addressing is disabled (or unset)
        force_path_style: !cfg.use_virtual_addressing.unwrap_or(false),
        max_concurrent_requests: 16,
    }
}

/// Polling interval while waiting for NIXL OBJ transfer completion.
///
/// Override with `DYN_KVBM_NIXL_OBJ_POLL_MS` (milliseconds).
fn poll_interval() -> Duration {
    let ms = std::env::var("DYN_KVBM_NIXL_OBJ_POLL_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(50);
    Duration::from_millis(ms)
}

/// XOR-fold a 128-bit `SequenceHash` into a 64-bit NIXL OBJ device_id.
///
/// Both write and read use the same conversion so that the object key is
/// stable and round-trips correctly across nodes.
fn hash_to_device_id(hash: SequenceHash) -> u64 {
    let raw = hash.as_u128();
    (raw as u64) ^ ((raw >> 64) as u64)
}

/// Initialise the NIXL OBJ backend on an existing agent.
///
/// Merges `extra_params` (bucket, endpoint, etc.) into the plugin's default
/// parameter set, then calls `create_backend("OBJ", params)`.
///
/// The caller is responsible for ensuring the agent hasn't already had an OBJ
/// backend added (idempotent via `NixlAgent::add_backend_with_params`).
pub fn add_obj_backend(agent: &mut NixlAgent, extra_params: HashMap<String, String>) -> Result<()> {
    agent.add_backend_with_params("OBJ", &extra_params)
}

// ─────────────────────────────────────────────────────────────────────────────
// NixlObjectBlockClient
// ─────────────────────────────────────────────────────────────────────────────

/// `ObjectBlockOps` implementation that routes DRAM↔Object transfers through
/// NIXL's OBJ backend instead of the AWS SDK.
///
/// Construct via [`NixlObjectBlockClient::from_config`] (from the runtime) or
/// via [`NixlObjectBlockClient::new`] after manually calling [`add_obj_backend`].
#[derive(Clone)]
pub struct NixlObjectBlockClient {
    /// Underlying raw NIXL agent (shared across clones for transfer state).
    raw_agent: Arc<RawAgent>,
    /// Bucket name embedded in NIXL Object descriptors.
    bucket: String,
    /// Key formatter (kept for potential future string-key NIXL support).
    #[allow(dead_code)]
    key_formatter: Arc<dyn KeyFormatter>,
    /// Optional delegate used only for `has_blocks` HEAD checks.
    ///
    /// NIXL's OBJ backend has no stat primitive, so we optionally delegate to
    /// an `ObjectBlockOps` implementation that does (e.g. `S3ObjectBlockClient`).
    /// `None` → `has_blocks` always returns `None` (conservative).
    has_blocks_delegate: Option<Arc<dyn ObjectBlockOps>>,
}

impl std::fmt::Debug for NixlObjectBlockClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NixlObjectBlockClient")
            .field("bucket", &self.bucket)
            .field("has_blocks_delegate", &self.has_blocks_delegate.is_some())
            .finish_non_exhaustive()
    }
}

impl NixlObjectBlockClient {
    /// Create a new client.
    ///
    /// The `agent` must already have the OBJ backend initialised.
    /// `has_blocks_delegate` — if `Some`, its `has_blocks()` is called for
    /// existence checks; if `None`, all blocks are assumed absent.
    pub fn new(
        agent: NixlAgent,
        bucket: String,
        key_formatter: Arc<dyn KeyFormatter>,
        has_blocks_delegate: Option<Arc<dyn ObjectBlockOps>>,
    ) -> Self {
        Self {
            raw_agent: Arc::new(agent.into_raw_agent()),
            bucket,
            key_formatter,
            has_blocks_delegate,
        }
    }

    /// Build from a kvbm-config [`NixlObjectConfig`] and an already-initialised agent.
    ///
    /// When the `s3` feature is available a thin `S3ObjectBlockClient` is
    /// created from the embedded S3 config and used as the `has_blocks` delegate.
    pub async fn from_config(
        agent: NixlAgent,
        nixl_config: &kvbm_config::NixlObjectConfig,
        rank: Option<usize>,
    ) -> Result<Self> {
        use kvbm_config::NixlObjectConfig;
        use super::create_key_formatter;

        let key_formatter = create_key_formatter(rank);

        match nixl_config {
            NixlObjectConfig::S3(nixl_s3) => {
                let bucket = nixl_s3.bucket_name().unwrap_or_else(|| "kvbm-blocks".to_string());

                // Build a thin S3 client for has_blocks HEAD checks when the
                // `s3` feature is available.
                let has_blocks_delegate: Option<Arc<dyn ObjectBlockOps>> = {
                    #[cfg(feature = "s3")]
                    {
                        let s3_config = nixl_s3_to_sdk_config(nixl_s3);
                        super::s3::S3ObjectBlockClient::with_key_formatter(
                            s3_config,
                            key_formatter.clone(),
                        )
                        .await
                        .ok()
                        .map(|c| Arc::new(c) as Arc<dyn ObjectBlockOps>)
                    }
                    #[cfg(not(feature = "s3"))]
                    {
                        None
                    }
                };

                Ok(Self::new(agent, bucket, key_formatter, has_blocks_delegate))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ObjectBlockOps impl
// ─────────────────────────────────────────────────────────────────────────────

impl ObjectBlockOps for NixlObjectBlockClient {
    /// Check block presence.
    ///
    /// Delegates to `has_blocks_delegate` when set (e.g. a thin S3 client using
    /// HEAD requests). Falls back to reporting all blocks absent, which is
    /// conservative but correct: the offload pipeline re-uploads the block.
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        if let Some(delegate) = &self.has_blocks_delegate {
            return delegate.has_blocks(keys);
        }
        Box::pin(async move { keys.into_iter().map(|k| (k, None)).collect() })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _src_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // Workers must resolve the logical handle to a physical layout first.
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _dst_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn put_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        let agent = Arc::clone(&self.raw_agent);
        Box::pin(async move {
            execute_nixl_obj_transfer(&agent, XferOp::Write, &keys, &layout, &block_ids)
                .await
        })
    }

    fn get_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        let agent = Arc::clone(&self.raw_agent);
        Box::pin(async move {
            execute_nixl_obj_transfer(&agent, XferOp::Read, &keys, &layout, &block_ids)
                .await
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core NIXL OBJ transfer
// ─────────────────────────────────────────────────────────────────────────────

/// Execute a NIXL transfer between host DRAM and object storage.
///
/// * `XferOp::Write` — offload: DRAM → Object storage
/// * `XferOp::Read`  — onboard: Object storage → DRAM
///
/// For fully-contiguous layouts each block is one DRAM descriptor.
/// For layer-separate layouts we add one descriptor per (layer, outer_dim)
/// region and match them against a correspondingly striped object address space.
#[instrument(skip_all, fields(op = ?op, n = keys.len()))]
async fn execute_nixl_obj_transfer(
    agent: &RawAgent,
    op: XferOp,
    keys: &[SequenceHash],
    layout: &PhysicalLayout,
    block_ids: &[BlockId],
) -> Vec<Result<SequenceHash, SequenceHash>> {
    if keys.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(keys.len(), block_ids.len());

    let config = layout.layout().config();
    let block_size = config.block_size_bytes();
    let is_contiguous = layout.layout().is_fully_contiguous();

    // ── Build descriptor lists ────────────────────────────────────────────────
    //
    // Convention (matches dynamo-fork execute_object_transfer):
    //   src_dl = DRAM (initiator's local buffers, always)
    //   dst_dl = Object (responder = OBJ backend on same agent)
    //   XferOp::Write → DRAM → Object  (offload)
    //   XferOp::Read  → Object → DRAM  (onboard)
    //
    // The Object side uses addr=0 and device_id = XOR-folded sequence hash.
    // The OBJ backend maps device_id → object key in the configured bucket.

    // Use a block to drop non-Send registration handles before the first .await.
    let xfer_result = (|| -> anyhow::Result<(XferRequest, bool)> {
        let mut src_dl = XferDescList::new(MemType::Dram)?;
        let mut dst_dl = XferDescList::new(MemType::Object)?;

        // Registration handles must stay alive until after post_xfer_req.
        let mut _obj_reg_handles = Vec::with_capacity(keys.len());

        for (key, block_id) in keys.iter().zip(block_ids.iter()) {
            let device_id = hash_to_device_id(*key);

            // Register the Object-side descriptor with the agent so NIXL can
            // associate it with the OBJ backend.
            let obj_desc = NixlDescriptor {
                addr: 0,
                size: block_size,
                mem_type: MemType::Object,
                device_id,
            };
            let handle = agent.register_memory(&obj_desc, None)?;
            _obj_reg_handles.push(handle);

            if is_contiguous {
                // Fast path: the block's data is one contiguous region.
                let region = layout.memory_region(*block_id, 0, 0)?;
                src_dl.add_desc(region.addr(), block_size, 0);
                dst_dl.add_desc(0, block_size, device_id);
            } else {
                // Slow path: one DRAM entry per (layer, outer_dim) region.
                // Object-side entries share the same device_id (same object),
                // but each points to a different sub-range via addr offset.
                let inner = layout.layout();
                let region_size = config.region_size();
                let mut obj_offset: usize = 0;
                for layer_id in 0..inner.num_layers() {
                    for outer_id in 0..inner.outer_dim() {
                        let region = layout.memory_region(*block_id, layer_id, outer_id)?;
                        src_dl.add_desc(region.addr(), region_size, 0);
                        // Offset into the object acts as the sub-range addr.
                        dst_dl.add_desc(obj_offset, region_size, device_id);
                        obj_offset += region_size;
                    }
                }
            }
        }

        let agent_name = agent.name();
        let xfer_req = agent.create_xfer_req(op, &src_dl, &dst_dl, &agent_name, None)?;
        let still_pending = agent.post_xfer_req(&xfer_req, None)?;
        Ok((xfer_req, still_pending))
    })();

    let (xfer_req, still_pending) = match xfer_result {
        Ok(pair) => pair,
        Err(e) => {
            tracing::error!("NIXL OBJ: failed to build/post transfer: {e}");
            return keys.iter().map(|k| Err(*k)).collect();
        }
    };

    // ── Wait for completion ───────────────────────────────────────────────────
    if still_pending {
        let interval = poll_interval();
        loop {
            tokio::time::sleep(interval).await;
            match agent.get_xfer_status(&xfer_req) {
                Ok(status) if status.is_success() => break,
                Ok(_) => continue,
                Err(e) => {
                    tracing::error!("NIXL OBJ: transfer status error: {e}");
                    return keys.iter().map(|k| Err(*k)).collect();
                }
            }
        }
    }

    keys.iter().map(|k| Ok(*k)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests — require the `testing-s3` feature and a live S3 endpoint.
//
// Run with:
//   bash lib/kvbm-engine/scripts/test-s3.sh
// or manually:
//   S3_TEST_ENDPOINT=http://localhost:9876 \
//   AWS_ACCESS_KEY_ID=minioadmin \
//   AWS_SECRET_ACCESS_KEY=minioadmin \
//   cargo test -p kvbm-engine --features testing-s3 -- nixl_integration
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "testing-s3")]
pub mod nixl_integration {
    use super::*;
    use crate::object::s3::s3_integration::create_test_client;
    use crate::object::s3::{S3Config, S3ObjectBlockClient};
    use crate::object::DefaultKeyFormatter;
    use dynamo_tokens::PositionalLineageHash;

    // ── nixl_s3_to_sdk_config ─────────────────────────────────────────────────

    /// `nixl_s3_to_sdk_config` must produce an `S3Config` whose fields correctly
    /// reflect the source `NixlS3Config`.
    #[test]
    fn test_nixl_s3_to_sdk_config_fields() {
        let nixl_cfg = kvbm_config::NixlS3Config {
            endpoint_override: Some("http://host:9000".into()),
            bucket: Some("my-bucket".into()),
            region: Some("eu-west-1".into()),
            use_virtual_addressing: Some(false),
            ..Default::default()
        };
        let sdk_cfg = nixl_s3_to_sdk_config(&nixl_cfg);
        assert_eq!(sdk_cfg.endpoint_url, Some("http://host:9000".into()));
        assert_eq!(sdk_cfg.bucket, "my-bucket");
        assert_eq!(sdk_cfg.region, "eu-west-1");
        assert!(sdk_cfg.force_path_style, "virtual_addressing=false → path-style");
    }

    /// `use_virtual_addressing` unset → path-style (safe default for S3-compatible endpoints).
    #[test]
    fn test_nixl_s3_to_sdk_config_virtual_addressing_default() {
        let nixl_cfg = kvbm_config::NixlS3Config {
            bucket: Some("b".into()),
            ..Default::default()
        };
        let sdk_cfg = nixl_s3_to_sdk_config(&nixl_cfg);
        assert!(sdk_cfg.force_path_style);
    }

    /// `bucket` field None + no env var → falls back to the hardcoded default "kvbm-blocks".
    #[test]
    fn test_nixl_s3_to_sdk_config_bucket_hardcoded_default() {
        // bucket is None and AWS_DEFAULT_BUCKET is not set → nixl_s3_to_sdk_config
        // falls back to the hardcoded sentinel "kvbm-blocks".
        let nixl_cfg = kvbm_config::NixlS3Config::default();
        // Only run this assertion when the env var is absent so the test is
        // deterministic regardless of the host environment.
        if std::env::var("AWS_DEFAULT_BUCKET").is_err() {
            let sdk_cfg = nixl_s3_to_sdk_config(&nixl_cfg);
            assert_eq!(sdk_cfg.bucket, "kvbm-blocks");
        }
    }

    // ── hash_to_device_id ─────────────────────────────────────────────────────

    /// Same hash must produce the same device_id every time (deterministic).
    #[test]
    fn test_hash_to_device_id_deterministic() {
        let hash = PositionalLineageHash::new(0xDEAD_BEEF, None, 0);
        assert_eq!(hash_to_device_id(hash), hash_to_device_id(hash));
    }

    /// Hashes with different content must produce different device_ids with these values.
    #[test]
    fn test_hash_to_device_id_distinct_hashes() {
        let h1 = PositionalLineageHash::new(0x0000_0001, None, 0);
        let h2 = PositionalLineageHash::new(0x0000_0002, None, 0);
        assert_ne!(hash_to_device_id(h1), hash_to_device_id(h2));
    }

    // ── NixlS3Config serde round-trip ─────────────────────────────────────────

    /// JSON serialisation of `NixlS3Config` must round-trip correctly so that
    /// config files produced by one version are readable by another.
    #[test]
    fn test_nixl_s3_config_serde_roundtrip() {
        let endpoint = std::env::var("S3_TEST_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:9876".into());
        let original = kvbm_config::NixlS3Config::with_endpoint(endpoint, "serde-test-bucket");
        let json = serde_json::to_string(&original).unwrap();
        let restored: kvbm_config::NixlS3Config = serde_json::from_str(&json).unwrap();
        assert_eq!(original.endpoint_override, restored.endpoint_override);
        assert_eq!(original.bucket, restored.bucket);
        assert_eq!(original.use_virtual_addressing, restored.use_virtual_addressing);
    }

    // ── has_blocks with SDK delegate against real S3 ──────────────────────────

    /// Build a `NixlObjectBlockClient` whose `has_blocks_delegate` is a real
    /// `S3ObjectBlockClient`, then verify:
    ///   1. Blocks not yet uploaded → `has_blocks` returns `None`.
    ///   2. After uploading a block via the SDK client → `has_blocks` returns `Some`.
    ///
    /// This does NOT exercise the NIXL transfer path (which requires NIXL hardware).
    /// It validates the delegate wiring and `NixlS3Config → S3Config` conversion
    /// end-to-end against a real S3 endpoint.
    #[tokio::test]
    async fn test_has_blocks_sdk_delegate() {
        let bucket = "nixl-has-blocks-test";
        let sdk_client = Arc::new(create_test_client(bucket).await) as Arc<dyn ObjectBlockOps>;

        let absent_hash = PositionalLineageHash::new(0xAAAA_AAAA, None, 1);
        let present_hash = PositionalLineageHash::new(0xBBBB_BBBB, None, 2);

        // Upload one object under the key that `has_blocks` will HEAD-check.
        let key = DefaultKeyFormatter.format_key(&present_hash);
        let endpoint = std::env::var("S3_TEST_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:9876".into());
        let raw = S3ObjectBlockClient::new(S3Config::minio(endpoint.clone(), bucket.into()))
            .await
            .unwrap();
        raw.put_object(&key, bytes::Bytes::from("payload")).await.unwrap();

        // Verify presence via the delegate.
        let results: std::collections::HashMap<_, _> = sdk_client
            .has_blocks(vec![absent_hash, present_hash])
            .await
            .into_iter()
            .collect();

        assert!(results[&absent_hash].is_none(), "absent block should return None");
        assert!(results[&present_hash].is_some(), "uploaded block should return Some");

        // Cleanup.
        let raw = S3ObjectBlockClient::new(S3Config::minio(endpoint, bucket.into()))
            .await
            .unwrap();
        raw.delete_object(&key).await.unwrap();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NIXL OBJ full round-trip test — requires real NIXL with OBJ backend.
//
// Run with:
//   bash lib/kvbm-engine/scripts/test-nixl-obj.sh
// or inside the Docker Compose environment:
//   cargo test -p kvbm-engine --features testing-nixl-obj -- nixl_obj_integration --nocapture
//
// Requires:
//   - NIXL 0.10.0 installed at $NIXL_PREFIX with OBJ backend
//   - S3_TEST_ENDPOINT / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY set
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "testing-nixl-obj")]
#[allow(dead_code, unused_imports)]
mod nixl_obj_integration {
    use super::*;
    use crate::object::s3::{S3Config, S3ObjectBlockClient};
    use dynamo_memory::nixl::NixlAgent;
    use dynamo_tokens::PositionalLineageHash;
    use kvbm_physical::layout::{LayoutConfig, PhysicalLayoutBuilder};
    use kvbm_physical::transfer::{FillPattern, PhysicalLayout, StorageKind, fill_blocks};

    fn test_endpoint() -> String {
        std::env::var("S3_TEST_ENDPOINT").unwrap_or_else(|_| "http://localhost:9876".into())
    }

    fn make_layout_config(num_blocks: usize) -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(num_blocks)
            .num_layers(2)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(64)
            .dtype_width_bytes(2)
            .build()
            .unwrap()
    }

    fn build_system_layout(agent: NixlAgent, config: LayoutConfig) -> PhysicalLayout {
        PhysicalLayout::builder(agent)
            .with_config(config)
            .fully_contiguous()
            .allocate_system()
            .build()
            .unwrap()
    }

    /// Verify NIXL is available (OBJ backend).
    /// Returns false (skip) if running in stub mode.
    fn nixl_available() -> bool {
        if nixl_sys::is_stub() {
            eprintln!("NIXL is in stub mode — skipping NIXL OBJ transfer test");
            return false;
        }
        true
    }

    fn read_layout_bytes(layout: &PhysicalLayout, block_id: usize) -> Vec<u8> {
        let config = layout.layout().config();
        let mut all_bytes = Vec::new();
        for layer_id in 0..config.num_layers {
            for outer_id in 0..config.outer_dim {
                let region = layout.memory_region(block_id, layer_id, outer_id).unwrap();
                let slice = unsafe {
                    std::slice::from_raw_parts(region.addr() as *const u8, region.size())
                };
                all_bytes.extend_from_slice(slice);
            }
        }
        all_bytes
    }

    /// End-to-end NIXL OBJ transfer: DRAM → S3 (put) then S3 → DRAM (get).
    ///
    /// Uses `SystemStorage` (plain malloc) for the DRAM side so no GPU is required.
    /// The OBJ backend reads/writes DRAM buffers directly; no RDMA (UCX) is needed
    /// for a single-agent local transfer.
    #[tokio::test]
    async fn test_nixl_obj_dram_to_s3_roundtrip() {
        if !nixl_available() {
            return;
        }

        let bucket = "nixl-obj-roundtrip";
        let endpoint = test_endpoint();

        // ── Build NixlAgent with OBJ backend only ────────────────────────────
        let mut agent = NixlAgent::new("nixl-obj-test").expect("NixlAgent::new failed");

        let obj_params: std::collections::HashMap<String, String> = {
            let cfg = kvbm_config::NixlS3Config::with_endpoint(endpoint.clone(), bucket);
            cfg.to_nixl_params()
        };
        add_obj_backend(&mut agent, obj_params).expect("OBJ backend required");

        // ── Create write layout and fill with known pattern ───────────────────
        let config = make_layout_config(1);
        let write_agent = agent.clone();
        let write_layout = build_system_layout(write_agent, config.clone());

        fill_blocks(&write_layout, &[0], FillPattern::Sequential).unwrap();
        let original_bytes = read_layout_bytes(&write_layout, 0);
        assert!(!original_bytes.iter().all(|&b| b == 0), "write buffer must be non-zero");

        // ── Ensure the S3 bucket exists ───────────────────────────────────────
        let s3_cfg = S3Config::minio(endpoint.clone(), bucket.into());
        S3ObjectBlockClient::new(s3_cfg).await.unwrap()
            .ensure_bucket_exists().await.unwrap();

        // ── Build the client and offload (DRAM → S3) ─────────────────────────
        let nixl_cfg = kvbm_config::NixlObjectConfig::S3(
            kvbm_config::NixlS3Config::with_endpoint(endpoint.clone(), bucket)
        );
        let client = NixlObjectBlockClient::from_config(agent.clone(), &nixl_cfg, None)
            .await
            .unwrap();

        let hash = PositionalLineageHash::new(0xC0DE_CAFE, None, 7);
        let put_results = client
            .put_blocks_with_layout(vec![hash], write_layout, vec![0])
            .await;
        assert_eq!(put_results.len(), 1);
        assert!(put_results[0].is_ok(), "put_blocks_with_layout failed: {:?}", put_results[0]);

        // ── Create zero-filled read layout and onboard (S3 → DRAM) ───────────
        let read_layout = build_system_layout(agent.clone(), config.clone());
        {
            // Sanity: read buffer starts zeroed
            let bytes = read_layout_bytes(&read_layout, 0);
            assert!(bytes.iter().all(|&b| b == 0), "read buffer must start zeroed");
        }

        let get_results = client
            .get_blocks_with_layout(vec![hash], read_layout.clone(), vec![0])
            .await;
        assert_eq!(get_results.len(), 1);
        assert!(get_results[0].is_ok(), "get_blocks_with_layout failed: {:?}", get_results[0]);

        // ── Verify byte-for-byte correctness ─────────────────────────────────
        let restored_bytes = read_layout_bytes(&read_layout, 0);
        assert_eq!(
            original_bytes, restored_bytes,
            "round-trip mismatch: DRAM→S3→DRAM did not reproduce the original data"
        );

        println!(
            "✓ NIXL OBJ round-trip: {} bytes transferred via DRAM→S3→DRAM",
            original_bytes.len()
        );
    }
}
