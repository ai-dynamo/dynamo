// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport manager for local and remote physical layouts with transfer execution.

mod handle;
mod local;
mod metadata;
mod remote;

pub use handle::LayoutHandle;
pub use metadata::{SerializedLayout, WorkerAddress};

pub(crate) use local::LocalLayout;
pub(crate) use metadata::LocalLayoutDescriptor;
pub(crate) use remote::RemoteLayout;

use crate::block_manager::config::ObjectStorageConfig;
use crate::block_manager::v2::memory::{ObjectStorage, StorageKind};
use crate::block_manager::v2::physical::layout::PhysicalLayout;
use crate::block_manager::v2::physical::transfer::TransferContext;
use crate::block_manager::v2::physical::transfer::context::TransferCompleteNotification;
use crate::block_manager::v2::physical::transfer::nixl_agent::NixlAgent;
use crate::block_manager::v2::physical::transfer::options::TransferOptions;
use anyhow::{Result, anyhow, bail};
use nixl_sys::{MemType, XferDescList, XferOp};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, RwLock};

/// Direction of G4 (object storage) transfer.
#[derive(Debug, Clone, Copy)]
pub enum G4TransferDirection {
    /// Local → Object Storage (offload/persist)
    Offload,
    /// Object Storage → Local (onboard/restore)
    Onboard,
}

/// Result of a G4 transfer operation.
#[derive(Debug)]
pub struct G4TransferResult {
    /// Number of blocks successfully transferred
    pub transferred: usize,
}

/// Descriptor for a remote object in G4 (object storage) transfers.
///
/// This encapsulates all the information needed to identify and transfer
/// a single block to/from object storage.
#[derive(Debug, Clone)]
pub struct RemoteDescriptor {
    /// Object key (typically a sequence hash) identifying the object
    pub object_key: u64,
    /// Resolved bucket name for the object
    pub bucket: String,
}

impl RemoteDescriptor {
    /// Create a new RemoteDescriptor.
    ///
    /// # Arguments
    /// * `object_key` - The key identifying the object (typically sequence hash)
    /// * `bucket` - The resolved bucket name
    pub fn new(object_key: u64, bucket: impl Into<String>) -> Self {
        Self {
            object_key,
            bucket: bucket.into(),
        }
    }

    /// Create RemoteDescriptors from object keys and config.
    ///
    /// # Arguments
    /// * `object_keys` - Slice of object keys
    /// * `config` - Object storage configuration
    /// * `worker_id` - Worker ID for bucket resolution
    pub fn from_keys(
        object_keys: &[u64],
        config: &ObjectStorageConfig,
        worker_id: u32,
    ) -> Vec<Self> {
        let bucket = config.resolve_bucket(worker_id);
        object_keys
            .iter()
            .map(|&key| Self::new(key, bucket.clone()))
            .collect()
    }
}

/// Public entry point for layout and transfer management.
///
/// TransportManager combines layout registration/metadata management with
/// transfer execution capabilities, providing a unified API for:
/// - Registering local layouts and obtaining handles
/// - Exporting/importing layout metadata for remote workers
/// - Executing transfers between layouts using handles
/// - Managing CUDA, NIXL, and other execution resources
#[derive(Clone)]
pub struct TransportManager {
    registry: Arc<RwLock<LayoutRegistry>>,
    context: Arc<TransferContext>,
    // we need to have a way to provdie a g4 manager here?
    // it needs access to the external registry client
}

impl TransportManager {
    /// Create a new TransportManager builder.
    ///
    /// The builder configures the worker ID, NIXL agent, CUDA device,
    /// and other execution parameters before creating the manager.
    ///
    /// # Example
    /// ```ignore
    /// let manager = TransportManager::builder()
    ///     .worker_id(0)  // NIXL agent name defaults to "worker-0"
    ///     .nixl_backend("ucx")  // Optional: defaults to UCX from env
    ///     .cuda_device_id(0)
    ///     .build()?;
    ///
    /// // Or with custom agent name:
    /// let manager = TransportManager::builder()
    ///     .worker_id(0)
    ///     .nixl_agent_name("custom-agent")
    ///     .build()?;
    /// ```
    pub fn builder() -> crate::block_manager::v2::physical::transfer::context::TransferConfigBuilder
    {
        TransferContext::builder()
    }

    /// Create a TransportManager from a built TransferContext.
    ///
    /// This is used internally by the builder to wrap the context
    /// and create the associated registry.
    pub(crate) fn from_context(context: TransferContext) -> Self {
        let worker_id = context.worker_id();
        let nixl_agent = context.nixl_agent().clone();
        let registry = Arc::new(RwLock::new(LayoutRegistry::new(nixl_agent, worker_id)));

        Self {
            registry,
            context: Arc::new(context),
        }
    }

    // ===== Layout Registration and Metadata Management =====

    /// Register a local physical layout and return a unique handle.
    ///
    /// This registers the layout with the embedded memory manager, assigning
    /// it a unique handle that can be used for handle-based transfers.
    ///
    /// # Arguments
    /// * `layout` - Physical layout to register
    ///
    /// # Returns
    /// Unique handle for the registered layout
    ///
    /// # Errors
    /// Returns an error if layout IDs are exhausted (u16::MAX reached)
    pub fn register_layout(&self, layout: PhysicalLayout) -> Result<LayoutHandle> {
        self.registry.write().unwrap().register_local(layout)
    }

    /// Get a layout by handle, returning a clone.
    ///
    /// # Arguments
    /// * `handle` - Handle to the layout
    ///
    /// # Returns
    /// Returns Some(PhysicalLayout) if found, None otherwise
    pub fn get_layout(&self, handle: LayoutHandle) -> Option<PhysicalLayout> {
        self.registry
            .read()
            .unwrap()
            .get_layout(handle)
            .cloned()
    }

    /// Export layout metadata for transmission to remote workers.
    ///
    /// This exports all registered local layouts along with NIXL metadata
    /// needed for remote memory registration.
    ///
    /// # Returns
    /// Packed metadata ready for transmission to remote workers
    pub fn export_metadata(&self) -> Result<SerializedLayout> {
        self.registry.read().unwrap().export_metadata()
    }

    /// Import remote layout metadata.
    ///
    /// This loads NIXL metadata and reconstructs physical layouts from a remote
    /// worker's exported metadata.
    ///
    /// # Arguments
    /// * `metadata` - Packed metadata from remote worker
    ///
    /// # Returns
    /// Vector of handles for the imported remote layouts
    ///
    /// # Errors
    /// Returns an error if the remote worker was already loaded or if metadata
    /// loading/reconstruction fails
    pub fn import_metadata(&self, metadata: SerializedLayout) -> Result<Vec<LayoutHandle>> {
        self.registry.write().unwrap().import_metadata(metadata)
    }

    // ===== Handle-Based Transfer API =====

    /// Transfer complete blocks between layouts using handles.
    ///
    /// This function copies entire blocks (all layers and outer dimensions) between
    /// the source and destination layouts identified by their handles. The transfer
    /// strategy (memcpy, CUDA, NIXL) is automatically selected based on storage locations.
    ///
    /// The lock on the registry is held only briefly during layout lookup,
    /// then released before executing the actual transfer.
    ///
    /// # Arguments
    /// * `src_handle` - Handle to source layout
    /// * `src_blocks` - Source block IDs to transfer
    /// * `dst_handle` - Handle to destination layout
    /// * `dst_blocks` - Destination block IDs to transfer
    ///
    /// # Returns
    /// A notification handle that can be awaited for transfer completion
    ///
    /// # Errors
    /// Returns an error if:
    /// - Either handle is invalid
    /// - Block IDs are out of bounds
    /// - Transfer execution fails
    pub fn execute_transfer(
        &self,
        src_handle: LayoutHandle,
        src_blocks: &[usize],
        dst_handle: LayoutHandle,
        dst_blocks: &[usize],
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        // Clone layouts inside the lock, then drop lock before transfer
        let (src_layout, dst_layout) = {
            let registry = self.registry.read().unwrap();
            let src = registry
                .get_layout(src_handle)
                .ok_or_else(|| anyhow!("invalid source handle: {}", src_handle))?
                .clone(); // Cheap: just Arc refcount bump
            let dst = registry
                .get_layout(dst_handle)
                .ok_or_else(|| anyhow!("invalid destination handle: {}", dst_handle))?
                .clone();
            (src, dst)
        }; // Lock released here

        // Execute transfer with no lock held
        super::transfer::executor::execute_transfer(
            &src_layout,
            &dst_layout,
            src_blocks,
            dst_blocks,
            options,
            &self.context,
        )
    }



    /// Execute a G4 (object storage) transfer.
    ///
    /// This is the main entry point for object storage transfers. It delegates
    /// to either offload or onboard based on the direction.
    ///
    /// # Arguments
    /// * `direction` - Whether to offload (local→object) or onboard (object→local)
    /// * `local_handle` - Handle to the local layout (device or host)
    /// * `local_blocks` - Block indices in the local layout
    /// * `remote_descriptors` - Remote descriptors containing object keys and bucket info
    ///
    /// # Returns
    /// Result containing the number of blocks successfully transferred
    pub fn execute_g4_transfer(
        &self,
        direction: G4TransferDirection,
        local_handle: LayoutHandle,
        local_blocks: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        match direction {
            G4TransferDirection::Offload => {
                self.execute_g4_offload_with_descriptors(local_handle, local_blocks, remote_descriptors)
            }
            G4TransferDirection::Onboard => {
                self.execute_g4_onboard_with_descriptors(local_handle, local_blocks, remote_descriptors)
            }
        }
    }

    /// Async version of [`Self::execute_g4_transfer`].
    ///
    /// Use this in async contexts to avoid blocking the runtime.
    pub async fn execute_g4_transfer_async(
        &self,
        direction: G4TransferDirection,
        local_handle: LayoutHandle,
        local_blocks: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        match direction {
            G4TransferDirection::Offload => {
                self.execute_g4_offload_with_descriptors_async(local_handle, local_blocks, remote_descriptors).await
            }
            G4TransferDirection::Onboard => {
                self.execute_g4_onboard_with_descriptors_async(local_handle, local_blocks, remote_descriptors).await
            }
        }
    }

    /// Execute G4 offload: transfer blocks from local layout to object storage.
    ///
    /// Transfers blocks from a local layout (device/host) to object storage using
    /// NIXL's OBJ plugin. Each block is stored as a separate object with the key
    /// derived from the sequence hash.
    ///
    /// # Arguments
    /// * `local_handle` - Handle to the source layout (device or host)
    /// * `local_blocks` - Block indices in the source layout to offload
    /// * `object_keys` - Object keys (typically sequence hashes) for each block
    /// * `host_handle` - Host pool handle for bounce buffers (used if source is device)
    /// * `config` - Object storage configuration (bucket, endpoint, etc.)
    ///
    /// # Returns
    /// Result containing the number of blocks successfully transferred
    pub fn execute_g4_offload(
        &self,
        local_handle: LayoutHandle,
        local_blocks: &[usize],
        object_keys: &[u64],
        _host_handle: LayoutHandle,
        config: &ObjectStorageConfig,
    ) -> Result<G4TransferResult> {
        if local_blocks.len() != object_keys.len() {
            bail!(
                "local_blocks.len() ({}) != object_keys.len() ({})",
                local_blocks.len(),
                object_keys.len()
            );
        }

        if local_blocks.is_empty() {
            return Ok(G4TransferResult { transferred: 0 });
        }

        // Get source layout from registry
        let src_layout = {
            let registry = self.registry.read().unwrap();
            registry
                .get_layout(local_handle)
                .ok_or_else(|| anyhow!("invalid source handle: {}", local_handle))?
                .clone()
        };

        let nixl_agent = self.context.nixl_agent();

        // Validate NIXL has OBJ backend loaded
        if !nixl_agent.has_backend("OBJ") {
            bail!("NIXL OBJ backend not available for object storage transfers");
        }

        let src_layout_inner = src_layout.layout();
        let layout_config = src_layout_inner.config();

        // Calculate the size of one contiguous block
        // block_size = num_layers * outer_dim * page_size * inner_dim * dtype_width_bytes
        let block_size = layout_config.num_layers
            * layout_config.outer_dim
            * layout_config.page_size
            * layout_config.inner_dim
            * layout_config.dtype_width_bytes;

        // Get source NIXL metadata
        let src_metadata = src_layout.nixl_metadata();
        let src_mem_type = src_metadata.mem_type();
        let src_device_id = src_metadata.device_id();

        // Resolve the bucket name for this worker
        let bucket = config.resolve_bucket(self.worker_id() as u32);

        // Register object storage slots with NIXL
        // Each object is registered with its key as the device_id
        let mut obj_storages = Vec::with_capacity(object_keys.len());
        let mut registration_handles = Vec::with_capacity(object_keys.len());

        for &key in object_keys.iter() {
            let obj_storage = ObjectStorage::new(&bucket, key, block_size)
                .map_err(|e| anyhow!("Failed to create object storage: {:?}", e))?;

            let handle = nixl_agent
                .register_memory(&obj_storage, None)
                .map_err(|e| anyhow!("Failed to register object storage: {:?}", e))?;

            obj_storages.push(obj_storage);
            registration_handles.push(handle);
        }

        // Build transfer descriptor lists
        let mut src_dl = XferDescList::new(src_mem_type)?;
        let mut dst_dl = XferDescList::new(MemType::Object)?;

        // Add one descriptor per block
        for (&src_block_id, &key) in local_blocks.iter().zip(object_keys.iter()) {
            // Get the base address of the block (first layer, first outer dim)
            let src_region = src_layout.memory_region(src_block_id, 0, 0)?;

            // Add to source descriptor list - full contiguous block
            src_dl.add_desc(src_region.addr(), block_size, src_device_id);

            // Add to destination descriptor list
            // The object key is used as the device_id for the OBJ backend
            dst_dl.add_desc(0, block_size, key);
        }

        // Create transfer request using loopback (agent's own name for OBJ plugin)
        let agent_name = nixl_agent.name();
        let xfer_req = nixl_agent.create_xfer_req(
            XferOp::Write,
            &src_dl,
            &dst_dl,
            &agent_name,
            None,
        )?;

        // Post transfer request
        let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;

        // Keep registration handles alive until transfer completes
        // For sync completion, we can drop them immediately after posting
        let transferred = local_blocks.len();

        // Drop registration handles - safe because transfer has been posted
        drop(registration_handles);
        drop(obj_storages);

        if still_pending {
            // For async completion, we need to wait for it
            // Register for completion and block on it
            let notification = self.context.register_nixl_status(xfer_req);
            // Block synchronously since this method is not async
            notification.wait()?;
        }

        Ok(G4TransferResult { transferred })
    }

    /// Execute G4 onboard: transfer blocks from object storage to local layout.
    ///
    /// Transfers blocks from object storage to a local layout (device/host) using
    /// NIXL's OBJ plugin. Each object is identified by its key (sequence hash).
    ///
    /// # Arguments
    /// * `local_handle` - Handle to the destination layout (device or host)
    /// * `local_blocks` - Block indices in the destination layout
    /// * `object_keys` - Object keys (typically sequence hashes) to retrieve
    /// * `host_handle` - Host pool handle for bounce buffers (used if dest is device)
    /// * `config` - Object storage configuration (bucket, endpoint, etc.)
    ///
    /// # Returns
    /// Result containing the number of blocks successfully transferred
    pub fn execute_g4_onboard(
        &self,
        local_handle: LayoutHandle,
        local_blocks: &[usize],
        object_keys: &[u64],
        _host_handle: LayoutHandle,
        config: &ObjectStorageConfig,
    ) -> Result<G4TransferResult> {
        if local_blocks.len() != object_keys.len() {
            bail!(
                "local_blocks.len() ({}) != object_keys.len() ({})",
                local_blocks.len(),
                object_keys.len()
            );
        }

        if local_blocks.is_empty() {
            return Ok(G4TransferResult { transferred: 0 });
        }

        // Get destination layout from registry
        let dst_layout = {
            let registry = self.registry.read().unwrap();
            registry
                .get_layout(local_handle)
                .ok_or_else(|| anyhow!("invalid destination handle: {}", local_handle))?
                .clone()
        };

        let nixl_agent = self.context.nixl_agent();

        // Validate NIXL has OBJ backend loaded
        if !nixl_agent.has_backend("OBJ") {
            bail!("NIXL OBJ backend not available for object storage transfers");
        }

        let dst_layout_inner = dst_layout.layout();
        let layout_config = dst_layout_inner.config();

        // Calculate the size of one contiguous block
        let block_size = layout_config.num_layers
            * layout_config.outer_dim
            * layout_config.page_size
            * layout_config.inner_dim
            * layout_config.dtype_width_bytes;

        // Get destination NIXL metadata
        let dst_metadata = dst_layout.nixl_metadata();
        let dst_mem_type = dst_metadata.mem_type();
        let dst_device_id = dst_metadata.device_id();

        // Resolve the bucket name for this worker
        let bucket = config.resolve_bucket(self.worker_id() as u32);

        // Register object storage slots with NIXL
        let mut obj_storages = Vec::with_capacity(object_keys.len());
        let mut registration_handles = Vec::with_capacity(object_keys.len());

        for &key in object_keys.iter() {
            let obj_storage = ObjectStorage::new(&bucket, key, block_size)
                .map_err(|e| anyhow!("Failed to create object storage: {:?}", e))?;

            let handle = nixl_agent
                .register_memory(&obj_storage, None)
                .map_err(|e| anyhow!("Failed to register object storage: {:?}", e))?;

            obj_storages.push(obj_storage);
            registration_handles.push(handle);
        }

        // Build transfer descriptor lists
        // IMPORTANT: For OBJ backend, the first descriptor list (src_dl) must be local DRAM.
        // For READ operations, we put local DRAM in src_dl and object storage in dst_dl.
        let mut src_dl = XferDescList::new(dst_mem_type)?;  // Local DRAM (destination)
        let mut dst_dl = XferDescList::new(MemType::Object)?;  // Object storage (source)

        // Add one descriptor per block
        for (&dst_block_id, &key) in local_blocks.iter().zip(object_keys.iter()) {
            // Get the base address of the destination block (local DRAM)
            let dst_region = dst_layout.memory_region(dst_block_id, 0, 0)?;

            // src_dl = local DRAM buffer where data will be read INTO
            src_dl.add_desc(dst_region.addr(), block_size, dst_device_id);

            // dst_dl = object storage (the object key is used as device_id)
            dst_dl.add_desc(0, block_size, key);
        }

        // Create transfer request using loopback
        let agent_name = nixl_agent.name();
        let xfer_req = nixl_agent.create_xfer_req(
            XferOp::Read,
            &src_dl,
            &dst_dl,
            &agent_name,
            None,
        )?;

        // Post transfer request
        let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;

        let transferred = local_blocks.len();

        // Drop registration handles - safe because transfer has been posted
        drop(registration_handles);
        drop(obj_storages);

        if still_pending {
            let notification = self.context.register_nixl_status(xfer_req);
            notification.wait()?;
        }

        Ok(G4TransferResult { transferred })
    }

    /// Execute G4 offload using RemoteDescriptors.
    ///
    /// This is a convenience wrapper that extracts object keys and bucket info
    /// from RemoteDescriptors.
    fn execute_g4_offload_with_descriptors(
        &self,
        local_handle: LayoutHandle,
        local_blocks: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        if local_blocks.len() != remote_descriptors.len() {
            bail!(
                "local_blocks.len() ({}) != remote_descriptors.len() ({})",
                local_blocks.len(),
                remote_descriptors.len()
            );
        }

        if remote_descriptors.is_empty() {
            return Ok(G4TransferResult { transferred: 0 });
        }

        // Get source layout from registry
        let src_layout = {
            let registry = self.registry.read().unwrap();
            registry
                .get_layout(local_handle)
                .ok_or_else(|| anyhow!("invalid source handle: {}", local_handle))?
                .clone()
        };

        let nixl_agent = self.context.nixl_agent();

        // Validate NIXL has OBJ backend loaded
        if !nixl_agent.has_backend("OBJ") {
            bail!("NIXL OBJ backend not available for object storage transfers");
        }

        let src_layout_inner = src_layout.layout();
        let layout_config = src_layout_inner.config();

        // Calculate the size of one contiguous block
        let block_size = layout_config.num_layers
            * layout_config.outer_dim
            * layout_config.page_size
            * layout_config.inner_dim
            * layout_config.dtype_width_bytes;

        // Get source NIXL metadata
        let src_metadata = src_layout.nixl_metadata();
        let src_mem_type = src_metadata.mem_type();
        let src_device_id = src_metadata.device_id();

        // Register object storage slots with NIXL
        let mut obj_storages = Vec::with_capacity(remote_descriptors.len());
        let mut registration_handles = Vec::with_capacity(remote_descriptors.len());

        for desc in remote_descriptors.iter() {
            let obj_storage = ObjectStorage::new(&desc.bucket, desc.object_key, block_size)
                .map_err(|e| anyhow!("Failed to create object storage: {:?}", e))?;

            let handle = nixl_agent
                .register_memory(&obj_storage, None)
                .map_err(|e| anyhow!("Failed to register object storage: {:?}", e))?;

            obj_storages.push(obj_storage);
            registration_handles.push(handle);
        }

        // Build transfer descriptor lists
        let mut src_dl = XferDescList::new(src_mem_type)?;
        let mut dst_dl = XferDescList::new(MemType::Object)?;

        // Add one descriptor per block
        for (&src_block_id, desc) in local_blocks.iter().zip(remote_descriptors.iter()) {
            let src_region = src_layout.memory_region(src_block_id, 0, 0)?;
            src_dl.add_desc(src_region.addr(), block_size, src_device_id);
            dst_dl.add_desc(0, block_size, desc.object_key);
        }

        // Create transfer request
        let agent_name = nixl_agent.name();
        let xfer_req = nixl_agent.create_xfer_req(
            XferOp::Write,
            &src_dl,
            &dst_dl,
            &agent_name,
            None,
        )?;

        let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;
        let transferred = local_blocks.len();

        drop(registration_handles);
        drop(obj_storages);

        if still_pending {
            let notification = self.context.register_nixl_status(xfer_req);
            notification.wait()?;
        }

        Ok(G4TransferResult { transferred })
    }

    /// Async version of G4 offload using RemoteDescriptors.
    async fn execute_g4_offload_with_descriptors_async(
        &self,
        local_handle: LayoutHandle,
        local_blocks: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        if local_blocks.len() != remote_descriptors.len() {
            bail!(
                "local_blocks.len() ({}) != remote_descriptors.len() ({})",
                local_blocks.len(),
                remote_descriptors.len()
            );
        }

        if remote_descriptors.is_empty() {
            return Ok(G4TransferResult { transferred: 0 });
        }

        // Get source layout from registry
        let src_layout = {
            let registry = self.registry.read().unwrap();
            registry
                .get_layout(local_handle)
                .ok_or_else(|| anyhow!("invalid source handle: {}", local_handle))?
                .clone()
        };

        let nixl_agent = self.context.nixl_agent();

        // Validate NIXL has OBJ backend loaded
        if !nixl_agent.has_backend("OBJ") {
            bail!("NIXL OBJ backend not available for object storage transfers");
        }

        let src_layout_inner = src_layout.layout();
        let layout_config = src_layout_inner.config();

        // Calculate the size of one contiguous block
        let block_size = layout_config.num_layers
            * layout_config.outer_dim
            * layout_config.page_size
            * layout_config.inner_dim
            * layout_config.dtype_width_bytes;

        // Get source NIXL metadata
        let src_metadata = src_layout.nixl_metadata();
        let src_mem_type = src_metadata.mem_type();
        let src_device_id = src_metadata.device_id();

        // Use a scope block to ensure all non-Send types (XferDescList, ObjectStorage, etc.)
        // are dropped before the await point, making the future Send-safe
        let (xfer_req, transferred, still_pending) = {
            // Register object storage slots with NIXL
            let mut obj_storages = Vec::with_capacity(remote_descriptors.len());
            let mut registration_handles = Vec::with_capacity(remote_descriptors.len());

            for desc in remote_descriptors.iter() {
                let obj_storage = ObjectStorage::new(&desc.bucket, desc.object_key, block_size)
                    .map_err(|e| anyhow!("Failed to create object storage: {:?}", e))?;

                let handle = nixl_agent
                    .register_memory(&obj_storage, None)
                    .map_err(|e| anyhow!("Failed to register object storage: {:?}", e))?;

                obj_storages.push(obj_storage);
                registration_handles.push(handle);
            }

            // Build transfer descriptor lists
            let mut src_dl = XferDescList::new(src_mem_type)?;
            let mut dst_dl = XferDescList::new(MemType::Object)?;

            // Add one descriptor per block
            for (&src_block_id, desc) in local_blocks.iter().zip(remote_descriptors.iter()) {
                let src_region = src_layout.memory_region(src_block_id, 0, 0)?;
                src_dl.add_desc(src_region.addr(), block_size, src_device_id);
                dst_dl.add_desc(0, block_size, desc.object_key);
            }

            // Create transfer request
            let agent_name = nixl_agent.name();
            let xfer_req = nixl_agent.create_xfer_req(
                XferOp::Write,
                &src_dl,
                &dst_dl,
                &agent_name,
                None,
            )?;

            let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;
            let transferred = local_blocks.len();

            // All non-Send types (src_dl, dst_dl, registration_handles, obj_storages)
            // are dropped here when the block ends
            (xfer_req, transferred, still_pending)
        };

        if still_pending {
            let notification = self.context.register_nixl_status(xfer_req);
            notification.await?;
        }

        Ok(G4TransferResult { transferred })
    }

    /// Execute G4 onboard using RemoteDescriptors.
    ///
    /// This is a convenience wrapper that extracts object keys and bucket info
    /// from RemoteDescriptors.
    fn execute_g4_onboard_with_descriptors(
        &self,
        local_handle: LayoutHandle,
        local_blocks: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        if local_blocks.len() != remote_descriptors.len() {
            bail!(
                "local_blocks.len() ({}) != remote_descriptors.len() ({})",
                local_blocks.len(),
                remote_descriptors.len()
            );
        }

        if remote_descriptors.is_empty() {
            return Ok(G4TransferResult { transferred: 0 });
        }

        // Get destination layout from registry
        let dst_layout = {
            let registry = self.registry.read().unwrap();
            registry
                .get_layout(local_handle)
                .ok_or_else(|| anyhow!("invalid destination handle: {}", local_handle))?
                .clone()
        };

        let nixl_agent = self.context.nixl_agent();

        // Validate NIXL has OBJ backend loaded
        if !nixl_agent.has_backend("OBJ") {
            bail!("NIXL OBJ backend not available for object storage transfers");
        }

        let dst_layout_inner = dst_layout.layout();
        let layout_config = dst_layout_inner.config();

        // Calculate the size of one contiguous block
        let block_size = layout_config.num_layers
            * layout_config.outer_dim
            * layout_config.page_size
            * layout_config.inner_dim
            * layout_config.dtype_width_bytes;

        // Get destination NIXL metadata
        let dst_metadata = dst_layout.nixl_metadata();
        let dst_mem_type = dst_metadata.mem_type();
        let dst_device_id = dst_metadata.device_id();

        // Register object storage slots with NIXL
        let mut obj_storages = Vec::with_capacity(remote_descriptors.len());
        let mut registration_handles = Vec::with_capacity(remote_descriptors.len());

        for desc in remote_descriptors.iter() {
            let obj_storage = ObjectStorage::new(&desc.bucket, desc.object_key, block_size)
                .map_err(|e| anyhow!("Failed to create object storage: {:?}", e))?;

            let handle = nixl_agent
                .register_memory(&obj_storage, None)
                .map_err(|e| anyhow!("Failed to register object storage: {:?}", e))?;

            obj_storages.push(obj_storage);
            registration_handles.push(handle);
        }

        // Build transfer descriptor lists
        // IMPORTANT: For OBJ backend, the first descriptor list (src_dl) must be local DRAM.
        // For READ operations, we put local DRAM in src_dl and object storage in dst_dl.
        let mut src_dl = XferDescList::new(dst_mem_type)?;  // Local DRAM (destination)
        let mut dst_dl = XferDescList::new(MemType::Object)?;  // Object storage (source)

        // Add one descriptor per block
        for (&dst_block_id, desc) in local_blocks.iter().zip(remote_descriptors.iter()) {
            let dst_region = dst_layout.memory_region(dst_block_id, 0, 0)?;
            // src_dl = local DRAM buffer where data will be read INTO
            src_dl.add_desc(dst_region.addr(), block_size, dst_device_id);
            // dst_dl = object storage (the object key is used as device_id)
            dst_dl.add_desc(0, block_size, desc.object_key);
        }

        // Create transfer request
        let agent_name = nixl_agent.name();
        let xfer_req = nixl_agent.create_xfer_req(
            XferOp::Read,
            &src_dl,
            &dst_dl,
            &agent_name,
            None,
        )?;

        let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;
        let transferred = local_blocks.len();

        drop(registration_handles);
        drop(obj_storages);

        if still_pending {
            let notification = self.context.register_nixl_status(xfer_req);
            notification.wait()?;
        }

        Ok(G4TransferResult { transferred })
    }

    /// Async version of G4 onboard using RemoteDescriptors.
    async fn execute_g4_onboard_with_descriptors_async(
        &self,
        local_handle: LayoutHandle,
        local_blocks: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        if local_blocks.len() != remote_descriptors.len() {
            bail!(
                "local_blocks.len() ({}) != remote_descriptors.len() ({})",
                local_blocks.len(),
                remote_descriptors.len()
            );
        }

        if remote_descriptors.is_empty() {
            return Ok(G4TransferResult { transferred: 0 });
        }

        // Get destination layout from registry
        let dst_layout = {
            let registry = self.registry.read().unwrap();
            registry
                .get_layout(local_handle)
                .ok_or_else(|| anyhow!("invalid destination handle: {}", local_handle))?
                .clone()
        };

        let nixl_agent = self.context.nixl_agent();

        // Validate NIXL has OBJ backend loaded
        if !nixl_agent.has_backend("OBJ") {
            bail!("NIXL OBJ backend not available for object storage transfers");
        }

        let dst_layout_inner = dst_layout.layout();
        let layout_config = dst_layout_inner.config();

        // Calculate the size of one contiguous block
        let block_size = layout_config.num_layers
            * layout_config.outer_dim
            * layout_config.page_size
            * layout_config.inner_dim
            * layout_config.dtype_width_bytes;

        // Get destination NIXL metadata
        let dst_metadata = dst_layout.nixl_metadata();
        let dst_mem_type = dst_metadata.mem_type();
        let dst_device_id = dst_metadata.device_id();

        // Use a scope block to ensure all non-Send types (XferDescList, ObjectStorage, etc.)
        // are dropped before the await point, making the future Send-safe
        let (xfer_req, transferred, still_pending) = {
            // Register object storage slots with NIXL
            let mut obj_storages = Vec::with_capacity(remote_descriptors.len());
            let mut registration_handles = Vec::with_capacity(remote_descriptors.len());

            for desc in remote_descriptors.iter() {
                let obj_storage = ObjectStorage::new(&desc.bucket, desc.object_key, block_size)
                    .map_err(|e| anyhow!("Failed to create object storage: {:?}", e))?;

                let handle = nixl_agent
                    .register_memory(&obj_storage, None)
                    .map_err(|e| anyhow!("Failed to register object storage: {:?}", e))?;

                obj_storages.push(obj_storage);
                registration_handles.push(handle);
            }

            // Build transfer descriptor lists
            // IMPORTANT: For OBJ backend, the first descriptor list (src_dl) must be local DRAM.
            // For READ operations, we put local DRAM in src_dl and object storage in dst_dl.
            let mut src_dl = XferDescList::new(dst_mem_type)?;  // Local DRAM (destination)
            let mut dst_dl = XferDescList::new(MemType::Object)?;  // Object storage (source)

            // Add one descriptor per block
            for (&dst_block_id, desc) in local_blocks.iter().zip(remote_descriptors.iter()) {
                let dst_region = dst_layout.memory_region(dst_block_id, 0, 0)?;
                // src_dl = local DRAM buffer where data will be read INTO
                src_dl.add_desc(dst_region.addr(), block_size, dst_device_id);
                // dst_dl = object storage (the object key is used as device_id)
                dst_dl.add_desc(0, block_size, desc.object_key);
            }

            // Create transfer request
            let agent_name = nixl_agent.name();
            let xfer_req = nixl_agent.create_xfer_req(
                XferOp::Read,
                &src_dl,
                &dst_dl,
                &agent_name,
                None,
            )?;

            let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;
            let transferred = local_blocks.len();

            // All non-Send types (src_dl, dst_dl, registration_handles, obj_storages)
            // are dropped here when the block ends
            (xfer_req, transferred, still_pending)
        };

        if still_pending {
            let notification = self.context.register_nixl_status(xfer_req);
            notification.await?;
        }

        Ok(G4TransferResult { transferred })
    }

    // ===== Query Methods =====

    /// Get the worker ID for this manager.
    pub fn worker_id(&self) -> u64 {
        self.context.worker_id()
    }

    /// Get handles for all locally registered layouts.
    pub fn get_local_handles(&self) -> Vec<LayoutHandle> {
        self.registry.read().unwrap().local_handles()
    }

    /// Get handles for all imported remote layouts.
    pub fn get_remote_handles(&self) -> Vec<LayoutHandle> {
        self.registry.read().unwrap().remote_handles()
    }

    // ===== Internal Methods for Testing =====

    /// Get the internal transfer context (for testing only).
    pub fn context(&self) -> &Arc<TransferContext> {
        &self.context
    }

    /// Get the H2D stream (for testing only).
    #[cfg(all(test, feature = "testing-cuda"))]
    pub(crate) fn h2d_stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        self.context.h2d_stream()
    }

    /// Get the D2H stream (for testing only).
    #[cfg(all(test, feature = "testing-cuda"))]
    #[allow(dead_code)]
    pub(crate) fn d2h_stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        self.context.d2h_stream()
    }

    /// Get the CUDA context (for testing only).
    #[cfg(all(test, feature = "testing-cuda"))]
    pub(crate) fn cuda_context(&self) -> &std::sync::Arc<cudarc::driver::CudaContext> {
        self.context.cuda_context()
    }

    /// Register a CUDA event for completion (for testing only).
    #[cfg(all(test, feature = "testing-cuda"))]
    pub(crate) fn register_cuda_event(
        &self,
        event: cudarc::driver::CudaEvent,
    ) -> TransferCompleteNotification {
        self.context.register_cuda_event(event)
    }
}

/// Internal registry for local and remote physical layouts with NIXL integration.
///
/// The LayoutRegistry handles:
/// - Registering local layouts with unique handles
/// - Exporting local layout metadata for remote access
/// - Importing remote layout metadata and reconstructing layouts
/// - Managing NIXL metadata for RDMA operations
#[derive(Debug)]
pub(crate) struct LayoutRegistry {
    /// NIXL agent for memory registration
    nixl_agent: NixlAgent,
    /// Worker ID for this manager
    worker_id: u64,
    /// Next layout ID to assign (monotonically increasing)
    next_layout_id: AtomicU16,
    /// Local layouts registered on this worker
    local_layouts: HashMap<LayoutHandle, LocalLayout>,
    /// Remote layouts imported from other workers
    remote_layouts: HashMap<LayoutHandle, RemoteLayout>,
    /// Set of loaded remote workers (agent_name, worker_id) to prevent duplicates
    loaded_remotes: HashSet<(String, u64)>,
}

#[expect(dead_code)]
impl LayoutRegistry {
    /// Create a new layout manager.
    ///
    /// # Arguments
    /// * `nixl_agent` - NIXL agent for memory registration
    /// * `worker_id` - Unique identifier for this worker
    pub(crate) fn new(nixl_agent: NixlAgent, worker_id: u64) -> Self {
        Self {
            nixl_agent,
            worker_id,
            next_layout_id: AtomicU16::new(0),
            local_layouts: HashMap::new(),
            remote_layouts: HashMap::new(),
            loaded_remotes: HashSet::new(),
        }
    }

    /// Register a local physical layout.
    ///
    /// # Arguments
    /// * `layout` - Physical layout to register
    ///
    /// # Returns
    /// Unique handle for the registered layout
    ///
    /// # Errors
    /// Returns an error if layout IDs are exhausted (u16::MAX reached)
    pub(crate) fn register_local(&mut self, layout: PhysicalLayout) -> Result<LayoutHandle> {
        // Get next layout ID
        let layout_id = self.next_layout_id.fetch_add(1, Ordering::SeqCst);
        if layout_id == u16::MAX {
            bail!("Layout ID overflow: maximum number of layouts (65535) reached");
        }

        // Create handle
        let handle = LayoutHandle::new(self.worker_id, layout_id);

        // Wrap in LocalLayout
        let local_layout = LocalLayout::new(handle, layout);

        // Store
        self.local_layouts.insert(handle, local_layout);

        Ok(handle)
    }

    /// Export local layout metadata for transmission to remote workers.
    ///
    /// This exports:
    /// - NIXL agent metadata for remote memory registration
    /// - All host and device layouts (disk layouts are excluded)
    /// - Worker address information
    ///
    /// # Returns
    /// Packed metadata ready for transmission
    pub(crate) fn export_metadata(&self) -> Result<SerializedLayout> {
        // Get NIXL metadata from agent
        let nixl_metadata = self
            .nixl_agent
            .get_local_md()
            .map_err(|e| anyhow!("failed to get NIXL local metadata: {:?}", e))?;

        // Create worker address
        let worker_address = WorkerAddress::new(self.worker_id, self.nixl_agent.name().to_string());

        // Filter and serialize layouts (only host and device, skip disk)
        let mut serialized_layouts = Vec::new();
        for (handle, local_layout) in &self.local_layouts {
            let location = local_layout.layout().location();

            // Only export host and device layouts
            if matches!(
                location,
                StorageKind::System | StorageKind::Device(_) | StorageKind::Pinned
            ) {
                let serialized = local_layout
                    .layout()
                    .to_descriptor()
                    .map_err(|e| anyhow!("failed to serialize layout {}: {}", handle, e))?;

                serialized_layouts.push(LocalLayoutDescriptor::new(*handle, serialized));
            }
        }

        // Pack into managed metadata
        SerializedLayout::pack(worker_address, nixl_metadata, serialized_layouts)
    }

    /// Import remote layout metadata.
    ///
    /// This:
    /// - Validates the remote worker hasn't been loaded already
    /// - Loads NIXL metadata into the agent
    /// - Reconstructs physical layouts from serialized data
    /// - Stores them as remote layouts
    ///
    /// # Arguments
    /// * `metadata` - Packed metadata from remote worker
    ///
    /// # Returns
    /// Vector of handles for the imported layouts
    ///
    /// # Errors
    /// Returns an error if:
    /// - The remote worker was already loaded
    /// - NIXL metadata loading fails
    /// - Agent name mismatch after loading
    /// - Layout reconstruction fails
    pub(crate) fn import_metadata(
        &mut self,
        metadata: SerializedLayout,
    ) -> Result<Vec<LayoutHandle>> {
        // Unpack metadata
        let inner = metadata.unpack()?;

        // Validate not already loaded
        let remote_key = (
            inner.worker_address.nixl_agent_name.clone(),
            inner.worker_address.worker_id,
        );
        if self.loaded_remotes.contains(&remote_key) {
            bail!(
                "Remote worker already loaded: {} (worker_id={})",
                remote_key.0,
                remote_key.1
            );
        }

        // Load NIXL metadata
        let returned_agent_name = self
            .nixl_agent
            .load_remote_md(&inner.nixl_metadata)
            .map_err(|e| anyhow!("failed to load remote NIXL metadata: {:?}", e))?;

        // Verify agent name matches
        if returned_agent_name != inner.worker_address.nixl_agent_name {
            bail!(
                "Agent name mismatch: expected '{}', got '{}'",
                inner.worker_address.nixl_agent_name,
                returned_agent_name
            );
        }

        // Reconstruct layouts
        let mut imported_handles = Vec::new();
        for serialized_with_handle in inner.layouts {
            let handle = serialized_with_handle.handle;
            let layout = PhysicalLayout::from_descriptor(serialized_with_handle.layout)
                .map_err(|e| anyhow!("failed to reconstruct layout {}: {}", handle, e))?;

            let remote_layout = RemoteLayout::new(handle, layout);
            self.remote_layouts.insert(handle, remote_layout);
            imported_handles.push(handle);
        }

        // Mark remote as loaded
        self.loaded_remotes.insert(remote_key);

        Ok(imported_handles)
    }

    /// Get a local layout by handle.
    pub(crate) fn get_local(&self, handle: LayoutHandle) -> Option<&LocalLayout> {
        self.local_layouts.get(&handle)
    }

    /// Get a remote layout by handle.
    pub(crate) fn get_remote(&self, handle: LayoutHandle) -> Option<&RemoteLayout> {
        self.remote_layouts.get(&handle)
    }

    /// Get a layout by handle (either local or remote).
    ///
    /// # Returns
    /// Returns a reference to the PhysicalLayout if found
    pub fn get_layout(&self, handle: LayoutHandle) -> Option<&PhysicalLayout> {
        self.local_layouts
            .get(&handle)
            .map(|l| l.layout())
            .or_else(|| self.remote_layouts.get(&handle).map(|r| r.layout()))
    }

    /// Check if a handle refers to a local layout.
    pub(crate) fn is_local(&self, handle: LayoutHandle) -> bool {
        self.local_layouts.contains_key(&handle)
    }

    /// Check if a handle refers to a remote layout.
    pub(crate) fn is_remote(&self, handle: LayoutHandle) -> bool {
        self.remote_layouts.contains_key(&handle)
    }

    /// Get the number of local layouts.
    pub(crate) fn local_count(&self) -> usize {
        self.local_layouts.len()
    }

    /// Get the number of remote layouts.
    pub(crate) fn remote_count(&self) -> usize {
        self.remote_layouts.len()
    }

    /// Get the worker ID for this manager.
    pub(crate) fn worker_id(&self) -> u64 {
        self.worker_id
    }

    /// Get all local layout handles.
    pub(crate) fn local_handles(&self) -> Vec<LayoutHandle> {
        self.local_layouts.keys().copied().collect()
    }

    /// Get all remote layout handles.
    pub(crate) fn remote_handles(&self) -> Vec<LayoutHandle> {
        self.remote_layouts.keys().copied().collect()
    }
}

#[cfg(all(test, feature = "testing-nixl"))]
mod tests {
    use super::*;
    use crate::block_manager::v2::physical::layout::LayoutConfig;
    use crate::block_manager::v2::physical::transfer::nixl_agent::NixlAgent;

    fn make_test_agent(name: &str) -> NixlAgent {
        NixlAgent::require_backends(name, &[]).expect("failed to create wrapped agent")
    }

    fn make_test_layout(agent: &NixlAgent) -> PhysicalLayout {
        let config = LayoutConfig::builder()
            .num_blocks(2)
            .num_layers(2)
            .outer_dim(2)
            .page_size(4)
            .inner_dim(8)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        PhysicalLayout::builder(agent.clone())
            .with_config(config)
            .fully_contiguous()
            .allocate_system()
            .build()
            .unwrap()
    }

    #[test]
    fn test_manager_creation() {
        let agent = make_test_agent("test-manager");
        let manager = LayoutRegistry::new(agent, 42);

        assert_eq!(manager.worker_id(), 42);
        assert_eq!(manager.local_count(), 0);
        assert_eq!(manager.remote_count(), 0);
    }

    #[test]
    fn test_register_local() {
        let agent = make_test_agent("test-register");
        let mut manager = LayoutRegistry::new(agent.clone(), 100);

        let layout = make_test_layout(&agent);
        let handle = manager.register_local(layout).unwrap();

        assert_eq!(handle.worker_id(), 100);
        assert_eq!(handle.layout_id(), 0);
        assert_eq!(manager.local_count(), 1);
        assert!(manager.is_local(handle));
        assert!(!manager.is_remote(handle));
    }

    #[test]
    fn test_register_multiple_locals() {
        let agent = make_test_agent("test-multiple");
        let mut manager = LayoutRegistry::new(agent.clone(), 1);

        let handle1 = manager.register_local(make_test_layout(&agent)).unwrap();
        let handle2 = manager.register_local(make_test_layout(&agent)).unwrap();
        let handle3 = manager.register_local(make_test_layout(&agent)).unwrap();

        assert_eq!(handle1.layout_id(), 0);
        assert_eq!(handle2.layout_id(), 1);
        assert_eq!(handle3.layout_id(), 2);
        assert_eq!(manager.local_count(), 3);
    }

    #[test]
    #[ignore] // Requires actual NIXL memory registration
    fn test_export_import_roundtrip() {
        // Create source manager and register layouts
        let source_agent = make_test_agent("source");
        let mut source_manager = LayoutRegistry::new(source_agent.clone(), 1);

        let handle1 = source_manager
            .register_local(make_test_layout(&source_agent))
            .unwrap();
        let handle2 = source_manager
            .register_local(make_test_layout(&source_agent))
            .unwrap();

        // Export metadata
        let metadata = source_manager.export_metadata().unwrap();
        assert!(!metadata.is_empty());

        // Create destination manager and import
        let dest_agent = make_test_agent("dest");
        let mut dest_manager = LayoutRegistry::new(dest_agent, 2);

        let imported_handles = dest_manager.import_metadata(metadata).unwrap();

        // Verify
        assert_eq!(imported_handles.len(), 2);
        assert_eq!(dest_manager.remote_count(), 2);
        assert!(dest_manager.is_remote(handle1));
        assert!(dest_manager.is_remote(handle2));

        // Can get layouts
        assert!(dest_manager.get_remote(handle1).is_some());
        assert!(dest_manager.get_remote(handle2).is_some());
        assert!(dest_manager.get_layout(handle1).is_some());
    }

    #[test]
    #[ignore] // Requires actual NIXL memory registration
    fn test_import_duplicate_remote_fails() {
        let source_agent = make_test_agent("source2");
        let mut source_manager = LayoutRegistry::new(source_agent.clone(), 10);

        source_manager
            .register_local(make_test_layout(&source_agent))
            .unwrap();

        let metadata = source_manager.export_metadata().unwrap();

        let dest_agent = make_test_agent("dest2");
        let mut dest_manager = LayoutRegistry::new(dest_agent, 20);

        // First import succeeds
        let metadata_clone = SerializedLayout::from_bytes(metadata.as_bytes().clone());
        dest_manager.import_metadata(metadata).unwrap();

        // Second import should fail
        let result = dest_manager.import_metadata(metadata_clone);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already loaded"));
    }

    #[test]
    fn test_get_layout_handles() {
        let agent = make_test_agent("test-handles");
        let mut manager = LayoutRegistry::new(agent.clone(), 5);

        let h1 = manager.register_local(make_test_layout(&agent)).unwrap();
        let h2 = manager.register_local(make_test_layout(&agent)).unwrap();

        let handles = manager.local_handles();
        assert_eq!(handles.len(), 2);
        assert!(handles.contains(&h1));
        assert!(handles.contains(&h2));
    }
}

/// Tests for G4 (object storage) transfers.
///
/// These tests verify the `execute_g4_offload`, `execute_g4_onboard`, and
/// `execute_g4_transfer` methods using NIXL's OBJ backend.
///
/// # Requirements
/// - NIXL with OBJ backend support
/// - Object storage (e.g., MinIO, S3)
/// - Environment variables for connection configuration
///
/// # Environment Variables
/// ```bash
/// export DYN_KVBM_NIXL_BACKEND_OBJ_ENDPOINT="http://localhost:9000"
/// export DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET="test-bucket"
/// export DYN_KVBM_NIXL_BACKEND_OBJ_ACCESS_KEY=""
/// export DYN_KVBM_NIXL_BACKEND_OBJ_SECRET_KEY=""
/// ```
#[cfg(all(test, feature = "testing-nixl"))]
mod g4_transfer_tests {
    use super::*;
    use crate::block_manager::config::ObjectStorageConfig;
    use crate::block_manager::v2::physical::layout::LayoutConfig;
    use crate::block_manager::v2::physical::transfer::nixl_agent::NixlAgent;
    use crate::block_manager::v2::physical::transfer::{FillPattern, TransferCapabilities, fill_blocks, compute_block_checksums};
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Generate a unique object key for testing (based on timestamp).
    fn generate_test_key() -> u64 {
        let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        duration.as_nanos() as u64
    }

    /// Get bucket name from environment or use default.
    fn get_test_bucket() -> String {
        std::env::var("DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET")
            .unwrap_or_else(|_| "test-bucket".to_string())
    }

    /// Create a test agent with OBJ backend.
    fn make_g4_test_agent(name: &str) -> Result<NixlAgent> {
        NixlAgent::new_with_backends(name, &["OBJ", "POSIX"])
    }

    /// Create a standard layout config for testing.
    fn make_test_config(num_blocks: usize) -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(num_blocks)
            .num_layers(2)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(128)
            .dtype_width_bytes(2)
            .build()
            .unwrap()
    }

    /// Create an ObjectStorageConfig for testing.
    fn make_object_storage_config() -> ObjectStorageConfig {
        ObjectStorageConfig {
            bucket_template: get_test_bucket(),
            endpoint_override: std::env::var("DYN_KVBM_NIXL_BACKEND_OBJ_ENDPOINT").ok(),
            region: std::env::var("DYN_KVBM_NIXL_BACKEND_OBJ_REGION").ok(),
        }
    }

    /// Create a pinned memory layout.
    fn make_pinned_layout(agent: &NixlAgent, num_blocks: usize) -> Result<PhysicalLayout> {
        let config = make_test_config(num_blocks);
        PhysicalLayout::builder(agent.clone())
            .with_config(config)
            .fully_contiguous()
            .allocate_pinned(false)
            .build()
    }

    /// Create a TransportManager for testing.
    fn make_transport_manager(agent: NixlAgent, worker_id: u64) -> Result<TransportManager> {
        TransportManager::builder()
            .capabilities(TransferCapabilities::default())
            .worker_id(worker_id)
            .nixl_agent(agent)
            .cuda_device_id(0)
            .build()
    }

    /// Fill blocks with test data and compute checksums.
    fn fill_and_checksum(
        layout: &PhysicalLayout,
        block_ids: &[usize],
    ) -> Result<HashMap<usize, crate::block_manager::v2::physical::transfer::BlockChecksum>> {
        fill_blocks(layout, block_ids, FillPattern::Sequential)?;
        compute_block_checksums(layout, block_ids)
    }

    // =========================================================================
    // Basic G4 Transfer Tests
    // =========================================================================

    /// Test basic G4 offload operation.
    ///
    /// This test:
    /// 1. Creates a pinned memory layout with test data
    /// 2. Offloads blocks to object storage using execute_g4_offload
    /// 3. Verifies the transfer completes successfully
    #[test]
    #[ignore] // Requires OBJ backend and object storage
    fn test_g4_offload_basic() -> Result<()> {
        let test_name = format!("test-g4-offload-{}", generate_test_key());
        let agent = make_g4_test_agent(&test_name)?;
        let config = make_object_storage_config();
        let num_blocks = 4;

        // Create transport manager
        let transport = make_transport_manager(agent.clone(), 0)?;

        // Create source layout with test data
        let src_layout = make_pinned_layout(&agent, num_blocks)?;
        let src_handle = transport.register_layout(src_layout.clone())?;

        // Create dummy host handle (for bounce buffers - not used in pinned->object)
        let host_layout = make_pinned_layout(&agent, num_blocks)?;
        let host_handle = transport.register_layout(host_layout)?;

        let block_ids: Vec<usize> = (0..num_blocks).collect();

        // Fill with test data
        let _checksums = fill_and_checksum(&src_layout, &block_ids)?;

        // Generate unique object keys
        let base_key = generate_test_key();
        let object_keys: Vec<u64> = (0..num_blocks as u64).map(|i| base_key + i).collect();

        // Execute offload
        let result = transport.execute_g4_offload(
            src_handle,
            &block_ids,
            &object_keys,
            host_handle,
            &config,
        )?;

        assert_eq!(result.transferred, num_blocks);
        Ok(())
    }

    /// Test basic G4 onboard operation.
    ///
    /// This test:
    /// 1. Offloads test data to object storage
    /// 2. Onboards the data back to a new layout
    /// 3. Verifies the transfer completes successfully
    #[test]
    #[ignore] // Requires OBJ backend and object storage
    fn test_g4_onboard_basic() -> Result<()> {
        let test_name = format!("test-g4-onboard-{}", generate_test_key());
        let agent = make_g4_test_agent(&test_name)?;
        let config = make_object_storage_config();
        let num_blocks = 4;

        // Create transport manager
        let transport = make_transport_manager(agent.clone(), 0)?;

        // Create source layout and offload first
        let src_layout = make_pinned_layout(&agent, num_blocks)?;
        let src_handle = transport.register_layout(src_layout.clone())?;

        let host_layout = make_pinned_layout(&agent, num_blocks)?;
        let host_handle = transport.register_layout(host_layout)?;

        let block_ids: Vec<usize> = (0..num_blocks).collect();
        let _checksums = fill_and_checksum(&src_layout, &block_ids)?;

        let base_key = generate_test_key();
        let object_keys: Vec<u64> = (0..num_blocks as u64).map(|i| base_key + i).collect();

        // First offload
        transport.execute_g4_offload(
            src_handle,
            &block_ids,
            &object_keys,
            host_handle,
            &config,
        )?;

        // Create destination layout
        let dst_layout = make_pinned_layout(&agent, num_blocks)?;
        let dst_handle = transport.register_layout(dst_layout)?;

        // Execute onboard
        let result = transport.execute_g4_onboard(
            dst_handle,
            &block_ids,
            &object_keys,
            host_handle,
            &config,
        )?;

        assert_eq!(result.transferred, num_blocks);
        Ok(())
    }

    // =========================================================================
    // G4 Transfer with RemoteDescriptor Tests
    // =========================================================================

    /// Test G4 transfer using RemoteDescriptor API.
    #[test]
    #[ignore] // Requires OBJ backend and object storage
    fn test_g4_transfer_with_remote_descriptor() -> Result<()> {
        let test_name = format!("test-g4-descriptor-{}", generate_test_key());
        let agent = make_g4_test_agent(&test_name)?;
        let config = make_object_storage_config();
        let num_blocks = 4;

        let transport = make_transport_manager(agent.clone(), 0)?;

        let src_layout = make_pinned_layout(&agent, num_blocks)?;
        let src_handle = transport.register_layout(src_layout.clone())?;

        let block_ids: Vec<usize> = (0..num_blocks).collect();
        let _checksums = fill_and_checksum(&src_layout, &block_ids)?;

        let base_key = generate_test_key();
        let object_keys: Vec<u64> = (0..num_blocks as u64).map(|i| base_key + i).collect();

        // Create RemoteDescriptors
        let remote_descriptors = RemoteDescriptor::from_keys(
            &object_keys,
            &config,
            transport.worker_id() as u32,
        );

        // Execute offload using execute_g4_transfer
        let result = transport.execute_g4_transfer(
            G4TransferDirection::Offload,
            src_handle,
            &block_ids,
            &remote_descriptors,
        )?;

        assert_eq!(result.transferred, num_blocks);

        // Create destination and onboard
        let dst_layout = make_pinned_layout(&agent, num_blocks)?;
        let dst_handle = transport.register_layout(dst_layout)?;

        let result = transport.execute_g4_transfer(
            G4TransferDirection::Onboard,
            dst_handle,
            &block_ids,
            &remote_descriptors,
        )?;

        assert_eq!(result.transferred, num_blocks);
        Ok(())
    }

    // =========================================================================
    // Roundtrip Tests
    // =========================================================================

    /// Test G4 roundtrip: offload → onboard with data verification.
    ///
    /// This is the most important test - it verifies data integrity
    /// through a complete offload/onboard cycle.
    #[test]
    #[ignore] // Requires OBJ backend and object storage
    fn test_g4_roundtrip_with_verification() -> Result<()> {
        let test_name = format!("test-g4-roundtrip-{}", generate_test_key());
        let agent = make_g4_test_agent(&test_name)?;
        let config = make_object_storage_config();
        let num_blocks = 4;

        let transport = make_transport_manager(agent.clone(), 0)?;

        // Create source layout with test data
        let src_layout = make_pinned_layout(&agent, num_blocks)?;
        let src_handle = transport.register_layout(src_layout.clone())?;

        let host_layout = make_pinned_layout(&agent, num_blocks)?;
        let host_handle = transport.register_layout(host_layout)?;

        let block_ids: Vec<usize> = (0..num_blocks).collect();

        // Fill and compute original checksums
        let original_checksums = fill_and_checksum(&src_layout, &block_ids)?;

        let base_key = generate_test_key();
        let object_keys: Vec<u64> = (0..num_blocks as u64).map(|i| base_key + i).collect();

        // Step 1: Offload to object storage
        let offload_result = transport.execute_g4_offload(
            src_handle,
            &block_ids,
            &object_keys,
            host_handle,
            &config,
        )?;
        assert_eq!(offload_result.transferred, num_blocks);

        // Step 2: Create fresh destination and onboard
        let dst_layout = make_pinned_layout(&agent, num_blocks)?;
        let dst_handle = transport.register_layout(dst_layout.clone())?;

        let onboard_result = transport.execute_g4_onboard(
            dst_handle,
            &block_ids,
            &object_keys,
            host_handle,
            &config,
        )?;
        assert_eq!(onboard_result.transferred, num_blocks);

        // Step 3: Verify checksums match
        let final_checksums = compute_block_checksums(&dst_layout, &block_ids)?;

        for block_id in &block_ids {
            let original = original_checksums.get(block_id)
                .unwrap_or_else(|| panic!("Missing original checksum for block {}", block_id));
            let final_cs = final_checksums.get(block_id)
                .unwrap_or_else(|| panic!("Missing final checksum for block {}", block_id));

            assert_eq!(
                original, final_cs,
                "Checksum mismatch for block {}: original={}, final={}",
                block_id, original, final_cs
            );
        }

        Ok(())
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    /// Test G4 transfer with empty block list.
    #[test]
    fn test_g4_offload_empty_blocks() -> Result<()> {
        let test_name = format!("test-g4-empty-{}", generate_test_key());
        let agent = NixlAgent::require_backends(&test_name, &[])?;
        let config = make_object_storage_config();

        let transport = make_transport_manager(agent.clone(), 0)?;

        let src_layout = make_pinned_layout(&agent, 4)?;
        let src_handle = transport.register_layout(src_layout)?;

        let host_layout = make_pinned_layout(&agent, 4)?;
        let host_handle = transport.register_layout(host_layout)?;

        // Empty block list should succeed with 0 transferred
        let result = transport.execute_g4_offload(
            src_handle,
            &[],
            &[],
            host_handle,
            &config,
        )?;

        assert_eq!(result.transferred, 0);
        Ok(())
    }

    /// Test G4 transfer with mismatched block/key counts.
    #[test]
    fn test_g4_offload_mismatched_counts() -> Result<()> {
        let test_name = format!("test-g4-mismatch-{}", generate_test_key());
        let agent = NixlAgent::require_backends(&test_name, &[])?;
        let config = make_object_storage_config();

        let transport = make_transport_manager(agent.clone(), 0)?;

        let src_layout = make_pinned_layout(&agent, 4)?;
        let src_handle = transport.register_layout(src_layout)?;

        let host_layout = make_pinned_layout(&agent, 4)?;
        let host_handle = transport.register_layout(host_layout)?;

        // Mismatched counts should fail
        let result = transport.execute_g4_offload(
            src_handle,
            &[0, 1, 2],  // 3 blocks
            &[100, 200], // 2 keys
            host_handle,
            &config,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("!="));
        Ok(())
    }

    /// Test RemoteDescriptor::from_keys helper.
    #[test]
    fn test_remote_descriptor_from_keys() {
        let config = ObjectStorageConfig {
            bucket_template: "bucket-{worker_id}".to_string(),
            endpoint_override: None,
            region: None,
        };

        let keys = vec![100u64, 200, 300];
        let descriptors = RemoteDescriptor::from_keys(&keys, &config, 42);

        assert_eq!(descriptors.len(), 3);
        assert_eq!(descriptors[0].object_key, 100);
        assert_eq!(descriptors[0].bucket, "bucket-42");
        assert_eq!(descriptors[1].object_key, 200);
        assert_eq!(descriptors[2].object_key, 300);
    }
}
