// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport manager for local and remote physical layouts with transfer execution.

mod handle;
mod local;
mod metadata;
mod remote;

pub use handle::LayoutHandle;
pub use metadata::{LogicalLayoutDescriptor, SerializedLayout, WorkerAddress};

pub(crate) use local::LocalLayout;
pub(crate) use metadata::LocalLayoutDescriptor;
pub(crate) use remote::RemoteLayout;

use crate::physical::transfer::BounceBufferInternal;
use crate::v2::logical::LogicalLayoutHandle;
use crate::v2::physical::layout::PhysicalLayout;
use crate::v2::physical::transfer::TransferContext;
use crate::v2::physical::transfer::context::TransferCompleteNotification;
use crate::v2::physical::transfer::executor::TransferOptionsInternal;
use crate::v2::physical::transfer::options::TransferOptions;
use crate::{BlockId, SequenceHash};
use anyhow::{Result, anyhow, bail};
use dynamo_memory::StorageKind;
use dynamo_memory::nixl::NixlAgent;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, RwLock};

/// Public entry point for layout and transfer management.
///
/// TransferManager combines layout registration/metadata management with
/// transfer execution capabilities, providing a unified API for:
/// - Registering local layouts and obtaining handles
/// - Exporting/importing layout metadata for remote workers
/// - Executing transfers between layouts using handles
/// - Managing CUDA, NIXL, and other execution resources
#[derive(Clone)]
pub struct TransferManager {
    registry: Arc<RwLock<LayoutRegistry>>,
    context: Arc<TransferContext>,
}

impl TransferManager {
    /// Create a new TransferManager builder.
    ///
    /// The builder configures the worker ID, NIXL agent, CUDA device,
    /// and other execution parameters before creating the manager.
    ///
    /// # Example
    /// ```ignore
    /// let manager = TransferManager::builder()
    ///     .worker_id(0)  // NIXL agent name defaults to "worker-0"
    ///     .nixl_backend("ucx")  // Optional: defaults to UCX from env
    ///     .cuda_device_id(0)
    ///     .build()?;
    ///
    /// // Or with custom agent name:
    /// let manager = TransferManager::builder()
    ///     .worker_id(0)
    ///     .nixl_agent_name("custom-agent")
    ///     .build()?;
    /// ```
    pub fn builder() -> crate::v2::physical::transfer::context::TransferConfigBuilder {
        TransferContext::builder()
    }

    /// Create a TransferManager from a built TransferContext.
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

    /// Build a logical layout descriptor for a specific handle.
    ///
    /// This creates a descriptor that includes the logical layout type (G1, G2, G3, G4)
    /// for use in RDMA metadata exchange. The caller must provide the logical type
    /// mapping since only the caller (e.g., DirectWorker) knows which handle corresponds
    /// to which logical tier.
    ///
    /// # Arguments
    /// * `handle` - Handle to the local layout
    /// * `logical_type` - The logical tier (G1, G2, G3, G4) this handle represents
    ///
    /// # Returns
    /// A LogicalLayoutDescriptor ready for serialization
    ///
    /// # Errors
    /// Returns an error if the handle is not found or serialization fails
    pub fn build_logical_descriptor(
        &self,
        handle: LayoutHandle,
        logical_type: LogicalLayoutHandle,
    ) -> Result<LogicalLayoutDescriptor> {
        self.registry
            .read()
            .unwrap()
            .build_logical_descriptor(handle, logical_type)
    }

    /// Get the NIXL metadata for this worker.
    ///
    /// Returns the raw NIXL metadata bytes needed for remote registration.
    pub fn get_nixl_metadata(&self) -> Result<Vec<u8>> {
        self.registry.read().unwrap().get_nixl_metadata()
    }

    /// Get the worker address for this manager.
    pub fn worker_address(&self) -> WorkerAddress {
        self.registry.read().unwrap().worker_address()
    }

    /// Get a reference to the NIXL agent.
    ///
    /// This is useful for building layouts that need to register memory
    /// with the same agent that the TransferManager uses.
    pub fn nixl_agent(&self) -> &NixlAgent {
        self.context.nixl_agent()
    }

    /// Get the layout configuration for a registered layout.
    ///
    /// Returns a clone of the layout's configuration, which includes
    /// dimensions like num_blocks, num_layers, page_size, etc.
    ///
    /// # Arguments
    /// * `handle` - Handle to a registered layout (local or remote)
    ///
    /// # Returns
    /// A clone of the layout's configuration
    ///
    /// # Errors
    /// Returns an error if the handle is not found
    pub fn get_layout_config(
        &self,
        handle: LayoutHandle,
    ) -> Result<crate::v2::physical::layout::LayoutConfig> {
        let registry = self.registry.read().unwrap();
        let physical_layout = registry
            .get_layout(handle)
            .ok_or_else(|| anyhow!("invalid handle: {}", handle))?;
        Ok(physical_layout.layout().config().clone())
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
        src_blocks: &[BlockId],
        dst_handle: LayoutHandle,
        dst_blocks: &[BlockId],
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

        let TransferOptions {
            layer_range,
            nixl_write_notification,
            bounce_buffer,
        } = options;

        let mut internal_options = TransferOptionsInternal::builder();

        if let Some(range) = layer_range {
            internal_options = internal_options.layer_range(range);
        }

        if let Some(notification) = nixl_write_notification {
            internal_options = internal_options.nixl_write_notification(notification);
        }

        if let Some(bounce) = bounce_buffer {
            let (handle, block_ids) = bounce.into_parts();
            let bounce_buffer = self.create_bounce_buffer(handle, block_ids)?;
            internal_options = internal_options.bounce_buffer(bounce_buffer);
        }

        let options = internal_options.build()?;

        tracing::debug!(
            src_handle = src_handle.to_string(),
            dst_handle = dst_handle.to_string(),
            "Executing transfer; src_blocks = {:?}; dst_blocks = {:?}",
            src_blocks,
            dst_blocks,
        );

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

    /// Execute a G4 offload.
    ///
    /// Takes a LayoutHandle and a vector of block IDs for the source blocks and
    /// a list of SequenceHashes for the destination blocks.
    ///
    /// use an extension on TransferOptions to pass in the "rank/part" of the the object in a
    /// multi-worker/multi-tp scenario.
    pub fn execute_g4_offload(
        _src_handle: LayoutHandle,
        _src_blocks: &[BlockId],
        _dst_object: &[SequenceHash],
        _options: TransferOptions, // add rank/part to the options
    ) -> Result<TransferCompleteNotification> {
        // check registration cache for the remote object, if it's not found, register it with nixl
        // register all non-registered blocks with nixl in parallel
        // then extend super::transfer::executor to access the memory regions for the source
        // and generate a nixl descriptor
        todo!("implement remote offload")
    }

    pub fn execute_g4_onboard() {
        todo!("implement remote onboard")
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

    /// Create a bounce buffer specification from a layout handle and block IDs.
    ///
    /// This resolves the layout handle to a physical layout and wraps it in a
    /// BounceBufferSpec implementation for use in transfer options.
    pub(crate) fn create_bounce_buffer(
        &self,
        handle: LayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> Result<BounceBufferInternal> {
        let layout = {
            let registry = self.registry.read().unwrap();
            registry
                .get_layout(handle)
                .ok_or_else(|| anyhow!("invalid bounce buffer handle: {}", handle))?
                .clone()
        };

        Ok(BounceBufferInternal::from_layout(layout, block_ids))
    }

    // ===== Internal Methods for Testing =====

    /// Get the internal transfer context (for testing only).
    pub(crate) fn context(&self) -> &TransferContext {
        &self.context
    }

    /// Get access to the internal layout registry.
    ///
    /// This is primarily for testing utilities that need direct layout access
    /// (e.g., fill patterns, checksum computation).
    pub(crate) fn registry(&self) -> &RwLock<LayoutRegistry> {
        &self.registry
    }

    /// Get the H2D stream (for testing only).
    #[cfg(test)]
    pub(crate) fn h2d_stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        self.context.h2d_stream()
    }

    /// Get the D2H stream (for testing only).
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn d2h_stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        self.context.d2h_stream()
    }

    /// Get the CUDA context (for testing only).
    #[cfg(test)]
    pub(crate) fn cuda_context(&self) -> &std::sync::Arc<cudarc::driver::CudaContext> {
        self.context.cuda_context()
    }

    /// Register a CUDA event for completion (for testing only).
    #[cfg(test)]
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

                serialized_layouts.push(LocalLayoutDescriptor::new_with_default_type(
                    *handle, serialized,
                ));
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

    /// Build a logical layout descriptor for a specific handle.
    ///
    /// # Arguments
    /// * `handle` - Handle to the local layout
    /// * `logical_type` - The logical tier (G1, G2, G3, G4) this handle represents
    ///
    /// # Returns
    /// A LogicalLayoutDescriptor ready for serialization
    pub(crate) fn build_logical_descriptor(
        &self,
        handle: LayoutHandle,
        logical_type: LogicalLayoutHandle,
    ) -> Result<LogicalLayoutDescriptor> {
        let local_layout = self
            .local_layouts
            .get(&handle)
            .ok_or_else(|| anyhow!("Layout handle not found: {:?}", handle))?;

        let layout_descriptor = local_layout
            .layout()
            .to_descriptor()
            .map_err(|e| anyhow!("failed to serialize layout {}: {}", handle, e))?;

        Ok(LogicalLayoutDescriptor::new(
            handle,
            logical_type,
            layout_descriptor,
        ))
    }

    /// Get the NIXL metadata for this worker.
    pub(crate) fn get_nixl_metadata(&self) -> Result<Vec<u8>> {
        self.nixl_agent
            .get_local_md()
            .map_err(|e| anyhow!("failed to get NIXL local metadata: {:?}", e))
    }

    /// Get the worker address for this registry.
    pub(crate) fn worker_address(&self) -> WorkerAddress {
        WorkerAddress::new(self.worker_id, self.nixl_agent.name().to_string())
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
    pub(crate) fn get_layout(&self, handle: LayoutHandle) -> Option<&PhysicalLayout> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::physical::layout::LayoutConfig;
    use dynamo_memory::nixl::NixlAgent;

    fn make_test_agent(name: &str) -> NixlAgent {
        NixlAgent::new(name).expect("failed to create agent")
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
