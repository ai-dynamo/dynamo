// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed builder for constructing [`PhysicalLayout`](crate::block_manager::v2::layout::PhysicalLayout)
//! instances with strongly-typed configuration, layout selection, and memory provisioning.
//!
//! The builder enforces the three steps required to materialize a physical layout:
//! 1. Provide a [`LayoutConfig`]
//! 2. Select a concrete layout (fully contiguous or layer separate)
//! 3. Specify memory backing (either by allocating or by supplying existing regions)
//!
//! NIXL registration is always enabled. Callers must provide a [`nixl_sys::Agent`], and any memory
//! supplied to the builder must implement [`NixlCompatible`].

use crate::block_manager::v2::physical::layout::physical::PhysicalLayout;

use super::{
    BlockDimension, FullyContiguousLayout, LayerSeparateLayout, Layout, LayoutConfig, MemoryRegion,
    physical::NixlMetadata,
};

use crate::block_manager::v2::memory::{
    DiskStorage, NixlCompatible, NixlDescriptor, OffsetMemoryRegion, OwnedMemoryRegion,
    RegisteredView, StorageKind, SystemStorage, register_with_nixl,
};
use anyhow::{Result, anyhow, bail};
#[allow(unused_imports)]
use nixl_sys::Agent as RawNixlAgent;
use nixl_sys::MemType;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;

use crate::block_manager::v2::memory::{DeviceStorage, PinnedStorage};

use crate::block_manager::v2::physical::transfer::nixl_agent::NixlAgent;

const REGION_ALIGNMENT: usize = 512;

/// Layout selection exposed by the builder.
#[derive(Debug, Clone)]
pub enum LayoutKind {
    FullyContiguous,
    LayerSeparate { block_dim: BlockDimension },
}

/// Allocation strategies for builder-managed memory.
#[derive(Debug, Clone)]
enum AllocationKind {
    System,
    Pinned { numa_aware: bool },

    Device { device_id: u32 },
    Disk { path: Option<PathBuf> },
}

/// Memory provisioning plan (either provided regions or an allocation request).
#[derive(Debug, Clone)]
enum MemoryPlan {
    Provided(Vec<MemoryEntry>),
    Allocate(AllocationKind),
}

/// Memory tenancy captured during the build process.
#[derive(Debug, Clone)]
struct MemoryEntry {
    region: OwnedMemoryRegion,
    descriptor: Option<NixlDescriptor>,
}

impl MemoryEntry {
    fn new(region: OwnedMemoryRegion, descriptor: Option<NixlDescriptor>) -> Self {
        Self { region, descriptor }
    }

    fn ensure_registered(mut self) -> Result<Self> {
        if self.descriptor.is_none() {
            self.descriptor = self.region.nixl_descriptor();
        }

        #[cfg(not(test))]
        {
            // In production, require NIXL registration
            if self.descriptor.is_none() {
                bail!(
                    "memory region {} is not registered with NIXL",
                    self.region.addr()
                );
            }
        }

        // In test builds, allow None descriptors for local-only layouts
        Ok(self)
    }
}

/// Marker types for the builder state machine.
pub struct NoConfig;
pub struct HasConfig;

pub struct NoLayout;
pub struct HasLayout;

pub struct NoMemory;
pub struct HasMemory;

/// Default builder state type alias.
pub type PhysicalLayoutBuilderDefault = PhysicalLayoutBuilder<NoConfig, NoLayout, NoMemory>;

/// Typed builder enforcing configuration, layout selection, and memory provisioning phases.
pub struct PhysicalLayoutBuilder<C, L, M> {
    agent: NixlAgent,
    config: Option<LayoutConfig>,
    layout_kind: Option<LayoutKind>,
    memory_plan: Option<MemoryPlan>,
    _config: PhantomData<C>,
    _layout: PhantomData<L>,
    _memory: PhantomData<M>,
}

impl PhysicalLayoutBuilder<NoConfig, NoLayout, NoMemory> {
    /// Create a new builder in its initial state.
    pub fn new(agent: NixlAgent) -> Self {
        Self {
            agent,
            config: None,
            layout_kind: None,
            memory_plan: None,
            _config: PhantomData,
            _layout: PhantomData,
            _memory: PhantomData,
        }
    }
}

impl<C, L, M> PhysicalLayoutBuilder<C, L, M> {
    fn into_parts(
        self,
    ) -> (
        NixlAgent,
        Option<LayoutConfig>,
        Option<LayoutKind>,
        Option<MemoryPlan>,
    ) {
        (self.agent, self.config, self.layout_kind, self.memory_plan)
    }

    fn from_parts<C2, L2, M2>(
        agent: NixlAgent,
        config: Option<LayoutConfig>,
        layout_kind: Option<LayoutKind>,
        memory_plan: Option<MemoryPlan>,
    ) -> PhysicalLayoutBuilder<C2, L2, M2> {
        PhysicalLayoutBuilder {
            agent,
            config,
            layout_kind,
            memory_plan,
            _config: PhantomData,
            _layout: PhantomData,
            _memory: PhantomData,
        }
    }
}

impl<L, M> PhysicalLayoutBuilder<NoConfig, L, M> {
    /// Attach the [`LayoutConfig`] required to size the layout and allocations.
    pub fn with_config(self, config: LayoutConfig) -> PhysicalLayoutBuilder<HasConfig, L, M> {
        let (agent, _config, layout_kind, memory_plan) = self.into_parts();
        PhysicalLayoutBuilder::<HasConfig, L, M>::from_parts(
            agent,
            Some(config),
            layout_kind,
            memory_plan,
        )
    }
}

impl<M> PhysicalLayoutBuilder<HasConfig, NoLayout, M> {
    /// Select the fully contiguous layout variant.
    pub fn fully_contiguous(self) -> PhysicalLayoutBuilder<HasConfig, HasLayout, M> {
        let (agent, config, _layout, memory_plan) = self.into_parts();
        PhysicalLayoutBuilder::<HasConfig, HasLayout, M>::from_parts(
            agent,
            config,
            Some(LayoutKind::FullyContiguous),
            memory_plan,
        )
    }

    /// Select the layer-separate layout variant with the provided block dimension ordering.
    pub fn layer_separate(
        self,
        block_dim: BlockDimension,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, M> {
        let (agent, config, _layout, memory_plan) = self.into_parts();
        PhysicalLayoutBuilder::<HasConfig, HasLayout, M>::from_parts(
            agent,
            config,
            Some(LayoutKind::LayerSeparate { block_dim }),
            memory_plan,
        )
    }
}

impl PhysicalLayoutBuilder<HasConfig, HasLayout, NoMemory> {
    fn set_memory_plan(
        self,
        plan: MemoryPlan,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
        let (agent, config, layout_kind, _memory) = self.into_parts();
        PhysicalLayoutBuilder::<HasConfig, HasLayout, HasMemory>::from_parts(
            agent,
            config,
            layout_kind,
            Some(plan),
        )
    }

    pub fn allocate_system(self) -> PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
        self.set_memory_plan(MemoryPlan::Allocate(AllocationKind::System))
    }

    /// Allocate pinned (page-locked) host memory.
    pub fn allocate_pinned(
        self,
        numa_aware: bool,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
        self.set_memory_plan(MemoryPlan::Allocate(AllocationKind::Pinned { numa_aware }))
    }

    /// Allocate device memory on the specified CUDA device (or the context device if `None`).
    pub fn allocate_device(
        self,
        device_id: u32,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
        self.set_memory_plan(MemoryPlan::Allocate(AllocationKind::Device { device_id }))
    }

    /// Allocate disk-backed storage. When `path` is `None`, a temporary file is used.
    pub fn allocate_disk(
        self,
        path: Option<PathBuf>,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
        self.set_memory_plan(MemoryPlan::Allocate(AllocationKind::Disk { path }))
    }

    /// Use existing NIXL-compatible memory regions supplied by the caller.
    pub fn with_memory_regions<S>(
        self,
        regions: Vec<S>,
    ) -> Result<PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory>>
    where
        S: MemoryRegion + NixlCompatible + 'static,
    {
        let (agent, config, layout_kind, _memory) = self.into_parts();
        let entries = register_existing_regions(&agent, regions)?;
        Ok(
            PhysicalLayoutBuilder::<HasConfig, HasLayout, HasMemory>::from_parts(
                agent,
                config,
                layout_kind,
                Some(MemoryPlan::Provided(entries)),
            ),
        )
    }

    /// Use pre-registered memory regions (already wrapped in `Arc<dyn MemoryRegion>`).
    ///
    /// All regions must already expose a NIXL descriptor.
    pub fn with_registered_regions(
        self,
        regions: Vec<OwnedMemoryRegion>,
    ) -> Result<PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory>> {
        let entries = regions
            .into_iter()
            .enumerate()
            .map(|(index, region)| {
                let descriptor = region.nixl_descriptor().ok_or_else(|| {
                    anyhow!(
                        "provided memory region at index {} is not NIXL registered",
                        index
                    )
                })?;
                Ok(MemoryEntry::new(region, Some(descriptor)))
            })
            .collect::<Result<Vec<_>>>()?;

        let (agent, config, layout_kind, _memory) = self.into_parts();
        Ok(
            PhysicalLayoutBuilder::<HasConfig, HasLayout, HasMemory>::from_parts(
                agent,
                config,
                layout_kind,
                Some(MemoryPlan::Provided(entries)),
            ),
        )
    }
}

impl PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
    /// Finalize the builder, constructing the [`PhysicalLayout`].
    pub fn build(self) -> Result<PhysicalLayout> {
        let (agent, config, layout_kind, memory_plan) = self.into_parts();

        let config = config.ok_or_else(|| anyhow!("layout config missing despite type state"))?;
        let layout_kind =
            layout_kind.ok_or_else(|| anyhow!("layout kind missing despite type state"))?;
        let memory_plan =
            memory_plan.ok_or_else(|| anyhow!("memory plan missing despite type state"))?;

        let required_sizes = compute_allocation_sizes(&config, &layout_kind)?;
        let entries = resolve_memory_plan(&agent, memory_plan, &required_sizes)?;

        validate_memory_sizes(&entries, &required_sizes)?;
        let kind = derive_storage_kind(&entries)?;
        let metadata = derive_nixl_metadata(&agent, &entries)?;

        let layout: Arc<dyn Layout> = match layout_kind {
            LayoutKind::FullyContiguous => {
                let entry = entries.first().ok_or_else(|| {
                    anyhow!("fully contiguous layout requires a single memory region")
                })?;
                let layout = FullyContiguousLayout::new(config.clone(), Arc::clone(&entry.region))?;
                Arc::new(layout)
            }
            LayoutKind::LayerSeparate { block_dim } => {
                let regions: Vec<OwnedMemoryRegion> = entries
                    .iter()
                    .map(|entry| Arc::clone(&entry.region))
                    .collect();
                let layout = LayerSeparateLayout::new(config.clone(), regions, block_dim)?;
                Arc::new(layout)
            }
        };

        Ok(PhysicalLayout::new_local(layout, kind, metadata))
    }
}

fn register_existing_regions<S>(agent: &NixlAgent, regions: Vec<S>) -> Result<Vec<MemoryEntry>>
where
    S: MemoryRegion + NixlCompatible + 'static,
{
    regions
        .into_iter()
        .map(|region| register_storage(region, agent))
        .collect()
}

fn resolve_memory_plan(
    agent: &NixlAgent,
    plan: MemoryPlan,
    sizes: &[usize],
) -> Result<Vec<MemoryEntry>> {
    match plan {
        MemoryPlan::Provided(entries) => {
            if entries.len() != sizes.len() {
                bail!(
                    "provided memory count ({}) does not match required allocations ({})",
                    entries.len(),
                    sizes.len()
                );
            }
            entries
                .into_iter()
                .map(MemoryEntry::ensure_registered)
                .collect()
        }
        MemoryPlan::Allocate(strategy) => allocate_regions(agent, strategy, sizes),
    }
}

fn allocate_regions(
    agent: &NixlAgent,
    strategy: AllocationKind,
    sizes: &[usize],
) -> Result<Vec<MemoryEntry>> {
    if sizes.is_empty() {
        return Ok(Vec::new());
    }

    let reserve_size = total_allocation_size(sizes, REGION_ALIGNMENT)?;

    let base_entry = match strategy {
        AllocationKind::System => allocate_system_entry(reserve_size, agent)?,
        AllocationKind::Pinned { numa_aware } => {
            allocate_pinned_entry(reserve_size, agent, numa_aware)?
        }

        AllocationKind::Device { device_id } => {
            allocate_device_entry(reserve_size, agent, device_id)?
        }
        AllocationKind::Disk { path } => allocate_disk_entry(reserve_size, agent, path)?,
    };

    create_offset_entries(base_entry, sizes, REGION_ALIGNMENT)
}

fn allocate_system_entry(size: usize, agent: &NixlAgent) -> Result<MemoryEntry> {
    let storage = SystemStorage::new(size)
        .map_err(|e| anyhow!("failed to allocate system memory ({size} bytes): {e}"))?;
    register_storage(storage, agent)
}

fn allocate_pinned_entry(size: usize, agent: &NixlAgent, _numa_aware: bool) -> Result<MemoryEntry> {
    let storage = PinnedStorage::new(size)
        .map_err(|e| anyhow!("failed to allocate pinned memory ({size} bytes): {e}"))?;
    register_storage(storage, agent)
}

fn allocate_device_entry(size: usize, agent: &NixlAgent, device_id: u32) -> Result<MemoryEntry> {
    let storage = DeviceStorage::new(size, device_id).map_err(|e| {
        anyhow!("failed to allocate device memory ({size} bytes) on device {device_id}: {e}")
    })?;
    register_storage(storage, agent)
}

fn allocate_disk_entry(
    size: usize,
    agent: &NixlAgent,
    path: Option<PathBuf>,
) -> Result<MemoryEntry> {
    let storage = if let Some(path) = path {
        DiskStorage::new_at(&path, size)
            .map_err(|e| anyhow!("failed to allocate disk storage at {}: {e}", path.display()))?
    } else {
        DiskStorage::new(size).map_err(|e| anyhow!("failed to allocate disk storage: {e}"))?
    };
    register_storage(storage, agent)
}

// When testing, we allow unregistered layouts to help with test time. NIXL + UCX is very expensive to setup
// so we only use that backend when it's needed.
#[cfg(test)]
fn register_storage<S>(storage: S, agent: &NixlAgent) -> Result<MemoryEntry>
where
    S: MemoryRegion + NixlCompatible + 'static,
{
    let storage_kind = storage.storage_kind();

    // Determine if registration is needed based on storage type and available backends
    let should_register = match storage_kind {
        StorageKind::System | StorageKind::Pinned => {
            // System/Pinned memory needs UCX for remote transfers
            agent.has_backend("UCX") || agent.has_backend("POSIX")
        }
        StorageKind::Device(_) => {
            // Device memory needs UCX for remote transfers OR GDS for direct disk transfers
            agent.has_backend("UCX") || agent.has_backend("GDS_MT")
        }
        StorageKind::Disk(_) => {
            // Disk storage needs POSIX for regular I/O OR GDS for GPU direct I/O
            agent.has_backend("POSIX") || agent.has_backend("GDS_MT")
        }
    };

    if !should_register {
        // Skip registration - only local non-NIXL transfers will be used
        let region: OwnedMemoryRegion = Arc::new(storage);
        return Ok(MemoryEntry::new(region, None));
    }

    // Register with NIXL using the appropriate backend
    match register_with_nixl(storage, agent.raw_agent(), None) {
        Ok(registered) => {
            let descriptor = registered.descriptor();
            let region: OwnedMemoryRegion = Arc::new(registered);
            Ok(MemoryEntry::new(region, Some(descriptor)))
        }
        Err(_storage) => bail!("failed to register memory with NIXL agent {}", agent.name()),
    }
}

// Production builds always register
#[cfg(not(test))]
fn register_storage<S>(storage: S, agent: &NixlAgent) -> Result<MemoryEntry>
where
    S: MemoryRegion + NixlCompatible + 'static,
{
    // Production builds always register for safety
    match register_with_nixl(storage, agent.raw_agent(), None) {
        Ok(registered) => {
            let descriptor = registered.descriptor();
            let region: OwnedMemoryRegion = Arc::new(registered);
            Ok(MemoryEntry::new(region, Some(descriptor)))
        }
        Err(_storage) => bail!("failed to register memory with NIXL agent {}", agent.name()),
    }
}

fn create_offset_entries(
    base_entry: MemoryEntry,
    sizes: &[usize],
    alignment: usize,
) -> Result<Vec<MemoryEntry>> {
    if sizes.is_empty() {
        return Ok(Vec::new());
    }

    let base_region = base_entry.region;
    let base_descriptor = base_entry.descriptor;
    let base_addr = base_region.addr();
    let base_len = base_region.size();

    let mut entries = Vec::with_capacity(sizes.len());
    let mut offset = 0usize;

    for (index, &size) in sizes.iter().enumerate() {
        let region = if index == 0 && offset == 0 && size == base_len && sizes.len() == 1 {
            Arc::clone(&base_region)
        } else {
            let view = OffsetMemoryRegion::new(Arc::clone(&base_region), offset, size)
                .map_err(|e| anyhow!("failed to create offset region: {e}"))?;
            Arc::new(view) as OwnedMemoryRegion
        };

        let descriptor = base_descriptor
            .as_ref()
            .map(|descriptor| derive_descriptor(descriptor, offset, size))
            .transpose()?;

        entries.push(MemoryEntry::new(region, descriptor));

        offset = offset
            .checked_add(size)
            .ok_or_else(|| anyhow!("offset computation overflow"))?;

        if index + 1 < sizes.len() && alignment > 1 {
            let current_addr = base_addr
                .checked_add(offset)
                .ok_or_else(|| anyhow!("address computation overflow"))?;
            let aligned_addr = align_up(current_addr, alignment)?;
            offset = aligned_addr
                .checked_sub(base_addr)
                .ok_or_else(|| anyhow!("alignment subtraction overflow"))?;
        }
    }

    if offset > base_len {
        bail!(
            "allocated base region ({base_len} bytes) is insufficient for {offset} bytes with padding"
        );
    }

    Ok(entries)
}

fn derive_descriptor(base: &NixlDescriptor, offset: usize, size: usize) -> Result<NixlDescriptor> {
    let mut descriptor = base.clone();
    descriptor.size = size;
    if descriptor.mem_type != MemType::File {
        descriptor.addr = descriptor
            .addr
            .checked_add(offset as u64)
            .ok_or_else(|| anyhow!("descriptor address overflow"))?;
    }
    Ok(descriptor)
}

fn compute_allocation_sizes(config: &LayoutConfig, kind: &LayoutKind) -> Result<Vec<usize>> {
    match kind {
        LayoutKind::FullyContiguous => {
            let factors = [
                config.num_blocks,
                config.num_layers,
                config.outer_dim,
                config.page_size,
                config.inner_dim,
                config.dtype_width_bytes,
            ];
            let total = mul_chain(&factors)?;
            Ok(vec![total])
        }
        LayoutKind::LayerSeparate { .. } => {
            let factors = [
                config.num_blocks,
                config.outer_dim,
                config.page_size,
                config.inner_dim,
                config.dtype_width_bytes,
            ];
            let per_layer = mul_chain(&factors)?;
            Ok(vec![per_layer; config.num_layers])
        }
    }
}

fn mul_chain(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value)
            .ok_or_else(|| anyhow!("allocation size overflow during layout computation"))
    })
}

fn total_allocation_size(sizes: &[usize], alignment: usize) -> Result<usize> {
    if sizes.is_empty() {
        return Ok(0);
    }

    let mut total = *sizes
        .first()
        .ok_or_else(|| anyhow!("allocation requires at least one region"))?;

    for size in sizes.iter().skip(1) {
        total = total
            .checked_add(*size)
            .ok_or_else(|| anyhow!("allocation size overflow during aggregation"))?;
        if alignment > 1 {
            total = total
                .checked_add(alignment - 1)
                .ok_or_else(|| anyhow!("allocation alignment padding overflow"))?;
        }
    }

    Ok(total)
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    if alignment <= 1 {
        return Ok(value);
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| anyhow!("alignment overflow"))
    }
}

fn validate_memory_sizes(entries: &[MemoryEntry], required: &[usize]) -> Result<()> {
    for (entry, &required_size) in entries.iter().zip(required.iter()) {
        if entry.region.size() < required_size {
            bail!(
                "memory region too small: required {} bytes, available {} bytes",
                required_size,
                entry.region.size()
            );
        }
    }
    Ok(())
}

fn derive_storage_kind(entries: &[MemoryEntry]) -> Result<StorageKind> {
    let first = entries
        .first()
        .ok_or_else(|| anyhow!("no memory regions available to determine storage location"))?;
    let first_kind = first.region.storage_kind();

    for entry in entries.iter().skip(1) {
        let kind = entry.region.storage_kind();
        if kind != first_kind {
            bail!(
                "all memory regions must share the same storage location (found {:?} and {:?})",
                first_kind,
                kind
            );
        }
    }

    Ok(first_kind)
}

fn derive_nixl_metadata(agent: &NixlAgent, entries: &[MemoryEntry]) -> Result<NixlMetadata> {
    // Try to find a descriptor from entries
    let descriptor_opt = entries.iter().find_map(|entry| entry.descriptor.clone());

    #[cfg(test)]
    {
        // In test builds, allow layouts without NIXL registration
        // Use defaults for local-only transfers
        if let Some(descriptor) = descriptor_opt {
            Ok(NixlMetadata::new(
                agent.name().to_string(),
                descriptor.mem_type,
                descriptor.device_id,
            ))
        } else {
            // Use placeholder metadata for unregistered layouts
            let first_entry = entries
                .first()
                .ok_or_else(|| anyhow!("no memory entries"))?;
            let storage_kind = first_entry.region.storage_kind();
            let (mem_type, device_id) = match storage_kind {
                StorageKind::System => (MemType::Dram, 0),
                StorageKind::Pinned => (MemType::Dram, 0),
                StorageKind::Device(id) => (MemType::Vram, id as u64),
                StorageKind::Disk(id) => (MemType::File, id),
            };
            Ok(NixlMetadata::new(
                agent.name().to_string(),
                mem_type,
                device_id,
            ))
        }
    }

    #[cfg(not(test))]
    {
        let descriptor = descriptor_opt
            .ok_or_else(|| anyhow!("memory entries missing NIXL registration metadata"))?;
        Ok(NixlMetadata::new(
            agent.name().to_string(),
            descriptor.mem_type,
            descriptor.device_id,
        ))
    }
}

#[cfg(all(test, feature = "testing-nixl"))]
mod tests {
    use super::super::{BlockDimension, LayoutConfig};
    use super::*;

    use crate::block_manager::v2::memory::{MemoryRegion, OwnedMemoryRegion, StorageKind};
    use nixl_sys::MemType;
    use std::any::Any;
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestRegisteredRegion {
        data: Vec<u8>,
        kind: StorageKind,
        descriptor: NixlDescriptor,
    }

    impl TestRegisteredRegion {
        fn new(size: usize, kind: StorageKind, mem_type: MemType, device_id: u64) -> Self {
            let data = vec![0u8; size];
            let addr = data.as_ptr() as u64;
            let descriptor = NixlDescriptor {
                addr,
                size,
                mem_type,
                device_id,
            };
            Self {
                data,
                kind,
                descriptor,
            }
        }
    }

    impl MemoryRegion for TestRegisteredRegion {
        fn addr(&self) -> usize {
            self.data.as_ptr() as usize
        }

        fn size(&self) -> usize {
            self.data.len()
        }

        fn storage_kind(&self) -> StorageKind {
            self.kind
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
            Some(self.descriptor.clone())
        }
    }

    fn make_layout_config() -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(2)
            .num_layers(3)
            .outer_dim(2)
            .page_size(4)
            .inner_dim(8)
            .dtype_width_bytes(2)
            .build()
            .unwrap()
    }

    fn fully_contiguous_size(cfg: &LayoutConfig) -> usize {
        cfg.num_blocks
            * cfg.num_layers
            * cfg.outer_dim
            * cfg.page_size
            * cfg.inner_dim
            * cfg.dtype_width_bytes
    }

    fn per_layer_size(cfg: &LayoutConfig) -> usize {
        cfg.num_blocks * cfg.outer_dim * cfg.page_size * cfg.inner_dim * cfg.dtype_width_bytes
    }

    #[test]
    fn builds_fully_contiguous_from_registered_regions() {
        let agent = NixlAgent::require_backends("builder-test-fully", &[])
            .expect("failed to create wrapped agent");
        let cfg = make_layout_config();

        let required = fully_contiguous_size(&cfg);
        let region = Arc::new(TestRegisteredRegion::new(
            required,
            StorageKind::System,
            MemType::Dram,
            0,
        )) as OwnedMemoryRegion;

        let physical = PhysicalLayoutBuilder::new(agent.clone())
            .with_config(cfg.clone())
            .fully_contiguous()
            .with_registered_regions(vec![region])
            .expect("registered regions accepted")
            .build()
            .expect("builder should succeed");

        assert_eq!(physical.location(), StorageKind::System);
        assert!(physical.layout().as_ref().is_fully_contiguous());
        assert_eq!(physical.layout().config().num_blocks, cfg.num_blocks);
        assert_eq!(physical.layout().config().num_layers, cfg.num_layers);

        let metadata = physical.nixl_metadata();
        assert_eq!(metadata.agent_name(), agent.name());
        assert_eq!(metadata.mem_type(), MemType::Dram);
    }

    #[test]
    fn builds_layer_separate_from_registered_regions() {
        let agent = NixlAgent::require_backends("builder-test-layer", &[])
            .expect("failed to create wrapped agent");
        let cfg = make_layout_config();

        let per_layer = per_layer_size(&cfg);
        let regions: Vec<OwnedMemoryRegion> = (0..cfg.num_layers)
            .map(|_| {
                Arc::new(TestRegisteredRegion::new(
                    per_layer,
                    StorageKind::System,
                    MemType::Dram,
                    0,
                )) as OwnedMemoryRegion
            })
            .collect();

        let physical = PhysicalLayoutBuilder::new(agent.clone())
            .with_config(cfg.clone())
            .layer_separate(BlockDimension::BlockIsFirstDim)
            .with_registered_regions(regions)
            .expect("registered layer regions accepted")
            .build()
            .expect("builder should succeed");

        assert_eq!(physical.location(), StorageKind::System);
        assert!(!physical.layout().as_ref().is_fully_contiguous());
        assert_eq!(physical.layout().config().num_layers, cfg.num_layers);

        let metadata = physical.nixl_metadata();
        assert_eq!(metadata.agent_name(), agent.name());
        assert_eq!(metadata.mem_type(), MemType::Dram);
    }
}

// fn context_device_id(ctx: &TransferContext) -> u32 {
//     ctx.stream().context().ordinal() as u32
// }
