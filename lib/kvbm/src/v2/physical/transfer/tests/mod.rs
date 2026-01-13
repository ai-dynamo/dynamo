// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive transfer tests for verifying data integrity across storage types and layout configurations.

mod cuda_kernel_tests;
mod local_transfers;

/// Skip test if stub kernels are in use (no real CUDA available).
///
/// Call this at the start of any test that requires real CUDA operations.
/// When stubs are in use, the test will print a message and return early.
///
/// # Example
/// ```ignore
/// #[test]
/// fn my_cuda_test() {
///     skip_if_stubs!();
///     // ... test code that requires CUDA ...
/// }
/// ```
macro_rules! skip_if_stubs {
    () => {
        if dynamo_kvbm_kernels::is_using_stubs() {
            eprintln!(
                "Skipping test '{}': stub kernels in use (no real CUDA)",
                module_path!()
            );
            return;
        }
    };
}

/// Check if any of the storage kinds require CUDA, and skip if stubs are in use.
///
/// Call this at the start of parameterized tests that may or may not use Device storage.
macro_rules! skip_if_stubs_and_device {
    ($($kind:expr),+ $(,)?) => {
        if dynamo_kvbm_kernels::is_using_stubs() {
            let needs_cuda = false $(|| matches!($kind, StorageKind::Device(_)))+;
            if needs_cuda {
                eprintln!(
                    "Skipping test '{}': stub kernels in use and test requires Device storage",
                    module_path!()
                );
                return Ok(());
            }
        }
    };
}

/// Check if GDS (GPU Direct Storage) is available on this system.
///
/// This function performs a one-time test to verify GDS functionality.
/// The result is cached for subsequent calls.
#[allow(dead_code)] // Available for tests that need GDS detection
pub fn is_gds_available() -> bool {
    static GDS_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *GDS_AVAILABLE.get_or_init(|| TransferCapabilities::default().with_gds_if_supported().allow_gds)
}

/// Skip test if GDS (GPU Direct Storage) is not available.
///
/// Call this at the start of any test that requires real GDS operations.
/// When GDS is not available, the test will print a message and return early.
///
/// If `REQUIRE_GDS_TESTS=1` environment variable is set, the test will panic
/// instead of skipping, ensuring CI environments with GDS hardware enforce
/// these tests pass.
///
/// # Example
/// ```ignore
/// #[test]
/// fn my_gds_test() -> Result<()> {
///     skip_if_no_gds!();
///     // ... test code that requires GDS ...
/// }
/// ```
#[allow(unused_macros)] // Available for tests that require real GDS operations
macro_rules! skip_if_no_gds {
    () => {
        if !is_gds_available() {
            if std::env::var("REQUIRE_GDS_TESTS")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                panic!(
                    "Test '{}' requires GDS but GDS is not available. \
                     REQUIRE_GDS_TESTS=1 is set, so this is a failure.",
                    module_path!()
                );
            }
            eprintln!(
                "Skipping test '{}': GDS not available on this system. \
                 Set REQUIRE_GDS_TESTS=1 to make this a failure.",
                module_path!()
            );
            return Ok(());
        }
    };
}

// Make the macros available to submodules
pub(crate) use skip_if_no_gds;
pub(crate) use skip_if_stubs;
pub(crate) use skip_if_stubs_and_device;

use super::{
    BlockChecksum, FillPattern, NixlAgent, PhysicalLayout, StorageKind, TransferCapabilities,
    compute_block_checksums, compute_layer_checksums, fill_blocks, fill_layers,
};
use crate::{
    BlockId,
    v2::physical::layout::{
        BlockDimension, LayoutConfig,
        builder::{HasConfig, NoLayout, NoMemory, PhysicalLayoutBuilder},
    },
};
use anyhow::Result;
use cudarc::driver::sys::CUdevice_attribute_enum;
use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// Layout kind for parameterized testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutKind {
    /// Fully contiguous layout
    FC,
    /// Layer-wise (layer-separate) layout
    LW,
}

/// Storage and layout specification for creating test layouts.
#[derive(Debug, Clone, Copy)]
pub struct LayoutSpec {
    pub kind: LayoutKind,
    pub storage: StorageKind,
}

impl LayoutSpec {
    pub fn new(kind: LayoutKind, storage: StorageKind) -> Self {
        Self { kind, storage }
    }
}

/// Transfer mode for parameterized testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMode {
    /// Transfer entire blocks (all layers)
    FullBlocks,
    /// Transfer only the first layer
    FirstLayerOnly,
    /// Transfer only the second layer
    SecondLayerOnly,
}

impl TransferMode {
    /// Convert to optional layer range for execute_transfer.
    pub fn layer_range(&self) -> Option<Range<usize>> {
        match self {
            TransferMode::FullBlocks => None,
            TransferMode::FirstLayerOnly => Some(0..1),
            TransferMode::SecondLayerOnly => Some(1..2),
        }
    }

    /// Get a descriptive suffix for test names.
    pub fn suffix(&self) -> &'static str {
        match self {
            TransferMode::FullBlocks => "full",
            TransferMode::FirstLayerOnly => "layer0",
            TransferMode::SecondLayerOnly => "layer1",
        }
    }
}

/// Standard layout configuration for all tests.
pub fn standard_config(num_blocks: usize) -> LayoutConfig {
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

/// Helper function for creating a PhysicalLayout builder with standard config.
///
/// This is used by other test modules (fill, checksum, validation) for backwards compatibility.
pub fn builder(num_blocks: usize) -> PhysicalLayoutBuilder<HasConfig, NoLayout, NoMemory> {
    let agent = create_test_agent("test_agent");
    let config = standard_config(num_blocks);
    PhysicalLayout::builder(agent).with_config(config)
}

/// Create a test agent with no backends.
///
/// Use this for tests that don't require specific NIXL backends.
pub fn create_test_agent(name: &str) -> NixlAgent {
    NixlAgent::new(name).expect("Failed to create agent")
}

/// Create a test agent with specific backends (strict - all must succeed).
pub fn create_test_agent_with_backends(name: &str, backends: &[&str]) -> Result<NixlAgent> {
    NixlAgent::with_backends(name, backends)
}

/// Create a fully contiguous physical layout with the specified storage type.
pub fn create_fc_layout(
    agent: NixlAgent,
    storage_kind: StorageKind,
    num_blocks: usize,
) -> PhysicalLayout {
    let config = standard_config(num_blocks);
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous();

    match storage_kind {
        StorageKind::System => builder.allocate_system().build().unwrap(),
        StorageKind::Pinned => builder.allocate_pinned(None).build().unwrap(),
        StorageKind::Device(device_id) => builder.allocate_device(device_id).build().unwrap(),
        StorageKind::Disk(_) => builder.allocate_disk(None).build().unwrap(),
    }
}

/// Create a layer-separate physical layout with the specified storage type.
pub fn create_lw_layout(
    agent: NixlAgent,
    storage_kind: StorageKind,
    num_blocks: usize,
) -> PhysicalLayout {
    let config = standard_config(num_blocks);
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .layer_separate(BlockDimension::BlockIsFirstDim);

    match storage_kind {
        StorageKind::System => builder.allocate_system().build().unwrap(),
        StorageKind::Pinned => builder.allocate_pinned(None).build().unwrap(),
        StorageKind::Device(device_id) => builder.allocate_device(device_id).build().unwrap(),
        StorageKind::Disk(_) => builder.allocate_disk(None).build().unwrap(),
    }
}

/// Create a physical layout based on the specification.
///
/// This is a DRY helper that dispatches to create_fc_layout or create_lw_layout
/// based on the layout kind in the spec.
pub fn create_layout(agent: NixlAgent, spec: LayoutSpec, num_blocks: usize) -> PhysicalLayout {
    match spec.kind {
        LayoutKind::FC => create_fc_layout(agent, spec.storage, num_blocks),
        LayoutKind::LW => create_lw_layout(agent, spec.storage, num_blocks),
    }
}

/// Create a transport manager for testing with the specified agent.
///
/// Note: The agent should already have backends configured. Use `create_test_agent`
/// or `build_agent_with_backends` to create properly configured agents.
pub fn create_transfer_context(
    agent: NixlAgent,
    capabilities: Option<TransferCapabilities>,
) -> Result<crate::v2::physical::manager::TransferManager> {
    crate::v2::physical::manager::TransferManager::builder()
        .capabilities(capabilities.unwrap_or_default())
        .nixl_agent(agent)
        .cuda_device_id(0)
        .build()
}

/// Fill blocks and compute checksums.
///
/// This can only be called on System or Pinned layouts.
pub fn fill_and_checksum(
    layout: &PhysicalLayout,
    block_ids: &[BlockId],
    pattern: FillPattern,
) -> Result<HashMap<BlockId, BlockChecksum>> {
    fill_blocks(layout, block_ids, pattern)?;
    compute_block_checksums(layout, block_ids)
}

/// Fill blocks or layers based on transfer mode and compute checksums.
///
/// This is a mode-aware version of fill_and_checksum that handles both
/// full block transfers and layer-wise transfers.
pub fn fill_and_checksum_with_mode(
    layout: &PhysicalLayout,
    block_ids: &[BlockId],
    pattern: FillPattern,
    mode: TransferMode,
) -> Result<HashMap<BlockId, BlockChecksum>> {
    match mode {
        TransferMode::FullBlocks => {
            fill_blocks(layout, block_ids, pattern)?;
            compute_block_checksums(layout, block_ids)
        }
        TransferMode::FirstLayerOnly => {
            fill_layers(layout, block_ids, 0..1, pattern)?;
            compute_layer_checksums(layout, block_ids, 0..1)
        }
        TransferMode::SecondLayerOnly => {
            fill_layers(layout, block_ids, 1..2, pattern)?;
            compute_layer_checksums(layout, block_ids, 1..2)
        }
    }
}

/// Verify that destination block checksums match the expected source checksums.
///
/// This function compares checksums in order, assuming the source and destination
/// block arrays have a 1:1 correspondence (src[i] was transferred to dst[i]).
pub fn verify_checksums_by_position(
    src_checksums: &HashMap<BlockId, BlockChecksum>,
    src_block_ids: &[BlockId],
    dst_layout: &PhysicalLayout,
    dst_block_ids: &[BlockId],
) -> Result<()> {
    assert_eq!(
        src_block_ids.len(),
        dst_block_ids.len(),
        "Source and destination block arrays must have same length"
    );

    let dst_checksums = compute_block_checksums(dst_layout, dst_block_ids)?;

    for (src_id, dst_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        let src_checksum = src_checksums
            .get(src_id)
            .unwrap_or_else(|| panic!("Missing source checksum for block {}", src_id));
        let dst_checksum = dst_checksums
            .get(dst_id)
            .unwrap_or_else(|| panic!("Missing destination checksum for block {}", dst_id));

        assert_eq!(
            src_checksum, dst_checksum,
            "Checksum mismatch: src[{}] != dst[{}]: {} != {}",
            src_id, dst_id, src_checksum, dst_checksum
        );
    }

    Ok(())
}

/// Verify checksums with transfer mode awareness.
///
/// This is a mode-aware version that handles both full block and layer-wise verification.
pub fn verify_checksums_by_position_with_mode(
    src_checksums: &HashMap<BlockId, BlockChecksum>,
    src_block_ids: &[BlockId],
    dst_layout: &PhysicalLayout,
    dst_block_ids: &[BlockId],
    mode: TransferMode,
) -> Result<()> {
    assert_eq!(
        src_block_ids.len(),
        dst_block_ids.len(),
        "Source and destination block arrays must have same length"
    );

    let dst_checksums = match mode {
        TransferMode::FullBlocks => compute_block_checksums(dst_layout, dst_block_ids)?,
        TransferMode::FirstLayerOnly => compute_layer_checksums(dst_layout, dst_block_ids, 0..1)?,
        TransferMode::SecondLayerOnly => compute_layer_checksums(dst_layout, dst_block_ids, 1..2)?,
    };

    for (src_id, dst_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        let src_checksum = src_checksums
            .get(src_id)
            .unwrap_or_else(|| panic!("Missing source checksum for block {}", src_id));
        let dst_checksum = dst_checksums
            .get(dst_id)
            .unwrap_or_else(|| panic!("Missing destination checksum for block {}", dst_id));

        assert_eq!(
            src_checksum, dst_checksum,
            "Checksum mismatch (mode={:?}): src[{}] != dst[{}]: {} != {}",
            mode, src_id, dst_id, src_checksum, dst_checksum
        );
    }

    Ok(())
}

/// Fill guard blocks and return their checksums for later verification.
///
/// Guard blocks are blocks adjacent to transfer destinations that should
/// remain unchanged during transfers. This function fills them with a
/// distinctive pattern and returns their checksums for later validation.
///
/// # Arguments
/// * `layout` - The physical layout containing the guard blocks
/// * `guard_block_ids` - Block IDs to use as guards
/// * `pattern` - Fill pattern for guard blocks (typically a constant like 0xFF)
///
/// # Returns
/// A map of block ID to checksum for all guard blocks
pub fn create_guard_blocks(
    layout: &PhysicalLayout,
    guard_block_ids: &[usize],
    pattern: FillPattern,
) -> Result<HashMap<usize, BlockChecksum>> {
    fill_blocks(layout, guard_block_ids, pattern)?;
    compute_block_checksums(layout, guard_block_ids)
}

/// Verify that guard blocks remain unchanged after transfers.
///
/// This function compares the current checksums of guard blocks against
/// their expected values. Any mismatch indicates memory corruption or
/// unintended overwrites during transfer operations.
///
/// # Arguments
/// * `layout` - The physical layout containing the guard blocks
/// * `guard_block_ids` - Block IDs to verify
/// * `expected_checksums` - Expected checksums from create_guard_blocks
///
/// # Errors
/// Returns an error if any guard block checksum has changed
pub fn verify_guard_blocks_unchanged(
    layout: &PhysicalLayout,
    guard_block_ids: &[usize],
    expected_checksums: &HashMap<usize, BlockChecksum>,
) -> Result<()> {
    let current_checksums = compute_block_checksums(layout, guard_block_ids)?;

    for &block_id in guard_block_ids {
        let expected = expected_checksums
            .get(&block_id)
            .unwrap_or_else(|| panic!("Missing expected checksum for guard block {}", block_id));
        let current = current_checksums
            .get(&block_id)
            .unwrap_or_else(|| panic!("Missing current checksum for guard block {}", block_id));

        if expected != current {
            return Err(anyhow::anyhow!(
                "Guard block {} was modified during transfer! Expected: {}, Got: {}",
                block_id,
                expected,
                current
            ));
        }
    }

    Ok(())
}

/// CUDA sleep kernel source code.
const SLEEP_KERNEL_SRC: &str = r#"
extern "C" __global__ void sleep_kernel(unsigned long long min_cycles) {
    const unsigned long long start = clock64();
    while ((clock64() - start) < min_cycles) {
        asm volatile("");
    }
}
"#;

/// A reusable CUDA sleep utility for tests.
///
/// This struct provides a simple interface to execute GPU sleep operations
/// with calibrated timing. It compiles the sleep kernel once per CUDA context
/// and caches the calibration for reuse.
///
/// The calibration is conservative (prefers longer sleep durations over shorter)
/// to ensure minimum sleep times are met.
pub struct CudaSleep {
    function: cudarc::driver::CudaFunction,
    cycles_per_ms: f64,
}

impl CudaSleep {
    /// Get or create a CudaSleep instance for the given CUDA context.
    ///
    /// This function uses lazy initialization and caches instances per device ID.
    /// The first call for each device will compile the kernel and run calibration.
    ///
    /// # Arguments
    /// * `cuda_ctx` - The CUDA context to use
    ///
    /// # Returns
    /// A shared reference to the CudaSleep instance for this context's device.
    pub fn for_context(cuda_ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        static INSTANCES: OnceLock<parking_lot::Mutex<HashMap<usize, Arc<CudaSleep>>>> =
            OnceLock::new();

        let instances = INSTANCES.get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
        let device_ordinal = cuda_ctx.ordinal();

        // Fast path: check if instance already exists
        {
            let instances_guard = instances.lock();
            if let Some(instance) = instances_guard.get(&device_ordinal) {
                return Ok(Arc::clone(instance));
            }
        }

        // Slow path: create new instance with calibration
        let instance = Arc::new(Self::new(cuda_ctx)?);

        // Store in cache
        let mut instances_guard = instances.lock();
        instances_guard
            .entry(device_ordinal)
            .or_insert_with(|| Arc::clone(&instance));

        Ok(instance)
    }

    /// Create a new CudaSleep instance with calibration.
    ///
    /// This compiles the sleep kernel and runs a calibration loop to determine
    /// the relationship between clock cycles and wall-clock time.
    fn new(cuda_ctx: &Arc<CudaContext>) -> Result<Self> {
        // Get device compute capability
        let major = cuda_ctx
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = cuda_ctx
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;

        // Compile PTX for this device
        let mut compile_opts = CompileOptions {
            name: Some("sleep_kernel.cu".into()),
            ..Default::default()
        };
        compile_opts
            .options
            .push(format!("--gpu-architecture=compute_{}{}", major, minor));
        let ptx = compile_ptx_with_opts(SLEEP_KERNEL_SRC, compile_opts)?;
        let module = cuda_ctx.load_module(ptx)?;
        let function = module.load_function("sleep_kernel")?;

        // Get device clock rate
        let clock_rate_khz =
            cuda_ctx.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)? as u64;

        // Create a temporary stream for calibration
        let stream = cuda_ctx.new_stream()?;

        // Warm up to absorb JIT overhead
        let warm_cycles = clock_rate_khz.saturating_mul(10).max(1);
        Self::launch_kernel(&function, &stream, warm_cycles)?;
        stream.synchronize()?;

        // Run calibration loop
        let desired_delay = Duration::from_millis(600);
        let mut target_cycles = clock_rate_khz.saturating_mul(50).max(1); // ~50ms starting point
        let mut actual_duration = Duration::ZERO;

        for _ in 0..8 {
            let start = Instant::now();
            Self::launch_kernel(&function, &stream, target_cycles)?;
            stream.synchronize()?;
            actual_duration = start.elapsed();

            if actual_duration >= desired_delay {
                break;
            }

            target_cycles = target_cycles.saturating_mul(2);
        }

        // Calculate cycles per millisecond with conservative 20% margin
        // (prefer longer sleeps over shorter)
        let cycles_per_ms = if actual_duration.as_millis() > 0 {
            (target_cycles as f64 / actual_duration.as_millis() as f64) * 1.2
        } else {
            clock_rate_khz as f64 // Fallback to clock rate
        };

        Ok(Self {
            function,
            cycles_per_ms,
        })
    }

    /// Launch the sleep kernel with the specified number of cycles.
    fn launch_kernel(
        function: &cudarc::driver::CudaFunction,
        stream: &Arc<CudaStream>,
        cycles: u64,
    ) -> Result<()> {
        let launch_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launch = stream.launch_builder(function);
        unsafe {
            launch.arg(&cycles);
            launch.launch(launch_cfg)?;
        }

        Ok(())
    }

    /// Launch a sleep operation on the given stream.
    ///
    /// This queues a GPU kernel that will sleep for approximately the specified
    /// duration. The sleep is conservative and may take longer than requested.
    ///
    /// # Arguments
    /// * `duration` - The minimum duration to sleep
    /// * `stream` - The CUDA stream to launch the kernel on
    ///
    /// # Returns
    /// Ok(()) if the kernel was successfully queued
    pub fn launch(&self, duration: Duration, stream: &Arc<CudaStream>) -> Result<()> {
        let target_cycles = (duration.as_millis() as f64 * self.cycles_per_ms) as u64;
        Self::launch_kernel(&self.function, stream, target_cycles)
    }
}
