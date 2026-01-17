// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive transfer tests for verifying data integrity across storage types and layout configurations.

#[cfg(all(feature = "testing-cuda", feature = "testing-nixl"))]
mod local_transfers;

use super::{NixlAgent, PhysicalLayout};
use crate::block_manager::v2::physical::layout::{
    LayoutConfig,
    builder::{HasConfig, NoLayout, NoMemory, PhysicalLayoutBuilder},
};

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

/// Create a test agent with optimal backends for testing.
///
/// Attempts to initialize UCX, GDS, and POSIX backends. Falls back gracefully
/// if some backends are unavailable (e.g., GDS on non-DGX machines).
pub fn create_test_agent(name: &str) -> NixlAgent {
    NixlAgent::require_backends(name, &[]).expect("Failed to require backends")
}

#[cfg(feature = "testing-cuda")]
pub(crate) mod cuda {
    use anyhow::Result;
    use cudarc::driver::sys::CUdevice_attribute_enum;
    use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock};
    use std::time::{Duration, Instant};

    /// CUDA sleep kernel source code.
    pub const SLEEP_KERNEL_SRC: &str = r#"
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
}
