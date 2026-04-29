// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine profiler adapter trait.
//!
//! Each engine backend (vLLM, SGLang, TRT-LLM, PyTorch) implements
//! [`EngineProfilerAdapter`] to expose backend-specific profiling hooks
//! to the sysprofile capture agent.
//!
//! The adapter is responsible for:
//! - Starting/stopping backend-specific profiling (torch profiler, NVTX, etc.)
//! - Collecting engine-level metrics (batch sizes, KV cache occupancy)
//! - Providing per-rank GPU utilization data for TP imbalance detection

use crate::config::SysprofileConfig;

/// Backend-specific profiling hooks for engine components.
///
/// Implementations live outside this crate (in the engine Python bindings
/// or Rust engine wrappers). The sysprofile agent calls these methods
/// during capture lifecycle.
pub trait EngineProfilerAdapter: Send + Sync {
    fn backend_name(&self) -> &str;

    /// Start collecting profiling data for a capture run.
    ///
    /// Called once when a `StartRequest` is received. The adapter should
    /// enable any backend-specific profilers (torch profiler, NVTX push/pop,
    /// TRT-LLM profiler API, etc.) and begin accumulating data.
    fn start_capture(&self, config: &CaptureConfig) -> anyhow::Result<()>;

    /// Stop collecting and flush any buffered profiling data.
    ///
    /// Called when the capture window ends (duration or `StopRequest`).
    /// The adapter should write any remaining data to `output_dir` and
    /// disable backend profilers.
    fn stop_capture(&self) -> anyhow::Result<CaptureResult>;

    /// Return the TP world size for this engine instance.
    /// Used for View B straggler ratio computation.
    fn tp_world_size(&self) -> u32 {
        1
    }

    /// Return the current TP rank for this engine instance.
    fn tp_rank(&self) -> u32 {
        0
    }

    /// Return per-iteration GPU busy times (nanoseconds) for the capture
    /// window, one entry per iteration. Used to compute straggler ratio.
    ///
    /// Default returns empty (no per-rank data available).
    fn gpu_busy_times_ns(&self) -> Vec<u64> {
        vec![]
    }
}

/// Configuration passed to an engine adapter when starting a capture.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    pub run_id: String,
    pub output_dir: std::path::PathBuf,
    pub duration_s: u64,
    pub sampling: f64,
    pub sysprofile_config: SysprofileConfig,
    pub enable_nvtx: bool,
    pub enable_torch_profiler: bool,
}

/// Result returned by an engine adapter after stopping a capture.
#[derive(Debug, Clone)]
pub struct CaptureResult {
    pub files_written: Vec<std::path::PathBuf>,
    pub total_bytes: u64,
    pub iterations_captured: u64,
}

/// No-op adapter for components that don't have engine-specific profiling.
///
/// Used by frontend, router, and other non-engine components that only
/// produce sysprofile NVTX ranges (via `range()` / `range_with()`).
pub struct NoopAdapter {
    name: String,
}

impl NoopAdapter {
    pub fn new(component_name: &str) -> Self {
        Self {
            name: component_name.to_string(),
        }
    }
}

impl EngineProfilerAdapter for NoopAdapter {
    fn backend_name(&self) -> &str {
        &self.name
    }

    fn start_capture(&self, _config: &CaptureConfig) -> anyhow::Result<()> {
        Ok(())
    }

    fn stop_capture(&self) -> anyhow::Result<CaptureResult> {
        Ok(CaptureResult {
            files_written: vec![],
            total_bytes: 0,
            iterations_captured: 0,
        })
    }
}

/// Adapter for vLLM engine: uses torch.profiler under the hood.
///
/// Stub — actual implementation lives in the Python engine bindings
/// and communicates via the `dynamo.sysprofile.engine.*` NATS subjects.
pub struct VllmAdapter {
    pub tp_size: u32,
    pub tp_rank: u32,
}

impl EngineProfilerAdapter for VllmAdapter {
    fn backend_name(&self) -> &str {
        "vllm"
    }

    fn start_capture(&self, _config: &CaptureConfig) -> anyhow::Result<()> {
        tracing::info!(backend = "vllm", "vLLM adapter: start_capture is a no-op in Rust; torch.profiler is started via Python bindings");
        Ok(())
    }

    fn stop_capture(&self) -> anyhow::Result<CaptureResult> {
        Ok(CaptureResult {
            files_written: vec![],
            total_bytes: 0,
            iterations_captured: 0,
        })
    }

    fn tp_world_size(&self) -> u32 {
        self.tp_size
    }

    fn tp_rank(&self) -> u32 {
        self.tp_rank
    }
}

/// Adapter for TRT-LLM engine: uses NVTX ranges directly.
pub struct TrtLlmAdapter {
    pub tp_size: u32,
    pub tp_rank: u32,
}

impl EngineProfilerAdapter for TrtLlmAdapter {
    fn backend_name(&self) -> &str {
        "trt-llm"
    }

    fn start_capture(&self, _config: &CaptureConfig) -> anyhow::Result<()> {
        tracing::info!(backend = "trt-llm", "TRT-LLM adapter: NVTX ranges are always active; start_capture enables trace collection");
        Ok(())
    }

    fn stop_capture(&self) -> anyhow::Result<CaptureResult> {
        Ok(CaptureResult {
            files_written: vec![],
            total_bytes: 0,
            iterations_captured: 0,
        })
    }

    fn tp_world_size(&self) -> u32 {
        self.tp_size
    }

    fn tp_rank(&self) -> u32 {
        self.tp_rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_adapter_lifecycle() {
        let adapter = NoopAdapter::new("frontend");
        assert_eq!(adapter.backend_name(), "frontend");
        assert_eq!(adapter.tp_world_size(), 1);
        assert_eq!(adapter.tp_rank(), 0);
        assert!(adapter.gpu_busy_times_ns().is_empty());

        let config = CaptureConfig {
            run_id: "test".into(),
            output_dir: std::path::PathBuf::from("/tmp"),
            duration_s: 10,
            sampling: 0.1,
            sysprofile_config: SysprofileConfig::default(),
            enable_nvtx: false,
            enable_torch_profiler: false,
        };

        adapter.start_capture(&config).unwrap();
        let result = adapter.stop_capture().unwrap();
        assert!(result.files_written.is_empty());
    }

    #[test]
    fn vllm_adapter_has_tp_info() {
        let adapter = VllmAdapter {
            tp_size: 4,
            tp_rank: 2,
        };
        assert_eq!(adapter.backend_name(), "vllm");
        assert_eq!(adapter.tp_world_size(), 4);
        assert_eq!(adapter.tp_rank(), 2);
    }

    #[test]
    fn trait_is_object_safe() {
        let adapter: Box<dyn EngineProfilerAdapter> = Box::new(NoopAdapter::new("test"));
        assert_eq!(adapter.backend_name(), "test");
    }
}
