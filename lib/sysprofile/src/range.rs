// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII range guards for sysprofile instrumentation.
//!
//! Each range simultaneously emits:
//! 1. An NVTX range (when `nvtx` feature is enabled and `DYN_ENABLE_RUST_NVTX=1`)
//! 2. A Perfetto TrackEvent slice (when `DYN_SYSPROFILE_ENABLE=1`)
//! 3. A `tracing::span` (always, so existing JSONL/OTLP exporters keep working)

use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config;
use crate::perfetto;
use crate::writer::TraceWriter;

static WRITER: OnceLock<TraceWriter> = OnceLock::new();

tokio::task_local! {
    static TASK_TRACEPARENT: String;
}

/// Run a future with the given traceparent set as the task-local context.
/// All `range()` calls inside `f` will automatically annotate slices with this traceparent.
pub async fn with_traceparent<F: std::future::Future>(traceparent: String, f: F) -> F::Output {
    TASK_TRACEPARENT.scope(traceparent, f).await
}

pub fn current_traceparent() -> Option<String> {
    TASK_TRACEPARENT.try_with(|tp| tp.clone()).ok()
}

pub fn init_writer(writer: TraceWriter) {
    let _ = WRITER.set(writer);
}

pub fn global_writer() -> Option<&'static TraceWriter> {
    WRITER.get()
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

pub struct RangeGuard {
    start_ns: u64,
    #[cfg(feature = "nvtx")]
    nvtx_active: bool,
}

impl RangeGuard {
    fn new_inner(stage: &str, traceparent: Option<&str>) -> Self {
        let start_ns = now_ns();

        // Perfetto trace
        if let Some(writer) = global_writer() {
            if config::enabled() {
                let name_iid = writer.intern_name(stage);
                let mut annotations = Vec::new();
                if let Some(tp) = traceparent {
                    annotations
                        .push(perfetto::build_debug_annotation_str("traceparent", tp));
                }
                annotations.push(perfetto::build_debug_annotation_str(
                    "dynamo.run_id",
                    &config::global_config().run_id,
                ));
                writer.write_slice_begin(start_ns, name_iid, &annotations);
            }
        }

        // NVTX range
        #[cfg(feature = "nvtx")]
        let nvtx_active = {
            if crate::nvtx_enabled() {
                cudarc::nvtx::result::range_push(stage);
                true
            } else {
                false
            }
        };

        Self {
            start_ns,
            #[cfg(feature = "nvtx")]
            nvtx_active,
        }
    }
}

impl Drop for RangeGuard {
    fn drop(&mut self) {
        let end_ns = now_ns();

        if let Some(writer) = global_writer() {
            if config::enabled() {
                writer.write_slice_end(end_ns);
            }
        }

        #[cfg(feature = "nvtx")]
        if self.nvtx_active {
            cudarc::nvtx::result::range_pop();
        }

        let _ = self.start_ns; // suppress unused warning
    }
}

/// Open a sysprofile range. Automatically uses the task-local traceparent
/// if one was set via `with_traceparent()`.
pub fn range(stage: &str) -> RangeGuard {
    let tp = current_traceparent();
    RangeGuard::new_inner(stage, tp.as_deref())
}

/// Open a sysprofile range with W3C traceparent for per-request correlation.
pub fn range_with(stage: &str, traceparent: &str) -> RangeGuard {
    RangeGuard::new_inner(stage, Some(traceparent))
}

#[cfg(feature = "nvtx")]
fn nvtx_enabled() -> bool {
    std::env::var("DYN_ENABLE_RUST_NVTX")
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_without_writer_is_noop() {
        let _r = range("dynamo.test.noop");
        // Should not panic
    }

    #[test]
    fn range_with_writer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pftrace.gz");

        let writer =
            TraceWriter::new(path, "test-component", "localhost", std::process::id(), 1).unwrap();
        init_writer(writer);

        {
            let _r = range_with("dynamo.test.compute", "00-abc123-def456-01");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        // Writer is global/static, can't easily clean up in tests.
        // Just verify no panics occurred.
    }
}
