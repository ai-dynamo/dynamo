// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profiling configuration and low-overhead annotation helpers.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use validator::Validate;

static NVTX_ENABLED: AtomicBool = AtomicBool::new(false);

/// Canonical KVBM NVTX range names.
///
/// Keep these names stable so Nsight Systems traces are comparable across runs.
pub mod ranges {
    pub const TRANSFER_EXECUTE: &str = "kvbm.transfer.execute";

    pub const NIXL_EXECUTE: &str = "kvbm.nixl.execute";
    pub const NIXL_BUILD_DESC: &str = "kvbm.nixl.build_desc";
    pub const NIXL_CREATE_XFER_REQ: &str = "kvbm.nixl.create_xfer_req";
    pub const NIXL_POST_XFER_REQ: &str = "kvbm.nixl.post_xfer_req";

    pub const OBJECT_S3_PUT_OBJECT: &str = "kvbm.object.s3.put_object";
    pub const OBJECT_PUT_BLOCKS_SUBMIT: &str = "kvbm.object.put_blocks.submit";
    pub const OBJECT_PUT_BLOCKS: &str = "kvbm.object.put_blocks";
    pub const OBJECT_PUT_COPY_BLOCK_TO_BYTES: &str = "kvbm.object.put.copy_block_to_bytes";
    pub const OBJECT_PUT_S3_PUT_OBJECT: &str = "kvbm.object.put.s3_put_object";
    pub const OBJECT_GET_BLOCKS_SUBMIT: &str = "kvbm.object.get_blocks.submit";
    pub const OBJECT_GET_BLOCKS: &str = "kvbm.object.get_blocks";
    pub const OBJECT_GET_S3_GET_OBJECT: &str = "kvbm.object.get.s3_get_object";
    pub const OBJECT_GET_COPY_BYTES_TO_BLOCK: &str = "kvbm.object.get.copy_bytes_to_block";

    pub const WORKER_LOCAL_TRANSFER_AWAIT: &str = "kvbm.worker.local_transfer.await";
    pub const WORKER_REMOTE_ONBOARD_AWAIT: &str = "kvbm.worker.remote_onboard.await";
    pub const WORKER_REMOTE_OFFLOAD_AWAIT: &str = "kvbm.worker.remote_offload.await";
    pub const WORKER_REMOTE_ONBOARD_FOR_INSTANCE_AWAIT: &str =
        "kvbm.worker.remote_onboard_for_instance.await";
    pub const WORKER_OBJECT_HAS_BLOCKS_AWAIT: &str = "kvbm.worker.object_has_blocks.await";
    pub const WORKER_OBJECT_PUT_BLOCKS_AWAIT: &str = "kvbm.worker.object_put_blocks.await";
    pub const WORKER_OBJECT_GET_BLOCKS_AWAIT: &str = "kvbm.worker.object_get_blocks.await";

    pub const OFFLOAD_POLICY: &str = "offload::policy";
    pub const OFFLOAD_PRECONDITION: &str = "offload::precondition";
    pub const OFFLOAD_BATCH: &str = "offload::batch";
    pub const OFFLOAD_TRANSFER: &str = "offload::transfer";
}

/// Profiling configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct ProfilingConfig {
    /// Enable KVBM NVTX annotations for Nsight Systems.
    ///
    /// This requires the `nvtx` Cargo feature. When the feature is disabled,
    /// annotations compile to no-ops regardless of this value.
    #[serde(default)]
    pub nvtx_enabled: bool,
}

impl ProfilingConfig {
    /// Apply this profiling config to the process-local profiling switches.
    pub fn apply(&self) {
        NVTX_ENABLED.store(self.nvtx_enabled, Ordering::Relaxed);
    }
}

/// Returns true when KVBM NVTX annotations are enabled at runtime.
#[inline(always)]
pub fn nvtx_enabled() -> bool {
    NVTX_ENABLED.load(Ordering::Relaxed)
}

/// Process-wide NVTX range.
///
/// Use this for async work that may cross `.await` points. It maps to
/// `nvtx::Range`, not thread-local push/pop ranges, so it remains valid if
/// a Tokio future resumes on a different worker thread.
#[cfg(feature = "nvtx")]
pub struct NvtxRange {
    range: Option<nvtx::RangeGuard>,
}

#[cfg(feature = "nvtx")]
impl NvtxRange {
    #[inline(always)]
    pub fn new(name: impl Into<String>) -> Self {
        if nvtx_enabled() {
            let name = name.into();
            Self {
                range: Some(nvtx::range!("{}", name)),
            }
        } else {
            Self { range: None }
        }
    }
}

#[cfg(feature = "nvtx")]
impl Drop for NvtxRange {
    fn drop(&mut self) {
        let _ = self.range.take();
    }
}

/// No-op range used when the `nvtx` Cargo feature is disabled.
#[cfg(not(feature = "nvtx"))]
pub struct NvtxRange;

#[cfg(not(feature = "nvtx"))]
impl NvtxRange {
    #[inline(always)]
    pub fn new(_name: impl Into<String>) -> Self {
        Self
    }
}
