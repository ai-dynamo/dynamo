// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shutdown plumbing for the gRPC + HTTP servers.
//!
//! A single `CancellationToken` fans out to both serve loops. The UDS path is
//! tracked here so `Drop` on [`UdsGuard`] can unlink the socket file even on
//! ungraceful shutdown — without it a re-launch would fail with "address in
//! use" on the stale socket.

use std::path::{Path, PathBuf};

/// RAII wrapper that unlinks the UDS path on drop. Cheap, holds a clone of
/// the path; safe to drop after the listener has already been shut down.
pub struct UdsGuard {
    path: PathBuf,
}

impl UdsGuard {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for UdsGuard {
    fn drop(&mut self) {
        // Best-effort. If the file is already gone we don't care.
        let _ = std::fs::remove_file(&self.path);
    }
}
