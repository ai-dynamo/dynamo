// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Relays find the core through the velo peer info it writes to a shared file.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use velo::PeerInfo;

/// Wait for the core's peer file.
pub async fn read_peer_file(path: &Path, timeout: Duration) -> anyhow::Result<PeerInfo> {
    let deadline = Instant::now() + timeout;
    loop {
        match std::fs::read(path) {
            Ok(bytes) => {
                return rmp_serde::from_slice(&bytes).with_context(|| {
                    format!("peer file {} is not velo peer info", path.display())
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                anyhow::ensure!(
                    Instant::now() < deadline,
                    "no peer file at {} after {timeout:?}",
                    path.display()
                );
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
}
