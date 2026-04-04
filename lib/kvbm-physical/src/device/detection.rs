// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hardware detection and auto-configuration.

use super::DeviceBackend;
use anyhow::{Result, bail};
use std::str::FromStr;

impl DeviceBackend {
    /// Auto-detect the best available device backend.
    ///
    /// Priority order: CUDA → Level-Zero (XPU)
    pub fn auto_detect() -> Result<Self> {
        // 1. Check environment variable override
        if let Ok(backend_str) = std::env::var("KVBM_DEVICE_BACKEND") {
            return Self::from_str(&backend_str);
        }

        // 2. Probe hardware in priority order
        if Self::Cuda.is_available() {
            tracing::info!("Auto-detected CUDA backend");
            return Ok(Self::Cuda);
        }

        #[cfg(feature = "xpu")]
        if Self::Ze.is_available() {
            tracing::info!("Auto-detected Level-Zero (XPU) backend");
            return Ok(Self::Ze);
        }

        bail!("No supported device backend available on this system")
    }

    /// Get list of all available backends on current system.
    pub fn list_available() -> Vec<Self> {
        let mut backends = Vec::new();

        if Self::Cuda.is_available() {
            backends.push(Self::Cuda);
        }

        #[cfg(feature = "xpu")]
        if Self::Ze.is_available() {
            backends.push(Self::Ze);
        }

        backends
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detect() {
        match DeviceBackend::auto_detect() {
            Ok(backend) => {
                println!("Auto-detected: {:?}", backend);
                assert!(backend.is_available());
            }
            Err(e) => {
                println!("No backend available: {}", e);
            }
        }
    }

    #[test]
    fn test_list_available() {
        let backends = DeviceBackend::list_available();
        println!("Available backends: {:?}", backends);
    }
}
