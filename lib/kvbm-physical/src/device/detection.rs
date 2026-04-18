// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hardware detection and auto-configuration.

use super::DeviceBackend;
use anyhow::{Result, bail};

impl DeviceBackend {
    /// Auto-detect the best available device backend.
    ///
    /// Priority order: CUDA then Level-Zero (XPU).
    pub fn detect_backend() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if Self::Cuda.is_available() {
                tracing::info!("Auto-detected CUDA backend");
                return Ok(Self::Cuda);
            }
        }

        #[cfg(feature = "xpu-sycl")]
        {
            if Self::Sycl.is_available() {
                tracing::info!("Auto-detected SYCL (XPU) backend");
                return Ok(Self::Sycl);
            }
        }

        #[cfg(feature = "xpu-ze")]
        {
            if Self::Ze.is_available() {
                tracing::info!("Auto-detected Level-Zero (XPU) backend");
                return Ok(Self::Ze);
            }
        }

        bail!("No supported device backend available on this system")
    }

    /// Get list of all available backends on current system.
    pub fn list_available() -> Vec<Self> {
        let mut backends = Vec::new();

        #[cfg(feature = "cuda")]
        if Self::Cuda.is_available() {
            backends.push(Self::Cuda);
        }

        #[cfg(feature = "xpu-sycl")]
        if Self::Sycl.is_available() {
            backends.push(Self::Sycl);
        }

        #[cfg(feature = "xpu-ze")]
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
    fn test_detect_backend() {
        match DeviceBackend::detect_backend() {
            Ok(backend) => {
                println!("Detected: {:?}", backend);
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
