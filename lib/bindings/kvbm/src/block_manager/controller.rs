// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub use dynamo_llm::block_manager::controller::client::ControlClient;
pub use dynamo_llm::block_manager::controller::{CacheLevel, Controller};

#[pyclass]
pub struct BlockManagerClient {
    inner: ControlClient,
}

#[pymethods]
impl BlockManagerClient {
    #[new]
    fn new(component: Component, instance_id: i64) -> PyResult<Self> {
        let client = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(ControlClient::new(component.inner, instance_id))
            .map_err(to_pyerr)?;
        Ok(BlockManagerClient { inner: client })
    }

    fn reset_pool(&self, cache_level: String) -> PyResult<()> {
        let cache_level = Self::cache_level_from_str(&cache_level).map_err(to_pyerr)?;
        pyo3_async_runtimes::tokio::get_runtime()
            .block_on(self.inner.reset_pool(cache_level))
            .map_err(to_pyerr)
    }

    fn reset_blocks(&self, cache_level: String, blocks: Vec<u64>) -> PyResult<ResetBlocksResponse> {
        let cache_level = Self::cache_level_from_str(&cache_level).map_err(to_pyerr)?;
        let response = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(self.inner.reset_blocks(cache_level, blocks))
            .map_err(to_pyerr)?;
        Ok(ResetBlocksResponse { inner: response })
    }

    fn status(&self, cache_level: String) -> PyResult<BlockPoolStatus> {
        let cache_level = Self::cache_level_from_str(&cache_level).map_err(to_pyerr)?;
        let status = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(self.inner.status(cache_level))
            .map_err(to_pyerr)?;
        Ok(BlockPoolStatus { inner: status })
    }

    fn reset_all_pools(&self) -> PyResult<()> {
        pyo3_async_runtimes::tokio::get_runtime()
            .block_on(self.inner.reset_all_pools())
            .map_err(to_pyerr)
    }

    /// Clear (wipe) all KV cache entries from a specific pool.
    ///
    /// `pool` must be one of: "gpu" / "device" (G1), "cpu" / "host" (G2), or "disk" (G3).
    ///
    /// Requires KVBM_DEV_MODE=TRUE. Raises an exception if dev-mode is not enabled
    /// or if the pool name is invalid.
    fn clear_pool(&self, pool: String) -> PyResult<()> {
        let is_dev = std::env::var(dynamo_runtime::config::environment_names::kvbm::KVBM_DEV_MODE)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        if !is_dev {
            return Err(to_pyerr(
                "clear_pool called but KVBM_DEV_MODE is not enabled. \
                 Set KVBM_DEV_MODE=TRUE to allow destructive pool operations.",
            ));
        }

        let cache_level = match pool.to_lowercase().as_str() {
            "gpu" | "device" => CacheLevel::G1,
            "cpu" | "host" => CacheLevel::G2,
            "disk" => CacheLevel::G3,
            other => {
                return Err(to_pyerr(format!(
                    "unknown pool '{other}': expected one of 'gpu', 'device', 'cpu', 'host', 'disk'"
                )));
            }
        };

        tracing::warn!("clear_pool({pool}): wiping pool via controller (dev-mode)");

        pyo3_async_runtimes::tokio::get_runtime()
            .block_on(self.inner.reset_pool(cache_level))
            .map_err(to_pyerr)?;

        tracing::info!("clear_pool({pool}): pool wiped successfully via controller");
        Ok(())
    }
}

impl BlockManagerClient {
    // convert string to cache level
    fn cache_level_from_str(cache_level: &str) -> anyhow::Result<CacheLevel> {
        match cache_level.to_uppercase().as_str() {
            "G1" => Ok(CacheLevel::G1),
            "G2" => Ok(CacheLevel::G2),
            "G3" => Ok(CacheLevel::G3),
            _ => anyhow::bail!("Invalid cache level: allowed values are G1, G2, G3"),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct BlockPoolStatus {
    inner: dynamo_llm::block_manager::pool::BlockPoolStatus,
}

#[pymethods]
impl BlockPoolStatus {
    fn active_blocks(&self) -> usize {
        self.inner.active_blocks
    }

    fn inactive_blocks(&self) -> usize {
        self.inner.inactive_blocks
    }

    fn empty_blocks(&self) -> usize {
        self.inner.empty_blocks
    }
}

#[pyclass]
pub struct ResetBlocksResponse {
    inner: dynamo_llm::block_manager::pool::ResetBlocksResponse,
}

#[pymethods]
impl ResetBlocksResponse {
    fn reset_blocks(&self) -> Vec<u64> {
        self.inner.reset_blocks.clone()
    }

    fn not_found_blocks(&self) -> Vec<u64> {
        self.inner.not_found.clone()
    }

    fn not_reset_blocks(&self) -> Vec<u64> {
        self.inner.not_reset.clone()
    }
}
