// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_runtime::traits::DistributedRuntimeProvider;
use pyo3::prelude::*;

use super::*;

#[pyclass]
pub(crate) struct KvCacheSolEstimator {
    inner: llm_rs::kv_cache_sol::KvCacheSolEstimator,
}

#[pymethods]
impl KvCacheSolEstimator {
    #[new]
    #[pyo3(signature = (endpoint, horizon_secs=3600, max_cache_blocks=5_000_000, max_pending_requests=100_000))]
    fn new(
        endpoint: Endpoint,
        horizon_secs: u64,
        max_cache_blocks: usize,
        max_pending_requests: usize,
    ) -> PyResult<Self> {
        if horizon_secs == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "horizon_secs must be greater than zero",
            ));
        }
        if max_cache_blocks == 0 || max_pending_requests == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_cache_blocks and max_pending_requests must be greater than zero",
            ));
        }
        let component = endpoint.inner.component().clone();
        let runtime = component.drt().runtime().secondary();
        let inner = runtime
            .block_on(llm_rs::kv_cache_sol::KvCacheSolEstimator::start(
                component,
                Duration::from_secs(horizon_secs),
                max_cache_blocks,
                max_pending_requests,
            ))
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn shutdown(&self) {
        self.inner.shutdown();
    }
}
