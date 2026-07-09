// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::tokenizer_cache::TokenizerCacheConfig as RsTokenizerCacheConfig;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction]
pub fn normalize_router_valkey_config(raw: &str) -> PyResult<String> {
    dynamo_llm::router_valkey_config::RouterValkeyConfig::normalized_json(raw)
        .map_err(|error| PyValueError::new_err(format!("{error:#}")))
}

#[pyclass]
#[derive(Default, Clone, Debug)]
pub struct TokenizerCacheConfig {
    pub(super) inner: RsTokenizerCacheConfig,
}

#[pymethods]
impl TokenizerCacheConfig {
    #[new]
    #[pyo3(signature = (router_valkey_config=None))]
    fn new(router_valkey_config: Option<&str>) -> PyResult<Self> {
        let Some(raw) = router_valkey_config else {
            return Ok(Self::default());
        };
        let contract = dynamo_llm::router_valkey_config::RouterValkeyConfig::parse(raw)
            .map_err(|error| PyValueError::new_err(format!("{error:#}")))?;
        let inner = contract
            .tokenizer_cache_config()
            .map_err(|error| PyValueError::new_err(format!("{error:#}")))?;
        Ok(Self { inner })
    }
}
