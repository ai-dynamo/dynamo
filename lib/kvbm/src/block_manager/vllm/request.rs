// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde::{Deserialize, Serialize};

use dynamo_llm::tokens::compute_hash_v2;

/// Request Inputs
#[pyclass]
#[derive(Debug, Clone, Dissolve, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct KvbmRequest {
    pub request_id: String,
    pub lora_name: Option<String>,
    pub salt_hash: u64,
}

#[pymethods]
impl KvbmRequest {
    #[new]
    #[pyo3(signature = (request_id, lora_name=None, salt_hash=None))]
    pub fn new(request_id: String, lora_name: Option<String>, salt_hash: Option<String>) -> Self {
        // compute salt
        #[derive(Debug, serde::Serialize)]
        struct Salt {
            #[serde(skip_serializing_if = "Option::is_none")]
            salt: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            lora_name: Option<String>,
        }

        let salt = Salt {
            salt: salt_hash,
            lora_name: lora_name.clone(),
        };

        tracing::trace!("salt: {:?}", salt);

        let salt_bytes = serde_json::to_vec(&salt).unwrap();
        let salt_hash = compute_hash_v2(&salt_bytes, 0);

        tracing::trace!("salt_hash: {:?}", salt_hash);

        Self {
            request_id,
            lora_name,
            salt_hash,
        }
    }
}
