// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::{Tokens, compute_hash_v2};
use serde::Serialize;

/// Minimal representation of a scheduler slot request.
#[derive(Clone, Debug)]
pub struct Request {
    pub request_id: String,
    pub tokens: Tokens,
    pub lora_name: Option<String>,
    pub salt_hash: u64,
    pub max_tokens: Option<usize>,
}

impl Request {
    pub fn new(
        request_id: impl Into<String>,
        tokens: impl Into<Tokens>,
        lora_name: Option<String>,
        salt: Option<String>,
        max_tokens: Option<usize>,
    ) -> Self {
        // Pack any data that needs to be included in the salt hash into [`SaltPayload`]
        #[derive(Serialize)]
        struct SaltPayload<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            salt: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            lora_name: Option<&'a str>,
        }

        let request_id = request_id.into();
        let payload = SaltPayload {
            salt: salt.as_deref(),
            lora_name: lora_name.as_deref(),
        };
        let salt_bytes = serde_json::to_vec(&payload).expect("failed to serialize salt payload");
        let salt_hash = compute_hash_v2(&salt_bytes, 0);

        Self {
            request_id,
            tokens: tokens.into(),
            lora_name,
            salt_hash,
            max_tokens,
        }
    }
}
