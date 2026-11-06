// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sha2::{Digest, Sha256};

pub const RUNTIME_LORA_KEY_PREFIX: &str = "dyn-lora-";
// Adapter identity is versioned independently of the worker wire protocol.
const IDENTITY_DOMAIN: &[u8] = b"dynamo-runtime-lora-v1\0";

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RuntimeLoraError {
    #[error("{0}")]
    InvalidConfiguration(String),
    #[error("runtime LoRA model id is invalid")]
    InvalidModelId,
    #[error("runtime LoRA base model must be specified when it cannot be inferred")]
    BaseModelRequired,
    #[error("runtime LoRA base model was not found")]
    BaseModelNotFound,
    #[error("runtime LoRA is unsupported for base model '{0}'")]
    BaseModelUnsupported(String),
}

pub fn identity_digest(base_model_name: &str, source_uri: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(IDENTITY_DOMAIN);
    hasher.update(base_model_name.as_bytes());
    hasher.update(b"\0");
    hasher.update(source_uri.as_bytes());
    hasher.finalize().into()
}

pub fn adapter_key(base_model_name: &str, source_uri: &str) -> String {
    let digest = identity_digest(base_model_name, source_uri);
    let mut key = String::with_capacity(RUNTIME_LORA_KEY_PREFIX.len() + 32);
    key.push_str(RUNTIME_LORA_KEY_PREFIX);
    for byte in &digest[..16] {
        use std::fmt::Write;
        write!(key, "{byte:02x}").expect("writing to String cannot fail");
    }
    key
}

#[cfg(test)]
mod tests {
    use super::adapter_key;

    #[test]
    fn adapter_key_matches_v1_golden_vectors() {
        assert_eq!(
            adapter_key(
                "meta-llama/Llama-3.1-8B-Instruct",
                "wandb-artifact:///team/project/adapter:v7",
            ),
            "dyn-lora-c7246b154263a336c006517e4bc6d7c8"
        );
        assert_eq!(
            adapter_key("base", "custom://adapter@sha256:abc"),
            "dyn-lora-e3cfcd02f2e62cb70f45999324e7d4de"
        );
        assert_eq!(
            adapter_key("base", "wandb-artifact:///entity/project/artifact:v1"),
            "dyn-lora-27056eb300024fbfc91334895bba1e26"
        );
        assert_eq!(
            adapter_key("base", "wandb-artifact:///a|b:v1"),
            "dyn-lora-74bbe88c562d9d177d098e3ba118851a"
        );
    }
}
