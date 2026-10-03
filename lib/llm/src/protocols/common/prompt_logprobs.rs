// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Signed public counts versus the existing unsigned cross-process contract.
//! Never infer sentinel support from the ability to deserialize a u32: Dynamo
//! 1.4/1.5 vLLM workers assign that integer directly to SamplingParams.

use serde::{Deserialize, Serialize};

use crate::local_model::runtime_config::ModelRuntimeConfig;

use super::invalid_argument_error;

pub const FULL_VOCAB_COUNT: u32 = u32::MAX;
pub const VLLM_PROMPT_LOGPROBS_CAPABILITY: &str = "vllm_prompt_logprobs";

/// Additive, worker-published facts from its effective engine configuration.
#[derive(Debug, Deserialize, Serialize)]
pub struct VllmPromptLogprobsCapability {
    pub schema_version: u32,
    pub wire_count: String,
    pub max_logprobs: i64,
    pub vocab_size: u32,
}

pub(crate) fn public_count_to_wire(count: Option<i64>) -> anyhow::Result<Option<u32>> {
    match count {
        None => Ok(None),
        Some(-1) => Ok(Some(FULL_VOCAB_COUNT)),
        Some(n) if (0..i64::from(FULL_VOCAB_COUNT)).contains(&n) => Ok(Some(n as u32)),
        Some(_) => Err(invalid_argument_error(
            "`prompt_logprobs` must be -1 or a non-negative count below 4294967295",
        )),
    }
}

/// Validate the actual selected hop, not another member's capability. Ordinary
/// finite counts retain their existing N-2 path. The old Generate marker cannot
/// authorize this sentinel: its first released reader did not understand it.
pub(crate) fn validate_full_vocab_count(
    count: Option<u32>,
    runtime: Option<&ModelRuntimeConfig>,
) -> anyhow::Result<()> {
    if count != Some(FULL_VOCAB_COUNT) {
        return Ok(());
    }
    let capability = runtime
        .map(|runtime| {
            runtime.get_engine_specific::<VllmPromptLogprobsCapability>(
                VLLM_PROMPT_LOGPROBS_CAPABILITY,
            )
        })
        .transpose()
        .map_err(|_| rejected("malformed worker capability"))?
        .flatten()
        .ok_or_else(|| {
            rejected("selected worker has not advertised full-vocabulary wire support")
        })?;
    if capability.schema_version != 1
        || capability.wire_count != "u32_max"
        || capability.vocab_size == 0
        || capability.vocab_size == FULL_VOCAB_COUNT
        || capability.max_logprobs < -1
    {
        return Err(rejected("malformed or unsupported worker capability"));
    }
    if capability.max_logprobs != -1 && capability.max_logprobs < i64::from(capability.vocab_size) {
        return Err(rejected(
            "requested full vocabulary exceeds the worker's max_logprobs limit",
        ));
    }
    Ok(())
}

fn rejected(reason: &str) -> anyhow::Error {
    invalid_argument_error(format!(
        "`prompt_logprobs=-1` rejected: {reason}. Use a compatible worker configured \
         to allow full-vocabulary prompt logprobs, or request a finite count."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn public_signed_count_preserves_the_existing_wire_without_wrapping() {
        for (public, wire) in [
            (None, None),
            (Some(-1), Some(FULL_VOCAB_COUNT)),
            (Some(0), Some(0)),
            (Some(1), Some(1)),
        ] {
            assert_eq!(public_count_to_wire(public).unwrap(), wire);
        }
        for count in [-2, i64::MIN, i64::MAX, i64::from(u32::MAX)] {
            assert!(public_count_to_wire(Some(count)).is_err());
        }
        assert_eq!(
            public_count_to_wire(Some(i64::from(u32::MAX) - 1)).unwrap(),
            Some(u32::MAX - 1)
        );
    }

    #[test]
    fn full_vocab_requires_exact_wire_support_and_sufficient_engine_limit() {
        for (capability, allowed) in [
            (
                json!({"schema_version":1,"wire_count":"u32_max","max_logprobs":-1,"vocab_size":32}),
                true,
            ),
            (
                json!({"schema_version":1,"wire_count":"u32_max","max_logprobs":32,"vocab_size":32}),
                true,
            ),
            (
                json!({"schema_version":1,"wire_count":"u32_max","max_logprobs":20,"vocab_size":32}),
                false,
            ),
            (
                json!({"schema_version":1,"wire_count":"u32_max","max_logprobs":-2,"vocab_size":32}),
                false,
            ),
            (
                json!({"schema_version":1,"wire_count":"u32_max","max_logprobs":-1,"vocab_size":0}),
                false,
            ),
            (
                json!({"schema_version":1,"wire_count":"signed","max_logprobs":-1,"vocab_size":32}),
                false,
            ),
            (
                json!({"schema_version":2,"wire_count":"u32_max","max_logprobs":-1,"vocab_size":32}),
                false,
            ),
            (json!({"schema_version":true}), false),
            (json!(null), false),
        ] {
            let mut runtime = ModelRuntimeConfig::default();
            runtime
                .set_engine_specific(VLLM_PROMPT_LOGPROBS_CAPABILITY, capability)
                .unwrap();
            assert_eq!(
                validate_full_vocab_count(Some(FULL_VOCAB_COUNT), Some(&runtime)).is_ok(),
                allowed
            );
            validate_full_vocab_count(Some(1), Some(&runtime)).unwrap();
        }
        validate_full_vocab_count(None, None).unwrap();
        validate_full_vocab_count(Some(0), None).unwrap();
        assert!(validate_full_vocab_count(Some(FULL_VOCAB_COUNT), None).is_err());
        assert!(
            validate_full_vocab_count(Some(FULL_VOCAB_COUNT), Some(&ModelRuntimeConfig::default()))
                .is_err()
        );
    }
}
