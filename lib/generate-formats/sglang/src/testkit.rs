// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testing SGLang-backed `/generate` formats without GPUs (feature `testkit`).
//!
//! - [`sim`] models SGLang's incremental stream and answers the body a format
//!   builds, including the retry bodies of migrated requests.
//! - [`driver`] plays the frontend, prefill router, migration, and worker
//!   around a format, and fails attempts on cue.
//! - [`laws`] are property tests a format must pass: streamed chunks join into
//!   the unary response, P/D matches aggregated serving, migration is
//!   invisible, blocked requests fail cleanly, malformed responses never panic.
//!
//! A format that reshapes the client side implements [`ClientShape`] to write
//! a [`RequestSpec`] in its own body, then runs [`laws::Harness::check_all`].

use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::response_merge::RuleSet;
use dynamo_llm::generate_format::Worker;

/// A worker set advertising `engine_info`.
pub fn worker(engine_info: &Value) -> Worker<'_> {
    Worker {
        model: "test-model",
        engine_info,
        kv_cache_block_size: 16,
        lora_name: None,
        tower_connector_lora_enabled: false,
    }
}

pub mod driver;
pub mod laws;
pub mod sim;

/// One request, in SGLang's terms.
#[derive(Debug, Clone, Default)]
pub struct RequestSpec {
    pub input_ids: Vec<u32>,
    pub stream: bool,
    pub max_new_tokens: Option<u32>,
    pub min_new_tokens: Option<u32>,
    pub return_logprob: bool,
    pub logprob_start_len: Option<i64>,
    pub top_logprobs_num: u32,
    pub return_routed_experts: bool,
    /// Further top-level and `sampling_params` fields.
    pub extra: Map<String, Value>,
    pub extra_sampling: Map<String, Value>,
}

/// How a format's clients write requests and join streamed responses.
pub trait ClientShape: Send + Sync {
    /// The client body for `spec`, or `None` when the format cannot express it.
    fn body(&self, spec: &RequestSpec) -> Option<Vec<u8>>;

    /// Rules joining streamed client chunks into the unary response; the
    /// stream-join law is skipped without them.
    fn client_rules(&self) -> Option<Arc<RuleSet>>;
}

/// Clients of SGLang's native format.
pub struct NativeShape;

impl ClientShape for NativeShape {
    fn body(&self, spec: &RequestSpec) -> Option<Vec<u8>> {
        let mut sampling = spec.extra_sampling.clone();
        for (name, value) in [
            ("max_new_tokens", spec.max_new_tokens),
            ("min_new_tokens", spec.min_new_tokens),
        ] {
            if let Some(value) = value {
                sampling.insert(name.into(), value.into());
            }
        }
        let mut body = spec.extra.clone();
        body.extend([
            ("input_ids".into(), json!(spec.input_ids)),
            ("stream".into(), json!(spec.stream)),
            ("sampling_params".into(), Value::Object(sampling)),
            ("return_logprob".into(), json!(spec.return_logprob)),
            ("logprob_start_len".into(), json!(spec.logprob_start_len)),
            ("top_logprobs_num".into(), json!(spec.top_logprobs_num)),
            (
                "return_routed_experts".into(),
                json!(spec.return_routed_experts),
            ),
        ]);
        Some(Value::Object(body).to_string().into_bytes())
    }

    fn client_rules(&self) -> Option<Arc<RuleSet>> {
        Some(crate::rules())
    }
}
