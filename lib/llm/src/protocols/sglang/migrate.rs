// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Continuing an SGLang `/generate` request on another worker.
//!
//! A retry's prompt is the client's prompt followed by the delivered tokens,
//! so its body asks only for the rest of the budget and stops recomputing
//! prompt-side outputs the client already has. Requests whose engine state a
//! retry cannot rebuild (penalties, seeds, grammars, sessions) do not migrate.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Map, Value};

/// SGLang's `SamplingParams.max_new_tokens` default.
pub const DEFAULT_MAX_NEW_TOKENS: u32 = 128;

/// The request side of `merge_rules.json`.
#[derive(Deserialize)]
pub(super) struct RequestRules {
    prefill_request_flags: BTreeMap<String, Vec<Vec<String>>>,
    other_request_flags: Vec<String>,
    migration: MigrationRules,
}

#[derive(Deserialize)]
struct MigrationRules {
    /// Fields whose engine state a retry cannot rebuild, unless at default.
    block_unless_default: BTreeMap<String, Value>,
    /// Top-level values a retry sets once the client has prompt-side outputs.
    retry_without_prompt_outputs: Map<String, Value>,
}

fn is_default(value: &Value, default: &Value) -> bool {
    match (value, default) {
        (Value::Null, _) => true,
        (Value::Array(values), _) if values.is_empty() => true,
        (Value::String(text), _) if text.is_empty() => true,
        (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
        _ => value == default,
    }
}

/// The engine's budget for a request that omits `max_new_tokens`:
/// `--preferred-sampling-params`, then SGLang's default. `None` when the
/// worker does not advertise its engine.
fn engine_default(engine_info: &Value) -> Option<u32> {
    let preferred = match engine_info.as_object()?.get("preferred_sampling_params") {
        Some(Value::String(json)) => serde_json::from_str(json).ok(),
        other => other.cloned(),
    };
    let preferred = preferred
        .and_then(|p: Value| p.get("max_new_tokens")?.as_u64())
        .and_then(|n| u32::try_from(n).ok());
    Some(preferred.unwrap_or(DEFAULT_MAX_NEW_TOKENS))
}

/// How later attempts continue one request.
pub(super) struct Continuation {
    original: Map<String, Value>,
    max_new_tokens: u32,
    min_new_tokens: Option<u32>,
}

impl Continuation {
    /// `Err` says why a failed attempt cannot continue on another worker.
    pub(super) fn new(
        rules: &RequestRules,
        body: &Map<String, Value>,
        limits: (Option<u32>, Option<u32>),
        engine_info: &Value,
    ) -> Result<Self, String> {
        let request = Value::Object(body.clone());
        if let Some(pointer) = rules
            .migration
            .block_unless_default
            .iter()
            .find(|(p, default)| request.pointer(p).is_some_and(|v| !is_default(v, default)))
            .map(|(p, _)| p)
        {
            return Err(format!(
                "`{pointer}` keeps engine state a retry cannot rebuild"
            ));
        }
        let known: Vec<&String> = rules
            .prefill_request_flags
            .values()
            .flatten()
            .flatten()
            .chain(&rules.other_request_flags)
            .collect();
        if let Some(flag) = body
            .keys()
            .find(|key| key.starts_with("return_") && !known.contains(key))
        {
            return Err(format!("`{flag}` is not classified for migration"));
        }
        if engine_info.get("skip_tokenizer_init") == Some(&Value::Bool(false)) {
            return Err("the engine detokenizes, and text cannot continue across workers".into());
        }
        let max_new_tokens = limits
            .0
            .or_else(|| engine_default(engine_info))
            .ok_or("max_new_tokens is omitted and the worker does not report its default")?;
        Ok(Self {
            original: body.clone(),
            max_new_tokens,
            min_new_tokens: limits.1,
        })
    }

    pub(super) fn returns_logprobs(&self) -> bool {
        self.original.get("return_logprob") == Some(&Value::Bool(true))
    }

    /// The body after `delivered` tokens, rebuilt from the original so repeated
    /// migrations never compound; `None` when the budget is spent.
    pub(super) fn retry(
        &self,
        rules: &RequestRules,
        delivered: usize,
        prompt_delivered: bool,
    ) -> Option<Map<String, Value>> {
        let delivered = u32::try_from(delivered).unwrap_or(u32::MAX);
        let remaining = self.max_new_tokens.saturating_sub(delivered);
        if remaining == 0 {
            return None;
        }
        let mut body = self.original.clone();
        let sampling = body.entry("sampling_params").or_insert(Value::Null);
        if !sampling.is_object() {
            *sampling = Value::Object(Map::new());
        }
        let sampling = sampling
            .as_object_mut()
            .expect("sampling_params is an object");
        sampling.insert("max_new_tokens".into(), remaining.into());
        match self.min_new_tokens.map(|min| min.saturating_sub(delivered)) {
            Some(min) if min > 0 => sampling.insert("min_new_tokens".into(), min.into()),
            _ => sampling.remove("min_new_tokens"),
        };
        if prompt_delivered {
            body.extend(rules.migration.retry_without_prompt_outputs.clone());
        }
        Some(body)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn continuation(body: Value, engine_info: Value) -> Result<Continuation, String> {
        let body = body.as_object().unwrap();
        let sampling = &body["sampling_params"];
        let limit = |key| sampling.get(key).and_then(Value::as_u64).map(|n| n as u32);
        let limits = (limit("max_new_tokens"), limit("min_new_tokens"));
        Continuation::new(&super::super::REQUEST_RULES, body, limits, &engine_info)
    }

    #[test]
    fn stateful_requests_cannot_migrate() {
        for field in [
            json!({"sampling_params": {"frequency_penalty": 0.5}}),
            json!({"sampling_params": {"sampling_seed": 7}}),
            json!({"sampling_params": {"stop": ["x"]}}),
            json!({"sampling_params": {}, "session_params": {"id": "s"}}),
            json!({"sampling_params": {}, "return_future_thing": true}),
        ] {
            assert!(continuation(field, json!({})).is_err());
        }
        let defaults = json!({"sampling_params": {"frequency_penalty": 0, "stop": []},
            "return_logprob": true, "return_routed_experts": true});
        assert!(continuation(defaults, json!({})).is_ok());
        let detokenizing = json!({"skip_tokenizer_init": false});
        assert!(continuation(json!({"sampling_params": {}}), detokenizing).is_err());
    }

    #[test]
    fn omitted_budgets_continue_from_the_engine_default() {
        let body = json!({"sampling_params": {}});
        assert!(continuation(body.clone(), Value::Null).is_err());
        let budget = |info| continuation(body.clone(), info).unwrap().max_new_tokens;
        assert_eq!(budget(json!({})), DEFAULT_MAX_NEW_TOKENS);
        let preferred = r#"{"max_new_tokens": 9}"#;
        assert_eq!(budget(json!({"preferred_sampling_params": preferred})), 9);
    }

    #[test]
    fn retries_rebuild_the_budget_from_the_original_body() {
        let body = json!({"logprob_start_len": 0,
            "sampling_params": {"max_new_tokens": 10, "min_new_tokens": 4}});
        let continuation = continuation(body, Value::Null).unwrap();
        let rules = &super::super::REQUEST_RULES;
        let first = continuation.retry(rules, 3, false).unwrap();
        assert_eq!(
            first["sampling_params"],
            json!({"max_new_tokens": 7, "min_new_tokens": 1})
        );
        assert_eq!(first["logprob_start_len"], 0);
        let second = continuation.retry(rules, 6, true).unwrap();
        assert_eq!(second["sampling_params"], json!({"max_new_tokens": 4}));
        assert_eq!(second["logprob_start_len"], -1);
        assert!(continuation.retry(rules, 10, true).is_none());
    }
}
