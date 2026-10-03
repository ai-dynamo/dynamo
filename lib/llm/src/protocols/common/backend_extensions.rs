// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capability-gated lowering at the frontend/worker boundary.
//!
//! The v1 envelope lives under `PreprocessedRequest.extra_args.backend_extensions`
//! so existing wire readers remain tolerant. It is never a public passthrough API.
//! These three fields affect decode sampling/output, not prompt processing or KV
//! identity, and are representable on aggregated and disaggregated token paths.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::local_model::runtime_config::{
    ModelRuntimeConfig, VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
};
use crate::protocols::common::invalid_argument_error;
use crate::protocols::common::legacy_vllm::LegacyVllmRelease;

pub const VLLM_PROTOCOL_EXTENSIONS_CAPABILITY: &str = "vllm_protocol_extensions";
pub const MAX_EXTENSION_BYTES: usize = 65536;
pub const MAX_EXTENSION_VALUES: usize = 16384;
pub const MAX_EXTENSION_DEPTH: usize = 8;
pub(crate) const SAMPLING_FIELDS: &[&str] = &[
    "allowed_token_ids",
    "bad_words_token_ids",
    "logprob_token_ids",
];

/// Frontend-local causal identity, not a new worker-wire error representation.
/// Only the reviewed public sampling vocabulary may enter this annotation.
#[derive(Debug, thiserror::Error)]
#[error("{source}")]
pub(crate) struct SamplingCapabilityFailure {
    pub field: &'static str,
    #[source]
    source: anyhow::Error,
}

fn sampling_capability_failure(field: &str, source: anyhow::Error) -> anyhow::Error {
    match SAMPLING_FIELDS
        .iter()
        .copied()
        .find(|known| *known == field)
    {
        Some(field) => SamplingCapabilityFailure { field, source }.into(),
        None => source,
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VllmExtensionCapability {
    pub schema_version: u32,
    pub target: String,
    pub engine_version: String,
    pub sampling_fields: Vec<String>,
}

/// Resolved once for lowering and its diagnostic projection. A catalog must not
/// implement its own capability precedence or treat an invalid v1 declaration as
/// permission to fall back to a legacy capability.
pub(crate) enum SamplingTarget {
    Current(Vec<String>),
    Legacy(Option<LegacyVllmRelease>),
    Unavailable,
}

impl SamplingTarget {
    pub(crate) fn resolve(
        runtime: &ModelRuntimeConfig,
        legacy: Option<LegacyVllmRelease>,
    ) -> anyhow::Result<Self> {
        if let Some(capability) = runtime
            .get_engine_specific::<VllmExtensionCapability>(VLLM_PROTOCOL_EXTENSIONS_CAPABILITY)
            .map_err(|_| rejected("backend_extensions", "malformed worker capability"))?
        {
            if capability.schema_version != 1 || capability.target != "vllm" {
                return Err(rejected(
                    "backend_extensions",
                    "worker does not support vLLM schema v1",
                ));
            }
            return Ok(Self::Current(capability.sampling_fields));
        }
        if let Some(legacy) = legacy {
            return Ok(Self::Legacy(Some(legacy)));
        }
        if runtime
            .get_engine_specific::<bool>(VLLM_INFERENCE_V1_GENERATE_CAPABILITY)
            .map_err(|_| rejected("backend_extensions", "malformed legacy worker capability"))?
            == Some(true)
        {
            return Ok(Self::Legacy(None));
        }
        Ok(Self::Unavailable)
    }

    pub(crate) fn validate_field(&self, field: &str) -> anyhow::Result<()> {
        let reason = match self {
            Self::Current(fields) if !fields.iter().any(|name| name == field) => {
                Some("selected worker cannot preserve this field")
            }
            Self::Legacy(Some(target)) if !target.supports_field(field) => {
                Some("declared legacy vLLM release cannot preserve this field")
            }
            Self::Unavailable => Some("selected worker has no verified vLLM extension capability"),
            _ => None,
        };
        match reason {
            Some(reason) => Err(sampling_capability_failure(field, rejected(field, reason))),
            None => Ok(()),
        }
    }
}

fn rejected(field: &str, reason: &str) -> anyhow::Error {
    invalid_argument_error(format!(
        "vLLM protocol extension `{field}` rejected at preprocessing: {reason}. \
         Remove the field or use a compatible worker."
    ))
}

/// Whether this hop must verify a sampling-extension contract. Ordinary requests
/// and unrelated extra_args retain their existing dispatch path.
pub(crate) fn requires_worker_validation(extra: Option<&Value>) -> bool {
    let Some(extra) = extra.and_then(Value::as_object) else {
        return false;
    };
    extra
        .get("backend_extensions")
        .is_some_and(|v| !v.is_null())
        || extra
            .get("sampling_options")
            .and_then(Value::as_object)
            .is_some_and(|fields| {
                SAMPLING_FIELDS
                    .iter()
                    .any(|name| fields.get(*name).is_some_and(|value| !value.is_null()))
            })
}

/// Recheck the actual dispatch target, not the representative WorkerSet card.
/// The same bounded legacy subset must be readable at every prefill/decode hop.
/// Callers must obtain runtime from their selected worker's live discovery data.
#[cfg(test)]
pub(crate) fn validate_worker_extensions(
    extra: Option<&Value>,
    runtime: Option<&ModelRuntimeConfig>,
) -> anyhow::Result<()> {
    validate_worker_extensions_for_target(extra, runtime, None)
}

/// The explicit fallback is scoped by the caller to this committed routing hop.
/// It cannot replace a missing/closed discovery watch or a malformed capability.
pub(crate) fn validate_worker_extensions_for_target(
    extra: Option<&Value>,
    runtime: Option<&ModelRuntimeConfig>,
    legacy_target: Option<LegacyVllmRelease>,
) -> anyhow::Result<()> {
    if !requires_worker_validation(extra) {
        return Ok(());
    }
    let extra = extra.and_then(Value::as_object).expect("checked above");
    let mut fields = Map::new();
    if let Some(legacy) = extra.get("sampling_options") {
        validate_extension_bounds(legacy)?;
    }
    if let Some(legacy) = extra.get("sampling_options").and_then(Value::as_object) {
        for name in SAMPLING_FIELDS {
            if let Some(value) = legacy.get(*name) {
                fields.insert((*name).to_owned(), value.clone());
            }
        }
    }
    if let Some(envelope) = extra.get("backend_extensions").filter(|v| !v.is_null()) {
        validate_extension_bounds(envelope)?;
        let envelope = envelope
            .as_object()
            .ok_or_else(|| rejected("backend_extensions", "expected a versioned object"))?;
        if envelope.len() != 2
            || envelope.get("schema_version").and_then(Value::as_u64) != Some(1)
            || !envelope.contains_key("vllm")
        {
            return Err(rejected(
                "backend_extensions",
                "unsupported backend or schema version",
            ));
        }
        let current = envelope["vllm"]
            .as_object()
            .ok_or_else(|| rejected("vllm", "expected an object"))?;
        if current
            .keys()
            .any(|name| !SAMPLING_FIELDS.contains(&name.as_str()))
        {
            return Err(rejected(
                "vllm",
                "unknown or canonical field in extension map",
            ));
        }
        for (name, value) in current {
            if fields
                .get(name)
                .is_some_and(|old| !old.is_null() && old != value)
            {
                return Err(rejected(name, "conflicting current and legacy values"));
            }
            fields.insert(name.clone(), value.clone());
        }
    }
    let runtime = runtime.ok_or_else(|| {
        rejected(
            "backend_extensions",
            "dispatch target has no verified runtime capability",
        )
    })?;
    // Reuse the writer's exact field/type/bounds/capability policy. The result is
    // discarded: dispatch must not overwrite either representation on the wire.
    lower_sampling_passthrough_for_target(fields, runtime, legacy_target)?;
    Ok(())
}

/// Enforce bounds before serializing, logging, or copying extension payloads.
pub fn validate_extension_bounds(value: &Value) -> anyhow::Result<()> {
    let mut pending = vec![(value, 0usize)];
    let mut visited = 0usize;
    while let Some((item, depth)) = pending.pop() {
        visited += 1;
        if depth > MAX_EXTENSION_DEPTH || visited + pending.len() > MAX_EXTENSION_VALUES {
            return Err(rejected(
                "backend_extensions",
                "nesting or value-count limit exceeded",
            ));
        }
        match item {
            Value::Array(values) => {
                if visited + pending.len() + values.len() > MAX_EXTENSION_VALUES {
                    return Err(rejected("backend_extensions", "value-count limit exceeded"));
                }
                pending.extend(values.iter().map(|v| (v, depth + 1)));
            }
            Value::Object(values) => {
                if visited + pending.len() + values.len() > MAX_EXTENSION_VALUES {
                    return Err(rejected("backend_extensions", "value-count limit exceeded"));
                }
                pending.extend(values.values().map(|v| (v, depth + 1)));
            }
            _ => {}
        }
    }
    // A limited writer avoids allocating another unbounded JSON buffer.
    struct ByteLimit(usize);
    impl std::io::Write for ByteLimit {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > self.0 {
                return Err(std::io::Error::other("extension size limit exceeded"));
            }
            self.0 -= bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(ByteLimit(MAX_EXTENSION_BYTES), value)
        .map_err(|_| rejected("backend_extensions", "size limit exceeded"))
}

/// Lower only the reviewed sampling-extension subset for the selected worker card.
/// Default requests do not require new metadata. Missing capabilities never grant
/// permission to send an opaque field to an unidentified backend.
pub fn lower_sampling_passthrough(
    fields: Map<String, Value>,
    runtime: &ModelRuntimeConfig,
) -> anyhow::Result<Map<String, Value>> {
    lower_sampling_passthrough_for_target(fields, runtime, None)
}

/// Explicit legacy identification is frontend-owned, never synthetic runtime
/// metadata. Resolve its exact scope before calling. A live v1 advertisement
/// remains authoritative, even if it rejects a field supported by the fallback.
pub fn lower_sampling_passthrough_for_target(
    mut fields: Map<String, Value>,
    runtime: &ModelRuntimeConfig,
    legacy_target: Option<LegacyVllmRelease>,
) -> anyhow::Result<Map<String, Value>> {
    let mut extra = Map::new();
    // detokenize is a legacy token-path control, not an opaque engine extension.
    let detokenize = fields.remove("detokenize");
    fields.retain(|_, value| !value.is_null());
    if fields
        .keys()
        .any(|name| !SAMPLING_FIELDS.contains(&name.as_str()))
    {
        return Err(rejected(
            "vllm",
            "unknown or canonical field in extension map",
        ));
    }
    let valid_ids = |value: &Value| {
        value.as_array().is_some_and(|ids| {
            ids.iter()
                .all(|id| id.as_u64().is_some_and(|id| u32::try_from(id).is_ok()))
        })
    };
    for (name, value) in &fields {
        let valid = if name == "bad_words_token_ids" {
            value
                .as_array()
                .is_some_and(|groups| groups.iter().all(&valid_ids))
        } else {
            valid_ids(value)
        };
        if !valid {
            return Err(rejected(name, "expected unsigned 32-bit token ID arrays"));
        }
    }
    if detokenize
        .as_ref()
        .is_some_and(|value| !value.is_boolean() && !value.is_null())
    {
        return Err(rejected("detokenize", "expected a boolean"));
    }
    if !fields.is_empty() {
        let target = SamplingTarget::resolve(runtime, legacy_target).map_err(|error| {
            // A malformed declaration rejects all requested fields. Identify the
            // first actual directive deterministically, never an internal key.
            sampling_capability_failure(fields.keys().min().expect("nonempty fields"), error)
        })?;
        for name in fields.keys() {
            target.validate_field(name)?;
        }
        if matches!(target, SamplingTarget::Current(_)) {
            let envelope = serde_json::json!({"schema_version": 1, "vllm": fields});
            validate_extension_bounds(&envelope)?;
            // N-2: a selected decode worker may delegate prefill to an older
            // worker. Its card does not prove every downstream hop reads v1.
            // Dual-write only this already-representable legacy subset.
            // TODO(1.8): remove when Dynamo 1.5 leaves the support window.
            extra.insert("sampling_options".into(), envelope["vllm"].clone());
            extra.insert("backend_extensions".into(), envelope);
        } else {
            // A legacy target is identified by an explicit release declaration
            // or the legacy Generate marker. Dynamo 1.4 has no such marker and
            // therefore requires an explicit declaration, not missing-means-vLLM.
            // Only the previously supported subset uses the legacy map.
            // TODO(1.8): remove when Dynamo 1.5 leaves the N-2 support window.
            let legacy = Value::Object(fields);
            validate_extension_bounds(&legacy)?;
            extra.insert("sampling_options".into(), legacy);
        }
    }
    if let Some(detokenize) = detokenize {
        extra
            .entry("sampling_options")
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
            .expect("sampling_options constructed as an object")
            .insert("detokenize".into(), detokenize);
    }
    Ok(extra)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn missing_capability_error_identifies_the_public_directive() {
        for name in SAMPLING_FIELDS {
            let value = if *name == "bad_words_token_ids" {
                json!([[31415]])
            } else {
                json!([31415])
            };
            let fields = Map::from_iter([((*name).to_owned(), value)]);
            let failure =
                lower_sampling_passthrough(fields, &ModelRuntimeConfig::default()).unwrap_err();
            assert_eq!(
                failure
                    .downcast_ref::<SamplingCapabilityFailure>()
                    .unwrap()
                    .field,
                *name
            );
            let error = failure.to_string();
            assert!(error.contains(&format!("`{name}`")), "{error}");
            assert!(
                !error.contains("31415"),
                "request values must not be logged"
            );
        }
    }

    fn fields() -> Map<String, Value> {
        json!({"allowed_token_ids": [0, 1], "logprob_token_ids": []})
            .as_object()
            .unwrap()
            .clone()
    }

    #[test]
    fn malformed_capability_identifies_first_requested_public_field_without_payloads() {
        let mut runtime = ModelRuntimeConfig::default();
        runtime
            .set_engine_specific(
                VLLM_PROTOCOL_EXTENSIONS_CAPABILITY,
                json!("private-capability"),
            )
            .unwrap();
        let requested = fields();
        for fields in [
            requested.clone(),
            requested.into_iter().rev().collect::<Map<String, Value>>(),
        ] {
            let error = lower_sampling_passthrough(fields, &runtime).unwrap_err();
            assert_eq!(
                error
                    .downcast_ref::<SamplingCapabilityFailure>()
                    .unwrap()
                    .field,
                "allowed_token_ids"
            );
            assert!(!error.to_string().contains("private-capability"));
        }
        let unrecognized =
            sampling_capability_failure("private-key", invalid_argument_error("safe message"));
        assert!(
            unrecognized
                .downcast_ref::<SamplingCapabilityFailure>()
                .is_none()
        );
    }

    fn capable() -> ModelRuntimeConfig {
        let mut runtime = ModelRuntimeConfig::default();
        runtime
            .set_engine_specific(
                VLLM_PROTOCOL_EXTENSIONS_CAPABILITY,
                json!({
                    "schema_version": 1, "target": "vllm", "engine_version": "0.30.0",
                    "sampling_fields": SAMPLING_FIELDS,
                }),
            )
            .unwrap();
        runtime
    }

    #[test]
    fn current_writer_uses_namespaced_versioned_envelope_and_keeps_empty_arrays() {
        let extra = lower_sampling_passthrough(fields(), &capable()).unwrap();
        assert_eq!(
            extra["backend_extensions"],
            json!({"schema_version":1, "vllm":fields()})
        );
        assert_eq!(extra["sampling_options"], json!(fields()));
    }

    #[test]
    fn legacy_vllm_worker_uses_old_wire_but_unknown_backend_fails_closed() {
        let mut runtime = ModelRuntimeConfig::default();
        assert!(lower_sampling_passthrough(fields(), &runtime).is_err());
        assert!(
            lower_sampling_passthrough(Map::new(), &runtime)
                .unwrap()
                .is_empty()
        );
        runtime
            .set_engine_specific(VLLM_INFERENCE_V1_GENERATE_CAPABILITY, true)
            .unwrap();
        let extra = lower_sampling_passthrough(fields(), &runtime).unwrap();
        assert_eq!(extra["sampling_options"], json!(fields()));
        assert!(!extra.contains_key("backend_extensions"));
    }

    #[test]
    fn current_capability_is_authoritative_and_does_not_fall_back_on_error() {
        let mut runtime = capable();
        runtime
            .set_engine_specific(VLLM_INFERENCE_V1_GENERATE_CAPABILITY, true)
            .unwrap();
        runtime.set_engine_specific(VLLM_PROTOCOL_EXTENSIONS_CAPABILITY, json!({
            "schema_version": 2, "target":"vllm", "engine_version":"future", "sampling_fields":[]
        })).unwrap();
        assert!(lower_sampling_passthrough(fields(), &runtime).is_err());
    }

    #[test]
    fn canonical_field_injection_and_unadvertised_fields_fail() {
        let mut injected = fields();
        injected.insert("temperature".into(), json!(0.7));
        assert!(lower_sampling_passthrough(injected, &capable()).is_err());
        let mut runtime = capable();
        runtime.set_engine_specific(VLLM_PROTOCOL_EXTENSIONS_CAPABILITY, json!({
            "schema_version":1,"target":"vllm","engine_version":"0.30.0","sampling_fields":[]
        })).unwrap();
        assert!(lower_sampling_passthrough(fields(), &runtime).is_err());
    }

    #[test]
    fn size_depth_and_value_count_are_bounded() {
        assert!(
            validate_extension_bounds(&json!({"value":"x".repeat(MAX_EXTENSION_BYTES)})).is_err()
        );
        assert!(validate_extension_bounds(&json!(vec![0; MAX_EXTENSION_VALUES])).is_err());
        let mut nested = json!(0);
        for _ in 0..=MAX_EXTENSION_DEPTH {
            nested = json!([nested]);
        }
        assert!(validate_extension_bounds(&nested).is_err());
        validate_extension_bounds(&json!({"schema_version":1,"vllm":fields()})).unwrap();
    }

    #[test]
    fn dispatch_rechecks_selected_worker_and_preserves_default_requests() {
        let extra = Value::Object(lower_sampling_passthrough(fields(), &capable()).unwrap());
        validate_worker_extensions(Some(&extra), Some(&capable())).unwrap();
        assert!(validate_worker_extensions(Some(&extra), None).is_err());
        assert!(
            validate_worker_extensions(Some(&extra), Some(&ModelRuntimeConfig::default())).is_err()
        );
        let mut restricted = capable();
        restricted
            .runtime_data
            .get_mut(VLLM_PROTOCOL_EXTENSIONS_CAPABILITY)
            .unwrap()["sampling_fields"] = json!([]);
        assert!(validate_worker_extensions(Some(&extra), Some(&restricted)).is_err());
        validate_worker_extensions(None, None).unwrap();
        validate_worker_extensions(Some(&json!({"other_backend_setting": true})), None).unwrap();
    }

    #[test]
    fn dispatch_rejects_conflicting_or_malformed_envelopes() {
        for extra in [
            json!({"backend_extensions":{"schema_version":2,"vllm":{}}}),
            json!({"backend_extensions":{"schema_version":1,"sglang":{}}}),
            json!({"backend_extensions":{"schema_version":1,"vllm":{"temperature":0.5}}}),
            json!({"backend_extensions":{"schema_version":1,"vllm":{"allowed_token_ids":null}},
                "sampling_options":{"allowed_token_ids":[0]}}),
        ] {
            assert!(validate_worker_extensions(Some(&extra), Some(&capable())).is_err());
        }
    }

    #[test]
    fn explicit_legacy_lowering_keeps_runtime_facts_and_wire_identity_separate() {
        let runtime = ModelRuntimeConfig::default();
        let before = serde_json::to_value(&runtime).unwrap();
        let fields = json!({"allowed_token_ids":[0],"bad_words_token_ids":[[1,2]]})
            .as_object()
            .unwrap()
            .clone();
        for release in [LegacyVllmRelease::Dynamo14, LegacyVllmRelease::Dynamo15] {
            let extra =
                lower_sampling_passthrough_for_target(fields.clone(), &runtime, Some(release))
                    .unwrap();
            assert_eq!(
                extra,
                json!({"sampling_options":fields})
                    .as_object()
                    .unwrap()
                    .clone()
            );
            assert_eq!(serde_json::to_value(&runtime).unwrap(), before);
            assert!(
                lower_sampling_passthrough_for_target(Map::new(), &runtime, Some(release))
                    .unwrap()
                    .is_empty()
            );
            let extra = Value::Object(extra);
            validate_worker_extensions_for_target(Some(&extra), Some(&runtime), Some(release))
                .unwrap();
            // An operator declaration does not revive a closed or absent watch.
            assert!(
                validate_worker_extensions_for_target(Some(&extra), None, Some(release)).is_err()
            );
        }
        assert!(lower_sampling_passthrough(fields, &runtime).is_err());
    }

    #[test]
    fn explicit_14_profile_cannot_gain_token_selection_from_generic_marker() {
        let mut runtime = ModelRuntimeConfig::default();
        runtime
            .set_engine_specific(VLLM_INFERENCE_V1_GENERATE_CAPABILITY, true)
            .unwrap();
        let selected = json!({"logprob_token_ids":[0]})
            .as_object()
            .unwrap()
            .clone();
        assert!(
            lower_sampling_passthrough_for_target(
                selected.clone(),
                &runtime,
                Some(LegacyVllmRelease::Dynamo14)
            )
            .is_err()
        );
        let actual = lower_sampling_passthrough_for_target(
            selected.clone(),
            &runtime,
            Some(LegacyVllmRelease::Dynamo15),
        )
        .unwrap();
        assert_eq!(actual["sampling_options"], json!(selected));
    }

    #[test]
    fn explicit_fallback_never_overrides_present_worker_capability() {
        let simple = json!({"allowed_token_ids":[0]})
            .as_object()
            .unwrap()
            .clone();
        for capability in [
            Value::Null,
            json!({}),
            json!({"schema_version":2,"target":"vllm","engine_version":"future","sampling_fields":[]}),
            json!({"schema_version":1,"target":"other","engine_version":"0.30.0","sampling_fields":["allowed_token_ids"]}),
            json!({"schema_version":1,"target":"vllm","engine_version":"0.30.0","sampling_fields":[]}),
        ] {
            let mut runtime = ModelRuntimeConfig::default();
            runtime
                .set_engine_specific(VLLM_PROTOCOL_EXTENSIONS_CAPABILITY, capability)
                .unwrap();
            assert!(
                lower_sampling_passthrough_for_target(
                    simple.clone(),
                    &runtime,
                    Some(LegacyVllmRelease::Dynamo14)
                )
                .is_err()
            );
        }
        let actual = lower_sampling_passthrough_for_target(
            fields(),
            &capable(),
            Some(LegacyVllmRelease::Dynamo14),
        )
        .unwrap();
        assert!(actual.contains_key("backend_extensions"));
    }
}
