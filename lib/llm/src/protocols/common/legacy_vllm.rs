// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit, frontend-owned identification for pre-capability vLLM workers.
//!
//! This is operator configuration, never worker runtime data or public request
//! data. Resolve it independently for each committed routing hop. It must not
//! change MDC checksums, admitted cohorts, or advertised engine facts.

use std::collections::HashSet;

use dynamo_runtime::protocols::EndpointId;
use serde::{Deserialize, Serialize};

use crate::{model_type::ModelInput, worker_type::WorkerType};

/// Exact release adapters covered by the N-2 boundary fixtures. This identifies
/// legacy lowering only, not full native-server or disaggregated conformance.
/// TODO(1.8): remove after both releases leave the N-2 window.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LegacyVllmRelease {
    #[serde(rename = "1.4.0")]
    Dynamo14,
    #[serde(rename = "1.5.0")]
    Dynamo15,
}

impl LegacyVllmRelease {
    pub fn engine_version(self) -> &'static str {
        match self {
            Self::Dynamo14 => "0.26.0",
            Self::Dynamo15 => "0.28.0",
        }
    }

    pub(crate) fn supports_field(self, name: &str) -> bool {
        match name {
            "allowed_token_ids" | "bad_words_token_ids" => true,
            // The 1.4 adapter can assign this attribute, but retains a competing
            // logprob count. Attribute presence alone cannot authorize the field.
            "logprob_token_ids" => self == Self::Dynamo15,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    namespace: String,
    component: String,
    endpoint: String,
    model: String,
    worker_type: WorkerType,
    dynamo_release: LegacyVllmRelease,
}

/// Validated immutable startup declarations. There are no global defaults,
/// wildcard/prefix matches, inferred roles, or endpoint-name backend heuristics.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(transparent)]
pub struct LegacyVllmTargets(Vec<Declaration>);

impl LegacyVllmTargets {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn from_json(source: &str) -> anyhow::Result<Self> {
        anyhow::ensure!(
            source.len() <= 65536,
            "legacy vLLM target config is too large"
        );
        let declarations: Vec<Declaration> = serde_json::from_str(source)?;
        anyhow::ensure!(declarations.len() <= 128, "too many legacy vLLM targets");
        let mut scopes = HashSet::new();
        for declaration in &declarations {
            for value in [
                &declaration.namespace,
                &declaration.component,
                &declaration.endpoint,
                &declaration.model,
            ] {
                anyhow::ensure!(
                    !value.is_empty()
                        && value.len() <= 512
                        && value.trim() == value
                        && !value.contains('*')
                        && !value.chars().any(char::is_control),
                    "legacy vLLM targets require nonempty exact scope strings"
                );
            }
            anyhow::ensure!(
                declaration.worker_type != WorkerType::Encode,
                "legacy vLLM sampling targets do not cover encoder workers"
            );
            anyhow::ensure!(
                scopes.insert((
                    &declaration.namespace,
                    &declaration.component,
                    &declaration.endpoint,
                    &declaration.model,
                    declaration.worker_type,
                )),
                "duplicate legacy vLLM target scope"
            );
        }
        Ok(Self(declarations))
    }

    /// Resolve against the selected card's exact identity, not any nearby card.
    /// An absent worker role or a text/tensor RPC cannot inherit token coverage.
    pub fn resolve(
        &self,
        endpoint: &EndpointId,
        model: &str,
        worker_type: Option<WorkerType>,
        model_input: ModelInput,
    ) -> Option<LegacyVllmRelease> {
        if model_input != ModelInput::Tokens {
            return None;
        }
        self.0.iter().find_map(|declaration| {
            (declaration.namespace == endpoint.namespace
                && declaration.component == endpoint.component
                && declaration.endpoint == endpoint.name
                && declaration.model == model
                && Some(declaration.worker_type) == worker_type)
                .then_some(declaration.dynamo_release)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn declaration() -> serde_json::Value {
        json!({
            "namespace":"release-a", "component":"worker", "endpoint":"generate",
            "model":"model-a", "worker_type":"aggregated", "dynamo_release":"1.4.0",
        })
    }

    #[test]
    fn every_scope_dimension_is_exact_and_roles_are_not_inferred() {
        let targets = LegacyVllmTargets::from_json(&json!([declaration()]).to_string()).unwrap();
        let endpoint = EndpointId::from("release-a/worker/generate");
        assert_eq!(
            targets.resolve(
                &endpoint,
                "model-a",
                Some(WorkerType::Aggregated),
                ModelInput::Tokens
            ),
            Some(LegacyVllmRelease::Dynamo14)
        );
        for different in [
            "release-a-new/worker/generate",
            "release-a/other/generate",
            "release-a/worker/other",
        ] {
            assert_eq!(
                targets.resolve(
                    &EndpointId::from(different),
                    "model-a",
                    Some(WorkerType::Aggregated),
                    ModelInput::Tokens
                ),
                None
            );
        }
        assert_eq!(
            targets.resolve(
                &endpoint,
                "model-b",
                Some(WorkerType::Aggregated),
                ModelInput::Tokens
            ),
            None
        );
        assert_eq!(
            targets.resolve(&endpoint, "model-a", None, ModelInput::Tokens),
            None
        );
        assert_eq!(
            targets.resolve(
                &endpoint,
                "model-a",
                Some(WorkerType::Prefill),
                ModelInput::Tokens
            ),
            None
        );
        assert_eq!(
            targets.resolve(
                &endpoint,
                "model-a",
                Some(WorkerType::Aggregated),
                ModelInput::Text
            ),
            None
        );
        assert_eq!(
            LegacyVllmTargets::default().resolve(
                &endpoint,
                "model-a",
                Some(WorkerType::Aggregated),
                ModelInput::Tokens
            ),
            None
        );
    }

    #[test]
    fn malformed_ambiguous_or_unverified_declarations_fail_startup_validation() {
        for (field, value) in [
            ("namespace", json!("*")),
            ("model", json!("")),
            ("endpoint", json!(" generate")),
            ("component", json!("worker\n")),
            ("worker_type", json!("encode")),
            ("worker_type", json!(null)),
            ("dynamo_release", json!("1.4")),
            ("dynamo_release", json!("1.3.0")),
            ("dynamo_release", json!("1.5.0+local")),
            ("unknown", json!(true)),
        ] {
            let mut entry = declaration();
            entry[field] = value;
            assert!(
                LegacyVllmTargets::from_json(&json!([entry]).to_string()).is_err(),
                "{field}"
            );
        }
        assert!(
            LegacyVllmTargets::from_json(&json!([declaration(), declaration()]).to_string())
                .is_err()
        );
        let mut second = declaration();
        second["dynamo_release"] = json!("1.5.0");
        assert!(LegacyVllmTargets::from_json(&json!([declaration(), second]).to_string()).is_err());
        assert!(
            LegacyVllmTargets::from_json(&json!(vec![declaration(); 129]).to_string()).is_err()
        );
        assert!(LegacyVllmTargets::from_json(&" ".repeat(65537)).is_err());
    }

    #[test]
    fn independent_hops_and_serialized_declarations_keep_release_identity() {
        let mut prefill = declaration();
        prefill["component"] = json!("prefill");
        prefill["worker_type"] = json!("prefill");
        prefill["dynamo_release"] = json!("1.5.0");
        let targets =
            LegacyVllmTargets::from_json(&json!([declaration(), prefill]).to_string()).unwrap();
        let roundtrip =
            LegacyVllmTargets::from_json(&serde_json::to_string(&targets).unwrap()).unwrap();
        assert_eq!(
            roundtrip.resolve(
                &EndpointId::from("release-a/prefill/generate"),
                "model-a",
                Some(WorkerType::Prefill),
                ModelInput::Tokens
            ),
            Some(LegacyVllmRelease::Dynamo15)
        );
        assert_eq!(LegacyVllmRelease::Dynamo14.engine_version(), "0.26.0");
        assert_eq!(LegacyVllmRelease::Dynamo15.engine_version(), "0.28.0");
    }
}
