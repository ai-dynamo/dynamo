// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded observations of compatibility decisions, not conformance claims.
//!
//! Count field decisions at each named boundary, not unique requests or successful
//! generations. Validation can run before target selection, so unresolved context
//! stays explicit. User values and arbitrary field/model/version strings never
//! become labels. The runtime admission descriptor supplies selected context.

use std::sync::LazyLock;

use prometheus::{IntCounterVec, Opts};

use super::profile::{CompatibilityProfile, Deployment, Endpoint, Processor, Transport};

pub(crate) static PROTOCOL_DECISIONS: LazyLock<IntCounterVec> =
    LazyLock::new(|| new_counter(crate::http::service::metrics::process_metrics_prefix()));

pub(super) fn new_counter(prefix: &str) -> IntCounterVec {
    IntCounterVec::new(
        Opts::new(
            format!("{prefix}_protocol_decisions_total"),
            "Protocol field decisions at named boundaries; not unique requests or conformance",
        ),
        &[
            "endpoint",
            "target",
            "processor",
            "transport",
            "deployment",
            "field",
            "stage",
            "decision",
            "reason",
        ],
    )
    .expect("fixed protocol decision labels and sanitized frontend prefix are valid")
}

#[derive(Clone, Copy)]
pub(crate) enum Event {
    RejectKnownField,
    RejectUnknownField,
    RejectWrongEndpoint,
    MalformedPromptLogprobs,
    IgnoreUnknownField,
    RejectCombination,
    AdmitPromptLogprobs,
    AdmitExtensions,
    RejectExtensions,
    LegacyExtensions,
    DualWriteExtensions,
}

impl Event {
    fn labels(self) -> [&'static str; 3] {
        match self {
            Self::RejectKnownField => ["validation", "reject", "unhandled_native_field"],
            Self::RejectUnknownField => ["validation", "reject", "unknown_field"],
            Self::RejectWrongEndpoint => ["validation", "reject", "wrong_endpoint"],
            Self::MalformedPromptLogprobs => {
                ["response_decode", "error", "malformed_backend_payload"]
            }
            Self::IgnoreUnknownField => ["validation", "ignore", "migration_ignore"],
            Self::RejectCombination => ["admission", "reject", "prompt_stream_combination"],
            Self::AdmitPromptLogprobs => ["admission", "admit", "prompt_admission_passed"],
            Self::AdmitExtensions => ["admission", "admit", "extension_contract_passed"],
            // This is a bundle-level rejection, not an accusation that every
            // requested field is individually unsupported.
            Self::RejectExtensions => ["admission", "reject", "extension_bundle_rejected"],
            Self::LegacyExtensions => ["admission", "legacy", "legacy_extension_envelope"],
            Self::DualWriteExtensions => ["admission", "legacy", "dual_write_extension_envelope"],
        }
    }
}

/// Only generated names and the fixed extension/bundle vocabulary are exposed.
fn field_label(field: &str) -> &'static str {
    match field {
        "bad_words_token_ids" => "bad_words_token_ids",
        "sampling_extensions" => "sampling_extensions",
        name => super::vllm_fields::VLLM_REQUEST_FIELDS
            .binary_search(&name)
            .map(|index| super::vllm_fields::VLLM_REQUEST_FIELDS[index])
            .unwrap_or("unknown"),
    }
}

pub(crate) fn record_validation(endpoint: Option<Endpoint>, field: &str, event: Event) {
    record(&PROTOCOL_DECISIONS, endpoint, None, field, event);
}

/// Decode only when the caller has requested prompt data. Missing/null data is
/// not a decoding failure: workers may send it on a later chunk. Profile context
/// is intentionally unresolved here; model names cannot establish its identity.
pub(crate) fn decode_prompt_logprobs(
    endpoint: Endpoint,
    engine_data: Option<&serde_json::Value>,
) -> anyhow::Result<Option<crate::protocols::common::llm_backend::PromptLogprobs>> {
    crate::protocols::common::llm_backend::prompt_logprobs_from_engine_data(engine_data)
        .inspect_err(|_| {
            record(
                &PROTOCOL_DECISIONS,
                Some(endpoint),
                None,
                "prompt_logprobs",
                Event::MalformedPromptLogprobs,
            );
        })
}

pub(super) fn record(
    counter: &IntCounterVec,
    endpoint: Option<Endpoint>,
    profile: Option<CompatibilityProfile>,
    field: &str,
    event: Event,
) {
    let (target, processor, transport, deployment) = match profile {
        Some(profile) => (
            profile.target.label(),
            match profile.pipeline.processor {
                Processor::Rust => "rust",
                Processor::Vllm => "vllm",
                Processor::Sglang => "sglang",
                Processor::ExternalFactory => "external_factory",
                Processor::Backend => "backend",
                Processor::Unknown => "unknown",
            },
            match profile.pipeline.transport {
                Transport::PreprocessedRpc => "preprocessed_rpc",
                Transport::OpenaiRpc => "openai_rpc",
                Transport::Unknown => "unknown",
            },
            match profile.pipeline.deployment {
                Deployment::Aggregated => "aggregated",
                Deployment::Disaggregated => "disaggregated",
                Deployment::Unknown => "unknown",
            },
        ),
        None => ("unresolved", "unresolved", "unresolved", "unresolved"),
    };
    let [stage, decision, reason] = event.labels();
    counter
        .with_label_values(&[
            profile
                .map(|value| value.endpoint)
                .or(endpoint)
                .map_or("unresolved", Endpoint::as_str),
            target,
            processor,
            transport,
            deployment,
            field_label(field),
            stage,
            decision,
            reason,
        ])
        .inc();
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::core::Collector;

    #[test]
    fn unknown_names_share_one_series_without_recording_the_input() {
        let counter = new_counter("test");
        for index in 0..1000 {
            record(
                &counter,
                Some(Endpoint::Chat),
                None,
                &format!("private-{index}"),
                Event::IgnoreUnknownField,
            );
        }
        let families = counter.collect();
        let metrics = families[0].get_metric();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].get_counter().as_ref().unwrap().value(), 1000.0);
        let labels: Vec<_> = metrics[0]
            .get_label()
            .iter()
            .map(|label| (label.name(), label.value()))
            .collect();
        assert!(labels.contains(&("field", "unknown")));
        assert!(labels.contains(&("endpoint", "/v1/chat/completions")));
        assert!(labels.contains(&("target", "unresolved")));
        assert!(!format!("{labels:?}").contains("private"));
    }

    #[test]
    fn selected_profile_labels_are_derived_from_the_admission_descriptor() {
        let counter = new_counter("test");
        let profile = CompatibilityProfile::new(
            super::super::admission::TargetAdmission::Vllm030,
            Endpoint::Completion,
            super::super::profile::PipelineContext::from_pipeline(
                Endpoint::Completion,
                crate::model_type::ModelInput::Tokens,
                Some(crate::worker_type::WorkerType::Aggregated),
                Some(crate::entrypoint::ChatProcessorIdentity::Vllm),
            ),
        );
        record(
            &counter,
            None,
            Some(profile),
            "prompt_logprobs",
            Event::RejectCombination,
        );
        let families = counter.collect();
        let labels: Vec<_> = families[0].get_metric()[0]
            .get_label()
            .iter()
            .map(|label| (label.name(), label.value()))
            .collect();
        for pair in [
            ("target", "vllm/0.30.0"),
            ("processor", "rust"),
            ("endpoint", "/v1/completions"),
            ("field", "prompt_logprobs"),
            ("decision", "reject"),
        ] {
            assert!(labels.contains(&pair), "{labels:?}");
        }
    }
}
