// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Target-version-specific admission at the selected WorkerSet pipeline boundary.
//!
//! This is a narrow rule set, not a claim of complete endpoint compatibility.
//! Resolve authoritative engine facts once from the admitted card, without changing
//! discovery identity. Never infer a backend from a model name or tokenizer.
//! Rules run before preprocessing changes `stream`, including for Python processors.

use std::sync::Arc;

use dynamo_runtime::pipeline::{
    AsyncEngine, Data, Error, ManyOut, ServerStreamingEngine, SingleIn, async_trait,
};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::protocols::common::{
    backend_extensions::{
        SamplingCapabilityFailure, VLLM_PROTOCOL_EXTENSIONS_CAPABILITY, VllmExtensionCapability,
        lower_sampling_passthrough_for_target,
    },
    invalid_argument_error,
    legacy_vllm::LegacyVllmRelease,
    prompt_logprobs::{FULL_VOCAB_COUNT, public_count_to_wire},
};

use super::super::{
    chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
};
#[cfg(test)]
use super::profile::PipelineContext;
use super::profile::{CompatibilityProfile, Endpoint, PromptLogprobsAdmission, Transport};
use super::rejection::{CompatibilityRejection, RejectionKind, RejectionStage};
use super::telemetry::{self, Event, PROTOCOL_DECISIONS};

/// Only exact releases whose relevant validators have been inspected are selected.
/// Development/local-version suffixes must not silently inherit release semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub(crate) enum TargetAdmission {
    /// No new metadata: preserve ordinary N-2 and other-backend serving behavior.
    #[serde(rename = "unidentified")]
    Unidentified,
    /// Capability exists but its schema, target, or release cannot select a rule.
    #[serde(rename = "vllm/unverified")]
    Unverified,
    #[serde(rename = "vllm/0.29.0")]
    Vllm029,
    #[serde(rename = "vllm/0.30.0")]
    Vllm030,
}

impl TargetAdmission {
    pub(crate) fn from_runtime(runtime: &ModelRuntimeConfig) -> Self {
        match runtime
            .get_engine_specific::<VllmExtensionCapability>(VLLM_PROTOCOL_EXTENSIONS_CAPABILITY)
        {
            Ok(None) => Self::Unidentified,
            Ok(Some(capability))
                if capability.schema_version == 1 && capability.target == "vllm" =>
            {
                match capability.engine_version.as_str() {
                    "0.29.0" => Self::Vllm029,
                    "0.30.0" => Self::Vllm030,
                    _ => Self::Unverified,
                }
            }
            _ => Self::Unverified,
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Unidentified => "unidentified",
            Self::Unverified => "vllm/unverified",
            Self::Vllm029 => "vllm/0.29.0",
            Self::Vllm030 => "vllm/0.30.0",
        }
    }

    #[cfg(test)]
    pub(crate) fn wrap<Req: AdmissionRequest, Resp: Data>(
        self,
        inner: ServerStreamingEngine<Req, Resp>,
        pipeline: PipelineContext,
        runtime: &ModelRuntimeConfig,
        legacy_target: Option<LegacyVllmRelease>,
    ) -> ServerStreamingEngine<Req, Resp> {
        CompatibilityProfile::new(self, Req::ENDPOINT, pipeline).wrap(inner, runtime, legacy_target)
    }
}

impl CompatibilityProfile {
    pub(crate) fn wrap<Req: AdmissionRequest, Resp: Data>(
        self,
        inner: ServerStreamingEngine<Req, Resp>,
        runtime: &ModelRuntimeConfig,
        legacy_target: Option<LegacyVllmRelease>,
    ) -> ServerStreamingEngine<Req, Resp> {
        assert_eq!(
            self.endpoint,
            Req::ENDPOINT,
            "pipeline profile endpoint must match engine type"
        );
        // Token pipelines must validate extensions even without target metadata:
        // unknown capability is not permission to accept a semantic directive.
        let extension_target = (self.pipeline.transport == Transport::PreprocessedRpc)
            .then(|| (runtime.clone(), legacy_target));
        // Keep the boundary even for unidentified text workers: a signed public
        // -1 must not bypass capability admission through an unverified path.
        Arc::new(AdmissionEngine {
            inner,
            profile: self,
            extension_target,
        })
    }
}

impl CompatibilityProfile {
    fn validate(self, request: &impl AdmissionRequest) -> anyhow::Result<()> {
        if !request.client_stream() || !request.prompt_logprobs().is_some_and(|n| n > 0) {
            return Ok(());
        }
        let reason = match self.prompt_logprobs_admission {
            // Absent metadata is not evidence of vLLM. A separate legacy target
            // resolution mechanism is needed before applying native rules here.
            PromptLogprobsAdmission::UnidentifiedTarget => return Ok(()),
            PromptLogprobsAdmission::UnverifiedTarget => {
                "the selected worker has no verified admission rule for this combination"
            }
            // Both pinned sources reject positive prompt counts with stream=true.
            // Zero is deliberately accepted (not equivalent to positive counts).
            // Full-vocabulary -1 is checked separately with its wire capability.
            PromptLogprobsAdmission::RejectPositiveStreaming => {
                "positive prompt_logprobs are unavailable when stream=true"
            }
        };
        Err(CompatibilityRejection::prompt_logprobs(
            self,
            if self.prompt_logprobs_admission == PromptLogprobsAdmission::UnverifiedTarget {
                RejectionKind::UnsupportedField
            } else {
                RejectionKind::UnsafeCombination
            },
            RejectionStage::Admission,
            &["Use stream=false", "Omit prompt_logprobs"],
        )
        .attach(invalid_argument_error(format!(
            "`prompt_logprobs` rejected by profile {self}: {reason}. \
             Use stream=false or omit prompt_logprobs.",
        ))))
    }
}

pub(crate) trait AdmissionRequest: Data {
    const ENDPOINT: Endpoint;
    fn client_stream(&self) -> bool;
    fn prompt_logprobs(&self) -> Option<i64>;
    fn extra_fields(&self) -> &std::collections::HashMap<String, Value>;

    fn sampling_extensions(&self) -> Map<String, Value> {
        [
            "allowed_token_ids",
            "bad_words_token_ids",
            "logprob_token_ids",
        ]
        .into_iter()
        .filter_map(|name| {
            self.extra_fields()
                .get(name)
                .map(|value| (name.to_owned(), value.clone()))
        })
        .collect()
    }
}

impl AdmissionRequest for NvCreateChatCompletionRequest {
    const ENDPOINT: Endpoint = Endpoint::Chat;
    fn client_stream(&self) -> bool {
        self.inner.stream.unwrap_or(false)
    }
    fn prompt_logprobs(&self) -> Option<i64> {
        self.common.prompt_logprobs
    }
    fn extra_fields(&self) -> &std::collections::HashMap<String, Value> {
        &self.unsupported_fields
    }
}

impl AdmissionRequest for NvCreateCompletionRequest {
    const ENDPOINT: Endpoint = Endpoint::Completion;
    fn client_stream(&self) -> bool {
        self.inner.stream.unwrap_or(false)
    }
    fn prompt_logprobs(&self) -> Option<i64> {
        self.common.prompt_logprobs
    }
    fn extra_fields(&self) -> &std::collections::HashMap<String, Value> {
        &self.unsupported_fields
    }
}

struct AdmissionEngine<Req: Data, Resp: Data> {
    inner: ServerStreamingEngine<Req, Resp>,
    profile: CompatibilityProfile,
    extension_target: Option<(ModelRuntimeConfig, Option<LegacyVllmRelease>)>,
}

impl<Req: AdmissionRequest, Resp: Data> AdmissionEngine<Req, Resp> {
    fn validate_full_prompt_count(&self, request: &Req) -> anyhow::Result<()> {
        let count = public_count_to_wire(request.prompt_logprobs()).map_err(|error| {
            CompatibilityRejection::prompt_logprobs(
                self.profile,
                RejectionKind::InvalidValue,
                RejectionStage::Admission,
                &["Use -1 or a non-negative count below 4294967295"],
            )
            .attach(error)
        })?;
        if count != Some(FULL_VOCAB_COUNT) {
            return Ok(());
        }
        if request.client_stream() {
            return Err(CompatibilityRejection::prompt_logprobs(
                self.profile,
                RejectionKind::UnsafeCombination,
                RejectionStage::Admission,
                &["Use stream=false", "Omit prompt_logprobs"],
            )
            .attach(invalid_argument_error(
                "`prompt_logprobs=-1` is unavailable when stream=true. Use stream=false.",
            )));
        }
        self.profile.validate_full_vocab_pipeline(
            self.extension_target.as_ref().map(|(runtime, _)| runtime),
        )
    }

    fn validate_with_metrics(
        &self,
        request: &Req,
        counter: &prometheus::IntCounterVec,
    ) -> anyhow::Result<()> {
        let record =
            |field, event| telemetry::record(counter, None, Some(self.profile), field, event);
        if let Err(error) = self
            .profile
            .validate(request)
            .and_then(|_| self.validate_full_prompt_count(request))
        {
            record("prompt_logprobs", Event::RejectCombination);
            return Err(error);
        }
        if request.prompt_logprobs().is_some() {
            record("prompt_logprobs", Event::AdmitPromptLogprobs);
        }
        if let Some((runtime, legacy_target)) = &self.extension_target {
            // Check before calling a lazy Python engine: errors must precede
            // HTTP/SSE commitment. Live selected-worker checks still run later.
            let fields = request.sampling_extensions();
            let lowered =
                match lower_sampling_passthrough_for_target(fields, runtime, *legacy_target) {
                    Ok(lowered) => lowered,
                    Err(error) => {
                        record("sampling_extensions", Event::RejectExtensions);
                        let field = error
                            .downcast_ref::<SamplingCapabilityFailure>()
                            .map(|failure| failure.field);
                        return Err(match field {
                            Some(field) => CompatibilityRejection::unsupported_sampling_field(
                                self.profile,
                                field,
                            )
                            .attach(error),
                            None => error,
                        });
                    }
                };
            if let Some(sampling) = lowered.get("sampling_options").and_then(Value::as_object) {
                for name in sampling.keys() {
                    record(name, Event::AdmitExtensions);
                    record(
                        name,
                        if lowered.contains_key("backend_extensions") {
                            Event::DualWriteExtensions
                        } else {
                            Event::LegacyExtensions
                        },
                    );
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl<Req: AdmissionRequest, Resp: Data> AsyncEngine<SingleIn<Req>, ManyOut<Resp>, Error>
    for AdmissionEngine<Req, Resp>
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Resp>, Error> {
        self.validate_with_metrics(&request, &PROTOCOL_DECISIONS)?;
        self.inner.generate(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entrypoint::ChatProcessorIdentity;
    use dynamo_runtime::pipeline::{AsyncEngineContextProvider, ResponseStream};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn context(endpoint: Endpoint) -> PipelineContext {
        PipelineContext::from_pipeline(
            endpoint,
            crate::model_type::ModelInput::Tokens,
            Some(crate::worker_type::WorkerType::Aggregated),
            None,
        )
    }

    #[test]
    fn full_vocab_admission_is_feature_scoped_and_precedes_inner_engine() {
        use crate::protocols::common::prompt_logprobs::VLLM_PROMPT_LOGPROBS_CAPABILITY;
        for limit in [None, Some(20), Some(-1), Some(32)] {
            let mut runtime = runtime("0.30.0");
            if let Some(limit) = limit {
                runtime.set_engine_specific(VLLM_PROMPT_LOGPROBS_CAPABILITY,json!({
                    "schema_version":1,"wire_count":"u32_max","max_logprobs":limit,"vocab_size":32,
                })).unwrap();
            }
            let profile = CompatibilityProfile::new(
                TargetAdmission::Vllm030,
                Endpoint::Chat,
                context(Endpoint::Chat),
            );
            let admits = matches!(limit, Some(-1 | 32));
            let engine = AdmissionEngine {
                inner: Arc::new(Recorder(AtomicUsize::new(0)))
                    as ServerStreamingEngine<NvCreateChatCompletionRequest, ()>,
                profile,
                extension_target: Some((runtime, None)),
            };
            for stream in [false, true] {
                for count in [-2, -1, 0] {
                    let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                        "model":"test", "messages":[], "stream":stream,"prompt_logprobs":count,
                    }))
                    .unwrap();
                    assert_eq!(
                        engine
                            .validate_with_metrics(&request, &telemetry::new_counter("test"))
                            .is_ok(),
                        count == 0 || (count == -1 && !stream && admits)
                    );
                }
            }
        }
    }

    fn runtime(version: &str) -> ModelRuntimeConfig {
        let mut runtime = ModelRuntimeConfig::default();
        runtime
            .set_engine_specific(
                VLLM_PROTOCOL_EXTENSIONS_CAPABILITY,
                json!({
                    "schema_version": 1, "target": "vllm", "engine_version": version,
                    "sampling_fields": [],
                }),
            )
            .unwrap();
        runtime
    }

    #[test]
    fn telemetry_tracks_real_admission_results_without_payload_labels() {
        use prometheus::TextEncoder;
        use prometheus::core::Collector;

        let counter = telemetry::new_counter("test");
        let mut current = runtime("0.30.0");
        current
            .set_engine_specific(
                VLLM_PROTOCOL_EXTENSIONS_CAPABILITY,
                json!({
                    "schema_version": 1, "target": "vllm", "engine_version": "0.30.0",
                    "sampling_fields": ["allowed_token_ids"],
                }),
            )
            .unwrap();
        for (runtime, legacy, payload, should_pass) in [
            (
                current.clone(),
                None,
                json!({"prompt_logprobs": 1, "stream": true}),
                false,
            ),
            (
                current.clone(),
                None,
                json!({"prompt_logprobs": 0, "stream": true}),
                true,
            ),
            (
                current.clone(),
                None,
                json!({"allowed_token_ids": [31415]}),
                true,
            ),
            (
                current,
                None,
                json!({"bad_words_token_ids": [[31415]]}),
                false,
            ),
            (
                ModelRuntimeConfig::default(),
                Some(LegacyVllmRelease::Dynamo14),
                json!({"allowed_token_ids": [31415]}),
                true,
            ),
        ] {
            let engine = AdmissionEngine {
                inner: Arc::new(Recorder(AtomicUsize::new(0)))
                    as ServerStreamingEngine<NvCreateChatCompletionRequest, ()>,
                profile: CompatibilityProfile::new(
                    TargetAdmission::from_runtime(&runtime),
                    Endpoint::Chat,
                    context(Endpoint::Chat),
                ),
                extension_target: Some((runtime, legacy)),
            };
            let mut body = json!({"model": "private-model", "messages": []});
            body.as_object_mut()
                .unwrap()
                .extend(payload.as_object().unwrap().clone());
            let request = serde_json::from_value(body).unwrap();
            assert_eq!(
                engine.validate_with_metrics(&request, &counter).is_ok(),
                should_pass
            );
        }
        let families = counter.collect();
        let metrics = families[0].get_metric();
        assert_eq!(metrics.len(), 7);
        for metric in metrics {
            assert_eq!(metric.get_counter().as_ref().unwrap().value(), 1.0);
        }
        let text = TextEncoder::new().encode_to_string(&families).unwrap();
        for expected in [
            "reason=\"prompt_stream_combination\"",
            "reason=\"prompt_admission_passed\"",
            "reason=\"extension_contract_passed\"",
            "reason=\"dual_write_extension_envelope\"",
            "reason=\"legacy_extension_envelope\"",
            "reason=\"extension_bundle_rejected\"",
            "field=\"sampling_extensions\"",
        ] {
            assert!(text.contains(expected), "{text}");
        }
        assert!(!text.contains("31415"));
        assert!(!text.contains("private-model"));
        assert!(
            !text.contains("field=\"bad_words_token_ids\""),
            "bundle rejection must not claim per-field attribution"
        );
    }

    #[test]
    fn selects_exact_versions_without_inventing_legacy_or_other_backend_identity() {
        for (version, expected) in [
            ("0.29.0", TargetAdmission::Vllm029),
            ("0.30.0", TargetAdmission::Vllm030),
            ("0.30.0+custom", TargetAdmission::Unverified),
            ("0.30.0.dev1", TargetAdmission::Unverified),
            ("0.31.0", TargetAdmission::Unverified),
            ("", TargetAdmission::Unverified),
        ] {
            assert_eq!(TargetAdmission::from_runtime(&runtime(version)), expected);
        }
        assert_eq!(
            TargetAdmission::from_runtime(&ModelRuntimeConfig::default()),
            TargetAdmission::Unidentified
        );
        for bad in [
            json!(null),
            json!(false),
            json!({"schema_version": 2}),
            json!({"schema_version":1,"target":"sglang","engine_version":"0.30.0","sampling_fields":[]}),
        ] {
            let mut runtime = ModelRuntimeConfig::default();
            runtime
                .set_engine_specific(VLLM_PROTOCOL_EXTENSIONS_CAPABILITY, bad)
                .unwrap();
            assert_eq!(
                TargetAdmission::from_runtime(&runtime),
                TargetAdmission::Unverified
            );
        }
    }

    #[test]
    fn exact_rule_releases_match_inspected_source_pins() {
        // Guard against a release bump leaving stale admission selectors. The
        // inventory and policy are intentionally separate: a declaration is not
        // proof of end-to-end support.
        let pins: serde_json::Value = serde_json::from_str(include_str!("vllm_pins.json")).unwrap();
        for (version, revision) in [
            ("0.29.0", "98dff2a81d747d1dba01a47f939f48c3526d4206"),
            ("0.30.0", "ced6857afa0ea7b2e3f0846a62e1394e90f15607"),
        ] {
            let profile = CompatibilityProfile::new(
                TargetAdmission::from_runtime(&runtime(version)),
                Endpoint::Chat,
                context(Endpoint::Chat),
            );
            assert_eq!(profile.upstream_commit, Some(revision));
            assert!(
                pins["versions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|pin| pin["version"] == version && pin["commit"] == revision)
            );
        }
    }

    #[test]
    fn validates_original_client_mode_for_both_endpoints_and_preserves_zero() {
        for target in [
            TargetAdmission::Unidentified,
            TargetAdmission::Unverified,
            TargetAdmission::Vllm029,
            TargetAdmission::Vllm030,
        ] {
            for stream in [None, Some(false), Some(true)] {
                for count in [None, Some(0), Some(1), Some(u32::MAX)] {
                    let chat: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                        "model":"test", "messages":[], "stream":stream, "prompt_logprobs":count,
                        "nvext":{"extra_fields":["prompt_logprobs"]},
                    }))
                    .unwrap();
                    let completion: NvCreateCompletionRequest = serde_json::from_value(json!({
                        "model":"test", "prompt":"test", "stream":stream, "prompt_logprobs":count,
                    }))
                    .unwrap();
                    let reject = target != TargetAdmission::Unidentified
                        && stream == Some(true)
                        && count.is_some_and(|n| n > 0);
                    assert_eq!(
                        CompatibilityProfile::new(target, Endpoint::Chat, context(Endpoint::Chat))
                            .validate(&chat)
                            .is_err(),
                        reject
                    );
                    assert_eq!(
                        CompatibilityProfile::new(
                            target,
                            Endpoint::Completion,
                            context(Endpoint::Completion)
                        )
                        .validate(&completion)
                        .is_err(),
                        reject
                    );
                }
            }
        }
    }

    struct Recorder(AtomicUsize);

    async fn check_extension_admission<Req>(endpoint: Endpoint)
    where
        Req: AdmissionRequest + serde::de::DeserializeOwned,
    {
        use crate::{model_type::ModelInput, worker_type::WorkerType};

        for factory in [
            None,
            Some(ChatProcessorIdentity::Custom),
            Some(ChatProcessorIdentity::Vllm),
            Some(ChatProcessorIdentity::Sglang),
        ] {
            let pipeline = PipelineContext::from_pipeline(
                endpoint,
                ModelInput::Tokens,
                Some(WorkerType::Aggregated),
                factory,
            );
            for legacy_target in [None, Some(LegacyVllmRelease::Dynamo14)] {
                for capability in [
                    None,
                    Some(json!(null)),
                    Some(json!({
                        "schema_version": 1, "target": "vllm", "engine_version": "0.30.0",
                        "sampling_fields": [],
                    })),
                    Some(json!({
                        "schema_version": 1, "target": "vllm", "engine_version": "0.30.0",
                        "sampling_fields": ["allowed_token_ids", "bad_words_token_ids", "logprob_token_ids"],
                    })),
                ] {
                    let mut runtime = ModelRuntimeConfig::default();
                    if let Some(value) = &capability {
                        runtime
                            .set_engine_specific(VLLM_PROTOCOL_EXTENSIONS_CAPABILITY, value)
                            .unwrap();
                    }
                    let recorder = Arc::new(Recorder(AtomicUsize::new(0)));
                    let engine = TargetAdmission::from_runtime(&runtime).wrap(
                        recorder.clone(),
                        pipeline,
                        &runtime,
                        legacy_target,
                    );
                    for field in [
                        "allowed_token_ids",
                        "bad_words_token_ids",
                        "logprob_token_ids",
                    ] {
                        for stream in [false, true] {
                            for present in [false, true] {
                                let mut value = json!({
                                    "model": "test", "messages": [], "prompt": "test", "stream": stream,
                                });
                                if present {
                                    value[field] = if field == "bad_words_token_ids" {
                                        json!([[31415]])
                                    } else {
                                        json!([31415])
                                    };
                                }
                                let request: Req = serde_json::from_value(value).unwrap();
                                let supports = match &capability {
                                    Some(value) => value["sampling_fields"]
                                        .as_array()
                                        .is_some_and(|fields| fields.contains(&json!(field))),
                                    None => legacy_target
                                        .is_some_and(|target| target.supports_field(field)),
                                };
                                let before = recorder.0.load(Ordering::SeqCst);
                                let result = engine
                                    .generate(SingleIn::with_id_and_metadata(
                                        request,
                                        "admission-context".into(),
                                        Default::default(),
                                    ))
                                    .await;
                                if present && !supports {
                                    let Err(error) = result else {
                                        panic!("unsupported extension reached lazy engine")
                                    };
                                    assert!(!error.to_string().contains("31415"));
                                    assert_eq!(recorder.0.load(Ordering::SeqCst), before);
                                } else {
                                    assert!(result.is_ok());
                                    assert_eq!(recorder.0.load(Ordering::SeqCst), before + 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn extensions_are_admitted_before_rust_or_python_stream_creation() {
        check_extension_admission::<NvCreateChatCompletionRequest>(Endpoint::Chat).await;
        check_extension_admission::<NvCreateCompletionRequest>(Endpoint::Completion).await;
    }

    #[test]
    fn rule_is_pipeline_independent_but_errors_identify_the_actual_pipeline() {
        use crate::{model_type::ModelInput, worker_type::WorkerType};
        let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model":"private-model-must-not-appear", "messages":[],
            "stream":true, "prompt_logprobs":1,
        }))
        .unwrap();
        for target in [
            TargetAdmission::Vllm029,
            TargetAdmission::Vllm030,
            TargetAdmission::Unverified,
        ] {
            for input in [ModelInput::Tokens, ModelInput::Text] {
                for worker in [None, Some(WorkerType::Aggregated), Some(WorkerType::Decode)] {
                    for factory in [
                        None,
                        Some(ChatProcessorIdentity::Custom),
                        Some(ChatProcessorIdentity::Vllm),
                        Some(ChatProcessorIdentity::Sglang),
                    ] {
                        let pipeline =
                            PipelineContext::from_pipeline(Endpoint::Chat, input, worker, factory);
                        let profile = CompatibilityProfile::new(target, Endpoint::Chat, pipeline);
                        let message = profile.validate(&request).unwrap_err().to_string();
                        assert!(message.contains(&profile.to_string()), "{message}");
                        assert!(!message.contains("private-model"));
                        assert!(message.contains("stream=false"));
                        if target == TargetAdmission::Unverified {
                            assert!(message.contains("no verified admission rule"));
                        } else {
                            assert!(message.contains("positive prompt_logprobs are unavailable"));
                        }
                    }
                }
            }
        }
    }

    #[async_trait]
    impl<Req: AdmissionRequest> AsyncEngine<SingleIn<Req>, ManyOut<()>, Error> for Recorder {
        async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<()>, Error> {
            self.0.fetch_add(1, Ordering::SeqCst);
            assert_eq!(request.context().id(), "admission-context");
            Ok(ResponseStream::new(
                Box::pin(futures::stream::empty()),
                request.context(),
            ))
        }
    }

    #[tokio::test]
    async fn rejection_precedes_generate_and_preserves_context_on_admitted_requests() {
        let recorder = Arc::new(Recorder(AtomicUsize::new(0)));
        let engine = TargetAdmission::Vllm030.wrap(
            recorder.clone(),
            context(Endpoint::Chat),
            &runtime("0.30.0"),
            None,
        );
        for count in [1, 0] {
            let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                "model":"test", "messages":[], "stream":true, "prompt_logprobs":count,
            }))
            .unwrap();
            let request = SingleIn::with_id_and_metadata(
                request,
                "admission-context".into(),
                Default::default(),
            );
            let result = engine.generate(request).await;
            if count == 1 {
                let Err(error) = result else {
                    panic!("expected rejection before stream creation")
                };
                let message = error.to_string();
                assert!(
                    message.contains("prompt_logprobs")
                        && message.contains("vllm/0.30.0")
                        && message.contains("/v1/chat/completions"),
                    "{message}"
                );
                assert_eq!(recorder.0.load(Ordering::SeqCst), 0);
            } else {
                assert!(result.is_ok());
                assert_eq!(recorder.0.load(Ordering::SeqCst), 1);
            }
        }
    }
}
