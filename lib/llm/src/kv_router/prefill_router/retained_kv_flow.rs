// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connect public tokenized requests to engine-owned, one-shot decoder exports.

use std::sync::Arc;
use std::time::Instant;

use arc_swap::ArcSwapOption;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, ManyOut, ResponseStream};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::retained_kv::{Catalog, Donor, MAX_TOKENS, PRIVATE_KEY, Scope, Source, VERSION};
use super::{PrefillBinding, PrefillRouter};
use crate::discovery::WorkerSetTargetId;
use crate::kv_router::RoutingHost;
use crate::protocols::common::FinishReason;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use crate::protocols::common::timing::RequestTracker;

pub(super) fn strip_private_request(request: &mut PreprocessedRequest) {
    remove_private(&mut request.extra_args);
}

fn remove_private(extra: &mut Option<Value>) -> Option<Value> {
    extra.as_mut()?.as_object_mut()?.remove(PRIVATE_KEY)
}

fn set_private(request: &mut PreprocessedRequest, params: Option<&Value>) {
    let extra = request.extra_args.get_or_insert_with(|| json!({}));
    if let Some(extra) = extra.as_object_mut() {
        let mut hint = json!({"version": VERSION, "capture": true});
        hint["cache_salt"] = request
            .routing
            .as_ref()
            .and_then(|routing| routing.cache_namespace.as_ref())
            .map_or(Value::Null, |salt| Value::String(salt.clone()));
        if let Some(params) = params {
            hint["kv_transfer_params"] = params.clone();
        }
        extra.insert(PRIVATE_KEY.into(), hint);
    }
}

fn eligible(request: &PreprocessedRequest) -> bool {
    request.mdc_sum.is_some()
        && !request.token_ids.is_empty()
        && request.token_ids.len() <= MAX_TOKENS
        && request.prompt_embeds.is_none()
        && request.multi_modal_data.is_none()
        && request.multi_modal_uuids.is_none()
        && request.mm_routing_info.is_none()
        && request.encoder_result.is_none()
        && request.sampling_options.n.unwrap_or(1) == 1
        && request.sampling_options.best_of.unwrap_or(1) == 1
        && request
            .routing
            .as_ref()
            .is_none_or(|r| r.lora_name.is_none())
        && request
            .extra_args
            .as_ref()
            .is_none_or(|extra| extra.is_object() && extra.get("vllm_tito").is_none())
}

impl PrefillRouter {
    pub(super) fn prepare_retained_kv(
        &self,
        request: &mut PreprocessedRequest,
        binding: &PrefillBinding,
        request_id: &str,
    ) -> Option<RetainedFlow> {
        let catalog = self.retained_kv.as_ref()?.clone();
        if !eligible(request) {
            return None;
        }
        let decode_host = self.decode_routing_host.get()?.clone();
        // The operator's domain bounds reuse authorization. A caller's salt only
        // subdivides that domain and is never treated as an authenticated identity.
        let cache_namespace = serde_json::to_string(&(
            catalog.trust_domain(),
            request
                .routing
                .as_ref()
                .and_then(|r| r.cache_namespace.as_ref()),
        ))
        .ok()?;
        if cache_namespace.len() > 4096 {
            return None;
        }
        let generation = format!(
            "{}:{}:{}:{:?}",
            self.namespace,
            self.model_name,
            request.mdc_sum.as_deref()?,
            binding.target_id
        );
        let scope = Scope {
            pipeline: generation.clone(),
            cache_namespace: Some(cache_namespace.clone()),
        };
        request.routing_mut().cache_namespace = Some(cache_namespace);
        let donor = catalog.claim(&scope, &request.token_ids, Instant::now());
        let mut flow = RetainedFlow {
            catalog,
            scope,
            generation,
            binding: self.binding.clone(),
            target_id: binding.target_id.clone(),
            cancel_token: self.cancel_token.clone(),
            decode_host,
            prompt: request.token_ids.clone(),
            tracker: request.tracker.clone()?,
            donor,
            request_id: request_id.to_owned(),
        };
        if flow
            .donor
            .as_ref()
            .is_some_and(|donor| !flow.valid_source(&donor.source))
        {
            flow.donor = None;
        }
        set_private(request, None);
        tracing::debug!(
            request_id,
            event = "retained_kv_claim",
            hit = flow.donor.is_some(),
            input_tokens = flow.prompt.len(),
            "Checked server-managed retained KV"
        );
        Some(flow)
    }
}

pub(super) struct RetainedFlow {
    catalog: Arc<Catalog>,
    scope: Scope,
    generation: String,
    binding: Arc<ArcSwapOption<PrefillBinding>>,
    target_id: WorkerSetTargetId,
    cancel_token: CancellationToken,
    decode_host: Arc<RoutingHost>,
    prompt: Arc<Vec<u32>>,
    tracker: Arc<RequestTracker>,
    donor: Option<Donor>,
    request_id: String,
}

impl RetainedFlow {
    fn current(&self) -> bool {
        !self.cancel_token.is_cancelled()
            && self
                .binding
                .load()
                .as_ref()
                .is_some_and(|binding| binding.target_id == self.target_id)
    }

    fn valid_source(&self, source: &Source) -> bool {
        self.current()
            && source.generation == self.generation
            && self.decode_host.is_worker_available(source.worker_id)
    }

    pub(super) fn prepare_prefill(&self, request: &mut PreprocessedRequest) {
        request.routing_mut().cache_namespace = self.scope.cache_namespace.clone();
        set_private(
            request,
            self.donor.as_ref().map(|donor| &donor.kv_transfer_params),
        );
    }

    pub(super) fn revalidate_prefill(&self, request: &mut PreprocessedRequest) {
        if self
            .donor
            .as_ref()
            .is_some_and(|donor| !self.valid_source(&donor.source))
        {
            set_private(request, None);
        }
    }

    pub(super) fn observe(
        self,
        mut response: ManyOut<Annotated<LLMEngineOutput>>,
    ) -> ManyOut<Annotated<LLMEngineOutput>> {
        let context = response.context();
        let stream_context = context.clone();
        let stream = async_stream::stream! {
            let mut capture = Capture::default();
            while let Some(mut item) = response.next().await {
                capture.observe(&mut item);
                yield item;
            }
            if capture.successful && !capture.invalid
                && !stream_context.is_stopped() && !stream_context.is_killed()
                && self.current()
                && let (Some(payload), Some(worker_id)) = (capture.payload, self.tracker.decode_worker_id())
            {
                let source = Source { worker_id, generation: self.generation.clone() };
                if self.valid_source(&source) {
                    let stored = self.catalog.insert(
                        self.scope, source, &self.prompt, &capture.generated, &payload, Instant::now(),
                    );
                    tracing::debug!(
                        request_id = %self.request_id,
                        event = "retained_kv_export",
                        stored,
                        prompt_tokens = self.prompt.len(),
                        generated_tokens = capture.generated.len(),
                        "Observed private decoder export"
                    );
                }
            }
        };
        ResponseStream::new(Box::pin(stream), context)
    }
}

#[derive(Default)]
struct Capture {
    generated: Vec<u32>,
    payload: Option<Value>,
    successful: bool,
    invalid: bool,
}

impl Capture {
    fn observe(&mut self, item: &mut Annotated<LLMEngineOutput>) {
        if item.error.is_some() || item.event.as_deref() == Some("error") {
            self.invalid = true;
        }
        let Some(data) = item.data.as_mut() else {
            return;
        };
        let payload = remove_private(&mut data.extra_args);
        // Prefill annotations contain metadata, not decoder output tokens.
        if item
            .event
            .as_deref()
            .is_some_and(|event| event.starts_with("prefill"))
        {
            return;
        }
        if self.successful
            || data.index.is_some_and(|index| index != 0)
            || self.generated.len().saturating_add(data.token_ids.len()) > MAX_TOKENS
        {
            self.invalid = true;
            return;
        }
        self.generated.extend_from_slice(&data.token_ids);
        if let Some(reason) = &data.finish_reason {
            self.successful = matches!(
                reason,
                FinishReason::EoS | FinishReason::Length | FinishReason::Stop
            );
            self.invalid |= !self.successful;
            self.payload = payload;
        } else if payload.is_some() {
            self.invalid = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        component::Client,
        distributed::DistributedConfig,
        pipeline::{RouterMode, SingleIn},
    };

    use super::*;
    use crate::{
        discovery::{LoadThresholdHandle, ModelManager},
        kv_router::{RouterLoadSource, RoutingLoadContext, prefill_router::retained_kv::Config},
        protocols::common::{
            preprocessor::MmRoutingInfo,
            timing::{RequestPhase, WORKER_TYPE_DECODE},
        },
        session_affinity::SessionAffinityMode,
    };

    fn request() -> PreprocessedRequest {
        let mut request = PreprocessedRequest::builder()
            .model("test-model".to_owned())
            .token_ids(vec![1, 2])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        request.mdc_sum = Some("selected-mdc".into());
        request.tracker = Some(Arc::new(RequestTracker::new()));
        request.routing_mut().cache_namespace = Some("caller-partition".into());
        request.extra_args = Some(json!({
            "nvext": {"cache_salt": "caller-partition", "extra_fields": ["engine_data"]},
            "sampling_options": {"detokenize": false}
        }));
        request
    }

    fn payload() -> Value {
        json!({
            "version": VERSION,
            "prompt_token_ids": [1, 2],
            "kv_transfer_params": {
                "do_remote_decode": true,
                "do_remote_prefill": false,
                "remote_engine_id": "private-engine-canary",
                "remote_request_id": "private-request-canary",
                "remote_host": "10.0.0.7",
                "remote_port": 5600,
                "remote_block_ids": [[4, 5]],
                "remote_num_tokens": 3,
                "remote_blocks_expiry_time": 12345.0
            }
        })
    }

    fn output(
        ids: Vec<u32>,
        reason: Option<FinishReason>,
        private: Option<Value>,
    ) -> Annotated<LLMEngineOutput> {
        let mut extra = json!({"ordinary": "preserved"});
        if let Some(private) = private {
            extra[PRIVATE_KEY] = private;
        }
        Annotated::from_data(LLMEngineOutput {
            token_ids: ids,
            finish_reason: reason,
            index: Some(0),
            text: Some("unchanged output text".into()),
            extra_args: Some(extra),
            engine_data: Some(json!({"public_diagnostic": 8})),
            ..Default::default()
        })
    }

    fn terminal() -> Annotated<LLMEngineOutput> {
        output(vec![3], Some(FinishReason::Length), Some(payload()))
    }

    fn response(items: Vec<Annotated<LLMEngineOutput>>) -> ManyOut<Annotated<LLMEngineOutput>> {
        ResponseStream::new(
            Box::pin(futures::stream::iter(items)),
            SingleIn::new(()).context(),
        )
    }

    struct Fixture {
        router: Arc<PrefillRouter>,
        binding: Arc<PrefillBinding>,
        client: Client,
        _distributed: DistributedRuntime,
    }

    async fn fixture() -> Fixture {
        let runtime = Runtime::from_current().unwrap();
        let distributed = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = distributed
            .namespace(format!("retained-flow-{}", uuid::Uuid::new_v4()))
            .unwrap()
            .component("workers")
            .unwrap()
            .endpoint("generate");
        let client = endpoint.client().await.unwrap();
        client.override_discovered_instances(vec![7]);
        client.override_instance_avail(vec![7]);
        let load = RoutingLoadContext::start(
            client.clone(),
            RouterLoadSource::Decode,
            LoadThresholdHandle::new(Default::default()),
            &distributed.child_token(),
            None,
        )
        .await
        .unwrap();
        let host = Arc::new(
            RoutingHost::builder(client.clone(), RouterMode::RoundRobin)
                .load_context(load)
                .build()
                .await
                .unwrap(),
        );
        let mut router = PrefillRouter::disabled_with_session_affinity_mode(
            Arc::new(ModelManager::new()),
            RouterMode::RoundRobin,
            None,
            SessionAffinityMode::Hard,
        );
        let mutable = Arc::get_mut(&mut router).unwrap();
        mutable.retained_kv = Some(Arc::new(Catalog::new(
            Config::new("operator-domain".into()).unwrap(),
        )));
        mutable.namespace = "selected-namespace".into();
        mutable.model_name = "selected-model".into();
        let binding = Arc::new(PrefillBinding {
            target_id: WorkerSetTargetId::Committed {
                group: endpoint.id().to_string(),
                generation: 1,
            },
            endpoint_id: endpoint.id(),
            router: host.clone(),
            prefill_router_mode: RouterMode::RoundRobin,
        });
        router.binding.store(Some(binding.clone()));
        router.set_decode_routing_host(host).unwrap();
        // Initial endpoint monitoring can publish its empty discovery snapshot
        // while the async routing setup runs. Install the synthetic workers last.
        tokio::task::yield_now().await;
        client.override_discovered_instances(vec![7]);
        client.override_instance_avail(vec![7]);
        Fixture {
            router,
            binding,
            client,
            _distributed: distributed,
        }
    }

    async fn flow(fixture: &Fixture) -> (RetainedFlow, PreprocessedRequest) {
        let mut request = request();
        let flow = fixture
            .router
            .prepare_retained_kv(&mut request, &fixture.binding, "request")
            .unwrap();
        drop(flow.tracker.set_phase(RequestPhase::Decode).await);
        flow.tracker.record_worker(7, None, WORKER_TYPE_DECODE);
        (flow, request)
    }

    fn replace_binding(fixture: &Fixture) {
        fixture.router.binding.store(Some(Arc::new(PrefillBinding {
            target_id: WorkerSetTargetId::Committed {
                group: fixture.binding.endpoint_id.to_string(),
                generation: 2,
            },
            endpoint_id: fixture.binding.endpoint_id.clone(),
            router: fixture.binding.router.clone(),
            prefill_router_mode: RouterMode::RoundRobin,
        })));
    }

    #[tokio::test]
    async fn round_robin_pipeline_installs_decode_membership_for_retained_reuse() {
        tokio::time::timeout(Duration::from_secs(5), async {
            let fixture = fixture().await;
            let mut router = PrefillRouter::disabled_with_session_affinity_mode(
                fixture.router.model_manager.clone(),
                RouterMode::RoundRobin,
                None,
                SessionAffinityMode::Hard,
            );
            Arc::get_mut(&mut router).unwrap().retained_kv = fixture.router.retained_kv.clone();
            assert!(router.decode_routing_host.get().is_none());
            let load = RoutingLoadContext::start(
                fixture.client.clone(),
                RouterLoadSource::Decode,
                LoadThresholdHandle::new(Default::default()),
                &fixture._distributed.child_token(),
                None,
            )
            .await
            .unwrap();

            let _pipeline = crate::entrypoint::input::build_preprocessed_routing(
                &fixture.client,
                fixture.router.model_manager.clone(),
                RouterMode::RoundRobin,
                load,
                None,
                Some(router.clone()),
                None,
                false,
                None,
            )
            .await
            .unwrap();

            assert!(router.decode_routing_host.get().is_some());
        })
        .await
        .expect("round-robin routing pipeline construction timed out");
    }

    #[test]
    fn spoofed_private_request_is_removed_without_losing_supported_arguments() {
        let mut request = request();
        request.extra_args.as_mut().unwrap()[PRIVATE_KEY] = json!({
            "version": VERSION,
            "capture": true,
            "kv_transfer_params": {"remote_host": "attacker.invalid"},
            "cache_salt": "forged-domain"
        });
        strip_private_request(&mut request);
        let extra = request.extra_args.as_ref().unwrap();
        assert!(extra.get(PRIVATE_KEY).is_none());
        assert_eq!(extra["nvext"]["cache_salt"], "caller-partition");
        assert_eq!(extra["sampling_options"]["detokenize"], false);
        set_private(&mut request, None);
        let private = &request.extra_args.as_ref().unwrap()[PRIVATE_KEY];
        assert_eq!(private["version"], VERSION);
        assert_eq!(private["capture"], true);
        assert!(private.get("kv_transfer_params").is_none());
        assert_ne!(private["cache_salt"], "forged-domain");
    }

    #[test]
    fn capture_removes_private_metadata_before_yielding_and_preserves_public_data() {
        let mut capture = Capture::default();
        let mut item = terminal();
        capture.observe(&mut item);
        let data = item.data.as_ref().unwrap();
        assert!(data.extra_args.as_ref().unwrap().get(PRIVATE_KEY).is_none());
        assert_eq!(data.extra_args.as_ref().unwrap()["ordinary"], "preserved");
        assert_eq!(data.engine_data, Some(json!({"public_diagnostic": 8})));
        assert_eq!(data.token_ids, [3]);
        assert_eq!(data.text.as_deref(), Some("unchanged output text"));
        let serialized = serde_json::to_string(&item).unwrap();
        for private in [PRIVATE_KEY, "private-engine-canary", "10.0.0.7"] {
            assert!(!serialized.contains(private));
        }
        assert!(capture.successful);
        assert!(!capture.invalid);
        assert_eq!(capture.generated, [3]);
        assert_eq!(capture.payload, Some(payload()));
    }

    #[test]
    fn captures_raw_disjoint_deltas_and_rejects_nonterminal_exports_or_extra_choices() {
        let mut capture = Capture::default();
        capture.observe(&mut output(vec![3], None, None));
        capture.observe(&mut output(
            vec![4],
            Some(FinishReason::Stop),
            Some(payload()),
        ));
        assert_eq!(capture.generated, [3, 4]);
        assert!(capture.successful && !capture.invalid);

        let mut early = Capture::default();
        early.observe(&mut output(vec![3], None, Some(payload())));
        assert!(early.invalid);
        for index in [1, 2] {
            let mut ambiguous = Capture::default();
            let mut item = terminal();
            item.data.as_mut().unwrap().index = Some(index);
            ambiguous.observe(&mut item);
            assert!(ambiguous.invalid && !ambiguous.successful);
            assert!(
                item.data
                    .unwrap()
                    .extra_args
                    .unwrap()
                    .get(PRIVATE_KEY)
                    .is_none()
            );
        }
        let mut duplicate = Capture::default();
        duplicate.observe(&mut terminal());
        duplicate.observe(&mut terminal());
        assert!(duplicate.invalid);
    }

    #[test]
    fn late_typed_and_legacy_errors_reject_an_earlier_successful_export() {
        let mut legacy =
            Annotated::<LLMEngineOutput>::from_annotation("error", &"late failure").unwrap();
        legacy.error = None;
        for mut error in [Annotated::from_error("typed failure"), legacy] {
            let mut capture = Capture::default();
            capture.observe(&mut terminal());
            capture.observe(&mut error);
            assert!(capture.invalid);
        }
        for reason in [
            FinishReason::Cancelled,
            FinishReason::Error("decode failure".into()),
            FinishReason::ContentFilter,
        ] {
            let mut capture = Capture::default();
            capture.observe(&mut output(vec![3], Some(reason), Some(payload())));
            assert!(capture.invalid && !capture.successful);
        }
    }

    #[test]
    fn prefill_annotations_are_scrubbed_and_do_not_supply_decoder_tokens_or_exports() {
        let mut capture = Capture::default();
        let mut annotation = terminal();
        annotation.event = Some("prefill_output".into());
        capture.observe(&mut annotation);
        assert!(capture.generated.is_empty());
        assert!(capture.payload.is_none());
        assert!(!capture.successful && !capture.invalid);
        assert!(
            annotation
                .data
                .unwrap()
                .extra_args
                .unwrap()
                .get(PRIVATE_KEY)
                .is_none()
        );
    }

    #[test]
    fn generated_token_capture_is_bounded_and_preserves_the_original_item() {
        let mut capture = Capture::default();
        capture.observe(&mut output(vec![3; MAX_TOKENS], None, None));
        let mut excess = terminal();
        capture.observe(&mut excess);
        assert!(capture.invalid);
        assert_eq!(capture.generated.len(), MAX_TOKENS);
        assert_eq!(excess.data.as_ref().unwrap().token_ids, [3]);
        assert!(
            excess
                .data
                .unwrap()
                .extra_args
                .unwrap()
                .get(PRIVATE_KEY)
                .is_none()
        );
    }

    #[test]
    fn eligibility_excludes_unsupported_identity_media_and_choice_profiles() {
        assert!(eligible(&request()));
        let mut variants = Vec::new();
        let mut missing_identity = request();
        missing_identity.mdc_sum = None;
        variants.push(missing_identity);
        let mut empty = request();
        empty.token_ids = Arc::new(Vec::new());
        variants.push(empty);
        let mut large = request();
        large.token_ids = Arc::new(vec![1; MAX_TOKENS + 1]);
        variants.push(large);
        let mut embeds = request();
        embeds.prompt_embeds = Some("opaque".into());
        variants.push(embeds);
        let mut media = request();
        media.multi_modal_data = Some(HashMap::new());
        variants.push(media);
        let mut uuids = request();
        uuids.multi_modal_uuids = Some(HashMap::new());
        variants.push(uuids);
        let mut mm_routing = request();
        mm_routing.mm_routing_info = Some(MmRoutingInfo {
            routing_token_ids: vec![1],
            block_mm_infos: vec![None],
            expanded_prompt_len: 1,
        });
        variants.push(mm_routing);
        let mut encoder = request();
        encoder.encoder_result = Some(json!({}));
        variants.push(encoder);
        let mut lora = request();
        lora.routing_mut().lora_name = Some("adapter".into());
        variants.push(lora);
        let mut choices = request();
        choices.sampling_options.n = Some(2);
        variants.push(choices);
        let mut best_of = request();
        best_of.sampling_options.best_of = Some(2);
        variants.push(best_of);
        let mut native = request();
        native.extra_args = Some(json!({"vllm_tito": {}}));
        variants.push(native);
        let mut malformed = request();
        malformed.extra_args = Some(json!([]));
        variants.push(malformed);
        for unsupported in variants {
            assert!(!eligible(&unsupported));
        }
    }

    #[tokio::test]
    async fn public_salt_is_preserved_and_effective_domain_salt_reaches_both_role_hints() {
        let fixture = fixture().await;
        let (flow, decode) = flow(&fixture).await;
        let mut prefill = request();
        flow.prepare_prefill(&mut prefill);
        let effective = decode
            .routing
            .as_ref()
            .unwrap()
            .cache_namespace
            .as_ref()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(effective).unwrap(),
            json!(["operator-domain", "caller-partition"])
        );
        for role in [&decode, &prefill] {
            assert_eq!(
                role.routing.as_ref().unwrap().cache_namespace.as_ref(),
                Some(effective)
            );
            let extra = role.extra_args.as_ref().unwrap();
            assert_eq!(extra[PRIVATE_KEY]["cache_salt"], effective.as_str());
            assert_eq!(extra["nvext"]["cache_salt"], "caller-partition");
        }
        let mut other = request();
        other.routing_mut().cache_namespace = Some("other-partition".into());
        let other = fixture
            .router
            .prepare_retained_kv(&mut other, &fixture.binding, "other")
            .unwrap();
        assert_ne!(flow.scope.cache_namespace, other.scope.cache_namespace);
    }

    #[tokio::test]
    async fn source_worker_and_generation_are_revalidated_before_native_dispatch() {
        let fixture = fixture().await;
        let (mut flow, _) = flow(&fixture).await;
        let source = Source {
            worker_id: 7,
            generation: flow.generation.clone(),
        };
        assert!(flow.valid_source(&source));
        assert!(!flow.valid_source(&Source {
            worker_id: 99,
            ..source.clone()
        }));
        assert!(!flow.valid_source(&Source {
            generation: "retired-generation".into(),
            ..source.clone()
        }));
        flow.donor = Some(Donor {
            source,
            kv_transfer_params: payload()["kv_transfer_params"].clone(),
        });
        let mut prefill = request();
        flow.prepare_prefill(&mut prefill);
        assert!(
            prefill.extra_args.as_ref().unwrap()[PRIVATE_KEY]
                .get("kv_transfer_params")
                .is_some()
        );
        fixture.client.override_discovered_instances(vec![]);
        fixture.client.override_instance_avail(vec![]);
        flow.revalidate_prefill(&mut prefill);
        let private = &prefill.extra_args.as_ref().unwrap()[PRIVATE_KEY];
        assert_eq!(private["capture"], true);
        assert!(private.get("kv_transfer_params").is_none());
    }

    #[tokio::test]
    async fn target_succession_and_router_cancellation_fence_existing_flow() {
        let fixture = fixture().await;
        let (flow, _) = flow(&fixture).await;
        assert!(flow.current());
        replace_binding(&fixture);
        assert!(!flow.current());
        fixture.router.binding.store(Some(fixture.binding.clone()));
        assert!(flow.current());
        fixture.router.cancel_token.cancel();
        assert!(!flow.current());
    }

    #[tokio::test]
    async fn export_waits_for_clean_stream_end_and_uses_router_observed_worker() {
        let fixture = fixture().await;
        let (flow, _) = flow(&fixture).await;
        let catalog = flow.catalog.clone();
        let scope = flow.scope.clone();
        let mut response = flow.observe(response(vec![terminal()]));
        let item = response.next().await.unwrap();
        assert!(
            item.data
                .unwrap()
                .extra_args
                .unwrap()
                .get(PRIVATE_KEY)
                .is_none()
        );
        assert!(catalog.claim(&scope, &[1, 2, 3], Instant::now()).is_none());
        assert!(response.next().await.is_none());
        let donor = catalog
            .claim(&scope, &[1, 2, 3, 9], Instant::now())
            .unwrap();
        assert_eq!(donor.source.worker_id, 7);
        assert!(catalog.claim(&scope, &[1, 2, 3], Instant::now()).is_none());
    }

    #[tokio::test]
    async fn late_error_or_retired_binding_prevents_terminal_export() {
        for retire in [false, true] {
            let fixture = fixture().await;
            let (flow, _) = flow(&fixture).await;
            let catalog = flow.catalog.clone();
            let scope = flow.scope.clone();
            let mut items = vec![terminal()];
            if !retire {
                items.push(Annotated::from_annotation("error", &"late legacy error").unwrap());
            }
            let mut response = flow.observe(response(items));
            assert!(response.next().await.is_some());
            if retire {
                replace_binding(&fixture);
            }
            while response.next().await.is_some() {}
            assert!(catalog.claim(&scope, &[1, 2, 3], Instant::now()).is_none());
        }
    }

    #[tokio::test]
    async fn dropping_or_cancelling_the_response_does_not_commit_the_export() {
        for cancellation in ["drop", "kill", "stop"] {
            let fixture = fixture().await;
            let (flow, _) = flow(&fixture).await;
            let catalog = flow.catalog.clone();
            let scope = flow.scope.clone();
            let mut response = flow.observe(response(vec![terminal()]));
            assert!(response.next().await.is_some());
            if cancellation != "drop" {
                if cancellation == "kill" {
                    response.context().kill();
                } else {
                    response.context().stop_generating();
                }
                assert!(response.next().await.is_none());
            }
            drop(response);
            assert!(catalog.claim(&scope, &[1, 2, 3], Instant::now()).is_none());
        }
    }
}
