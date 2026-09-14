// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_kv_router::conditional_disagg::ConditionalDisaggDecisionInput;
use dynamo_kv_router::selector::PrefillAction;
use dynamo_runtime::pipeline::SingleIn;

use super::PrefillRouter;
use crate::kv_router::routing_host::{RoutePlan, RoutePlanSignals, is_cancelled};
use crate::protocols::common::{llm_backend::PreprocessedRequest, timing::RequestPhase};

/// An explicit plugin choice must not silently fall back to another execution path.
#[derive(Debug, thiserror::Error)]
#[error("prefill path action failed: {0}")]
pub(super) struct PrefillActionError(#[source] pub anyhow::Error);

/// An admitted decode route selected by `RoutingHost` together with the
/// conditional-disagg policy's diagnostic signals.
pub(super) struct ConditionalDisaggDecodeDecision {
    pub plan: RoutePlan,
    pub overlap_tokens: usize,
    pub net_new_tokens: usize,
}

fn decode_gate_allows_bypass(
    policy_says_bypass: bool,
    decode_gate_configured: bool,
    decode_busy: Option<bool>,
) -> bool {
    policy_says_bypass && (!decode_gate_configured || matches!(decode_busy, Some(false)))
}

impl PrefillRouter {
    /// Preview one decode route, then admit it only when the topology policy
    /// chooses local decode.
    pub(super) async fn plan_conditional_disagg_decode(
        &self,
        request: &mut SingleIn<PreprocessedRequest>,
        request_id: &str,
    ) -> Result<Option<ConditionalDisaggDecodeDecision>> {
        // Conditional disagg chooses a cache-hot decode worker, so it only
        // applies to a KV-routed decode set.
        if !self.decode_router_mode.is_kv_routing() {
            return Ok(None);
        }

        let requirements = self.path_planning();
        let prefill_pinned = request
            .routing
            .as_ref()
            .is_some_and(|routing| routing.prefill_worker_id.is_some());
        if prefill_pinned && requirements.is_none() {
            tracing::debug!(
                request_id,
                "Skipping conditional disagg because request has a preselected prefill worker"
            );
            return Ok(None);
        }

        let Some(decode_host) = self.decode_routing_host.get() else {
            tracing::debug!(
                request_id,
                "Skipping conditional disagg because decode RoutingHost is unavailable"
            );
            return Ok(None);
        };

        let (routing_token_ids, _) = request.block_mm_routing_info();
        if routing_token_ids.is_empty() {
            return Ok(None);
        }

        let probed_prefill_busy = if requirements.is_some_and(|r| r.prefill_load) {
            self.probe_prefill_busy(request, request_id).await?
        } else {
            None
        };
        let preview = if requirements.is_some() {
            decode_host
                .preview_prefill_path(request, probed_prefill_busy)
                .await
                .map_err(PrefillActionError)?
        } else {
            decode_host
                .preview_kv_route(request, RequestPhase::Decode)
                .await?
        };
        let action = preview.prefill;
        if action == PrefillAction::Remote {
            return Ok(None);
        }
        if prefill_pinned {
            if action == PrefillAction::LocalOnDecode {
                return Err(PrefillActionError(anyhow::anyhow!(
                    "local prefill conflicts with explicit prefill worker pin"
                ))
                .into());
            }
            return Ok(None);
        }
        let signals = preview.signals;
        let mut input =
            ConditionalDisaggDecisionInput::new(routing_token_ids.len(), signals.cached_tokens);
        if action == PrefillAction::Default
            && self.conditional_disagg_policy.needs_prefill_worker_busy()
        {
            let busy = if requirements.is_some_and(|r| r.prefill_load) {
                probed_prefill_busy
            } else {
                self.probe_prefill_busy(request, request_id).await?
            };
            input = input.with_prefill_chosen_worker_busy(busy);
        }
        let net_new_tokens = input.net_new_tokens();
        let overlap_tokens =
            (signals.overlap_blocks as usize) * (decode_host.kv_router().block_size() as usize);

        let policy_says_bypass = action == PrefillAction::LocalOnDecode
            || self
                .conditional_disagg_policy
                .should_bypass_remote_prefill(input)
                .await;
        let decode_gate_configured = action == PrefillAction::Default
            && self.conditional_disagg_decode_busy_threshold.is_some();
        let decode_busy = if policy_says_bypass && decode_gate_configured {
            self.conditional_disagg_decode_busy_threshold
                .and_then(|threshold| signals.decode_load_exceeds(threshold))
        } else {
            None
        };
        input = input.with_decode_chosen_worker_busy(decode_busy);

        let bypass =
            decode_gate_allows_bypass(policy_says_bypass, decode_gate_configured, decode_busy);
        let decode_gate_decision = if !policy_says_bypass {
            "bypass_declined_by_policy"
        } else if !decode_gate_configured {
            "bypass_allowed_decode_gate_disabled"
        } else if decode_busy.is_none() {
            "bypass_denied_decode_busy_unknown"
        } else if decode_busy == Some(true) {
            "bypass_denied_decode_busy"
        } else {
            "bypass_allowed_decode_not_busy"
        };

        log_conditional_disagg_decision(
            request_id,
            signals,
            net_new_tokens,
            overlap_tokens,
            input,
            decode_busy,
            self.conditional_disagg_decode_busy_threshold,
            decode_gate_decision,
            bypass,
        );

        if bypass {
            if action == PrefillAction::LocalOnDecode {
                // Charge the local prefill even if a caller requested decode-only accounting.
                request
                    .router_config_override
                    .get_or_insert_with(Default::default)
                    .track_prefill_tokens = Some(true);
            }
            let plan = decode_host
                .plan_kv_route_from_preview(request, preview)
                .await;
            let plan = if action == PrefillAction::LocalOnDecode {
                plan.map_err(PrefillActionError)?
            } else {
                plan?
            };
            return Ok(Some(ConditionalDisaggDecodeDecision {
                plan,
                overlap_tokens,
                net_new_tokens,
            }));
        }

        Ok(None)
    }

    async fn probe_prefill_busy(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        request_id: &str,
    ) -> Result<Option<bool>> {
        match self.peek_prefill_chosen_worker_busy(request).await {
            Ok(busy) => Ok(busy),
            Err(error) if is_cancelled(&error) => Err(error),
            Err(error) => {
                tracing::debug!(request_id, %error, "Conditional disagg prefill-load probe failed; treating load as unavailable");
                Ok(None)
            }
        }
    }

    async fn peek_prefill_chosen_worker_busy(
        &self,
        request: &SingleIn<PreprocessedRequest>,
    ) -> Result<Option<bool>> {
        let Some(threshold) = self.conditional_disagg_prefill_busy_threshold else {
            return Ok(None);
        };
        let Some(binding) = self.binding.load_full() else {
            return Ok(None);
        };
        Ok(Some(
            binding
                .router
                .prefill_worker_busy(request, threshold)
                .await?,
        ))
    }
}

#[expect(clippy::too_many_arguments)]
fn log_conditional_disagg_decision(
    request_id: &str,
    signals: RoutePlanSignals,
    net_new_tokens: usize,
    overlap_tokens: usize,
    input: ConditionalDisaggDecisionInput,
    decode_busy: Option<bool>,
    decode_busy_threshold: Option<f64>,
    decode_gate_decision: &str,
    bypass: bool,
) {
    tracing::debug!(
        request_id,
        worker_id = signals.worker.worker_id,
        dp_rank = signals.worker.dp_rank,
        prompt_tokens = input.prompt_tokens,
        net_new_tokens,
        overlap_tokens,
        prefill_chosen_worker_busy = ?input.prefill_chosen_worker_busy,
        decode_chosen_worker_busy = ?decode_busy,
        cached_tokens = signals.cached_tokens,
        potential_decode_blocks = signals.potential_decode_blocks,
        decode_busy_threshold = ?decode_busy_threshold,
        decode_gate_decision,
        bypass,
        "Conditional disagg decision"
    );
}

#[cfg(test)]
mod tests {
    use super::decode_gate_allows_bypass;

    #[test]
    fn decode_gate_calm_and_policy_bypass_allows_bypass() {
        assert!(decode_gate_allows_bypass(true, true, Some(false)));
    }

    #[test]
    fn decode_gate_busy_vetoes_policy_bypass() {
        assert!(!decode_gate_allows_bypass(true, true, Some(true)));
    }

    #[test]
    fn decode_gate_does_not_bypass_when_policy_declines() {
        assert!(!decode_gate_allows_bypass(false, true, Some(false)));
        assert!(!decode_gate_allows_bypass(false, true, Some(true)));
        assert!(!decode_gate_allows_bypass(false, true, None));
    }

    #[test]
    fn disabled_decode_gate_does_not_block_bypass() {
        assert!(decode_gate_allows_bypass(true, false, None));
        assert!(decode_gate_allows_bypass(true, false, Some(true)));
    }

    #[test]
    fn configured_decode_gate_signal_unavailable_vetoes_bypass() {
        assert!(!decode_gate_allows_bypass(true, true, None));
    }
}

#[cfg(test)]
mod plugin_tests {
    use super::*;
    use crate::{
        discovery::ModelManager,
        kv_router::{
            SelectionPolicySource,
            routing_host::{RoutingHost, tests::router_with_worker_policy_updates},
        },
        local_model::runtime_config::ModelRuntimeConfig,
        session_affinity::SessionAffinityMode,
    };
    use dynamo_kv_router::{
        config::KvRouterConfig,
        scheduling::WorkerSelectionPolicyError,
        selector::{
            PathPlanningRequirements, RouteChoice, WorkerInputView, WorkerPicker,
            WorkerSelectionContext, WorkerSelectionPolicy,
        },
    };
    use dynamo_runtime::{
        Runtime,
        pipeline::{AsyncEngineContextProvider, Context, RouterMode},
    };
    use std::{
        collections::HashMap,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };
    use tokio_util::sync::CancellationToken;

    struct Picker {
        action: PrefillAction,
        calls: Arc<AtomicUsize>,
        prefill_load: bool,
        expected_prefill_busy: Option<bool>,
    }
    impl WorkerPicker for Picker {
        fn path_planning(&self) -> Option<PathPlanningRequirements> {
            Some(PathPlanningRequirements {
                prefill_load: self.prefill_load,
            })
        }
        fn pick(
            &mut self,
            _: &WorkerSelectionContext<'_>,
            _: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            Ok(0)
        }
        fn pick_route(
            &mut self,
            ctx: &WorkerSelectionContext<'_>,
            _: WorkerInputView<'_>,
        ) -> Result<RouteChoice, WorkerSelectionPolicyError> {
            assert!(ctx.is_path_planning());
            assert_eq!(ctx.prefill_worker_busy(), self.expected_prefill_busy);
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(RouteChoice {
                candidate: 0,
                prefill: self.action,
            })
        }
    }

    async fn setup(
        action: PrefillAction,
        capability: Option<bool>,
        enabled: bool,
    ) -> (
        Arc<PrefillRouter>,
        Arc<RoutingHost>,
        Runtime,
        Arc<AtomicUsize>,
        tokio::sync::watch::Sender<HashMap<u64, ModelRuntimeConfig>>,
    ) {
        setup_with_probe_requirements(
            action,
            capability,
            KvRouterConfig {
                conditional_disagg_enabled: enabled,
                conditional_disagg_decode_busy_threshold: Some(0.0),
                ..Default::default()
            },
            false,
            None,
        )
        .await
    }

    async fn setup_with_probe_requirements(
        action: PrefillAction,
        capability: Option<bool>,
        router_config: KvRouterConfig,
        prefill_load: bool,
        expected_prefill_busy: Option<bool>,
    ) -> (
        Arc<PrefillRouter>,
        Arc<RoutingHost>,
        Runtime,
        Arc<AtomicUsize>,
        tokio::sync::watch::Sender<HashMap<u64, ModelRuntimeConfig>>,
    ) {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let policy = SelectionPolicySource::Factory(Arc::new(move |config, role, _| {
            WorkerSelectionPolicy::new(
                config.clone(),
                role.as_str(),
                vec![],
                Box::new(Picker {
                    action,
                    calls: observed.clone(),
                    prefill_load,
                    expected_prefill_busy,
                }),
            )
        }));
        let mut config = ModelRuntimeConfig::default();
        if let Some(capability) = capability {
            config
                .runtime_data
                .insert("local_prefill".into(), serde_json::json!(capability));
        }
        let (host, runtime, updates) = router_with_worker_policy_updates(
            None,
            HashMap::from([(7, config)]),
            policy,
            Some(crate::worker_type::WorkerType::Decode),
        )
        .await;
        let host = Arc::new(host);
        let (_tx, rx) = tokio::sync::oneshot::channel();
        let router = PrefillRouter::new(
            rx,
            Arc::new(ModelManager::new()),
            RouterMode::KV,
            16,
            Some(router_config),
            None,
            None,
            SessionAffinityMode::Hard,
            "test".into(),
            "test".into(),
            crate::discovery::LoadThresholdHandle::new(Default::default()),
            CancellationToken::new(),
        );
        router.set_decode_routing_host(host.clone()).unwrap();
        (router, host, runtime, calls, updates)
    }

    fn request() -> Context<PreprocessedRequest> {
        Context::new(
            PreprocessedRequest::builder()
                .model("test".into())
                .token_ids(vec![1; 32])
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .build()
                .unwrap(),
        )
    }

    async fn assert_no_booking(host: &RoutingHost) {
        let loads = host
            .kv_router()
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert!(loads.iter().all(|l| l.active_requests == 0), "{loads:?}");
    }

    struct PrefillProbePicker(Arc<AtomicUsize>);

    impl WorkerPicker for PrefillProbePicker {
        fn pick(
            &mut self,
            context: &WorkerSelectionContext<'_>,
            _: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            assert!(!context.is_path_planning());
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(0)
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn plugin_prefill_load_probe_is_optional_and_reused_by_default() {
        for (action, declared, builtin_load, expected_probes) in [
            (PrefillAction::Remote, false, false, 0),
            (PrefillAction::Remote, true, false, 1),
            (PrefillAction::Default, true, true, 1),
            (PrefillAction::Default, false, true, 1),
        ] {
            let (router, decode_host, decode_runtime, route_calls, _decode_updates) =
                setup_with_probe_requirements(
                    action,
                    Some(true),
                    KvRouterConfig {
                        conditional_disagg_enabled: builtin_load,
                        conditional_disagg_policy:
                            dynamo_kv_router::config::ConditionalDisaggPolicyKind::PrefillLoad,
                        conditional_disagg_prefill_busy_threshold: Some(0.5),
                        ..Default::default()
                    },
                    declared,
                    declared.then_some(false),
                )
                .await;
            let probe_calls = Arc::new(AtomicUsize::new(0));
            let observed = probe_calls.clone();
            let policy = SelectionPolicySource::Factory(Arc::new(move |config, role, _| {
                WorkerSelectionPolicy::new(
                    config.clone(),
                    role.as_str(),
                    vec![],
                    Box::new(PrefillProbePicker(observed.clone())),
                )
            }));
            let (prefill_host, prefill_runtime, _prefill_updates) =
                router_with_worker_policy_updates(
                    None,
                    HashMap::from([(
                        9,
                        ModelRuntimeConfig {
                            max_num_batched_tokens: Some(1024),
                            ..Default::default()
                        },
                    )]),
                    policy,
                    Some(crate::worker_type::WorkerType::Prefill),
                )
                .await;
            let prefill_host = Arc::new(prefill_host);
            let endpoint_id = dynamo_runtime::protocols::EndpointId {
                namespace: "prefill-probe".into(),
                component: "workers".into(),
                name: "generate".into(),
            };
            router.binding.store(Some(Arc::new(
                crate::kv_router::prefill_router::PrefillBinding {
                    target_id: crate::discovery::WorkerSetTargetId::Legacy(endpoint_id.clone()),
                    endpoint_id,
                    router: prefill_host.clone(),
                    prefill_router_mode: RouterMode::KV,
                },
            )));
            let mut request = request();
            let id = request.context().id().to_string();
            assert!(
                router
                    .plan_conditional_disagg_decode(&mut request, &id)
                    .await
                    .unwrap()
                    .is_none()
            );
            assert_eq!(probe_calls.load(Ordering::Relaxed), expected_probes);
            assert_eq!(route_calls.load(Ordering::Relaxed), 1);
            assert_no_booking(&prefill_host).await;
            assert_no_booking(&decode_host).await;
            drop(router);
            drop(prefill_host);
            drop(decode_host);
            prefill_runtime.shutdown();
            decode_runtime.shutdown();
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn plugin_local_prefill_works_when_builtin_disabled_and_charges_local_work() {
        let (router, host, runtime, calls, _) =
            setup(PrefillAction::LocalOnDecode, Some(true), false).await;
        let mut request = request();
        request.router_config_override = Some(dynamo_kv_router::RouterConfigOverride {
            track_prefill_tokens: Some(false),
            ..Default::default()
        });
        let id = request.context().id().to_string();
        let decision = router
            .plan_conditional_disagg_decode(&mut request, &id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(decision.plan.signals.worker.worker_id, 7);
        assert_eq!(
            request
                .router_config_override
                .as_ref()
                .unwrap()
                .track_prefill_tokens,
            Some(true)
        );
        let loads = host
            .kv_router()
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert_eq!(loads[0].active_requests, 1);
        assert_eq!(loads[0].potential_prefill_tokens, 32);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        decision.plan.abort().await;
        assert_no_booking(&host).await;
        drop(router);
        drop(host);
        runtime.shutdown();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn plugin_actions_validate_capability_and_prefill_pins_without_booking() {
        for (capability, pin) in [(None, false), (Some(false), false), (Some(true), true)] {
            let (router, host, runtime, _, _) =
                setup(PrefillAction::LocalOnDecode, capability, true).await;
            let mut request = request();
            if pin {
                request.routing_mut().prefill_worker_id = Some(9);
            }
            let id = request.context().id().to_string();
            let result = router
                .plan_conditional_disagg_decode(&mut request, &id)
                .await;
            assert!(matches!(result, Err(ref e) if e.is::<PrefillActionError>()));
            assert_no_booking(&host).await;
            drop(router);
            drop(host);
            runtime.shutdown();
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn plugin_local_plan_rejects_revoked_capability_and_releases_booking() {
        let (router, host, runtime, calls, updates) =
            setup(PrefillAction::LocalOnDecode, Some(true), false).await;
        let mut request = request();
        let id = request.context().id().to_string();
        let decision = router
            .plan_conditional_disagg_decode(&mut request, &id)
            .await
            .unwrap()
            .unwrap();
        updates.send_modify(|workers| {
            workers
                .get_mut(&7)
                .unwrap()
                .runtime_data
                .insert("local_prefill".into(), serde_json::json!(false));
        });
        let result = host.dispatch_kv_plan(request, decision.plan).await;
        assert!(matches!(result, Err(ref e) if e.to_string().contains("local prefill support")));
        assert_no_booking(&host).await;
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        drop(router);
        drop(host);
        runtime.shutdown();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn plugin_default_preserves_unknown_decode_load_veto() {
        let (router, host, runtime, calls, _) =
            setup(PrefillAction::Default, Some(true), true).await;
        let mut request = request();
        let id = request.context().id().to_string();
        assert!(
            router
                .plan_conditional_disagg_decode(&mut request, &id)
                .await
                .unwrap()
                .is_none()
        );
        assert_no_booking(&host).await;
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        drop(router);
        drop(host);
        runtime.shutdown();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn plugin_remote_and_disabled_default_do_not_admit() {
        for action in [PrefillAction::Remote, PrefillAction::Default] {
            let (router, host, runtime, calls, _) = setup(action, Some(true), false).await;
            let mut request = request();
            let id = request.context().id().to_string();
            assert!(
                router
                    .plan_conditional_disagg_decode(&mut request, &id)
                    .await
                    .unwrap()
                    .is_none()
            );
            assert_no_booking(&host).await;
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            drop(router);
            drop(host);
            runtime.shutdown();
        }
    }
}
