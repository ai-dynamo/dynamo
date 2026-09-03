// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_kv_router::conditional_disagg::ConditionalDisaggDecisionInput;
use dynamo_kv_router::selector::WorkerSelector;
use dynamo_runtime::pipeline::SingleIn;

use super::PrefillRouter;
use crate::kv_router::routing_host::{RoutePlan, RoutePlanSignals, is_cancelled};
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::protocols::common::{llm_backend::PreprocessedRequest, timing::RequestPhase};

/// An admitted decode route selected by `RoutingHost` together with the
/// conditional-disagg policy's diagnostic signals.
pub(super) struct ConditionalDisaggDecodeDecision<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    pub plan: RoutePlan<Sel>,
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

impl<Sel> PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    /// Preview one decode route, then admit it only when the topology policy
    /// chooses local decode.
    pub(super) async fn plan_conditional_disagg_decode(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        request_id: &str,
    ) -> Result<Option<ConditionalDisaggDecodeDecision<Sel>>> {
        // Conditional disagg chooses a cache-hot decode worker, so it only
        // applies to a KV-routed decode set.
        if !self.decode_router_mode.is_kv_routing() {
            return Ok(None);
        }

        if request
            .routing
            .as_ref()
            .is_some_and(|routing| routing.prefill_worker_id.is_some())
        {
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

        let preview = decode_host
            .preview_kv_route(request, RequestPhase::Decode)
            .await?;
        let signals = preview.signals();
        let mut input =
            ConditionalDisaggDecisionInput::new(routing_token_ids.len(), signals.cached_tokens);
        if self.conditional_disagg_policy.needs_prefill_worker_busy() {
            let busy = match self.peek_prefill_chosen_worker_busy(request).await {
                Ok(busy) => busy,
                Err(error) if is_cancelled(&error) => return Err(error),
                Err(error) => {
                    tracing::debug!(
                        request_id,
                        %error,
                        "Conditional disagg prefill-load probe failed; treating load as unavailable"
                    );
                    None
                }
            };
            tracing::debug!(
                request_id,
                prefill_chosen_worker_busy = ?busy,
                "Conditional disagg prefill-load condition inspected selected prefill worker"
            );
            input = input.with_prefill_chosen_worker_busy(busy);
        }
        let net_new_tokens = input.net_new_tokens();
        let overlap_tokens =
            (signals.overlap_blocks as usize) * (decode_host.kv_router().block_size() as usize);

        let policy_says_bypass = self
            .conditional_disagg_policy
            .should_bypass_remote_prefill(input)
            .await;
        let decode_gate_configured = self.conditional_disagg_decode_busy_threshold.is_some();
        let decode_busy = if policy_says_bypass {
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
            let plan = decode_host
                .plan_kv_route_from_preview(request, preview)
                .await?;
            return Ok(Some(ConditionalDisaggDecodeDecision {
                plan,
                overlap_tokens,
                net_new_tokens,
            }));
        }

        Ok(None)
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
