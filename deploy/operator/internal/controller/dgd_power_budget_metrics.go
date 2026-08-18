/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"strings"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/prometheus/client_golang/prometheus"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

const powerMetricsNamespace = "dynamo_operator"

const (
	powerRecoveryActionScaleDown     = "scale_down"
	powerRecoveryActionStabilityHold = "stability_hold"
	powerRecoveryActionReopen        = "reopen"
	powerRecoveryActionInfeasible    = "infeasible"
)

var (
	powerRequestVectors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: powerMetricsNamespace,
			Name:      "power_request_vectors_total",
			Help:      "Transactional replica-vector admission outcomes.",
		},
		[]string{"result", "reason"},
	)
	powerBudgetPhases = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: powerMetricsNamespace,
			Name:      "power_budget_phase_observations_total",
			Help:      "Observed DynamoGraphPowerBudget phases.",
		},
		[]string{"phase"},
	)
	powerChargedWatts = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: powerMetricsNamespace,
			Name:      "power_charged_watts",
			Help:      "Conservative charged watts observed by ledger class.",
			Buckets:   []float64{0, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000},
		},
		[]string{"class"},
	)
	powerEnforcementFreshness = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: powerMetricsNamespace,
			Name:      "power_enforcement_freshness_total",
			Help:      "Bounded enforcement-report freshness observations.",
		},
		[]string{"state"},
	)
	powerGateWaitOutcomes = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: powerMetricsNamespace,
			Name:      "power_gate_wait_outcomes_total",
			Help:      "Power-gate wait termination outcomes.",
		},
		[]string{"outcome"},
	)
	powerGateWaitSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: powerMetricsNamespace,
			Name:      "power_gate_wait_seconds",
			Help:      "Elapsed power-gate wait by bounded success or failure outcome.",
			Buckets:   []float64{.1, .25, .5, 1, 2.5, 5, 10, 15, 30, 60, 120, 300},
		},
		[]string{"outcome"},
	)
	powerRecoveryActions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: powerMetricsNamespace,
			Name:      "power_recovery_actions_total",
			Help:      "Replica-only recovery actions.",
		},
		[]string{"action"},
	)
	powerReportFailures = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: powerMetricsNamespace,
			Name:      "power_report_failures_total",
			Help:      "Rejected power Agent reports by bounded reason.",
		},
		[]string{"reason"},
	)
	powerCapClamped = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: powerMetricsNamespace,
			Name:      "power_cap_clamped_total",
			Help:      "Immutable transactional power intents clamped to a qualified product range.",
		},
		[]string{"direction", "product"},
	)

	powerMetricCollectors = []prometheus.Collector{
		powerRequestVectors,
		powerBudgetPhases,
		powerChargedWatts,
		powerEnforcementFreshness,
		powerGateWaitOutcomes,
		powerGateWaitSeconds,
		powerRecoveryActions,
		powerReportFailures,
		powerCapClamped,
	}
	powerMetricVariableLabels = [][]string{
		{"result", "reason"},
		{"phase"},
		{"class"},
		{"state"},
		{"outcome"},
		{"outcome"},
		{"action"},
		{"reason"},
		{"direction", "product"},
	}
)

func init() {
	ctrlmetrics.Registry.MustRegister(powerMetricCollectors...)
}

func recordPowerRequestVector(accepted bool, reason nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReason) {
	result := "pending"
	if accepted {
		result = "admitted"
	}
	powerRequestVectors.WithLabelValues(result, boundedPowerPendingReason(reason)).Inc()
}

func boundedPowerPendingReason(reason nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReason) string {
	switch reason {
	case "":
		return "none"
	case nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonBudgetExceeded,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnsupportedTopology,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnqualifiedHardware,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonBelowMinimum,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonInvalidTarget:
		return string(reason)
	default:
		return "unknown"
	}
}

func recordPowerBudgetStatus(status nvidiacomv1beta1.DynamoGraphPowerBudgetStatus) {
	powerBudgetPhases.WithLabelValues(boundedPowerPhase(status.Phase)).Inc()
	for class, watts := range map[string]int64{
		"enforced":         status.Ledger.EnforcedWatts,
		"unknown":          status.Ledger.UnknownWatts,
		"in_gate_reserved": status.Ledger.InGateReservedWatts,
		"rollout_extra":    status.Ledger.RolloutExtraWatts,
	} {
		powerChargedWatts.WithLabelValues(class).Observe(float64(watts))
	}
}

func boundedPowerPhase(phase nvidiacomv1beta1.DynamoGraphPowerBudgetPhase) string {
	switch phase {
	case nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInfeasible,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseUnqualified:
		return string(phase)
	default:
		return "Unknown"
	}
}

func recordPowerReportEvidence(evidence podPowerEvidence) {
	state := "fresh"
	if !evidence.Accepted {
		state = "rejected"
		if evidence.Reason == "report missing" {
			state = "missing"
		}
		powerReportFailures.WithLabelValues(boundedPowerReportFailure(evidence.Reason)).Inc()
	}
	powerEnforcementFreshness.WithLabelValues(state).Inc()
}

func boundedPowerReportFailure(reason string) string {
	switch reason {
	case "report missing":
		return "missing"
	case "report identity mismatch":
		return "identity_mismatch"
	case "report GPU count mismatch":
		return "gpu_count_mismatch"
	case "report GPU allocation mismatch":
		return "allocation_mismatch"
	case "report power intent mismatch":
		return "power_intent_mismatch"
	case "not every assigned GPU has healthy evidence":
		return "stale_or_unhealthy"
	default:
		return "invalid"
	}
}

func recordPowerGateWait(outcome string, elapsed time.Duration, durationKnown bool) {
	switch outcome {
	case "success":
		outcome = "success"
	case "PowerGateConfigurationError":
		outcome = "configuration_error"
	case "PowerGateEnforcementTimeout":
		outcome = "enforcement_timeout"
	case "PowerGateExecFailed":
		outcome = "exec_failed"
	default:
		outcome = "failed"
	}
	powerGateWaitOutcomes.WithLabelValues(outcome).Inc()
	if durationKnown && elapsed >= 0 {
		powerGateWaitSeconds.WithLabelValues(outcome).Observe(elapsed.Seconds())
	}
}

func recordPowerRecoveryAction(action string) {
	switch action {
	case powerRecoveryActionScaleDown,
		powerRecoveryActionStabilityHold,
		powerRecoveryActionReopen,
		powerRecoveryActionInfeasible:
	default:
		action = "unknown"
	}
	powerRecoveryActions.WithLabelValues(action).Inc()
}

func recordPowerCapClamp(direction, product string) {
	if direction != "below_min" && direction != "above_max" {
		direction = "unknown"
	}
	powerCapClamped.WithLabelValues(direction, product).Inc()
}

func powerMetricLabelIsIdentity(label string) bool {
	lower := strings.ToLower(label)
	for _, prohibited := range []string{"gpu", "uuid", "pod", "node", "allocation", "container"} {
		if strings.Contains(lower, prohibited) {
			return true
		}
	}
	return false
}
