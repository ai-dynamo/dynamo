// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	maxConditionMessageSize = 32768
	// SchedulingFailed is durable expiry/retry evidence, separate from aggregate Ready.
	schedulingFailedCondition             = "SchedulingFailed"
	pipelineRequestDeadlineExceededReason = "LPXSchedulingDeadlineExceeded"
)

// updateStatus persists the updated status on the deployment.
func (r *graphReconciler) updateStatus(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, previous *v1alpha1.LPXGraphDeploymentStatus) error {
	for name, component := range deployment.Status.Components {
		ready := meta.FindStatusCondition(component.Conditions, v1alpha1.LPXReadyCondition)
		if ready == nil {
			ready = &metav1.Condition{
				Type: v1alpha1.LPXReadyCondition, Status: metav1.ConditionUnknown,
				ObservedGeneration: deployment.Generation, Reason: "NotObserved",
				Message: "Component readiness has not been observed",
			}
		}
		// Compare the final observation to persisted history, not intermediate states.
		component.Conditions = slices.Clone(previous.Components[name].Conditions)
		meta.SetStatusCondition(&component.Conditions, *ready)
		deployment.Status.Components[name] = component
	}

	// Persist the complete observation without overwriting per-workload readiness.
	if apiequality.Semantic.DeepEqual(previous, &deployment.Status) {
		return nil
	}

	if err := r.Status().Update(ctx, deployment); err != nil {
		return fmt.Errorf("persist LPXGraphDeployment status: %w", err)
	}

	return nil
}

// setReadyCondition updates the deployment's Ready condition.
func setReadyCondition(deployment *v1alpha1.LPXGraphDeployment, state v1beta1.DGDState, message string) {
	meta.SetStatusCondition(&deployment.Status.Conditions, readyCondition(deployment.Generation, state, message))
}

func readyCondition(generation int64, state v1beta1.DGDState, message string) metav1.Condition {
	status, reason := metav1.ConditionFalse, v1alpha1.LPXReadyReasonPending
	switch state {
	case v1beta1.DGDStateSuccessful:
		status, reason = metav1.ConditionTrue, v1alpha1.LPXReadyReasonReady
	case v1beta1.DGDStateFailed:
		reason = v1alpha1.LPXReadyReasonFailed
	}
	return metav1.Condition{
		Type: v1alpha1.LPXReadyCondition, Status: status, ObservedGeneration: generation,
		Reason: reason, Message: truncateConditionMessage(message),
	}
}

// setPipelineRequestReadyCondition reports the first current failure, otherwise the first
// non-Bound request, and returns whether every desired request is Bound.
// Absent receipts and stale receipts are still pending.
func setPipelineRequestReadyCondition(conditions *[]metav1.Condition, generation int64, requests map[string]*lpxv1alpha1.LPUPipelineRequest) bool {
	// Failures outrank pending receipts; stable name order keeps diagnostics consistent.
	allBound := true
	var pending *lpxv1alpha1.LPUPipelineRequestStatus
	for _, name := range slices.Sorted(maps.Keys(requests)) {
		request := requests[name]
		// Only receipts for the request's current generation can prove Bound or failure.
		var status *lpxv1alpha1.LPUPipelineRequestStatus
		if request.Status != nil && request.Status.ObservedGeneration != nil && *request.Status.ObservedGeneration == request.Generation {
			status = request.Status
		}

		if status != nil {
			switch status.Phase {
			case lpxv1alpha1.RequestPhaseBound:
				continue
			case lpxv1alpha1.RequestPhasePending, lpxv1alpha1.RequestPhaseNoFit,
				lpxv1alpha1.RequestPhasePlanned, lpxv1alpha1.RequestPhaseReserving,
				lpxv1alpha1.RequestPhaseBinding, lpxv1alpha1.RequestPhaseDegraded,
				lpxv1alpha1.RequestPhaseReleasing, lpxv1alpha1.RequestPhaseReleased:
				// Keep the first pending receipt while checking the remaining requests for failure.
			default:
				setPipelineRequestPhaseCondition(conditions, generation, status)
				return false
			}
		}

		if allBound {
			pending = status
		}
		allBound = false
	}

	if !allBound {
		setPipelineRequestPhaseCondition(conditions, generation, pending)
	}
	return allBound
}

// setPipelineRequestPhaseCondition formats a non-Bound receipt without copying it.
// Nil status means no current scheduler receipt exists.
func setPipelineRequestPhaseCondition(conditions *[]metav1.Condition, generation int64, status *lpxv1alpha1.LPUPipelineRequestStatus) {
	// A missing or stale receipt cannot establish scheduler progress.
	if status == nil {
		meta.SetStatusCondition(conditions, readyCondition(generation, v1beta1.DGDStatePending, "LPUPipelineRequest is published"))
		return
	}

	// Scheduler diagnostics describe only the selected current receipt.
	var (
		state   = v1beta1.DGDStatePending
		summary string
	)

	switch status.Phase {
	case lpxv1alpha1.RequestPhasePending:
		summary = "LPX scheduler is waiting to plan the current request"
	case lpxv1alpha1.RequestPhaseNoFit:
		summary = "LPX scheduler found no placement for the current request"
	case lpxv1alpha1.RequestPhaseUnsupported:
		state = v1beta1.DGDStateFailed
		summary = "LPX scheduler cannot support the current request"
	case lpxv1alpha1.RequestPhasePlanned:
		summary = "LPX scheduler committed a placement plan for the current request"
	case lpxv1alpha1.RequestPhaseReserving:
		summary = "LPX scheduler is reserving the planned LPU allocation"
	case lpxv1alpha1.RequestPhaseBinding:
		summary = "LPX scheduler is binding the planned workload Pods"
	case lpxv1alpha1.RequestPhaseDegraded:
		summary = "LPX scheduler is repairing or releasing a degraded placement"
	case lpxv1alpha1.RequestPhaseReleasing:
		summary = "LPX scheduler is releasing the current plan"
	case lpxv1alpha1.RequestPhaseReleased:
		summary = "LPX scheduler released the current plan"
	default:
		state = v1beta1.DGDStateFailed
		summary = fmt.Sprintf("LPX scheduler reported unknown phase %q", status.Phase)
	}

	meta.SetStatusCondition(conditions, readyCondition(generation, state, pipelineRequestDiagnosticMessage(summary, status.Diagnostics)))
}

// isSchedulingFailedConditionCurrent reports whether the current generation must remain
// failed until a later input revision explicitly authorizes another publication.
func isSchedulingFailedConditionCurrent(deployment *v1alpha1.LPXGraphDeployment) bool {
	failed := meta.FindStatusCondition(deployment.Status.Conditions, schedulingFailedCondition)
	return failed != nil && failed.Status == metav1.ConditionTrue && failed.Reason == pipelineRequestDeadlineExceededReason &&
		failed.ObservedGeneration >= deployment.Generation
}

// schedulingFailureCoversPipelineRequests reports whether every expired scheduling cycle
// predates the recorded failure, even after an input edit. A later cycle needs
// its own durable failure before cleanup; cleanup must not consume a retry edit.
func schedulingFailureCoversPipelineRequests(
	deployment *v1alpha1.LPXGraphDeployment,
	requests []*lpxv1alpha1.LPUPipelineRequest,
) bool {
	failed := meta.FindStatusCondition(deployment.Status.Conditions, schedulingFailedCondition)
	if failed == nil || failed.Status != metav1.ConditionTrue || failed.Reason != pipelineRequestDeadlineExceededReason {
		return false
	}
	if failed.LastTransitionTime.IsZero() {
		return false
	}
	for _, request := range requests {
		startedAt := pipelineRequestSchedulingStartedAt(request)
		if startedAt.IsZero() || !startedAt.Before(failed.LastTransitionTime.Time) {
			return false
		}
	}
	return true
}

// setSchedulingFailedCondition updates a non-nil deployment's in-memory failure.
// renew starts a new durable fence; otherwise the original failure time and
// generation survive retries. Reconcile persists this before a later pass deletes LPRs.
func setSchedulingFailedCondition(deployment *v1alpha1.LPXGraphDeployment, renew bool) {
	observedGeneration := deployment.Generation
	prior := meta.FindStatusCondition(deployment.Status.Conditions, schedulingFailedCondition)
	if prior != nil && prior.Status == metav1.ConditionTrue && !renew {
		observedGeneration = prior.ObservedGeneration
	}

	// Even True-to-True renewal needs a fresh timestamp to cover the newly expired cycle.
	if renew {
		meta.RemoveStatusCondition(&deployment.Status.Conditions, schedulingFailedCondition)
	}

	// Aggregate failure and retry authorization have independent condition lifetimes.
	const message = "An LPX request exceeded its scheduling deadline"
	setReadyCondition(deployment, v1beta1.DGDStateFailed, message)
	meta.SetStatusCondition(&deployment.Status.Conditions, metav1.Condition{
		Type: schedulingFailedCondition, Status: metav1.ConditionTrue,
		ObservedGeneration: observedGeneration, Reason: pipelineRequestDeadlineExceededReason, Message: message,
	})
}

// acknowledgeSchedulingRetry marks an older failure superseded by a later generation.
// deployment is non-nil; callers must not use this after observing a current expiry.
func acknowledgeSchedulingRetry(deployment *v1alpha1.LPXGraphDeployment) {
	prior := meta.FindStatusCondition(deployment.Status.Conditions, schedulingFailedCondition)
	if prior == nil || deployment.Generation <= prior.ObservedGeneration {
		return
	}
	meta.SetStatusCondition(&deployment.Status.Conditions, metav1.Condition{
		Type: schedulingFailedCondition, Status: metav1.ConditionFalse,
		ObservedGeneration: deployment.Generation, Reason: "LPXSchedulingRetryAuthorized",
		Message: "The current input revision supersedes the previous scheduling failure",
	})
}

func pipelineRequestDiagnosticMessage(summary string, diagnostics []lpxv1alpha1.StatusDiagnostic) string {
	if len(diagnostics) == 0 {
		return summary
	}
	var message strings.Builder
	message.WriteString(summary)
	message.WriteString(". LPX diagnostics: ")
	for index, diagnostic := range diagnostics {
		if index > 0 {
			message.WriteString("; ")
		}
		message.WriteString(diagnostic.Code)
		message.WriteString(" [")
		message.WriteString(diagnostic.Subject)
		message.WriteString("]: ")
		message.WriteString(diagnostic.Detail)

		// Later diagnostics cannot change the retained prefix.
		if message.Len() > maxConditionMessageSize {
			break
		}
	}
	return truncateConditionMessage(message.String())
}

func truncateConditionMessage(message string) string {
	if len(message) <= maxConditionMessageSize {
		return message
	}
	const suffix = "..."
	last := maxConditionMessageSize - len(suffix)
	for last > 0 && !utf8.ValidString(message[:last]) {
		last--
	}
	return message[:last] + suffix
}
